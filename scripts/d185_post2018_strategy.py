#!/usr/bin/env python3
"""D185 — re-optimise the trading strategy for the POST-2018 frame, with the
owner's December skip pre-committed, and characterise which conditions are
structurally hard for THIS model rather than merely low-scoring.

Owner: "we will optimize for post 2018, when injury reports come into play. lets
redo trading strategy, taking out high drawdown periods and explaining why they
are structurally difficult for our model" + "so skip december".

THE DISTINCTION THIS SCRIPT ENFORCES.  Removing a stretch BECAUSE it drew down is
curve-fitting — the stretch is identified by the endpoint.  Removing it because a
condition OBSERVABLE AT BET TIME says the model is weak there is a strategy.  So
every filter below is (a) computable before the game, (b) motivated by a named
mechanism in the model's own construction, and (c) validated walk-forward against
the unfiltered incumbent (D176: net-of-null is not sufficient).

FILTERS, each with its mechanism, declared before scoring:
  F0  DECEMBER SKIP — owner pre-commitment (D184).  Registered as a judgment
      override, not a gated finding.  Scored here so its live cost is known.
  F1  COLD START — a team's first C games of the season.  MECHANISM: the
      four-factors leg is an opponent-adjusted ridge solve on in-season
      possessions; with few games it is prior-dominated, which is why the
      October bridge (D84-A) exists at all.  Half the margin is therefore
      running on a prior rather than on data.
  F2  ABSENCE LOAD — games where the two teams' TRAILING absence burden is high.
      MECHANISM: the availability leg is a linear sum of DARKO x trailing
      minutes over available players.  When many players are out, (i) the sum
      extrapolates to roster states it was never calibrated on, and (ii) the
      replacements have no stable DARKO rating or minutes history, so the leg's
      inputs are least reliable exactly when it matters most.
      Measured on the TRAILING 5 games, never tonight's report, because at the
      OPEN tonight's 5pm report does not yet exist (LEAKAGE.md).

Read-only.  Nothing ships.  No production default changed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

import oc_capacity as oc                                          # noqa: E402
from nbapred.db import connect                                    # noqa: E402

MODERN = "2019-20"   # D186: 2018-19 is only 63.7% injury-covered (reports
                     # begin 2018-12-17, MID-season) -> first fully covered season
K_START = 3            # need >=3 post-2018 seasons of history before selecting
SEED = 20260805


def trailing_absence(df):
    """Mean Out/Doubtful listings for the two teams over their trailing 5 games.

    Strictly prior to tonight: uses only reports dated BEFORE game_date.
    """
    con = connect(read_only=True)
    try:
        rows = con.execute("""
            SELECT game_date, team, count(*) n
            FROM injury_reports_pit
            WHERE status IN ('Out','Doubtful')
            GROUP BY 1,2""").fetchall()
    except Exception as e:
        con.close()
        print(f"  (absence probe unavailable: {str(e)[:70]})")
        return None
    con.close()
    if not rows:
        return None
    a = pd.DataFrame(rows, columns=["game_date", "team", "n"])
    # D185 FIX (5th instance of the team-name bug class): injury_reports_pit
    # stores FULL names ("Toronto Raptors"); the frame carries abbrevs ("TOR").
    # Resolve through nbapred/teams.py, which REPORTS what it cannot map.
    from nbapred import teams as T
    amap, unresolved = T.resolve_map(sorted(a["team"].unique()))
    if unresolved:
        print(f"  [teams] {len(unresolved)} unresolvable report names, "
              f"REPORTED not dropped: {unresolved[:6]}"
              f"{' ...' if len(unresolved) > 6 else ''}")
    a["team"] = a["team"].map(amap)
    a = a[a["team"].notna()]
    a["game_date"] = pd.to_datetime(a["game_date"])
    a = a.sort_values("game_date")
    # trailing mean per team, shifted so tonight is excluded
    a["tr"] = (a.groupby("team")["n"]
                 .transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean()))
    m = dict(zip(zip(a["team"], a["game_date"]), a["tr"]))
    d = pd.to_datetime(df["game_date"])
    out = []
    for h, aw, dt_ in zip(df["home"], df["away"], d):
        v = [m.get((h, dt_)), m.get((aw, dt_))]
        v = [x for x in v if x is not None and np.isfinite(x)]
        out.append(np.mean(v) if v else np.nan)
    return np.array(out, float)


def game_index(df):
    """0-based index of each game within its team's season (max of the two)."""
    idx = np.zeros(len(df), int)
    cnt = {}
    for i, (s, h, a) in enumerate(zip(df["season"], df["home"], df["away"])):
        ih = cnt.get((s, h), 0)
        ia = cnt.get((s, a), 0)
        idx[i] = max(ih, ia)
        cnt[(s, h)] = ih + 1
        cnt[(s, a)] = ia + 1
    return idx


def walk_forward(cnt, pay, K, k_start):
    """All-history walk-forward selection on the supplied season slice."""
    from wf_equity import select
    steps = []
    for k in range(k_start, K):
        c = cnt[:, :k].sum(1)
        p = pay[:, :k].sum(1)
        best = select(c, p, 100.0 * k)
        if best is None or cnt[best, k] <= 0:
            continue
        steps.append(dict(k=k, cfg=best, roi=float(pay[best, k] / cnt[best, k]),
                          n=float(cnt[best, k]), pay=float(pay[best, k])))
    return steps


def score(steps, seasons, label):
    if not steps:
        print(f"  {label:38s}  (no scoreable steps)")
        return None
    n = sum(s["n"] for s in steps)
    p = sum(s["pay"] for s in steps)
    per = np.array([s["roi"] for s in steps], float)
    K = len(per)
    se = per.std(ddof=1) / np.sqrt(K) if K > 1 else np.nan
    tc = oc.t_crit(K - 1) if K > 1 else np.nan
    roi = 100 * p / n
    print(f"  {label:38s} n={n:6.0f}  ROI {roi:+7.2f}%  "
          f"cum {p:+7.1f}u  K={K}  "
          f"95%CI [{roi-100*tc*se:+7.2f},{roi+100*tc*se:+7.2f}]")
    return dict(label=label, n=float(n), roi=float(roi), pay=float(p),
                K=K, per=[float(x) for x in per],
                ci=[float(roi - 100 * tc * se), float(roi + 100 * tc * se)])


def main():
    df, seasons = oc.load()
    keep_s = [s for s in seasons if s >= MODERN]
    print(f"POST-2018 FRAME: {len(keep_s)} seasons {keep_s[0]}..{keep_s[-1]}")

    mask_season = df["season"].isin(keep_s).to_numpy()
    dfm = df[mask_season].reset_index(drop=True)
    # D185 FIX: s_i still carries ORIGINAL 19-season indices. agg() scatters on
    # s_i, so without re-indexing every bet lands in columns 11..18 and the
    # walk-forward slice cnt[:, :k] reads empty columns and finds nothing.
    dfm["s_i"] = dfm["season"].map({s: i for i, s in enumerate(keep_s)}).astype(int)
    print(f"  games {len(dfm)}  (of {len(df)} all-era)\n")

    st = oc.build_static(dfm)
    payoff, M, keys, win, push, bet_home = oc.payoff_and_masks(
        dfm["m_us"].to_numpy(float), dfm["p_us"].to_numpy(float), st)
    K = len(keep_s)

    # ---- covariates, all computable before tip
    month = dfm["month"].to_numpy(int)
    gidx = game_index(dfm)
    absl = trailing_absence(dfm)
    print("COVARIATES")
    print(f"  December games: {(month == 12).sum()} "
          f"({100*(month==12).mean():.1f}%)")
    print(f"  game index within season: median {np.median(gidx):.0f}, "
          f"first-10 share {100*(gidx < 10).mean():.1f}%")
    if absl is not None:
        ok = np.isfinite(absl)
        print(f"  trailing absence load: coverage {100*ok.mean():.1f}%, "
              f"median {np.nanmedian(absl):.2f}, "
              f"p80 {np.nanpercentile(absl[ok], 80):.2f}")

    def run(extra_keep, label):
        km = mask_season.copy()
        sub = np.ones(len(dfm), bool) if extra_keep is None else extra_keep
        Msub = M * sub[None, :]
        cnt, pay = oc.agg(Msub, payoff, st)
        return score(walk_forward(cnt, pay, K, K_START), keep_s, label)

    print("\n" + "=" * 78)
    print("WALK-FORWARD ON THE POST-2018 FRAME  (train 1..k, score k+1)")
    print("=" * 78)
    res = {}
    res["incumbent"] = run(None, "INCUMBENT (no filter)")
    res["F0_dec"] = run(month != 12, "F0  skip December  [owner pre-commit]")

    print("\n  F1 COLD START — drop a team's first C games (four-factors is")
    print("     prior-dominated early; half the margin runs on a prior)")
    for C in (5, 10, 15, 20):
        res[f"F1_{C}"] = run((gidx >= C) & (month != 12),
                             f"     +F1 drop first {C:2d} games")

    if absl is not None:
        print("\n  F2 ABSENCE LOAD — drop games whose TRAILING absence burden is")
        print("     high (composition leg extrapolating on unrated replacements)")
        ok = np.isfinite(absl)
        for q in (90, 80, 70):
            thr = np.nanpercentile(absl[ok], q)
            k = (~ok) | (absl < thr)
            res[f"F2_{q}"] = run(k & (month != 12),
                                 f"     +F2 drop top {100-q:2d}% absence "
                                 f"(>{thr:.1f})")

    print("\n" + "=" * 78)
    print("DRAWDOWN STRUCTURE — are the bad stretches even localisable?")
    print("=" * 78)
    pb = json.load(open(ROOT / "data" / "wf_perbet_D181.json"))
    b = sorted([x for x in pb["k=9 raw"] if x["season"] >= MODERN],
               key=lambda x: (x["date"], x["gid"]))
    ev = np.array([x["ev"] * x["keep"] for x in b])
    cum = np.cumsum(ev)
    dd = cum - np.maximum.accumulate(cum)
    print(f"  post-2018 equity: {len(b)} bets, final {cum[-1]:+.1f}u, "
          f"max DD {dd.min():+.1f}u")
    i = 0
    epi = []
    while i < len(dd):
        if dd[i] < -1e-9:
            j = i
            while j < len(dd) and dd[j] < -1e-9:
                j += 1
            epi.append((b[i]["date"], b[j - 1]["date"], float(dd[i:j].min()),
                        j - i))
            i = j
        else:
            i += 1
    epi = sorted([e for e in epi if e[2] < -5], key=lambda z: z[2])
    for s, e, d, n in epi:
        print(f"    {s} -> {e}  depth {d:+7.2f}u over {n:4d} bets")
    long2 = sum(n for *_, n in epi[:2])
    print(f"\n  the two deepest episodes span {long2} of {len(b)} bets "
          f"({100*long2/len(b):.0f}%).")
    print("  They are multi-SEASON underwater stretches, not excisable events —")
    print("  'remove the high-drawdown periods' would remove most of the sample.")

    json.dump(res, open(ROOT / "data" / "d185_post2018.json", "w"), indent=1)
    print("\nwrote data/d185_post2018.json")


if __name__ == "__main__":
    main()
