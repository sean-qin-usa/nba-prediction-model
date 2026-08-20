#!/usr/bin/env python3
"""D247c — TURN T6 INTO A BET, BECAUSE T6 CANNOT BE TRUE AS STATED.

T6 said: controlling for the team's current-season market price, prior-season
margin predicts the market residual at -0.1608 pts per point (t -9.46). Prior
margins span roughly -10 to +10, so that is a +-1.6 pt/game edge available from
two public numbers -- last season's point differential and tonight's closing
line. An edge that size, from information every bettor has, would be among the
largest inefficiencies ever documented in these markets.

The register's own D239 measured how easily this pipeline manufactures results:
best-of-N random subsets buy +2.54 ROI points from nothing. So the correct
response to a too-large coefficient is not to admire it but to try to spend it.

THE TEST. No fitting of any kind. A fixed, sign-only rule:

    delta_prev = prev_season_margin(home) - prev_season_margin(away)
    T6 implies  resid_home ~ -0.16 * delta_prev
    -> bet AWAY when delta_prev > 0, HOME when delta_prev < 0
       i.e. ALWAYS BACK THE TEAM THAT WAS WORSE LAST SEASON, against the CLOSE.

Score: ATS cover rate versus the closing line, which is the price actually
available. Break-even at -110 is 52.38%.

If T6 is a real, exploitable anchoring effect the implied cover rate is roughly
53.5-54%. If it is an artefact of team-season aggregation, the rule lands on
50% and T6 dies here.

Three further discriminations, all pre-specified:
  A. the same rule at the OPENING line -- an anchoring story predicts a LARGER
     edge at the open, since the close has had all day to absorb it;
  B. the extremes only (|delta_prev| in the top tercile), where the coefficient
     says the edge is concentrated;
  C. the team-season regression rerun at GAME level with game-level controls,
     to see whether the -0.16 survives leaving the aggregate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from scipy import stats                                           # noqa: E402


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def clus(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    k = len(v)
    if k < 2:
        return np.nan, np.nan, np.nan, np.nan, k
    se = v.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return v.mean(), v.mean() - tc * se, v.mean() + tc * se, v.mean() / se, k


def report(name, cover, seasons_arr, be=0.5238):
    """Season-clustered cover rate with the break-even comparison."""
    df = pd.DataFrame({"s": seasons_arr, "c": cover})
    per = df.groupby("s")["c"].mean()
    m, lo, hi, t, k = clus(per)
    edge = m - be
    print(f"  {name:44} {100*m:6.2f}%  CI [{100*lo:6.2f}%, {100*hi:6.2f}%]  "
          f"k={k}  n={len(df):,}")
    print(f"  {'':44} vs 50.00%: {100*(m-0.5):+5.2f}pp   "
          f"vs -110 break-even: {100*edge:+5.2f}pp   "
          f"{'PROFITABLE' if lo > be else ('ABOVE 50% but not tradable' if lo > 0.5 else 'NULL')}")
    return dict(cover=float(m), ci=[float(lo), float(hi)], k=int(k), n=len(df))


def main():
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
    f["game_id"] = zf(f["game_id"])
    f["game_date"] = pd.to_datetime(f["game_date"])
    f = f.dropna(subset=["margin_actual", "close_margin", "open_margin"]).copy()
    seasons = sorted(f.season.unique())
    idx = {s: i for i, s in enumerate(seasons)}
    out = {}

    # prior-season mean margin per team, from realised results only
    h = f.assign(team=f.home, act=f.margin_actual)
    a = f.assign(team=f.away, act=-f.margin_actual)
    L = pd.concat([h[["season", "team", "act"]], a[["season", "team", "act"]]],
                  ignore_index=True)
    ts = L.groupby(["season", "team"])["act"].mean().reset_index()
    ts["nxt"] = ts.season.map(lambda s: seasons[idx[s] + 1]
                              if idx[s] + 1 < len(seasons) else None)
    prev = ts.dropna(subset=["nxt"]).set_index(["nxt", "team"])["act"]

    f["prev_h"] = [prev.get((s, t), np.nan)
                   for s, t in zip(f.season, f.home)]
    f["prev_a"] = [prev.get((s, t), np.nan)
                   for s, t in zip(f.season, f.away)]
    g = f.dropna(subset=["prev_h", "prev_a"]).copy()
    g["dprev"] = g.prev_h - g.prev_a
    print(f"frame {len(g):,} games with both prior seasons "
          f"({g.season.nunique()} seasons)")
    print(f"E|delta_prev| = {g.dprev.abs().mean():.3f} pts   "
          f"sd {g.dprev.std():.3f}")
    print(f"T6 implies a mean |edge| of 0.1608 * {g.dprev.abs().mean():.3f} "
          f"= {0.1608*g.dprev.abs().mean():.3f} pts/game\n")

    # ---- the bet: always back the team that was worse last season -----
    side = np.where(g.dprev > 0, -1.0, 1.0)          # +1 = home, -1 = away
    for lab, line in (("CLOSING line", g.close_margin.to_numpy(float)),
                      ("OPENING line", g.open_margin.to_numpy(float))):
        cov = (np.sign(g.margin_actual.to_numpy(float) - line) == side)
        print(f"=== BACK LAST SEASON'S WORSE TEAM, {lab} ===")
        out[f"bet_{lab.split()[0]}"] = report(f"all games ({lab})",
                                              cov.astype(float),
                                              g.season.to_numpy())
        # extremes only
        q = g.dprev.abs().quantile(2 / 3)
        m3 = g.dprev.abs() >= q
        out[f"bet_{lab.split()[0]}_ext"] = report(
            f"top tercile |delta_prev| >= {q:.2f}",
            cov[m3.to_numpy()].astype(float), g.season.to_numpy()[m3.to_numpy()])
        print()

    # ---- realised residual, the direct version of the coefficient ----
    print("=== DIRECT: mean signed residual of the rule, vs the CLOSE ===")
    r = side * (g.margin_actual - g.close_margin).to_numpy(float)
    per = pd.DataFrame({"s": g.season, "r": r}).groupby("s")["r"].mean()
    m, lo, hi, t, k = clus(per)
    print(f"  realised {m:+.4f} pts/game  CI [{lo:+.4f}, {hi:+.4f}]  "
          f"t {t:+.2f}  k={k}")
    print(f"  T6 predicted {0.1608*g.dprev.abs().mean():+.4f} pts/game")
    ratio = m / (0.1608 * g.dprev.abs().mean())
    print(f"  realised / predicted = {ratio:.3f}")
    out["direct"] = dict(realised=float(m), ci=[float(lo), float(hi)],
                         predicted=float(0.1608 * g.dprev.abs().mean()),
                         ratio=float(ratio))

    # ---- C. the regression, at GAME level ----------------------------
    print("\n=== C. THE SAME REGRESSION AT GAME LEVEL ===")
    print("  resid_home ~ dprev + close_margin, clustered by season")
    per = []
    for s, gg in g.groupby("season"):
        y = (gg.margin_actual - gg.close_margin).to_numpy(float)
        X = np.column_stack([np.ones(len(gg)), gg.dprev.to_numpy(float),
                             gg.close_margin.to_numpy(float)])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        X2 = np.column_stack([np.ones(len(gg)), gg.dprev.to_numpy(float)])
        b2 = np.linalg.lstsq(X2, y, rcond=None)[0]
        per.append(dict(season=s, b_dprev_ctrl=float(b[1]),
                        b_close=float(b[2]), b_dprev_only=float(b2[1])))
    dp = pd.DataFrame(per)
    for c, lab in (("b_dprev_only", "dprev alone"),
                   ("b_dprev_ctrl", "dprev | close  (the T6 analogue)"),
                   ("b_close", "close_margin")):
        m, lo, hi, t, k = clus(dp[c])
        print(f"  {lab:34} {m:+.5f}  CI [{lo:+.5f}, {hi:+.5f}]  "
              f"t {t:+6.2f}  {'SIG' if (hi<0 or lo>0) else 'ns'}")
        out[f"game_{c}"] = dict(mean=float(m), ci=[float(lo), float(hi)])
    print("\n  If the game-level dprev|close coefficient is near zero while the")
    print("  team-season one was -0.161, the team-season result was an artefact")
    print("  of aggregating 82 games into one point per team.")

    json.dump(out, open(ROOT / "data" / "d247c_bet_it.json", "w"), default=float)
    print("\nwrote data/d247c_bet_it.json")


if __name__ == "__main__":
    main()
