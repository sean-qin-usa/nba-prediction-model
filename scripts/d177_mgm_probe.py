#!/usr/bin/env python3
"""D177 — ingest the Kaggle `caseydurfee/mgm-grand-nba-betting-data` BetMGM
closing spread and use it as a CALIBRATED ONE-BOOK PROBE of the 2019-20..2022-23
per-book hole.

READ THIS FIRST, BECAUSE IT IS THE POINT OF THE ENTRY:
**THIS IS NOT A PANEL.**  The file carries exactly ONE operator (BetMGM, via
Yahoo's internal JSON API).  A best-of-k line-shopping ladder CANNOT be computed
from one book, and this script never pretends otherwise.  What one book DOES
give, when the other side of the comparison is a consensus, is an estimate of
the cross-book scale parameter sigma - and sigma is what the ladder is built
from.  The estimator is FITTED on the seasons where both quantities are
MEASURED, and only then applied to the hole.  Every output cell is labelled.

Phases:
  1  ingest + team-name audit (nbapred.teams, D171 - REPORTS, never drops)
  2  pair-join to odds_market (D174's +/-1 day tolerance), match rate both ways
  3  validation vs odds_market close and odds_open open/close
  4  same-operator control vs data/bkp_panel_rows.csv.gz's own `mgm` rows
     (D174 s3's method: one operator through two independent scrapers)
  5  the calibrated sigma bridge, fitted on MEASURED seasons, applied to the hole

DB is READ-ONLY throughout (read_only=True, retry_s=60); ZERO writes.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                      # noqa: E402
nbapred.threads.pin(1)                      # BEFORE numpy

import collections                          # noqa: E402
import json                                 # noqa: E402

import numpy as np                          # noqa: E402
import pandas as pd                         # noqa: E402

from nbapred.db import connect              # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from nbapred.teams import abbrev_for, modern, resolve_map   # noqa: E402

MGM_CSV = (ROOT / "data" / "raw" / "kaggle" /
           "caseydurfee__mgm-grand-nba-betting-data" / "all_odds.csv")
PANEL = ROOT / "data" / "bkp_panel_rows.csv.gz"
OUT_JSON = ROOT / "data" / "d177_mgm_probe.json"
OUT_ROWS = ROOT / "data" / "d177_mgm_rows.csv.gz"

HOLE = ["2019-20", "2020-21", "2021-22", "2022-23"]
OUT: dict = {}


def sea(y: int) -> str:
    return f"{y - 1}-{str(y)[-2:]}"


# ---------------------------------------------------------------- 1. ingest
def load_mgm() -> pd.DataFrame:
    d = pd.read_csv(MGM_CSV)
    d["game_date"] = pd.to_datetime(d.game_date.str.slice(0, 10)).dt.date
    d = d[d.spread_home_points.notna()].copy()

    names = sorted(set(d.home_team) | set(d.away_team))
    ok, bad = resolve_map(names)
    cnt = collections.Counter()
    for c in ("home_team", "away_team"):
        cnt.update(d[c].tolist())
    OUT["name_audit"] = {
        "distinct_strings": len(names),
        "resolved": len(ok),
        "unresolved": {n: int(cnt[n]) for n in bad},
        "map": ok,
    }
    print(f"[1] team strings: {len(names)} distinct, {len(ok)} resolved, "
          f"{len(bad)} UNRESOLVED {bad}")
    if bad:
        for n in bad:
            print(f"       UNRESOLVED {n!r}: {cnt[n]} rows")

    d["home_abbr"] = d.home_team.map(lambda x: modern(abbrev_for(x)))
    d["away_abbr"] = d.away_team.map(lambda x: modern(abbrev_for(x)))
    d = d[d.home_abbr.notna() & d.away_abbr.notna()].copy()
    # feed handicap is ON THE HOME TEAM (negative = home favored) -> margin
    d["mgm_margin_raw"] = -d.spread_home_points.astype(float)
    return d


# ------------------------------------------------------- 2. join to spine
def pair_join(d: pd.DataFrame, con) -> pd.DataFrame:
    """D174's rule: unordered pair + date with +/-1 day tolerance. The tolerance
    is load-bearing whenever the two feeds disagree about the timezone of a
    late tip-off; the effect is MEASURED below rather than assumed."""
    mk = con.execute("""SELECT season_end, game_date, home, away, score_home,
                               score_away, home_exp_margin
                        FROM odds_market""").df()
    mk["game_date"] = pd.to_datetime(mk.game_date).dt.date
    mk["pair"] = [frozenset(p) for p in zip(mk.home, mk.away)]
    d = d.copy()
    d["pair"] = [frozenset(p) for p in zip(d.home_abbr, d.away_abbr)]

    hits = {}
    for off in (0, -1, 1):                        # exact first, then tolerance
        cur = d[~d.index.isin(hits)].copy()
        if not len(cur):
            break
        cur["jd"] = cur.game_date + pd.Timedelta(days=off)
        cur["jd"] = cur["jd"].map(lambda x: x.date() if hasattr(x, "date") else x)
        m = cur.reset_index().merge(mk, left_on=["jd", "pair"],
                                    right_on=["game_date", "pair"],
                                    how="inner", suffixes=("", "_mk"))
        for _, r in m.iterrows():
            hits.setdefault(int(r["index"]), (off, r))
        print(f"[2] offset {off:+d}d: cumulative {len(hits)}/{len(d)} matched")
    OUT["join"] = {"mgm_rows": int(len(d)), "matched": len(hits),
                   "by_offset": dict(collections.Counter(v[0] for v in hits.values()))}

    rows = []
    for i, (off, r) in hits.items():
        flip = d.at[i, "home_abbr"] != r["home"]
        mm = d.at[i, "mgm_margin_raw"]
        rows.append({
            "season_end": int(r["season_end"]), "season": sea(int(r["season_end"])),
            "game_date": r["game_date"], "home": r["home"], "away": r["away"],
            "mgm_margin": -mm if flip else mm,
            "mkt_margin": r["home_exp_margin"],
            "score_home": r["score_home"], "score_away": r["score_away"],
            "join_offset": off,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------ 3. validation
def validate(p: pd.DataFrame, con) -> None:
    op = con.execute("""SELECT season_end, game_date, home, away, open_margin,
                               close_margin, source FROM odds_open""").df()
    op["game_date"] = pd.to_datetime(op.game_date).dt.date
    j = p.merge(op, on=["season_end", "game_date", "home", "away"], how="left")

    rep = []
    for s, g in j.groupby("season"):
        gm = g.dropna(subset=["mkt_margin"])
        go = g.dropna(subset=["open_margin"])
        gc = g.dropna(subset=["close_margin"])
        rep.append({
            "season": s, "n": int(len(g)),
            "n_mkt": int(len(gm)),
            "corr_mkt": float(np.corrcoef(gm.mgm_margin, gm.mkt_margin)[0, 1]) if len(gm) > 2 else None,
            "mad_mkt": float((gm.mgm_margin - gm.mkt_margin).abs().mean()) if len(gm) else None,
            "bias_mkt": float((gm.mgm_margin - gm.mkt_margin).mean()) if len(gm) else None,
            "tie_mkt": float((gm.mgm_margin == gm.mkt_margin).mean()) if len(gm) else None,
            "n_open": int(len(go)),
            "mad_open": float((go.mgm_margin - go.open_margin).abs().mean()) if len(go) else None,
            "tie_open": float((go.mgm_margin == go.open_margin).mean()) if len(go) else None,
            "n_close": int(len(gc)),
            "mad_close": float((gc.mgm_margin - gc.close_margin).abs().mean()) if len(gc) else None,
            "tie_close": float((gc.mgm_margin == gc.close_margin).mean()) if len(gc) else None,
        })
    OUT["validate"] = rep
    print("\n[3] MGM vs our consensus prices (mkt = odds_market CLOSE)")
    print(pd.DataFrame(rep).to_string(index=False))


# ------------------------------------- 4. same-operator control vs the panel
def same_operator(p: pd.DataFrame) -> None:
    pan = pd.read_csv(PANEL)
    pan["game_date"] = pd.to_datetime(pan.game_date).dt.date
    mg = pan[(pan.operator == "mgm")].copy()
    rep = []
    for (s, ph), g in mg.groupby(["season", "phase"]):
        g = g.groupby(["season", "game_date", "home", "away"], as_index=False).home_margin.first()
        j = p.merge(g, on=["season", "game_date", "home", "away"], how="inner")
        if len(j) < 20:
            continue
        rep.append({"season": s, "phase": ph, "n": int(len(j)),
                    "tie_pct": round(100 * float((j.mgm_margin == j.home_margin).mean()), 2),
                    "mad": round(float((j.mgm_margin - j.home_margin).abs().mean()), 4)})
    OUT["same_operator"] = rep
    print("\n[4] SAME-OPERATOR CONTROL - Kaggle/Yahoo BetMGM vs the panel's own `mgm`")
    print(pd.DataFrame(rep).to_string(index=False))


# ------------------------------------------------- 5. calibrated sigma bridge
# D174's OWN ladder machinery is imported, not reimplemented, so every number
# below is comparable to D163/D174 cell for cell.
from bkp_ladder import (DCOVER_PER_PT, KS, clustered, ladder_vals,  # noqa: E402
                        load as load_panel)

TRIM = 1.5      # D163's outlier-haircut threshold, applied to m1 as well


def m1_of(diffs) -> dict:
    """The one-book scale statistic. `trim` mirrors D163's ladder haircut
    (drop a quote more than 1.5 pts off) so the numerator and denominator of
    the ratio are haircut on the SAME rule; `mean` is kept as the sensitivity."""
    d = np.asarray(diffs, dtype=float)
    return {"mean": float(d.mean()),
            "trim": float(d[d <= TRIM].mean()) if (d <= TRIM).any() else float("nan")}


def bridge(p: pd.DataFrame) -> dict:
    """On MEASURED seasons relate m1 = mean|one book - consensus close| (the ONE
    statistic a single-book season can supply) to the MEASURED shopping ladder,
    then read the hole seasons off that fit. Raw AND D163 outlier-haircut."""
    pan = load_panel()
    con = connect(read_only=True, retry_s=60)
    mk = con.execute("SELECT game_date, home, away, home_exp_margin "
                     "FROM odds_market").df()
    con.close()
    mk["game_date"] = mk.game_date.astype(str)
    cons = {(r.game_date, r.home, r.away): r.home_exp_margin
            for r in mk.itertuples()}

    rows = []
    for s in sorted({k[0] for k in pan}):
        sub = {k: v for k, v in pan.items()
               if k[0] == s and k[1] == "close" and len(v) >= 2}
        if not sub:
            continue
        diffs = [abs(d[o]["home_margin"] - cons[(k[2], k[3], k[4])])
                 for k, d in sub.items() if (k[2], k[3], k[4]) in cons for o in d]
        if len(diffs) < 100:
            continue
        m1 = m1_of(diffs)
        row = {"season": s, "n_games": len(sub),
               "n_books": len({o for d in sub.values() for o in d}),
               "m1_mean": m1["mean"], "m1_trim": m1["trim"]}
        for hc, tag in ((None, ""), ("outlier", "_hc")):
            per_k, _ = ladder_vals(sub, haircut=hc)
            for k in KS:
                if k == 1:
                    continue
                row[f"k{k}{tag}"] = float(np.mean(per_k[k]))
        rows.append(row)
    cal = pd.DataFrame(rows).sort_values("season")
    OUT["calibration"] = cal.to_dict("records")
    print("\n[5a] CALIBRATION on MEASURED seasons (CLOSE phase) — every raw "
          "cell beside its D163 outlier-haircut twin")
    print(cal.round(4).to_string(index=False))

    kcols = [c for c in cal.columns if c.startswith("k")]
    out = {}
    for base in ("m1_trim", "m1_mean"):
        for c in kcols:
            r = (cal[c] / cal[base]).values
            m, lo, hi, K = clustered(r, cal.season.tolist())
            out[f"{c}|{base}"] = {"ratio": m, "ci": [lo, hi], "K": K,
                                  "sd": float(np.std(r, ddof=1)),
                                  "cv": float(np.std(r, ddof=1)) / m}
    OUT["ratios"] = out
    print("\n[5b] ratio (MEASURED ladder_k)/(one-book m1), season-clustered K=10."
          "\n     m1_trim is PRIMARY: same 1.5pt haircut rule as the ladder it "
          "predicts.")
    for c in kcols:
        a, b = out[f"{c}|m1_trim"], out[f"{c}|m1_mean"]
        print(f"   {c:8s} trim {a['ratio']:.4f} (cv {a['cv']:.3f}) "
              f"CI [{a['ci'][0]:.4f},{a['ci'][1]:.4f}]   |   "
              f"mean {b['ratio']:.4f} (cv {b['cv']:.3f})")

    print("\n[5c] APPLIED — 2021-22/2022-23 do NOT become MEASURED. The "
          "extrapolation becomes ANCHORED to a real book inside the season.")
    ap = []
    for s, g in p.groupby("season"):
        g = g.dropna(subset=["mkt_margin"])
        if not len(g):
            continue
        m1 = m1_of((g.mgm_margin - g.mkt_margin).abs().values)
        row = {"season": s, "n": int(len(g)), "m1_trim": m1["trim"],
               "m1_mean": m1["mean"],
               "label": "ANCHORED-EXTRAP" if s in HOLE else "control"}
        for c in kcols:
            v = out[f"{c}|m1_trim"]
            row[c] = m1["trim"] * v["ratio"]
            row[c + "_lo"] = m1["trim"] * v["ci"][0]
            row[c + "_hi"] = m1["trim"] * v["ci"][1]
        ap.append(row)
    apd = pd.DataFrame(ap)
    OUT["applied"] = apd.to_dict("records")
    print(apd[["season", "n", "label", "m1_trim", "k2", "k2_hc", "k3_hc",
               "k5", "k5_hc", "k5_hc_lo", "k5_hc_hi", "k8_hc"]]
          .round(4).to_string(index=False))

    print("\n[5d] HARNESS CHECK — the anchored estimate against the MEASURED "
          "ladder on the three seasons where BOTH exist. THIS IS THE TEST THAT "
          "DECIDES WHETHER THE BRIDGE MAY BE QUOTED AT ALL.")
    chk = []
    for _, r in apd.iterrows():
        m = cal[cal.season == r.season]
        if not len(m):
            continue
        chk.append({"season": r.season,
                    "k5_hc_MEASURED": float(m.k5_hc.iloc[0]),
                    "k5_hc_ANCHORED": float(r.k5_hc),
                    "ratio": float(r.k5_hc) / float(m.k5_hc.iloc[0])})
    OUT["harness"] = chk
    ck = pd.DataFrame(chk)
    print(ck.round(4).to_string(index=False))
    err = float(np.max(np.abs(ck.ratio - 1.0))) if len(ck) else float("nan")
    OUT["harness_max_err"] = err
    print(f"   WORST CONTROL-SEASON ERROR: {100*err:.1f}%  — every anchored "
          f"cell must be read with that band in front of it.")
    return {r["season"]: r for r in OUT["applied"]}


# ------------------------------------ 6. SENSITIVITY (not a re-price) on D174
def sensitivity(app: dict) -> None:
    """D174 §12's path, with 2021-22/2022-23's applied gain swapped for the
    ANCHORED estimate. This is a SENSITIVITY, NOT a re-price: neither season
    becomes MEASURED, so the headline path does NOT move and the official
    record stays D174's +3.13% / +48.6u on 9-of-14 MEASURED."""
    D174 = json.loads((ROOT / "data" / "bkp_ladder.json").read_text())["repriced"]
    CONV = DCOVER_PER_PT * 1.084

    def roi(c):
        return (100.0 / 110.0) * c - (1.0 - c)

    rows, cum_new, cum_sens, N = [], 0.0, 0.0, 0
    for r in D174["rows"]:
        s, n = r["season"], r["n"]
        c1 = r["cover_d166"] - 100.0 * r["g_d166"] * CONV        # one-book base
        g_sens = r["g_meas"]
        lab = r["label"]
        if s in ("2021-22", "2022-23") and s in app:
            g_sens = app[s]["k5_hc"]
            lab = "ANCHORED-EXTRAP"
        c_sens = c1 + 100.0 * g_sens * CONV
        r_new, r_s = r["roi_new"], 100 * roi(c_sens / 100.0)
        cum_new += n * r_new / 100.0
        cum_sens += n * r_s / 100.0
        N += n
        rows.append({"season": s, "n": n, "g_D174": r["g_meas"], "g_sens": g_sens,
                     "roi_D174": r_new, "roi_sens": r_s, "label": lab})
    d = pd.DataFrame(rows)
    print("\n[6] SENSITIVITY ONLY — D174's path with 2021-22/2022-23 anchored")
    print(d.round(4).to_string(index=False))
    mo, lo_o, hi_o, K = clustered(d.roi_D174.tolist(), d.season.tolist())
    ms, lo_s, hi_s, _ = clustered(d.roi_sens.tolist(), d.season.tolist())
    print(f"\n  POOLED  D174 (OFFICIAL) {100*cum_new/N:+.2f}% ({cum_new:+.1f}u)"
          f"   sensitivity {100*cum_sens/N:+.2f}% ({cum_sens:+.1f}u)")
    print(f"  season-clustered K={K}  D174 {mo:+.2f}% [{lo_o:+.2f},{hi_o:+.2f}]"
          f"   sens {ms:+.2f}% [{lo_s:+.2f},{hi_s:+.2f}]")
    OUT["sensitivity"] = {"rows": rows, "pooled_D174": 100 * cum_new / N,
                          "pooled_sens": 100 * cum_sens / N,
                          "units_D174": cum_new, "units_sens": cum_sens,
                          "clustered_D174": [mo, lo_o, hi_o],
                          "clustered_sens": [ms, lo_s, hi_s], "K": K}


def main() -> None:
    con = connect(read_only=True, retry_s=60)
    d = load_mgm()
    p = pair_join(d, con)
    print(f"\n[2] matched {len(p)} game-rows; seasons "
          f"{sorted(p.season.unique())}")
    OUT["by_season"] = {s: int(n) for s, n in p.season.value_counts().items()}
    validate(p, con)
    con.close()
    same_operator(p)
    app = bridge(p)
    sensitivity(app)
    p.to_csv(OUT_ROWS, index=False, compression="gzip")
    OUT_JSON.write_text(json.dumps(OUT, indent=1, default=str))
    print(f"\nwrote {OUT_ROWS} and {OUT_JSON}")


if __name__ == "__main__":
    main()
