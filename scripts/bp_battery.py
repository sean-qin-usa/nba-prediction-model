#!/usr/bin/env python3
"""BP-BATTERY — the full V3 battery (docs/GATE_POLICY_V2.md §8-§11) on every
rung of the BIGPLAYER information ladder.

Reads data/bp_ladder_pergame.csv (produced by scripts/bp_ladder.py) and scores
every pre-registered comparison: each rung's INCREMENT over the previous rung,
the full T5-T0 stack, the market gap at every rung, the MAE sensitivity, and
the PART C clairvoyant arms (reported separately and labelled).

Season-clustered CI is the reporting statistic.  ICC / design effects /
cluster-mean t at K-1 dof are reported for every comparison per §9.1, and §9.3's
small-K warning is live throughout (K = 3).

Read-only.  Out: data/bp_battery.json, data/logs/bp_battery.log
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

from nbapred.eval.splits import (Panel, full_report,              # noqa: E402
                                 paired_bootstrap)

CSV = REPO / "data" / "bp_ladder_pergame.csv"
OUT = REPO / "data" / "bp_battery.json"
B = 4000
SEED = 20260803
LN2 = float(np.log(2.0))


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ---- the ladder, in order ------------------------------------------------
BUYABLE = [
    ("T0", "p_T0", "availability-BLIND floor"),
    ("T1", "p_T1", "+ 5PM official injury report (injury_reports_pit)"),
    ("T2", "p_T2", "+ official inactive list T-30 (game_inactives)"),
    ("T3", "p_T3", "+ purchased minutes projections (MAE 4.0, SIMULATED)"),
    ("T4", "p_T4", "+ tracking from PRIOR season (D36/D72 on-ball)"),
    ("T5", "p_T5", "+ DARKO+EPM talent as of the prior day"),
]
CLAIR = [
    ("C1", "p_C1", "CLAIRVOYANT: T2 + who actually played (= D132 default)"),
    ("C2", "p_C2", "CLAIRVOYANT: C1 + realised minutes"),
    ("C3", "p_C3", "CLAIRVOYANT: C2 + 60-day CENTRED talent (D97)"),
    ("C4", "p_C4", "CLAIRVOYANT: T3 + SAME-season tracking (D72's own arm)"),
]


def gaps(df, col):
    """per-season and pooled (ll_us, ll_mkt, raw, normalized) for one column."""
    out = {}
    for s, g in df.groupby("season"):
        lu, lm = ll(g.y.values, g[col].values).mean(), ll(g.y.values, g.p_mkt.values).mean()
        out[s] = dict(n=int(len(g)), ll=round(float(lu), 5), ll_mkt=round(float(lm), 5),
                      raw=round(float(lu - lm), 5),
                      norm=round(100.0 * float(lu - lm) / (LN2 - lm), 2))
    lu = ll(df.y.values, df[col].values).mean()
    lm = ll(df.y.values, df.p_mkt.values).mean()
    out["POOLED"] = dict(n=int(len(df)), ll=round(float(lu), 5),
                         ll_mkt=round(float(lm), 5), raw=round(float(lu - lm), 5),
                         norm=round(100.0 * float(lu - lm) / (LN2 - lm), 2))
    return out


def compare(df, ctrl, treat, label):
    """Full V3 battery for ctrl -> treat.  POSITIVE delta = treat is better."""
    y = df.y.values
    p = Panel.from_logloss(df.season.values, y, df[ctrl].values, df[treat].values,
                           date=df.game_date.values, cluster=df.season.values,
                           label=label)
    rep = full_report(p, B=B, seed=SEED)
    rep["iid"] = paired_bootstrap(p.d, B, SEED, None)     # secondary, §9.1
    rep["pct_moved"] = float(np.mean(np.abs(df[treat].values - df[ctrl].values) > 1e-9))
    rep["max_abs_dp"] = float(np.max(np.abs(df[treat].values - df[ctrl].values)))
    return rep


def brief(rep):
    c = rep["clustering"]
    t = c["season_mean_t"]
    return dict(
        label=rep["label"], n=rep["n"],
        est=round(rep["pooled"]["est"], 6),
        iid_ci=[round(rep["iid"]["lo"], 6), round(rep["iid"]["hi"], 6)],
        clustered_ci=[round(rep["pooled"]["lo"], 6), round(rep["pooled"]["hi"], 6)],
        clustered_sig=bool(rep["pooled"]["sig"]),
        t_ci=[round(t["lo"], 6), round(t["hi"], 6)], t_sig=bool(t["sig"]),
        icc=round(c["icc_season"]["icc"], 6),
        deff_anova=round(c["icc_season"]["deff"], 3),
        deff_boot=round(c["design_effect_season"], 3),
        mde80=round(rep["pooled_mde80"], 6),
        per_season=[(f["season"], round(f["est"], 6)) for f in rep["per_season"]],
        ro_folds=[(f["test"], round(f["fold"]["est"], 6))
                  for f in rep["rolling_origin"]["folds"]],
        block_ci=[round(rep["block_bootstrap"]["lo"], 6),
                  round(rep["block_bootstrap"]["hi"], 6)],
        I2=round(rep["era"]["I2"], 4), p_Q=round(rep["era"]["p_heterogeneity"], 4),
        era_stable=bool(rep["era"]["era_stable"]),
        verdict=rep["verdict"]["tier"], flags=rep["verdict"].get("flags", []),
        ro_sign=rep["verdict"]["rolling_origin_sign"],
        pct_moved=round(rep["pct_moved"], 4), max_abs_dp=round(rep["max_abs_dp"], 5))


def main():
    df = pd.read_csv(CSV, dtype={"game_id": str})
    res = {"frame": dict(n=int(len(df)),
                         seasons={s: int(len(g)) for s, g in df.groupby("season")}),
           "gaps": {}, "increments": {}, "stack": {}, "clairvoyant": {},
           "sensitivity": {}}

    print("=" * 100)
    print("FRAME:", len(df), "games |", dict(df.groupby("season").size()))
    print("=" * 100)

    # ---------------- normalized gap at every rung -------------------------
    print("\n[1] LOG LOSS AND NORMALIZED GAP VS THE CLOSE, EVERY RUNG")
    print(f"{'tier':6} {'2023-24':>28} {'2024-25':>28} {'2025-26p':>28} {'POOLED':>28}")
    for tag, col, _ in BUYABLE + CLAIR:
        g = gaps(df, col)
        res["gaps"][tag] = g
        cells = "".join(f"{g[s]['ll']:.5f} {g[s]['raw']:+.5f} {g[s]['norm']:6.2f}%".rjust(29)
                        for s in ("2023-24", "2024-25", "2025-26", "POOLED"))
        print(f"{tag:6}{cells}")
    gm = gaps(df, "p_mkt")
    res["gaps"]["MARKET"] = gm
    print("market ll: " + " ".join(f"{s}={gm[s]['ll_mkt']:.5f}" for s in
                                   ("2023-24", "2024-25", "2025-26", "POOLED")))

    # ---------------- rung-by-rung increments ------------------------------
    print("\n[2] INCREMENTS — each rung OVER THE PREVIOUS RUNG (buyable stack)")
    for i in range(1, len(BUYABLE)):
        a, b = BUYABLE[i - 1], BUYABLE[i]
        rep = compare(df, a[1], b[1], f"{a[0]}->{b[0]}  {b[2]}")
        res["increments"][f"{a[0]}->{b[0]}"] = brief(rep)
        r = res["increments"][f"{a[0]}->{b[0]}"]
        print(f"  {r['label'][:60]:60} est {r['est']:+.5f}  "
              f"clust[{r['clustered_ci'][0]:+.5f},{r['clustered_ci'][1]:+.5f}] "
              f"{'SIG' if r['clustered_sig'] else 'ns '}  "
              f"t[{r['t_ci'][0]:+.5f},{r['t_ci'][1]:+.5f}] "
              f"{'SIG' if r['t_sig'] else 'ns '}  MDE80 {r['mde80']:.5f}  "
              f"moved {100 * r['pct_moved']:.1f}%")

    # ---------------- the whole stack --------------------------------------
    print("\n[3] THE STACK")
    for a, b in (("p_T0", "p_T5"), ("p_T0", "p_T2"), ("p_T2", "p_T5"),
                 ("p_T5", "p_C1"), ("p_T0", "p_C1")):
        rep = compare(df, a, b, f"{a[2:]}->{b[2:]}")
        res["stack"][f"{a[2:]}->{b[2:]}"] = brief(rep)
        r = res["stack"][f"{a[2:]}->{b[2:]}"]
        print(f"  {r['label']:14} est {r['est']:+.6f}  "
              f"clust[{r['clustered_ci'][0]:+.6f},{r['clustered_ci'][1]:+.6f}] "
              f"{'SIG' if r['clustered_sig'] else 'ns '}  "
              f"t[{r['t_ci'][0]:+.6f},{r['t_ci'][1]:+.6f}] "
              f"{'SIG' if r['t_sig'] else 'ns '}  ICC {r['icc']:+.5f} "
              f"DEFF {r['deff_anova']:.2f}/{r['deff_boot']:.2f}  "
              f"seasons {r['per_season']}")

    # ---------------- MAE sensitivity --------------------------------------
    print("\n[4] MINUTES-PROJECTION MAE SENSITIVITY (T3 and T5)")
    for tag, col in (("T3@MAE3.0", "p_T3_m3"), ("T3@MAE4.0", "p_T3"),
                     ("T3@MAE5.0", "p_T3_m5"), ("T5@MAE3.0", "p_T5_m3"),
                     ("T5@MAE4.0", "p_T5"), ("T5@MAE5.0", "p_T5_m5")):
        g = gaps(df, col)["POOLED"]
        base = "p_T2" if tag.startswith("T3") else "p_T4"
        rep = compare(df, base, col, tag)
        res["sensitivity"][tag] = dict(gap=g, inc=brief(rep))
        r = res["sensitivity"][tag]["inc"]
        print(f"  {tag:10} ll {g['ll']:.5f} raw {g['raw']:+.5f} norm {g['norm']:6.2f}%"
              f"   inc over {base[2:]}: {r['est']:+.5f} "
              f"clust[{r['clustered_ci'][0]:+.5f},{r['clustered_ci'][1]:+.5f}]")

    # ---------------- PART C ------------------------------------------------
    print("\n[5] PART C — CLAIRVOYANT, UNATTAINABLE, NOT PART OF THE BUYABLE STACK")
    for a, b, lab in (("p_T2", "p_C1", "T2->C1 perfect availability"),
                      ("p_C1", "p_C2", "C1->C2 perfect minutes"),
                      ("p_C2", "p_C3", "C2->C3 perfect talent (D97)"),
                      ("p_T3", "p_C4", "T3->C4 same-season tracking (D72)"),
                      ("p_T0", "p_C3", "T0->C3 the WHOLE clairvoyant stack")):
        rep = compare(df, a, b, lab)
        res["clairvoyant"][lab] = brief(rep)
        r = res["clairvoyant"][lab]
        print(f"  {lab:38} est {r['est']:+.6f}  "
              f"clust[{r['clustered_ci'][0]:+.6f},{r['clustered_ci'][1]:+.6f}] "
              f"{'SIG' if r['clustered_sig'] else 'ns '}  "
              f"t[{r['t_ci'][0]:+.6f},{r['t_ci'][1]:+.6f}] "
              f"{'SIG' if r['t_sig'] else 'ns '}")

    # ---------------- T4 parameter cost ------------------------------------
    n = len(df)
    cost = 1.0 / n
    inc4 = res["increments"]["T3->T4"]["est"]
    res["parameter_cost"] = dict(n=n, aic_charge_1param=round(cost, 6),
                                 T4_increment=round(inc4, 6),
                                 T4_survives=bool(inc4 > cost))
    print(f"\n[6] PARAMETER COST (D154).  T4 adds ONE parameter (k). "
          f"AIC-style charge at n={n} is {cost:.6f}; T4 increment {inc4:+.6f} -> "
          f"{'SURVIVES' if inc4 > cost else 'DOES NOT SURVIVE'} its own cost. "
          f"T1/T2/T3/T5 add ZERO parameters (existing slots).")

    # ---------------- pockets (D72 heavy-fav, D77 mid-band) ----------------
    print("\n[7] POCKETS — D72's heavy-favourite conditional and D77's mid band")
    res["pockets"] = {}
    conf = (df.p_mkt - 0.5).abs()
    for pname, mask in (("HEAVY-FAV |p_mkt-.5|>0.35", conf > 0.35),
                        ("MID-DIST |p_mkt-.5|<=0.35", conf <= 0.35)):
        sub = df[mask]
        res["pockets"][pname] = dict(n=int(len(sub)))
        for a, b, lab in (("p_T3", "p_T4", "T3->T4 prior-season tracking"),
                          ("p_T3", "p_C4", "T3->C4 same-season tracking"),
                          ("p_T0", "p_T5", "T0->T5 whole buyable stack"),
                          ("p_T0", "p_C3", "T0->C3 whole clairvoyant stack")):
            r = brief(compare(sub, a, b, lab))
            res["pockets"][pname][lab] = r
            print(f"  {pname:26} n={len(sub):5} {lab:32} est {r['est']:+.5f} "
                  f"clust[{r['clustered_ci'][0]:+.5f},{r['clustered_ci'][1]:+.5f}] "
                  f"{'SIG' if r['clustered_sig'] else 'ns '}")
        g5, g0, gm_ = gaps(sub, "p_T5")["POOLED"], gaps(sub, "p_T0")["POOLED"], None
        res["pockets"][pname]["gap_T0"] = g0
        res["pockets"][pname]["gap_T5"] = g5
        print(f"  {pname:26} gap vs close: T0 {g0['raw']:+.5f} ({g0['norm']:.2f}%)"
              f"  T5 {g5['raw']:+.5f} ({g5['norm']:.2f}%)")

    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
