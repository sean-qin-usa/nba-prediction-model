"""HA-(2b) SCHEDULE-ASYMMETRY COUNTERFACTUAL — the correct way to ask
"how much of the home edge is just that the visitor is more often tired?"

ha_decomp.py measures the schedule terms against a ZERO-schedule baseline
(nobody has travelled, nobody is on a b2b). That answers a different question.
The question Sean asked is a CONTRASTIVE one, so the counterfactual has to be:
give the AWAY team the HOME team's schedule state and see how much of the edge
survives.

    asymmetry_j = b_(away side, j) * ( E[X_away,j] - E[X_home,j] )

with the rest-differential term carried whole. Travel-km and |tz shift| are
0.77-0.86 correlated (see the correlation block printed below), so their
individual coefficients are not separately interpretable -- they are reported
as a JOINT block, which is the only identified quantity.

DESCRIPTIVE, FULL-SAMPLE.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ha_core import boot_ci, load_panel
from ha_decomp import SCHED, build, contributions, fit

SEED = 20260801
OUT = Path("/tmp/claude-1004/-hdd-steveqin-sean-dev/"
           "4912ce58-69cd-488d-a5f4-0d3ed2eed30c/scratchpad")
NORMAL = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

PAIRS = [("b2b", "h_b2b", "a_b2b"), ("3in4", "h_3in4", "a_3in4"),
         ("travel_km", "h_travel_k", "a_travel_k"),
         ("tz_shift", "h_tz_abs", "a_tz_abs")]


def asym(df, b):
    """points of home edge from the visitor's schedule being WORSE."""
    out = {}
    for name, hc, ac in PAIRS:
        out[name] = float(b[ac] * (df[ac].mean() - df[hc].mean()))
    out["rest_diff"] = float(b["rest_diff"] * df["rest_diff"].mean())
    return out


def main():
    rng = np.random.default_rng(SEED)
    d = load_panel()
    res = {}
    for label, sub in (("ALL 7 seasons", d),
                       ("NORMAL 5 seasons", d[d.season.isin(NORMAL)].reset_index(drop=True))):
        hfa, b, *_ = fit(sub)
        a = asym(sub, b)
        full = contributions(sub, b)
        home_side = sum(v for k, v in full.items() if k.startswith("h_"))
        away_side = sum(v for k, v in full.items() if k.startswith("a_"))
        raw = sub.margin.mean()
        # bootstrap
        B = 1500
        boots = {k: [] for k in list(a) + ["total_asym", "travel_block",
                                           "fatigue_block", "pure_hfa", "raw"]}
        for _ in range(B):
            s = sub.iloc[rng.integers(0, len(sub), len(sub))]
            try:
                h2, b2, *_ = fit(s)
            except Exception:
                continue
            a2 = asym(s, b2)
            for k, v in a2.items():
                boots[k].append(v)
            boots["total_asym"].append(sum(a2.values()))
            boots["travel_block"].append(a2["travel_km"] + a2["tz_shift"])
            boots["fatigue_block"].append(a2["b2b"] + a2["3in4"] + a2["rest_diff"])
            boots["pure_hfa"].append(h2.get("normal", np.nan))
            boots["raw"].append(s.margin.mean())
        ci = {k: boot_ci(np.array(v)) for k, v in boots.items() if len(v) > 50}

        print(f"\n=== SCHEDULE ASYMMETRY DECOMPOSITION — {label} ===")
        print(f"  raw home edge = {raw:+.4f} pts")
        print(f"  {'component':16s} {'pts':>8s}  {'95% CI':>20s}   "
              f"{'away-minus-home X':>18s}")
        for name, hc, ac in PAIRS:
            lo, hi = ci[name]
            print(f"  {name:16s} {a[name]:>+8.4f}  ({lo:+.4f},{hi:+.4f})   "
                  f"{sub[ac].mean()-sub[hc].mean():>+18.4f}")
        lo, hi = ci["rest_diff"]
        print(f"  {'rest differential':16s} {a['rest_diff']:>+8.4f}  "
              f"({lo:+.4f},{hi:+.4f})   {sub['rest_diff'].mean():>+18.4f}")
        for k, nm in (("fatigue_block", "FATIGUE block (b2b+3in4+rest)"),
                      ("travel_block", "TRAVEL block (km + tz, joint)"),
                      ("total_asym", "TOTAL schedule asymmetry")):
            lo, hi = ci[k]
            v = (a["b2b"] + a["3in4"] + a["rest_diff"] if k == "fatigue_block"
                 else a["travel_km"] + a["tz_shift"] if k == "travel_block"
                 else sum(a.values()))
            print(f"  {nm:34s} {v:>+8.4f}  ({lo:+.4f},{hi:+.4f})  "
                  f"{'SIG' if lo>0 or hi<0 else 'NS'}  = "
                  f"{100*v/raw:5.1f}% of the raw home edge")
        print(f"  [home team's OWN schedule burden vs a rested baseline: "
              f"{home_side:+.4f}; visitor's: {away_side:+.4f}]")
        res[label] = dict(raw=float(raw), asym=a, ci=ci, home_side=float(home_side),
                          away_side=float(away_side), coef=b, hfa=hfa)
    (OUT / "ha_decomp2.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {OUT/'ha_decomp2.json'}")


if __name__ == "__main__":
    main()
