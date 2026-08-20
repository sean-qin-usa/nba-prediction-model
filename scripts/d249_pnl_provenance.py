#!/usr/bin/env python3
"""D249 — HOW MUCH OF THE HEADLINE P&L IS OBSERVED, AND HOW MUCH IS MODELLED?

D248 showed the odds feed has three recording regimes and that 2024-25's opens
come from a single book posting 100% half-point spreads. That raises a question
the register has partly answered and never quantified end-to-end: is the
reported +16.62% / 460 bets an observed result?

WHAT THE TAG MEANS (scripts/wf_equity.py, `era_of`). It is a SEASON-level label
for which multi-book PANEL priced the shopping tier, not a claim about the base
price. So:

  * k=1 is a REAL OBSERVED PRICE in every season. The single-book ladder is not
    extrapolated in any season, whatever the tag says.
  * k>1 uplift is measured only where that season's panel exists.

WHY k=9 CANNOT BE MEASURED IN 2024-25 AT ALL. D174 counted the panels: ESPN
providers collapse 16 -> 2 -> 4, and 2024-25 is ESPN BET plus its own in-game
feed and nothing else. D174's own fixed-basket ladder therefore stops at k=3 --
only three books span the seasons. Best-of-NINE is not merely unmeasured in
2024-25; there are not nine books to measure.

THE PUSH ASYMMETRY, WHICH IS NEW HERE. A half-point line cannot push. The
ledger shows 3 pushes in 2023-24 and ZERO in 2024-25 and 2025-26. Any shopping
model that converts a line improvement into a change in cover probability gets
a free boost on a grid where no improvement is ever absorbed by a push.

This script does not re-derive the shopping law. It reports what is observed,
what is modelled, and what the headline becomes under stated alternatives, so a
reader can see the load each assumption carries.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from scipy import stats                                           # noqa: E402

BLOCK = ("2023-24", "2024-25", "2025-26")


def clus(v):
    v = np.asarray(v, float); k = len(v)
    se = v.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return v.mean(), v.mean() - tc * se, v.mean() + tc * se, k


def main():
    pb = json.load(open(ROOT / "data" / "wf_perbet_OFFSET.json"))
    k1 = pd.DataFrame(pb["k=1 raw"]); k9 = pd.DataFrame(pb["k=9 raw"])
    k1, k9 = k1[k1.keep > 0], k9[k9.keep > 0]
    g = pd.DataFrame({
        "n": k1.groupby("season").size(),
        "tag": k1.groupby("season").tag.first(),
        "roi_k1": 100 * k1.groupby("season").ev.mean(),
        "roi_k9": 100 * k9.groupby("season").ev.mean(),
    })
    g["uplift"] = g.roi_k9 - g.roi_k1
    g["pushes_k1"] = k1.assign(p=k1.ev.abs() < 1e-12).groupby("season").p.sum()
    print("=" * 78)
    print("PER-SEASON: OBSERVED SINGLE BOOK vs MODELLED BEST-OF-NINE")
    print("=" * 78)
    print(g.to_string(float_format=lambda v: f"{v:8.2f}"))

    meas = g[g.tag == "MEASURED"]
    extr = g[g.tag == "EXTRAPOLATED"]
    print(f"\n  shopping uplift where the panel EXISTS  ({len(meas)} seasons): "
          f"mean {meas.uplift.mean():+.2f} pts  "
          f"[{meas.uplift.min():+.2f}, {meas.uplift.max():+.2f}]")
    print(f"  shopping uplift where it is MODELLED    ({len(extr)} seasons): "
          f"mean {extr.uplift.mean():+.2f} pts  "
          f"[{extr.uplift.min():+.2f}, {extr.uplift.max():+.2f}]")
    for s in ("2024-25", "2025-26"):
        if s in g.index:
            print(f"    {s}: {g.uplift[s]:+.2f} pts = "
                  f"{g.uplift[s]/meas.uplift.mean():.1f}x the measured mean")

    # ---------------- the headline block, three ways --------------------
    print("\n" + "=" * 78)
    print("THE 460-BET HEADLINE UNDER THREE PRICING ASSUMPTIONS")
    print("=" * 78)
    b = g.loc[list(BLOCK)]
    up_meas = meas.uplift.mean()
    scen = {
        "AS REPORTED (modelled k=9)":
            b.roi_k9.to_dict(),
        "k=9 in 2023-24 (measured panel), "
        "historical mean uplift elsewhere":
            {s: (b.roi_k9[s] if b.tag[s] == "MEASURED"
                 else b.roi_k1[s] + up_meas) for s in BLOCK},
        "SINGLE OBSERVED BOOK ONLY (k=1)":
            b.roi_k1.to_dict(),
    }
    rows = []
    for lab, per in scen.items():
        pnl = sum(b.n[s] * per[s] / 100 for s in BLOCK)
        roi = 100 * pnl / b.n.sum()
        m, lo, hi, k = clus([per[s] for s in BLOCK])
        print(f"\n  {lab}")
        print(f"    per season: " + "  ".join(f"{s} {per[s]:+6.2f}%"
                                              for s in BLOCK))
        print(f"    P&L {pnl:+7.2f}u on {int(b.n.sum())} bets   "
              f"ROI {roi:+6.2f}%   season-clustered CI "
              f"[{lo:+.2f}%, {hi:+.2f}%]")
        rows.append(dict(scenario=lab, pnl=float(pnl), roi=float(roi),
                         ci=[float(lo), float(hi)]))

    print("\n" + "-" * 78)
    print("  Every interval spans zero on three seasons, as REVIEW.md already")
    print("  states. The point here is the SPREAD BETWEEN SCENARIOS: the")
    print(f"  headline moves {rows[0]['roi'] - rows[2]['roi']:.2f} ROI points")
    print("  between 'one book we can see' and 'nine books we model'.")

    print("\n" + "=" * 78)
    print("PUSH ASYMMETRY — a half-point line cannot push")
    print("=" * 78)
    print(g[["n", "pushes_k1", "tag"]].to_string())
    print("\n  Zero pushes in 2024-25 and 2025-26 is the granularity finding of")
    print("  D248 visible directly in the ledger, and it flatters any shopping")
    print("  model that turns line improvements into cover-probability changes.")

    json.dump({"per_season": g.reset_index().to_dict("records"),
               "scenarios": rows}, open(ROOT / "data" / "d249_pnl.json", "w"),
              default=float)
    print("\nwrote data/d249_pnl.json")


if __name__ == "__main__":
    main()
