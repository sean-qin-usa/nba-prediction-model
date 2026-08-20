#!/usr/bin/env python3
"""Engine calibration battery v0 (handoff III.2): does the Monte Carlo engine
reproduce league marginals with league-average inputs? Run after any engine
change. This is how we know the possession LOOP is sound before real skills
replace the placeholder rates.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.engine.possession import LEAGUE, simulate_matchup, simulate_team

TARGETS = {  # (empirical target, tolerance)
    "pts_per_team": (116.0, 4.0),
    "3pa_share": (0.418, 0.02),
    "fta_fga": (0.291, 0.03),
    "ppp": (1.15, 0.05),
    "margin_sd": (13.5, 3.0),
}


def main():
    rng = np.random.default_rng(7)
    pts, fga, fg3a, fta = [], [], [], []
    for _ in range(4000):
        p, a, a3, ft = simulate_team(LEAGUE, rng)
        pts.append(p); fga.append(a); fg3a.append(a3); fta.append(ft)
    res = simulate_matchup(LEAGUE, LEAGUE, n=4000, seed=7)
    got = {
        "pts_per_team": float(np.mean(pts)),
        "3pa_share": float(np.sum(fg3a) / np.sum(fga)),
        "fta_fga": float(np.sum(fta) / np.sum(fga)),
        "ppp": float(np.mean(pts) / LEAGUE["pace"]),
        "margin_sd": res["margin_sd"],
    }
    print(f"{'metric':14} {'got':>8} {'target':>8} {'tol':>6}  status")
    for k, (tgt, tol) in TARGETS.items():
        ok = abs(got[k] - tgt) <= tol
        print(f"{k:14} {got[k]:8.3f} {tgt:8.3f} {tol:6.2f}  {'PASS' if ok else 'FAIL'}")
    print(f"\nhome win% {res['p_home_win']:.3f}  total {res['total_mean']:.1f}"
          f"  margin {res['margin_mean']:+.1f}")
    print("\nKnown v0 gaps (structural, fixed during real calibration, not by")
    print("tuning placeholder rates): margin_sd too high -> pace is Poisson")
    print("(over-dispersed); real pace is under-dispersed and the two teams'")
    print("paces are shared/correlated. Fix when the clock model (II.3.7) lands.")


if __name__ == "__main__":
    main()
