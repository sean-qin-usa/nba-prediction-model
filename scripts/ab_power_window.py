"""ABSENCE AUDIT — §5.5 power arithmetic for candidate ARM-R designs and
effect-concentration windows. Uses ONLY the control probabilities p (never y).

MDE80 = 2.802 * sd / sqrt(n) on the paired per-game log-loss delta, with sd
computed as the exact outcome-averaged sd under the control's own p (so no
outcome is read):
    d1 = -log(p'/p), d0 = -log((1-p')/(1-p)), m = p*d1+(1-p)*d0
    Var = E[p*d1^2+(1-p)*d0^2 - m^2] + Var(m)
best_case_dLogLoss = 0.5 * E[d^2 * p(1-p)] restricted to the same window.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

SCALE = 7.2


def stats(dm, p, mask):
    dm, p = dm[mask], p[mask]
    n = len(dm)
    if n < 30:
        return None
    d = dm / SCALE
    lg = np.log(p / (1 - p)) + d
    pn = 1.0 / (1.0 + np.exp(-lg))
    d1 = -np.log(pn / p)
    d0 = -np.log((1 - pn) / (1 - p))
    m = p * d1 + (1 - p) * d0
    v = p * d1 ** 2 + (1 - p) * d0 ** 2 - m ** 2
    sd = float(np.sqrt(v.mean() + m.var()))
    best = float(0.5 * np.mean(d ** 2 * p * (1 - p)))
    mde = 2.802 * sd / np.sqrt(n)
    return {"n": int(n), "rms_dmargin": round(float(np.sqrt((dm ** 2).mean())), 4),
            "sd": round(sd, 5), "MDE80": round(mde, 5),
            "best_case": round(best, 6), "best_over_MDE80": round(best / mde, 3)}


def main():
    caps = pd.read_csv("data/capstone_pergame_d132.csv", dtype={"game_id": str})
    p = caps.p_us.to_numpy(float)
    out = {}
    for N in (14, 18, 21, 25, 30, 45, 9999):
        dm = np.load(f"data/ab_dmargin_R{N}.npy")
        rec = {}
        rec["pooled"] = stats(dm, p, np.ones(len(dm), bool))
        for thr in (0.0, 0.25, 0.5, 1.0, 1.5):
            rec[f"|dmargin|>{thr}"] = stats(dm, p, np.abs(dm) > thr)
        # mid-distribution window (D77 precedent), intersected with moved games
        mid = (np.abs(caps.p_mkt.to_numpy(float) - 0.5) <= 0.35)
        rec["moved & mid|pmkt-.5|<=.35"] = stats(dm, p, (np.abs(dm) > 1e-9) & mid)
        out[f"R{N}"] = rec
        print(f"\n=== ARM R, ROSTER_DAYS = {N} ===")
        for k, v in rec.items():
            print(f"  {k:28s} {v}")
    json.dump(out, open("data/ab_power_window.json", "w"), indent=1)
    print("\nwrote data/ab_power_window.json")


if __name__ == "__main__":
    main()
