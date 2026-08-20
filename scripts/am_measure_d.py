"""AM MEASUREMENT M-D / M-E — artifact-free persistence + implied half-life.

M-D  PAST-vs-FUTURE persistence by phase, no SRS residualisation (so no
     sum-to-zero artifact): for each team-season and cut point t,
     A = mean margin over games [t-10, t), B = mean margin over [t, t+10).
     corr(A, B) across all team-seasons, by phase of t.  Falling corr late =
     strength is LESS persistent late = more drift = shorter memory is better.

M-E  IMPLIED HALF-LIFE from M-B's fitted per-game weight ratio.  Under a
     CONSTANT exponential half-life h, the ratio of the mean weight of the
     last 5 games to the mean weight of the older gp-5 games is a rising
     function of gp all by itself.  Solve for the h that reproduces the
     OBSERVED ratio at each phase.  If the implied h is flat, the rising
     ratio is mechanical, not a phase-varying learning rate.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from nbapred.db import connect
from am_measure import load_games  # noqa

SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")


def implied_h(ratio, gp, w=5):
    """Solve mean_{a<w} 0.5^(a/h) / mean_{w<=a<gp} 0.5^(a/h) = ratio for h."""
    def f(h):
        a1 = np.arange(0, w)
        a2 = np.arange(w, int(round(gp)))
        if len(a2) == 0:
            return np.nan
        return (0.5 ** (a1 / h)).mean() / (0.5 ** (a2 / h)).mean()
    lo, hi = 0.5, 5000.0
    if f(hi) > ratio:      # even infinite memory gives a bigger ratio
        return float("inf") if ratio < f(hi) else hi
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if f(mid) > ratio:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def main():
    con = connect(read_only=True)
    games = load_games(con)
    seq = {}
    for g in games:
        s = g["season"]
        ht, hp = g["h"]; at, ap = g["a"]
        seq.setdefault((s, ht), []).append(hp - ap)
        seq.setdefault((s, at), []).append(ap - hp)
    W = 10
    out = {"M_D_past_future_persistence": {}, "window_games": W}
    for lo, hi in [(10, 20), (20, 30), (30, 41), (41, 55), (55, 72)]:
        A, B = [], []
        for k, ms in seq.items():
            for t in range(max(lo, W), min(hi, len(ms) - W)):
                A.append(float(np.mean(ms[t - W:t])))
                B.append(float(np.mean(ms[t:t + W])))
        if len(A) > 100:
            r = float(np.corrcoef(A, B)[0, 1])
            # season-clustered-ish SE via Fisher z on the number of INDEPENDENT
            # team-seasons contributing (not the overlapping cut points)
            n_ts = len(seq)
            se = 1.0 / np.sqrt(max(n_ts - 3, 1))
            out["M_D_past_future_persistence"][f"gp[{lo},{hi})"] = {
                "n_pairs": len(A), "n_team_seasons": n_ts,
                "corr": round(r, 4),
                "ci_fisher_on_team_seasons": [
                    round(float(np.tanh(np.arctanh(r) - 1.96 * se)), 4),
                    round(float(np.tanh(np.arctanh(r) + 1.96 * se)), 4)]}
    mb = json.load(open(ROOT / "data" / "am_measure.json"))[
        "M_B_recent_vs_old_weight_by_phase"]
    out["M_E_implied_halflife"] = {}
    for k, v in mb.items():
        gp = v["mean_gp"]
        r = v["per_game_weight_ratio_recent_over_old"]
        ci = v["ratio_ci"]
        out["M_E_implied_halflife"][k] = {
            "mean_gp": gp, "observed_ratio": r,
            "implied_halflife_games": round(implied_h(r, gp), 2),
            "implied_halflife_ci": [round(implied_h(ci[1], gp), 2),
                                    round(implied_h(ci[0], gp), 2)],
            "ratio_predicted_by_constant_h21": round(
                (0.5 ** (np.arange(0, 5) / 21.0)).mean() /
                (0.5 ** (np.arange(5, int(round(gp))) / 21.0)).mean(), 3)}
    (ROOT / "data" / "am_measure_d.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
