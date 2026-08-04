#!/usr/bin/env python3
"""IG probe (no DB): live path seeds simulate_player with seed=pid every day
(predict_today.py:69) while every backtest uses seed=n (fresh draws per row).

Consequences quantified:
  1. MC noise at n=4000 -> per-player P(over) offset that NEVER averages out
     across the season live, but DOES average out in the backtest (asymmetry).
  2. Day-over-day correlation of the MC error under seed=pid with slowly
     drifting rates vs seed=fresh.
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nbapred.engine.props import simulate_player

RATES = dict(proj_min=32.0, sd_min=4.5,
             minutes_hist=np.array([34, 30, 36, 28, 33, 35, 31, 29, 32, 34,
                                    30, 36, 24, 33, 35, 31, 38, 32, 30, 33], float),
             rate_rim=0.18, rate_mid=0.10, rate_thr=0.24,
             fg_rim=0.62, fg_mid=0.44, fg_thr=0.36,
             fta_per_min=0.14, ft_pct=0.80, reb_per_min=0.15, ast_per_min=0.12)

line = None
# ground truth from a big sim
truth = simulate_player(RATES, n=400000, seed=999)["points"]
line = float(np.median(truth)) + 0.5
p_true = float(np.mean(truth > line))
print(f"line {line}, p_true {p_true:.4f}")

# 1. spread of p_hat across seeds at live n=4000
ps = np.array([np.mean(simulate_player(RATES, n=4000, seed=s)["points"] > line)
               for s in range(200)])
print(f"n=4000 MC sd of p_over across seeds: {ps.std():.4f} "
      f"(persistent per-player offset live; averages out in backtest)")
print(f"implied per-prop logloss cost E[dp^2]/(p(1-p)): "
      f"{ps.var() / (p_true * (1 - p_true)):.5f}")

# 2. day-over-day error correlation, drifting rates
rng = np.random.default_rng(5)
err_fixed, err_fresh = [], []
for day in range(60):
    r = dict(RATES)
    drift = 1 + 0.02 * rng.standard_normal()
    r["proj_min"] = RATES["proj_min"] + 0.5 * rng.standard_normal()
    for k in ("rate_rim", "rate_mid", "rate_thr", "fta_per_min"):
        r[k] = RATES[k] * drift
    big = simulate_player(r, n=200000, seed=1000 + day)["points"]
    pt = np.mean(big > line)
    e_fixed = np.mean(simulate_player(r, n=4000, seed=203954)["points"] > line) - pt
    e_fresh = np.mean(simulate_player(r, n=4000, seed=day)["points"] > line) - pt
    err_fixed.append(e_fixed); err_fresh.append(e_fresh)
ef, eh = np.array(err_fixed), np.array(err_fresh)
print(f"day-over-day MC error: seed=pid mean {ef.mean():+.4f} sd {ef.std():.4f} "
      f"(a bias, not noise)")
print(f"                       seed=day mean {eh.mean():+.4f} sd {eh.std():.4f}")
# lag-1 autocorr of the error series
print(f"lag-1 autocorr: seed=pid {np.corrcoef(ef[:-1], ef[1:])[0,1]:+.3f}  "
      f"seed=day {np.corrcoef(eh[:-1], eh[1:])[0,1]:+.3f}")
