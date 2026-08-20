#!/usr/bin/env python3
"""Validate the prop simulator on held-out player-games. The honest test for a
predicted DISTRIBUTION: PIT calibration — the actual outcome's percentile in the
predicted distribution should be ~Uniform(0,1) (mean ~0.5, and coverage of the
central interval ~ nominal). Also reports CRPS vs a naive "recent-average"
baseline. Leakage-safe: each player-game predicted from that player's games
strictly before it.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect
from nbapred.engine.props import player_rates_from_stats, simulate_player


def crps(samples, y):
    s = np.sort(samples); n = len(s)
    return float(np.mean(np.abs(s - y)) - 0.5 * (2 * np.arange(1, n + 1) - n - 1) @ s / n**2)


def main(min_prior_games=8, min_proj_min=20, sims=3000, max_eval=1500):
    con = connect(read_only=True)
    # candidate player-games: rotation players, later in the season
    pg = con.execute("""
        SELECT s.player_id, g.game_date, s.pts, s.seconds
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '002%' AND s.seconds >= 720
        ORDER BY g.game_date
    """).fetchdf()

    pit, crps_model, crps_base, cover80, n = [], [], [], 0, 0
    for r in pg.itertuples():
        if n >= max_eval:
            break
        rates = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
        if rates is None or rates["n_games"] < min_prior_games or rates["proj_min"] < min_proj_min:
            continue
        sim = simulate_player(rates, n=sims, seed=n)["points"]
        y = r.pts
        pit.append(float(np.mean(sim < y) + 0.5 * np.mean(sim == y)))
        crps_model.append(crps(sim, y))
        crps_base.append(abs(rates["proj_min"] * (rates["rate_rim"]*rates["fg_rim"]*2
                          + rates["rate_mid"]*rates["fg_mid"]*2
                          + rates["rate_thr"]*rates["fg_thr"]*3
                          + rates["fta_per_min"]*rates["ft_pct"]) - y))  # point-est baseline
        lo, hi = np.percentile(sim, [10, 90])
        cover80 += int(lo <= y <= hi)
        n += 1
    con.close()

    pit = np.array(pit)
    print(f"player-games evaluated: {n}")
    print(f"\nPIT calibration (want mean~0.5, std~0.29 for uniform):")
    print(f"  mean {pit.mean():.3f}  std {pit.std():.3f}")
    print(f"  80% central interval coverage: {cover80/n:.3f}  (want ~0.80)")
    print(f"\nCRPS (lower=better sharper+calibrated):")
    print(f"  prop simulator     : {np.mean(crps_model):.3f}")
    print(f"  point-est baseline : {np.mean(crps_base):.3f}  (|mean - actual|, a floor)")
    # PIT histogram (uniformity)
    h, _ = np.histogram(pit, bins=10, range=(0, 1))
    print("\nPIT histogram (flat = calibrated):", (h / n * 10).round(2).tolist())


if __name__ == "__main__":
    main()
