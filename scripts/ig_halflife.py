#!/usr/bin/env python3
"""IG probe (read-only): tune half_life_games (shipped 10.0, never gated).

Paired one-step-ahead points-CRPS across hl in {4,7,10,15,25}, same seed per row.
Hypothesis (not blind grid): rate half-life should be LONGER than minutes
half-life because per-minute skill moves slower than role/minutes; hl=10 was a
guess applied to both jointly, so the sweep separates 'too fast' vs 'too slow'.
"""
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from nbapred.db import connect
from nbapred.engine.props import player_rates_from_stats, simulate_player

HLS = [4.0, 7.0, 10.0, 15.0, 25.0]


def crps(s, y):
    s = np.sort(s); n = len(s)
    return float(np.mean(np.abs(s - y)) - 0.5 * (2 * np.arange(1, n + 1) - n - 1) @ s / n**2)


def main(sims=1500, max_eval=450):
    con = connect(read_only=True)
    pg = con.execute("""
        SELECT s.player_id, g.game_date, s.pts FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '00225%' AND s.seconds >= 720
        ORDER BY g.game_date, s.player_id""").fetchdf()
    stride = max(1, len(pg) // (max_eval * 2))
    rows = pg.iloc[::stride]
    out = {hl: [] for hl in HLS}
    n = 0
    for r in rows.itertuples():
        if n >= max_eval:
            break
        base = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
        if base is None or base["n_games"] < 8 or base["proj_min"] < 20:
            continue
        y = float(r.pts)
        for hl in HLS:
            rates = base if hl == 10.0 else player_rates_from_stats(
                con, int(r.player_id), before=r.game_date, half_life_games=hl)
            out[hl].append(crps(simulate_player(rates, sims, seed=n)["points"], y))
        n += 1
    con.close()
    print(f"rows: {n}")
    for hl in HLS:
        print(f"  hl={hl:5.1f}: CRPS {np.mean(out[hl]):.4f}")
    rng = np.random.default_rng(0)
    ref = np.array(out[10.0])
    for hl in HLS:
        if hl == 10.0:
            continue
        d = ref - np.array(out[hl])
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  hl10 - hl{hl:g}: {d.mean():+.4f}  CI ({lo:+.4f},{hi:+.4f})")


if __name__ == "__main__":
    main()
