"""Kalman-filtered prop rates vs the current EWMA, on held-out player-games.
Paired-bootstrap CRPS + PIT. Keep Kalman only if it improves OOS.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from nbapred.db import connect
from nbapred.engine.props import player_rates_from_stats, player_rates_kalman, simulate_player


def crps(s, y):
    s = np.sort(s); n = len(s)
    return float(np.mean(np.abs(s - y)) - 0.5 * (2 * np.arange(1, n + 1) - n - 1) @ s / n**2)


def main(sims=3000, max_eval=1200):
    con = connect(read_only=True)
    pg = con.execute("""SELECT s.player_id, g.game_date, s.pts FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id,game_date FROM nba_games) g USING(game_id)
        WHERE s.game_id LIKE '002%' AND s.seconds>=720 ORDER BY g.game_date""").fetchdf()
    ew_c, ka_c, ew_pit, ka_pit, n = [], [], [], [], 0
    for r in pg.itertuples():
        if n >= max_eval:
            break
        ew = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
        ka = player_rates_kalman(con, int(r.player_id), before=r.game_date)
        if not ew or not ka or ew["n_games"] < 8 or ew["proj_min"] < 20:
            continue
        y = r.pts
        se = simulate_player(ew, sims, seed=n)["points"]
        sk = simulate_player(ka, sims, seed=n)["points"]
        ew_c.append(crps(se, y)); ka_c.append(crps(sk, y))
        ew_pit.append(float(np.mean(se < y) + 0.5*np.mean(se == y)))
        ka_pit.append(float(np.mean(sk < y) + 0.5*np.mean(sk == y)))
        n += 1
    con.close()
    ec, kc = np.mean(ew_c), np.mean(ka_c)
    print(f"player-games: {n}")
    print(f"CRPS  EWMA {ec:.4f}   Kalman {kc:.4f}   delta {ec-kc:+.4f}")
    d = np.array(ew_c) - np.array(ka_c); rng = np.random.default_rng(0)
    boot = np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)])
    lo, hi = np.percentile(boot,[2.5,97.5])
    print(f"CRPS improvement 95% CI ({lo:+.4f},{hi:+.4f}) -> {'KEEP Kalman' if lo>0 else 'not significant'}")
    print(f"PIT mean  EWMA {np.mean(ew_pit):.3f}   Kalman {np.mean(ka_pit):.3f}  (want 0.5)")


if __name__ == "__main__":
    main()
