"""Does opponent-pace adjustment improve props? base vs pace-adjusted, held-out
CRPS, paired bootstrap. Marginal improvements stack — this one should help
counting stats vs fast/slow opponents.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from nbapred.db import connect
from nbapred.engine.props import player_rates_from_stats, simulate_player, team_pace, apply_pace


def crps(s, y):
    s = np.sort(s); n = len(s)
    return float(np.mean(np.abs(s - y)) - 0.5 * (2*np.arange(1, n+1) - n - 1) @ s / n**2)


def main(sims=3000, max_eval=1200):
    con = connect(read_only=True)
    pg = con.execute("""SELECT s.player_id, s.team_id, g.game_date, g.matchup, g.team_abbrev, s.pts
        FROM player_game_stats s JOIN nba_games g ON g.game_id=s.game_id AND g.team_id=s.team_id
        WHERE s.game_id LIKE '002%' AND s.seconds>=720 ORDER BY g.game_date""").fetchdf()
    ab2id = {r[1]: r[0] for r in con.execute(
        "SELECT DISTINCT team_id, team_abbrev FROM nba_games WHERE game_id LIKE '002%'").fetchall()}
    pace_cache = {}
    base_c, pace_c, n = [], [], 0
    for r in pg.itertuples():
        if n >= max_eval:
            break
        rates = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
        if not rates or rates["n_games"] < 8 or rates["proj_min"] < 20:
            continue
        opp = None
        for tok in r.matchup.replace("vs.", "@").split("@"):
            tok = tok.strip()
            if tok and tok != r.team_abbrev:
                opp = tok
        oid = ab2id.get(opp)
        if oid is None:
            continue
        key_o, key_s = (oid, str(r.game_date)), (r.team_id, str(r.game_date))
        if key_o not in pace_cache:
            pace_cache[key_o] = team_pace(con, int(oid), before=r.game_date)
        if key_s not in pace_cache:
            pace_cache[key_s] = team_pace(con, int(r.team_id), before=r.game_date)
        y = r.pts
        base_c.append(crps(simulate_player(rates, sims, seed=n)["points"], y))
        adj = apply_pace(rates, pace_cache[key_o], pace_cache[key_s])
        pace_c.append(crps(simulate_player(adj, sims, seed=n)["points"], y))
        n += 1
    con.close()
    bc, pc = np.mean(base_c), np.mean(pace_c)
    print(f"player-games: {n}")
    print(f"CRPS base {bc:.4f}  pace-adjusted {pc:.4f}  delta {bc-pc:+.4f}")
    d = np.array(base_c) - np.array(pace_c); rng = np.random.default_rng(0)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"95% CI ({lo:+.4f},{hi:+.4f}) -> {'KEEP pace' if lo > 0 else 'not significant'}")


if __name__ == "__main__":
    main()
