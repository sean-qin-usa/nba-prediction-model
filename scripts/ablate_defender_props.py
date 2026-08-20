#!/usr/bin/env python3
"""Ablation gate for defender-aware props: does conditioning a player's shooting
on the ACTUAL opponent defense improve held-out prop accuracy? Compares base
prop model vs defender-aware (opponent zone-defense applied) on the same
player-games by CRPS + PIT. If defender-aware wins OOS, the richness earned it.
Leakage-safe: opponent zone-defense computed from games strictly before.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect
from nbapred.engine.props import (apply_opp_defense, player_rates_from_stats,
                                  simulate_player, team_zone_defense)


def crps(samples, y):
    s = np.sort(samples); n = len(s)
    return float(np.mean(np.abs(s - y)) - 0.5 * (2 * np.arange(1, n + 1) - n - 1) @ s / n**2)


def main(min_prior=8, min_min=20, sims=3000, max_eval=1200):
    con = connect(read_only=True)
    pg = con.execute("""
        SELECT s.player_id, s.team_id, g.game_date, g.game_id, g.matchup, g.team_abbrev, s.pts
        FROM player_game_stats s
        JOIN nba_games g ON g.game_id=s.game_id AND g.team_id=s.team_id
        WHERE s.game_id LIKE '002%' AND s.seconds >= 720
        ORDER BY g.game_date
    """).fetchdf()
    abbr2id = {r[1]: r[0] for r in con.execute(
        "SELECT DISTINCT team_id, team_abbrev FROM nba_games WHERE game_id LIKE '002%'").fetchall()}

    base_crps, def_crps, base_pit, def_pit, n = [], [], [], [], 0
    stride = max(1, len(pg) // (max_eval * 3))   # spread sample across full timeline
    pg = pg.iloc[::stride]
    for r in pg.itertuples():
        if n >= max_eval:
            break
        rates = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
        if rates is None or rates["n_games"] < min_prior or rates["proj_min"] < min_min:
            continue
        # opponent team
        m = r.matchup
        opp_abbr = None
        for tok in m.replace("vs.", "@").split("@"):
            tok = tok.strip()
            if tok and tok != r.team_abbrev:
                opp_abbr = tok
        opp_id = abbr2id.get(opp_abbr)
        if opp_id is None:
            continue
        shift = team_zone_defense(con, int(opp_id), before=r.game_date)

        base = simulate_player(rates, n=sims, seed=n)["points"]
        dfr = simulate_player(apply_opp_defense(rates, shift), n=sims, seed=n)["points"]
        y = r.pts
        base_crps.append(crps(base, y)); def_crps.append(crps(dfr, y))
        base_pit.append(float(np.mean(base < y) + 0.5 * np.mean(base == y)))
        def_pit.append(float(np.mean(dfr < y) + 0.5 * np.mean(dfr == y)))
        n += 1
    con.close()

    bc, dc = np.mean(base_crps), np.mean(def_crps)
    print(f"player-games: {n}")
    print(f"\nCRPS (lower=better):  base {bc:.4f}   defender-aware {dc:.4f}   delta {bc-dc:+.4f}")
    # paired bootstrap on the CRPS improvement
    d = np.array(base_crps) - np.array(def_crps)
    rng = np.random.default_rng(0)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"CRPS improvement 95% CI: ({lo:+.4f}, {hi:+.4f})  -> "
          f"{'KEEP (defender-aware helps)' if lo > 0 else 'not significant'}")
    print(f"\nPIT mean  base {np.mean(base_pit):.3f}  defender-aware {np.mean(def_pit):.3f} (want 0.5)")


if __name__ == "__main__":
    main()
