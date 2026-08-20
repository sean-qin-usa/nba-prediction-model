"""Does DE-CONFOUNDED zone-split defensive RAPM beat raw team allowance for
defender-aware props? Single temporal split (fit on first 60% of the season,
test on the last 40%). Compares base vs raw-allowance-aware vs defRAPM-aware
props by held-out CRPS. If defRAPM wins, the de-confounding pays off downstream.
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
from nbapred.model.def_rapm import fit_all

LG = {"rim": 0.613, "mid": 0.438, "thr": 0.355}


def crps(s, y):
    s = np.sort(s); n = len(s)
    return float(np.mean(np.abs(s - y)) - 0.5 * (2 * np.arange(1, n + 1) - n - 1) @ s / n**2)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def main(sims=2500, max_eval=1000):
    con = connect(read_only=True)
    games = con.execute("""SELECT DISTINCT game_id, game_date FROM nba_games
        WHERE season='2025-26' AND game_id LIKE '002%' ORDER BY game_date""").fetchdf()
    cut_date = games.game_date.quantile(0.6)
    train_gids = set(games[games.game_date <= cut_date].game_id)

    # team -> minute-weighted player minutes on TRAIN (for aggregating def-RAPM)
    mins = con.execute("""SELECT team_id, player_id, sum(seconds)/60.0 m
        FROM player_game_stats WHERE game_id IN (SELECT DISTINCT game_id FROM nba_games
        WHERE season='2025-26' AND game_date <= ?) GROUP BY 1,2""", [cut_date]).fetchdf()
    abbr2id = {r[1]: r[0] for r in con.execute(
        "SELECT DISTINCT team_id, team_abbrev FROM nba_games WHERE game_id LIKE '002%'").fetchall()}

    # test player-games (after cut)
    pg = con.execute("""SELECT s.player_id, g.game_date, g.matchup, g.team_abbrev, s.pts
        FROM player_game_stats s JOIN nba_games g ON g.game_id=s.game_id AND g.team_id=s.team_id
        WHERE s.game_id LIKE '002%' AND s.seconds>=720 AND g.game_date > ? ORDER BY g.game_date""",
        [cut_date]).fetchdf()
    con.close()

    print(f"fitting def-RAPM on {len(train_gids)} train games...")
    dr, _, counts = fit_all(allowed_gids=train_gids)
    print(f"def-RAPM shots: {counts}")

    # team zone-D from def-RAPM: minute-weighted avg of rostered players' zone ratings,
    # converted to a logit shift (positive rating = good D => negative make shift)
    team_defrapm = {}
    for tid, grp in mins.groupby("team_id"):
        acc = {"rim": 0.0, "mid": 0.0, "thr": 0.0}; wsum = 0.0
        for r in grp.itertuples():
            pr = dr.get(r.player_id)
            if pr:
                for z in acc:
                    acc[z] += r.m * pr.get(z, 0.0)
                wsum += r.m
        if wsum > 0:
            # rating (pts/100 suppressed) -> approx make-prob logit shift: scale down
            team_defrapm[tid] = {z: -acc[z] / wsum / 100.0 * 4 for z in acc}

    con = connect(read_only=True)
    base_c, raw_c, rapm_c, n = [], [], [], 0
    for r in pg.itertuples():
        if n >= max_eval:
            break
        rates = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
        if rates is None or rates["n_games"] < 8 or rates["proj_min"] < 20:
            continue
        opp = None
        for tok in r.matchup.replace("vs.", "@").split("@"):
            tok = tok.strip()
            if tok and tok != r.team_abbrev:
                opp = tok
        opp_id = abbr2id.get(opp)
        if opp_id is None:
            continue
        raw_shift = team_zone_defense(con, int(opp_id), before=r.game_date, league=LG)
        rapm_shift = team_defrapm.get(opp_id, {"rim": 0, "mid": 0, "thr": 0})
        y = r.pts
        base_c.append(crps(simulate_player(rates, sims, seed=n)["points"], y))
        raw_c.append(crps(simulate_player(apply_opp_defense(rates, raw_shift), sims, seed=n)["points"], y))
        rapm_c.append(crps(simulate_player(apply_opp_defense(rates, rapm_shift), sims, seed=n)["points"], y))
        n += 1
    con.close()

    print(f"\ntest player-games: {n}")
    print(f"CRPS base             : {np.mean(base_c):.4f}")
    print(f"CRPS raw-allowance    : {np.mean(raw_c):.4f}  (delta {np.mean(base_c)-np.mean(raw_c):+.4f})")
    print(f"CRPS def-RAPM         : {np.mean(rapm_c):.4f}  (delta {np.mean(base_c)-np.mean(rapm_c):+.4f})")


if __name__ == "__main__":
    main()
