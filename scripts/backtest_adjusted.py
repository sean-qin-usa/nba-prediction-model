#!/usr/bin/env python3
"""Backtest the OPPONENT-ADJUSTED possession engine, walk-forward, vs baselines.

For each game (after a warm-up), using ONLY games played before it, we fit team
ratings, scale the engine to each team's adjusted efficiency, simulate the game
many times, and read off P(home win). We then score those probabilities against
what actually happened (log loss). Compares: adjusted engine (MC) vs the
adjusted-ratings model (direct) vs Elo vs naive baselines.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect
from nbapred.engine.adjusted import EfficiencyCalibrator, matchup_engine_rates
from nbapred.engine.fast import simulate_matchup_fast
from nbapred.eval.metrics import summary
from nbapred.eval.walkforward import Elo
from nbapred.model.team_ratings import TeamRatings

# reuse the fast preloader
from scripts.eval_team_ratings import preload, sigmoid, SCALE


def main(min_train=120, refit_every=5, ridge=25.0, sims=1500):
    con = connect(read_only=True)
    games, fit_rows = preload(con)
    con.close()
    fdates = np.array([r[0] for r in fit_rows])
    cal = EfficiencyCalibrator()

    elo = Elo()
    y, p_eng, p_adj, p_elo = [], [], [], []
    tr, since = None, 10**9
    for i, (d, gid, hid, aid, habbr, aabbr, hw) in enumerate(games):
        if i >= min_train:
            if tr is None or since >= refit_every:
                cut = np.searchsorted(fdates, d)
                tr = TeamRatings(ridge=ridge).fit([r[1:] for r in fit_rows[:cut]])
                since = 0
            since += 1
            # direct adjusted-ratings win prob
            margin = tr.pred_margin(hid, aid)
            p_adj.append(float(sigmoid(margin / SCALE)))
            # engine win prob from simulated games
            hr, ar = matchup_engine_rates(tr, hid, aid, cal)
            res = simulate_matchup_fast(hr, ar, n=sims, seed=i)
            p_eng.append(res["p_home_win"])
            p_elo.append(elo.p_home(habbr, aabbr))
            y.append(hw)
        elo.update(habbr, aabbr, hw)

    y = np.array(y)
    print(f"games scored: {len(y)}  (sims/game={sims}, ridge={ridge})")
    print("\nADJUSTED ENGINE (MC):", {k: round(v, 4) if isinstance(v, float) else v
                                     for k, v in summary(y, p_eng).items()})
    print("adjusted ratings (direct):", {k: round(v, 4) if isinstance(v, float) else v
                                         for k, v in summary(y, p_adj).items()})
    print("Elo                     :", {k: round(v, 4) if isinstance(v, float) else v
                                       for k, v in summary(y, p_elo).items()})
    print("\nref: coin flip 0.6931 | raw (unadjusted) engine 0.6925 | market bar ~0.589")


if __name__ == "__main__":
    main()
