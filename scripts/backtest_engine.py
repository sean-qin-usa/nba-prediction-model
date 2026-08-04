#!/usr/bin/env python3
"""End-to-end backtest: run the composition-driven MC engine through a
walk-forward split and score its win-probabilities against outcomes + baselines.

This is the capstone that ties fit -> compose -> simulate -> evaluate together.
Leakage-safe: each test game is predicted from rates composed on games STRICTLY
before it (compose.matchup_rates(before=game_date)).

Caveats on current data: single season (2025-26), partial backfill, and NO
market odds for these games (SBR ends 2023, live logger is offseason-empty), so
the honest comparison here is engine vs naive baselines (coin flip, home base
rate). The market/Elo comparison needs multi-season history + logged odds and
runs once those exist. Treat the numbers as a pipeline smoke test, not a verdict.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect
from nbapred.engine.compose import matchup_rates
from nbapred.engine.fast import simulate_matchup_fast as simulate_matchup
from nbapred.eval.metrics import summary


def main(min_train_games=120, sims=400):
    con = connect(read_only=True)
    # regular-season games with a home/away parse and a decided result
    games = con.execute("""
        SELECT g.game_id, g.game_date, g.matchup, g.team_abbrev, g.team_id, g.wl
        FROM nba_games g
        WHERE g.season='2025-26' AND g.game_id LIKE '002%' AND g.wl IS NOT NULL
        ORDER BY g.game_date, g.game_id
    """).fetchdf()

    # pair rows into games (home = team after '@' or before 'vs.')
    by_game = {}
    for r in games.itertuples():
        by_game.setdefault(r.game_id, []).append(r)
    ordered = [gid for gid in dict.fromkeys(games.game_id)]

    y, p_eng, used = [], [], 0
    home_wins = 0
    for i, gid in enumerate(ordered):
        if i < min_train_games:
            continue  # need trailing history to compose from
        recs = by_game[gid]
        if len(recs) != 2:
            continue
        m = recs[0].matchup
        host = m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        home = next((x for x in recs if x.team_abbrev == host), None)
        away = next((x for x in recs if x.team_abbrev != host), None)
        if home is None or away is None:
            continue
        gdate = home.game_date
        hr = matchup_rates(con, int(home.team_id), int(away.team_id), before=gdate)
        ar = matchup_rates(con, int(away.team_id), int(home.team_id), before=gdate)
        res = simulate_matchup(hr, ar, n=sims, seed=i)
        hw = int(home.wl == "W")
        y.append(hw); p_eng.append(res["p_home_win"]); home_wins += hw; used += 1

    con.close()
    if used < 20:
        print(f"only {used} test games available — need more backfill. Skipping verdict.")
        return
    y = np.array(y)
    base_rate = home_wins / used
    print(f"test games: {used}  home win rate: {base_rate:.3f}")
    print("\n=== engine (composition-driven MC, walk-forward) ===")
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in summary(y, p_eng).items()})
    print("\n=== naive baselines on same games ===")
    print("coin flip   :", round(0.6931, 4))
    print("home base-rate:", {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in summary(y, [base_rate] * used).items()})


if __name__ == "__main__":
    main()
