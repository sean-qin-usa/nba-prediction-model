"""Do we MAKE MONEY betting win-prob? Walk-forward our adjusted-ratings model on
2025-26, bet the side where our edge over the market (spread-implied prob)
exceeds a threshold, and compute record + ROI at standard -110 vig. Also the
honest edge test: among games where we disagree with the market, are we
vindicated by outcomes, or is the market right (null zone)?
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect
from nbapred.model.team_ratings import TeamRatings
from scripts.eval_team_ratings import preload, sigmoid, SCALE

VIG_ODDS = 1.909  # -110 decimal payout; breakeven win rate 52.38%


def main(min_train=120, refit_every=5, ridge=25.0, edge_thr=0.05):
    con = connect(read_only=True)
    games, fit_rows = preload(con)
    # market spread-implied prob by (date, home, away)
    mkt = {}
    for r in con.execute("SELECT game_date, home, away, p_home_spread FROM odds_market "
                         "WHERE season_end=2026").fetchall():
        mkt[(str(r[0])[:10], r[1], r[2])] = r[3]
    con.close()

    fdates = np.array([r[0] for r in fit_rows])
    from nba_api.stats.static import teams as T
    id2ab = {t["id"]: t["abbreviation"] for t in T.get_teams()}

    tr, since = None, 10**9
    bets, wins, pnl = 0, 0, 0.0
    dis_n, dis_hit = 0, 0
    for i, (d, gid, hid, aid, habbr, aabbr, hw) in enumerate(games):
        if i < min_train:
            continue
        if tr is None or since >= refit_every:
            cut = np.searchsorted(fdates, d)
            tr = TeamRatings(ridge=ridge).fit([r[1:] for r in fit_rows[:cut]])
            since = 0
        since += 1
        p_us = float(sigmoid(tr.pred_margin(hid, aid) / SCALE))
        key = (str(d)[:10], id2ab.get(hid), id2ab.get(aid))
        p_mkt = mkt.get(key)
        if p_mkt is None:
            continue
        # edge test: do our disagreements beat the market's implied rate?
        if abs(p_us - p_mkt) > 0.03:
            dis_n += 1
            dis_hit += (hw if p_us > p_mkt else (1 - hw))
        # betting: take the side we like more than market by > threshold
        if p_us - p_mkt > edge_thr:      # bet home
            bets += 1; won = hw
        elif p_mkt - p_us > edge_thr:    # bet away
            bets += 1; won = 1 - hw
        else:
            continue
        wins += won
        pnl += (VIG_ODDS - 1) if won else -1

    print(f"edge threshold: {edge_thr:.2f}")
    print(f"bets: {bets}  win rate: {wins/max(bets,1):.3f}  (breakeven 0.524 at -110)")
    print(f"ROI: {pnl/max(bets,1)*100:+.2f}%   total units: {pnl:+.1f}")
    print(f"\nedge test — games where we disagree w/ market by >3%:")
    print(f"  n={dis_n}  our side hit {dis_hit/max(dis_n,1):.3f} of the time")
    print(f"  (>0.50 = our disagreements beat the market; ~0.50 or less = null zone)")


if __name__ == "__main__":
    for thr in (0.03, 0.05, 0.08):
        main(edge_thr=thr); print()
