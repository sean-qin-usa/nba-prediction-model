"""Does adding features raise prediction power? Walk-forward compare:
  base            = opponent-adjusted ratings (0.601)
  +recency        = recency-weighted ratings (recent games count more)
  +rest           = base margin + rest-advantage adjustment (schedule_features)
  +both
against the market (spread-implied, 0.578) on 2025-26. Each feature judged by
OOS log loss — the ablation gate for prediction power.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect
from nbapred.eval.metrics import log_loss
from nbapred.model.team_ratings import TeamRatings
from scripts.eval_team_ratings import preload, sigmoid, SCALE

HALF_LIFE_DAYS = 21.0
REST_COEF = 0.45   # points of margin per day of rest advantage (basketball prior)


def main(min_train=120, refit_every=5, ridge=25.0):
    con = connect(read_only=True)
    games, fit_rows = preload(con)
    # rest advantage per (game_id, is it home's) — from schedule_features
    rest = {}
    for r in con.execute("SELECT game_id, team, is_home, days_rest FROM schedule_features "
                        "WHERE season='2025-26'").fetchall():
        rest[(r[0], r[3] if False else r[1])] = (r[2], r[3])
    rest_by_game = {}
    for r in con.execute("""SELECT game_id, is_home, days_rest FROM schedule_features
                          WHERE season='2025-26'""").fetchall():
        rest_by_game.setdefault(r[0], {})[bool(r[1])] = r[2]
    mkt = {}
    for r in con.execute("SELECT game_date, home, away, p_home_spread FROM odds_market "
                        "WHERE season_end=2026").fetchall():
        mkt[(str(r[0])[:10], r[1], r[2])] = r[3]
    con.close()

    from nba_api.stats.static import teams as T
    id2ab = {t["id"]: t["abbreviation"] for t in T.get_teams()}
    fdates = np.array([r[0] for r in fit_rows])
    fdays = np.array([(d - fdates.min()).days for d in fdates])

    y = []
    p = {"base": [], "recency": [], "rest": [], "both": [], "market": []}
    tr, trw, since = None, None, 10**9
    for i, (d, gid, hid, aid, habbr, aabbr, hw) in enumerate(games):
        if i < min_train:
            continue
        if tr is None or since >= refit_every:
            cut = np.searchsorted(fdates, d)
            rowsub = [r[1:] for r in fit_rows[:cut]]
            tr = TeamRatings(ridge=ridge).fit(rowsub)
            age = (d - fdates[:cut])
            w = 0.5 ** (np.array([a.days for a in age]) / HALF_LIFE_DAYS)
            trw = TeamRatings(ridge=ridge).fit(rowsub, weights=w)
            since = 0
        since += 1
        # rest advantage (home - away days rest)
        rg = rest_by_game.get(gid, {})
        radv = 0.0
        if True in rg and False in rg and rg[True] is not None and rg[False] is not None:
            radv = np.clip(rg[True] - rg[False], -3, 3)
        m_base = tr.pred_margin(hid, aid)
        m_rec = trw.pred_margin(hid, aid)
        y.append(hw)
        p["base"].append(sigmoid(m_base / SCALE))
        p["recency"].append(sigmoid(m_rec / SCALE))
        p["rest"].append(sigmoid((m_base + REST_COEF * radv) / SCALE))
        p["both"].append(sigmoid((m_rec + REST_COEF * radv) / SCALE))
        p["market"].append(mkt.get((str(d)[:10], id2ab.get(hid), id2ab.get(aid)), 0.5))

    y = np.array(y)
    print(f"games: {len(y)}  (half-life {HALF_LIFE_DAYS}d, rest {REST_COEF} pt/day)")
    for name in ["base", "recency", "rest", "both", "market"]:
        print(f"  {name:10}: log loss {log_loss(y, p[name]):.4f}")


if __name__ == "__main__":
    main()
