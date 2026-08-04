"""2024-25 head-to-head: PRODUCTION model (efficiency ratings + cold-start
prior) walk-forward vs the market (0.5863). Second season of evidence.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect
from nbapred.eval.metrics import log_loss
from nbapred.model.production import SCALE, fit_production, last_season_prior, sigmoid
from nbapred.model.team_ratings import TeamRatings, game_rows


def main(season="2024-25", refit_every=5):
    con = connect(read_only=True)
    meta = con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl, game_date
        FROM nba_games WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL""",
        [season]).fetchdf()
    # NOTE: key dates as str[:10] — fetchdf gives Timestamps ('... 00:00:00'),
    # fetchall gives dates; raw str() keys silently never match (bug that also
    # zeroed backtest_betting and eval_enhanced's market column).
    mkt = {(str(r[0])[:10], r[1], r[2]): r[3] for r in con.execute(
        "SELECT game_date, home, away, p_home_spread FROM odds_market WHERE season_end=?",
        [int(season[:4]) + 1]).fetchall()}
    prior = last_season_prior(con, season)
    games_played = {}
    id2ab = dict(con.execute(
        "SELECT DISTINCT team_id, team_abbrev FROM nba_games WHERE season=?", [season]).fetchall())

    by_game = {}
    for x in meta.itertuples():
        by_game.setdefault(x.game_id, []).append(x)
    order = sorted(by_game, key=lambda g: (by_game[g][0].game_date, g))

    y, p_us, p_mkt, tr, since = [], [], [], None, 10**9
    for i, gid in enumerate(order):
        recs = by_game[gid]
        if len(recs) != 2:
            continue
        m = recs[0].matchup
        host = m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        h = next((x for x in recs if x.team_abbrev == host), None)
        a = next((x for x in recs if x.team_abbrev != host), None)
        if not h or not a:
            continue
        d = h.game_date
        if tr is None or since >= refit_every:
            tr = TeamRatings(ridge=25.0).fit(game_rows(con, before=d, season=season))
            since = 0
        since += 1
        gh, ga = games_played.get(h.team_id, 0), games_played.get(a.team_id, 0)
        wh, wa = max(0.0, 1 - gh / 20.0), max(0.0, 1 - ga / 20.0)
        margin = tr.pred_margin(h.team_id, a.team_id) \
            + wh * prior.get(id2ab.get(h.team_id, ""), 0.0) \
            - wa * prior.get(id2ab.get(a.team_id, ""), 0.0)
        pm = mkt.get((str(d)[:10], h.team_abbrev, a.team_abbrev))
        if pm is not None:
            y.append(int(h.wl == "W"))
            p_us.append(float(sigmoid(margin / SCALE)))
            p_mkt.append(pm)
        games_played[h.team_id] = gh + 1
        games_played[a.team_id] = ga + 1
    con.close()

    y = np.array(y)
    print(f"2024-25 games scored: {len(y)} (FULL season incl. early — cold-start covers it)")
    print(f"  PRODUCTION model: {log_loss(y, p_us):.4f}")
    print(f"  MARKET (spread) : {log_loss(y, p_mkt):.4f}")
    print(f"  (2025-26 was: ours 0.601 no-prior full / 0.5815 mature; market 0.578)")


if __name__ == "__main__":
    main()
