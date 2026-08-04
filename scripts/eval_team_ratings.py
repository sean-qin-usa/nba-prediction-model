#!/usr/bin/env python3
"""Does opponent-adjustment fix the engine's failure? Walk-forward the
opponent-adjusted team ratings on 2025-26 and compare to Elo + the failed engine
on the same games. Confirms (or refutes) the diagnosis that the v0 engine lost
because its inputs were schedule-unadjusted.

Fast: all team-game rows are pulled ONCE, then the walk-forward filters in
memory (no per-refit DB round trip). Win prob from predicted efficiency margin
via a fixed logistic scale (a 10-pt favorite ~ 80% => scale 7.2), from
basketball priors, NOT tuned on test.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect
from nbapred.eval.metrics import summary
from nbapred.eval.walkforward import Elo
from nbapred.model.team_ratings import TeamRatings, possessions

SCALE = 7.2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def preload(con, season="2025-26"):
    """One query -> per-game (date, home, away, home_won) and the team-game rows
    (off_team, def_team, is_home, ortg, date) for fitting."""
    df = con.execute("""
        SELECT s.game_id, s.team_id, g.game_date, g.matchup, g.team_abbrev, g.wl,
               sum(s.pts) pts, sum(s.fga) fga, sum(s.fta) fta, sum(s.oreb) oreb, sum(s.tov) tov
        FROM player_game_stats s
        JOIN nba_games g ON g.game_id=s.game_id AND g.team_id=s.team_id
        WHERE g.season=? AND s.game_id LIKE '002%' AND g.wl IS NOT NULL
        GROUP BY s.game_id, s.team_id, g.game_date, g.matchup, g.team_abbrev, g.wl
    """, [season]).fetchdf()
    by_game = {}
    for r in df.itertuples():
        by_game.setdefault(r.game_id, []).append(r)
    games, fit_rows = [], []
    for gid, recs in by_game.items():
        if len(recs) != 2:
            continue
        m = recs[0].matchup
        host = m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        h = next((x for x in recs if x.team_abbrev == host), None)
        a = next((x for x in recs if x.team_abbrev != host), None)
        if not h or not a:
            continue
        d = h.game_date
        games.append((d, gid, int(h.team_id), int(a.team_id), h.team_abbrev, a.team_abbrev,
                      int(h.wl == "W")))
        for r in recs:
            opp = a if r.team_id == h.team_id else h
            poss = possessions(r.fga, r.fta, r.oreb, r.tov)
            if poss >= 50:
                fit_rows.append((d, int(r.team_id), int(opp.team_id),
                                 r.team_abbrev == host, 100.0 * r.pts / poss))
    games.sort(key=lambda x: (x[0], x[1]))
    fit_rows.sort(key=lambda x: x[0])
    return games, fit_rows


def main(min_train=120, refit_every=5, ridge=25.0):
    con = connect(read_only=True)
    games, fit_rows = preload(con)
    con.close()

    fdates = np.array([r[0] for r in fit_rows])
    elo = Elo()
    y, p_adj, p_elo = [], [], []
    tr, since = None, 10**9
    for i, (d, gid, hid, aid, habbr, aabbr, hw) in enumerate(games):
        if i >= min_train:
            if tr is None or since >= refit_every:
                cut = np.searchsorted(fdates, d)          # rows strictly before date d
                tr = TeamRatings(ridge=ridge).fit([r[1:] for r in fit_rows[:cut]])
                since = 0
            since += 1
            margin = tr.pred_margin(hid, aid)
            y.append(hw); p_adj.append(float(sigmoid(margin / SCALE)))
            p_elo.append(elo.p_home(habbr, aabbr))
        elo.update(habbr, aabbr, hw)

    y = np.array(y)
    print(f"games: {len(y)}  (ridge={ridge}, scale={SCALE})")
    print("OPP-ADJUSTED :", {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in summary(y, p_adj).items()})
    print("ELO          :", {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in summary(y, p_elo).items()})
    print("ref: v0 engine 0.6925 | home base-rate 0.6882 | coin flip 0.6931")


if __name__ == "__main__":
    main()
