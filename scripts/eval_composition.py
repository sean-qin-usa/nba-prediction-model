"""Player-composition-aware win-prob: team strength = sum of AVAILABLE players'
RAPM (minute-weighted), so when a player sits the rating drops. This is the
mechanism that prices injuries/rest — the thing our team-average model is blind
to. Single split: fit RAPM on train, rate test games by who played, compare to
the team-average model and the market.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect
from nbapred.eval.metrics import log_loss
from nbapred.model.rapm import RAPM, load_stints

SCALE = 7.2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def main():
    con = connect(read_only=True)
    games = con.execute("""SELECT DISTINCT game_id, game_date FROM nba_games
        WHERE season='2025-26' AND game_id LIKE '002%' ORDER BY game_date""").fetchdf()
    cut = games.game_date.quantile(0.6)
    # RAPM on TRAIN stints only (leakage-safe)
    stints = load_stints(con, before=cut)
    r = RAPM(ridge=2000).fit(stints)
    net = {p: r.net(p) for p in r.off}

    # per-game team player-minutes (who played)
    pm = con.execute("""SELECT s.game_id, s.team_id, s.player_id, s.seconds
        FROM player_game_stats s WHERE s.game_id LIKE '002%'""").fetchdf()
    meta = con.execute("""SELECT g.game_id, g.team_id, g.team_abbrev, g.matchup, g.wl, g.game_date
        FROM nba_games g WHERE g.season='2025-26' AND g.game_id LIKE '002%' AND g.wl IS NOT NULL""").fetchdf()
    con.close()

    # composition rating per (game, team) = Σ net_RAPM * minutes/48
    pm["min"] = pm.seconds / 60.0
    pm["contrib"] = pm.player_id.map(lambda p: net.get(p, 0.0)) * pm["min"] / 48.0
    team_rate = pm.groupby(["game_id", "team_id"])["contrib"].sum().to_dict()

    by_game = {}
    for x in meta.itertuples():
        by_game.setdefault(x.game_id, []).append(x)

    y, p_comp = [], []
    for gid, recs in by_game.items():
        if len(recs) != 2 or recs[0].game_date <= cut:
            continue
        m = recs[0].matchup
        host = m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        h = next((x for x in recs if x.team_abbrev == host), None)
        a = next((x for x in recs if x.team_abbrev != host), None)
        if not h or not a:
            continue
        hr = team_rate.get((gid, h.team_id), 0.0)
        ar = team_rate.get((gid, a.team_id), 0.0)
        margin = (hr - ar) + 3.0        # +3 home edge
        y.append(int(h.wl == "W"))
        p_comp.append(float(sigmoid(margin / SCALE)))

    y = np.array(y)
    print(f"test games: {len(y)}")
    print(f"composition-aware (RAPM sum of who-played): log loss {log_loss(y, p_comp):.4f}")
    print("ref: team-average ratings 0.601 | market 0.578 | Elo 0.618")
    print("\n(uses actual minutes as availability proxy; real forecast uses projected")
    print(" minutes from injury reports ~1hr pre-tip, same info the market has.)")


if __name__ == "__main__":
    main()
