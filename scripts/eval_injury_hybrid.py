"""Injury-pricing done right: clean team-average rating as the BASE, then adjust
by the RAPM of players whose minutes deviate from normal (i.e. who's OUT). On
full-strength games the adjustment is ~0 (keeps the clean base); on injury games
it drops the team by the missing star's impact. Tests overall AND on the subset
of games with real roster changes — where the edge should live.
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
from nbapred.model.team_ratings import TeamRatings, game_rows
from scripts.eval_team_ratings import preload

SCALE = 7.2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def main():
    con = connect(read_only=True)
    games = con.execute("""SELECT DISTINCT game_id, game_date FROM nba_games
        WHERE season='2025-26' AND game_id LIKE '002%' ORDER BY game_date""").fetchdf()
    cut = games.game_date.quantile(0.6)

    tr = TeamRatings(ridge=25).fit(game_rows(con, before=cut))
    r = RAPM(ridge=2000).fit(load_stints(con, before=cut))
    net = {p: r.net(p) for p in r.off}

    # normal minutes per player from train
    norm = con.execute("""SELECT s.player_id, avg(s.seconds)/60.0 nm
        FROM player_game_stats s JOIN (SELECT DISTINCT game_id,game_date FROM nba_games) g USING(game_id)
        WHERE s.game_id LIKE '002%' AND g.game_date <= ? AND s.seconds>0 GROUP BY 1""", [cut]).fetchdf()
    normal_min = dict(zip(norm.player_id, norm.nm))
    pm = con.execute("SELECT game_id, team_id, player_id, seconds FROM player_game_stats WHERE game_id LIKE '002%'").fetchdf()
    meta = con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl, game_date FROM nba_games
        WHERE season='2025-26' AND game_id LIKE '002%' AND wl IS NOT NULL""").fetchdf()
    con.close()

    pm["min"] = pm.seconds / 60.0
    # adjustment per (game,team): Σ net_RAPM * (actual_min - normal_min)/48 over the
    # UNION of rostered players (a missing regular has actual 0, normal ~30 -> big neg)
    adj = {}
    rosters = pm.groupby("team_id")["player_id"].apply(lambda s: set(s)).to_dict()
    played = {(g, t): dict(zip(grp.player_id, grp["min"]))
              for (g, t), grp in pm.groupby(["game_id", "team_id"])}
    for (g, t), mins in played.items():
        a = 0.0
        for p in rosters.get(t, set()):
            nm = normal_min.get(p, 0.0)
            if nm < 8:
                continue
            am = mins.get(p, 0.0)
            a += net.get(p, 0.0) * (am - nm) / 48.0
        adj[(g, t)] = a

    from nba_api.stats.static import teams as T
    ab2id = {v["abbreviation"]: v["id"] for v in T.get_teams()}
    by_game = {}
    for x in meta.itertuples():
        by_game.setdefault(x.game_id, []).append(x)

    y, p_base, p_hybrid, changed = [], [], [], []
    for gid, recs in by_game.items():
        if len(recs) != 2 or recs[0].game_date <= cut:
            continue
        m = recs[0].matchup
        host = m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        h = next((x for x in recs if x.team_abbrev == host), None)
        a = next((x for x in recs if x.team_abbrev != host), None)
        if not h or not a:
            continue
        base_m = tr.pred_margin(h.team_id, a.team_id)
        hadj, aadj = adj.get((gid, h.team_id), 0.0), adj.get((gid, a.team_id), 0.0)
        y.append(int(h.wl == "W"))
        p_base.append(float(sigmoid(base_m / SCALE)))
        p_hybrid.append(float(sigmoid((base_m + hadj - aadj) / SCALE)))
        changed.append(abs(hadj) + abs(aadj) > 2.0)   # meaningful roster change

    y, changed = np.array(y), np.array(changed)
    pb, ph = np.array(p_base), np.array(p_hybrid)
    print(f"test games: {len(y)}  (roster-changed: {changed.sum()})")
    print(f"\nALL games:            base {log_loss(y, pb):.4f}   hybrid {log_loss(y, ph):.4f}")
    if changed.sum() > 20:
        print(f"ROSTER-CHANGED games: base {log_loss(y[changed], pb[changed]):.4f}   "
              f"hybrid {log_loss(y[changed], ph[changed]):.4f}   (n={changed.sum()})")
    print("\nref: market 0.578 | team-avg 0.601 | the injury edge should show on changed games")


if __name__ == "__main__":
    main()
