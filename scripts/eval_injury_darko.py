"""Injury impact done right: value missing players by DARKO DPM (stable, multi-
season) instead of noisy single-season RAPM. Base team-avg rating MINUS the DARKO
impact of players whose minutes are below normal. Tests overall + on roster-
changed games — where the injury edge lives. Hypothesis: DARKO's stability fixes
what killed the RAPM version.
Caveat: DARKO snapshot is current (mildly leaky as a talent proxy); acceptable
since DARKO is stable and this tests the MECHANISM.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from nbapred.db import connect
from nbapred.eval.metrics import log_loss
from nbapred.model.team_ratings import TeamRatings, game_rows

SCALE = 7.2
sig = lambda x: 1/(1+np.exp(-np.asarray(x)))


def main():
    con = connect(read_only=True)
    games = con.execute("""SELECT DISTINCT game_id, game_date FROM nba_games
        WHERE season='2025-26' AND game_id LIKE '002%' ORDER BY game_date""").fetchdf()
    cut = games.game_date.quantile(0.6)
    tr = TeamRatings(ridge=25).fit(game_rows(con, before=cut))
    darko = {p: o+d for p,o,d in con.execute(
        "SELECT nba_player_id,o_dpm,d_dpm FROM darko_dpm WHERE snapshot_date=(SELECT max(snapshot_date) FROM darko_dpm)").fetchall()}
    norm = con.execute("""SELECT s.player_id, avg(s.seconds)/60.0 nm FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id,game_date FROM nba_games) g USING(game_id)
        WHERE s.game_id LIKE '002%' AND g.game_date<=? AND s.seconds>0 GROUP BY 1""",[cut]).fetchdf()
    normal_min = dict(zip(norm.player_id, norm.nm))
    pm = con.execute("SELECT game_id, team_id, player_id, seconds FROM player_game_stats WHERE game_id LIKE '002%'").fetchdf()
    meta = con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl, game_date FROM nba_games
        WHERE season='2025-26' AND game_id LIKE '002%' AND wl IS NOT NULL""").fetchdf()
    con.close()

    pm["min"] = pm.seconds/60.0
    rosters = pm.groupby("team_id")["player_id"].apply(set).to_dict()
    played = {(g,t): dict(zip(grp.player_id, grp["min"])) for (g,t),grp in pm.groupby(["game_id","team_id"])}
    adj = {}
    for (g,t), mins in played.items():
        a = 0.0
        for p in rosters.get(t,set()):
            nm = normal_min.get(p,0.0)
            if nm < 12: continue
            am = mins.get(p,0.0)
            a += darko.get(p,0.0)*(am-nm)/48.0   # DARKO impact of minute delta
        adj[(g,t)] = a

    by_game={}
    for x in meta.itertuples(): by_game.setdefault(x.game_id,[]).append(x)
    y,pb,ph,ch=[],[],[],[]
    for gid,recs in by_game.items():
        if len(recs)!=2 or recs[0].game_date<=cut: continue
        m=recs[0].matchup; host=m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        h=next((x for x in recs if x.team_abbrev==host),None); a=next((x for x in recs if x.team_abbrev!=host),None)
        if not h or not a: continue
        bm=tr.pred_margin(h.team_id,a.team_id)
        ha,aa=adj.get((gid,h.team_id),0.0),adj.get((gid,a.team_id),0.0)
        y.append(int(h.wl=="W")); pb.append(float(sig(bm/SCALE))); ph.append(float(sig((bm+ha-aa)/SCALE)))
        ch.append(abs(ha)+abs(aa)>1.5)
    y,ch=np.array(y),np.array(ch); pb,ph=np.array(pb),np.array(ph)
    print(f"test games {len(y)} (roster-changed {ch.sum()})")
    print(f"ALL:     base {log_loss(y,pb):.4f}  DARKO-injury {log_loss(y,ph):.4f}")
    if ch.sum()>20:
        print(f"CHANGED: base {log_loss(y[ch],pb[ch]):.4f}  DARKO-injury {log_loss(y[ch],ph[ch]):.4f} (n={ch.sum()})")
    print("ref: RAPM-injury version was WORSE (0.5727 vs 0.5667 on changed)")


if __name__ == "__main__":
    main()
