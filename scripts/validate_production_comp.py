"""Validate integrated production model, REFIT WALK-FORWARD (weekly) exactly as
production runs daily — the frozen-asof artifact in v1 of this script made
rosters decay. OUT sets from the availability oracle.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from nbapred.db import connect
from nbapred.eval.metrics import log_loss
from nbapred.model.production import fit_production
from nbapred.model.composition import CompositionModel

def main():
    con=connect(read_only=True)
    games=con.execute("""SELECT DISTINCT game_id, game_date FROM nba_games
        WHERE season='2025-26' AND game_id LIKE '002%' ORDER BY game_date""").fetchdf()
    cut=games.game_date.quantile(0.6)
    pm=con.execute("""SELECT game_id, team_id, player_id FROM player_game_stats
        WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    played={(g,t):set(grp.player_id) for (g,t),grp in pm.groupby(["game_id","team_id"])}
    meta=con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl, game_date FROM nba_games
        WHERE season='2025-26' AND game_id LIKE '002%' AND wl IS NOT NULL AND game_date > ?
        ORDER BY game_date""",[cut]).fetchdf()
    by={}
    order=[]
    for x in meta.itertuples():
        if x.game_id not in by: order.append(x.game_id)
        by.setdefault(x.game_id,[]).append(x)
    results={w:[] for w in (1.0,0.85,0.7,0.5,0.0)}
    ys=[]
    model=None; comp=None; last_fit=None
    for gid in order:
        recs=by[gid]
        if len(recs)!=2: continue
        m=recs[0].matchup; host=m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        h=next((x for x in recs if x.team_abbrev==host),None); a=next((x for x in recs if x.team_abbrev!=host),None)
        if not h or not a: continue
        gd=h.game_date.date() if hasattr(h.game_date,"date") else h.game_date
        if last_fit is None or (gd-last_fit).days>=7:
            model=fit_production(con,"2025-26",before=gd,w_comp=0.7)
            comp=CompositionModel(con,before=gd)
            last_fit=gd
        outs={}
        for t in (h.team_id,a.team_id):
            pl=played.get((gid,t),set())
            outs[t]={p for p,d in comp.players.items()
                     if d["team_id"]==t and (gd-d["last_played"]).days<=12 and p not in pl}
        ys.append(int(h.wl=="W"))
        cm=comp.margin(h.team_id,a.team_id,outs[h.team_id],outs[a.team_id],gd)
        rm=model.ratings_margin(h.team_id,a.team_id)
        for w in results:
            pmarg=w*cm+(1-w)*rm
            results[w].append(1/(1+np.exp(-pmarg/7.2)))
    y=np.array(ys)
    print(f"n={len(y)} (weekly refit, availability oracle)")
    for w in sorted(results,reverse=True):
        print(f"  w_comp={w:.2f}: {log_loss(y,results[w]):.4f}")
    print("refs: standalone comp 0.5455 | ratings 0.5815 | market(subset) 0.4959")

if __name__=="__main__": main()
