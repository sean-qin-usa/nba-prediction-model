"""Three-way blend (four-factors + composition + ratings) on late-2025-26, plus
four-factors replication on 2024-25. Equal-weight blend first (no tuning), then
a few fixed weight sets for reference.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from nbapred.db import connect
from nbapred.eval.metrics import log_loss
from nbapred.model.team_ratings import TeamRatings
from nbapred.model.composition import CompositionModel
from scripts.test_ff_gbm import factor_rows
SCALE=7.2
sig=lambda x:1/(1+np.exp(-np.asarray(x)))
FACTORS=["efg","tovr","orbr","ftr"]

def run(season, do_comp=True):
    con=connect(read_only=True)
    rows=factor_rows(con,season)
    pm=con.execute("""SELECT game_id, team_id, player_id FROM player_game_stats
        WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    played={(g,t):set(grp.player_id) for (g,t),grp in pm.groupby(["game_id","team_id"])}
    dates=sorted({r["date"] for r in rows}); cutoff=dates[len(dates)*3//5]
    games={}
    for r in rows: games.setdefault(r["gid"],[]).append(r)
    order=[g for g in dict.fromkeys(r["gid"] for r in rows)]
    y=[]; M={"ff":[], "rt":[], "cp":[]}
    last=None; fms={}; rm=None; W=None; comp=None; hist=[]
    for gid in order:
        recs=games[gid]
        if len(recs)!=2: continue
        h=next(x for x in recs if x["home"]); a=next(x for x in recs if not x["home"])
        d=h["date"]; dd=d.date() if hasattr(d,"date") else d
        if last is None or (dd-last).days>=7:
            if len(hist)>200:
                fms={f:TeamRatings(ridge=25.0,team_home_ridge=None).fit(
                    [(x["tid"],x["oid"],x["home"],x[f]*100) for x in hist]) for f in FACTORS}
                rm=TeamRatings(ridge=25.0).fit([(x["tid"],x["oid"],x["home"],x["ortg"]) for x in hist])
                X=np.array([[fms[f].pred_ortg(x["tid"],x["oid"],x["home"]) for f in FACTORS] for x in hist])
                yy=np.array([x["ortg"] for x in hist])
                W=np.linalg.lstsq(np.c_[X,np.ones(len(X))],yy,rcond=None)[0]
                if do_comp: comp=CompositionModel(con,before=dd)
            last=dd
        if d>cutoff and fms and rm is not None:
            def eortg(t,o,hm):
                xf=np.array([fms[f].pred_ortg(t,o,hm) for f in FACTORS])
                return float(xf@W[:4]+W[4])
            y.append(h["win"])
            M["ff"].append(eortg(h["tid"],a["tid"],True)-eortg(a["tid"],h["tid"],False))
            M["rt"].append(rm.pred_ortg(h["tid"],a["tid"],True)-rm.pred_ortg(a["tid"],h["tid"],False))
            if do_comp and comp:
                outs={}
                for t in (h["tid"],a["tid"]):
                    pl=played.get((gid,t),set())
                    outs[t]={p for p,d0 in comp.players.items()
                             if d0["team_id"]==t and (dd-d0["last_played"]).days<=12 and p not in pl}
                M["cp"].append(comp.margin(h["tid"],a["tid"],outs[h["tid"]],outs[a["tid"]],dd))
        hist.append(h); hist.append(a)
    con.close()
    y=np.array(y)
    print(f"{season} n={len(y)}: ff {log_loss(y,sig(np.array(M['ff'])/SCALE)):.4f}  "
          f"ratings {log_loss(y,sig(np.array(M['rt'])/SCALE)):.4f}", end="")
    if do_comp and M["cp"]:
        cp=np.array(M["cp"]); ff=np.array(M["ff"]); rt=np.array(M["rt"])
        print(f"  comp {log_loss(y,sig(cp/SCALE)):.4f}")
        for w in ((1/3,1/3,1/3),(0.45,0.35,0.2),(0.5,0.5,0.0)):
            m=w[0]*ff+w[1]*cp+w[2]*rt
            print(f"   blend ff/comp/rt {w}: {log_loss(y,sig(m/SCALE)):.4f}")
    else:
        print()

if __name__=="__main__":
    run("2025-26"); run("2024-25", do_comp=False)
