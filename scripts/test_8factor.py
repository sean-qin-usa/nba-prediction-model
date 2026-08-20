"""Deeper decomposition (Sean: as much as possible): 8 factors using our zone
data — rim/mid/3 accuracy, rim/3 shot-share, TOV rate, OREB rate, FT rate.
Same protocol as four-factors; gate vs the 4F model.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from nbapred.db import connect
from nbapred.eval.metrics import log_loss
from nbapred.model.team_ratings import TeamRatings
SCALE=7.2
sig=lambda x:1/(1+np.exp(-np.asarray(x)))

def rows8(con, season):
    df=con.execute("""SELECT s.game_id, s.team_id, g.game_date, g.matchup, g.team_abbrev,
        sum(s.rima) rima, sum(s.rimm) rimm, sum(s.mida) mida, sum(s.midm) midm,
        sum(s.thra) thra, sum(s.thrm) thrm, sum(s.tov) tov, sum(s.oreb) oreb,
        sum(s.dreb) dreb, sum(s.fta) fta, sum(s.ftm) ftm, sum(s.fga) fga, sum(s.pts) pts
        FROM player_game_stats s JOIN nba_games g ON g.game_id=s.game_id AND g.team_id=s.team_id
        WHERE g.season=? AND s.game_id LIKE '002%' GROUP BY 1,2,3,4,5""",[season]).fetchdf()
    by={}
    for r in df.itertuples(): by.setdefault(r.game_id,[]).append(r)
    out=[]
    for gid,recs in by.items():
        if len(recs)!=2: continue
        m=recs[0].matchup; host=m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        h=next((x for x in recs if x.team_abbrev==host),None); a=next((x for x in recs if x.team_abbrev!=host),None)
        if not h or not a: continue
        for t,o,hm in ((h,a,True),(a,h,False)):
            poss=t.fga+0.44*t.fta-t.oreb+t.tov
            if poss<50 or t.fga<40: continue
            out.append(dict(date=h.game_date,gid=gid,tid=t.team_id,oid=o.team_id,home=hm,
                rim_acc=t.rimm/max(t.rima,1), mid_acc=t.midm/max(t.mida,1), thr_acc=t.thrm/max(t.thra,1),
                rim_sh=t.rima/t.fga, thr_sh=t.thra/t.fga,
                tovr=t.tov/poss, orbr=t.oreb/(t.oreb+o.dreb), ftr=t.fta/t.fga,
                ortg=100*t.pts/poss, win=int(t.pts>o.pts)))
    out.sort(key=lambda r:(r["date"],r["gid"]))
    return out

F8=["rim_acc","mid_acc","thr_acc","rim_sh","thr_sh","tovr","orbr","ftr"]
F4=["efg4","tovr","orbr","ftr"]

def run(season):
    con=connect(read_only=True); rows=rows8(con,season); con.close()
    for r in rows:  # derive 4F efg from zones for the baseline
        r["efg4"]=(r["rim_acc"]*r["rim_sh"]+r["mid_acc"]*(1-r["rim_sh"]-r["thr_sh"])+1.5*r["thr_acc"]*r["thr_sh"])
    dates=sorted({r["date"] for r in rows}); cutoff=dates[len(dates)*3//5]
    games={}
    for r in rows: games.setdefault(r["gid"],[]).append(r)
    order=[g for g in dict.fromkeys(r["gid"] for r in rows)]
    def walk(fs):
        y,P=[],[]; last=None; fms={}; W=None; hist=[]
        for gid in order:
            recs=games[gid]
            if len(recs)!=2: continue
            h=next(x for x in recs if x["home"]); a=next(x for x in recs if not x["home"])
            d=h["date"]; dd=d.date() if hasattr(d,"date") else d
            if last is None or (dd-last).days>=7:
                if len(hist)>200:
                    fms={f:TeamRatings(ridge=25.0,team_home_ridge=None).fit(
                        [(x["tid"],x["oid"],x["home"],x[f]*100) for x in hist]) for f in fs}
                    X=np.array([[fms[f].pred_ortg(x["tid"],x["oid"],x["home"]) for f in fs] for x in hist])
                    yy=np.array([x["ortg"] for x in hist])
                    W=np.linalg.lstsq(np.c_[X,np.ones(len(X))],yy,rcond=None)[0]
                last=dd
            if d>cutoff and fms:
                def e(t,o,hm):
                    xf=np.array([fms[f].pred_ortg(t,o,hm) for f in fs])
                    return float(xf@W[:len(fs)]+W[len(fs)])
                y.append(h["win"]); P.append(float(sig((e(h["tid"],a["tid"],True)-e(a["tid"],h["tid"],False))/SCALE)))
            hist.append(h); hist.append(a)
        return np.array(y), np.array(P)
    y4,p4=walk(F4); y8,p8=walk(F8)
    print(f"{season} n={len(y4)}: 4-factor {log_loss(y4,p4):.4f}   8-factor {log_loss(y8,p8):.4f}")
    d=(-(y4*np.log(np.clip(p4,1e-9,1))+(1-y4)*np.log(np.clip(1-p4,1e-9,1))))-(
       -(y8*np.log(np.clip(p8,1e-9,1))+(1-y8)*np.log(np.clip(1-p8,1e-9,1))))
    rng=np.random.default_rng(0)
    boot=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)])
    lo,hi=np.percentile(boot,[2.5,97.5])
    print(f"   8F vs 4F delta {d.mean():+.5f} CI ({lo:+.5f},{hi:+.5f}) -> {'KEEP 8F' if lo>0 else ('KEEP 4F (8F worse)' if hi<0 else 'tie')}")

if __name__=="__main__":
    run("2025-26"); run("2024-25")
