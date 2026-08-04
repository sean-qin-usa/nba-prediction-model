"""A) Four-factors: model eFG/TOV/OREB/FTr per team (opponent-adjusted ridge each),
map predicted factors -> expected efficiency margin (weights fit on train only).
B) GBM challenger (handoff-mandated): HistGradientBoosting on [rating-diff,
comp-diff, rest-diff, home] — does nonlinearity add anything? Both vs production
margin, walk-forward 2025-26, log loss gate.
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

def factor_rows(con, season):
    df=con.execute("""SELECT s.game_id, s.team_id, g.game_date, g.matchup, g.team_abbrev,
        sum(s.fgm) fgm, sum(s.fga) fga, sum(s.thrm) thrm, sum(s.tov) tov,
        sum(s.oreb) oreb, sum(s.dreb) dreb, sum(s.fta) fta, sum(s.ftm) ftm, sum(s.pts) pts
        FROM player_game_stats s JOIN nba_games g ON g.game_id=s.game_id AND g.team_id=s.team_id
        WHERE g.season=? AND s.game_id LIKE '002%' GROUP BY 1,2,3,4,5""",[season]).fetchdf()
    by={}
    for r in df.itertuples(): by.setdefault(r.game_id,[]).append(r)
    rows=[]
    for gid,recs in by.items():
        if len(recs)!=2: continue
        m=recs[0].matchup; host=m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        h=next((x for x in recs if x.team_abbrev==host),None); a=next((x for x in recs if x.team_abbrev!=host),None)
        if not h or not a: continue
        for t,o,is_home in ((h,a,True),(a,h,False)):
            poss=t.fga+0.44*t.fta-t.oreb+t.tov
            if poss<50: continue
            rows.append(dict(date=h.game_date, gid=gid, tid=t.team_id, oid=o.team_id, home=is_home,
                efg=(t.fgm+0.5*t.thrm)/t.fga, tovr=t.tov/poss,
                orbr=t.oreb/(t.oreb+o.dreb), ftr=t.fta/t.fga, ortg=100*t.pts/poss,
                win=int(t.pts>o.pts)))
    rows.sort(key=lambda r:(r["date"],r["gid"]))
    return rows

def main(season="2025-26"):
    con=connect(read_only=True)
    rows=factor_rows(con,season)
    con.close()
    dates=sorted({r["date"] for r in rows})
    cutoff=dates[len(dates)*3//5]
    factors=["efg","tovr","orbr","ftr"]
    # walk-forward: refit weekly; factor models = TeamRatings reused per factor (scale x100)
    y,pf,pr=[],[],[]
    last=None; fms={}; rm=None; W=None
    hist=[]
    games={}  # gid -> (home_row, away_row)
    for r in rows: games.setdefault(r["gid"],[]).append(r)
    order=[gid for gid in dict.fromkeys(r["gid"] for r in rows)]
    for gid in order:
        recs=games[gid]
        if len(recs)!=2: continue
        h=next(x for x in recs if x["home"]); a=next(x for x in recs if not x["home"])
        d=h["date"]
        train=[x for x in hist]
        if last is None or (d-last).days>=7:
            if len(train)>200:
                fms={f:TeamRatings(ridge=25.0,team_home_ridge=None).fit(
                    [(x["tid"],x["oid"],x["home"],x[f]*100) for x in train]) for f in factors}
                rm=TeamRatings(ridge=25.0).fit([(x["tid"],x["oid"],x["home"],x["ortg"]) for x in train])
                # map factors->ortg on train (linear, incl intercept)
                X=np.array([[fms[f].pred_ortg(x["tid"],x["oid"],x["home"]) for f in factors] for x in train])
                yy=np.array([x["ortg"] for x in train])
                W=np.linalg.lstsq(np.c_[X,np.ones(len(X))],yy,rcond=None)[0]
            last=d
        if d>cutoff and fms and rm is not None:
            def eortg(t,o,hm):
                xf=np.array([fms[f].pred_ortg(t,o,hm) for f in factors])
                return float(xf@W[:4]+W[4])
            mf=eortg(h["tid"],a["tid"],True)-eortg(a["tid"],h["tid"],False)
            mr=rm.pred_ortg(h["tid"],a["tid"],True)-rm.pred_ortg(a["tid"],h["tid"],False)
            y.append(h["win"]); pf.append(float(sig(mf/SCALE))); pr.append(float(sig(mr/SCALE)))
        hist.append(h); hist.append(a)
    y=np.array(y)
    print(f"A) FOUR-FACTORS n={len(y)}: ratings {log_loss(y,pr):.4f}  four-factors {log_loss(y,pf):.4f}")
    d=-(y*np.log(np.clip(pf,1e-9,1))+(1-y)*np.log(np.clip(1-np.array(pf),1e-9,1)))
    b=-(y*np.log(np.clip(pr,1e-9,1))+(1-y)*np.log(np.clip(1-np.array(pr),1e-9,1)))
    diff=b-d; rng=np.random.default_rng(0)
    boot=np.array([diff[rng.integers(0,len(diff),len(diff))].mean() for _ in range(2000)])
    lo,hi=np.percentile(boot,[2.5,97.5])
    print(f"   delta {diff.mean():+.5f} CI ({lo:+.5f},{hi:+.5f}) -> {'KEEP' if lo>0 else 'no'}")

if __name__=="__main__": main()
