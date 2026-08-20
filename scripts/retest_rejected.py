"""Re-test the three MEDIUM false-rejection-risk features with fixed construction:
 A) recency-weighted ratings: half-life 60d (not 21) + ridge scaled by sum(w)/n
 B) opponent-pace props: EB-shrunk pace (K=15 games toward league)
 C) teammate-out: MINUTES-ONLY lift (x1.038), no rate lift
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from nbapred.db import connect
from nbapred.eval.metrics import log_loss
from nbapred.model.team_ratings import TeamRatings
from nbapred.engine.props import player_rates_from_stats, simulate_player, team_pace, apply_pace, LEAGUE_PACE
from scripts.eval_team_ratings import preload, sigmoid, SCALE

def crps(s,y):
    s=np.sort(s); n=len(s); return float(np.mean(np.abs(s-y))-0.5*(2*np.arange(1,n+1)-n-1)@s/n**2)

def testA():
    con=connect(read_only=True); games,fit_rows=preload(con); con.close()
    fdates=np.array([r[0] for r in fit_rows])
    y,pb,pr=[],[],[]
    trb,trw,since=None,None,10**9
    for i,(d,gid,hid,aid,habbr,aabbr,hw) in enumerate(games):
        if i<120: continue
        if trb is None or since>=5:
            cut=np.searchsorted(fdates,d); rows=[r[1:] for r in fit_rows[:cut]]
            trb=TeamRatings(ridge=25.0).fit(rows)
            age=np.array([(d-x).days for x in fdates[:cut]])
            w=0.5**(age/60.0)
            # rescale ridge so effective shrinkage matches unweighted fit
            trw=TeamRatings(ridge=25.0*w.mean()).fit(rows,weights=w)
            since=0
        since+=1
        y.append(hw); pb.append(sigmoid(trb.pred_margin(hid,aid)/SCALE)); pr.append(sigmoid(trw.pred_margin(hid,aid)/SCALE))
    y=np.array(y)
    print(f"A) RECENCY hl=60d ridge-rescaled: base {log_loss(y,pb):.4f}  recency {log_loss(y,pr):.4f}  (old 21d attempt: 0.624)")

def testB():
    con=connect(read_only=True)
    pg=con.execute("""SELECT s.player_id, s.team_id, g.game_date, g.matchup, g.team_abbrev, s.pts
        FROM player_game_stats s JOIN nba_games g ON g.game_id=s.game_id AND g.team_id=s.team_id
        WHERE s.game_id LIKE '002%' AND s.seconds>=720 ORDER BY g.game_date""").fetchdf()
    ab2id={r[1]:r[0] for r in con.execute("SELECT DISTINCT team_id, team_abbrev FROM nba_games WHERE game_id LIKE '002%'").fetchall()}
    stride=max(1,len(pg)//3000); pg=pg.iloc[::stride]
    cache={}; b,pv,n=[],[],0
    def shrunk_pace(tid,date):
        k=(tid,str(date))
        if k not in cache:
            raw=team_pace(con,int(tid),before=date)
            # EB shrink toward league with prior weight ~15 games
            gp=con.execute("""SELECT count(DISTINCT game_id) FROM player_game_stats s
              JOIN (SELECT DISTINCT game_id,game_date FROM nba_games) g USING(game_id)
              WHERE s.team_id=? AND g.game_date<?""",[int(tid),date]).fetchone()[0]
            cache[k]=(raw*gp+LEAGUE_PACE*15)/(gp+15)
        return cache[k]
    for r in pg.itertuples():
        if n>=1000: break
        rates=player_rates_from_stats(con,int(r.player_id),before=r.game_date)
        if not rates or rates["n_games"]<8 or rates["proj_min"]<20: continue
        opp=None
        for tok in r.matchup.replace("vs.","@").split("@"):
            tok=tok.strip()
            if tok and tok!=r.team_abbrev: opp=tok
        oid=ab2id.get(opp)
        if oid is None: continue
        rn=dict(rates); rn.pop("minutes_hist",None)
        y=r.pts
        b.append(crps(simulate_player(rn,2500,seed=n)["points"],y))
        adj=apply_pace(rn, shrunk_pace(oid,r.game_date), shrunk_pace(r.team_id,r.game_date))
        pv.append(crps(simulate_player(adj,2500,seed=n)["points"],y))
        n+=1
    con.close()
    d=np.array(b)-np.array(pv); rng=np.random.default_rng(0)
    boot=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)])
    lo,hi=np.percentile(boot,[2.5,97.5])
    print(f"B) PACE shrunk: base {np.mean(b):.4f} pace {np.mean(pv):.4f} delta {d.mean():+.4f} CI({lo:+.4f},{hi:+.4f}) -> {'KEEP' if lo>0 else 'no'}")

def testC():
    # minutes-only lift on star-out (reuse ablate_teammate_out plumbing, lift minutes x1.038 only)
    import scripts.ablate_teammate_out as T
    # inline lightweight version
    con=connect(read_only=True)
    pg=con.execute("""SELECT s.game_id, s.player_id, s.team_id, g.game_date, s.seconds, s.pts,
        s.rima+s.mida+s.thra shots FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING(game_id)
        WHERE s.game_id LIKE '002%' ORDER BY g.game_date""").fetchdf()
    pg["mins"]=pg.seconds/60.0
    pg=pg.sort_values(["player_id","game_date"])
    pg["avg10"]=pg.groupby("player_id")["mins"].transform(lambda s:s.shift(1).rolling(10,min_periods=5).mean())
    played=pg[pg.mins>=8].groupby(["game_id","team_id"])["player_id"].apply(set)
    stars=pg[pg.avg10>=28.0]
    sbt={}
    for r in stars[["player_id","team_id","game_date"]].itertuples():
        sbt.setdefault(r.team_id,[]).append((r.game_date,r.player_id))
    games=pg[["game_id","team_id","game_date"]].drop_duplicates()
    star_out={}
    for r in games.itertuples():
        recent={p for (d0,p) in sbt.get(r.team_id,[]) if 0<(r.game_date-d0).days<=12}
        star_out[(r.game_id,r.team_id)]=recent-played.get((r.game_id,r.team_id),set())
    rot=pg[(pg.avg10>=15)&(pg.mins>=12)].copy()
    rot["sa"]=[len(star_out.get((g,t),set())-{p})>0 for g,t,p in zip(rot.game_id,rot.team_id,rot.player_id)]
    cut=pg.game_date.quantile(0.6)
    test=rot[(rot.game_date>cut)&(rot.sa)]
    test=test.iloc[::max(1,len(test)//600)]
    b,e,n=[],[],0
    for r in test.itertuples():
        if n>=500: break
        rates=player_rates_from_stats(con,int(r.player_id),before=r.game_date)
        if not rates or rates["n_games"]<8 or rates["proj_min"]<15: continue
        rn=dict(rates); rn.pop("minutes_hist",None)
        y=r.pts
        b.append(crps(simulate_player(rn,2500,seed=n)["points"],y))
        r2=dict(rn); r2["proj_min"]=min(rn["proj_min"]*1.038,44)
        e.append(crps(simulate_player(r2,2500,seed=n)["points"],y))
        n+=1
    con.close()
    d=np.array(b)-np.array(e); rng=np.random.default_rng(0)
    boot=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)])
    lo,hi=np.percentile(boot,[2.5,97.5])
    print(f"C) TEAMMATE-OUT minutes-only: star-out n={n} base {np.mean(b):.4f} adj {np.mean(e):.4f} delta {d.mean():+.4f} CI({lo:+.4f},{hi:+.4f}) -> {'KEEP' if lo>0 else 'no'}")

if __name__=="__main__":
    testA(); testB(); testC()
