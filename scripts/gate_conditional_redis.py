"""Conditional (C&S-tilted) redistribution gate — from G3's twice-replicated
finding: star-out gains concentrate in high-C&S players. Tilt the softmax lift
by C&S frequency and gate on star-out SHOT-count Poisson LL vs the untilted
softmax lift (D33's champion).
  lift_i = 1 + (L-1) * tilt_i,  tilt_i = cs_i / mean(cs_team)  (capped [0.3, 2.5])
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, glob, orjson
from nbapred.db import connect
from nbapred.ingest.nba_stats import _frames

def load_cs(season):
    out={}
    for f in glob.glob("data/raw/nba_api/ptshot/*.json"):
        d=orjson.loads(open(f,"rb").read())
        if d["params"].get("general_range_nullable")=="Catch and Shoot" and d["params"].get("season")==season:
            df=list(_frames(d["response"]).values())[0]
            for r in df.itertuples(): out[int(r.PLAYER_ID)]=float(np.nan_to_num(r.FGA_FREQUENCY or 0))
    return out

def main():
    con=connect(read_only=True)
    pg=con.execute("""SELECT s.game_id, s.player_id, s.team_id, g.game_date,
        s.seconds/60.0 mins, s.rima+s.mida+s.thra shots
        FROM player_game_stats s JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING(game_id)
        WHERE s.game_id LIKE '002%' ORDER BY g.game_date""").fetchdf()
    con.close()
    uz=np.load("data/v2_usage.npz"); u=dict(zip(uz["player_ids"].tolist(),uz["u"].tolist()))
    cs=load_cs("2024-25")
    pg=pg.sort_values(["player_id","game_date"])
    pg["avg_min"]=pg.groupby("player_id")["mins"].transform(lambda s:s.shift(1).rolling(10,min_periods=5).mean())
    pg["avg_shots"]=pg.groupby("player_id")["shots"].transform(lambda s:s.shift(1).rolling(10,min_periods=5).mean())
    played=pg[pg.mins>=8].groupby(["game_id","team_id"])["player_id"].apply(set)
    stars=pg[pg.avg_min>=28.0]; sbt={}
    for r in stars[["player_id","team_id","game_date"]].itertuples():
        sbt.setdefault(r.team_id,[]).append((r.game_date,r.player_id))
    rot=pg[(pg.avg_min>=15)&(pg.mins>=12)&pg.avg_shots.notna()].copy()

    rows=[]
    for r in rot.itertuples():
        recent={p for (d0,p) in sbt.get(r.team_id,[]) if 0<(r.game_date-d0).days<=12}
        outs=(recent-played.get((r.game_id,r.team_id),set()))-{r.player_id}
        if not outs: continue
        star=max(outs,key=lambda p:u.get(p,0.0))
        team_now=rot[(rot.team_id==r.team_id)&(rot.game_date==r.game_date)]
        pool={int(p) for p in team_now.player_id}|{star}
        S=sum(np.exp(u.get(p,0.0)) for p in pool)
        L=float(min(S/max(S-np.exp(u.get(star,0.0)),1e-9),1.5))
        # tilt by C&S freq relative to teammates present
        cs_i=cs.get(int(r.player_id))
        team_cs=[cs.get(int(p)) for p in team_now.player_id if cs.get(int(p)) is not None]
        if cs_i is None or len(team_cs)<3: continue
        tilt=float(np.clip(cs_i/max(np.mean(team_cs),1e-6),0.3,2.5))
        rows.append((r.avg_shots, r.shots, L, 1+(L-1)*tilt))
    a=np.array(rows)
    print(f"star-out player-games {len(a)}  mean uniform-lift {a[:,2].mean():.3f}  mean tilted {a[:,3].mean():.3f}")
    def ll(pred,y):
        pred=np.clip(pred,0.2,None); return y*np.log(pred)-pred
    lu=ll(a[:,0]*a[:,2],a[:,1]); lt=ll(a[:,0]*a[:,3],a[:,1])
    d=lt-lu; rng=np.random.default_rng(0)
    boot=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)])
    lo,hi=np.percentile(boot,[2.5,97.5])
    print(f"Poisson LL uniform {lu.mean():.5f}  tilted {lt.mean():.5f}")
    print(f"tilted-vs-uniform delta {d.mean():+.5f} CI ({lo:+.5f},{hi:+.5f}) -> {'KEEP tilt' if lo>0 else 'uniform stands'}")
    print("COND_REDIS_DONE",flush=True)

if __name__=="__main__": main()
