"""D32 follow-up: star-out shot redistribution. Predict rotation players' SHOT
COUNTS in star-out games three ways:
  a) baseline: trailing shots avg (no adjustment)
  b) flat lift x1.020 (D30's global empirical lift)
  c) SOFTMAX renorm: lift = S/(S - exp(u_star)) over the team's rotation —
     star-specific magnitude from the fitted usage propensities
Score: Poisson log-lik of actual shots. (c)>(b) means the fitted usage model
prices redistribution better than folklore.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from nbapred.db import connect

def main():
    con=connect(read_only=True)
    pg=con.execute("""SELECT s.game_id, s.player_id, s.team_id, g.game_date,
        s.seconds/60.0 mins, s.rima+s.mida+s.thra shots
        FROM player_game_stats s JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING(game_id)
        WHERE s.game_id LIKE '002%' ORDER BY g.game_date""").fetchdf()
    con.close()
    uz=np.load("data/v2_usage.npz")
    u=dict(zip(uz["player_ids"].tolist(), uz["u"].tolist()))

    pg=pg.sort_values(["player_id","game_date"])
    pg["avg_min"]=pg.groupby("player_id")["mins"].transform(lambda s:s.shift(1).rolling(10,min_periods=5).mean())
    pg["avg_shots"]=pg.groupby("player_id")["shots"].transform(lambda s:s.shift(1).rolling(10,min_periods=5).mean())
    played=pg[pg.mins>=8].groupby(["game_id","team_id"])["player_id"].apply(set)
    stars=pg[pg.avg_min>=28.0]
    sbt={}
    for r in stars[["player_id","team_id","game_date"]].itertuples():
        sbt.setdefault(r.team_id,[]).append((r.game_date,r.player_id))
    rot=pg[(pg.avg_min>=15)&(pg.mins>=12)&pg.avg_shots.notna()].copy()

    def star_out_set(gid,tid,gd):
        recent={p for (d0,p) in sbt.get(tid,[]) if 0<(gd-d0).days<=12}
        return recent-played.get((gid,tid),set())

    rows=[]
    for r in rot.itertuples():
        outs=star_out_set(r.game_id,r.team_id,r.game_date)-{r.player_id}
        if not outs: continue
        star=max(outs,key=lambda p:u.get(p,0.0))
        # softmax renorm over team rotation present (top by avg_min incl star)
        team_rot=rot[(rot.team_id==r.team_id)&(rot.game_date==r.game_date)].player_id.tolist()
        pool=set(team_rot)|{star}
        S=sum(np.exp(u.get(p,0.0)) for p in pool)
        Sx=S-np.exp(u.get(star,0.0))
        lift=min(S/max(Sx,1e-9),1.5)
        rows.append((r.avg_shots, r.shots, lift))
    a=np.array(rows)
    print(f"star-out rotation player-games: {len(a)}  mean softmax lift {a[:,2].mean():.3f} (flat=1.020)")
    def pois_ll(pred, y):
        pred=np.clip(pred,0.2,None)
        return float(np.mean(y*np.log(pred)-pred))
    base=pois_ll(a[:,0],a[:,1])
    flat=pois_ll(a[:,0]*1.020,a[:,1])
    soft=pois_ll(a[:,0]*a[:,2],a[:,1])
    print(f"Poisson LL (higher=better): baseline {base:.5f}  flat x1.02 {flat:.5f}  SOFTMAX {soft:.5f}")
    # bootstrap softmax vs flat
    d=(a[:,1]*np.log(np.clip(a[:,0]*a[:,2],0.2,None))-a[:,0]*a[:,2]) - \
      (a[:,1]*np.log(np.clip(a[:,0]*1.020,0.2,None))-a[:,0]*1.020)
    rng=np.random.default_rng(0)
    boot=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)])
    lo,hi=np.percentile(boot,[2.5,97.5])
    print(f"softmax-vs-flat delta {d.mean():+.5f} CI ({lo:+.5f},{hi:+.5f}) -> {'KEEP softmax' if lo>0 else ('flat suffices' if hi>0 else 'softmax WORSE')}")

if __name__=="__main__": main()
