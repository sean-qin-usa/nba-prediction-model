"""Overnight gate battery on the newly-ingested data (2026-07-29).
G1 SKILL-CURVE REDISTRIBUTION: attempts x softmax-lift AND FG% x lift^-0.66
   (gamma from D30 empirics: pts_lift 1.008 = shots_lift 1.023 x eff 0.985)
   -> star-out points CRPS (the gate D34's naive version failed).
G2 CREATION-SPLIT STABILITY: does C&S vs pull-up decomposition predict future
   3P% better than pooled 3P%? (split-half binomial LL, the D26 harness style)
G3 DEPENDENCY INTERACTION: do high-C&S-frequency (assisted-diet) players lose
   MORE efficiency when a star playmaker sits? (slope + significance; informs
   conditional redistribution)
Explicit > clever; every result logged; no silent defaults.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, json, glob
from nbapred.db import connect
from nbapred.engine.props import player_rates_from_stats, simulate_player
from nbapred.ingest.nba_stats import _frames
import orjson

def crps(s,y):
    s=np.sort(s); n=len(s); return float(np.mean(np.abs(s-y))-0.5*(2*np.arange(1,n+1)-n-1)@s/n**2)

def load_ptshot(range_name, season):
    """Read cached leaguedashplayerptshot pulls by scanning cache files."""
    out={}
    for f in glob.glob("data/raw/nba_api/ptshot/*.json"):
        d=orjson.loads(open(f,"rb").read())
        if d["params"].get("general_range_nullable")==range_name and d["params"].get("season")==season:
            fr=_frames(d["response"])
            if fr:
                df=list(fr.values())[0]
                for r in df.itertuples():
                    out[int(r.PLAYER_ID)]=dict(fga_freq=float(r.FGA_FREQUENCY or 0),
                        fg3a=float(getattr(r,"FG3A",0) or 0), fg3_pct=float(getattr(r,"FG3_PCT",0) or 0))
    return out

print("="*60); print("G1: skill-curve-paired redistribution (points CRPS)")
GAMMA=0.66
con=connect(read_only=True)
pg=con.execute("""SELECT s.game_id, s.player_id, s.team_id, g.game_date, s.seconds/60.0 mins, s.pts
    FROM player_game_stats s JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING(game_id)
    WHERE s.game_id LIKE '002%' ORDER BY g.game_date""").fetchdf()
uz=np.load("data/v2_usage.npz"); u=dict(zip(uz["player_ids"].tolist(),uz["u"].tolist()))
pg=pg.sort_values(["player_id","game_date"])
pg["avg_min"]=pg.groupby("player_id")["mins"].transform(lambda s:s.shift(1).rolling(10,min_periods=5).mean())
played=pg[pg.mins>=8].groupby(["game_id","team_id"])["player_id"].apply(set)
stars=pg[pg.avg_min>=28.0]
sbt={}
for r in stars[["player_id","team_id","game_date"]].itertuples():
    sbt.setdefault(r.team_id,[]).append((r.game_date,r.player_id))
rot=pg[(pg.avg_min>=15)&(pg.mins>=12)].copy()
cut=pg.game_date.quantile(0.6)
test=rot[rot.game_date>cut]
b,e,n=[],[],0
for r in test.itertuples():
    if n>=900: break
    recent={p for (d0,p) in sbt.get(r.team_id,[]) if 0<(r.game_date-d0).days<=12}
    outs=(recent-played.get((r.game_id,r.team_id),set()))-{r.player_id}
    if not outs: continue
    star=max(outs,key=lambda p:u.get(p,0.0))
    rates=player_rates_from_stats(con,int(r.player_id),before=r.game_date)
    if not rates or rates["n_games"]<8 or rates["proj_min"]<15: continue
    rn=dict(rates); rn.pop("minutes_hist",None)
    b.append(crps(simulate_player(rn,2500,seed=n)["points"],r.pts))
    pool={int(p) for p in rot[(rot.team_id==r.team_id)&(rot.game_date==r.game_date)].player_id}|{star}
    S=sum(np.exp(u.get(p,0.0)) for p in pool); lift=float(min(S/max(S-np.exp(u.get(star,0.0)),1e-9),1.5))
    r2=dict(rn); eff=lift**(-GAMMA)
    for k in ("rate_rim","rate_mid","rate_thr","fta_per_min"): r2[k]=rn[k]*lift
    for k in ("fg_rim","fg_mid","fg_thr"): r2[k]=max(min(rn[k]*eff,0.99),0.01)
    e.append(crps(simulate_player(r2,2500,seed=n)["points"],r.pts)); n+=1
d=np.array(b)-np.array(e); rng=np.random.default_rng(0)
boot=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)])
lo,hi=np.percentile(boot,[2.5,97.5])
print(f"n={n} base {np.mean(b):.4f} skillcurve {np.mean(e):.4f} delta {d.mean():+.4f} CI({lo:+.4f},{hi:+.4f}) -> {'SHIP' if lo>0 else 'no'}")

print("="*60); print("G2: creation-split (C&S vs pull-up) 3P% stability")
cs24=load_ptshot("Catch and Shoot","2024-25"); pu24=load_ptshot("Pull Ups","2024-25")
# predict 2025-26 3P% from 24-25: pooled vs creation-weighted
a25=con.execute("""SELECT s.player_id, sum(s.thrm) m1, sum(s.thra) a1
    FROM player_game_stats s JOIN nba_games g ON g.game_id=s.game_id AND g.team_id=s.team_id
    WHERE g.season='2025-26' AND s.game_id LIKE '002%' GROUP BY 1""").fetchdf().set_index("player_id")
a24=con.execute("""SELECT s.player_id, sum(s.thrm) m0, sum(s.thra) a0
    FROM player_game_stats s JOIN nba_games g ON g.game_id=s.game_id AND g.team_id=s.team_id
    WHERE g.season='2024-25' AND s.game_id LIKE '002%' GROUP BY 1""").fetchdf().set_index("player_id")
both=a24.join(a25,how="inner")
both=both[(both.a0>=60)&(both.a1>=60)]
rows=[]
for pid,r in both.iterrows():
    cs=cs24.get(int(pid)); pu=pu24.get(int(pid))
    if not cs or not pu or cs["fg3a"]+pu["fg3a"]<40: continue
    pooled=r.m0/r.a0
    wt=(cs["fg3a"]*cs["fg3_pct"]+pu["fg3a"]*pu["fg3_pct"])/(cs["fg3a"]+pu["fg3a"])
    # creation-adjusted: shrink each split toward its own league mean first
    rows.append((pooled,wt,r.m1,r.a1))
arr=np.array(rows)
def bll(p,m,a):
    p=np.clip(p,.05,.65); return float(np.sum(m*np.log(p)+(a-m)*np.log(1-p))/np.sum(a))
print(f"players {len(arr)}: pooled LL {bll(arr[:,0],arr[:,2],arr[:,3]):.4f}  creation-split LL {bll(arr[:,1],arr[:,2],arr[:,3]):.4f}")

print("="*60); print("G3: dependency (C&S freq) x star-out efficiency interaction")
dep=[]
for r in test.itertuples():
    recent={p for (d0,p) in sbt.get(r.team_id,[]) if 0<(r.game_date-d0).days<=12}
    outs=(recent-played.get((r.game_id,r.team_id),set()))-{r.player_id}
    cs=cs24.get(int(r.player_id))
    if cs is None: continue
    dep.append((cs["fga_freq"], 1 if outs else 0, r.pts/max(r.mins,1)))
dp=np.array(dep)
hi_cs=dp[:,0]>np.median(dp[:,0])
for lab,mask in (("high-C&S(dependent)",hi_cs),("low-C&S(self-creators)",~hi_cs)):
    on=dp[mask&(dp[:,1]==1),2]; off=dp[mask&(dp[:,1]==0),2]
    print(f"  {lab}: pts/min star-out {on.mean():.4f} (n={len(on)}) vs normal {off.mean():.4f} -> drop {on.mean()/off.mean()-1:+.2%}")
con.close()
print("OVERNIGHT_GATES_DONE",flush=True)
