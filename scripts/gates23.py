import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, glob, orjson
from nbapred.db import connect
from nbapred.ingest.nba_stats import _frames

def load_ptshot(range_name, season):
    out={}
    for f in glob.glob("data/raw/nba_api/ptshot/*.json"):
        d=orjson.loads(open(f,"rb").read())
        if d["params"].get("general_range_nullable")==range_name and d["params"].get("season")==season:
            fr=_frames(d["response"])
            if fr:
                df=list(fr.values())[0]
                for r in df.itertuples():
                    out[int(r.PLAYER_ID)]=dict(fga_freq=float(r.FGA_FREQUENCY or 0),
                        fg3a=float(np.nan_to_num(getattr(r,"FG3A",0) or 0))*float(np.nan_to_num(getattr(r,"GP",1) or 1)),
                        fg3_pct=float(np.nan_to_num(getattr(r,"FG3_PCT",0) or 0)))
    return out

con=connect(read_only=True)
print("G2: creation-split 3P% stability (predict 25-26 from 24-25)")
cs=load_ptshot("Catch and Shoot","2024-25"); pu=load_ptshot("Pullups","2024-25")
a25=con.execute("""SELECT s.player_id, sum(s.thrm) m1, sum(s.thra) a1
    FROM player_game_stats s JOIN nba_games g ON g.game_id=s.game_id AND g.team_id=s.team_id
    WHERE g.season='2025-26' AND s.game_id LIKE '002%' GROUP BY 1""").fetchdf().set_index("player_id")
a24=con.execute("""SELECT s.player_id, sum(s.thrm) m0, sum(s.thra) a0
    FROM player_game_stats s JOIN nba_games g ON g.game_id=s.game_id AND g.team_id=s.team_id
    WHERE g.season='2024-25' AND s.game_id LIKE '002%' GROUP BY 1""").fetchdf().set_index("player_id")
both=a24.join(a25,how="inner"); both=both[(both.a0>=60)&(both.a1>=60)]
rows=[]
for pid,r in both.iterrows():
    c=cs.get(int(pid)); p=pu.get(int(pid))
    if not c or not p or (c["fg3a"]+p["fg3a"])<40: continue
    pooled=r.m0/r.a0
    wt=(c["fg3a"]*c["fg3_pct"]+p["fg3a"]*p["fg3_pct"])/(c["fg3a"]+p["fg3a"])
    rows.append((pooled,wt,r.m1,r.a1))
arr=np.array(rows)
if len(arr)>10:
    def bll(pv,m,a):
        pv=np.clip(pv,.05,.65); return float(np.sum(m*np.log(pv)+(a-m)*np.log(1-pv))/np.sum(a))
    print(f"  players {len(arr)}: pooled {bll(arr[:,0],arr[:,2],arr[:,3]):.4f}  creation-wt {bll(arr[:,1],arr[:,2],arr[:,3]):.4f}")
else: print("  insufficient overlap", len(arr))

print("G3: C&S-dependency x star-out efficiency")
pg=con.execute("""SELECT s.game_id, s.player_id, s.team_id, g.game_date, s.seconds/60.0 mins, s.pts
    FROM player_game_stats s JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING(game_id)
    WHERE s.game_id LIKE '002%' ORDER BY g.game_date""").fetchdf()
pg=pg.sort_values(["player_id","game_date"])
pg["avg_min"]=pg.groupby("player_id")["mins"].transform(lambda s:s.shift(1).rolling(10,min_periods=5).mean())
played=pg[pg.mins>=8].groupby(["game_id","team_id"])["player_id"].apply(set)
stars=pg[pg.avg_min>=28.0]; sbt={}
for r in stars[["player_id","team_id","game_date"]].itertuples():
    sbt.setdefault(r.team_id,[]).append((r.game_date,r.player_id))
rot=pg[(pg.avg_min>=15)&(pg.mins>=12)]
dep=[]
for r in rot.itertuples():
    recent={p for (d0,p) in sbt.get(r.team_id,[]) if 0<(r.game_date-d0).days<=12}
    outs=(recent-played.get((r.game_id,r.team_id),set()))-{r.player_id}
    c=cs.get(int(r.player_id))
    if c is None: continue
    dep.append((c["fga_freq"], 1 if outs else 0, r.pts/max(r.mins,1)))
dp=np.array(dep); hi=dp[:,0]>np.median(dp[:,0])
for lab,mask in (("high-C&S (dependent) ",hi),("low-C&S (self-create)",~hi)):
    on=dp[mask&(dp[:,1]==1),2]; off=dp[mask&(dp[:,1]==0),2]
    print(f"  {lab}: star-out {on.mean():.4f} (n={len(on)}) vs normal {off.mean():.4f} -> {on.mean()/off.mean()-1:+.2%}")
con.close(); print("G23_DONE",flush=True)
