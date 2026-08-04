"""THE D33 payoff gate: softmax star-out redistribution wired into the prop sim,
scored on held-out star-out player-games by points CRPS (the test flat-lift
failed). Pass -> ships to prop production for October."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from nbapred.db import connect
from nbapred.engine.props import player_rates_from_stats, simulate_player

def crps(s,y):
    s=np.sort(s); n=len(s); return float(np.mean(np.abs(s-y))-0.5*(2*np.arange(1,n+1)-n-1)@s/n**2)

def main(sims=2500,max_eval=900):
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
        if n>=max_eval: break
        recent={p for (d0,p) in sbt.get(r.team_id,[]) if 0<(r.game_date-d0).days<=12}
        outs=(recent-played.get((r.game_id,r.team_id),set()))-{r.player_id}
        if not outs: continue
        star=max(outs,key=lambda p:u.get(p,0.0))
        rates=player_rates_from_stats(con,int(r.player_id),before=r.game_date)
        if not rates or rates["n_games"]<8 or rates["proj_min"]<15: continue
        rn=dict(rates); rn.pop("minutes_hist",None)
        y=r.pts
        b.append(crps(simulate_player(rn,sims,seed=n)["points"],y))
        # softmax lift on this team's rotation pool
        pool={int(p) for p in rot[(rot.team_id==r.team_id)&(rot.game_date==r.game_date)].player_id}|{star}
        S=sum(np.exp(u.get(p,0.0)) for p in pool); Sx=S-np.exp(u.get(star,0.0))
        lift=float(min(S/max(Sx,1e-9),1.5))
        r2=dict(rn)
        for k in ("rate_rim","rate_mid","rate_thr","fta_per_min"): r2[k]=rn[k]*lift
        e.append(crps(simulate_player(r2,sims,seed=n)["points"],y)); n+=1
    con.close()
    d=np.array(b)-np.array(e); rng=np.random.default_rng(0)
    boot=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)])
    lo,hi=np.percentile(boot,[2.5,97.5])
    print(f"star-out n={n}  CRPS base {np.mean(b):.4f}  softmax-lift {np.mean(e):.4f}")
    print(f"delta {d.mean():+.4f} CI ({lo:+.4f},{hi:+.4f}) -> {'SHIP to props' if lo>0 else 'no'}")
    print("GATE_DONE",flush=True)

if __name__=="__main__": main()
