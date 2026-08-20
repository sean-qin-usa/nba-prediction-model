"""Sean's progression-arc idea (handoff II.1 age curves, never built): does
age-adjusting LAST season's rates improve prediction of THIS season's rates?
Gate: fit per-dimension age deltas on a random half of two-season players,
test on the other half — age-adjusted 2024-25 rate vs raw 2024-25 rate
predicting 2025-26, binomial LL/attempt. Also prints the raw aging profile.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from nbapred.db import connect
from scripts.validate_bayes_updating import binom_ll

DIMS={"thr":("thrm","thra"),"ft":("ftm","fta"),"rim":("rimm","rima"),"mid":("midm","mida")}

def agg(con, season):
    return con.execute(f"""SELECT s.player_id,
        sum(s.thrm) thrm, sum(s.thra) thra, sum(s.ftm) ftm, sum(s.fta) fta,
        sum(s.rimm) rimm, sum(s.rima) rima, sum(s.midm) midm, sum(s.mida) mida
        FROM player_game_stats s JOIN nba_games g ON g.game_id=s.game_id AND g.team_id=s.team_id
        WHERE g.season=? AND s.game_id LIKE '002%' GROUP BY 1""",[season]).fetchdf().set_index("player_id")

def main():
    con=connect(read_only=True)
    a24,a25=agg(con,"2024-25"),agg(con,"2025-26")
    age=dict(con.execute("SELECT nba_player_id, age FROM darko_dpm "
        "WHERE snapshot_date=(SELECT max(snapshot_date) FROM darko_dpm)").fetchall())
    con.close()
    both=a24.join(a25,how="inner",lsuffix="_0",rsuffix="_1")
    both["age"]=[float(age.get(int(p)) or np.nan)-1.0 for p in both.index]  # age during 24-25
    both=both.dropna(subset=["age"])
    rng=np.random.default_rng(0)
    half=rng.random(len(both))<0.5
    print(f"two-season players: {len(both)} (fit {half.sum()} / test {(~half).sum()})")
    for dim,(mk,at) in DIMS.items():
        m=both[(both[f"{at}_0"]>=30)&(both[f"{at}_1"]>=30)].copy()
        hm=half[:len(both)][both.index.isin(m.index)] if False else rng.random(len(m))<0.5
        r0=m[f"{mk}_0"]/m[f"{at}_0"]; r1=m[f"{mk}_1"]/m[f"{at}_1"]
        d=(r1-r0).to_numpy(); ag=m["age"].to_numpy()
        # aging profile (all players, descriptive)
        for lo,hi in ((19,24),(24,28),(28,32),(32,45)):
            sel=(ag>=lo)&(ag<hi)
            if sel.sum()>5: print(f"  {dim} age {lo}-{hi}: mean delta {d[sel].mean():+.4f} (n={sel.sum()})")
        # gate: quadratic age curve fit on half, applied to other half
        fit_sel=hm; te=~hm
        A=np.c_[np.ones(fit_sel.sum()), ag[fit_sel], ag[fit_sel]**2]
        beta=np.linalg.lstsq(A,d[fit_sel],rcond=None)[0]
        adj=beta[0]+beta[1]*ag[te]+beta[2]*ag[te]**2
        p_raw=np.clip(r0.to_numpy()[te],1e-3,1-1e-3)
        p_adj=np.clip(r0.to_numpy()[te]+adj,1e-3,1-1e-3)
        n1=m[f"{at}_1"].to_numpy()[te]; m1=m[f"{mk}_1"].to_numpy()[te]
        ll_raw=binom_ll(m1,n1,p_raw).sum()/n1.sum()
        ll_adj=binom_ll(m1,n1,p_adj).sum()/n1.sum()
        print(f"  {dim}: predict 25-26 | raw last-season {ll_raw:.4f}  AGE-ADJ {ll_adj:.4f}  delta {ll_adj-ll_raw:+.5f}\n")

if __name__=="__main__": main()
