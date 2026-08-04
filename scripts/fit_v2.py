#!/usr/bin/env python3
"""v2 FIT (first cut): possession-level Poisson-points RAPM on GPU."""
import sys, warnings, time
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from nbapred.db import connect

def main(max_poss=200_000):
    con=connect(read_only=True)
    # D107: 002 filter added (the D29 run absorbed 131 playoff + 11 preseason
    # games). SUPERSEDED by scripts/cg_v2_sufficiency.py — D108 re-ran this
    # model on the rebuilt table and D29's rejection REVERSED (+0.541 vs the
    # registered -0.059). Kept for provenance only.
    df=con.execute("SELECT off_lineup, def_lineup, points FROM possessions_v2 "
                   "WHERE game_id LIKE '002%'").fetchdf()
    darko=dict(con.execute("SELECT nba_player_id, o_dpm+d_dpm FROM darko_dpm "
        "WHERE snapshot_date=(SELECT max(snapshot_date) FROM darko_dpm)").fetchall())
    con.close()
    pids={}; O=[];D=[];Y=[]
    for r in df.itertuples():
        try:
            o=[int(x) for x in r.off_lineup.split(",")]; d=[int(x) for x in r.def_lineup.split(",")]
        except Exception: continue
        if len(o)!=5 or len(d)!=5: continue
        for p in o+d: pids.setdefault(p,len(pids))
        O.append([pids[p] for p in o]); D.append([pids[p] for p in d]); Y.append(min(r.points,4))
    if len(Y)>max_poss:
        idx=np.random.default_rng(0).choice(len(Y),max_poss,replace=False)
        O=[O[i] for i in idx]; D=[D[i] for i in idx]; Y=[Y[i] for i in idx]
    O=np.array(O); D=np.array(D); Y=np.array(Y,float); P=len(pids)
    print(f"fit possessions {len(Y)}  players {P}", flush=True)
    import jax.numpy as jnp, numpyro
    import numpyro.distributions as dist
    from jax import random
    from numpyro.infer import MCMC, NUTS
    def model():
        mu=numpyro.sample("mu",dist.Normal(0.1,0.5))
        so=numpyro.sample("so",dist.HalfNormal(0.1))
        sd=numpyro.sample("sd",dist.HalfNormal(0.1))
        with numpyro.plate("po",P): off=numpyro.sample("off",dist.Normal(0.0,so))
        with numpyro.plate("pd",P): deff=numpyro.sample("deff",dist.Normal(0.0,sd))
        lograte=mu+off[jnp.asarray(O)].sum(1)-deff[jnp.asarray(D)].sum(1)
        numpyro.sample("y",dist.Poisson(jnp.exp(lograte)),obs=jnp.asarray(Y))
    t0=time.time()
    mcmc=MCMC(NUTS(model,max_tree_depth=8),num_warmup=400,num_samples=500,progress_bar=True)
    mcmc.run(random.PRNGKey(0))
    print(f"fit {time.time()-t0:.0f}s", flush=True)
    s=mcmc.get_samples()
    net=np.asarray(s["off"]).mean(0)+np.asarray(s["deff"]).mean(0)
    inv={v:k for k,v in pids.items()}
    a=np.array([(net[i],darko[inv[i]]) for i in range(P) if inv[i] in darko])
    print(f"V2 net vs DARKO corr: {np.corrcoef(a[:,0],a[:,1])[0,1]:.3f} (n={len(a)})")
    try:
        v1=np.load("data/v1_posterior.npz")
        v1net=dict(zip(v1["player_ids"].tolist(), v1["net"].tolist()))
        b=np.array([(net[i],v1net[inv[i]]) for i in range(P) if inv[i] in v1net])
        print(f"V2 net vs V1 net corr: {np.corrcoef(b[:,0],b[:,1])[0,1]:.3f}")
    except Exception as e: print("v1 compare skipped:", e)
    np.savez("data/v2_posterior.npz", player_ids=np.array([inv[i] for i in range(P)]), net=net)
    print("V2_FIT_DONE", flush=True)

if __name__=="__main__": main()
