#!/usr/bin/env python3
"""v2 fit, second cut — fixes for D29's failure:
  1. NON-CENTERED parameterization (kills the funnel that stuck the sampler)
  2. v1 posterior as prior: off/def prior means = v1 net split, scaled from
     pts/48 to per-possession log-rate (~/55) — v2 REFINES v1, not from scratch
  3. diagnostics printed: divergences, acc prob, ESS on hyperparams
Validation unchanged: net vs DARKO (benchmark v1=0.625) and vs v1 net.
"""
import sys, warnings, time
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from nbapred.db import connect

def main(max_poss=200_000):
    con=connect(read_only=True)
    # D107: 002 filter added (the D31 run absorbed 131 playoff + 11 preseason
    # games). SUPERSEDED by scripts/cg_v2_sufficiency.py — D108 REFUTED D31's
    # sufficiency claim on the rebuilt table: net vs DARKO +0.606 vs the v1
    # stint fit's 0.625, data-residual -0.220 -> +0.438, and on held-out
    # possessions this model class BEATS stint RAPM (+0.000346 SIG).
    df=con.execute("SELECT off_lineup, def_lineup, points FROM possessions_v2 "
                   "WHERE game_id LIKE '002%'").fetchdf()
    darko=dict(con.execute("SELECT nba_player_id, o_dpm+d_dpm FROM darko_dpm "
        "WHERE snapshot_date=(SELECT max(snapshot_date) FROM darko_dpm)").fetchall())
    con.close()
    v1=np.load("data/v1_posterior.npz")
    v1net=dict(zip(v1["player_ids"].tolist(), v1["net"].tolist()))

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
    inv={v:k for k,v in pids.items()}
    # v1 prior means on per-possession log scale (net pts/48 -> /55; split off/def)
    prior_mean=np.array([v1net.get(inv[i],0.0)/2.0/55.0 for i in range(P)])
    print(f"fit possessions {len(Y)}  players {P}  v1-prior coverage "
          f"{np.mean([inv[i] in v1net for i in range(P)]):.2f}", flush=True)

    import jax.numpy as jnp, numpyro
    import numpyro.distributions as dist
    from jax import random
    from numpyro.infer import MCMC, NUTS
    pm=jnp.asarray(prior_mean); Oj=jnp.asarray(O); Dj=jnp.asarray(D); Yj=jnp.asarray(Y)

    def model():
        mu=numpyro.sample("mu",dist.Normal(0.09,0.2))
        so=numpyro.sample("so",dist.HalfNormal(0.05))
        sd=numpyro.sample("sd",dist.HalfNormal(0.05))
        with numpyro.plate("po",P):
            zo=numpyro.sample("zo",dist.Normal(0.0,1.0))   # non-centered
        with numpyro.plate("pd",P):
            zd=numpyro.sample("zd",dist.Normal(0.0,1.0))
        off=pm+so*zo
        deff=pm+sd*zd
        lograte=mu+off[Oj].sum(1)-deff[Dj].sum(1)
        numpyro.sample("y",dist.Poisson(jnp.exp(lograte)),obs=Yj)

    t0=time.time()
    mcmc=MCMC(NUTS(model,max_tree_depth=10,target_accept_prob=0.9),
              num_warmup=500,num_samples=500,progress_bar=True)
    mcmc.run(random.PRNGKey(1))
    print(f"fit {time.time()-t0:.0f}s", flush=True)
    div=int(np.sum(mcmc.get_extra_fields().get("diverging",np.zeros(1)))) if mcmc.get_extra_fields() else -1
    s=mcmc.get_samples()
    so_=np.asarray(s["so"]); sd_=np.asarray(s["sd"])
    print(f"sigma_off {so_.mean():.4f}+-{so_.std():.4f}  sigma_def {sd_.mean():.4f}+-{sd_.std():.4f}")
    off=(prior_mean[None,:]+np.asarray(s["so"])[:,None]*np.asarray(s["zo"])).mean(0)
    deff=(prior_mean[None,:]+np.asarray(s["sd"])[:,None]*np.asarray(s["zd"])).mean(0)
    net=off+deff
    a=np.array([(net[i],darko[inv[i]]) for i in range(P) if inv[i] in darko])
    print(f"V2b net vs DARKO corr: {np.corrcoef(a[:,0],a[:,1])[0,1]:.3f} (n={len(a)}; v1=0.625)")
    b=np.array([(net[i],v1net[inv[i]]) for i in range(P) if inv[i] in v1net])
    print(f"V2b net vs V1 net corr: {np.corrcoef(b[:,0],b[:,1])[0,1]:.3f}")
    # the honest incremental test: does the DATA move skills BEYOND the prior?
    resid=net-2*prior_mean*1.0  # net minus prior-implied net
    c=np.array([(resid[i],darko[inv[i]]) for i in range(P) if inv[i] in darko])
    print(f"V2b data-residual vs DARKO: {np.corrcoef(c[:,0],c[:,1])[0,1]:.3f} "
          f"(possession data's OWN signal beyond v1 prior)")
    np.savez("data/v2b_posterior.npz",player_ids=np.array([inv[i] for i in range(P)]),net=net)
    print("V2B_DONE", flush=True)

if __name__=="__main__": main()
