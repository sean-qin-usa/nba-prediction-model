#!/usr/bin/env python3
"""v2 event-level cut #1: usage softmax — P(shooter=i | offensive lineup).
Conditional logit over the 5 on-floor players with per-player usage propensity
u_i. THIS is information stints can't carry (who absorbs shots), and the
softmax makes usage competitive by construction (handoff II.3.2).
Validation: held-out shooter log loss vs (a) uniform 1/5, (b) unconditional
season usage shares renormalized over the lineup. Beat (b) = the conditional
structure earns its place.
"""
import sys, warnings, time
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, orjson
from nbapred.db import connect
from nbapred.features.cache_index import game_index
from nbapred.features.defense_zone import _game_segments
from nbapred.features.possessions_v2 import _team_ids
from nbapred.features.stints import _elapsed

def collect(limit=None):
    rots=game_index("gamerotation"); pbps=game_index("playbyplayv3")
    gids=sorted(set(rots)&set(pbps))
    if limit: gids=gids[:limit]
    shots=[]  # (off5 tuple, shooter)
    for gid in gids:
        try:
            pbp=orjson.loads(open(pbps[gid],"rb").read())["response"]
            rot=orjson.loads(open(rots[gid],"rb").read())["response"]
        except Exception: continue
        # D100: cached playbyplayv3 `game` has NO homeTeamId -> this was ALWAYS
        # None, so `off5 = h5 if teamId==home else a5` returned the AWAY five for
        # every shot and the `if pid in off5` guard below then silently DROPPED
        # every home-team shot (424,285 of 847,142 kept = 49.9% loss). Derive the
        # ids from the rotation feed (same fix D81 applied to possessions_v2).
        home,_away=_team_ids(rot,pbp)
        if home is None: continue
        segs=_game_segments(rot,pbp)
        if not segs: continue
        t0=np.array([s[0] for s in segs])
        for a in pbp.get("game",{}).get("actions",[]):
            if a.get("actionType") not in ("Made Shot","Missed Shot"): continue
            t=_elapsed(a.get("period"),a.get("clock")); pid=a.get("personId")
            if t is None or not pid: continue
            k=int(np.searchsorted(t0,t,side="right")-1)
            if k<0 or k>=len(segs): continue
            _,_,h5,a5=segs[k]
            off5=h5 if a.get("teamId")==home else a5
            if pid in off5: shots.append((tuple(off5),int(pid)))
    return shots

def main():
    t0=time.time()
    shots=collect()
    print(f"shots {len(shots)}  collect {time.time()-t0:.0f}s",flush=True)
    rng=np.random.default_rng(0)
    idx=rng.permutation(len(shots)); cut=int(len(shots)*0.7)
    train=[shots[i] for i in idx[:cut]]; test=[shots[i] for i in idx[cut:]]
    pids={}
    for l,s in train:
        for p in l: pids.setdefault(p,len(pids))
    P=len(pids)
    # unconditional usage share baseline (train): shots_i / on-floor shots_i
    took=np.zeros(P); onfloor=np.zeros(P)
    for l,s in train:
        for p in l:
            onfloor[pids[p]]+=1
        if s in [x for x in l]: took[pids[s]]+=1
    share=(took+1.0)/(onfloor+5.0)   # smoothed per-shot take rate while on floor
    # conditional logit via numpyro SVI (fast MAP): u_i free, softmax over lineup
    L=np.array([[pids[p] for p in l] for l,s in train])
    S=np.array([ [pids[p] for p in l].index(pids[s]) for l,s in train])
    import jax, jax.numpy as jnp
    import numpyro, numpyro.distributions as dist
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoDelta
    Lj=jnp.asarray(L); Sj=jnp.asarray(S)
    def model():
        u=numpyro.sample("u",dist.Normal(jnp.zeros(P),1.5))
        logits=u[Lj]
        numpyro.sample("s",dist.Categorical(logits=logits),obs=Sj)
    guide=AutoDelta(model)
    svi=SVI(model,guide,numpyro.optim.Adam(0.05),Trace_ELBO())
    res=svi.run(jax.random.PRNGKey(0),1500)
    u=np.array(res.params["u_auto_loc"])
    print(f"fit {time.time()-t0:.0f}s total",flush=True)
    # held-out scoring
    llc=llu=llb=0.0; n=0
    for l,s in test:
        ids=[pids.get(p) for p in l]
        if any(i is None for i in ids) or pids.get(s) is None: continue
        si=ids.index(pids[s])
        # conditional logit
        z=u[ids]; pz=np.exp(z-z.max()); pz=pz/pz.sum()
        llc+=np.log(max(pz[si],1e-9))
        # unconditional shares renormalized
        sh=share[ids]; sh=sh/sh.sum()
        llu+=np.log(max(sh[si],1e-9))
        llb+=np.log(0.2); n+=1
    print(f"held-out shooter log loss (n={n}):")
    print(f"  uniform 1/5        : {-llb/n:.4f}")
    print(f"  uncond usage share : {-llu/n:.4f}")
    print(f"  CONDITIONAL LOGIT  : {-llc/n:.4f}")
    np.savez("data/v2_usage.npz",player_ids=np.array(sorted(pids,key=pids.get)),u=u)
    print("V2USAGE_DONE",flush=True)

if __name__=="__main__": main()
