#!/usr/bin/env python3
"""Run the full v1 Bayesian skill fit (GPU) and validate: 2K betas ordering,
net-RAPM correlation with DARKO, posterior summaries saved."""
import sys, warnings, time
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from nbapred.db import connect
from nbapred.model.v1_fit import build_dataset, fit, BINOM_DIMS, POIS_DIMS

def main():
    con = connect(read_only=True)
    data = build_dataset(con)
    darko = dict(con.execute("SELECT nba_player_id, o_dpm+d_dpm FROM darko_dpm "
        "WHERE snapshot_date=(SELECT max(snapshot_date) FROM darko_dpm)").fetchall())
    con.close()
    print(f"players {data['n_players']}  stints {len(data['stint_margin'])}", flush=True)
    t0 = time.time()
    mcmc = fit(data)
    print(f"fit time {time.time()-t0:.0f}s", flush=True)
    s = mcmc.get_samples()
    print("\n2K trust betas (posterior mean +- sd):")
    for dim in list(BINOM_DIMS) + list(POIS_DIMS):
        b = np.asarray(s[f"beta_{dim}"])
        print(f"  {dim:5} {b.mean():+.3f} +- {b.std():.3f}")
    net = np.asarray(s["net"]).mean(axis=0)
    ids = data["player_ids"]
    common = [(net[i], darko[int(p)]) for i, p in enumerate(ids) if int(p) in darko]
    a = np.array(common)
    print(f"\nBayes net vs DARKO corr: {np.corrcoef(a[:,0],a[:,1])[0,1]:.3f} (n={len(a)})")
    np.savez("data/v1_posterior.npz", player_ids=ids, net=net,
             **{f"theta_{d}": np.asarray(s[f"a_{d}"]).mean(axis=0) for d in
                list(BINOM_DIMS) + list(POIS_DIMS)})
    print("posterior saved -> data/v1_posterior.npz")

if __name__ == "__main__":
    main()
