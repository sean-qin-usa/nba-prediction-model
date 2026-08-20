#!/usr/bin/env python3
"""BH q=0.10 recount for ARM R, append-only against the D141 family (K=106).

§9.1 makes the CLUSTERED p primary and demotes the i.i.d. p to a secondary, so
the governing row is the season-mean t (K-1 dof) p. Both are reported.

Writes data/ad_bh.json and appends the new member to data/bh_family.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

FAM = ROOT / "data" / "bh_family.csv"
BAT = ROOT / "data" / "ad_role_v3_battery.json"
Q = 0.10


def bh(ps, q=Q):
    """Benjamini-Hochberg step-up. Returns (thresholds, rejected_mask, k_max)."""
    p = np.asarray(ps, float)
    order = np.argsort(p)
    m = len(p)
    thr = q * (np.arange(1, m + 1)) / m
    below = p[order] <= thr
    kmax = int(np.max(np.where(below)[0]) + 1) if below.any() else 0
    rej = np.zeros(m, bool)
    if kmax:
        rej[order[:kmax]] = True
    return thr, rej, kmax, order


def main():
    fam = pd.read_csv(FAM)
    K0 = len(fam)
    rep = json.loads(BAT.read_text())
    prim = rep["R|ROLE-ACTIVE"]
    p_iid = prim["pooled"]["p_wrongside"]
    t = prim["clustering"]["season_mean_t"]
    # splits.cluster_mean_t_interval returns the interval, not a p — build the
    # one-sided p from the cluster means at K-1 dof, the D141 convention.
    from scipy import stats
    tstat = t["unweighted_mean"] / t["se"] if t["se"] else 0.0
    p_t = float(1 - stats.t.cdf(tstat, t["K"] - 1))
    print(f"  cluster-mean t: mean {t['unweighted_mean']:+.5f} se {t['se']:.5f} "
          f"t={tstat:.3f} dof={t['K']-1}")
    sc = prim["clustering"]["season_cluster_boot"]

    new = dict(name="ARM R GameRotation role-transition props minutes (D142)",
               source="D142", est=prim["pooled"]["est"],
               lo=prim["pooled"]["lo"], hi=prim["pooled"]["hi"],
               se=prim["pooled"]["se"],
               z=prim["pooled"]["est"] / prim["pooled"]["se"]
               if prim["pooled"]["se"] else np.nan,
               p_onesided=p_t, basis="published_p_clustered")
    print(f"family before {K0}; adding 1 -> {K0+1}")
    print(f"  ARM R player-cluster p_wrongside {p_iid:.6f}")
    print(f"  ARM R season-cluster boot CI ({sc['lo']:+.5f},{sc['hi']:+.5f}) "
          f"{'SIG' if sc['sig'] else 'ns'}")
    print(f"  ARM R season-mean t (dof={t['K']-1}) p_onesided {p_t:.6f}")

    out = {"K_before": K0, "K_after": K0 + 1, "q": Q,
           "p_iid": p_iid, "p_clustered": p_t}
    for lab, pv in (("player_cluster_iid", p_iid), ("season_mean_t", p_t)):
        ps = list(fam.p_onesided.fillna(1.0).to_numpy()) + [pv]
        names = list(fam.name) + ["ARM R (this gate)"]
        thr, rej, kmax, order = bh(ps)
        idx = len(ps) - 1
        rank = int(np.where(order == idx)[0][0]) + 1
        my_thr = Q * rank / len(ps)
        out[lab] = dict(p=pv, rank=rank, K=len(ps), threshold=my_thr,
                        rejected=bool(rej[idx]), n_rejected=kmax,
                        largest_rejected_p=float(np.sort(ps)[kmax - 1]) if kmax else None)
        print(f"  BH {lab:20s}: p {pv:.6f} rank {rank}/{len(ps)} "
              f"thr {my_thr:.5f} -> {'SURVIVES' if rej[idx] else 'FAILS'} "
              f"(BH rejects {kmax})")

    fam2 = pd.concat([fam, pd.DataFrame([new])], ignore_index=True)
    fam2.to_csv(FAM, index=False)
    (ROOT / "data" / "ad_bh.json").write_text(json.dumps(out, indent=2, default=float))
    print("AD_BH_DONE")


if __name__ == "__main__":
    main()
