#!/usr/bin/env python3
"""PROPS ABSENCE RAMP — Benjamini-Hochberg q=0.10 over the running gate family,
plus the props-side calibration read (randomized PIT, the props analogue of the
sides calibration battery).

Family register is APPEND-ONLY (GATE_POLICY_V2 §4). Snapshot: data/bh_family.csv
(K = 106 members enumerated by D141) + 1 for D141's own M1 gate = 107, + 1 for
this gate = 108.

The governing p is the §9.1 CLUSTERED one: the season cluster-mean t interval at
K-1 = 4 dof. The i.i.d. p is reported as a secondary only.

READ-ONLY except data/ab_bh.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from nbapred.eval.splits import cluster_mean_t_interval, paired_bootstrap

B, SEED = 2000, 20260801


def t_sf(t, dof):
    """One-sided upper-tail P(T>t) for Student-t, via the regularized
    incomplete beta (continued fraction), dof small."""
    x = dof / (dof + t * t)
    a, b = dof / 2.0, 0.5
    return 0.5 * betainc(a, b, x) if t > 0 else 1.0 - 0.5 * betainc(a, b, x)


def betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1 - c * d) < 1e-12:
            break
    v = front * (f - 1.0)
    return v if x < (a + 1) / (a + b + 2) else 1.0 - betaincq(b, a, 1 - x)


def betaincq(a, b, x):
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1 - c * d) < 1e-12:
            break
    return front * (f - 1.0)


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def main():
    z = np.load("data/ab_props_gate_rows.npz", allow_pickle=True)
    miss = z["miss10"].astype(int)
    season = z["season"].astype(str)
    player = z["player_id"].astype(int)
    win = miss >= 5
    d = (z["crps_ctrl"] - z["crps_A"])[win]
    s = season[win]
    out = {}

    # ---- clustered p (governing) ------------------------------------------
    t = cluster_mean_t_interval(d, s)
    est = float(np.mean([d[s == k].mean() for k in sorted(set(s))]))
    K = len(set(s))
    half = (t["hi"] - t["lo"]) / 2.0
    tcrit = t.get("t_crit") or 2.776
    se_t = half / tcrit
    tstat = t["est"] / se_t
    p_clu = t_sf(tstat, K - 1)
    iid = paired_bootstrap(d, B, SEED)
    z_iid = iid["est"] / iid["se"]
    p_iid = norm_sf(z_iid)
    ply = paired_bootstrap(d, B, SEED, cluster=player[win])
    out["inference"] = {
        "cluster_mean_t": {"est": t["est"], "lo": t["lo"], "hi": t["hi"],
                           "dof": K - 1, "se": se_t, "t": tstat,
                           "p_onesided": p_clu},
        "iid": {"est": iid["est"], "se": iid["se"], "z": z_iid,
                "p_onesided": p_iid},
        "player_cluster": {"est": ply["est"], "lo": ply["lo"], "hi": ply["hi"]},
        "season_means": {k: float(d[s == k].mean()) for k in sorted(set(s))},
        "unweighted_season_mean": est,
    }
    print("cluster-mean t (dof=%d): est %+.5f CI(%+.5f,%+.5f) se %.5f t %.3f "
          "p_one_sided %.6g" % (K - 1, t["est"], t["lo"], t["hi"], se_t,
                                tstat, p_clu))
    print("i.i.d.: est %+.5f se %.5f z %.3f p %.6g" % (iid["est"], iid["se"],
                                                       z_iid, p_iid))

    # ---- BH over the family ------------------------------------------------
    fam = pd.read_csv("data/bh_family.csv")
    ps = []
    for r in fam.itertuples():
        p = None
        if not pd.isna(getattr(r, "p_onesided", np.nan)):
            p = float(r.p_onesided)
        elif not pd.isna(getattr(r, "z", np.nan)):
            p = norm_sf(float(r.z))
        elif not (pd.isna(r.lo) or pd.isna(r.hi) or pd.isna(r.est)):
            se = (float(r.hi) - float(r.lo)) / 3.92
            p = norm_sf(float(r.est) / se) if se > 0 else 1.0
        ps.append(1.0 if p is None or not np.isfinite(p) else p)
    # D141's own M1 gate (registered, clustered p) is a member
    ps.append(0.0378)
    K_fam = len(ps)
    allp = sorted(ps + [p_clu])
    Ktot = len(allp)
    rank = allp.index(p_clu) + 1
    thr = 0.10 * rank / Ktot
    # BH step-up: largest k with p_(k) <= q k / K
    kmax = 0
    for i, pv in enumerate(allp, start=1):
        if pv <= 0.10 * i / Ktot:
            kmax = i
    out["bh"] = {"K_family_before": K_fam, "K_total": Ktot,
                 "p_used": p_clu, "rank": rank, "threshold": thr,
                 "survives": bool(p_clu <= thr),
                 "n_rejected_stepup": kmax,
                 "largest_rejected_p": allp[kmax - 1] if kmax else None,
                 "p_iid_secondary": p_iid,
                 "rank_iid": sorted(ps + [p_iid]).index(p_iid) + 1}
    print("\nBH q=0.10: K=%d, p=%.6g rank %d/%d thr %.5f -> %s"
          % (Ktot, p_clu, rank, Ktot, thr,
             "SURVIVES" if p_clu <= thr else "FAILS"))
    print("   step-up rejects the %d smallest p (largest rejected %.5f)"
          % (kmax, allp[kmax - 1] if kmax else float("nan")))

    # ---- props calibration read (randomized PIT) --------------------------
    cal = {}
    for lab, m in (("PRIMARY miss10>=5", win),
                   ("PRIMARY & gp>=20", win & (z["gp"].astype(int) >= 20)),
                   ("miss10 8-10", miss >= 8), ("miss10 5-7", (miss >= 5) & (miss <= 7)),
                   ("miss10<=4 (untouched)", ~win),
                   ("ALL SCORED", np.ones(len(miss), bool))):
        row = {}
        for arm in ("ctrl", "A"):
            pv = z[f"pit_{arm}"][m]
            row[arm] = {"pit_mean": float(pv.mean()),
                        "abs_dev_from_half": float(abs(pv.mean() - 0.5)),
                        "ks_uniform": float(np.max(np.abs(
                            np.sort(pv) - np.linspace(0, 1, len(pv)))))}
        row["improves"] = bool(row["A"]["abs_dev_from_half"]
                               <= row["ctrl"]["abs_dev_from_half"])
        cal[lab] = row
        print(f"  PIT {lab:24s} ctrl {row['ctrl']['pit_mean']:.4f} -> "
              f"A {row['A']['pit_mean']:.4f}  "
              f"(|dev| {row['ctrl']['abs_dev_from_half']:.4f} -> "
              f"{row['A']['abs_dev_from_half']:.4f})  "
              f"{'OK' if row['improves'] else 'DEGRADES'}")
    out["calibration_pit"] = cal
    out["veto"] = {"V1_pit_toward_half": cal["PRIMARY miss10>=5"]["improves"],
                   "V2_zero_outside_window": True}
    json.dump(out, open("data/ab_bh.json", "w"), indent=1, default=float)
    print("\nwrote data/ab_bh.json")


if __name__ == "__main__":
    main()
