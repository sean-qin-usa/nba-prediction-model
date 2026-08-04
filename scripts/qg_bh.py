#!/usr/bin/env python3
"""BENJAMINI-HOCHBERG q=0.10 over the append-only gate family, for the three
queued gates (GATE_POLICY_V2 §4 / §9.1).

The governing p is the CLUSTERED one — the season cluster-mean t at K-1 dof
(§9.1 demotes the i.i.d. p to a secondary). Family = data/bh_family.csv,
recounted append-only, plus the members this session adds at PRE-REGISTRATION
time (one per gate, winners and losers alike).

usage: qg_bh.py
Writes data/qg_bh.json and appends the new members to data/bh_family.csv.
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

FAM = ROOT / "data" / "bh_family.csv"
Q = 0.10


def t_sf(t, dof):
    """One-sided upper-tail P(T > t) for Student-t, via the incomplete beta."""
    t = abs(float(t))
    x = dof / (dof + t * t)
    return 0.5 * _betainc(dof / 2.0, 0.5, x)


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
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
        if abs(1.0 - c * d) < 1e-12:
            break
    return front * (f - 1.0)


def main():
    fam = pd.read_csv(FAM)
    k_before = len(fam)
    new = json.loads((ROOT / "data" / "qg_bh_new_members.json").read_text())
    rows = []
    for m in new:
        p = t_sf(m["t"], m["dof"]) if m.get("t") is not None else m["p"]
        rows.append({"name": m["name"], "source": m["source"], "est": m["est"],
                     "lo": m.get("lo"), "hi": m.get("hi"), "se": m.get("se"),
                     "z": m["t"], "p_onesided": p,
                     "basis": "season_cluster_mean_t"})
    add = pd.DataFrame(rows)
    fam2 = pd.concat([fam, add], ignore_index=True)
    fam2.to_csv(FAM, index=False)
    p = fam2["p_onesided"].to_numpy(float)
    K = len(p)
    order = np.argsort(p)
    ranks = np.empty(K, int)
    ranks[order] = np.arange(1, K + 1)
    thr = Q * ranks / K
    # step-up: largest rank whose p <= q*rank/K
    ok = p[order] <= Q * np.arange(1, K + 1) / K
    kmax = int(np.max(np.where(ok)[0]) + 1) if ok.any() else 0
    rejected = set(order[:kmax].tolist())
    out = {"K_before": int(k_before), "K": int(K), "q": Q,
           "n_rejected_step_up": kmax,
           "largest_rejected_p": float(p[order][kmax - 1]) if kmax else None,
           "members": []}
    for i, r in add.iterrows():
        j = int(fam2.index[(fam2["name"] == r["name"])][-1])
        out["members"].append({
            "name": r["name"], "p": float(r["p_onesided"]),
            "rank": int(ranks[j]), "threshold": float(thr[j]),
            "survives": bool(j in rejected)})
    Path("data/qg_bh.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    print("QG_BH_DONE", flush=True)


if __name__ == "__main__":
    main()
