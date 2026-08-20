"""OVERFIT-CAPACITY robustness + decomposition (POST-HOC, clearly labelled).

Everything here is SECONDARY to data/oc_capacity.json and is reported as
sensitivity, not as a headline. The prereg's guard (100 bets/season) and
min-history (5) are the registered values; this file varies them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nbapred.threads  # noqa: E402
nbapred.threads.pin(1)

import numpy as np  # noqa: E402

import oc_capacity as OC  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "oc_robust.json"
NDRAWS_R = 60          # smaller null for the sensitivity grid


def main():
    df, seasons = OC.load()
    K = len(seasons)
    st = OC.build_static(df)
    m_us, p_us = df["m_us"].to_numpy(float), df["p_us"].to_numpy(float)
    payoff, M, keys, win, push, _ = OC.payoff_and_masks(m_us, p_us, st)
    cnt, pay = OC.agg(M, payoff, st)
    res = {"seasons": seasons}

    # ---------------------------------------------- fixed-config comparators
    fixed = {}
    for cfg in [(0, "ALL", "ALL", "ALL"), (3, "ALL", "ALL", "ALL"),
                (3, "ALL", "ALL", "GT18"), (3, "ALL", "ALL", "GT08"),
                (3, "AWAY", "ALL", "GT08"), (3, "DOG", "ALL", "GT18")]:
        i = keys.index(cfg)
        scored = list(range(OC.MIN_HISTORY, K))
        fixed["/".join(map(str, cfg))] = {
            "roi_19": float(pay[i].sum() / cnt[i].sum()), "n_19": float(cnt[i].sum()),
            "roi_scored14": float(pay[i, scored].sum() / cnt[i, scored].sum()),
            "n_scored14": float(cnt[i, scored].sum()),
            "ci_scored14": OC.cluster_mean_t(
                [pay[i, j] / cnt[i, j] for j in scored if cnt[i, j] > 0]),
            "per_season": [float(pay[i, j] / cnt[i, j]) if cnt[i, j] > 0 else None
                           for j in range(K)],
        }
    res["fixed_configs"] = fixed

    # ------------------------------------------ sensitivity to the n-guard
    rng = np.random.default_rng(OC.SEED + 1)
    slate = st["slate"]
    order = np.argsort(slate, kind="stable")
    bounds = np.searchsorted(slate[order], np.arange(slate.max() + 2))

    def null_draws(nd, guard, minhist):
        caps, wfs = [], []
        r2 = np.random.default_rng(OC.SEED + 1)
        for _ in range(nd):
            perm = order.copy()
            for gi in range(len(bounds) - 1):
                a, b = bounds[gi], bounds[gi + 1]
                if b - a > 1:
                    perm[a:b] = r2.permutation(perm[a:b])
            idx = np.empty_like(perm)
            idx[order] = perm
            po, Mn, _, _, _, _ = OC.payoff_and_masks(m_us[idx], p_us[idx], st)
            c, p = OC.agg(Mn, po, st)
            old = OC.MIN_PER_SEASON
            OC.MIN_PER_SEASON = guard
            A = OC.arm_a(c, p, K)
            B = OC.arm_b(c, p, K, minhist)
            OC.MIN_PER_SEASON = old
            caps.append(float(np.mean([a["is_roi"] - a["oos_roi_pooled"]
                                       for a in A if a])))
            wfs.append(float(OC.wf_pooled(B)[0]))
        return np.array(caps), np.array(wfs)

    grid = []
    for guard in [50, 100, 200, 300, 500]:
        old = OC.MIN_PER_SEASON
        OC.MIN_PER_SEASON = guard
        A = OC.arm_a(cnt, pay, K)
        B = OC.arm_b(cnt, pay, K)
        OC.MIN_PER_SEASON = old
        dec = [a["is_roi"] - a["oos_roi_pooled"] for a in A if a]
        wf, nwf = OC.wf_pooled(B)
        nc, nw = null_draws(NDRAWS_R, guard, OC.MIN_HISTORY)
        grid.append({
            "guard": guard,
            "mean_is": float(np.mean([a["is_roi"] for a in A if a])),
            "mean_oos": float(np.mean([a["oos_roi_pooled"] for a in A if a])),
            "capacity": float(np.mean(dec)),
            "capacity_null": float(nc.mean()),
            "capacity_net": float(np.mean(dec) - nc.mean()),
            "hit_rate": int(sum(1 for a in A if a["oos_roi_pooled"] > 0)),
            "mean_sel_n": float(np.mean([a["is_n"] for a in A if a])),
            "wf_roi": float(wf), "wf_n": float(nwf),
            "wf_null_mean": float(nw.mean()), "wf_null_p95": float(np.percentile(nw, 95)),
            "wf_net": float(wf - nw.mean()),
            "wf_ci": OC.cluster_mean_t([b["test_roi"] for b in B]),
            "wf_changes": sum(1 for i in range(1, len(B))
                              if B[i]["cfg"] != B[i - 1]["cfg"]),
            "wf_distinct": len({b["cfg"] for b in B}),
            "wf_cfgs": [list(map(str, keys[b["cfg"]])) for b in B],
        })
        print(f"  guard={guard} cap={grid[-1]['capacity']*100:+.2f} "
              f"null={grid[-1]['capacity_null']*100:+.2f} "
              f"wf={wf*100:+.2f} wfnull={nw.mean()*100:+.2f}", flush=True)
    res["guard_grid"] = grid

    # ------------------------------------------ sensitivity to min-history
    mh = []
    for minhist in [3, 4, 5, 6, 8, 10]:
        B = OC.arm_b(cnt, pay, K, minhist)
        wf, nwf = OC.wf_pooled(B)
        nc, nw = null_draws(NDRAWS_R, OC.MIN_PER_SEASON, minhist)
        mh.append({
            "min_history": minhist, "n_scored": len(B),
            "wf_roi": float(wf), "wf_n": float(nwf),
            "wf_ci": OC.cluster_mean_t([b["test_roi"] for b in B]),
            "wf_null_mean": float(nw.mean()),
            "wf_null_p95": float(np.percentile(nw, 95)),
            "wf_net": float(wf - nw.mean()),
            "seasons_pos": int(sum(1 for b in B if b["test_roi"] > 0)),
            "changes": sum(1 for i in range(1, len(B))
                           if B[i]["cfg"] != B[i - 1]["cfg"]),
        })
        print(f"  minhist={minhist} wf={wf*100:+.2f} "
              f"null={nw.mean()*100:+.2f}", flush=True)
    res["minhist_grid"] = mh

    # ------------------------------------- search-space-size scaling of capacity
    # capacity as a function of how many cells the selector may see
    axes = {"T only (6)": [k for k in keys if k[1] == "ALL" and k[2] == "ALL" and k[3] == "ALL"],
            "T x SIDE (30)": [k for k in keys if k[2] == "ALL" and k[3] == "ALL"],
            "T x SIDE x PHASE (120)": [k for k in keys if k[3] == "ALL"],
            "FULL (600)": keys}
    scal = []
    for nm, sub in axes.items():
        ii = np.array([keys.index(k) for k in sub])
        c2, p2 = cnt[ii], pay[ii]
        A = OC.arm_a(c2, p2, K)
        dec = [a["is_roi"] - a["oos_roi_pooled"] for a in A if a]
        scal.append({"space": nm, "cells": len(sub),
                     "mean_is": float(np.mean([a["is_roi"] for a in A if a])),
                     "mean_oos": float(np.mean([a["oos_roi_pooled"] for a in A if a])),
                     "capacity": float(np.mean(dec)),
                     "hit_rate": int(sum(1 for a in A if a["oos_roi_pooled"] > 0))})
        print(f"  {nm}: cap {scal[-1]['capacity']*100:+.2f}", flush=True)
    res["space_scaling"] = scal

    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
