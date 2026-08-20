#!/usr/bin/env python3
"""NEWSTRAT BATTERY — GATE_POLICY_V2 §8-§11 on the pre-registered arms, plus the
matched-q diagnostic that isolates SELECTOR QUALITY from the selection layer.

Prereg data/newstrat_prereg.md sha256
db293ad8548a2e7099586a66847d6d1a71d8e83e4ab171e0917584980e9f1fc7.

MATCHED-q: the same 12 pre-registered cells read with q FROZEN instead of
walk-forward-selected.  The cut-point is still learned PIT-safely on seasons
[0,k) and applied to season k, so nothing is in-sample; only the selection layer
is removed.  This answers "is the SELECTOR better?" separately from "did the
walk-forward pick well?".

READ-ONLY.  No DB.  No default changed.
  python3 scripts/ns_battery.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from ns_score import (                                            # noqa: E402
    ALL_IDS, ARMS, MIN_PER_SEASON, QS, WIN, cmt, icc_deff, load,
    payoff_of, selectors, t_crit,
)

SEED = 20260805
B_BLOCK = 2000
OUT = ROOT / "data" / "ns_battery.json"
ERAS = {"2007-08": "K-A", "2008-09": "K-A", "2009-10": "K-A", "2010-11": "K-A",
        "2011-12": "K-B", "2012-13": "K-B", "2013-14": "K-B",
        "2014-15": "K-C", "2015-16": "K-C", "2016-17": "K-C", "2017-18": "K-C",
        "2018-19": "K-C", "2019-20": "K-D", "2020-21": "K-D",
        "2021-22": "K-E", "2022-23": "K-E", "2023-24": "K-E", "2024-25": "K-E",
        "2025-26": "K-E"}
# D174 bkp_ladder.json repriced.rows labels (artifact, NOT D174 §15's prose)
LADDER = {"2012-13": ("MEASURED", 0.1210), "2013-14": ("MEASURED", 0.1013),
          "2014-15": ("MEASURED", 0.1136), "2015-16": ("MEASURED", 0.1186),
          "2016-17": ("MEASURED", 0.1223), "2017-18": ("MEASURED", 0.1631),
          "2018-19": ("MEASURED", 0.1107),
          "2019-20": ("EXTRAPOLATED", 0.2153), "2020-21": ("EXTRAPOLATED", 0.2153),
          "2021-22": ("EXTRAPOLATED", 0.2153), "2022-23": ("EXTRAPOLATED", 0.2153),
          "2023-24": ("MEASURED", 0.4261), "2024-25": ("MEASURED", 0.2527),
          "2025-26": ("MEASURED", 0.2154)}
DROI_PER_PT = 1.909 * 0.0317276          # D163/D174 conversion, ROI per spread pt


def chi2_sf(x, k):
    """Upper tail of chi-square, series form (no scipy)."""
    if x <= 0:
        return 1.0
    if k % 2 == 0:
        t = np.exp(-x / 2.0)
        s = t
        for i in range(1, k // 2):
            t *= x / (2.0 * i)
            s += t
        return float(min(1.0, s))
    from math import erfc, sqrt, exp, pi
    s = erfc(sqrt(x / 2.0))
    t = sqrt(2.0 * x / pi) * exp(-x / 2.0)
    for i in range(1, (k - 1) // 2 + 1):
        s += t
        t *= x / (2.0 * i + 1.0)
    return float(min(1.0, s))


def wf_fixed_q(sel, pay, s_i, K, minh, q, clv, season_of):
    """Walk-forward with q FROZEN: cut-point learned on [0,k), applied to k."""
    steps = []
    for k in range(minh, K):
        win_m = s_i < k
        cut = float(np.quantile(sel[win_m], 1.0 - q))
        mt = (s_i == k) & (sel >= cut)
        if mt.sum() < 5:
            continue
        steps.append({"k": k, "season": season_of[k], "cut": cut,
                      "n": int(mt.sum()), "pay": pay[mt], "clv": clv[mt],
                      "roi": float(pay[mt].mean()),
                      "clv_mean": float(np.nanmean(clv[mt])),
                      "mask": mt})
    return steps


def block_boot(pay, dates, B=B_BLOCK, seed=SEED):
    """7-day calendar block bootstrap on per-bet payoffs (GATE_POLICY_V2 §8.1)."""
    d = pd.to_datetime(pd.Series(dates))
    blk = ((d - d.min()).dt.days // 7).to_numpy()
    keys = np.unique(blk)
    idx = {k: np.where(blk == k)[0] for k in keys}
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        sel = np.concatenate([idx[keys[j]] for j in pick])
        out[b] = pay[sel].mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def era_decomp(per_season_roi, seasons):
    """DerSimonian-Laird Q / I^2 / tau on the era means."""
    by = {}
    for s, r in zip(seasons, per_season_roi):
        by.setdefault(ERAS[s], []).append(r)
    codes = [c for c in sorted(by) if len(by[c]) >= 1]
    mu = np.array([np.mean(by[c]) for c in codes])
    nn = np.array([len(by[c]) for c in codes], float)
    sd = np.array([np.std(by[c], ddof=1) if len(by[c]) > 1 else np.nan
                   for c in codes])
    var = np.where(np.isfinite(sd), (sd ** 2) / nn, np.nan)
    ok = np.isfinite(var) & (var > 0)
    out = {"eras": codes, "means": mu.tolist(), "n_seasons": nn.tolist()}
    if ok.sum() >= 2:
        w = 1.0 / var[ok]
        mbar = float((w * mu[ok]).sum() / w.sum())
        Q = float((w * (mu[ok] - mbar) ** 2).sum())
        dof = int(ok.sum() - 1)
        I2 = max(0.0, (Q - dof) / Q) * 100 if Q > 0 else 0.0
        tau2 = max(0.0, (Q - dof) / (w.sum() - (w ** 2).sum() / w.sum()))
        out.update({"Q": Q, "dof": dof, "p_Q": chi2_sf(Q, dof), "I2": I2,
                    "tau": float(np.sqrt(tau2)), "pooled": mbar})
        out["verdict"] = ("ERA-STABLE" if I2 < 50 and out["p_Q"] > 0.10
                          else ("ERA-SPECIFIC"
                                if (mu[ok] > 0).any() and (mu[ok] < 0).any()
                                else "ERA-CONDITIONAL"))
    else:
        out["verdict"] = "UNDETERMINED (too few eras with >1 season)"
    return out


def bh(pvals, alpha=0.05):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    crit = alpha * (np.arange(1, m + 1)) / m
    passed = p[order] <= crit
    kmax = np.where(passed)[0].max() + 1 if passed.any() else 0
    rej = set(order[:kmax].tolist())
    return {"m": m, "alpha": alpha,
            "crit": crit.tolist(), "sorted_p": p[order].tolist(),
            "rejected_idx": sorted(rej), "n_rejected": len(rej)}


def main():
    d = load()
    res = {"prereg_sha256": "db293ad8548a2e7099586a66847d6d1a71d8e83e4ab171e0917584980e9f1fc7"}
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    allseasons = sorted(d.season.unique())
    report8 = [s for s in allseasons if s >= "2018-19"]
    prior = json.load(open(ROOT / "data" / "ns_score.json"))

    for label, keep, minh in [("PRIMARY_REPORT8", report8, 3),
                              ("SECONDARY_POOL19", allseasons, 5)]:
        sub = d[d.season.isin(keep)].reset_index(drop=True)
        seasons = sorted(sub.season.unique())
        s_i = sub.season.map({s: i for i, s in enumerate(seasons)}).to_numpy()
        K = len(seasons)
        om = sub.open_margin.to_numpy(float)
        act = sub.margin_actual.to_numpy(float)
        m_us = sub.m_us.to_numpy(float)
        close = sub.close_margin.to_numpy(float)
        pay, win, push, bh_ = payoff_of(m_us, om, act)
        clv = np.where(bh_, 1.0, -1.0) * (close - om)
        sels = selectors(m_us, sub.m_us_blind.to_numpy(float),
                         sub.pred_dm.to_numpy(float),
                         sub.retmin_h.to_numpy(float),
                         sub.retmin_a.to_numpy(float), om)
        R = {"K": K, "matched_q": {}, "battery": {}}

        # ---------------- MATCHED-q DIAGNOSTIC -------------------------------
        log(f"\n=== {label} — MATCHED-q DIAGNOSTIC (q frozen, cut-point still "
            f"learned on [0,k) only) ===")
        log(f"    {'q':<6}{'arm':<11}{'n':>7}{'ROI%':>9}{'cover%':>9}"
            f"{'CLV pts':>10}   {'K-1 t CI on ROI':<24}{'paired ROI vs A0':<24}"
            f"{'paired CLV vs A0':<20}")
        for q in QS:
            steps = {a: wf_fixed_q(sels[a], pay, s_i, K, minh, q, clv, seasons)
                     for a in ALL_IDS}
            a0r = [x["roi"] for x in steps["A0_EDGE"]]
            a0c = [x["clv_mean"] for x in steps["A0_EDGE"]]
            R["matched_q"][str(q)] = {}
            for a in ALL_IDS:
                st = steps[a]
                n = sum(x["n"] for x in st)
                allpay = np.concatenate([x["pay"] for x in st])
                allclv = np.concatenate([x["clv"] for x in st])
                mask = np.zeros(len(sub), bool)
                for x in st:
                    mask |= x["mask"]
                cov = float(win[mask][~push[mask]].mean())
                per = [x["roi"] for x in st]
                perc = [x["clv_mean"] for x in st]
                ci = cmt(per)
                cic = cmt(perc)
                pr = cmt([per[i] - a0r[i] for i in range(len(per))]) \
                    if a != "A0_EDGE" else None
                pc = cmt([perc[i] - a0c[i] for i in range(len(perc))]) \
                    if a != "A0_EDGE" else None
                R["matched_q"][str(q)][a] = {
                    "n": int(n), "roi": float(allpay.mean()), "cover": cov,
                    "clv": float(np.nanmean(allclv)),
                    "per_fold_roi": per, "per_fold_clv": perc,
                    "ci_roi": ci, "ci_clv": cic,
                    "paired_roi_vs_A0": pr, "paired_clv_vs_A0": pc,
                }
                ps = (f"{100*pr['mean']:+6.2f}[{100*pr['lo']:+6.2f},"
                      f"{100*pr['hi']:+6.2f}]{'S' if pr['sig'] else 'n'}") \
                    if pr else "      (benchmark)      "
                cs = (f"{pc['mean']:+6.3f}[{pc['lo']:+6.3f},{pc['hi']:+6.3f}]"
                      f"{'S' if pc['sig'] else 'n'}") if pc else ""
                log(f"    {q:<6.2f}{a:<11}{n:>7}{100*allpay.mean():>+9.2f}"
                    f"{100*cov:>9.3f}{np.nanmean(allclv):>+10.3f}   "
                    f"[{100*ci['lo']:+7.2f},{100*ci['hi']:+7.2f}]"
                    f"{'SIG' if ci['sig'] else ' ns'}  {ps} {cs}")

        # ---------------- V3 BATTERY on the walk-forward arms ----------------
        log(f"\n=== {label} — V3 BATTERY (GATE_POLICY_V2 §8-§11) ===")
        P = prior["frames"][label]
        folds = P["scored_folds"]
        a0per = P["arms"]["A0_EDGE"]["per_fold_roi"]
        pvals, pnames = [], []
        for a in ALL_IDS:
            A = P["arms"][a]
            per = A["per_fold_roi"]
            # rolling-origin IS the walk-forward: sign consistency across folds
            signs = [1 if x > 0 else -1 for x in per]
            # LOSO on the pooled per-fold mean and on the paired delta
            loso = [float(np.mean([per[j] for j in range(len(per)) if j != i]))
                    for i in range(len(per))]
            if a != "A0_EDGE":
                dl = [per[i] - a0per[i] for i in range(len(per))]
                loso_p = [float(np.mean([dl[j] for j in range(len(dl))
                                         if j != i])) for i in range(len(dl))]
                flip = any((np.mean(dl) > 0) != (x > 0) for x in loso_p)
            else:
                dl, loso_p, flip = None, None, None
            # block bootstrap on the realised per-bet payoffs
            steps = None
            era = era_decomp(per, folds)
            B = {"rolling_origin_folds": len(per),
                 "folds_positive": int(sum(1 for x in per if x > 0)),
                 "sign_consistent": bool(len(set(signs)) == 1),
                 "loso_pooled_lo": float(min(loso)), "loso_pooled_hi": float(max(loso)),
                 "era": era}
            if dl is not None:
                B.update({"paired_mean": float(np.mean(dl)),
                          "paired_loso_lo": float(min(loso_p)),
                          "paired_loso_hi": float(max(loso_p)),
                          "paired_loso_sign_flip": bool(flip),
                          "paired_folds_positive":
                              int(sum(1 for x in dl if x > 0))})
                pvals.append(P["arms"][a]["p_null_s_paired"])
                pnames.append(a)
            R["battery"][a] = B
            _ = steps
            log(f"    {a:<11} folds+={B['folds_positive']}/{len(per)}  "
                f"sign_consistent={B['sign_consistent']}  "
                f"LOSO pooled [{100*min(loso):+.2f},{100*max(loso):+.2f}]  "
                f"era {era['verdict']}"
                + (f"  | paired {100*B['paired_mean']:+.2f} folds+="
                   f"{B['paired_folds_positive']}/{len(dl)} LOSO "
                   f"[{100*min(loso_p):+.2f},{100*max(loso_p):+.2f}]"
                   f" flip={B['paired_loso_sign_flip']}" if dl is not None else ""))
            for c, m_ in zip(era["eras"], era["means"]):
                log(f"                era {c}: {100*m_:+.2f}%")

        # block bootstrap needs the realised bets: recompute from matched q=0.10
        for a in ALL_IDS:
            st = wf_fixed_q(sels[a], pay, s_i, K, minh, 0.10, clv, seasons)
            allpay = np.concatenate([x["pay"] for x in st])
            dts = np.concatenate([sub.game_date.to_numpy()[x["mask"]] for x in st])
            lo, hi = block_boot(allpay, dts)
            icc, deff = icc_deff(
                allpay,
                np.concatenate([np.full(x["n"], x["season"]) for x in st]))
            R["battery"][a]["block_boot_q10"] = {"lo": lo, "hi": hi}
            R["battery"][a]["icc_q10"] = icc
            R["battery"][a]["deff_q10"] = deff
            log(f"    {a:<11} block-bootstrap(7d) on the q=0.10 cell: "
                f"[{100*lo:+.2f},{100*hi:+.2f}]  ICC={icc:+.5f} DEFF={deff:.3f}")

        R["bh_across_family"] = bh(pvals) if pvals else None
        R["bh_names"] = pnames
        if pvals:
            log(f"    BH across the family of {len(pvals)} paired p-values "
                f"{[round(x,3) for x in pvals]}: "
                f"{R['bh_across_family']['n_rejected']} rejected at alpha=0.05")

        # ---------------- line-shopping LEVEL twin ---------------------------
        tw = {}
        for a in ALL_IDS:
            A = P["arms"][a]
            gain = np.array([LADDER.get(s, ("EXTRAPOLATED", 0.2153))[1]
                             for s in folds])
            nper = np.array(A["per_fold_n"], float)
            add = float((gain * nper).sum() / nper.sum()) * DROI_PER_PT
            tw[a] = {"bet_weighted_gain_pts": float((gain * nper).sum() / nper.sum()),
                     "roi_with_ladder": A["pooled_roi"] + add,
                     "delta_roi": add,
                     "labels": {s: LADDER.get(s, ("EXTRAPOLATED", 0.2153))[0]
                                for s in folds}}
        R["ladder_twin"] = tw
        log(f"\n    LINE-SHOPPING LEVEL TWIN (k=5 haircut ladder, D174 artifact "
            f"labels; uniform across arms so it CANNOT move the paired contrast)")
        for a in ALL_IDS:
            log(f"      {a:<11} +{100*tw[a]['delta_roi']:.2f}pp -> "
                f"{100*tw[a]['roi_with_ladder']:+.2f}%  "
                f"(bet-weighted gain {tw[a]['bet_weighted_gain_pts']:.4f} pts)")
        res[label] = R

    OUT.write_text(json.dumps(res, indent=1, default=float))
    (ROOT / "data" / "logs").mkdir(exist_ok=True)
    (ROOT / "data" / "logs" / "ns_battery.log").write_text("\n".join(lines))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
