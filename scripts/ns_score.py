#!/usr/bin/env python3
"""NEWSTRAT SCORE — 3 pre-registered arms + 1 control, walk-forward selection.

Pre-registered in data/newstrat_prereg.md
sha256 db293ad8548a2e7099586a66847d6d1a71d8e83e4ab171e0917584980e9f1fc7,
WRITTEN AND HASHED BEFORE THIS FILE WAS RUN.

Selection design is D164 ARM B verbatim (select on seasons [0,k), freeze, score
k, roll; eligibility n >= 100*k; tie-break larger-n then lower index; pool
bet-weighted; interval = equal-weight K-1 dof cluster-mean t).

Nulls, both >=200 draws, identical procedure per arm:
  NULL-S  permute the arm's OWN selector within date-slate  -> does the SELECTOR
          carry information?
  NULL-M  permute every MODEL-derived quantity jointly within date-slate
          (m_us, p_us, m_us_blind, pred_dm) -> D164/D173 convention, prices the
          LEVEL.
Family-wise: all 3 arms through the SAME permuted realisation per draw; per-draw
max -> E[max | null] and FWER p (D165's prescription).

READ-ONLY. No DB. nbapred/ untouched. No production default changed.

  python3 scripts/ns_score.py
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

PREREG = "db293ad8548a2e7099586a66847d6d1a71d8e83e4ab171e0917584980e9f1fc7"
WIN = 100.0 / 110.0
MIN_PER_SEASON = 100
QS = [0.50, 0.25, 0.10]
NDRAWS = 400
SEED = 20260805
ARMS = ["A1_CLV", "A2_AVAIL", "A3_RETURN"]
ALL_IDS = ["A0_EDGE"] + ARMS
OUT = ROOT / "data" / "ns_score.json"


def t_crit(dof: int) -> float:
    tbl = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
           7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
           13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
           19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000}
    if dof in tbl:
        return tbl[dof]
    ks = sorted(tbl)
    if dof < ks[0]:
        return tbl[ks[0]]
    if dof > ks[-1]:
        return 1.960
    lo = max(k for k in ks if k <= dof)
    hi = min(k for k in ks if k >= dof)
    if lo == hi:
        return tbl[lo]
    w = (dof - lo) / (hi - lo)
    return tbl[lo] * (1 - w) + tbl[hi] * w


def cmt(vals) -> dict:
    """Equal-weight K-1 dof cluster-mean t interval (GATE_POLICY_V2 §9.1(4))."""
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    K = len(v)
    if K < 2:
        return {"K": int(K), "mean": float(v[0]) if K else float("nan"),
                "lo": float("nan"), "hi": float("nan"), "sig": False}
    m, sd = float(v.mean()), float(v.std(ddof=1))
    se = sd / np.sqrt(K)
    h = t_crit(K - 1) * se
    return {"K": K, "mean": m, "sd": sd, "se": float(se),
            "lo": m - h, "hi": m + h, "sig": bool((m - h) * (m + h) > 0)}


def icc_deff(vals, clus):
    df = pd.DataFrame({"v": np.asarray(vals, float), "c": np.asarray(clus)})
    g = df.groupby("c").v
    n_i = g.size().values.astype(float)
    K = len(n_i)
    if K < 2:
        return float("nan"), float("nan")
    N = n_i.sum()
    gm = df.v.mean()
    msb = (n_i * (g.mean().values - gm) ** 2).sum() / (K - 1)
    msw = (g.var(ddof=1).values * (n_i - 1)).sum() / (N - K)
    n0 = (N - (n_i ** 2).sum() / N) / (K - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    return float(icc), float(1.0 + (n0 - 1) * icc)


# --------------------------------------------------------------------- data --
def load():
    f = pd.read_csv(ROOT / "data" / "ats19_frame.csv.gz")
    b = pd.read_csv(ROOT / "data" / "ats19_frame_D173blind.csv.gz")[
        ["game_id", "m_us"]].rename(columns={"m_us": "m_us_blind"})
    x = pd.read_csv(ROOT / "data" / "ns_features.csv.gz")[
        ["game_id", "pred_dm", "retmin_h", "retmin_a"]]
    d = f.merge(b, on="game_id").merge(x, on="game_id")
    assert len(d) == len(f) == 22742, (len(d), len(f))
    d["game_date"] = pd.to_datetime(d["game_date"])
    d = d.sort_values(["season", "game_date", "game_id"]).reset_index(drop=True)
    d["slate"] = d.groupby(["season", "game_date"], sort=False).ngroup()
    return d


def selectors(m_us, m_blind, pred_dm, retmin_h, retmin_a, om):
    """The four pre-registered selectors.  s = +1 home bet, -1 away bet."""
    s = np.where(m_us > om, 1.0, -1.0)
    return {
        "A0_EDGE": np.abs(m_us - om),
        "A1_CLV": s * pred_dm,
        "A2_AVAIL": s * (m_us - m_blind),
        # retmin(side we back) - retmin(the other side)
        "A3_RETURN": s * (retmin_h - retmin_a),
    }


def payoff_of(m_us, om, act):
    bet_home = m_us > om
    diff = act - om
    push = diff == 0
    win = np.where(bet_home, diff > 0, diff < 0)
    return np.where(push, 0.0, np.where(win, WIN, -1.0)), win, push, bet_home


# ------------------------------------------------------- walk-forward select --
def wf(sel, pay, s_i, K, minh, clv=None):
    """D164 ARM B on the 3 quantile cells.  Returns per-fold records."""
    steps = []
    for k in range(minh, K):
        win_m = s_i < k
        selw = sel[win_m]
        paw = pay[win_m]
        best = None
        for qi, q in enumerate(QS):
            cut = float(np.quantile(selw, 1.0 - q))
            mm = selw >= cut
            n = int(mm.sum())
            if n < MIN_PER_SEASON * k:
                continue
            roi = float(paw[mm].mean())
            key = (roi, n, -qi)
            if best is None or key > best[0]:
                best = (key, qi, cut, n, roi)
        if best is None:
            steps.append(None)
            continue
        _, qi, cut, seln, selroi = best
        te = s_i == k
        mt = te & (sel >= cut)
        nt = int(mt.sum())
        if nt < 5:
            steps.append(None)
            continue
        steps.append({
            "k": int(k), "q": QS[qi], "cut": cut,
            "sel_n": seln, "sel_roi": selroi,
            "test_n": nt, "test_pay": float(pay[mt].sum()),
            "test_roi": float(pay[mt].mean()),
            "test_clv": float(clv[mt].mean()) if clv is not None else np.nan,
            "mask": mt,
        })
    return steps


def pooled(steps):
    n = sum(s["test_n"] for s in steps if s)
    p = sum(s["test_pay"] for s in steps if s)
    return (p / n) if n > 0 else np.nan, n


def run_all(sels, pay, s_i, K, minh, clv=None):
    return {a: wf(sels[a], pay, s_i, K, minh, clv) for a in ALL_IDS}


# ------------------------------------------------------------------- frames --
def frame_view(d, seasons_keep, minh):
    m = d.season.isin(seasons_keep).to_numpy()
    sub = d[m].reset_index(drop=True)
    seasons = sorted(sub.season.unique())
    s_i = sub.season.map({s: i for i, s in enumerate(seasons)}).to_numpy()
    return sub, seasons, s_i, minh


def score_frame(sub, seasons, s_i, minh, label, log):
    K = len(seasons)
    om = sub.open_margin.to_numpy(float)
    act = sub.margin_actual.to_numpy(float)
    m_us = sub.m_us.to_numpy(float)
    m_bl = sub.m_us_blind.to_numpy(float)
    pdm = sub.pred_dm.to_numpy(float)
    rh = sub.retmin_h.to_numpy(float)
    ra = sub.retmin_a.to_numpy(float)
    p_us = sub.p_us.to_numpy(float)
    close = sub.close_margin.to_numpy(float)
    slate = sub.slate.to_numpy(np.int64)

    pay, win, push, bh = payoff_of(m_us, om, act)
    clv = np.where(bh, 1.0, -1.0) * (close - om)
    sels = selectors(m_us, m_bl, pdm, rh, ra, om)
    real = run_all(sels, pay, s_i, K, minh, clv)

    scored = [seasons[st["k"]] for st in real["A0_EDGE"] if st]
    log(f"\n=== {label}  K={K}  min_history={minh}  "
        f"scored folds ({len(scored)}): {', '.join(scored)}")
    log(f"    universe: n={len(sub)}  ROI={100*pay.mean():+.3f}%  "
        f"cover={100*win[~push].mean():.3f}%  CLV={np.nanmean(clv):+.4f} pts")

    res = {"label": label, "K": K, "min_history": minh, "seasons": seasons,
           "scored_folds": scored, "n_frame": int(len(sub)),
           "universe_roi": float(pay.mean()),
           "universe_cover": float(win[~push].mean()),
           "universe_clv": float(np.nanmean(clv)), "arms": {}}

    a0 = [st["test_roi"] for st in real["A0_EDGE"] if st]
    log(f"\n    {'arm':<11}{'q-path':<18}{'n':>7}{'ROI%':>9}{'cover%':>9}"
        f"{'CLV':>8}   {'K-1 t CI on folds':<26}{'paired vs A0':<24}")
    for a in ALL_IDS:
        st = [x for x in real[a] if x]
        pr, n = pooled(st)
        per = [x["test_roi"] for x in st]
        qs = "/".join(f"{x['q']:.2f}"[1:] for x in st)
        wc = sum(win[x["mask"]].sum() for x in st)
        pc = sum(push[x["mask"]].sum() for x in st)
        cov = wc / (n - pc) if n > pc else np.nan
        cl = float(np.nanmean(np.concatenate([clv[x["mask"]] for x in st])))
        ci = cmt(per)
        pair = cmt([per[i] - a0[i] for i in range(len(per))]) if a != "A0_EDGE" else None
        icc, deff = icc_deff(
            np.concatenate([pay[x["mask"]] for x in st]),
            np.concatenate([np.full(x["test_n"], x["k"]) for x in st]))
        res["arms"][a] = {
            "pooled_roi": float(pr), "n": int(n), "cover": float(cov),
            "clv": cl, "per_fold_roi": per, "q_path": [x["q"] for x in st],
            "cut_path": [x["cut"] for x in st],
            "per_fold_n": [x["test_n"] for x in st],
            "per_fold_clv": [x["test_clv"] for x in st],
            "ci": ci, "paired_vs_A0": pair, "icc": icc, "deff": deff,
            "folds_positive": int(sum(1 for x in per if x > 0)),
            "n_distinct_q": len({x["q"] for x in st}),
        }
        ps = (f"{100*pair['mean']:+7.2f} [{100*pair['lo']:+6.2f},"
              f"{100*pair['hi']:+6.2f}]{'SIG' if pair['sig'] else ' ns'}") \
            if pair else "        (benchmark)"
        log(f"    {a:<11}{qs:<18}{n:>7}{100*pr:>+9.2f}{100*cov:>9.3f}"
            f"{cl:>+8.3f}   [{100*ci['lo']:+7.2f},{100*ci['hi']:+7.2f}]"
            f"{'SIG' if ci['sig'] else ' ns'}  {ps}")

    # ex-COVID twin on the scored folds
    for a in ALL_IDS:
        st = [x for x in real[a] if x]
        keep = [x for x in st if seasons[x["k"]] not in ("2019-20", "2020-21")]
        if keep and len(keep) < len(st):
            n = sum(x["test_n"] for x in keep)
            res["arms"][a]["ex_covid_roi"] = float(
                sum(x["test_pay"] for x in keep) / n)
            res["arms"][a]["ex_covid_n"] = int(n)
    return res, real, sels, pay, clv, slate, (m_us, p_us, m_bl, pdm, om, act)


# --------------------------------------------------------------------- null --
def slate_bounds(slate):
    order = np.argsort(slate, kind="stable")
    bounds = np.searchsorted(slate[order], np.arange(slate.max() + 2))
    return order, bounds


def perm_index(order, bounds, rng):
    perm = order.copy()
    for gi in range(len(bounds) - 1):
        a, b = bounds[gi], bounds[gi + 1]
        if b - a > 1:
            perm[a:b] = rng.permutation(perm[a:b])
    idx = np.empty_like(perm)
    idx[order] = perm
    return idx


def run_nulls(sub, seasons, s_i, minh, sels, pay, clv, slate, raw, log):
    K = len(seasons)
    m_us, p_us, m_bl, pdm, om, act = raw
    order, bounds = slate_bounds(slate)

    # ---- NULL-S : permute each arm's own selector within slate --------------
    rng = np.random.default_rng(SEED)
    ns = {a: [] for a in ARMS}
    ns_pair = {a: [] for a in ARMS}
    fam_max, fam_max_pair = [], []
    a0_real = [x["test_roi"] for x in wf(sels["A0_EDGE"], pay, s_i, K, minh) if x]
    a0_pool, _ = pooled([x for x in wf(sels["A0_EDGE"], pay, s_i, K, minh) if x])
    for _ in range(NDRAWS):
        idx = perm_index(order, bounds, rng)
        row, rowp = {}, {}
        for a in ARMS:
            st = [x for x in wf(sels[a][idx], pay, s_i, K, minh) if x]
            r, _n = pooled(st)
            per = [x["test_roi"] for x in st]
            ns[a].append(float(r))
            assert len(per) == len(a0_real), (len(per), len(a0_real))
            dl = cmt([per[i] - a0_real[i] for i in range(len(per))])
            ns_pair[a].append(float(dl["mean"]))
            row[a] = float(r)
            rowp[a] = float(dl["mean"])
        fam_max.append(max(row.values()))
        fam_max_pair.append(max(rowp.values()))

    # ---- NULL-M : permute every model-derived quantity jointly --------------
    rng2 = np.random.default_rng(SEED)
    nm = {a: [] for a in ALL_IDS}
    for _ in range(NDRAWS):
        idx = perm_index(order, bounds, rng2)
        mp, pp, mbp, pdp = m_us[idx], p_us[idx], m_bl[idx], pdm[idx]
        pay_n, _w, _p, _b = payoff_of(mp, om, act)
        sl_n = selectors(mp, mbp, pdp,
                         sub.retmin_h.to_numpy(float),
                         sub.retmin_a.to_numpy(float), om)
        for a in ALL_IDS:
            st = [x for x in wf(sl_n[a], pay_n, s_i, K, minh) if x]
            r, _n = pooled(st)
            nm[a].append(float(r))

    def dist(v):
        v = np.asarray(v, float)
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                "p05": float(np.percentile(v, 5)),
                "p50": float(np.percentile(v, 50)),
                "p95": float(np.percentile(v, 95)),
                "min": float(v.min()), "max": float(v.max())}

    out = {"n_draws": NDRAWS, "seed": SEED,
           "null_s": {a: dist(ns[a]) for a in ARMS},
           "null_s_paired": {a: dist(ns_pair[a]) for a in ARMS},
           "null_m": {a: dist(nm[a]) for a in ALL_IDS},
           "family_max": dist(fam_max),
           "family_max_paired": dist(fam_max_pair),
           "a0_pooled": float(a0_pool)}
    log(f"\n    NULL, {NDRAWS} draws, seed {SEED}.")
    log(f"      {'arm':<11}{'real ROI%':>11}{'NULL-S%':>10}{'net':>8}{'p':>7}"
        f"{'NULL-M%':>10}{'net':>8}{'p':>7}   {'paired real':>12}"
        f"{'NULL-Sp':>9}{'net':>8}{'p':>7}{'FWERp':>8}")
    return out, ns, ns_pair, nm, fam_max, fam_max_pair


def main():
    log_lines = []

    def log(s=""):
        print(s)
        log_lines.append(s)

    d = load()
    log(f"NEWSTRAT — prereg sha256 {PREREG}")
    log(f"frame n={len(d)}  seasons={d.season.nunique()}  "
        f"QS={QS}  NDRAWS={NDRAWS}  seed={SEED}")

    allseasons = sorted(d.season.unique())
    report8 = [s for s in allseasons if s >= "2018-19"]
    frames = [("PRIMARY_REPORT8", report8, 3), ("SECONDARY_POOL19", allseasons, 5)]

    res = {"prereg_sha256": PREREG, "qs": QS, "ndraws": NDRAWS, "seed": SEED,
           "frames": {}}
    for label, keep, minh in frames:
        sub, seasons, s_i, mh = frame_view(d, keep, minh)
        r, real, sels, pay, clv, slate, raw = score_frame(
            sub, seasons, s_i, mh, label, log)
        nl, ns, nsp, nm, fmax, fmaxp = run_nulls(
            sub, seasons, s_i, mh, sels, pay, clv, slate, raw, log)
        r["null"] = nl
        fmax = np.asarray(fmax)
        fmaxp = np.asarray(fmaxp)
        for a in ALL_IDS:
            A = r["arms"][a]
            A["null_m_mean"] = nl["null_m"][a]["mean"]
            A["net_of_null_m"] = A["pooled_roi"] - nl["null_m"][a]["mean"]
            A["p_null_m"] = float(np.mean(np.asarray(nm[a]) >= A["pooled_roi"]))
            if a in ARMS:
                A["null_s_mean"] = nl["null_s"][a]["mean"]
                A["net_of_null_s"] = A["pooled_roi"] - nl["null_s"][a]["mean"]
                A["p_null_s"] = float(np.mean(np.asarray(ns[a]) >= A["pooled_roi"]))
                pm = A["paired_vs_A0"]["mean"]
                A["null_s_paired_mean"] = nl["null_s_paired"][a]["mean"]
                A["net_of_null_paired"] = pm - nl["null_s_paired"][a]["mean"]
                A["p_null_s_paired"] = float(np.mean(np.asarray(nsp[a]) >= pm))
                A["fwer_p"] = float(np.mean(fmax >= A["pooled_roi"]))
                A["fwer_p_paired"] = float(np.mean(fmaxp >= pm))
                log(f"      {a:<11}{100*A['pooled_roi']:>+11.2f}"
                    f"{100*A['null_s_mean']:>+10.2f}{100*A['net_of_null_s']:>+8.2f}"
                    f"{A['p_null_s']:>7.3f}"
                    f"{100*A['null_m_mean']:>+10.2f}{100*A['net_of_null_m']:>+8.2f}"
                    f"{A['p_null_m']:>7.3f}   {100*pm:>+12.2f}"
                    f"{100*A['null_s_paired_mean']:>+9.2f}"
                    f"{100*A['net_of_null_paired']:>+8.2f}"
                    f"{A['p_null_s_paired']:>7.3f}{A['fwer_p_paired']:>8.3f}")
            else:
                log(f"      {a:<11}{100*A['pooled_roi']:>+11.2f}"
                    f"{'--':>10}{'--':>8}{'--':>7}"
                    f"{100*A['null_m_mean']:>+10.2f}{100*A['net_of_null_m']:>+8.2f}"
                    f"{A['p_null_m']:>7.3f}")
        single = float(np.mean([nl["null_s"][a]["mean"] for a in ARMS]))
        r["family_burden"] = {
            "E_max_of_3": nl["family_max"]["mean"],
            "mean_single_arm_null": single,
            "burden_points": nl["family_max"]["mean"] - single,
            "E_max_of_3_paired": nl["family_max_paired"]["mean"],
            "mean_single_arm_null_paired": float(
                np.mean([nl["null_s_paired"][a]["mean"] for a in ARMS])),
            "burden_points_paired": nl["family_max_paired"]["mean"] - float(
                np.mean([nl["null_s_paired"][a]["mean"] for a in ARMS])),
        }
        fb = r["family_burden"]
        log(f"    FAMILY (F=3): E[max|null] = {100*fb['E_max_of_3']:+.3f}%  vs "
            f"mean single-arm null {100*fb['mean_single_arm_null']:+.3f}%  -> "
            f"burden {100*fb['burden_points']:+.3f} points")
        log(f"    FAMILY paired: E[max|null] = {100*fb['E_max_of_3_paired']:+.3f}%"
            f"  -> burden {100*fb['burden_points_paired']:+.3f} points")
        res["frames"][label] = r

    OUT.write_text(json.dumps(res, indent=1, default=float))
    (ROOT / "data" / "logs").mkdir(exist_ok=True)
    (ROOT / "data" / "logs" / "ns_score.log").write_text("\n".join(log_lines))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
