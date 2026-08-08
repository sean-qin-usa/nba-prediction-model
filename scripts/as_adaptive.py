"""ADAPTIVE / RECENCY-BASED CONFIG SELECTION.

Pre-registered in data/adaptive_sel_prereg.md
sha256 529f21effa39a52279a4e58eeb78514e44c0e5434fa6c59692fa523832f033fa

Extends D164's harness (scripts/oc_capacity.py): same frame, same 600-cell
space, same bet/price/ROI convention, same eligibility guard, same permutation
null machinery and the SAME SEED.

ARM A        RECENCY-1  : select on season k ONLY, score k+1        (PRIMARY)
ARM B2/3/5   RECENCY-W  : select on the last W seasons, score k+1
ARM C        ALL-HISTORY: select on 1..k, score k+1  (= D164 ARM B, anchor)
ARM D-ONLINE within-season, season-to-date only, re-selected on a cadence
ARM D-HYBRID within-season, all history + season-to-date
ARM E        null       : 200 within-date permutations through ALL SEVEN arms
ADJACENCY    the mechanism test: ROI on k+1 vs ROI on all other seasons

READ-ONLY. No DB. No default changed. Nothing ships.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads  # noqa: E402
nbapred.threads.pin(1)

import numpy as np  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import oc_capacity as oc  # noqa: E402  (D164 harness: load/masks/agg/CI)

import os as _os  # noqa: E402
# PATH OVERRIDE ONLY (D173 re-run on the D170/D171 backfilled data).  The
# default is byte-identical to D165's; no arm, window, guard or seed is touched.
OUT = ROOT / "data" / f"as_adaptive{_os.environ.get('AS_TAG', '')}.json"

PREREG_SHA = "529f21effa39a52279a4e58eeb78514e44c0e5434fa6c59692fa523832f033fa"
MEAN_SEASON = 22742.0 / 19.0          # 1196.947, prereg §3
BURNIN = 300                          # prereg §5, ARM D
CADENCE = 200                         # prereg §5, ARM D
K_COMMON = 5                          # prereg §4: score k+1 for k = 5..18
import os  # noqa: E402
# prereg §7 fixes 200 draws for the REGISTERED run; the env var exists only so
# the script can be smoke-tested cheaply and is never used for a reported number.
NDRAWS = int(os.environ.get("AS_DRAWS", "200"))
SEED = 20260804                       # prereg §7 (same as D164)
T80 = 2.802                           # prereg §8 MDE80 constant
COVID = ("2019-20", "2020-21")

ARMS = ["A_REC1", "B_REC2", "B_REC3", "B_REC5", "C_ALL", "D_ONLINE", "D_HYBRID"]


# ------------------------------------------------------------------ selection
def select(c_win, p_win, guard):
    """D164's selection rule verbatim: max ROI among eligible, det. tie-break."""
    elig = c_win >= guard
    if not elig.any():
        return None
    roi = np.where(c_win > 0, p_win / np.maximum(c_win, 1e-9), np.nan)
    cand = np.where(elig, roi, -np.inf)
    best = int(np.argmax(cand))
    ties = np.where(np.isclose(cand, cand[best], rtol=0, atol=1e-12))[0]
    if len(ties) > 1:
        best = int(ties[np.lexsort((ties, -c_win[ties]))[0]])
    return best


def arm_window(cnt, pay, K, W, k_start):
    """Season-window selection. W=None -> all history. Score season k."""
    steps = []
    for k in range(k_start, K):
        lo = 0 if W is None else max(0, k - W)
        nseas = k - lo
        c = cnt[:, lo:k].sum(1)
        p = pay[:, lo:k].sum(1)
        best = select(c, p, 100.0 * nseas)
        if best is None:
            continue
        steps.append({
            "k": k, "cfg": best, "lo": lo, "nseas": nseas,
            "sel_roi": float(p[best] / c[best]), "sel_n": float(c[best]),
            "test_roi": float(pay[best, k] / cnt[best, k]) if cnt[best, k] > 0 else np.nan,
            "test_n": float(cnt[best, k]), "test_pay": float(pay[best, k]),
        })
    return steps


# ------------------------------------------------------------------- within-season
def season_prefix(M, payoff, idx_j, extra=None):
    """Cumulative per-config counts / payoffs over one season, in game order."""
    Mj = M[:, idx_j]
    cc = np.cumsum(Mj, axis=1)
    cp = np.cumsum(Mj * payoff[idx_j].astype(np.float32), axis=1)
    ex = None
    if extra is not None:
        ex = [np.cumsum(Mj * e[idx_j].astype(np.float32), axis=1) for e in extra]
    return cc, cp, ex


def _seg(cum, cfg, a, b):
    return float(cum[cfg, b - 1] - (cum[cfg, a - 1] if a > 0 else 0.0))


def arm_online(M, payoff, cnt, pay, K, sidx, cfgC, mode, extra=None,
               only_season=None, prefix_cache=None):
    """Within-season online re-selection.

    Segment 0 (games 0..BURNIN) uses ARM C's frozen config for that season, so
    ARM D and ARM C are on identical bets until the first online decision.
    Decision points at BURNIN, BURNIN+CADENCE, ... ; each applies FORWARD only.
    """
    steps = []
    ks = range(K_COMMON, K) if only_season is None else [only_season]
    hist_c = None
    for k in ks:
        idx_j = sidx[k]
        n_j = len(idx_j)
        if prefix_cache is not None and k in prefix_cache:
            cc, cp, ex = prefix_cache[k]
        else:
            cc, cp, ex = season_prefix(M, payoff, idx_j, extra)
            if prefix_cache is not None:
                prefix_cache[k] = (cc, cp, ex)
        if mode == "hybrid":
            hist_c = cnt[:, :k].sum(1)
            hist_p = pay[:, :k].sum(1)
        dps = list(range(BURNIN, n_j, CADENCE))
        bounds = [0] + dps + [n_j]
        seg_cfg = [cfgC[k]]
        for d in dps:
            if mode == "online":
                c_w, p_w = cc[:, d - 1], cp[:, d - 1]
                guard = 100.0 * d / MEAN_SEASON
            else:
                c_w = hist_c + cc[:, d - 1]
                p_w = hist_p + cp[:, d - 1]
                guard = 100.0 * (k + d / MEAN_SEASON)
            b = select(c_w, p_w, guard)
            seg_cfg.append(seg_cfg[-1] if b is None else b)
        tot_n = tot_p = post_n = post_p = 0.0
        exs = [0.0, 0.0] if ex is not None else None
        for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            c = seg_cfg[i]
            nn, pp = _seg(cc, c, a, b), _seg(cp, c, a, b)
            tot_n += nn
            tot_p += pp
            if a >= BURNIN:
                post_n += nn
                post_p += pp
            if exs is not None:
                exs[0] += _seg(ex[0], c, a, b)
                exs[1] += _seg(ex[1], c, a, b)
        # ARM C comparator on exactly the post-burn-in games
        cC = cfgC[k]
        cpost_n = _seg(cc, cC, BURNIN, n_j) if n_j > BURNIN else 0.0
        cpost_p = _seg(cp, cC, BURNIN, n_j) if n_j > BURNIN else 0.0
        steps.append({
            "k": k, "n_dp": len(dps), "seg_cfg": seg_cfg,
            "changes": int(sum(1 for i in range(1, len(seg_cfg))
                               if seg_cfg[i] != seg_cfg[i - 1])),
            "test_n": tot_n, "test_pay": tot_p,
            "test_roi": (tot_p / tot_n) if tot_n > 0 else np.nan,
            "post_n": post_n, "post_pay": post_p,
            "post_roi": (post_p / post_n) if post_n > 0 else np.nan,
            "c_post_n": cpost_n, "c_post_pay": cpost_p,
            "c_post_roi": (cpost_p / cpost_n) if cpost_n > 0 else np.nan,
            "win": exs[0] if exs else None, "push": exs[1] if exs else None,
        })
    return steps


# ------------------------------------------------------------------ aggregation
def pooled(steps, nkey="test_n", pkey="test_pay"):
    n = sum(s[nkey] for s in steps)
    p = sum(s[pkey] for s in steps)
    return (p / n if n > 0 else np.nan), n


def arm_summary(steps, seasons, key="test", wcnt=None, pcnt=None):
    per = [s[f"{key}_roi"] for s in steps]
    roi, n = pooled(steps, f"{key}_n", f"{key}_pay")
    ci = oc.cluster_mean_t(per)
    sd = ci["sd"]
    out = {
        "pooled_roi": float(roi), "pooled_n": float(n),
        "per_season_roi": [float(x) for x in per],
        "seasons": [seasons[s["k"]] for s in steps],
        "ci": ci, "K": len(per),
        "seasons_positive": int(sum(1 for x in per if x > 0)),
        "mde80": float(T80 * sd / np.sqrt(len(per))) if len(per) > 1 else np.nan,
    }
    ex = [s for s in steps if seasons[s["k"]] not in COVID]
    if ex:
        r, _ = pooled(ex, f"{key}_n", f"{key}_pay")
        out["ex_covid_roi"] = float(r)
    return out


def dist(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "p05": float(np.percentile(x, 5)), "p50": float(np.percentile(x, 50)),
            "p95": float(np.percentile(x, 95)),
            "min": float(x.min()), "max": float(x.max())}


# ------------------------------------------------------------------- adjacency
def adjacency(cnt, pay, K):
    """For each tuning season k: ROI of c*(k) on k+1 vs its mean on the rest."""
    roi_mat = np.where(cnt > 0, pay / np.maximum(cnt, 1e-9), np.nan)
    rows = []
    for k in range(K):
        best = select(cnt[:, k], pay[:, k], 100.0)
        if best is None:
            continue
        r = roi_mat[best]
        rec = {"k": k, "cfg": best, "is_roi": float(r[k])}
        for lag in (1, 2):
            tgt = k + lag
            if tgt >= K:
                rec[f"t{lag}"] = np.nan
                rec[f"far{lag}"] = np.nan
                rec[f"prem{lag}"] = np.nan
                continue
            far_idx = [j for j in range(K) if j not in (k, tgt)]
            rec[f"t{lag}"] = float(r[tgt])
            rec[f"far{lag}"] = float(np.nanmean(r[far_idx]))
            rec[f"prem{lag}"] = rec[f"t{lag}"] - rec[f"far{lag}"]
        rows.append(rec)
    prem1 = [x["prem1"] for x in rows if np.isfinite(x["prem1"])]
    prem2 = [x["prem2"] for x in rows if np.isfinite(x["prem2"])]
    return {
        "rows": rows,
        "premium_lag1": {"mean": float(np.mean(prem1)), "n": len(prem1),
                         "ci": oc.cluster_mean_t(prem1),
                         "positive": int(sum(1 for x in prem1 if x > 0))},
        "premium_lag2": {"mean": float(np.mean(prem2)), "n": len(prem2),
                         "ci": oc.cluster_mean_t(prem2),
                         "positive": int(sum(1 for x in prem2 if x > 0))},
    }


# ------------------------------------------------------------------ one full pass
def run_all(M, payoff, st, K, sidx, seasons, want_prefix_extra=None):
    """All seven arms + adjacency on one (real or permuted) realisation."""
    cnt, pay = oc.agg(M, payoff, st)
    res = {}
    res["A_REC1"] = arm_window(cnt, pay, K, 1, K_COMMON)
    res["B_REC2"] = arm_window(cnt, pay, K, 2, K_COMMON)
    res["B_REC3"] = arm_window(cnt, pay, K, 3, K_COMMON)
    res["B_REC5"] = arm_window(cnt, pay, K, 5, K_COMMON)
    res["C_ALL"] = arm_window(cnt, pay, K, None, K_COMMON)
    cfgC = {s["k"]: s["cfg"] for s in res["C_ALL"]}
    cache = {}
    res["D_ONLINE"] = arm_online(M, payoff, cnt, pay, K, sidx, cfgC, "online",
                                 extra=want_prefix_extra, prefix_cache=cache)
    res["D_HYBRID"] = arm_online(M, payoff, cnt, pay, K, sidx, cfgC, "hybrid",
                                 extra=want_prefix_extra, prefix_cache=cache)
    return res, cnt, pay


# ------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    df, seasons = oc.load()
    K = len(seasons)
    st = oc.build_static(df)
    m_us = df["m_us"].to_numpy(float)
    p_us = df["p_us"].to_numpy(float)
    sidx = {j: np.where(st["s_i"] == j)[0] for j in range(K)}

    payoff, M, keys, win, push, _ = oc.payoff_and_masks(m_us, p_us, st)
    winf = win.astype(np.float32)
    pushf = push.astype(np.float32)

    real, cnt, pay = run_all(M, payoff, st, K, sidx, seasons,
                             want_prefix_extra=[winf, pushf])
    wcnt = (M * winf) @ st["S"]
    pcnt = (M * pushf) @ st["S"]

    out = {
        "prereg_sha256": PREREG_SHA,
        "parent": "D164 / scripts/oc_capacity.py",
        "seasons": seasons, "n_games": int(st["n"]), "cardinality": len(keys),
        "burnin": BURNIN, "cadence": CADENCE, "k_common": K_COMMON,
        "seed": SEED, "ndraws": NDRAWS,
    }

    # ---------------------------------------------------------------- anchors
    ref = 0
    ref_roi_season = [float(pay[ref, j] / cnt[ref, j]) for j in range(K)]
    out["anchor"] = {
        "d162_pool19_roi": float(pay[ref].sum() / cnt[ref].sum()),
        "d162_pool19_cover": float(wcnt[ref].sum() / (cnt[ref].sum() - pcnt[ref].sum())),
        "n_games": int(st["n"]), "pushes": float(pcnt[ref].sum()),
    }

    # ---------------------------------------------------------------- the arms
    out["arms"] = {}
    for a in ARMS:
        steps = real[a]
        s = arm_summary(steps, seasons, "test")
        s["steps"] = [{
            "k": st_["k"], "test_season": seasons[st_["k"]],
            "cfg": list(map(str, keys[st_["cfg"]])) if "cfg" in st_ else
                   [list(map(str, keys[c])) for c in st_["seg_cfg"]],
            "cfg_idx": st_.get("cfg", st_.get("seg_cfg")),
            "sel_roi": st_.get("sel_roi"), "sel_n": st_.get("sel_n"),
            "test_roi": st_["test_roi"], "test_n": st_["test_n"],
        } for st_ in steps]
        if "cfg" in steps[0]:
            cfgs = [x["cfg"] for x in steps]
            s["selection_changes"] = int(sum(1 for i in range(1, len(cfgs))
                                             if cfgs[i] != cfgs[i - 1]))
            s["n_transitions"] = len(cfgs) - 1
            s["n_distinct_cfg"] = len(set(cfgs))
            wc = sum(wcnt[x["cfg"], x["k"]] for x in steps)
            pc = sum(pcnt[x["cfg"], x["k"]] for x in steps)
            s["pooled_cover"] = float(wc / (s["pooled_n"] - pc))
            s["mean_sel_n"] = float(np.mean([x["sel_n"] for x in steps]))
            s["mean_sel_roi"] = float(np.mean([x["sel_roi"] for x in steps]))
            # in-sample-to-out-of-sample decay of THIS arm's selector
            s["decay"] = float(s["mean_sel_roi"] - s["pooled_roi"])
        else:
            s["selection_changes"] = int(sum(x["changes"] for x in steps))
            s["n_transitions"] = int(sum(x["n_dp"] for x in steps))
            s["n_distinct_cfg"] = len({c for x in steps for c in x["seg_cfg"]})
            wc = sum(x["win"] for x in steps)
            pc = sum(x["push"] for x in steps)
            s["pooled_cover"] = float(wc / (s["pooled_n"] - pc))
            s["post_burnin"] = arm_summary(steps, seasons, "post")
            s["post_burnin_C"] = arm_summary(
                [{**x, "c_post_roi": x["c_post_roi"], "k": x["k"]} for x in steps],
                seasons, "c_post")
            s["paired_post_vs_C"] = oc.cluster_mean_t(
                [x["post_roi"] - x["c_post_roi"] for x in steps])
        out["arms"][a] = s

    # paired deltas against ARM C on the common track
    cper = out["arms"]["C_ALL"]["per_season_roi"]
    for a in ARMS:
        if a == "C_ALL":
            continue
        d = [x - y for x, y in zip(out["arms"][a]["per_season_roi"], cper)]
        out["arms"][a]["paired_vs_C"] = oc.cluster_mean_t(d)

    # ------------------------------------------------------ maximal tracks (2ry)
    out["maximal_tracks"] = {}
    for a, W, ks in [("A_REC1", 1, 1), ("B_REC2", 2, 2), ("B_REC3", 3, 3),
                     ("B_REC5", 5, 5), ("C_ALL", None, 5)]:
        stp = arm_window(cnt, pay, K, W, ks)
        out["maximal_tracks"][a] = arm_summary(stp, seasons, "test")

    # ---------------------------------------------------------------- adjacency
    out["adjacency"] = adjacency(cnt, pay, K)

    # ------------------------------------------------- the named 24-25 -> 25-26
    k2425 = seasons.index("2024-25")
    b = select(cnt[:, k2425], pay[:, k2425], 100.0)
    j = seasons.index("2025-26")
    idx = sidx[j]
    mrow = M[b, idx].astype(bool)
    pay_g = payoff[idx][mrow]
    n1 = int(mrow.sum())
    roi1 = float(pay_g.mean())
    se1 = float(pay_g.std(ddof=1) / np.sqrt(n1))
    out["named_case"] = {
        "tune_season": "2024-25", "score_season": "2025-26",
        "cfg": list(map(str, keys[b])), "cfg_idx": int(b),
        "is_roi": float(pay[b, k2425] / cnt[b, k2425]),
        "is_n": float(cnt[b, k2425]),
        "roi": roi1, "n": n1,
        "cover": float(win[idx][mrow].sum() / (n1 - push[idx][mrow].sum())),
        "iid_lo": roi1 - 1.96 * se1, "iid_hi": roi1 + 1.96 * se1,
        "iid_se": se1,
        "mde80_iid": float(T80 * pay_g.std(ddof=1) / np.sqrt(n1)),
        "K_clusters": 1,
    }

    # -------------------------------------------- CAUSALITY: permute the future
    rngc = np.random.default_rng(SEED + 1)
    cfgC = {s["k"]: s["cfg"] for s in real["C_ALL"]}
    ntest, nfail = 0, 0
    for k in range(K_COMMON, K):
        idx_j = sidx[k]
        n_j = len(idx_j)
        base_on = arm_online(M, payoff, cnt, pay, K, sidx, cfgC, "online",
                             only_season=k)[0]["seg_cfg"]
        base_hy = arm_online(M, payoff, cnt, pay, K, sidx, cfgC, "hybrid",
                             only_season=k)[0]["seg_cfg"]
        for di, d in enumerate(range(BURNIN, n_j, CADENCE)):
            mp, pp = m_us.copy(), p_us.copy()
            act = st["actual"].copy()
            tail = idx_j[d:]
            perm = rngc.permutation(len(tail))
            mp[tail] = m_us[tail][perm]
            pp[tail] = p_us[tail][perm]
            act[tail] = st["actual"][tail][perm]
            st2 = dict(st)
            st2["actual"] = act
            po2, M2, _, _, _, _ = oc.payoff_and_masks(mp, pp, st2)
            c2, p2 = oc.agg(M2, po2, st2)
            cC2 = {kk: cfgC[kk] for kk in cfgC}
            on2 = arm_online(M2, po2, c2, p2, K, sidx, cC2, "online",
                             only_season=k)[0]["seg_cfg"]
            hy2 = arm_online(M2, po2, c2, p2, K, sidx, cC2, "hybrid",
                             only_season=k)[0]["seg_cfg"]
            ntest += 1
            ok = (on2[:di + 2] == base_on[:di + 2]) and (hy2[:di + 2] == base_hy[:di + 2])
            if not ok:
                nfail += 1
                print(f"  CAUSALITY FAIL season {seasons[k]} dp {d}", flush=True)
    out["causality"] = {"pairs_tested": ntest, "failures": nfail,
                        "pass": bool(nfail == 0)}
    print(f"causality: {ntest} (season, decision-point) pairs, {nfail} failures "
          f"[{time.time()-t0:.1f}s]", flush=True)

    # ------------------------------------------------------------- ARM E: null
    rng = np.random.default_rng(SEED)
    slate = st["slate"]
    order = np.argsort(slate, kind="stable")
    bounds = np.searchsorted(slate[order], np.arange(slate.max() + 2))
    null_roi = {a: [] for a in ARMS}
    null_post = {a: [] for a in ("D_ONLINE", "D_HYBRID")}
    null_prem1, null_prem2, null_maxes = [], [], []
    null_named = []
    for d in range(NDRAWS):
        perm = order.copy()
        for gi in range(len(bounds) - 1):
            a_, b_ = bounds[gi], bounds[gi + 1]
            if b_ - a_ > 1:
                perm[a_:b_] = rng.permutation(perm[a_:b_])
        idxp = np.empty_like(perm)
        idxp[order] = perm
        mp, pp = m_us[idxp], p_us[idxp]
        po_n, M_n, _, _, _, _ = oc.payoff_and_masks(mp, pp, st)
        rn, c_n, p_n = run_all(M_n, po_n, st, K, sidx, seasons)
        row = {}
        for a in ARMS:
            r, _ = pooled(rn[a])
            null_roi[a].append(float(r))
            row[a] = float(r)
        for a in ("D_ONLINE", "D_HYBRID"):
            r, _ = pooled(rn[a], "post_n", "post_pay")
            null_post[a].append(float(r))
        null_maxes.append(max(row.values()))
        adj = adjacency(c_n, p_n, K)
        null_prem1.append(adj["premium_lag1"]["mean"])
        null_prem2.append(adj["premium_lag2"]["mean"])
        bb = select(c_n[:, k2425], p_n[:, k2425], 100.0)
        null_named.append(float(p_n[bb, j] / c_n[bb, j]) if c_n[bb, j] > 0 else np.nan)
        if (d + 1) % 25 == 0:
            print(f"  null draw {d+1}/{NDRAWS}  [{time.time()-t0:.1f}s]", flush=True)

    out["null"] = {"n_draws": NDRAWS, "seed": SEED}
    for a in ARMS:
        nd = dist(null_roi[a])
        real_r = out["arms"][a]["pooled_roi"]
        out["null"][a] = nd
        out["arms"][a]["null"] = nd
        out["arms"][a]["net_of_null"] = float(real_r - nd["mean"])
        out["arms"][a]["p_value"] = float(np.mean(np.asarray(null_roi[a]) >= real_r))
        out["arms"][a]["beats_null_p95"] = bool(real_r > nd["p95"])
    for a in ("D_ONLINE", "D_HYBRID"):
        nd = dist(null_post[a])
        rr = out["arms"][a]["post_burnin"]["pooled_roi"]
        out["arms"][a]["post_burnin"]["null"] = nd
        out["arms"][a]["post_burnin"]["net_of_null"] = float(rr - nd["mean"])
        out["arms"][a]["post_burnin"]["p_value"] = float(
            np.mean(np.asarray(null_post[a]) >= rr))
    out["null"]["family_max"] = dist(null_maxes)
    out["null"]["family_max_draws"] = [float(x) for x in null_maxes]
    out["null"]["draws"] = {a: [float(x) for x in null_roi[a]] for a in ARMS}
    for a in ARMS:
        out["arms"][a]["fwer_p"] = float(
            np.mean(np.asarray(null_maxes) >= out["arms"][a]["pooled_roi"]))
    out["null"]["premium_lag1"] = dist(null_prem1)
    out["null"]["premium_lag2"] = dist(null_prem2)
    out["null"]["named_case"] = dist(null_named)
    out["adjacency"]["premium_lag1"]["net_of_null"] = (
        out["adjacency"]["premium_lag1"]["mean"] - out["null"]["premium_lag1"]["mean"])
    out["adjacency"]["premium_lag1"]["p_value"] = float(
        np.mean(np.asarray(null_prem1) >= out["adjacency"]["premium_lag1"]["mean"]))
    out["adjacency"]["premium_lag2"]["net_of_null"] = (
        out["adjacency"]["premium_lag2"]["mean"] - out["null"]["premium_lag2"]["mean"])
    out["named_case"]["net_of_null"] = (out["named_case"]["roi"]
                                        - out["null"]["named_case"]["mean"])
    out["named_case"]["p_value"] = float(
        np.mean(np.asarray(null_named) >= out["named_case"]["roi"]))

    # family-wise: how many arms beat E[max]?
    em = out["null"]["family_max"]
    out["family"] = {
        "n_arms": len(ARMS),
        "E_max_null": em["mean"], "p95_max_null": em["p95"], "max_max": em["max"],
        "arms_beating_E_max": [a for a in ARMS
                               if out["arms"][a]["pooled_roi"] > em["mean"]],
        "arms_beating_p95_max": [a for a in ARMS
                                 if out["arms"][a]["pooled_roi"] > em["p95"]],
        "arms_passing_prereg": [a for a in ARMS
                                if out["arms"][a]["ci"]["lo"] > 0
                                and out["arms"][a]["beats_null_p95"]],
    }

    out["runtime_s"] = time.time() - t0
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {OUT}  [{out['runtime_s']:.1f}s]")
    return out


if __name__ == "__main__":
    r = main()
    for a in ARMS:
        s = r["arms"][a]
        print(f"{a:10s} ROI {100*s['pooled_roi']:+7.2f}%  n {s['pooled_n']:6.0f}  "
              f"cover {100*s['pooled_cover']:.3f}%  CI [{100*s['ci']['lo']:+.2f},"
              f"{100*s['ci']['hi']:+.2f}]  null {100*s['null']['mean']:+.2f}%  "
              f"net {100*s['net_of_null']:+.2f}  p {s['p_value']:.3f}  "
              f"MDE80 {100*s['mde80']:.2f}", file=sys.stderr)
