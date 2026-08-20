"""OVERFITTING CAPACITY + WALK-FORWARD CONFIG SELECTION.

Pre-registered in data/overfit_capacity_prereg.md
sha256 c0ec86dfe86ace509024f966b5a7943d83470136efb772009cd2de5153c53d00

ARM A  capacity      : tune on season s, score on s (IS) and on the other 18 (OOS)
ARM B  walk-forward  : tune on seasons 1..k, score on k+1, roll (PRIMARY)
ARM C  null          : 200 within-date permutations of (m_us, p_us) through A and B
ARM D  season phase  : cover/ROI by phase and month, profile correlation

READ-ONLY. No DB. No default changed. Nothing ships.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nbapred.threads  # noqa: E402
nbapred.threads.pin(1)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# --- PATH OVERRIDES ONLY (D173 re-run on the D170/D171 backfilled data).
# --- Defaults are byte-identical to D164's; no statistic is touched.
import os  # noqa: E402
FRAME = Path(os.environ.get("OC_FRAME") or (ROOT / "data" / "ats19_frame.csv.gz"))
_TAG = os.environ.get("OC_TAG", "")
OUT = ROOT / "data" / f"oc_capacity{_TAG}.json"

WIN = 100.0 / 110.0          # -110 payoff on a win
MIN_PER_SEASON = 100         # prereg §3 eligibility guard
MIN_HISTORY = 5              # prereg §4 ARM B
NDRAWS = 200                 # prereg §4 ARM C
SEED = 20260804

T_LEVELS = [0, 1, 2, 3, 4, 5]
SIDE_LEVELS = ["ALL", "HOME", "AWAY", "FAV", "DOG"]
PHASE_LEVELS = ["ALL", "EARLY", "MID", "LATE"]
BAND_LEVELS = ["ALL", "LE08", "M0818", "GT18", "GT08"]


# ---------------------------------------------------------------- statistics
def t_crit(dof: int) -> float:
    """Two-sided 95% t critical value (table; no scipy dependency)."""
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


def cluster_mean_t(vals) -> dict:
    """K-1 dof cluster-mean t interval on a vector of per-cluster estimates."""
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    K = len(v)
    if K < 2:
        return {"K": int(K), "mean": float(v[0]) if K else float("nan"),
                "sd": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "sig": False}
    m, sd = float(v.mean()), float(v.std(ddof=1))
    se = sd / np.sqrt(K)
    h = t_crit(K - 1) * se
    return {"K": K, "mean": m, "sd": sd, "se": float(se),
            "lo": m - h, "hi": m + h, "sig": bool((m - h) * (m + h) > 0)}


# ------------------------------------------------------------------ the frame
def load():
    df = pd.read_csv(FRAME)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["season", "game_date", "game_id"]).reset_index(drop=True)
    seasons = sorted(df["season"].unique())
    sidx = {s: i for i, s in enumerate(seasons)}
    df["s_i"] = df["season"].map(sidx)

    # phase: equal-count terciles of the within-season chronological index
    ph = np.empty(len(df), dtype=np.int8)
    for s, g in df.groupby("season", sort=False):
        n = len(g)
        r = np.arange(n)
        p = np.where(r < n / 3.0, 0, np.where(r < 2 * n / 3.0, 1, 2))
        ph[g.index.values] = p
    df["phase"] = ph
    df["month"] = df["game_date"].dt.month

    # date-slate id, for the within-date permutation null
    df["slate"] = df.groupby(["season", "game_date"], sort=False).ngroup()
    return df, seasons


def build_static(df):
    """Everything that does NOT depend on m_us / p_us."""
    n = len(df)
    S = np.zeros((n, df["s_i"].max() + 1), dtype=np.float32)
    S[np.arange(n), df["s_i"].values] = 1.0
    return {
        "n": n,
        "S": S,
        "open_margin": df["open_margin"].to_numpy(float),
        "actual": df["margin_actual"].to_numpy(float),
        "phase": df["phase"].to_numpy(np.int8),
        "slate": df["slate"].to_numpy(np.int64),
        "s_i": df["s_i"].to_numpy(np.int64),
    }


def payoff_and_masks(m_us, p_us, st):
    """Per-game payoff and the 600 config masks, for one (possibly permuted) run.

    Returns payoff (n,), M (600, n) uint8, and won/lost/push flags.
    """
    om, act = st["open_margin"], st["actual"]
    bet_home = m_us > om
    diff = act - om
    win = np.where(bet_home, diff > 0, diff < 0)
    push = diff == 0
    payoff = np.where(push, 0.0, np.where(win, WIN, -1.0))

    edge = np.abs(m_us - om)
    conf = np.abs(p_us - 0.5)

    # side masks
    fav = (bet_home & (om > 0)) | ((~bet_home) & (om < 0))
    dog = (bet_home & (om < 0)) | ((~bet_home) & (om > 0))
    side_masks = {
        "ALL": np.ones(st["n"], bool),
        "HOME": bet_home,
        "AWAY": ~bet_home,
        "FAV": fav,
        "DOG": dog,
    }
    band_masks = {
        "ALL": np.ones(st["n"], bool),
        "LE08": conf <= 0.08,
        "M0818": (conf > 0.08) & (conf <= 0.18),
        "GT18": conf > 0.18,
        "GT08": conf > 0.08,
    }
    ph = st["phase"]
    phase_masks = {"ALL": np.ones(st["n"], bool), "EARLY": ph == 0,
                   "MID": ph == 1, "LATE": ph == 2}
    t_masks = {t: edge >= t for t in T_LEVELS}

    M = np.empty((600, st["n"]), dtype=np.float32)
    keys = []
    i = 0
    for t in T_LEVELS:
        for sd in SIDE_LEVELS:
            for phn in PHASE_LEVELS:
                for bd in BAND_LEVELS:
                    M[i] = (t_masks[t] & side_masks[sd] & phase_masks[phn]
                            & band_masks[bd])
                    keys.append((t, sd, phn, bd))
                    i += 1
    return payoff, M, keys, win, push, bet_home


def agg(M, payoff, st):
    """(600, K) bet counts, payoff sums, win counts, push counts."""
    S = st["S"]
    cnt = M @ S
    pay = (M * payoff.astype(np.float32)) @ S
    return cnt, pay


def roi_of(pay, cnt):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, pay / np.maximum(cnt, 1e-9), np.nan)


# --------------------------------------------------------------------- ARM A
def arm_a(cnt, pay, K):
    """Tune on season s; report IS on s and OOS on the other K-1."""
    tot_c, tot_p = cnt.sum(1), pay.sum(1)
    out = []
    for s in range(K):
        elig = cnt[:, s] >= MIN_PER_SEASON
        if not elig.any():
            out.append(None)
            continue
        roi_s = roi_of(pay[:, s], cnt[:, s])
        cand = np.where(elig, roi_s, -np.inf)
        best = int(np.argmax(cand))
        # deterministic tie-break: larger n, then lexicographic (= lower index)
        ties = np.where(np.isclose(cand, cand[best], rtol=0, atol=1e-12))[0]
        if len(ties) > 1:
            best = int(ties[np.lexsort((ties, -cnt[ties, s]))[0]])
        oc = tot_c[best] - cnt[best, s]
        op = tot_p[best] - pay[best, s]
        per = [float(roi_of(pay[best, j], cnt[best, j])) if cnt[best, j] > 0
               else np.nan for j in range(K) if j != s]
        out.append({
            "cfg": best,
            "is_roi": float(roi_s[best]), "is_n": float(cnt[best, s]),
            "oos_roi_pooled": float(op / oc) if oc > 0 else np.nan,
            "oos_n": float(oc),
            "oos_roi_meanseason": float(np.nanmean(per)),
            "oos_per_season": per,
        })
    return out


# --------------------------------------------------------------------- ARM B
def arm_b(cnt, pay, K, min_hist=MIN_HISTORY):
    """Select on seasons 0..k-1, score on season k. k = min_hist .. K-1."""
    steps = []
    for k in range(min_hist, K):
        c_sel = cnt[:, :k].sum(1)
        p_sel = pay[:, :k].sum(1)
        elig = c_sel >= MIN_PER_SEASON * k
        if not elig.any():
            steps.append(None)
            continue
        roi_sel = roi_of(p_sel, c_sel)
        cand = np.where(elig, roi_sel, -np.inf)
        best = int(np.argmax(cand))
        ties = np.where(np.isclose(cand, cand[best], rtol=0, atol=1e-12))[0]
        if len(ties) > 1:
            best = int(ties[np.lexsort((ties, -c_sel[ties]))[0]])
        steps.append({
            "k": k, "cfg": best,
            "sel_roi": float(roi_sel[best]), "sel_n": float(c_sel[best]),
            "test_season_idx": k,
            "test_roi": float(roi_of(pay[best, k], cnt[best, k])),
            "test_n": float(cnt[best, k]),
            "test_pay": float(pay[best, k]),
        })
    return steps


def wf_pooled(steps):
    n = sum(s["test_n"] for s in steps if s)
    p = sum(s["test_pay"] for s in steps if s)
    return (p / n) if n > 0 else np.nan, n


# ----------------------------------------------------------------------- main
def main():
    df, seasons = load()
    K = len(seasons)
    st = build_static(df)
    m_us = df["m_us"].to_numpy(float)
    p_us = df["p_us"].to_numpy(float)

    payoff, M, keys, win, push, bet_home = payoff_and_masks(m_us, p_us, st)
    cnt, pay = agg(M, payoff, st)

    # ---- win/push counts per config-season, for cover rates
    Mw = M * win.astype(np.float32)
    Mp = M * push.astype(np.float32)
    wcnt = Mw @ st["S"]
    pcnt = Mp @ st["S"]

    res = {
        "prereg_sha256": "c0ec86dfe86ace509024f966b5a7943d83470136efb772009cd2de5153c53d00",
        "seasons": seasons, "n_games": int(st["n"]),
        "cardinality": len(keys),
        "keys": [list(map(str, k)) for k in keys],
    }

    # ============================================================== reference
    ref = 0  # cell 1 of 600 = T0/ALL/ALL/ALL  (D162 primary)
    ref_roi_season = [float(pay[ref, j] / cnt[ref, j]) for j in range(K)]
    ref_cov_season = [float(wcnt[ref, j] / (cnt[ref, j] - pcnt[ref, j]))
                      for j in range(K)]
    res["reference"] = {
        "cfg": keys[ref],
        "n": float(cnt[ref].sum()),
        "roi_pooled": float(pay[ref].sum() / cnt[ref].sum()),
        "cover_pooled": float(wcnt[ref].sum() / (cnt[ref].sum() - pcnt[ref].sum())),
        "roi_per_season": ref_roi_season,
        "cover_per_season": ref_cov_season,
        "ci": cluster_mean_t(ref_roi_season),
        "push_total": float(pcnt[ref].sum()),
    }
    # the D162 sub-windows, reproduced on this frame
    dev5 = [j for j, s in enumerate(seasons) if s >= "2021-22"]
    oos14 = [j for j, s in enumerate(seasons) if s < "2021-22"]
    for nm, idx in [("DEV5", dev5), ("OOS14", oos14)]:
        res["reference"][nm] = {
            "roi": float(pay[ref, idx].sum() / cnt[ref, idx].sum()),
            "n": float(cnt[ref, idx].sum()),
            "ci": cluster_mean_t([ref_roi_season[j] for j in idx]),
        }
    # the T>=3 cell (D162 §7), as the secondary fixed reference
    ref3 = keys.index((3, "ALL", "ALL", "ALL"))
    res["reference_T3"] = {
        "cfg": keys[ref3], "idx": ref3,
        "roi_pooled": float(pay[ref3].sum() / cnt[ref3].sum()),
        "n": float(cnt[ref3].sum()),
        "roi_per_season": [float(pay[ref3, j] / cnt[ref3, j]) for j in range(K)],
    }

    # ================================================================== ARM A
    A = arm_a(cnt, pay, K)
    res["arm_a"] = []
    for s, a in enumerate(A):
        res["arm_a"].append({
            "season": seasons[s], "cfg": list(map(str, keys[a["cfg"]])),
            "cfg_idx": a["cfg"],
            "is_roi": a["is_roi"], "is_n": a["is_n"],
            "oos_roi_pooled": a["oos_roi_pooled"],
            "oos_roi_meanseason": a["oos_roi_meanseason"],
            "oos_n": a["oos_n"],
            "decay": a["is_roi"] - a["oos_roi_pooled"],
            "oos_positive": bool(a["oos_roi_pooled"] > 0),
            "oos_seasons_positive": int(sum(1 for x in a["oos_per_season"]
                                            if np.isfinite(x) and x > 0)),
        })
    decays = [r["decay"] for r in res["arm_a"]]
    res["capacity"] = {
        "mean_is": float(np.mean([r["is_roi"] for r in res["arm_a"]])),
        "mean_oos_pooled": float(np.mean([r["oos_roi_pooled"] for r in res["arm_a"]])),
        "mean_oos_meanseason": float(np.mean([r["oos_roi_meanseason"] for r in res["arm_a"]])),
        "mean_decay": float(np.mean(decays)),
        "ci_decay": cluster_mean_t(decays),
        "hit_rate": int(sum(r["oos_positive"] for r in res["arm_a"])),
        "n_targets": K,
        "n_distinct_cfg": len({r["cfg_idx"] for r in res["arm_a"]}),
    }
    # jackknife over the 19 targets
    jk = [float(np.mean([d for i, d in enumerate(decays) if i != j]))
          for j in range(K)]
    res["capacity"]["jackknife"] = {"lo": float(min(jk)), "hi": float(max(jk)),
                                    "se": float(np.sqrt((K - 1) / K * np.sum(
                                        (np.array(jk) - np.mean(jk)) ** 2)))}

    # ================================================================== ARM B
    B = arm_b(cnt, pay, K)
    roi_wf, n_wf = wf_pooled(B)
    per = [b["test_roi"] for b in B]
    cfgs = [b["cfg"] for b in B]
    changes = sum(1 for i in range(1, len(cfgs)) if cfgs[i] != cfgs[i - 1])
    scored_idx = [b["test_season_idx"] for b in B]
    wcov = sum(wcnt[b["cfg"], b["test_season_idx"]] for b in B)
    pcov = sum(pcnt[b["cfg"], b["test_season_idx"]] for b in B)
    res["arm_b"] = {
        "min_history": MIN_HISTORY,
        "steps": [{"k": b["k"], "select_through": seasons[b["k"] - 1],
                   "test_season": seasons[b["test_season_idx"]],
                   "cfg": list(map(str, keys[b["cfg"]])), "cfg_idx": b["cfg"],
                   "sel_roi": b["sel_roi"], "sel_n": b["sel_n"],
                   "test_roi": b["test_roi"], "test_n": b["test_n"]} for b in B],
        "pooled_roi": float(roi_wf), "pooled_n": float(n_wf),
        "pooled_cover": float(wcov / (n_wf - pcov)),
        "ci": cluster_mean_t(per),
        "selection_changes": changes, "n_transitions": len(cfgs) - 1,
        "n_distinct_cfg": len(set(cfgs)),
        "seasons_positive": int(sum(1 for x in per if x > 0)),
        "n_scored": len(per),
        # §7 discriminator
        "early_block_roi": float(np.mean(per[:7])),
        "late_block_roi": float(np.mean(per[7:])),
        "slope_on_k": float(np.polyfit([b["k"] for b in B], per, 1)[0]),
        # ex-COVID (drop 2019-20, 2020-21 as SCORED seasons)
        "ex_covid_roi": float(
            sum(b["test_pay"] for b in B if seasons[b["test_season_idx"]]
                not in ("2019-20", "2020-21"))
            / sum(b["test_n"] for b in B if seasons[b["test_season_idx"]]
                  not in ("2019-20", "2020-21"))),
        # comparator (b): shipped fixed config on the SAME scored seasons
        "ref_same_seasons_roi": float(pay[ref, scored_idx].sum()
                                      / cnt[ref, scored_idx].sum()),
        "ref_same_seasons_ci": cluster_mean_t([ref_roi_season[j] for j in scored_idx]),
        "refT3_same_seasons_roi": float(pay[ref3, scored_idx].sum()
                                        / cnt[ref3, scored_idx].sum()),
        "paired_vs_ref_ci": cluster_mean_t(
            [per[i] - ref_roi_season[j] for i, j in enumerate(scored_idx)]),
    }

    # ================================================================== ARM D
    ph = st["phase"]
    s_i = st["s_i"]
    phase_names = ["EARLY", "MID", "LATE"]
    D = {"phase": {}, "month": {}, "per_season_profile": []}
    for pi, pn in enumerate(phase_names):
        m = ph == pi
        rows = []
        for j in range(K):
            sel = m & (s_i == j)
            n = sel.sum()
            rows.append({"roi": float(payoff[sel].sum() / n),
                         "cover": float(win[sel].sum() / (n - push[sel].sum())),
                         "n": int(n)})
        D["phase"][pn] = {
            "roi_pooled": float(payoff[m].sum() / m.sum()),
            "cover_pooled": float(win[m].sum() / (m.sum() - push[m].sum())),
            "n": int(m.sum()),
            "per_season_roi": [r["roi"] for r in rows],
            "per_season_cover": [r["cover"] for r in rows],
            "ci": cluster_mean_t([r["roi"] for r in rows]),
            "ci_cover": cluster_mean_t([r["cover"] for r in rows]),
            "seasons_positive": int(sum(1 for r in rows if r["roi"] > 0)),
        }
    mon = df["month"].to_numpy()
    for mm in sorted(set(mon.tolist())):
        m = mon == mm
        per_s = []
        for j in range(K):
            sel = m & (s_i == j)
            if sel.sum() >= 20:
                per_s.append(float(payoff[sel].sum() / sel.sum()))
        D["month"][int(mm)] = {
            "n": int(m.sum()),
            "roi_pooled": float(payoff[m].sum() / m.sum()),
            "cover_pooled": float(win[m].sum() / (m.sum() - push[m].sum())),
            "ci": cluster_mean_t(per_s), "K": len(per_s),
            "thin": bool(m.sum() < 200),
        }
    prof = np.array([[D["phase"][pn]["per_season_roi"][j] for pn in phase_names]
                     for j in range(K)])
    D["profile_matrix"] = prof.tolist()
    pairs = []
    for a in range(K):
        for b in range(a + 1, K):
            x, y = prof[a], prof[b]
            if x.std() > 0 and y.std() > 0:
                pairs.append(float(np.corrcoef(x, y)[0, 1]))
    D["profile_corr"] = {"mean": float(np.mean(pairs)), "sd": float(np.std(pairs, ddof=1)),
                         "n_pairs": len(pairs),
                         "frac_negative": float(np.mean([p < 0 for p in pairs])),
                         "median": float(np.median(pairs))}
    # D98's own three seasons, reproduced on the ATS profile
    d98 = {}
    for a, b in [("2023-24", "2024-25"), ("2024-25", "2025-26"), ("2023-24", "2025-26")]:
        ia, ib = seasons.index(a), seasons.index(b)
        d98[f"{a}|{b}"] = float(np.corrcoef(prof[ia], prof[ib])[0, 1])
    D["d98_pairs"] = d98
    res["arm_d"] = D

    # ================================================================== ARM C
    rng = np.random.default_rng(SEED)
    slate = st["slate"]
    order = np.argsort(slate, kind="stable")
    bounds = np.searchsorted(slate[order], np.arange(slate.max() + 2))
    null_cap, null_wf, null_is, null_oos, null_hit = [], [], [], [], []
    null_wf_per, null_prof = [], []
    for d in range(NDRAWS):
        perm = order.copy()
        for gi in range(len(bounds) - 1):
            a, b = bounds[gi], bounds[gi + 1]
            if b - a > 1:
                perm[a:b] = rng.permutation(perm[a:b])
        idx = np.empty_like(perm)
        idx[order] = perm
        mp, pp = m_us[idx], p_us[idx]
        po_n, M_n, _, win_n, push_n, _ = payoff_and_masks(mp, pp, st)
        c_n, p_n = agg(M_n, po_n, st)
        An = arm_a(c_n, p_n, K)
        dn = [a["is_roi"] - a["oos_roi_pooled"] for a in An if a]
        null_cap.append(float(np.mean(dn)))
        null_is.append(float(np.mean([a["is_roi"] for a in An if a])))
        null_oos.append(float(np.mean([a["oos_roi_pooled"] for a in An if a])))
        null_hit.append(int(sum(1 for a in An if a and a["oos_roi_pooled"] > 0)))
        Bn = arm_b(c_n, p_n, K)
        r, _ = wf_pooled(Bn)
        null_wf.append(float(r))
        null_wf_per.append([b["test_roi"] for b in Bn])
        # null phase profile correlation
        pm = np.array([[float(po_n[(ph == pi) & (s_i == j)].mean())
                        for pi in range(3)] for j in range(K)])
        pc = [float(np.corrcoef(pm[a], pm[b])[0, 1])
              for a in range(K) for b in range(a + 1, K)
              if pm[a].std() > 0 and pm[b].std() > 0]
        null_prof.append(float(np.mean(pc)))
        if (d + 1) % 25 == 0:
            print(f"  null draw {d+1}/{NDRAWS}", flush=True)

    def dist(x):
        x = np.asarray(x, float)
        return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)),
                "p05": float(np.percentile(x, 5)), "p50": float(np.percentile(x, 50)),
                "p95": float(np.percentile(x, 95)),
                "min": float(x.min()), "max": float(x.max())}

    res["arm_c"] = {
        "n_draws": NDRAWS, "seed": SEED,
        "capacity": dist(null_cap), "is_roi": dist(null_is),
        "oos_roi": dist(null_oos), "hit_rate": dist(null_hit),
        "wf_roi": dist(null_wf),
        "profile_corr": dist(null_prof),
        "wf_p_value": float(np.mean(np.asarray(null_wf) >= res["arm_b"]["pooled_roi"])),
        "capacity_p_value": float(np.mean(np.asarray(null_cap)
                                          >= res["capacity"]["mean_decay"])),
        "wf_per_step_mean": np.asarray(null_wf_per).mean(0).tolist(),
    }
    res["capacity"]["net_of_null"] = (res["capacity"]["mean_decay"]
                                      - res["arm_c"]["capacity"]["mean"])
    res["arm_b"]["net_of_null"] = (res["arm_b"]["pooled_roi"]
                                   - res["arm_c"]["wf_roi"]["mean"])
    res["arm_b"]["beats_null_p95"] = bool(
        res["arm_b"]["pooled_roi"] > res["arm_c"]["wf_roi"]["p95"])

    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {OUT}")
    return res


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: r[k] for k in ("cardinality", "capacity")},
                     indent=1, default=float)[:1500], file=sys.stderr)
