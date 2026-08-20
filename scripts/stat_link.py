#!/usr/bin/env python3
"""stat_link.py -- LINK-FUNCTION family test on the margin -> P(win) map.

Motivation: favorite-tail hole (heavy favorites cost ~+0.038/game vs market).
Is that a LINK problem (wrong shape/scale mapping margin to probability) or a
MARGIN problem (wrong point forecast)?  This script holds the margin FIXED and
varies only the link, so any delta is attributable to the link family alone.

Margin: margin_hat = 0.5*m_ff + 0.5*m_comp from data/component_pergame.csv
(rows with both non-null).  This file predates the schedule layer; that is fine
here because EVERY variant consumes the identical margin -- the comparison is
within-margin.  (It does mean margin_hat carries the known pre-D46 home bias:
mean error ~ -1.1 pts.  Documented, not corrected.)

Variants (all walk-forward: shape parameters fit on strictly-past games only,
refit at each calendar-month boundary, 400-game burn-in):
  (a) CONTROL   p = sigmoid(m / 7.2)                       [shipped link]
  (b) STUDENT-T p = T_nu(m / s_nu), nu fit walk-forward, SHAPE-ONLY:
                s_nu is pinned so dp/dm at m=0 equals the control's 1/(4*7.2).
                Same steepness at the money; only the TAILS differ.
  (c) POSTPRED  p = sigmoid(m / (7.2 * sqrt(1 + s2_param/171.61)))
                s2_param = max(0, Var_past(actual_margin - margin_hat) - 13.1^2)
                = the share of prediction-error variance in EXCESS of the
                irreducible game-noise floor, i.e. uncertainty about the true
                expected margin, which widens the effective link scale.
Supplementary (diagnostic, same gate, reported but not primary):
  (d) FREESCALE p = sigmoid(m / k), k fit walk-forward  -- separates "the 7.2
                scale is stale" from "the logistic FAMILY is wrong".
  (e) T-FREE    p = T_nu(m / s), nu AND s both fit walk-forward.

Gate: variant-vs-control PAIRED bootstrap, 2000 resamples, 95% percentile CI on
mean per-game log-loss delta.  Sign convention: delta = LL_control - LL_variant,
so POSITIVE = variant IMPROVES on control.  Shared bootstrap indices across
variants.  Also reported: heavy-favorite subset |p_ctrl - 0.5| > 0.35.

Read-only: DuckDB opened read_only=True (actual final scores only -- an OUTCOME,
not a market input).  Market p_mkt is carried as a REFERENCE benchmark only and
never enters any fit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SCALE = 7.2                 # shipped logistic scale
NOISE_SD = 13.1             # irreducible game-noise floor (prompt spec)
NOISE_VAR = NOISE_SD ** 2   # 171.61 -- used for BOTH the floor subtraction and
                            # the /171 denominator (they are the same quantity;
                            # the prompt's "171" is 13.1^2 rounded)
BURN_IN = 400
NBOOT = 2000
SEED = 20260730
EPS = 1e-6


# ---------------------------------------------------------------- utilities
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def logloss(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def t_scale_slope_matched(nu):
    """Scale s such that d/dm T_nu(m/s) at m=0 == d/dm sigmoid(m/7.2) at m=0.

    control slope = 1/(4*7.2);  t link slope = f_nu(0)/s  =>  s = 28.8*f_nu(0).
    This is the SHAPE-ONLY normalization: identical steepness at pick'em, so the
    only thing that can move log-loss is tail behaviour.
    """
    return 4.0 * SCALE * stats.t.pdf(0.0, nu)


# nu grid: log-spaced 1 -> 1000 plus the Gaussian limit (nu=1e6 ~ probit)
NU_GRID = np.unique(np.concatenate([
    np.round(np.exp(np.linspace(np.log(1.0), np.log(1000.0), 60)), 4),
    np.array([1e6]),
]))
# free-scale grid for (d)/(e).  Range must comfortably contain BOTH the logistic
# scale (~7-8) and the t scale (~11-13, since a t with df~13 needs s ~ SD/1.09)
# or the argmin pins to a grid edge.
K_GRID = np.round(np.arange(4.0, 18.01, 0.05), 3)


# ---------------------------------------------------------------- data load
def load() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / "component_pergame.csv", dtype={"game_id": str})
    n_all = len(d)
    d = d.dropna(subset=["m_ff", "m_comp"]).copy()
    d["m"] = 0.5 * d["m_ff"] + 0.5 * d["m_comp"]
    d["game_date"] = pd.to_datetime(d["game_date"])

    con = duckdb.connect(str(ROOT / "data" / "nba.duckdb"), read_only=True)
    g = con.execute(
        "SELECT game_id, team_abbrev, pts FROM nba_games "
        "WHERE game_id LIKE '002%' AND pts IS NOT NULL"
    ).fetchdf()
    con.close()
    pts = {(r.game_id, r.team_abbrev): r.pts for r in g.itertuples()}
    d["hp"] = [pts.get((r.game_id, r.home)) for r in d.itertuples()]
    d["ap"] = [pts.get((r.game_id, r.away)) for r in d.itertuples()]
    miss = int(d["hp"].isna().sum() + d["ap"].isna().sum())
    d = d.dropna(subset=["hp", "ap"]).copy()
    d["actual_margin"] = d["hp"] - d["ap"]
    # integrity: y must equal sign(actual margin)
    assert ((d["actual_margin"] > 0).astype(int) == d["y"]).all(), "y/margin mismatch"

    d = d.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    d["month"] = d["game_date"].dt.strftime("%Y-%m")          # calendar-month key
    print(f"[load] csv rows={n_all}  both-components={len(d) + 0}  "
          f"score-join-misses={miss}  span={d.game_date.min().date()}..{d.game_date.max().date()}")
    return d


# -------------------------------------------------- walk-forward parameters
def fit_nu(m, y, scale_free=False):
    """Grid-search nu (and optionally scale) minimizing log-loss on PAST games."""
    best = (np.inf, None, None)
    if scale_free:
        for nu in NU_GRID:
            z = np.divide.outer(m, K_GRID)                    # (n, len(K))
            p = stats.t.cdf(z, nu)
            ll = logloss(y[:, None], p).mean(axis=0)
            j = int(np.argmin(ll))
            if ll[j] < best[0]:
                best = (float(ll[j]), float(nu), float(K_GRID[j]))
    else:
        for nu in NU_GRID:
            s = t_scale_slope_matched(nu)
            p = stats.t.cdf(m / s, nu)
            ll = logloss(y, p).mean()
            if ll < best[0]:
                best = (float(ll), float(nu), float(s))
    return best[1], best[2]


def fit_k(m, y):
    """Grid-search logistic scale k on PAST games."""
    p = sigmoid(np.divide.outer(m, K_GRID))
    ll = logloss(y[:, None], p).mean(axis=0)
    return float(K_GRID[int(np.argmin(ll))])


def walk_forward(d: pd.DataFrame):
    """For every calendar month, fit each variant's shape parameter on games
    STRICTLY BEFORE the month start.  Months whose past-sample is < BURN_IN are
    undefined -> those games are dropped from the evaluation set."""
    months = sorted(d["month"].unique())
    m_all, y_all, am_all = d["m"].values, d["y"].values, d["actual_margin"].values
    dates = d["game_date"].values

    params = {}
    for mo in months:
        start = np.datetime64(f"{mo}-01")
        past = dates < start                    # STRICTLY past (< month start)
        n = int(past.sum())
        if n < BURN_IN:
            params[mo] = None
            continue
        mp, yp, ap = m_all[past], y_all[past], am_all[past]
        err = ap - mp
        s2_raw = float(np.var(err, ddof=1))     # variance about its own mean
        s2_mse = float(np.mean(err ** 2))       # robustness alt (bias included)
        s2_param = max(0.0, s2_raw - NOISE_VAR)
        s2_param_mse = max(0.0, s2_mse - NOISE_VAR)
        nu_b, s_b = fit_nu(mp, yp, scale_free=False)
        k_d = fit_k(mp, yp)
        nu_e, s_e = fit_nu(mp, yp, scale_free=True)
        params[mo] = dict(
            n_fit=n, err_mean=float(err.mean()), err_var=s2_raw, err_mse=s2_mse,
            s2_param=s2_param, s2_param_mse=s2_param_mse,
            widen=float(np.sqrt(1.0 + s2_param / NOISE_VAR)),
            widen_mse=float(np.sqrt(1.0 + s2_param_mse / NOISE_VAR)),
            nu_b=nu_b, s_b=s_b, k_d=k_d, nu_e=nu_e, s_e=s_e,
        )
    return params


def apply_links(d: pd.DataFrame, params: dict) -> pd.DataFrame:
    keep, rows = [], []
    for i, r in enumerate(d.itertuples()):
        pr = params.get(r.month)
        if pr is None:
            continue
        m = r.m
        p_a = float(sigmoid(m / SCALE))
        p_b = float(stats.t.cdf(m / pr["s_b"], pr["nu_b"]))
        p_c = float(sigmoid(m / (SCALE * pr["widen"])))
        p_c2 = float(sigmoid(m / (SCALE * pr["widen_mse"])))
        p_d = float(sigmoid(m / pr["k_d"]))
        p_e = float(stats.t.cdf(m / pr["s_e"], pr["nu_e"]))
        keep.append(i)
        rows.append((p_a, p_b, p_c, p_c2, p_d, p_e,
                     pr["nu_b"], pr["widen"], pr["k_d"], pr["nu_e"], pr["s_e"]))
    out = d.iloc[keep].reset_index(drop=True).copy()
    cols = ["p_a", "p_b", "p_c", "p_c_mse", "p_d", "p_e",
            "nu_b", "widen_c", "k_d", "nu_e", "s_e"]
    for j, c in enumerate(cols):
        out[c] = [x[j] for x in rows]
    return out


# ------------------------------------------------------------------- gating
def paired_boot(ll_ctrl, ll_var, rng, nboot=NBOOT):
    """Paired bootstrap on per-game log-loss delta (ctrl - variant)."""
    d = np.asarray(ll_ctrl) - np.asarray(ll_var)
    n = len(d)
    if n == 0:
        return dict(n=0, delta=float("nan"), lo=float("nan"), hi=float("nan"))
    idx = rng.integers(0, n, size=(nboot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return dict(n=int(n), delta=float(d.mean()), lo=float(lo), hi=float(hi),
                p_wrongside=float(np.mean(means <= 0)))


def verdict(g):
    if g["n"] == 0:
        return "BLOCKED"
    if g["lo"] > 0:
        return "PASS"
    if g["hi"] < 0:
        return "FAIL"
    return "NS"


def load_shipped() -> pd.DataFrame:
    """ROBUSTNESS ARM: the ACTUAL shipped margin, recovered exactly by inverting
    the production link (p_us = sigmoid(m/7.2) => m = 7.2*logit(p_us)).  This
    includes the D46 schedule layer that component_pergame.csv predates, so it
    answers the production question directly: should PRODUCTION change its link?
    """
    d = pd.read_csv(ROOT / "data" / "capstone_pergame.csv", dtype={"game_id": str})
    d["game_date"] = pd.to_datetime(d["game_date"])
    d["m"] = SCALE * np.log(d["p_us"] / (1 - d["p_us"]))
    con = duckdb.connect(str(ROOT / "data" / "nba.duckdb"), read_only=True)
    g = con.execute(
        "SELECT game_id, team_abbrev, pts FROM nba_games "
        "WHERE game_id LIKE '002%' AND pts IS NOT NULL"
    ).fetchdf()
    con.close()
    pts = {(r.game_id, r.team_abbrev): r.pts for r in g.itertuples()}
    d["hp"] = [pts.get((r.game_id, r.home)) for r in d.itertuples()]
    d["ap"] = [pts.get((r.game_id, r.away)) for r in d.itertuples()]
    d = d.dropna(subset=["hp", "ap"]).copy()
    d["actual_margin"] = d["hp"] - d["ap"]
    assert ((d["actual_margin"] > 0).astype(int) == d["y"]).all(), "y/margin mismatch"
    # round-trip check: our reconstructed control must reproduce p_us exactly
    assert np.allclose(sigmoid(d["m"] / SCALE), d["p_us"], atol=1e-9), "invert failed"
    d = d.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    d["month"] = d["game_date"].dt.strftime("%Y-%m")
    print(f"[load-shipped] n={len(d)} span={d.game_date.min().date()}"
          f"..{d.game_date.max().date()} (margin recovered by logit inversion)")
    return d


def arm(d: pd.DataFrame, label: str):
    params = walk_forward(d)
    ev = apply_links(d, params)
    print(f"[eval] burn-in {BURN_IN} games -> evaluated n={len(ev)} "
          f"({ev.game_date.min().date()}..{ev.game_date.max().date()}); "
          f"dropped {len(d) - len(ev)} burn-in games")

    variants = {"b_studentt": "p_b", "c_postpred": "p_c",
                "d_freescale": "p_d", "e_t_free": "p_e",
                "c2_postpred_mse": "p_c_mse"}
    ll = {"a_control": logloss(ev.y.values, ev.p_a.values)}
    for k, c in variants.items():
        ll[k] = logloss(ev.y.values, ev[c].values)
    ll_mkt = logloss(ev.y.values, ev.p_mkt.values)

    heavy = (np.abs(ev.p_a.values - 0.5) > 0.35)
    print(f"[heavy-fav] |p_ctrl-0.5|>0.35 subset n={int(heavy.sum())} "
          f"({100 * heavy.mean():.1f}% of eval)")

    res = {"meta": {
        "n_eval": int(len(ev)), "n_heavy": int(heavy.sum()),
        "burn_in": BURN_IN, "nboot": NBOOT, "seed": SEED,
        "noise_var_used": NOISE_VAR,
        "ll_control_pooled": float(ll["a_control"].mean()),
        "ll_market_pooled": float(ll_mkt.mean()),
        "gap_to_market": float(ll["a_control"].mean() - ll_mkt.mean()),
        "ll_control_heavy": float(ll["a_control"][heavy].mean()),
        "ll_market_heavy": float(ll_mkt[heavy].mean()),
        "gap_to_market_heavy": float(ll["a_control"][heavy].mean() - ll_mkt[heavy].mean()),
    }, "gates": {}, "per_season": {}, "params": {}}

    # parameter trajectory
    for c in ("nu_b", "widen_c", "k_d", "nu_e", "s_e"):
        res["params"][c] = dict(first=float(ev[c].iloc[0]), last=float(ev[c].iloc[-1]),
                                median=float(ev[c].median()),
                                min=float(ev[c].min()), max=float(ev[c].max()))
    print("[params] " + "  ".join(
        f"{c}: {res['params'][c]['first']:.3f}->{res['params'][c]['last']:.3f} "
        f"(med {res['params'][c]['median']:.3f})" for c in ("nu_b", "widen_c", "k_d")))

    seasons = list(dict.fromkeys(ev.season))
    for name in variants:
        rng = np.random.default_rng(SEED)          # shared indices across variants
        g_all = paired_boot(ll["a_control"], ll[name], rng)
        rng_h = np.random.default_rng(SEED + 1)
        g_h = paired_boot(ll["a_control"][heavy], ll[name][heavy], rng_h)
        per = {}
        for s in seasons:
            msk = (ev.season == s).values
            rng_s = np.random.default_rng(SEED + 7)
            per[s] = paired_boot(ll["a_control"][msk], ll[name][msk], rng_s)
            per[s]["verdict"] = verdict(per[s])
        res["gates"][name] = {
            "pooled": g_all | {"verdict": verdict(g_all)},
            "heavy_fav": g_h | {"verdict": verdict(g_h)},
            "per_season": per,
            "ll_variant_pooled": float(ll[name].mean()),
            "ll_variant_heavy": float(ll[name][heavy].mean()),
        }
        print(f"\n[gate] {name}: pooled d={g_all['delta']:+.5f} "
              f"CI({g_all['lo']:+.5f},{g_all['hi']:+.5f}) {verdict(g_all)}   "
              f"heavy d={g_h['delta']:+.5f} CI({g_h['lo']:+.5f},{g_h['hi']:+.5f}) "
              f"{verdict(g_h)} [n={g_h['n']}]")
        for s in seasons:
            q = per[s]
            print(f"        {s}: d={q['delta']:+.5f} CI({q['lo']:+.5f},{q['hi']:+.5f}) "
                  f"{q['verdict']} [n={q['n']}]")

    for s in seasons:
        msk = (ev.season == s).values
        res["per_season"][s] = {
            "n": int(msk.sum()),
            "ll_control": float(ll["a_control"][msk].mean()),
            "ll_market": float(ll_mkt[msk].mean()),
            **{f"ll_{k}": float(ll[k][msk].mean()) for k in variants},
        }

    # ---------------------------------------------------------------------
    # ORACLE LINK CEILING (non-PIT, in-sample -> an OPTIMISTIC UPPER BOUND on
    # what ANY link can buy).  If even a link fit ON the evaluation data with
    # hindsight recovers ~nothing, the favorite-tail hole is definitively a
    # MARGIN problem, not a LINK problem.  Not gated; a bound, not a candidate.
    from sklearn.isotonic import IsotonicRegression
    mv, yv = ev.m.values, ev.y.values
    iso = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip")
    p_iso = iso.fit(mv, yv).predict(mv)                 # best monotone map, in-sample
    k_or = fit_k(mv, yv)                                # best logistic scale, in-sample
    p_kor = sigmoid(mv / k_or)
    nu_or, s_or = fit_nu(mv, yv, scale_free=True)       # best t, in-sample
    p_tor = stats.t.cdf(mv / s_or, nu_or)
    orc = {}
    for lbl, p in [("isotonic_monotone", p_iso), ("logistic_bestk", p_kor),
                   ("studentt_best", p_tor)]:
        l = logloss(yv, p)
        orc[lbl] = dict(ll=float(l.mean()),
                        gain_vs_control=float(ll["a_control"].mean() - l.mean()),
                        ll_heavy=float(l[heavy].mean()),
                        gain_heavy=float(ll["a_control"][heavy].mean() - l[heavy].mean()))
    orc["fitted"] = dict(k_oracle=k_or, nu_oracle=nu_or, s_oracle=s_or)
    res["oracle_link_ceiling_in_sample"] = orc
    print("\n[ORACLE link ceiling — in-sample, hindsight, NOT achievable]:")
    for k2 in ("isotonic_monotone", "logistic_bestk", "studentt_best"):
        v = orc[k2]
        print(f"   {k2:20s} ll={v['ll']:.5f} gain={v['gain_vs_control']:+.5f}  "
              f"heavy ll={v['ll_heavy']:.5f} gain={v['gain_heavy']:+.5f}")
    print(f"   fitted: k={k_or}  nu={nu_or}  s={s_or}   (control k=7.2)")

    # ---------------------------------------------------------------------
    # WHERE THE HOLE ACTUALLY IS: our-confidence tail vs market-confidence tail.
    # A monotone link can only re-map p as a function of m -- it CANNOT move one
    # game without moving every game at the same margin.  So if the hole lives
    # in games the MARKET calls heavy favorites but WE DO NOT, no link can reach.
    tails = {}
    for lbl, msk in [
        ("ours_gt35", np.abs(ev.p_a.values - 0.5) > 0.35),
        ("mkt_gt35", np.abs(ev.p_mkt.values - 0.5) > 0.35),
        ("mkt_gt30", np.abs(ev.p_mkt.values - 0.5) > 0.30),
        ("mkt_gt40", np.abs(ev.p_mkt.values - 0.5) > 0.40),
        ("mkt_gt35_and_ours_lt35", (np.abs(ev.p_mkt.values - 0.5) > 0.35)
         & (np.abs(ev.p_a.values - 0.5) <= 0.35)),
    ]:
        if msk.sum() == 0:
            continue
        tails[lbl] = dict(n=int(msk.sum()),
                          ll_control=float(ll["a_control"][msk].mean()),
                          ll_market=float(ll_mkt[msk].mean()),
                          gap=float(ll["a_control"][msk].mean() - ll_mkt[msk].mean()),
                          ll_iso_oracle=float(logloss(yv, p_iso)[msk].mean()),
                          gap_iso_oracle=float(logloss(yv, p_iso)[msk].mean()
                                               - ll_mkt[msk].mean()))
    res["tail_localisation"] = tails
    print("\n[tail localisation] where the favorite-tail hole actually lives:")
    for k2, v in tails.items():
        print(f"   {k2:24s} n={v['n']:4d} ctrl={v['ll_control']:.4f} "
              f"mkt={v['ll_market']:.4f} gap={v['gap']:+.4f}   "
              f"iso-oracle gap={v['gap_iso_oracle']:+.4f}")

    # calibration of the favorite tail, favorite-side folded
    fold = lambda p, y: (np.where(p > .5, p, 1 - p), np.where(p > .5, y, 1 - y))
    cal = {}
    for lbl, col in [("control", "p_a"), ("studentt", "p_b"), ("postpred", "p_c"),
                     ("freescale", "p_d"), ("market", "p_mkt")]:
        pf, yf = fold(ev[col].values, ev.y.values)
        t = pf > 0.85
        cal[lbl] = dict(n=int(t.sum()), pred=float(pf[t].mean()), act=float(yf[t].mean()))
    res["fav_tail_calibration_p_gt_085"] = cal
    print("\n[calibration] favorite-side p>0.85 (folded):")
    for k, v in cal.items():
        print(f"   {k:10s} n={v['n']:4d} pred={v['pred']:.4f} act={v['act']:.4f} "
              f"bias={v['pred'] - v['act']:+.4f}")

    ev.to_csv(ROOT / "data" / f"stat_link_pergame_{label}.csv", index=False)
    return res


def main():
    all_res = {}
    print("=" * 74)
    print("ARM 1 (PRIMARY, as specified): margin_hat = 0.5*m_ff + 0.5*m_comp")
    print("=" * 74)
    all_res["primary_component_margin"] = arm(load(), "component")

    print("\n" + "=" * 74)
    print("ARM 2 (ROBUSTNESS): the ACTUAL SHIPPED margin (incl. D46 schedule")
    print("layer), recovered by inverting production's logistic. Same links.")
    print("=" * 74)
    all_res["robustness_shipped_margin"] = arm(load_shipped(), "shipped")

    out = ROOT / "data" / "stat_link_results.json"
    out.write_text(json.dumps(all_res, indent=2, default=str))
    print(f"\n[done] -> {out}")
    print("LINK_DONE")


if __name__ == "__main__":
    sys.exit(main())
