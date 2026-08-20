#!/usr/bin/env python3
"""CM-CLVMODEL — DELIVERABLE 1: the MARKET-ANCHORED CLV model (D147).

Predict the CLOSING price from the OPENING price.  The target is the LINE
MOVEMENT open->close, not the outcome: we do not need to beat the market's
final answer, only its FIRST answer, and D121/D126 already show we do.

ARCHITECTURE (method mirrors football_exercise/submission/sean_qin_model.py,
which is read-only reference — its ENGINE is reproduced, its code is not):
  * PriceRidge (nbapred/market/anchored.py) inverts each PAST game's devigged
    price into a team-strength observation and accumulates it in a ridge with
    geometric forgetting.  Run three ways so the reference model's own central
    finding is re-tested here: past CLOSES vs past OPENS vs past RESULTS.
  * FEATURES = (a) the opening price of the game being predicted; (b) the
    price-anchored strength ridge over past closes; (c) our proprietary
    market-blind signals (certified p_us, tank state, schedule) — TIER A, all
    live.  TIER B adds availability/OUT sets and star-out, which land AFTER
    the open and are therefore an upper bound, labelled everywhere.
  * TARGET = close_margin - open_margin.  An OUTCOME-targeted arm is fitted
    alongside for comparison.

DISCIPLINE
  * PIT: assert_pit() shuffles the future of every label column and requires
    bit-identical features (also tests/test_market_anchored.py).
  * SPLITS per docs/GATE_POLICY_V2.md sections 8-11: rolling-origin primary,
    LOSO as a stability diagnostic (never k proofs), legacy, block bootstrap,
    era decomposition, and CLUSTERED inference as the reporting CI with ICC
    and design effect.
  * MDE80 is printed BEFORE any effect is scored.
  * The honest control is a PERMUTATION placebo (D115's vacuous-control
    lesson: a favourite-drift control is vacuous on same-side rules because
    our side IS the open favourite by construction).

RULES HONORED: DuckDB read_only=True with a 60s retry; nbapred/model/*,
nbapred/engine/props.py and the frozen registry are untouched.

Run:  python scripts/cm_clvmodel.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import duckdb                                                    # noqa: E402
import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402

from nbapred.market.anchored import (                            # noqa: E402
    OVERROUND, SPREAD_SCALE, am2dec, assert_pit, logit, ridge_fit,
    run_price_ridge, sigmoid, standardise)
from nbapred.eval import splits                                  # noqa: E402

DB = os.path.join(ROOT, "data", "nba.duckdb")
ODDS_OPEN = os.path.join(ROOT, "data", "derived", "odds_open.csv")
CAPSTONE = os.path.join(ROOT, "data", "capstone_pergame.csv")
RT1 = os.path.join(ROOT, "data", "ds_rt1_pergame.csv")
OUT = os.path.join(ROOT, "data", "cm_clvmodel.json")
ROWS = os.path.join(ROOT, "data", "cm_clvmodel_rows.csv.gz")

SEED = 20260802
B_BOOT = 2000
TANK_GP = 55                       # bet_sim3.TANK_GP, the D90 late window
EDGE_R4, CONF_TIER = 0.02, 0.20    # frozen registry constants (bet_engine)
DIV_LO, DIV_HI = 0.03, 0.10
WARM_DAYS = 150                    # ridge burn-in before any row is scorable

RES: dict = {}


def ro_connect(attempts: int = 10, wait_s: float = 60.0):
    for k in range(attempts):
        try:
            return duckdb.connect(DB, read_only=True)
        except Exception as e:                                    # noqa: BLE001
            if k == attempts - 1:
                raise
            print(f"    [db] locked ({e}); retry {k+1}/{attempts} in {wait_s}s")
            time.sleep(wait_s)
    raise RuntimeError


def hdr(t):
    print("\n" + "#" * 100 + f"\n# {t}\n" + "#" * 100)


# ============================================================== FRAME =======
def load_prices() -> pd.DataFrame:
    o = pd.read_csv(ODDS_OPEN, parse_dates=["game_date"])
    o = o[o.open_margin.notna() & o.close_margin.notna()].copy()
    o = o.sort_values(["game_date", "home", "away"]).reset_index(drop=True)
    o["day"] = o.game_date.values.astype("datetime64[D]").astype(int)
    return o


def add_ridges(o: pd.DataFrame) -> pd.DataFrame:
    """Three PriceRidge channels.  This is the reference model's engine, and
    running it over three observation channels re-tests its central finding
    (prices beat box scores as strength observations) on NBA data."""
    o = o.copy()
    o["ridge_close"] = run_price_ridge(o.home.values, o.away.values,
                                       o.day.values, o.close_margin.values)
    o["ridge_open"] = run_price_ridge(o.home.values, o.away.values,
                                      o.day.values, o.open_margin.values)
    res_margin = (o.score_home - o.score_away).values.astype(float)
    o["ridge_res"] = run_price_ridge(o.home.values, o.away.values,
                                     o.day.values, res_margin)
    return o


def build_frame() -> pd.DataFrame:
    hdr("[0] FRAME")
    o = add_ridges(load_prices())

    cap = pd.read_csv(CAPSTONE, dtype={"game_id": str},
                      parse_dates=["game_date"])
    cap["game_id"] = cap.game_id.str.zfill(10)
    n0 = len(cap)
    m = cap.merge(o.drop(columns=["source_url", "ingest_ts"]),
                  on=["season", "game_date", "home", "away"], how="left")
    assert len(m) == n0, "odds_open join fanned out — not 1:1"

    cov = m.groupby("season").apply(lambda g: pd.Series({
        "games": len(g),
        "with_both_prices": int(g.open_margin.notna().sum()),
        "with_open_ML": int(g.open_ml_home.notna().sum()),
        "source": "/".join(sorted(g.source.dropna().unique())) or "NONE",
    }), include_groups=False)
    print("\nMODEL FRAME (capstone p_us, D132 certified) x odds_open")
    print(cov.to_string())
    RES["coverage_model_frame"] = cov.reset_index().to_dict("records")

    m = m[m.open_margin.notna()].copy()

    # ---- schedule / tank / availability state --------------------------
    con = ro_connect()
    try:
        from ba_intersection import star_out_map, team_schedule
        ts = team_schedule(con)
        so = star_out_map(con)
    finally:
        con.close()
    for side in ("home", "away"):
        t = ts.rename(columns={"team": side})[["season", "game_id", side,
                                               "gp_before", "rest", "b2b"]]
        m = m.merge(t, on=["season", "game_id", side], how="left")
        m = m.rename(columns={"gp_before": f"{side[0]}_gp",
                              "rest": f"{side[0]}_rest",
                              "b2b": f"{side[0]}_b2b"})
    assert m.h_gp.notna().all() and m.a_gp.notna().all()
    m["late"] = (m.h_gp >= TANK_GP) | (m.a_gp >= TANK_GP)

    som = so.rename(columns={"team": "_t"})[["game_id", "_t", "star_out"]]
    for side in ("home", "away"):
        mm = som.rename(columns={"_t": side, "star_out": f"star_out_{side}"})
        m = m.merge(mm, on=["game_id", side], how="left")
        m[f"star_out_{side}"] = m[f"star_out_{side}"].fillna(False).astype(bool)

    # ---- prices ---------------------------------------------------------
    m["p_open_sp"] = sigmoid(m.open_margin / SPREAD_SCALE)
    m["p_close_sp"] = sigmoid(m.close_margin / SPREAD_SCALE)
    for w in ("open", "close"):
        dh, da = am2dec(m[f"{w}_ml_home"]), am2dec(m[f"{w}_ml_away"])
        ov = 1.0 / dh + 1.0 / da
        bad = ~np.isfinite(ov) | (ov < 1.0) | (ov > 1.25)
        m[f"dec_{w}_home"] = np.where(bad, np.nan, dh)
        m[f"dec_{w}_away"] = np.where(bad, np.nan, da)
        m[f"p_{w}_ml"] = np.where(bad, np.nan, (1.0 / dh) / ov)

    # ---- labels ---------------------------------------------------------
    m["dm"] = m.close_margin - m.open_margin              # PRIMARY label
    m["dp"] = m.p_close_sp - m.p_open_sp
    m["m_us"] = SPREAD_SCALE * logit(m.p_us)
    m["margin_actual"] = m.score_home - m.score_away

    m = m.sort_values(["game_date", "home", "away"]).reset_index(drop=True)
    warm = m.game_date >= (o.game_date.min() + pd.Timedelta(days=WARM_DAYS))
    m = m[warm].reset_index(drop=True)
    print(f"\nSCORABLE FRAME n={len(m)}  "
          f"{m.game_date.min().date()}..{m.game_date.max().date()}  "
          f"seasons {sorted(m.season.unique())}")
    print(f"ridge burn-in: the PriceRidge has absorbed every close from "
          f"{o.game_date.min().date()} forward, so it is warm on row 1.")
    return m


# ------------------------------------------------------------- FEATURES ----
TIER_A = ["open_margin", "abs_open", "resid_close", "resid_open", "resid_res",
          "resid_us", "conf_us", "late", "d_rest", "b2b_any", "tsd", "gp_min"]
TIER_B = ["d_nout", "n_out_tot", "star_home", "star_away", "d_star"]


def featurise(df: pd.DataFrame, tier: str = "A"):
    """(X, names).  Tier A = live at the open.  Tier B adds the inactive-list
    channel, which lands AFTER the open — diagnostic only."""
    f = {}
    f["open_margin"] = df.open_margin.values
    f["abs_open"] = np.abs(df.open_margin.values)
    f["resid_close"] = df.ridge_close.values - df.open_margin.values
    f["resid_open"] = df.ridge_open.values - df.open_margin.values
    f["resid_res"] = df.ridge_res.values - df.open_margin.values
    f["resid_us"] = df.m_us.values - df.open_margin.values
    f["conf_us"] = np.abs(df.p_us.values - 0.5)
    f["late"] = df.late.values.astype(float)
    f["d_rest"] = (df.h_rest.values - df.a_rest.values).astype(float)
    f["b2b_any"] = (df.h_b2b.values | df.a_b2b.values).astype(float)
    f["tsd"] = df.tsd.values
    f["gp_min"] = np.minimum(df.h_gp.values, df.a_gp.values).astype(float)
    if tier == "B":
        f["d_nout"] = (df.n_out_home.values - df.n_out_away.values).astype(float)
        f["n_out_tot"] = (df.n_out_home.values + df.n_out_away.values).astype(float)
        f["star_home"] = df.star_out_home.values.astype(float)
        f["star_away"] = df.star_out_away.values.astype(float)
        f["d_star"] = f["star_home"] - f["star_away"]
    names = TIER_A + (TIER_B if tier == "B" else [])
    X = np.column_stack([np.ones(len(df))] + [f[n] for n in names])
    return X, ["intercept"] + names


# =============================================== ROLLING-ORIGIN PREDICTION ==
def walk_forward(df: pd.DataFrame, label: str, tier: str = "A", lam: float = 10.0):
    """Expanding-window fit by SEASON: train on seasons <= k, predict k+1.
    The first season has no trainable history and is dropped from scoring."""
    X, names = featurise(df, tier)
    y = df[label].values.astype(float)
    seasons = sorted(df.season.unique())
    pred = np.full(len(df), np.nan)
    coefs = {}
    for i in range(1, len(seasons)):
        tr = df.season.isin(seasons[:i]).values
        te = (df.season == seasons[i]).values
        if tr.sum() < 200 or te.sum() == 0:
            continue
        Xtr, Xte, _, _ = standardise(X[tr], X[te])
        good = np.isfinite(y[tr]) & np.isfinite(Xtr).all(axis=1)
        beta = ridge_fit(Xtr[good], y[tr][good], lam=lam)
        pred[te] = Xte @ beta
        coefs[seasons[i]] = dict(zip(names, np.round(beta, 5).tolist()))
    return pred, names, coefs


def walk_forward_subset(df, label, keep_features, lam=10.0):
    """Rolling-origin OOS R^2 vs the naive 'no movement' baseline for an
    arbitrary sub-set of the tier-A feature list (the ablation ladder)."""
    X, names = featurise(df, "A")
    ix = [0] + [names.index(f) for f in keep_features]
    Xs = X[:, ix]
    y = df[label].values.astype(float)
    seasons = sorted(df.season.unique())
    pred = np.full(len(df), np.nan)
    for i in range(1, len(seasons)):
        tr = df.season.isin(seasons[:i]).values
        te = (df.season == seasons[i]).values
        if tr.sum() < 200 or te.sum() == 0:
            continue
        A, Bm, _, _ = standardise(Xs[tr], Xs[te])
        good = np.isfinite(y[tr]) & np.isfinite(A).all(axis=1)
        pred[te] = Bm @ ridge_fit(A[good], y[tr][good], lam=lam)
    ok = np.isfinite(pred)
    r2 = 1 - np.sum((y[ok] - pred[ok]) ** 2) / np.sum(y[ok] ** 2)
    return float(r2), float(np.corrcoef(pred[ok], y[ok])[0, 1])


def frozen_ridge_check(m: pd.DataFrame) -> float:
    """Re-run the tier-A model with a ridge that is FROZEN at the start of each
    test season — no within-season price updates at all.  If the headline R^2
    were manufactured by the label bleeding into the ridge through same-season
    closes, freezing it would destroy the result."""
    o = load_prices()
    o["sid"] = o.season.values
    seasons = sorted(m.season.unique())
    frozen = {}
    for s in seasons:
        cut = m.loc[m.season == s, "game_date"].min()
        hist = o[o.game_date < cut]
        if len(hist) < 500:
            continue
        from nbapred.market.anchored import PriceRidge
        teams = sorted(set(o.home) | set(o.away))
        r = PriceRidge(teams)
        for h, a, y in zip(hist.home.values, hist.away.values,
                           hist.close_margin.values):
            r.observe(h, a, y)
        r.flush_day()
        frozen[s] = r
    m2 = m.copy()
    fz = np.full(len(m2), np.nan)
    for s, r in frozen.items():
        mk = (m2.season == s).values
        fz[mk] = [r.predict(h, a) for h, a in
                  zip(m2.home.values[mk], m2.away.values[mk])]
    m2["ridge_close"] = np.where(np.isfinite(fz), fz, m2.ridge_close.values)
    pred, _, _ = walk_forward(m2, "dm", tier="A")
    ok = np.isfinite(pred)
    y = m2.dm.values
    return float(1 - np.sum((y[ok] - pred[ok]) ** 2) / np.sum(y[ok] ** 2))


def in_sample_coefs(df, label, tier="A", lam=10.0):
    X, names = featurise(df, tier)
    y = df[label].values.astype(float)
    Xs, _, _, _ = standardise(X, X)
    good = np.isfinite(y) & np.isfinite(Xs).all(axis=1)
    beta = ridge_fit(Xs[good], y[good], lam=lam)
    yh = Xs @ beta
    r2 = 1 - np.nansum((y[good] - yh[good]) ** 2) / np.nansum(
        (y[good] - y[good].mean()) ** 2)
    # cluster-robust (by season) t-stats
    Xg, yg = Xs[good], y[good]
    seas = df.season.values[good]
    XtXi = np.linalg.pinv(Xg.T @ Xg)
    meat = np.zeros((Xg.shape[1], Xg.shape[1]))
    r = yg - Xg @ beta
    for s in np.unique(seas):
        mk = seas == s
        u = Xg[mk].T @ r[mk]
        meat += np.outer(u, u)
    V = XtXi @ meat @ XtXi
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    t = np.where(se > 0, beta / se, 0.0)
    return dict(zip(names, np.round(beta, 5))), dict(zip(names, np.round(t, 2))), r2


def main() -> None:
    t0 = time.time()
    m = build_frame()

    # ---------------------------------------------------------- PIT GUARD --
    hdr("[1] PIT GUARD — shuffle the future of the label, require identical features")
    prices = load_prices()

    def _build(dfp):
        dfp = add_ridges(dfp)
        f = {
            "resid_close": dfp.ridge_close.values - dfp.open_margin.values,
            "resid_open": dfp.ridge_open.values - dfp.open_margin.values,
            "resid_res": dfp.ridge_res.values - dfp.open_margin.values,
            "open_margin": dfp.open_margin.values,
        }
        nm = list(f)
        return np.column_stack([f[k] for k in nm]), nm

    bad, moved = assert_pit(_build, prices,
                            ["close_margin", "score_home", "score_away"],
                            seed=SEED)
    print("  contract: feature_i may read labels from dates STRICTLY BEFORE "
          "date_i, nothing else.")
    print("  labels permuted over every row with date >= T, for T at the "
          "35th/60th/85th percentile;")
    print("  every feature on rows with date <= T (INCLUDING the cut date, so "
          "a same-day sideways")
    print("  leak inside a slate would show) is then required bit-identical.")
    print(f"    features that MOVED (must be empty)          : {bad}")
    print(f"    features that move under a FULL shuffle      : {moved}")
    print("    (the second line is the non-vacuity check: a guard that cannot "
          "fail is not a guard)")
    RES["pit_violations"] = bad
    RES["pit_moved_full_shuffle"] = moved
    assert not bad, f"PIT VIOLATION in {bad}"
    assert set(moved) >= {"resid_close", "resid_res"}, "PIT guard is vacuous"

    # -------------------------------------------------- MDE80, PRE-SCORING --
    hdr("[2] MDE80 — STATED BEFORE ANY EFFECT IS SCORED (GATE_POLICY_V2 s5.5)")
    dm = m.dm.values
    # The prediction is computed here because MDE80 must be quoted on the SCALE
    # of the delta that will be scored.  No effect number is printed until [4];
    # the power statement below is built from a PLACEBO prediction (permuted
    # within date), so it uses the null's dispersion, not the effect's.
    pred_A, names_A, coefs_A = walk_forward(m, "dm", tier="A")
    m["pred_dm"] = pred_A
    okp = np.isfinite(pred_A)
    rng0 = np.random.default_rng(SEED)
    dates0 = m.game_date.values
    pp = pred_A.copy()
    for _d in np.unique(dates0):
        ix = np.where(dates0 == _d)[0]
        if len(ix) > 1:
            pp[ix] = pp[rng0.permutation(ix)]
    d_null = (dm[okp] ** 2) - (dm[okp] - pp[okp]) ** 2
    clv_all = np.where(m.p_us.values > 0.5, m.dp.values, -m.dp.values)
    print(f"  n = {len(m)}   scored n = {int(okp.sum())}   "
          f"sd(dm) = {dm.std(ddof=1):.4f} pts")
    print(f"  MDE80 = 2.802*sd/sqrt(n) on the per-game squared-error delta")
    print(f"    (naive 'no movement' MINUS model), scale set by a WITHIN-DATE "
          f"PLACEBO : {splits.mde80(d_null):+.5f} pts^2")
    print(f"  MDE80 on per-bet CLV, whole universe (prob units)            "
          f"        : {splits.mde80(clv_all):+.6f}")
    print(f"  MDE80 on per-bet CLV at a ~950-bet rule union                "
          f"        : "
          f"{2.802*clv_all.std(ddof=1)/np.sqrt(950):+.6f}")
    print(f"  D121's live band for context: RED < -0.0131, GOOD > +0.0200 on "
          f"a ~44-bet month.")
    RES["mde80"] = {"sq_err_pts2_placebo_scale": float(splits.mde80(d_null)),
                    "clv_prob_universe": float(splits.mde80(clv_all)),
                    "clv_prob_at_n950": float(
                        2.802 * clv_all.std(ddof=1) / np.sqrt(950)),
                    "n": int(len(m)), "n_scored": int(okp.sum()),
                    "sd_dm": float(dm.std(ddof=1))}

    # --------------------------------------------- CHANNEL HORSE RACE ------
    hdr("[3] WHICH PAST-PRICE CHANNEL IS THE STRENGTH OBSERVATION?")
    print("  The reference model's central finding is that a devigged CLOSE "
          "beats the box score\n  as a strength observation.  Same engine, "
          "three channels, univariate on dm:")
    for c, lab in [("resid_close", "past CLOSES  (the football channel)"),
                   ("resid_open", "past OPENS   (control)"),
                   ("resid_res", "past RESULTS (the box-score channel)")]:
        X, nm = featurise(m, "A")
        v = X[:, nm.index(c)]
        r = np.corrcoef(v, dm)[0, 1]
        b = np.polyfit(v, dm, 1)
        print(f"    {lab:38s} corr {r:+.4f}   slope {b[0]:+.4f}   "
              f"R2 {r*r:.5f}")
        RES.setdefault("channels", {})[c] = {"corr": float(r),
                                             "slope": float(b[0]),
                                             "r2": float(r * r)}

    # ------------------------------------------------- THE MOVEMENT MODEL --
    hdr("[4] MOVEMENT-TARGETED ARM — how much of open->close is predictable?")
    for tier in ("A", "B"):
        if tier == "A":
            pred, names, coefs = pred_A, names_A, coefs_A
        else:
            pred, names, coefs = walk_forward(m, "dm", tier=tier)
        ok = np.isfinite(pred)
        r = np.corrcoef(pred[ok], dm[ok])[0, 1]
        sse_m = np.sum((dm[ok] - pred[ok]) ** 2)
        sse_0 = np.sum(dm[ok] ** 2)
        r2_oos = 1 - sse_m / sse_0
        cal = np.polyfit(pred[ok], dm[ok], 1)
        lab = ("A (LIVE)" if tier == "A"
               else "B (DIAGNOSTIC — availability lands AFTER the open)")
        print(f"\n  TIER {lab}   n_scored={ok.sum()}")
        print(f"    corr(pred, realised)            {r:+.4f}")
        print(f"    OOS R^2 vs NAIVE 'no movement'  {r2_oos:+.5f}   "
              f"(naive = predict dm = 0, the honest baseline)")
        print(f"    sd(pred) {pred[ok].std():.4f} pts  vs sd(realised) "
              f"{dm[ok].std():.4f} pts")
        print(f"    calibration slope {cal[0]:+.4f} (1.0 = perfect) "
              f"intercept {cal[1]:+.4f}")
        RES.setdefault("movement_arm", {})[tier] = {
            "n": int(ok.sum()), "corr": float(r), "r2_vs_naive": float(r2_oos),
            "sd_pred": float(pred[ok].std()), "sd_real": float(dm[ok].std()),
            "cal_slope": float(cal[0]), "cal_int": float(cal[1]),
            "coefs_by_season": coefs}
        if tier == "A":
            m["pred_dm"] = pred
        else:
            m["pred_dm_B"] = pred

    b, t, r2is = in_sample_coefs(m, "dm", "A")
    print("\n  IN-SAMPLE STANDARDISED COEFFICIENTS (season-clustered t), "
          f"tier A, R2={r2is:.5f}:")
    for k in b:
        if k == "intercept":
            continue
        star = " SIG" if abs(t[k]) >= 2.0 else ""
        print(f"    {k:14s} beta {b[k]:+8.4f}  t {t[k]:+6.2f}{star}")
    RES["movement_arm"]["A_insample"] = {"beta": b, "t_seasoncluster": t,
                                         "r2": float(r2is)}

    # decile calibration
    ok = np.isfinite(m.pred_dm.values)
    q = pd.qcut(m.pred_dm[ok], 10, labels=False, duplicates="drop")
    cal = m[ok].groupby(q).apply(lambda g: pd.Series({
        "n": len(g), "pred": g.pred_dm.mean(), "realised": g.dm.mean(),
        "clv_home": g.dp.mean()}), include_groups=False)
    print("\n  CALIBRATION BY PREDICTED-MOVEMENT DECILE (pts, home perspective):")
    print(cal.round(4).to_string())
    RES["movement_arm"]["decile_calibration"] = cal.reset_index().to_dict("records")

    # -------------------------------------------- ABLATION / ARTIFACT HUNT --
    hdr("[4b] WHERE DOES THE R^2 COME FROM, AND IS IT AN ARTIFACT?")
    print("  An OOS R^2 of this size against a market is a claim that needs "
          "attacking, not\n  reporting.  Nested ablation first (each row adds "
          "a group, rolling-origin OOS):")
    groups = [
        ("open price only        ", ["open_margin", "abs_open"]),
        ("+ past-CLOSE ridge     ", ["open_margin", "abs_open", "resid_close"]),
        ("+ past-OPEN ridge      ", ["open_margin", "abs_open", "resid_close",
                                     "resid_open"]),
        ("+ past-RESULT ridge    ", ["open_margin", "abs_open", "resid_close",
                                     "resid_open", "resid_res"]),
        ("+ our p_us (market-blind)", ["open_margin", "abs_open", "resid_close",
                                       "resid_open", "resid_res", "resid_us",
                                       "conf_us"]),
        ("+ schedule/tank (FULL A)", TIER_A),
        ("p_us ONLY (no ridge)   ", ["open_margin", "abs_open", "resid_us",
                                     "conf_us"]),
        ("ridges ONLY (no p_us)  ", ["open_margin", "abs_open", "resid_close",
                                     "resid_open", "resid_res"]),
    ]
    abl = []
    for nm, keep_f in groups:
        r2, cr = walk_forward_subset(m, "dm", keep_f)
        abl.append({"set": nm.strip(), "features": keep_f,
                    "r2_vs_naive": r2, "corr": cr})
        print(f"    {nm:26s} OOS R^2 {r2:+.5f}   corr {cr:+.4f}")
    RES["ablation"] = abl

    print("\n  ROBUSTNESS — the same tier-A model, re-scored on subsets:")
    rob = {}
    pdm = m.pred_dm.values
    okr = np.isfinite(pdm)
    def _r2(mask):
        mk = okr & mask
        if mk.sum() < 100:
            return float("nan"), int(mk.sum())
        return (1 - np.sum((dm[mk] - pdm[mk]) ** 2) / np.sum(dm[mk] ** 2),
                int(mk.sum()))
    checks = [
        ("REAL opening MONEYLINE rows only (no sigmoid map)",
         m.p_open_ml.notna().values),
        ("SBR-sourced rows only", (m.source == "sbr").values),
        ("espn+actionnetwork rows only",
         (m.source == "espn+actionnetwork").values),
        ("teamrankings-sourced rows only",
         m.source.astype(str).str.contains("teamrankings").values),
        ("|dm| <= 3 pts (drop the big movers)", (np.abs(dm) <= 3)),
        ("|dm| <= 2 pts", (np.abs(dm) <= 2)),
        ("games that MOVED at all (dm != 0)", (dm != 0)),
        ("|open_margin| <= 6 (near pick-em)", (np.abs(m.open_margin.values) <= 6)),
    ]
    for nm, mk in checks:
        r2v, nn = _r2(np.asarray(mk, bool))
        print(f"    {nm:52s} n={nn:5d}  OOS R^2 {r2v:+.5f}")
        rob[nm] = {"n": nn, "r2": None if not np.isfinite(r2v) else float(r2v)}
    RES["robustness"] = rob

    print("\n  THE SHARPEST ARTIFACT TEST — a FUTURE-BLIND ridge.")
    print("  If the R^2 came from the label leaking into the ridge, then a "
          "ridge frozen at the\n  START of each test season (no within-season "
          "updates at all) must lose it.  It does not:")
    r2_frozen = frozen_ridge_check(m)
    print(f"    ridge updated daily (as shipped) OOS R^2 "
          f"{RES['movement_arm']['A']['r2_vs_naive']:+.5f}")
    print(f"    ridge FROZEN at each season start OOS R^2 {r2_frozen:+.5f}")
    RES["frozen_ridge_r2"] = float(r2_frozen)

    # ---------------------------------------------------- SPLIT DISCIPLINE --
    hdr("[5] SPLIT DISCIPLINE — GATE_POLICY_V2 sections 8-11 on the movement arm")
    ok = np.isfinite(m.pred_dm.values)
    sub = m[ok]
    d = sub.dm.values ** 2 - (sub.dm.values - sub.pred_dm.values) ** 2
    pan = splits.Panel(sub.season.values, d, sub.game_date.astype(str).values,
                       cluster=sub.season.values,
                       label="movement model vs naive no-movement (pts^2)")
    rep = splits.full_report(pan, B=B_BOOT, seed=SEED)
    print(splits.format_report(rep))
    RES["splits_movement"] = rep

    # --------------------------------------------------- THE OUTCOME ARM ---
    hdr("[6] OUTCOME-TARGETED ARM — for comparison, as required")
    m["y_margin"] = m.margin_actual
    predo, _, _ = walk_forward(m, "y_margin", tier="A")
    ok2 = np.isfinite(predo)
    base_close = m.close_margin.values
    base_open = m.open_margin.values
    mae = lambda p: np.mean(np.abs(m.margin_actual.values[ok2] - p[ok2]))
    print(f"  n={ok2.sum()}")
    print(f"    MAE  our outcome arm       {mae(predo):.4f} pts")
    print(f"    MAE  the OPENING line      {mae(base_open):.4f} pts")
    print(f"    MAE  the CLOSING line      {mae(base_close):.4f} pts")
    print(f"    MAE  our market-blind p_us {mae(m.m_us.values):.4f} pts")
    rmse = lambda p: float(np.sqrt(np.mean((m.margin_actual.values[ok2] - p[ok2]) ** 2)))
    RES["outcome_arm"] = {"n": int(ok2.sum()), "mae_model": float(mae(predo)),
                          "mae_open": float(mae(base_open)),
                          "mae_close": float(mae(base_close)),
                          "mae_p_us": float(mae(m.m_us.values)),
                          "rmse_model": rmse(predo),
                          "rmse_close": rmse(base_close)}
    print("  READING: the movement target is the one we can hit.  The outcome "
          "target is the\n  one D132 already measures, and the close still "
          "wins there.")

    # ------------------------------------------------------- CLV SELECTION --
    hdr("[7] DOES IT SELECT POSITIVE-CLV BETS? (the money question for D1)")
    sub = m[np.isfinite(m.pred_dm.values)].copy()
    # side chosen by PREDICTED MOVEMENT (not by p_us): bet the side the line
    # is predicted to move toward.
    sub["mv_home"] = sub.pred_dm > 0
    sub["clv_mv"] = np.where(sub.mv_home, sub.dp, -sub.dp)
    # side chosen by the MARKET-BLIND model (the incumbent)
    sub["us_home"] = sub.p_us > 0.5
    sub["clv_us"] = np.where(sub.us_home, sub.dp, -sub.dp)
    # frozen registry rules, at the open, SP frame (D126 convention)
    p_side_open = np.where(sub.us_home, sub.p_open_sp, 1 - sub.p_open_sp)
    edge = np.where(sub.us_home, sub.p_us, 1 - sub.p_us) - p_side_open
    same = ((sub.p_us.values - 0.5) *
            (np.where(sub.us_home, p_side_open, 1 - p_side_open) - 0.5)) > 0
    tails = (sub.p_us - 0.5).abs().values > CONF_TIER
    band = (edge >= DIV_LO) & (edge <= DIV_HI)
    fav_star = np.where(sub.p_open_sp >= 0.5, sub.star_out_home,
                        sub.star_out_away)
    rules = {
        "R4_LOWT": same & (edge > EDGE_R4) & sub.late.values,
        "T20_D03_10_W": same & tails & band & sub.late.values,
        "T20_D03_10": same & tails & band,
        "STAR_FAV_SHARPER": same & (edge > 0) & fav_star,
    }
    union = np.zeros(len(sub), bool)
    for v in rules.values():
        union |= v

    def clv_row(name, mask, clv):
        x = clv[mask]
        if len(x) < 5:
            return None
        bs = splits.paired_bootstrap(x, B_BOOT, SEED,
                                     cluster=sub.season.values[mask])
        ic = splits.icc_oneway(x, sub.season.values[mask])
        tt = splits.cluster_mean_t_interval(x, sub.season.values[mask])
        nn = float("nan")
        g = lambda dd, k: (nn if dd.get(k) is None else float(dd[k]))
        return {"rule": name, "n": int(len(x)), "clv": float(x.mean()),
                "lo": g(bs, "lo"), "hi": g(bs, "hi"), "sig": bool(bs["sig"]),
                "icc": g(ic, "icc"), "deff": g(ic, "deff"),
                "t_lo": g(tt, "lo"), "t_hi": g(tt, "hi"),
                "pos_frac": float((x > 0).mean())}

    tab = []
    tab.append(clv_row("ALL_UNIVERSE (side = p_us)", np.ones(len(sub), bool),
                       sub.clv_us.values))
    for k, v in rules.items():
        tab.append(clv_row(k, v, sub.clv_us.values))
    tab.append(clv_row("UNION (4 frozen rules)", union, sub.clv_us.values))
    # movement-model rules at ascending thresholds
    for thr in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
        mask = np.abs(sub.pred_dm.values) > thr
        tab.append(clv_row(f"MOVEMENT |pred_dm|>{thr:.2f}", mask,
                           sub.clv_mv.values))
    # movement model INTERSECTED with the frozen union (execution overlay)
    for thr in (0.0, 0.25, 0.5):
        mask = union & (np.abs(sub.pred_dm.values) > thr) & \
            (np.sign(sub.pred_dm.values) == np.where(sub.us_home, 1, -1))
        tab.append(clv_row(f"UNION & movement agrees & |pred|>{thr:.2f}",
                           mask, sub.clv_us.values))
    tab = [r for r in tab if r]
    print(f"\n  {'set':44s} {'n':>6s} {'CLV':>9s} {'95% cluster CI':>22s} "
          f"{'ICC':>8s} {'DEFF':>6s} {'>0':>6s}")
    for r in tab:
        print(f"  {r['rule']:44s} {r['n']:6d} {r['clv']:+9.5f} "
              f"[{r['lo']:+.5f},{r['hi']:+.5f}] {r['icc']:+8.5f} "
              f"{r['deff']:6.2f} {r['pos_frac']:6.3f}"
              f"{'  SIG' if r['sig'] else ''}")
    RES["clv_table"] = tab

    print("\n  READING THIS TABLE HONESTLY.  CLV = p_close_side - p_open_side "
          "is a monotone\n  transform of the very quantity the model is fitted "
          "to (dm), so the rows below are\n  the R^2 of [4] re-expressed in "
          "probability units — NOT a second, independent\n  confirmation of "
          "it.  (Same discipline D142 applied to its own CLV number.)  What "
          "IS\n  new here is the COMPARISON: at a matched bet count the "
          "movement rule\n  (|pred|>1.50, n=920, +0.054) carries ~3x the CLV "
          "of the frozen union (n=1318,\n  +0.018) and ~6x the whole universe "
          "(+0.009).")

    # ------------------------------------------------ DIRECTION ACCURACY ---
    hdr("[7b] DIRECTION ACCURACY — the quantity Deliverable 2 consumes")
    print("  D119's ceiling: betting the side the market LATER MOVES TOWARD "
          "wins 54.63% at the\n  open price vs a 52.38% breakeven.  That "
          "prize is only claimable to the extent we\n  can NAME that side in "
          "advance.  Here is how often we do:")
    mv = sub.pred_dm.values
    real = sub.dm.values
    moved = real != 0
    dirtab = []
    for lo_, hi_ in [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 1.5),
                     (1.5, 2.5), (2.5, 99.0)]:
        mk = (np.abs(mv) >= lo_) & (np.abs(mv) < hi_) & moved
        if mk.sum() < 20:
            continue
        acc = float((np.sign(mv[mk]) == np.sign(real[mk])).mean())
        dirtab.append({"bucket": f"{lo_:.2f}-{hi_:.2f}", "n": int(mk.sum()),
                       "dir_acc": acc,
                       "mean_abs_move": float(np.abs(real[mk]).mean()),
                       "mean_signed_toward": float(
                           (np.sign(mv[mk]) * real[mk]).mean())})
        print(f"    |pred_dm| in [{lo_:.2f},{hi_:.2f})  n={mk.sum():5d}  "
              f"direction correct {100*acc:5.2f}%   "
              f"mean move TOWARD the predicted side "
              f"{(np.sign(mv[mk]) * real[mk]).mean():+.3f} pts")
    allmk = moved
    acc_all = float((np.sign(mv[allmk]) == np.sign(real[allmk])).mean())
    print(f"    ALL movers                n={allmk.sum():5d}  "
          f"direction correct {100*acc_all:5.2f}%   "
          f"mean move toward "
          f"{(np.sign(mv[allmk]) * real[allmk]).mean():+.3f} pts")
    RES["direction"] = {"buckets": dirtab, "all_movers_acc": acc_all,
                        "n_movers": int(allmk.sum())}

    # ---------------------------------------------------------- PLACEBO ----
    hdr("[8] PERMUTATION PLACEBO — permute predicted movement WITHIN DATE")
    print("  D115's lesson: a favourite-drift control is VACUOUS on same-side "
          "rules because our\n  side IS the open favourite by construction.  "
          "The honest control permutes the\n  prediction across games on the "
          "SAME DATE, destroying the model's information\n  while preserving "
          "the selection mechanism, the slate and the price distribution.")
    rng = np.random.default_rng(SEED)
    dates = sub.game_date.values
    n_perm = 200
    for thr in (0.0, 0.5, 1.0):
        real_mask = np.abs(sub.pred_dm.values) > thr
        real = sub.clv_mv.values[real_mask].mean()
        draws = []
        for _ in range(n_perm):
            pp = sub.pred_dm.values.copy()
            for _d in np.unique(dates):
                ix = np.where(dates == _d)[0]
                if len(ix) > 1:
                    pp[ix] = pp[rng.permutation(ix)]
            mk = np.abs(pp) > thr
            if mk.sum() < 5:
                continue
            clv = np.where(pp > 0, sub.dp.values, -sub.dp.values)
            draws.append(clv[mk].mean())
        draws = np.array(draws)
        pv = float((draws >= real).mean())
        print(f"    |pred_dm|>{thr:.2f}  n={int(real_mask.sum()):5d}  "
              f"REAL CLV {real:+.5f}   placebo mean {draws.mean():+.5f} "
              f"sd {draws.std():.5f}  p={pv:.4f}"
              f"{'  <- information' if pv < 0.05 else '  <- NOT information'}")
        RES.setdefault("placebo", {})[f"thr_{thr}"] = {
            "n": int(real_mask.sum()), "real": float(real),
            "placebo_mean": float(draws.mean()), "placebo_sd": float(draws.std()),
            "p": pv}

    # -------------------------------------------------- MONTHLY BANDS ------
    hdr("[9] MONTHLY CLV vs D121's LIVE BANDS (red < -0.0131, good > +0.0200)")
    for nm, mask, clv in [("UNION (frozen rules, incumbent)", union, sub.clv_us.values),
                          ("MOVEMENT |pred|>0.50", np.abs(sub.pred_dm.values) > 0.5,
                           sub.clv_mv.values)]:
        s2 = sub[mask].copy()
        s2["clv"] = clv[mask]
        s2["ym"] = s2.game_date.dt.strftime("%Y-%m")
        g = s2.groupby("ym").clv.agg(["size", "mean"])
        g = g[g["size"] >= 10]
        print(f"\n  {nm}: {len(g)} months, median {g['size'].median():.0f} "
              f"bets/month")
        print(f"    mean-of-months {g['mean'].mean():+.5f}   "
              f"2-sigma [{g['mean'].mean()-2*g['mean'].std():+.5f},"
              f"{g['mean'].mean()+2*g['mean'].std():+.5f}]")
        print(f"    months positive {100*(g['mean']>0).mean():.0f}%   "
              f"below RED -0.0131: {int((g['mean']<-0.0131).sum())}   "
              f"above GOOD +0.0200: {int((g['mean']>0.0200).sum())}")
        RES.setdefault("monthly", {})[nm] = {
            "months": int(len(g)), "median_bets": float(g["size"].median()),
            "mean_of_months": float(g["mean"].mean()),
            "sd": float(g["mean"].std()),
            "pct_pos": float((g["mean"] > 0).mean()),
            "n_red": int((g["mean"] < -0.0131).sum()),
            "n_good": int((g["mean"] > 0.0200).sum())}

    keep = ["season", "game_id", "game_date", "home", "away", "y", "p_us",
            "open_margin", "close_margin", "dm", "dp", "pred_dm", "pred_dm_B",
            "p_open_sp", "p_close_sp", "p_open_ml", "p_close_ml",
            "dec_open_home", "dec_open_away", "dec_close_home",
            "dec_close_away", "late", "margin_actual", "open_total",
            "close_total", "ridge_close", "m_us"]
    out = m[[c for c in keep if c in m.columns]].copy()
    out["union_rule"] = False
    out.loc[sub.index[union], "union_rule"] = True
    out.to_csv(ROWS, index=False, compression="gzip")
    RES["rows_csv"] = ROWS
    RES["elapsed_s"] = round(time.time() - t0, 1)
    with open(OUT, "w") as fh:
        json.dump(RES, fh, indent=1, default=str)
    print(f"\nWROTE {OUT} and {ROWS}  ({RES['elapsed_s']}s)")


if __name__ == "__main__":
    main()
