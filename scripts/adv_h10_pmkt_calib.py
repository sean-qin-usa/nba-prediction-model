"""H10 audit (docs/ADVERSE_HYPOTHESES.md): is p_mkt = sigmoid(spread/6.96)
miscalibrated at extremes, so that part of the "market-confident-we-not" hole
(n=208, +0.030/gm) and heavy-fav gap is OUR measurement artifact?

A) Reliability of p_mkt by favorite-prob bucket on the 3 eval seasons.
B) Spread-conversion vs de-vigged moneyline on 2008-2023 (ml available) by
   spread size — direct bias test of the fixed logistic scale at big spreads.
C) Recompute region gaps with (i) a PIT recalibrated conversion fit on
   pre-eval seasons and (ii) a hindsight logit recalibration fit on the eval
   seasons themselves (upper bound of artifact share).

Read-only DuckDB. Output: printed report only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import duckdb

DB = "/hdd/steveqin/sean_dev/nba_model/data/nba.duckdb"
CSV = "/hdd/steveqin/sean_dev/nba_model/data/capstone_pergame_tank.csv"
RNG = np.random.default_rng(7)


def ll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def boot_mean_ci(x, B=4000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(axis=1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def devig_american(mh, ma):
    def imp(m):
        m = float(m)
        return 100.0 / (m + 100.0) if m > 0 else -m / (-m + 100.0)
    ih, ia = imp(mh), imp(ma)
    return ih / (ih + ia)


def main():
    con = duckdb.connect(DB, read_only=True)
    df = pd.read_csv(CSV, dtype={"game_id": str})
    df["game_id"] = df.game_id.str.zfill(10)
    om = con.execute("""SELECT season_end, game_date, home, away, score_home,
        score_away, home_win, home_exp_margin, ml_home, ml_away, p_home_spread
        FROM odds_market""").df()
    om["game_date"] = pd.to_datetime(om.game_date).dt.date.astype(str)
    df = df.merge(om[["game_date", "home", "away", "home_exp_margin"]],
                  on=["game_date", "home", "away"], how="left")
    n_miss = df.home_exp_margin.isna().sum()
    print(f"capstone rows {len(df)}, unmatched to odds_market: {n_miss}")
    # sanity: p_mkt should equal sigmoid(margin/6.96)
    chk = np.abs(sigmoid(df.home_exp_margin / 6.96) - df.p_mkt).max()
    print(f"max |sigmoid(m/6.96) - p_mkt| = {chk:.2e}\n")

    # ---------- A) reliability of p_mkt on eval seasons ----------
    print("=" * 72)
    print("A) RELIABILITY of p_mkt (favorite-folded), eval seasons 2023-26")
    pf = np.maximum(df.p_mkt, 1 - df.p_mkt)
    yf = np.where(df.p_mkt >= 0.5, df.y, 1 - df.y)
    edges = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pf >= lo) & (pf < hi)
        n, k = int(m.sum()), int(yf[m].sum())
        w = wilson(k, n)
        rows.append([f"[{lo:.2f},{hi:.2f})", n, round(pf[m].mean(), 4),
                     round(k / n, 4) if n else np.nan,
                     round(k / n - pf[m].mean(), 4) if n else np.nan,
                     f"({w[0]:.3f},{w[1]:.3f})"])
    print(pd.DataFrame(rows, columns=[
        "p_fav bucket", "n", "mean p_mkt", "emp win", "gap", "wilson95"]
    ).to_string(index=False))
    m = pf > 0.85
    n, k = int(m.sum()), int(yf[m].sum())
    w = wilson(k, n)
    print(f"\nEXTREME |p_mkt-0.5|>0.35: n={n} mean_p={pf[m].mean():.4f} "
          f"emp={k / n:.4f} wilson=({w[0]:.3f},{w[1]:.3f})")
    # hindsight logit recalibration on eval seasons (a,b)
    from numpy.polynomial import polynomial  # noqa: F401  (no sklearn dep)
    X = logit(df.p_mkt.values)
    yv = df.y.values.astype(float)
    a, b = fit_logreg(X, yv)
    print(f"logit recal on eval:  y ~ sigmoid({a:+.4f} + {b:.4f}*logit(p_mkt))"
          f"   [a=0,b=1 = perfectly calibrated]")

    # ---------- B) conversion vs ML devig, 2008-2023 ----------
    print("\n" + "=" * 72)
    print("B) sigmoid(spread/6.96) vs ML-devig, seasons with moneylines")
    h = om[(om.ml_home.notna()) & (om.ml_away.notna())].copy()
    h["p_ml"] = [devig_american(r.ml_home, r.ml_away) for r in h.itertuples()]
    h["p_sp"] = sigmoid(h.home_exp_margin / 6.96)
    h["fav_margin"] = np.abs(h.home_exp_margin)
    # fold to favorite by SPREAD side
    h["pf_sp"] = np.maximum(h.p_sp, 1 - h.p_sp)
    h["pf_ml"] = np.where(h.p_sp >= 0.5, h.p_ml, 1 - h.p_ml)
    h["yf"] = np.where(h.p_sp >= 0.5, h.home_win, ~h.home_win).astype(float)
    bins = [0, 3, 6, 9, 12, 15, 30]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (h.fav_margin >= lo) & (h.fav_margin < hi)
        n = int(m.sum())
        rows.append([f"[{lo},{hi})", n,
                     round(h.pf_sp[m].mean(), 4), round(h.pf_ml[m].mean(), 4),
                     round((h.pf_sp[m] - h.pf_ml[m]).mean(), 4),
                     round(h.yf[m].mean(), 4),
                     f"({wilson(int(h.yf[m].sum()), n)[0]:.3f},"
                     f"{wilson(int(h.yf[m].sum()), n)[1]:.3f})"])
    print(pd.DataFrame(rows, columns=[
        "|spread|", "n", "mean p_spread", "mean p_ml_devig", "sp-ml bias",
        "emp win", "emp wilson95"]).to_string(index=False))
    print("LL(spread conv) vs LL(ml devig) on ml sample: "
          f"{ll(h.p_sp, h.home_win.astype(float)).mean():.5f} vs "
          f"{ll(h.p_ml, h.home_win.astype(float)).mean():.5f}")

    # ---------- C) region gaps under recalibrated conversions ----------
    print("\n" + "=" * 72)
    print("C) REGION GAPS under alternative p_mkt")
    # (i) PIT refit on pre-eval seasons (<=2023): scale + probit + a,b logistic
    pre = om[om.season_end <= 2023]
    yh = pre.home_win.astype(float).values
    mg = pre.home_exp_margin.values
    # refit single logistic scale
    from scipy.optimize import minimize_scalar
    s_fit = minimize_scalar(lambda s: ll(sigmoid(mg / s), yh).mean(),
                            bounds=(4, 12), method="bounded").x
    a2, b2 = fit_logreg(mg, yh, init=(0.0, 1 / 7.0))  # margin logistic w/ icpt
    # probit fit
    from scipy.stats import norm
    s_pr = minimize_scalar(lambda s: ll(norm.cdf(mg / s), yh).mean(),
                           bounds=(6, 20), method="bounded").x
    bt, ct = fit_tailboost(mg, yh)  # tail-shape corrected logistic
    print(f"PIT refits on 2008-23 (n={len(pre)}): logistic scale={s_fit:.3f} "
          f"(shipped 6.96); a+b*m: a={a2:+.4f} b={b2:.4f}; probit sd={s_pr:.3f}; "
          f"tailboost b={bt:.4f} (1/{1 / bt:.2f}) c={ct:+.4f}")
    # fit ML-devig-matching conversion too: regress logit(p_ml) on [m, tail]
    tml = np.sign(h.home_exp_margin.values) * np.maximum(
        np.abs(h.home_exp_margin.values) - 9.0, 0.0)
    A = np.column_stack([h.home_exp_margin.values, tml])
    coef, *_ = np.linalg.lstsq(A, logit(h.p_ml.values), rcond=None)
    print(f"ML-devig-matching: logit(p_ml) ~ {coef[0]:.4f}*m "
          f"{coef[1]:+.4f}*(|m|-9)+  (i.e. scale 1/{1 / coef[0]:.2f})")
    mge = df.home_exp_margin.values
    tge = np.sign(mge) * np.maximum(np.abs(mge) - 9.0, 0.0)
    variants = {
        "shipped 6.96": df.p_mkt.values,
        f"refit scale {s_fit:.2f}": sigmoid(mge / s_fit),
        "refit a+b*m": sigmoid(a2 + b2 * mge),
        f"probit sd {s_pr:.1f}": norm.cdf(mge / s_pr),
        "tailboost (emp fit)": sigmoid(bt * mge + ct * tge),
        "ML-devig-matching": sigmoid(coef[0] * mge + coef[1] * tge),
        "hindsight logit recal": sigmoid(a + b * logit(df.p_mkt.values)),
    }
    y = df.y.values.astype(float)
    Lus = ll(df.p_us.values, y)
    reg_mkconf = ((np.abs(df.p_mkt - 0.5) > 0.35)
                  & (np.abs(df.p_us - 0.5) <= 0.35)).values
    reg_heavy = (np.abs(df.p_mkt - 0.5) > 0.35).values
    reg_opp = ((df.p_us - 0.5) * (df.p_mkt - 0.5) < 0).values
    print(f"\nregions (fixed at shipped defs): mkconf n={reg_mkconf.sum()}, "
          f"heavy n={reg_heavy.sum()}, opp n={reg_opp.sum()}")
    rows = []
    for name, pm in variants.items():
        Lm = ll(pm, y)
        d = Lus - Lm
        rows.append([name, round(Lm.mean(), 5),
                     round(d[reg_mkconf].mean(), 4),
                     round(d[reg_heavy].mean(), 4),
                     round(d[reg_opp].mean(), 4),
                     round(d.mean(), 5)])
    out = pd.DataFrame(rows, columns=[
        "p_mkt variant", "mkt LL(all)", "gap mkconf/gm", "gap heavy/gm",
        "gap opp/gm", "gap all/gm"])
    print(out.to_string(index=False))
    # bootstrap CI of artifact share: gap_shipped - gap_recal on mkconf
    d_ship = Lus - ll(variants["shipped 6.96"], y)
    d_hind = Lus - ll(variants["hindsight logit recal"], y)
    diff = (d_ship - d_hind)[reg_mkconf]
    lo, hi = boot_mean_ci(diff)
    print(f"\nARTIFACT SHARE (hindsight bound), mkconf region: "
          f"gap_shipped - gap_recal = {diff.mean():+.4f}/gm "
          f"CI95 ({lo:+.4f},{hi:+.4f}); "
          f"share of +{d_ship[reg_mkconf].mean():.4f} hole = "
          f"{diff.mean() / d_ship[reg_mkconf].mean() * 100:.1f}%")
    d_pit = Lus - ll(variants[f"refit scale {s_fit:.2f}"], y)
    diffp = (d_ship - d_pit)[reg_mkconf]
    lo2, hi2 = boot_mean_ci(diffp)
    print(f"ARTIFACT SHARE (PIT refit scale), mkconf region: {diffp.mean():+.4f}"
          f"/gm CI95 ({lo2:+.4f},{hi2:+.4f})")
    con.close()


def fit_logreg(x, y, init=(0.0, 1.0)):
    """2-param logistic regression y ~ sigmoid(a + b*x) via scipy BFGS."""
    from scipy.optimize import minimize
    X = np.column_stack([np.ones_like(x), x])

    def nll(w):
        return ll(sigmoid(X @ w), y).mean()
    r = minimize(nll, np.array(init), method="Nelder-Mead",
                 options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 5000})
    return float(r.x[0]), float(r.x[1])


def fit_tailboost(m, y, knot=9.0):
    """logit p = b*m + c*sign(m)*(|m|-knot)+  (tail-shape correction)."""
    from scipy.optimize import minimize
    t = np.sign(m) * np.maximum(np.abs(m) - knot, 0.0)

    def nll(w):
        return ll(sigmoid(w[0] * m + w[1] * t), y).mean()
    r = minimize(nll, np.array([1 / 7.0, 0.0]), method="Nelder-Mead",
                 options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 5000})
    return float(r.x[0]), float(r.x[1])


if __name__ == "__main__":
    main()
