"""The football project's OWN inefficiency tests, run on the NBA market, plus
the two tests that only exist here (total-as-second-dimension, spread-vs-ML).

Football reference (football_exercise/DECISIONS.md §F, COMPARISON.md §3):
  smoothing the market across a season   -0.0031 CI[-0.0069,+0.0008]  ns
  sharpening the market's own delta      -0.0006 CI[-0.0013,+0.0001]  ns
  tempo beyond the market's P(draw)      non-monotone                 no effect
  bookmaker margin                        2.82%  (sharp-book class)
  the market is 2-D: 70% of draw-parameter variation is independent of strength

Read-only on data/nba.duckdb.  Run: python scripts/lb_football_analogues.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import nbapred.threads                                           # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402

from lb_longshot import (DEVIGS, am2dec, cluster_boot, cluster_mean_t,  # noqa
                         icc_oneway, load_panel, logit, sides, sigmoid)

OUT = os.path.join(ROOT, "data", "lb_football_analogues.json")
SEED = 20260802
N_BOOT = 4000


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def logistic_fit(X, y, ridge=0.0):
    b = np.zeros(X.shape[1])
    for _ in range(80):
        mu = sigmoid(X @ b)
        w = np.clip(mu * (1 - mu), 1e-9, None)
        H = (X.T * w) @ X + ridge * np.eye(X.shape[1])
        gsc = X.T @ (y - mu) - ridge * b
        try:
            step = np.linalg.solve(H, gsc)
        except np.linalg.LinAlgError:
            break
        b = b + step
        if np.max(np.abs(step)) < 1e-11:
            break
    mu = sigmoid(X @ b)
    w = np.clip(mu * (1 - mu), 1e-9, None)
    H = (X.T * w) @ X + ridge * np.eye(X.shape[1])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return b, se


# ------------------------------------------------------- (iii) SHARPEN ------
def sharpen(gm, pcol, tag, res):
    """p_s = sigmoid(s * logit(p_mkt)).  Football: -0.0006 CI[-0.0013,+0.0001].
    s > 1 helping == the market is UNDER-confident == favourite-longshot bias
    in log-loss units.  Same statistic as lb_longshot's logit_b, different
    loss."""
    z = logit(gm[pcol].values)
    y = gm.y.values
    cl = gm.season.values
    base = ll(y, sigmoid(z))
    grid = np.round(np.arange(0.80, 1.351, 0.01), 3)
    curve = [(float(s), float((ll(y, sigmoid(s * z)) - base).mean()))
             for s in grid]
    best_s = float(min(curve, key=lambda t: t[1])[0])
    d = ll(y, sigmoid(best_s * z)) - base
    lo, hi, se = cluster_boot(d, cl, n_boot=N_BOOT)
    tlo, thi, K = cluster_mean_t(d, cl)
    # honest version: pick s out-of-sample by leave-one-season-out
    seasons = sorted(set(cl))
    d_loso = np.empty(len(y))
    picks = {}
    for s0 in seasons:
        tr = cl != s0
        c2 = [(float(ss), float((ll(y[tr], sigmoid(ss * z[tr]))
                                 - base[tr]).mean())) for ss in grid]
        sb = float(min(c2, key=lambda t: t[1])[0])
        picks[s0] = sb
        te = ~tr
        d_loso[te] = ll(y[te], sigmoid(sb * z[te])) - base[te]
    llo, lhi, lse = cluster_boot(d_loso, cl, n_boot=N_BOOT)
    ltlo, lthi, _ = cluster_mean_t(d_loso, cl)
    res[f"sharpen|{tag}"] = dict(
        n=int(len(y)), base_ll=float(base.mean()), best_s=best_s,
        d_insample=float(d.mean()), lo=lo, hi=hi, se=se, tlo=tlo, thi=thi, K=K,
        d_loso=float(d_loso.mean()), loso_lo=llo, loso_hi=lhi,
        loso_tlo=ltlo, loso_thi=lthi, loso_picks=picks,
        curve=curve,
        n_seasons_improved=int(sum(
            1 for s0 in seasons if d[cl == s0].mean() < 0)),
    )
    print(f"[SHARPEN {tag}] base LL {base.mean():.5f}  best s={best_s:.2f}  "
          f"d={d.mean():+.5f} CI[{lo:+.5f},{hi:+.5f}]  "
          f"t-CI[{tlo:+.5f},{thi:+.5f}]  K={K}  "
          f"LOSO d={d_loso.mean():+.5f} CI[{llo:+.5f},{lhi:+.5f}]")
    return res[f"sharpen|{tag}"]


# -------------------------------------------------------- (ii) SMOOTH -------
def smooth_season(gm, pcol, tag, res, lams=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0)):
    """The football project's retracted-then-null test, NBA form.  Within each
    SEASON fit z_g = s_home - s_away + h to the market's OWN logit prices by
    ridge, leave-one-game-out, and ask whether the smoothed price beats the raw
    one.  This is an ORACLE (it uses the whole season, including the future),
    so a null here is a strong statement."""
    y = gm.y.values
    z = logit(gm[pcol].values)
    cl = gm.season.values
    base = ll(y, sigmoid(z))
    out = {}
    for lam in lams:
        zhat = np.empty(len(z))
        for s0 in sorted(set(cl)):
            m = cl == s0
            zz = z[m]
            teams = sorted(set(gm.home.values[m]) | set(gm.away.values[m]))
            ti = {t: i for i, t in enumerate(teams)}
            T = len(teams)
            n = int(m.sum())
            X = np.zeros((n, T + 1))
            hs = gm.home.values[m]
            aw = gm.away.values[m]
            X[np.arange(n), [ti[t] for t in hs]] = 1.0
            X[np.arange(n), [ti[t] for t in aw]] = -1.0
            X[:, T] = 1.0
            A = X.T @ X + lam * np.eye(T + 1)
            Ainv = np.linalg.inv(A)
            b = Ainv @ (X.T @ zz)
            fit = X @ b
            hlev = np.einsum("ij,jk,ik->i", X, Ainv, X)
            # exact leave-one-out for ridge: r_loo = r / (1 - h)
            r = zz - fit
            zhat[m] = zz - r / np.clip(1.0 - hlev, 1e-6, None)
        d = ll(y, sigmoid(zhat)) - base
        lo, hi, se = cluster_boot(d, cl, n_boot=N_BOOT)
        tlo, thi, K = cluster_mean_t(d, cl)
        nwin = sum(1 for s0 in sorted(set(cl)) if d[cl == s0].mean() < 0)
        out[str(lam)] = dict(d=float(d.mean()), lo=lo, hi=hi, se=se,
                             tlo=tlo, thi=thi, K=K, seasons_improved=nwin)
        print(f"[SMOOTH {tag}] lam={lam:5.1f}  d={d.mean():+.5f} "
              f"CI[{lo:+.5f},{hi:+.5f}]  t-CI[{tlo:+.5f},{thi:+.5f}]  "
              f"{nwin}/{K} seasons improved")
    # BLEND form: z_w = (1-w)*z + w*zhat at the best lam.  The pure-replacement
    # test above is the football project's; this is the more forgiving version,
    # and if the optimum is w=0 the null is total.
    best_lam = min(out, key=lambda k: out[k]["d"])
    zhat = np.empty(len(z))
    lam = float(best_lam)
    for s0 in sorted(set(cl)):
        m = cl == s0
        zz = z[m]
        teams = sorted(set(gm.home.values[m]) | set(gm.away.values[m]))
        ti = {t: i for i, t in enumerate(teams)}
        T = len(teams)
        n = int(m.sum())
        X = np.zeros((n, T + 1))
        X[np.arange(n), [ti[t] for t in gm.home.values[m]]] = 1.0
        X[np.arange(n), [ti[t] for t in gm.away.values[m]]] = -1.0
        X[:, T] = 1.0
        A = X.T @ X + lam * np.eye(T + 1)
        Ainv = np.linalg.inv(A)
        fit = X @ (Ainv @ (X.T @ zz))
        hlev = np.einsum("ij,jk,ik->i", X, Ainv, X)
        zhat[m] = zz - (zz - fit) / np.clip(1.0 - hlev, 1e-6, None)
    blend = []
    for w in np.round(np.arange(0.0, 1.01, 0.05), 2):
        d = ll(y, sigmoid((1 - w) * z + w * zhat)) - base
        blend.append((float(w), float(d.mean())))
    wbest = float(min(blend, key=lambda t: t[1])[0])
    dw = ll(y, sigmoid((1 - wbest) * z + wbest * zhat)) - base
    blo, bhi, _ = cluster_boot(dw, cl, n_boot=N_BOOT)
    btlo, bthi, K = cluster_mean_t(dw, cl)
    print(f"[SMOOTH-BLEND {tag}] lam={lam}  best w={wbest:.2f}  "
          f"d={dw.mean():+.5f} CI[{blo:+.5f},{bhi:+.5f}] t-CI[{btlo:+.5f},"
          f"{bthi:+.5f}]  {sum(1 for s0 in sorted(set(cl)) if dw[cl==s0].mean()<0)}/{K}")
    res[f"smooth|{tag}"] = dict(base_ll=float(base.mean()), n=int(len(y)),
                                by_lam=out, blend_curve=blend,
                                blend_lam=lam, blend_w=wbest,
                                blend_d=float(dw.mean()), blend_lo=blo,
                                blend_hi=bhi, blend_tlo=btlo, blend_thi=bthi)
    return out


def era_split(gm, res):
    """Is any of this era-dependent?  Vendor/era universe: SBR closes
    2007-08..2022-23 vs ESPN/ActionNetwork closes 2023-24..2025-26."""
    out = {}
    for tag, m in (("SBR_2008_2023", gm.source.values == "sbr"),
                   ("MODERN_2024_2026", gm.source.values != "sbr")):
        sub = gm[m]
        if len(sub) < 500:
            continue
        z = logit(sub.p_prop.values)
        y = sub.y.values
        cl = sub.season.values
        base = ll(y, sigmoid(z))
        grid = np.round(np.arange(0.80, 1.351, 0.01), 3)
        curve = [(float(s), float((ll(y, sigmoid(s * z)) - base).mean()))
                 for s in grid]
        bs = float(min(curve, key=lambda t: t[1])[0])
        d = ll(y, sigmoid(bs * z)) - base
        lo, hi, _ = cluster_boot(d, cl, n_boot=N_BOOT)
        e = y - sub.p_prop.values
        x = sub.p_prop.values - 0.5
        vx = x - x.mean()
        lin = float((vx * (e - e.mean())).sum() / (vx * vx).sum())
        out[tag] = dict(n=int(len(sub)), seasons=int(sub.season.nunique()),
                        best_s=bs, d=float(d.mean()), lo=lo, hi=hi,
                        lin_b=lin, base_ll=float(base.mean()),
                        overround=float(sub.ov.mean()))
        print(f"[ERA {tag}] n={len(sub)} K={sub.season.nunique()} "
              f"overround={sub.ov.mean():.4f} best_s={bs:.2f} "
              f"d={d.mean():+.5f} CI[{lo:+.5f},{hi:+.5f}]  lin_b={lin:+.4f}")
    res["era_split"] = out
    return out


# ------------------------------------- (i) IS THE NBA MARKET 2-D? -----------
def two_d(gm, pcol, tag, res):
    """Football's second dimension was the DRAW parameter — 70% of its variation
    independent of strength, and genuinely predictive.  A 2-outcome moneyline is
    EXACTLY 1-D: one number determines the whole book.  The two nearest NBA
    candidates:
      (A) the TOTAL as a second-moment dimension (higher total -> more margin
          variance -> win prob compressed toward 0.5);
      (B) spread-vs-moneyline consistency (two prices for the same event)."""
    y = gm.y.values
    cl = gm.season.values
    z = logit(gm[pcol].values)
    mg = gm.margin_home.values
    tot = gm.total.values
    ok = np.isfinite(tot) & np.isfinite(mg) & np.isfinite(z)
    y, cl, z, mg, tot = y[ok], cl[ok], z[ok], mg[ok], tot[ok]
    # de-season the total: it drifts hugely across the 3-point era
    dft = pd.DataFrame({"s": cl, "t": tot})
    tz = tot - dft.groupby("s").t.transform("mean").values
    tz = tz / dft.groupby("s").t.transform("std").values

    o = {}
    # how much of the TOTAL is independent of the strength axis (football: 70%)
    r = np.corrcoef(np.abs(mg), tz)[0, 1]
    o["total_indep_of_strength_pct"] = float(100.0 * (1.0 - r * r))
    o["corr_absmargin_total"] = float(r)

    # (A1) does the total actually move outcome DISPERSION?  (the mechanism)
    resid = gm.margin_actual.values[ok] - mg
    q = np.quantile(tz, [1 / 3, 2 / 3])
    ter = np.digitize(tz, q)
    o["resid_sd_by_total_tercile"] = [
        [int((ter == k).sum()), float(resid[ter == k].std(ddof=1))]
        for k in range(3)]

    # (A2) does the total add to the market's OWN moneyline?
    X0 = np.c_[np.ones_like(z), z]
    X1 = np.c_[np.ones_like(z), z, tz, z * tz]
    b0, _ = logistic_fit(X0, y)
    b1, se1 = logistic_fit(X1, y)
    d = ll(y, sigmoid(X1 @ b1)) - ll(y, sigmoid(X0 @ b0))
    lo, hi, _ = cluster_boot(d, cl, n_boot=N_BOOT)
    tlo, thi, K = cluster_mean_t(d, cl)
    o["total_beyond_ml"] = dict(
        coef_total=float(b1[2]), t_total=float(b1[2] / se1[2]),
        coef_inter=float(b1[3]), t_inter=float(b1[3] / se1[3]),
        d_ll=float(d.mean()), lo=lo, hi=hi, tlo=tlo, thi=thi, K=K,
        n=int(len(y)))
    print(f"[2D-A {tag}] total beyond ML: coef {b1[2]:+.4f} (t={b1[2]/se1[2]:+.2f}) "
          f"inter {b1[3]:+.4f} (t={b1[3]/se1[3]:+.2f})  dLL={d.mean():+.5f} "
          f"CI[{lo:+.5f},{hi:+.5f}]")

    # (A3) realised win rate by total tercile INSIDE market-price bands — the
    # football project's own diagnostic form (does the second dimension move
    # realised outcomes the way it should?)
    pb = np.digitize(gm[pcol].values[ok], [0.35, 0.5, 0.65, 0.8])
    tabA = []
    for band in range(5):
        for k in range(3):
            m = (pb == band) & (ter == k)
            if m.sum() < 100:
                continue
            tabA.append(dict(band=int(band), tercile=int(k), n=int(m.sum()),
                             implied=float(gm[pcol].values[ok][m].mean()),
                             realised=float(y[m].mean()),
                             err=float((y[m] - gm[pcol].values[ok][m]).mean())))
    o["realised_by_priceband_x_total"] = tabA

    # (B) spread-vs-moneyline consistency
    fit_scale = None
    zs = mg / 6.96
    X2 = np.c_[np.ones_like(z), z, (z - zs)]
    b2, se2 = logistic_fit(X2, y)
    d2 = ll(y, sigmoid(X2 @ b2)) - ll(y, sigmoid(X0 @ b0))
    lo2, hi2, _ = cluster_boot(d2, cl, n_boot=N_BOOT)
    t2lo, t2hi, _ = cluster_mean_t(d2, cl)
    # which single price is better on its own?
    Xa, _ = logistic_fit(np.c_[np.ones_like(z), z], y)
    Xb, _ = logistic_fit(np.c_[np.ones_like(zs), zs], y)
    ll_ml = ll(y, sigmoid(np.c_[np.ones_like(z), z] @ Xa)).mean()
    ll_sp = ll(y, sigmoid(np.c_[np.ones_like(zs), zs] @ Xb)).mean()
    o["spread_vs_ml"] = dict(
        coef_disagree=float(b2[2]), t_disagree=float(b2[2] / se2[2]),
        d_ll=float(d2.mean()), lo=lo2, hi=hi2, tlo=t2lo, thi=t2hi,
        ll_ml_recalib=float(ll_ml), ll_spread_recalib=float(ll_sp),
        corr=float(np.corrcoef(z, zs)[0, 1]),
        sd_disagree=float((z - zs).std(ddof=1)), n=int(len(y)),
        fit_scale=fit_scale)
    print(f"[2D-B {tag}] spread-vs-ML disagreement: coef {b2[2]:+.4f} "
          f"(t={b2[2]/se2[2]:+.2f})  dLL={d2.mean():+.5f} "
          f"CI[{lo2:+.5f},{hi2:+.5f}]  |  recalibrated LL: ML {ll_ml:.5f} vs "
          f"SPREAD {ll_sp:.5f}")
    res[f"twod|{tag}"] = o
    return o


def main():
    res = {}
    g = load_panel()
    sc = sides(g, "close")
    # one row per game, HOME side only, for the log-loss tests
    gm = sc[sc.side == "home"].copy()
    gm = gm.rename(columns={"margin": "margin_home"})
    gg = g[["game_date", "home", "away", "score_home", "score_away"]].copy()
    gg["margin_actual"] = gg.score_home - gg.score_away
    n0 = len(gm)
    gm = gm.merge(gg[["game_date", "home", "away", "margin_actual"]],
                  on=["game_date", "home", "away"], how="left")
    assert len(gm) == n0, "score join fanned out"
    print(f"game frame (home rows, real closing MLs): n={len(gm)}  "
          f"seasons={gm.season.nunique()}")
    res["n_games"] = int(len(gm))
    res["seasons"] = sorted(gm.season.unique().tolist())

    for dv in ("prop", "shin", "goto"):
        sharpen(gm, "p_" + dv, dv, res)
    smooth_season(gm, "p_prop", "prop", res,
                  lams=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0))
    two_d(gm, "p_prop", "prop", res)
    era_split(gm, res)

    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(f"\nwrote {OUT}")
    return res


if __name__ == "__main__":
    main()
