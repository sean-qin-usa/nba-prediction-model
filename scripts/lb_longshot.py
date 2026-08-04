"""Favourite-longshot bias (FLB) in the NBA moneyline market — the direct test.

Answers, on OUR data, the question the sister football project never asked:
  1. is the market miscalibrated ALONG THE PRICE AXIS (longshots losing more
     often than priced, favourites winning more often than priced)?
  2. if so, does the bias clear the overround anywhere?
  3. how much of our frozen rules' measured edge is just favourite exposure?
  4. the football project's OWN three inefficiency tests, NBA analogues.

Read-only on data/nba.duckdb (read_only=True, 60s retry).  Writes only
data/lb_longshot.json + charts (a separate script).  nbapred/, the frozen
registry and scripts/bet_engine.py are untouched.

Run:  python scripts/lb_longshot.py
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

import duckdb                                                    # noqa: E402
import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402

DB = os.path.join(ROOT, "data", "nba.duckdb")
OUT = os.path.join(ROOT, "data", "lb_longshot.json")

SEED = 20260802
N_BOOT = 4000
OVERROUND = 1.043          # D121-validated proportional overround
SPREAD_SCALE = 6.96        # the D126-frozen synthetic map p = sigmoid(m/6.96)

# probability bins.  Finer in the tails, where the FLB signature lives.
EDGES = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                  0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
                  1.0])


# --------------------------------------------------------------- utilities --
def am2dec(a):
    """American -> decimal.  NaN-safe.  (D120: a sign slip here once produced
    decimals < 1 and 107% breakevens.)"""
    a = np.asarray(a, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(a > 0, 1.0 + a / 100.0, 1.0 + 100.0 / np.abs(a))
    return np.where(np.isnan(a) | (a == 0), np.nan, d)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------ devig --
# Every method takes r = raw implied probs, shape (n, 2), rows summing to the
# overround, and returns fair probs summing to 1.

def dv_prop(r):
    """Proportional / multiplicative.  The convention D75/D120/D121 use."""
    return r / r.sum(1, keepdims=True)


def dv_add(r):
    """Additive: subtract the excess equally."""
    k = r.shape[1]
    return r - (r.sum(1, keepdims=True) - 1.0) / k


def dv_power(r):
    """Power: p_i = r_i ** k, solve k so the row sums to 1.  k > 1, and since
    x**k shrinks small x proportionally MORE, this is FLB-aware."""
    lo = np.ones(len(r)) * 0.2
    hi = np.ones(len(r)) * 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        s = (r ** mid[:, None]).sum(1)
        hi = np.where(s > 1.0, hi, mid)
        lo = np.where(s > 1.0, mid, lo)
    k = 0.5 * (lo + hi)
    p = r ** k[:, None]
    return p / p.sum(1, keepdims=True)


def dv_or(r):
    """Odds-ratio (Cheung): p_i/(1-p_i) = (r_i/(1-r_i)) / c, solve c >= 1.
    The `goto_conversion` family's older cousin; also FLB-aware."""
    o = r / (1.0 - r)
    lo = np.ones(len(r)) * 1e-6
    hi = np.ones(len(r)) * 50.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        oo = o / mid[:, None]
        s = (oo / (1.0 + oo)).sum(1)
        hi = np.where(s > 1.0, hi, mid)
        lo = np.where(s > 1.0, mid, lo)
    c = 0.5 * (lo + hi)
    oo = o / c[:, None]
    p = oo / (1.0 + oo)
    return p / p.sum(1, keepdims=True)


def dv_shin(r):
    """Shin (1993): an insider-trading model.  p_i = (sqrt(z^2 + 4(1-z) r_i^2/O)
    - z) / (2(1-z)); solve z in [0, 0.5).  The canonical FLB-aware devig."""
    O = r.sum(1, keepdims=True)
    lo = np.zeros(len(r))
    hi = np.ones(len(r)) * 0.6

    def s_of(z):
        z = z[:, None]
        num = np.sqrt(z * z + 4.0 * (1.0 - z) * r * r / O) - z
        return (num / (2.0 * (1.0 - z))).sum(1)

    for _ in range(90):
        mid = 0.5 * (lo + hi)
        s = s_of(mid)
        hi = np.where(s > 1.0, hi, mid)
        lo = np.where(s > 1.0, mid, lo)
    z = (0.5 * (lo + hi))[:, None]
    num = np.sqrt(z * z + 4.0 * (1.0 - z) * r * r / O) - z
    p = num / (2.0 * (1.0 - z))
    return p / p.sum(1, keepdims=True)


def dv_goto(r):
    """goto_conversion: shift every outcome by the SAME number of its own
    standard errors, se_i = sqrt((r_i - r_i^2)/r_i^2) = sqrt((1-r_i)/r_i).
    Longshots have the larger se, so they are cut hardest — FLB-aware by
    construction."""
    se = np.sqrt(np.clip((1.0 - r) / r, 0.0, None))
    step = (r.sum(1, keepdims=True) - 1.0) / se.sum(1, keepdims=True)
    p = r - se * step
    return np.clip(p, 1e-6, 1 - 1e-6)


DEVIGS = {"prop": dv_prop, "add": dv_add, "power": dv_power,
          "or": dv_or, "shin": dv_shin, "goto": dv_goto}


# ------------------------------------------------------ clustered inference --
def cluster_boot(vals, clus, stat=None, n_boot=N_BOOT, seed=SEED):
    """Season-clustered bootstrap CI for the mean of `vals` (or `stat(v)`)."""
    vals = np.asarray(vals, float)
    clus = np.asarray(clus)
    keys = np.unique(clus)
    idx = {k: np.where(clus == k)[0] for k in keys}
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        sel = np.concatenate([idx[keys[j]] for j in pick])
        out[b] = vals[sel].mean() if stat is None else stat(sel)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi), float(out.std(ddof=1))


def cluster_mean_t(vals, clus):
    """Cluster-mean t interval at K-1 dof — GATE_POLICY_V2 §9.1(4), the
    conservative bound that counts at small K."""
    from math import sqrt
    df = pd.DataFrame({"v": np.asarray(vals, float), "c": np.asarray(clus)})
    mu = df.groupby("c").v.mean().values
    K = len(mu)
    if K < 2:
        return float("nan"), float("nan"), K
    se = mu.std(ddof=1) / sqrt(K)
    # two-sided 95% t quantile at K-1 dof
    try:
        from statistics import NormalDist
        tq = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
              7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
              13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
              18: 2.101, 19: 2.093, 20: 2.086}.get(
                  K - 1, NormalDist().inv_cdf(0.975))
    except Exception:                                            # pragma: no cover
        tq = 1.96
    m = float(mu.mean())
    return m - tq * se, m + tq * se, K


def icc_oneway(vals, clus):
    """One-way ICC and the ANOVA design effect."""
    df = pd.DataFrame({"v": np.asarray(vals, float), "c": np.asarray(clus)})
    g = df.groupby("c").v
    n_i = g.size().values.astype(float)
    K = len(n_i)
    if K < 2:
        return float("nan"), float("nan")
    N = n_i.sum()
    gm = df.v.mean()
    msb = (n_i * (g.mean().values - gm) ** 2).sum() / (K - 1)
    msw = ((g.var(ddof=1).values * (n_i - 1)).sum()) / (N - K)
    n0 = (N - (n_i ** 2).sum() / N) / (K - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    return float(icc), float(1.0 + (n0 - 1) * icc)


# ------------------------------------------------------------------- data ---
def load_panel():
    """One row per GAME with every price we have; then exploded to one row per
    SIDE.  odds_open is the unified panel: it carries SBR closes 2007-08..
    2022-23 and ESPN/ActionNetwork closes+opens 2023-24..2025-26, and its
    close_ml agrees with odds_market's ml on 100% of the 18,494 overlapping
    games (verified, see notes)."""
    con = duckdb.connect(DB, read_only=True)
    try:
        g = con.execute("""
            select season, season_end, game_date, home, away,
                   score_home, score_away, home_win,
                   open_margin, close_margin, open_total, close_total,
                   open_ml_home, open_ml_away, close_ml_home, close_ml_away,
                   source
            from odds_open
            where season_end >= 2008 and home_win is not null
            order by game_date, home
        """).fetchdf()
    finally:
        con.close()
    return g


def sides(g, when):
    """Explode a game panel into 2 rows/game for price epoch `when`."""
    mlh = g[f"{when}_ml_home"].values.astype(float)
    mla = g[f"{when}_ml_away"].values.astype(float)
    dh, da = am2dec(mlh), am2dec(mla)
    r = np.c_[1.0 / dh, 1.0 / da]
    ov = r.sum(1)
    ok = np.isfinite(ov) & (ov > 1.0) & (ov < 1.25)
    out = {}
    for name, fn in DEVIGS.items():
        p = np.full_like(r, np.nan)
        if ok.any():
            p[ok] = fn(r[ok])
        out[name] = p
    y = g.home_win.values.astype(float)
    rows = []
    for k, side in enumerate(("home", "away")):
        d = pd.DataFrame({
            "season": g.season.values, "season_end": g.season_end.values,
            "game_date": g.game_date.values,
            "home": g.home.values, "away": g.away.values,
            "side": side,
            "y": y if side == "home" else 1.0 - y,
            "dec": dh if side == "home" else da,
            "raw": r[:, k], "ov": ov, "ok": ok,
            "margin": g[f"{when}_margin"].values * (1 if side == "home" else -1),
            "total": g[f"{when}_total"].values,
            "source": g.source.values,
        })
        for name in DEVIGS:
            d["p_" + name] = out[name][:, k]
        rows.append(d)
    s = pd.concat(rows, ignore_index=True)
    return s[s.ok & np.isfinite(s.dec) & np.isfinite(s.p_prop)].copy()


# ------------------------------------------------------------ calibration ---
def calib_table(s, pcol, tag):
    """Per-bin implied vs realised with season-clustered CIs, plus the blind
    ROI of backing that bin at the RAW transactable price."""
    p = s[pcol].values
    b = np.clip(np.digitize(p, EDGES) - 1, 0, len(EDGES) - 2)
    rows = []
    for i in range(len(EDGES) - 1):
        m = b == i
        n = int(m.sum())
        if n < 30:
            continue
        yy = s.y.values[m]
        pp = p[m]
        cl = s.season.values[m]
        err = yy - pp
        lo, hi, se = cluster_boot(err, cl)
        tlo, thi, K = cluster_mean_t(err, cl)
        dd = s.dec.values[m]
        pnl = np.where(yy > 0.5, dd - 1.0, -1.0)
        rlo, rhi, rse = cluster_boot(pnl, cl)
        rtlo, rthi, _ = cluster_mean_t(pnl, cl)
        rows.append(dict(
            tag=tag, bin=f"[{EDGES[i]:.2f},{EDGES[i+1]:.2f})", lo_edge=EDGES[i],
            n=n, implied=float(pp.mean()), realised=float(yy.mean()),
            err=float(err.mean()), err_lo=lo, err_hi=hi, err_se=se,
            err_tlo=tlo, err_thi=thi, K=K,
            mean_dec=float(dd.mean()), breakeven=float((1.0 / dd).mean()),
            roi=float(pnl.mean()), roi_lo=rlo, roi_hi=rhi,
            roi_tlo=rtlo, roi_thi=rthi,
            sig_err=bool(lo > 0 or hi < 0), sig_roi=bool(rlo > 0 or rhi < 0),
        ))
    return pd.DataFrame(rows)


def flb_slopes(s, pcol):
    """Two summary statistics for the whole curve.
      lin_b : OLS slope of (y - p) on (p - 0.5).  > 0 = FLB (favourites win
              more than priced, longshots less).
      logit_b : y ~ sigmoid(a + b*logit(p)).  b > 1 = market UNDER-confident
              = FLB.  This is exactly the football project's SHARPEN test in
              probability space.
    Both season-clustered."""
    p = s[pcol].values
    y = s.y.values
    cl = s.season.values
    x = p - 0.5

    def lin(idx):
        xx, yy, pp = x[idx], y[idx], p[idx]
        e = yy - pp
        vx = xx - xx.mean()
        return float((vx * (e - e.mean())).sum() / (vx * vx).sum())

    def lgt(idx):
        z = logit(p[idx])
        yy = y[idx]
        a, bb = 0.0, 1.0
        for _ in range(60):
            eta = a + bb * z
            mu = sigmoid(eta)
            w = np.clip(mu * (1 - mu), 1e-9, None)
            r = yy - mu
            X = np.c_[np.ones_like(z), z]
            XtW = X.T * w
            H = XtW @ X
            gsc = X.T @ r
            try:
                step = np.linalg.solve(H, gsc)
            except np.linalg.LinAlgError:
                break
            a += step[0]
            bb += step[1]
            if max(abs(step)) < 1e-10:
                break
        return float(bb), float(a)

    allidx = np.arange(len(p))
    lb = lin(allidx)
    gb, ga = lgt(allidx)
    llo, lhi, lse = cluster_boot(np.zeros(len(p)), cl, stat=lin)
    glo, ghi, gse = cluster_boot(np.zeros(len(p)), cl, stat=lambda i: lgt(i)[0])
    # cluster-mean t on the per-season slope
    dfx = pd.DataFrame({"c": cl})
    per = []
    for k, idx in dfx.groupby("c").groups.items():
        idx = np.asarray(idx)
        if len(idx) > 50:
            per.append((k, lin(idx), lgt(idx)[0]))
    pl = np.array([q[1] for q in per])
    pg = np.array([q[2] for q in per])
    K = len(per)
    from math import sqrt
    tq = {18: 2.101, 19: 2.093, 17: 2.110, 2: 4.303, 3: 3.182}.get(K - 1, 2.1)
    return dict(
        lin_b=lb, lin_lo=llo, lin_hi=lhi, lin_se=lse,
        lin_tlo=float(pl.mean() - tq * pl.std(ddof=1) / sqrt(K)),
        lin_thi=float(pl.mean() + tq * pl.std(ddof=1) / sqrt(K)),
        logit_b=gb, logit_a=ga, logit_lo=glo, logit_hi=ghi, logit_se=gse,
        logit_tlo=float(pg.mean() - tq * pg.std(ddof=1) / sqrt(K)),
        logit_thi=float(pg.mean() + tq * pg.std(ddof=1) / sqrt(K)),
        K=K, per_season_lin=[[q[0], q[1]] for q in per],
        per_season_logit=[[q[0], q[2]] for q in per],
        n=int(len(p)),
    )


# ------------------------------------------------------------------- main ---
def main():
    res = {}
    g = load_panel()
    print(f"odds_open panel: {len(g)} games, "
          f"{g.season.nunique()} seasons {g.season.min()}..{g.season.max()}")

    for when in ("close", "open"):
        ok = np.isfinite(am2dec(g[f"{when}_ml_home"].values.astype(float)))
        print(f"  real {when} moneylines: {int(ok.sum())} games")

    sc = sides(g, "close")
    so = sides(g, "open")
    res["coverage"] = dict(
        games=int(len(g)), seasons=int(g.season.nunique()),
        close_sides=int(len(sc)), open_sides=int(len(so)),
        close_games=int(len(sc) / 2), open_games=int(len(so) / 2),
        close_seasons=sorted(sc.season.unique().tolist()),
        open_seasons=sorted(so.season.unique().tolist()),
    )
    print(f"  usable close sides {len(sc)} ({len(sc)//2} games); "
          f"open sides {len(so)} ({len(so)//2} games)")

    # -------- overround, the football 2.82% benchmark --------------------
    ovr = {}
    for tag, s in (("close", sc), ("open", so)):
        u = s.drop_duplicates(["game_date", "home", "away"])
        ovr[tag] = dict(mean=float(u.ov.mean()), median=float(u.ov.median()),
                        n=int(len(u)))
        by = u.groupby("season").ov.agg(["mean", "size"])
        ovr[tag]["by_season"] = {k: [float(v["mean"]), int(v["size"])]
                                 for k, v in by.iterrows()}
    # favourite vs dog overround share (is the vig loaded on the dog?)
    u = sc.copy()
    u["fav"] = u.p_prop > 0.5
    ovr["close_vig_share"] = {
        "fav_raw_minus_fair": float((u[u.fav].raw - u[u.fav].p_prop).mean()),
        "dog_raw_minus_fair": float((u[~u.fav].raw - u[~u.fav].p_prop).mean()),
    }
    res["overround"] = ovr
    print(f"\nOVERROUND  close {ovr['close']['mean']:.4f} "
          f"(margin {100*(ovr['close']['mean']-1):.2f}%)  n={ovr['close']['n']}")
    print(f"           open  {ovr['open']['mean']:.4f} "
          f"(margin {100*(ovr['open']['mean']-1):.2f}%)  n={ovr['open']['n']}")

    # -------- TASK 1: calibration by bin, every devig --------------------
    res["calib"] = {}
    res["slopes"] = {}
    for tag, s in (("close", sc), ("open", so)):
        for dv in DEVIGS:
            t = calib_table(s, "p_" + dv, f"{tag}|{dv}")
            res["calib"][f"{tag}|{dv}"] = t.to_dict("records")
            res["slopes"][f"{tag}|{dv}"] = flb_slopes(s, "p_" + dv)
        print(f"\n{'='*100}\nCALIBRATION — {tag} (real moneylines)\n{'='*100}")
        for dv in DEVIGS:
            sl = res["slopes"][f"{tag}|{dv}"]
            print(f"  devig={dv:6s}  lin_b={sl['lin_b']:+.4f} "
                  f"[{sl['lin_lo']:+.4f},{sl['lin_hi']:+.4f}]  "
                  f"logit_b={sl['logit_b']:.4f} "
                  f"[{sl['logit_lo']:.4f},{sl['logit_hi']:.4f}]  n={sl['n']}")
        t = pd.DataFrame(res["calib"][f"{tag}|prop"])
        print("\n  PROPORTIONAL devig, per bin:")
        print(t[["bin", "n", "implied", "realised", "err", "err_lo", "err_hi",
                 "mean_dec", "roi", "roi_lo", "roi_hi"]].to_string(
                     index=False, float_format=lambda v: f"{v:+.4f}"))

    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(f"\nwrote {OUT}")
    return res


if __name__ == "__main__":
    main()
