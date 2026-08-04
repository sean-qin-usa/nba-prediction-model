"""REGIME D follow-up: tank-k ADAPTIVITY (walk-forward, honest) + Feb pocket.

Findings from rw_regimeD_lateres.py this feeds on:
  - post-tank residual slope on tsd is -1.10 +- 0.82 (under-correction);
  - per-season hindsight (log-loss MLE) k: -2.5 / -2.5 / -6.4 vs shipped
    ~-1.9..-2.2 -> k is NONSTATIONARY (|k| growing), estimator is
    all-history + n/(n+600) shrink -> structurally lags;
  - shape tests: linear in tsd is right (no saturation/acceleration gain).

This script:
  P1  Sub-window diagnostics: d/gm + resid~tsd slope by calendar month and by
      min(gp) bucket inside the active window (Feb front-edge pocket probe).
  P2  HONEST WALK-FORWARD adaptive-k replay. Mirrors the production estimator
      (OLS home_margin ~ [1, tsd, wdiff] on all completed ACTIVE games,
      2022-23 burn-in included, k=0 until 20 active rows) then varies ONLY
      the weighting/shrink (hypothesis: nonstationary k -> discount old
      seasons):
        v0 ship-mirror  : equal weights, shrink n/(n+600)   [validation]
        v1 noshrink     : equal weights, no shrink
        v2 shrink150    : equal weights, shrink n/(n+150)
        v3 ewma400      : per-row weight 0.5^(age_rows/400), shrink neff/(neff+150)
        v4 ewma150      : per-row weight 0.5^(age_rows/150), shrink neff/(neff+150)
      Counterfactual p' = sigmoid((m_base + k_t*tsd)/7.2) on active games,
      paired vs shipped p_us, per-season + pooled bootstrap CI.
  P3  Feb-pocket attribution: is the front-edge excess tsd-linked (slope) or
      not; residual by |tsd| tercile within Feb.

PIT strict: k_t at date D uses only games completed strictly before D.
DuckDB read_only. Market close used only as benchmark.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

DB = "/hdd/steveqin/sean_dev/nba_model/data/nba.duckdb"
CSV = "/hdd/steveqin/sean_dev/nba_model/data/capstone_pergame_tank.csv"
TANKSTATS = "/hdd/steveqin/sean_dev/nba_model/data/apr_tank_stats.csv"
SCALE = 7.2
K_MIN_ACTIVE = 20
RNG = np.random.default_rng(46)


def ll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def boot_ci(x, B=4000):
    x = np.asarray(x, float)
    idx = RNG.integers(0, len(x), (B, len(x)))
    return tuple(np.percentile(x[idx].mean(axis=1), [2.5, 97.5]))


def slope_ci(x, r):
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, r, rcond=None)
    e = r - X @ beta
    se = np.sqrt((e @ e) / (len(x) - 2) * np.linalg.inv(X.T @ X)[1, 1])
    return beta[1], 1.96 * se


def main():
    con = duckdb.connect(DB, read_only=True)
    tg = con.execute("""SELECT season, game_id, game_date, team_abbrev, wl,
        pts, is_home, matchup FROM nba_games WHERE game_id LIKE '002%'
          AND season IN ('2022-23','2023-24','2024-25','2025-26')""").df()
    con.close()
    tg["game_date"] = pd.to_datetime(tg.game_date)
    # neutral-site games have is_home=False for BOTH rows; matchup string is
    # authoritative: 'X @ Y' -> home=Y, 'X vs. Y' -> home=X
    tg["is_home"] = np.where(
        tg.matchup.str.contains(" @ "),
        tg.matchup.str.split(" @ ").str[1] == tg.team_abbrev,
        tg.matchup.str.split(" vs. ").str[0] == tg.team_abbrev)

    df = pd.read_csv(CSV, dtype={"game_id": str})
    df["game_id"] = df.game_id.str.zfill(10)
    df["game_date"] = pd.to_datetime(df.game_date)
    df["month"] = df.game_date.dt.month
    df["m_us"] = SCALE * logit(df.p_us)
    df["m_base"] = df.m_us - df.k * df.tsd
    df["d"] = ll(df.p_us.values, df.y.values) - ll(df.p_mkt.values, df.y.values)

    ts = pd.read_csv(TANKSTATS, dtype={"game_id": str})
    ts["game_id"] = ts.game_id.str.zfill(10)
    side = ts[["season", "game_id", "team_abbrev", "gp_before", "tank_score"]]
    df = df.merge(side.rename(columns={"team_abbrev": "home",
                                       "gp_before": "gp_h",
                                       "tank_score": "tank_h"})
                  .drop(columns="season"),
                  on=["game_id", "home"], how="left")
    df = df.merge(side.rename(columns={"team_abbrev": "away",
                                       "gp_before": "gp_a",
                                       "tank_score": "tank_a"})
                  .drop(columns="season"),
                  on=["game_id", "away"], how="left")

    # realized home margin from nba_games (also covers 2022-23 burn-in)
    gh = tg[tg.is_home][["season", "game_id", "game_date", "team_abbrev",
                         "pts"]].rename(columns={"team_abbrev": "home",
                                                 "pts": "pts_h"})
    ga = tg[~tg.is_home][["game_id", "team_abbrev", "pts"]].rename(
        columns={"team_abbrev": "away", "pts": "pts_a"})
    games = gh.merge(ga, on="game_id")
    games["margin_home"] = games.pts_h - games.pts_a
    df = df.merge(games[["game_id", "margin_home"]], on="game_id", how="left")
    assert df.margin_home.notna().all()

    # standings for wdiff (PIT strictly-before)
    hist = {}
    for (season, team), g in tg.groupby(["season", "team_abbrev"]):
        g = g.sort_values("game_date")
        hist[(season, team)] = (g.game_date.values,
                                np.cumsum((g.wl == "W").values))

    def wpct_before(season, team, D):
        dates, wins = hist[(season, team)]
        i = int(np.searchsorted(dates, D))
        return (wins[i - 1] / i) if i > 0 else 0.5

    # ---------------- P1 sub-window diagnostics ------------------------- #
    act = df[df.tsd != 0].copy()
    act["resid_pre"] = act.margin_home - act.m_base
    act["resid_post"] = act.margin_home - act.m_us
    print("=" * 100)
    print("P1 SUB-WINDOWS of active (tsd!=0) games")
    act["cal"] = np.where(act.month.isin([1, 2]), "Jan-Feb",
                          np.where(act.month == 3, "Mar", "Apr"))
    for cal in ["Jan-Feb", "Mar", "Apr"]:
        g = act[act.cal == cal]
        b, w = slope_ci(g.tsd.values, g.resid_pre.values)
        b2, w2 = slope_ci(g.tsd.values, g.resid_post.values)
        lo, hi = boot_ci(g.d.values)
        print(f"  {cal:7s} n={len(g):4d}  d/gm {g.d.mean():+.4f} "
              f"CI({lo:+.4f},{hi:+.4f})  slope_pre {b:+.2f}+-{w:.2f} "
              f"slope_post {b2:+.2f}+-{w2:.2f}  mean|tsd| {g.tsd.abs().mean():.3f}")
    act["gpmin"] = act[["gp_h", "gp_a"]].min(axis=1)
    for lo_, hi_ in [(40, 55), (55, 60), (60, 65), (65, 70), (70, 76), (76, 83)]:
        g = act[(act.gpmin >= lo_) & (act.gpmin < hi_)]
        if len(g) < 30:
            continue
        b, w = slope_ci(g.tsd.values, g.resid_post.values)
        print(f"  gpmin [{lo_},{hi_}) n={len(g):4d}  d/gm {g.d.mean():+.4f}  "
              f"slope_post {b:+.2f}+-{w:.2f}")

    # ---------------- P3 Feb pocket attribution ------------------------- #
    print("=" * 100)
    print("P3 FEB FRONT-EDGE POCKET (active & pre-Mar, n=%d, d/gm %+.4f)"
          % ((act.cal == "Jan-Feb").sum(), act[act.cal == "Jan-Feb"].d.mean()))
    g = act[act.cal == "Jan-Feb"].copy()
    g["ter"] = pd.qcut(g.tsd.abs(), 3, labels=["lo|tsd|", "mid", "hi|tsd|"])
    for t, gg in g.groupby("ter", observed=True):
        lo, hi = boot_ci(gg.d.values)
        print(f"  {t}: n={len(gg):3d}  d/gm {gg.d.mean():+.4f} "
              f"CI({lo:+.4f},{hi:+.4f})  resid_post {gg.resid_post.mean():+.2f}")
    # is it outs-driven / p_mkt divergence-driven instead?
    g["div"] = (g.p_us - g.p_mkt).abs()
    print(f"  mean |p_us-p_mkt| Feb {g['div'].mean():.4f} vs Mar/Apr "
          f"{(act[act.cal != 'Jan-Feb'].p_us - act[act.cal != 'Jan-Feb'].p_mkt).abs().mean():.4f}")

    # ---------------- P2 walk-forward adaptive-k replay ------------------ #
    print("=" * 100)
    print("P2 WALK-FORWARD ADAPTIVE k (production-mirror design, "
          "y=home_margin, X=[1,tsd,wdiff]; only weighting/shrink varies)")
    # build full active-row history incl 2022-23 burn-in (merge by abbrev,
    # not is_home — neutral-site rows have is_home False on both sides)
    tk_h = side.rename(columns={"team_abbrev": "home", "gp_before": "bgp_h",
                                "tank_score": "btank_h"})
    tk_a = side.rename(columns={"team_abbrev": "away", "gp_before": "bgp_a",
                                "tank_score": "btank_a"}).drop(
        columns="season")
    allg = games.merge(tk_h, on=["season", "game_id", "home"], how="inner")
    allg = allg.merge(tk_a, on=["game_id", "away"], how="inner")
    allg["tsd"] = (np.where(allg.bgp_h >= 55, allg.btank_h, 0.0)
                   - np.where(allg.bgp_a >= 55, allg.btank_a, 0.0))
    allg = allg.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    allg["wdiff"] = [wpct_before(r.season, r.home, np.datetime64(r.game_date))
                     - wpct_before(r.season, r.away, np.datetime64(r.game_date))
                     for r in allg.itertuples()]
    # sanity: our reconstructed tsd matches the capstone tsd on 2023-24+
    chk = allg.merge(df[["game_id", "tsd"]], on="game_id", how="inner",
                     suffixes=("_rec", "_cap"))
    mism = (chk.tsd_rec - chk.tsd_cap).abs().max()
    print(f"  tsd reconstruction max|diff| vs capstone: {mism:.2e} "
          f"(n={len(chk)})")

    variants = {
        "v0 ship-mirror shrink600": dict(hl=None, n0=600),
        "v1 noshrink": dict(hl=None, n0=0),
        "v2 shrink150": dict(hl=None, n0=150),
        "v3 ewma400+shrink150": dict(hl=400, n0=150),
        "v4 ewma150+shrink150": dict(hl=150, n0=150),
    }
    active_hist_idx = np.where(allg.tsd.values != 0)[0]
    dates = allg.game_date.values
    y_m = allg.margin_home.values
    X_all = np.column_stack([np.ones(len(allg)), allg.tsd.values,
                             allg.wdiff.values])

    # daily refit: k for date D from active rows strictly before D
    uniq_dates = np.unique(df.game_date.values)
    kpaths = {v: {} for v in variants}
    for D in uniq_dates:
        m = active_hist_idx[dates[active_hist_idx] < D]
        for vn, cfg in variants.items():
            if len(m) < K_MIN_ACTIVE:
                kpaths[vn][D] = 0.0
                continue
            X, ym = X_all[m], y_m[m]
            if cfg["hl"] is None:
                w = np.ones(len(m))
            else:
                age = len(m) - 1 - np.arange(len(m))
                w = 0.5 ** (age / cfg["hl"])
            W = np.sqrt(w)
            beta, *_ = np.linalg.lstsq(X * W[:, None], ym * W, rcond=None)
            k = beta[1]
            neff = w.sum() ** 2 / (w ** 2).sum()
            if cfg["n0"] > 0:
                k *= neff / (neff + cfg["n0"])
            kpaths[vn][D] = float(min(0.0, k))   # sign guard: tank k <= 0

    amask = (df.tsd != 0).values
    yv = df.y.values
    print(f"  shipped active d/gm {df[amask].d.mean():+.4f} "
          f"(n={int(amask.sum())})")
    for vn in variants:
        kk = df.game_date.map(kpaths[vn]).values
        p = sigmoid((df.m_base.values + kk * df.tsd.values) / SCALE)
        delta = ll(p, yv) - ll(df.p_us.values, yv)
        dd = ll(p, yv) - ll(df.p_mkt.values, yv)
        lo, hi = boot_ci(delta[amask])
        by = " ".join(
            f"{s[-5:]}:{delta[amask & (df.season == s).values].mean():+.4f}"
            for s in sorted(df.season.unique()))
        kapr = np.mean([kpaths[vn][D] for D in uniq_dates
                        if pd.Timestamp(D).month == 4])
        print(f"  {vn:26s} d/gm {dd[amask].mean():+.4f}  vs shipped "
              f"{delta[amask].mean():+.5f} CI({lo:+.5f},{hi:+.5f}) "
              f"nats {delta[amask].sum():+5.1f}  by-season {by}  "
              f"mean k(Apr) {kapr:+.2f}")


if __name__ == "__main__":
    main()
