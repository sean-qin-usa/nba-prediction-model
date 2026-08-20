"""REGIME D follow-up 2: pocket anatomy (Feb front-edge / April) and
FORM-ON-TOP-OF-TANK additivity.

Inputs from rw_regimeD_lateres.py / rw_regimeD_tankk.py:
  - active window (tsd!=0) residual 20.9 nats splits: Jan-Feb +0.0336 (n=219),
    Mar +0.0033 AT PAR (n=707), Apr +0.0353 (n=316);
  - honest adaptive-k recovers ~1.9 nats (25-26-heavy).

Sections:
  Q1  April anatomy: outs buckets (oracle outs in CSV), elimination-involving
      games (PIT max-wins < 10th-place wins proxy), market-confidence terciles,
      form5-diff slope by sub-window.
  Q2  Feb anatomy: days-since-trade-deadline windows, form5 slope.
  Q3  FORM ON TOP OF TANK (D71 F1 term, post-tank additivity):
      hindsight pooled c; HONEST walk-forward c (production-style OLS
      y=home_margin, X=[1, fdiff, wdiff, tsd] on accumulated late-active rows,
      2022-23 burn-in, shrink n/(n+600) and n/(n+150)); overlap corr(tsd,fdiff);
      per-pocket recovery. Combined with adaptive-k v2 (shrink150).
  NOTE: D71 froze the SOLO form term as F1 for 2026-27 live confirm; this is a
  spent-data diagnostic of post-tank additivity, not a ship gate.

PIT strict. DuckDB read_only. Market data benchmark/subset-definition only.
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
FORM_N = 5
RNG = np.random.default_rng(46)
DEADLINES = {"2023-24": "2024-02-08", "2024-25": "2025-02-06",
             "2025-26": "2026-02-05"}


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
    if len(x) < 2:
        return (np.nan, np.nan)
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
        pts, matchup FROM nba_games WHERE game_id LIKE '002%'
          AND season IN ('2022-23','2023-24','2024-25','2025-26')""").df()
    con.close()
    tg["game_date"] = pd.to_datetime(tg.game_date)
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

    # margins + home/away per game (incl 2022-23)
    gh = tg[tg.is_home][["season", "game_id", "game_date", "team_abbrev",
                         "pts"]].rename(columns={"team_abbrev": "home",
                                                 "pts": "pts_h"})
    ga = tg[~tg.is_home][["game_id", "team_abbrev", "pts"]].rename(
        columns={"team_abbrev": "away", "pts": "pts_a"})
    games = gh.merge(ga, on="game_id")
    games["margin_home"] = games.pts_h - games.pts_a
    df = df.merge(games[["game_id", "margin_home"]], on="game_id", how="left")

    # per-team signed-margin history for form5 + standings (PIT)
    long = pd.concat([
        games.rename(columns={"home": "team"})[
            ["season", "game_id", "game_date", "team"]].assign(
            sm=games.margin_home.values),
        games.rename(columns={"away": "team"})[
            ["season", "game_id", "game_date", "team"]].assign(
            sm=-games.margin_home.values)]).sort_values("game_date")
    long["form5"] = (long.groupby(["season", "team"]).sm
                     .transform(lambda s: s.shift(1)
                                .rolling(FORM_N, min_periods=FORM_N).mean()))
    fkey = long[["season", "game_id", "team", "form5"]]
    df = df.merge(fkey.rename(columns={"team": "home", "form5": "f5_h"}),
                  on=["season", "game_id", "home"], how="left")
    df = df.merge(fkey.rename(columns={"team": "away", "form5": "f5_a"}),
                  on=["season", "game_id", "away"], how="left")
    df["fdiff"] = (df.f5_h - df.f5_a).fillna(0.0)

    hist = {}
    for (season, team), g in tg.groupby(["season", "team_abbrev"]):
        g = g.sort_values("game_date")
        hist[(season, team)] = (g.game_date.values,
                                np.cumsum((g.wl == "W").values))

    def rec(season, team, D):
        dates, wins = hist[(season, team)]
        i = int(np.searchsorted(dates, D))
        w = int(wins[i - 1]) if i > 0 else 0
        return w, i

    EAST = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA",
            "MIL", "NYK", "ORL", "PHI", "TOR", "WAS"}
    ccache = {}

    def w10(season, is_east, D):
        ck = (season, is_east, D)
        if ck not in ccache:
            ws = sorted((rec(season, t, D)[0] for (s, t) in hist
                         if s == season and ((t in EAST) == is_east)),
                        reverse=True)
            ccache[ck] = ws[9]
        return ccache[ck]

    def elim(season, team, D):
        w, gp = rec(season, team, D)
        return (w + 82 - gp) < w10(season, team in EAST, D)

    df["elim_h"] = [elim(r.season, r.home, np.datetime64(r.game_date))
                    for r in df.itertuples()]
    df["elim_a"] = [elim(r.season, r.away, np.datetime64(r.game_date))
                    for r in df.itertuples()]

    act = df[df.tsd != 0].copy()
    act["resid_post"] = act.margin_home - act.m_us
    act["cal"] = np.where(act.month.isin([1, 2]), "Feb",
                          np.where(act.month == 3, "Mar", "Apr"))
    act["n_out"] = act.n_out_home + act.n_out_away

    # ---------------- Q1 April anatomy ----------------------------------- #
    print("=" * 100)
    print("Q1 APRIL POCKET (active, n=%d, d/gm %+.4f)"
          % ((act.cal == "Apr").sum(), act[act.cal == "Apr"].d.mean()))
    ap = act[act.cal == "Apr"]
    for lo_, hi_, nm in [(0, 1, "outs 0"), (1, 3, "outs 1-2"),
                         (3, 6, "outs 3-5"), (6, 99, "outs 6+")]:
        g = ap[(ap.n_out >= lo_) & (ap.n_out < hi_)]
        if len(g) == 0:
            continue
        lo, hi = boot_ci(g.d.values)
        print(f"  {nm:10s} n={len(g):3d}  d/gm {g.d.mean():+.4f} "
              f"CI({lo:+.4f},{hi:+.4f})")
    for nm, m in [("elim one side", ap.elim_h ^ ap.elim_a),
                  ("elim both", ap.elim_h & ap.elim_a),
                  ("elim none", ~ap.elim_h & ~ap.elim_a)]:
        g = ap[m]
        lo, hi = boot_ci(g.d.values)
        print(f"  {nm:14s} n={len(g):3d}  d/gm {g.d.mean():+.4f} "
              f"CI({lo:+.4f},{hi:+.4f})")
    ap2 = ap.copy()
    ap2["conf"] = pd.qcut((ap2.p_mkt - 0.5).abs(), 3,
                          labels=["mkt-tossup", "mkt-mid", "mkt-heavy"])
    for t, g in ap2.groupby("conf", observed=True):
        lo, hi = boot_ci(g.d.values)
        print(f"  {t:11s} n={len(g):3d}  d/gm {g.d.mean():+.4f} "
              f"CI({lo:+.4f},{hi:+.4f})")
    for cal in ["Feb", "Mar", "Apr"]:
        g = act[act.cal == cal]
        b, w = slope_ci(g.fdiff.values, g.resid_post.values)
        print(f"  resid_post ~ fdiff slope {cal}: {b:+.3f}+-{w:.3f} "
              f"(n={len(g)})  corr(tsd,fdiff)={np.corrcoef(g.tsd, g.fdiff)[0, 1]:+.2f}")

    # ---------------- Q2 Feb anatomy ------------------------------------- #
    print("=" * 100)
    print("Q2 FEB POCKET vs trade deadline")
    act["dl_days"] = [(r.game_date - pd.Timestamp(DEADLINES[r.season])).days
                      for r in act.itertuples()]
    for nm, m in [("pre-deadline", act.dl_days < 0),
                  ("deadline +0-13d", (act.dl_days >= 0) & (act.dl_days < 14)),
                  ("deadline +14-27d", (act.dl_days >= 14) & (act.dl_days < 28)),
                  ("deadline +28d..Mar31", (act.dl_days >= 28)
                   & (act.cal != "Apr"))]:
        g = act[m]
        if len(g) == 0:
            continue
        lo, hi = boot_ci(g.d.values)
        print(f"  {nm:20s} n={len(g):4d}  d/gm {g.d.mean():+.4f} "
              f"CI({lo:+.4f},{hi:+.4f})")

    # ---------------- Q3 form on top of tank ----------------------------- #
    print("=" * 100)
    print("Q3 FORM-ON-TOP-OF-TANK (F1 additivity diagnostic)")
    amask = (df.tsd != 0).values
    yv = df.y.values
    # hindsight pooled c
    cs = np.linspace(-1, 1, 201)
    best, bc = 1e18, 0.0
    for c in cs:
        p = sigmoid((df.m_us.values
                     + np.where(amask, c * df.fdiff.values, 0.0)) / SCALE)
        L = ll(p, yv)[amask].sum()
        if L < best:
            best, bc = L, c
    p = sigmoid((df.m_us.values
                 + np.where(amask, bc * df.fdiff.values, 0.0)) / SCALE)
    delta = ll(p, yv) - ll(df.p_us.values, yv)
    lo, hi = boot_ci(delta[amask])
    print(f"  hindsight pooled c*={bc:+.3f}  vs shipped "
          f"{delta[amask].mean():+.5f} CI({lo:+.5f},{hi:+.5f}) "
          f"nats {delta[amask].sum():+5.1f}")

    # honest walk-forward c (and adaptive k) from accumulated late-active rows
    tk_h = side.rename(columns={"team_abbrev": "home", "gp_before": "bgp_h",
                                "tank_score": "btank_h"})
    tk_a = side.rename(columns={"team_abbrev": "away", "gp_before": "bgp_a",
                                "tank_score": "btank_a"}).drop(columns="season")
    allg = games.merge(tk_h, on=["season", "game_id", "home"], how="inner")
    allg = allg.merge(tk_a, on=["game_id", "away"], how="inner")
    allg["tsd"] = (np.where(allg.bgp_h >= 55, allg.btank_h, 0.0)
                   - np.where(allg.bgp_a >= 55, allg.btank_a, 0.0))
    allg = allg.merge(fkey.rename(columns={"team": "home", "form5": "bf5_h"}),
                      on=["season", "game_id", "home"], how="left")
    allg = allg.merge(fkey.rename(columns={"team": "away", "form5": "bf5_a"}),
                      on=["season", "game_id", "away"], how="left")
    allg["fdiff"] = (allg.bf5_h - allg.bf5_a).fillna(0.0)
    allg = allg.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    def wpct_before(season, team, D):
        w, gp = rec(season, team, D)
        return (w / gp) if gp > 0 else 0.5

    allg["wdiff"] = [wpct_before(r.season, r.home, np.datetime64(r.game_date))
                     - wpct_before(r.season, r.away,
                                   np.datetime64(r.game_date))
                     for r in allg.itertuples()]
    aidx = np.where(allg.tsd.values != 0)[0]
    dts = allg.game_date.values
    X_all = np.column_stack([np.ones(len(allg)), allg.tsd.values,
                             allg.fdiff.values, allg.wdiff.values])
    y_m = allg.margin_home.values
    uniq_dates = np.unique(df.game_date.values)
    paths = {}
    for n0 in (600, 150):
        kf, cf = {}, {}
        for D in uniq_dates:
            m = aidx[dts[aidx] < D]
            if len(m) < K_MIN_ACTIVE:
                kf[D], cf[D] = 0.0, 0.0
                continue
            beta, *_ = np.linalg.lstsq(X_all[m], y_m[m], rcond=None)
            sh = len(m) / (len(m) + n0)
            kf[D] = float(min(0.0, beta[1] * sh))
            cf[D] = float(max(0.0, beta[2] * sh))
        paths[n0] = (kf, cf)

    for n0 in (600, 150):
        kf, cf = paths[n0]
        kk = df.game_date.map(kf).values
        cc = df.game_date.map(cf).values
        for nm, m_new in [
                (f"wf c only (shrink{n0}), tank=ship",
                 df.m_us.values + cc * df.fdiff.values),
                (f"wf k+c joint (shrink{n0})",
                 df.m_base.values + kk * df.tsd.values
                 + cc * df.fdiff.values)]:
            p = sigmoid(np.where(amask, m_new, df.m_us.values) / SCALE)
            delta = ll(p, yv) - ll(df.p_us.values, yv)
            dd = ll(p, yv) - ll(df.p_mkt.values, yv)
            lo, hi = boot_ci(delta[amask])
            by = " ".join(
                f"{s[-5:]}:{delta[amask & (df.season == s).values].mean():+.4f}"
                for s in sorted(df.season.unique()))
            pockets = " ".join(
                f"{cal}:{delta[amask & (act.reindex(df.index).cal == cal).fillna(False).values].mean():+.4f}"
                for cal in ["Feb", "Mar", "Apr"])
            capr = np.mean([cf[D] for D in uniq_dates
                            if pd.Timestamp(D).month == 4])
            print(f"  {nm:34s} d/gm {dd[amask].mean():+.4f}  vs shipped "
                  f"{delta[amask].mean():+.5f} CI({lo:+.5f},{hi:+.5f}) "
                  f"nats {delta[amask].sum():+5.1f}  by-season {by}  "
                  f"pockets {pockets}  mean c(Apr) {capr:+.3f}")


if __name__ == "__main__":
    main()
