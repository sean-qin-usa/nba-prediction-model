"""REGIME D synthesis: JOINT WALK-FORWARD LATE-STATE LAYER.

Combines the three surviving margin terms found by the regime-D decomposition
(rw_regimeD_lateres/tankk/pockets/outsterm) into one honestly-fit late layer:

    m' = m_base + [k*tsd + c_f*fdiff + c_o*outdiff]   (active window only)

Fit daily, production idiom (fit_schedule_layer / tanking.py style): OLS
y=home_margin, X=[1, tsd, fdiff, outdiff, wdiff] on ALL completed active
(either gp>=55) games strictly before D, 2022-23 burn-in (outdiff=0 there —
conservative), shrink n/(n+n0), sign guards k<=0, c_f>=0, c_o<=0.

Arms: L2 outs-only, L3 form+outs, L4 k+form+outs (full), each n0 in {600,150}.
Deltas paired vs shipped p_us on the active window; per-season; Feb/Mar/Apr
pockets; final d/gm vs market (parity check).

PIT strict; oracle-tier outs (same tier as the headline capstone).
DuckDB read_only. Market close benchmark only.
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
    df["outdiff"] = (df.n_out_home - df.n_out_away).astype(float)

    ts = pd.read_csv(TANKSTATS, dtype={"game_id": str})
    ts["game_id"] = ts.game_id.str.zfill(10)
    side = ts[["season", "game_id", "team_abbrev", "gp_before", "tank_score"]]

    gh = tg[tg.is_home][["season", "game_id", "game_date", "team_abbrev",
                         "pts"]].rename(columns={"team_abbrev": "home",
                                                 "pts": "pts_h"})
    ga = tg[~tg.is_home][["game_id", "team_abbrev", "pts"]].rename(
        columns={"team_abbrev": "away", "pts": "pts_a"})
    games = gh.merge(ga, on="game_id")
    games["margin_home"] = games.pts_h - games.pts_a

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

    def wpct_before(season, team, D):
        dates, wins = hist[(season, team)]
        i = int(np.searchsorted(dates, D))
        return (wins[i - 1] / i) if i > 0 else 0.5

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
    allg = allg.merge(df[["game_id", "outdiff"]], on="game_id", how="left")
    allg["outdiff"] = allg.outdiff.fillna(0.0)   # 2022-23 burn-in: no outs
    allg = allg.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    allg["wdiff"] = [wpct_before(r.season, r.home, np.datetime64(r.game_date))
                     - wpct_before(r.season, r.away,
                                   np.datetime64(r.game_date))
                     for r in allg.itertuples()]

    aidx = np.where(allg.tsd.values != 0)[0]
    dts = allg.game_date.values
    y_m = allg.margin_home.values
    X_all = np.column_stack([np.ones(len(allg)), allg.tsd.values,
                             allg.fdiff.values, allg.outdiff.values,
                             allg.wdiff.values])
    uniq_dates = np.unique(df.game_date.values)

    def fit_paths(n0):
        out = {}
        for D in uniq_dates:
            m = aidx[dts[aidx] < D]
            if len(m) < K_MIN_ACTIVE:
                out[D] = (0.0, 0.0, 0.0)
                continue
            beta, *_ = np.linalg.lstsq(X_all[m], y_m[m], rcond=None)
            sh = len(m) / (len(m) + n0)
            out[D] = (float(min(0.0, beta[1] * sh)),
                      float(max(0.0, beta[2] * sh)),
                      float(min(0.0, beta[3] * sh)))
        return out

    amask = (df.tsd != 0).values
    yv = df.y.values
    act_cal = np.where(df.month.isin([1, 2]), "Feb",
                       np.where(df.month == 3, "Mar", "Apr"))
    print(f"shipped active d/gm {df[amask].d.mean():+.4f} "
          f"(n={int(amask.sum())}, nats {df[amask].d.sum():+.1f})")
    for n0 in (600, 150):
        paths = fit_paths(n0)
        kk = np.array([paths[D][0] for D in df.game_date.values])
        cf = np.array([paths[D][1] for D in df.game_date.values])
        co = np.array([paths[D][2] for D in df.game_date.values])
        arms = {
            f"L2 outs-only (n0={n0})":
                df.m_us.values + co * df.outdiff.values,
            f"L3 form+outs (n0={n0})":
                df.m_us.values + cf * df.fdiff.values + co * df.outdiff.values,
            f"L4 k+form+outs (n0={n0})":
                df.m_base.values + kk * df.tsd.values + cf * df.fdiff.values
                + co * df.outdiff.values,
        }
        for nm, marg in arms.items():
            p = sigmoid(np.where(amask, marg, df.m_us.values) / SCALE)
            delta = ll(p, yv) - ll(df.p_us.values, yv)
            dd = ll(p, yv) - ll(df.p_mkt.values, yv)
            lo, hi = boot_ci(delta[amask])
            by = " ".join(
                f"{s[-5:]}:{delta[amask & (df.season == s).values].mean():+.4f}"
                for s in sorted(df.season.unique()))
            pk = " ".join(
                f"{cal}:{delta[amask & (act_cal == cal)].mean():+.4f}"
                for cal in ["Feb", "Mar", "Apr"])
            print(f"{nm:26s} active-d/gm {dd[amask].mean():+.4f}  vs shipped "
                  f"{delta[amask].mean():+.5f} CI({lo:+.5f},{hi:+.5f}) "
                  f"nats {delta[amask].sum():+5.1f}  {by}  pockets {pk}")
        capr = np.mean([paths[D] for D in uniq_dates
                        if pd.Timestamp(D).month == 4], axis=0)
        print(f"  mean coefs April (n0={n0}): k={capr[0]:+.2f} "
              f"c_form={capr[1]:+.3f} c_out={capr[2]:+.3f}")


if __name__ == "__main__":
    main()
