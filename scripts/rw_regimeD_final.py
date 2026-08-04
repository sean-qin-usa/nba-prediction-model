"""REGIME D follow-up 3 (final probes): April k-inflation, deadline-churn term,
April market-heavy contributor eyeball.

  R1  Two-k hindsight test: does April want a bigger |k| than Feb-Mar?
      (k_FebMar, k_Apr) vs single k; also k scaled by season-progress.
  R2  Deadline-churn term: PIT churn proxy per team-date = share of last-5-game
      minutes played by players with <=10 games for the team this season
      ("new-face integration share"). Term m += c*(churn_a - churn_h), active
      deadline..deadline+35d only. Hindsight c; walk-forward honesty note.
  R3  April mkt-heavy (top-tercile |p_mkt-.5|) worst contributors listed for
      mechanism eyeballing (teams, tsd, outs, elim flags).

PIT strict; DuckDB read_only; market data benchmark/subset-definition only.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

DB = "/hdd/steveqin/sean_dev/nba_model/data/nba.duckdb"
CSV = "/hdd/steveqin/sean_dev/nba_model/data/capstone_pergame_tank.csv"
SCALE = 7.2
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
    idx = RNG.integers(0, len(x), (B, len(x)))
    return tuple(np.percentile(x[idx].mean(axis=1), [2.5, 97.5]))


def main():
    con = duckdb.connect(DB, read_only=True)
    tg = con.execute("""SELECT season, game_id, game_date, team_abbrev,
        matchup, pts FROM nba_games WHERE game_id LIKE '002%'
          AND season IN ('2023-24','2024-25','2025-26')""").df()
    pg = con.execute("""SELECT game_id, team_id, player_id, seconds
        FROM player_game_stats WHERE game_id LIKE '002%'
          AND seconds IS NOT NULL""").df()
    tid = con.execute("""SELECT DISTINCT season, game_id, game_date, team_id,
        team_abbrev FROM nba_games WHERE game_id LIKE '002%'
          AND season IN ('2023-24','2024-25','2025-26')""").df()
    con.close()
    tg["game_date"] = pd.to_datetime(tg.game_date)
    tid["game_date"] = pd.to_datetime(tid.game_date)

    df = pd.read_csv(CSV, dtype={"game_id": str})
    df["game_id"] = df.game_id.str.zfill(10)
    df["game_date"] = pd.to_datetime(df.game_date)
    df["month"] = df.game_date.dt.month
    df["m_us"] = SCALE * logit(df.p_us)
    df["m_base"] = df.m_us - df.k * df.tsd
    df["d"] = ll(df.p_us.values, df.y.values) - ll(df.p_mkt.values, df.y.values)
    tg["is_home"] = np.where(
        tg.matchup.str.contains(" @ "),
        tg.matchup.str.split(" @ ").str[1] == tg.team_abbrev,
        tg.matchup.str.split(" vs. ").str[0] == tg.team_abbrev)
    gh = tg[tg.is_home][["game_id", "pts"]].rename(columns={"pts": "pts_h"})
    ga = tg[~tg.is_home][["game_id", "pts"]].rename(columns={"pts": "pts_a"})
    df = df.merge(gh, on="game_id").merge(ga, on="game_id")
    df["margin_home"] = df.pts_h - df.pts_a
    amask = (df.tsd != 0).values
    yv = df.y.values

    # ---------------- R1 two-k hindsight --------------------------------- #
    print("=" * 100)
    print("R1 APRIL K-INFLATION (hindsight, active games)")
    apr = (df.month == 4).values & amask
    febmar = amask & ~apr

    def mle_1k(mask):
        ks = np.linspace(-9, 1, 401)
        L = [ll(sigmoid((df.m_base.values + k * df.tsd.values) / SCALE),
                yv)[mask].sum() for k in ks]
        return ks[int(np.argmin(L))]

    k1 = mle_1k(amask)
    kfm, kap = mle_1k(febmar), mle_1k(apr)
    for nm, marg in [
            ("single k*", df.m_base.values + k1 * df.tsd.values),
            ("two-k (FebMar/Apr)", df.m_base.values + np.where(
                apr, kap, kfm) * df.tsd.values)]:
        p = sigmoid(np.where(amask, marg, df.m_us.values) / SCALE)
        delta = ll(p, yv) - ll(df.p_us.values, yv)
        lo, hi = boot_ci(delta[amask])
        print(f"  {nm:22s} k={k1 if 'single' in nm else (kfm, kap)}  vs "
              f"shipped {delta[amask].mean():+.5f} CI({lo:+.5f},{hi:+.5f})  "
              f"April-only {delta[apr].mean():+.5f}")

    # ---------------- R2 deadline-churn term ----------------------------- #
    print("=" * 100)
    print("R2 DEADLINE-CHURN TERM (new-face integration share)")
    pg = pg.merge(tid, on=["game_id", "team_id"], how="inner")
    pg = pg.sort_values("game_date")
    pg["gp_for_team"] = pg.groupby(
        ["season", "team_id", "player_id"]).cumcount()  # games BEFORE this one
    pg["is_new"] = pg.gp_for_team <= 10
    gsec = pg.groupby(["season", "game_id", "game_date", "team_abbrev"]).agg(
        sec_tot=("seconds", "sum"),
        sec_new=("seconds", lambda s: 0)).reset_index()
    newsec = pg[pg.is_new].groupby(
        ["season", "game_id", "game_date", "team_abbrev"]).seconds.sum()
    gsec = gsec.set_index(["season", "game_id", "game_date", "team_abbrev"])
    gsec["sec_new"] = newsec
    gsec = gsec.fillna(0.0).reset_index()
    gsec["new_share"] = gsec.sec_new / gsec.sec_tot
    gsec = gsec.sort_values("game_date")
    gsec["churn5"] = (gsec.groupby(["season", "team_abbrev"]).new_share
                      .transform(lambda s: s.shift(1)
                                 .rolling(5, min_periods=3).mean()))
    ck = gsec[["season", "game_date", "team_abbrev", "churn5"]]
    df = df.merge(ck.rename(columns={"team_abbrev": "home",
                                     "churn5": "ch_h"}),
                  on=["season", "game_date", "home"], how="left")
    df = df.merge(ck.rename(columns={"team_abbrev": "away",
                                     "churn5": "ch_a"}),
                  on=["season", "game_date", "away"], how="left")
    df["dl_days"] = [(r.game_date - pd.Timestamp(DEADLINES[r.season])).days
                     for r in df.itertuples()]
    dlwin = (df.dl_days >= 0) & (df.dl_days < 35)
    x = (df.ch_a.fillna(0) - df.ch_h.fillna(0)).values  # + => away churned
    print(f"  churn5 in dl window: mean {df.ch_h[dlwin & pd.Series(amask)].mean():.3f} "
          f"p90 {df.ch_h[dlwin & pd.Series(amask)].quantile(0.9):.3f}")
    # does the churned side underperform our margin in the dl window?
    g = df[dlwin & amask]
    chd = (g.ch_h.fillna(0) - g.ch_a.fillna(0))
    rp = g.margin_home - g.m_us
    X = np.column_stack([np.ones(len(g)) if False else np.ones(0)])
    b = np.polyfit(chd, rp, 1)[0] if len(g) > 10 else np.nan
    print(f"  dl-window active n={len(g)}  resid_post ~ churn_diff slope "
          f"{b:+.2f} pts per unit share")
    cs = np.linspace(-30, 30, 241)
    best, bc = 1e18, 0.0
    act_dl = amask & dlwin.values
    for c in cs:
        p = sigmoid((df.m_us.values + np.where(act_dl, c * x, 0.0)) / SCALE)
        L = ll(p, yv)[act_dl].sum()
        if L < best:
            best, bc = L, c
    p = sigmoid((df.m_us.values + np.where(act_dl, bc * x, 0.0)) / SCALE)
    delta = ll(p, yv) - ll(df.p_us.values, yv)
    lo, hi = boot_ci(delta[act_dl])
    by = " ".join(f"{s[-5:]}:{delta[act_dl & (df.season == s).values].mean():+.4f}"
                  for s in sorted(df.season.unique()))
    print(f"  hindsight c*={bc:+.1f}  dl-window active vs shipped "
          f"{delta[act_dl].mean():+.5f} CI({lo:+.5f},{hi:+.5f}) "
          f"nats {delta[act_dl].sum():+.1f}  by-season {by}")

    # ---------------- R3 April mkt-heavy eyeball ------------------------- #
    print("=" * 100)
    print("R3 APRIL MKT-HEAVY worst 20 contributors (by d)")
    ap = df[(df.month == 4) & (df.tsd != 0)].copy()
    thr = (ap.p_mkt - 0.5).abs().quantile(2 / 3)
    aph = ap[(ap.p_mkt - 0.5).abs() >= thr].sort_values("d", ascending=False)
    cols = ["season", "game_date", "home", "away", "y", "p_us", "p_mkt",
            "tsd", "n_out_home", "n_out_away", "d", "pts_h", "pts_a"]
    aph["game_date"] = aph.game_date.dt.date
    print(aph[cols].head(20).to_string(index=False,
                                       float_format=lambda v: f"{v:+.3f}"))
    print(f"  mkt-heavy April: n={len(aph)}  d/gm {aph.d.mean():+.4f}  "
          f"we-too-heavy share "
          f"{((aph.p_us - .5).abs() >= thr).mean():.2f}")
    lost = aph[aph.d > 0.05]
    print(f"  games we lost badly (d>.05): n={len(lost)}, of which market "
          f"favored winner & we underfavored: "
          f"{(np.round(lost.p_mkt) == lost.y).mean():.2f}")


if __name__ == "__main__":
    main()
