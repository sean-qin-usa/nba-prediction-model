#!/usr/bin/env python3
"""NATIONAL-TV STAR-DNP MECHANISM — MEASUREMENT ONLY (no endpoint touched).

The claim under test (D135 item 1): the NBA Player Participation Policy, in
force from 2023-24 (era E5), PROHIBITS resting healthy star players in
nationally televised games, so the star-DNP hazard should be MECHANICALLY
SUPPRESSED on the ~23% of the slate that is national.

Confound the measurement must beat: national-TV games are SELECTED for good
teams and marquee matchups, whose stars are healthier and rested differently.
Every headline number below is therefore WITHIN-PLAYER-SEASON (player-season
fixed effect absorbed), which compares the same player against himself in the
same season, and additionally controls rest / b2b / home / calendar.

Identification check: the effect must APPEAR at the 2023-24 boundary. 2022-23
is the only PRE-policy season for which the flag exists in our cache, so this
is a 1-vs-3 difference-in-differences, reported with its own CI.

Outputs data/ad_natltv_mechanism.json. DB read_only=True. No production file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from nbapred.db import connect

SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")
PRE = ("2022-23",)
POST = ("2023-24", "2024-25", "2025-26")
NBOOT = 2000
SEED = 20260801


# --------------------------------------------------------------- inference
def cluster_boot(y, x, groups, stat, B=NBOOT, seed=SEED):
    """Non-parametric cluster bootstrap: resample CLUSTERS with replacement and
    recompute `stat(y_b, x_b)`.  Returns (point, lo, hi, se)."""
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_by = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    pt = stat(y, x)
    out = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx_by[i] for i in pick])
        out[b] = stat(y[sel], x[sel])
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(pt), float(lo), float(hi), float(out.std(ddof=1))


def mean_diff(y, x):
    a = y[x == 1]
    b = y[x == 0]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(a.mean() - b.mean())


def ols_beta(Y, X):
    """beta on column 0 of X (X already includes an intercept as last col)."""
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return float(b[0])


def cluster_boot_ols(Y, X, groups, B=NBOOT, seed=SEED):
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_by = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    pt = ols_beta(Y, X)
    out = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx_by[i] for i in pick])
        try:
            out[b] = ols_beta(Y[sel], X[sel])
        except np.linalg.LinAlgError:
            out[b] = np.nan
    out = out[~np.isnan(out)]
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(pt), float(lo), float(hi), float(out.std(ddof=1))


def sig(lo, hi):
    return "SIG" if (lo > 0 or hi < 0) else "ns"


def demean(v, key):
    """Within-group demeaning (absorbs the group fixed effect)."""
    s = pd.Series(v)
    return (s - s.groupby(key).transform("mean")).to_numpy()


# ------------------------------------------------------------------- corpus
def build(con, tv):
    games = con.execute("""
        SELECT game_id, season, game_date, team_id, is_home,
               pts
        FROM nba_games WHERE game_id LIKE '002%'
    """).fetchdf()
    games["game_id"] = games["game_id"].astype(str)
    games = games[games.season.isin(SEASONS)].copy()

    # opponent
    opp = games[["game_id", "team_id", "pts"]].rename(
        columns={"team_id": "opp_id", "pts": "opp_pts"})
    games = games.merge(opp, on="game_id")
    games = games[games.team_id != games.opp_id].copy()

    # rest / b2b from the team's own schedule
    games = games.sort_values(["season", "team_id", "game_date"])
    games["prev"] = games.groupby(["season", "team_id"])["game_date"].shift(1)
    games["days_rest"] = (pd.to_datetime(games.game_date)
                          - pd.to_datetime(games.prev)).dt.days
    games["is_b2b"] = (games.days_rest == 1).astype(float)
    games["days_rest"] = games["days_rest"].fillna(7).clip(upper=7)
    games["tgp"] = games.groupby(["season", "team_id"]).cumcount()

    # core cohort, era_measure.py:112-130 definition VERBATIM
    core = con.execute("""
        WITH ps AS (
          SELECT g.season, p.player_id, p.team_id, count(*) gp, avg(p.seconds) sec
          FROM player_game_stats p
          JOIN (SELECT DISTINCT game_id, season FROM nba_games
                WHERE game_id LIKE '002%') g USING (game_id)
          WHERE p.seconds > 0 GROUP BY 1,2,3
        ), agg AS (
          SELECT season, player_id, sum(gp) gp, sum(gp*sec)/sum(gp) sec,
                 arg_max(team_id, gp) team_id FROM ps GROUP BY 1,2
        )
        SELECT season, player_id, team_id, gp, sec FROM agg
        WHERE gp >= 20 AND sec >= 1680
    """).fetchdf()
    core = core[core.season.isin(SEASONS)]

    played = con.execute("""
        SELECT p.game_id, p.team_id, p.player_id
        FROM player_game_stats p
        WHERE p.game_id LIKE '002%'
    """).fetchdf()
    played["game_id"] = played["game_id"].astype(str)
    played["_played"] = 1

    # cross core x team schedule
    panel = core[["season", "player_id", "team_id"]].merge(
        games[["game_id", "season", "team_id", "game_date", "is_home",
               "days_rest", "is_b2b", "tgp", "opp_id"]],
        on=["season", "team_id"], how="inner")
    panel = panel.merge(played, on=["game_id", "team_id", "player_id"], how="left")
    panel["dnp"] = panel["_played"].isna().astype(float)
    panel = panel.drop(columns=["_played"])

    panel = panel.merge(tv[["game_id", "is_natl_tv", "natl_tv"]],
                        on="game_id", how="left")
    panel = panel[panel.is_natl_tv.notna()].copy()
    panel["is_natl_tv"] = panel["is_natl_tv"].astype(float)

    # opponent strength (season win pct of the opponent) -- the marquee confound
    wl = con.execute("""
        SELECT season, team_id, avg(CASE WHEN wl='W' THEN 1.0 ELSE 0.0 END) wpct
        FROM nba_games WHERE game_id LIKE '002%' GROUP BY 1,2
    """).fetchdf()
    panel = panel.merge(wl.rename(columns={"team_id": "opp_id", "wpct": "opp_wpct"}),
                        on=["season", "opp_id"], how="left")
    panel = panel.merge(wl.rename(columns={"wpct": "own_wpct"}),
                        on=["season", "team_id"], how="left")
    panel["month"] = pd.to_datetime(panel.game_date).dt.month
    panel["ps"] = panel.player_id.astype(str) + "_" + panel.season
    panel["post"] = panel.season.isin(POST).astype(float)
    return panel, games


def rest_flag(con, panel):
    """PPP's own object: the 'Rest' DNP.  injury_reports_pit starts 2023-10 so
    this is an E5+E6-ONLY secondary (ERAS.md §5 availability trap, stated)."""
    inj = con.execute("""
        SELECT game_date, team, player, status, reason FROM injury_reports_pit
        WHERE status='Out'
    """).fetchdf()
    inj["is_rest"] = inj.reason.str.strip().str.lower().eq("rest").astype(int)
    return inj


def report(panel, label, out, cluster="player_id"):
    y = panel.dnp.to_numpy(float)
    x = panel.is_natl_tv.to_numpy(float)
    g = panel[cluster].to_numpy()
    blk = {"n": int(len(panel)),
           "n_players": int(panel.player_id.nunique()),
           "n_natl_rows": int(x.sum()),
           "rate_natl": float(y[x == 1].mean()) if (x == 1).any() else None,
           "rate_local": float(y[x == 0].mean()) if (x == 0).any() else None}

    # E1 raw
    pt, lo, hi, se = cluster_boot(y, x, g, mean_diff)
    blk["E1_raw"] = dict(delta=pt, lo=lo, hi=hi, se=se, sig=sig(lo, hi))

    # E2 within player-season (FE absorbed)
    key = panel.ps.to_numpy()
    yd = demean(y, key)
    xd = demean(x, key)
    X = np.column_stack([xd, np.ones(len(xd))])
    pt, lo, hi, se = cluster_boot_ols(yd, X, g)
    blk["E2_playerseason_FE"] = dict(delta=pt, lo=lo, hi=hi, se=se, sig=sig(lo, hi))

    # E3 + rest/home/calendar/opponent controls, still FE-absorbed
    ctrl = np.column_stack([
        panel.is_b2b.to_numpy(float),
        panel.days_rest.to_numpy(float),
        panel.is_home.astype(float).to_numpy(),
        panel.tgp.to_numpy(float),
        panel.opp_wpct.fillna(0.5).to_numpy(float),
    ])
    ctrld = np.column_stack([demean(ctrl[:, j], key) for j in range(ctrl.shape[1])])
    X3 = np.column_stack([xd, ctrld, np.ones(len(xd))])
    pt, lo, hi, se = cluster_boot_ols(yd, X3, g)
    blk["E3_FE_plus_controls"] = dict(delta=pt, lo=lo, hi=hi, se=se, sig=sig(lo, hi))

    out[label] = blk
    print(f"\n--- {label}  n={blk['n']} ({blk['n_players']} players, "
          f"{blk['n_natl_rows']} natl rows) ---")
    print(f"  DNP rate  natl {blk['rate_natl']:.4f}  local {blk['rate_local']:.4f}")
    for k in ("E1_raw", "E2_playerseason_FE", "E3_FE_plus_controls"):
        v = blk[k]
        print(f"  {k:24s} {v['delta']:+.5f} CI[{v['lo']:+.5f},{v['hi']:+.5f}]"
              f" se {v['se']:.5f} {v['sig']}")
    return blk


def main():
    tv = pd.read_csv(ROOT / "data" / "ad_natl_tv.csv", dtype={"game_id": str})
    con = connect(read_only=True)
    panel, games = build(con, tv)
    print(f"panel {len(panel)} core player-games, "
          f"{panel.game_id.nunique()} games, {panel.player_id.nunique()} players")

    out = {"seasons": list(SEASONS), "nboot": NBOOT, "seed": SEED,
           "core_def": "gp>=20 & mean_sec>=1680 (era_measure.py:112-130)"}

    # ------------------------------------------------- confound descriptives
    gtv = games.merge(tv[["game_id", "is_natl_tv"]], on="game_id", how="inner")
    wl = con.execute("""
        SELECT season, team_id, avg(CASE WHEN wl='W' THEN 1.0 ELSE 0.0 END) wpct
        FROM nba_games WHERE game_id LIKE '002%' GROUP BY 1,2
    """).fetchdf()
    gtv = gtv.merge(wl, on=["season", "team_id"], how="left")
    conf = gtv.groupby("is_natl_tv").agg(
        team_wpct=("wpct", "mean"), b2b=("is_b2b", "mean"),
        rest=("days_rest", "mean"), n=("game_id", "size")).round(4)
    print("\nCONFOUND CHECK (team-game level):")
    print(conf.to_string())
    out["confound"] = conf.to_dict("index")

    # ------------------------------------------------------------ headline
    report(panel, "ALL 2022-26 (E4+E5+E6)", out)
    for s in SEASONS:
        report(panel[panel.season == s], f"season {s}", out)
    report(panel[panel.season.isin(PRE)], "PRE-PPP (E4, 2022-23)", out)
    report(panel[panel.season.isin(POST)], "POST-PPP (E5+E6)", out)

    # ------------------------------------------------- DiD at the E5 boundary
    y = panel.dnp.to_numpy(float)
    x = panel.is_natl_tv.to_numpy(float)
    post = panel.post.to_numpy(float)
    key = panel.ps.to_numpy()
    yd = demean(y, key)
    xd = demean(x, key)
    ixd = demean(x * post, key)
    postd = demean(post, key)   # collinear inside player-season; kept for safety
    X = np.column_stack([ixd, xd, postd, np.ones(len(y))])
    pt, lo, hi, se = cluster_boot_ols(yd, X, panel.player_id.to_numpy())
    out["DiD_natlTV_x_postPPP"] = dict(delta=pt, lo=lo, hi=hi, se=se, sig=sig(lo, hi))
    print(f"\nDiD natlTV x post-PPP (player-season FE): {pt:+.5f} "
          f"CI[{lo:+.5f},{hi:+.5f}] se {se:.5f} {sig(lo,hi)}")

    # season-cluster / season-mean t on the FE estimate
    per = {}
    for s in SEASONS:
        p = panel[panel.season == s]
        k = p.ps.to_numpy()
        b = ols_beta(demean(p.dnp.to_numpy(float), k),
                     np.column_stack([demean(p.is_natl_tv.to_numpy(float), k),
                                      np.ones(len(p))]))
        per[s] = b
    v = np.array(list(per.values()))
    m, sd, K = v.mean(), v.std(ddof=1), len(v)
    tcrit = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(K - 1, 2.0)
    out["season_mean_t"] = dict(per_season=per, mean=float(m),
                                sd=float(sd), K=K,
                                lo=float(m - tcrit * sd / np.sqrt(K)),
                                hi=float(m + tcrit * sd / np.sqrt(K)))
    print(f"\nper-season E2: " + "  ".join(f"{k} {x:+.5f}" for k, x in per.items()))
    print(f"season-mean t (K={K}, dof={K-1}): {m:+.5f} "
          f"CI[{out['season_mean_t']['lo']:+.5f},{out['season_mean_t']['hi']:+.5f}]")

    # ------------------------------------ SECONDARY: the 'Rest' DNP directly
    inj = rest_flag(con, panel)
    # name-match: injury reports carry "Last, First"; join on team+date+lastname
    nm = con.execute("SELECT player_id, full_name FROM nba_players").fetchdf()
    nm["last"] = nm.full_name.str.split().str[-1].str.lower()
    nm["first"] = nm.full_name.str.split().str[0].str.lower()
    inj["last"] = inj.player.str.split(",").str[0].str.strip().str.lower()
    inj["first"] = inj.player.str.split(",").str[-1].str.strip().str.lower()
    inj["game_date"] = pd.to_datetime(inj.game_date).dt.date
    key_rest = set()
    j = inj[inj.is_rest == 1].merge(nm, on=["last", "first"], how="inner")
    for r in j.itertuples():
        key_rest.add((r.player_id, r.game_date))
    p2 = panel[panel.season.isin(POST)].copy()
    p2["gd"] = pd.to_datetime(p2.game_date).dt.date
    p2["rest_dnp"] = [1.0 if (int(a), b) in key_rest else 0.0
                      for a, b in zip(p2.player_id, p2.gd)]
    print(f"\nREST-DNP rows matched: {int(p2.rest_dnp.sum())} of {len(p2)} "
          f"(E5+E6 only; injury_reports_pit starts 2023-10)")
    yr = p2.rest_dnp.to_numpy(float)
    xr = p2.is_natl_tv.to_numpy(float)
    kr = p2.ps.to_numpy()
    pt, lo, hi, se = cluster_boot(yr, xr, p2.player_id.to_numpy(), mean_diff)
    blk = dict(n=len(p2), n_rest=int(yr.sum()),
               rate_natl=float(yr[xr == 1].mean()), rate_local=float(yr[xr == 0].mean()),
               E1_raw=dict(delta=pt, lo=lo, hi=hi, se=se, sig=sig(lo, hi)))
    ptf, lof, hif, sef = cluster_boot_ols(
        demean(yr, kr), np.column_stack([demean(xr, kr), np.ones(len(yr))]),
        p2.player_id.to_numpy())
    blk["E2_playerseason_FE"] = dict(delta=ptf, lo=lof, hi=hif, se=sef, sig=sig(lof, hif))
    out["REST_DNP_E5E6"] = blk
    print(f"  rest-DNP  natl {blk['rate_natl']:.5f}  local {blk['rate_local']:.5f}")
    print(f"  E1_raw {pt:+.6f} CI[{lo:+.6f},{hi:+.6f}] {sig(lo,hi)}")
    print(f"  E2_FE  {ptf:+.6f} CI[{lof:+.6f},{hif:+.6f}] {sig(lof,hif)}")

    con.close()
    (ROOT / "data" / "ad_natltv_mechanism.json").write_text(
        json.dumps(out, indent=2, default=float))
    panel.to_csv(ROOT / "data" / "ad_natltv_panel.csv.gz", index=False,
                 compression="gzip")
    print("\nAD_NATLTV_MECH_DONE")


if __name__ == "__main__":
    main()
