#!/usr/bin/env python3
"""APRIL-COLLAPSE (D65) TANK-BEHAVIOR PROGRAM — build + validate + gate.

D65 localized the heavy-favorite hole to a late-season collapse cluster
(market prices CURRENT-STATE capitulation — dead/tanking dogs, mass
non-star outs, recently-crushed — while we price season aggregates).
This script builds PIT-predictable "will they try / who will play"
team-date stats and tests whether they (1) predict next-game margin
underperformance vs our model's expectation, (2) predict D65-cluster
membership, and (3) pass a pre-registered walk-forward margin gate.

STATS (per team-date, everything strictly < game date):
  a. VET-MINUTES SHARE SHIFT: season-baseline share of team minutes to
     age>=27 players (darko_history age, asof) MINUS last-5 share.
     Higher = vets recently benched = tanking.
  b. ROTATION EXPERIMENTATION: b1 = distinct proxy-starters used in the
     last 10 games; b2 = season-first proxy-starts in the last 5 games.
     Proxy-starter = top-5 by seconds in a game (player_game_stats has
     no starter flag; proxy validated vs lineup_stints stint 0:
     mean overlap 4.15/5, 83% of team-games >=4/5 — noise averages
     over the 10-game window). Higher = churn.
  c. SHUTDOWN LISTINGS: distinct (player, game_date) injury-report
     'Out' entries in the trailing 14 days with rest / injury-
     management / maintenance reasons, on players with season-to-date
     mpg >= 25 (report_date strictly < date). NaN (-> neutral z) when
     the report feed has no coverage in the trailing 7 days — the PIT
     feed ends 2025-12-21, so c is DEAD for late 2025-26 (incl. the
     April-2026 target window); a/b/d carry that season.
  d. STANDINGS INCENTIVE: games-back from the conference play-in (10th
     seed), lottery-lock flag (GB > games remaining), seed-locked flag
     (clinch approx: GB gap to the neighbors above AND below both >
     games remaining).

STANDARDIZATION (PIT): expanding z over ALL team-dates strictly before
the current date, pooled across teams/seasons, 2022-23 included as
burn-in; z=0 until 300 prior obs; z clipped to +-4; NaN raw -> z=0.
tank_score = mean(z_a, (z_b1+z_b2)/2, z_c, (z_d1+z_d2+z_d3)/3).

PRE-REGISTERED GATE (ONE config, no sweeps):
  margin_adj = margin + k * tank_score_diff,
  tank_score_diff = act(home) - act(away), act(t) = tank_score if the
  team's gp >= 55 else 0 (term ACTIVE only late-season).
  k fit walk-forward by 1-D MLE (Newton) on this run's accumulated
  (margin, tsd, y) history strictly before the refit date, pooled
  across seasons in season order; k=0 until >=20 active rows; k
  clipped +-15; refit at the production weekly-refit cadence.
  Full 3-season capstone (loop copied from prod_by_season.py, same-run
  control p_ctrl = sigmoid(margin/SCALE)), paired bootstrap 2000x
  seed 7 on per-game logloss deltas (ctrl - treat; + = improvement),
  reported pooled + per-season + late-season (Mar/Apr) + April +
  D65-intersection (|p_mkt-.5|>.35 & |p_ctrl-.5|<=.35) + active.

Usage: python scripts/apr_program.py [stats|gate|all]
Artifacts: data/apr_tank_stats.parquet, data/apr_capstone_pergame.csv,
           data/apr_results.json.  DB opened read_only=True; no edits
           to nbapred/ or existing scripts.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nbapred.db import connect                        # read_only used below
from nbapred.model.production import SCALE, sigmoid, fit_production
from nbapred.model.composition import CompositionModel

SEASONS_BURN = ["2022-23", "2023-24", "2024-25", "2025-26"]
SEASONS_GATE = ["2023-24", "2024-25", "2025-26"]
STATS_PARQUET = os.path.join(ROOT, "data", "apr_tank_stats.csv")
CAPSTONE_CSV = os.path.join(ROOT, "data", "apr_capstone_pergame.csv")
RESULTS_JSON = os.path.join(ROOT, "data", "apr_results.json")
CARRY_CSV = os.path.join(ROOT, "data", "capstone_pergame_carry.csv")

VET_AGE = 27.0
SHUTDOWN_MPG = 25.0
SHUTDOWN_WINDOW = 14          # days
Z_MIN_N = 300                 # prior obs before z activates
Z_CLIP = 4.0
GP_ACTIVE = 55                # gate: term active only when scored team gp>=55
K_MIN_ACTIVE = 20             # gate: k stays 0 until this many active rows
K_CLIP = 15.0
NBOOT, SEED = 2000, 7
HEAVY = 0.35                  # D65 slice definition
EPS = 1e-12

EAST = {"ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DET", "IND", "MIA",
        "MIL", "NYK", "ORL", "PHI", "TOR", "WAS"}
WEST = {"DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "MIN", "NOP",
        "OKC", "PHX", "POR", "SAC", "SAS", "UTA"}


def logloss_vec(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# --------------------------------------------------------------------------
# 1. STAT CONSTRUCTION
# --------------------------------------------------------------------------

def team_games(con) -> pd.DataFrame:
    tg = con.execute("""
        SELECT season, game_id, game_date, team_id, team_abbrev, pts, is_home
        FROM nba_games WHERE game_id LIKE '002%' AND wl IS NOT NULL
          AND season >= '2022-23'
        ORDER BY team_id, game_date, game_id""").fetchdf()
    tg["game_date"] = pd.to_datetime(tg["game_date"])
    tg["gp_before"] = tg.groupby(["season", "team_id"]).cumcount()
    return tg


def player_minutes(con) -> pd.DataFrame:
    pm = con.execute("""
        SELECT s.game_id, s.team_id, s.player_id, s.seconds/60.0 AS mins,
               g.game_date, g.season
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games
              WHERE season >= '2022-23') g USING (game_id)
        WHERE s.game_id LIKE '002%' AND s.seconds > 0""").fetchdf()
    pm["game_date"] = pd.to_datetime(pm["game_date"])
    return pm


def attach_age(con, pm: pd.DataFrame) -> pd.DataFrame:
    dk = con.execute("""
        SELECT player_id, date, age FROM darko_history
        WHERE date >= '2022-06-01' AND age IS NOT NULL
        ORDER BY date""").fetchdf()
    dk["date"] = pd.to_datetime(dk["date"])
    pm = pm.sort_values("game_date").reset_index(drop=True)
    pm = pd.merge_asof(pm, dk.rename(columns={"date": "game_date"}),
                       on="game_date", by="player_id", direction="backward")
    pm["is_vet"] = pm["age"] >= VET_AGE          # NaN age -> False (<27)
    return pm


def comp_a_vetshift(pm: pd.DataFrame) -> pd.DataFrame:
    """Per team-game: a_raw = season-baseline vet-minute share (strictly
    before) minus last-5 share.  Positive = vets recently benched."""
    g = (pm.groupby(["season", "team_id", "game_id", "game_date"])
           .apply(lambda s: s.loc[s.is_vet, "mins"].sum() / max(s.mins.sum(), 1e-9))
           .rename("vet_share").reset_index()
           .sort_values(["season", "team_id", "game_date", "game_id"]))
    grp = g.groupby(["season", "team_id"], sort=False)
    prev = grp["vet_share"].shift(1)
    base = (prev.groupby([g.season, g.team_id]).expanding(min_periods=8)
            .mean().reset_index(level=[0, 1], drop=True))
    last5 = (prev.groupby([g.season, g.team_id]).rolling(5, min_periods=5)
             .mean().reset_index(level=[0, 1], drop=True))
    g["a_raw"] = base - last5
    return g[["season", "team_id", "game_id", "a_raw"]]


def comp_b_rotation(pm: pd.DataFrame) -> pd.DataFrame:
    """b1 = distinct proxy-starters in last 10 games; b2 = season-first
    proxy-starts in last 5 games.  Proxy-starter = top-5 seconds."""
    pm = pm.sort_values(["game_id", "team_id", "mins", "player_id"],
                        ascending=[True, True, False, True])
    top5 = (pm.groupby(["season", "team_id", "game_date", "game_id"])
              .head(5)
              .groupby(["season", "team_id", "game_date", "game_id"])["player_id"]
              .apply(lambda s: frozenset(int(x) for x in s))
              .rename("starters").reset_index()
              .sort_values(["season", "team_id", "game_date", "game_id"]))
    rows = []
    for (season, tid), sub in top5.groupby(["season", "team_id"], sort=False):
        starters = sub["starters"].tolist()
        gids = sub["game_id"].tolist()
        first_idx: dict[int, int] = {}
        for i, st in enumerate(starters):
            for p in st:
                first_idx.setdefault(p, i)
        for i, gid in enumerate(gids):
            b1 = float(len(set().union(*starters[max(0, i - 10):i]))) \
                if i >= 10 else np.nan
            if i >= 5:
                b2 = float(sum(1 for j in range(i - 5, i)
                               for p in starters[j] if first_idx[p] == j))
            else:
                b2 = np.nan
            rows.append((season, tid, gid, b1, b2))
    return pd.DataFrame(rows, columns=["season", "team_id", "game_id",
                                       "b1_raw", "b2_raw"])


def comp_c_shutdown(con, tg: pd.DataFrame, pm: pd.DataFrame) -> pd.DataFrame:
    """Distinct (player, game_date) rest/management/maintenance 'Out'
    listings on mpg>=25 players in the trailing 14 days, report_date < d.
    NaN when the report feed has no coverage in the trailing 7 days."""
    from nbapred.teams import abbrev_for     # D171: "LA Clippers" resolves
    ent = con.execute("""
        SELECT DISTINCT i.report_date, i.game_date, i.team, p.player_id
        FROM injury_reports_pit i
        JOIN (SELECT player_id,
                     lower(first_name||' '||last_name) fn FROM nba_players) p
          ON p.fn = lower(trim(split_part(i.player,',',2))||' '||
                          trim(split_part(i.player,',',1)))
        WHERE i.status = 'Out'
          AND (lower(i.reason) LIKE '%rest%'
               OR lower(i.reason) LIKE '%management%'
               OR lower(i.reason) LIKE '%maintenance%')""").fetchdf()
    ent["report_date"] = pd.to_datetime(ent["report_date"])
    ent["game_date"] = pd.to_datetime(ent["game_date"])
    ent["team_ab"] = ent["team"].map(abbrev_for)     # D171
    ent = ent.dropna(subset=["team_ab", "player_id"])
    by_team: dict[str, pd.DataFrame] = {
        ab: sub.sort_values("game_date") for ab, sub in ent.groupby("team_ab")}
    report_days = np.sort(pd.to_datetime(con.execute(
        "SELECT DISTINCT report_date FROM injury_reports_pit").fetchdf()
        ["report_date"]).values)

    # per (player, season): cumulative minutes history for mpg-before-d
    pmin: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for (pid, season), sub in pm.groupby(["player_id", "season"], sort=False):
        sub = sub.sort_values("game_date")
        pmin[(int(pid), season)] = (sub.game_date.values,
                                    np.cumsum(sub.mins.values))

    def mpg_before(pid, season, d):
        h = pmin.get((int(pid), season))
        if h is None:
            return 0.0
        i = np.searchsorted(h[0], np.datetime64(d))
        return (h[1][i - 1] / i) if i > 0 else 0.0

    out = []
    for r in tg.itertuples():
        d = r.game_date
        i = np.searchsorted(report_days, np.datetime64(d))
        covered = i > 0 and (d - pd.Timestamp(report_days[i - 1])).days <= 7
        if not covered:
            out.append((r.season, r.team_id, r.game_id, np.nan))
            continue
        sub = by_team.get(r.team_abbrev)
        c = 0
        if sub is not None:
            w = sub[(sub.game_date >= d - pd.Timedelta(days=SHUTDOWN_WINDOW))
                    & (sub.game_date < d) & (sub.report_date < d)]
            seen = set()
            for e in w.itertuples():
                key = (int(e.player_id), e.game_date)
                if key in seen:
                    continue
                seen.add(key)
                if mpg_before(e.player_id, r.season, d) >= SHUTDOWN_MPG:
                    c += 1
        out.append((r.season, r.team_id, r.game_id, float(c)))
    return pd.DataFrame(out, columns=["season", "team_id", "game_id", "c_raw"])


def comp_d_standings(tg: pd.DataFrame) -> pd.DataFrame:
    """Play-in games-back, lottery-lock, seed-locked per team-date."""
    out = []
    for season, sub in tg.groupby("season", sort=False):
        teams = sub[["team_id", "team_abbrev"]].drop_duplicates()
        ab = dict(zip(teams.team_id, teams.team_abbrev))
        res = sub.copy()
        # win flag needs opponent pts
        opp = sub[["game_id", "team_id", "pts"]].rename(
            columns={"team_id": "opp_id", "pts": "opp_pts"})
        res = res.merge(opp, on="game_id")
        res = res[res.team_id != res.opp_id]
        res["win"] = (res.pts > res.opp_pts).astype(int)
        res = res.sort_values(["game_date", "game_id"])
        dates = np.sort(res.game_date.unique())
        W = {t: 0 for t in ab}
        L = {t: 0 for t in ab}
        state_by_date = {}
        for d in dates:
            state_by_date[d] = (dict(W), dict(L))
            day = res[res.game_date == d]
            for r in day.itertuples():
                if r.win:
                    W[r.team_id] += 1
                else:
                    L[r.team_id] += 1

        def wpct(w, l):
            return w / (w + l) if (w + l) else 0.5

        for d in dates:
            Wd, Ld = state_by_date[d]
            for conf in (EAST, WEST):
                ids = [t for t in ab if ab[t] in conf]
                order = sorted(ids, key=lambda t: (-wpct(Wd[t], Ld[t]),
                                                   -Wd[t], ab[t]))
                rank = {t: i for i, t in enumerate(order)}
                t10 = order[9]
                for t in ids:
                    gp = Wd[t] + Ld[t]
                    rem = 82 - gp
                    gb10 = ((Wd[t10] - Wd[t]) + (Ld[t] - Ld[t10])) / 2.0
                    d1 = float(np.clip(max(gb10, 0.0), 0, 20))
                    d2 = float(max(gb10, 0.0) > rem)
                    i = rank[t]

                    def gap(u, v):   # GB distance between u (better) and v
                        return ((Wd[u] - Wd[v]) + (Ld[v] - Ld[u])) / 2.0
                    lock_below = i == len(order) - 1 or \
                        gap(t, order[i + 1]) > rem
                    lock_above = i == 0 or gap(order[i - 1], t) > rem
                    d3 = float(lock_below and lock_above)
                    out.append((season, t, d, d1, d2, d3))
    st = pd.DataFrame(out, columns=["season", "team_id", "game_date",
                                    "d1_raw", "d2_raw", "d3_raw"])
    return st


def expanding_z(df: pd.DataFrame, cols) -> pd.DataFrame:
    """PIT expanding z per metric: stats from rows with date < d, pooled
    across teams/seasons (2022-23 burn-in included)."""
    df = df.sort_values(["game_date", "game_id", "team_id"]).reset_index(drop=True)
    date_groups = df.groupby("game_date", sort=True).indices
    date_keys = sorted(date_groups.keys())
    for c in cols:
        x = df[c].values.astype(float)
        z = np.zeros(len(df))
        n, s, ss = 0, 0.0, 0.0
        for key in date_keys:
            idx = np.asarray(date_groups[key])
            if n >= Z_MIN_N:
                mu = s / n
                sd = math.sqrt(max(ss / n - mu * mu, 1e-12))
                sd = max(sd, 1e-6)
                for i in idx:
                    if not math.isnan(x[i]):
                        z[i] = float(np.clip((x[i] - mu) / sd, -Z_CLIP, Z_CLIP))
            for i in idx:
                if not math.isnan(x[i]):
                    n += 1
                    s += x[i]
                    ss += x[i] * x[i]
        df["z_" + c[0:2].rstrip("_")] = z
    return df


def build_stats() -> pd.DataFrame:
    con = connect(read_only=True)
    try:
        tg = team_games(con)
        pm = player_minutes(con)
        pm = attach_age(con, pm)
        print(f"[stats] team-games={len(tg)}  player-games={len(pm)}  "
              f"age coverage={pm.age.notna().mean():.3f}")
        a = comp_a_vetshift(pm)
        b = comp_b_rotation(pm)
        c = comp_c_shutdown(con, tg, pm)
        d = comp_d_standings(tg)
    finally:
        con.close()
    df = tg.merge(a, on=["season", "team_id", "game_id"], how="left") \
           .merge(b, on=["season", "team_id", "game_id"], how="left") \
           .merge(c, on=["season", "team_id", "game_id"], how="left") \
           .merge(d, on=["season", "team_id", "game_date"], how="left")
    df = expanding_z(df, ["a_raw", "b1_raw", "b2_raw", "c_raw",
                          "d1_raw", "d2_raw", "d3_raw"])
    df = df.rename(columns={"z_a_": "z_a", "z_b1": "z_b1", "z_b2": "z_b2",
                            "z_c_": "z_c", "z_d1": "z_d1", "z_d2": "z_d2",
                            "z_d3": "z_d3"})
    zc = {c: "z_" + c[:2].rstrip("_") for c in
          ["a_raw", "b1_raw", "b2_raw", "c_raw", "d1_raw", "d2_raw", "d3_raw"]}
    za, zb1, zb2 = df[zc["a_raw"]], df[zc["b1_raw"]], df[zc["b2_raw"]]
    zcc, zd1 = df[zc["c_raw"]], df[zc["d1_raw"]]
    zd2, zd3 = df[zc["d2_raw"]], df[zc["d3_raw"]]
    df["z_b"] = (zb1 + zb2) / 2.0
    df["z_d"] = (zd1 + zd2 + zd3) / 3.0
    df["tank_score"] = (za + df["z_b"] + zcc + df["z_d"]) / 4.0
    df.to_csv(STATS_PARQUET, index=False)
    print(f"[stats] wrote {STATS_PARQUET}  rows={len(df)}")
    return df


# --------------------------------------------------------------------------
# 2. DESCRIPTIVE VALIDATION
# --------------------------------------------------------------------------

def auc(score, label):
    """Rank AUC (ties averaged)."""
    score = np.asarray(score, float)
    label = np.asarray(label, bool)
    ok = ~np.isnan(score)
    score, label = score[ok], label[ok]
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return np.nan, 0, 0
    r = pd.Series(score).rank().values
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0), int(n1), int(n0)


def boot_corr(x, y, n=NBOOT, seed=SEED):
    from scipy.stats import pearsonr, spearmanr
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = np.asarray(x, float)[ok], np.asarray(y, float)[ok]
    if len(x) < 30:
        return dict(n=len(x))
    r = float(pearsonr(x, y)[0])
    rho = float(spearmanr(x, y)[0])
    rng = np.random.default_rng(seed)
    bs = np.empty(n)
    for b in range(n):
        i = rng.integers(0, len(x), len(x))
        sx = x[i]
        bs[b] = np.corrcoef(sx, y[i])[0, 1] if sx.std() > 0 else 0.0
    return dict(n=len(x), pearson=round(r, 4), rho=round(rho, 4),
                ci_lo=round(float(np.percentile(bs, 2.5)), 4),
                ci_hi=round(float(np.percentile(bs, 97.5)), 4))


def validate(df: pd.DataFrame) -> dict:
    res = {}
    cs = pd.read_csv(CARRY_CSV, dtype={"game_id": str})
    cs["game_date"] = pd.to_datetime(cs["game_date"])
    cs["exp_margin"] = SCALE * np.log(
        np.clip(cs.p_us, EPS, 1 - EPS) / np.clip(1 - cs.p_us, EPS, 1 - EPS))
    con = connect(read_only=True)
    try:
        hm = con.execute("""
            WITH g AS (SELECT game_id, team_abbrev, pts, is_home FROM nba_games
                       WHERE game_id LIKE '002%')
            SELECT h.game_id, h.pts - a.pts AS home_margin
            FROM g h JOIN g a ON h.game_id = a.game_id
                 AND h.team_abbrev <> a.team_abbrev
            WHERE h.is_home""").fetchdf()
    finally:
        con.close()
    cs = cs.merge(hm, on="game_id")
    cs["resid_home"] = cs.home_margin - cs.exp_margin
    cs["ll_us"] = logloss_vec(cs.p_us, cs.y)
    cs["ll_mkt"] = logloss_vec(cs.p_mkt, cs.y)
    cs["d_excess"] = cs.ll_us - cs.ll_mkt
    cs["month"] = cs.game_date.dt.month
    cs["late"] = cs.month.isin([3, 4])
    cs["april"] = cs.month == 4

    keep = ["game_id", "team_abbrev", "gp_before", "tank_score",
            "z_a", "z_b", "z_c", "z_d"]
    sub = df[keep]
    for side in ("home", "away"):
        m = sub.rename(columns={"team_abbrev": side}).rename(
            columns={c: f"{side[0]}_{c}" for c in keep[2:]})
        cs = cs.merge(m, on=["game_id", side], how="left")

    # (a) team-level: does tank_score predict next-game margin
    #     underperformance vs our model's expectation?
    rows = []
    for _, r in cs.iterrows():
        rows.append((r.season, r.month, r.h_gp_before, r.h_tank_score,
                     r.h_z_a, r.h_z_b, r.h_z_c, r.h_z_d, r.resid_home))
        rows.append((r.season, r.month, r.a_gp_before, r.a_tank_score,
                     r.a_z_a, r.a_z_b, r.a_z_c, r.a_z_d, -r.resid_home))
    tl = pd.DataFrame(rows, columns=["season", "month", "gp", "tank",
                                     "z_a", "z_b", "z_c", "z_d", "resid"])
    res["margin_resid"] = {}
    for name, m in [("all", np.ones(len(tl), bool)),
                    ("gp>=55", (tl.gp >= GP_ACTIVE).values),
                    ("gp>=55 & late", ((tl.gp >= GP_ACTIVE)
                                       & tl.month.isin([3, 4])).values),
                    ("april", (tl.month == 4).values)]:
        res["margin_resid"][name] = boot_corr(tl.tank.values[m],
                                              tl.resid.values[m])
    res["margin_resid_components_gp55"] = {
        c: boot_corr(tl[c].values[(tl.gp >= GP_ACTIVE).values],
                     tl.resid.values[(tl.gp >= GP_ACTIVE).values])
        for c in ["z_a", "z_b", "z_c", "z_d"]}
    for s, g in tl[tl.gp >= GP_ACTIVE].groupby("season"):
        res["margin_resid"][f"gp>=55 {s}"] = boot_corr(g.tank.values,
                                                       g.resid.values)

    # game-level differential form (active-gated, as the gate uses it)
    act_h = np.where(cs.h_gp_before >= GP_ACTIVE, cs.h_tank_score, 0.0)
    act_a = np.where(cs.a_gp_before >= GP_ACTIVE, cs.a_tank_score, 0.0)
    cs["tsd"] = act_h - act_a
    m = (cs.tsd != 0).values
    res["tsd_vs_resid_active"] = boot_corr(cs.tsd.values[m],
                                           cs.resid_home.values[m])

    # (b) D65 collapse-cluster membership
    heavy = (cs.p_mkt - 0.5).abs() > HEAVY
    cs["I"] = heavy & ((cs.p_us - 0.5).abs() <= HEAVY)
    cs["C"] = heavy & ((cs.p_us - 0.5).abs() > HEAVY)
    fav_home = cs.p_mkt >= 0.5
    cs["dog_tank"] = np.where(fav_home, cs.a_tank_score, cs.h_tank_score)
    cs["dog_gp"] = np.where(fav_home, cs.a_gp_before, cs.h_gp_before)
    cs["dog_tank_act"] = np.where(cs.dog_gp >= GP_ACTIVE, cs.dog_tank, 0.0)
    hv = cs[heavy]
    res["d65_auc"] = {}
    for nm, sc, lb in [
        ("dogtank I-vs-C (heavy)", hv.dog_tank, hv.I),
        ("dogtank_act I-vs-C (heavy)", hv.dog_tank_act, hv.I),
        ("dogtank I-vs-rest (all)", cs.dog_tank, cs.I),
        ("dogtank I&late-vs-C&late",
         hv[hv.late].dog_tank, hv[hv.late].I),
    ]:
        a_, n1, n0 = auc(sc.values, lb.values.astype(bool))
        res["d65_auc"][nm] = dict(auc=round(float(a_), 4) if a_ == a_ else None,
                                  n_pos=n1, n_neg=n0)
    # does dog tank score track the per-game excess loss inside heavy games?
    res["dogtank_vs_excess_heavy"] = boot_corr(hv.dog_tank.values,
                                               hv.d_excess.values)
    res["dogtank_vs_excess_I"] = boot_corr(cs[cs.I].dog_tank.values,
                                           cs[cs.I].d_excess.values)

    # descriptive sanity: top tank team-dates in each April
    tops = (df[(df.game_date.dt.month == 4) & (df.gp_before >= GP_ACTIVE)]
            .sort_values("tank_score", ascending=False)
            .groupby("season").head(5)
            [["season", "team_abbrev", "game_date", "tank_score",
              "z_a", "z_b", "z_c", "z_d"]])
    res["top_april_tankers"] = [
        dict(season=r.season, team=r.team_abbrev,
             date=str(r.game_date.date()), tank=round(r.tank_score, 2),
             z_a=round(r.z_a, 2), z_b=round(r.z_b, 2),
             z_c=round(r.z_c, 2), z_d=round(r.z_d, 2))
        for r in tops.itertuples()]
    return res


# --------------------------------------------------------------------------
# 3. PRE-REGISTERED GATE (capstone loop copied from prod_by_season.py)
# --------------------------------------------------------------------------

def fit_k(hist):
    """1-D MLE for k on accumulated (margin, tsd, y); Newton, clipped."""
    act = [(m, t, y) for m, t, y in hist if t != 0.0]
    if len(act) < K_MIN_ACTIVE:
        return 0.0
    m = np.array([r[0] for r in act])
    t = np.array([r[1] for r in act])
    y = np.array([r[2] for r in act], float)
    k = 0.0
    for _ in range(60):
        p = sigmoid((m + k * t) / SCALE)
        g = float(np.sum((p - y) * t) / SCALE)
        h = float(np.sum(p * (1 - p) * t * t) / (SCALE * SCALE))
        if h < 1e-12:
            break
        step = g / h
        k = float(np.clip(k - step, -K_CLIP, K_CLIP))
        if abs(step) < 1e-9:
            break
    return k


def season_run_gate(season, tank_map, hist, k_traj):
    """Copy of prod_by_season.season_run (default flags: oracle OUT sets,
    b2b, weekly refit, w_comp=0.7) + same-run control and the tank term."""
    con = connect(read_only=True)
    pm = con.execute("""SELECT game_id, team_id, player_id, seconds/60.0 AS mins
        FROM player_game_stats WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    played = {(g, t): set(grp.player_id)
              for (g, t), grp in pm.groupby(["game_id", "team_id"])}
    meta = con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl, game_date
        FROM nba_games WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL
        ORDER BY game_date""", [season]).fetchdf()
    mkt = {(str(r[0])[:10], r[1], r[2]): r[3] for r in con.execute(
        "SELECT game_date, home, away, p_home_spread FROM odds_market WHERE season_end=?",
        [int(season[:4]) + 1]).fetchall()}
    by = {}
    order = []
    for x in meta.itertuples():
        if x.game_id not in by:
            order.append(x.game_id)
        by.setdefault(x.game_id, []).append(x)
    tdates = {}
    for x in meta.itertuples():
        d = x.game_date.date() if hasattr(x.game_date, "date") else x.game_date
        tdates.setdefault(x.team_id, set()).add(d)

    def b2b(tid, d):
        return (d - dt.timedelta(days=1)) in tdates.get(tid, set())

    rows = []
    model = comp = None
    last = None
    k = fit_k(hist)
    for gid in order:
        recs = by[gid]
        if len(recs) != 2:
            continue
        m_ = recs[0].matchup
        host = m_.split("@")[-1].strip() if "@" in m_ else \
            m_.split("vs.")[0].strip()
        h = next((x for x in recs if x.team_abbrev == host), None)
        a = next((x for x in recs if x.team_abbrev != host), None)
        if not h or not a:
            continue
        gd = h.game_date.date() if hasattr(h.game_date, "date") else h.game_date
        if last is None or (gd - last).days >= 7:
            model = fit_production(con, season, before=gd, w_comp=0.7)
            comp = CompositionModel(con, before=gd)
            last = gd
            k = fit_k(hist)
            k_traj.append((season, str(gd), round(k, 4),
                           sum(1 for r in hist if r[1] != 0.0)))
        pmv = mkt.get((str(gd)[:10], h.team_abbrev, a.team_abbrev))
        if pmv is None:
            continue
        outs = {}
        for t in (h.team_id, a.team_id):
            pl = played.get((gid, t), set())
            outs[t] = {p for p, d0 in comp.players.items()
                       if d0["team_id"] == t
                       and (gd - d0["last_played"]).days <= 12 and p not in pl}
        margin = model.margin(h.team_id, a.team_id, outs[h.team_id],
                              outs[a.team_id], gd,
                              b2b_home=b2b(h.team_id, gd),
                              b2b_away=b2b(a.team_id, gd))
        th = tank_map.get((gid, int(h.team_id)), (0.0, 0))
        ta = tank_map.get((gid, int(a.team_id)), (0.0, 0))
        act_h = th[0] if th[1] >= GP_ACTIVE else 0.0
        act_a = ta[0] if ta[1] >= GP_ACTIVE else 0.0
        tsd = act_h - act_a
        p_ctrl = float(sigmoid(margin / SCALE))
        p_trt = float(sigmoid((margin + k * tsd) / SCALE))
        yv = int(h.wl == "W")
        rows.append((season, gid, str(gd)[:10], h.team_abbrev, a.team_abbrev,
                     yv, p_ctrl, p_trt, float(pmv), float(tsd), round(k, 4)))
        hist.append((float(margin), float(tsd), yv))
    con.close()
    return rows


def boot_delta(d, n=NBOOT, seed=SEED):
    d = np.asarray(d, float)
    if len(d) == 0:
        return dict(n=0)
    rng = np.random.default_rng(seed)
    bs = np.empty(n)
    for b in range(n):
        bs[b] = d[rng.integers(0, len(d), len(d))].mean()
    return dict(n=int(len(d)), delta=round(float(d.mean()), 5),
                ci_lo=round(float(np.percentile(bs, 2.5)), 5),
                ci_hi=round(float(np.percentile(bs, 97.5)), 5))


def run_gate(df: pd.DataFrame) -> dict:
    tank_map = {(r.game_id, int(r.team_id)): (float(r.tank_score),
                                              int(r.gp_before))
                for r in df.itertuples()}
    hist: list[tuple[float, float, int]] = []
    k_traj: list = []
    all_rows = []
    for season in SEASONS_GATE:
        rows = season_run_gate(season, tank_map, hist, k_traj)
        all_rows.extend(rows)
        print(f"[gate] {season}: n={len(rows)}  k_end={k_traj[-1][2]}")
    g = pd.DataFrame(all_rows, columns=["season", "game_id", "game_date",
                                        "home", "away", "y", "p_ctrl",
                                        "p_trt", "p_mkt", "tsd", "k"])
    g.to_csv(CAPSTONE_CSV, index=False)
    g["ll_ctrl"] = logloss_vec(g.p_ctrl, g.y)
    g["ll_trt"] = logloss_vec(g.p_trt, g.y)
    g["ll_mkt"] = logloss_vec(g.p_mkt, g.y)
    g["delta"] = g.ll_ctrl - g.ll_trt          # + = tank term improves
    g["month"] = pd.to_datetime(g.game_date).dt.month
    heavy = (g.p_mkt - 0.5).abs() > HEAVY
    g["I"] = heavy & ((g.p_ctrl - 0.5).abs() <= HEAVY)

    res = {"k_trajectory": k_traj[-20:], "k_final": k_traj[-1][2],
           "n_active_rows": int((g.tsd != 0).sum())}
    # carry cross-check (control fidelity vs shipped capstone)
    try:
        cc = pd.read_csv(CARRY_CSV, dtype={"game_id": str})
        mm = g.merge(cc[["game_id", "p_us"]], on="game_id")
        res["ctrl_vs_carry_maxabs"] = round(
            float((mm.p_ctrl - mm.p_us).abs().max()), 6)
    except Exception as e:               # pragma: no cover
        res["ctrl_vs_carry_maxabs"] = str(e)
    subs = {"pooled": np.ones(len(g), bool),
            "late (Mar/Apr)": g.month.isin([3, 4]).values,
            "april": (g.month == 4).values,
            "D65-I": g.I.values,
            "active (tsd!=0)": (g.tsd != 0).values,
            "active & late": ((g.tsd != 0) & g.month.isin([3, 4])).values}
    for s in SEASONS_GATE:
        subs[s] = (g.season == s).values
    res["gate"] = {nm: boot_delta(g.delta.values[m]) for nm, m in subs.items()}
    res["logloss"] = {s: dict(ctrl=round(float(gg.ll_ctrl.mean()), 4),
                              trt=round(float(gg.ll_trt.mean()), 4),
                              mkt=round(float(gg.ll_mkt.mean()), 4))
                      for s, gg in g.groupby("season")}
    return res


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    results = {}
    if os.path.exists(STATS_PARQUET) and mode == "gate":
        df = pd.read_csv(STATS_PARQUET, dtype={"game_id": str})
        df["game_date"] = pd.to_datetime(df["game_date"])
    else:
        df = build_stats()
    if mode in ("stats", "all"):
        results["validation"] = validate(df)
        print(json.dumps(results["validation"], indent=2, default=str))
    if mode in ("gate", "all"):
        results["gate"] = run_gate(df)
        print(json.dumps(results["gate"], indent=2, default=str))
    if os.path.exists(RESULTS_JSON):
        old = json.load(open(RESULTS_JSON))
        old.update(results)
        results = old
    json.dump(results, open(RESULTS_JSON, "w"), indent=2, default=str)
    print(f"[done] wrote {RESULTS_JSON}")


if __name__ == "__main__":
    main()
