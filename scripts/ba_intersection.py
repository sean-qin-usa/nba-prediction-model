#!/usr/bin/env python3
"""TASK B (D61 follow-up): Market-confidence intersection profile.

I = games where the MARKET is heavy (|p_mkt-0.5| > 0.35) but WE are not
(|p_us-0.5| <= 0.35).  D61 localized ~0.004 pooled logloss (the entire
heavy-favorite hole) to this slice.  This script profiles I against the
matched control C = market-heavy games where we AGREE (|p_us-0.5| > 0.35),
then decomposes the per-game excess loss d_i = LL_us - LL_mkt by factor.

Data: data/capstone_pergame_carry.csv (p_us/p_mkt/y; cross-checked vs
carry2) + nba.duckdb (read_only).  Market data is used ONLY to define the
analysis subsets (benchmark/diagnostic), never as a model input.

Factors profiled (oriented to the MARKET DOG where directional):
  season phase, dead/tanking team (entering wpct<.35 & gp>=60), n_outs,
  star-out (>=28 trailing-min inactive), B2B, chaos teams
  (PHI/DAL/BKN/POR/SAS), rest differential, blowout recency (last-5 avg
  |margin|), roster continuity (production continuity_map), coach-change
  windows (best-effort registry; production registry is empty).

Outputs (stdout only): matched over-representation table, within-I loss
decomposition (counterfactual: F-games repriced at non-F excess), joint
OLS, and a margin-mechanism probe.  Paired bootstrap 2000x, seed 7.
"""
from __future__ import annotations

import datetime as dt
import math
import os
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DB = os.path.join(ROOT, "data", "nba.duckdb")
CSV = os.path.join(ROOT, "data", "capstone_pergame_carry.csv")
CSV2 = os.path.join(ROOT, "data", "capstone_pergame_carry2.csv")

HEAVY = 0.35          # |p-0.5| threshold defining "heavy" confidence
STAR_MIN = 28.0       # trailing avg minutes for "star" (task spec; windows.py uses 30)
STAR_TRAIL = 10       # trailing games window for star minutes
STAR_MIN_GP = 3       # min qualifying games to trust the trailing average
DEAD_WPCT, DEAD_GP = 0.35, 60
CHAOS = {"PHI", "DAL", "BKN", "POR", "SAS"}
NBOOT, SEED = 2000, 7
EPS = 1e-12

# Best-effort midseason coach-change registry (production COACH_CHANGES in
# nbapred/market/windows.py is EMPTY).  First regular-season game under the
# new/interim coach.  KNOWN GAP: no 2025-26 entries verified -> the coach
# factor is a lower bound and covers 2023-24/2024-25 only.
COACH_CHANGES = [
    ("MIL", dt.date(2024, 1, 24)),   # Griffin out 2024-01-23 (Prunty/Rivers)
    ("WAS", dt.date(2024, 1, 26)),   # Unseld Jr. out 2024-01-25 (Keefe)
    ("SAC", dt.date(2024, 12, 28)),  # Brown out 2024-12-27 (Christie)
    ("MEM", dt.date(2025, 3, 29)),   # Jenkins out 2025-03-28 (Iisalo)
    ("DEN", dt.date(2025, 4, 9)),    # Malone out 2025-04-08 (Adelman)
]
COACH_WINDOW_GAMES = 15


def logloss(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def team_schedule(con) -> pd.DataFrame:
    """Per team-game PIT schedule state: entering wins/gp, rest, b2b,
    trailing-5 signed margin and |margin| (strictly before the game)."""
    tg = con.execute("""
        WITH g AS (
          SELECT season, game_id, game_date, team_abbrev AS team, pts, is_home
          FROM nba_games WHERE game_id LIKE '002%'),
        opp AS (
          SELECT a.season, a.game_id, a.game_date, a.team, a.is_home,
                 a.pts - b.pts AS margin,
                 CASE WHEN a.pts > b.pts THEN 1 ELSE 0 END AS win
          FROM g a JOIN g b ON a.game_id = b.game_id AND a.team <> b.team)
        SELECT * FROM opp ORDER BY team, game_date, game_id
    """).fetchdf()
    tg["game_date"] = pd.to_datetime(tg["game_date"])
    grp = tg.groupby(["season", "team"], sort=False)
    tg["gp_before"] = grp.cumcount()
    tg["wins_before"] = grp["win"].cumsum() - tg["win"]
    tg["wpct_before"] = np.where(
        tg.gp_before > 0, tg.wins_before / tg.gp_before.clip(lower=1), 0.5)
    tg["prev_date"] = grp["game_date"].shift(1)
    tg["rest"] = (tg.game_date - tg.prev_date).dt.days.clip(upper=7).fillna(7)
    tg["b2b"] = tg["rest"] == 1
    prev_m = grp["margin"].shift(1)
    tg["m5_signed"] = (prev_m.groupby([tg.season, tg.team])
                       .rolling(5, min_periods=3).mean()
                       .reset_index(level=[0, 1], drop=True))
    tg["m5_abs"] = (prev_m.abs().groupby([tg.season, tg.team])
                    .rolling(5, min_periods=3).mean()
                    .reset_index(level=[0, 1], drop=True))
    return tg


def star_out_map(con) -> pd.DataFrame:
    """Per (game_id, team_abbrev): n inactives and star-out flag, where star
    = trailing avg minutes >= STAR_MIN over last STAR_TRAIL games with 12+
    minutes played, strictly before the game date (windows.py convention)."""
    ina = con.execute("""
        SELECT i.game_id, i.player_id, g.team_abbrev AS team, g.game_date
        FROM game_inactives i
        JOIN nba_games g ON g.game_id = i.game_id AND g.team_id = i.team_id
        WHERE i.game_id LIKE '002%'
    """).fetchdf()
    pmin = con.execute("""
        SELECT s.player_id, g.game_date, s.seconds/60.0 AS mins
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '002%' AND s.seconds >= 720
        ORDER BY s.player_id, g.game_date
    """).fetchdf()
    hist: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for pid, sub in pmin.groupby("player_id", sort=False):
        hist[int(pid)] = (sub.game_date.values.astype("datetime64[D]"),
                          sub.mins.values)
    trail = np.full(len(ina), np.nan)
    gd = ina.game_date.values.astype("datetime64[D]")
    for k, (pid, d) in enumerate(zip(ina.player_id.astype(int).values, gd)):
        h = hist.get(pid)
        if h is None:
            continue
        i = np.searchsorted(h[0], d)          # games strictly before d
        if i >= STAR_MIN_GP:
            trail[k] = h[1][max(0, i - STAR_TRAIL):i].mean()
    ina["trail_min"] = trail
    out = (ina.groupby(["game_id", "team"])
           .agg(n_inact=("player_id", "size"),
                star_out=("trail_min", lambda s: bool((s >= STAR_MIN).any())),
                max_trail=("trail_min", "max"))
           .reset_index())
    return out


def build_frame() -> pd.DataFrame:
    df = pd.read_csv(CSV, dtype={"game_id": str})
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["ll_us"] = logloss(df.p_us, df.y)
    df["ll_mkt"] = logloss(df.p_mkt, df.y)
    df["d"] = df.ll_us - df.ll_mkt

    con = duckdb.connect(DB, read_only=True)
    try:
        ts = team_schedule(con)
        so = star_out_map(con)
        margins = con.execute("""
            WITH g AS (SELECT game_id, team_abbrev, pts, is_home
                       FROM nba_games WHERE game_id LIKE '002%')
            SELECT h.game_id, h.pts - a.pts AS home_margin
            FROM g h JOIN g a ON h.game_id = a.game_id AND h.team_abbrev <> a.team_abbrev
            WHERE h.is_home
        """).fetchdf()
        from nbapred.model.production import continuity_map  # import-only (no edits)
        id2ab = dict(con.execute(
            "SELECT DISTINCT team_id, team_abbrev FROM nba_games").fetchall())
        cont: dict[tuple[str, str], float] = {}
        for season in sorted(df.season.unique()):
            cm = continuity_map(con, season, before=None) or {}
            for tid, v in cm.items():
                ab = id2ab.get(tid)
                if ab:
                    cont[(season, ab)] = v
    finally:
        con.close()

    keys = ["season", "game_id"]
    tcols = ["gp_before", "wpct_before", "rest", "b2b", "m5_signed", "m5_abs"]
    for side in ("home", "away"):
        m = ts.rename(columns={"team": side})
        df = df.merge(m[keys + [side] + tcols], on=keys + [side], how="left",
                      suffixes=("", "_x"))
        df = df.rename(columns={c: f"{side[0]}_{c}" for c in tcols})
        s = so.rename(columns={"team": side})
        df = df.merge(s[["game_id", side, "n_inact", "star_out", "max_trail"]],
                      on=["game_id", side], how="left")
        df = df.rename(columns={"n_inact": f"{side[0]}_n_inact",
                                "star_out": f"{side[0]}_star_out",
                                "max_trail": f"{side[0]}_max_trail"})
        df[f"{side[0]}_star_out"] = df[f"{side[0]}_star_out"].fillna(False)
        df[f"{side[0]}_n_inact"] = df[f"{side[0]}_n_inact"].fillna(0)
        df[f"{side[0]}_cont"] = [cont.get((se, t), np.nan)
                                 for se, t in zip(df.season, df[side])]
    df = df.merge(margins, on="game_id", how="left")

    # Orientation: market favorite side (p_mkt is home win prob).
    fav_home = df.p_mkt >= 0.5
    def pick(hcol, acol):
        return np.where(fav_home, df[hcol], df[acol]), \
               np.where(fav_home, df[acol], df[hcol])
    for name, (h, a) in {
        "wpct": ("h_wpct_before", "a_wpct_before"),
        "gp": ("h_gp_before", "a_gp_before"),
        "rest": ("h_rest", "a_rest"),
        "b2b": ("h_b2b", "a_b2b"),
        "m5s": ("h_m5_signed", "a_m5_signed"),
        "m5a": ("h_m5_abs", "a_m5_abs"),
        "star": ("h_star_out", "a_star_out"),
        "cont": ("h_cont", "a_cont"),
        "nout": ("n_out_home", "n_out_away"),
    }.items():
        df[f"fav_{name}"], df[f"dog_{name}"] = pick(h, a)
    df["fav_team"] = np.where(fav_home, df.home, df.away)
    df["dog_team"] = np.where(fav_home, df.away, df.home)

    # Coach-change windows (best-effort registry).
    ts_idx = ts.set_index(["season", "team"])
    def in_coach_window(season, team, date):
        for t, start in COACH_CHANGES:
            if t != team or date.date() < start:
                continue
            sub = ts_idx.loc[(season, team)]
            n_since = int(((sub.game_date >= pd.Timestamp(start)) &
                           (sub.game_date < date)).sum())
            return n_since < COACH_WINDOW_GAMES
        return False
    df["h_coach"] = [in_coach_window(s, t, d)
                     for s, t, d in zip(df.season, df.home, df.game_date)]
    df["a_coach"] = [in_coach_window(s, t, d)
                     for s, t, d in zip(df.season, df.away, df.game_date)]

    # Derived factors.
    df["month"] = df.game_date.dt.month
    df["late_season"] = df.month.isin([3, 4])
    df["april"] = df.month == 4
    for side in ("fav", "dog"):
        df[f"dead_{side}"] = (df[f"{side}_wpct"] < DEAD_WPCT) & \
                             (df[f"{side}_gp"] >= DEAD_GP)
    # home/away-oriented flags for the candidate-feature sizing (section F)
    df["dead_h"] = (df.h_wpct_before < DEAD_WPCT) & (df.h_gp_before >= DEAD_GP)
    df["dead_a"] = (df.a_wpct_before < DEAD_WPCT) & (df.a_gp_before >= DEAD_GP)
    df["massout_h"] = (df.n_out_home >= 4) & ~df.h_star_out.astype(bool)
    df["massout_a"] = (df.n_out_away >= 4) & ~df.a_star_out.astype(bool)
    df["dog_massout_nostar"] = np.where(fav_home, df.massout_a, df.massout_h)
    df["dead_any"] = df.dead_fav | df.dead_dog
    df["star_any"] = df.fav_star | df.dog_star
    df["b2b_any"] = df.fav_b2b.astype(bool) | df.dog_b2b.astype(bool)
    df["chaos_any"] = df.home.isin(CHAOS) | df.away.isin(CHAOS)
    df["chaos_dog"] = df.dog_team.isin(CHAOS)
    df["chaos_fav"] = df.fav_team.isin(CHAOS)
    df["rest_diff_fav"] = df.fav_rest - df.dog_rest
    df["fav_rest_adv2"] = df.rest_diff_fav >= 2
    df["dog_rest_adv2"] = df.rest_diff_fav <= -2
    df["n_out_tot"] = df.n_out_home + df.n_out_away
    df["blow_max5"] = np.nanmax(np.c_[df.h_m5_abs, df.a_m5_abs], axis=1)
    df["coach_any"] = df.h_coach | df.a_coach
    df["mkt_correct"] = ((df.p_mkt >= 0.5) == (df.y == 1))
    return df


def boot_diff(x, y, wfun=None, n=NBOOT, seed=SEED):
    """Bootstrap CI + two-sided p for mean(x)-mean(y) (weighted y optional)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); y = np.asarray(y, float)
    wy = np.ones(len(y)) if wfun is None else np.asarray(wfun, float)
    diffs = np.empty(n)
    for b in range(n):
        xi = rng.integers(0, len(x), len(x))
        yi = rng.integers(0, len(y), len(y))
        diffs[b] = np.nanmean(x[xi]) - \
            (np.nansum(y[yi] * wy[yi]) / np.nansum(wy[yi] * ~np.isnan(y[yi])))
    obs = np.nanmean(x) - np.nansum(y * wy) / np.nansum(wy * ~np.isnan(y))
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return obs, lo, hi, min(max(p, 1 / n), 1.0)


def main():
    df = build_frame()
    heavy = (df.p_mkt - 0.5).abs() > HEAVY
    df["I"] = heavy & ((df.p_us - 0.5).abs() <= HEAVY)
    df["C"] = heavy & ((df.p_us - 0.5).abs() > HEAVY)
    I, C = df[df.I].copy(), df[df.C].copy()

    # carry2 cross-check
    c2 = pd.read_csv(CSV2)
    d2 = logloss(c2.p_us, c2.y) - logloss(c2.p_mkt, c2.y)
    i2 = ((c2.p_mkt - 0.5).abs() > HEAVY) & ((c2.p_us - 0.5).abs() <= HEAVY)
    print("=" * 88)
    print("TASK B — MARKET-CONFIDENT-WE-NOT INTERSECTION (I) PROFILE  "
          f"[carry CSV, {len(df)} games]")
    print("=" * 88)
    tot_excess = I.d.sum()
    print(f"I: n={len(I)}  mean d=+{I.d.mean():.4f}/game  "
          f"pooled contribution={tot_excess/len(df):.5f}")
    print(f"C (mkt-heavy, we agree): n={len(C)}  mean d={C.d.mean():+.4f}")
    print(f"carry2 cross-check: I n={int(i2.sum())} mean d={d2[i2].mean():+.4f}")
    rng = np.random.default_rng(SEED)
    bs = np.array([I.d.values[rng.integers(0, len(I), len(I))].mean()
                   for _ in range(NBOOT)])
    print(f"paired bootstrap {NBOOT}x mean d in I: "
          f"CI({np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f})")
    print(f"seasons: " + "  ".join(
        f"{s}: n={g.d.size} d=+{g.d.mean():.4f}"
        for s, g in I.groupby("season")))

    # p_mkt-extremity matching weights for C (5 quantile bins of I).
    ext = (df.p_mkt - 0.5).abs()
    qs = np.quantile(ext[df.I], [0.2, 0.4, 0.6, 0.8])
    bin_I = np.digitize(ext[df.I], qs)
    bin_C = np.digitize(ext[df.C], qs)
    wC = np.ones(len(C))
    for b in range(5):
        nI, nC = (bin_I == b).sum(), (bin_C == b).sum()
        if nC:
            wC[bin_C == b] = (nI / len(I)) / (nC / len(C))
    print(f"\nmatching: |p_mkt-.5| mean I={ext[df.I].mean():.3f} "
          f"C={ext[df.C].mean():.3f} C-matched="
          f"{np.average(ext[df.C], weights=wC):.3f}")

    binary = [
        "late_season", "april", "dead_any", "dead_dog", "dead_fav",
        "star_any", "dog_star", "fav_star", "b2b_any", "dog_b2b", "fav_b2b",
        "chaos_any", "chaos_dog", "chaos_fav", "fav_rest_adv2",
        "dog_rest_adv2", "coach_any", "dog_massout_nostar"]
    contin = ["n_out_tot", "dog_nout", "fav_nout", "rest_diff_fav",
              "blow_max5", "dog_m5s", "fav_m5s", "dog_cont", "fav_cont",
              "dog_wpct", "fav_wpct"]

    print("\n" + "-" * 88)
    print("A. OVER-REPRESENTATION: I vs C (C matched on |p_mkt-0.5| quintiles)"
          f"\n{'factor':<16}{'rate_I':>8}{'rate_Cm':>9}{'diff':>8}"
          f"{'CI_lo':>8}{'CI_hi':>8}{'p':>8}")
    over = {}
    for f in binary:
        obs, lo, hi, p = boot_diff(I[f].astype(float), C[f].astype(float), wC)
        over[f] = (obs, p)
        print(f"{f:<16}{I[f].mean():>8.3f}"
              f"{np.average(C[f].astype(float), weights=wC):>9.3f}"
              f"{obs:>+8.3f}{lo:>+8.3f}{hi:>+8.3f}{p:>8.3f}")
    print(f"\n{'continuous':<16}{'mean_I':>8}{'mean_Cm':>9}{'diff':>8}"
          f"{'CI_lo':>8}{'CI_hi':>8}{'p':>8}")
    for f in contin:
        obs, lo, hi, p = boot_diff(I[f], C[f], wC)
        mI = np.nanmean(I[f])
        mC = np.nansum(C[f] * wC) / np.nansum(wC * ~np.isnan(C[f]))
        print(f"{f:<16}{mI:>8.2f}{mC:>9.2f}{obs:>+8.2f}"
              f"{lo:>+8.2f}{hi:>+8.2f}{p:>8.3f}")

    # B. Loss decomposition within I: counterfactual repricing of F-games.
    print("\n" + "-" * 88)
    print("B. LOSS DECOMPOSITION within I (total excess = "
          f"{tot_excess:.2f} nats = {tot_excess/len(df):.5f} pooled)\n"
          f"{'factor':<16}{'n_F':>5}{'d_F':>9}{'d_notF':>9}{'delta':>9}"
          f"{'CI_lo':>8}{'CI_hi':>8}{'expl_nats':>10}{'expl_%':>8}"
          f"{'d_C_F':>8}")
    decomp = []
    thresh_blow = np.nanquantile(df.loc[heavy, "blow_max5"], 0.75)
    I["blow_hi"] = I.blow_max5 >= thresh_blow
    C["blow_hi"] = C.blow_max5 >= thresh_blow
    I["dog_crushed"] = I.dog_m5s <= -10
    C["dog_crushed"] = C.dog_m5s <= -10
    med_cont = np.nanmedian(df.loc[heavy, "dog_cont"])
    I["low_cont_dog"] = I.dog_cont < med_cont
    C["low_cont_dog"] = C.dog_cont < med_cont
    I["outs_hi"] = I.n_out_tot >= 4
    C["outs_hi"] = C.n_out_tot >= 4
    for f in binary + ["blow_hi", "dog_crushed", "low_cont_dog", "outs_hi"]:
        mF, mN = I[I[f].astype(bool)], I[~I[f].astype(bool)]
        if len(mF) < 8 or len(mN) < 8:
            continue
        obs, lo, hi, p = boot_diff(mF.d, mN.d)
        expl = len(mF) * obs
        dCF = C.loc[C[f].astype(bool), "d"].mean() if C[f].astype(bool).any() \
            else np.nan
        decomp.append((f, len(mF), mF.d.mean(), mN.d.mean(), obs, lo, hi,
                       expl, 100 * expl / tot_excess, dCF))
    decomp.sort(key=lambda r: -r[7])
    for f, nF, dF, dN, obs, lo, hi, expl, pct, dCF in decomp:
        print(f"{f:<16}{nF:>5}{dF:>+9.4f}{dN:>+9.4f}{obs:>+9.4f}"
              f"{lo:>+8.3f}{hi:>+8.3f}{expl:>10.2f}{pct:>8.1f}{dCF:>+8.3f}")

    # C. Joint OLS of d on factors within I (unique contributions).
    feats = ["late_season", "dead_dog", "dead_fav", "dog_star", "fav_star",
             "b2b_any", "chaos_any", "dog_crushed", "low_cont_dog", "outs_hi"]
    X = I[feats].astype(float).fillna(0).values
    X = np.c_[np.ones(len(X)), X]
    beta, *_ = np.linalg.lstsq(X, I.d.values, rcond=None)
    resid = I.d.values - X @ beta
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) *
                 (resid @ resid) / (len(X) - X.shape[1]))
    print("\n" + "-" * 88)
    print("C. JOINT OLS of d on factors within I (t>|2| = unique signal)")
    for name, b, s in zip(["const"] + feats, beta, se):
        print(f"  {name:<16}{b:>+9.4f}  t={b/s:>+6.2f}")

    # D. Margin mechanism probe: is it direction or magnitude?
    print("\n" + "-" * 88)
    print("D. MECHANISM PROBE (margin magnitude vs direction)")
    for lbl, g in (("I", I), ("C", C)):
        mkt_fav_won = g.mkt_correct
        fav_margin = np.where(g.p_mkt >= 0.5, g.home_margin, -g.home_margin)
        print(f"  {lbl}: mkt-fav win rate={mkt_fav_won.mean():.3f} "
              f"(implied {np.where(g.p_mkt>=.5, g.p_mkt, 1-g.p_mkt).mean():.3f}) "
              f" our fav prob={np.where(g.p_mkt>=.5, g.p_us, 1-g.p_us).mean():.3f}"
              f"  avg fav margin={np.nanmean(fav_margin):+.1f}")
    sub = I[I.mkt_correct]
    print(f"  I where mkt right (n={len(sub)}): d=+{sub.d.mean():.4f}; "
          f"I where mkt wrong (n={len(I)-len(sub)}): "
          f"d={I.d[~I.mkt_correct].mean():+.4f}"
          f"  -> upset games PAY us; the bleed is when the blowout lands")
    print(f"  our fav-prob shortfall in I: {np.mean(np.where(I.p_mkt>=.5, I.p_mkt, 1-I.p_mkt) - np.where(I.p_mkt>=.5, I.p_us, 1-I.p_us)):.3f}")

    # E. Where dead_dog lives inside I (phase interaction, for the feature pick)
    print("\n" + "-" * 88)
    print("E. FACTOR INTERSECTIONS within I (top pairs by joint excess)")
    pairs = [("dead_dog", "late_season"), ("dead_dog", "dog_star"),
             ("dead_dog", "chaos_any"), ("dog_star", "late_season"),
             ("dead_dog", "dog_crushed"), ("dog_crushed", "late_season")]
    for a, b in pairs:
        m = I[a].astype(bool) & I[b].astype(bool)
        if m.sum() >= 5:
            print(f"  {a} & {b}: n={m.sum()}  d=+{I.d[m].mean():.4f}  "
                  f"nats={I.d[m].sum():.2f} ({100*I.d[m].sum()/tot_excess:.0f}%)")
    unexplained = I.d[~(I.dead_dog | I.dog_star | I.late_season |
                        I.dog_crushed)].sum()
    print(f"  none-of-(dead_dog,dog_star,late,dog_crushed): "
          f"nats={unexplained:.2f} ({100*unexplained/tot_excess:.0f}%)")

    # F. Candidate margin-side features: hindsight single-coefficient sizing.
    #    p_adj = sigmoid(logit(p_us) + b*f); b fit by hindsight on ALL 3690
    #    games (upper bound; walk-forward will keep less).  Features are
    #    home-minus-away, built from DB only (market never an input).
    print("\n" + "-" * 88)
    print("F. CANDIDATE FEATURE SIZING (hindsight 1-param logit add-on; "
          "upper bounds)")
    late_gate = ((df.h_gp_before >= DEAD_GP) |
                 (df.a_gp_before >= DEAD_GP)).astype(float)
    m5h = df.h_m5_signed.fillna(0).values
    m5a = df.a_m5_signed.fillna(0).values
    cands = {
        "form5_lateGated": late_gate.values * (m5h - m5a),
        "form5_ungated": (m5h - m5a),
        "dead_flag": (df.dead_a.astype(float) - df.dead_h.astype(float)).values,
        "outs_diff": (df.n_out_away - df.n_out_home).values.astype(float),
        "massout_nostar": (df.massout_a.astype(float) -
                           df.massout_h.astype(float)).values,
        "deadXform5": (late_gate.values *
                       (df.dead_a.astype(float) * np.maximum(-m5a, 0) -
                        df.dead_h.astype(float) * np.maximum(-m5h, 0))),
    }
    z = np.log(np.clip(df.p_us, EPS, 1 - EPS) /
               np.clip(1 - df.p_us, EPS, 1 - EPS)).values
    y = df.y.values.astype(float)
    base_pool = df.ll_us.mean()
    Imask = df.I.values
    print(f"  baseline: pooled LL={base_pool:.5f}  mean d in I=+{I.d.mean():.4f}"
          f"  (hole total {tot_excess:.2f} nats)")
    print(f"  {'feature':<18}{'b*':>9}{'pool_dLL':>10}{'d_I_after':>10}"
          f"{'holeclosed%':>12}")
    for name, f in cands.items():
        f = np.asarray(f, float)
        def nll(b):
            p = 1 / (1 + np.exp(-(z + b * f)))
            return logloss(p, y).mean()
        bs_grid = np.linspace(-0.5, 0.5, 201)
        b0 = bs_grid[int(np.argmin([nll(b) for b in bs_grid]))]
        for span in (0.02, 0.002):
            bg = np.linspace(b0 - span * 5, b0 + span * 5, 41)
            b0 = bg[int(np.argmin([nll(b) for b in bg]))]
        p_new = 1 / (1 + np.exp(-(z + b0 * f)))
        ll_new = logloss(p_new, y)
        d_new = ll_new - df.ll_mkt.values
        closed = (tot_excess - d_new[Imask].sum()) / tot_excess * 100
        print(f"  {name:<18}{b0:>9.4f}{ll_new.mean()-base_pool:>+10.5f}"
              f"{d_new[Imask].mean():>+10.4f}{closed:>12.1f}")


if __name__ == "__main__":
    main()
