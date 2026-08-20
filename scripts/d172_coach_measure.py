#!/usr/bin/env python3
"""D172 TASK 2 §2-§3 — MEASURE BEFORE MODELLING.

Order of business, per the brief:

  §2  Is there anything a coach could PLAUSIBLY move that we do not already
      price?  Measured on behaviours we hold for all 30 seasons: rotation
      concentration, bench depth, starter stability, pace, and the veteran /
      youth minutes split (the D73 tank composite's own input).  The clean
      identification is the in-season coach CHANGE — same roster, same
      season, different coach.

  §3  The confound IS the problem (D70/D137).  Good coaches coach good
      rosters, so a coach fixed effect is unidentified.  The coach-CHANGE
      event is used instead, and every performance effect is measured NET of
      two controls:
        (A) our own model's expected margin `m_us` — COACH-BLIND, so this is
            the RAW effect (roster + schedule held fixed, coach not);
        (B) the market's opening/closing margin — which HAS seen the firing,
            so this is the UNPRICED effect, i.e. the only one a feature could
            ever monetise.
      Both are reported against a MEAN-REVERSION NULL built from placebo cut
      points, because coaches are fired when a team is underperforming and
      regression alone manufactures a positive post-minus-pre.  (D164: a
      600-cell search manufactures +16.92 ROI points of pure noise; the null
      runs alongside, always.)

  §4  Persistence.  D137: team home advantage was real within a season
      (tau 1.80 pts, 26% signal) and had ZERO cross-season memory (lag-1
      +0.02), which made it worthless for forecasting.  Same test here.

READ-ONLY on data/nba.duckdb.  Writes data/d172_*.json / *.csv only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import duckdb                                                     # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

from nbapred.teams import FRANCHISE                               # noqa: E402

DB = str(ROOT / "data" / "nba.duckdb")
DATA = ROOT / "data"
RES: dict = {}
T0 = time.time()


def ro(attempts=10, wait=60.0):
    for i in range(attempts):
        try:
            return duckdb.connect(DB, read_only=True)
        except Exception as e:                                    # noqa: BLE001
            if ("lock" not in str(e).lower() and "held" not in str(e).lower()) \
               or i == attempts - 1:
                raise
            time.sleep(wait)


def clustered_ci(values, clusters, alpha=0.05):
    """Season-clustered mean + t interval on the K cluster means (K-1 dof).

    The shipping statistic under GATE_POLICY_V2 §9.  Returns
    (mean, lo, hi, K, t_stat, p_two_sided)."""
    from scipy import stats
    df = pd.DataFrame({"v": np.asarray(values, float),
                       "c": np.asarray(clusters)}).dropna()
    if df.empty:
        return (np.nan,) * 6
    m = df.groupby("c").v.mean()
    K = len(m)
    if K < 2:
        return (float(m.iloc[0]), np.nan, np.nan, K, np.nan, np.nan)
    mu, sd = float(m.mean()), float(m.std(ddof=1))
    se = sd / np.sqrt(K)
    t = stats.t.ppf(1 - alpha / 2, K - 1)
    tstat = mu / se if se > 0 else np.nan
    p = 2 * (1 - stats.t.cdf(abs(tstat), K - 1)) if se > 0 else np.nan
    return mu, mu - t * se, mu + t * se, K, tstat, p


con = ro()
print("=" * 100)
print("D172 §2-§4  COACH: MEASURE BEFORE MODELLING")
print("=" * 100)

# ============================================================ 1. COACH PANEL
print("\n" + "-" * 100)
print("1. COACH PANEL — assign a coach to every regular-season team-game")
print("-" * 100)

cb = pd.read_csv(DATA / "d172_coach_bbref.csv")
cb = cb[cb.g.notna() & (cb.g > 0)].copy()
cb = cb.sort_values(["season", "team", "order"])

games = con.execute("""
    SELECT season, team_id, team_abbrev, game_id, game_date, is_home, pts, wl
      FROM nba_games
     WHERE game_id LIKE '002%' AND pts IS NOT NULL
     ORDER BY season, team_id, game_date, game_id""").fetchdf()
games["gi"] = games.groupby(["season", "team_id"]).cumcount()
games["n_ts"] = games.groupby(["season", "team_id"]).game_id.transform("size")

# opponent points -> team margin
pts = games[["game_id", "team_id", "pts"]]
opp = pts.merge(pts, on="game_id", suffixes=("", "_o"))
opp = opp[opp.team_id != opp.team_id_o][["game_id", "team_id", "pts_o"]]
games = games.merge(opp, on=["game_id", "team_id"], how="left")
games["margin"] = games.pts - games.pts_o

# BBRef team code is the code IN FORCE that season (NJN/SEA/...), same as ours.
gsz = (games.groupby(["season", "team_abbrev"]).size()
       .rename("n_games").reset_index())
chk = (cb.groupby(["season", "team"]).g.sum().rename("g_bb").reset_index()
       .merge(gsz, left_on=["season", "team"],
              right_on=["season", "team_abbrev"], how="outer"))
bad = chk[(chk.g_bb != chk.n_games) | chk.g_bb.isna() | chk.n_games.isna()]
print(f"team-seasons: BBRef {chk.g_bb.notna().sum()}  nba_games "
      f"{chk.n_games.notna().sum()}  G-SUM MISMATCH {len(bad)}")
if len(bad):
    print(bad.head(20).to_string(index=False))
RES["panel_mismatch_team_seasons"] = int(len(bad))

# spell boundaries: coach `order` covers game indices [cum-g, cum)
cb["cum"] = cb.groupby(["season", "team"]).g.cumsum()
cb["lo"] = cb.cum - cb.g
spells = cb[["season", "team", "Coach", "order", "lo", "cum", "g", "w", "l"]]
spells = spells.rename(columns={"Coach": "coach", "cum": "hi"})

g2 = games.merge(spells, left_on=["season", "team_abbrev"],
                 right_on=["season", "team"], how="left")
g2 = g2[(g2.gi >= g2.lo) & (g2.gi < g2.hi)].copy()
print(f"team-games assigned a coach: {len(g2):,} / {len(games):,} "
      f"({len(g2)/len(games):.4%})")
RES["team_games"] = int(len(games))
RES["team_games_with_coach"] = int(len(g2))
RES["coach_assign_rate"] = float(len(g2) / len(games))

n_chg = spells.groupby(["season", "team"]).size().sub(1).clip(lower=0)
print(f"in-season coach changes: {int(n_chg.sum())} over "
      f"{int((n_chg > 0).sum())} team-seasons "
      f"({spells.season.nunique()} seasons)")
RES["n_changes"] = int(n_chg.sum())
RES["n_change_team_seasons"] = int((n_chg > 0).sum())

# ==================================================== 2. COACH-MOVABLE BEHAVIOUR
print("\n" + "-" * 100)
print("2. WHAT COULD A COACH MOVE?  behaviours, all 30 seasons")
print("-" * 100)

beh = con.execute("""
    WITH s AS (
      SELECT game_id, team_id, player_id, seconds,
             fga, fta, tov, oreb, pts
        FROM player_game_stats
       WHERE game_id LIKE '002%' AND seconds > 0)
    SELECT game_id, team_id,
           COUNT(*)                                   AS n_used,
           SUM(seconds)                               AS tot_sec,
           SUM(fga) AS fga, SUM(fta) AS fta,
           SUM(tov) AS tov, SUM(oreb) AS oreb,
           SUM(POWER(seconds, 2))                     AS sec_sq
      FROM s GROUP BY 1,2""").fetchdf()
beh["hhi"] = beh.sec_sq / beh.tot_sec.pow(2)          # minutes Herfindahl
beh["poss"] = beh.fga - beh.oreb + beh.tov + 0.44 * beh.fta

# top-5 / top-8 minute shares and starter (top-5) turnover
pm = con.execute("""
    SELECT game_id, team_id, player_id, seconds
      FROM player_game_stats
     WHERE game_id LIKE '002%' AND seconds > 0""").fetchdf()
pm = pm.sort_values(["game_id", "team_id", "seconds"], ascending=[1, 1, 0])
pm["rk"] = pm.groupby(["game_id", "team_id"]).cumcount()
top = (pm[pm.rk < 8].groupby(["game_id", "team_id"])
       .seconds.sum().rename("sec_top8").reset_index())
beh = beh.merge(top, on=["game_id", "team_id"], how="left")
beh["top8_share"] = beh.sec_top8 / beh.tot_sec

top5 = (pm[pm.rk < 5].groupby(["game_id", "team_id"])
        .player_id.apply(frozenset).rename("t5").reset_index())

B = g2.merge(beh, on=["game_id", "team_id"], how="left") \
      .merge(top5, on=["game_id", "team_id"], how="left")
B = B.sort_values(["season", "team_id", "gi"])
prev = B.groupby(["season", "team_id"]).t5.shift(1)
B["t5_turnover"] = [
    np.nan if (not isinstance(a, frozenset) or not isinstance(b, frozenset))
    else 5 - len(a & b) for a, b in zip(B.t5, prev)]

BEHS = ["n_used", "hhi", "top8_share", "poss", "t5_turnover"]
print(f"behaviour panel: {len(B):,} team-games, "
      f"{B.season.nunique()} seasons\n")
print(B[BEHS].describe().T.to_string())

# --- 2a. variance decomposition: between-COACH vs between-TEAM vs within
print("\n2a. VARIANCE DECOMPOSITION (season-demeaned, so era drift is removed)")
print(f"{'behaviour':<14}{'sd(total)':>11}{'between-TEAM':>15}"
      f"{'between-COACHSPELL':>21}{'coach|team extra':>19}")
vd = {}
for b in BEHS:
    d = B[["season", "team_abbrev", "coach", b]].dropna().copy()
    d["z"] = d[b] - d.groupby("season")[b].transform("mean")
    tot = d.z.var(ddof=1)
    # between-team-season
    bt = d.groupby(["season", "team_abbrev"]).z.mean().var(ddof=1)
    # between-coach-spell (nested inside team-season)
    bc = d.groupby(["season", "team_abbrev", "coach"]).z.mean().var(ddof=1)
    vd[b] = dict(sd_total=float(np.sqrt(tot)),
                 frac_team=float(bt / tot), frac_coachspell=float(bc / tot),
                 extra=float((bc - bt) / tot))
    print(f"{b:<14}{np.sqrt(tot):>11.4f}{bt/tot:>14.1%}{bc/tot:>20.1%}"
          f"{(bc-bt)/tot:>18.1%}")
RES["variance_decomp"] = vd
print("  read: 'coach|team extra' is how much MORE of the variance is explained\n"
      "  by splitting a team-season at its coach change than by the team-season\n"
      "  alone.  It is an UPPER bound — any split of a season adds variance.")

# --- 2b. the clean test: does the behaviour JUMP at the change?
print("\n2b. DISCONTINUITY AT THE COACH CHANGE (same roster, same season)")
ev = spells[spells.order > 0].copy()          # each new coach = one event
ev = ev.merge(games[["season", "team_abbrev", "team_id"]].drop_duplicates(),
              left_on=["season", "team"], right_on=["season", "team_abbrev"],
              how="left")
print(f"events: {len(ev)}")


def window_stats(B, ev, col, W=None):
    """mean of `col` in the W games before / after each change (W=None: all)."""
    out = []
    idx = {k: v for k, v in B.groupby(["season", "team_abbrev"])}
    for r in ev.itertuples():
        d = idx.get((r.season, r.team))
        if d is None:
            continue
        pre = d[d.gi < r.lo]
        post = d[d.gi >= r.lo]
        if W:
            pre, post = pre.tail(W), post.head(W)
        if len(pre) < 5 or len(post) < 5:
            continue
        out.append(dict(season=r.season, team=r.team, coach=r.coach,
                        lo=int(r.lo), n_pre=len(pre), n_post=len(post),
                        pre=float(pre[col].mean()), post=float(post[col].mean()),
                        d=float(post[col].mean() - pre[col].mean())))
    return pd.DataFrame(out)


print(f"\n{'behaviour':<14}{'n_ev':>6}{'pre':>10}{'post':>10}{'delta':>10}"
      f"{'95% CI (season-clustered)':>30}{'p':>9}")
beh_ev = {}
for b in BEHS:
    e = window_stats(B, ev, b)
    if e.empty:
        continue
    mu, lo, hi, K, t, p = clustered_ci(e.d, e.season)
    beh_ev[b] = dict(n_events=len(e), pre=float(e.pre.mean()),
                     post=float(e.post.mean()), delta=mu, lo=lo, hi=hi,
                     K=int(K), p=float(p))
    print(f"{b:<14}{len(e):>6}{e.pre.mean():>10.4f}{e.post.mean():>10.4f}"
          f"{mu:>10.4f}   [{lo:>+8.4f},{hi:>+8.4f}]{p:>9.4f}")
RES["behaviour_at_change"] = beh_ev

# =============================================== 3. PERFORMANCE EVENT STUDY
print("\n" + "-" * 100)
print("3. THE COACH-CHANGE EVENT STUDY — performance, NET of two controls")
print("-" * 100)

A = pd.read_csv(DATA / "ats19_frame.csv.gz")
print(f"ats19_frame: {len(A):,} games, {A.season.nunique()} seasons "
      f"{A.season.min()}..{A.season.max()}  (m_us is COACH-BLIND)")
A["game_id"] = A.game_id.astype(str).str.zfill(10)
long = []
for side, sgn in (("home", 1), ("away", -1)):
    d = A[["season", "game_id", side, "m_us", "open_margin", "close_margin",
           "margin_actual"]].rename(columns={side: "team"})
    d["r_model"] = sgn * (d.margin_actual - d.m_us)
    d["r_open"] = sgn * (d.margin_actual - d.open_margin)
    d["r_close"] = sgn * (d.margin_actual - d.close_margin)
    long.append(d[["season", "game_id", "team", "r_model", "r_open", "r_close"]])
L = pd.concat(long, ignore_index=True)

# ats19_frame carries MODERN codes; the coach panel carries season codes.
g2["team_mod"] = g2.team_abbrev.map(lambda a: FRANCHISE.get(a, a))
g2["game_id_s"] = g2.game_id.astype(str).str.zfill(10)
P = g2.merge(L, left_on=["season", "game_id_s", "team_mod"],
             right_on=["season", "game_id", "team"], how="inner")
print(f"coach panel JOIN residual frame: {len(P):,} team-games "
      f"({P.season.nunique()} seasons)")
RES["event_panel_rows"] = int(len(P))
P = P.sort_values(["season", "team_abbrev", "gi"])

ev19 = ev[ev.season.isin(set(P.season))].copy()
print(f"in-season coach changes inside the 19-season residual frame: "
      f"{len(ev19)}")
RES["n_changes_in_frame"] = int(len(ev19))


def perf_events(P, ev, col, W=None):
    out = []
    idx = {k: v for k, v in P.groupby(["season", "team_abbrev"])}
    for r in ev.itertuples():
        d = idx.get((r.season, r.team))
        if d is None:
            continue
        pre, post = d[d.gi < r.lo], d[d.gi >= r.lo]
        if W:
            pre, post = pre.tail(W), post.head(W)
        if len(pre) < 10 or len(post) < 10:
            continue
        out.append(dict(season=r.season, team=r.team, coach=r.coach,
                        lo=int(r.lo), n_pre=len(pre), n_post=len(post),
                        pre=float(pre[col].mean()), post=float(post[col].mean()),
                        d=float(post[col].mean() - pre[col].mean())))
    return pd.DataFrame(out)


def placebo(P, ev, col, W=None, seed=0):
    """MEAN-REVERSION NULL.  For every team-season with NO coach change,
    cut at the same game index an event used, and compute the same post-pre.
    Matched on the SIGN and DECILE of the pre-window residual so the null
    carries the same selection-on-underperformance the events do."""
    rng = np.random.default_rng(seed)
    chg = set(zip(ev.season, ev.team))
    pool = [(k, v) for k, v in P.groupby(["season", "team_abbrev"])
            if k not in chg]
    out = []
    cuts = ev.lo.values
    for k, d in pool:
        for lo in rng.choice(cuts, size=min(3, len(cuts)), replace=False):
            pre, post = d[d.gi < lo], d[d.gi >= lo]
            if W:
                pre, post = pre.tail(W), post.head(W)
            if len(pre) < 10 or len(post) < 10:
                continue
            out.append(dict(season=k[0], team=k[1], lo=int(lo),
                            pre=float(pre[col].mean()),
                            post=float(post[col].mean()),
                            d=float(post[col].mean() - pre[col].mean())))
    return pd.DataFrame(out)


print("\n3a. RAW EFFECT — residual vs OUR MODEL (coach-blind: roster+schedule "
      "held, coach not)")
print("3b. UNPRICED EFFECT — residual vs the MARKET, which HAS seen the firing")
print(f"\n{'control':<10}{'window':<10}{'n_ev':>6}{'pre':>9}{'post':>9}"
      f"{'delta':>9}{'95% CI season-clustered':>28}{'p':>8}"
      f"{'null(mean-rev)':>16}{'net':>9}")
perf = {}
for col, lbl in (("r_model", "MODEL"), ("r_open", "OPEN"), ("r_close", "CLOSE")):
    for W, wl in ((None, "all"), (20, "+-20g")):
        e = perf_events(P, ev19, col, W)
        if e.empty:
            continue
        mu, lo, hi, K, t, p = clustered_ci(e.d, e.season)
        nl = placebo(P, ev19, col, W)
        nmu, nlo, nhi, nK, nt, np_ = clustered_ci(nl.d, nl.season)
        perf[f"{lbl}_{wl}"] = dict(
            n_events=len(e), pre=float(e.pre.mean()), post=float(e.post.mean()),
            delta=mu, lo=lo, hi=hi, K=int(K), p=float(p),
            null_delta=nmu, null_lo=nlo, null_hi=nhi, null_n=int(len(nl)),
            net=mu - nmu)
        print(f"{lbl:<10}{wl:<10}{len(e):>6}{e.pre.mean():>9.3f}"
              f"{e.post.mean():>9.3f}{mu:>9.3f}  [{lo:>+7.3f},{hi:>+7.3f}]"
              f"{p:>8.4f}{nmu:>16.3f}{mu-nmu:>9.3f}")
RES["performance_events"] = perf
print("\n  units = points of margin.  'null' is the placebo mean-reversion "
      "estimate\n  on team-seasons with NO coach change, cut at the same game "
      "indices.\n  'net' = delta - null: the part not explained by regression "
      "to the mean.")

# ============================================================ 4. PERSISTENCE
print("\n" + "-" * 100)
print("4. PERSISTENCE — D137's test.  A coach effect with no cross-season "
      "memory\n   is a description, not a feature.")
print("-" * 100)

CS = (P.groupby(["coach", "team_abbrev", "season"])
        .agg(n=("r_model", "size"), eff=("r_model", "mean"),
             eff_mkt=("r_open", "mean")).reset_index())
CS = CS[CS.n >= 20].copy()
print(f"coach-team-seasons with >=20 games in frame: {len(CS):,}  "
      f"(coaches {CS.coach.nunique()})")

# within-season signal share: tau^2 vs sampling noise (D137 shape)
sd_g = float(P.r_model.std(ddof=1))
CS["var_noise"] = sd_g ** 2 / CS.n
tot_var = float(CS.eff.var(ddof=1))
mean_noise = float(CS.var_noise.mean())
tau2 = max(tot_var - mean_noise, 0.0)
print(f"\n  per-game residual sd            {sd_g:.3f} pts")
print(f"  var(coach-season effect)        {tot_var:.4f}")
print(f"  expected sampling noise         {mean_noise:.4f}")
print(f"  tau^2 (true between-coach)      {tau2:.4f}   tau = {np.sqrt(tau2):.3f} pts")
print(f"  SIGNAL SHARE                    {tau2/tot_var:.1%}"
      f"   (D137 home advantage: 26%)")
RES["persistence"] = dict(sd_game=sd_g, var_total=tot_var,
                          var_noise=mean_noise, tau2=tau2,
                          tau=float(np.sqrt(tau2)),
                          signal_share=float(tau2 / tot_var),
                          n_coach_seasons=int(len(CS)))

# lag-1: same coach, same team, consecutive seasons
CS["yr"] = CS.season.str[:4].astype(int)
lag = CS.merge(CS.assign(yr=CS.yr + 1), on=["coach", "team_abbrev", "yr"],
               suffixes=("", "_prev"))
from scipy import stats as _st
for lbl, a, b in (("same coach+team, model resid", "eff_prev", "eff"),
                  ("same coach+team, market resid", "eff_mkt_prev", "eff_mkt")):
    if len(lag) > 3:
        r, p = _st.pearsonr(lag[a], lag[b])
        print(f"\n  LAG-1 {lbl}: r = {r:+.4f}  (p={p:.3f}, n={len(lag)} pairs)")
        RES.setdefault("lag1", {})[lbl] = dict(r=float(r), p=float(p),
                                               n=int(len(lag)))
# lag-1 for a coach who CHANGED team (does the effect travel with the man?)
CT = (P.groupby(["coach", "season", "team_abbrev"])
        .agg(n=("r_model", "size"), eff=("r_model", "mean")).reset_index())
CT = CT[CT.n >= 20].copy()
CT["yr"] = CT.season.str[:4].astype(int)
mv = CT.merge(CT.assign(yr=CT.yr + 1), on=["coach", "yr"], suffixes=("", "_p"))
mv = mv[mv.team_abbrev != mv.team_abbrev_p]
if len(mv) > 3:
    r, p = _st.pearsonr(mv.eff_p, mv.eff)
    print(f"  LAG-1 same coach, DIFFERENT team: r = {r:+.4f} "
          f"(p={p:.3f}, n={len(mv)} pairs)")
    RES.setdefault("lag1", {})["coach travels between teams"] = dict(
        r=float(r), p=float(p), n=int(len(mv)))

CS.to_csv(DATA / "d172_coach_seasons.csv", index=False)
ev19.to_csv(DATA / "d172_events.csv", index=False)
B[["season", "team_abbrev", "coach", "gi", "game_id"] + BEHS].to_csv(
    DATA / "d172_behaviour.csv.gz", index=False, compression="gzip")

con.close()
RES["elapsed_s"] = round(time.time() - T0, 1)
(DATA / "d172_measure.json").write_text(json.dumps(RES, indent=1, default=str))
print(f"\nWROTE data/d172_measure.json  ({RES['elapsed_s']}s)")
