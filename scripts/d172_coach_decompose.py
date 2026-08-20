#!/usr/bin/env python3
"""D172 TASK 2 §3-§4 (corrected) — separating the coach from the confound.

TWO DEFECTS IN THE FIRST PASS (d172_coach_measure.py), found by reading its
own numbers, and both fixed here:

  (1) THE PLACEBO WAS NOT MATCHED.  It cut random non-event team-seasons at
      event game indices, but those teams were not underperforming: their
      pre-window residual is ~0 while the events' is -2.28.  A null that does
      not carry the SELECTION cannot price the selection.  Fixed by matching
      each event to non-event team-seasons with a similarly bad pre-window
      residual (calliper on pre-window mean).

  (2) POST-MINUS-PRE IS THE WRONG STATISTIC ANYWAY.  A coach is fired AFTER a
      run of bad results, so pre is negative BY SELECTION and post-minus-pre is
      mechanically positive whatever the new coach does.  The only statistic a
      FEATURE could ever use is the POST-CHANGE residual itself, tested against
      zero — because the post period is the only one you can trade.

  (3) THE PERSISTENCE TEST NEEDED ITS OWN CONTROL.  lag-1 r = +0.389 for
      "same coach, same team" is not evidence of a coach effect: it is
      whatever persists in a TEAM across seasons.  The decisive contrast is
      SAME TEAM / DIFFERENT COACH.  If that is just as persistent, the
      persistence is the team's, and the coach contributes nothing.

READ-ONLY.  Writes data/d172_decompose.json.
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
from scipy import stats as st                                     # noqa: E402

from nbapred.teams import FRANCHISE                               # noqa: E402

DATA = ROOT / "data"
RES: dict = {}
T0 = time.time()


def ro(attempts=10, wait=60.0):
    for i in range(attempts):
        try:
            return duckdb.connect(str(DATA / "nba.duckdb"), read_only=True)
        except Exception as e:                                    # noqa: BLE001
            if ("lock" not in str(e).lower() and "held" not in str(e).lower()) \
               or i == attempts - 1:
                raise
            time.sleep(wait)


def cci(v, cl, alpha=0.05):
    d = pd.DataFrame({"v": np.asarray(v, float), "c": np.asarray(cl)}).dropna()
    if d.empty:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, K=0, p=np.nan, n=0)
    m = d.groupby("c").v.mean()
    K = len(m)
    mu, sd = float(m.mean()), float(m.std(ddof=1)) if K > 1 else np.nan
    se = sd / np.sqrt(K) if K > 1 else np.nan
    t = st.t.ppf(0.975, K - 1) if K > 1 else np.nan
    ts = mu / se if se and se > 0 else np.nan
    p = 2 * (1 - st.t.cdf(abs(ts), K - 1)) if se and se > 0 else np.nan
    return dict(mean=mu, lo=mu - t * se, hi=mu + t * se, K=int(K),
                p=float(p) if p == p else np.nan, n=int(len(d)))


def fmt(d, w=9):
    return (f"{d['mean']:>{w}.3f}  [{d['lo']:>+7.3f},{d['hi']:>+7.3f}] "
            f"p={d['p']:.4f}  n={d['n']:>4} K={d['K']}")


con = ro()
print("=" * 100)
print("D172 §3-§4 CORRECTED — IS THE COACH ANYTHING BUT THE ROSTER?")
print("=" * 100)

# ------------------------------------------------------------------ rebuild
cb = pd.read_csv(DATA / "d172_coach_bbref.csv")
cb = cb[cb.g.notna() & (cb.g > 0)].sort_values(["season", "team", "order"])
cb["cum"] = cb.groupby(["season", "team"]).g.cumsum()
cb["lo"] = cb.cum - cb.g
spells = cb.rename(columns={"Coach": "coach", "cum": "hi"})[
    ["season", "team", "coach", "order", "lo", "hi", "g"]]

games = con.execute("""
    SELECT season, team_id, team_abbrev, game_id, game_date
      FROM nba_games
     WHERE game_id LIKE '002%' AND pts IS NOT NULL
     ORDER BY season, team_id, game_date, game_id""").fetchdf()
games["gi"] = games.groupby(["season", "team_id"]).cumcount()
g2 = games.merge(spells, left_on=["season", "team_abbrev"],
                 right_on=["season", "team"], how="left")
g2 = g2[(g2.gi >= g2.lo) & (g2.gi < g2.hi)].copy()

A = pd.read_csv(DATA / "ats19_frame.csv.gz")
A["game_id"] = A.game_id.astype(str).str.zfill(10)
long = []
for side, sgn in (("home", 1), ("away", -1)):
    d = A[["season", "game_id", side, "m_us", "open_margin",
           "margin_actual"]].rename(columns={side: "tm"})
    d["r_model"] = sgn * (d.margin_actual - d.m_us)
    d["r_open"] = sgn * (d.margin_actual - d.open_margin)
    long.append(d[["season", "game_id", "tm", "r_model", "r_open"]])
L = pd.concat(long, ignore_index=True)

g2["team_mod"] = g2.team_abbrev.map(lambda a: FRANCHISE.get(a, a))
g2["gid"] = g2.game_id.astype(str).str.zfill(10)
P = g2.merge(L, left_on=["season", "gid", "team_mod"],
             right_on=["season", "game_id", "tm"], how="inner") \
      .sort_values(["season", "team_abbrev", "gi"])
ev = spells[(spells.order > 0) & spells.season.isin(set(P.season))].copy()
IDX = {k: v for k, v in P.groupby(["season", "team_abbrev"])}
print(f"panel {len(P):,} team-games, {P.season.nunique()} seasons; "
      f"in-season coach changes in frame {len(ev)}")

# ==================================================== 1. MATCHED EVENT STUDY
print("\n" + "-" * 100)
print("1. THE STATISTIC THAT MATTERS: the POST-CHANGE residual, against zero")
print("-" * 100)
print("   post-minus-pre is mechanically positive (a coach is fired AFTER a\n"
      "   bad run).  Only the POST period is tradeable, so that is the test.\n")

W = 20
rows = []
for r in ev.itertuples():
    d = IDX.get((r.season, r.team))
    if d is None:
        continue
    pre, post = d[d.gi < r.lo], d[d.gi >= r.lo]
    if len(pre) < 10 or len(post) < 10:
        continue
    rows.append(dict(season=r.season, team=r.team, coach=r.coach, lo=int(r.lo),
                     n_pre=len(pre), n_post=len(post),
                     pre_model=pre.r_model.mean(), post_model=post.r_model.mean(),
                     pre_open=pre.r_open.mean(), post_open=post.r_open.mean(),
                     pre20_model=pre.tail(W).r_model.mean(),
                     post20_model=post.head(W).r_model.mean(),
                     pre20_open=pre.tail(W).r_open.mean(),
                     post20_open=post.head(W).r_open.mean()))
E = pd.DataFrame(rows)
print(f"{'quantity':<44}{'mean':>9}  {'95% CI (season-clustered)':<26}")
out1 = {}
for lbl, col in (("PRE-change  residual vs OUR MODEL", "pre_model"),
                 ("POST-change residual vs OUR MODEL", "post_model"),
                 ("PRE-change  residual vs the MARKET (open)", "pre_open"),
                 ("POST-change residual vs the MARKET (open)", "post_open"),
                 ("POST-change, first 20 games, vs MARKET", "post20_open")):
    r = cci(E[col], E.season)
    out1[col] = r
    print(f"{lbl:<44}{fmt(r)}")
RES["post_vs_zero"] = out1
print("\n  READ: the market residual BEFORE the firing is strongly negative —\n"
      "  that is the selection (coaches are fired after bad runs), not news.\n"
      "  AFTER the firing it is the POST number above.  If that is zero, the\n"
      "  market prices the new coach correctly and there is nothing to sell.")

# ---- matched placebo: same bad pre-run, no coach change
print("\n" + "-" * 100)
print("2. MATCHED PLACEBO — teams that ran equally badly and did NOT fire "
      "the coach")
print("-" * 100)
chg = set(zip(ev.season, ev.team))
pool = []
cuts = sorted(set(ev.lo.astype(int)))
for k, d in IDX.items():
    if k in chg:
        continue
    for lo in cuts:
        pre, post = d[d.gi < lo], d[d.gi >= lo]
        if len(pre) < 10 or len(post) < 10:
            continue
        pool.append(dict(season=k[0], team=k[1], lo=lo,
                         pre_model=pre.r_model.mean(),
                         post_model=post.r_model.mean(),
                         pre_open=pre.r_open.mean(),
                         post_open=post.r_open.mean()))
POOL = pd.DataFrame(pool)
print(f"placebo pool: {len(POOL):,} (team-season, cut) cells, "
      f"{POOL.groupby(['season','team']).ngroups} team-seasons")

CAL = 0.75          # points of pre-window residual
matched = []
for r in E.itertuples():
    c = POOL[(POOL.lo == r.lo) & (POOL.pre_open.sub(r.pre_open).abs() <= CAL)]
    if len(c) == 0:
        c = POOL[(POOL.lo.sub(r.lo).abs() <= 5) &
                 (POOL.pre_open.sub(r.pre_open).abs() <= CAL)]
    if len(c) == 0:
        continue
    matched.append(dict(season=r.season, team=r.team, n_ctrl=len(c),
                        ev_pre_open=r.pre_open, ev_post_open=r.post_open,
                        ct_pre_open=c.pre_open.mean(),
                        ct_post_open=c.post_open.mean(),
                        ev_pre_model=r.pre_model, ev_post_model=r.post_model,
                        ct_pre_model=c.pre_model.mean(),
                        ct_post_model=c.post_model.mean()))
M = pd.DataFrame(matched)
print(f"events matched: {len(M)} / {len(E)}  "
      f"(calliper {CAL} pts on the pre-window market residual, "
      f"mean {M.n_ctrl.mean():.0f} controls each)\n")
print(f"{'':<30}{'EVENT (coach fired)':>22}{'CONTROL (kept)':>20}"
      f"{'DIFF-IN-DIFF':>16}")
did = {}
for lbl, ev_pre, ev_po, ct_pre, ct_po in (
        ("vs MARKET (open)", "ev_pre_open", "ev_post_open",
         "ct_pre_open", "ct_post_open"),
        ("vs OUR MODEL", "ev_pre_model", "ev_post_model",
         "ct_pre_model", "ct_post_model")):
    e_d = M[ev_po] - M[ev_pre]
    c_d = M[ct_po] - M[ct_pre]
    r = cci(e_d - c_d, M.season)
    did[lbl] = dict(event_pre=float(M[ev_pre].mean()),
                    event_post=float(M[ev_po].mean()),
                    ctrl_pre=float(M[ct_pre].mean()),
                    ctrl_post=float(M[ct_po].mean()),
                    event_delta=float(e_d.mean()), ctrl_delta=float(c_d.mean()),
                    did=r)
    print(f"{lbl:<30}pre {M[ev_pre].mean():>6.2f} post {M[ev_po].mean():>6.2f}"
          f"   pre {M[ct_pre].mean():>6.2f} post {M[ct_po].mean():>6.2f}")
    print(f"{'':<30}{'delta ' + f'{e_d.mean():+.3f}':>22}"
          f"{'delta ' + f'{c_d.mean():+.3f}':>20}   {fmt(r)}")
RES["matched_did"] = did
print("\n  DIFF-IN-DIFF is the coach-change effect NET of mean reversion.")

# ================================================= 3. PERSISTENCE, DECOMPOSED
print("\n" + "-" * 100)
print("3. PERSISTENCE — is the memory the COACH's or the TEAM's?  (D137)")
print("-" * 100)

TS = (P.groupby(["team_abbrev", "season"])
        .agg(n=("r_model", "size"), eff=("r_model", "mean"),
             eff_m=("r_open", "mean")).reset_index())
TS["yr"] = TS.season.str[:4].astype(int)
# principal coach of the team-season = the one with the most games
prin = (P.groupby(["team_abbrev", "season", "coach"]).size()
          .rename("n").reset_index().sort_values("n", ascending=False)
          .drop_duplicates(["team_abbrev", "season"]))
TS = TS.merge(prin[["team_abbrev", "season", "coach"]],
              on=["team_abbrev", "season"], how="left")
lag = TS.merge(TS.assign(yr=TS.yr + 1), on=["team_abbrev", "yr"],
               suffixes=("", "_p"))
same = lag[lag.coach == lag.coach_p]
diff = lag[lag.coach != lag.coach_p]
print(f"consecutive team-season pairs: {len(lag)}  "
      f"(same head coach {len(same)}, coach CHANGED between seasons {len(diff)})\n")
print(f"{'contrast':<46}{'lag-1 r':>10}{'p':>9}{'n':>7}")
pers = {}
for lbl, d, a, b in (
        ("SAME coach, same team — vs our model", same, "eff_p", "eff"),
        ("DIFFERENT coach, same team — vs our model", diff, "eff_p", "eff"),
        ("SAME coach, same team — vs the MARKET", same, "eff_m_p", "eff_m"),
        ("DIFFERENT coach, same team — vs the MARKET", diff, "eff_m_p", "eff_m")):
    if len(d) > 3:
        r, p = st.pearsonr(d[a], d[b])
        pers[lbl] = dict(r=float(r), p=float(p), n=int(len(d)))
        print(f"{lbl:<46}{r:>+10.4f}{p:>9.4f}{len(d):>7}")
RES["persistence_decomposed"] = pers
print("\n  If DIFFERENT-coach persistence matches SAME-coach persistence, the\n"
      "  memory belongs to the TEAM (roster, market miscalibration), not the man.")

# ---- does the effect TRAVEL with the coach? the only coach-specific test
CT = (P.groupby(["coach", "season", "team_abbrev"])
        .agg(n=("r_model", "size"), eff=("r_model", "mean"),
             eff_m=("r_open", "mean")).reset_index())
CT = CT[CT.n >= 20].copy()
CT["yr"] = CT.season.str[:4].astype(int)
mv = CT.merge(CT.assign(yr=CT.yr + 1), on=["coach", "yr"], suffixes=("", "_p"))
mv = mv[mv.team_abbrev != mv.team_abbrev_p]
print(f"\n  COACH MOVES TEAM (the only coach-specific persistence test): "
      f"n={len(mv)}")
for lbl, a, b in (("vs our model", "eff_p", "eff"),
                  ("vs the MARKET", "eff_m_p", "eff_m")):
    if len(mv) > 3:
        r, p = st.pearsonr(mv[a], mv[b])
        RES.setdefault("coach_travels", {})[lbl] = dict(
            r=float(r), p=float(p), n=int(len(mv)))
        print(f"    {lbl:<20}r = {r:+.4f}  p={p:.3f}")

# ---- signal share on the MARKET residual (the honest one)
print("\n  SIGNAL SHARE of a coach-season effect (D137 shape):")
for lbl, col in (("vs our model", "r_model"), ("vs the MARKET", "r_open")):
    CS = (P.groupby(["coach", "team_abbrev", "season"])
            .agg(n=(col, "size"), eff=(col, "mean")).reset_index())
    CS = CS[CS.n >= 20]
    sd = float(P[col].std(ddof=1))
    tot = float(CS.eff.var(ddof=1))
    noise = float((sd ** 2 / CS.n).mean())
    tau2 = max(tot - noise, 0.0)
    RES.setdefault("signal_share", {})[lbl] = dict(
        sd_game=sd, var_total=tot, var_noise=noise, tau2=tau2,
        tau=float(np.sqrt(tau2)), share=float(tau2 / tot), n=int(len(CS)))
    print(f"    {lbl:<16}tau = {np.sqrt(tau2):.3f} pts, "
          f"signal share {tau2/tot:>6.1%}  (n={len(CS)})")
print("  D137 reference: team home advantage tau 1.80 pts, 26% signal, and\n"
      "  lag-1 +0.02 — real within a season, worthless across one.")

con.close()
RES["elapsed_s"] = round(time.time() - T0, 1)
(DATA / "d172_decompose.json").write_text(json.dumps(RES, indent=1, default=str))
E.to_csv(DATA / "d172_event_rows.csv", index=False)
M.to_csv(DATA / "d172_matched.csv", index=False)
print(f"\nWROTE data/d172_decompose.json ({RES['elapsed_s']}s)")
