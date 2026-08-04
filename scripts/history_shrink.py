#!/usr/bin/env python3
"""IS AN ERA-SPECIFIC TERM FREE TO CARRY? (D153 deliverable 7 — the owner's
question, verbatim: "if things like altitude work 10 years ago but not today,
we can keep it in the model as long as there is no negative result — the term
will just go to 0".)

THE DISTINCTION THIS SCRIPT EXISTS TO MAKE.  D20/D70 (team home), D96
(altitude) and D136 (travel) were all tested as FIXED ADDITIONS behind a
pass/fail gate: fit the term, add it, does log loss improve?  Under a SHRINKAGE
estimator the question is different and better posed —

    cost    = the variance an extra fitted parameter injects where its true
              effect is zero (it is estimated as 0 +- se, not as 0);
    benefit = option value in the eras / subsets where the effect is live;
    verdict = free / cheap / costs X, not pass / fail.

Shrinkage DAMPS the cost.  It does not eliminate it, and this script measures
what is left.

THE ESTIMATOR IS THE THING UNDER TEST, not just the feature.  Every arm here is
fitted INSIDE the shipped walk-forward schedule-layer machinery:
`fit_schedule_layer_ext` (D136), whose new regressors shrink toward **0.0** by
w = n/(n+600), refit weekly on a trailing 730 days, with the D46 b2b/dead terms
and the wpct-diff control in the design matrix so a new coefficient is by
construction the INCREMENT over what already ships.  Applying the jointly refit
layer (rather than bolting a term onto unchanged b2b coefficients) is what
stops double counting.

ARMS
  TRAV      D136 arm A — dtrav_kkm, points per 1,000 great-circle km, home
            minus away.  Registered pre-registration sign: NEGATIVE.
  ALT       D96 retried as a game-level PHYSICAL FACT rather than a city
            dummy: the VISITOR's elevation gain in km on the trip into
            tonight's venue.  Predicted sign POSITIVE on home margin (the
            visitor climbs, the host does not).  Registered here as an
            ARM_TERMS entry at RUN TIME — nbapred/ is not modified.
  TEAMHOME  D20/D70 restored as 30 per-team home deviations under an explicit
            RIDGE (the `team_home_ridge=200` pattern from team_ratings.py),
            then the same w = n/(n+600) shrink.  This is the shrinkage form of
            the term D70 killed as a fixed effect.
  ALL       all three at once.

Reported per season and per corpus: paired delta (positive = the arm is
BETTER), i.i.d. and SEASON-CLUSTERED CIs, the cluster-mean t interval, and
MDE80 so a null is interpretable.

READ-ONLY.  Nothing in nbapred/ is modified; no default is flipped.

  python scripts/history_shrink.py            # both corpora
  python scripts/history_shrink.py 2016-17 ...
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nbapred import threads  # noqa: E402
threads.pin(1)

import numpy as np  # noqa: E402

from nbapred.db import connect  # noqa: E402
from nbapred.eval import splits as S  # noqa: E402
from nbapred.model import travel as TV  # noqa: E402
from nbapred.model.composition import CompositionModel  # noqa: E402
from nbapred.model.production import (  # noqa: E402
    DEAD_GP, DEAD_WPCT, SCALE, SCHED_PRIOR, SCHED_SHRINK, fit_production,
    fit_schedule_layer, fit_schedule_layer_ext, sigmoid)

# ---- ALT registered at run time (nbapred/ untouched) ----------------------
TV.ARM_TERMS["ALT"] = [("aelev_km", lambda s: s["elev_gain_m"] / 1000.0)]
TV.TERM_SIDE["aelev_km"] = "a"          # the VISITOR's climb
TV.TERM_PRED["aelev_km"] = +1           # home margin up when the visitor climbs

ARMSETS = {"TRAV": ("A",), "ALT": ("ALT",), "ALL": ("A", "ALT")}
TEAMHOME_RIDGE = 200.0                  # D20's shipped team_home_ridge
SEED = 20260801
B = 2000


# ---------------------------------------------------------------------------
# TEAMHOME: the D46 layer + a ridge-penalised per-team home block.
# Deliberately a verbatim structural copy of fit_schedule_layer (same frame,
# same trailing window, same wpct control, same w = n/(n+600) shrink) with one
# extra block, so the arm differs from the shipped layer in EXACTLY one way.
# ---------------------------------------------------------------------------

def fit_teamhome_layer(con, before, ridge=TEAMHOME_RIDGE):
    _lo = before - _dt.timedelta(days=730)
    g = con.execute("""
        WITH t AS (SELECT game_id, game_date, team_id, is_home, pts
                   FROM nba_games WHERE game_id LIKE '002%' AND pts IS NOT NULL
                   AND game_date < ? AND game_date >= ?)
        SELECT h.game_date, h.team_id ht, a.team_id a_t, h.pts - a.pts AS margin
        FROM t h JOIN t a USING (game_id) WHERE h.is_home AND NOT a.is_home
    """, [before, _lo]).fetchdf()
    n = len(g)
    if n == 0:
        return SCHED_PRIOR, {}
    lastg = {}
    for r in con.execute("""
        SELECT team_id, game_date FROM nba_games WHERE game_id LIKE '002%'
        AND pts IS NOT NULL AND game_date < ? AND game_date >= ?
        ORDER BY game_date""", [before, before - _dt.timedelta(days=760)]).fetchall():
        lastg.setdefault(r[0], []).append(r[1])
    prev = {t: {d: i for i, d in enumerate(ds)} for t, ds in lastg.items()}

    def is_b2b(t, d):
        ds = lastg.get(t, [])
        i = prev[t].get(d) if t in prev else None
        return i is not None and i > 0 and (d - ds[i - 1]).days == 1
    wl = con.execute("""
        SELECT season, team_id, game_date, wl FROM nba_games
        WHERE game_id LIKE '002%' AND wl IS NOT NULL AND game_date < ?
        AND game_date >= ? ORDER BY game_date""",
        [before, before - _dt.timedelta(days=760)]).fetchall()
    gp, wins, stand = {}, {}, {}
    for season, t, d0, w0 in wl:
        k = (season, t)
        stand[(t, d0)] = (gp.get(k, 0), wins.get(k, 0) / max(gp.get(k, 1), 1))
        gp[k] = gp.get(k, 0) + 1
        wins[k] = wins.get(k, 0) + (w0 == "W")

    def dead(t, d0):
        s = stand.get((t, d0))
        return s is not None and s[0] >= DEAD_GP and s[1] < DEAD_WPCT
    hb, ab, hd, ad, qd, hts = [], [], [], [], [], []
    for r in g.itertuples():
        d = r.game_date.date() if hasattr(r.game_date, "date") else r.game_date
        hb.append(is_b2b(r.ht, d)); ab.append(is_b2b(r.a_t, d))
        hd.append(dead(r.ht, d)); ad.append(dead(r.a_t, d))
        sh_, sa_ = stand.get((r.ht, d)), stand.get((r.a_t, d))
        qd.append((sh_[1] if sh_ else 0.5) - (sa_[1] if sa_ else 0.5))
        hts.append(r.ht)
    teams = sorted(set(hts))
    tix = {t: i for i, t in enumerate(teams)}
    TH = np.zeros((n, len(teams)))
    for i, t in enumerate(hts):
        TH[i, tix[t]] = 1.0
    X = np.c_[np.ones(n), np.array(hb, float), np.array(ab, float),
              np.array(hd, float), np.array(ad, float), TH,
              np.array(qd, float)]
    P = np.zeros(X.shape[1])
    P[5:5 + len(teams)] = ridge          # penalise ONLY the team-home block
    beta = np.linalg.solve(X.T @ X + np.diag(P), X.T @ g.margin.values)
    w = n / (n + SCHED_SHRINK)
    base5 = tuple(w * beta[i] + (1 - w) * SCHED_PRIOR[i] for i in range(5))
    dev = {t: w * float(beta[5 + tix[t]]) for t in teams}
    return base5, dev


# ---------------------------------------------------------------------------

def season_run(con, season, state):
    pm = con.execute("""SELECT game_id, team_id, player_id FROM player_game_stats
        WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    played = {(g, t): set(grp.player_id)
              for (g, t), grp in pm.groupby(["game_id", "team_id"])}
    meta = con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl, game_date
        FROM nba_games WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL
        ORDER BY game_date""", [season]).fetchdf()
    mkt = {(str(r[0])[:10], r[1], r[2]): r[3] for r in con.execute(
        "SELECT game_date, home, away, p_home_spread FROM odds_market "
        "WHERE season_end=?", [int(season[:4]) + 1]).fetchall()}
    by, order = {}, []
    for x in meta.itertuples():
        if x.game_id not in by:
            order.append(x.game_id)
        by.setdefault(x.game_id, []).append(x)
    tdates = {}
    for x in meta.itertuples():
        d = x.game_date.date() if hasattr(x.game_date, "date") else x.game_date
        tdates.setdefault(x.team_id, set()).add(d)

    def b2b(tid, d):
        return (d - _dt.timedelta(days=1)) in tdates.get(tid, set())

    rows, coefs = [], []
    model = comp = sh5 = None
    ext, th = {}, (None, {})
    last = None
    n_invalid = n_nostate = 0
    for gid in order:
        recs = by[gid]
        if len(recs) != 2:
            continue
        mm = recs[0].matchup
        host = mm.split("@")[-1].strip() if "@" in mm else mm.split("vs.")[0].strip()
        h = next((x for x in recs if x.team_abbrev == host), None)
        a = next((x for x in recs if x.team_abbrev != host), None)
        if not h or not a:
            continue
        gd = h.game_date.date() if hasattr(h.game_date, "date") else h.game_date
        if last is None or (gd - last).days >= 7:
            model = fit_production(con, season, before=gd, w_comp=0.7)
            comp = CompositionModel(con, before=gd)
            sh5 = fit_schedule_layer(con, gd)
            ext = {k: fit_schedule_layer_ext(con, gd, arms=v, state=state)
                   for k, v in ARMSETS.items()}
            th = fit_teamhome_layer(con, gd)
            coefs.append({"date": str(gd), "shipped_he": round(float(sh5[0]), 5),
                          **{k: {kk: round(float(vv), 5)
                                 for kk, vv in v[1].items()} for k, v in ext.items()},
                          "teamhome_sd": round(float(np.std(list(th[1].values()))), 5)
                          if th[1] else 0.0})
            last = gd
        pmv = mkt.get((str(gd)[:10], h.team_abbrev, a.team_abbrev))
        if pmv is None:
            continue
        sh_st, sa_st = state.get((h.team_id, gd)), state.get((a.team_id, gd))
        if sh_st is None or sa_st is None:
            n_nostate += 1
            continue
        if not (sh_st["travel_valid"] and sa_st["travel_valid"]):
            n_invalid += 1
            continue
        outs = {}
        for t in (h.team_id, a.team_id):
            pl = played.get((gid, t), set())
            outs[t] = {p for p, d0 in comp.players.items()
                       if d0["team_id"] == t
                       and (gd - d0["last_played"]).days <= 12 and p not in pl}
        bh, ba = b2b(h.team_id, gd), b2b(a.team_id, gd)
        m_base = model.margin(h.team_id, a.team_id, outs[h.team_id],
                              outs[a.team_id], gd, b2b_home=bh, b2b_away=ba)
        sched_shipped = sh5[0] + (sh5[1] if bh else 0.0) + (sh5[2] if ba else 0.0)
        rec = {"season": season, "game_id": gid, "game_date": str(gd)[:10],
               "home": h.team_abbrev, "away": a.team_abbrev,
               "y": int(h.wl == "W"), "p_mkt": float(pmv),
               "p_ctrl": float(sigmoid(m_base / SCALE))}
        for k, arms in ARMSETS.items():
            b5, exv = ext[k]
            s_arm = b5[0] + (b5[1] if bh else 0.0) + (b5[2] if ba else 0.0)
            for aa in arms:
                for c, fn in TV.ARM_TERMS[aa]:
                    s_arm += exv[c] * TV.term_value(c, fn, sh_st, sa_st)
            rec[f"dm_{k}"] = float(s_arm - sched_shipped)
            rec[f"p_{k}"] = float(sigmoid((m_base + s_arm - sched_shipped) / SCALE))
        b5, dev = th
        s_arm = (b5[0] + (b5[1] if bh else 0.0) + (b5[2] if ba else 0.0)
                 + dev.get(h.team_id, 0.0))
        rec["dm_TEAMHOME"] = float(s_arm - sched_shipped)
        rec["p_TEAMHOME"] = float(sigmoid((m_base + s_arm - sched_shipped) / SCALE))
        # ALL3 = travel + altitude + team home, jointly
        b5t, exv = ext["ALL"]
        s_all = (b5t[0] + (b5t[1] if bh else 0.0) + (b5t[2] if ba else 0.0)
                 + dev.get(h.team_id, 0.0))
        for aa in ARMSETS["ALL"]:
            for c, fn in TV.ARM_TERMS[aa]:
                s_all += exv[c] * TV.term_value(c, fn, sh_st, sa_st)
        rec["dm_ALL3"] = float(s_all - sched_shipped)
        rec["p_ALL3"] = float(sigmoid((m_base + s_all - sched_shipped) / SCALE))
        rows.append(rec)
    return rows, coefs, n_invalid, n_nostate


ARMS_OUT = ["TRAV", "ALT", "TEAMHOME", "ALL", "ALL3"]
CERT = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
STRATA = ("2011-12", "2019-20", "2020-21")


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def report(rows, arm, seasons, label):
    rr = [r for r in rows if r["season"] in seasons]
    if len(rr) < 100:
        return None
    y = np.array([r["y"] for r in rr])
    d = ll(y, [r["p_ctrl"] for r in rr]) - ll(y, [r[f"p_{arm}"] for r in rr])
    pan = S.Panel(np.array([r["season"] for r in rr]), d,
                  date=np.array([r["game_date"] for r in rr]), label=label)
    iid = S.paired_bootstrap(pan.d, B, SEED)
    sc = S.paired_bootstrap(pan.d, B, SEED, cluster=pan.season)
    t = S.cluster_mean_t_interval(pan.d, pan.season)
    dm = np.array([r[f"dm_{arm}"] for r in rr])
    return dict(arm=arm, group=label, n=len(rr), seasons=sorted(set(
        r["season"] for r in rr)),
        est=round(float(d.mean()), 6),
        iid=[round(iid["lo"], 6), round(iid["hi"], 6)], iid_sig=iid["sig"],
        season_cluster=[round(sc["lo"], 6), round(sc["hi"], 6)], sc_sig=sc["sig"],
        # a single-season stratum has K=1, so there is no cluster-mean t
        t_interval=[round(t.get("lo", float("nan")), 6),
                    round(t.get("hi", float("nan")), 6)],
        t_sig=bool(t.get("sig", False)), t_dof=int(t.get("dof", 0)),
        mde80=round(S.mde80(pan.d), 6),
        rms_dm=round(float(np.sqrt((dm ** 2).mean())), 4),
        per_season={s: round(float(
            (ll(np.array([r["y"] for r in rr if r["season"] == s]),
                [r["p_ctrl"] for r in rr if r["season"] == s])
             - ll(np.array([r["y"] for r in rr if r["season"] == s]),
                  [r[f"p_{arm}"] for r in rr if r["season"] == s])).mean()), 6)
            for s in sorted(set(r["season"] for r in rr))})


def verdict(cost):
    """The owner's terms.  `cost` is the CURRENT-corpus report (negative est =
    the arm hurts today)."""
    if cost is None:
        return "unmeasured"
    lo, hi = cost["season_cluster"]
    if lo > 0:
        return "FREE (positive today)"
    if hi < 0:
        return f"COSTS {-cost['est']:.5f} log loss (season-clustered CI excludes 0)"
    if abs(cost["est"]) < cost["mde80"] / 2:
        return (f"FREE TO CARRY within resolution: {cost['est']:+.5f}, "
                f"|est| < MDE80/2 = {cost['mde80']/2:.5f}")
    return (f"CHEAP: {cost['est']:+.5f}, season-clustered CI "
            f"({lo:+.5f},{hi:+.5f}) straddles 0, MDE80 {cost['mde80']:.5f}")


def main():
    con = connect(read_only=True)
    from history_scorable import sets
    pool, strat, _ = sets(con)
    want = sys.argv[1:] or (pool + strat)
    print("building travel state ...", flush=True)
    t0 = time.time()
    state = TV.build_state(con, since=_dt.date(2008, 7, 1))
    print(f"  {len(state)} team-games in {time.time()-t0:.0f}s", flush=True)
    allrows, meta = [], {}
    for s in want:
        t0 = time.time()
        rr, coefs, ninv, nost = season_run(con, s, state)
        allrows += rr
        meta[s] = {"n": len(rr), "n_travel_invalid": ninv, "n_no_state": nost,
                   "secs": round(time.time() - t0, 1), "coefs": coefs}
        print(f"{s}: n={len(rr)} travel_invalid={ninv} no_state={nost} "
              f"({meta[s]['secs']}s)", flush=True)
    con.close()
    hdr = list(allrows[0].keys())
    with open(ROOT / "data" / "history_shrink_pergame.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerows(allrows)

    seasons = sorted({r["season"] for r in allrows})
    hist = [s for s in seasons if s not in CERT and s not in STRATA]
    cert = [s for s in seasons if s in CERT]
    res = {"arms": ARMS_OUT, "teamhome_ridge": TEAMHOME_RIDGE,
           "meta": {s: {k: v for k, v in meta[s].items() if k != "coefs"}
                    for s in meta},
           "coefs": {s: meta[s]["coefs"] for s in meta},
           "results": {}}
    for arm in ARMS_OUT:
        res["results"][arm] = {
            "CERTIFIED_5 (cost — effect believed dead)": report(allrows, arm, cert, "certified"),
            "HISTORICAL_NEW (benefit — effect may be live)": report(allrows, arm, hist, "historical"),
            "ALL_POOLABLE": report(allrows, arm, cert + hist, "all"),
        }
        for s in STRATA:
            if s in seasons:
                res["results"][arm][f"STRATUM {s}"] = report(allrows, arm, [s], s)
        res["results"][arm]["OWNER_VERDICT"] = verdict(
            res["results"][arm]["CERTIFIED_5 (cost — effect believed dead)"])
    json.dump(res, open(ROOT / "data" / "history_shrink.json", "w"),
              indent=1, default=str)

    print("\n=== CARRY COST / BENEFIT ===")
    for arm in ARMS_OUT:
        print(f"\n{arm}")
        for g, r in res["results"][arm].items():
            if g == "OWNER_VERDICT" or r is None:
                continue
            print(f"  {g:48s} {r['est']:+.6f} "
                  f"iid({r['iid'][0]:+.6f},{r['iid'][1]:+.6f}){'S' if r['iid_sig'] else ' '} "
                  f"seas({r['season_cluster'][0]:+.6f},{r['season_cluster'][1]:+.6f})"
                  f"{'S' if r['sc_sig'] else ' '} "
                  f"MDE80 {r['mde80']:.6f} rms(dm) {r['rms_dm']:.3f}")
        print(f"  VERDICT: {res['results'][arm]['OWNER_VERDICT']}")
    print("\nwrote data/history_shrink.json, data/history_shrink_pergame.csv")


if __name__ == "__main__":
    main()
