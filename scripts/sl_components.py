#!/usr/bin/env python3
"""SL-COMPONENTS — emit the PRODUCTION MARGIN's separable components, per game,
on all 19 contiguous scorable seasons, so the D166 walk-forward loop can be
re-run on progressively more primitive MODEL STRUCTURES.

WHY THIS EXISTS.  D166's loop re-selects only the BETTING CONFIG walk-forward.
The MODEL underneath is fixed and its ARCHITECTURE was chosen with full
knowledge of 2021-26.  This file makes the ablation ladder computable.

THE ONE STRUCTURAL FACT IT EXPLOITS: `production.Predictor.margin` is an EXACT
LINEAR SUM

    margin = 0.5*ff.margin_neutral + 0.5*comp.margin(home_edge=0)
             + he + b_hb2b*[home b2b] + b_ab2b*[away b2b]
             + tank_k*tank_diff  (+ latestate, DEFAULT OFF)

so one pass that emits the pieces supports every term-deletion variant.

HARNESS: `scripts/k19_model.py` verbatim in every respect that touches the
model — same weekly refit cadence, same AVAILABILITY-BLIND empty OUT sets on
every season, same franchise crosswalk, same TANK_SEASON_FLOOR pin, same
market join.  The ONLY additions are (i) a second FourFactors fit with the D62
carry disabled and (ii) an explicit `fit_schedule_layer` call so the schedule
coefficients are observable.  NO played-set oracle anywhere.

ANCHOR: `p_us = sigmoid(full/7.2)` must reproduce `data/k19_pergame.csv` to
1e-9 on all 22,804 games, and 2021-22 must reproduce D158's certified cell.

READ-ONLY on data/nba.duckdb.  nbapred/ UNTOUCHED.  Nothing ships.

  TANK_SEASON_FLOOR=2020-21 python3 scripts/sl_components.py
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

import nbapred.threads                                          # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                              # noqa: E402

from nbapred.db import connect                                  # noqa: E402
from nbapred.eval.metrics import log_loss                       # noqa: E402
from nbapred.model.composition import CompositionModel, ROSTER_DAYS  # noqa: E402
from nbapred.model.four_factors import FourFactors, factor_game_rows  # noqa: E402
from nbapred.model.production import (CARRY_CONT_DEFAULT, CARRY_W0,  # noqa: E402
                                      SCALE, _prev_season, continuity_map,
                                      fit_production, fit_schedule_layer,
                                      sigmoid)

from prod_by_season import report_out_map                       # noqa: E402

SEASONS = ["%d-%02d" % (y, (y + 1) % 100) for y in range(2007, 2026)]
LN2 = 0.6931471805599453
# --- D173: AVAILABILITY-TIER ARM.  SL_TIER defaults to "blind", which is
# --- D168's behaviour EXACTLY (empty out-sets on every season).  SL_TIER=t2
# --- uses the honest out-sets D170/D171 made available, built by the code
# --- imported VERBATIM from k19_t2.py::season_run.  Nothing else changes:
# --- same refit cadence, same crosswalk, same linear decomposition, same
# --- bridge-residual check.
TIER = os.environ.get("SL_TIER", "blind")
_TAG = os.environ.get("SL_TAG", "")
OUT_CSV = ROOT / "data" / f"sl_components{_TAG}.csv.gz"
OUT_JSON = ROOT / "data" / f"sl_components{_TAG}.json"

FRANCHISE = {"SEA": "OKC", "NJN": "BKN", "NOH": "NOP", "NOK": "NOP",
             "VAN": "MEM", "CHH": "CHA"}


def fx(ab: str) -> str:
    return FRANCHISE.get(ab, ab)


D158_ANCHOR = {"2021-22": (1228, 0.63053, 0.60429)}


def season_run(con, season: str, tier=None, rout=None, rcov=None,
               inact=None) -> dict:
    tier = tier or TIER
    rout = rout if rout is not None else {}
    rcov = rcov if rcov is not None else set()
    inact = inact if inact is not None else {}
    meta = con.execute(
        """SELECT game_id, team_id, team_abbrev, matchup, wl, game_date
           FROM nba_games WHERE season=? AND game_id LIKE '002%'
             AND wl IS NOT NULL ORDER BY game_date""", [season]).fetchdf()
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

    EMPTY: set = set()
    cov = {"report": 0, "inactives": 0, "either": 0}
    n_out_h, n_out_a = [], []
    y, pp, pmk, rows = [], [], [], []
    model = comp = ffnc = None
    he = bhb = bab = 0.0
    last = None
    prev_rows_cache = None
    dk_frac = []
    n_bridge = 0
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
            # ---- the SHIPPED predictor, verbatim -------------------------
            model = fit_production(con, season, before=gd, w_comp=0.7)
            comp = CompositionModel(con, before=gd)
            he, bhb, bab, _hd, _ad = fit_schedule_layer(con, gd)
            # ---- the four-factors leg WITH the D62 carry, rebuilt EXPLICITLY
            # (production.py:428-439 verbatim) so the October-bridge residual
            # is MEASURED rather than absorbed into a recovered coefficient.
            cont = continuity_map(con, season, before=gd)
            carry_rows = carry_w = None
            if cont is not None:
                if prev_rows_cache is None:
                    prev_rows_cache = factor_game_rows(
                        con, _prev_season(season), before=None)
                if prev_rows_cache:
                    carry_rows = prev_rows_cache
                    carry_w = [CARRY_W0 * cont.get(x["tid"], CARRY_CONT_DEFAULT)
                               for x in prev_rows_cache]
            ffc = FourFactors().fit(con, season, before=gd,
                                    carry_rows=carry_rows, carry_weights=carry_w)
            # ---- the SAME four-factors leg with the D62 CARRY DISABLED ----
            # (production.py:428-439 with carry_rows=None)
            ffnc = FourFactors().fit(con, season, before=gd)
            if not ffnc.ready:
                # <200 current rows and no carry: the carry-free structure
                # genuinely cannot fit yet.  Fall back to the prior season's
                # rows at FULL weight (i.e. a pure last-season model), which
                # is what a 2007-era builder would have had.  Flagged.
                if prev_rows_cache is None:
                    prev_rows_cache = factor_game_rows(
                        con, _prev_season(season), before=None)
                ffnc = FourFactors().fit(
                    con, season, before=gd, carry_rows=prev_rows_cache,
                    carry_weights=[1.0] * len(prev_rows_cache))
            last = gd
            act = [v for v in comp.players.values()
                   if (gd - v["last_played"]).days <= ROSTER_DAYS]
            if act:
                mtot = sum(v["trail_min"] for v in act)
                mcov = sum(v["trail_min"] for v in act if v["talent"] != 0.0)
                dk_frac.append(mcov / mtot if mtot else 0.0)
        pmv = mkt.get((str(gd)[:10], fx(h.team_abbrev), fx(a.team_abbrev)))
        if pmv is None:
            continue
        bh, ba = b2b(h.team_id, gd), b2b(a.team_id, gd)
        hid, aid = h.team_id, a.team_id
        ds = str(gd)[:10]
        if tier == "blind":
            # >>> AVAILABILITY-BLIND: EMPTY OUT SETS, EVERY SEASON <<<
            oh, oa = EMPTY, EMPTY
        else:
            # VERBATIM from k19_t2.py::season_run (itself verbatim from
            # prod_by_season.py's honest default branch, D158).
            has_r, has_i = ds in rcov, gid in inact
            cov["report"] += int(has_r)
            cov["inactives"] += int(has_i)
            cov["either"] += int(has_r or has_i)
            rot = {t: {pl for pl, d0 in comp.players.items()
                       if d0["team_id"] == t
                       and (gd - d0["last_played"]).days <= ROSTER_DAYS}
                   for t in (hid, aid)}
            os_ = {}
            for t, ab_ in ((hid, h.team_abbrev), (aid, a.team_abbrev)):
                o = rout.get((ds, fx(ab_)), set()) & rot[t]      # T1: 5PM report
                o = o | (inact.get(gid, set()) & rot[t])         # T2: UNION inactives
                os_[t] = o                                       # empty where no feed
            oh, oa = os_[hid], os_[aid]
        n_out_h.append(len(oh))
        n_out_a.append(len(oa))
        full = float(model.margin(hid, aid, oh, oa, gd,
                                  b2b_home=bh, b2b_away=ba))
        p = float(sigmoid(full / SCALE))
        tk = float(model.tank_k) * float(model.tank_diff(hid, aid, gd))
        sch_b2b = (bhb if bh else 0.0) + (bab if ba else 0.0)
        cm = float(comp.margin(hid, aid, oh, oa, gd, home_edge=0.0))
        fmnc = float(ffnc.margin_neutral(hid, aid))
        fmc = float(ffc.margin_neutral(hid, aid))
        # measured D84-A October-bridge residual: EXACTLY 0.0 on every game
        # where the bridge does not fire (the comp rotation is non-empty).
        brg = full - (0.5 * fmc + 0.5 * cm + he + sch_b2b + tk)
        if abs(brg) > 1e-9:
            n_bridge += 1
        y.append(int(h.wl == "W"))
        pp.append(p)
        pmk.append(float(pmv))
        rows.append((season, gid, str(gd)[:10], fx(h.team_abbrev),
                     fx(a.team_abbrev), y[-1], p, float(pmv),
                     round(full, 9), round(fmc, 9), round(fmnc, 9),
                     round(cm, 9), round(he, 9), round(sch_b2b, 9),
                     round(tk, 9), round(brg, 9)))
    ya = np.array(y)
    llu = float(log_loss(ya, np.array(pp)))
    llm = float(log_loss(ya, np.array(pmk)))
    n = len(ya)
    # tier label, VERBATIM from k19_t2.py::season_run
    if tier == "blind":
        lab = "BLIND"
    elif cov["either"] == 0:
        lab = "BLIND(no feed)"
    elif cov["report"] == 0:
        lab = "T2i" if cov["inactives"] == n else "T2i-partial"
    else:
        lab = "T2" if cov["either"] == n else "T2-partial"
    return {"season": season, "n": n, "tier": tier, "tier_label": lab,
            "cov_report": cov["report"], "cov_inactives": cov["inactives"],
            "mean_outs_per_team": round(
                float(np.mean(n_out_h + n_out_a)), 4) if n_out_h else 0.0,
            "ll_us": round(llu, 5), "ll_mkt": round(llm, 5),
            "raw_gap": round(llu - llm, 5),
            "norm_gap_pct": round(100.0 * (llu - llm) / (LN2 - llm), 2),
            "darko_frac": round(float(np.mean(dk_frac)), 4) if dk_frac else 0.0,
            "n_bridge_games": n_bridge, "rows": rows}


def main():
    print("=" * 100)
    print("SL-COMPONENTS — production margin decomposed, 19 seasons, "
          f"tier={TIER.upper()}")
    print(f"  TANK_SEASON_FLOOR={os.environ.get('TANK_SEASON_FLOOR', '(UNSET!)')}")
    print("  NO played-set oracle is constructed anywhere in this file.")
    print("=" * 100, flush=True)
    con = connect(read_only=True)
    from nbapred.model.tanking import floor_audit, season_floor
    print(f"  tanking.season_floor(con)={season_floor(con)} "
          f"floor_audit={floor_audit(con)}", flush=True)

    rout, rcov, inact = {}, set(), {}
    if TIER != "blind":
        rout, rcov = report_out_map(con)
        for g, p in con.execute(
                "SELECT game_id, player_id FROM game_inactives").fetchall():
            inact.setdefault(g, set()).add(int(p))
        print(f"  feeds loaded: report days={len(rcov)}, "
              f"inactive games={len(inact)}", flush=True)

    seasons = sys.argv[1:] or SEASONS
    out, allrows = [], []
    for s in seasons:
        t0 = time.time()
        r = season_run(con, s, TIER, rout, rcov, inact)
        allrows += r.pop("rows")
        r["secs"] = round(time.time() - t0, 1)
        out.append(r)
        anc = ""
        if s in D158_ANCHOR and TIER == "blind":
            n0, u0, m0 = D158_ANCHOR[s]
            ok = (r["n"] == n0 and abs(r["ll_us"] - u0) < 5e-5
                  and abs(r["ll_mkt"] - m0) < 5e-5)
            anc = "   [D158 ANCHOR " + ("EXACT" if ok else "*** MISMATCH ***") + "]"
        print(f"  {s}  n={r['n']:5d} ll_us={r['ll_us']:.5f} "
              f"ll_mkt={r['ll_mkt']:.5f} norm={r['norm_gap_pct']:+6.2f}% "
              f"tier={r['tier_label']:<12s} outs/tm={r['mean_outs_per_team']:.3f} "
              f"darko={100*r['darko_frac']:5.1f}%  ({r['secs']}s){anc}",
              flush=True)
    con.close()

    import gzip
    with gzip.open(OUT_CSV, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["season", "game_id", "game_date", "home", "away", "y",
                    "p_us", "p_mkt", "full", "fm_carry", "fm_nocarry", "cm",
                    "he", "b2b", "tk", "bridge"])
        w.writerows(allrows)
    json.dump({"tier": ("AVAILABILITY-BLIND (empty out sets), all seasons"
                        if TIER == "blind" else
                        "T2 where the feeds exist (5PM report UNION official "
                        "pregame inactives); labelled per season"),
               "tier_arg": TIER,
               "scale": SCALE, "seasons": out}, open(OUT_JSON, "w"), indent=1)
    print(f"\nwrote {OUT_CSV} ({len(allrows)} games) and {OUT_JSON}")


if __name__ == "__main__":
    main()
