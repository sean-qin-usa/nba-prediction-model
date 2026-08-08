#!/usr/bin/env python3
"""K19-T2 — D161's 19-season model evaluation RE-RUN AT THE T2 AVAILABILITY
TIER, now that D170 has put `game_inactives` on all 19 seasons.

D161 ran every season AVAILABILITY-BLIND and called every level a LOWER BOUND,
because `game_inactives` began 2022-23 and `injury_reports_pit` 2023-10-24.
D170 backfilled inactives to 2006-07 from BoxScoreSummaryV2 (the gap was
INGEST, not source) and extended the report feed. This script scores the same
model, the same way, at the honest T2 tier.

TIER DISCIPLINE (D158's rule: never silently mix tiers).  Each season is
LABELLED with the tier that was actually achievable on it:
    T2  = 5PM report OUT-set UNION official pregame inactives
    T2i = inactives only (no report feed exists for that season)
    BLIND = neither feed covers the season (out-sets EMPTY)
The out-set construction is COPIED VERBATIM from the honest default branch of
scripts/prod_by_season.py (D158), and `report_out_map` is IMPORTED from it so
the two cannot drift.  NO PLAYED-SET ORACLE: `player_game_stats` is never used
to build an out set anywhere in this file.

TWO ARMS, so the availability effect and the DARKO effect are not confounded:
    --tier blind   empty out-sets, CURRENT darko_history  (isolates DARKO)
    --tier t2      honest out-sets, CURRENT darko_history (the payoff)
D161's registered numbers are a THIRD point (blind + PRE-backfill DARKO) and
are not reproducible after the backfill; they are quoted from
data/k19_model.json, never recomputed.

HARNESS VALIDATION: `--tier blind` must reproduce scripts/k19_model.py run on
the SAME DB state to 5 decimals. That is the check that these edits changed
the out-sets and nothing else.

READ-ONLY on data/nba.duckdb.  Nothing in nbapred/ is modified, no default is
flipped, no gate is re-run, the eval corpus is not widened.

  TANK_SEASON_FLOOR=2020-21 python3 scripts/k19_t2.py --tier t2
"""
from __future__ import annotations

import argparse
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
from nbapred.model.production import fit_production             # noqa: E402
from prod_by_season import report_out_map                       # noqa: E402

SEASONS = ["%d-%02d" % (y, (y + 1) % 100) for y in range(2007, 2026)]
LN2 = 0.6931471805599453

# D161's franchise crosswalk, unchanged (see k19_model.py for the derivation).
FRANCHISE = {"SEA": "OKC", "NJN": "BKN", "NOH": "NOP",
             "NOK": "NOP", "VAN": "MEM", "CHH": "CHA"}


def fx(ab: str) -> str:
    return FRANCHISE.get(ab, ab)


def season_run(con, season: str, tier: str, rout, rcov, inact) -> dict:
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
    y, pp, pmk, rows = [], [], [], []
    n_sched = n_nomkt = 0
    model = comp = None
    last = None
    dk_frac, dk_abs, dk_n = [], [], []
    cov = {"report": 0, "inactives": 0, "either": 0, "neither": 0}
    n_out = []
    for gid in order:
        recs = by[gid]
        if len(recs) != 2:
            continue
        n_sched += 1
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
            last = gd
            act = [v for v in comp.players.values()
                   if (gd - v["last_played"]).days <= ROSTER_DAYS]
            if act:
                mtot = sum(v["trail_min"] for v in act)
                mcov = sum(v["trail_min"] for v in act if v["talent"] != 0.0)
                dk_frac.append(mcov / mtot if mtot else 0.0)
                dk_abs.append(float(np.mean([abs(v["talent"]) for v in act])))
                dk_n.append(int(sum(1 for v in act if v["talent"] != 0.0)))
        pmv = mkt.get((str(gd)[:10], fx(h.team_abbrev), fx(a.team_abbrev)))
        if pmv is None:
            n_nomkt += 1
            continue
        ds = str(gd)[:10]
        has_r = ds in rcov
        has_i = gid in inact
        cov["report"] += int(has_r)
        cov["inactives"] += int(has_i)
        cov["either"] += int(has_r or has_i)
        cov["neither"] += int(not (has_r or has_i))
        if tier == "blind":
            outs = {h.team_id: EMPTY, a.team_id: EMPTY}
        else:
            # VERBATIM from prod_by_season.py's honest default branch (D158)
            rot = {t: {p for p, d0 in comp.players.items()
                       if d0["team_id"] == t and (gd - d0["last_played"]).days <= ROSTER_DAYS}
                   for t in (h.team_id, a.team_id)}
            outs = {}
            for t, ab_ in ((h.team_id, h.team_abbrev), (a.team_id, a.team_abbrev)):
                # D171: `t2i` = inactives ONLY. It exists to PRICE the injury
                # report at each era's own absence density: on 2007-08..2017-18
                # t2i is identical to t2 (no report exists), so t2-vs-t2i on the
                # MODERN seasons is what the pre-2018-12-17 seasons are missing,
                # measured rather than extrapolated from the C-B2 contrast.
                o = EMPTY if tier == "t2i" else (
                    rout.get((ds, fx(ab_)), set()) & rot[t])     # T1: 5PM report
                o = o | (inact.get(gid, set()) & rot[t])         # T2: UNION inactives
                outs[t] = o                                      # empty where no feed
        bh, ba = b2b(h.team_id, gd), b2b(a.team_id, gd)
        p = model.p_home(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id],
                         gd, b2b_home=bh, b2b_away=ba)
        y.append(int(h.wl == "W"))
        pp.append(float(p))
        pmk.append(float(pmv))
        n_out.append(len(outs[h.team_id]))
        n_out.append(len(outs[a.team_id]))
        tsd = model.tank_diff(h.team_id, a.team_id, gd)
        rows.append((season, gid, ds, fx(h.team_abbrev), fx(a.team_abbrev),
                     y[-1], float(p), float(pmv), len(outs[h.team_id]),
                     len(outs[a.team_id]), round(float(tsd), 6),
                     round(float(model.tank_k), 4),
                     h.team_abbrev, a.team_abbrev))
    ya = np.array(y)
    llu = float(log_loss(ya, np.array(pp)))
    llm = float(log_loss(ya, np.array(pmk)))
    n = len(ya)
    if tier == "blind":
        lab = "BLIND"
    elif tier == "t2i":                       # D171: inactives-only arm
        lab = ("BLIND(no feed)" if cov["inactives"] == 0 else
               "T2i" if cov["inactives"] == n else "T2i-partial")
    elif cov["either"] == 0:
        lab = "BLIND(no feed)"
    elif cov["report"] == 0:
        lab = "T2i" if cov["inactives"] == n else "T2i-partial"
    else:
        lab = "T2" if cov["either"] == n else "T2-partial"
    return {"season": season, "n": n, "n_scheduled": n_sched,
            "n_no_market": n_nomkt, "tier_label": lab,
            "cov_report": cov["report"], "cov_inactives": cov["inactives"],
            "cov_either": cov["either"], "cov_neither": cov["neither"],
            "mean_outs_per_team": round(float(np.mean(n_out)) if n_out else 0.0, 3),
            "ll_us": round(llu, 5), "ll_mkt": round(llm, 5),
            "raw_gap": round(llu - llm, 5),
            "norm_gap_pct": round(100.0 * (llu - llm) / (LN2 - llm), 2),
            "darko_frac_roster_nonzero": round(float(np.mean(dk_frac)), 4)
            if dk_frac else 0.0,
            "darko_mean_abs_talent": round(float(np.mean(dk_abs)), 4)
            if dk_abs else 0.0,
            "darko_n_players_nonzero": int(np.mean(dk_n)) if dk_n else 0,
            "n_refits": len(dk_frac), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["blind", "t2", "t2i"], required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("seasons", nargs="*")
    a = ap.parse_args()
    tag = a.tag or a.tier

    print("=" * 100)
    print(f"K19-T2 — 19 SEASONS, AVAILABILITY TIER = {a.tier.upper()}")
    print("  out-sets: " + ("EMPTY on every season (D161's arm, re-run on the "
                            "CURRENT darko_history)" if a.tier == "blind" else
                            "5PM report UNION official pregame inactives, "
                            f"roster window {ROSTER_DAYS}d; EMPTY where no feed"))
    print("  NO played-set oracle is constructed anywhere in this file.")
    print(f"  TANK_SEASON_FLOOR={os.environ.get('TANK_SEASON_FLOOR', '(UNSET!)')}")
    for k in ("LATE_STATE", "TANK_TERM", "ORACLE_MINUTES", "INACTIVE_OUTS",
              "REPORT_OUTS", "ORACLE_PLAYED_OUTS", "OCT_BRIDGE",
              "OCT_BRIDGE_TRAIL", "COVID_GUARD", "FF_LUCK"):
        v = os.environ.get(k)
        if v is not None:
            print(f"  *** ENV {k}={v} ***")
    print("=" * 100, flush=True)

    con = connect(read_only=True)
    from nbapred.model.tanking import season_floor, floor_audit
    print(f"  tanking.season_floor(con) = {season_floor(con)}   "
          f"floor_audit = {floor_audit(con)}", flush=True)
    rout, rcov = report_out_map(con)
    inact = {}
    for g, p in con.execute(
            "SELECT game_id, player_id FROM game_inactives").fetchall():
        inact.setdefault(g, set()).add(int(p))
    print(f"  feeds loaded: report days={len(rcov)}, "
          f"inactive games={len(inact)}", flush=True)

    seasons = a.seasons or SEASONS
    out, allrows = [], []
    for s in seasons:
        t0 = time.time()
        r = season_run(con, s, a.tier, rout, rcov, inact)
        allrows += r.pop("rows")
        r["secs"] = round(time.time() - t0, 1)
        out.append(r)
        print(f"  {s}  n={r['n']:5d}  ll_us={r['ll_us']:.5f} "
              f"ll_mkt={r['ll_mkt']:.5f}  norm={r['norm_gap_pct']:+6.2f}%  "
              f"tier={r['tier_label']:<12s} rep={r['cov_report']:4d} "
              f"inact={r['cov_inactives']:4d} outs/tm={r['mean_outs_per_team']:.2f}  "
              f"darko={100*r['darko_frac_roster_nonzero']:5.1f}% "
              f"({r['darko_n_players_nonzero']:4d} pl)  ({r['secs']}s)",
              flush=True)
    con.close()

    oc = ROOT / "data" / f"k19_{tag}_pergame.csv"
    oj = ROOT / "data" / f"k19_{tag}.json"
    with open(oc, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["season", "game_id", "game_date", "home", "away", "y",
                    "p_us", "p_mkt", "n_out_home", "n_out_away", "tsd", "k",
                    "home_nba", "away_nba"])
        w.writerows(allrows)
    json.dump({"tier": a.tier, "seasons": out}, open(oj, "w"), indent=1)
    print(f"\nwrote {oc} ({len(allrows)} games) and {oj}")


if __name__ == "__main__":
    main()
