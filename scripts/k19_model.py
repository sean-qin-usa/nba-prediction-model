#!/usr/bin/env python3
"""K19-MODEL — score the certified production stack on ALL 19 CONTIGUOUS
SCORABLE SEASONS (2007-08..2025-26, D160) at ONE CONSTANT AVAILABILITY TIER.

THE TIER IS **AVAILABILITY-BLIND** (empty OUT sets) ON EVERY SEASON.
`injury_reports_pit` starts 2023-10-24 and `game_inactives` starts 2022-23, so
honest availability information does NOT exist for 15 of the 19 seasons.  D158
showed that mixing tiers across seasons yields an uninterpretable pooled
number.  Running the whole frame blind is CONSISTENT, HONEST, and STRICTLY
WEAKER than what production ships in October (T2 on every game).  Every number
this script produces is therefore a LOWER BOUND on the model.

**NO PLAYED-SET ORACLE.**  No `player_game_stats`-derived OUT set is built
anywhere in this file.  D158's `ORACLE_PLAYED_OUTS` path is not reachable from
here.

The predictor is the LITERAL production predictor —
`fit_production(con, season, before=gd, w_comp=0.7).p_home(...)`, refit weekly,
exactly as `scripts/prod_by_season.py` does — with `outs` forced EMPTY.  The
market column is `odds_market.p_home_spread`, the same column every certified
capstone has used.

VALIDATION ANCHOR: 2021-22 has NO availability feed in existence, so the D158
certified run scored it BLIND already.  This script must reproduce D158's
2021-22 cell (n=1228, ll_us 0.63053, ll_mkt 0.60429) to five decimals.  If it
does not, the harness is wrong and nothing else here is believed.

ALSO MEASURED: **DARKO COVERAGE PER SEASON**, because D153 found the historical
readout partly measures our own talent-feed ramp (corr +0.79).  `darko_history`
floors at 2003-10-29 (D160 §5) but its player coverage RAMPS across the frame
(measured: 10 distinct players in 2007, 58 in 2012, 150 in 2016, 668 in 2024).
Per season this script records the fraction of rostered players carrying a
NON-ZERO DARKO talent value and the mean |talent| — the confound is measured,
not assumed away.

READ-ONLY on data/nba.duckdb.  Nothing in nbapred/ is modified, no default is
flipped, no gate is re-run, the eval corpus is not widened.

  TANK_SEASON_FLOOR=2020-21 python3 scripts/k19_model.py
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
from nbapred.model.production import fit_production             # noqa: E402

SEASONS = ["%d-%02d" % (y, (y + 1) % 100) for y in range(2007, 2026)]
LN2 = 0.6931471805599453
OUT_CSV = ROOT / "data" / "k19_pergame.csv"
OUT_JSON = ROOT / "data" / "k19_model.json"

# ---------------------------------------------------------------------------
# FRANCHISE-CODE CROSSWALK — a JOIN DEFECT FOUND BY THIS RUN, NOT A MODEL
# CHANGE.  `nba_games.team_abbrev` carries the abbreviation IN FORCE THAT
# SEASON; `odds_market` and `data/derived/odds_open.csv` carry the MODERN
# franchise code on every row.  The market join is
# (game_date, home_abbrev, away_abbrev), so before this fix EVERY GAME played
# by a relocated/renamed franchise silently dropped out of the historical
# frame.  Measured against the un-crosswalked run: 2007-08 lost 239 games
# (SEA+NJN+NOH), 2008-09..2011-12 lost ~164/season (NJN+NOH) and 2012-13 lost
# 83 (NOH) — ~1,100 games, i.e. most of a season, and NOT a random sample
# (three specific franchises, two of them mid-relocation and therefore exactly
# the tanking/roster-churn population the D73 term is about).
#   D153 (1) reported this as "MARKET COVERAGE is thin on the three oldest
#   seasons and the loss is not random" with counts 162 / 131 / 83 for
#   2010-11 / 2011-12 / 2012-13 — those are EXACTLY the NJN+NOH and NOH
#   schedules.  The mechanism was never named.  It is named here.
# Verified from the data, not asserted: for every season 2007-08..2025-26 the
# set difference (nba_games codes - odds codes) is exactly {} after this map.
FRANCHISE = {"SEA": "OKC",   # Seattle SuperSonics -> Oklahoma City, 2008-09
             "NJN": "BKN",   # New Jersey Nets     -> Brooklyn,      2012-13
             "NOH": "NOP",   # New Orleans Hornets -> Pelicans,      2013-14
             "NOK": "NOP",   # NO/Oklahoma City Hornets (2005-07; pre-frame)
             "VAN": "MEM",   # Vancouver Grizzlies (pre-frame)
             "CHH": "CHA"}   # original Charlotte Hornets (pre-frame)


def fx(ab: str) -> str:
    """Season abbreviation -> modern franchise code."""
    return FRANCHISE.get(ab, ab)

# D158's certified 2021-22 cell — that season ran BLIND (no feed exists), so it
# is the one cell where this blind harness MUST agree with the certified one.
D158_ANCHOR = {"2021-22": (1228, 0.63053, 0.60429)}


def season_run(con, season: str) -> dict:
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
            # DARKO COVERAGE AT THIS REFIT — the D153 confound, measured on the
            # population that actually enters the composition sum: players
            # inside the ROSTER WINDOW at this cutoff, MINUTE-WEIGHTED (D153's
            # own denominator, "minute coverage of the players who played").
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
        bh, ba = b2b(h.team_id, gd), b2b(a.team_id, gd)
        # >>> AVAILABILITY-BLIND: EMPTY OUT SETS, EVERY SEASON, NO EXCEPTIONS <<<
        p = model.p_home(h.team_id, a.team_id, EMPTY, EMPTY, gd,
                         b2b_home=bh, b2b_away=ba)
        y.append(int(h.wl == "W"))
        pp.append(float(p))
        pmk.append(float(pmv))
        tsd = model.tank_diff(h.team_id, a.team_id, gd)
        # home/away are emitted as MODERN franchise codes so the odds joins
        # downstream are 1:1; the season-of-record code is kept alongside.
        rows.append((season, gid, str(gd)[:10], fx(h.team_abbrev),
                     fx(a.team_abbrev), y[-1], float(p), float(pmv), 0, 0,
                     round(float(tsd), 6), round(float(model.tank_k), 4),
                     h.team_abbrev, a.team_abbrev))
    ya = np.array(y)
    llu = float(log_loss(ya, np.array(pp)))
    llm = float(log_loss(ya, np.array(pmk)))
    return {"season": season, "n": len(ya), "n_scheduled": n_sched,
            "n_no_market": n_nomkt,
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
    print("=" * 100)
    print("K19-MODEL — 19 CONTIGUOUS SEASONS AT ONE CONSTANT AVAILABILITY TIER")
    print("  AVAILABILITY TIER: **BLIND** (EMPTY out sets) on EVERY season.")
    print("  Honest availability data does not exist before 2022-23; a mixed")
    print("  tier is uninterpretable (D158).  These numbers are a LOWER BOUND.")
    print("  NO played-set oracle is constructed anywhere in this file.")
    print(f"  TANK_SEASON_FLOOR={os.environ.get('TANK_SEASON_FLOOR', '(UNSET!)')}"
          f"   ROSTER_DAYS={ROSTER_DAYS} (unused: out sets are empty)")
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

    seasons = sys.argv[1:] or SEASONS
    out, allrows = [], []
    for s in seasons:
        t0 = time.time()
        r = season_run(con, s)
        allrows += r.pop("rows")
        r["secs"] = round(time.time() - t0, 1)
        out.append(r)
        anc = ""
        if s in D158_ANCHOR:
            n0, u0, m0 = D158_ANCHOR[s]
            ok = (r["n"] == n0 and abs(r["ll_us"] - u0) < 5e-5
                  and abs(r["ll_mkt"] - m0) < 5e-5)
            anc = ("   [D158 ANCHOR " + ("EXACT" if ok else "*** MISMATCH ***")
                   + f" n={n0} us={u0} mkt={m0}]")
        print(f"  {s}  n={r['n']:5d}  ll_us={r['ll_us']:.5f} "
              f"ll_mkt={r['ll_mkt']:.5f}  raw={r['raw_gap']:+.5f} "
              f"norm={r['norm_gap_pct']:+6.2f}%  darko_cov="
              f"{100*r['darko_frac_roster_nonzero']:5.1f}% "
              f"({r['darko_n_players_nonzero']:4d} pl)  "
              f"nomkt={r['n_no_market']:4d}  ({r['secs']}s){anc}", flush=True)
    con.close()

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["season", "game_id", "game_date", "home", "away", "y",
                    "p_us", "p_mkt", "n_out_home", "n_out_away", "tsd", "k",
                    "home_nba", "away_nba"])
        w.writerows(allrows)
    json.dump({"tier": "AVAILABILITY-BLIND (empty out sets), all seasons",
               "lower_bound": True,
               "tank_season_floor_env": os.environ.get("TANK_SEASON_FLOOR"),
               "seasons": out}, open(OUT_JSON, "w"), indent=1)
    print(f"\nwrote {OUT_CSV} ({len(allrows)} games) and {OUT_JSON}")


if __name__ == "__main__":
    main()
