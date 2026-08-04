#!/usr/bin/env python3
"""GATE 1 DESIGN DIAGNOSTIC, PART 2 — before the pre-registration.

Three questions raised by part 1 (scripts/qg_starout_design.py), all still at
the DETECTOR/POOL level, no endpoint number:

  (1) CONTINUITY. Reproduce D145 §5's published table under D145's OWN
      definitions (clean star = >=12-min trailing mean >= 28 + clean freshness,
      no rotation-membership clause) so the new numbers can be read against the
      registered ones.

  (2) IS ARM B ARITHMETICALLY INERT ON THE STAR SIDE?  trail_min >= 28 over a
      10-ROW window requires >= 280 trailing minutes; a game cannot exceed 48
      minutes (the simulator's own ceiling; realized max in the corpus is
      measured here), so a firing player must already have >= ceil(280/max_min)
      played rows. If that is >= PLAYED_FLOOR the floor CANNOT bind and ARM B
      is a null BY ARITHMETIC on the star side. Measured, not argued.

  (3) THE POOL SIDE. ARM B also drops sub-floor players from the ROTATION POOL,
      which changes S and therefore the softmax lift. How many pool
      memberships move, and by how much does the applied (residual-scaled)
      lift move?  This is the only channel through which ARM B can act at all.

READ-ONLY. Writes data/qg_starout_design2.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from nbapred.engine import starout
from nbapred.db import connect
from nbapred.engine.starout import (FRESH_DAYS, MIN_TRAIL_GAMES, PLAYED_FLOOR,
                                    RESID_ATT_SCALE, ROT_TRAILING_MIN,
                                    STAR_TRAILING_MIN, TRAIL_GAMES)

SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
BIG = 12.0


def _d(x):
    return x.date() if hasattr(x, "date") else x


def main():
    con = connect(read_only=True)
    pm = con.execute("""
        SELECT s.game_id, s.team_id, s.player_id, s.seconds/60.0 AS m,
               s.rima + s.mida + s.thra AS att, g.game_date, g.season
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games
              WHERE wl IS NOT NULL) g USING (game_id)
        WHERE s.game_id LIKE '002%'""").fetchdf()
    pm["game_date"] = [_d(x) for x in pm["game_date"]]
    sched = defaultdict(list)
    for s, t, d in con.execute("""
        SELECT season, team_id, game_date FROM nba_games
        WHERE game_id LIKE '002%' AND wl IS NOT NULL""").fetchall():
        sched[(s, int(t))].append(_d(d))
    for k in sched:
        sched[k] = sorted(set(sched[k]))
    weights = starout.load_usage_weights()
    con.close()
    pm = pm[pm["season"].isin(SEASONS)]
    out = {}

    # ---- (2) arithmetic bound -------------------------------------------
    mx = float(pm["m"].max())
    need_star = int(np.ceil(STAR_TRAILING_MIN * TRAIL_GAMES / mx))
    need_rot = int(np.ceil(ROT_TRAILING_MIN * TRAIL_GAMES / mx))
    out["arithmetic_floor_bound"] = {
        "max_minutes_observed": round(mx, 2),
        "min_played_rows_implied_by_STAR_28": need_star,
        "min_played_rows_implied_by_ROT_15": need_rot,
        "PLAYED_FLOOR": PLAYED_FLOOR,
        "floor_can_bind_on_star_side": bool(need_star < PLAYED_FLOOR),
        "floor_can_bind_on_pool_side": bool(need_rot < PLAYED_FLOOR)}
    print("(2) arithmetic floor bound:", json.dumps(out["arithmetic_floor_bound"]))

    per = defaultdict(list)
    for r in pm.itertuples():
        per[(r.season, int(r.team_id), int(r.player_id))].append(
            (r.game_date, float(r.m or 0.0), float(r.att or 0.0)))
    for k in per:
        per[k].sort()

    # ---- (1) D145 continuity: clean star = >=12-min trailing mean --------
    d145 = []
    for (s, t, p), ent in per.items():
        dates = [d for d, _, _ in ent]
        mins = {d: m for d, m, _ in ent}
        big = [d for d, m, _ in ent if m >= BIG]
        if len(big) < MIN_TRAIL_GAMES:
            continue
        k_miss = 0
        for gd in sched[(s, t)]:
            if mins.get(gd, 0.0) > 0:
                k_miss = 0
                continue
            k_miss += 1
            prior = [d for d in dates if d < gd]
            if len(prior) < MIN_TRAIL_GAMES:
                continue
            tail = prior[-TRAIL_GAMES:]
            tm_cur = float(np.mean([mins[d] for d in tail]))
            fresh_cur = 0 < (gd - prior[-1]).days <= FRESH_DAYS
            pbig = [d for d in big if d < gd]
            if len(pbig) < MIN_TRAIL_GAMES:
                continue
            tm_cln = float(np.mean([mins[d] for d in pbig[-TRAIL_GAMES:]]))
            fresh_cln = 0 < (gd - pbig[-1]).days <= FRESH_DAYS
            d145.append((k_miss, tm_cur, tm_cln,
                         int(tm_cur >= STAR_TRAILING_MIN and fresh_cur),
                         int(tm_cln >= STAR_TRAILING_MIN and fresh_cln)))
    dd = pd.DataFrame(d145, columns=["k", "tm_cur", "tm_clean", "star_cur",
                                     "star_clean"])
    tab = []
    for lo, hi in [(1, 1), (2, 2), (3, 3), (4, 5), (6, 9)]:
        s = dd[(dd.k >= lo) & (dd.k <= hi) & (dd.star_clean == 1)]
        tab.append([f"{lo}-{hi}", int(len(s)), round(float(s.tm_clean.mean()), 2),
                    round(float(s.tm_cur.mean()), 2),
                    round(float(s.star_cur.mean()), 4)])
    out["d145_continuity"] = {
        "table": tab,
        "recall": round(float((dd.star_cur == 1).sum()
                              / max(int((dd.star_clean == 1).sum()), 1)), 4),
        "clean_star_out_total": int((dd.star_clean == 1).sum()),
        "false_negatives": int(((dd.star_cur == 0) & (dd.star_clean == 1)).sum()),
        "false_positives": int(((dd.star_cur == 1) & (dd.star_clean == 0)).sum()),
        "note": "D145 §5 registered recall 0.789 and P(fire) 0.842/0.760/0.708/"
                "0.689/0.705 BEFORE the D146 freshness fix; the guard now "
                "requires an actually-PLAYED last game, which is the only "
                "construction difference."}
    print("\n(1) D145 continuity (k, n clean stars, clean tm, cur tm, P(fire)):")
    for r in tab:
        print("   ", r)
    print("    recall", out["d145_continuity"]["recall"],
          "FN", out["d145_continuity"]["false_negatives"],
          "FP", out["d145_continuity"]["false_positives"])

    # ---- (3) pool side: how often does the floor drop a pool member? ------
    # For each (season, team, date) with >=1 out player, build ctrl and floor
    # pools and compare the applied lift.
    rows, nchg, npool_c, npool_f = [], 0, 0, 0
    for (s, t), sc in sched.items():
        if s not in SEASONS:
            continue
        ppl = {p: ent for (s2, t2, p), ent in per.items() if s2 == s and t2 == t}
        if not ppl:
            continue
        for gd in sc:
            stats = {}
            for p, ent in ppl.items():
                prior = [(d, m, a) for d, m, a in ent if d < gd]
                if len(prior) < MIN_TRAIL_GAMES:
                    continue
                tail = prior[-TRAIL_GAMES:]
                tm = float(np.mean([m for _, m, _ in tail]))
                ta = float(np.mean([a for _, _, a in tail]))
                npl = sum(1 for _, m, _ in tail if m > 0)
                played = [d for d, m, _ in prior if m > 0]
                fresh = bool(played) and 0 < (gd - played[-1]).days <= FRESH_DAYS
                stats[p] = (tm, len(tail), ta, npl, fresh)
            if not stats:
                continue
            outs = {p for p, ent in ppl.items()
                    if not any(d == gd and m > 0 for d, m, _ in ent)
                    and p in stats}
            cands = [p for p in outs
                     if stats[p][1] >= MIN_TRAIL_GAMES
                     and stats[p][0] >= STAR_TRAILING_MIN and stats[p][4]]
            if not cands:
                continue
            w = weights or {}
            star = max(cands, key=lambda p: w.get(p, 1.0))
            pool_c = {p for p, v in stats.items()
                      if p not in outs and v[1] >= MIN_TRAIL_GAMES
                      and v[0] >= ROT_TRAILING_MIN and v[4]}
            pool_f = {p for p in pool_c if stats[p][3] >= PLAYED_FLOOR}
            Lc = starout.compute_lift(w, pool_c, star, 1.0)
            Lf = starout.compute_lift(w, pool_f, star, 1.0)
            npool_c += len(pool_c); npool_f += len(pool_f)
            if pool_c != pool_f:
                nchg += 1
            rows.append((len(pool_c), len(pool_f), Lc, Lf,
                         1 + RESID_ATT_SCALE * (Lc - 1),
                         1 + RESID_ATT_SCALE * (Lf - 1)))
    R = pd.DataFrame(rows, columns=["np_c", "np_f", "L_c", "L_f", "app_c", "app_f"])
    out["pool_side"] = {
        "star_out_team_games": int(len(R)),
        "team_games_with_a_dropped_pool_member": int(nchg),
        "share": round(float(nchg / max(len(R), 1)), 4),
        "mean_pool_ctrl": round(float(R.np_c.mean()), 3),
        "mean_pool_floor": round(float(R.np_f.mean()), 3),
        "mean_softmax_lift_ctrl": round(float(R.L_c.mean()), 5),
        "mean_softmax_lift_floor": round(float(R.L_f.mean()), 5),
        "mean_applied_lift_ctrl": round(float(R.app_c.mean()), 6),
        "mean_applied_lift_floor": round(float(R.app_f.mean()), 6),
        "max_abs_applied_lift_diff": round(float((R.app_f - R.app_c).abs().max()), 6),
        "mean_abs_applied_lift_diff": round(float((R.app_f - R.app_c).abs().mean()), 8),
    }
    print("\n(3) pool side:", json.dumps(out["pool_side"], indent=1))
    json.dump(out, open("data/qg_starout_design2.json", "w"), indent=1, default=float)
    print("\nwrote data/qg_starout_design2.json")
    print("QG_STAROUT_DESIGN2_DONE", flush=True)


if __name__ == "__main__":
    main()
