#!/usr/bin/env python3
"""GATE 2 DESIGN DIAGNOSTIC — softmax vs pool-arithmetic (null_u).

LIFT LEVEL ONLY. No CRPS / PIT endpoint number is produced here.

D129 registered, on post-cut rows with the PIT usage vector (n=9,315, 479
players): the softmax lift beats every naive baseline (vs flat-1.020 +0.04457
SIG; vs c_mean +0.02594 SIG; vs shuffled-lift +0.05017) but does NOT beat the
in-sample-optimal CONSTANT lift (+0.00194 ns) and LOSES to the
pool-arithmetic-only null (null_u lift 1.147): -0.00747 CI(-0.01469,-0.00038).
Reading: what may be load-bearing is the LIFT MAGNITUDE (pool arithmetic), not
the player-specific weighting from data/v2_usage.npz.

Before pre-registering the gate, measure how far apart the two lifts actually
are ON THE LIVE PATH — i.e. after the D82 cap [1.0,1.6] and the D83 residual
scaling (applied = 1 + 0.16*(L-1)), which is where the live stakes live:

  L_softmax = S/(S - w_star),  w = exp(u) from data/v2_usage.npz
  L_null    = N/(N - 1),       every player weighs 1  ->  PURE POOL ARITHMETIC
  L_trailatt= S/(S - a_star),  a = trailing-10 mean attempts (module fallback)

Also sizes the eval universe (star-out team-games, remaining-rotation rows) and
reports the mean absolute applied-lift difference, which bounds how much the
endpoint can possibly move.

READ-ONLY. Writes data/qg_nullu_design.json.
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

from nbapred.db import connect
from nbapred.engine import starout
from nbapred.engine.starout import (FRESH_DAYS, MIN_TRAIL_GAMES,
                                    RESID_ATT_SCALE, ROT_TRAILING_MIN,
                                    STAR_TRAILING_MIN, TRAIL_GAMES)

SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")


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
    assert weights, "data/v2_usage.npz must load for this diagnostic"
    print(f"loaded {len(pm)} 002 rows; v2_usage.npz has {len(weights)} players",
          flush=True)

    per = defaultdict(list)
    for r in pm.itertuples():
        per[(r.season, int(r.team_id), int(r.player_id))].append(
            (r.game_date, float(r.m or 0.0), float(r.att or 0.0)))
    for k in per:
        per[k].sort()

    recs = []
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
                played = [d for d, m, _ in prior if m > 0]
                stats[p] = (float(np.mean([m for _, m, _ in tail])), len(tail),
                            float(np.mean([a for _, _, a in tail])),
                            bool(played) and 0 < (gd - played[-1]).days <= FRESH_DAYS)
            if not stats:
                continue
            outs = {p for p, ent in ppl.items()
                    if not any(d == gd and m > 0 for d, m, _ in ent) and p in stats}
            cands = [p for p in outs if stats[p][1] >= MIN_TRAIL_GAMES
                     and stats[p][0] >= STAR_TRAILING_MIN and stats[p][3]]
            if not cands:
                continue
            pool = {p for p, v in stats.items()
                    if p not in outs and v[1] >= MIN_TRAIL_GAMES
                    and v[0] >= ROT_TRAILING_MIN and v[3]}
            att = {p: max(stats[p][2], 1e-3) for p in stats}
            att_def = float(np.median(list(att.values()))) if att else 1.0
            star_sm = max(cands, key=lambda p: weights.get(p, 1.0))
            star_ta = max(cands, key=lambda p: att.get(p, 0.0))
            L_sm = starout.compute_lift(weights, pool, star_sm, 1.0)
            L_nl = starout.compute_lift({}, pool, star_ta, 1.0)
            L_ta = starout.compute_lift(att, pool, star_ta, att_def)
            recs.append((s, int(t), gd, len(pool), int(star_sm), int(star_ta),
                         int(star_sm == star_ta), L_sm, L_nl, L_ta,
                         float(weights.get(star_sm, 1.0)),
                         int(star_sm in weights)))
    R = pd.DataFrame(recs, columns=["season", "team_id", "date", "n_pool",
                                    "star_sm", "star_ta", "same_star", "L_sm",
                                    "L_nl", "L_ta", "w_star", "star_in_npz"])
    for c in ("sm", "nl", "ta"):
        R[f"app_{c}"] = 1.0 + RESID_ATT_SCALE * (R[f"L_{c}"] - 1.0)
    R.to_csv("data/qg_nullu_rows.csv.gz", index=False, compression="gzip")

    out = {"star_out_team_games": int(len(R)), "seasons": list(SEASONS)}
    out["lifts"] = {
        "mean_L_softmax": round(float(R.L_sm.mean()), 5),
        "mean_L_null_u": round(float(R.L_nl.mean()), 5),
        "mean_L_trailatt": round(float(R.L_ta.mean()), 5),
        "corr_softmax_null": round(float(np.corrcoef(R.L_sm, R.L_nl)[0, 1]), 4),
        "corr_softmax_trailatt": round(float(np.corrcoef(R.L_sm, R.L_ta)[0, 1]), 4),
        "mean_pool": round(float(R.n_pool.mean()), 3),
        "capped_at_1.6_softmax": int((R.L_sm >= 1.5999).sum()),
        "capped_at_1.6_null": int((R.L_nl >= 1.5999).sum()),
        "same_star_share": round(float(R.same_star.mean()), 4),
        "star_in_npz_share": round(float(R.star_in_npz.mean()), 4)}
    out["applied"] = {
        "mean_applied_softmax": round(float(R.app_sm.mean()), 6),
        "mean_applied_null_u": round(float(R.app_nl.mean()), 6),
        "mean_applied_trailatt": round(float(R.app_ta.mean()), 6),
        "mean_abs_diff_sm_nl": round(float((R.app_sm - R.app_nl).abs().mean()), 6),
        "p95_abs_diff_sm_nl": round(float((R.app_sm - R.app_nl).abs().quantile(0.95)), 6),
        "max_abs_diff_sm_nl": round(float((R.app_sm - R.app_nl).abs().max()), 6),
        "mean_signed_diff_sm_nl": round(float((R.app_sm - R.app_nl).mean()), 6)}
    print("\nlifts:", json.dumps(out["lifts"], indent=1))
    print("applied (post D83 residual scaling):", json.dumps(out["applied"], indent=1))

    byp = []
    for lo, hi in [(0, 5), (6, 7), (8, 9), (10, 11), (12, 30)]:
        s = R[(R.n_pool >= lo) & (R.n_pool <= hi)]
        if not len(s):
            continue
        byp.append([f"{lo}-{hi}", int(len(s)), round(float(s.L_sm.mean()), 4),
                    round(float(s.L_nl.mean()), 4),
                    round(float((s.L_sm - s.L_nl).mean()), 4)])
    out["by_pool_size"] = byp
    print("\nby pool size [bin, n, L_softmax, L_null, diff]:")
    for r in byp:
        print("   ", r)

    ss = {}
    for s_ in SEASONS:
        s = R[R.season == s_]
        ss[s_] = [int(len(s)), round(float(s.L_sm.mean()), 4),
                  round(float(s.L_nl.mean()), 4),
                  round(float((s.app_sm - s.app_nl).abs().mean()), 6)]
    out["by_season"] = ss
    print("\nby season [n, L_sm, L_nl, mean|applied diff|]:")
    for k, v in ss.items():
        print("   ", k, v)

    json.dump(out, open("data/qg_nullu_design.json", "w"), indent=1, default=float)
    print("\nwrote data/qg_nullu_design.json")
    print("QG_NULLU_DESIGN_DONE", flush=True)


if __name__ == "__main__":
    main()
