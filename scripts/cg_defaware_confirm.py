#!/usr/bin/env python3
"""cg_defaware_confirm.py — confirmation arm for RE-RUN #1.

cg_zone_props.py re-gated the defender-aware props axis on 002-clean rates and
the RAW (team zone-allowance) arm PASSED: +0.0084 CI(+0.0029,+0.0143), n=1200,
2025-26, first-1200-after-the-cut sample.  The pre-D79 verdict on the same
construction was +0.0049 CI(-0.0067,+0.0162) NS (data/logs/defaware_stride.log,
2026-07-28) -> the reject may be a contamination artifact.

Before believing a reversal this script kills the two ways it could still be
fake:

1. SAMPLE.  Reproduces scripts/ablate_defender_props.py's own construction --
   stride-sampled across the FULL season timeline, not the weeks after a cut --
   and runs it on 2023-24, 2024-25 and 2025-26 separately.  A real effect is
   sign-consistent across seasons.

2. LEVEL BIAS (the D88 trap).  team_zone_defense measures allowance against
   HARDCODED league constants {rim .613, mid .44, thr .359}.  If those constants
   are stale the shift carries a uniform component and the "defense" arm is
   really a global shooting recalibration.  Arms:

     CTL    shipped props, no defense term
     RAW    + team_zone_defense shift                    (the gated arm)
     LEVEL  + the CROSS-SECTIONAL MEAN shift only, same for every opponent
            -> carries zero defensive information; whatever it earns is
               calibration, not defense
     REL    + (shift - mean shift)                       -> pure relative
            defense with the level component removed

   RAW passing while REL fails would mean the axis is still dead and we just
   re-derived a calibration constant.

Metric: points CRPS, paired bootstrap 2000x, 95% pct CI.  delta = CTL - arm, so
positive = the arm helps.  Read-only.  Writes data/cg_defaware_confirm.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import nbapred.db as _db

if os.environ.get("CG_DB"):
    _db.DB_PATH = Path(os.environ["CG_DB"])

from nbapred.db import connect                            # noqa: E402
from nbapred.engine.props import (apply_opp_defense,      # noqa: E402
                                  player_rates_from_stats, simulate_player,
                                  team_zone_defense)

OUT = Path("data/cg_defaware_confirm.json")
ZONES = ("rim", "mid", "thr")
ARMS = ("CTL", "RAW", "LEVEL", "REL")


def crps(samples, y):
    s = np.sort(samples); n = len(s)
    return float(np.mean(np.abs(s - y))
                 - 0.5 * (2 * np.arange(1, n + 1) - n - 1) @ s / n ** 2)


def paired(a, b, n_boot=2000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean()
                     for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def run_season(con, season, sims, max_eval, min_prior=8, min_min=20):
    pg = con.execute("""
        SELECT s.player_id, g.game_date, g.matchup, g.team_abbrev, s.pts
        FROM player_game_stats s JOIN nba_games g
          ON g.game_id=s.game_id AND g.team_id=s.team_id
        WHERE s.game_id LIKE '002%' AND s.seconds >= 720 AND g.season = ?
        ORDER BY g.game_date""", [season]).fetchdf()
    abbr2id = {r[1]: r[0] for r in con.execute(
        "SELECT DISTINCT team_id, team_abbrev FROM nba_games "
        "WHERE game_id LIKE '002%'").fetchall()}
    stride = max(1, len(pg) // (max_eval * 3))     # ablate_defender_props.py
    pg = pg.iloc[::stride]

    # PIT-safe level term: the mean shift is recomputed from the shifts already
    # seen at this point in the walk (expanding mean), never from the future.
    shifts_seen = {z: [] for z in ZONES}
    arms = {k: [] for k in ARMS}
    raw_shift_log = {z: [] for z in ZONES}
    n = 0
    for r in pg.itertuples():
        if n >= max_eval:
            break
        rates = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
        if rates is None or rates["n_games"] < min_prior or rates["proj_min"] < min_min:
            continue
        opp = None
        for tok in r.matchup.replace("vs.", "@").split("@"):
            tok = tok.strip()
            if tok and tok != r.team_abbrev:
                opp = tok
        opp_id = abbr2id.get(opp)
        if opp_id is None:
            continue
        shift = team_zone_defense(con, int(opp_id), before=r.game_date)
        if len(shifts_seen["rim"]) < 30:
            lvl = {z: 0.0 for z in ZONES}          # not enough history: no-op
        else:
            lvl = {z: float(np.mean(shifts_seen[z])) for z in ZONES}
        rel = {z: shift[z] - lvl[z] for z in ZONES}
        for z in ZONES:
            shifts_seen[z].append(shift[z]); raw_shift_log[z].append(shift[z])

        y = r.pts
        arms["CTL"].append(crps(simulate_player(rates, sims, seed=n)["points"], y))
        arms["RAW"].append(crps(simulate_player(
            apply_opp_defense(rates, shift), sims, seed=n)["points"], y))
        arms["LEVEL"].append(crps(simulate_player(
            apply_opp_defense(rates, lvl), sims, seed=n)["points"], y))
        arms["REL"].append(crps(simulate_player(
            apply_opp_defense(rates, rel), sims, seed=n)["points"], y))
        n += 1
    stats = {z: dict(mean=float(np.mean(raw_shift_log[z])),
                     sd=float(np.std(raw_shift_log[z]))) for z in ZONES} if n else {}
    return arms, n, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2023-24,2024-25,2025-26")
    ap.add_argument("--sims", type=int, default=3000)
    ap.add_argument("--max-eval", type=int, default=1200)
    a = ap.parse_args()

    t0 = time.time()
    con = connect(read_only=True)
    res, pooled = {}, {k: [] for k in ARMS}
    for season in a.seasons.split(","):
        arms, n, stats = run_season(con, season, a.sims, a.max_eval)
        if not n:
            print(f"[{season}] no rows"); continue
        blk = {"n": n, "shift_stats": stats,
               "mean_crps": {k: float(np.mean(v)) for k, v in arms.items()},
               "deltas_vs_CTL": {}}
        print(f"\n[{season}] n={n}   ({time.time()-t0:.0f}s)")
        print("  mean applied logit shift (0 => the hardcoded league constants "
              "are unbiased):")
        for z in ZONES:
            print(f"    {z}: mean {stats[z]['mean']:+.4f}  sd {stats[z]['sd']:.4f}")
        for k in ARMS:
            print(f"  CRPS {k:<6} {np.mean(arms[k]):.4f}")
            pooled[k].extend(arms[k])
        for k in ("RAW", "LEVEL", "REL"):
            d, lo, hi = paired(arms["CTL"], arms[k])
            v = "PASS" if lo > 0 else ("HARMS" if hi < 0 else "NS")
            blk["deltas_vs_CTL"][k] = dict(delta=d, lo=lo, hi=hi, verdict=v)
            print(f"    delta CTL-{k:<6} {d:+.4f} CI ({lo:+.4f},{hi:+.4f}) -> {v}")
        res[season] = blk

    print(f"\n[POOLED] n={len(pooled['CTL'])}")
    res["POOLED"] = {"n": len(pooled["CTL"]),
                     "mean_crps": {k: float(np.mean(v)) for k, v in pooled.items()},
                     "deltas_vs_CTL": {}}
    for k in ("RAW", "LEVEL", "REL"):
        d, lo, hi = paired(pooled["CTL"], pooled[k])
        v = "PASS" if lo > 0 else ("HARMS" if hi < 0 else "NS")
        res["POOLED"]["deltas_vs_CTL"][k] = dict(delta=d, lo=lo, hi=hi, verdict=v)
        print(f"  delta CTL-{k:<6} {d:+.4f} CI ({lo:+.4f},{hi:+.4f}) -> {v}")
    con.close()
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT}  ({time.time()-t0:.0f}s)")
    print("CG_DEFAWARE_DONE", flush=True)


if __name__ == "__main__":
    main()
