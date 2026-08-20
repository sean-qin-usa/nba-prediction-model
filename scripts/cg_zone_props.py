#!/usr/bin/env python3
"""cg_zone_props.py — RE-RUN #1 of the contaminated-gates audit.

Re-gates the DEFENDER-AWARE / ZONE-DEFENSE props axis on data that is clean of
BOTH bugs, and isolates how much of the old verdict the bugs owned.

Why this gate is on the list
---------------------------
* FEATURE_LEDGER: "Defender-aware (props) | NS x4, provisional"; D58 removed the
  opponent-defense shift from predict_today on that evidence.
* FEATURE_LEDGER: "def-RAPM team-aggregate (props) | worse than raw allowance".
* Both verdicts were produced BEFORE D79 (002 filters, 2026-07-30 21:37), so the
  shooter's rates AND the opponent-allowance aggregation both absorbed preseason
  and playoff games.
* nbapred/model/def_rapm.py reads `pbp['game']['homeTeamId']`, which the cached
  playbyplayv3 payload does not contain (cg_forensics: 0 of 4,800 games).  So
  `defenders = a5 if teamId == home_id else h5` handed 100% of 850,797 shots to
  the HOME five and 49.9% of them were the wrong five.  D81 fixed this in
  defense_zone.py / possessions_v2.py but NOT in def_rapm.py -- the def-RAPM
  rejection was fitted on a design matrix where the "defenders" column was a
  coin flip.

Arms (same eval rows, same seeds, paired)
-----------------------------------------
  CTL      shipped props, no opponent-defense term
  RAW      + props.team_zone_defense shift  (box-score path; D79-clean now)
  RAPM     + def-RAPM team aggregate, home ids derived from the ROTATION feed
           (the D81 fix ported into the def-RAPM collector)
  RAPMBUG  + def-RAPM team aggregate with the LITERAL shipped def_rapm.py
           collector (home_id=None) -- the same-run control that measures what
           the bug was worth

Eval: 2025-26 regular season, train = games <= 60th-pct date (def-RAPM fit),
test = later player-games with seconds>=720, n_games>=8, proj_min>=20.
Metric: points CRPS (lower better); reported deltas are CTL - arm, so POSITIVE
means the defense arm helped.  Paired bootstrap, 2000 resamples, 95% pct CI.

Read-only.  Writes data/cg_zone_props.json + charts nothing.
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
import orjson

import nbapred.db as _db

if os.environ.get("CG_DB"):
    _db.DB_PATH = Path(os.environ["CG_DB"])

from nbapred.db import connect                                   # noqa: E402
from nbapred.engine.props import (apply_opp_defense,             # noqa: E402
                                  player_rates_from_stats,
                                  simulate_player, team_zone_defense)
from nbapred.features.cache_index import game_index              # noqa: E402
from nbapred.features.defense_zone import _game_segments, _zone  # noqa: E402
from nbapred.features.possessions_v2 import _team_ids            # noqa: E402
from nbapred.features.stints import _elapsed                     # noqa: E402
from nbapred.model.def_rapm import fit_zone                      # noqa: E402

OUT = Path("data/cg_zone_props.json")
ZONES = ("rim", "mid", "thr")
# league zone make-rates used by ablate_defrapm_props.py (kept identical so the
# RAW arm reproduces the old gate's construction exactly)
LG = {"rim": 0.613, "mid": 0.44, "thr": 0.359}


def crps(samples, y):
    s = np.sort(samples); n = len(s)
    return float(np.mean(np.abs(s - y))
                 - 0.5 * (2 * np.arange(1, n + 1) - n - 1) @ s / n ** 2)


# ------------------------------------------------------- def-RAPM shot collect
def collect_shots(allowed_gids, mode: str):
    """Per-zone (def5, made) records.

    mode='fixed'  home id from the rotation feed  (the D81 fix)
    mode='buggy'  home id from pbp['game']['homeTeamId'] (always None) --
                  byte-for-byte the shipped nbapred/model/def_rapm.py logic
    """
    rots, pbps = game_index("gamerotation"), game_index("playbyplayv3")
    gids = sorted(g for g in (set(rots) & set(pbps)) if g in allowed_gids)
    recs = {z: [] for z in ZONES}
    for gid in gids:
        try:
            rot = orjson.loads(open(rots[gid], "rb").read())["response"]
            pbp = orjson.loads(open(pbps[gid], "rb").read())["response"]
        except Exception:
            continue
        if mode == "fixed":
            home_id, _ = _team_ids(rot, pbp)
            if home_id is None:
                continue
        else:
            home_id = pbp.get("game", {}).get("homeTeamId")
        segs = _game_segments(rot, pbp)
        if not segs:
            continue
        t0 = np.array([s[0] for s in segs])
        for a in pbp.get("game", {}).get("actions", []):
            if a.get("actionType") not in ("Made Shot", "Missed Shot"):
                continue
            t = _elapsed(a.get("period"), a.get("clock"))
            if t is None:
                continue
            k = int(np.searchsorted(t0, t, side="right") - 1)
            if k < 0 or k >= len(segs):
                continue
            _, _, h5, a5 = segs[k]
            defenders = a5 if a.get("teamId") == home_id else h5
            z = _zone(a.get("shotDistance"), a.get("shotValue"), a.get("subType"))
            if z:
                recs[z].append((tuple(defenders),
                                int(a.get("shotResult") == "Made")))
    return recs


def fit_defrapm(allowed_gids, mode, ridge=800.0):
    recs = collect_shots(allowed_gids, mode)
    out = {}
    for z in ZONES:
        ratings, _mu = fit_zone(recs[z], ridge)
        for p, v in ratings.items():
            out.setdefault(p, {})[z] = v
    return out, {z: len(recs[z]) for z in ZONES}


def team_shift_from_rapm(dr, mins_df):
    """Minute-weighted team zone rating -> logit make-shift, exactly the
    conversion ablate_defrapm_props.py used (rating/100*4, negated)."""
    team = {}
    for tid, grp in mins_df.groupby("team_id"):
        acc = {z: 0.0 for z in ZONES}; wsum = 0.0
        for r in grp.itertuples():
            pr = dr.get(r.player_id)
            if pr:
                for z in acc:
                    acc[z] += r.m * pr.get(z, 0.0)
                wsum += r.m
        if wsum > 0:
            team[tid] = {z: -acc[z] / wsum / 100.0 * 4 for z in acc}
    return team


def paired(a, b, n_boot=2000, seed=0):
    """delta = mean(a-b) with a percentile bootstrap CI (positive => b better
    when a=CTL crps and b=arm crps)."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean()
                     for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2025-26")
    ap.add_argument("--sims", type=int, default=3000)
    ap.add_argument("--max-eval", type=int, default=1200)
    a = ap.parse_args()

    t_start = time.time()
    con = connect(read_only=True)
    games = con.execute("""SELECT DISTINCT game_id, game_date FROM nba_games
        WHERE season=? AND game_id LIKE '002%' ORDER BY game_date""",
        [a.season]).fetchdf()
    cut = games.game_date.quantile(0.6)
    train_gids = set(games[games.game_date <= cut].game_id)
    mins = con.execute("""SELECT team_id, player_id, sum(seconds)/60.0 m
        FROM player_game_stats WHERE game_id IN (SELECT DISTINCT game_id
        FROM nba_games WHERE season=? AND game_date <= ? AND game_id LIKE '002%')
        GROUP BY 1,2""", [a.season, cut]).fetchdf()
    abbr2id = {r[1]: r[0] for r in con.execute(
        "SELECT DISTINCT team_id, team_abbrev FROM nba_games "
        "WHERE game_id LIKE '002%'").fetchall()}
    pg = con.execute("""SELECT s.player_id, g.game_date, g.matchup, g.team_abbrev, s.pts
        FROM player_game_stats s JOIN nba_games g
          ON g.game_id=s.game_id AND g.team_id=s.team_id
        WHERE s.game_id LIKE '002%' AND s.seconds>=720 AND g.game_date > ?
          AND g.season=? ORDER BY g.game_date""", [cut, a.season]).fetchdf()
    con.close()

    print(f"train games {len(train_gids)}  test player-games {len(pg)}", flush=True)
    dr_fix, cnt_fix = fit_defrapm(train_gids, "fixed")
    print(f"def-RAPM FIXED  shots {cnt_fix}  players {len(dr_fix)}", flush=True)
    dr_bug, cnt_bug = fit_defrapm(train_gids, "buggy")
    print(f"def-RAPM BUGGY  shots {cnt_bug}  players {len(dr_bug)}", flush=True)

    # how different are the two rating vectors? (bug-impact on the feature itself)
    common = sorted(set(dr_fix) & set(dr_bug))
    corr = {}
    for z in ZONES:
        x = np.array([dr_fix[p].get(z, 0.0) for p in common])
        y = np.array([dr_bug[p].get(z, 0.0) for p in common])
        corr[z] = float(np.corrcoef(x, y)[0, 1]) if len(x) > 3 else None
    print(f"fixed-vs-buggy def-RAPM rating corr: {corr}", flush=True)

    t_fix = team_shift_from_rapm(dr_fix, mins)
    t_bug = team_shift_from_rapm(dr_bug, mins)

    con = connect(read_only=True)
    arms = {"CTL": [], "RAW": [], "RAPM": [], "RAPMBUG": []}
    n = 0
    zero = {z: 0.0 for z in ZONES}
    for r in pg.itertuples():
        if n >= a.max_eval:
            break
        rates = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
        if rates is None or rates["n_games"] < 8 or rates["proj_min"] < 20:
            continue
        opp = None
        for tok in r.matchup.replace("vs.", "@").split("@"):
            tok = tok.strip()
            if tok and tok != r.team_abbrev:
                opp = tok
        opp_id = abbr2id.get(opp)
        if opp_id is None:
            continue
        raw = team_zone_defense(con, int(opp_id), before=r.game_date, league=LG)
        y = r.pts
        arms["CTL"].append(crps(simulate_player(rates, a.sims, seed=n)["points"], y))
        arms["RAW"].append(crps(simulate_player(
            apply_opp_defense(rates, raw), a.sims, seed=n)["points"], y))
        arms["RAPM"].append(crps(simulate_player(
            apply_opp_defense(rates, t_fix.get(opp_id, zero)),
            a.sims, seed=n)["points"], y))
        arms["RAPMBUG"].append(crps(simulate_player(
            apply_opp_defense(rates, t_bug.get(opp_id, zero)),
            a.sims, seed=n)["points"], y))
        n += 1
        if n % 200 == 0:
            print(f"  {n} rows  ({time.time()-t_start:.0f}s)", flush=True)
    con.close()

    res = {"season": a.season, "n": n, "sims": a.sims,
           "train_games": len(train_gids),
           "defrapm_shots_fixed": cnt_fix, "defrapm_shots_buggy": cnt_bug,
           "defrapm_fixed_vs_buggy_corr": corr,
           "mean_crps": {k: float(np.mean(v)) for k, v in arms.items()},
           "deltas_vs_CTL": {}}
    print(f"\nn={n}   mean points CRPS")
    for k, v in arms.items():
        print(f"  {k:8s} {np.mean(v):.4f}")
    print("\ndelta = CTL - arm   (positive => the defense arm HELPS)")
    for k in ("RAW", "RAPM", "RAPMBUG"):
        d, lo, hi = paired(arms["CTL"], arms[k])
        res["deltas_vs_CTL"][k] = dict(delta=d, lo=lo, hi=hi,
                                       verdict="PASS" if lo > 0 else
                                       ("HARMS" if hi < 0 else "NS"))
        print(f"  {k:8s} {d:+.4f}  CI ({lo:+.4f},{hi:+.4f})  -> "
              f"{res['deltas_vs_CTL'][k]['verdict']}")
    # did the bug own the def-RAPM verdict?
    d, lo, hi = paired(arms["RAPMBUG"], arms["RAPM"])
    res["fixed_minus_buggy"] = dict(delta=d, lo=lo, hi=hi)
    print(f"\nRAPM(fixed) vs RAPM(buggy): {d:+.4f} CI ({lo:+.4f},{hi:+.4f}) "
          f"(positive => the D81 fix improved the arm)")

    OUT.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT}  ({time.time()-t_start:.0f}s)")
    print("CG_ZONE_DONE", flush=True)


if __name__ == "__main__":
    main()
