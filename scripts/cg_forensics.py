#!/usr/bin/env python3
"""cg_forensics.py — CONTAMINATED-GATES forensics (bias class (c)).

Mechanically measures the blast radius of the bugs we later fixed, so the
contamination map in docs/CONTAMINATED_GATES.md rests on numbers, not on
reading the diff.  READ-ONLY: opens DuckDB with read_only=True, writes nothing
to the DB and nothing to nbapred/.

Probes
------
P1  D79 (missing 002 filters in props.py rate queries)
    How much non-regular-season history sat inside the trailing windows that
    every pre-2026-07-30 props gate consumed.

P2  D81 (cached playbyplayv3 has no homeTeamId)
    possessions_v2 as it currently sits in the DB: def_team constant, and the
    off_lineup <-> off_team agreement rate (a correct build is ~1.0; a build
    that swapped home-offense rows is ~0.5).

P3  D81 on defense_zone.accumulate: replay the BUGGY attribution
    (home_id=None) against the FIXED one on cached games and report what
    fraction of shots the buggy path handed to the home five.

P4  D81-family, STILL UNFIXED: scripts/fit_v2_usage.py + scripts/audit_usage_pit.py
    + nbapred/model/def_rapm.py all re-implement the same `homeTeamId` lookup.
    Measures the shot loss / misattribution those cause.

P5  Kalman-vs-EWMA universe (D12/D54): how the two shipped rate builders'
    history universes differed, pre- and post-D79.

Usage: python scripts/cg_forensics.py [--games N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import orjson

import nbapred.db as _db

if os.environ.get("CG_DB"):                      # forensic snapshot override
    _db.DB_PATH = Path(os.environ["CG_DB"])

from nbapred.db import connect                        # noqa: E402
from nbapred.features.cache_index import game_index    # noqa: E402
from nbapred.features.defense_zone import _game_segments, _zone   # noqa: E402
from nbapred.features.possessions_v2 import _team_ids             # noqa: E402
from nbapred.features.stints import _elapsed           # noqa: E402

OUT = Path("data/cg_forensics.json")


# ------------------------------------------------------------------ P1: D79
def p1_props_universe(con) -> dict:
    """Non-002 contamination of the props rate history."""
    by_kind = con.execute("""
        SELECT CASE WHEN game_id LIKE '002%' THEN 'regular'
                    WHEN game_id LIKE '001%' THEN 'preseason'
                    WHEN game_id LIKE '004%' THEN 'playoff'
                    ELSE 'other' END kind,
               count(*) nrows, count(DISTINCT game_id) ngames,
               count(DISTINCT player_id) nplayers
        FROM player_game_stats GROUP BY 1 ORDER BY 2 DESC
    """).fetchall()
    tot = sum(r[1] for r in by_kind)
    nonreg = sum(r[1] for r in by_kind if r[0] != 'regular')

    # The headline D79 statistic: at each season's Nov 1, how many players had
    # non-regular-season games inside the last 20 games props.py would have read
    # (pre-fix the query had no game-type filter and no >=720s filter on kalman).
    q = """
    WITH h AS (
      SELECT s.player_id, s.game_id, g.game_date,
             row_number() OVER (PARTITION BY s.player_id
                                ORDER BY g.game_date DESC, s.game_id DESC) rk
      FROM player_game_stats s
      JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g USING (game_id)
      WHERE g.game_date < CAST(? AS DATE) {floor}
    )
    SELECT count(DISTINCT player_id) nplayers,
           count(DISTINCT CASE WHEN game_id NOT LIKE '002%' THEN player_id END) dirty,
           sum(CASE WHEN game_id NOT LIKE '002%' THEN 1 ELSE 0 END) dirty_rows,
           count(*) nrows
    FROM h WHERE rk <= 20
    """
    windows = {}
    for season, nov1, floor in (("2023-24", "2023-11-01", "2023-09-01"),
                                ("2024-25", "2024-11-01", "2024-09-01"),
                                ("2025-26", "2025-11-01", "2025-09-01")):
        # (a) props.py's real window: trailing 20 over ALL history (what the
        #     pre-fix query saw); (b) season-to-date view, isolating October.
        ra = con.execute(q.format(floor=""), [nov1]).fetchone()
        rb = con.execute(q.format(floor="AND g.game_date >= CAST(? AS DATE)"),
                         [nov1, floor]).fetchone()
        windows[season] = dict(
            all_history=dict(players=ra[0], players_with_nonreg=ra[1],
                             nonreg_rows=ra[2], rows=ra[3],
                             frac_players=None if not ra[0] else round(ra[1] / ra[0], 4),
                             frac_rows=None if not ra[3] else round(ra[2] / ra[3], 4)),
            season_to_date=dict(players=rb[0], players_with_nonreg=rb[1],
                                nonreg_rows=rb[2], rows=rb[3],
                                frac_players=None if not rb[0] else round(rb[1] / rb[0], 4),
                                frac_rows=None if not rb[3] else round(rb[2] / rb[3], 4)))
    return {"rows_by_kind": [dict(kind=k, rows=n, games=g, players=p)
                             for k, n, g, p in by_kind],
            "rows_total": tot, "rows_non_regular": nonreg,
            "frac_non_regular": round(nonreg / tot, 5),
            "nov1_trailing20": windows}


# ------------------------------------------------------------------ P2: D81
def p2_possessions_v2(con) -> dict:
    n = con.execute("SELECT count(*) FROM possessions_v2").fetchone()[0]
    if not n:
        return {"rows": 0}
    dt = con.execute("""SELECT def_team, count(*) c FROM possessions_v2
                        GROUP BY 1 ORDER BY c DESC LIMIT 3""").fetchall()
    kinds = con.execute("""
        SELECT CASE WHEN game_id LIKE '002%' THEN 'regular'
                    WHEN game_id LIKE '001%' THEN 'preseason'
                    WHEN game_id LIKE '004%' THEN 'playoff' ELSE 'other' END,
               count(DISTINCT game_id), count(*)
        FROM possessions_v2 GROUP BY 1 ORDER BY 3 DESC""").fetchall()
    # off_lineup players should belong to off_team.  Unnest and join to the
    # authoritative (game_id, player_id) -> team_id map.
    agree = con.execute("""
        WITH pt AS (SELECT DISTINCT game_id, player_id, team_id FROM player_game_stats),
        x AS (SELECT p.game_id, p.off_team, CAST(trim(u.pid) AS BIGINT) pid
              FROM possessions_v2 p, UNNEST(str_split(p.off_lineup, ',')) AS u(pid))
        SELECT avg(CASE WHEN pt.team_id = x.off_team THEN 1.0 ELSE 0.0 END), count(*)
        FROM x JOIN pt ON pt.game_id = x.game_id AND pt.player_id = x.pid
    """).fetchone()
    return {"rows": n, "games": con.execute(
                "SELECT count(DISTINCT game_id) FROM possessions_v2").fetchone()[0],
            "def_team_top": [dict(def_team=d, rows=c) for d, c in dt],
            "def_team_all_zero": len(dt) == 1 and dt[0][0] == 0,
            "by_kind": [dict(kind=k, games=g, rows=r) for k, g, r in kinds],
            "off_lineup_matches_off_team": round(float(agree[0]), 4),
            "checked_player_slots": int(agree[1])}


# --------------------------------------------------- P3/P4: cache-side replay
def p3p4_cache_replay(n_games: int) -> dict:
    rots, pbps = game_index("gamerotation"), game_index("playbyplayv3")
    gids = [g for g in sorted(set(rots) & set(pbps)) if g.startswith("002")]
    rng = np.random.default_rng(11)
    if n_games and n_games < len(gids):
        gids = [gids[i] for i in sorted(rng.choice(len(gids), n_games, replace=False))]

    zone_buggy_home = zone_shots = 0            # defense_zone.accumulate
    zone_wrong = 0
    usage_kept_buggy = usage_kept_fixed = 0     # fit_v2_usage.collect
    hid_present = 0
    for gid in gids:
        try:
            rot = orjson.loads(open(rots[gid], "rb").read())["response"]
            pbp = orjson.loads(open(pbps[gid], "rb").read())["response"]
        except Exception:
            continue
        if pbp.get("game", {}).get("homeTeamId") is not None:
            hid_present += 1
        home_true, _ = _team_ids(rot, pbp)      # the D81 fix: ids from rotation
        segs = _game_segments(rot, pbp)
        if not segs or home_true is None:
            continue
        t0 = np.array([s[0] for s in segs])
        for a in pbp.get("game", {}).get("actions", []):
            if a.get("actionType") not in ("Made Shot", "Missed Shot"):
                continue
            t = _elapsed(a.get("period"), a.get("clock"))
            pid = a.get("personId")
            if t is None:
                continue
            k = int(np.searchsorted(t0, t, side="right") - 1)
            if k < 0 or k >= len(segs):
                continue
            _, _, h5, a5 = segs[k]
            shooter_team = a.get("teamId")
            # --- P3 defense_zone.accumulate: `a5 if shooter_team == home_id else h5`
            if _zone(a.get("shotDistance"), a.get("shotValue"), a.get("subType")):
                zone_shots += 1
                buggy_def = h5                              # home_id was None
                fixed_def = a5 if shooter_team == home_true else h5
                zone_buggy_home += 1                        # always the home five
                if buggy_def is not fixed_def:
                    zone_wrong += 1
            # --- P4 fit_v2_usage.collect: `off5 = h5 if teamId==home else a5`
            if pid:
                if pid in a5:                               # buggy: off5 == a5 always
                    usage_kept_buggy += 1
                off5_fixed = h5 if shooter_team == home_true else a5
                if pid in off5_fixed:
                    usage_kept_fixed += 1
    return {"games_replayed": len(gids),
            "pbp_with_homeTeamId": hid_present,
            "zone_shots": zone_shots,
            "zone_attributed_to_home_five_buggy": zone_buggy_home,
            "zone_misattributed_frac": None if not zone_shots
                                       else round(zone_wrong / zone_shots, 4),
            "usage_shots_kept_buggy": usage_kept_buggy,
            "usage_shots_kept_fixed": usage_kept_fixed,
            "usage_shot_loss_frac": None if not usage_kept_fixed else
                round(1 - usage_kept_buggy / usage_kept_fixed, 4)}


# ------------------------------------------------------------------ P5: D12
def p5_kalman_universe(con) -> dict:
    """History rows each shipped rate builder absorbed, pre- vs post-D79."""
    r = con.execute("""
        SELECT
          sum(CASE WHEN seconds > 0 THEN 1 ELSE 0 END)                            kal_pre,
          sum(CASE WHEN seconds > 0 AND game_id LIKE '002%' THEN 1 ELSE 0 END)    kal_post,
          sum(CASE WHEN seconds >= 720 THEN 1 ELSE 0 END)                         ewma_pre,
          sum(CASE WHEN seconds >= 720 AND game_id LIKE '002%' THEN 1 ELSE 0 END) ewma_post
        FROM player_game_stats
    """).fetchone()
    return {"kalman_hist_rows_pre_d79": r[0], "kalman_hist_rows_post_d79": r[1],
            "ewma_hist_rows_pre_d79": r[2], "ewma_hist_rows_post_d79": r[3],
            "universe_ratio_kalman_over_ewma_post_d79": round(r[1] / r[3], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400,
                    help="cached games to replay for P3/P4 (0 = all)")
    a = ap.parse_args()

    con = connect(read_only=True)
    res = {"P1_d79_props_universe": p1_props_universe(con),
           "P2_d81_possessions_v2": p2_possessions_v2(con),
           "P5_kalman_universe": p5_kalman_universe(con)}
    con.close()
    res["P3P4_d81_cache_replay"] = p3p4_cache_replay(a.games)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
