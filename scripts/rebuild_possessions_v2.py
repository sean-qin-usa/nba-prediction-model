#!/usr/bin/env python3
"""D100 JOB-2a: force-rebuild possessions_v2 with the D81-fixed parser.

The table in the DB was written 2026-07-28 (pre-D81): `def_team = 0` on 100% of
375,295 rows and off_lineup/off_team agreement 0.5004 (a coin flip). Because
`build()` skipped every game already present, no ordinary run could ever repair
it. This calls the new force path, then re-measures the two forensic statistics
`scripts/cg_forensics.py::p2_possessions_v2` reports, so the rebuild is
self-verifying.

SINGLE-WRITER: run only when no loader/backfill holds the DuckDB write lock.

  python3 scripts/rebuild_possessions_v2.py            # full corpus, forced
  python3 scripts/rebuild_possessions_v2.py --verify   # measure only, no write
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nbapred.db import connect                       # noqa: E402
from nbapred.features import possessions_v2 as p2    # noqa: E402


def measure(con) -> dict:
    n = con.execute("SELECT count(*) FROM possessions_v2").fetchone()[0]
    if not n:
        return {"rows": 0}
    games = con.execute("SELECT count(DISTINCT game_id) FROM possessions_v2").fetchone()[0]
    zero = con.execute("SELECT count(*) FROM possessions_v2 WHERE def_team = 0 "
                       "OR def_team IS NULL").fetchone()[0]
    kinds = con.execute("""
        SELECT CASE WHEN game_id LIKE '002%' THEN 'regular'
                    WHEN game_id LIKE '001%' THEN 'preseason'
                    WHEN game_id LIKE '004%' THEN 'playoff' ELSE 'other' END k,
               count(DISTINCT game_id), count(*)
        FROM possessions_v2 GROUP BY 1 ORDER BY 3 DESC""").fetchall()
    agree = con.execute("""
        WITH pt AS (SELECT DISTINCT game_id, player_id, team_id FROM player_game_stats),
        x AS (SELECT p.game_id, p.off_team, CAST(trim(u.pid) AS BIGINT) pid
              FROM possessions_v2 p, UNNEST(str_split(p.off_lineup, ',')) AS u(pid))
        SELECT avg(CASE WHEN pt.team_id = x.off_team THEN 1.0 ELSE 0.0 END), count(*)
        FROM x JOIN pt ON pt.game_id = x.game_id AND pt.player_id = x.pid""").fetchone()
    dagree = con.execute("""
        WITH pt AS (SELECT DISTINCT game_id, player_id, team_id FROM player_game_stats),
        x AS (SELECT p.game_id, p.def_team, CAST(trim(u.pid) AS BIGINT) pid
              FROM possessions_v2 p, UNNEST(str_split(p.def_lineup, ',')) AS u(pid))
        SELECT avg(CASE WHEN pt.team_id = x.def_team THEN 1.0 ELSE 0.0 END), count(*)
        FROM x JOIN pt ON pt.game_id = x.game_id AND pt.player_id = x.pid""").fetchone()
    ppp = con.execute("SELECT avg(points), avg(seconds) FROM possessions_v2").fetchone()
    return {"rows": n, "games": games, "def_team_zero_or_null": zero,
            "by_kind": [dict(kind=k, games=g, rows=r) for k, g, r in kinds],
            "off_lineup_matches_off_team": round(float(agree[0]), 4),
            "def_lineup_matches_def_team": round(float(dagree[0]), 4),
            "checked_off_slots": int(agree[1]), "checked_def_slots": int(dagree[1]),
            "mean_points": round(float(ppp[0]), 4), "mean_seconds": round(float(ppp[1]), 2)}


def show(tag, m):
    print(f"\n=== {tag} ===")
    for k, v in m.items():
        print(f"  {k}: {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="measure only, do not write")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=500)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    con = connect(read_only=True)
    before = measure(con)
    con.close()
    show("BEFORE", before)
    if a.verify:
        return
    t0 = time.time()
    res = p2.build(connect, limit=a.limit, force=True, chunk_games=a.chunk)
    print(f"\nbuild: {res} in {time.time()-t0:.0f}s")
    con = connect(read_only=True)
    after = measure(con)
    con.close()
    show("AFTER", after)
    ok = (after["def_team_zero_or_null"] == 0
          and after["off_lineup_matches_off_team"] > 0.98
          and after["def_lineup_matches_def_team"] > 0.98)
    print(f"\nREBUILD_{'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
