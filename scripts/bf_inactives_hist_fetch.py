#!/usr/bin/env python3
"""D170 phase A: NETWORK-ONLY fetch of BoxScoreSummaryV2 for historical games.

Writes nothing to DuckDB. Every response lands in the immutable raw cache
(data/raw/nba_api/boxscoresummaryv2/) which is the ground truth; the loader
(bf_inactives_hist_load.py) reads the cache and does ONE batched write.

Rationale (D170): game_inactives covered 2022-23..2025-26 only, but V2's
InactivePlayers result set is populated live back to 2006-07 (probed
2026-08-04: 2005-06 and earlier return 0 rows, 2006-07 onward return real
rows). That is an INGEST gap, not a source gap.

Sharding: --shard i --nshards N takes game_ids where index %% N == i, so
several workers can run without touching each other's work. The rate limiter
is per-process, so keep N small and --interval >= 1.0 to stay polite.

Usage:
  python3 scripts/bf_inactives_hist_fetch.py --shard 0 --nshards 3 --interval 1.0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import nbapred.threads  # noqa: E402
nbapred.threads.pin(1)

import duckdb  # noqa: E402

from nbapred.ingest import nba_stats  # noqa: E402
from nbapred.ingest.nba_stats import cached_endpoint  # noqa: E402

DB = REPO / "data" / "nba.duckdb"
# V2 InactivePlayers is empty for every game before 2006-07 (probed 2026-08-04).
FIRST_SEASON = "2006-07"


def _ro():
    for i in range(60):
        try:
            return duckdb.connect(str(DB), read_only=True)
        except duckdb.IOException as e:
            if "lock" not in str(e).lower():
                raise
            print(f"write lock held; yielding 60s ({i+1}/60)", flush=True)
            time.sleep(60)
    raise RuntimeError("could not open db read-only")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    nba_stats.STATS_MIN_INTERVAL = a.interval

    con = _ro()
    rows = con.execute(
        "SELECT DISTINCT season, game_id FROM nba_games "
        "WHERE game_id LIKE '002%' AND season >= ? ORDER BY game_id", [FIRST_SEASON]
    ).fetchall()
    have = {r[0] for r in con.execute(
        "SELECT DISTINCT game_id FROM game_inactives").fetchall()}
    con.close()

    todo = [g for _, g in rows if g not in have]
    todo = [g for i, g in enumerate(todo) if i % a.nshards == a.shard]
    if a.limit:
        todo = todo[:a.limit]
    print(f"shard {a.shard}/{a.nshards}: {len(todo)} games to fetch "
          f"(interval {a.interval}s)", flush=True)

    from nba_api.stats.endpoints import boxscoresummaryv2
    ok = fail = empty = cached = 0
    consec_fail = 0
    t0 = time.time()
    bucket = nba_stats.RAW_NBA / "boxscoresummaryv2"
    for i, gid in enumerate(todo):
        pre = (bucket / f"{nba_stats._cache_key({'game_id': gid})}.json").exists()
        try:
            raw = cached_endpoint(boxscoresummaryv2.BoxScoreSummaryV2,
                                  "boxscoresummaryv2", immutable=True,
                                  attempts=3, game_id=gid)
            consec_fail = 0
        except Exception as e:  # noqa: BLE001
            fail += 1
            consec_fail += 1
            print(f"FAIL {gid}: {type(e).__name__} {str(e)[:100]}", flush=True)
            if consec_fail >= 12:
                print(f"12 consecutive failures -> cooling off 300s", flush=True)
                time.sleep(300)
                consec_fail = 0
            continue
        cached += pre
        n = 0
        for rs in raw.get("resultSets", []):
            if rs.get("name") == "InactivePlayers":
                n = len(rs.get("rowSet") or [])
        ok += 1
        empty += (n == 0)
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            rate = (i + 1) / el
            print(f"  {i+1}/{len(todo)} ok={ok} empty={empty} fail={fail} "
                  f"cachehit={cached} {rate:.2f} g/s eta={(len(todo)-i-1)/rate/60:.0f}m",
                  flush=True)
    print(f"SHARD {a.shard} DONE ok={ok} empty={empty} fail={fail} "
          f"elapsed={(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
