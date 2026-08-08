#!/usr/bin/env python3
"""D170 phase A: NETWORK-ONLY fetch of darko.app player pages for the FULL
historical player universe. Writes nothing to DuckDB.

Root cause of the `darko_history` coverage ramp (D161 measured DARKO minute
coverage at 3.2% in 2007-08 rising to 100% by 2023-24): we only ever fetched
1,009 player pages — the modern roster universe — while `player_game_stats`
holds 3,934 distinct player_ids over 1996-97..2025-26. It was never a DARKO
limit. Probed 2026-08-04: darko.app server-renders the full daily series for
RETIRED players too (Kobe 1996-11-01..2016-04-13, n=1777; Duncan
1997-10-31..2016-05-12, n=1747; Iverson, Nash, Dirk, Yao, KG all likewise).

Each page is cached to data/raw/darko_history/{player_id}.json (ground truth,
including 404s so we never re-ask). bf_darko_hist_load.py does the one batched
write.

Usage: python3 scripts/bf_darko_hist_fetch.py --shard 0 --nshards 2
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
import requests  # noqa: E402

from nbapred.ingest.darko_history import RAW_DH, fetch_player  # noqa: E402

DB = REPO / "data" / "nba.duckdb"


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
    ap.add_argument("--interval", type=float, default=1.5)
    a = ap.parse_args()

    con = _ro()
    pids = [r[0] for r in con.execute(
        "SELECT DISTINCT player_id FROM player_game_stats ORDER BY 1").fetchall()]
    con.close()
    todo = [p for p in pids if not (RAW_DH / f"{p}.json").exists()]
    todo = [p for i, p in enumerate(todo) if i % a.nshards == a.shard]
    print(f"shard {a.shard}/{a.nshards}: universe={len(pids)} uncached={len(todo)}",
          flush=True)

    sess = requests.Session()
    ok = empty = fail = 0
    rows = 0
    t0 = time.time()
    import random
    for i, pid in enumerate(todo):
        try:
            series = fetch_player(pid, sess)
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"FAIL {pid}: {type(e).__name__} {str(e)[:90]}", flush=True)
            time.sleep(5)
            continue
        if series:
            ok += 1
            rows += len(series)
        else:
            empty += 1
        time.sleep(a.interval + random.uniform(0, 0.6))
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(todo)} ok={ok} empty={empty} fail={fail} "
                  f"rows={rows} eta={(len(todo)-i-1)*el/(i+1)/60:.0f}m", flush=True)
    print(f"SHARD {a.shard} DONE ok={ok} empty={empty} fail={fail} rows={rows} "
          f"elapsed={(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
