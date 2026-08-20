#!/usr/bin/env python3
"""D170 phase B: load game_inactives from the raw BoxScoreSummaryV2 cache.

FIXED BATCHED WRITER (D152/D160): register a DataFrame and INSERT..SELECT.
Per-row executemany on this build was measured at >600s for 60k rows against
0.82s batched, and a held write lock BLOCKS READERS. All parsing happens with
NO connection open; the lock is taken once, for one statement, and released.

Idempotent: DELETE the game_ids we are about to write, then insert.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import nbapred.threads  # noqa: E402
nbapred.threads.pin(1)

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402

DB = REPO / "data" / "nba.duckdb"
V2 = REPO / "data" / "raw" / "nba_api" / "boxscoresummaryv2"


def _connect(read_only: bool):
    for i in range(120):
        try:
            return duckdb.connect(str(DB), read_only=read_only)
        except duckdb.IOException as e:
            if "lock" not in str(e).lower():
                raise
            print(f"write lock held; yielding 60s ({i+1}/120)", flush=True)
            time.sleep(60)
    raise RuntimeError("lock held too long")


def main() -> None:
    # STRICTLY ADDITIVE. V2's InactivePlayers is empty for 35 games of 2024-25
    # and 1,203 of 2025-26 (the documented ">= 4/10/2025" V2 breakage); those
    # games were populated from BoxScoreSummaryV3 by
    # scripts/backfill_inactives_v3.py. Re-deriving them from V2 here would
    # replace good V3 rows with nothing. So games already present in
    # game_inactives are skipped outright.
    con = _connect(read_only=True)
    have = {r[0] for r in con.execute(
        "SELECT DISTINCT game_id FROM game_inactives").fetchall()}
    con.close()
    print(f"{len(have)} games already in game_inactives; those are left alone",
          flush=True)

    # ---- parse phase: no DB connection open ---------------------------------
    recs = []
    seen_games = set()
    files = sorted(V2.iterdir())
    print(f"parsing {len(files)} cached V2 responses", flush=True)
    for n, p in enumerate(files):
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        gid = (j.get("params") or {}).get("game_id", "")
        if not gid or gid in seen_games or gid in have:
            continue
        seen_games.add(gid)
        for rs in (j.get("response") or {}).get("resultSets", []):
            if rs.get("name") != "InactivePlayers":
                continue
            rows = rs.get("rowSet") or []
            if not rows:
                continue
            hdr = rs.get("headers") or []
            ip, it = hdr.index("PLAYER_ID"), hdr.index("TEAM_ID")
            for r in rows:
                recs.append((gid, int(r[ip]), int(r[it])))
        if (n + 1) % 5000 == 0:
            print(f"  {n+1}/{len(files)} parsed, {len(recs)} rows", flush=True)

    df = pd.DataFrame(recs, columns=["game_id", "player_id", "team_id"])
    df = df.drop_duplicates(subset=["game_id", "player_id"])
    gids = sorted(df.game_id.unique())
    print(f"parsed {len(df)} inactive rows across {len(gids)} games "
          f"({len(seen_games)} cached games, {len(seen_games)-len(gids)} with an "
          f"EMPTY InactivePlayers set)", flush=True)
    if df.empty:
        print("nothing to write"); return
    gdf = pd.DataFrame({"game_id": gids})

    # ---- write phase: one lock, two statements ------------------------------
    t0 = time.time()
    con = _connect(read_only=False)
    try:
        con.register("df", df)
        con.register("gdf", gdf)
        con.execute("BEGIN")
        con.execute("DELETE FROM game_inactives WHERE game_id IN (SELECT game_id FROM gdf)")
        con.execute("INSERT INTO game_inactives SELECT game_id, player_id, team_id FROM df")
        con.execute("COMMIT")
    finally:
        con.close()
    print(f"WRITE DONE in {time.time()-t0:.2f}s", flush=True)

    con = _connect(read_only=True)
    print(con.execute("""
        SELECT g.season, count(DISTINCT g.game_id) tot,
               count(DISTINCT i.game_id) cov,
               round(100.0*count(DISTINCT i.game_id)/count(DISTINCT g.game_id),1) pct
        FROM (SELECT DISTINCT season, game_id FROM nba_games WHERE game_id LIKE '002%') g
        LEFT JOIN (SELECT DISTINCT game_id FROM game_inactives) i USING(game_id)
        GROUP BY 1 ORDER BY 1""").fetchdf().to_string(), flush=True)
    con.close()
    print("INACTIVES_HIST_LOAD_DONE", flush=True)


if __name__ == "__main__":
    main()
