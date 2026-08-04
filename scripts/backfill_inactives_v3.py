#!/usr/bin/env python3
"""Backfill game_inactives (+ game_officials top-up) via BoxScoreSummaryV3.

ROOT CAUSE (2026-07-31 loose-ends sweep, thread-6): the D69 data gotcha —
game_inactives covering only 20/1230 of 2025-26 — is NOT a load gap. The raw
cache holds 1223/1230 boxscoresummaryv2 responses for 2025-26 and every one
was faithfully loaded: the SOURCE 'InactivePlayers' result set is empty
(19 of the 20 non-empty games are the last games of the season plus the two
openers). nba_api itself warns: "BoxScoreSummaryV2 has known data availability
issues. Data may be missing for games on or after 4/10/2025" — exactly
matching the 35 missing late-2024-25 games and the 1203 empty 2025-26 games.
A live refetch (2026-07-31) confirmed V2 is still empty for those games.
BoxScoreSummaryV3 (boxScoreSummary.{home,away}Team.inactives, .officials)
HAS the data — verified on 0022500003 (4 home / 3 away inactives).

Phases (network only in phase 2; ONE short write at the end):
  1. V2 cache sweep — collect InactivePlayers rows for any 002 game in
     nba_games missing from game_inactives whose cached V2 response is
     non-empty (this picks up 2022-23 if the chained officials retry job
     fetched it over V2, which works for pre-4/10/2025 games).
  2. V3 fetch via cached_endpoint (throttled >=0.65s, raw JSON cached as
     ground truth) for games still missing; parse inactives + officials.
     Failures are logged and skipped — coverage is reported, never assumed.

Single-writer discipline: read snapshot up front, all network before the
write, write connection opened once at the end with lock-retry (concurrent
agents hold short write locks on nba.duckdb).

Run standalone anytime; it is idempotent (INSERT OR REPLACE, immutable cache).
Intended chaining: after data/logs/summary_retry.log prints
SUMMARY_RETRY_DONE (the V2 officials retry that follows backfill_multi).
Marker on completion: INACTIVES_V3_DONE
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import duckdb  # noqa: E402

from nbapred.db import connect  # noqa: E402
from nbapred.ingest.nba_stats import RAW_NBA, cached_endpoint  # noqa: E402

V2_DIR = RAW_NBA / "boxscoresummaryv2"
LOCK_RETRIES = 60          # x 60s = up to 1h waiting on other writers
FETCH_FAIL_CAP = 200       # stop hitting the network after this many failures


def _connect_retry(read_only: bool):
    for i in range(LOCK_RETRIES):
        try:
            return connect(read_only=read_only) if not read_only else \
                duckdb.connect(str(REPO / "data" / "nba.duckdb"), read_only=True)
        except duckdb.IOException as e:
            if "lock" not in str(e).lower():
                raise
            print(f"db locked ({'ro' if read_only else 'rw'}), retry {i+1}/{LOCK_RETRIES} in 60s",
                  flush=True)
            time.sleep(60)
    raise RuntimeError("could not acquire duckdb connection (lock held too long)")


def main() -> None:
    con = _connect_retry(read_only=True)
    all_games = {r[0] for r in con.execute(
        "SELECT DISTINCT game_id FROM nba_games WHERE game_id LIKE '002%'").fetchall()}
    have_inact = {r[0] for r in con.execute(
        "SELECT DISTINCT game_id FROM game_inactives").fetchall()}
    have_off = {r[0] for r in con.execute(
        "SELECT DISTINCT game_id FROM game_officials").fetchall()}
    con.close()
    missing = sorted(all_games - have_inact)
    print(f"games in nba_games: {len(all_games)}; missing from game_inactives: {len(missing)}",
          flush=True)

    inact_rows: dict[str, list[tuple[str, int, int]]] = {}   # gid -> [(gid, player, team)]
    off_rows: dict[str, list[tuple]] = {}                    # gid -> officials tuples

    # ---- phase 1: V2 cache sweep -------------------------------------------
    missing_set = set(missing)
    swept = 0
    if V2_DIR.exists():
        for p in V2_DIR.iterdir():
            try:
                j = json.loads(p.read_text())
            except Exception:
                continue
            gid = (j.get("params") or {}).get("game_id", "")
            if gid not in missing_set or gid in inact_rows:
                continue
            for rs in (j.get("response") or {}).get("resultSets", []):
                if rs.get("name") != "InactivePlayers":
                    continue
                hdr = rs.get("headers") or []
                rows = rs.get("rowSet") or []
                if not rows:
                    continue
                ip, it = hdr.index("PLAYER_ID"), hdr.index("TEAM_ID")
                inact_rows[gid] = [(gid, int(r[ip]), int(r[it])) for r in rows]
                swept += 1
    print(f"phase 1 (V2 cache sweep): {swept} games recovered from cache", flush=True)

    # ---- phase 2: V3 fetch for the rest ------------------------------------
    still = [g for g in missing if g not in inact_rows]
    print(f"phase 2 (V3): fetching {len(still)} games", flush=True)
    from nba_api.stats.endpoints import boxscoresummaryv3
    fails = 0
    for i, gid in enumerate(still):
        if fails >= FETCH_FAIL_CAP:
            print(f"fail cap {FETCH_FAIL_CAP} reached; stopping network phase", flush=True)
            break
        try:
            raw = cached_endpoint(boxscoresummaryv3.BoxScoreSummaryV3,
                                  "boxscoresummaryv3", immutable=True, game_id=gid)
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"V3 fetch failed {gid}: {type(e).__name__} {str(e)[:120]}", flush=True)
            continue
        bs = raw.get("boxScoreSummary") or {}
        rows = []
        for side in ("homeTeam", "awayTeam"):
            t = bs.get(side) or {}
            tid = t.get("teamId")
            for pl in t.get("inactives") or []:
                if pl.get("personId") is not None and tid is not None:
                    rows.append((gid, int(pl["personId"]), int(tid)))
        if rows:
            inact_rows[gid] = rows
        if gid not in have_off:
            offs = [(gid, int(o["personId"]), o.get("firstName"), o.get("familyName"),
                     str(o.get("jerseyNum", "")).strip(), "boxsummary")
                    for o in bs.get("officials") or [] if o.get("personId") is not None]
            if offs:
                off_rows[gid] = offs
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(still)} fetched (ok={len(inact_rows)} fail={fails})",
                  flush=True)

    # ---- single short write -------------------------------------------------
    n_i = sum(len(v) for v in inact_rows.values())
    n_o = sum(len(v) for v in off_rows.values())
    print(f"writing: {len(inact_rows)} games / {n_i} inactive rows; "
          f"{len(off_rows)} games / {n_o} officials rows", flush=True)
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc)
    wcon = _connect_retry(read_only=False)
    try:
        for rows in inact_rows.values():
            for gid, pid, tid in rows:
                wcon.execute("INSERT OR REPLACE INTO game_inactives VALUES (?,?,?)",
                             [gid, pid, tid])
        for offs in off_rows.values():
            for gid, oid, fn, ln, jn, src in offs:
                wcon.execute("INSERT OR REPLACE INTO game_officials VALUES (?,?,?,?,?,?,?)",
                             [gid, oid, fn, ln, jn, src, now])
        print("== game_inactives coverage after load ==", flush=True)
        for sea, games, nrows in wcon.execute(
                "SELECT substr(game_id,1,5), count(DISTINCT game_id), count(*) "
                "FROM game_inactives GROUP BY 1 ORDER BY 1").fetchall():
            print(f"  {sea}: {games} games, {nrows} rows", flush=True)
    finally:
        wcon.close()
    print("INACTIVES_V3_DONE", flush=True)


if __name__ == "__main__":
    main()
