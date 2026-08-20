#!/usr/bin/env python3
"""D170 phase B: load darko_history from the raw darko.app page cache.

FIXED BATCHED WRITER: parse every cached player series with NO DB connection
open, build one DataFrame, take the lock once for a DELETE + INSERT..SELECT.

Context: darko_history held 837 distinct players (354,600 rows) and its player
coverage RAMPED (32 distinct in 2010, 123 in 2015, 354 in 2020, 668 in 2024)
against seasons that carry 450-780 players. D161 measured the resulting DARKO
MINUTE coverage at 3.2% in 2007-08 rising to 100% by 2023-24, and
CompositionModel scores unrated players as league-average, so the historical
composition leg was largely inert. The cause was NOT a DARKO limit: we had only
ever fetched the modern roster universe. darko.app renders full daily series
for retired players back to 1996-11-01.
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

from nbapred.ingest.darko_history import FIELDS, RAW_DH, SCHEMA  # noqa: E402

DB = REPO / "data" / "nba.duckdb"


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
    files = sorted(RAW_DH.glob("*.json"))
    print(f"parsing {len(files)} cached darko player pages", flush=True)
    rows = []
    ok = empty = 0
    for n, f in enumerate(files):
        try:
            j = json.loads(f.read_text())
        except Exception:
            continue
        pid = int(j["player_id"])
        ser = j.get("series") or []
        if not ser:
            empty += 1
            continue
        ok += 1
        for r in ser:
            rows.append([pid, r["date"]] + [r.get(k) for k in FIELDS])
        if (n + 1) % 500 == 0:
            print(f"  {n+1}/{len(files)} parsed, {len(rows)} rows", flush=True)

    df = pd.DataFrame(rows, columns=["player_id", "date", *FIELDS])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.drop_duplicates(subset=["player_id", "date"])
    print(f"{ok} players with a series, {empty} empty/404; {len(df)} rows; "
          f"span {df.date.min()} .. {df.date.max()}", flush=True)

    t0 = time.time()
    con = _connect(read_only=False)
    try:
        con.execute(SCHEMA)
        con.register("df", df)
        con.execute("BEGIN")
        con.execute("DELETE FROM darko_history WHERE player_id IN "
                    "(SELECT DISTINCT player_id FROM df)")
        con.execute("INSERT INTO darko_history SELECT player_id, date, "
                    + ", ".join(FIELDS) + " FROM df")
        con.execute("COMMIT")
    finally:
        con.close()
    print(f"WRITE DONE in {time.time()-t0:.2f}s", flush=True)

    con = _connect(read_only=True)
    print(con.execute(
        "SELECT count(*) rows_, count(DISTINCT player_id) players_, "
        "min(date), max(date) FROM darko_history").fetchdf().to_string(),
        flush=True)
    print(con.execute(
        "SELECT year(date) y, count(DISTINCT player_id) pl FROM darko_history "
        "GROUP BY 1 ORDER BY 1").fetchdf().to_string(), flush=True)
    con.close()
    print("DARKO_HIST_LOAD_DONE", flush=True)


if __name__ == "__main__":
    main()
