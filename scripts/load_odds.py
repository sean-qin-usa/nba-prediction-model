#!/usr/bin/env python3
"""Batch-load odds JSONL (raw capture) into DuckDB odds_quotes. Idempotent:
re-loads a file by deleting its rows first (provenance column raw_file).

Usage: python scripts/load_odds.py [YYYY-MM-DD ...]   (default: all files)
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nbapred.config import RAW_ODDS
from nbapred.db import connect


def _ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def flatten(record: dict, raw_file: str, ingest_ts) -> list[tuple]:
    rows = []
    snap = _ts(record["snapshot_ts"])
    events = record["data"] if isinstance(record["data"], list) else [record["data"]]
    for ev in events:
        for bk in ev.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                for oc in mkt.get("outcomes", []):
                    rows.append((
                        snap, ingest_ts, "the-odds-api", ev.get("id"),
                        _ts(ev.get("commence_time")), ev.get("home_team"),
                        ev.get("away_team"), bk.get("key"),
                        _ts(mkt.get("last_update") or bk.get("last_update")),
                        mkt.get("key"), oc.get("name"), oc.get("description"),
                        float(oc.get("price")), oc.get("point"), raw_file,
                    ))
    return rows


def _connect_retry(attempts: int = 8, wait_s: int = 120):
    """build_features (09:40, up to 60 min) can hold the writer lock; retry
    instead of dying so the cron slot is robust to any schedule drift."""
    import time
    for i in range(attempts):
        try:
            return connect()
        except Exception as e:
            if "lock" not in str(e).lower() or i == attempts - 1:
                raise
            print(f"writer lock held, retry {i + 1}/{attempts} in {wait_s}s")
            time.sleep(wait_s)


def main(days: list[str]) -> None:
    files = sorted(RAW_ODDS.glob("*.jsonl")) if not days else \
        [RAW_ODDS / f"{d}.jsonl" for d in days]
    con = _connect_retry()
    ingest_ts = dt.datetime.now(dt.timezone.utc)
    for path in files:
        if not path.exists():
            print(f"skip missing {path.name}")
            continue
        rows = []
        with path.open() as f:
            for line in f:
                rows.extend(flatten(json.loads(line), path.name, ingest_ts))
        con.execute("DELETE FROM odds_quotes WHERE raw_file = ?", [path.name])
        if rows:   # offseason files can flatten to zero quotes
            con.executemany(
                "INSERT INTO odds_quotes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows)
        print(f"{path.name}: {len(rows)} quote rows")
    con.close()


if __name__ == "__main__":
    main(sys.argv[1:])
