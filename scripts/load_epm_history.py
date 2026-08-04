#!/usr/bin/env python3
"""D85 step 1c — load the identity-bearing EPM PIT history into DuckDB.

Source: parsed Wayback captures of the live /epm page
(data/raw/ext_epm/wayback/parsed_{ts}.json, built by pull_epm_wayback.py)
plus today's live-page parse when present. The masked ?date= endpoint grid
(data/raw/ext_epm/{date}.json) is NOT loaded — identity is anonymized beyond
the top 5 players (lock finding 2026-07-31); it stays raw-cached as the
value hedge.

Table: epm_history — one row per (asof_date, player_id), asof_date = last
game date included in that run (PIT consumers must filter asof_date < d).
When several captures share an asof_date the latest capture wins.

SINGLE-WRITER DISCIPLINE: refuses to write while a known DB-writing script
(build_features / backfill_* / load_*) is running; DuckDB lock conflicts are
retried briefly, then the load aborts cache-only with a note (rerun later).
"""
from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import duckdb

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from nbapred.config import DB_PATH  # noqa: E402

WB_DIR = REPO / "data/raw/ext_epm/wayback"
LIVE = REPO / "data/raw/ext_epm/live_page_2026-07-31.html"
# match only an EXECUTING python writer (args begin with python), not bash
# watcher loops that merely embed a future writer command in their -c string
WRITERS = re.compile(r"^\s*\d+\s+python[0-9.]*\s+\S*(build_features|"
                     r"backfill_\w+|load_\w+|build_player_stats)\.py")

DDL = """
CREATE TABLE IF NOT EXISTS epm_history (
    asof_date   DATE NOT NULL,
    capture_ts  VARCHAR NOT NULL,
    era         VARCHAR,
    player_id   BIGINT NOT NULL,
    player_name VARCHAR,
    team_id     BIGINT,
    team_alias  VARCHAR,
    off_epm     DOUBLE,
    def_epm     DOUBLE,
    tot_epm     DOUBLE,
    ingest_ts   TIMESTAMPTZ
);
"""


def writers_running() -> list[str]:
    me = str(Path(__file__).name)
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True,
                         text=True).stdout
    hits = []
    for line in out.splitlines():
        if WRITERS.search(line) and me not in line:
            hits.append(line.strip()[:120])
    return hits


def collect() -> list[dict]:
    per_asof: dict[str, dict] = {}
    metas = []
    for f in sorted(WB_DIR.glob("parsed_*.json")):
        metas.append(json.loads(f.read_text()))
    if LIVE.exists():
        from pull_epm_wayback import parse
        p = parse("20260731000000", LIVE.read_text(errors="replace"))
        if p:
            p["ts"] = "live_2026-07-31"
            metas.append(p)
    for meta in metas:
        cur = per_asof.get(meta["asof"])
        if cur is None or meta["ts"] > cur["ts"]:
            per_asof[meta["asof"]] = meta
    return [per_asof[k] for k in sorted(per_asof)]


def main() -> None:
    hits = writers_running()
    if hits:
        print("WRITER ACTIVE — cache-only, load deferred:")
        for h in hits:
            print("  ", h)
        sys.exit(2)
    metas = collect()
    n_rows = sum(m["n"] for m in metas)
    print(f"captures to load: {len(metas)} asof-dates, {n_rows} rows "
          f"({metas[0]['asof']} .. {metas[-1]['asof']})")
    con = None
    for attempt in range(6):
        try:
            con = duckdb.connect(str(DB_PATH))
            break
        except duckdb.IOException as e:
            print(f"  lock busy ({e}); retry {attempt + 1}/6 in 10s")
            time.sleep(10)
    if con is None:
        print("DB LOCKED — load deferred (raw cache intact; rerun later)")
        sys.exit(2)
    con.execute(DDL)
    now = dt.datetime.now(dt.timezone.utc)
    rows = []
    for m in metas:
        for r in m["rows"]:
            rows.append((dt.date.fromisoformat(m["asof"]), m["ts"], m["era"],
                         r["player_id"], r["player_name"], r["team_id"],
                         r["team_alias"], r["off"], r["def_"], r["tot"], now))
    import pandas as pd
    df = pd.DataFrame(rows, columns=[
        "asof_date", "capture_ts", "era", "player_id", "player_name",
        "team_id", "team_alias", "off_epm", "def_epm", "tot_epm",
        "ingest_ts"])
    con.register("epm_stage", df)
    con.execute("BEGIN")
    con.execute("DELETE FROM epm_history")
    con.execute("INSERT INTO epm_history SELECT * FROM epm_stage")
    con.execute("COMMIT")
    chk = con.execute("""
        SELECT count(*), count(DISTINCT asof_date), count(DISTINCT player_id),
               min(asof_date), max(asof_date) FROM epm_history""").fetchone()
    con.close()
    print(f"epm_history loaded: rows={chk[0]} asof_dates={chk[1]} "
          f"players={chk[2]} span {chk[3]}..{chk[4]}")


if __name__ == "__main__":
    main()
