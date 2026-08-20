#!/usr/bin/env python3
"""D170: reparse the whole data/raw/injury_reports/ archive into
injury_reports_pit with the FIXED batched writer.

Why it is needed even with no new download: the league renamed the PDF on
2026-01-01 (`_05PM` -> `_05_00PM`); nbapred/ingest/injury_pdf.py's filename
regex did not match, load_all() swallowed the AttributeError as a parse
failure, and 97 archived report-days (2026-01-01..2026-04-12) never reached
the table. injury_reports_pit therefore stopped at 2025-12-22 while the PDFs
sat on disk. The regex is fixed; this reloads everything.

ALL parsing happens with NO DB connection open. The lock is taken once, for a
DELETE + one INSERT..SELECT off a registered DataFrame, and released.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import nbapred.threads  # noqa: E402
nbapred.threads.pin(1)

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402

from nbapred.ingest.injury_pdf import SCHEMA, parse_pdf  # noqa: E402

DB = REPO / "data" / "nba.duckdb"
RAW = REPO / "data" / "raw" / "injury_reports"


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
    files = sorted(RAW.glob("Injury-Report_*.pdf"))
    rows, bad = [], []
    for n, f in enumerate(files):
        try:
            rows.extend(parse_pdf(f))
        except Exception as e:  # noqa: BLE001
            bad.append((f.name, f"{type(e).__name__}: {str(e)[:80]}"))
        if (n + 1) % 100 == 0:
            print(f"  parsed {n+1}/{len(files)} ({len(rows)} rows, {len(bad)} bad)",
                  flush=True)
    print(f"parsed {len(files)} files -> {len(rows)} rows, {len(bad)} failures",
          flush=True)
    for nm, err in bad[:20]:
        print(f"  BAD {nm}: {err}", flush=True)

    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df.game_date, format="%m/%d/%Y",
                                     errors="coerce").dt.date
    df["report_date"] = pd.to_datetime(df.report_date).dt.date
    df = df.dropna(subset=["game_date"]).drop_duplicates(
        subset=["report_date", "edition", "game_date", "team", "player"])
    print(f"{len(df)} rows after dedup; span "
          f"{df.report_date.min()} .. {df.report_date.max()}", flush=True)

    t0 = time.time()
    con = _connect(read_only=False)
    try:
        con.execute(SCHEMA)
        con.register("df", df)
        con.execute("BEGIN")
        con.execute("DELETE FROM injury_reports_pit")
        con.execute("""INSERT INTO injury_reports_pit
            SELECT report_date, edition, game_date, matchup, team, player,
                   status, reason FROM df""")
        con.execute("COMMIT")
    finally:
        con.close()
    print(f"WRITE DONE in {time.time()-t0:.2f}s", flush=True)

    con = _connect(read_only=True)
    print(con.execute("""
      SELECT CASE WHEN month(game_date)>=10
                  THEN year(game_date)||'-'||substr(cast(year(game_date)+1 as varchar),3,2)
                  ELSE cast(year(game_date)-1 as varchar)||'-'||substr(cast(year(game_date) as varchar),3,2)
             END season,
             count(DISTINCT report_date) report_days, count(*) n_rows,
             min(game_date) first_gd, max(game_date) last_gd
      FROM injury_reports_pit GROUP BY 1 ORDER BY 1""").fetchdf().to_string(),
          flush=True)
    con.close()
    print("INJURY_LOAD_DONE", flush=True)


if __name__ == "__main__":
    main()
