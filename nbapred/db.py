"""DuckDB store. Point-in-time discipline: every row carries created_ts (source
event time where knowable) and ingest_ts (when we captured it). Raw files on
disk are the ground truth; DuckDB tables are derived and rebuildable.

Single-writer rule: the always-on odds logger NEVER touches DuckDB — it appends
JSONL to data/raw/odds/. Only batch loaders (scripts/) open the DB for writing.
"""
from __future__ import annotations

import duckdb

from .config import DB_PATH

SCHEMA = """
CREATE SEQUENCE IF NOT EXISTS seq_generic START 1;

-- odds snapshots, flattened one row per (snapshot, event, book, market, outcome)
CREATE TABLE IF NOT EXISTS odds_quotes (
    snapshot_ts    TIMESTAMPTZ NOT NULL,   -- when the logger polled
    ingest_ts      TIMESTAMPTZ NOT NULL,   -- when loaded into DuckDB
    source         VARCHAR NOT NULL,       -- 'the-odds-api'
    event_id       VARCHAR NOT NULL,
    commence_time  TIMESTAMPTZ,
    home_team      VARCHAR,
    away_team      VARCHAR,
    bookmaker      VARCHAR NOT NULL,
    book_last_update TIMESTAMPTZ,          -- book's own update time (created_ts analog)
    market         VARCHAR NOT NULL,       -- h2h / spreads / totals / player_points / ...
    outcome_name   VARCHAR NOT NULL,       -- team name, Over/Under, player name
    outcome_desc   VARCHAR,                -- player name for props
    price_decimal  DOUBLE NOT NULL,
    point          DOUBLE,                 -- spread / total / prop line
    raw_file       VARCHAR                 -- provenance: JSONL file this came from
);

CREATE TABLE IF NOT EXISTS nba_games (
    season         VARCHAR NOT NULL,
    game_id        VARCHAR NOT NULL,
    game_date      DATE,
    team_id        BIGINT,
    team_abbrev    VARCHAR,
    matchup        VARCHAR,
    is_home        BOOLEAN,
    wl             VARCHAR,
    pts            INTEGER,
    ingest_ts      TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS nba_players (
    player_id      BIGINT NOT NULL,
    full_name      VARCHAR NOT NULL,
    first_name     VARCHAR,
    last_name      VARCHAR,
    is_active      BOOLEAN,
    ingest_ts      TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS ratings_2k (
    scrape_date    DATE NOT NULL,          -- version key
    edition        VARCHAR NOT NULL,       -- e.g. 'NBA 2K26'
    player_name    VARCHAR NOT NULL,
    team_slug      VARCHAR,
    overall        INTEGER,
    attributes     JSON,                   -- {attr_name: value}
    badges         JSON,
    source_url     VARCHAR,
    raw_file       VARCHAR,
    ingest_ts      TIMESTAMPTZ NOT NULL
);

-- crosswalk: one row per NBA player, matched names elsewhere
CREATE TABLE IF NOT EXISTS player_xwalk (
    nba_player_id  BIGINT NOT NULL,
    nba_name       VARCHAR NOT NULL,
    norm_name      VARCHAR NOT NULL,       -- normalized join key
    name_2k        VARCHAR,                -- as it appears on 2kratings
    match_2k_method VARCHAR,               -- exact / normalized / manual / unmatched
    name_odds      VARCHAR,                -- as seen in prop outcome_desc (filled as observed)
    updated_ts     TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS injury_reports (
    report_ts      TIMESTAMPTZ,            -- report publication time (from filename/header)
    ingest_ts      TIMESTAMPTZ NOT NULL,
    game_date      DATE,
    matchup        VARCHAR,
    team           VARCHAR,
    player_name    VARCHAR,
    status         VARCHAR,                -- Out/Doubtful/Questionable/Probable/Available
    reason         VARCHAR,
    raw_file       VARCHAR
);
"""


def connect(read_only: bool = False, retry_s: float | None = None
            ) -> duckdb.DuckDBPyConnection:
    """Single-writer DuckDB. Reads retry-on-lock so analysis never fails while a
    batch loader/backfill holds the write lock (the recurring contention).

    Default deadline: 120 s for readers, 600 s for WRITERS. Writers wait longer
    because a crashed daily loader silently loses that day's data, while a slow
    one only finishes late — and long-running gate scripts routinely hold the
    write lock for 10-20 min. Measured 2026-08-02: the 120 s writer deadline
    had already cost load_darko 3 runs, load_odds 4 and scrape_2k 1 (each an
    `IOException: Could not set lock`, all pre-dating any backfill). 600 s also
    matches the shortest cron `timeout` on a writer job, so the wait can never
    outlive the job's own budget. Pass retry_s explicitly to override
    (backfills use 0 and poll themselves, so they can yield politely).
    """
    import time
    if retry_s is None:
        retry_s = 120.0 if read_only else 600.0
    deadline = time.monotonic() + retry_s
    while True:
        try:
            con = duckdb.connect(str(DB_PATH), read_only=read_only)
            break
        except duckdb.IOException:
            if time.monotonic() >= deadline:
                raise
            time.sleep(2)
    if not read_only:
        con.execute(SCHEMA)
    return con
