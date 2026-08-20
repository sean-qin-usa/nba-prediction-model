"""v3 DuckDB tables + the guarded single-writer connection.

The v3 tables live in the SAME DuckDB file as production (data/nba.duckdb),
exactly as V3_SPEC 2.6 declares, but v3 NEVER writes any production table.
DuckDB enforces one writer per file, so a v3 write phase must (a) be brief,
(b) only start when no other writer holds the lock. `v3_writer()` implements
that: it attempts a read-write connection with a SHORT deadline — if a batch
loader / ingest cron holds the lock it raises loudly instead of queueing up
behind production work (respecting the single-writer rule D1 rather than
fighting it). All v3 writes go through this context manager.
"""
from __future__ import annotations

import contextlib
import time

import duckdb

from ..config import DB_PATH

V3_SCHEMA = """
CREATE TABLE IF NOT EXISTS player_states (
    "asof"     DATE   NOT NULL,
    player_id  BIGINT NOT NULL,      -- team_id for team-level dims (dim LIKE 'team_%' / 'pace')
    team_id    BIGINT,
    dim        VARCHAR NOT NULL,
    mean       DOUBLE,
    var        DOUBLE,
    PRIMARY KEY ("asof", player_id, dim)
);
CREATE TABLE IF NOT EXISTS state_shocks (
    event_date DATE    NOT NULL,
    entity_id  BIGINT  NOT NULL,     -- player_id or team_id, per kind
    kind       VARCHAR NOT NULL,     -- trade | return | coach | season_boundary
    source     VARCHAR,
    flagged_ts TIMESTAMP,
    PRIMARY KEY (event_date, entity_id, kind)
);
CREATE TABLE IF NOT EXISTS v3_predictions (
    game_id  VARCHAR NOT NULL,
    asof_ts  TIMESTAMP,
    head     VARCHAR NOT NULL,       -- side | total | team_total | prop:<stat>
    mu       DOUBLE,
    sigma    DOUBLE,
    p        DOUBLE,
    version  VARCHAR NOT NULL,
    PRIMARY KEY (game_id, head, version)
);
"""

_V3_TABLES = ("player_states", "state_shocks", "v3_predictions")


def ensure_v3_tables(con) -> None:
    con.execute(V3_SCHEMA)


@contextlib.contextmanager
def v3_writer(retry_s: float = 10.0):
    """Short-deadline read-write connection for v3 tables ONLY.

    Raises duckdb.IOException if another writer still holds the lock after
    `retry_s` — fail loud, never silently queue behind ingest crons.
    """
    deadline = time.monotonic() + retry_s
    while True:
        try:
            con = duckdb.connect(str(DB_PATH), read_only=False)
            break
        except duckdb.IOException:
            if time.monotonic() >= deadline:
                raise duckdb.IOException(
                    "v3_writer: another writer holds the DuckDB lock "
                    f"(waited {retry_s:.0f}s). Single-writer rule: retry when "
                    "the ingest/batch job finishes.")
            time.sleep(1.0)
    try:
        ensure_v3_tables(con)
        yield con
    finally:
        con.close()
