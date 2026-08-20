"""NBA draft history via nba_api drafthistory (free, one call for all years).

Purpose (D84-A candidate #4): rookies are invisible to the October composition
bridge (no prior-season 002 minutes, no pre-debut DARKO row — verified: first
darko_history row is always the day AFTER debut). Draft slot is the only
PIT-available talent signal on opening night, so we ingest the full draft
history and fit slot -> rookie-season DPM elsewhere.

Raw rule: response cached to data/raw/nba_api/drafthistory/ before parsing
(cached_endpoint). Table is derived + rebuildable; single-writer discipline —
only call load_draft_history from a batch script with no other writer running.
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from .nba_stats import _frames, cached_endpoint

log = logging.getLogger("draft")

SCHEMA = """
CREATE TABLE IF NOT EXISTS draft_history (
    player_id    BIGINT  NOT NULL,   -- PERSON_ID (joins player_game_stats / darko_history)
    player_name  VARCHAR,
    draft_year   INTEGER NOT NULL,   -- SEASON field, e.g. 2023 = 2023-24 rookie season
    round_number INTEGER,
    round_pick   INTEGER,
    overall_pick INTEGER,            -- 1-60 (draft slot); NULL never occurs in source
    team_id      BIGINT,
    organization VARCHAR,            -- college / club
    ingest_ts    TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (player_id, draft_year)
);
"""


def pull_draft_history() -> pd.DataFrame:
    """All drafts, one request. immutable=False: re-pull picks up each new draft."""
    from nba_api.stats.endpoints import drafthistory
    raw = cached_endpoint(drafthistory.DraftHistory, "drafthistory",
                          immutable=False, league_id="00")
    return _frames(raw)["DraftHistory"]


def load_draft_history(con, df: pd.DataFrame | None = None) -> int:
    """Idempotent full reload (source is tiny: ~8k rows across all years)."""
    if df is None:
        df = pull_draft_history()
    con.execute(SCHEMA)
    now = dt.datetime.now(dt.timezone.utc)
    rows = [
        [int(r.PERSON_ID), r.PLAYER_NAME, int(r.SEASON),
         int(r.ROUND_NUMBER) if pd.notna(r.ROUND_NUMBER) else None,
         int(r.ROUND_PICK) if pd.notna(r.ROUND_PICK) else None,
         int(r.OVERALL_PICK) if pd.notna(r.OVERALL_PICK) else None,
         int(r.TEAM_ID) if pd.notna(r.TEAM_ID) else None,
         r.ORGANIZATION, now]
        for _, r in df.iterrows()
    ]
    con.execute("DELETE FROM draft_history")
    con.executemany("INSERT INTO draft_history VALUES (?,?,?,?,?,?,?,?,?)", rows)
    log.info("loaded %d draft rows (%s-%s)", len(rows),
             df.SEASON.min(), df.SEASON.max())
    return len(rows)
