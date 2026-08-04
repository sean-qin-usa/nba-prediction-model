"""Referee data — a prediction signal not in the original handoff.

Refs measurably shift foul rates, free-throw volume, pace, and totals; crews
are assigned and published PREGAME, so the assignment is knowable before the
line settles. This feeds the foul-draw step of the possession engine (II.3.3)
as a crew fixed-effect, and is a totals/props signal in its own right.

Two feeds:
  * retrospective (who officiated past games): nba_api BoxScoreSummaryV2
    'Officials' result set -> ref crew per historical game, for building
    per-ref foul/FT/pace tendencies from our PBP corpus.
  * pregame assignments (who WILL officiate tonight): official.nba.com wp-json
    get-game-officials feed -> the forward-looking signal. Empty in offseason;
    poller archives raw regardless.
"""
from __future__ import annotations

import datetime as dt
import json
import logging

import requests

from ..config import RAW
from .nba_stats import _frames, cached_endpoint

log = logging.getLogger("referees")

RAW_REF = RAW / "referees"
RAW_REF.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0"}
ASSIGN_URL = "https://official.nba.com/wp-json/api/v1/get-game-officials"

SCHEMA = """
CREATE TABLE IF NOT EXISTS game_officials (
    game_id     VARCHAR NOT NULL,
    official_id BIGINT NOT NULL,
    first_name  VARCHAR, last_name VARCHAR, jersey_num VARCHAR,
    source      VARCHAR,          -- 'boxsummary' (retro) / 'assignment' (pregame)
    ingest_ts   TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (game_id, official_id, source)
);
"""


def officials_for_game(game_id: str) -> list[dict]:
    """Retrospective crew via BoxScoreSummaryV2."""
    from nba_api.stats.endpoints import boxscoresummaryv2
    raw = cached_endpoint(boxscoresummaryv2.BoxScoreSummaryV2, "boxscoresummaryv2",
                          immutable=True, game_id=game_id)
    df = _frames(raw).get("Officials")
    if df is None or df.empty:
        return []
    return [{"official_id": r.OFFICIAL_ID, "first_name": r.FIRST_NAME,
             "last_name": r.LAST_NAME, "jersey_num": str(r.JERSEY_NUM).strip()}
            for r in df.itertuples()]


def poll_assignments(season: str | None = None) -> int:
    """Pregame assignments feed; archive raw. Returns row count captured.
    season=None -> current season by calendar (dynamic)."""
    if season is None:
        from ..config import current_season
        season = current_season()
    r = requests.get(ASSIGN_URL, headers=UA, params={"season": season}, timeout=20)
    r.raise_for_status()
    body = r.json()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    (RAW_REF / f"assignments_{stamp}.json").write_text(json.dumps(body))
    table = (body.get("nba") or {}).get("Table") or {}
    rows = table.get("rows") or []
    log.info("assignments feed: %d rows", len(rows))
    return len(rows)


def load_retro_officials(connect_fn, game_ids: list[str]) -> int:
    """Fetch officials for the given games (network), then one short write."""
    crews = {}
    for gid in game_ids:
        try:
            crews[gid] = officials_for_game(gid)
        except Exception:
            log.exception("officials fetch failed: %s", gid)
    now = dt.datetime.now(dt.timezone.utc)
    con = connect_fn()
    con.execute(SCHEMA)
    n = 0
    for gid, crew in crews.items():
        for o in crew:
            con.execute("INSERT OR REPLACE INTO game_officials VALUES (?,?,?,?,?,?,?)",
                        [gid, o["official_id"], o["first_name"], o["last_name"],
                         o["jersey_num"], "boxsummary", now])
            n += 1
    con.close()
    log.info("loaded officials for %d games -> %d rows", len(crews), n)
    return n
