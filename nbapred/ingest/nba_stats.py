"""nba_api ingestion with mandatory raw caching and rate limiting.

Every stats.nba.com response is written to data/raw/nba_api/<endpoint>/<key>.json
before anything parses it. Cache is keyed by endpoint+params; completed-game data
is immutable so cache hits skip the network. Rate limit: >=0.65s + jitter between
calls, exponential backoff on failure.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import random
import time

import pandas as pd

from ..config import CURRENT_SEASON, RAW_NBA, STATS_MIN_INTERVAL, STATS_TIMEOUT

log = logging.getLogger("nba_stats")
_last_call = 0.0


def _throttle() -> None:
    global _last_call
    wait = STATS_MIN_INTERVAL + random.uniform(0, 0.4) - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()


def _cache_key(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def cached_endpoint(endpoint_cls, cache_bucket: str, immutable: bool = True,
                    attempts: int = 4, **params) -> dict:
    """Call an nba_api endpoint class with caching. Returns the raw dict response.

    `attempts` exists for endpoints that are EXPECTED to fail on part of the
    corpus rather than fail transiently. GameRotation is the case: it returns
    an empty body for most pre-2019 games and for scattered later ones, and the
    default 4-try exponential backoff then burns ~22s per known-absent game
    (measured 7.9 s/game on the 2018-19 backfill vs 2.5 s/game without it).
    Callers that treat a miss as data-absent pass attempts=1. Default is
    unchanged, so the cron pullers behave exactly as before.
    """
    bucket = RAW_NBA / cache_bucket
    bucket.mkdir(parents=True, exist_ok=True)
    path = bucket / f"{_cache_key(params)}.json"
    if immutable and path.exists():
        return json.loads(path.read_text())["response"]

    for attempt in range(attempts):
        try:
            _throttle()
            ep = endpoint_cls(timeout=STATS_TIMEOUT, **params)
            raw = ep.get_dict()
            break
        except Exception as e:  # noqa: BLE001
            if attempt == attempts - 1:
                raise
            backoff = 2 ** (attempt + 1) + random.uniform(0, 2)
            log.warning("%s %s failed (%s); retry in %.0fs", cache_bucket, params, e, backoff)
            time.sleep(backoff)

    path.write_text(json.dumps({
        "ingest_ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "endpoint": cache_bucket,
        "params": params,
        "response": raw,
    }))
    return raw


def _frames(raw: dict) -> dict[str, pd.DataFrame]:
    """resultSets -> {name: DataFrame} (handles both resultSets and resultSet)."""
    out = {}
    sets_ = raw.get("resultSets") or raw.get("resultSet") or []
    if isinstance(sets_, dict):
        sets_ = [sets_]
    for rs in sets_:
        if "rowSet" in rs:
            out[rs.get("name", "res")] = pd.DataFrame(rs["rowSet"], columns=rs["headers"])
    return out


# ---- concrete pulls ---------------------------------------------------------

def pull_season_games(season: str = CURRENT_SEASON) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamefinder
    raw = cached_endpoint(
        leaguegamefinder.LeagueGameFinder, "leaguegamefinder",
        immutable=False,  # in-season this grows; offseason it's stable
        season_nullable=season, league_id_nullable="00",
    )
    return _frames(raw)["LeagueGameFinderResults"]


def pull_play_by_play(game_id: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import playbyplayv3
    raw = cached_endpoint(playbyplayv3.PlayByPlayV3, "playbyplayv3", game_id=game_id)
    # v3 returns a different shape: {'game': {'actions': [...]}}
    actions = raw.get("game", {}).get("actions")
    if actions is not None:
        return pd.DataFrame(actions)
    return next(iter(_frames(raw).values()))


def pull_boxscore(game_id: str) -> dict[str, pd.DataFrame]:
    from nba_api.stats.endpoints import boxscoretraditionalv3
    raw = cached_endpoint(boxscoretraditionalv3.BoxScoreTraditionalV3,
                          "boxscoretraditionalv3", game_id=game_id)
    return raw  # v3 nested dict; parse downstream as needed


def pull_rotations(game_id: str, attempts: int = 4) -> dict[str, pd.DataFrame]:
    """Stint data: GameRotation gives per-player IN/OUT wall-clock spans.

    Coverage is PARTIAL at the source (see cached_endpoint's `attempts` note);
    backfills pass attempts=1 so a known-absent game costs one call, not four.
    """
    from nba_api.stats.endpoints import gamerotation
    raw = cached_endpoint(gamerotation.GameRotation, "gamerotation",
                          attempts=attempts, game_id=game_id)
    return _frames(raw)


def sync_static_players_teams(con) -> None:
    from nba_api.stats.static import players
    now = dt.datetime.now(dt.timezone.utc)
    rows = [(p["id"], p["full_name"], p["first_name"], p["last_name"], p["is_active"], now)
            for p in players.get_players()]
    con.execute("DELETE FROM nba_players")
    con.executemany("INSERT INTO nba_players VALUES (?,?,?,?,?,?)", rows)
    log.info("synced %d players", len(rows))


def load_season_games(con, season: str = CURRENT_SEASON, df=None) -> int:
    if df is None:
        df = pull_season_games(season)   # network — prefer passing df pre-fetched
    now = dt.datetime.now(dt.timezone.utc)
    # D160: LeagueGameFinder emits rows with a NULL TEAM_ID for exhibition
    # entries that are not franchises — measured on 2004-05: 6 rows, all of
    # them All-Star-weekend (0030400002 "RKE @ SPH") or preseason games whose
    # team could not be resolved. `int(NaN)` raised ValueError and killed the
    # whole season write AFTER a 35-minute artifact pull had already
    # succeeded, which is the sole reason 2004-05 was absent from nba_games.
    # Drop them loudly; NO 002 row is ever affected (2004-05 keeps all 1,230).
    bad = df["TEAM_ID"].isna() | df["GAME_ID"].isna()
    if bad.any():
        drop = df[bad]
        log.warning("%s: dropping %d LeagueGameFinder rows with NULL TEAM_ID/"
                    "GAME_ID (%d of them 002): %s", season, len(drop),
                    int(drop["GAME_ID"].astype(str).str.startswith("002").sum()),
                    list(drop["GAME_ID"].astype(str))[:10])
        df = df[~bad]
    rows = [
        [season, r.GAME_ID, r.GAME_DATE, int(r.TEAM_ID), r.TEAM_ABBREVIATION,
         r.MATCHUP, "vs." in str(r.MATCHUP), r.WL,
         int(r.PTS) if pd.notna(r.PTS) else None, now]
        for _, r in df.iterrows()
    ]
    # D160: same fix D152 applied to possessions._write_rows, for the same
    # reason. `executemany` runs the statement once per row and on this DuckDB
    # build a held write lock BLOCKS READERS, so every other job on the box
    # stalls for the duration. MEASURED on the 2004-05 pull: 2,728 rows took
    # **114 s** via executemany vs **0.03 s** via register + INSERT..SELECT
    # (>3,000x). Semantics are unchanged: DELETE the season, insert its rows.
    cols = ["season", "game_id", "game_date", "team_id", "team_abbrev",
            "matchup", "is_home", "wl", "pts", "ingest_ts"]
    ins = pd.DataFrame(rows, columns=cols)
    ins["team_id"] = ins["team_id"].astype("Int64")
    ins["pts"] = ins["pts"].astype("Int64")
    con.execute("DELETE FROM nba_games WHERE season = ?", [season])
    con.register("nba_games_new", ins)
    con.execute("INSERT INTO nba_games SELECT * FROM nba_games_new")
    con.unregister("nba_games_new")
    return len(df)
