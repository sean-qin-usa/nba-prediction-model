"""Play-by-play + boxscore -> per-player-game sufficient statistics.

Handoff II.6 (v1): the skill model fits on "aggregated sufficient statistics
(binomial counts per player x context bucket)". This module produces the
finest re-aggregatable unit: one row per (game, player) with the counts that
map onto the latent skills in II.1. Downstream fitting sums these into whatever
context buckets it needs.

Source split (chosen after inspecting the v3 feeds):
  * boxscoretraditionalv3 = clean per-player totals (AST/STL/BLK/OREB/DREB/PF/
    TOV/FT + minutes). Backbone.
  * playbyplayv3 = the one thing the box lacks: the SHOT-ZONE split
    (rim / midrange / 3PT makes+attempts) and the shooting-foul split, parsed
    from shotValue + shotDistance + subType. AST/STL/BLK are NOT re-derived
    from PBP text (v3 encodes them by name-in-description, no id) — the box has
    them clean.

Skill -> statistic mapping (II.1):
  3PT / midrange / rim finish  <- FGM/FGA by zone (PBP)
  FT                            <- FTM/FTA (box)
  creation/passing              <- AST (box)
  TO propensity                 <- TOV, usage-denominated (box)
  OREB / DREB                   <- box
  steal pressure / rim protect  <- STL / BLK (box)
  foul propensity               <- PF, shooting-foul split (box + PBP)
  usage                         <- FGA + 0.44*FTA + TOV (box)

Runs over the cached raw JSON, so it processes the backfill without network.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import orjson
import logging

from ..config import RAW_NBA

log = logging.getLogger("possessions")

SCHEMA = """
CREATE TABLE IF NOT EXISTS player_game_stats (
    game_id     VARCHAR NOT NULL,
    player_id   BIGINT NOT NULL,
    team_id     BIGINT,
    seconds     INTEGER,                 -- minutes played, in seconds
    -- totals from boxscore
    fga INTEGER, fgm INTEGER, fg3a INTEGER, fg3m INTEGER,
    fta INTEGER, ftm INTEGER,
    ast INTEGER, tov INTEGER,
    oreb INTEGER, dreb INTEGER,
    stl INTEGER, blk INTEGER, pf INTEGER,
    pts INTEGER, plus_minus INTEGER,
    -- shot-zone split from PBP (attempts/makes); sum(zone a) reconciles to fga
    rima INTEGER, rimm INTEGER, mida INTEGER, midm INTEGER, thra INTEGER, thrm INTEGER,
    shooting_fouls INTEGER,
    ingest_ts   TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (game_id, player_id)
);
"""


def _mins_to_sec(m) -> int | None:
    if not m or m in ("", "0:00"):
        return 0
    try:
        if ":" in str(m):
            a, b = str(m).split(":")
            return int(float(a)) * 60 + int(float(b))
        return int(round(float(m) * 60))
    except (ValueError, TypeError):
        return None


def _zone(dist, shot_value, subtype: str) -> str:
    st = (subtype or "").lower()
    if shot_value == 3:
        return "thr"
    if any(k in st for k in ("layup", "dunk", "tip", "hook", "putback")) or (dist is not None and dist <= 4):
        return "rim"
    return "mid"


def parse_box(gid_dir: str) -> dict[int, dict]:
    raw = orjson.loads(open(gid_dir,"rb").read())["response"]
    bg = raw["boxScoreTraditional"]
    out: dict[int, dict] = {}
    for side in ("homeTeam", "awayTeam"):
        team = bg[side]
        tid = team["teamId"]
        for p in team["players"]:
            s = p["statistics"]
            out[p["personId"]] = dict(
                team_id=tid, seconds=_mins_to_sec(s.get("minutes")),
                fga=s.get("fieldGoalsAttempted", 0), fgm=s.get("fieldGoalsMade", 0),
                fg3a=s.get("threePointersAttempted", 0), fg3m=s.get("threePointersMade", 0),
                fta=s.get("freeThrowsAttempted", 0), ftm=s.get("freeThrowsMade", 0),
                ast=s.get("assists", 0), tov=s.get("turnovers", 0),
                oreb=s.get("reboundsOffensive", 0), dreb=s.get("reboundsDefensive", 0),
                stl=s.get("steals", 0), blk=s.get("blocks", 0), pf=s.get("foulsPersonal", 0),
                pts=s.get("points", 0), plus_minus=s.get("plusMinusPoints", 0),
                rima=0, rimm=0, mida=0, midm=0, thra=0, thrm=0, shooting_fouls=0,
            )
    return out


def apply_pbp_zones(pbp_path: str, players: dict[int, dict]) -> None:
    """Add shot-zone splits + shooting-foul counts from PBP onto box rows.

    D160 — `actionType` / `shotResult` are `.strip()`ped because stats.nba.com
    SPACE-PADS them to a fixed width on part of the archive. MEASURED: 11 games
    in 2000-01 (`0020000374`..`0020000383` and `0020000389`, all 2000-12-22/23)
    return `'Missed Shot                             '` — a normal 432-529
    action PBP, but every exact-match test here failed, so all 1,745 of their
    field-goal attempts landed with EVERY ZONE COUNT ZERO and zero shooting
    fouls. That is the D152 pre-1996-97 failure mode arriving through a
    different door: the file is present and parses, so `load_corpus`'s
    "no PBP cached" deferral never fires, and eFG silently degrades to
    fgm/fga for the affected games. Whole-cache audit after the fix: every
    season 1996-97..2025-26 sits at zone-attempts/fga = 1.0000.
    """
    raw = orjson.loads(open(pbp_path,"rb").read())["response"]
    for a in raw.get("game", {}).get("actions", []):
        pid = a.get("personId")
        if not pid or pid not in players:
            continue
        at = (a.get("actionType") or "").strip()
        if at in ("Made Shot", "Missed Shot", "Heave"):
            z = _zone(a.get("shotDistance"), a.get("shotValue"), a.get("subType"))
            players[pid][f"{z}a"] += 1
            if (a.get("shotResult") or "").strip() == "Made":
                players[pid][f"{z}m"] += 1
        elif at == "Foul" and "shooting" in (a.get("subType") or "").lower():
            players[pid]["shooting_fouls"] += 1


COLS = ["game_id", "player_id", "team_id", "seconds",
        "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "ast", "tov",
        "oreb", "dreb", "stl", "blk", "pf", "pts", "plus_minus",
        "rima", "rimm", "mida", "midm", "thra", "thrm",
        "shooting_fouls", "ingest_ts"]


def _write_rows(connect_fn, rows: list) -> None:
    """ONE vectorised statement, not a parameter-set-per-row executemany.

    duckdb's `executemany` runs the prepared statement once PER parameter set,
    and against a PRIMARY KEY the INSERT OR REPLACE costs an index probe each
    time. Measured 2026-08-02 on this box: 60,000 rows took **>600 s** that
    way (and the whole time the write lock is held, which on this DuckDB build
    blocks READERS too — every other job on the machine stalls). Registering
    the batch and doing a single INSERT..SELECT does the same 60,000 rows in
    **0.82 s**, a >700x cut in lock-hold time, with NULLs preserved.

    The backfill made this load-bearing: a full-corpus build is ~340,000 rows,
    i.e. hours of held lock on the old path.
    """
    if not rows:
        return
    import pandas as pd
    df = pd.DataFrame(rows, columns=COLS)
    for c in COLS[1:26]:                    # nullable ints (seconds can be None)
        df[c] = df[c].astype("Int64")
    con = connect_fn()
    con.execute(SCHEMA)
    con.register("pgs_new", df)
    con.execute("INSERT OR REPLACE INTO player_game_stats SELECT * FROM pgs_new")
    con.unregister("pgs_new")
    con.close()


def _game_ids() -> dict[str, str]:
    from .cache_index import game_index
    return game_index("boxscoretraditionalv3")


def _pbp_index() -> dict[str, str]:
    from .cache_index import game_index
    return game_index("playbyplayv3")


def load_corpus(connect_fn, limit: int | None = None, incremental: bool = True) -> dict:
    boxes = _game_ids()
    pbps = _pbp_index()
    gids = sorted(boxes)
    if incremental:
        con = connect_fn()
        con.execute(SCHEMA)
        have = {r[0] for r in con.execute("SELECT DISTINCT game_id FROM player_game_stats").fetchall()}
        con.close()
        gids = [g for g in gids if g not in have]
    if limit:
        gids = gids[:limit]
    if not gids:
        log.info("player_game_stats already current (no new games)")
        return {"games": 0, "zoned": 0, "rows": 0}
    rows, n_games, n_zoned, n_dead = [], 0, 0, 0
    now = dt.datetime.now(dt.timezone.utc)
    for gid in gids:
        try:
            players = parse_box(boxes[gid])
        except Exception:
            log.exception("box parse failed: %s", gid)
            continue
        # review6 #2: a PBP miss used to log-and-insert the game anyway with
        # rima/../thra all ZERO, and the incremental `have` check then marked
        # the game_id loaded FOREVER — one transient failure permanently
        # poisoned eFG (four_factors consumes thrm/thra) and every zone share.
        # Now: defer the game (skip insert) so the next incremental run
        # retries it once the PBP cache is present/parseable.
        if gid not in pbps:
            log.warning("no PBP cached for %s; game deferred (retry next run)", gid)
            continue
        try:
            apply_pbp_zones(pbps[gid], players)
            n_zoned += 1
        except Exception:
            log.exception("pbp zone failed: %s — game deferred (retry next run)", gid)
            continue
        # D160 ZONE-DEAD REFUSAL. The check above only defers a game whose PBP
        # FILE is missing. D152 named the gap and it was live: an
        # empty-but-present PBP parses fine and lands every zone count 0, which
        # silently degrades four_factors' eFG to fgm/fga — and the incremental
        # `have` set then marks the game done FOREVER. Two real populations hit
        # this: (a) pre-1996-97, where playbyplayv3 returns HTTP 200 with ZERO
        # actions (13 probe-sample games had leaked into player_game_stats this
        # way and were purged by D160); (b) 11 games in 2000-01 whose
        # actionType was space-padded (fixed in apply_pbp_zones). REFUSE rather
        # than insert: a deferred game is visible as missing coverage, a
        # zone-zero game is invisible corruption.
        za = sum(c["rima"] + c["mida"] + c["thra"] for c in players.values())
        if za == 0 and sum(c["fga"] for c in players.values()) > 0:
            n_dead += 1
            log.warning("ZONE-DEAD %s: %d fga but zero PBP zone attempts; game "
                        "REFUSED (not inserted)", gid,
                        sum(c["fga"] for c in players.values()))
            continue
        n_games += 1
        for pid, c in players.items():
            rows.append([gid, pid, c["team_id"], c["seconds"],
                         c["fga"], c["fgm"], c["fg3a"], c["fg3m"], c["fta"], c["ftm"],
                         c["ast"], c["tov"], c["oreb"], c["dreb"], c["stl"], c["blk"], c["pf"],
                         c["pts"], c["plus_minus"],
                         c["rima"], c["rimm"], c["mida"], c["midm"], c["thra"], c["thrm"],
                         c["shooting_fouls"], now])

    _write_rows(connect_fn, rows)
    log.info("parsed %d games (%d with PBP zones, %d REFUSED zone-dead) -> "
             "%d player-game rows", n_games, n_zoned, n_dead, len(rows))
    return {"games": n_games, "zoned": n_zoned, "zone_dead_refused": n_dead,
            "rows": len(rows)}
