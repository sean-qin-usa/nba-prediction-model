"""GameRotation + PBP -> lineup-stint point margins (RAPM substrate).

Handoff II.2: the stint-margin likelihood is "identification-critical" — a
RAPM-style term on lineup-stint point margins that pins each player's TOTAL
two-way impact (defense is underidentified by events alone). This module
produces its training data: one row per stint (a maximal interval over which
all ten on-court players are constant), with both 5-man lineups, duration, and
the home-perspective point margin scored during the stint.

Time bases (verified against cached v3 data):
  * GameRotation IN_TIME_REAL/OUT_TIME_REAL: tenths of a second (regulation
    max 28800 = 2880s). Divide by 10.
  * PBP clock: ISO8601 'PTmmMss.ssS' = time REMAINING in the period. Elapsed
    real seconds = period_start + (period_len - remaining), period_len 720
    (regulation) / 300 (OT).
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import orjson
import logging
import re

from ..config import RAW_NBA

log = logging.getLogger("stints")

SCHEMA = """
CREATE TABLE IF NOT EXISTS lineup_stints (
    game_id     VARCHAR NOT NULL,
    stint_idx   INTEGER NOT NULL,
    home_team_id BIGINT, away_team_id BIGINT,
    t_start     DOUBLE, t_end DOUBLE, seconds DOUBLE,
    home_lineup VARCHAR,          -- sorted comma-joined player_ids (5)
    away_lineup VARCHAR,
    home_pts    INTEGER,          -- points scored by home during stint
    away_pts    INTEGER,
    margin      INTEGER,          -- home_pts - away_pts
    ingest_ts   TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (game_id, stint_idx)
);
"""

_CLOCK = re.compile(r"PT(\d+)M([\d.]+)S")


def _elapsed(period: int, clock: str) -> float:
    m = _CLOCK.match(clock or "")
    if not m:
        return None
    remaining = int(m.group(1)) * 60 + float(m.group(2))
    if period <= 4:
        start, length = (period - 1) * 720, 720
    else:
        start, length = 4 * 720 + (period - 5) * 300, 300
    return start + (length - remaining)


def _score_fn(pbp_actions: list):
    """Return sorted [(elapsed_sec, home, away)] score checkpoints."""
    pts = []
    for a in pbp_actions:
        sh, sa = a.get("scoreHome"), a.get("scoreAway")
        if sh in (None, "") or sa in (None, ""):
            continue
        t = _elapsed(a.get("period"), a.get("clock"))
        if t is not None:
            pts.append((t, int(sh), int(sa)))
    pts.sort()
    return pts


def _score_at(checkpoints, t: float) -> tuple[int, int]:
    """(home, away) as of elapsed time t (last checkpoint <= t)."""
    h, a = 0, 0
    for ct, ch, ca in checkpoints:
        if ct <= t + 1e-6:
            h, a = ch, ca
        else:
            break
    return h, a


def build_game(rot_raw: dict, pbp_raw: dict) -> list[dict]:
    sides = {}  # team_id -> ("home"/"away", [(pid, in_s, out_s)])
    home_id = pbp_raw.get("game", {}).get("homeTeamId")
    for rs in rot_raw["resultSets"]:
        side = "home" if rs["name"] == "HomeTeam" else "away"
        for row in rs["rowSet"]:
            d = dict(zip(rs["headers"], row))
            sides.setdefault(side, {"team_id": d["TEAM_ID"], "spans": []})
            sides[side]["spans"].append(
                (d["PERSON_ID"], d["IN_TIME_REAL"] / 10.0, d["OUT_TIME_REAL"] / 10.0))

    # breakpoints from every sub in/out
    bps = set()
    for side in sides.values():
        for _, i, o in side["spans"]:
            bps.add(round(i, 1))
            bps.add(round(o, 1))
    bps = sorted(bps)

    checkpoints = _score_fn(pbp_raw.get("game", {}).get("actions", []))
    stints = []
    for k in range(len(bps) - 1):
        t0, t1 = bps[k], bps[k + 1]
        if t1 - t0 < 0.5:
            continue
        mid = (t0 + t1) / 2

        def on_court(side_key):
            return sorted(pid for pid, i, o in sides[side_key]["spans"] if i <= mid < o)

        home5, away5 = on_court("home"), on_court("away")
        if len(home5) != 5 or len(away5) != 5:
            continue  # transient inconsistency at a breakpoint; skip
        h0, a0 = _score_at(checkpoints, t0)
        h1, a1 = _score_at(checkpoints, t1)
        stints.append(dict(
            home_team_id=sides["home"]["team_id"], away_team_id=sides["away"]["team_id"],
            t_start=t0, t_end=t1, seconds=t1 - t0,
            home_lineup=",".join(map(str, home5)), away_lineup=",".join(map(str, away5)),
            home_pts=h1 - h0, away_pts=a1 - a0, margin=(h1 - h0) - (a1 - a0)))
    return stints


def load_corpus(connect_fn, limit: int | None = None, incremental: bool = True) -> dict:
    from .cache_index import game_index
    rots = game_index("gamerotation")
    pbps = game_index("playbyplayv3")
    gids = sorted(set(rots) & set(pbps))
    if incremental:
        con = connect_fn()
        con.execute(SCHEMA)
        have = {r[0] for r in con.execute("SELECT DISTINCT game_id FROM lineup_stints").fetchall()}
        con.close()
        gids = [g for g in gids if g not in have]
    if limit:
        gids = gids[:limit]
    if not gids:
        log.info("lineup_stints already current (no new games)")
        return {"games": 0, "stints": 0}
    rows, n_games = [], 0
    now = dt.datetime.now(dt.timezone.utc)
    for gid in gids:
        try:
            rot = orjson.loads(open(rots[gid],"rb").read())["response"]
            pbp = orjson.loads(open(pbps[gid],"rb").read())["response"]
            stints = build_game(rot, pbp)
        except Exception:
            log.exception("stint build failed: %s", gid)
            continue
        n_games += 1
        for idx, s in enumerate(stints):
            rows.append([gid, idx, s["home_team_id"], s["away_team_id"],
                         s["t_start"], s["t_end"], s["seconds"], s["home_lineup"],
                         s["away_lineup"], s["home_pts"], s["away_pts"], s["margin"], now])

    # D160 — two fixes, both hit live on 2026-08-03.
    # (a) EMPTY BATCH CRASHED THE CRON. When every candidate game's rotation
    #     JSON fails to build a stint (measured: 28 gamerotation files the
    #     backfill added, all of which produce zero stints), `rows` is empty and
    #     duckdb raises "executemany requires a non-empty list of parameter
    #     sets" — so `scripts/build_features.py` DIED before reaching
    #     schedule.build(). Return early instead.
    # (b) executemany -> register + INSERT..SELECT, the D152 rule: one
    #     statement, not one per row, so the write lock (which blocks READERS
    #     on this build) is held for a fraction of the time. lineup_stints is
    #     ~152k rows, i.e. a full rebuild was minutes of held lock.
    if not rows:
        log.info("built %d games -> 0 stints (nothing to write)", n_games)
        return {"games": n_games, "stints": 0}
    import pandas as pd
    cols = ["game_id", "stint_idx", "home_team_id", "away_team_id", "t_start",
            "t_end", "seconds", "home_lineup", "away_lineup", "home_pts",
            "away_pts", "margin", "ingest_ts"]
    df = pd.DataFrame(rows, columns=cols)
    con = connect_fn()
    con.execute(SCHEMA)
    con.register("stints_new", df)
    con.execute("INSERT OR REPLACE INTO lineup_stints SELECT * FROM stints_new")
    con.unregister("stints_new")
    con.close()
    log.info("built %d games -> %d stints", n_games, len(rows))
    return {"games": n_games, "stints": len(rows)}
