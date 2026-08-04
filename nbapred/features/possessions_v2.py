"""v2 possession dataset: segment PBP into individual possessions with the
on-court lineups — the substrate for the possession-level likelihood (v2's
whole point: fit skills on WHO did WHAT against WHOM, possession by possession).

A possession ends on: made FG (plus trailing FTs from and-one), defensive
rebound, turnover, made final FT of a trip, or period end. Offensive rebounds
CONTINUE the possession. Lineups come from the stint segments (rotation data).

Output row: (game_id, poss_idx, off_team, def_team, off_lineup, def_lineup,
points, ended_by, seconds). Written to DuckDB table possessions_v2.
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import orjson

from ..features.defense_zone import _game_segments
from ..features.stints import _elapsed

log = logging.getLogger("poss_v2")

SCHEMA = """
CREATE TABLE IF NOT EXISTS possessions_v2 (
    game_id VARCHAR, poss_idx INTEGER,
    off_team BIGINT, def_team BIGINT,
    off_lineup VARCHAR, def_lineup VARCHAR,
    points INTEGER, ended_by VARCHAR, seconds DOUBLE,
    PRIMARY KEY (game_id, poss_idx)
);
"""

END_EVENTS = {"Made Shot", "Turnover"}


def _team_ids(rot_raw: dict, pbp_raw: dict):
    """(home_id, away_id). The cached playbyplayv3 payload's `game` object has
    NO homeTeamId (probe-verified: only gameId/videoAvailable), so the old
    `pbp_raw['game'].get('homeTeamId')` was ALWAYS None -> is_home was False
    for every possession and home-offense rows got swapped off/def lineups.
    The rotation feed's HomeTeam/AwayTeam result sets carry TEAM_ID."""
    home_id = pbp_raw.get("game", {}).get("homeTeamId")
    away_id = pbp_raw.get("game", {}).get("awayTeamId")
    for rs in rot_raw.get("resultSets", []):
        try:
            tid = rs["rowSet"][0][rs["headers"].index("TEAM_ID")]
        except (KeyError, IndexError, ValueError):
            continue
        if rs.get("name") == "HomeTeam":
            home_id = home_id or tid
        elif rs.get("name") == "AwayTeam":
            away_id = away_id or tid
    return home_id, away_id


def parse_game(pbp_raw: dict, rot_raw: dict):
    """Yield possession dicts for one game."""
    home_id, away_id = _team_ids(rot_raw, pbp_raw)
    segs = _game_segments(rot_raw, pbp_raw)
    if not segs:
        return []
    seg_t0 = np.array([s[0] for s in segs])
    acts = pbp_raw.get("game", {}).get("actions", [])

    poss = []
    cur_team = None
    cur_pts = 0
    t_start = 0.0
    pending_ft = False   # trip continues through free throws

    def lineup_at(t, team_is_home):
        k = int(np.searchsorted(seg_t0, t, side="right") - 1)
        if k < 0 or k >= len(segs):
            return None, None
        _, _, h5, a5 = segs[k]
        return (h5, a5) if team_is_home else (a5, h5)

    def close(t_end, ended_by):
        nonlocal cur_team, cur_pts, t_start
        if cur_team is None:
            cur_team, cur_pts, t_start = None, 0, t_end
            return
        is_home = cur_team == home_id
        off5, def5 = lineup_at((t_start + t_end) / 2, is_home)
        if off5 and def5 and len(off5) == 5 and len(def5) == 5:
            poss.append(dict(off_team=cur_team,
                             # review6 #6: was None and the caller wrote a
                             # literal 0 for every row, killing all defender/
                             # opponent conditioning off this table
                             def_team=(away_id if is_home else home_id),
                             off_lineup=",".join(map(str, off5)),
                             def_lineup=",".join(map(str, def5)),
                             points=cur_pts, ended_by=ended_by,
                             seconds=max(t_end - t_start, 0.0)))
        cur_team, cur_pts, t_start = None, 0, t_end

    for a in acts:
        at = a.get("actionType", "")
        team = a.get("teamId")
        t = _elapsed(a.get("period"), a.get("clock"))
        if t is None or at in ("Substitution", "Timeout", "Instant Replay"):
            continue
        if cur_team is None and team and at in ("Made Shot", "Missed Shot", "Turnover",
                                                "Free Throw", "Foul"):
            cur_team = team; t_start = t
        if at == "Made Shot" and team == cur_team:
            sv = a.get("shotValue")
            cur_pts += 2 if sv is None else int(sv)  # explicit None check
                       # (`or 2` also coerced an explicit 0 shotValue to 2)
            desc = a.get("description") or ""
            # and-one: foul on the make -> FTs continue the trip
            pending_ft = "AST" not in desc and False  # conservative: close unless FT follows
            close(t, "made_fg")
        elif at == "Turnover" and team == cur_team:
            close(t, "turnover")
        elif at == "Rebound":
            # defensive rebound by the OTHER team ends the possession
            if cur_team is not None and team and team != cur_team:
                close(t, "dreb")
                cur_team = team; t_start = t     # new possession starts
        elif at == "Free Throw" and team == cur_team:
            desc = a.get("description") or ""
            made = "MISS" not in desc
            if made:
                cur_pts += 1
            # last FT of trip? crude: '2 of 2', '3 of 3', '1 of 1'
            tag = a.get("subType") or ""
            if any(k in tag for k in ("1 of 1", "2 of 2", "3 of 3")):
                if made:
                    close(t, "made_ft")
                # missed final FT -> rebound decides; leave possession open
        elif at == "period":
            close(t, "period_end")
    return poss


COLS = ["game_id", "poss_idx", "off_team", "def_team", "off_lineup",
        "def_lineup", "points", "ended_by", "seconds"]


def build(connect_fn, limit=None, force: bool = False, chunk_games: int = 500,
          only_002: bool = False) -> dict:
    """Build/refresh possessions_v2 from the rotation+PBP cache.

    D100: the incremental `have` guard below is correct for *appending* new
    games but it made the table UNREPAIRABLE — every one of the 2,123 games
    written on 2026-07-28 (pre-D81, `def_team = 0` on 100% of rows and
    off_lineup/off_team agreement 0.5004) was permanently skipped. `force=True`
    truncates the table first so a fixed parser can regenerate it. Writes are
    CHUNKED (`chunk_games`) so a full-corpus rebuild never holds one giant
    transaction; the DB is a single-writer store, so check no loader is running
    before calling with force=True.

    only_002: restrict to regular-season game_ids (the table itself keeps the
    game_id, so consumers can also filter downstream; the default keeps the
    substrate complete).
    """
    import pandas as pd
    from .cache_index import game_index
    rots = game_index("gamerotation")
    pbps = game_index("playbyplayv3")
    gids = sorted(set(rots) & set(pbps))
    if only_002:
        gids = [g for g in gids if g.startswith("002")]
    if limit:
        gids = gids[:limit]

    con = connect_fn()
    con.execute(SCHEMA)
    if force:
        n_old = con.execute("SELECT count(*) FROM possessions_v2").fetchone()[0]
        con.execute("DELETE FROM possessions_v2")
        log.info("possessions_v2: force rebuild, dropped %d stale rows", n_old)
        have = set()
    else:
        have = {r[0] for r in
                con.execute("SELECT DISTINCT game_id FROM possessions_v2").fetchall()}
    con.close()
    todo = [g for g in gids if g not in have]

    n_games = n_rows = 0
    for start in range(0, len(todo), chunk_games):
        rows = []
        for gid in todo[start:start + chunk_games]:
            try:
                pbp = orjson.loads(open(pbps[gid], "rb").read())["response"]
                rot = orjson.loads(open(rots[gid], "rb").read())["response"]
                ps = parse_game(pbp, rot)
            except Exception:
                log.exception("possession parse failed %s", gid)
                continue
            n_games += 1
            for i, p in enumerate(ps):
                rows.append([gid, i, p["off_team"], p["def_team"], p["off_lineup"],
                             p["def_lineup"], p["points"], p["ended_by"], p["seconds"]])
        if not rows:
            continue
        dfp = pd.DataFrame(rows, columns=COLS)  # noqa: F841 (duckdb scans locals)
        con = connect_fn()
        con.execute("INSERT OR REPLACE INTO possessions_v2 SELECT * FROM dfp")  # bulk
        con.close()
        n_rows += len(rows)
        log.info("possessions_v2: %d/%d games, %d possessions written",
                 n_games, len(todo), n_rows)
    log.info("possessions_v2: %d games -> %d possessions", n_games, n_rows)
    return {"games": n_games, "possessions": n_rows}
