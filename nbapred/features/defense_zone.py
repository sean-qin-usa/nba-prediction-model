"""Per-player zone-split defense: rim protection vs perimeter defense as
SEPARATE skills (a first step toward the handoff's K-dimensional skills).

Box scores collapse defense to STL/BLK; that's too coarse. Here every shot
(from PBP, with its zone) is attributed to the DEFENSIVE lineup on the floor at
that moment (reusing the stint time-segmentation), so we can measure each
player's opponent FG% ALLOWED by zone. Shrunk toward league (EB) so thin
minutes don't produce absurd ratings.

Output per player: rim_def, mid_def, thr_def = points-below-average the player's
presence is associated with allowing in that zone (positive = good defense).
This is what makes a prop defender-aware: a shooter vs an elite rim protector
gets a lower rim make-prob.

Confound (stated, as the handoff does): on-floor allowance is contaminated by
the other four defenders + who the offense chooses to attack. This is the raw
version; the principled fix is zone-split defensive RAPM (ridge over lineups),
a straightforward extension once this is validated.
"""
from __future__ import annotations

import glob

import orjson
import numpy as np

from ..config import RAW_NBA
from .stints import _elapsed


def _defensive_lineup_at(segments, t):
    """segments: sorted [(t0,t1,off_team,def5)]. Return def5 covering time t."""
    for t0, t1, off_team, hdef, adef in segments:
        if t0 <= t < t1:
            return hdef, adef, off_team
    return None, None, None


def _game_segments(rot_raw, pbp_raw):
    """Build [(t0,t1, home5, away5)] from rotation (who's on floor when)."""
    sides = {}
    for rs in rot_raw["resultSets"]:
        side = "home" if rs["name"] == "HomeTeam" else "away"
        for row in rs["rowSet"]:
            d = dict(zip(rs["headers"], row))
            sides.setdefault(side, []).append(
                (d["PERSON_ID"], d["IN_TIME_REAL"] / 10.0, d["OUT_TIME_REAL"] / 10.0))
    bps = sorted({round(x, 1) for s in sides.values() for _, i, o in s for x in (i, o)})
    segs = []
    for k in range(len(bps) - 1):
        t0, t1 = bps[k], bps[k + 1]
        mid = (t0 + t1) / 2
        h = sorted(p for p, i, o in sides.get("home", []) if i <= mid < o)
        a = sorted(p for p, i, o in sides.get("away", []) if i <= mid < o)
        if len(h) == 5 and len(a) == 5:
            segs.append((t0, t1, h, a))
    return segs


def _zone(dist, val, subtype):
    st = (subtype or "").lower()
    if val == 3:
        return "thr"
    if any(k in st for k in ("layup", "dunk", "tip", "hook", "putback")) or (dist is not None and dist <= 4):
        return "rim"
    return "mid"


def accumulate(rot_raw, pbp_raw, home_id, allowed):
    """Attribute each shot to the DEFENDING five; tally attempts/makes by zone
    into allowed[player_id][zone] = [att, made]."""
    segs = _game_segments(rot_raw, pbp_raw)
    if not segs:
        return
    seg_t0 = np.array([s[0] for s in segs])
    for a in pbp_raw.get("game", {}).get("actions", []):
        if a.get("actionType") not in ("Made Shot", "Missed Shot"):
            continue
        t = _elapsed(a.get("period"), a.get("clock"))
        if t is None:
            continue
        k = int(np.searchsorted(seg_t0, t, side="right") - 1)
        if k < 0 or k >= len(segs):
            continue
        _, _, h5, a5 = segs[k]
        shooter_team = a.get("teamId")
        # defenders = the team that is NOT shooting
        defenders = a5 if shooter_team == home_id else h5
        z = _zone(a.get("shotDistance"), a.get("shotValue"), a.get("subType"))
        made = a.get("shotResult") == "Made"
        for d in defenders:
            cell = allowed.setdefault(d, {"rim": [0, 0], "mid": [0, 0], "thr": [0, 0]})
            cell[z][0] += 1
            cell[z][1] += int(made)


def games_before(before, only_002: bool = True, con=None) -> set:
    """game_ids strictly before `before` (date/str) — the PIT universe.

    D100 hazard #3: `build_zone_defense` used to scan the ENTIRE cache with no
    game-type filter and no date cutoff, so any backtest calling it consumed
    future information *and* preseason/playoff games. This resolves the cutoff
    against `nba_games`; games missing from the schedule table are excluded
    (safe direction: never admit a game we cannot date)."""
    import datetime as _dt
    own = con is None
    if own:
        from ..db import connect
        con = connect(read_only=True)
    try:
        q = "SELECT DISTINCT game_id FROM nba_games WHERE game_date < ?"
        if only_002:
            q += " AND game_id LIKE '002%'"
        if isinstance(before, str):
            before = _dt.date.fromisoformat(before)
        return {r[0] for r in con.execute(q, [before]).fetchall()}
    finally:
        if own:
            con.close()


def build_zone_defense(limit=None, only_002: bool = True, before=None,
                       allowed_gids=None, con=None):
    """Scan cached rotation+PBP, return per-player shrunk zone-defense ratings
    (positive = suppresses that zone's FG% below league).

    only_002    : regular season only (D100: was unfiltered — preseason/playoff
                  games entered every rating).
    before      : date/ISO string; keep only games strictly before it (PIT).
                  Resolved against `nba_games`, so it needs the DB.
    allowed_gids: explicit game-id whitelist (leakage-safe train split), same
                  contract as `def_rapm.collect_shots`. Applied on top of the
                  other two filters.
    """
    from .cache_index import game_index
    rots = game_index("gamerotation")
    pbps = game_index("playbyplayv3")
    gids = sorted(set(rots) & set(pbps))
    if only_002:
        gids = [g for g in gids if g.startswith("002")]
    if before is not None:
        ok = games_before(before, only_002=only_002, con=con)
        gids = [g for g in gids if g in ok]
    if allowed_gids is not None:
        allowed_gids = set(allowed_gids)
        gids = [g for g in gids if g in allowed_gids]
    if limit:
        gids = gids[:limit]

    # probe-verified: cached playbyplayv3 `game` has NO homeTeamId, so the old
    # lookup was ALWAYS None -> `defenders = a5 if shooter == home_id else h5`
    # attributed EVERY shot to the home five. Derive ids from the rotation feed
    # instead (same fix as possessions_v2). Imported here, not at module scope,
    # because possessions_v2 imports this module.
    from .possessions_v2 import _team_ids

    allowed = {}
    league = {"rim": [0, 0], "mid": [0, 0], "thr": [0, 0]}
    for gid in gids:
        try:
            rot = orjson.loads(open(rots[gid], "rb").read())["response"]
            pbp = orjson.loads(open(pbps[gid], "rb").read())["response"]
            home_id, _ = _team_ids(rot, pbp)
            if home_id is None:
                continue
            accumulate(rot, pbp, home_id, allowed)
        except Exception:
            continue
    # league baselines
    for cells in allowed.values():
        for z, (att, made) in cells.items():
            league[z][0] += att; league[z][1] += made
    lg = {z: (league[z][1] / league[z][0] if league[z][0] else 0.5) for z in league}

    # EB shrinkage: opponent FG% allowed vs league, per zone. Positive rating =
    # allowed BELOW league (good D). Units: FG% points * 100 -> ~ pts scale.
    K = {"rim": 60, "mid": 80, "thr": 100}   # attempts-of-prior per zone
    out = {}
    for pid, cells in allowed.items():
        r = {}
        for z, (att, made) in cells.items():
            if att < 10:
                r[z] = 0.0
                continue
            shrunk = (made + K[z] * lg[z]) / (att + K[z])
            r[z] = float((lg[z] - shrunk) * 100)   # >0 = suppresses zone
            r[z + "_att"] = att
        out[pid] = r
    return out, lg
