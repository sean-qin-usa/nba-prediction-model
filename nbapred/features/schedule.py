"""Rest / schedule / travel fatigue features (docs/SIGNALS.md #1).

Per team-game, entirely from the schedule (nba_games) + static arena geo:
  days_rest, is_b2b, is_3in4, is_4in5, games_last_7,
  travel_km (haversine from previous game city), tz_shift,
  and the opponent-differenced versions (rest_adv, travel_adv) — the actual
  edge lives in the *asymmetry* between the two teams, not the absolute.

Home/away is parsed from the matchup string (robust: "A @ B" -> B home,
"A vs. B" -> A home), not the stored is_home flag (which mis-set on some
preseason rows where both team-rows carried the "@" string).

NEUTRAL COURTS (D140 fix, the defect D137 flagged here): `matchup` names a
NOMINAL host even when the game is played at a neutral site, so this module
used to record those as ordinary home games AND carry the nominal host forward
as the origin of each team's next game. Neutral games are now detected
mechanically by `nbapred.model.travel.neutral_game_venues` (the 2020 Orlando
bubble window + the feed's `is_home`-FALSE-on-both-rows marker); at a neutral
court `is_home` is FALSE for both teams and the geo chain follows the VENUE.
Travel across a hiatus longer than `travel.HIATUS_RESET_DAYS` resets to 0 —
the acute load has washed out. `travel_km`/`tz_shift` are NULL (not 0) when a
neutral venue's coordinates are unknown, so a fiction is never stored.
"""
from __future__ import annotations

import datetime as dt
import logging
import math

from ..config import CURRENT_SEASON

log = logging.getLogger("schedule")

# (lat, lon, utc_offset_hours) per team. Offsets ignore DST (used only for
# tz_shift deltas, where the DST error cancels between same-country cities).
ARENAS = {
    "ATL": (33.757, -84.396, -5), "BOS": (42.366, -71.062, -5), "BKN": (40.683, -73.975, -5),
    "CHA": (35.225, -80.839, -5), "CHI": (41.881, -87.674, -6), "CLE": (41.497, -81.688, -5),
    "DAL": (32.790, -96.810, -6), "DEN": (39.749, -105.007, -7), "DET": (42.341, -83.055, -5),
    "GSW": (37.768, -122.388, -8), "HOU": (29.751, -95.362, -6), "IND": (39.764, -86.155, -5),
    "LAC": (34.043, -118.267, -8), "LAL": (34.043, -118.267, -8), "MEM": (35.138, -90.051, -6),
    "MIA": (25.781, -80.187, -5), "MIL": (43.045, -87.917, -6), "MIN": (44.979, -93.276, -6),
    "NOP": (29.949, -90.082, -6), "NYK": (40.751, -73.993, -5), "OKC": (35.463, -97.515, -6),
    "ORL": (28.539, -81.384, -5), "PHI": (39.901, -75.172, -5), "PHX": (33.446, -112.071, -7),
    "POR": (45.532, -122.667, -8), "SAC": (38.580, -121.500, -8), "SAS": (29.427, -98.437, -6),
    "TOR": (43.643, -79.379, -5), "UTA": (40.768, -111.901, -7), "WAS": (38.898, -77.021, -5),
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS schedule_features (
    season      VARCHAR NOT NULL,
    game_id     VARCHAR NOT NULL,
    game_date   DATE,
    team        VARCHAR NOT NULL,
    opponent    VARCHAR,
    is_home     BOOLEAN,
    days_rest   INTEGER,          -- NULL for a team's first game of season
    is_b2b      BOOLEAN,
    is_3in4     BOOLEAN,
    is_4in5     BOOLEAN,
    games_last_7 INTEGER,
    travel_km   DOUBLE,           -- from previous game's host city
    tz_shift    INTEGER,          -- host-tz change vs previous game (signed hrs)
    rest_adv    INTEGER,          -- days_rest - opponent days_rest
    travel_adv  DOUBLE,           -- opponent travel_km - own travel_km (positive = we're fresher)
    ingest_ts   TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (season, game_id, team)
);
"""


# Neutral venues that DO have coordinates here, keyed exactly as
# nbapred.model.travel keys them. Anything neutral and absent from this table
# has unknown geo and yields NULL travel/tz rather than a fabricated 0.
from ..model.travel import (BUBBLE_VENUE, HIATUS_RESET_DAYS,  # noqa: E402
                            neutral_game_venues)

NEUTRAL_ARENAS = {
    BUBBLE_VENUE: (28.3382, -81.5494, -5),   # ESPN WWoS, Bay Lake FL
}


def _geo(key: str):
    return ARENAS.get(key) or NEUTRAL_ARENAS.get(key)


def _haversine(a: str, b: str) -> float:
    if _geo(a) is None or _geo(b) is None:
        return None
    (la1, lo1, _), (la2, lo2, _) = _geo(a), _geo(b)
    r = 6371.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dphi, dl = math.radians(la2 - la1), math.radians(lo2 - lo1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _parse_host(matchup: str) -> str | None:
    if not matchup:
        return None
    if "@" in matchup:
        return matchup.split("@")[-1].strip()
    if "vs." in matchup:
        return matchup.split("vs.")[0].strip()
    return None


def build(connect_fn, season: str = CURRENT_SEASON) -> int:
    con = connect_fn()
    con.execute(SCHEMA)
    # regular season only (002*); one row per (game, team)
    rows = con.execute("""
        SELECT game_id, game_date, team_abbrev, matchup
        FROM nba_games
        WHERE season = ? AND game_id LIKE '002%'
        ORDER BY game_date, game_id
    """, [season]).fetchall()

    # group by game to get opponent + host
    by_game: dict[str, list] = {}
    for gid, gdate, team, matchup in rows:
        by_game.setdefault(gid, []).append((gdate, team, matchup))

    neutral = neutral_game_venues(con)

    # per-team chronological history for rest/travel
    hist: dict[str, list] = {}
    feats: dict[tuple, dict] = {}
    for gid in sorted(by_game, key=lambda g: (by_game[g][0][0], g)):
        recs = by_game[gid]
        if len(recs) != 2:
            continue
        (d0, t0, m0), (d1, t1, m1) = recs
        host = _parse_host(m0) or _parse_host(m1)
        nv = neutral.get(gid)
        venue = nv if nv is not None else host   # geo chain follows the VENUE
        gdate = d0
        for team, opp in ((t0, t1), (t1, t0)):
            past = hist.get(team, [])
            days_rest = (gdate - past[-1][0]).days if past else None
            games_last_7 = sum(1 for pd_, _ in past if 0 <= (gdate - pd_).days <= 7)
            last4 = [pd_ for pd_, _ in past if 0 <= (gdate - pd_).days <= 3]
            last5 = [pd_ for pd_, _ in past if 0 <= (gdate - pd_).days <= 4]
            prev_venue = past[-1][1] if past else None
            if prev_venue is None or days_rest > HIATUS_RESET_DAYS:
                travel, tz_shift = 0.0, 0
            elif _geo(venue) is not None and _geo(prev_venue) is not None:
                travel = _haversine(prev_venue, venue)
                tz_shift = _geo(venue)[2] - _geo(prev_venue)[2]
            else:
                # neutral venue with unknown coordinates -> unknown, not zero
                travel, tz_shift = None, None
            feats[(gid, team)] = dict(
                season=season, game_id=gid, game_date=gdate, team=team, opponent=opp,
                # nobody is the home team at a neutral court
                is_home=(team == host and nv is None), days_rest=days_rest,
                is_b2b=(days_rest == 1), is_3in4=(len(last4) >= 2), is_4in5=(len(last5) >= 3),
                games_last_7=games_last_7, travel_km=travel, tz_shift=tz_shift)
            hist.setdefault(team, []).append((gdate, venue))

    # opponent-differenced fields
    now = dt.datetime.now(dt.timezone.utc)
    out = []
    for (gid, team), f in feats.items():
        opp_f = feats.get((gid, f["opponent"]))
        dr, odr = f["days_rest"], (opp_f or {}).get("days_rest")
        rest_adv = (dr - odr) if (dr is not None and odr is not None) else None
        otrav = (opp_f or {}).get("travel_km")
        travel_adv = (otrav - f["travel_km"]
                      if (opp_f and otrav is not None and f["travel_km"] is not None)
                      else None)
        out.append([f["season"], f["game_id"], f["game_date"], f["team"], f["opponent"],
                    f["is_home"], f["days_rest"], f["is_b2b"], f["is_3in4"], f["is_4in5"],
                    f["games_last_7"], f["travel_km"], f["tz_shift"], rest_adv, travel_adv, now])

    con.execute("DELETE FROM schedule_features WHERE season = ?", [season])
    con.executemany("INSERT INTO schedule_features VALUES (" + ",".join("?" * 16) + ")", out)
    con.close()
    log.info("schedule_features %s: %d team-game rows", season, len(out))
    return len(out)
