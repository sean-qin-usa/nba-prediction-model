"""MECHANISM-BASED travel / circadian / schedule-density state, per team-game.

Everything here is a GAME-LEVEL PHYSICAL FACT derived from the schedule plus a
static arena table (data/arenas.csv). That is the whole point of the module:
D20/D70 (team-specific home advantage) and D96 (altitude-as-a-city-dummy) both
died of NONSTATIONARITY because they tested team/city IDENTITY, whose meaning
drifts across eras. Boston->LA is 4,169 km in 2021 and 4,169 km in 2026, and a
3-zone eastward flight is the same phase shift in both. Nothing here is fit per
team or per city.

PIT: every field is a pure schedule fact, published months before tip. No
outcome, no box score, no market input touches this file.

NEUTRAL SITES (D140 fix; the bug D139 found). A game played at a neutral court
is NOT played at the nominal host's arena, so deriving the venue from `matchup`
is wrong twice over: it invents travel INTO the game, and it then carries the
wrong origin OUT of it into the team's next game. Two neutral classes exist in
this corpus and both are detected mechanically, never by hand:
  * the 2020 ORLANDO BUBBLE — all 88 regular-season restart games
    (2020-07-30..2020-08-14, era E1) were played at one complex at Walt Disney
    World. The old code assigned 1,505.5 km/team-game; the truth is 0.
  * the feed's own NEUTRAL-COURT FLAG — `is_home` is FALSE on BOTH team-rows
    (D137: 10 such games, Mexico City / Paris / NBA Cup semifinals in Vegas).
Venue chaining is on the VENUE, not the nominal host, so the origin of a team's
next game is where it actually played.

KNOWN LIMITATIONS (registered, not hidden):
  * The 10 feed-flagged neutral games have no venue coordinates in this repo,
    so their travel is UNKNOWN, not zero. Those team-games — and the next
    team-game after each, whose origin is equally unknown — carry
    `travel_valid=False` and travel_km/tz_east/elev_gain_m of 0.0. Consumers
    must DROP them, not score them. `is_home=FALSE-on-both` is also INCOMPLETE
    before 2024-25 (D137: Dec-2023 Cup semifinals in Vegas are unmarked),
    leaving ~2-4 games/season undetected in 2021-22..2023-24.
  * SAS played a few nominal home games in Austin (2022-23/2023-24); Moody
    Center is 120 km from Frost Bank Center, immaterial at this resolution.
  * LAC uses Intuit Dome (current arena rule); it is 12.9 km from the
    Crypto.com Arena they shared with LAL through 2023-24.
"""
from __future__ import annotations

import csv
import datetime as dt
import math
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

_ARENA_CSV = Path(__file__).resolve().parents[2] / "data" / "arenas.csv"

EARTH_R_KM = 6371.0


@lru_cache(maxsize=1)
def arenas() -> dict:
    """{abbrev: dict(lat, lon, elev_m, tz)} from data/arenas.csv."""
    out = {}
    with open(_ARENA_CSV) as fh:
        for r in csv.DictReader(fh):
            out[r["team"]] = dict(lat=float(r["lat"]), lon=float(r["lon"]),
                                  elev_m=float(r["elev_m"]),
                                  tz=ZoneInfo(r["tz"]), arena=r["arena"])
    return out


# ---------------------------------------------------------- neutral-site data
# The 2020 restart: 88 regular-season games, one complex, no travel between
# them. Dates are the corpus's own first/last restart game; the window is
# written wide enough to cover the scheduled restart period.
BUBBLE_SEASON = "2019-20"
BUBBLE_FROM = dt.date(2020, 7, 30)
BUBBLE_TO = dt.date(2020, 10, 11)
BUBBLE_VENUE = "WWOS_ORL"

# Venue keys are deliberately NOT valid team abbreviations, so they can never
# collide with an arenas.csv row.
NEUTRAL_VENUES = {
    BUBBLE_VENUE: dict(lat=28.3382, lon=-81.5494, elev_m=21.0,
                       tz=ZoneInfo("America/New_York"),
                       arena="ESPN Wide World of Sports Complex, Bay Lake FL"),
}

# A gap this long fully washes out acute travel load, so the next game starts
# a fresh chain — the same assumption build_state already makes at the start of
# a season ("training camp is at home"). NOT a tuned parameter: the longest
# inter-game gap in every scorable season (2020-21..2025-26) is 13 days, so on
# this corpus ANY threshold in [14, 140] selects exactly the 22 bubble restart
# games and nothing else.
HIATUS_RESET_DAYS = 14


@lru_cache(maxsize=1)
def venues() -> dict:
    """arenas() plus the known neutral venues, keyed the same way."""
    return {**arenas(), **NEUTRAL_VENUES}


def haversine_km(a: str, b: str) -> float:
    A = venues()
    if a not in A or b not in A or a == b:
        return 0.0
    p1, p2 = math.radians(A[a]["lat"]), math.radians(A[b]["lat"])
    dphi = math.radians(A[b]["lat"] - A[a]["lat"])
    dl = math.radians(A[b]["lon"] - A[a]["lon"])
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(h))


def utc_offset_h(team: str, day: dt.date) -> float:
    """True UTC offset in hours at a 19:30 local tip (DST-correct). PHX never
    observes DST, so it sits with Mountain in winter and Pacific in October and
    late March/April — the fixed-offset table in nbapred/features/schedule.py
    gets that wrong. Accepts a neutral VENUE key as well as a team abbrev."""
    A = venues()
    if team not in A:
        return 0.0
    naive = dt.datetime(day.year, day.month, day.day, 19, 30)
    return naive.replace(tzinfo=A[team]["tz"]).utcoffset().total_seconds() / 3600.0


def neutral_game_venues(con) -> dict:
    """{game_id: venue_key} for every regular-season game at a neutral court.

    Two mechanical sources, no hand-maintained game list:
      1. the 2020 Orlando bubble date window (season 2019-20);
      2. the feed's own neutral-court marker — `is_home` FALSE on BOTH
         team-rows (D137). Those venues are not in NEUTRAL_VENUES, so they get
         a per-game key with unknown coordinates and are reported
         `travel_valid=False` rather than silently scored as 0 km of real
         travel.
    """
    out = {}
    for (gid,) in con.execute(
            """SELECT DISTINCT game_id FROM nba_games
               WHERE game_id LIKE '002%' AND season = ?
                 AND game_date >= ? AND game_date <= ?""",
            [BUBBLE_SEASON, BUBBLE_FROM, BUBBLE_TO]).fetchall():
        out[gid] = BUBBLE_VENUE
    for (gid,) in con.execute(
            """SELECT game_id FROM nba_games WHERE game_id LIKE '002%'
               GROUP BY game_id
               HAVING count(*) = 2
                  AND sum(CASE WHEN is_home THEN 1 ELSE 0 END) = 0""").fetchall():
        out.setdefault(gid, "NEUTRAL:" + str(gid))
    return out


def parse_host(matchup: str) -> str | None:
    """'A @ B' -> B is host; 'A vs. B' -> A is host. Robust to the preseason
    is_home mis-set noted in nbapred/features/schedule.py."""
    if not matchup:
        return None
    if "@" in matchup:
        return matchup.split("@")[-1].strip()
    if "vs." in matchup:
        return matchup.split("vs.")[0].strip()
    return None


# ---------------------------------------------------------------- state build

def build_state(con, since: dt.date | None = None) -> dict:
    """{(team_id, game_date): state-dict} for every regular-season team-game.

    Sequence is per (team, season): a team's first game of a season has its own
    arena as origin (training camp is at home), travel 0, tz shift 0. A gap of
    more than HIATUS_RESET_DAYS does the same (the 2020 restart).

    Fields (all schedule-only):
      travel_km   great-circle km from the VENUE of the team's PREVIOUS game in
                  this season to tonight's VENUE (neutral courts included)
      tz_east     signed UTC-offset change, hours, POSITIVE = travelled EAST
      road_len    n-th consecutive road game including tonight (0 at home);
                  nobody is at home at a neutral site
      home_return 1 if tonight is a home game whose previous game was on the
                  road at the end of a >=3-game road trip
      is_3in4     tonight is the 3rd game in a 4-night window
      is_5in7     tonight is the 5th game in a 7-night window
      b2b         previous game was yesterday
      elev_gain_m tonight's venue elevation minus the previous venue's
      neutral     tonight is at a neutral court (neither team is home)
      venue       the venue key actually used for the geo chain
      host        the NOMINAL host abbrev parsed from `matchup` (unchanged;
                  kept for callers that need the fixture's nominal side)
      travel_valid False when travel_km / tz_east / elev_gain_m could not be
                  computed because this venue or the previous one has unknown
                  coordinates. They are reported as 0.0; DROP these team-games
                  rather than scoring them.
    """
    q = """SELECT season, game_id, game_date, team_id, team_abbrev, matchup
           FROM nba_games WHERE game_id LIKE '002%'"""
    params = []
    if since is not None:
        q += " AND game_date >= ?"
        params.append(since)
    q += " ORDER BY game_date, game_id"
    rows = con.execute(q, params).fetchall()

    by_game = {}
    for season, gid, gd, tid, ab, mu in rows:
        gd = gd.date() if hasattr(gd, "date") else gd
        by_game.setdefault(gid, []).append((season, gd, tid, ab, mu))

    A = venues()
    neutral = neutral_game_venues(con)
    # (team_id, season) -> [(date, venue_key, was_home, venue_known)]
    hist = {}
    out = {}
    for gid in sorted(by_game, key=lambda g: (by_game[g][0][1], g)):
        recs = by_game[gid]
        if len(recs) != 2:
            continue
        host = parse_host(recs[0][4]) or parse_host(recs[1][4])
        nv = neutral.get(gid)
        # A neutral court is the venue; otherwise the nominal host's arena is.
        venue = nv if nv is not None else host
        venue_known = venue in A
        season, gd = recs[0][0], recs[0][1]
        for _, _, tid, ab, _mu in recs:
            key = (tid, season)
            past = hist.get(key, [])
            # nobody is the home team at a neutral court
            at_home = (ab == host) and nv is None
            if past:
                pdate, pvenue, p_home, p_known = past[-1]
                b2b = (gd - pdate).days == 1
                if (gd - pdate).days > HIATUS_RESET_DAYS:
                    # acute travel load has fully washed out — start a fresh
                    # chain, exactly as at the start of a season
                    travel, tz_e, elev, valid = 0.0, 0.0, 0.0, True
                elif venue_known and p_known:
                    travel = haversine_km(pvenue, venue)
                    tz_e = utc_offset_h(venue, gd) - utc_offset_h(pvenue, pdate)
                    elev = A[venue]["elev_m"] - A[pvenue]["elev_m"]
                    valid = True
                else:
                    # a neutral venue whose coordinates we do not have: the
                    # honest answer is "unknown", not "zero km travelled"
                    travel, tz_e, elev, valid = 0.0, 0.0, 0.0, False
            else:
                travel, tz_e, b2b, elev, valid = 0.0, 0.0, False, 0.0, True
            # consecutive road games including tonight
            if at_home:
                road_len = 0
            else:
                road_len = 1
                for _pd, _pv, p_home, _pk in reversed(past):
                    if p_home:
                        break
                    road_len += 1
            # homestand return: home tonight, away last game, trip was >=3 long
            home_return = 0.0
            if at_home and past and not past[-1][2]:
                trip = 0
                for _pd, _pv, p_home, _pk in reversed(past):
                    if p_home:
                        break
                    trip += 1
                home_return = 1.0 if trip >= 3 else 0.0
            d3 = sum(1 for pdate, *_ in past if 1 <= (gd - pdate).days <= 3)
            d7 = sum(1 for pdate, *_ in past if 1 <= (gd - pdate).days <= 6)
            out[(tid, gd)] = dict(
                travel_km=travel, tz_east=tz_e, road_len=float(road_len),
                home_return=home_return, is_3in4=float(d3 >= 2),
                is_5in7=float(d7 >= 4), b2b=bool(b2b), elev_gain_m=elev,
                host=host, at_home=at_home, venue=venue,
                neutral=bool(nv is not None), travel_valid=bool(valid))
            hist.setdefault(key, []).append((gd, venue, at_home, venue_known))
    return out


# ------------------------------------------------------- registered arm terms
# EXACTLY the gated forms of data/travel_prereg.md section 2
# (sha256 d3d334b92665af13dae7914133af626c9d8c1993982a67023a40810c2fbb5a3e).
#
# The response is HOME MARGIN. ARM A/B/D are ANTISYMMETRIC (home minus away):
# 1 dof instead of 2, and the antisymmetry is itself the physiological claim —
# the insult belongs to the body that suffered it, not to the uniform. ARM C's
# two regressors are naturally one-sided (a team at home has road_len 0, and a
# team on the road cannot be in the homestand-return state), which is confirmed
# empirically: the mirrored columns are identically zero.
#
# side: "d" = home value minus away value, "h" = home team's own state only,
#       "a" = away team's own state only.

ARM_TERMS = {
    # A — acute travel load, POINTS PER 1,000 GREAT-CIRCLE KM.  PREDICT NEGATIVE
    "A": [("dtrav_kkm", lambda s: s["travel_km"] / 1000.0)],
    # B — signed circadian phase shift, POINTS PER ZONE CROSSED EASTWARD.
    #     PREDICT POSITIVE (acute phase-at-tip beats re-entrainment cost)
    "B": [("dtz_east", lambda s: s["tz_east"])],
    # C — road-trip state.  PREDICT hret_h NEGATIVE, rlen_extra_a POSITIVE
    "C": [("hret_h", lambda s: s["home_return"]),
          ("rlen_extra_a", lambda s: max(s["road_len"] - 1.0, 0.0))],
    # D — schedule density beyond b2b.  PREDICT BOTH NEGATIVE
    "D": [("d3in4", lambda s: s["is_3in4"]), ("d5in7", lambda s: s["is_5in7"])],
}

TERM_SIDE = {"dtrav_kkm": "d", "dtz_east": "d", "hret_h": "h",
             "rlen_extra_a": "a", "d3in4": "d", "d5in7": "d"}

# pre-registered sign predictions on HOME MARGIN (+1 = positive, -1 = negative)
TERM_PRED = {"dtrav_kkm": -1, "dtz_east": +1, "hret_h": -1,
             "rlen_extra_a": +1, "d3in4": -1, "d5in7": -1}


def arm_columns(arms) -> list:
    cols = []
    for a in arms:
        cols += ARM_TERMS[a]
    return cols


def term_value(name, fn, sh, sa) -> float:
    """Value of the named regressor from the home-team and away-team states.
    Used identically at FIT time and at APPLY time (no second construction)."""
    side = TERM_SIDE.get(name)
    if side == "d":
        return fn(sh) - fn(sa)
    return fn(sh) if side == "h" else fn(sa)
