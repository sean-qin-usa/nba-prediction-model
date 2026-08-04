"""Event-shock detection (V3_SPEC M4; M1 uses only season_boundary).

Shock kinds and sources (the full set is M4 scope):
    season_boundary  nba_games season calendar          [ACTIVE at M1]
    trade            roster first-appearance            [M4]
    return           injury feed, absence >= N games    [M4]
    coach            COACH_CHANGES registry (D40)       [M4]
"""
from __future__ import annotations

import dataclasses
import datetime as dt


@dataclasses.dataclass(frozen=True)
class Shock:
    entity_id: int          # player_id or team_id (team-wide kinds)
    kind: str               # season_boundary | trade | return | coach
    date: dt.date
    source: str = ""


def detect_shocks(con, date: dt.date) -> list[Shock]:
    """Shocks EFFECTIVE on `date`. M1: season boundary only — if `date` is a
    season's first regular-season game date, every team gets one."""
    row = con.execute("""
        SELECT min(game_date) FROM nba_games
        WHERE game_id LIKE '002%'
          AND season = (SELECT any_value(season) FROM nba_games
                        WHERE game_id LIKE '002%' AND game_date = ?)""",
        [date]).fetchone()
    if not row or row[0] != date:
        return []
    tids = [r[0] for r in con.execute(
        """SELECT DISTINCT team_id FROM nba_games
           WHERE game_id LIKE '002%' AND game_date = ?""", [date]).fetchall()]
    return [Shock(int(t), "season_boundary", date, "season_calendar")
            for t in tids]


def log_shocks(con_rw, shocks: list[Shock]) -> None:
    """Persist to state_shocks (idempotent upsert; v3_writer connection)."""
    for s in shocks:
        con_rw.execute(
            """INSERT OR REPLACE INTO state_shocks VALUES
               (?, ?, ?, ?, current_timestamp)""",
            [s.date, s.entity_id, s.kind, s.source])
