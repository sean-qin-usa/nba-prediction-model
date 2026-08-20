"""Flagged-window rule engine (handoff I.6): games qualify for H-A bets by
PRE-REGISTERED rules evaluated BEFORE seeing any line. No hindsight selection.

RULES (pre-registered 2026-07-28, before any live line was ever captured):
  W1 STAR_STATUS: a player averaging >= STAR_MIN minutes (last 10 games) appears
     on the injury report as OUT / DOUBTFUL / QUESTIONABLE for tonight, or was
     OUT last game and is now expected back (return flag).
  W2 MINUTES_LOAD: a rotation player (>= ROT_MIN min avg) played neither of the
     team's last two games (unannounced absence / rest pattern).
  W3 DEBUT: a player's first game after a trade/signing (roster first-appearance)
     or season debut of a >= STAR_MIN player.
  W4 B2B_STAR_REST_RISK: team on a back-to-back AND has a >= 34-min star aged
     >= 32 (load-management risk profile).
  W5 NEW_COACH (pre-registered 2026-07-29, before any live line): a team's first
     COACH_WINDOW_GAMES regular-season games under a new head coach (in-season
     replacement OR offseason hire with a documented scheme overhaul). Rationale:
     coach changes shift usage/scheme faster than trailing-window rates adapt
     (regime-change audit graded coach-change adaptation C, slowest of the three
     regime types) — flag the window rather than pretend the model has caught up.
     Coach changes come from the manually-maintained COACH_CHANGES registry
     (free news; no feed exists) — each entry must be added BEFORE the games it
     flags are bet, enforced by flagged_ts PIT discipline.
Window scoring: a game is FLAGGED if any rule fires for either team. The rule
that fired is recorded (auditability + per-rule CLV attribution later).

The engine takes an availability snapshot (from the injury-report feed, live in
October) plus our minutes history. Until October, fires only on W2/W4 (derivable
from games alone) — W1/W3 activate when the report feed has data.
"""
from __future__ import annotations

import datetime as dt

STAR_MIN = 30.0   # avg minutes for "star" status
ROT_MIN = 18.0    # rotation player
COACH_WINDOW_GAMES = 15   # W5: games flagged after a head-coach change

# W5 registry: {team_abbrev: first regular-season game date under the new coach}.
# Manually maintained from news (no free feed). PIT rule: an entry only counts
# for games flagged AFTER the entry was added (flagged_ts precedes the bet).
COACH_CHANGES: dict[str, dt.date] = {
    # e.g. "LAL": dt.date(2024, 10, 22),  # JJ Redick first game (historical example)
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS flagged_windows (
    game_date   DATE NOT NULL,
    game_id     VARCHAR,
    team        VARCHAR NOT NULL,
    rule        VARCHAR NOT NULL,        -- W1..W4
    player_id   BIGINT,
    detail      VARCHAR,
    flagged_ts  TIMESTAMPTZ NOT NULL,    -- when flagged (PIT: must precede bet)
    PRIMARY KEY (game_date, team, rule, player_id)
);
"""


def recent_minutes(con, before: dt.date, n_games: int = 10):
    """{player_id: (team_id, avg_min, games_missed_last2)} from trailing data."""
    df = con.execute("""
        WITH pg AS (
          SELECT s.player_id, s.team_id, s.seconds/60.0 m, g.game_date,
                 row_number() OVER (PARTITION BY s.player_id ORDER BY g.game_date DESC) rn
          FROM player_game_stats s
          JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
          WHERE s.game_id LIKE '002%' AND g.game_date < ? AND s.seconds >= 720
        )
        SELECT player_id, arg_max(team_id, game_date) team_id, avg(m) avg_min, max(game_date) last_played
        FROM pg WHERE rn <= ? GROUP BY player_id
    """, [before, n_games]).fetchdf()
    return df


def team_last_games(con, team_id: int, before: dt.date, k: int = 2):
    # seasons derived from `before` (current + previous): the old hardcoded
    # ('2024-25','2025-26') literal would have made W2 look back to April 2026
    # for "last two games" once 2026-27 started (external review round 6 #4).
    from ..config import current_season, prev_season
    cur = current_season(before)
    return [r[0] for r in con.execute("""
        SELECT DISTINCT g.game_date FROM nba_games g
        WHERE g.team_id = ? AND g.season IN (?, ?) AND g.game_id LIKE '002%'
          AND g.game_date < ? ORDER BY g.game_date DESC LIMIT ?""",
        [team_id, prev_season(cur), cur, before, k]).fetchall()]


def flag_games(con, game_date: dt.date, availability: dict | None = None) -> list[dict]:
    """Evaluate rules for all games on `game_date`. `availability` is the parsed
    injury-report snapshot {player_id: status} (None until the feed is live).
    Returns flag records; caller persists + uses to gate bets."""
    flags = []
    rm = recent_minutes(con, game_date)
    games = con.execute("""SELECT DISTINCT game_id, team_id, team_abbrev FROM nba_games
        WHERE game_date = ? AND game_id LIKE '002%'""", [game_date]).fetchdf()
    now = dt.datetime.now(dt.timezone.utc)

    for r in games.itertuples():
        team_players = rm[rm.team_id == r.team_id]
        last2 = team_last_games(con, int(r.team_id), game_date, 2)
        # W5: within first COACH_WINDOW_GAMES games under a new head coach
        ch = COACH_CHANGES.get(r.team_abbrev)
        if ch and ch <= game_date:
            n_since = con.execute("""
                SELECT count(DISTINCT game_id) FROM nba_games
                WHERE team_id = ? AND game_id LIKE '002%'
                  AND game_date >= ? AND game_date < ?""",
                [int(r.team_id), ch, game_date]).fetchone()[0]
            if n_since < COACH_WINDOW_GAMES:
                flags.append(dict(game_date=game_date, game_id=r.game_id,
                                  team=r.team_abbrev, rule="W5", player_id=0,  # team-level: PK forbids NULL
                                  detail=f"new coach game {n_since + 1}/{COACH_WINDOW_GAMES}",
                                  flagged_ts=now))
        for p in team_players.itertuples():
            # W1: injury-report status on a star (needs availability feed)
            if availability and p.avg_min >= STAR_MIN:
                st = availability.get(int(p.player_id))
                if st in ("OUT", "DOUBTFUL", "QUESTIONABLE"):
                    flags.append(dict(game_date=game_date, game_id=r.game_id,
                                      team=r.team_abbrev, rule="W1",
                                      player_id=int(p.player_id),
                                      detail=f"{st} star {p.avg_min:.0f}min",
                                      flagged_ts=now))
            # W2: rotation player missed both of last two team games
            last_played = p.last_played.date() if hasattr(p.last_played, "date") else p.last_played
            days_out = (game_date - last_played).days
            # NEW absences only (onset <=10 days): long-term injuries are stale
            # news the market fully prices; the window is the fresh change.
            if p.avg_min >= ROT_MIN and len(last2) == 2 and last_played < last2[-1] and days_out <= 10:
                flags.append(dict(game_date=game_date, game_id=r.game_id,
                                  team=r.team_abbrev, rule="W2",
                                  player_id=int(p.player_id),
                                  detail=f"missed last2, avg {p.avg_min:.0f}min",
                                  flagged_ts=now))
    return flags


# W6 MODEL-EDGE WINDOW (pre-registered 2026-07-30, before any live line):
# EARLY SEASON — either team <20 games played. Evidence: carry-seeded model
# BEAT the close outright in the 2024-25 early window (0.5973 vs 0.6048) and
# gp[0,5) carry value +0.0154 CI-solid; the market's implicit ratings converge
# SLOWER than our continuity-carried model in October. Bet eligibility only —
# sizing waits for live CLV validation (evidence base: 1 season absolute beat;
# W6 must show positive CLV in October before real stakes).
# Star-out note (measured 2026-07-30): star-out games are a RELATIVE strength
# (market edge over us 44% smaller: +0.0125 vs +0.0223) -> routed to the PROPS
# program (redistribution science vs soft books), not main-line betting.
EARLY_SEASON_GP = 20

def w6_early_season(games_played_home: int, games_played_away: int) -> bool:
    return min(games_played_home, games_played_away) < EARLY_SEASON_GP
