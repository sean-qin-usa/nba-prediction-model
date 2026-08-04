"""October composition bridge — D84-A live path.

DEFECT FIXED (measured, rw_early_decomp / rw_early_analysis): during each
season's first refit window (~week 1) CompositionModel has an EMPTY rotation
(ROSTER_DAYS=12 excludes everyone in October; comp is 002-only), so the
shipped blend averages a LITERAL 0-margin at 50% weight AND the outs channel
is dead (outs are drawn from comp.players). That is 49% of the entire
early-season (gp<20) loss; the bridge construction reached MARKET PARITY on
week 1 in discovery (active +0.0340 CI(+0.0049,+0.0631), D84-A).

CONSTRUCTION (verbatim the pre-registered gate construction in
scripts/rw_early_v1_gate.py — no fitting, no tuned params, all inputs
strictly before the cutoff):
  roster(t) = {p appeared (seconds>0) in a 001 preseason game for t this
               season; team = argmax 001 minutes}
            UNION {p whose PRIOR-season primary 002 team (arg_max game_date)
               is t and who appears in no team's 001 this season}
  contrib(p) = DARKO_dpm(as-of < cutoff) x trail_min(p)/48
               trail_min = avg of last 10 games with seconds>=720 in 002 data
               strictly before the cutoff (spans the prior season; rookies
               with no 002 history contribute nothing)
  cm_ps(h,a) = sum contrib over roster(h) minus outs  -  same for away
               (live outs = injury feed / departed filter, exactly the sets
               predict_today already passes to Predictor.margin)

ACTIVATION: ONLY when the composition rotation is empty for BOTH teams of a
game (the dead-zero state; equivalently cm == 0 via empty roster). All other
games are bitwise unchanged — see production.fit_production wiring and
tests/test_october_bridge.py (D73-style parity + zero-outside-window).

STATUS: shipped as the D84-A live package; the gate itself is the FREEZE-LIST
October-2026 one-shot confirm (F6, discovery-data disclosure in the gate
script). Env kill-switch OCT_BRIDGE=0 restores the dead-zero behavior for
same-run controls in the one-shot.
"""
from __future__ import annotations

import datetime as dt

from .composition import CompositionModel, ROSTER_DAYS


def _ref_date(before: dt.date | None) -> dt.date:
    return before or dt.date.today()


def rotation_empty(comp: CompositionModel, home_id: int, away_id: int,
                   game_date: dt.date | None = None) -> bool:
    """True iff NO rostered comp player of either team is inside the
    ROSTER_DAYS window as of `game_date` — the D84-A dead state, in which
    comp.margin(home_edge=0) is exactly 0.0 and outs are inert. Ignores outs
    on purpose: an all-outs team still has a live rotation."""
    ref = game_date or comp.asof or dt.date.today()
    for p in comp.players.values():
        if (p["team_id"] in (home_id, away_id)
                and (ref - p["last_played"]).days <= ROSTER_DAYS):
            return False
    return True


def missing_rotation_teams(con, comp: CompositionModel, season: str,
                           before: dt.date | None = None) -> set[int]:
    """Teams (this season's 30, via current+prev nba_games rows) with an
    EMPTY comp rotation as of the fit cutoff — nonempty => fit_production
    builds the bridge. Mid-season this is empty and costs one dict scan."""
    from ..config import prev_season
    ref = comp.asof or _ref_date(before)
    alive = {p["team_id"] for p in comp.players.values()
             if (ref - p["last_played"]).days <= ROSTER_DAYS}
    teams = {int(t) for (t,) in con.execute(
        "SELECT DISTINCT team_id FROM nba_games WHERE season IN (?, ?) "
        "AND game_id LIKE '002%'",     # 002 only: preseason 001 rows include
        [season, prev_season(season)]).fetchall()}   # non-NBA exhibition opps
    return teams - alive


class OctoberBridge:
    """cm_ps table for one (season, cutoff): per-player (team, contrib).

    trail_seasons (D100/D105): cap the trailing-minutes lookback to the N most
    recent seasons before `season`. The roster UNION leg is already season-
    pinned to prev_season, but the trailing leg reaches player_game_stats
    through an UNQUALIFIED join on nba_games, so it spans the entire corpus.
    D101's schedule backfill made 2021-22/2020-21 rows visible and the roster
    grew by 6 players whose most recent >=12-minute regular-season game is
    2022-04-10 or older (Tim Frazier: 2022-01-09) — preseason camp bodies
    projected as rotation players off four-year-old minutes.

    MEASURED (scripts/cg_bridge_reconfirm.py, 2025-26 week-1, 53 games, vs the
    pre-registered gate table data/rw_week1_psroster.csv):

        cap        n moved   max |dcm_ps|
        1 season       30      1.16
        2 seasons       5      0.418      <- D105's declared primary
        3 seasons       0      3.55e-15   <- BIT-EXACT to the registered gate
        4/5/none        5      0.604

    A 3-season cap reproduces the pre-registered construction EXACTLY, because
    `nba_games` held exactly 2022-23..2025-26 when that table was built. The
    constructor default stays None, but the PRODUCTION call site
    (production.py fit_production) is FROZEN at trail_seasons=2 for 2026-27 —
    D105's declared F6 primary — via OCT_BRIDGE_TRAIL (default "2"; set "" for
    the uncapped legacy construction used by the D122 certification).
    """

    def __init__(self, con, season: str, before: dt.date | None = None,
                 trail_seasons: int | None = None):
        from ..config import prev_season
        ref = _ref_date(before)
        floor = None
        if trail_seasons is not None:
            y = int(season[:4]) - int(trail_seasons)
            floor = f"{y}-{(y + 1) % 100:02d}"
        # 001 preseason roster this season, argmax-minutes team assignment
        ps_ros: dict[int, dict[int, float]] = {}
        for t, p, m in con.execute("""
            SELECT s.team_id, s.player_id, sum(s.seconds)/60.0
            FROM player_game_stats s
            JOIN nba_games g ON g.game_id = s.game_id AND g.team_id = s.team_id
            WHERE g.season = ? AND s.game_id LIKE '001%' AND s.seconds > 0
              AND g.game_date < ?
            GROUP BY 1, 2""", [season, ref]).fetchall():
            ps_ros.setdefault(int(p), {})[int(t)] = float(m)
        assign = {p: max(d, key=d.get) for p, d in ps_ros.items()}
        # UNION: prior-season primary 002 team for players with no 001 minutes
        for p, t in con.execute("""
            SELECT s.player_id, arg_max(s.team_id, g.game_date)
            FROM player_game_stats s
            JOIN nba_games g ON g.game_id = s.game_id AND g.team_id = s.team_id
            WHERE g.season = ? AND s.game_id LIKE '002%' AND s.seconds > 0
            GROUP BY 1""", [prev_season(season)]).fetchall():
            if int(p) not in assign:
                assign[int(p)] = int(t)
        # trailing minutes: last 10 002 games with >=12 min, strictly < ref
        # (season-capped iff trail_seasons was given — see the class docstring)
        trail = dict(con.execute(f"""
            WITH pg AS (
              SELECT s.player_id, s.seconds/60.0 m,
                     row_number() OVER (PARTITION BY s.player_id
                                        ORDER BY g.game_date DESC) rn
              FROM player_game_stats s
              JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g
                USING (game_id)
              WHERE s.game_id LIKE '002%' AND s.seconds >= 720
                AND g.game_date < ?
                {"AND g.season >= '" + floor + "'" if floor else ""})
            SELECT player_id, avg(m) FROM pg WHERE rn <= 10 GROUP BY 1""",
            [ref]).fetchall())
        darko = CompositionModel._darko_asof(con, before)
        self.season = season
        self.asof = before
        self.trail_seasons = trail_seasons
        self.trail_floor = floor
        self.contrib: dict[int, tuple[int, float]] = {
            int(p): (int(t), float(darko.get(p, 0.0)) * float(trail[p]) / 48.0)
            for p, t in assign.items() if p in trail}

    def strength(self, team_id: int, out: set | None = None) -> float:
        out = out or set()
        return sum(c for p, (t, c) in self.contrib.items()
                   if t == team_id and p not in out)

    def margin(self, home_id: int, away_id: int, out_home: set | None = None,
               out_away: set | None = None) -> float:
        """cm_ps home-away strength diff; NO home edge (sched layer owns it),
        mirroring the comp leg it replaces (comp.margin(..., home_edge=0))."""
        return self.strength(home_id, out_home) - self.strength(away_id, out_away)
