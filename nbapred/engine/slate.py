"""Tonight's slate — the SHARED model-side assembly for the live entrypoints.

Extracted verbatim from scripts/predict_today.py (2026-07-31) so that the
daily human-readable printout and the paper-trade bet engine price tonight's
games through ONE code path: tank priming -> fit_production (which carries
the D73 tank term and the D84-A October bridge) -> injury-feed OUT sets +
departed filter -> b2b flags. Any live-parity fix lands here once.
"""
from __future__ import annotations

import datetime as dt

# ---------------------------------------------------------------------------
# D178 FIX 1 — THE GAME-TYPE FILTER, THE LIVE PATH'S LAST UNFILTERED HOLE.
# D172 audited every production consumer for `game_id LIKE '002%'` and found
# EXACTLY ONE that lacked it: this module — the LIVE path.  NBA game ids encode
# the game type in the first three characters:
#     001 preseason (incl. international exhibitions)   004 playoffs
#     002 REGULAR SEASON  <- the only thing this model is fit on and bets
#     003 All-Star weekend                              005 play-in
#     006 NBA Cup final (does not count toward standings)
# The spine carries 315 rows / 97 team codes that are not NBA franchises at all
# (167 All-Star, 148 international preseason).  Every backtest frame the model
# was fit and priced on is 002-only (`prod_by_season.py:111`,
# `ds_rt1_capstone.py`, `production.fit_schedule_layer`), so anything else
# reaching the live path is a live/backtest PARITY BREAK, not just noise.
# The filter is applied at BOTH chokepoints — `todays_games()` (the source) and
# `slate_context()` (so a caller supplying its own game list, or a monkeypatch,
# still cannot smuggle one in).
REGULAR_SEASON_PREFIX = "002"

GAME_TYPE = {"001": "preseason", "002": "regular season", "003": "all-star",
             "004": "playoffs", "005": "play-in", "006": "nba-cup-final"}


def is_regular_season(game_id) -> bool:
    """True iff `game_id` is an NBA REGULAR-SEASON game (prefix 002)."""
    return str(game_id).startswith(REGULAR_SEASON_PREFIX)


def filter_regular_season(games, where: str = "slate"):
    """Drop non-002 games from [(game_id, home_id, away_id)], REPORTING what
    was dropped (D171's teams.py law: report, never silently drop)."""
    keep, drop = [], []
    for g in games:
        (keep if is_regular_season(g[0]) else drop).append(g)
    for g in drop:
        gid = str(g[0])
        print(f"  [{where}] EXCLUDED non-regular-season game {gid} "
              f"({GAME_TYPE.get(gid[:3], 'unknown type')}) — the model is fit "
              f"and priced on 002 only (D178)")
    return keep


def todays_games():
    """[(game_id, home_team_id, away_team_id)] from the NBA scoreboard,
    REGULAR SEASON ONLY (D178 — the scoreboard returns preseason in early
    October and All-Star games in February);
    empty list offseason / on scoreboard failure (offseason-safe)."""
    from nba_api.stats.endpoints import scoreboardv2
    today = dt.date.today().strftime("%Y-%m-%d")
    try:
        sb = scoreboardv2.ScoreboardV2(game_date=today, timeout=30)
        hdr = sb.game_header.get_data_frame()
    except Exception as e:  # noqa: BLE001
        print(f"scoreboard unavailable: {e}")
        return []
    return filter_regular_season(
        [(r.GAME_ID, r.HOME_TEAM_ID, r.VISITOR_TEAM_ID)
         for r in hdr.itertuples()], where="scoreboard")


def b2b_teams(con, day: dt.date) -> set:
    """{team_id} that played a REGULAR-SEASON game on `day` (i.e. are on a
    back-to-back for `day + 1`).  Extracted from `slate_context` by D178 so the
    game-type filter is directly testable: `production.fit_schedule_layer` and
    every backtest builder compute b2b off 002 rows only, so this must too."""
    return {r[0] for r in con.execute(
        "SELECT DISTINCT team_id FROM nba_games "
        "WHERE game_date = ? AND game_id LIKE '002%'", [day]).fetchall()}


def slate_context(con, season: str, games, today: dt.date | None = None):
    """Fit the production model for tonight and assemble per-game inputs.

    Returns dict with:
      model      fit_production Predictor (tank primed for tonight's slate)
      comp_view  CompositionModel as-of today (props / departed filter)
      outs       {game_id: {team_id: OUT set}} (injury feed 'Out' + departed)
      b2b        {team_id playing yesterday}
      gp         {team_id: completed 002 games entering today} (window flags)
    """
    today = today or dt.date.today()
    # D178 FIX 1: second chokepoint — a caller that assembled its own game list
    # (or a test/monkeypatch of todays_games) cannot smuggle a preseason or
    # All-Star game into the tank priming, the model, or the bet engine.
    games = filter_regular_season(list(games), where="slate_context")
    # D73 tank term, live parity: prime the tank model with tonight's slate
    # (virtual team-date rows flow through the same composite construction the
    # backtest uses) BEFORE fit_production, which picks up the primed cache.
    from ..model.tanking import get_tank_model
    get_tank_model(con, virtual_games=[
        (season, tid, today) for _, hid, aid in games for tid in (hid, aid)])
    from ..model.production import fit_production
    model = fit_production(con, season, before=today)
    # D67 R1 live-parity fix: b2b flags from yesterday's schedule (PIT-known),
    # OUT sets from the official injury report (5PM PIT feed), departed-player
    # filter (most-recent team != this team -> not on this roster anymore).
    # D178 FIX 1: `game_id LIKE '002%'` — the b2b flag must mean what it meant
    # in the fit.  `production.fit_schedule_layer` (line 131) and every
    # backtest builder (`prod_by_season.py:111`, `ds_rt1_capstone.py`) compute
    # b2b off 002 rows only; without the filter the live path flagged a b2b off
    # a preseason or NBA Cup-final game and applied a coefficient that was
    # never fit on one.  MEASURED across the whole spine: 1 date (2025-12-17,
    # the day after the 2025 Cup final) x 2 teams would have been wrongly
    # flagged — small, but it is a parity break and it is free to close.
    yday = b2b_teams(con, today - dt.timedelta(days=1))
    routs = {}
    for gd, team, pid in con.execute("""
        SELECT i.game_date, i.team, p.player_id FROM injury_reports_pit i
        JOIN (SELECT player_id, lower(first_name||' '||last_name) fn FROM nba_players) p
          ON p.fn = lower(trim(split_part(i.player,',',2))||' '||trim(split_part(i.player,',',1)))
        WHERE i.status = 'Out' AND i.game_date = ?""", [today]).fetchall():
        routs.setdefault(team, set()).add(int(pid))
    # D171: the PDFs say "LA Clippers"; the old `{full_name: id}` lookup
    # dropped them, so the LIVE slate has been predicting Clippers games with
    # an empty injury-report out-set. `team_id_for` resolves the alias.
    from ..teams import team_id_for
    outs_by_tid = {}
    for k, v in routs.items():
        tid = team_id_for(k)
        if tid is not None:
            outs_by_tid[tid] = v
    last_team = dict(con.execute("""
        SELECT player_id, arg_max(s.team_id, g.game_date) FROM player_game_stats s
        JOIN nba_games g ON g.game_id = s.game_id AND g.team_id = s.team_id
        WHERE s.game_id LIKE '002%' GROUP BY player_id""").fetchall())
    from ..model.composition import CompositionModel
    comp_view = CompositionModel(con, before=today)
    outs = {}
    for gid, hid, aid in games:
        g_outs = {}
        for t in (hid, aid):
            # departed filter: comp roster says team t, but the player's most
            # recent game was for another team (traded away, lingering in the
            # 12-day roster window) -> treat as OUT for team t
            departed = {p for p, info in comp_view.players.items()
                        if info["team_id"] == t and last_team.get(p, t) != t}
            g_outs[t] = outs_by_tid.get(t, set()) | departed
        outs[gid] = g_outs
    gp = dict(con.execute("""
        SELECT team_id, count(*) FROM nba_games
        WHERE season = ? AND game_id LIKE '002%' AND wl IS NOT NULL
          AND game_date < ? GROUP BY 1""", [season, today]).fetchall())
    return dict(model=model, comp_view=comp_view, outs=outs, b2b=yday, gp=gp)
