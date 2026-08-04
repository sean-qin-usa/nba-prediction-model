"""Tonight's slate — the SHARED model-side assembly for the live entrypoints.

Extracted verbatim from scripts/predict_today.py (2026-07-31) so that the
daily human-readable printout and the paper-trade bet engine price tonight's
games through ONE code path: tank priming -> fit_production (which carries
the D73 tank term and the D84-A October bridge) -> injury-feed OUT sets +
departed filter -> b2b flags. Any live-parity fix lands here once.
"""
from __future__ import annotations

import datetime as dt


def todays_games():
    """[(game_id, home_team_id, away_team_id)] from the NBA scoreboard;
    empty list offseason / on scoreboard failure (offseason-safe)."""
    from nba_api.stats.endpoints import scoreboardv2
    today = dt.date.today().strftime("%Y-%m-%d")
    try:
        sb = scoreboardv2.ScoreboardV2(game_date=today, timeout=30)
        hdr = sb.game_header.get_data_frame()
    except Exception as e:  # noqa: BLE001
        print(f"scoreboard unavailable: {e}")
        return []
    return [(r.GAME_ID, r.HOME_TEAM_ID, r.VISITOR_TEAM_ID)
            for r in hdr.itertuples()]


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
    yday = {r[0] for r in con.execute(
        "SELECT DISTINCT team_id FROM nba_games WHERE game_date = ?",
        [today - dt.timedelta(days=1)]).fetchall()}
    routs = {}
    for gd, team, pid in con.execute("""
        SELECT i.game_date, i.team, p.player_id FROM injury_reports_pit i
        JOIN (SELECT player_id, lower(first_name||' '||last_name) fn FROM nba_players) p
          ON p.fn = lower(trim(split_part(i.player,',',2))||' '||trim(split_part(i.player,',',1)))
        WHERE i.status = 'Out' AND i.game_date = ?""", [today]).fetchall():
        routs.setdefault(team, set()).add(int(pid))
    from nba_api.stats.static import teams as _t
    name2id = {t["full_name"]: t["id"] for t in _t.get_teams()}
    outs_by_tid = {name2id[k]: v for k, v in routs.items() if k in name2id}
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
