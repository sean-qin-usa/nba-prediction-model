"""Availability-composition team strength — the validated player-level model
(0.5455 vs team ratings 0.5815 on the leak-checked late-season split).

Strength(team) = Σ over AVAILABLE rostered players: DARKO_net × trailing_min/48.
 - roster: players who appeared for the team within `roster_days`
 - trailing_min: avg of last 10 games actually played (>=12 min)
 - availability: production = injury feed marks OUT players (drop them);
   backtest = oracle (who appeared). No feed -> assume full roster.

Margin(home, away) = strength diff + HOME_EDGE. This intentionally REPLACES the
team rating rather than adjusting it (adjustment double-counts — three failed
variants proved it; see journal 2026-07-28).
"""
from __future__ import annotations

import datetime as dt

HOME_EDGE = 3.0
ROSTER_DAYS = 12


class CompositionModel:
    def __init__(self, con, before: dt.date | None = None):
        """Build talent × trailing-minutes table as of `before` (None = now)."""
        date_clause = "AND g.game_date < ?" if before else ""
        params = [before] if before else []
        df = con.execute(f"""
            WITH pg AS (
              SELECT s.player_id, s.team_id, s.seconds/60.0 m, g.game_date,
                     row_number() OVER (PARTITION BY s.player_id ORDER BY g.game_date DESC) rn
              FROM player_game_stats s
              JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
              WHERE s.game_id LIKE '002%' AND s.seconds >= 720 {date_clause}
            )
            SELECT player_id, arg_max(team_id, game_date) team_id, avg(m) trail_min,
                   max(game_date) last_played
            FROM pg WHERE rn <= 10 GROUP BY player_id
        """, params).fetchdf()
        darko = self._darko_asof(con, before)
        self.asof = before
        self.players = {}
        for r in df.itertuples():
            lp = r.last_played.date() if hasattr(r.last_played, "date") else r.last_played
            self.players[int(r.player_id)] = dict(
                team_id=int(r.team_id), trail_min=float(r.trail_min),
                last_played=lp, talent=float(darko.get(r.player_id, 0.0)))

    @staticmethod
    def _darko_asof(con, before: dt.date | None) -> dict:
        """PIT talent: darko_history as-of `before` (D43 — kills the staleness
        artifact in backtests); falls back to the current darko_dpm snapshot
        when history is absent (fresh clone / pre-pull)."""
        try:
            cond = "WHERE date < ?" if before else ""
            params = [before] if before else []
            rows = con.execute(f"""
                SELECT player_id, dpm FROM (
                  SELECT player_id, dpm,
                         row_number() OVER (PARTITION BY player_id
                                            ORDER BY date DESC) rn
                  FROM darko_history {cond}
                ) WHERE rn = 1
            """, params).fetchall()
            if rows:
                return dict(rows)
        except Exception:
            pass  # table missing -> snapshot fallback
        if before is not None:
            # AUDIT FIX: never silently substitute TODAY's snapshot for a past
            # cutoff (latent lookahead if history is missing) — fail loud.
            raise RuntimeError(f"darko_history empty before {before}; "
                               "refusing snapshot fallback for a past cutoff")
        return dict(con.execute(
            "SELECT nba_player_id, o_dpm + d_dpm FROM darko_dpm "
            "WHERE snapshot_date = (SELECT max(snapshot_date) FROM darko_dpm)"
        ).fetchall())

    def strength(self, team_id: int, out: set | None = None,
                 game_date: dt.date | None = None) -> float:
        out = out or set()
        ref = game_date or self.asof or dt.date.today()
        s = 0.0
        for pid, p in self.players.items():
            if p["team_id"] != team_id or pid in out:
                continue
            if (ref - p["last_played"]).days > ROSTER_DAYS:
                continue   # not currently in rotation (long-term out / departed)
            s += p["talent"] * p["trail_min"] / 48.0
        return s

    def margin(self, home_id: int, away_id: int, out_home: set | None = None,
               out_away: set | None = None, game_date: dt.date | None = None,
               home_edge: float = HOME_EDGE) -> float:
        return (self.strength(home_id, out_home, game_date)
                - self.strength(away_id, out_away, game_date) + home_edge)
