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
import os

HOME_EDGE = 3.0
ROSTER_DAYS = 12


#: D242 joint minute-conserving allocation. UNSET = current behaviour,
#: byte-identical. Values: N1 proportional, N2 capped, N3 role-tiered,
#: N4 half-conserved.
_ALLOC = os.environ.get("COMP_ALLOC", "")



_ALPHA_CACHE = None


def _alpha_for(asof):
    """Walk-forward tier share for the season containing `asof` (D245)."""
    global _ALPHA_CACHE
    if _ALPHA_CACHE is None:
        import json as _j
        from pathlib import Path as _P
        f = _P(__file__).resolve().parent.parent.parent / "data" / "d245_alpha.json"
        _ALPHA_CACHE = _j.load(open(f)) if f.exists() else {}
    if asof is None or not _ALPHA_CACHE:
        return None
    y = asof.year if asof.month >= 10 else asof.year - 1
    return _ALPHA_CACHE.get(f"{y}-{(y + 1) % 100:02d}")


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

    def _alloc(self, u: dict, mode: str) -> dict:
        """D242: redistribute expected minutes so the team conserves 240.

        NBA teams play ~240 regulation minutes regardless of who is absent, so a
        routine DNP REALLOCATES minutes rather than destroying them. The
        incumbent sum-of-independent-discounts lets them vanish; every mode here
        conserves the total. Four constructions are implemented and all four are
        reported in D242 -- reporting only the winner would make a four-way
        search look like one test.
        """
        tot = sum(u.values())
        if tot <= 0:
            return dict(u)
        if mode == "N1":                              # proportional
            k = 240.0 / tot
            return {i: v * k for i, v in u.items()}
        if mode == "N2":                              # capped water-fill
            m = {i: v * 240.0 / tot for i, v in u.items()}
            for _ in range(20):
                over = {i: v for i, v in m.items() if v > 40.0}
                if not over:
                    break
                excess = sum(v - 40.0 for v in over.values())
                free = {i: v for i, v in m.items() if v < 40.0}
                fsum = sum(free.values())
                if fsum <= 0:
                    break
                for i in over:
                    m[i] = 40.0
                for i, v in free.items():
                    m[i] = v + excess * v / fsum
            return m
        if mode == "N3":                              # starter/bench tiers
            order = sorted(u, key=lambda i: (-u[i], i))
            top, rest = order[:5], order[5:]
            out_ = {}
            ts = sum(u[i] for i in top) or 1.0
            rs = sum(u[i] for i in rest) or 1.0
            for i in top:
                out_[i] = 160.0 * u[i] / ts
            for i in rest:
                out_[i] = 80.0 * u[i] / rs
            return out_
        if mode == "ALPHA":                           # D245 walk-forward alpha
            # The tier share is READ from a walk-forward artifact fitted on
            # realised minute shares of PRIOR seasons only -- never tuned
            # against margin or log loss. Falls back to raw weights if the
            # season is absent, so a missing key degrades to CONTROL rather
            # than silently using some other split.
            a = _alpha_for(self.asof)
            if a is None:
                return dict(u)
            order = sorted(u, key=lambda i: (-u[i], i))
            top, rest = order[:5], order[5:]
            if not rest:
                return {i: v * 240.0 / tot for i, v in u.items()}
            ts = sum(u[i] for i in top) or 1.0
            rs = sum(u[i] for i in rest) or 1.0
            out_ = {}
            for i in top:
                out_[i] = 240.0 * a * u[i] / ts
            for i in rest:
                out_[i] = 240.0 * (1 - a) * u[i] / rs
            return out_
        if mode == "N4":                              # half-conserved
            k = 240.0 / tot
            return {i: 0.5 * v * k + 0.5 * v for i, v in u.items()}
        return dict(u)

    def strength(self, team_id: int, out=None,
                 game_date: dt.date | None = None) -> float:
        """Team strength = sum over rostered players of talent x minutes / 48,
        weighted by availability.

        `out` accepts EITHER form, and the set form is byte-identical to the
        pre-D201 behaviour:
          * a SET of player_ids            -> hard availability, weight 0 or 1
          * a DICT {player_id: p_out}      -> SOFT availability, weight 1-p_out

        The soft form exists because at the OPEN we do not know tonight's
        out-set (D199: 18.1% of it is published after the line). A player last
        listed Questionable is out ~28.9% of the time; the hard rule scores him
        0.0 and the hard rule is wrong in both directions (D200).
        """
        out = out or set()
        soft = isinstance(out, dict)
        ref = game_date or self.asof or dt.date.today()
        s = 0.0
        u, tal = {}, {}
        # SORTED, BECAUSE FLOAT ADDITION IS NOT ASSOCIATIVE (D230).  `players` is
        # populated from a DuckDB scan whose outer row order is not pinned (the
        # ORDER BY clauses sit inside window functions), and DuckDB parallelises
        # it, so insertion order varied BETWEEN RUNS.  Measured: the same code,
        # same settings, two runs, differed by max|dp| 1.6e-14 on 5,616 of 6,148
        # games.  That is harmless in size but it put a NOISE FLOOR under every
        # control-hash check in the register — "max|dp| = 0" was unreachable, so
        # a genuinely tiny defect had somewhere to hide.  Summing in player_id
        # order makes the pipeline reproducible and those controls exact.
        for pid in sorted(self.players):
            p = self.players[pid]
            if p["team_id"] != team_id:
                continue
            if soft:
                w = 1.0 - float(out.get(pid, 0.0))
                if w <= 0.0:
                    continue
            elif pid in out:
                continue
            else:
                w = 1.0
            if (ref - p["last_played"]).days > ROSTER_DAYS:
                continue   # not currently in rotation (long-term out / departed)
            if _ALLOC:
                u[pid] = w * p["trail_min"]
                tal[pid] = p["talent"]
            else:
                s += w * p["talent"] * p["trail_min"] / 48.0
        if _ALLOC:
            m = self._alloc(u, _ALLOC)
            for pid in sorted(m):
                s += m[pid] * tal[pid] / 48.0
        return s

    def margin(self, home_id: int, away_id: int, out_home=None,
               out_away=None, game_date: dt.date | None = None,
               home_edge: float = HOME_EDGE) -> float:
        return (self.strength(home_id, out_home, game_date)
                - self.strength(away_id, out_away, game_date) + home_edge)
