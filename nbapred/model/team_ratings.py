"""Opponent-adjusted team efficiency ratings (the fix the backtest diagnosed).

The v0 engine failed (log loss 0.693 = coin flip, vs Elo 0.618) because it
composed team rates from RAW, opponent-UNADJUSTED box stats — it couldn't rank
teams. This solves each team's offensive and defensive rating controlling for
opponent strength + home court, ridge-regularized (SRS/adjusted-efficiency
style). It is the TEAM-level analog of the player RAPM/stint term, and its
output is what the engine's inputs should be scaled by.

Model, one row per team-game (points scored s, possessions poss):
    ortg_game = 100*s/poss
    ortg_game ~ mu + off_i - def_j + home*is_home
ridge penalty on {off, def} (sum-to-zero) shrinks thin/early-season teams toward
league average — the regularization is the anti-overfit defense (one lambda,
chosen by walk-forward, not a grid on the test set).

Leakage-safe: `fit_through(date)` uses only games strictly before `date`.
"""
from __future__ import annotations

import numpy as np


class TeamRatings:
    def __init__(self, ridge: float = 25.0, home: float | None = None,
                 team_home_ridge: float | None = 200.0):
        """team_home_ridge: per-team home-advantage deviations (Denver altitude
        etc.), heavily shrunk toward the league edge. Gate-passed 2026-07-28
        (+0.0020, CI +0.0004..+0.0036). None disables."""
        self.ridge = ridge
        self.home_fixed = home
        self.team_home_ridge = team_home_ridge
        self.teams: list[int] = []
        self.off: dict[int, float] = {}
        self.deff: dict[int, float] = {}
        self.home_dev: dict[int, float] = {}
        self.mu = 112.0
        self.home = 2.0

    def fit(self, rows: list[tuple], weights=None):
        """rows: (off_team, def_team, is_home, ortg) — one per team-game.
        weights: optional per-row weight (e.g. recency) — recent games count
        more, so ratings track current form."""
        if len(rows) < 30:
            return self
        teams = sorted({t for r in rows for t in (r[0], r[1])})
        idx = {t: i for i, t in enumerate(teams)}
        n, T = len(rows), len(teams)
        w = np.ones(n) if weights is None else np.asarray(weights, float)
        th = self.team_home_ridge is not None
        ncol = 2 + 2 * T + (T if th else 0)
        X = np.zeros((n, ncol))
        y = np.zeros(n)
        for r, (ot, dt, is_home, ortg) in enumerate(rows):
            X[r, 0] = 1.0
            X[r, 1] = 1.0 if is_home else 0.0
            X[r, 2 + idx[ot]] = 1.0
            X[r, 2 + T + idx[dt]] = -1.0
            if th and is_home:
                X[r, 2 + 2 * T + idx[ot]] = 1.0   # offense-at-home team's deviation
            y[r] = ortg
        Xw = X * w[:, None]
        P = np.zeros(ncol)
        P[2:2 + 2 * T] = self.ridge
        if th:
            P[2 + 2 * T:] = self.team_home_ridge
        A = X.T @ Xw + np.diag(P)
        b = Xw.T @ y
        beta = np.linalg.solve(A, b)
        if th:
            hd = beta[2 + 2 * T:]
            self.home_dev = {t: float(hd[idx[t]]) for t in teams}
        self.teams = teams
        self.mu = float(beta[0])
        self.home = float(self.home_fixed if self.home_fixed is not None else beta[1])
        off = beta[2:2 + T]
        deff = beta[2 + T:]
        off -= off.mean(); deff -= deff.mean()   # sum-to-zero identification
        self.off = {t: float(off[idx[t]]) for t in teams}
        self.deff = {t: float(deff[idx[t]]) for t in teams}
        return self

    def pred_ortg(self, off_team: int, def_team: int, is_home: bool) -> float:
        return (self.mu + self.off.get(off_team, 0.0) - self.deff.get(def_team, 0.0)
                + ((self.home + self.home_dev.get(off_team, 0.0)) if is_home else 0.0))

    def pred_margin(self, home: int, away: int) -> float:
        """Predicted home minus away efficiency margin (per 100 poss ~ per game)."""
        return self.pred_ortg(home, away, True) - self.pred_ortg(away, home, False)


def possessions(fga, fta, oreb, tov):
    return fga + 0.44 * fta - oreb + tov


def game_rows(con, before=None, season=None):
    """Build (off_team, def_team, is_home, ortg) rows from player_game_stats,
    strictly before `before` if given. Home parsed from nba_games.matchup.
    season=None -> the current season by calendar (dynamic; the old '2025-26'
    literal default would have gone stale in October 2026)."""
    if season is None:
        from ..config import current_season
        season = current_season()
    date_clause = "AND g.game_date < ?" if before else ""
    params = [season] + ([before] if before else [])
    df = con.execute(f"""
        SELECT s.game_id, s.team_id,
               sum(s.pts) pts, sum(s.fga) fga, sum(s.fta) fta,
               sum(s.oreb) oreb, sum(s.tov) tov,
               any_value(g.matchup) matchup, any_value(g.team_abbrev) abbr,
               any_value(g.game_date) gdate
        FROM player_game_stats s
        JOIN nba_games g ON g.game_id = s.game_id AND g.team_id = s.team_id
        WHERE g.season = ? AND s.game_id LIKE '002%' {date_clause}
        GROUP BY s.game_id, s.team_id
    """, params).fetchdf()
    rows, by_game = [], {}
    for r in df.itertuples():
        by_game.setdefault(r.game_id, []).append(r)
    for gid, recs in by_game.items():
        if len(recs) != 2:
            continue
        m = recs[0].matchup
        host = m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        for r in recs:
            opp = next(x for x in recs if x.team_id != r.team_id)
            poss = possessions(r.fga, r.fta, r.oreb, r.tov)
            if poss < 50:
                continue
            rows.append((int(r.team_id), int(opp.team_id), r.abbr == host,
                         100.0 * r.pts / poss))
    return rows
