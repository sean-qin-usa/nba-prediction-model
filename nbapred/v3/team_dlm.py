"""M1 — team-level dynamic linear model (the DLM between the two degenerate
corners: equal-weight season pooling vs pure recency).

State (units: ortg points per 100 poss), one joint Gaussian:
    mu            league scoring environment (random walk, slow)
    home          home-court ortg edge (random walk, very slow)
    off[t]        team offense above league (AR(1) toward 0, daily)
    def[t]        team defense above league (positive = GOOD defense)

Observations (both closed-form scalar Kalman updates on the full covariance —
opponent adjustment IS the joint covariance):
    efficiency pair (from possession data, 2022-23+):
        ortg_home = mu + home + off_h - def_a + eps,   eps ~ N(0, r_eff)
        ortg_away = mu +        off_a - def_h + eps'
    margin (warm-up seasons / playoff rows without possession data):
        margin    = home + (off_h + def_h) - (off_a + def_a) + e, e ~ N(0, r_margin)

Evolution, per day: off/def <- phi * off/def, var += q (closed form over any
gap). Season boundary = EVENT SHOCK, not a refit: off/def <- kappa * off/def
and var += v_bound (the structural replacement for the 0.75-regress cold-start
prior + carry — D16/D62 absorbed, per V3_SPEC M1).

The filter never sees market data (G2) and is strictly causal: a driver must
call predict_to(date) BEFORE reading predictions for that date's games and
only then feed that date's results (see filter_run.py).
"""
from __future__ import annotations

import datetime as dt
import math

import numpy as np

from .hyper import TeamHyper

LOG2PI = math.log(2.0 * math.pi)


class TeamDLM:
    def __init__(self, teams, hyper: TeamHyper | None = None,
                 season_boundaries=None, start: dt.date | None = None,
                 mu0: float = 112.0, home0: float = 2.3,
                 var_mu0: float = 25.0, var_home0: float = 1.0,
                 var_team0: float = 9.0):
        """teams: iterable of stable team keys (abbrevs). season_boundaries:
        sorted dates (first game date of each season); crossing one applies
        the boundary shock BEFORE that date's games."""
        self.h = hyper or TeamHyper()
        self.teams = sorted(teams)
        self.ti = {t: i for i, t in enumerate(self.teams)}
        T = len(self.teams)
        self.n = 2 + 2 * T
        self.x = np.zeros(self.n)
        self.x[0], self.x[1] = mu0, home0
        Pd = np.full(self.n, var_team0)
        Pd[0], Pd[1] = var_mu0, var_home0
        self.P = np.diag(Pd)
        self._team_sl = slice(2, self.n)
        self.boundaries = sorted(season_boundaries or [])
        self._bi = 0                       # next boundary not yet applied
        self.asof = start                  # date the state is valid AT (pre-games)
        if start is not None:
            while self._bi < len(self.boundaries) and self.boundaries[self._bi] <= start:
                self._bi += 1              # boundaries at/before start are baked into priors
        self.loglik = 0.0

    # ---------------------------------------------------------------- indices
    def _io(self, team) -> int:
        return 2 + self.ti[team]

    def _id(self, team) -> int:
        return 2 + len(self.teams) + self.ti[team]

    # -------------------------------------------------------------- evolution
    def _evolve(self, days: int) -> None:
        if days <= 0:
            return
        h = self.h
        d = phi_dt = h.phi ** days
        # exact integrated process noise for AR(1): q * sum_{k<dt} phi^{2k}
        if h.phi < 1.0:
            q_add = h.q * (1.0 - h.phi ** (2 * days)) / (1.0 - h.phi ** 2)
        else:
            q_add = h.q * days
        D = np.ones(self.n)
        D[self._team_sl] = d
        self.x[self._team_sl] *= phi_dt
        self.P *= np.outer(D, D)
        idx = np.arange(self.n)
        diag_add = np.full(self.n, q_add)
        diag_add[0], diag_add[1] = h.q_mu * days, h.q_home * days
        self.P[idx, idx] += diag_add

    def _season_shock(self) -> None:
        """off/def <- kappa*off/def, var += v_bound; mu var += v_mu_bound."""
        h = self.h
        K = np.ones(self.n)
        K[self._team_sl] = h.kappa
        self.x[self._team_sl] *= h.kappa
        self.P *= np.outer(K, K)
        idx = np.arange(self.n)
        add = np.full(self.n, h.v_bound)
        add[0], add[1] = h.v_mu_bound, 0.0
        self.P[idx, idx] += add

    def predict_to(self, date: dt.date) -> None:
        """Advance the state to `date` (exclusive of that date's games),
        applying any season-boundary shocks crossed on the way."""
        if self.asof is None:
            self.asof = date
        while (self._bi < len(self.boundaries)
               and self.boundaries[self._bi] <= date):
            b = self.boundaries[self._bi]
            self._evolve((b - self.asof).days)
            self._season_shock()
            self.asof = b
            self._bi += 1
        self._evolve((date - self.asof).days)
        self.asof = date

    # ------------------------------------------------------------ observation
    def _scalar_update(self, cols, vals, y: float, r: float) -> float:
        """One scalar Kalman update with sparse H; returns loglik contrib."""
        Ph = self.P[:, cols] @ vals
        S = float(vals @ Ph[cols]) + r
        innov = y - float(vals @ self.x[cols])
        K = Ph / S
        self.x += K * innov
        self.P -= np.outer(K, Ph)
        ll = -0.5 * (LOG2PI + math.log(S) + innov * innov / S)
        self.loglik += ll
        return ll

    def update_margin(self, home, away, margin: float) -> float:
        (cols, vals, y, r), = self._rows_for((None, "margin", home, away,
                                              margin, None))
        return self._scalar_update(cols, vals, y, r)

    def update_eff(self, home, away, ortg_home: float, ortg_away: float) -> float:
        ll = 0.0
        for cols, vals, y, r in self._rows_for((None, "eff", home, away,
                                                ortg_home, ortg_away)):
            ll += self._scalar_update(cols, vals, y, r)
        # guard symmetry drift from repeated rank-1 downdates
        self.P = 0.5 * (self.P + self.P.T)
        return ll

    def update(self, ob) -> float:
        """ob: (date, kind, home, away, y1, y2) — kind 'eff' uses (y1, y2) =
        (ortg_home, ortg_away); kind 'margin' uses y1."""
        _, kind, home, away, y1, y2 = ob
        if kind == "eff":
            return self.update_eff(home, away, y1, y2)
        return self.update_margin(home, away, y1)

    def _rows_for(self, ob):
        """Scalar-observation rows (cols, vals, y, r) for one game obs."""
        _, kind, home, away, y1, y2 = ob
        if kind == "eff":
            return [(np.array([0, 1, self._io(home), self._id(away)]),
                     np.array([1.0, 1.0, 1.0, -1.0]), y1, self.h.r_eff),
                    (np.array([0, self._io(away), self._id(home)]),
                     np.array([1.0, 1.0, -1.0]), y2, self.h.r_eff)]
        return [(np.array([1, self._io(home), self._id(home),
                           self._io(away), self._id(away)]),
                 np.array([1.0, 1.0, 1.0, -1.0, -1.0]), y1, self.h.r_margin)]

    def update_batch(self, obs_batch) -> float:
        """Joint Kalman update for one date's games (conditionally independent
        given the state, so the joint update equals the sequential one — chain
        rule keeps the marginal loglik identical; this exists because a
        per-date block is ~10x cheaper in Python than per-obs scalars,
        which is what makes the monthly marginal-likelihood hyperfit cheap)."""
        rows = [r for ob in obs_batch for r in self._rows_for(ob)]
        m = len(rows)
        if m == 0:
            return 0.0
        H = np.zeros((m, self.n))
        Rd = np.empty(m)
        y = np.empty(m)
        for i, (cols, vals, yi, r) in enumerate(rows):
            H[i, cols] = vals
            Rd[i] = r
            y[i] = yi
        PHt = self.P @ H.T
        S = H @ PHt
        S[np.arange(m), np.arange(m)] += Rd
        innov = y - H @ self.x
        L = np.linalg.cholesky(S)
        alpha = np.linalg.solve(S, innov)
        self.x += PHt @ alpha
        self.P -= PHt @ np.linalg.solve(S, PHt.T)
        self.P = 0.5 * (self.P + self.P.T)
        ll = -0.5 * (m * LOG2PI + 2.0 * float(np.log(np.diag(L)).sum())
                     + float(innov @ alpha))
        self.loglik += ll
        return ll

    def run(self, obs, loglik_from: dt.date | None = None) -> float:
        """Filter a chronological observation stream (per-date batches);
        returns the marginal loglik (prediction-error decomposition) — the
        hyperfit objective. loglik_from: only score obs on/after this date
        (trailing-window fit) while still replaying the full stream."""
        total = 0.0
        i, n = 0, len(obs)
        while i < n:
            d = obs[i][0]
            j = i
            while j < n and obs[j][0] == d:
                j += 1
            self.predict_to(d)
            ll = self.update_batch(obs[i:j])
            if loglik_from is None or d >= loglik_from:
                total += ll
            i = j
        return total

    # ------------------------------------------------------------- prediction
    def margin_neutral(self, home, away) -> float:
        """Home-away net-rating diff, NO home edge (the schedule layer owns
        it) — the drop-in for FourFactors.margin_neutral in the blend."""
        return float(self.x[self._io(home)] + self.x[self._id(home)]
                     - self.x[self._io(away)] - self.x[self._id(away)])

    def margin_neutral_var(self, home, away) -> float:
        """Filter variance of the neutral margin (M2's variance-head input)."""
        cols = np.array([self._io(home), self._id(home),
                         self._io(away), self._id(away)])
        vals = np.array([1.0, 1.0, -1.0, -1.0])
        return float(vals @ self.P[np.ix_(cols, cols)] @ vals)

    def net(self, team) -> float:
        return float(self.x[self._io(team)] + self.x[self._id(team)])

    # ------------------------------------------------------------ persistence
    def snapshot_rows(self, team_ids: dict) -> list[tuple]:
        """player_states rows (asof, entity_id, team_id, dim, mean, var) for
        the team dims; team_ids maps team key -> BIGINT id."""
        rows = []
        d = self.asof
        for t in self.teams:
            tid = int(team_ids.get(t, -1))
            for dim, i in (("team_off", self._io(t)), ("team_def", self._id(t))):
                rows.append((d, tid, tid, dim, float(self.x[i]),
                             float(self.P[i, i])))
        rows.append((d, -1, None, "league_mu", float(self.x[0]), float(self.P[0, 0])))
        rows.append((d, -1, None, "home_edge", float(self.x[1]), float(self.P[1, 1])))
        return rows
