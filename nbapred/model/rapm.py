"""Player RAPM — Regularized Adjusted Plus-Minus (handoff II.2 stint likelihood).

The player-level analog of team_ratings: solve each player's OFFENSIVE and
DEFENSIVE impact (points per 100 possessions) controlling for the other nine
players on the floor, ridge-regularized. This is the identification-critical
term — defense is nearly invisible in box events, and RAPM pins it from lineup
stint margins. Its output is the defensive skill the possession engine needs
(answering "who guards" at the player level) and a strong two-way rating.

Data: lineup_stints. Each stint -> TWO observations:
  home on offense: home_pts/100poss ~ mu + Σ O(home) - Σ D(away)
  away on offense: away_pts/100poss ~ mu + Σ O(away) - Σ D(home)
possessions ≈ seconds / 14.4 (one team-possession every ~14.4s). Weighted ridge
(weight = possessions) so long stints count more. Sum-to-zero per rating type.

Sparse normal equations: X is (2*stints) x (1 + 2*P) with ~10 nonzeros/row.
Solve (XᵀWX + λR) β = XᵀWy. P~500 -> a 1001x1001 dense solve, trivial.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

SEC_PER_POSS = 14.4


class RAPM:
    """ridge: shrinkage strength. off_prior/def_prior: {player_id: prior impact
    per 100} to shrink TOWARD (prior-informed RAPM, a la RPM/DARKO). Defaults to
    shrinking toward zero (classic RAPM). Prior-informed is far more stable in a
    single season — defense is underidentified, so a box/DARKO prior carries the
    load where stint data is thin."""

    def __init__(self, ridge: float = 2000.0, off_prior: dict | None = None,
                 def_prior: dict | None = None):
        self.ridge = ridge
        self.off_prior = off_prior or {}
        self.def_prior = def_prior or {}
        self.mu = 112.0
        self.off: dict[int, float] = {}
        self.deff: dict[int, float] = {}

    def fit(self, stints: list[dict]):
        """stints: [{home:[5 ids], away:[5 ids], seconds, home_pts, away_pts}]."""
        players = sorted({p for s in stints for p in s["home"] + s["away"]})
        if len(players) < 10 or len(stints) < 50:
            return self
        idx = {p: i for i, p in enumerate(players)}
        P = len(players)
        ncol = 1 + 2 * P  # [mu, off_0..off_{P-1}, def_0..def_{P-1}]

        rows, cols, vals, y, w = [], [], [], [], []
        r = 0

        def add(off5, def5, pts, poss):
            nonlocal r
            if poss < 1:
                return
            rate = 100.0 * pts / poss
            rows.append(r); cols.append(0); vals.append(1.0)          # mu
            for p in off5:
                rows.append(r); cols.append(1 + idx[p]); vals.append(1.0)
            for p in def5:
                rows.append(r); cols.append(1 + P + idx[p]); vals.append(-1.0)
            y.append(rate); w.append(poss); r += 1

        for s in stints:
            poss = s["seconds"] / SEC_PER_POSS
            add(s["home"], s["away"], s["home_pts"], poss)
            add(s["away"], s["home"], s["away_pts"], poss)

        X = sparse.csr_matrix((vals, (rows, cols)), shape=(r, ncol))
        W = sparse.diags(np.asarray(w))
        y = np.asarray(y)

        # prior vector aligned to columns. Coding: rate = mu + Σoff - Σdef, so a
        # POSITIVE def coefficient already means good defense (subtracts from the
        # opponent's rate). Priors enter as the shrinkage center.
        prior = np.zeros(ncol)
        for p, i in idx.items():
            prior[1 + i] = self.off_prior.get(p, 0.0)
            prior[1 + P + i] = self.def_prior.get(p, 0.0)

        # ridge toward `prior`: minimize ||W^.5 (y - Xβ)||^2 + λ||β - prior||^2
        # => solve for δ = β - prior on residual y - X·prior, then β = prior + δ.
        y_res = y - X @ prior
        A = (X.T @ W @ X).toarray()
        reg = np.full(ncol, self.ridge); reg[0] = 0.0
        A[np.diag_indices_from(A)] += reg
        b = X.T @ (W @ y_res)
        delta = np.linalg.solve(A, b)
        beta = prior + delta

        self.mu = float(beta[0])
        off = beta[1:1 + P]; deff = beta[1 + P:]
        off -= off.mean(); deff -= deff.mean()
        # CORRECT SIGN: positive def coefficient = good defense already. (Earlier
        # code negated it, which inverted the defensive ratings.)
        self.off = {players[i]: float(off[i]) for i in range(P)}
        self.deff = {players[i]: float(deff[i]) for i in range(P)}
        return self

    def net(self, player_id: int) -> float:
        return self.off.get(player_id, 0.0) + self.deff.get(player_id, 0.0)

    def table(self, top: int = 20):
        rows = [(p, self.off[p], self.deff[p], self.net(p)) for p in self.off]
        return sorted(rows, key=lambda x: -x[3])[:top]


def load_stints(con, before=None) -> list[dict]:
    date_clause = ""
    params = []
    if before:
        date_clause = "AND s.game_id IN (SELECT DISTINCT game_id FROM nba_games WHERE game_date < ?)"
        params = [before]
    df = con.execute(f"""
        SELECT home_lineup, away_lineup, seconds, home_pts, away_pts
        FROM lineup_stints s
        WHERE seconds > 0 {date_clause}
    """, params).fetchdf()
    out = []
    for r in df.itertuples():
        try:
            home = [int(x) for x in r.home_lineup.split(",")]
            away = [int(x) for x in r.away_lineup.split(",")]
        except (ValueError, AttributeError):
            continue
        if len(home) == 5 and len(away) == 5:
            out.append(dict(home=home, away=away, seconds=r.seconds,
                            home_pts=r.home_pts, away_pts=r.away_pts))
    return out
