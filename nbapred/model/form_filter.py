"""Kalman / state-space form filter — the principled 'form = AR(1) toward player
mean' (handoff II.1), replacing the EWMA shortcut. Answers three things EWMA
can't:

  * FORM/TREND: latent skill theta_t reverts toward the player's mean and tracks
    genuine level changes (injury recovery, role change).
  * BAD GAMES: the Kalman gain weights each game by its reliability — a poor game
    in few minutes (high measurement noise) barely moves the estimate; a poor
    game in 36 minutes moves it more, but still tempered by current confidence.
    This IS Bayesian sequential updating (posterior -> next prior).
  * TIME UNCERTAINTY: process variance grows with the DAY GAP since the last
    game, so a player who hasn't played in two weeks has a wider posterior.

State per player: theta (latent per-minute rate), P (variance).
Predict (gap of dt days):  theta <- mean + phi**dt * (theta - mean);  P += Q*dt
Update (game rate y over m minutes, meas. noise R = base/m):
  K = P/(P+R);  theta += K*(y - theta);  P *= (1-K)
"""
from __future__ import annotations

import numpy as np


class FormFilter:
    def __init__(self, prior_mean: float, prior_var: float = 0.02,
                 phi: float = 0.985, Q: float = 2e-4, meas_base: float = 8.0):
        self.mean = prior_mean          # player's long-run mean (EB prior center)
        self.theta = prior_mean
        self.P = prior_var
        self.phi = phi                  # daily reversion toward mean (<1)
        self.Q = Q                      # process noise per day (time uncertainty)
        self.meas_base = meas_base      # measurement noise scale (÷ minutes)

    def predict(self, dt_days: float):
        dt = max(dt_days, 0.0)
        self.theta = self.mean + (self.phi ** dt) * (self.theta - self.mean)
        self.P += self.Q * dt
        return self.theta, self.P

    def update(self, y: float, minutes: float):
        R = self.meas_base / max(minutes, 1.0)     # more minutes -> trust the game more
        K = self.P / (self.P + R)                  # Kalman gain
        self.theta += K * (y - self.theta)
        self.P *= (1 - K)
        return self.theta, self.P


def filter_series(games: list[dict], prior_mean: float, **kw):
    """games: [{date_ordinal, minutes, rate}] chronological. Returns the
    one-step-AHEAD predicted rate for each game (predict BEFORE seeing it) —
    the leakage-safe quantity to score."""
    f = FormFilter(prior_mean, **kw)
    preds = []
    last = None
    for g in games:
        dt = 0.0 if last is None else (g["date_ordinal"] - last)
        theta_pred, _ = f.predict(dt)
        preds.append(theta_pred)                   # forecast for THIS game
        f.update(g["rate"], g["minutes"])          # then absorb it
        last = g["date_ordinal"]
    return preds
