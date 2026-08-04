"""Walk-forward evaluation (handoff III.3: "walk-forward by season only").

The ONLY honest way to test whether an added feature/term helps: fit on the
past, score the strictly-future, never let the test set touch development.
This module provides the season splitter and an Elo baseline good enough to
(a) be one of the mandated baselines and (b) exercise the whole harness before
the simulator exists.
"""
from __future__ import annotations

import math

from .metrics import summary


def season_splits(seasons: list[str], min_train: int = 1):
    """Yield (train_seasons, test_season) walking forward one season at a time."""
    s = sorted(seasons)
    for i in range(min_train, len(s)):
        yield s[:i], s[i]


class Elo:
    """Standard game-level Elo with home edge — a market-blind baseline.
    Not part of the player model; it exists to benchmark against (I.5) and to
    give the harness a working forecaster today."""

    def __init__(self, k=20.0, home_edge=100.0, mean=1500.0, regress=0.25):
        self.k, self.home_edge, self.mean, self.regress = k, home_edge, mean, regress
        self.r: dict[str, float] = {}

    def _get(self, t):
        return self.r.get(t, self.mean)

    def new_season(self):
        for t in self.r:
            self.r[t] = self.mean + (1 - self.regress) * (self.r[t] - self.mean)

    def p_home(self, home, away):
        d = (self._get(home) + self.home_edge) - self._get(away)
        return 1.0 / (1.0 + 10 ** (-d / 400.0))

    def update(self, home, away, home_won: int):
        p = self.p_home(home, away)
        delta = self.k * (home_won - p)
        self.r[home] = self._get(home) + delta
        self.r[away] = self._get(away) - delta


def elo_walkforward(games_by_season: dict[str, list]) -> dict:
    """games_by_season[season] = [(date, home, away, home_won), ...] sorted.
    Returns OOS metrics pooled across all test seasons (train-then-predict:
    every game is predicted BEFORE its result updates the ratings)."""
    elo = Elo()
    y, p = [], []
    seasons = sorted(games_by_season)
    for si, season in enumerate(seasons):
        if si > 0:
            elo.new_season()
        for _, home, away, hw in games_by_season[season]:
            if si > 0:  # first season is burn-in (no ratings yet)
                y.append(hw)
                p.append(elo.p_home(home, away))
            elo.update(home, away, hw)
    return summary(y, p) if y else {"n": 0}
