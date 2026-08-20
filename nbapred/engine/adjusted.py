"""Inject opponent-adjusted efficiency into the possession engine.

The raw engine failed because its team rates were schedule-unadjusted. Here each
team's shooting is scaled so the engine's expected points-per-possession matches
its OPPONENT-ADJUSTED offensive rating (from team_ratings, which beats Elo). The
engine then reproduces the adjusted-ratings win probabilities WHILE still
emitting the full joint stat distribution (its real value for props).

Scaling: a single logit shift on all zone FG% moves expected PPP roughly
linearly; we calibrate the slope once on league rates and invert to hit each
team's target PPP = adjusted_ortg / 100.
"""
from __future__ import annotations

import numpy as np

from .fast import _team_points
from .possession import LEAGUE


def _expected_ppp(rates: dict, seed: int = 0, n: int = 20000) -> float:
    rng = np.random.default_rng(seed)
    pts, _, _, _ = _team_points(rates, np.array([n]), rng)
    return float(pts[0] / n)


def _shift_zone_fg(rates: dict, delta: float) -> dict:
    out = dict(rates)
    zf = {}
    for z, p in rates["zone_fg"].items():
        lg = np.log(p / (1 - p)) + delta
        zf[z] = float(1 / (1 + np.exp(-lg)))
    out["zone_fg"] = zf
    return out


class EfficiencyCalibrator:
    """One-time slope of PPP vs logit shift, on league rates."""
    def __init__(self, base_rates: dict = LEAGUE):
        self.base = base_rates
        p0 = _expected_ppp(base_rates, seed=1)
        pp = _expected_ppp(_shift_zone_fg(base_rates, 0.20), seed=1)
        self.ppp0 = p0
        self.slope = (pp - p0) / 0.20

    def rates_for_ppp(self, target_ppp: float, base: dict | None = None) -> dict:
        base = base or self.base
        delta = (target_ppp - self.ppp0) / self.slope
        delta = float(np.clip(delta, -0.9, 0.9))
        return _shift_zone_fg(base, delta)


def matchup_engine_rates(tr, home_id: int, away_id: int, cal: EfficiencyCalibrator):
    """(home_rates, away_rates) scaled to each team's opponent-adjusted ortg.
    tr = a fitted TeamRatings; cal = EfficiencyCalibrator."""
    home_ortg = tr.pred_ortg(home_id, away_id, is_home=True)
    away_ortg = tr.pred_ortg(away_id, home_id, is_home=False)
    hr = cal.rates_for_ppp(home_ortg / 100.0)
    ar = cal.rates_for_ppp(away_ortg / 100.0)
    return hr, ar
