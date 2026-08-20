"""Team margin from v3 states.

M1: margin from the team DLM — (mu, sigma) with mu the neutral net-rating
diff (schedule layer added by the caller, exactly like FourFactors
.margin_neutral) and sigma the filter-propagated uncertainty
(sqrt(w'Pw + sigma_game^2), V3_SPEC 2.2 — consumed at M2).

M3 replaces the team states with the minutes-weighted composition of player
states (team_margin signature below, per V3_SPEC 2.6).
"""
from __future__ import annotations

import math


def dlm_margin(dlm, home_key, away_key, sigma_game: float = 11.0) -> tuple[float, float]:
    """(mu_neutral, sigma) from a TeamDLM."""
    mu = dlm.margin_neutral(home_key, away_key)
    var = dlm.margin_neutral_var(home_key, away_key)
    return mu, math.sqrt(max(var, 0.0) + sigma_game ** 2)


def team_margin(bank, home_id: int, away_id: int, out_home: set, out_away: set,
                sched: float, date) -> tuple[float, float]:
    """M3 — composition of player states (minutes-state renorm over the active
    roster, positional tilt for OUT players per D39)."""
    raise NotImplementedError("M3 — player-state composition margin")
