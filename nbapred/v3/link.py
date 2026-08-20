"""Margin -> P(home win) link.

M1 deliberately keeps the production link (sigmoid(m / 7.2), D48 stands) so
the DLM gate isolates the MARGIN change. The matchup-conditioned Student-T
link (V3_SPEC 2.3) is M2: p = T_nu.cdf(mu / sigma) with (a, b, nu) fit
walk-forward on top of whichever margin wins M1.
"""
from __future__ import annotations

import math

SCALE = 7.2     # production link scale (D48: held-out recal made it worse)


def p_home(mu: float, sigma: float | None = None, nu: float | None = None) -> float:
    if sigma is None or nu is None:
        return 1.0 / (1.0 + math.exp(-mu / SCALE))
    from scipy.stats import t as student_t
    return float(student_t.cdf(mu / sigma, df=nu))
