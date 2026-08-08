"""ABSENCE-RESPONSE TERM — the second-order cost of missing players (D232).

    margin += beta * (E[absences_home] - E[absences_away])          beta < 0

WHY THIS EXISTS.  The composition leg already removes each absent player's own
`talent x minutes`.  D231b measured what is left over: regressing the production
model's margin residual on the expected-absence differential gives a slope of
-0.5367 with a season-clustered 95% CI of [-0.7348, -0.3385], the SAME SIGN IN
7/7 SEASONS, t = -6.63.  **An absence costs about half a point of margin MORE
than the departing player's own contribution.**

That is a second-order effect the composition leg cannot express: it carries
every remaining player at his unchanged trailing rate, so it never charges for
the fact that the minutes get absorbed by worse players in worse lineups.  D133
arm C and D144 both predicted exactly this when they found promoted replacements
underperforming the production implied by their bench rates — this term is the
first time that mechanism has been priced rather than only diagnosed.

GATE (D232, prereg sha256 fe77ff1e...): season-clustered mean delta -0.002174
nats, 95% CI [-0.003299, -0.001048] excluding zero, t = -4.96, better in 6/6
scored seasons, calibration veto passed, MDE80 0.00107.  Under a control arm
that also regresses on the model's own margin — the test of whether this is
really availability rather than team quality in disguise — beta does not shrink,
it GROWS (-0.645 -> -0.704), and the arm still ships.

NO NEW INFORMATION ENTERS.  `out` is the same as-of-open P(out) structure the
composition leg already consumes (D201), so this is a recalibration of how the
model aggregates availability, not a new feature and not a new leakage surface.

THE COEFFICIENT TRENDS.  Walk-forward fits run -0.28, -0.62, -0.71, -0.71,
-0.81, -0.74 across the scored seasons.  The shipped constant is the full-frame
fit (-0.8284), which is what the walk-forward rule "fit on all prior seasons"
prescribes for the next one; the per-fold values are kept in the artifact so the
trend stays visible rather than being averaged away silently.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_COEF_PATH = _ROOT / "data" / "absence_coefs.json"

#: set ABSENCE_TERM=0 to fall back to the pre-D232 margin
ENABLED = os.environ.get("ABSENCE_TERM", "1") == "1"

_CACHE: dict | None = None


def coefficients() -> dict:
    global _CACHE
    if _CACHE is None:
        if not _COEF_PATH.exists():
            raise FileNotFoundError(f"{_COEF_PATH} missing")
        _CACHE = json.load(open(_COEF_PATH))
    return _CACHE


def expected_out(o) -> float:
    """Expected absences from either `out` form.

    A DICT is the soft form {player_id: P(out)} and its expectation is the SUM of
    the probabilities — NOT len(), which counts everyone carrying any doubt at
    all and runs about 1.70 per team against a true expectation of 0.96.  That
    exact confusion sat in `prod_by_season.py`'s comment and produced a weaker
    first version of this gate (t -2.82 against the pre-registered -4.96).
    """
    if not o:
        return 0.0
    return float(sum(o.values())) if isinstance(o, dict) else float(len(o))


def term(out_home=None, out_away=None) -> float:
    """The margin adjustment. Returns 0.0 when disabled or nothing is known."""
    if not ENABLED:
        return 0.0
    diff = expected_out(out_home) - expected_out(out_away)
    if diff == 0.0:
        return 0.0
    return float(coefficients()["coefs"][0] * diff)


def describe() -> str:
    c = coefficients()
    return (f"absence term {'ON' if ENABLED else 'OFF'} — "
            f"beta={c['coefs'][0]:+.4f} (n={c['n']}, {c['gate']})")
