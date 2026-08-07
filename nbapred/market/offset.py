"""MARKET-OFFSET LAYER — the shipped correction to the opening line (D224).

    m_offset = m_open + f(m_blind - m_open, rest_diff, |m_open|)

`f` is a ridge shrunk hard toward zero. Its coefficient on the fundamental
disagreement is ~0.35, so the market-blind model's stated edge is spent at
roughly a third of face value.

WHY THIS EXISTS.  The market-blind model does not beat the opening line it would
transact at (D193: capture -0.035 on honest inputs; D199: -0.104 before the
availability leak was closed).  Its disagreement nevertheless carries
information, and shrunk hard against a market anchor it improves on the opener
in six of seven scored seasons.

GATE (D224): season-clustered mean delta -0.006378 nats, 95% CI
[-0.010621, -0.002134] excluding zero, t = -3.68, better in 6/7 seasons,
calibration veto passed.  Incumbent was the market-blind margin carrying D202
soft availability.

WHAT THIS DOES NOT DO.  It does not retire the market-blind model.  That model
is this layer's dominant input and its degraded-mode fallback: `apply()` returns
the blind margin unchanged when no opening price is available, so a dead odds
feed costs the correction, not the prediction.

MARKET-BLINDNESS.  The blind model still never sees a price.  This layer is the
only place a market number enters the forecast, and it enters after the blind
margin is formed.  That boundary is what makes the comparison in D193/D224 mean
anything, so it is enforced here by construction: `apply()` takes the blind
margin as an argument and cannot reach back into the model that produced it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_COEF_PATH = _ROOT / "data" / "offset_coefs.json"

#: set OFFSET_LAYER=0 to fall back to the raw market-blind margin
ENABLED = os.environ.get("OFFSET_LAYER", "1") == "1"

_CACHE: dict | None = None


def coefficients() -> dict:
    """The frozen production fit (scripts/d225_fit_offset_prod.py)."""
    global _CACHE
    if _CACHE is None:
        if not _COEF_PATH.exists():
            raise FileNotFoundError(
                f"{_COEF_PATH} missing; run scripts/d225_fit_offset_prod.py")
        _CACHE = json.load(open(_COEF_PATH))
    return _CACHE


def apply(m_blind: float, open_margin: float | None,
          rest_diff: float = 0.0) -> float:
    """Correct the OPENING line using the market-blind margin.

    m_blind      the market-blind model's home margin
    open_margin  the opening line as a home margin, or None if unavailable
    rest_diff    home rest days minus away rest days, both capped at 7

    Returns the corrected margin, or `m_blind` unchanged when the layer is
    disabled or no opening price exists — the degraded mode is the incumbent,
    never a silent zero.
    """
    if not ENABLED or open_margin is None:
        return float(m_blind)
    c = coefficients()
    b = dict(zip(c["features"], c["coefs"]))
    edge = float(m_blind) - float(open_margin)
    return float(open_margin
                 + b["edge"] * edge
                 + b["rest_diff"] * float(rest_diff)
                 + b["abs_open"] * abs(float(open_margin)))


def describe() -> str:
    c = coefficients()
    b = ", ".join(f"{k}={v:+.4f}" for k, v in zip(c["features"], c["coefs"]))
    return (f"offset layer {'ON' if ENABLED else 'OFF'} — {b} "
            f"(fitted through {c['fitted_through']}, n={c['n']}, {c['gate']})")
