"""Ablation harness — the operational answer to "did this addition earn its
place, or is it just complexity?" (see docs/COMPLEXITY.md).

A candidate feature/term is kept ONLY if, on walk-forward OOS data, it improves
the scoring rule by more than the noise in that improvement. We estimate the
noise by block bootstrap over test games, so "improvement" is judged against
its own sampling error, not eyeballed.
"""
from __future__ import annotations

import numpy as np

from .metrics import log_loss


def paired_bootstrap_delta(y, p_base, p_new, n_boot=2000, seed_offset=0):
    """Bootstrap the OOS log-loss REDUCTION of p_new vs p_base (positive = new
    is better). Returns (mean_delta, ci_low, ci_high, p_worse_or_equal).

    Deterministic: seeded from a fixed base (the sandbox forbids Math.random-
    style nondeterminism in some contexts; we use an explicit RNG)."""
    y = np.asarray(y, float)
    pb, pn = np.asarray(p_base, float), np.asarray(p_new, float)
    n = len(y)
    rng = np.random.default_rng(12345 + seed_offset)
    base_delta = log_loss(y, pb) - log_loss(y, pn)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[b] = log_loss(y[idx], pb[idx]) - log_loss(y[idx], pn[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_null = float(np.mean(deltas <= 0))  # fraction of resamples where new is NOT better
    return {"delta_logloss": float(base_delta), "ci95": (float(lo), float(hi)),
            "p_not_better": p_null, "keep": bool(lo > 0)}


def ablation_report(y, forecasts: dict) -> dict:
    """forecasts = {name: p_array}; scores each vs the 'base' entry.
    A variant is flagged keep=True only if its 95% CI on the OOS log-loss
    improvement excludes zero."""
    assert "base" in forecasts, "provide a 'base' forecast to compare against"
    out = {"base_logloss": log_loss(y, forecasts["base"])}
    for name, p in forecasts.items():
        if name == "base":
            continue
        out[name] = paired_bootstrap_delta(y, forecasts["base"], p)
    return out
