"""Scoring metrics for probabilistic forecasts (handoff III.3).

log loss + Brier for binary (win/loss); reliability/ECE for calibration;
CRPS for continuous props. Pure numpy — no sklearn dependency.
"""
from __future__ import annotations

import numpy as np


def _clip(p, eps=1e-15):
    return np.clip(np.asarray(p, float), eps, 1 - eps)


def log_loss(y, p) -> float:
    p = _clip(p)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def reliability_table(y, p, bins=10):
    """Calibration: per-bin (mean predicted, empirical rate, count)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            out.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return out


def ece(y, p, bins=10) -> float:
    """Expected calibration error: sum_k (n_k/N) |acc_k - conf_k|."""
    tbl = reliability_table(y, p, bins)
    n = len(y)
    return float(sum(cnt / n * abs(emp - conf) for conf, emp, cnt in tbl))


def crps_samples(y_true: float, samples) -> float:
    """CRPS of a scalar observation against an empirical sample distribution
    (for prop distributions). E|X-y| - 0.5 E|X-X'|."""
    s = np.sort(np.asarray(samples, float))
    n = len(s)
    term1 = np.mean(np.abs(s - y_true))
    # E|X-X'| via sorted-array closed form
    term2 = (2.0 / n ** 2) * np.sum((2 * np.arange(1, n + 1) - n - 1) * s)
    return float(term1 - 0.5 * term2)


def summary(y, p) -> dict:
    return {"n": int(len(y)), "log_loss": log_loss(y, p),
            "brier": brier(y, p), "ece": ece(y, p),
            "base_rate": float(np.mean(y))}
