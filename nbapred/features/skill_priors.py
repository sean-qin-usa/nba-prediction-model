"""Empirical-Bayes skill priors from trailing box stats (docs/PRIORS.md).

Turns per-player sufficient statistics into SHRUNK per-dimension rate estimates
— the stats-derived prior centers the MCMC skill model uses (and a leakage-safe
feature in their own right). Shrinkage is data-driven (empirical Bayes, no
tuning knob), honoring prefer-hypothesis-tuning: the shrinkage strength is
estimated from the population, not grid-searched.

Two families:
  * rate-on-attempts (fg3, rim, mid, ft): Beta-Binomial EB. Fit a Beta(a,b)
    prior to the league via method of moments, posterior mean per player is
    (makes + a) / (attempts + a + b). Thin data -> pulled to league mean.
  * rate-on-minutes (ast, tov, oreb, dreb, stl, blk, foul): Gamma-Poisson EB.
    Same idea per-minute.

Leakage: feed it TRAILING rows only (pit.trailing_player_stats). The estimator
itself is time-agnostic; the caller owns the as-of cut.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("skill_priors")

# (skill_name, makes_col, attempts_col)
RATE_ON_ATTEMPTS = [
    ("fg3_pct", "fg3m", "fg3a"),
    ("rim_pct", "rimm", "rima"),
    ("mid_pct", "midm", "mida"),
    ("ft_pct", "ftm", "fta"),
]
# (skill_name, count_col) — exposure is minutes (seconds/60)
RATE_ON_MINUTES = ["ast", "tov", "oreb", "dreb", "stl", "blk", "pf"]


def _beta_mom(makes: np.ndarray, attempts: np.ndarray, min_att: int = 20):
    """Method-of-moments Beta(a,b) prior from players with >= min_att attempts."""
    m = attempts >= min_att
    if m.sum() < 5:
        # too little data to estimate spread; fall back to a weak prior at league mean
        p = makes.sum() / max(attempts.sum(), 1)
        return p * 5.0, (1 - p) * 5.0
    p = makes[m] / attempts[m]
    mean, var = float(np.mean(p)), float(np.var(p))
    # subtract binomial sampling variance to isolate true-talent spread
    n_bar = float(np.mean(attempts[m]))
    samp = mean * (1 - mean) / n_bar
    tau2 = max(var - samp, 1e-6)
    strength = mean * (1 - mean) / tau2 - 1.0
    strength = float(np.clip(strength, 2.0, 2000.0))
    return mean * strength, (1 - mean) * strength


def _gamma_mom(counts: np.ndarray, exposure: np.ndarray, min_exp: float = 100.0):
    """Method-of-moments Gamma(k,theta) prior on per-minute rate."""
    m = exposure >= min_exp
    if m.sum() < 5:
        r = counts.sum() / max(exposure.sum(), 1)
        return r * 20.0, 1.0 / 20.0  # weak prior at league rate (k=r*20, theta=1/20)
    r = counts[m] / exposure[m]
    mean, var = float(np.mean(r)), float(np.var(r))
    # Poisson sampling var of the rate ~ mean / exposure
    samp = mean / float(np.mean(exposure[m]))
    tau2 = max(var - samp, 1e-9)
    k = mean * mean / tau2
    k = float(np.clip(k, 1.0, 5000.0))
    theta = mean / k
    return k, theta


def estimate(df: pd.DataFrame, min_minutes: float = 0.0) -> pd.DataFrame:
    """df = per-player AGGREGATE sufficient stats (already summed over the
    trailing window), with the columns from player_game_stats plus 'seconds'.
    Returns one row per player with shrunk skill estimates + a sample-size col."""
    g = df.groupby("player_id").sum(numeric_only=True).reset_index()
    g["minutes"] = g["seconds"] / 60.0
    g = g[g["minutes"] >= min_minutes].copy()
    if g.empty:
        return g

    out = g[["player_id", "minutes"]].copy()
    for name, mk, at in RATE_ON_ATTEMPTS:
        a, b = _beta_mom(g[mk].to_numpy(float), g[at].to_numpy(float))
        out[name] = (g[mk] + a) / (g[at] + a + b)
        out[f"{name}_n"] = g[at]
    for name in RATE_ON_MINUTES:
        k, theta = _gamma_mom(g[name].to_numpy(float), g["minutes"].to_numpy(float))
        prior_rate = k * theta
        # posterior mean of per-minute rate, then per-36 for readability
        rate = (g[name] + k) / (g["minutes"] + 1.0 / theta)
        out[f"{name}_per36"] = rate * 36.0
    return out


def build_asof(con, before, min_minutes: float = 50.0):
    """Convenience: shrunk skill priors from ALL games strictly before `before`.
    Leakage-safe (trailing). `before` = a date; None = use everything cached."""
    if before is None:
        df = con.execute(
            "SELECT * FROM player_game_stats WHERE game_id LIKE '002%'").fetchdf()
    else:
        df = con.execute("""
            SELECT s.* FROM player_game_stats s
            JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
            WHERE g.game_date < ? AND s.game_id LIKE '002%'
        """, [before]).fetchdf()
    return estimate(df, min_minutes=min_minutes)
