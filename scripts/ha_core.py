"""HA-CORE — shared estimation helpers for the home-advantage investigation.

The identifying regression (one per season, or pooled with season-team FE):

    margin_g = sum_t s_t * (1{home=t} - 1{away=t})        # team strength
             + sum_t a_t * 1{home=t}                      # team TOTAL home adv
             + C_g . beta                                 # schedule controls
             + e_g

The H block spans the intercept, so no separate intercept is fitted and a_t is
the team's own home advantage in points. The Z block has an exact null
direction (adding c to every s_t changes nothing, since sum_t z_t = 0), so the
minimum-norm lstsq solution automatically satisfies sum_t s_t = 0 -- i.e.
strength is measured against the league mean and a_t is fully opponent- and
own-quality-controlled.

    league HFA (this season) = mean_t(a_t)
    team home DEVIATION      = d_t = a_t - mean_t(a_t),  sum_t d_t = 0

DESCRIPTIVE / FULL-SAMPLE. Nothing here is point-in-time; no number produced by
this module may be presented as a forecastable edge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# schedule controls: the confound that contaminates a raw home-margin split
CONTROLS = ["h_b2b", "a_b2b", "h_3in4", "a_3in4", "rest_diff",
            "h_travel_k", "a_travel_k", "h_tz_abs", "a_tz_abs"]


def add_controls(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    hr = d["h_rest"].fillna(3.0).clip(upper=6.0)
    ar = d["a_rest"].fillna(3.0).clip(upper=6.0)
    d["rest_diff"] = (hr - ar).clip(-4, 4)
    d["h_travel_k"] = d["h_travel"] / 1000.0
    d["a_travel_k"] = d["a_travel"] / 1000.0
    d["h_tz_abs"] = d["h_tz"].abs()
    d["a_tz_abs"] = d["a_tz"].abs()
    return d


def design(df: pd.DataFrame, teams: list[str], controls: list[str] | None,
           extra: np.ndarray | None = None):
    """[Z | H | C | extra] design matrix."""
    n = len(df)
    idx = {t: i for i, t in enumerate(teams)}
    k = len(teams)
    Z = np.zeros((n, k))
    H = np.zeros((n, k))
    hi = df["home"].map(idx).to_numpy()
    ai = df["away"].map(idx).to_numpy()
    rows = np.arange(n)
    Z[rows, hi] += 1.0
    Z[rows, ai] -= 1.0
    H[rows, hi] = 1.0
    blocks = [Z, H]
    if controls:
        blocks.append(df[controls].to_numpy(float))
    if extra is not None:
        blocks.append(extra)
    return np.hstack(blocks), k


def fit_season(df: pd.DataFrame, controls: list[str] | None = CONTROLS,
               fast_rank: bool = False):
    """Per-season fit. Returns dict with a_t, d_t, SEs, league HFA."""
    teams = sorted(set(df["home"]) | set(df["away"]))
    X, k = design(df, teams, controls)
    y = df["margin"].to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    # the Z block has exactly one null direction (sum_t z_t == 0 identically);
    # the H block spans the intercept. So rank = (k-1) + k + n_controls.
    rank = (2 * k - 1 + (len(controls) if controls else 0)) if fast_rank \
        else np.linalg.matrix_rank(X)
    dof = len(y) - rank
    sigma2 = resid @ resid / dof
    XtXp = np.linalg.pinv(X.T @ X)
    cov = sigma2 * XtXp
    a = coef[k:2 * k]
    cov_a = cov[k:2 * k, k:2 * k]
    # d = (I - J/k) a
    M = np.eye(k) - np.ones((k, k)) / k
    d = M @ a
    cov_d = M @ cov_a @ M.T
    se_d = np.sqrt(np.clip(np.diag(cov_d), 0, None))
    se_a = np.sqrt(np.clip(np.diag(cov_a), 0, None))
    hfa = float(a.mean())
    se_hfa = float(np.sqrt(np.ones(k) @ cov_a @ np.ones(k)) / k)
    ctrl = dict(zip(controls, coef[2 * k:2 * k + len(controls)])) if controls else {}
    return dict(teams=teams, a=a, d=d, se_a=se_a, se_d=se_d, cov_d=cov_d,
                hfa=hfa, se_hfa=se_hfa, sigma=float(np.sqrt(sigma2)),
                n=len(y), dof=dof, controls=ctrl,
                s=coef[:k], resid=resid)


def eb_shrink(d: np.ndarray, se: np.ndarray):
    """Method-of-moments EB. Returns (tau2, shrunk_d, signal_share)."""
    v_obs = float(np.var(d, ddof=1))
    v_noise = float(np.mean(se ** 2))
    tau2 = max(0.0, v_obs - v_noise)
    w = tau2 / (tau2 + se ** 2)
    return tau2, w * d, (tau2 / v_obs if v_obs > 0 else 0.0)


def ols(X: np.ndarray, y: np.ndarray):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    rank = np.linalg.matrix_rank(X)
    dof = len(y) - rank
    s2 = r @ r / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    return coef, cov, float(np.sqrt(s2)), dof


def boot_ci(vals: np.ndarray, lo=2.5, hi=97.5):
    return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))


def load_panel(path=None) -> pd.DataFrame:
    path = path or ("data/scratch/ha_panel.csv")
    d = pd.read_csv(path)
    d = d[d["neutral"] == 0].reset_index(drop=True)
    return add_controls(d)
