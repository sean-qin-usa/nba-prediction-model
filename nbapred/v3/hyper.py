"""v3 hyperparameters.

Two tiers (V3_SPEC 5 compute plan):
  * `HyperParams` — the player-level core's ~60 scalars (phi/Q per block,
    R per obs type, anchor coefs, shock lam/K, sigma_game, nu). Fit RARELY by
    SVI on GPU (M3); dataclass + defaults defined now so the StateBank
    evolution step is real code from M0.
  * `TeamHyper` — the M1 team-DLM's few scalars, fit MONTHLY walk-forward by
    closed-form marginal likelihood (innovation decomposition) on trailing
    observations. No grid: optimizer starts from hypothesis values grounded in
    the data (per-game ortg sd ~11 -> r_eff ~ 120; D16's 0.75 season regress
    -> kappa ~ 0.7) and moves by MLE.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

# ---------------------------------------------------------------- player tier
BLOCKS = ("impact", "minutes", "usage", "shooting", "volume")


@dataclasses.dataclass
class HyperParams:
    """Player-core hyperparameters (M3 fits these; defaults are priors).

    phi/Q: ONE value per block {impact, minutes, usage, shooting, volume}
    (5 numbers each, not 18 — V3_SPEC 2.1). lam/K: event-shock Q inflation
    multiplier and window length (days) per shock kind.
    """
    phi: dict = dataclasses.field(default_factory=lambda: {
        "impact": 0.999, "minutes": 0.99, "usage": 0.995,
        "shooting": 0.999, "volume": 0.997})
    Q: dict = dataclasses.field(default_factory=lambda: {
        "impact": 1e-4, "minutes": 5e-3, "usage": 1e-4,
        "shooting": 5e-5, "volume": 2e-4})
    R_obs: dict = dataclasses.field(default_factory=lambda: {
        "binomial": None,       # Fisher information (per-obs, closed form)
        "poisson": None,        # Fisher information
        "minutes": 36.0,        # TruncNormal var
        "stint": 16.0,          # per sqrt(min) margin noise (rapm.py scale)
        "darko": 0.5})          # DARKO treated as noisy obs of net_o/net_d
    lam: dict = dataclasses.field(default_factory=lambda: {
        "trade": 6.0, "return": 4.0, "coach": 3.0, "season_boundary": 10.0})
    K: dict = dataclasses.field(default_factory=lambda: {
        "trade": 21, "return": 10, "coach": 30, "season_boundary": 30})
    sigma_game: float = 11.0    # irreducible game margin noise (sd, pts)
    nu: float = 8.0             # Student-T link dof (M2 fits)


def fit_hyperparams(con, seasons, device: str = "cuda") -> HyperParams:
    """M3: numpyro SVI over the joint filtered likelihood (V3_SPEC 5)."""
    raise NotImplementedError("M3 — player-level hyperfit (SVI) not built yet")


# ------------------------------------------------------------------ team tier
@dataclasses.dataclass
class TeamHyper:
    """Team-DLM (M1) hyperparameters — the 'few hyperparams' the compute plan
    allows monthly MLE for. Units: ortg points (per 100 poss).

    phi       daily AR(1) reversion of off/def toward league mean (<1)
    q         per-day process VARIANCE of each off/def state (the DLM knob
              between the two degenerate corners: q->0 = equal-weight pooling,
              q large = pure recency)
    r_eff     obs variance of a per-team-game efficiency (ortg) observation
    r_margin  obs variance of a game margin observation (warm-up/playoff rows
              without possession data)
    kappa     season-boundary continuity: off/def <- kappa * off/def
    v_bound   season-boundary variance inflation added to each off/def state
              (the event-shock: integrated offseason Q inflation, replacing
              the 0.75-regress/20-game-ramp cold start structurally)
    q_mu/q_home/v_mu_bound: league-environment and home-edge drift constants —
              hypothesis-set, NOT fit (kept out of the MLE by design).
    """
    phi: float = 0.998
    q: float = 0.02
    r_eff: float = 120.0
    r_margin: float = 165.0
    kappa: float = 0.70
    v_bound: float = 4.0
    q_mu: float = 2e-3
    q_home: float = 2e-4
    v_mu_bound: float = 2.25

    FIT_KEYS = ("phi", "q", "r_eff", "r_margin", "kappa", "v_bound")

    def to_vector(self) -> np.ndarray:
        """Unconstrained transform for the optimizer: logit for (phi, kappa),
        log for variances."""
        def logit(p):
            return math.log(p / (1 - p))
        return np.array([logit(self.phi), math.log(self.q),
                         math.log(self.r_eff), math.log(self.r_margin),
                         logit(self.kappa), math.log(self.v_bound)])

    @classmethod
    def from_vector(cls, v, template: "TeamHyper | None" = None) -> "TeamHyper":
        t = template or cls()
        def ilogit(x):
            return 1.0 / (1.0 + math.exp(-x))
        return dataclasses.replace(
            t, phi=ilogit(v[0]), q=math.exp(v[1]), r_eff=math.exp(v[2]),
            r_margin=math.exp(v[3]), kappa=ilogit(v[4]), v_bound=math.exp(v[5]))


def fit_team_hyper(obs, teams, hyper0: TeamHyper | None = None,
                   maxiter: int = 80, seed_boundaries=None,
                   loglik_from=None) -> TeamHyper:
    """MLE of TeamHyper by the filter's own marginal likelihood.

    obs: chronological observation list (see team_obs.build_team_obs).
    Closed-form: each Kalman update contributes log N(innov; 0, S). The full
    stream is always REPLAYED (state continuity) but only obs on/after
    `loglik_from` are scored — the 'trailing data' window of the compute plan.
    Nelder-Mead over the 6 transformed params, warm-started at hyper0 —
    hypothesis-driven starting point, no grid (user rule).
    """
    from scipy.optimize import minimize

    from .team_dlm import TeamDLM

    h0 = hyper0 or TeamHyper()
    start = obs[0][0] if obs else None

    def negloglik(vec):
        try:
            h = TeamHyper.from_vector(vec, h0)
        except OverflowError:
            return 1e12
        if not (0.9 < h.phi < 0.99999) or not (0.01 < h.kappa < 0.999):
            return 1e12
        dlm = TeamDLM(teams, h, season_boundaries=seed_boundaries, start=start)
        ll = dlm.run(obs, loglik_from=loglik_from)
        return -ll if np.isfinite(ll) else 1e12

    res = minimize(negloglik, h0.to_vector(), method="Nelder-Mead",
                   options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-2))
    return TeamHyper.from_vector(res.x, h0)
