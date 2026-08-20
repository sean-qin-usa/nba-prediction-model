"""First real Bayesian skill fit: hierarchical player shooting model (II.1/II.2).

Scope: the shooting dimensions (3PT / rim / mid / FT) as binomial event
likelihoods. This is a genuine slice of the skill model — full-Bayes (NUTS via
numpyro), not the empirical-Bayes approximation in skill_priors.py. It produces
a POSTERIOR per player per dimension (mean + uncertainty), which is the
load-bearing feature: thin-data players get wide posteriors.

Model per dimension k (makes m_i of attempts n_i for player i):
    logit(p_{i}) = mu_k + beta_k * z_rating_{i,k} + sigma_k * a_i     # a_i ~ N(0,1)
    m_i ~ Binomial(n_i, p_i)
- mu_k        league logit-mean
- beta_k      LEARNED trust in the external rating (z-scored). Hypothesis
              (handoff II.1): FT shows the highest beta.
- sigma_k     residual talent spread beyond what the rating explains
- a_i         per-player standardized skill; posterior mean*sigma+... is theta

Rating z-score comes in as `z` (2K / DARKO / trailing-stat prior center). When
absent (rookie, no rating), z=0 -> player leans on the hierarchical mean, which
is exactly the intended behavior.

Leakage: caller supplies makes/attempts from a TRAILING window and z from an
as-of rating (see nbapred/pit.py). This module is time-agnostic.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS


def _model(z, n, m=None):
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.5))
    beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    with numpyro.plate("players", len(n)):
        a = numpyro.sample("a", dist.Normal(0.0, 1.0))
        logit_p = mu + beta * z + sigma * a
        numpyro.sample("obs", dist.Binomial(total_count=n, logits=logit_p), obs=m)


def fit_dimension(z, n, m, num_warmup=500, num_samples=1000, seed=0):
    """z, n, m: 1-D arrays (rating z-score, attempts, makes) per player.
    Returns posterior summaries incl. per-player p and the learned beta."""
    z = jnp.asarray(np.nan_to_num(np.asarray(z, float)))
    n = jnp.asarray(np.asarray(n, int))
    m = jnp.asarray(np.asarray(m, int))
    mcmc = MCMC(NUTS(_model), num_warmup=num_warmup, num_samples=num_samples,
                progress_bar=False)
    mcmc.run(random.PRNGKey(seed), z=z, n=n, m=m)
    s = mcmc.get_samples()
    mu, beta, sigma, a = s["mu"], s["beta"], s["sigma"], s["a"]
    # per-player posterior make-probability
    logit_p = mu[:, None] + beta[:, None] * z[None, :] + sigma[:, None] * a
    p = 1.0 / (1.0 + jnp.exp(-logit_p))
    return {
        "beta_mean": float(beta.mean()), "beta_sd": float(beta.std()),
        "mu_mean": float(mu.mean()), "sigma_mean": float(sigma.mean()),
        "p_mean": np.asarray(p.mean(axis=0)),          # per-player posterior mean prob
        "p_sd": np.asarray(p.std(axis=0)),             # per-player posterior uncertainty
        "n_eff_beta": float(numpyro.diagnostics.effective_sample_size(
            np.asarray(beta)[None, :])),
    }
