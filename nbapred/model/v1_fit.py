"""Full v1 Bayesian skill fit (handoff II.1-II.2, completed at last):
hierarchical player skills across ALL dimensions, fit JOINTLY with:
  * event likelihoods — binomial makes (rim/mid/3/FT), Poisson counts per
    minute (TOV, OREB, DREB, AST, STL, BLK)
  * the stint-margin likelihood (RAPM term) — Normal margin per stint driven by
    on-court players' net two-way theta. Identification-critical for defense.
Priors: theta_k ~ N(alpha_k + beta_k * z2k_k, sigma_k) where a crosswalked 2K
attribute exists (beta learned = per-dimension 2K trust); else hierarchical mean.

numpyro NUTS on GPU (jax cuda). v1 = season-aggregated sufficient statistics.
"""
from __future__ import annotations

import json

import numpy as np

BINOM_DIMS = {   # dim: (makes col, att col, 2K attribute)
    "rim": ("rimm", "rima", "Close Shot"),
    "mid": ("midm", "mida", "Mid-Range Shot"),
    "thr": ("thrm", "thra", "Three-Point Shot"),
    "ft": ("ftm", "fta", "Free Throw"),
}
POIS_DIMS = {    # dim: (count col, 2K attribute or None)
    "tov": ("tov", "Ball Handle"),
    "oreb": ("oreb", "Offensive Rebound"),
    "dreb": ("dreb", "Defensive Rebound"),
    "ast": ("ast", "Pass Accuracy"),
    "stl": ("stl", "Steal"),
    "blk": ("blk", "Block"),
}


def build_dataset(con, season=None, min_minutes=100.0, max_stints=25000):
    """Aggregate sufficient stats + stints, aligned to one player index.
    season=None -> current season by calendar (dynamic)."""
    if season is None:
        from ..config import current_season
        season = current_season()
    df = con.execute("""SELECT s.player_id,
        sum(s.rimm) rimm, sum(s.rima) rima, sum(s.midm) midm, sum(s.mida) mida,
        sum(s.thrm) thrm, sum(s.thra) thra, sum(s.ftm) ftm, sum(s.fta) fta,
        sum(s.tov) tov, sum(s.oreb) oreb, sum(s.dreb) dreb, sum(s.ast) ast,
        sum(s.stl) stl, sum(s.blk) blk, sum(s.seconds)/60.0 mins
        FROM player_game_stats s JOIN nba_games g ON g.game_id=s.game_id AND g.team_id=s.team_id
        WHERE g.season=? AND s.game_id LIKE '002%' GROUP BY 1""", [season]).fetchdf()
    df = df[df.mins >= min_minutes].reset_index(drop=True)
    pid2i = {int(p): i for i, p in enumerate(df.player_id)}

    # 2K z-scores per dimension via crosswalk
    xw = con.execute("SELECT nba_player_id, name_2k FROM player_xwalk WHERE name_2k IS NOT NULL").fetchdf()
    r2k = con.execute("SELECT player_name, attributes FROM ratings_2k "
                      "WHERE scrape_date=(SELECT max(scrape_date) FROM ratings_2k)").fetchdf()
    name2attr = {r.player_name: json.loads(r.attributes) for r in r2k.itertuples()}
    pid2attr = {int(r.nba_player_id): name2attr.get(r.name_2k) for r in xw.itertuples()}

    def zvec(attr_name):
        # explicit None check: `or np.nan` coerced a legitimate 0 rating to NaN
        def _get(p):
            v = (pid2attr.get(int(p)) or {}).get(attr_name)
            return np.nan if v is None else float(v)
        vals = np.array([_get(p) for p in df.player_id])
        mu, sd = np.nanmean(vals), np.nanstd(vals)
        return np.nan_to_num((vals - mu) / (sd or 1.0))

    data = {"n_players": len(df), "mins": df.mins.to_numpy(float),
            "player_ids": df.player_id.to_numpy(int)}
    for d, (mk, at, a2k) in BINOM_DIMS.items():
        data[f"{d}_m"] = df[mk].to_numpy(int)
        data[f"{d}_n"] = df[at].to_numpy(int)
        data[f"{d}_z"] = zvec(a2k)
    for d, (col, a2k) in POIS_DIMS.items():
        data[f"{d}_c"] = df[col].to_numpy(int)
        data[f"{d}_z"] = zvec(a2k) if a2k else np.zeros(len(df))

    # stints (subsample for tractability), map lineups to indices
    st = con.execute("""SELECT home_lineup, away_lineup, seconds, margin FROM lineup_stints s
        WHERE s.game_id IN (SELECT game_id FROM nba_games WHERE season=?)
          AND seconds > 30""", [season]).fetchdf()
    H, A, w, m = [], [], [], []
    for r in st.itertuples():
        try:
            h = [pid2i[int(x)] for x in r.home_lineup.split(",")]
            a = [pid2i[int(x)] for x in r.away_lineup.split(",")]
        except (KeyError, ValueError):
            continue
        if len(h) == 5 and len(a) == 5:
            H.append(h); A.append(a)
            w.append(r.seconds / 60.0); m.append(float(r.margin))
    if len(H) > max_stints:
        idx = np.random.default_rng(0).choice(len(H), max_stints, replace=False)
        H = [H[i] for i in idx]; A = [A[i] for i in idx]
        w = [w[i] for i in idx]; m = [m[i] for i in idx]
    data["stint_home"] = np.array(H, int)
    data["stint_away"] = np.array(A, int)
    data["stint_min"] = np.array(w, float)
    data["stint_margin"] = np.array(m, float)
    return data


def model(d):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    P = d["n_players"]
    mins = jnp.asarray(d["mins"])

    thetas = {}
    for dim in BINOM_DIMS:
        mu = numpyro.sample(f"mu_{dim}", dist.Normal(0.0, 1.5))
        beta = numpyro.sample(f"beta_{dim}", dist.Normal(0.0, 1.0))
        sig = numpyro.sample(f"sig_{dim}", dist.HalfNormal(1.0))
        with numpyro.plate(f"pl_{dim}", P):
            a = numpyro.sample(f"a_{dim}", dist.Normal(0.0, 1.0))
        logit = mu + beta * jnp.asarray(d[f"{dim}_z"]) + sig * a
        numpyro.sample(f"obs_{dim}", dist.Binomial(
            total_count=jnp.asarray(d[f"{dim}_n"]), logits=logit),
            obs=jnp.asarray(d[f"{dim}_m"]))
        thetas[dim] = logit

    for dim in POIS_DIMS:
        mu = numpyro.sample(f"mu_{dim}", dist.Normal(-3.0, 1.5))
        beta = numpyro.sample(f"beta_{dim}", dist.Normal(0.0, 1.0))
        sig = numpyro.sample(f"sig_{dim}", dist.HalfNormal(1.0))
        with numpyro.plate(f"pl_{dim}", P):
            a = numpyro.sample(f"a_{dim}", dist.Normal(0.0, 1.0))
        lograte = mu + beta * jnp.asarray(d[f"{dim}_z"]) + sig * a
        numpyro.sample(f"obs_{dim}", dist.Poisson(jnp.exp(lograte) * mins),
                       obs=jnp.asarray(d[f"{dim}_c"]))
        thetas[dim] = lograte

    # stint-margin likelihood: net two-way impact per player (per-48 pts scale)
    sig_net = numpyro.sample("sig_net", dist.HalfNormal(3.0))
    with numpyro.plate("pl_net", P):
        net = numpyro.sample("net", dist.Normal(0.0, sig_net))
    home_edge = numpyro.sample("home_edge", dist.Normal(2.0, 2.0))
    hsum = net[jnp.asarray(d["stint_home"])].sum(axis=1)
    asum = net[jnp.asarray(d["stint_away"])].sum(axis=1)
    mins_st = jnp.asarray(d["stint_min"])
    mu_m = (hsum - asum + home_edge) * mins_st / 48.0
    sd_m = 4.0 * jnp.sqrt(jnp.maximum(mins_st, 0.5))     # scoring noise scales sqrt(t)
    numpyro.sample("obs_margin", dist.Normal(mu_m, sd_m),
                   obs=jnp.asarray(d["stint_margin"]))


def fit(data, num_warmup=500, num_samples=600, seed=0):
    import numpyro
    from jax import random
    from numpyro.infer import MCMC, NUTS
    numpyro.set_host_device_count(1)
    mcmc = MCMC(NUTS(model, max_tree_depth=8), num_warmup=num_warmup,
                num_samples=num_samples, progress_bar=True)
    mcmc.run(random.PRNGKey(seed), d=data)
    return mcmc
