#!/usr/bin/env python3
"""D100 JOB-2c: RE-RUN THE VOIDED D29/D31 SUFFICIENCY PROOF.

D31 registered: "v2b STILL fails -> sufficiency conclusion: possession-points-
RAPM == stint RAPM; v2 value must be EVENT-level. Possession-margin fitting
RETIRED with proof." D29 registered the first-cut failure (net vs DARKO
-0.059).

Both were fitted on `possessions_v2` as it sat in the DB, where (D99 forensics)
`def_team = 0` on 100% of rows and off_lineup matched off_team on 0.5004 of
player-slots — i.e. on ~half the rows the "defensive" lineup WAS the offensive
lineup. A RAPM design matrix that is a coin flip cannot reject a model class,
so the proof was VOID. possessions_v2 has now been force-rebuilt with the
D81-fixed parser (886,836 possessions, off/def lineup-team agreement 1.0000).

ARMS — everything paired, same seeds, same sample, same models:

  A. REPRODUCTION (verbatim D29 / D31 models)
     FIXED  : rebuilt table, 002 only
     BUGGY  : the exact pre-fix design matrix, reconstructed by swapping
              off_lineup/def_lineup on every HOME-offense row (that is
              precisely what `is_home = cur_team == None` did).
     Metric : net-vs-DARKO corr (D29 registered -0.059; v1 stint = 0.625) and
              net-vs-v1 corr.

  B. SUFFICIENCY (the claim itself, tested directly -- new, because the
     registered "proof" never had a held-out arm)
     Game-level 70/30 split (leakage-safe: no game spans the split). Poisson
     ridge RAPM fitted by IRLS on train; scored by held-out mean Poisson
     log-likelihood per possession, paired bootstrap 2000x CLUSTERED BY GAME.
       NULL   : intercept only
       V1     : v1 stint-RAPM net as a fitted-scale offset (+off -def)
       STINT  : stint RAPM REFITTED on exactly the same training games -- the
                honest opponent, since v1_posterior.npz is a fit from another
                era on another universe
       V2     : possession-fitted off/def, free
       V2+V1  : possession fit shrunk toward the v1 prior
     BOTH ridges are chosen by `select_ridges()` on a train-internal 80/20
     validation split, NEVER on test. This is load-bearing: the head-to-head
     reads WORSE at possession ridge 50, NS at 200 and BETTER at 800, so a
     hand-picked value is a coin toss dressed as a result.

  RESULT (D108): the possession arm WINS. vs same-data stint RAPM +0.000346
  CI(+0.000199,+0.000508); vs v1 +0.000619 CI(+0.000434,+0.000805); vs null
  +0.001137. Both stint arms beat null too, so it is not a weak baseline. On
  the BUGGY design the advantage halves and the ratings correlate +0.018 with
  stint RAPM (fixed: +0.591). D31's sufficiency claim is REFUTED.
  SCOPE: possession log-likelihood only -- nothing shipped, no gate run.

Read-only against DuckDB. Writes data/cg_v2_sufficiency.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                    # noqa: E402
from scipy import sparse                              # noqa: E402
from scipy.sparse.linalg import cg as sp_cg           # noqa: E402

from nbapred.db import connect                        # noqa: E402

NBOOT = 2000
MAX_POSS = 200_000          # fit_v2 / fit_v2b default
V1_SCALE = 2.0 * 55.0       # fit_v2b: net pts/48 -> per-possession log-rate


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------- data
def load():
    """Return (games, off[N,5], deff[N,5], y[N], pid_index, darko, v1net) for
    the FIXED design, plus the boolean home-offense mask that generates BUGGY."""
    con = connect(read_only=True)
    df = con.execute("""
        SELECT p.game_id, p.off_team, p.off_lineup, p.def_lineup, p.points
        FROM possessions_v2 p
        WHERE p.game_id LIKE '002%'""").fetchdf()
    home = dict(con.execute("""
        SELECT DISTINCT game_id, team_id FROM nba_games WHERE is_home""").fetchall())
    darko = dict(con.execute("""
        SELECT nba_player_id, o_dpm+d_dpm FROM darko_dpm
        WHERE snapshot_date=(SELECT max(snapshot_date) FROM darko_dpm)""").fetchall())
    con.close()
    v1 = np.load(ROOT / "data/v1_posterior.npz")
    v1net = dict(zip(v1["player_ids"].tolist(), v1["net"].tolist()))

    pids = {}
    G, O, D, Y, H = [], [], [], [], []
    miss_home = 0
    for r in df.itertuples():
        try:
            o = [int(x) for x in r.off_lineup.split(",")]
            d = [int(x) for x in r.def_lineup.split(",")]
        except Exception:
            continue
        if len(o) != 5 or len(d) != 5:
            continue
        for p in o + d:
            pids.setdefault(p, len(pids))
        G.append(r.game_id)
        O.append([pids[p] for p in o])
        D.append([pids[p] for p in d])
        Y.append(min(r.points, 4))
        h = home.get(r.game_id)
        if h is None:
            miss_home += 1
        H.append(h is not None and int(r.off_team) == int(h))
    log(f"loaded {len(Y)} 002 possessions, {len(pids)} players, "
        f"{len(set(G))} games; home-offense {np.mean(H):.4f}; "
        f"rows with no schedule home row {miss_home}")
    return (np.array(G), np.array(O), np.array(D), np.array(Y, float),
            np.array(H), pids, darko, v1net)


def buggy_view(O, D, H):
    """The pre-D81 design: `is_home` was always False, so lineup_at() returned
    (away5, home5) for EVERY possession -> home-offense rows got off/def
    swapped. Away-offense rows are identical to the fixed build."""
    Ob, Db = O.copy(), D.copy()
    Ob[H], Db[H] = D[H], O[H]
    return Ob, Db


# ------------------------------------------------------- A: verbatim D29/D31
def fit_nuts(O, D, Y, P, prior_mean=None, seed=0, label=""):
    """fit_v2.py model (prior_mean=None) or fit_v2b.py model (non-centered with
    the v1 prior). Both verbatim, including hyperpriors and sampler settings."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from jax import random
    from numpyro.infer import MCMC, NUTS

    Oj, Dj, Yj = jnp.asarray(O), jnp.asarray(D), jnp.asarray(Y)
    if prior_mean is None:
        def model():
            mu = numpyro.sample("mu", dist.Normal(0.1, 0.5))
            so = numpyro.sample("so", dist.HalfNormal(0.1))
            sd = numpyro.sample("sd", dist.HalfNormal(0.1))
            with numpyro.plate("po", P):
                off = numpyro.sample("off", dist.Normal(0.0, so))
            with numpyro.plate("pd", P):
                deff = numpyro.sample("deff", dist.Normal(0.0, sd))
            lr = mu + off[Oj].sum(1) - deff[Dj].sum(1)
            numpyro.sample("y", dist.Poisson(jnp.exp(lr)), obs=Yj)
        kern = NUTS(model, max_tree_depth=8)
        nw, ns = 400, 500
    else:
        pm = jnp.asarray(prior_mean)

        def model():
            mu = numpyro.sample("mu", dist.Normal(0.09, 0.2))
            so = numpyro.sample("so", dist.HalfNormal(0.05))
            sd = numpyro.sample("sd", dist.HalfNormal(0.05))
            with numpyro.plate("po", P):
                zo = numpyro.sample("zo", dist.Normal(0.0, 1.0))
            with numpyro.plate("pd", P):
                zd = numpyro.sample("zd", dist.Normal(0.0, 1.0))
            off = pm + so * zo
            deff = pm + sd * zd
            lr = mu + off[Oj].sum(1) - deff[Dj].sum(1)
            numpyro.sample("y", dist.Poisson(jnp.exp(lr)), obs=Yj)
        kern = NUTS(model, max_tree_depth=10, target_accept_prob=0.9)
        nw, ns = 500, 500
    t0 = time.time()
    mcmc = MCMC(kern, num_warmup=nw, num_samples=ns, progress_bar=False)
    mcmc.run(random.PRNGKey(seed))
    s = mcmc.get_samples()
    if prior_mean is None:
        net = np.asarray(s["off"]).mean(0) + np.asarray(s["deff"]).mean(0)
        extra = {}
    else:
        off = (prior_mean[None, :] + np.asarray(s["so"])[:, None] * np.asarray(s["zo"])).mean(0)
        deff = (prior_mean[None, :] + np.asarray(s["sd"])[:, None] * np.asarray(s["zd"])).mean(0)
        net = off + deff
        extra = {"sigma_off": float(np.asarray(s["so"]).mean()),
                 "sigma_def": float(np.asarray(s["sd"]).mean())}
    log(f"  [{label}] NUTS {time.time()-t0:.0f}s")
    return net, extra


def corr_vs(net, inv, ref):
    a = np.array([(net[i], ref[inv[i]]) for i in range(len(net)) if inv[i] in ref])
    if len(a) < 10:
        return None, 0
    return float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), int(len(a))


def arm_reproduction(O, D, Y, pids, darko, v1net, tag):
    inv = {v: k for k, v in pids.items()}
    P = len(pids)
    if len(Y) > MAX_POSS:
        idx = np.random.default_rng(0).choice(len(Y), MAX_POSS, replace=False)
        Os, Ds, Ys = O[idx], D[idx], Y[idx]
    else:
        Os, Ds, Ys = O, D, Y
    out = {"n_poss": int(len(Ys)), "players": P}
    log(f"\n[{tag}] D29 arm (fit_v2 model) on {len(Ys)} possessions, {P} players")
    net, _ = fit_nuts(Os, Ds, Ys, P, None, seed=0, label=f"{tag}/D29")
    c, n = corr_vs(net, inv, darko)
    c1, n1 = corr_vs(net, inv, v1net)
    out["D29_v2"] = {"net_vs_darko": c, "n_darko": n, "net_vs_v1": c1, "n_v1": n1}
    log(f"  D29 net vs DARKO {c:+.3f} (n={n})   net vs v1 {c1:+.3f} (n={n1})")

    prior_mean = np.array([v1net.get(inv[i], 0.0) / V1_SCALE for i in range(P)])
    log(f"[{tag}] D31 arm (fit_v2b model), v1-prior coverage "
        f"{np.mean([inv[i] in v1net for i in range(P)]):.2f}")
    netb, ex = fit_nuts(Os, Ds, Ys, P, prior_mean, seed=1, label=f"{tag}/D31")
    c, n = corr_vs(netb, inv, darko)
    c1, n1 = corr_vs(netb, inv, v1net)
    resid = netb - 2 * prior_mean
    cr, nr = corr_vs(resid, inv, darko)
    out["D31_v2b"] = {"net_vs_darko": c, "n_darko": n, "net_vs_v1": c1,
                      "resid_vs_darko": cr, "n_resid": nr, **ex}
    log(f"  D31 net vs DARKO {c:+.3f} (n={n})   net vs v1 {c1:+.3f}   "
        f"data-residual vs DARKO {cr:+.3f}")
    return out


# ------------------------------------------------- B: held-out sufficiency
def design(O, D, P):
    """Sparse [1 | +off | -def] design, one row per possession."""
    n = len(O)
    rows = np.repeat(np.arange(n), 11)
    cols = np.concatenate([np.zeros((n, 1), int), 1 + O, 1 + P + D], axis=1).ravel()
    vals = np.concatenate([np.ones((n, 1)), np.ones((n, 5)), -np.ones((n, 5))],
                          axis=1).ravel()
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, 1 + 2 * P))


def irls_poisson(X, y, ridge, b0=None, offset=None, iters=25, tol=1e-8):
    """Newton/IRLS for Poisson log-link with an L2 penalty (intercept free).
    Solves via CG on the normal equations — P ~ 1k so this is seconds."""
    p = X.shape[1]
    b = np.zeros(p) if b0 is None else b0.copy()
    off = np.zeros(X.shape[0]) if offset is None else offset
    pen = np.full(p, ridge); pen[0] = 0.0
    prev = None
    for _ in range(iters):
        eta = X @ b + off
        np.clip(eta, -8, 4, out=eta)
        mu = np.exp(eta)
        # gradient / Hessian of the penalised negative log-lik
        g = X.T @ (mu - y) + pen * b
        W = sparse.diags(mu)
        H = (X.T @ W @ X).tocsc()
        H = H + sparse.diags(pen)
        step, _ = sp_cg(H, -g, rtol=1e-7, maxiter=500)
        b = b + step
        ll = float((y * (X @ b + off) - np.exp(np.clip(X @ b + off, -8, 4))).mean())
        if prev is not None and abs(ll - prev) < tol:
            break
        prev = ll
    return b


def poisll(eta, y):
    eta = np.clip(eta, -8, 4)
    return y * eta - np.exp(eta)


def boot_ci(d, groups, nboot=NBOOT, seed=0):
    rng = np.random.default_rng(seed)
    _, inv = np.unique(groups, return_inverse=True)
    Gn = inv.max() + 1
    s = np.bincount(inv, weights=d, minlength=Gn)
    c = np.bincount(inv, minlength=Gn).astype(float)
    pick = rng.integers(0, Gn, size=(nboot, Gn))
    b = s[pick].sum(1) / c[pick].sum(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return [float(lo), float(hi)]


def fit_stint_rapm(train_games, pids, ridge=2000.0):
    """Stint RAPM fitted on EXACTLY the training games — the honest opponent for
    the "possession-margin fitting == stint RAPM" claim. The registered proof
    compared against `data/v1_posterior.npz`, a fit from a different era on a
    different universe; here both sides see the same games.

    Weighted ridge of stint margin-per-second on (home five) - (away five);
    returns a net rating per player on the per-possession log-rate scale used by
    the possession arms."""
    con = connect(read_only=True)
    df = con.execute("""SELECT game_id, home_lineup, away_lineup, margin, seconds
                        FROM lineup_stints
                        WHERE game_id LIKE '002%' AND seconds > 0""").fetchdf()
    con.close()
    tg = set(train_games)
    P = len(pids)
    rows, cols, vals, y, w = [], [], [], [], []
    r = 0
    for t in df.itertuples():
        if t.game_id not in tg:
            continue
        try:
            h = [pids[int(x)] for x in t.home_lineup.split(",")]
            a = [pids[int(x)] for x in t.away_lineup.split(",")]
        except (KeyError, ValueError):
            continue
        if len(h) != 5 or len(a) != 5:
            continue
        for i in h:
            rows.append(r); cols.append(i); vals.append(1.0)
        for i in a:
            rows.append(r); cols.append(i); vals.append(-1.0)
        y.append(t.margin / t.seconds)          # margin per second
        w.append(t.seconds)
        r += 1
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(r, P))
    y = np.asarray(y); w = np.asarray(w)
    W = sparse.diags(w)
    A = (X.T @ W @ X).toarray()
    A[np.diag_indices_from(A)] += ridge
    b = np.linalg.solve(A, X.T @ (w * y))
    log(f"  stint RAPM: {r} train stints, ridge {ridge}, "
        f"rating sd {b.std():.5f} (margin/sec scale)")
    return b


def arm_sufficiency(G, O, D, Y, pids, v1net, tag, ridge=200.0, stint_net=None):
    inv = {v: k for k, v in pids.items()}
    P = len(pids)
    ug = np.unique(G)
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(ug))
    tr_g = set(ug[perm[:int(len(ug) * 0.7)]])
    m = np.array([g in tr_g for g in G])
    log(f"\n[{tag}] SUFFICIENCY split: train {m.sum()} poss / {len(tr_g)} games, "
        f"test {(~m).sum()} poss / {len(ug)-len(tr_g)} games")
    Xtr, Xte = design(O[m], D[m], P), design(O[~m], D[~m], P)
    ytr, yte = Y[m], Y[~m]
    gte = G[~m]

    # ---- NULL: intercept only
    mu0 = np.log(max(ytr.mean(), 1e-6))
    ll_null = poisll(np.full(len(yte), mu0), yte)

    # ---- V1: stint net as FIXED offsets, only the intercept + one scale fitted
    v1v = np.array([v1net.get(inv[i], 0.0) / V1_SCALE for i in range(P)])
    # SIGN: v1 "net" is a net-impact rating (positive = good PLAYER). Predicting
    # the OFFENSE's points on a possession, good offensive players RAISE it and
    # good defensive players LOWER it -> +off, -def. (A `+` here makes the
    # regressor the sum of all ten ratings, which is nearly constant and
    # carries no offence-vs-defence signal — it would score as "v1 adds
    # nothing" for a purely mechanical reason.)
    z_tr = v1v[O[m]].sum(1) - v1v[D[m]].sum(1)
    z_te = v1v[O[~m]].sum(1) - v1v[D[~m]].sum(1)
    Z = np.column_stack([np.ones(len(ytr)), z_tr])
    bz = irls_poisson(sparse.csr_matrix(Z), ytr, 0.0)
    ll_v1 = poisll(bz[0] + bz[1] * z_te, yte)

    # ---- STINT: RAPM fitted on the SAME train games (the honest opponent)
    ll_st = None
    if stint_net is not None:
        s_tr = stint_net[O[m]].sum(1) - stint_net[D[m]].sum(1)
        s_te = stint_net[O[~m]].sum(1) - stint_net[D[~m]].sum(1)
        Zs = np.column_stack([np.ones(len(ytr)), s_tr])
        bs = irls_poisson(sparse.csr_matrix(Zs), ytr, 0.0)
        ll_st = poisll(bs[0] + bs[1] * s_te, yte)

    # ---- V2: free possession RAPM
    b2 = irls_poisson(Xtr, ytr, ridge)
    ll_v2 = poisll(Xte @ b2, yte)

    # ---- V2+V1: possession fit shrunk toward the v1 prior
    prior = np.concatenate([[0.0], v1v, v1v])
    off_tr = Xtr @ prior
    off_te = Xte @ prior
    b3 = irls_poisson(Xtr, ytr, ridge, offset=off_tr)
    ll_v3 = poisll(Xte @ b3 + off_te, yte)

    res = {"tag": tag, "ridge": ridge, "n_train": int(m.sum()), "n_test": int((~m).sum()),
           "ll_null": float(ll_null.mean()), "ll_v1": float(ll_v1.mean()),
           "ll_v2": float(ll_v2.mean()), "ll_v2v1": float(ll_v3.mean())}
    pairs = [("v1_vs_null", ll_v1, ll_null),
             ("v2_vs_null", ll_v2, ll_null),
             ("V2_vs_V1", ll_v2, ll_v1),
             ("V2V1_vs_V1", ll_v3, ll_v1),
             ("V2V1_vs_V2", ll_v3, ll_v2)]
    if ll_st is not None:
        res["ll_stint"] = float(ll_st.mean())
        pairs += [("stint_vs_null", ll_st, ll_null),
                  ("V2_vs_STINT", ll_v2, ll_st),          # <- THE claim
                  ("V2V1_vs_STINT", ll_v3, ll_st)]
    for name, a, b in pairs:
        d = a - b
        ci = boot_ci(d, gte)
        verd = "BETTER" if ci[0] > 0 else ("NS" if ci[1] > 0 else "WORSE")
        res[name] = {"delta": float(d.mean()), "ci_game_cluster": ci, "verdict": verd}
        log(f"  {name:12s} {d.mean():+.6f}  CI({ci[0]:+.6f},{ci[1]:+.6f})  {verd}")
    # fitted-rating correlations for context
    net2 = b2[1:1 + P] + b2[1 + P:]
    c, n = corr_vs(net2, inv, v1net)
    res["ridge_net_vs_v1"] = {"corr": c, "n": n}
    log(f"  ridge possession-net vs v1 stint-net corr {c:+.3f} (n={n})")
    return res


def select_ridges(G, O, D, Y, pids, tr_games,
                  p_grid=(100., 200., 400., 800., 1600., 3200., 6400., 12800., 25600.),
                  s_grid=(500., 2000., 8000., 32000., 128000.)):
    """Pick both ridges on a VALIDATION split carved out of TRAIN ONLY.

    Necessary, not cosmetic: a sensitivity display showed the possession-vs-
    stint verdict flips with the possession ridge (WORSE at 50, NS at 200,
    BETTER at 800), so any single hand-picked value is a coin toss dressed as a
    result. The test set is scored exactly once, at the chosen pair."""
    P = len(pids)
    tr = list(tr_games)
    n_fit = int(len(tr) * 0.8)
    fit_g, val_g = set(tr[:n_fit]), set(tr[n_fit:])
    mf = np.array([g in fit_g for g in G])
    mv = np.array([g in val_g for g in G])
    log(f"\nRIDGE SELECTION (train-internal): fit {mf.sum()} poss / {len(fit_g)} games, "
        f"val {mv.sum()} poss / {len(val_g)} games")
    Xf, Xv = design(O[mf], D[mf], P), design(O[mv], D[mv], P)
    best_p, best_ll = None, -np.inf
    for r in p_grid:
        b = irls_poisson(Xf, Y[mf], r)
        ll = float(poisll(Xv @ b, Y[mv]).mean())
        log(f"  possession ridge {r:8.0f}  val LL {ll:.7f}")
        if ll > best_ll:
            best_ll, best_p = ll, r
    best_s, best_sll = None, -np.inf
    for r in s_grid:
        sn = fit_stint_rapm(fit_g, pids, ridge=r)
        zf = sn[O[mf]].sum(1) - sn[D[mf]].sum(1)
        zv = sn[O[mv]].sum(1) - sn[D[mv]].sum(1)
        b = irls_poisson(sparse.csr_matrix(
            np.column_stack([np.ones(int(mf.sum())), zf])), Y[mf], 0.0)
        ll = float(poisll(b[0] + b[1] * zv, Y[mv]).mean())
        log(f"  stint ridge      {r:8.0f}  val LL {ll:.7f}")
        if ll > best_sll:
            best_sll, best_s = ll, r
    log(f"  CHOSEN possession ridge {best_p}, stint ridge {best_s}")
    return best_p, best_s


def main():
    stage = os.environ.get("STAGE", "all")
    t0 = time.time()
    G, O, D, Y, H, pids, darko, v1net = load()
    Ob, Db = buggy_view(O, D, H)
    out = {"corpus": {"possessions": int(len(Y)), "games": int(len(set(G.tolist()))),
                      "players": len(pids), "home_offense_frac": float(H.mean())}}

    if stage in ("suff", "suff2", "all"):
        ug = np.unique(G)
        perm = np.random.default_rng(7).permutation(len(ug))
        tr_g = ug[perm[:int(len(ug) * 0.7)]]        # same split arm_sufficiency uses
        pr, sr = select_ridges(G, O, D, Y, pids, tr_g)
        out["chosen_ridges"] = {"possession": pr, "stint": sr,
                                "selected_on": "train-internal 80/20 validation"}
        snet = fit_stint_rapm(tr_g, pids, ridge=sr)
        out["sufficiency_fixed"] = arm_sufficiency(G, O, D, Y, pids, v1net,
                                                   "FIXED", ridge=pr, stint_net=snet)
        out["sufficiency_buggy"] = arm_sufficiency(G, Ob, Db, Y, pids, v1net,
                                                   "BUGGY", ridge=pr, stint_net=snet)
    if stage in ("repro", "all"):
        out["repro_fixed"] = arm_reproduction(O, D, Y, pids, darko, v1net, "FIXED")
        out["repro_buggy"] = arm_reproduction(Ob, Db, Y, pids, darko, v1net, "BUGGY")

    p = ROOT / "data/cg_v2_sufficiency.json"
    if p.exists():
        prev = json.loads(p.read_text()); prev.update(out); out = prev
    p.write_text(json.dumps(out, indent=1))
    log(f"\nwrote {p}  ({time.time()-t0:.0f}s)\nSUFFICIENCY_DONE stage={stage}")


if __name__ == "__main__":
    main()
