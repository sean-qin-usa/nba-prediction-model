#!/usr/bin/env python3
"""cg_usage_redis_clean.py — RE-RUN #3 of the contaminated-gates audit.

The whole star-out usage/redistribution family (D32 usage softmax, D33 attempt
lift PASS, D34/D35 points-CRPS rejections, D38 conditional tilt rejection, and
D57's "D33 survives PIT refit") rests on ONE fitted artifact: the conditional-
logit usage vector `u` in data/v2_usage.npz, produced by scripts/fit_v2_usage.py.

That collector contains the SAME bug D81 fixed elsewhere and it is STILL
UNFIXED here (and in scripts/audit_usage_pit.py, which re-implements it
verbatim for the D57 PIT refit):

    home = pbp["game"].get("homeTeamId")     # ALWAYS None -- key does not exist
    off5 = h5 if a["teamId"] == home else a5 # -> off5 is ALWAYS the AWAY five
    if pid in off5: shots.append(...)        # -> every HOME shot is DROPPED

cg_forensics measured it over the full cache: 424,285 of 847,142 shots kept =
49.9% silently discarded, and the survivors are the away half of the league.
So `u` was fitted on an away-only sample at half the advertised n.

This script isolates that bug and re-runs the family on the clean sample.

Design
------
ONE collection pass with the FIXED home-id derivation (from the rotation feed,
as D81 did for defense_zone/possessions_v2).  The buggy sample is exactly the
away-team subset of the clean sample, so both arms come from identical parsing
and the only difference is the selection.

  u_fixed  conditional logit MAP on ALL pre-cut shots
  u_buggy  same model / optimizer / seed on the AWAY-ONLY pre-cut shots

Cut = 60th-percentile game_date of the 2025-26 regular season (the cut
scripts/gate_redistribution_crps.py uses), so u is PIT for every gate below.

Gates re-run
  G0 (D32) held-out shooter log-loss on post-cut shots: uniform / unconditional
     share / conditional logit, u_fixed vs u_buggy.
  G1 (D33) star-out attempt lift, Poisson LL vs flat x1.020 -- the gate that
     PASSED.  Risk here is a false ACCEPT.
  G2 (D34/D35) star-out points CRPS with the softmax lift applied to the prop
     sim, on POST-D79 002-clean rates -- the gate that FAILED at -0.40.

Read-only.  Writes data/cg_usage_redis_clean.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import orjson

import nbapred.db as _db

if os.environ.get("CG_DB"):
    _db.DB_PATH = Path(os.environ["CG_DB"])

from nbapred.db import connect                               # noqa: E402
from nbapred.engine.props import (player_rates_from_stats,   # noqa: E402
                                  simulate_player)
from nbapred.features.cache_index import game_index          # noqa: E402
from nbapred.features.defense_zone import _game_segments     # noqa: E402
from nbapred.features.possessions_v2 import _team_ids        # noqa: E402
from nbapred.features.stints import _elapsed                 # noqa: E402

OUT = Path("data/cg_usage_redis_clean.json")
SHOT_CACHE = Path("data/cg_usage_shots.npz")


# --------------------------------------------------------------- shot collect
def collect(gid_dates: dict):
    """(lineups[N,5], shooters[N], is_away[N], ordinal[N]) with the CORRECT
    offensive five for every shot."""
    if SHOT_CACHE.exists():
        z = np.load(SHOT_CACHE)
        print(f"shot cache hit: {len(z['shooter'])} shots", flush=True)
        return z["lineup"], z["shooter"], z["is_away"], z["ordinal"]
    rots, pbps = game_index("gamerotation"), game_index("playbyplayv3")
    gids = sorted(set(rots) & set(pbps))
    L, S, A, O = [], [], [], []
    for gid in gids:
        try:
            rot = orjson.loads(open(rots[gid], "rb").read())["response"]
            pbp = orjson.loads(open(pbps[gid], "rb").read())["response"]
        except Exception:
            continue
        home_id, _ = _team_ids(rot, pbp)
        segs = _game_segments(rot, pbp)
        if not segs or home_id is None:
            continue
        t0 = np.array([s[0] for s in segs])
        od = gid_dates.get(gid, 0)
        for a in pbp.get("game", {}).get("actions", []):
            if a.get("actionType") not in ("Made Shot", "Missed Shot"):
                continue
            t = _elapsed(a.get("period"), a.get("clock"))
            pid = a.get("personId")
            if t is None or not pid:
                continue
            k = int(np.searchsorted(t0, t, side="right") - 1)
            if k < 0 or k >= len(segs):
                continue
            _, _, h5, a5 = segs[k]
            is_away = a.get("teamId") != home_id
            off5 = a5 if is_away else h5
            if pid in off5:
                L.append(list(off5)); S.append(int(pid))
                A.append(int(is_away)); O.append(od)
    L = np.array(L, dtype=np.int64); S = np.array(S, dtype=np.int64)
    A = np.array(A, dtype=np.int8); O = np.array(O, dtype=np.int64)
    np.savez_compressed(SHOT_CACHE, lineup=L, shooter=S, is_away=A, ordinal=O)
    print(f"collected {len(S)} shots ({A.mean():.3f} away)", flush=True)
    return L, S, A, O


# ---------------------------------------------------------- conditional logit
def fit_u(L, S, steps=1500, seed=0):
    """fit_v2_usage.py's model, optimizer, prior and step count, verbatim."""
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoDelta

    pids = {}
    for row in L:
        for p in row:
            if p not in pids:
                pids[p] = len(pids)
    P = len(pids)
    Li = np.array([[pids[p] for p in row] for row in L])
    Si = np.array([list(row).index(s) for row, s in zip(Li, [pids[x] for x in S])])
    Lj, Sj = jnp.asarray(Li), jnp.asarray(Si)

    def model():
        u = numpyro.sample("u", dist.Normal(jnp.zeros(P), 1.5))
        numpyro.sample("s", dist.Categorical(logits=u[Lj]), obs=Sj)

    svi = SVI(model, AutoDelta(model), numpyro.optim.Adam(0.05), Trace_ELBO())
    res = svi.run(jax.random.PRNGKey(seed), steps, progress_bar=False)
    u = np.array(res.params["u_auto_loc"])
    # unconditional smoothed take-rate baseline (fit_v2_usage.py)
    took = np.zeros(P); onfloor = np.zeros(P)
    for row, s in zip(Li, Si):
        for i in row:
            onfloor[i] += 1
        took[row[s]] += 1
    share = (took + 1.0) / (onfloor + 5.0)
    return {int(p): float(u[i]) for p, i in pids.items()}, \
           {int(p): float(share[i]) for p, i in pids.items()}


def heldout_ll(L, S, u, share):
    llc = llu = llb = 0.0; n = 0
    for row, s in zip(L, S):
        if s not in u or any(p not in u for p in row):
            continue
        z = np.array([u[p] for p in row]); z = z - z.max()
        pz = np.exp(z); pz /= pz.sum()
        sh = np.array([share[p] for p in row]); sh = sh / sh.sum()
        i = list(row).index(s)
        llc += np.log(max(pz[i], 1e-9)); llu += np.log(max(sh[i], 1e-9))
        llb += np.log(0.2); n += 1
    return dict(n=n, uniform=-llb / n, uncond_share=-llu / n,
                conditional_logit=-llc / n)


def boot(d, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


# ------------------------------------------------------------------- G1 / G2
def star_frame(con):
    pg = con.execute("""SELECT s.game_id, s.player_id, s.team_id, g.game_date,
        s.seconds/60.0 mins, s.rima+s.mida+s.thra shots, s.pts
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING(game_id)
        WHERE s.game_id LIKE '002%' ORDER BY g.game_date""").fetchdf()
    pg = pg.sort_values(["player_id", "game_date"])
    pg["avg_min"] = pg.groupby("player_id")["mins"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=5).mean())
    pg["avg_shots"] = pg.groupby("player_id")["shots"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=5).mean())
    played = pg[pg.mins >= 8].groupby(["game_id", "team_id"])["player_id"].apply(set)
    stars = pg[pg.avg_min >= 28.0]
    sbt = {}
    for r in stars[["player_id", "team_id", "game_date"]].itertuples():
        sbt.setdefault(r.team_id, []).append((r.game_date, r.player_id))
    return pg, played, sbt


def lift_for(u, pool, star):
    S = sum(np.exp(u.get(p, 0.0)) for p in pool)
    Sx = S - np.exp(u.get(star, 0.0))
    return float(min(S / max(Sx, 1e-9), 1.5))


def g1_attempts(pg, played, sbt, us: dict, cut):
    """D33: Poisson LL of star-out shot counts, softmax lift vs flat 1.020.
    Restricted to POST-CUT games so u is out-of-sample."""
    rot = pg[(pg.avg_min >= 15) & (pg.mins >= 12) & pg.avg_shots.notna()]
    rot = rot[rot.game_date > cut].copy()
    out = {}
    rows = {k: [] for k in us}
    for r in rot.itertuples():
        recent = {p for (d0, p) in sbt.get(r.team_id, [])
                  if 0 < (r.game_date - d0).days <= 12}
        outs = (recent - played.get((r.game_id, r.team_id), set())) - {r.player_id}
        if not outs:
            continue
        pool_base = set(rot[(rot.team_id == r.team_id)
                            & (rot.game_date == r.game_date)].player_id.tolist())
        for k, u in us.items():
            star = max(outs, key=lambda p: u.get(p, 0.0))
            rows[k].append((r.avg_shots, r.shots,
                            lift_for(u, pool_base | {star}, star)))

    def pll(pred, y):
        pred = np.clip(pred, 0.2, None)
        return y * np.log(pred) - pred
    for k, rr in rows.items():
        a = np.array(rr)
        if not len(a):
            continue
        d = pll(a[:, 0] * a[:, 2], a[:, 1]) - pll(a[:, 0] * 1.020, a[:, 1])
        pt, lo, hi = boot(d)
        out[k] = dict(n=len(a), mean_lift=float(a[:, 2].mean()),
                      base_ll=float(pll(a[:, 0], a[:, 1]).mean()),
                      flat_ll=float(pll(a[:, 0] * 1.020, a[:, 1]).mean()),
                      softmax_ll=float(pll(a[:, 0] * a[:, 2], a[:, 1]).mean()),
                      softmax_minus_flat=pt, lo=lo, hi=hi,
                      verdict="KEEP softmax" if lo > 0 else
                              ("softmax WORSE" if hi < 0 else "NS"))
    return out


def crps(samples, y):
    s = np.sort(samples); n = len(s)
    return float(np.mean(np.abs(s - y))
                 - 0.5 * (2 * np.arange(1, n + 1) - n - 1) @ s / n ** 2)


def g2_points(con, pg, played, sbt, us: dict, cut, sims, max_eval):
    """D34/D35: points CRPS of the softmax lift on POST-D79 002-clean rates."""
    rot = pg[(pg.avg_min >= 15) & (pg.mins >= 12)]
    test = rot[rot.game_date > cut]
    base, exp = [], {k: [] for k in us}
    n = 0
    for r in test.itertuples():
        if n >= max_eval:
            break
        recent = {p for (d0, p) in sbt.get(r.team_id, [])
                  if 0 < (r.game_date - d0).days <= 12}
        outs = (recent - played.get((r.game_id, r.team_id), set())) - {r.player_id}
        if not outs:
            continue
        rates = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
        if not rates or rates["n_games"] < 8 or rates["proj_min"] < 15:
            continue
        rn = dict(rates); rn.pop("minutes_hist", None)
        y = r.pts
        base.append(crps(simulate_player(rn, sims, seed=n)["points"], y))
        pool_base = {int(p) for p in rot[(rot.team_id == r.team_id)
                                         & (rot.game_date == r.game_date)].player_id}
        for k, u in us.items():
            star = max(outs, key=lambda p: u.get(p, 0.0))
            lf = lift_for(u, pool_base | {star}, star)
            r2 = dict(rn)
            for key in ("rate_rim", "rate_mid", "rate_thr", "fta_per_min"):
                r2[key] = rn[key] * lf
            exp[k].append(crps(simulate_player(r2, sims, seed=n)["points"], y))
        n += 1
    out = {"n": n, "base_crps": float(np.mean(base))}
    for k, v in exp.items():
        pt, lo, hi = boot(np.array(base) - np.array(v))
        out[k] = dict(crps=float(np.mean(v)), delta=pt, lo=lo, hi=hi,
                      verdict="SHIP" if lo > 0 else
                              ("HARMS" if hi < 0 else "NS"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2500)
    ap.add_argument("--max-eval", type=int, default=900)
    ap.add_argument("--steps", type=int, default=1500)
    a = ap.parse_args()
    t0 = time.time()

    con = connect(read_only=True)
    gd = con.execute("SELECT DISTINCT game_id, game_date FROM nba_games").fetchdf()
    gid_dates = {r.game_id: r.game_date.toordinal() for r in gd.itertuples()}
    cut = con.execute("""SELECT quantile_cont(game_date, 0.6) FROM
        (SELECT DISTINCT game_id, game_date FROM nba_games
         WHERE season='2025-26' AND game_id LIKE '002%')""").fetchone()[0]
    print(f"PIT cut = {cut}", flush=True)

    L, S, A, O = collect(gid_dates)
    cut_ord = cut.toordinal()
    tr = O < cut_ord
    te = ~tr
    print(f"shots: train {tr.sum()} (away-only {int((tr & (A == 1)).sum())})  "
          f"test {te.sum()}  ({time.time()-t0:.0f}s)", flush=True)

    u_fix, sh_fix = fit_u(L[tr], S[tr], steps=a.steps)
    print(f"u_fixed fitted: {len(u_fix)} players  ({time.time()-t0:.0f}s)", flush=True)
    m = tr & (A == 1)
    u_bug, sh_bug = fit_u(L[m], S[m], steps=a.steps)
    print(f"u_buggy fitted: {len(u_bug)} players  ({time.time()-t0:.0f}s)", flush=True)

    common = sorted(set(u_fix) & set(u_bug))
    rho = float(np.corrcoef([u_fix[p] for p in common],
                            [u_bug[p] for p in common])[0, 1])
    res = {"cut": str(cut), "shots_train": int(tr.sum()),
           "shots_train_away_only": int(m.sum()), "shots_test": int(te.sum()),
           "players_fixed": len(u_fix), "players_buggy": len(u_bug),
           "u_corr_fixed_vs_buggy": rho}
    print(f"u corr fixed-vs-buggy: {rho:.4f} on {len(common)} shared players")

    res["G0_shooter_ll_heldout"] = {
        "fixed": heldout_ll(L[te], S[te], u_fix, sh_fix),
        "buggy": heldout_ll(L[te], S[te], u_bug, sh_bug)}
    print("\nG0 held-out shooter log loss (post-cut shots, lower better):")
    for k, v in res["G0_shooter_ll_heldout"].items():
        print(f"  u_{k:<6} n={v['n']}  uniform {v['uniform']:.4f}  "
              f"share {v['uncond_share']:.4f}  COND-LOGIT {v['conditional_logit']:.4f}")

    pg, played, sbt = star_frame(con)
    us = {"fixed": u_fix, "buggy": u_bug}
    res["G1_attempts_poisson"] = g1_attempts(pg, played, sbt, us, cut)
    print("\nG1 (D33) star-out attempts, softmax vs flat x1.020:")
    for k, v in res["G1_attempts_poisson"].items():
        print(f"  u_{k:<6} n={v['n']} lift {v['mean_lift']:.3f}  "
              f"delta {v['softmax_minus_flat']:+.5f} "
              f"CI ({v['lo']:+.5f},{v['hi']:+.5f}) -> {v['verdict']}")

    res["G2_points_crps"] = g2_points(con, pg, played, sbt, us, cut,
                                      a.sims, a.max_eval)
    print("\nG2 (D34/D35) star-out points CRPS on 002-clean rates:")
    g2 = res["G2_points_crps"]
    print(f"  n={g2['n']}  base {g2['base_crps']:.4f}")
    for k in ("fixed", "buggy"):
        v = g2[k]
        print(f"  u_{k:<6} lifted {v['crps']:.4f}  delta {v['delta']:+.4f} "
              f"CI ({v['lo']:+.4f},{v['hi']:+.4f}) -> {v['verdict']}")
    con.close()

    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {OUT}  ({time.time()-t0:.0f}s)")
    print("CG_USAGE_DONE", flush=True)


if __name__ == "__main__":
    main()
