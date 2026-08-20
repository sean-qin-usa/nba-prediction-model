#!/usr/bin/env python3
"""AUDIT (read-only DB): PIT refit of the v2 usage conditional-logit, then a
re-run of the D33 star-out redistribution gate on POST-CUT games only.

Why: data/v2_usage.npz was fit by scripts/fit_v2_usage.py on a RANDOM 70% of
ALL shots (rng.permutation, no date cutoff). Every consumer of that file --
scripts/test_usage_redistribution.py (D33's Poisson-LL test) and
scripts/gate_redistribution_crps.py (the D33 payoff gate) -- therefore scores
"held-out" games with usage propensities u_i that already saw those games'
shots. Both the star pick (argmax_u over the absent set) and the softmax lift
S/(S - exp(u_star)) are functions of that contaminated u.

This script:
  1. reproduces the gate's temporal cut -- cut = pg.game_date.quantile(0.6),
     exactly as computed in gate_redistribution_crps.py:29 over the
     player_game_stats x nba_games join (also reports the alternative
     "quantile 0.6 of star-out rotation player-games" cut as a sensitivity);
  2. re-collects shot events and refits the SAME conditional-logit model
     (copied from fit_v2_usage.py: u_i ~ Normal(0, 1.5), softmax over the five
     on-floor offensive players, numpyro SVI/AutoDelta MAP, Adam(0.05), 1500
     steps, PRNGKey(0)) using ONLY shots from games strictly before the cut;
  3. writes data/v2_usage_pit.npz;
  4. re-runs the D33 construction of test_usage_redistribution.py (star = 28+
     min trailing avg, fresh absence <= 12d, rotation avg_min >= 15 & mins >=
     12, Poisson LL of shot counts under trailing / flat-1.02 / softmax-renorm
     lift) as a 2x2: {contaminated u, PIT u} x {all games, post-cut only};
  5. forensically reconstructs the original random 70/30 split to measure how
     many post-cut shots were literally in the contaminated fit's training set.

Reporting: pooled + per-season, paired bootstrap 2000x 95% CI. Because the unit
is a player-game the primary CI is CLUSTERED BY PLAYER; the iid-row CI the
original test printed is shown alongside for comparability.

Stages (env STAGE=collect|fit|gate|leak|all, default all) exist only so each
step fits in a foreground call; every stage caches to AUDIT_SHOT_CACHE's dir.

Writes: data/v2_usage_pit.npz, data/logs/audit_usage_pit.json. Touches nothing
else. DB is opened read_only.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import orjson
import pandas as pd

from nbapred.db import connect
from nbapred.features.cache_index import game_index
from nbapred.features.defense_zone import _game_segments
from nbapred.features.possessions_v2 import _team_ids
from nbapred.features.stints import _elapsed

SCRATCH = Path(os.environ.get(
    "AUDIT_SCRATCH",
    "data/scratch"))
SHOT_CACHE = SCRATCH / "audit_usage_shots.npz"
ROW_CACHE = SCRATCH / "audit_usage_rows.pkl"
NBOOT = 2000


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- shot events
def _game_shots(job):
    """EXACT event logic of fit_v2_usage.collect(), one game at a time."""
    gid, pbp_path, rot_path = job
    try:
        pbp = orjson.loads(open(pbp_path, "rb").read())["response"]
        rot = orjson.loads(open(rot_path, "rb").read())["response"]
    except Exception:
        return gid, []
    # D100: cached playbyplayv3 `game` has NO homeTeamId -> always None, so the
    # off5 pick below returned the AWAY five for every shot and the `pid in off5`
    # guard silently dropped every home-team shot (49.9% loss). Fixed to match
    # D81's rotation-feed derivation; fit_v2_usage.collect() carries the same fix.
    home, _away = _team_ids(rot, pbp)
    if home is None:
        return gid, []
    segs = _game_segments(rot, pbp)
    if not segs:
        return gid, []
    t0 = np.array([s[0] for s in segs])
    out = []
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
        off5 = h5 if a.get("teamId") == home else a5
        if pid in off5:
            out.append((tuple(off5), int(pid)))
    return gid, out


def collect_shots():
    """-> (game_ids[N], lineups[N,5], shooters[N]) over every cached game, in
    the same sorted-game_id order fit_v2_usage.collect() used."""
    if SHOT_CACHE.exists():
        z = np.load(SHOT_CACHE, allow_pickle=True)
        log(f"shot cache hit: {len(z['shooter'])} shots")
        return z["gid"], z["lineup"], z["shooter"]
    t0 = time.time()
    rots = game_index("gamerotation")
    pbps = game_index("playbyplayv3")
    gids = sorted(set(rots) & set(pbps))
    jobs = [(g, pbps[g], rots[g]) for g in gids]
    G, L, S = [], [], []
    with ThreadPoolExecutor(max_workers=12) as ex:
        for i, (gid, shots) in enumerate(ex.map(_game_shots, jobs)):
            for off5, sh in shots:
                G.append(gid)
                L.append(off5)
                S.append(sh)
            if (i + 1) % 500 == 0:
                log(f"  {i+1}/{len(jobs)} games  {len(S)} shots  {time.time()-t0:.0f}s")
    G = np.array(G)
    L = np.array(L, dtype=np.int64)
    S = np.array(S, dtype=np.int64)
    SHOT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(SHOT_CACHE, gid=G, lineup=L, shooter=S)
    log(f"collected {len(S)} shots from {len(gids)} games in {time.time()-t0:.0f}s")
    return G, L, S


# ------------------------------------------------------------ conditional logit
def fit_logit(lineups, shooters, steps=1500, seed=0):
    """Model copied verbatim from scripts/fit_v2_usage.py (MAP via AutoDelta)."""
    pids = {}
    for row in lineups:
        for p in row:
            pids.setdefault(int(p), len(pids))
    P = len(pids)
    L = np.vectorize(pids.get)(lineups)
    S = np.array([list(r).index(int(s)) for r, s in zip(lineups, shooters)])
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoDelta

    Lj = jnp.asarray(L)
    Sj = jnp.asarray(S)

    def model():
        u = numpyro.sample("u", dist.Normal(jnp.zeros(P), 1.5))
        logits = u[Lj]
        numpyro.sample("s", dist.Categorical(logits=logits), obs=Sj)

    guide = AutoDelta(model)
    svi = SVI(model, guide, numpyro.optim.Adam(0.05), Trace_ELBO())
    res = svi.run(jax.random.PRNGKey(seed), steps, progress_bar=False)
    u = np.array(res.params["u_auto_loc"])
    # unconditional on-floor take-rate baseline (same smoothing as fit_v2_usage)
    took = np.zeros(P)
    onfloor = np.zeros(P)
    np.add.at(onfloor, L.ravel(), 1.0)
    np.add.at(took, np.array([pids[int(s)] for s in shooters]), 1.0)
    share = (took + 1.0) / (onfloor + 5.0)
    return pids, u, share


def shooter_ll(lineups, shooters, pids, u, share):
    """Held-out shooter log loss: uniform / uncond share / conditional logit."""
    ids = np.array([[pids.get(int(p), -1) for p in r] for r in lineups])
    tgt = np.array([pids.get(int(s), -1) for s in shooters])
    ok = (ids >= 0).all(1) & (tgt >= 0)
    ids, tgt = ids[ok], tgt[ok]
    if len(tgt) == 0:
        return None
    si = np.argmax(ids == tgt[:, None], axis=1)
    z = u[ids]
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(1, keepdims=True)
    sh = share[ids]
    sh = sh / sh.sum(1, keepdims=True)
    r = np.arange(len(tgt))
    return {"n": int(len(tgt)), "skipped": int((~ok).sum()),
            "uniform": float(-np.log(0.2)),
            "uncond_share": float(-np.log(np.clip(sh[r, si], 1e-9, None)).mean()),
            "cond_logit": float(-np.log(np.clip(p[r, si], 1e-9, None)).mean())}


# -------------------------------------------------------------- D33 construction
def build_frames(con):
    """test_usage_redistribution.py construction, verbatim semantics."""
    pg = con.execute("""SELECT s.game_id, s.player_id, s.team_id, g.game_date,
        s.seconds/60.0 mins, s.rima+s.mida+s.thra shots
        FROM player_game_stats s JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING(game_id)
        WHERE s.game_id LIKE '002%' ORDER BY g.game_date""").fetchdf()
    seas = con.execute("""SELECT DISTINCT game_id, season FROM nba_games
        WHERE game_id LIKE '002%'""").fetchdf()
    cut = pg.game_date.quantile(0.6)          # <- gate_redistribution_crps.py:29
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
    rot = pg[(pg.avg_min >= 15) & (pg.mins >= 12) & pg.avg_shots.notna()].copy()
    rot = rot.merge(seas, on="game_id", how="left")
    return pg, rot, played, sbt, cut


def star_out_table(rot, played, sbt):
    """One row per (rotation player-game) that has a fresh star absence.
    Identical membership to test_usage_redistribution.py's inner loop; the
    per-(team, date) 'recent stars' set is computed once instead of per row."""
    recent_by = {}
    for r in rot[["team_id", "game_date"]].drop_duplicates().itertuples():
        recent_by[(r.team_id, r.game_date)] = {
            int(p) for (d0, p) in sbt.get(r.team_id, [])
            if 0 < (r.game_date - d0).days <= 12}
    pool_by = rot.groupby(["team_id", "game_date"])["player_id"].apply(
        lambda s: [int(x) for x in s]).to_dict()
    recs = []
    for r in rot.itertuples():
        recent = recent_by[(r.team_id, r.game_date)]
        outs = (recent - {int(p) for p in played.get((r.game_id, r.team_id), set())}
                ) - {int(r.player_id)}
        if not outs:
            continue
        recs.append((r.game_id, int(r.player_id), r.team_id, r.game_date, r.season,
                     float(r.avg_shots), int(r.shots), sorted(outs),
                     pool_by.get((r.team_id, r.game_date), [])))
    return pd.DataFrame(recs, columns=["game_id", "player_id", "team_id", "game_date",
                                       "season", "avg_shots", "shots", "outs", "pool"])


def add_lift(so, u):
    """star = argmax_u(outs); lift = min(S/(S-exp(u_star)), 1.5)."""
    exp_cache = {}

    def eu(p):
        if p not in exp_cache:
            exp_cache[p] = float(np.exp(u.get(p, 0.0)))
        return exp_cache[p]

    stars, lifts, miss = [], [], 0
    for outs, pool in zip(so.outs.values, so.pool.values):
        star = max(outs, key=lambda p: u.get(p, 0.0))
        if star not in u:
            miss += 1
        P = set(pool) | {star}
        S = sum(eu(p) for p in P)
        Sx = S - eu(star)
        stars.append(star)
        lifts.append(min(S / max(Sx, 1e-9), 1.5))
    out = so.copy()
    out["star"] = stars
    out["lift"] = lifts
    out.attrs["star_missing_u"] = miss
    return out


# ------------------------------------------------------------------- scoring
def pois_ll_vec(pred, y):
    return y * np.log(np.clip(pred, 0.2, None)) - np.clip(pred, 0.2, None)


def boot_ci(d, groups=None, nboot=NBOOT, seed=0):
    """Paired bootstrap of mean(d). groups=None -> iid rows; else cluster."""
    rng = np.random.default_rng(seed)
    if groups is None:
        n = len(d)
        idx = rng.integers(0, n, size=(nboot, n))
        b = d[idx].mean(1)
    else:
        _, inv = np.unique(groups, return_inverse=True)
        G = inv.max() + 1
        s = np.bincount(inv, weights=d, minlength=G)
        c = np.bincount(inv, minlength=G).astype(float)
        pick = rng.integers(0, G, size=(nboot, G))
        b = s[pick].sum(1) / c[pick].sum(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return [float(lo), float(hi)]


def score(df, label):
    y = df.shots.to_numpy(float)
    base = df.avg_shots.to_numpy(float)
    lift = df.lift.to_numpy(float)
    base_v = pois_ll_vec(base, y)
    flat_v = pois_ll_vec(base * 1.020, y)
    soft_v = pois_ll_vec(base * lift, y)
    d_sf = soft_v - flat_v
    d_sb = soft_v - base_v
    pid = df.player_id.to_numpy()
    out = {"label": label, "n": int(len(df)), "n_players": int(df.player_id.nunique()),
           "mean_lift": float(lift.mean()), "frac_capped": float((lift >= 1.4999).mean()),
           "ll_base": float(base_v.mean()), "ll_flat": float(flat_v.mean()),
           "ll_soft": float(soft_v.mean()),
           "d_soft_vs_flat": float(d_sf.mean()), "d_soft_vs_base": float(d_sb.mean()),
           "ci_iid": boot_ci(d_sf), "ci_cluster": boot_ci(d_sf, pid),
           "ci_cluster_soft_vs_base": boot_ci(d_sb, pid), "by_season": {}}
    for s, g in df.groupby("season"):
        gy = g.shots.to_numpy(float)
        gb = g.avg_shots.to_numpy(float)
        gl = g.lift.to_numpy(float)
        dv = pois_ll_vec(gb * gl, gy) - pois_ll_vec(gb * 1.020, gy)
        out["by_season"][str(s)] = {
            "n": int(len(g)), "n_players": int(g.player_id.nunique()),
            "mean_lift": float(gl.mean()), "d_soft_vs_flat": float(dv.mean()),
            "ci_cluster": boot_ci(dv, g.player_id.to_numpy())}
    return out


def show(r):
    log(f"\n=== {r['label']} ===")
    log(f"  n={r['n']} player-games / {r['n_players']} players | mean softmax lift "
        f"{r['mean_lift']:.3f} (flat=1.020) | capped@1.5 {100*r['frac_capped']:.1f}%")
    log(f"  Poisson LL  base {r['ll_base']:.5f}  flat {r['ll_flat']:.5f}  softmax {r['ll_soft']:.5f}")
    ci, cc = r["ci_iid"], r["ci_cluster"]
    verdict = "KEEP softmax" if cc[0] > 0 else ("NS" if cc[1] > 0 else "softmax WORSE")
    log(f"  softmax-vs-flat {r['d_soft_vs_flat']:+.5f}  iid CI ({ci[0]:+.5f},{ci[1]:+.5f})  "
        f"PLAYER-CLUSTER CI ({cc[0]:+.5f},{cc[1]:+.5f}) -> {verdict}")
    cb = r["ci_cluster_soft_vs_base"]
    log(f"  softmax-vs-baseline {r['d_soft_vs_base']:+.5f}  cluster CI ({cb[0]:+.5f},{cb[1]:+.5f})")
    for s, v in sorted(r["by_season"].items()):
        log(f"    {s}: n={v['n']:6d} ({v['n_players']:3d} plyr) lift {v['mean_lift']:.3f} "
            f"d {v['d_soft_vs_flat']:+.5f} CI ({v['ci_cluster'][0]:+.5f},{v['ci_cluster'][1]:+.5f})")


# ---------------------------------------------------------------------- stages
def stage_collect():
    collect_shots()


def stage_rows():
    """DB-side D33 star-out table + the cut, cached for the gate stage."""
    if ROW_CACHE.exists():
        return pd.read_pickle(ROW_CACHE)
    con = connect(read_only=True)
    pg, rot, played, sbt, cut = build_frames(con)
    con.close()
    log(f"pg rows {len(pg)}  games {pg.game_id.nunique()}  rot rows {len(rot)}")
    log(f"gate cut  pg.game_date.quantile(0.6) = {cut.date()}")
    so = star_out_table(rot, played, sbt)
    alt_cut = so.game_date.quantile(0.6)
    log(f"alt cut (q0.6 of star-out rotation player-games) = {alt_cut.date()}")
    so.attrs["cut"] = cut
    so.attrs["alt_cut"] = alt_cut
    so.to_pickle(ROW_CACHE)
    return so


def stage_fit(so):
    cut = so.attrs["cut"]
    con = connect(read_only=True)
    gd = con.execute("""SELECT DISTINCT game_id, game_date FROM nba_games
        WHERE game_id LIKE '002%'""").fetchdf()
    con.close()
    G, L, S = collect_shots()
    sdate = pd.to_datetime(pd.Series(G).map(dict(zip(gd.game_id, gd.game_date))))
    pre = (sdate < cut).to_numpy()
    post = (sdate > cut).to_numpy()
    log(f"shots {len(S)} | pre-cut {pre.sum()} | post-cut {post.sum()} | "
        f"undated {int(sdate.isna().sum())}")
    t1 = time.time()
    pids, u_arr, share = fit_logit(L[pre], S[pre])
    log(f"PIT fit: {len(pids)} players on {int(pre.sum())} pre-cut shots, {time.time()-t1:.0f}s")
    np.savez(ROOT / "data/v2_usage_pit.npz",
             player_ids=np.array(sorted(pids, key=pids.get)), u=u_arr,
             cut=np.array([str(cut.date())]), n_train_shots=np.array([int(pre.sum())]))
    log(f"wrote {ROOT/'data/v2_usage_pit.npz'}")
    ll = shooter_ll(L[post], S[post], pids, u_arr, share)
    log(f"POST-CUT shooter log loss with PIT u (n={ll['n']}, skipped {ll['skipped']}): "
        f"uniform {ll['uniform']:.4f}  uncond {ll['uncond_share']:.4f}  COND {ll['cond_logit']:.4f}")
    return ll


def stage_leak():
    """Forensics: rebuild fit_v2_usage.py's random split over the games that
    were cached when it ran, and measure post-cut contamination directly."""
    con = connect(read_only=True)
    gd = con.execute("""SELECT DISTINCT game_id, game_date FROM nba_games
        WHERE game_id LIKE '002%'""").fetchdf()
    con.close()
    dmap = dict(zip(gd.game_id, gd.game_date))
    so = stage_rows()
    cut = so.attrs["cut"]
    import glob
    idx = json.loads((ROOT / "data/raw/nba_api/gamerotation/_index.json").read_bytes())
    fit_mtime = (ROOT / "data/v2_usage.npz").stat().st_mtime
    old = {idx[os.path.basename(f)] for f in glob.glob(str(ROOT / "data/raw/nba_api/gamerotation/*.json"))
           if os.path.getmtime(f) < fit_mtime and idx.get(os.path.basename(f))}
    G, L, S = collect_shots()
    m = np.isin(G, list(old))
    n = int(m.sum())
    rng = np.random.default_rng(0)     # fit_v2_usage.py seed
    perm = rng.permutation(n)
    train = perm[:int(n * 0.7)]
    gsub = G[m]
    tr_dates = pd.to_datetime(pd.Series(gsub[train]).map(dmap))
    log(f"\n--- leak forensics (original fit) ---")
    log(f"games cached at fit time: {len(old)}; shots re-collected from them: {n} "
        f"(fit_v2_usage.log recorded 188328)")
    log(f"original TRAIN shots: {len(train)}; of those, from POST-cut games: "
        f"{int((tr_dates > cut).sum())} ({100*(tr_dates>cut).mean():.1f}%)")
    post_games = set(gsub[train][(tr_dates > cut).to_numpy()])
    so_post = so[so.game_date > cut]
    log(f"post-cut games with shots in the original TRAIN set: {len(post_games)}")
    log(f"D33 post-cut star-out player-games whose game_id is in that train set: "
        f"{int(so_post.game_id.isin(post_games).sum())}/{len(so_post)} "
        f"({100*so_post.game_id.isin(post_games).mean():.1f}%)")
    return {"fit_games": len(old), "recollected_shots": n,
            "train_shots": len(train), "train_shots_postcut": int((tr_dates > cut).sum()),
            "postcut_games_in_train": len(post_games),
            "d33_postcut_rows_in_train_games": int(so_post.game_id.isin(post_games).sum()),
            "d33_postcut_rows": int(len(so_post))}


def stage_gate(ll=None, leak=None):
    so = stage_rows()
    cut = so.attrs["cut"]
    uz0 = np.load(ROOT / "data/v2_usage.npz")
    u_old = {int(k): float(v) for k, v in zip(uz0["player_ids"].tolist(), uz0["u"].tolist())}
    uz1 = np.load(ROOT / "data/v2_usage_pit.npz")
    u_pit = {int(k): float(v) for k, v in zip(uz1["player_ids"].tolist(), uz1["u"].tolist())}
    old_all = add_lift(so, u_old)
    pit_all = add_lift(so, u_pit)
    post = so.game_date > cut
    res = {}
    res["A_contaminated_allgames"] = score(old_all, "A  contaminated u, ALL games (published D33)")
    res["B_contaminated_postcut"] = score(old_all[post.values], "B  contaminated u, POST-CUT only")
    res["C_pit_postcut"] = score(pit_all[post.values], "C  PIT u, POST-CUT only   <-- HONEST TEST")
    res["D_pit_allgames"] = score(pit_all, "D  PIT u, ALL games")
    for k in ("A_contaminated_allgames", "B_contaminated_postcut", "C_pit_postcut", "D_pit_allgames"):
        show(res[k])
    common = sorted(set(u_old) & set(u_pit))
    a = np.array([u_old[p] for p in common])
    b = np.array([u_pit[p] for p in common])
    corr = float(np.corrcoef(a, b)[0, 1])
    log(f"\nu overlap {len(common)} players (contaminated {len(u_old)}, PIT {len(u_pit)}); "
        f"corr {corr:.3f}  mean|diff| {np.abs(a-b).mean():.3f}")
    sp = so[post.values]
    log(f"post-cut star coverage: contaminated u has {100*np.mean([s in u_old for s in old_all[post.values].star]):.1f}% "
        f"of picked stars, PIT u has {100*np.mean([s in u_pit for s in pit_all[post.values].star]):.1f}%")
    logp = ROOT / "data/logs/audit_usage_pit.json"
    logp.parent.mkdir(parents=True, exist_ok=True)
    logp.write_text(json.dumps({
        "cut": str(so.attrs["cut"].date()), "alt_cut_starout": str(so.attrs["alt_cut"].date()),
        "postcut_shooter_ll_pit": ll, "leak_forensics": leak,
        "u_corr_old_vs_pit": corr, "arms": res}, indent=2, default=str))
    log(f"\nwrote {logp}")
    return res


def stage_extra():
    """Attribution: D33's claim is that the FITTED usage model prices star-out
    redistribution better than folklore. flat-1.02 is a weak strawman -- it is
    the wrong MAGNITUDE, not the wrong shape. These arms separate the two:
      c_hat   : in-sample-OPTIMAL constant lift (Poisson MLE, sum y / sum base)
                -- a steelman folklore that only knows "shots go up by c"
      c_mean  : constant lift equal to the mean softmax lift
      shuffled: softmax lifts permuted across rows (same marginal distribution,
                cross-sectional pairing destroyed)
      null_u  : lift computed with u == 0 for everyone (pure pool arithmetic)
    If softmax does not beat c_hat / shuffled, the win is a recalibrated
    constant, not the usage model."""
    so = stage_rows()
    cut = so.attrs["cut"]
    uz1 = np.load(ROOT / "data/v2_usage_pit.npz")
    u_pit = {int(k): float(v) for k, v in zip(uz1["player_ids"].tolist(), uz1["u"].tolist())}
    df = add_lift(so, u_pit)
    df = df[(so.game_date > cut).values]
    y = df.shots.to_numpy(float)
    base = df.avg_shots.to_numpy(float)
    lift = df.lift.to_numpy(float)
    pid = df.player_id.to_numpy()
    c_hat = float(y.sum() / base.sum())
    c_mean = float(lift.mean())
    null = add_lift(so, {})
    null = null[(so.game_date > cut).values]
    lift0 = null.lift.to_numpy(float)
    soft_v = pois_ll_vec(base * lift, y)
    log(f"\n--- D33 attribution on POST-CUT rows (PIT u), n={len(df)}, "
        f"{df.player_id.nunique()} players ---")
    log(f"  in-sample-optimal constant lift c_hat = {c_hat:.3f} | mean softmax lift "
        f"{c_mean:.3f} | flat folklore 1.020 | null-u (pool arithmetic only) {lift0.mean():.3f}")
    out = {"c_hat": c_hat, "c_mean": c_mean, "n": int(len(df)),
           "mean_lift_null_u": float(lift0.mean())}
    for name, pred in (("flat_1.020", base * 1.020), ("c_mean", base * c_mean),
                       ("c_hat_optimal", base * c_hat), ("null_u_lift", base * lift0)):
        d = soft_v - pois_ll_vec(pred, y)
        ci = boot_ci(d, pid)
        v = "softmax WINS" if ci[0] > 0 else ("NS" if ci[1] > 0 else "softmax LOSES")
        log(f"  softmax vs {name:14s}: {d.mean():+.5f}  cluster CI ({ci[0]:+.5f},{ci[1]:+.5f})  -> {v}")
        out[f"vs_{name}"] = {"delta": float(d.mean()), "ci_cluster": ci}
    # shuffled lifts: same marginal, pairing destroyed
    ds = []
    for s in range(20):
        rs = np.random.default_rng(100 + s)
        d = soft_v - pois_ll_vec(base * rs.permutation(lift), y)
        ds.append(d.mean())
    rs = np.random.default_rng(100)
    d = soft_v - pois_ll_vec(base * rs.permutation(lift), y)
    ci = boot_ci(d, pid)
    log(f"  softmax vs shuffled_lift : {np.mean(ds):+.5f} (mean of 20 perms; seed-100 perm "
        f"{d.mean():+.5f} cluster CI ({ci[0]:+.5f},{ci[1]:+.5f}))")
    out["vs_shuffled_lift"] = {"delta_mean20": float(np.mean(ds)),
                               "delta_seed100": float(d.mean()), "ci_cluster": ci}
    # per-season vs the steelman
    out["by_season_vs_c_hat"] = {}
    for s, g in df.groupby("season"):
        gy = g.shots.to_numpy(float)
        gb = g.avg_shots.to_numpy(float)
        gl = g.lift.to_numpy(float)
        d = pois_ll_vec(gb * gl, gy) - pois_ll_vec(gb * c_hat, gy)
        ci = boot_ci(d, g.player_id.to_numpy())
        log(f"    {s}: n={len(g):6d} softmax-vs-c_hat {d.mean():+.5f} CI ({ci[0]:+.5f},{ci[1]:+.5f})")
        out["by_season_vs_c_hat"][str(s)] = {"n": int(len(g)), "delta": float(d.mean()),
                                             "ci_cluster": ci}
    p = ROOT / "data/logs/audit_usage_pit.json"
    if p.exists():
        j = json.loads(p.read_text())
        j["attribution"] = out
        p.write_text(json.dumps(j, indent=2, default=str))
    return out


def stage_control():
    """Confound check: D33's rotation universe requires mins >= 12 IN THE GAME
    BEING SCORED, while the baseline is a trailing 10-game shot average that
    includes the player's low-minute nights. That selection alone inflates
    actual/trailing. Measure the same optimal constant lift on rotation
    player-games with NO fresh star absence -- the control group."""
    CTRL = SCRATCH / "audit_usage_rows_all.pkl"
    if CTRL.exists():
        allrot = pd.read_pickle(CTRL)
    else:
        con = connect(read_only=True)
        pg, rot, played, sbt, cut = build_frames(con)
        con.close()
        recent_by = {}
        for r in rot[["team_id", "game_date"]].drop_duplicates().itertuples():
            recent_by[(r.team_id, r.game_date)] = {
                int(p) for (d0, p) in sbt.get(r.team_id, [])
                if 0 < (r.game_date - d0).days <= 12}
        has = []
        for r in rot.itertuples():
            outs = (recent_by[(r.team_id, r.game_date)]
                    - {int(p) for p in played.get((r.game_id, r.team_id), set())}
                    ) - {int(r.player_id)}
            has.append(bool(outs))
        allrot = rot[["game_id", "player_id", "game_date", "season", "avg_shots",
                      "shots", "mins"]].copy()
        allrot["has_star_out"] = has
        allrot.attrs["cut"] = cut
        allrot.to_pickle(CTRL)
    cut = allrot.attrs["cut"]
    post = allrot[allrot.game_date > cut]
    log(f"\n--- selection confound: optimal constant lift, POST-CUT rotation player-games ---")
    for lab, g in (("star-out    ", post[post.has_star_out]),
                   ("NO star-out ", post[~post.has_star_out])):
        c = g.shots.sum() / g.avg_shots.sum()
        log(f"  {lab}: n={len(g):6d}  sum(shots)/sum(trailing_avg) = {c:.4f}  "
            f"mean mins {g.mins.mean():.1f}")
    for s in sorted(post.season.dropna().unique()):
        ps = post[post.season == s]
        a = ps[ps.has_star_out]
        b = ps[~ps.has_star_out]
        log(f"    {s}: star-out {a.shots.sum()/a.avg_shots.sum():.4f} (n={len(a)})  "
            f"control {b.shots.sum()/b.avg_shots.sum():.4f} (n={len(b)})")
    out = {}
    for lab, g in (("star_out", post[post.has_star_out]),
                   ("control", post[~post.has_star_out])):
        out[lab] = {"n": int(len(g)), "c_hat": float(g.shots.sum() / g.avg_shots.sum())}
    out["excess_attributable_to_star_out"] = out["star_out"]["c_hat"] - out["control"]["c_hat"]
    # fully-PIT rival: constant lift estimated on PRE-cut star-out games only
    pre = allrot[(allrot.game_date < cut) & allrot.has_star_out]
    c_pre = float(pre.shots.sum() / pre.avg_shots.sum())
    so = stage_rows()
    uz1 = np.load(ROOT / "data/v2_usage_pit.npz")
    u_pit = {int(k): float(v) for k, v in zip(uz1["player_ids"].tolist(), uz1["u"].tolist())}
    df = add_lift(so, u_pit)
    df = df[(so.game_date > cut).values]
    y = df.shots.to_numpy(float)
    b = df.avg_shots.to_numpy(float)
    d = pois_ll_vec(b * df.lift.to_numpy(float), y) - pois_ll_vec(b * c_pre, y)
    ci = boot_ci(d, df.player_id.to_numpy())
    log(f"  PIT constant lift from pre-cut star-out games c_pre={c_pre:.4f} (n={len(pre)}); "
        f"softmax vs c_pre {d.mean():+.5f} cluster CI ({ci[0]:+.5f},{ci[1]:+.5f}) -> "
        f"{'softmax WINS' if ci[0]>0 else ('NS' if ci[1]>0 else 'softmax LOSES')}")
    out["c_pre_pit_constant"] = {"c_pre": c_pre, "n_pre": int(len(pre)),
                                 "softmax_vs_c_pre": float(d.mean()), "ci_cluster": ci}
    log(f"  => lift attributable to the star absence (difference): "
        f"{out['excess_attributable_to_star_out']:+.4f}  "
        f"(softmax prices it at {1.195-1.0:+.3f})")
    p = ROOT / "data/logs/audit_usage_pit.json"
    if p.exists():
        j = json.loads(p.read_text())
        j["selection_confound"] = out
        p.write_text(json.dumps(j, indent=2, default=str))
    return out


def main():
    stage = os.environ.get("STAGE", "all")
    t0 = time.time()
    if stage in ("collect", "all"):
        stage_collect()
    if stage in ("rows", "all"):
        stage_rows()
    ll = leak = None
    if stage in ("fit", "all"):
        ll = stage_fit(stage_rows())
    if stage in ("leak", "all"):
        leak = stage_leak()
    if stage in ("gate", "all"):
        if ll is None and (ROOT / "data/logs/audit_usage_pit.json").exists():
            prev = json.loads((ROOT / "data/logs/audit_usage_pit.json").read_text())
            ll = ll or prev.get("postcut_shooter_ll_pit")
            leak = leak or prev.get("leak_forensics")
        stage_gate(ll, leak)
    if stage in ("extra", "all"):
        stage_extra()
    if stage in ("control", "all"):
        stage_control()
    log(f"AUDIT_DONE stage={stage} {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
