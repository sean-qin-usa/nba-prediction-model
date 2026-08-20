#!/usr/bin/env python3
"""GAMEROTATION ROLE-TRANSITION MINUTES CORRECTION — pre-registered gate.

PRE-REGISTRATION: data/avail_depth_prereg.md
  sha256 4f7356e79b38521180e6792fbdf2f4faa997550cb4a4d11cf724f174c901a6d4
Nothing here may deviate from it.

Harness is D133's (scripts/pr_ramp_gate.py): same CRPS, same randomized PIT,
same paired same-seed MC draws, same 4,000 sims/row, same eval-universe
conditioning (002, seconds>=720, n_games>=8, proj_min>=20), same
cluster-bootstrap-by-player — so the numbers are directly comparable to D133's.

ARMS (control = production props.py verbatim, same run):
  ctrl  player_rates_from_stats + simulate_player, untouched
  R     proj_min -= b_R(role bucket)      GameRotation starter flag   (PRIMARY)
  M     proj_min -= b_M(minutes bucket)   minutes-only adversarial CONTROL

usage: ad_role_gate.py [dev|holdout] [--max-inactive N]
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from nbapred.db import connect
from nbapred.engine.props import (player_rates_from_stats, simulate_player,
                                  minutes_ramp)

DEV = ("2023-24", "2024-25", "2025-26")
HOLDOUT = ("2021-22", "2022-23")
HL = 10.0
SIMS = 4000
NBOOT = 2000
BOOT_SEED = 20260801
ARMS = ("ctrl", "R", "M")
PREREG = "4f7356e79b38521180e6792fbdf2f4faa997550cb4a4d11cf724f174c901a6d4"


# --------------------------------------------------------------- sim replica
def simulate_ad(rates: dict, n: int, seed: int, proj_over=None):
    """VERBATIM copy of props.simulate_player's generative model with one hook
    (proj_over replaces proj_min in the MINUTES draw only). With proj_over=None
    it must reproduce simulate_player BITWISE; main() asserts that before
    scoring anything (GATE_POLICY §6.6 implementation identity)."""
    rng = np.random.default_rng(seed)
    hist = rates.get("minutes_hist")
    pm = float(rates.get("proj_min", 30.0)) if proj_over is None else float(proj_over)
    if hist is not None and len(hist) >= 5:
        h = np.asarray(hist, float)
        base = rng.choice(h, n) - h.mean() + pm
        mins = np.clip(base + rng.normal(0, 2.0, n), 0, 48)
    else:
        sd = rates.get("sd_min", 4.0)
        mins = rng.normal(pm, sd, n)
        bad = mins < 10
        for _ in range(4):
            if not bad.any():
                break
            mins[bad] = rng.normal(pm, sd, bad.sum())
            bad = mins < 10
        mins = np.clip(mins, 10, 48)

    def zone_pts(rate, fg, val):
        att = rng.poisson(np.maximum(rate, 0) * mins)
        made = rng.binomial(att, np.clip(fg, 0, 1))
        return made, made * val

    rim_m, rim_p = zone_pts(rates["rate_rim"], rates["fg_rim"], 2)
    mid_m, mid_p = zone_pts(rates["rate_mid"], rates["fg_mid"], 2)
    thr_m, thr_p = zone_pts(rates["rate_thr"], rates["fg_thr"], 3)
    ft_att = rng.poisson(np.maximum(rates.get("fta_per_min", 0), 0) * mins)
    ft_m = rng.binomial(ft_att, np.clip(rates.get("ft_pct", 0.77), 0, 1))
    return rim_p + mid_p + thr_p + ft_m, mins


# ------------------------------------------------------------------ scoring
def crps(samples, y):
    s = np.sort(samples); n = len(s)
    return float(np.mean(np.abs(s - y)) - 0.5 * (2 * np.arange(1, n + 1) - n - 1) @ s / n ** 2)


def pit(samples, y, rng):
    lo = float(np.mean(samples < y))
    ties = float(np.mean(samples == y))
    return lo + rng.uniform() * ties


def cluster_boot(deltas, players, iters=NBOOT, seed=BOOT_SEED):
    uniq, inv = np.unique(players, return_inverse=True)
    sums = np.bincount(inv, weights=deltas, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), (iters, len(uniq)))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(np.mean(deltas)), float(lo), float(hi), float(means.std(ddof=1)))


# ------------------------------------------------------------- corpus / role
def load_corpus(con):
    df = con.execute("""
        SELECT s.player_id, s.team_id, g.season, g.game_date, s.seconds/60.0 AS mins,
               s.pts
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '002%' AND s.seconds >= 720
        ORDER BY s.player_id, g.game_date
    """).fetchdf()
    df["ord"] = df["game_date"].astype("datetime64[ns]").values.astype(
        "datetime64[D]").astype(int)
    return df


def load_roles():
    z = np.load(ROOT / "data" / "ad_role_flags.npz")
    return {(int(p), int(o)): int(s)
            for p, o, s in zip(z["player_id"], z["ord"], z["starter"])}


def build_index(df, roles):
    byp = {}
    for pid, sub in df.groupby("player_id", sort=False):
        ords = sub["ord"].to_numpy()
        mins = sub["mins"].to_numpy(float)
        seas = sub["season"].to_numpy(object)
        star = np.array([roles.get((int(pid), int(o)), np.nan) for o in ords], float)
        byp[int(pid)] = (ords, mins, seas, star)
    return byp


def row_state(byp, pid, day, season):
    """(proj, n_hist, gp, role_bucket, minutes_bucket) as of `day`, using ONLY
    games strictly before `day`.  PIT: i = searchsorted(ords, day) is the count
    of strictly-prior games, and every array is sliced [:i]."""
    ords, mins, seas, star = byp[pid]
    i = int(np.searchsorted(ords, day))
    if i < 3:
        return None
    assert i == 0 or ords[i - 1] < day, "PIT VIOLATION: history includes the scored day"
    w = 0.5 ** (np.arange(i)[::-1] / HL)
    proj_raw = float(np.sum(w * mins[:i]) / np.sum(w))
    gp = int((seas[:i] == season).sum())
    proj = max(proj_raw - minutes_ramp(gp), 0.0)

    # --- ARM R state: GameRotation starter flag
    hs = star[:i]
    cov = np.where(~np.isnan(hs))[0]
    rb = "NA"
    if len(cov) >= 5 and cov[-1] == i - 1:            # gap == 0 guard
        sr_last = hs[i - 1]
        sr5 = float(np.mean(hs[cov[-5:]]))
        if sr_last == 1.0 and sr5 < 0.5:
            rb = "PROMOTED"
        elif sr_last == 0.0 and sr5 > 0.5:
            rb = "DEMOTED"
        else:
            rb = "STABLE"

    # --- ARM M state: minutes-only analogue, no rotation input
    med = float(np.median(mins[max(0, i - 20):i]))
    ml_last = 1.0 if mins[i - 1] >= med else 0.0
    ml5 = float(np.mean(mins[max(0, i - 5):i] >= med))
    if ml_last == 1.0 and ml5 < 0.5:
        mb = "PROMOTED"
    elif ml_last == 0.0 and ml5 > 0.5:
        mb = "DEMOTED"
    else:
        mb = "STABLE"
    return proj, i, gp, rb, mb


def fit_bias(df, byp, fit_seasons):
    """Walk-forward b(bucket) tables from `fit_seasons` only (strictly prior)."""
    accR, accM = {}, {}
    sub = df[df["season"].isin(fit_seasons)]
    for r in sub.itertuples():
        st = row_state(byp, int(r.player_id), r.ord, r.season)
        if st is None:
            continue
        proj, nh, gp, rb, mb = st
        if nh < 8 or proj < 20:
            continue
        b = proj - float(r.mins)
        accR.setdefault(rb, []).append(b)
        accM.setdefault(mb, []).append(b)
    bR = {k: (float(np.mean(v)) if k in ("PROMOTED", "DEMOTED") else 0.0)
          for k, v in accR.items()}
    bM = {k: (float(np.mean(v)) if k in ("PROMOTED", "DEMOTED") else 0.0)
          for k, v in accM.items()}
    for k in ("PROMOTED", "DEMOTED", "STABLE", "NA"):
        bR.setdefault(k, 0.0); bM.setdefault(k, 0.0)
    meta = dict(fit_seasons=list(fit_seasons),
                nR={k: len(v) for k, v in accR.items()},
                nM={k: len(v) for k, v in accM.items()},
                bR=bR, bM=bM)
    return meta, bR, bM


# ----------------------------------------------------------------------- main
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    seasons = DEV if mode == "dev" else HOLDOUT
    max_inact = (int(sys.argv[sys.argv.index("--max-inactive") + 1])
                 if "--max-inactive" in sys.argv else 8000)
    out_path = ROOT / "data" / f"ad_role_{mode}.json"
    rows_path = ROOT / "data" / f"ad_role_{mode}_rows.npz"
    t0 = time.time()

    con = connect(read_only=True)
    df = load_corpus(con)
    roles = load_roles()
    byp = build_index(df, roles)
    all_seasons = sorted(set(df["season"]))
    print(f"corpus {len(df)} rows, seasons {all_seasons}, role flags {len(roles)}",
          flush=True)

    # ---- §6.6 replica assertion: local sim must equal production sim BITWISE
    probe = None
    for r in df[df["season"] == seasons[0]].iloc[::997].itertuples():
        rr = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
        if rr is None or rr["n_games"] < 8 or rr["proj_min"] < 20:
            continue
        a = simulate_player(rr, SIMS, seed=7)["points"]
        b, _ = simulate_ad(rr, SIMS, seed=7)
        assert np.array_equal(a, b), "local sim replica != production simulate_player"
        probe = True
        break
    assert probe, "replica assertion never ran"
    print("replica assertion PASS (bitwise == props.simulate_player)", flush=True)

    # ---- walk-forward bias tables (one per scored season)
    fits = {}
    for s in seasons:
        prior = [x for x in all_seasons if x < s]
        meta, bR, bM = fit_bias(df, byp, prior)
        fits[s] = (meta, bR, bM)
        print(f"FIT {s} <- {prior}\n   bR PROM {bR['PROMOTED']:+.4f} "
              f"(n={meta['nR'].get('PROMOTED',0)})  DEM {bR['DEMOTED']:+.4f} "
              f"(n={meta['nR'].get('DEMOTED',0)})\n   bM PROM {bM['PROMOTED']:+.4f} "
              f"(n={meta['nM'].get('PROMOTED',0)})  DEM {bM['DEMOTED']:+.4f} "
              f"(n={meta['nM'].get('DEMOTED',0)})  ({time.time()-t0:.0f}s)", flush=True)

    # ---- row selection: ALL role-active rows + stride sample of inactive
    rows = []
    for season in seasons:
        meta, bR, bM = fits[season]
        sub = df[df["season"] == season]
        cands, inact = [], []
        for r in sub.itertuples():
            st = row_state(byp, int(r.player_id), r.ord, season)
            if st is None:
                continue
            proj, nh, gp, rb, mb = st
            if nh < 8 or proj < 20:
                continue
            (cands if rb in ("PROMOTED", "DEMOTED") else inact).append((r, rb, mb))
        cap = max_inact // len(seasons)
        if len(inact) > cap:
            inact = inact[::max(1, len(inact) // cap)][:cap]
        cand = cands + inact
        print(f"{season}: {len(cands)} role-active + {len(inact)} inactive "
              f"= {len(cand)} candidates ({time.time()-t0:.0f}s)", flush=True)
        n_ok = 0
        for r, rb, mb in cand:
            base = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
            if base is None or base["n_games"] < 8 or base["proj_min"] < 20:
                continue
            i = len(rows)
            pm = float(base["proj_min"])
            bRi = bR.get(rb, 0.0)
            bMi = bM.get(mb, 0.0)
            rec = dict(player_id=int(r.player_id), season=season,
                       ord=int(r.ord), month=int(r.game_date.month),
                       rb={"NA": 0, "STABLE": 1, "PROMOTED": 2, "DEMOTED": 3}[rb],
                       mb={"STABLE": 1, "PROMOTED": 2, "DEMOTED": 3}[mb],
                       y=float(r.pts), ymin=float(r.mins), proj=pm,
                       bR=bRi, bM=bMi)
            prng = np.random.default_rng(10_000 + i)
            specs = (("ctrl", None), ("R", max(pm - bRi, 0.0)),
                     ("M", max(pm - bMi, 0.0)))
            for arm, po in specs:
                pts, mn = simulate_ad(base, SIMS, seed=i, proj_over=po)
                rec[f"crps_{arm}"] = crps(pts, rec["y"])
                rec[f"pit_{arm}"] = pit(pts, rec["y"], prng)
                rec[f"mae_{arm}"] = abs(float(pts.mean()) - rec["y"])
                rec[f"mcrps_{arm}"] = crps(mn, rec["ymin"])
                rec[f"mmae_{arm}"] = abs(float(mn.mean()) - rec["ymin"])
            rows.append(rec)
            n_ok += 1
            if n_ok % 500 == 0:
                print(f"  {season} {n_ok} rows ({time.time()-t0:.0f}s)", flush=True)
        print(f"{season}: scored {n_ok}", flush=True)
    con.close()

    # ---------------------------------------------------------------- report
    players = np.array([r["player_id"] for r in rows])
    seas = np.array([r["season"] for r in rows])
    rb = np.array([r["rb"] for r in rows])
    out = {"n": len(rows), "sims": SIMS, "mode": mode, "seasons": list(seasons),
           "boot_seed": BOOT_SEED, "nboot": NBOOT, "prereg_sha256": PREREG,
           "fits": {s: fits[s][0] for s in seasons}, "strata": {}}

    def get(metric, arm):
        return np.array([r[f"{metric}_{arm}"] for r in rows])

    def report(mask, label):
        if mask.sum() < 30:
            return
        blk = {"n": int(mask.sum()), "players": int(len(set(players[mask])))}
        print(f"\n--- {label} (n={mask.sum()}, {blk['players']} players) ---")
        for m, mn in (("crps", "points CRPS"), ("mae", "points MAE"),
                      ("mcrps", "minutes CRPS"), ("mmae", "minutes MAE")):
            blk[m] = {}
            base_mean = float(get(m, "ctrl")[mask].mean())
            blk[m]["ctrl_mean"] = base_mean
            for arm in ("R", "M"):
                d = (get(m, "ctrl") - get(m, arm))[mask]      # + = arm BETTER
                pt_, lo, hi, se = cluster_boot(d, players[mask])
                blk[m][arm] = dict(delta=pt_, lo=lo, hi=hi, se=se,
                                   rel_pct=100 * pt_ / base_mean if base_mean else 0.0,
                                   sig="SIG" if (lo > 0 or hi < 0) else "ns",
                                   mde80=2.802 * se)
                print(f"  {m:6s} {arm} {pt_:+.5f} ({100*pt_/base_mean:+.3f}%) "
                      f"CI[{lo:+.5f},{hi:+.5f}] se {se:.5f} "
                      f"MDE80 {2.802*se:.5f} {blk[m][arm]['sig']}")
            # R - M contrast (the "is the rotation source load-bearing" test)
            d = (get(m, "M") - get(m, "R"))[mask]
            pt_, lo, hi, se = cluster_boot(d, players[mask])
            blk[m]["R_minus_M"] = dict(delta=pt_, lo=lo, hi=hi, se=se,
                                       sig="SIG" if (lo > 0 or hi < 0) else "ns")
            print(f"  {m:6s} R-M {pt_:+.5f} CI[{lo:+.5f},{hi:+.5f}] "
                  f"{blk[m]['R_minus_M']['sig']}")
        blk["pit_mean"] = {a: float(get("pit", a)[mask].mean()) for a in ARMS}
        blk["crps_mean"] = {a: float(get("crps", a)[mask].mean()) for a in ARMS}
        print("  PIT: " + "  ".join(f"{a} {blk['pit_mean'][a]:.4f}" for a in ARMS))
        out["strata"][label] = blk

    ones = np.ones(len(rows), bool)
    active = np.isin(rb, (2, 3))
    report(active, "ROLE-ACTIVE (PRIMARY)")
    report(rb == 2, "PROMOTED")
    report(rb == 3, "DEMOTED")
    report(~active, "STABLE+NA (must be exactly 0)")
    report(ones, "ALL SCORED pooled")
    for s in seasons:
        report(active & (seas == s), f"ROLE-ACTIVE {s}")

    # zero-outside-window, exactly
    zmask = ~active
    out["zero_outside_window"] = {
        "n": int(zmask.sum()),
        "max_abs_dcrps": float(np.max(np.abs((get("crps", "ctrl") - get("crps", "R"))[zmask]))) if zmask.sum() else 0.0,
        "max_abs_dpit": float(np.max(np.abs((get("pit", "ctrl") - get("pit", "R"))[zmask]))) if zmask.sum() else 0.0,
    }
    print(f"\nZERO-OUTSIDE-WINDOW: n={out['zero_outside_window']['n']} "
          f"max|dCRPS|={out['zero_outside_window']['max_abs_dcrps']:.3e} "
          f"max|dPIT|={out['zero_outside_window']['max_abs_dpit']:.3e}")

    out_path.write_text(json.dumps(out, indent=2, default=float))
    np.savez_compressed(rows_path, **{k: np.array([r[k] for r in rows])
                                      for k in rows[0]})
    print(f"\nwrote {out_path} and {rows_path}  ({time.time()-t0:.0f}s)")
    print("AD_ROLE_GATE_DONE", flush=True)


if __name__ == "__main__":
    main()
