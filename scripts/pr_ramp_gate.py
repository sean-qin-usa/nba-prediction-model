#!/usr/bin/env python3
"""PROPS EARLY-MINUTES RAMP — pre-registered gate.

PRE-REGISTRATION: data/props_ramp_prereg.md
  sha256 010947be0ed97a09a8e6035bb72514197b0e94b2383129175b7489e766a12444
Nothing here may deviate from it.

Scoring/stratification REUSES the D128 harness (scripts/d79_reversal_review.py):
same CRPS, same randomized PIT, same cluster-bootstrap-by-player, same 4,000
sims per player-game, same conditioning (002 targets, seconds>=720, n_games>=8,
proj_min>=20), same paired same-seed MC draws — so the numbers are directly
comparable to D128's table.

ARMS (control = production props.py verbatim, same run):
  ctrl  player_rates_from_stats + simulate_player, untouched
  A     proj_min -= b_A(gp)              walk-forward bucket bias (PRIMARY)
  B     spread widened by s(gp), mean untouched  (uncertainty-only ablation)
  C     proj_min -= b_C(gp, avail)       two-axis (availability-conditioned)
  A0    proj_min -= b_const              DIAGNOSTIC single level knob (not gated)

Read-only DB. Writes data/pr_ramp_<tag>.json and the per-row parquet/npz.

usage: pr_ramp_gate.py [dev|holdout] [--max-octnov N] [--max-rest N]
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
from nbapred.engine.props import player_rates_from_stats, simulate_player

DEV = ("2023-24", "2024-25", "2025-26")
HOLDOUT = ("2021-22", "2022-23")
HL = 10.0
SIMS = 4000
NBOOT = 2000
BOOT_SEED = 20260801
ARMS = ("ctrl", "A", "B", "C", "A0")
GATED = ("A", "B", "C")

# pre-registered bucket edges: {0},{1,2},{3,5},{6,9},{10,14},{15,19},{>=20 -> 0}
BUCKETS = ((0, 1), (1, 3), (3, 6), (6, 10), (10, 15), (15, 20))
K_HORIZON = 20
MIN_CELL = 100          # ARM C thin-cell fallback to b_A (prereg §7)


def bucket_of(gp: int):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= gp < hi:
            return i
    return None                      # gp >= 20 -> no correction


# --------------------------------------------------------------- sim (replica)
def simulate_ramp(rates: dict, n: int, seed: int, proj_over=None, spread=1.0):
    """VERBATIM copy of props.simulate_player's generative model, with two
    hooks: proj_over replaces proj_min in the MINUTES draw only, and spread
    scales the empirical minutes deviations + jitter (ARM B). With
    proj_over=None and spread=1.0 it must reproduce simulate_player bitwise;
    main() asserts this before scoring anything."""
    rng = np.random.default_rng(seed)
    hist = rates.get("minutes_hist")
    pm = float(rates.get("proj_min", 30.0)) if proj_over is None else float(proj_over)
    if hist is not None and len(hist) >= 5:
        h = np.asarray(hist, float)
        base = (rng.choice(h, n) - h.mean()) * spread + pm
        mins = np.clip(base + rng.normal(0, 2.0 * spread, n), 0, 48)
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
    points = rim_p + mid_p + thr_p + ft_m
    return points, mins


# ------------------------------------------------------------------- scoring
def crps(samples, y):
    s = np.sort(samples); n = len(s)
    return float(np.mean(np.abs(s - y)) - 0.5 * (2 * np.arange(1, n + 1) - n - 1) @ s / n ** 2)


def pit(samples, y, rng):
    lo = float(np.mean(samples < y))
    ties = float(np.mean(samples == y))
    return lo + rng.uniform() * ties


def cluster_boot(deltas, players, iters=NBOOT, seed=BOOT_SEED):
    """Same estimator as D128's cluster_boot (resample player clusters with
    replacement, take the mean of the concatenated deltas) — vectorised."""
    uniq, inv = np.unique(players, return_inverse=True)
    sums = np.bincount(inv, weights=deltas, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), (iters, len(uniq)))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(np.mean(deltas)), float(lo), float(hi),
            float(means.std(ddof=1)))


# ------------------------------------------------------------- corpus / fits
def load_corpus(con):
    df = con.execute("""
        SELECT s.player_id, s.team_id, g.season, g.game_date, s.seconds/60.0 AS mins,
               s.pts
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '002%' AND s.seconds >= 720
        ORDER BY s.player_id, g.game_date
    """).fetchdf()
    tg = con.execute("""
        SELECT season, team_id, game_date FROM nba_games
        WHERE game_id LIKE '002%' ORDER BY season, team_id, game_date
    """).fetchdf()
    df["ord"] = df["game_date"].astype("datetime64[ns]").values.astype("datetime64[D]").astype(int)
    tg["ord"] = tg["game_date"].astype("datetime64[ns]").values.astype("datetime64[D]").astype(int)
    return df, tg


def build_index(df, tg):
    byp = {}
    for pid, sub in df.groupby("player_id", sort=False):
        byp[int(pid)] = (sub["ord"].to_numpy(), sub["mins"].to_numpy(float),
                         sub["season"].to_numpy(object))
    tsched = {}
    for (s, t), sub in tg.groupby(["season", "team_id"], sort=False):
        tsched[(s, int(t))] = np.sort(sub["ord"].to_numpy())
    return byp, tsched


def row_features(byp, tsched, pid, day, season, team_id):
    """(proj_ewma, n_hist, gp, tgp) exactly as props.py would see them."""
    dates, mins, seas = byp[pid]
    i = int(np.searchsorted(dates, day))
    if i < 3:
        return None
    w = 0.5 ** (np.arange(i)[::-1] / HL)
    proj = float(np.sum(w * mins[:i]) / np.sum(w))
    gp = int((seas[:i] == season).sum())
    sch = tsched.get((season, int(team_id)))
    tgp = int(np.searchsorted(sch, day)) if sch is not None else gp
    return proj, i, gp, tgp


def fit_bias(df, byp, tsched, fit_seasons):
    """Walk-forward bias tables from `fit_seasons` only (strictly-prior)."""
    gpb, avb, bias = [], [], []
    sub = df[df["season"].isin(fit_seasons)]
    for r in sub.itertuples():
        f = row_features(byp, tsched, int(r.player_id), r.ord, r.season, r.team_id)
        if f is None:
            continue
        proj, nh, gp, tgp = f
        if nh < 8 or proj < 20:
            continue
        gpb.append(gp); avb.append(1 if gp >= tgp - 1 else 0)
        bias.append(proj - float(r.mins))
    gpb = np.array(gpb); avb = np.array(avb); bias = np.array(bias)
    bk = np.array([bucket_of(int(g)) if bucket_of(int(g)) is not None else -1
                   for g in gpb])
    bA, nA = {}, {}
    for i in range(len(BUCKETS)):
        m = bk == i
        bA[i] = float(bias[m].mean()) if m.sum() else 0.0
        nA[i] = int(m.sum())
    bC, nC = {}, {}
    for i in range(len(BUCKETS)):
        for a in (0, 1):
            m = (bk == i) & (avb == a)
            if m.sum() >= MIN_CELL:
                bC[(i, a)] = float(bias[m].mean())
            else:
                bC[(i, a)] = bA[i]
            nC[(i, a)] = int(m.sum())
    mconst = bk >= 0
    b_const = float(bias[mconst].mean()) if mconst.sum() else 0.0
    return dict(bA=bA, nA=nA, bC={f"{k[0]}_{k[1]}": v for k, v in bC.items()},
                nC={f"{k[0]}_{k[1]}": v for k, v in nC.items()},
                b_const=b_const, n_fit=int(len(bias)),
                fit_seasons=list(fit_seasons)), bA, bC, b_const


# ----------------------------------------------------------------------- main
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    seasons = DEV if mode == "dev" else HOLDOUT
    max_on = int(sys.argv[sys.argv.index("--max-octnov") + 1]) if "--max-octnov" in sys.argv else 10**9
    max_rest = int(sys.argv[sys.argv.index("--max-rest") + 1]) if "--max-rest" in sys.argv else 6000
    out_path = ROOT / "data" / f"pr_ramp_{mode}.json"
    rows_path = ROOT / "data" / f"pr_ramp_{mode}_rows.npz"
    t0 = time.time()

    con = connect(read_only=True)
    df, tg = load_corpus(con)
    byp, tsched = build_index(df, tg)
    all_seasons = sorted(set(df["season"]))
    print(f"corpus {len(df)} rows, seasons {all_seasons}", flush=True)

    # ---- control-replication assertion: local sim must equal production sim
    probe = None
    for r in df[df["season"] == seasons[0]].iloc[::997].itertuples():
        rr = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
        if rr is None or rr["n_games"] < 8 or rr["proj_min"] < 20:
            continue
        a = simulate_player(rr, SIMS, seed=7)["points"]
        b, _ = simulate_ramp(rr, SIMS, seed=7)
        assert np.array_equal(a, b), "local sim replica != production simulate_player"
        probe = True
        break
    assert probe, "replica assertion never ran"
    print("replica assertion PASS (bitwise == props.simulate_player)", flush=True)

    # ---- walk-forward bias tables (one per scored season)
    fits = {}
    for s in seasons:
        prior = [x for x in all_seasons if x < s]
        meta, bA, bC, bconst = fit_bias(df, byp, tsched, prior)
        fits[s] = (meta, bA, bC, bconst)
        print(f"FIT {s} <- {prior}  n={meta['n_fit']}  "
              f"bA={[round(bA[i],3) for i in range(len(BUCKETS))]}  "
              f"b_const={bconst:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- row selection: ALL eligible Oct-Nov + stride sample of the rest
    rows = []
    for season in seasons:
        meta, bA, bC, bconst = fits[season]
        sub = df[df["season"] == season]
        mon = sub["game_date"].astype("datetime64[ns]").dt.month
        on = sub[mon.isin((10, 11))]
        rest = sub[~mon.isin((10, 11))]
        if len(rest) > max_rest // len(seasons):
            rest = rest.iloc[::max(1, len(rest) // (max_rest // len(seasons)))]
        cand = list(on.itertuples())[:max_on] + list(rest.itertuples())
        n_ok = 0
        for r in cand:
            f = row_features(byp, tsched, int(r.player_id), r.ord, season, r.team_id)
            if f is None:
                continue
            proj_fast, nh, gp, tgp = f
            if nh < 8 or proj_fast < 20:
                continue
            base = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
            if base is None or base["n_games"] < 8 or base["proj_min"] < 20:
                continue
            i = len(rows)
            bk = bucket_of(gp)
            av = 1 if gp >= tgp - 1 else 0
            bAi = bA[bk] if bk is not None else 0.0
            bCi = bC[(bk, av)] if bk is not None else 0.0
            bK = bconst if bk is not None else 0.0
            h = np.asarray(base["minutes_hist"], float)
            sd_tot = float(np.sqrt(h.var() + 4.0))
            spread = float(np.sqrt(1.0 + (bAi / sd_tot) ** 2)) if bk is not None else 1.0
            pm = float(base["proj_min"])
            rec = dict(player_id=int(r.player_id), season=season,
                       month=int(r.game_date.month), gp=gp, tgp=tgp, avail=av,
                       y=float(r.pts), ymin=float(r.mins), proj=pm,
                       bA=bAi, bC=bCi, bK=bK, spread=spread)
            prng = np.random.default_rng(10_000 + i)
            specs = (("ctrl", None, 1.0), ("A", max(pm - bAi, 0.0), 1.0),
                     ("B", None, spread), ("C", max(pm - bCi, 0.0), 1.0),
                     ("A0", max(pm - bK, 0.0), 1.0))
            for arm, po, sp in specs:
                pts, mn = simulate_ramp(base, SIMS, seed=i, proj_over=po, spread=sp)
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
    month = np.array([r["month"] for r in rows])
    gp = np.array([r["gp"] for r in rows])
    ramped = np.array([r["bA"] != 0.0 for r in rows])
    out = {"n": len(rows), "sims": SIMS, "mode": mode, "seasons": list(seasons),
           "boot_seed": BOOT_SEED, "nboot": NBOOT,
           "prereg_sha256": "010947be0ed97a09a8e6035bb72514197b0e94b2383129175b7489e766a12444",
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
            for arm in ("A", "B", "C", "A0"):
                d = (get(m, "ctrl") - get(m, arm))[mask]     # + = arm BETTER
                pt_, lo, hi, se = cluster_boot(d, players[mask])
                sig = "SIG" if (lo > 0 or hi < 0) else "ns"
                blk[m][arm] = dict(delta=pt_, lo=lo, hi=hi, se=se,
                                   rel_pct=100 * pt_ / base_mean if base_mean else 0.0,
                                   sig=sig, mde80=2.802 * se)
                print(f"  {m:6s} {arm:2s} {pt_:+.5f} ({100*pt_/base_mean:+.3f}%) "
                      f"CI[{lo:+.5f},{hi:+.5f}] se {se:.5f} MDE80 {2.802*se:.5f} {sig}")
        blk["pit_mean"] = {a: float(get("pit", a)[mask].mean()) for a in ARMS}
        blk["crps_mean"] = {a: float(get("crps", a)[mask].mean()) for a in ARMS}
        blk["proj_mean"] = {"ctrl": float(np.array([r["proj"] for r in rows])[mask].mean())}
        blk["ramped_share"] = float(ramped[mask].mean())
        print("  PIT: " + "  ".join(f"{a} {blk['pit_mean'][a]:.4f}" for a in ARMS)
              + f"   ramped {blk['ramped_share']:.2f}")
        out["strata"][label] = blk

    ones = np.ones(len(rows), bool)
    octnov = np.isin(month, (10, 11))
    report(octnov, "OCT+NOV (PRIMARY)")
    report(month == 10, "OCT only")
    report(ones, "ALL ROWS pooled")
    report(~octnov, "DEC-JUN (veto)")
    for s in seasons:
        report(octnov & (seas == s), f"OCT+NOV {s}")
    report(octnov & ramped, "OCT+NOV ramp-active")
    report(ramped, "ALL ramp-active")
    report(~ramped, "ALL ramp-inactive (must be exactly 0)")
    for lo, hi in BUCKETS:
        report((gp >= lo) & (gp < hi), f"gp[{lo},{hi})")

    out_path.write_text(json.dumps(out, indent=2, default=float))
    np.savez_compressed(rows_path, **{k: np.array([r[k] for r in rows])
                                      for k in rows[0]})
    print(f"\nwrote {out_path} and {rows_path}  ({time.time()-t0:.0f}s)")
    print("PR_RAMP_GATE_DONE", flush=True)


if __name__ == "__main__":
    main()
