#!/usr/bin/env python3
"""PROPS ABSENCE RAMP — pre-registered gate (the D133 successor).

PRE-REGISTRATION: data/absence_prereg.md
  sha256 c6c7044cc494346aa00d85b84dfacc05f68e3c6e0ece95204a7e0a8d09c5031e
Nothing here may deviate from it.

Harness functions are IMPORTED from scripts/pr_ramp_gate.py (D133/D128), so the
CRPS, the randomized PIT, the by-player cluster bootstrap, the 4,000-sim
generative replica and the eval conditioning are bit-identical to D133's.

ARMS (control = production props.py verbatim, same run, D133 ramp ON):
  ctrl  player_rates_from_stats + simulate_player, untouched
  A     proj_min -= b_miss(miss10), ZERO for miss10<=4     (PRIMARY / ship)
  A0    proj_min -= b_const on the same window             (adversarial control)
  C     proj_raw  -= b_missonly(miss10), gp ramp REPLACED  (mechanism test)

Read-only DB. Writes data/ab_props_gate.json + data/ab_props_gate_rows.npz.
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
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from nbapred.db import connect
from nbapred.engine.props import (minutes_ramp, player_rates_from_stats,
                                  simulate_player)
from pr_ramp_gate import (BOOT_SEED, HL, NBOOT, SIMS, build_index, cluster_boot,
                          crps, load_corpus, pit, simulate_ramp)

PREREG_SHA = "c6c7044cc494346aa00d85b84dfacc05f68e3c6e0ece95204a7e0a8d09c5031e"
SCORED = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
MISS_B = ((0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, 10))
WINDOW_LO = 4            # bucket index at which ARM A becomes non-zero (miss>=5)
ARMS = ("ctrl", "A", "A0", "C")
MAX_NONWINDOW = 6000     # stride sample of miss10<=4 rows (ARM A is 0 there)


def bucket_of(m: int) -> int:
    for i, (lo, hi) in enumerate(MISS_B):
        if lo <= m <= hi:
            return i
    return len(MISS_B) - 1


def miss10_of(byp, tsched, pid, day, season):
    """PRODUCTION-IDENTICAL absence axis (prereg §2). None if underivable."""
    dates, mins, seas, teams = byp[pid]
    i = int(np.searchsorted(dates, day))
    if i == 0:
        return None, None
    lt = int(teams[i - 1])
    sch = tsched.get((season, lt))
    if sch is None:
        return 0, lt
    prior = sch[sch < day]
    if prior.size == 0:
        return 0, lt
    prior = prior[-10:]
    m = (seas[:i] == season) & (teams[:i] == lt)
    own = set(dates[:i][m].tolist())
    if not own:
        return 0, lt
    first = min(own)
    return int(sum(1 for d in prior if d >= first and d not in own)), lt


def build_index2(df, tg):
    """build_index + team ids (needed by the absence axis)."""
    byp = {}
    for pid, sub in df.groupby("player_id", sort=False):
        byp[int(pid)] = (sub["ord"].to_numpy(), sub["mins"].to_numpy(float),
                         sub["season"].to_numpy(object),
                         sub["team_id"].to_numpy())
    tsched = {}
    for (s, t), sub in tg.groupby(["season", "team_id"], sort=False):
        tsched[(s, int(t))] = np.sort(sub["ord"].to_numpy())
    return byp, tsched


def row_feats(byp, tsched, pid, day, season):
    dates, mins, seas, teams = byp[pid]
    i = int(np.searchsorted(dates, day))
    if i < 3:
        return None
    w = 0.5 ** (np.arange(i)[::-1] / HL)
    proj_raw = float(np.sum(w * mins[:i]) / np.sum(w))
    gp = int((seas[:i] == season).sum())
    mi, lt = miss10_of(byp, tsched, pid, day, season)
    if mi is None:
        return None
    return proj_raw, i, gp, mi, lt


def fit_tables(df, byp, tsched, fit_seasons):
    """Walk-forward b tables from `fit_seasons` only."""
    mb, res_after, res_raw = [], [], []
    sub = df[df["season"].isin(fit_seasons)]
    for r in sub.itertuples():
        f = row_feats(byp, tsched, int(r.player_id), r.ord, r.season)
        if f is None:
            continue
        proj_raw, nh, gp, mi, lt = f
        if nh < 8 or proj_raw < 20:
            continue
        after = max(proj_raw - minutes_ramp(gp), 0.0)
        if after < 20:
            continue
        mb.append(bucket_of(mi))
        res_after.append(after - float(r.mins))
        res_raw.append(proj_raw - float(r.mins))
    mb = np.array(mb)
    ra = np.array(res_after)
    rr = np.array(res_raw)
    bA = {}
    bC = {}
    nb = {}
    for i in range(len(MISS_B)):
        m = mb == i
        nb[i] = int(m.sum())
        bA[i] = float(ra[m].mean()) if m.sum() else 0.0
        bC[i] = float(rr[m].mean()) if m.sum() else 0.0
    win = mb >= WINDOW_LO
    b_const = float(ra[win].mean()) if win.sum() else 0.0
    # ARM A is ZERO outside the pre-registered window
    bA_applied = {i: (bA[i] if i >= WINDOW_LO else 0.0) for i in bA}
    return dict(bA_fit=bA, bA_applied=bA_applied, bC=bC, b_const=b_const,
                n_bucket=nb, n_fit=int(len(ra)),
                fit_seasons=list(fit_seasons)), bA_applied, bC, b_const


def main():
    assert os.environ.get("PROPS_MIN_RAMP", "1") != "0", \
        "control must be production (D133 ramp ON)"
    t0 = time.time()
    con = connect(read_only=True)
    df, tg = load_corpus(con)
    byp, tsched = build_index2(df, tg)
    all_seasons = sorted(set(df["season"]))
    print(f"corpus {len(df)} rows, seasons {all_seasons}", flush=True)

    # ---- replica assertion (harness identity) ------------------------------
    probe = False
    for r in df[df["season"] == SCORED[-1]].iloc[::997].itertuples():
        rr = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
        if rr is None or rr["n_games"] < 8 or rr["proj_min"] < 20:
            continue
        a = simulate_player(rr, SIMS, seed=7)["points"]
        b, _ = simulate_ramp(rr, SIMS, seed=7)
        assert np.array_equal(a, b), "replica != production simulate_player"
        probe = True
        break
    assert probe
    print("replica assertion PASS (bitwise == props.simulate_player)", flush=True)

    fits = {}
    for s in SCORED:
        prior = [x for x in all_seasons if x < s]
        meta, bA, bC, bk = fit_tables(df, byp, tsched, prior)
        fits[s] = (meta, bA, bC, bk)
        print(f"FIT {s} <- {prior} n={meta['n_fit']}\n"
              f"    bA_fit   = {[round(meta['bA_fit'][i], 4) for i in range(6)]}\n"
              f"    bA_APPLY = {[round(bA[i], 4) for i in range(6)]}\n"
              f"    bC       = {[round(bC[i], 4) for i in range(6)]}  "
              f"b_const={bk:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- row selection -----------------------------------------------------
    rows = []
    for season in SCORED:
        meta, bA, bC, bk = fits[season]
        sub = df[df["season"] == season]
        cand, nonwin = [], []
        for r in sub.itertuples():
            f = row_feats(byp, tsched, int(r.player_id), r.ord, season)
            if f is None:
                continue
            proj_raw, nh, gp, mi, lt = f
            if nh < 8 or proj_raw < 20:
                continue
            (cand if bucket_of(mi) >= WINDOW_LO else nonwin).append((r, gp, mi))
        step = max(1, len(nonwin) // (MAX_NONWINDOW // len(SCORED)))
        cand = cand + nonwin[::step]
        n_ok = 0
        for r, gp, mi in cand:
            base = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
            if base is None or base["n_games"] < 8 or base["proj_min"] < 20:
                continue
            i = len(rows)
            bi = bucket_of(mi)
            pm = float(base["proj_min"])                 # already D133-ramped
            pr = pm + minutes_ramp(gp)                   # exact: max() never binds
            rec = dict(player_id=int(r.player_id), season=season,
                       date=str(r.game_date)[:10], month=int(r.game_date.month),
                       gp=gp, miss10=mi, bucket=bi, y=float(r.pts),
                       ymin=float(r.mins), proj=pm, proj_raw=pr,
                       bA=bA[bi], bK=(bk if bi >= WINDOW_LO else 0.0), bC=bC[bi])
            prng = np.random.default_rng(10_000 + i)
            specs = (("ctrl", None),
                     ("A", max(pm - rec["bA"], 0.0)),
                     ("A0", max(pm - rec["bK"], 0.0)),
                     ("C", max(pr - rec["bC"], 0.0)))
            for arm, po in specs:
                pts, mn = simulate_ramp(base, SIMS, seed=i, proj_over=po)
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

    players = np.array([r["player_id"] for r in rows])
    seas = np.array([r["season"] for r in rows])
    month = np.array([r["month"] for r in rows])
    gp = np.array([r["gp"] for r in rows])
    miss = np.array([r["miss10"] for r in rows])
    win = miss >= 5
    out = {"n": len(rows), "sims": SIMS, "boot_seed": BOOT_SEED, "nboot": NBOOT,
           "prereg_sha256": PREREG_SHA, "seasons": list(SCORED),
           "fits": {s: fits[s][0] for s in SCORED}, "strata": {}}

    def get(metric, arm):
        return np.array([r[f"{metric}_{arm}"] for r in rows])

    def report(mask, label):
        if mask.sum() < 30:
            return
        blk = {"n": int(mask.sum()), "players": int(len(set(players[mask])))}
        print(f"\n--- {label} (n={mask.sum()}, {blk['players']} players) ---")
        for m in ("crps", "mae", "mcrps", "mmae"):
            blk[m] = {}
            bm = float(get(m, "ctrl")[mask].mean())
            blk[m]["ctrl_mean"] = bm
            for arm in ("A", "A0", "C"):
                d = (get(m, "ctrl") - get(m, arm))[mask]
                pt_, lo, hi, se = cluster_boot(d, players[mask])
                sig = "SIG" if (lo > 0 or hi < 0) else "ns"
                blk[m][arm] = dict(delta=pt_, lo=lo, hi=hi, se=se, sig=sig,
                                   rel_pct=100 * pt_ / bm if bm else 0.0,
                                   mde80=2.802 * se)
                print(f"  {m:6s} {arm:2s} {pt_:+.5f} ({100*pt_/bm:+.3f}%) "
                      f"CI[{lo:+.5f},{hi:+.5f}] se {se:.5f} "
                      f"MDE80 {2.802*se:.5f} {sig}")
        blk["pit_mean"] = {a: float(get("pit", a)[mask].mean()) for a in ARMS}
        print("  PIT: " + "  ".join(f"{a} {blk['pit_mean'][a]:.4f}" for a in ARMS))
        out["strata"][label] = blk

    report(win, "WINDOW miss10>=5 (PRIMARY)")
    report(win & (gp >= 20), "PRIMARY & gp>=20 (D133-inert region)")
    report(win & (gp < 20), "PRIMARY & gp<20")
    report(miss >= 8, "miss10 8-10")
    report((miss >= 5) & (miss <= 7), "miss10 5-7")
    report(np.ones(len(rows), bool), "ALL SCORED ROWS pooled")
    report(~win, "miss10<=4 (must be EXACTLY 0 for A/A0)")
    report(win & ~np.isin(month, (10, 11)), "PRIMARY Dec-Jun")
    report(win & np.isin(month, (10, 11)), "PRIMARY Oct-Nov")
    for s in SCORED:
        report(win & (seas == s), f"PRIMARY {s}")

    # ---- V2 zero-outside-window (bitwise) ---------------------------------
    nz = ~win
    z = {a: float(np.abs(get("crps", "ctrl")[nz] - get("crps", a)[nz]).max())
         for a in ("A", "A0")}
    out["zero_outside_window_max_abs_dcrps"] = z
    print(f"\nV2 zero-outside-window max|dCRPS| on miss10<=4: {z}")

    # ---- A vs A0 (shape vs level) -----------------------------------------
    d = (get("crps", "A0") - get("crps", "A"))[win]
    pt_, lo, hi, se = cluster_boot(d, players[win])
    out["A_minus_A0_primary"] = dict(delta=pt_, lo=lo, hi=hi, se=se,
                                     sig="SIG" if (lo > 0 or hi < 0) else "ns")
    print(f"A - A0 (primary): {pt_:+.5f} CI[{lo:+.5f},{hi:+.5f}] "
          f"{out['A_minus_A0_primary']['sig']}")
    dC = (get("crps", "C") - get("crps", "A"))[win]
    pt2, lo2, hi2, se2 = cluster_boot(dC, players[win])
    out["A_minus_C_primary"] = dict(delta=pt2, lo=lo2, hi=hi2, se=se2,
                                    sig="SIG" if (lo2 > 0 or hi2 < 0) else "ns")
    print(f"A - C  (primary): {pt2:+.5f} CI[{lo2:+.5f},{hi2:+.5f}] "
          f"{out['A_minus_C_primary']['sig']}")

    Path("data/ab_props_gate.json").write_text(json.dumps(out, indent=2, default=float))
    np.savez_compressed("data/ab_props_gate_rows.npz",
                        **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    print(f"\nwrote data/ab_props_gate.json ({time.time()-t0:.0f}s)")
    print("AB_PROPS_GATE_DONE", flush=True)


if __name__ == "__main__":
    main()
