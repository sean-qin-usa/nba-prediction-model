#!/usr/bin/env python3
"""GATES 1 + 2 — pre-registered star-out gate (one scoring pass, four arms).

PRE-REGISTRATION: data/queued_gates_prereg.md  (sha256 recorded in the run json)
Nothing here may deviate from it.

Both queued star-out gates share the same eval universe and the same same-run
control, so they are scored in ONE pass with ONE set of paired MC draws:

  ctrl  production starout.py verbatim (STAROUT_TRAIL / STAROUT_USAGE unset)
  A     GATE 1  STAROUT_TRAIL=played   — trail_min over PLAYED rows only
  U     GATE 2  STAROUT_USAGE=null_u   — uniform weights => lift = N/(N-1)
  F     GATE 2  STAROUT_USAGE=trailatt — the module's documented fallback
                                         (DIAGNOSTIC, never a ship candidate)

IMPLEMENTATION IDENTITY (§6.6): every arm is produced by the SHIPPED functions
`starout.team_context` / `starout.adjust_player_rates` / `props.simulate_player`
under the env switch it is named after. No arm is reimplemented here.

There is NO fitted coefficient in any arm (they are construction changes), so
all five seasons are scored in one pass and the dev / holdout adjudication is
done downstream by scripts/qg_starout_report.py, which reads dev first and only
opens the holdout on a dev pass.

Read-only DB. Writes data/qg_starout_gate.json + data/qg_starout_gate_rows.npz.

usage: qg_starout_gate.py [--max-rows N] [--sims N]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from nbapred.db import connect
from nbapred.engine import starout
from nbapred.engine.props import player_rates_from_stats, simulate_player

SCORED = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
SIMS = 4000
NBOOT = 2000
BOOT_SEED = 20260801
BIG = 12.0
ARMS = ("ctrl", "A", "U", "F")
ARM_ENV = {"ctrl": {}, "A": {"STAROUT_TRAIL": "played"},
           "U": {"STAROUT_USAGE": "null_u"}, "F": {"STAROUT_USAGE": "trailatt"}}
MARKETS = ("points", "rebounds", "assists")


def _d(x):
    return x.date() if hasattr(x, "date") else x


def crps(samples, y):
    s = np.sort(samples); n = len(s)
    return float(np.mean(np.abs(s - y))
                 - 0.5 * (2 * np.arange(1, n + 1) - n - 1) @ s / n ** 2)


def pit(samples, y, rng):
    lo = float(np.mean(samples < y))
    ties = float(np.mean(samples == y))
    return lo + rng.uniform() * ties


def cluster_boot(deltas, groups, iters=NBOOT, seed=BOOT_SEED):
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=deltas, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), (iters, len(uniq)))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(np.mean(deltas)), float(lo), float(hi),
            float(means.std(ddof=1)))


def set_env(arm):
    for k in ("STAROUT_TRAIL", "STAROUT_USAGE"):
        os.environ.pop(k, None)
    for k, v in ARM_ENV[arm].items():
        os.environ[k] = v


def main():
    max_rows = int(sys.argv[sys.argv.index("--max-rows") + 1]) if "--max-rows" in sys.argv else 10 ** 9
    sims = int(sys.argv[sys.argv.index("--sims") + 1]) if "--sims" in sys.argv else SIMS
    t0 = time.time()
    prereg = (ROOT / "data" / "queued_gates_prereg.md").read_bytes()
    sha = hashlib.sha256(prereg).hexdigest()
    print(f"prereg sha256 {sha}", flush=True)
    # the props control must be production (D145 §15)
    assert os.environ.get("PROPS_MIN_RAMP", "1") != "0"
    assert os.environ.get("PROPS_ABSENCE_RAMP", "1") != "0"
    assert not os.environ.get("PROPS_CHANNEL_RAMP")

    con = connect(read_only=True)
    pg = con.execute("""
        SELECT s.game_id, s.team_id, s.player_id, s.seconds/60.0 AS m,
               s.pts, s.oreb + s.dreb AS reb, s.ast,
               s.rima + s.mida + s.thra AS shots,
               g.game_date, g.season
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games
              WHERE wl IS NOT NULL) g USING (game_id)
        WHERE s.game_id LIKE '002%'""").fetchdf()
    pg["game_date"] = [_d(x) for x in pg["game_date"]]
    pg = pg[pg["season"].isin(SCORED)].copy()
    sched = defaultdict(list)
    for s, t, d in con.execute("""
        SELECT season, team_id, game_date FROM nba_games
        WHERE game_id LIKE '002%' AND wl IS NOT NULL""").fetchall():
        sched[(s, int(t))].append(_d(d))
    for k in sched:
        sched[k] = sorted(set(sched[k]))
    weights = starout.load_usage_weights()
    positions = starout.load_positions()
    assert weights, "v2_usage.npz must load"
    print(f"corpus {len(pg)} rows / seasons {SCORED}; usage npz {len(weights)} "
          f"players; positions {len(positions)}", flush=True)

    # ---------- SCREEN: team-dates where ANY arm could fire ------------------
    # ARM A's played-only trailing mean dominates ctrl's row-based mean, so a
    # played-only screen is a superset of every arm's firing set.
    per = defaultdict(list)                      # (season, tid, pid) -> [(d, m)]
    for r in pg.itertuples():
        per[(r.season, int(r.team_id), int(r.player_id))].append(
            (r.game_date, float(r.m or 0.0)))
    for k in per:
        per[k].sort()
    gid_of = {}
    for r in pg.itertuples():
        gid_of[(r.season, int(r.team_id), r.game_date)] = r.game_id

    by_team = defaultdict(dict)
    for (s2, t2, p), e in per.items():
        by_team[(s2, t2)][p] = e
    cand = []                                    # (season, tid, date, out_ids)
    for (s, t), sc in sched.items():
        if s not in SCORED:
            continue
        ppl = by_team.get((s, int(t)))
        if not ppl:
            continue
        # per player: chronological arrays + running index (single sweep/date)
        prep = {}
        for p, ent in ppl.items():
            ds = [d for d, _ in ent]
            ms = [m for _, m in ent]
            pl_d = [d for d, m in ent if m > 0]
            pl_m = [m for d, m in ent if m > 0]
            prep[p] = (ds, ms, pl_d, pl_m, {d: m for d, m in ent})
        for gd in sc:
            outs, hot = set(), False
            for p, (ds, ms, pl_d, pl_m, dm) in prep.items():
                i = np.searchsorted(ds, gd)       # rows strictly before gd
                if i < starout.MIN_TRAIL_GAMES:
                    continue
                if dm.get(gd, 0.0) > 0:
                    continue
                outs.add(p)
                if hot:
                    continue
                j = np.searchsorted(pl_d, gd)     # played rows strictly before
                if j < starout.MIN_TRAIL_GAMES:
                    continue
                if not (0 < (gd - pl_d[j - 1]).days <= starout.FRESH_DAYS):
                    continue
                lo = max(0, j - starout.TRAIL_GAMES)
                if (sum(pl_m[lo:j]) / (j - lo)) >= starout.STAR_TRAILING_MIN:
                    hot = True
            if hot and outs:
                cand.append((s, int(t), gd, outs))
    print(f"screened {len(cand)} candidate team-dates ({time.time()-t0:.0f}s)",
          flush=True)

    # ---------- detect under each arm ---------------------------------------
    # cached: team_context is deterministic given (team, date, out set) and the
    # env mode, so a re-run reuses the detection pass verbatim.
    import pickle
    cache_p = ROOT / "data" / "qg_starout_ctx.pkl"
    ctxs = None
    if cache_p.exists():
        try:
            blob = pickle.loads(cache_p.read_bytes())
            if blob.get("n_cand") == len(cand) and blob.get("arms") == list(ARMS):
                ctxs = blob["ctxs"]
                print(f"  loaded cached detection ({time.time()-t0:.0f}s)", flush=True)
        except Exception:  # noqa: BLE001
            ctxs = None
    if ctxs is None:
        ctxs = {a: {} for a in ARMS}
        for a in ARMS:
            set_env(a)
            n_fire = 0
            for (s, t, gd, outs) in cand:
                c = starout.team_context(con, t, outs, gd, weights, positions)
                if c is not None:
                    c.pop("pool", None)          # not picklable-friendly / unused
                    ctxs[a][(s, t, gd)] = c
                    n_fire += 1
            print(f"  arm {a:4s} fires on {n_fire}/{len(cand)} team-dates "
                  f"({time.time()-t0:.0f}s)", flush=True)
        cache_p.write_bytes(pickle.dumps({"n_cand": len(cand),
                                          "arms": list(ARMS), "ctxs": ctxs}))
    else:
        for a in ARMS:
            print(f"  arm {a:4s} fires on {len(ctxs[a])}/{len(cand)} team-dates",
                  flush=True)
    set_env("ctrl")

    fire_union = set().union(*[set(ctxs[a]) for a in ARMS])
    print(f"union of firing team-dates: {len(fire_union)}", flush=True)

    # ---------- eval rows ----------------------------------------------------
    played12 = pg[pg.m >= BIG]
    by_td = defaultdict(list)
    for r in played12.itertuples():
        by_td[(r.season, int(r.team_id), r.game_date)].append(r)
    rows = []
    todo = []
    for (s, t, gd) in sorted(fire_union, key=lambda k: (k[2], k[1])):
        for r in by_td.get((s, t, gd), []):
            todo.append((s, t, gd, r))
    if len(todo) > max_rows:
        todo = todo[::max(1, len(todo) // max_rows)]
    print(f"candidate eval rows: {len(todo)}", flush=True)

    n_ok = 0
    for (s, t, gd, r) in todo:
        pid = int(r.player_id)
        base = player_rates_from_stats(con, pid, before=gd)
        if base is None or base["n_games"] < 8 or base["proj_min"] < 20:
            continue
        i = len(rows)
        rec = dict(player_id=pid, team_id=int(t), season=s, date=str(gd),
                   month=int(gd.month), y_points=float(r.pts),
                   y_rebounds=float(r.reb), y_assists=float(r.ast),
                   y_shots=float(r.shots), proj=float(base["proj_min"]))
        prng = np.random.default_rng(50_000 + i)
        for a in ARMS:
            c = ctxs[a].get((s, t, gd))
            rr = starout.adjust_player_rates(base, pid, c, positions) if c else base
            fired = int(c is not None and pid != c["star"])
            sim = simulate_player(rr, sims, seed=i)
            for mk in MARKETS:
                rec[f"crps_{mk}_{a}"] = crps(sim[mk], rec[f"y_{mk}"])
            rec[f"pit_points_{a}"] = pit(sim["points"], rec["y_points"], prng)
            rec[f"pit_rebounds_{a}"] = pit(sim["rebounds"], rec["y_rebounds"], prng)
            mu = max((rr["rate_rim"] + rr["rate_mid"] + rr["rate_thr"])
                     * rr["proj_min"], 0.2)
            rec[f"ll_{a}"] = float(rec["y_shots"] * np.log(mu) - mu)
            rec[f"fired_{a}"] = fired
            rec[f"lift_{a}"] = float(c["lift"]) if fired else 1.0
            rec[f"star_{a}"] = int(c["star"]) if c else -1
            rec[f"npool_{a}"] = int(c["n_pool"]) if c else 0
            rec[f"pm_{a}"] = float(rr["proj_min"])
        rows.append(rec)
        n_ok += 1
        if n_ok % 500 == 0:
            print(f"  {n_ok} rows ({time.time()-t0:.0f}s)", flush=True)
    con.close()
    print(f"scored {len(rows)} rows ({time.time()-t0:.0f}s)", flush=True)

    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    np.savez_compressed("data/qg_starout_gate_rows.npz", **R)
    out = {"prereg_sha256": sha, "n": len(rows), "sims": sims,
           "boot_seed": BOOT_SEED, "nboot": NBOOT, "seasons": list(SCORED),
           "n_candidate_team_dates": len(cand),
           "fires_by_arm": {a: len(ctxs[a]) for a in ARMS},
           "union_firing_team_dates": len(fire_union),
           "mean_lift_by_arm": {a: float(R[f"lift_{a}"][R[f"fired_{a}"] == 1].mean())
                                for a in ARMS},
           "fired_rows_by_arm": {a: int(R[f"fired_{a}"].sum()) for a in ARMS},
           "strata": {}}

    players = R["player_id"]
    seas = R["season"]
    diff_A = (R["fired_ctrl"] != R["fired_A"]) | (
        np.abs(R["lift_ctrl"] - R["lift_A"]) > 1e-12) | (
        np.abs(R["pm_ctrl"] - R["pm_A"]) > 1e-12)
    both = (R["fired_ctrl"] == 1) & (R["fired_A"] == 1)
    only_A = (R["fired_ctrl"] == 0) & (R["fired_A"] == 1)
    ctrl_fire = R["fired_ctrl"] == 1

    def report(mask, label, arms=("A", "U", "F")):
        if mask.sum() < 30:
            return
        blk = {"n": int(mask.sum()),
               "players": int(len(set(players[mask].tolist())))}
        print(f"\n--- {label} (n={mask.sum()}, {blk['players']} players) ---")
        for metric, key in (("points CRPS", "crps_points"),
                            ("rebounds CRPS", "crps_rebounds"),
                            ("assists CRPS", "crps_assists"),
                            ("attempts LL", "ll")):
            blk[key] = {}
            base = R[f"{key}_ctrl"][mask]
            blk[key]["ctrl_mean"] = float(base.mean())
            for a in arms:
                arm = R[f"{key}_{a}"][mask]
                d = (base - arm) if key != "ll" else (arm - base)
                pt_, lo, hi, se = cluster_boot(d, players[mask])
                sig = "SIG" if (lo > 0 or hi < 0) else "ns"
                blk[key][a] = dict(delta=pt_, lo=lo, hi=hi, se=se, sig=sig,
                                   mde80=2.802 * se)
                print(f"  {metric:14s} {a} {pt_:+.5f} CI[{lo:+.5f},{hi:+.5f}] "
                      f"se {se:.5f} MDE80 {2.802*se:.5f} {sig}")
        blk["pit_points"] = {a: float(R[f"pit_points_{a}"][mask].mean()) for a in ARMS}
        blk["pit_rebounds"] = {a: float(R[f"pit_rebounds_{a}"][mask].mean()) for a in ARMS}
        print("  PIT pts: " + "  ".join(f"{a} {blk['pit_points'][a]:.4f}" for a in ARMS))
        out["strata"][label] = blk

    ones = np.ones(len(rows), bool)
    report(diff_A, "G1 PRIMARY — ctrl vs A adjustment differs")
    report(only_A, "G1 newly fired (only A)")
    report(both, "G1 both fire")
    report(ctrl_fire, "G2 PRIMARY — ctrl fires")
    report(ones, "ALL scored rows pooled")
    for s in SCORED:
        report(diff_A & (seas == s), f"G1 PRIMARY {s}")
        report(ctrl_fire & (seas == s), f"G2 PRIMARY {s}")
    month = R["month"]
    report(diff_A & np.isin(month, (10, 11)), "G1 PRIMARY Oct-Nov")
    report(diff_A & ~np.isin(month, (10, 11)), "G1 PRIMARY Dec-Jun")
    report(ctrl_fire & np.isin(month, (10, 11)), "G2 PRIMARY Oct-Nov")
    report(ctrl_fire & ~np.isin(month, (10, 11)), "G2 PRIMARY Dec-Jun")

    # zero-outside checks
    same_A = ~diff_A
    out["zero_outside"] = {
        "A_max_abs_dcrps_points_on_unchanged_rows":
            float(np.abs(R["crps_points_ctrl"][same_A] - R["crps_points_A"][same_A]).max())
            if same_A.sum() else 0.0,
        "U_max_abs_dcrps_points_on_nonfiring_rows":
            float(np.abs(R["crps_points_ctrl"][~ctrl_fire] - R["crps_points_U"][~ctrl_fire]).max())
            if (~ctrl_fire).sum() else 0.0,
    }
    print("\nzero-outside:", json.dumps(out["zero_outside"]))

    Path("data/qg_starout_gate.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote data/qg_starout_gate.json ({time.time()-t0:.0f}s)")
    print("QG_STAROUT_GATE_DONE", flush=True)


if __name__ == "__main__":
    main()
