#!/usr/bin/env python3
"""cg_kalman_clean.py — RE-RUN #2 of the contaminated-gates audit.

The D12 verdict ("Kalman beats EWMA standalone; WASH in props") and the D57
re-gate ("Kalman stands") were both produced on a poisoned eval universe:

  * scripts/ablate_kalman_props.py ran 2026-07-28, and
    scripts/audit_kalman_720.py ran 2026-07-30 03:59 — BOTH before the D79 fix
    (props.py, 2026-07-30 21:37) that added `AND game_id LIKE '002%'` to the
    rate queries.  audit_kalman_720.py says so in its own docstring: "History is
    NOT filtered by game type (preseason/playoffs are absorbed) because props.py
    does not filter it either."  cg_forensics: 21,177 of 201,904 stat rows
    (10.5%) are non-regular-season, and 79.8% of players carried non-002 games
    inside the trailing-20 window at Nov 1.
  * D79 ALSO fixed the Kalman forward step, which was literally `predict(0)` —
    a no-op.  Every pre-D79 Kalman number is therefore the un-forwarded filter,
    i.e. not the estimator the design specifies and not the one shipped today.

This script re-runs the estimator comparison with both fixes in force, reusing
audit_kalman_720's own recursions (imported, not copied) so the arithmetic is
identical and its --verify fidelity check still applies.

ARMS (state carried in one pass; every arm sees only games strictly earlier)
  ewma720_002   EWMA hl=10, seconds>=720, 002-only        <- CTL, shipped today
  ewma720_all   EWMA hl=10, seconds>=720, any game type   <- pre-D79 incumbent
  kal720f_002   Kalman on the ALIGNED >=720 002 universe, WITH the forward
                step                                       <- PRIMARY challenger
  kal720_002    same, forward step suppressed (pre-D79 no-op behaviour)
  kal0f_002     Kalman on seconds>0, 002-only, with forward step
                                                           <- literally shipped
  kal0_all      Kalman on seconds>0, any game type, no forward step
                                                           <- pre-D79 shipped
  career720_002 minutes-weighted career-to-date mean       <- anchor diagnostic

PRE-REGISTERED PRIMARY: ewma720_002 - kal720f_002 (estimator isolated: same
universe, same 002 filter, Kalman with its designed forward step).  Positive =
Kalman better.  Ship bar = CI excludes zero (G1).
SECONDARY (the live question): ewma720_002 - kal0f_002.

Metric: minutes-weighted MAE of one-step-ahead per-minute attempt rates,
summed over rim/mid/thr.  Paired bootstrap 2000x clustered by player.
Read-only.  Writes data/cg_kalman_clean.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np

import nbapred.db as _db

if os.environ.get("CG_DB"):
    _db.DB_PATH = Path(os.environ["CG_DB"])

from nbapred.db import connect                                    # noqa: E402
from audit_kalman_720 import (COL, MIN_HIST, MIN_SEC, SEASONS,    # noqa: E402
                              ZONES, EwmaState, KalmanState,
                              cluster_boot)

OUT = Path("data/cg_kalman_clean.json")

ARMS = ("ewma720_002", "ewma720_all", "kal720f_002", "kal720_002",
        "kal0f_002", "kal0_all", "career720_002")


def build(con):
    df = con.execute("""
        SELECT s.player_id, g.game_date, g.season, s.game_id, s.seconds,
               s.rima, s.mida, s.thra
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g USING (game_id)
        ORDER BY s.player_id, g.game_date, s.game_id
    """).fetchdf()
    df = df[df["seconds"].notna()]
    df["ordinal"] = df["game_date"].map(lambda d: d.toordinal())

    rows = []
    for pid, grp in df.groupby("player_id", sort=False):
        e002, eall = EwmaState(), EwmaState()
        k720, k0_002, k0_all = KalmanState(), KalmanState(), KalmanState()
        secs = grp["seconds"].to_numpy()
        mins_all = secs / 60.0
        ords = grp["ordinal"].to_numpy()
        cnts = {z: grp[COL[z]].to_numpy().astype(float) for z in ZONES}
        seasons = grp["season"].to_numpy()
        gids = grp["game_id"].to_numpy()
        nrow = len(grp)

        i = 0
        while i < nrow:
            j = i
            while j < nrow and ords[j] == ords[i]:    # score the whole date block
                j += 1                                # BEFORE absorbing any of it
            for t in range(i, j):
                if secs[t] < MIN_SEC:
                    continue
                if not (str(gids[t]).startswith("002") and seasons[t] in SEASONS):
                    continue
                # both EWMA arms must be defined so the comparison is paired on
                # identical rows (props.py returns None below 3 history games)
                if e002.n < MIN_HIST or eall.n < MIN_HIST:
                    continue
                m_t = mins_all[t]
                o_t = int(ords[t])
                pred = {"ewma720_002": e002.rates(),
                        "ewma720_all": eall.rates(),
                        "kal720f_002": k720.theta_fwd(o_t),
                        "kal720_002": k720.theta(),
                        "kal0f_002": k0_002.theta_fwd(o_t),
                        "kal0_all": k0_all.theta(),
                        "career720_002": k720.career()}
                rec = {"player_id": pid, "season": seasons[t], "minutes": m_t,
                       "n_games": e002.n, "n_games_all": eall.n,
                       "proj_min": e002.proj_min(),
                       "S_kal720": float(np.mean([k720.S[z] for z in ZONES]))}
                for z in ZONES:
                    y = cnts[z][t] / m_t
                    rec[f"y_{z}"] = y
                    for arm in ARMS:
                        rec[f"{arm}_{z}"] = abs(pred[arm][z] - y)
                rows.append(rec)
            for t in range(i, j):
                c = {z: cnts[z][t] for z in ZONES}
                reg = str(gids[t]).startswith("002")
                if secs[t] >= MIN_SEC:
                    eall.absorb(mins_all[t], c)
                    if reg:
                        e002.absorb(mins_all[t], c)
                        k720.absorb(int(ords[t]), mins_all[t], c)
                if secs[t] > 0:
                    k0_all.absorb(int(ords[t]), mins_all[t], c)
                    if reg:
                        k0_002.absorb(int(ords[t]), mins_all[t], c)
            i = j
    return rows


COMPS = [
    ("PRIMARY  clean estimator   ewma720_002 - kal720f_002", "ewma720_002", "kal720f_002"),
    ("SECOND   as shipped today  ewma720_002 - kal0f_002  ", "ewma720_002", "kal0f_002"),
    ("DIAG     D79 on incumbent  ewma720_all - ewma720_002", "ewma720_all", "ewma720_002"),
    ("DIAG     D79+fwd on kalman kal0_all    - kal720f_002", "kal0_all", "kal720f_002"),
    ("DIAG     forward step      kal720_002  - kal720f_002", "kal720_002", "kal720f_002"),
    ("DIAG     universe (002)    kal0f_002   - kal720f_002", "kal0f_002", "kal720f_002"),
    ("DIAG     filter vs anchor  career720_002 - kal720f_002", "career720_002", "kal720f_002"),
    ("DIAG     metric has signal career720_002 - ewma720_002", "career720_002", "ewma720_002"),
]


def report(rows, label, out):
    if not rows:
        print(f"\n[{label}] no rows"); return
    pid = np.array([r["player_id"] for r in rows])
    w = np.array([r["minutes"] for r in rows])
    err = {(arm, z): np.array([r[f"{arm}_{z}"] for r in rows]) * w
           for arm in ARMS for z in ZONES}
    print(f"\n{'='*80}\n[{label}] player-games={len(rows)} players={len(set(pid))} "
          f"minutes={w.sum():,.0f}")
    print(f"  {'arm':<15}" + "".join(f"{z:>10}" for z in ZONES) + f"{'all3':>10}")
    tot = {}
    blk = {"n": len(rows), "players": int(len(set(pid))), "wmae": {}, "comps": {}}
    for arm in ARMS:
        vals = [err[(arm, z)].sum() / w.sum() for z in ZONES]
        tot[arm] = sum(vals)
        blk["wmae"][arm] = {z: float(v) for z, v in zip(ZONES, vals)}
        blk["wmae"][arm]["all3"] = float(tot[arm])
        print(f"  {arm:<15}" + "".join(f"{v:>10.5f}" for v in vals)
              + f"{tot[arm]:>10.5f}")
    print(f"  {'-'*76}\n  paired bootstrap 2000x clustered by player; "
          f"+ = SECOND arm better")
    for name, a, b in COMPS:
        na = sum(err[(a, z)] for z in ZONES)
        nb = sum(err[(b, z)] for z in ZONES)
        pt, lo, hi, P = cluster_boot(pid, w, na, nb)
        sig = "SIG" if (lo > 0 or hi < 0) else "ns "
        rel = 100.0 * pt / tot[a] if tot[a] else 0.0
        blk["comps"][f"{a}-{b}"] = dict(delta=float(pt), lo=float(lo),
                                        hi=float(hi), rel_pct=float(rel), sig=sig)
        print(f"  {name}: {pt:+.5f} ({rel:+.2f}%) CI [{lo:+.5f},{hi:+.5f}] {sig}")
    out[label] = blk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    con = connect(read_only=True)
    print("building rate paths (PIT: history strictly < target date) ...", flush=True)
    rows = build(con)
    con.close()
    print(f"scored targets: {len(rows)}  ({time.time()-t0:.0f}s)", flush=True)

    out = {}
    report([r for r in rows if r["season"] == "2025-26"], "2025-26 (PRIMARY)", out)
    for s in ("2023-24", "2024-25"):
        report([r for r in rows if r["season"] == s], f"{s} (secondary)", out)
    report(rows, "POOLED 2023-26", out)
    report([r for r in rows if r["n_games"] >= 8 and r["proj_min"] >= 20],
           "POOLED under the ORIGINAL ablation gate (n>=8, proj_min>=20)", out)
    Path(a.json).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.json}  ({time.time()-t0:.0f}s)")
    print("CG_KALMAN_DONE", flush=True)


if __name__ == "__main__":
    main()
