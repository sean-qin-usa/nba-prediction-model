#!/usr/bin/env python3
"""PROPS ABSENCE RAMP — SHIP-CONFIRM (GATE_POLICY_V2 §6.6 IMPLEMENTATION
IDENTITY: prove the code being shipped is the code that was gated).

D141 hall-of-shame 15: "a switch named after a hypothesis is not evidence that
it implements it." So this script does not inspect the switch — it RUNS it, on
the gate's own rows with the gate's own seeds, and checks three things:

  (1) SHIPPED-OFF (PROPS_ABSENCE_RAMP=0) must reproduce the gate's CONTROL
      **BITWISE** (max|dCRPS| == 0 over every scored row).
  (2) The shipped code's applied correction must land in the same ABSENCE
      BUCKET as the gate's, row by row (0 mismatches). The magnitudes differ by
      design: the gate scored WALK-FORWARD per-season tables, production ships
      the full-corpus fit — the D133 convention.
  (3) SHIPPED-ON must reproduce the gated ARM A estimate with the same sign and
      >= 50% of its magnitude (the V2 confirm rule).

Read-only DB. Writes data/ab_shipconfirm.json.
"""
from __future__ import annotations

import datetime as dt
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
from nbapred.engine import props as P
from pr_ramp_gate import BOOT_SEED, SIMS, cluster_boot, crps, pit, simulate_ramp


def bucket(m):
    return 0 if m < 5 else (1 if m < 8 else 2)


def main():
    t0 = time.time()
    z = np.load("data/ab_props_gate_rows.npz", allow_pickle=True)
    n = len(z["player_id"])
    pids = z["player_id"].astype(int)
    dates = [dt.date.fromisoformat(str(x)) for x in z["date"]]
    miss_gate = z["miss10"].astype(int)
    y = z["y"].astype(float)
    ymin = z["ymin"].astype(float)
    con = connect(read_only=True)

    res = {}
    for tag, env in (("off", "0"), ("on", "1")):
        os.environ["PROPS_ABSENCE_RAMP"] = env
        assert P.absence_ramp(9) == (0.0 if env == "0" else 2.987)
        c = np.empty(n)
        pm = np.empty(n)
        pt = np.empty(n)
        for i in range(n):
            r = P.player_rates_from_stats(con, int(pids[i]), before=dates[i])
            assert r is not None, f"row {i} lost its rates"
            pmv = float(r["proj_min"])
            pts, mn = simulate_ramp(r, SIMS, seed=i)
            c[i] = crps(pts, y[i])
            pm[i] = pmv
            pt[i] = pit(pts, y[i], np.random.default_rng(10_000 + i))
            if i % 3000 == 0:
                print(f"  {tag} {i}/{n} ({time.time()-t0:.0f}s)", flush=True)
        res[tag] = dict(crps=c, proj=pm, pit=pt)
        print(f"pass {tag} done ({time.time()-t0:.0f}s)", flush=True)
    con.close()
    os.environ.pop("PROPS_ABSENCE_RAMP", None)

    out = {}
    # ---- (1) shipped-OFF == gate control, BITWISE -------------------------
    d_off = np.abs(res["off"]["crps"] - z["crps_ctrl"])
    out["control_bitwise_max_abs_dcrps"] = float(d_off.max())
    out["control_rows_mismatched"] = int((d_off > 0).sum())
    print(f"\n(1) shipped-OFF vs gate control: max|dCRPS| {d_off.max():.3e}, "
          f"{int((d_off > 0).sum())}/{n} rows mismatched")

    # ---- (2) row identity: applied bucket must match the gate's -----------
    applied = res["off"]["proj"] - res["on"]["proj"]
    b_ship = np.array([P.MISS_RAMP[bucket(m)][1] for m in miss_gate])
    bk_ship = np.array([bucket(m) for m in miss_gate])
    bk_run = np.select([np.isclose(applied, 0.0),
                        np.isclose(applied, 0.858),
                        np.isclose(applied, 2.987)], [0, 1, 2], default=-1)
    mism = int((bk_run != bk_ship).sum())
    out["row_identity_bucket_mismatches"] = mism
    out["applied_matches_ship_table"] = int(np.isclose(applied, b_ship).sum())
    print(f"(2) row identity: {mism}/{n} bucket mismatches; "
          f"{int(np.isclose(applied, b_ship).sum())}/{n} rows carry exactly the "
          f"shipped table value")
    out["zero_outside_window_max_applied"] = float(
        np.abs(applied[miss_gate <= 4]).max())
    print(f"    zero-outside-window: max applied on miss10<=4 = "
          f"{np.abs(applied[miss_gate <= 4]).max():.3e}")

    # ---- (3) shipped estimate vs gated estimate ---------------------------
    win = miss_gate >= 5
    d_ship = (res["off"]["crps"] - res["on"]["crps"])[win]
    d_gate = (z["crps_ctrl"] - z["crps_A"])[win]
    a, lo, hi, se = cluster_boot(d_ship, pids[win])
    g = float(d_gate.mean())
    out["shipped"] = dict(est=a, lo=lo, hi=hi, se=se, n=int(win.sum()),
                          sig="SIG" if (lo > 0 or hi < 0) else "ns")
    out["gated"] = dict(est=g)
    out["reproduction_pct"] = 100.0 * a / g
    print(f"\n(3) SHIPPED ARM A on the window: {a:+.5f} CI[{lo:+.5f},{hi:+.5f}] "
          f"{out['shipped']['sig']}  vs GATED {g:+.5f}  -> "
          f"{100*a/g:.1f}% of the gated estimate")
    out["pit"] = {"ctrl": float(res["off"]["pit"][win].mean()),
                  "ship": float(res["on"]["pit"][win].mean())}
    print(f"    PIT on window: {out['pit']['ctrl']:.4f} -> {out['pit']['ship']:.4f}")
    for lab, m in (("gp>=20", win & (z["gp"].astype(int) >= 20)),
                   ("miss10 8-10", miss_gate >= 8),
                   ("miss10 5-7", (miss_gate >= 5) & (miss_gate <= 7))):
        dd = (res["off"]["crps"] - res["on"]["crps"])[m]
        aa, l2, h2, _ = cluster_boot(dd, pids[m])
        out[f"shipped_{lab}"] = dict(est=aa, lo=l2, hi=h2, n=int(m.sum()))
        print(f"    {lab:12s} {aa:+.5f} CI[{l2:+.5f},{h2:+.5f}] n={int(m.sum())}")

    out["verdict"] = ("PASS" if (out["control_rows_mismatched"] == 0
                                 and mism == 0
                                 and np.sign(a) == np.sign(g)
                                 and abs(a) >= 0.5 * abs(g)) else "FAIL")
    print(f"\n§6.6 IMPLEMENTATION IDENTITY: {out['verdict']}")
    Path("data/ab_shipconfirm.json").write_text(json.dumps(out, indent=1, default=float))
    print("wrote data/ab_shipconfirm.json")


if __name__ == "__main__":
    main()
