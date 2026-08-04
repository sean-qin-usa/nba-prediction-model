#!/usr/bin/env python3
"""THE FULL V3 BATTERY (GATE_POLICY_V2 §§8-11) for the three queued gates.

Re-scores the per-row artifacts produced by scripts/qg_starout_gate.py and
scripts/qg_channel_gate.py under the mandated multi-split + clustered
inference. No model is re-run: the per-row deltas are bit-identical to what the
gates scored; only the inference moves (the operation D139 §12 / D141 §2 /
D145 §9 ran).

  d[i] = loss_ctrl[i] - loss_ARM[i]        POSITIVE = arm better
  (for the attempts Poisson LL the sign is flipped, since higher LL is better)

§9.1: the SHIPPING CI is the CLUSTERED one. Two cluster levels are mechanically
justified and BOTH are reported:
  * SEASON — the shipping CI. GATE 3's lam table is fit per scored season on
    strictly-prior seasons, so a season shares one coefficient vector; GATES 1
    and 2 have no fitted coefficient but the detector's inputs (schedule,
    roster churn) are season-shared, and §9.1 requires the season level for
    anything fit or constructed on an expanding window.
  * PLAYER — D133/D145's own convention for props row-level effects.

DEV/HOLDOUT DISCIPLINE: dev = 2023-24..2025-26 is reported first. The holdout
(2021-22..2022-23) block is printed ONLY when --holdout is passed, which the
operator does once, after a dev pass.

usage: qg_v3_battery.py {starout|channel} [--holdout]
Writes data/qg_v3_battery_<which>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from nbapred.eval.splits import (Panel, cluster_mean_t_interval, format_report,
                                 full_report, icc_oneway, mde80)

B, SEED = 2000, 20260801
DEV = ("2023-24", "2024-25", "2025-26")
HOLDOUT = ("2021-22", "2022-23")


def run(z, specs, out, tag_seasons=None, label_suffix=""):
    season = z["season"].astype(str)
    date = z["date"].astype(str)
    player = z["player_id"].astype(int)
    for mask, arm, key, label, flip in specs:
        m = np.asarray(mask, bool)
        if tag_seasons is not None:
            m = m & np.isin(season, list(tag_seasons))
        if m.sum() < 60:
            print(f"  [skip {label}{label_suffix}: n={m.sum()}]")
            continue
        a = z[f"{key}_ctrl"][m]
        b = z[f"{key}_{arm}"][m]
        d = (b - a) if flip else (a - b)
        pan = Panel(season[m], d, date[m], player[m], label + label_suffix)
        rep = full_report(pan, B, SEED)
        rep["season_cluster_pooled"] = rep["clustering"]["season_cluster_boot"]
        rep["player_cluster_boot"] = rep["pooled"]
        rep["player_cluster_t"] = cluster_mean_t_interval(d, player[m])
        rep["icc_player"] = icc_oneway(d, player[m])
        rep["n_players"] = int(len(set(player[m].tolist())))
        rep["mde80_iid"] = mde80(d)
        out[label + label_suffix] = rep
        print("\n" + "=" * 78)
        print(f"### {label}{label_suffix}")
        print("=" * 78)
        print(format_report(rep))
        cl = rep["clustering"]
        p = rep["player_cluster_boot"]
        print(f"  player-cluster (D133 convention): {p['est']:+.5f} "
              f"CI[{p['lo']:+.5f},{p['hi']:+.5f}] {'SIG' if p['sig'] else 'ns'} "
              f"(ICC_player {rep['icc_player']['icc']:+.5f}, "
              f"{rep['n_players']} players)")
        print(f"  DEFF_anova(season) {cl['icc_season']['deff']:.3f}  "
              f"DEFF_boot(season) {cl['design_effect_season']:.3f}  "
              f"MDE80_iid {rep['mde80_iid']:.5f}")


def main():
    which = sys.argv[1]
    do_hold = "--holdout" in sys.argv
    out = {}
    if which == "starout":
        z = np.load("data/qg_starout_gate_rows.npz", allow_pickle=True)
        diff_A = ((z["fired_ctrl"] != z["fired_A"])
                  | (np.abs(z["lift_ctrl"] - z["lift_A"]) > 1e-12)
                  | (np.abs(z["pm_ctrl"] - z["pm_A"]) > 1e-12))
        only_A = (z["fired_ctrl"] == 0) & (z["fired_A"] == 1)
        ctrl_f = z["fired_ctrl"] == 1
        specs = [
            (diff_A, "A", "ll", "G1 ARM A — attempts LL (PRIMARY window)", True),
            (diff_A, "A", "crps_points", "G1 ARM A — points CRPS (PRIMARY window)", False),
            (only_A, "A", "crps_points", "G1 ARM A — points CRPS (newly fired)", False),
            (only_A, "A", "ll", "G1 ARM A — attempts LL (newly fired)", True),
            (diff_A, "A", "crps_rebounds", "G1 ARM A — rebounds CRPS (PRIMARY)", False),
            (ctrl_f, "U", "crps_points", "G2 ARM U null_u — points CRPS", False),
            (ctrl_f, "U", "ll", "G2 ARM U null_u — attempts LL", True),
            (ctrl_f, "F", "crps_points", "G2 ARM F trailatt — points CRPS", False),
            (ctrl_f, "F", "ll", "G2 ARM F trailatt — attempts LL", True),
        ]
    elif which == "channel":
        z = np.load("data/qg_channel_gate_rows.npz", allow_pickle=True)
        act = z["delta"] > 0
        specs = [
            (act, "A", "crps_rebounds", "G3 ARM A — rebounds CRPS (PRIMARY)", False),
            (act, "A", "crps_assists", "G3 ARM A — assists CRPS", False),
            (act, "B", "crps_rebounds", "G3 ARM B — rebounds CRPS (PRIMARY)", False),
            (act, "B", "crps_assists", "G3 ARM B — assists CRPS", False),
            (act & (z["delta"] >= 2), "A", "crps_rebounds",
             "G3 ARM A — rebounds CRPS (delta>=2)", False),
        ]
    else:
        raise SystemExit("which must be starout|channel")

    print("#" * 78)
    print(f"# DEV SPLIT {DEV}")
    print("#" * 78)
    run(z, specs, out, DEV, " [DEV]")
    if do_hold:
        print("\n" + "#" * 78)
        print(f"# HOLDOUT SPLIT {HOLDOUT} — run ONCE, config frozen")
        print("#" * 78)
        run(z, specs, out, HOLDOUT, " [HOLDOUT]")
        print("\n" + "#" * 78)
        print("# ALL SEASONS POOLED 2021-26")
        print("#" * 78)
        run(z, specs, out, None, " [ALL]")

    Path(f"data/qg_v3_battery_{which}.json").write_text(
        json.dumps(out, indent=1, default=float))
    print(f"\nwrote data/qg_v3_battery_{which}.json")
    print("QG_V3_BATTERY_DONE", flush=True)


if __name__ == "__main__":
    main()
