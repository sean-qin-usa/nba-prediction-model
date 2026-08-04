#!/usr/bin/env python3
"""PROPS ABSENCE RAMP — the GATE_POLICY_V2 (=V3) §§8-11 battery.

Re-scores the artifact produced by scripts/ab_props_gate.py under the mandated
multi-split + clustered inference. No model is re-run: the per-row CRPS deltas
are bit-identical to what the gate scored, only the inference moves (exactly
the operation D139 §12 / D141 §2 ran).

  d[i] = crps_ctrl[i] - crps_ARM[i]   (POSITIVE = arm better)

§9.1 requires the SHIPPING CI to be the CLUSTERED one. Two cluster levels are
mechanically justified here and BOTH are reported:
  * SEASON  — the b table is fit per scored season on strictly-prior seasons,
              so every row inside a season shares one coefficient vector. This
              is the §9 shipping CI.
  * PLAYER  — D133's own convention for props row-level effects; reported for
              direct comparability with D133's published numbers.

Writes data/ab_v3_battery.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from nbapred.eval.splits import (Panel, cluster_mean_t_interval, format_report,
                                 full_report, icc_oneway, mde80,
                                 paired_bootstrap)

B, SEED = 2000, 20260801


def main():
    z = np.load("data/ab_props_gate_rows.npz", allow_pickle=True)
    miss = z["miss10"].astype(int)
    season = z["season"].astype(str)
    date = z["date"].astype(str)
    player = z["player_id"].astype(int)
    gp = z["gp"].astype(int)
    win = miss >= 5
    out = {}

    def battery(mask, arm, label):
        d = (z["crps_ctrl"] - z[f"crps_{arm}"])[mask]
        # cluster=PLAYER on the Panel (D133 convention) so every SPLIT fold
        # gets a real CI; the SEASON-clustered shipping CI comes from
        # clustering_report(), which always clusters on panel.season.
        pan_s = Panel(season[mask], d, date[mask], player[mask], label)
        rep = full_report(pan_s, B, SEED)
        # player-cluster pooled (D133 convention) alongside
        rep["season_cluster_pooled"] = rep["clustering"]["season_cluster_boot"]
        rep["player_cluster_boot"] = rep["pooled"]
        rep["player_cluster_t"] = cluster_mean_t_interval(d, player[mask])
        rep["icc_player"] = icc_oneway(d, player[mask])
        rep["n_players"] = int(len(set(player[mask].tolist())))
        rep["mde80_iid"] = mde80(d)
        out[label] = rep
        print("\n" + "=" * 78)
        print(f"### {label}")
        print("=" * 78)
        print(format_report(rep))
        cl = rep["clustering"]
        print(f"  player-cluster (D133 convention): "
              f"{rep['player_cluster_boot']['est']:+.5f} "
              f"CI[{rep['player_cluster_boot']['lo']:+.5f},"
              f"{rep['player_cluster_boot']['hi']:+.5f}] "
              f"{'SIG' if rep['player_cluster_boot']['sig'] else 'ns'} "
              f"(ICC_player {rep['icc_player']['icc']:+.5f}, "
              f"{rep['n_players']} players)")
        print(f"  DEFF_anova(season) {cl['icc_season']['deff']:.3f}  "
              f"DEFF_boot(season) {cl['design_effect_season']:.3f}  "
              f"MDE80_iid {rep['mde80_iid']:.5f}")
        return rep

    battery(win, "A", "ARM A — PRIMARY (miss10>=5)")
    battery(win & (gp >= 20), "A", "ARM A — gp>=20 (D133-inert region)")
    battery(win, "A0", "ARM A0 — adversarial level control")
    battery(win, "C", "ARM C — absence axis REPLACING the gp ramp")

    Path("data/ab_v3_battery.json").write_text(json.dumps(out, indent=1, default=float))
    print("\nwrote data/ab_v3_battery.json")


if __name__ == "__main__":
    main()
