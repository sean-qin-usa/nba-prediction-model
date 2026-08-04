#!/usr/bin/env python3
"""ARM R — FULL V3 BATTERY (GATE_POLICY_V2 §§8-11) on the pre-registered
ROLE-ACTIVE stratum, plus BH bookkeeping inputs.

Consumes the two gate row artifacts (data/ad_role_dev_rows.npz and
data/ad_role_holdout_rows.npz) and re-analyses THE SAME per-row CRPS deltas the
gate scored — no re-simulation, so the pooled numbers are bit-identical to the
gate's. What this adds is the multi-split / clustered / era inference the gate
itself does not run.

Panel unit = one scored player-game. `cluster` = player_id (the props
convention, §9.1: "player for props row-level effects"); the SEASON-clustered
CI, ICC, design effect and cluster-mean t at K-1 dof come from
`clustering_report`, and both are reported side by side as §9.1 requires.

Writes data/ad_role_v3_battery.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from nbapred.eval import splits as S

RB = {0: "NA", 1: "STABLE", 2: "PROMOTED", 3: "DEMOTED"}


def load(mode):
    z = np.load(ROOT / "data" / f"ad_role_{mode}_rows.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def to_date(ordv):
    return np.array(ordv, dtype="datetime64[D]").astype(str)


def main():
    parts = []
    for mode in ("dev", "holdout"):
        p = ROOT / "data" / f"ad_role_{mode}_rows.npz"
        if not p.exists():
            print(f"MISSING {p} — run the {mode} gate first")
            continue
        parts.append(load(mode))
    if not parts:
        sys.exit(1)
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    print(f"combined rows {len(d['y'])}  seasons {sorted(set(d['season'].tolist()))}")

    out = {}
    active = np.isin(d["rb"].astype(int), (2, 3))
    print(f"ROLE-ACTIVE rows {int(active.sum())} / {len(active)}")

    for arm in ("R", "M"):
        for lab, mask in (("ROLE-ACTIVE", active), ("ALL SCORED", np.ones(len(active), bool)),
                          ("PROMOTED", d["rb"].astype(int) == 2),
                          ("DEMOTED", d["rb"].astype(int) == 3)):
            if mask.sum() < 100:
                continue
            panel = S.Panel.from_losses(
                season=d["season"][mask],
                loss_ctrl=d["crps_ctrl"][mask],
                loss_treat=d[f"crps_{arm}"][mask],
                date=to_date(d["ord"][mask]),
                cluster=d["player_id"][mask],
                label=f"ARM {arm} points CRPS — {lab}")
            rep = S.full_report(panel)
            key = f"{arm}|{lab}"
            out[key] = rep
            if lab == "ROLE-ACTIVE" or (arm == "R"):
                print("\n" + "=" * 78)
                print(S.format_report(rep))

    # ---- R - M contrast on ROLE-ACTIVE (is the rotation SOURCE load-bearing)
    panel = S.Panel.from_losses(
        season=d["season"][active],
        loss_ctrl=d["crps_M"][active], loss_treat=d["crps_R"][active],
        date=to_date(d["ord"][active]), cluster=d["player_id"][active],
        label="ARM R minus ARM M (rotation source vs minutes-only) — ROLE-ACTIVE")
    rep = S.full_report(panel)
    out["R_minus_M|ROLE-ACTIVE"] = rep
    print("\n" + "=" * 78)
    print(S.format_report(rep))

    # ---- BH inputs: one-sided p for the primary, both inference levels
    prim = out["R|ROLE-ACTIVE"]
    cl = prim["clustering"]
    bh = dict(
        pooled_player_cluster=dict(est=prim["pooled"]["est"], lo=prim["pooled"]["lo"],
                                   hi=prim["pooled"]["hi"],
                                   p_wrongside=prim["pooled"]["p_wrongside"],
                                   se=prim["pooled"]["se"]),
        season_cluster=cl["season_cluster_boot"],
        season_mean_t=cl["season_mean_t"],
        icc_season=cl["icc_season"],
        design_effect_season=cl["design_effect_season"],
        pooled_mde80=prim["pooled_mde80"],
    )
    # cluster-mean t -> one-sided p at K-1 dof
    t = cl["season_mean_t"]
    est, K = t["est"], t["K"]
    se_t = abs(est) / t["t_stat"] if t.get("t_stat") else None
    bh["season_mean_t_p"] = t.get("p_onesided")
    out["BH_inputs"] = bh
    print("\nBH INPUTS:", json.dumps(bh, indent=2, default=float))

    (ROOT / "data" / "ad_role_v3_battery.json").write_text(
        json.dumps(out, indent=2, default=float))
    print("\nAD_ROLE_V3_DONE")


if __name__ == "__main__":
    main()
