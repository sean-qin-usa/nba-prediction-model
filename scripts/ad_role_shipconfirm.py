#!/usr/bin/env python3
"""ARM R SHIP-CONFIRM — does the SHIPPED props.py reproduce the gated numbers?

GATE_POLICY §6.6 (D141's sixth condition) demands: diff the code the gate RAN
against the code the production switch REACHES, and prove a same-run control
reproduces the gate numbers bitwise.

The gate (scripts/ad_role_gate.py) scored ARM R by overriding proj_min inside a
local BITWISE replica of `simulate_player`, using per-season WALK-FORWARD
coefficient tables. Production ships ONE frozen table, applied INSIDE
`player_rates_from_stats`, switched by `PROPS_ROLE_STATE`. This re-runs the
identical rows with the identical seeds through the REAL production functions:

  PROPS_ROLE_STATE=0  must reproduce the gate's `ctrl` arm BITWISE
  PROPS_ROLE_STATE=1  is the shipped arm, compared to the gated ARM R

DISCLOSED NON-GATED CHANNELS: production `proj_min` also feeds rebounds and
assists. Both are measured here and reported, exactly as D133 did; they were
not part of the pre-registered gate and are not a pass criterion.

Read-only DB. Writes data/ad_role_shipconfirm.json.
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

import numpy as np

from nbapred.db import connect
from nbapred.engine.props import player_rates_from_stats, simulate_player
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("ad_role_gate", ROOT / "scripts" / "ad_role_gate.py")
G = _ilu.module_from_spec(_spec); _spec.loader.exec_module(G)

OUT = ROOT / "data" / "ad_role_shipconfirm.json"
RBMAP = {"NA": 0, "STABLE": 1, "PROMOTED": 2, "DEMOTED": 3}


def main():
    t0 = time.time()
    ref = np.load(ROOT / "data" / "ad_role_dev_rows.npz", allow_pickle=True)
    con = connect(read_only=True)
    df = G.load_corpus(con)
    roles = G.load_roles()
    byp = G.build_index(df, roles)
    all_seasons = sorted(set(df["season"]))
    ra = con.execute("""
        SELECT s.player_id, g.game_date, s.oreb + s.dreb AS reb, s.ast
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '002%' AND s.seconds >= 720""").fetchdf()
    ra["ord"] = ra["game_date"].astype("datetime64[ns]").values.astype(
        "datetime64[D]").astype(int)
    RA = {(int(a), int(b)): (float(c), float(d))
          for a, b, c, d in zip(ra["player_id"], ra["ord"], ra["reb"], ra["ast"])}

    fits = {}
    for s in G.DEV:
        fits[s] = G.fit_bias(df, byp, [x for x in all_seasons if x < s])
    print(f"corpus {len(df)}; reference rows {len(ref['player_id'])}", flush=True)

    rows, i, mismatch = [], 0, 0
    for season in G.DEV:
        meta, bR, bM = fits[season]
        sub = df[df["season"] == season]
        cands, inact = [], []
        for r in sub.itertuples():
            st = G.row_state(byp, int(r.player_id), r.ord, season)
            if st is None:
                continue
            proj, nh, gp, rb, mb = st
            if nh < 8 or proj < 20:
                continue
            (cands if rb in ("PROMOTED", "DEMOTED") else inact).append((r, rb, mb))
        cap = 8000 // len(G.DEV)
        if len(inact) > cap:
            inact = inact[::max(1, len(inact) // cap)][:cap]
        for r, rb, mb in cands + inact:
            os.environ["PROPS_ROLE_STATE"] = "0"
            off = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
            if off is None or off["n_games"] < 8 or off["proj_min"] < 20:
                continue
            if (int(ref["player_id"][i]) != int(r.player_id)
                    or float(ref["y"][i]) != float(r.pts)
                    or int(ref["rb"][i]) != RBMAP[rb]):
                mismatch += 1
            os.environ["PROPS_ROLE_STATE"] = "1"
            onr = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
            rec = {"player_id": int(r.player_id), "season": season, "seed": i,
                   "rb": RBMAP[rb], "y": float(r.pts),
                   "yreb": RA[(int(r.player_id), int(r.ord))][0],
                   "yast": RA[(int(r.player_id), int(r.ord))][1],
                   "ymin": float(r.mins),
                   "proj_off": float(off["proj_min"]),
                   "proj_on": float(onr["proj_min"]),
                   "gate_crps_ctrl": float(ref["crps_ctrl"][i]),
                   "gate_crps_R": float(ref["crps_R"][i])}
            prng = np.random.default_rng(10_000 + i)
            for arm, rr in (("off", off), ("on", onr)):
                sim = simulate_player(rr, G.SIMS, seed=i)
                for tgt, key, yv in (("pts", "points", rec["y"]),
                                     ("reb", "rebounds", rec["yreb"]),
                                     ("ast", "assists", rec["yast"])):
                    s = sim[key]
                    rec[f"{tgt}crps_{arm}"] = G.crps(s, yv)
                    rec[f"{tgt}pit_{arm}"] = G.pit(s, yv, prng)
                    rec[f"{tgt}mae_{arm}"] = abs(float(s.mean()) - yv)
            rows.append(rec)
            i += 1
            if len(rows) % 1000 == 0:
                print(f"  {len(rows)} rows ({time.time()-t0:.0f}s)", flush=True)
    con.close()
    os.environ.pop("PROPS_ROLE_STATE", None)
    print(f"enumerated {i} rows, row-identity mismatches vs gate npz: {mismatch}",
          flush=True)

    players = np.array([r["player_id"] for r in rows])
    seas = np.array([r["season"] for r in rows])
    rb = np.array([r["rb"] for r in rows])
    out = {"n": len(rows), "mismatch": mismatch, "sims": G.SIMS, "strata": {}}

    dctl = (np.array([r["ptscrps_off"] for r in rows])
            - np.array([r["gate_crps_ctrl"] for r in rows]))
    out["control_repro_max_abs"] = float(np.abs(dctl).max())
    print(f"CONTROL REPLICATION (shipped OFF vs gate ctrl): max|d| "
          f"{np.abs(dctl).max():.3e}")
    dproj_off = np.abs(np.array([r["proj_off"] for r in rows])
                       - np.array([r["proj_on"] for r in rows]))
    inact = ~np.isin(rb, (2, 3))
    out["proj_unchanged_outside_window_max"] = float(dproj_off[inact].max())
    print(f"proj_min max|change| on STABLE+NA rows: {dproj_off[inact].max():.3e}")

    def report(mask, label):
        if mask.sum() < 30:
            return
        blk = {"n": int(mask.sum()), "players": int(len(set(players[mask])))}
        print(f"\n--- {label} (n={mask.sum()}, {blk['players']} players) ---")
        for tgt in ("pts", "reb", "ast"):
            for m in ("crps", "mae"):
                d = (np.array([r[f"{tgt}{m}_off"] for r in rows])
                     - np.array([r[f"{tgt}{m}_on"] for r in rows]))[mask]
                pt_, lo, hi, se = G.cluster_boot(d, players[mask])
                base = float(np.array([r[f"{tgt}{m}_off"] for r in rows])[mask].mean())
                sig = "SIG" if (lo > 0 or hi < 0) else "ns"
                blk[f"{tgt}_{m}"] = dict(delta=pt_, lo=lo, hi=hi, se=se, sig=sig,
                                         rel_pct=100 * pt_ / base if base else 0.0)
                print(f"  {tgt} {m:4s} {pt_:+.5f} ({100*pt_/base:+.3f}%) "
                      f"CI[{lo:+.5f},{hi:+.5f}] {sig}")
            for a in ("off", "on"):
                blk[f"{tgt}_pit_{a}"] = float(
                    np.array([r[f"{tgt}pit_{a}"] for r in rows])[mask].mean())
            print(f"  {tgt} PIT {blk[f'{tgt}_pit_off']:.4f} -> {blk[f'{tgt}_pit_on']:.4f}")
        gate_d = (np.array([r["gate_crps_ctrl"] for r in rows])
                  - np.array([r["gate_crps_R"] for r in rows]))[mask]
        blk["gate_arm_R_delta"] = float(gate_d.mean())
        blk["ship_vs_gate_ratio"] = (blk["pts_crps"]["delta"] / gate_d.mean()
                                     if gate_d.mean() else None)
        print(f"  gated ARM R on same rows {gate_d.mean():+.5f} -> shipped "
              f"{blk['pts_crps']['delta']:+.5f}  "
              f"({100*blk['ship_vs_gate_ratio']:.1f}% of gated)"
              if gate_d.mean() else "")
        out["strata"][label] = blk

    report(np.isin(rb, (2, 3)), "ROLE-ACTIVE (PRIMARY)")
    report(rb == 2, "PROMOTED")
    report(rb == 3, "DEMOTED")
    report(~np.isin(rb, (2, 3)), "STABLE+NA (must be exactly 0)")
    report(np.ones(len(rows), bool), "ALL SCORED pooled")
    for s in G.DEV:
        report(np.isin(rb, (2, 3)) & (seas == s), f"ROLE-ACTIVE {s}")

    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {OUT}")
    print("AD_ROLE_SHIPCONFIRM_DONE", flush=True)


if __name__ == "__main__":
    main()
