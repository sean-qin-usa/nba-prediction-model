#!/usr/bin/env python3
"""D133 SHIP-CONFIRM — does the SHIPPED props.py reproduce the gated numbers?

The gate (scripts/pr_ramp_gate.py) scored ARM A by overriding proj_min inside a
local replica of simulate_player, using per-season WALK-FORWARD bias tables.
Production ships ONE frozen table (fit on the full 2019-20..2025-26 corpus), so
this re-runs the identical rows / identical seeds through the REAL
`player_rates_from_stats` + `simulate_player`, switched by PROPS_MIN_RAMP.

It also scores REBOUNDS and ASSISTS, which the gate did not: in production the
ramp lowers `rates["proj_min"]`, which reaches rebounds (Poisson(reb_per_min *
minutes)) and assists (ast_expo = clip(proj_min,10,44)) as well as points. Those
channels were NOT part of the pre-registered gate and are reported here as a
DISCLOSED non-gated measurement, not as a pass criterion.

Row enumeration/seeds are byte-identical to the gate loop, and every row is
cross-checked against data/pr_ramp_dev_rows.npz before scoring.

Read-only DB. Writes data/pr_ramp_shipconfirm.json.
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
_spec = _ilu.spec_from_file_location("pr_ramp_gate", ROOT / "scripts" / "pr_ramp_gate.py")
G = _ilu.module_from_spec(_spec); _spec.loader.exec_module(G)
DEV, SIMS = G.DEV, G.SIMS
build_index, cluster_boot, crps = G.build_index, G.cluster_boot, G.crps
fit_bias, load_corpus, pit, row_features = G.fit_bias, G.load_corpus, G.pit, G.row_features

OUT = ROOT / "data" / "pr_ramp_shipconfirm.json"
MAX_REST = 6000


def main():
    t0 = time.time()
    ref = np.load(ROOT / "data" / "pr_ramp_dev_rows.npz", allow_pickle=True)
    con = connect(read_only=True)
    df, tg = load_corpus(con)
    byp, tsched = build_index(df, tg)
    all_seasons = sorted(set(df["season"]))
    ra = con.execute("""
        SELECT s.player_id, g.game_date, s.oreb + s.dreb AS reb, s.ast
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '002%' AND s.seconds >= 720""").fetchdf()
    ra["ord"] = ra["game_date"].astype("datetime64[ns]").values.astype("datetime64[D]").astype(int)
    RA = {(int(a), int(b)): (float(c), float(d))
          for a, b, c, d in zip(ra["player_id"], ra["ord"], ra["reb"], ra["ast"])}
    for s in DEV:                                   # same fits, same order
        fit_bias(df, byp, tsched, [x for x in all_seasons if x < s])
    print(f"corpus {len(df)}; reference rows {len(ref['player_id'])}", flush=True)

    rows, i, mismatch = [], 0, 0
    for season in DEV:
        sub = df[df["season"] == season]
        mon = sub["game_date"].astype("datetime64[ns]").dt.month
        on = sub[mon.isin((10, 11))]
        rest = sub[~mon.isin((10, 11))]
        if len(rest) > MAX_REST // len(DEV):
            rest = rest.iloc[::max(1, len(rest) // (MAX_REST // len(DEV)))]
        for r in list(on.itertuples()) + list(rest.itertuples()):
            f = row_features(byp, tsched, int(r.player_id), r.ord, season, r.team_id)
            if f is None:
                continue
            proj_fast, nh, gp, tgp = f
            if nh < 8 or proj_fast < 20:
                continue
            os.environ["PROPS_MIN_RAMP"] = "0"
            off = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
            if off is None or off["n_games"] < 8 or off["proj_min"] < 20:
                continue
            # index i is now exactly the gate's seed for this row
            if (int(ref["player_id"][i]) != int(r.player_id)
                    or float(ref["y"][i]) != float(r.pts)):
                mismatch += 1
            if int(r.game_date.month) in (10, 11):
                os.environ["PROPS_MIN_RAMP"] = "1"
                onr = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
                rec = {"player_id": int(r.player_id), "season": season,
                       "month": int(r.game_date.month), "gp": gp, "seed": i,
                       "y": float(r.pts), "yreb": RA[(int(r.player_id), int(r.ord))][0],
                       "yast": RA[(int(r.player_id), int(r.ord))][1],
                       "ymin": float(r.mins),
                       "proj_off": float(off["proj_min"]),
                       "proj_on": float(onr["proj_min"]),
                       "gate_crps_ctrl": float(ref["crps_ctrl"][i]),
                       "gate_crps_A": float(ref["crps_A"][i])}
                prng = np.random.default_rng(10_000 + i)
                for arm, rr in (("off", off), ("on", onr)):
                    sim = simulate_player(rr, SIMS, seed=i)
                    for tgt, key, yv in (("pts", "points", rec["y"]),
                                         ("reb", "rebounds", rec["yreb"]),
                                         ("ast", "assists", rec["yast"])):
                        s = sim[key]
                        rec[f"{tgt}crps_{arm}"] = crps(s, yv)
                        rec[f"{tgt}pit_{arm}"] = pit(s, yv, prng)
                        rec[f"{tgt}mae_{arm}"] = abs(float(s.mean()) - yv)
                rows.append(rec)
                if len(rows) % 1000 == 0:
                    print(f"  {len(rows)} rows ({time.time()-t0:.0f}s)", flush=True)
            i += 1
    con.close()
    os.environ.pop("PROPS_MIN_RAMP", None)
    print(f"enumerated {i} rows, scored {len(rows)} Oct-Nov, "
          f"row-identity mismatches vs gate npz: {mismatch}", flush=True)

    players = np.array([r["player_id"] for r in rows])
    seas = np.array([r["season"] for r in rows])
    month = np.array([r["month"] for r in rows])
    out = {"n": len(rows), "mismatch": mismatch, "sims": SIMS, "strata": {}}

    # control replication: shipped OFF must equal the gate's ctrl arm bitwise
    dctl = np.array([r["ptscrps_off"] for r in rows]) - np.array(
        [r["gate_crps_ctrl"] for r in rows])
    out["control_repro_max_abs"] = float(np.abs(dctl).max())
    print(f"CONTROL REPLICATION (shipped OFF vs gate ctrl): max|d| "
          f"{np.abs(dctl).max():.3e}")

    def report(mask, label):
        if mask.sum() < 30:
            return
        blk = {"n": int(mask.sum()), "players": int(len(set(players[mask])))}
        print(f"\n--- {label} (n={mask.sum()}, {blk['players']} players) ---")
        for tgt in ("pts", "reb", "ast"):
            for m in ("crps", "mae"):
                d = (np.array([r[f"{tgt}{m}_off"] for r in rows])
                     - np.array([r[f"{tgt}{m}_on"] for r in rows]))[mask]
                pt_, lo, hi, se = cluster_boot(d, players[mask])
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
        # shipped-vs-gated ARM A agreement on points
        gate_d = (np.array([r["gate_crps_ctrl"] for r in rows])
                  - np.array([r["gate_crps_A"] for r in rows]))[mask]
        blk["gate_arm_a_delta"] = float(gate_d.mean())
        blk["ship_vs_gate_ratio"] = (blk["pts_crps"]["delta"] / gate_d.mean()
                                     if gate_d.mean() else None)
        print(f"  gated ARM A on same rows {gate_d.mean():+.5f} -> shipped "
              f"reproduces {100*blk['ship_vs_gate_ratio']:.1f}%")
        out["strata"][label] = blk

    report(np.ones(len(rows), bool), "OCT+NOV (PRIMARY)")
    report(month == 10, "OCT only")
    for s in DEV:
        report(seas == s, f"OCT+NOV {s}")
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {OUT} ({time.time()-t0:.0f}s)\nPR_RAMP_SHIPCONFIRM_DONE", flush=True)


if __name__ == "__main__":
    main()
