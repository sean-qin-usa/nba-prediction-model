#!/usr/bin/env python3
"""GATE_POLICY_V2 §6.6 SHIP-CONFIRM for the star-out gates (1 and 2).

The gate harness already runs the SHIPPED `starout.team_context` under the env
switch, so implementation identity of the ARM is structural. What ship-confirm
adds is the thing D141's hall-of-shame 15 actually demands: after a DEFAULT is
flipped in the module, verify that

  (i)  with the env var UNSET, the module now reproduces the gated ARM exactly
       (same firing set, same star, same lift, same applied proj_min), and
  (ii) with the env var set back to the OLD value, it reproduces the gate's
       CONTROL exactly.

Both are checked against data/qg_starout_ctx.pkl — the detection artifact the
gate actually scored — on a stride sample of the gate's own team-dates.

usage: qg_starout_shipconfirm.py --arm {A|U} [--n 400]
Writes data/qg_starout_shipconfirm_<arm>.json
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from nbapred.db import connect
from nbapred.engine import starout

# which env var each arm is controlled by, and the OLD (pre-ship) value
ARM_VAR = {"A": ("STAROUT_TRAIL", "current"), "U": ("STAROUT_USAGE", "softmax")}


def ctx_key(c):
    if c is None:
        return None
    return (int(c["star"]), round(float(c["lift"]), 12),
            round(float(c["lift_softmax"]), 12), int(c["n_pool"]),
            str(c["usage_source"]))


def main():
    arm = sys.argv[sys.argv.index("--arm") + 1]
    n_probe = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 400
    var, old = ARM_VAR[arm]
    blob = pickle.loads((ROOT / "data" / "qg_starout_ctx.pkl").read_bytes())
    ctxs = blob["ctxs"]
    con = connect(read_only=True)
    weights = starout.load_usage_weights()
    positions = starout.load_positions()

    # rebuild the out-sets exactly as the gate did (same screen)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qg_gate", ROOT / "scripts" / "qg_starout_gate.py")
    print("re-deriving candidate out-sets ...", flush=True)
    pg = con.execute("""
        SELECT s.team_id, s.player_id, s.seconds/60.0 AS m, g.game_date, g.season
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games
              WHERE wl IS NOT NULL) g USING (game_id)
        WHERE s.game_id LIKE '002%'""").fetchdf()
    pg["game_date"] = [d.date() if hasattr(d, "date") else d for d in pg["game_date"]]
    per = defaultdict(dict)
    for r in pg.itertuples():
        per[(r.season, int(r.team_id))].setdefault(int(r.player_id), []).append(
            (r.game_date, float(r.m or 0.0)))

    keys = sorted(set(ctxs["ctrl"]) | set(ctxs[arm]))
    step = max(1, len(keys) // n_probe)
    keys = keys[::step]
    t0 = time.time()
    mism_new, mism_old, n = [], [], 0
    for (s, t, gd) in keys:
        ppl = per.get((s, int(t)), {})
        outs = set()
        for p, ent in ppl.items():
            ent = sorted(ent)
            if sum(1 for d, _ in ent if d < gd) < starout.MIN_TRAIL_GAMES:
                continue
            tn = [m for d, m in ent if d == gd]
            if tn and tn[0] > 0:
                continue
            outs.add(p)
        # (i) DEFAULT (env unset) must equal the gated ARM
        os.environ.pop(var, None)
        c_new = starout.team_context(con, int(t), outs, gd, weights, positions)
        if ctx_key(c_new) != ctx_key(ctxs[arm].get((s, t, gd))):
            mism_new.append((s, int(t), str(gd), ctx_key(c_new),
                             ctx_key(ctxs[arm].get((s, t, gd)))))
        # (ii) OLD value must equal the gate's CONTROL
        os.environ[var] = old
        c_old = starout.team_context(con, int(t), outs, gd, weights, positions)
        if ctx_key(c_old) != ctx_key(ctxs["ctrl"].get((s, t, gd))):
            mism_old.append((s, int(t), str(gd), ctx_key(c_old),
                             ctx_key(ctxs["ctrl"].get((s, t, gd)))))
        os.environ.pop(var, None)
        n += 1
        if n % 100 == 0:
            print(f"  {n}/{len(keys)} ({time.time()-t0:.0f}s)", flush=True)
    con.close()
    out = {"arm": arm, "env_var": var, "old_value": old,
           "team_dates_checked": n,
           "default_reproduces_gated_arm_mismatches": len(mism_new),
           "old_value_reproduces_gate_control_mismatches": len(mism_old),
           "examples_new": mism_new[:3], "examples_old": mism_old[:3],
           "trail_mode_at_default": starout.trail_mode(),
           "usage_mode_at_default": starout.usage_mode()}
    Path(f"data/qg_starout_shipconfirm_{arm}.json").write_text(
        json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))
    print("QG_STAROUT_SHIPCONFIRM_DONE", flush=True)


if __name__ == "__main__":
    main()
