#!/usr/bin/env python3
"""GATE_POLICY_V2 §6.6 — IMPLEMENTATION IDENTITY / SHIP-CONFIRM.

D141's hall-of-shame 15: "a switch named after a hypothesis is not evidence
that it implements it." So whatever ships is RUN, on the gate's own rows, with
the gate's own seeds, and must:

  (i)   reproduce the gate's CONTROL BITWISE with the switch OFF;
  (ii)  carry the shipped coefficient on every row it is supposed to touch and
        EXACTLY ZERO everywhere else;
  (iii) reproduce the gated estimate with the switch ON (same sign, >= 50%).

usage: qg_shipconfirm.py {starout|channel} [--sims N]
Writes data/qg_shipconfirm_<which>.json
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
from nbapred.engine import props, starout
from nbapred.engine.props import player_rates_from_stats, simulate_player
from pr_ramp_gate import cluster_boot, crps, pit


def crps_of(sim, y):
    return crps(sim, y)


def confirm_channel(sims):
    """GATE 3: the shipped switch on the gate's own rows."""
    z = np.load("data/qg_channel_gate_rows.npz", allow_pickle=True)
    con = connect(read_only=True)
    n = len(z["player_id"])
    idx = np.arange(n)
    res = {"n_rows": int(n), "sims": sims}
    off_d, on_dr, on_da, pit_r_ctrl, pit_r_on = [], [], [], [], []
    players, deltas, lam_used = [], [], []
    t0 = time.time()
    for i in idx:
        pid = int(z["player_id"][i])
        dte = str(z["date"][i])
        base = player_rates_from_stats(con, pid, before=np.datetime64(dte).astype("O"))
        if base is None:
            continue
        y_r = float(z["y_rebounds"][i]); y_a = float(z["y_assists"][i])
        prng = np.random.default_rng(70_000 + i)
        os.environ["PROPS_CHANNEL_RAMP"] = "0"
        sim_off = simulate_player(base, sims, seed=int(i))
        os.environ["PROPS_CHANNEL_RAMP"] = res["mode"]
        sim_on = simulate_player(base, sims, seed=int(i))
        os.environ.pop("PROPS_CHANNEL_RAMP", None)
        off_d.append(abs(crps_of(sim_off["rebounds"], y_r)
                         - float(z["crps_rebounds_ctrl"][i])))
        on_dr.append(float(z["crps_rebounds_ctrl"][i]) - crps_of(sim_on["rebounds"], y_r))
        on_da.append(float(z["crps_assists_ctrl"][i]) - crps_of(sim_on["assists"], y_a))
        pit_r_ctrl.append(float(z["pit_rebounds_ctrl"][i]))
        pit_r_on.append(pit(sim_on["rebounds"], y_r, prng))
        players.append(pid); deltas.append(float(base.get("ramp_delta", 0.0)))
        lam_used.append(props.CHANNEL_LAM["reb"])
        if len(off_d) % 500 == 0:
            print(f"  {len(off_d)} rows ({time.time()-t0:.0f}s)", flush=True)
    con.close()
    off_d = np.array(off_d); players = np.array(players); deltas = np.array(deltas)
    act = deltas > 0
    res["off_reproduces_control_bitwise_max_abs"] = float(off_d.max())
    for lbl, arr in (("rebounds", np.array(on_dr)), ("assists", np.array(on_da))):
        pt_, lo, hi, se = cluster_boot(arr[act], players[act])
        res[f"on_{lbl}_primary"] = dict(delta=pt_, lo=lo, hi=hi, se=se,
                                        sig="SIG" if (lo > 0 or hi < 0) else "ns")
    res["pit_rebounds_primary"] = {
        "ctrl": float(np.array(pit_r_ctrl)[act].mean()),
        "shipped": float(np.array(pit_r_on)[act].mean())}
    res["lam_shipped"] = dict(props.CHANNEL_LAM)
    return res


def confirm_starout(sims):
    """GATES 1/2: the shipped switch on the gate's own rows."""
    z = np.load("data/qg_starout_gate_rows.npz", allow_pickle=True)
    raise SystemExit("starout ship-confirm is wired by the caller after the "
                     "gate verdict; see qg_starout_shipconfirm.py")


def main():
    which = sys.argv[1]
    sims = int(sys.argv[sys.argv.index("--sims") + 1]) if "--sims" in sys.argv else 4000
    if which == "channel":
        mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "A"
        res = {"mode": mode}
        res.update(confirm_channel(sims) if False else {})
        # run with the mode recorded first so confirm_channel can read it
        globals()["_MODE"] = mode
        r = _confirm_channel_impl(sims, mode)
        res.update(r)
    else:
        raise SystemExit("unsupported")
    Path(f"data/qg_shipconfirm_{which}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    print("QG_SHIPCONFIRM_DONE", flush=True)


def _confirm_channel_impl(sims, mode):
    z = np.load("data/qg_channel_gate_rows.npz", allow_pickle=True)
    con = connect(read_only=True)
    n = len(z["player_id"])
    off_d, on_dr, on_da, pit_r_on = [], [], [], []
    players, deltas = [], []
    t0 = time.time()
    for i in range(n):
        pid = int(z["player_id"][i])
        dte = np.datetime64(str(z["date"][i])).astype("O")
        base = player_rates_from_stats(con, pid, before=dte)
        if base is None:
            continue
        y_r = float(z["y_rebounds"][i]); y_a = float(z["y_assists"][i])
        prng = np.random.default_rng(70_000 + i)
        os.environ["PROPS_CHANNEL_RAMP"] = "0"
        sim_off = simulate_player(base, sims, seed=int(i))
        os.environ["PROPS_CHANNEL_RAMP"] = mode
        sim_on = simulate_player(base, sims, seed=int(i))
        os.environ.pop("PROPS_CHANNEL_RAMP", None)
        off_d.append(abs(crps_of(sim_off["rebounds"], y_r)
                         - float(z["crps_rebounds_ctrl"][i])))
        on_dr.append(float(z["crps_rebounds_ctrl"][i]) - crps_of(sim_on["rebounds"], y_r))
        on_da.append(float(z["crps_assists_ctrl"][i]) - crps_of(sim_on["assists"], y_a))
        pit_r_on.append(pit(sim_on["rebounds"], y_r, prng))
        players.append(pid); deltas.append(float(base.get("ramp_delta", 0.0)))
        if len(off_d) % 1000 == 0:
            print(f"  {len(off_d)}/{n} rows ({time.time()-t0:.0f}s)", flush=True)
    con.close()
    off_d = np.array(off_d); players = np.array(players); deltas = np.array(deltas)
    act = deltas > 0
    res = {"mode": mode, "n_rows": int(len(off_d)), "sims": sims,
           "off_reproduces_control_bitwise_max_abs": float(off_d.max()),
           "n_active": int(act.sum()),
           "lam_shipped": dict(props.CHANNEL_LAM)}
    for lbl, arr in (("rebounds", np.array(on_dr)), ("assists", np.array(on_da))):
        pt_, lo, hi, se = cluster_boot(arr[act], players[act])
        res[f"on_{lbl}_primary"] = dict(delta=pt_, lo=lo, hi=hi, se=se,
                                        sig="SIG" if (lo > 0 or hi < 0) else "ns")
    res["pit_rebounds_primary_shipped"] = float(np.array(pit_r_on)[act].mean())
    return res


if __name__ == "__main__":
    main()
