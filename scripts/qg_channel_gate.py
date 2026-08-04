#!/usr/bin/env python3
"""GATE 3 — per-channel props ramp, pre-registered gate.

PRE-REGISTRATION: data/queued_gates_prereg.md §3 (sha256 recorded in the json).
Nothing here may deviate from it.

ARMS (control = production props.py verbatim, same run, both shipped ramps ON):
  ctrl  PROPS_CHANNEL_RAMP unset          (lam_c == 1 for every channel)
  A     PROPS_CHANNEL_RAMP=A   LOCATION   exposure_c = m0 - lam_c*D
  B     PROPS_CHANNEL_RAMP=B   DISPERSION rate_c *= (m0-lam_c*D)/(m0-D)

IMPLEMENTATION IDENTITY (§6.6): the arms are produced by the SHIPPED
`props.simulate_player` under the switch they are named after; the walk-forward
lam tables are injected by rebinding `props.CHANNEL_LAM`, which is the exact
object production reads. No generative replica is used, so there is nothing to
diverge. The paired MC draws share a seed per row, and points/threes are drawn
BEFORE the channel block, which is why the points veto is checkable BITWISE.

lam_c is fit walk-forward (strictly-prior seasons only) by the closed-form
first-moment condition E[rate_c*(m0 - lam_c*D)] = E[y_c]. No sweep.

Read-only DB. Writes data/qg_channel_gate.json + data/qg_channel_gate_rows.npz.

usage: qg_channel_gate.py [--max-active N] [--sims N]
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
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from nbapred.db import connect
from nbapred.engine import props
from nbapred.engine.props import (absence_ramp, minutes_ramp,
                                  player_rates_from_stats, simulate_player)
from pr_ramp_gate import BOOT_SEED, NBOOT, cluster_boot, crps, pit

HL = 10.0
R_DECAY = 0.5 ** (1.0 / HL)
ALL_SEASONS = ("2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
               "2024-25", "2025-26")
SCORED = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
SIMS = 4000
ARMS = ("ctrl", "A", "B")
MAX_INACTIVE = 6000
MARKETS = ("points", "rebounds", "assists")


def ewma_prefix(x):
    out = np.empty(len(x) + 1)
    out[0] = 0.0
    acc = 0.0
    for i, v in enumerate(x):
        acc = v + R_DECAY * acc
        out[i + 1] = acc
    return out


def build_frame(con):
    df = con.execute("""
        SELECT s.player_id, s.team_id, g.season, g.game_date, s.seconds/60.0 AS mins,
               s.pts, s.oreb + s.dreb AS reb, s.ast
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '002%' AND s.seconds >= 720
        ORDER BY s.player_id, g.game_date""").fetchdf()
    df["ord"] = df["game_date"].values.astype("datetime64[D]").astype(int)
    sched = defaultdict(list)
    for s, t, d in con.execute("""
        SELECT season, team_id, game_date FROM nba_games
        WHERE game_id LIKE '002%'""").fetchall():
        sched[(s, int(t))].append(np.datetime64(d).astype("datetime64[D]").astype(int))
    for k in sched:
        sched[k] = np.array(sorted(set(sched[k])))
    recs = []
    for pid, sub in df.groupby("player_id", sort=False):
        o = sub["ord"].to_numpy()
        m = sub["mins"].to_numpy(float)
        seas = sub["season"].to_numpy(object)
        team = sub["team_id"].to_numpy()
        gdate = sub["game_date"].tolist()
        Sw = ewma_prefix(np.ones_like(m))
        Sm = ewma_prefix(m)
        Sr = ewma_prefix(sub["reb"].to_numpy(float))
        Sa = ewma_prefix(sub["ast"].to_numpy(float))
        y_reb = sub["reb"].to_numpy(float)
        y_ast = sub["ast"].to_numpy(float)
        y_pts = sub["pts"].to_numpy(float)
        for i in range(len(m)):
            if i < 8:
                continue
            m0 = Sm[i] / Sw[i]
            if m0 < 20:
                continue
            season = seas[i]
            gp = int((seas[:i] == season).sum())
            pm1 = max(m0 - minutes_ramp(gp), 0.0)
            lt = int(team[i - 1])
            sc = sched.get((season, lt))
            miss10 = 0
            if sc is not None:
                prior = sc[sc < o[i]][-10:]
                own = {d for d, s_, t_ in zip(o[:i], seas[:i], team[:i])
                       if s_ == season and int(t_) == lt}
                if own and prior.size:
                    first = min(own)
                    miss10 = int(sum(1 for d in prior if d >= first and d not in own))
            pm = max(pm1 - absence_ramp(miss10), 0.0)
            recs.append((int(pid), season, gdate[i], m0, pm, m0 - pm,
                         Sr[i] / Sm[i], Sa[i] / Sm[i], y_reb[i], y_ast[i],
                         y_pts[i], m[i], gp, miss10))
    return pd.DataFrame(recs, columns=[
        "player_id", "season", "game_date", "m0", "pm", "delta", "r_reb",
        "r_ast", "y_reb", "y_ast", "y_pts", "y_min", "gp", "miss10"])


def fit_lam(sub):
    a = sub[sub.delta > 0]
    lam = {}
    for c, ex, expo0 in (("reb", a.r_reb, a.m0),
                         ("ast", a.r_ast, np.clip(a.m0, 10, 44))):
        num = float((ex * expo0 - a[f"y_{c}"]).mean())
        den = float((ex * a.delta).mean())
        lam[c] = float(num / den) if den else 1.0
    lam["n_fit"] = int(len(a))
    return lam


def main():
    max_active = int(sys.argv[sys.argv.index("--max-active") + 1]) if "--max-active" in sys.argv else 10 ** 9
    sims = int(sys.argv[sys.argv.index("--sims") + 1]) if "--sims" in sys.argv else SIMS
    t0 = time.time()
    sha = hashlib.sha256((ROOT / "data" / "queued_gates_prereg.md").read_bytes()).hexdigest()
    print(f"prereg sha256 {sha}", flush=True)
    assert os.environ.get("PROPS_MIN_RAMP", "1") != "0"
    assert os.environ.get("PROPS_ABSENCE_RAMP", "1") != "0"
    os.environ.pop("PROPS_CHANNEL_RAMP", None)

    con = connect(read_only=True)
    F = build_frame(con)
    print(f"frame {len(F)} rows ({time.time()-t0:.0f}s)", flush=True)

    fits = {}
    for s in SCORED:
        prior = [x for x in ALL_SEASONS if x < s]
        lam = fit_lam(F[F.season.isin(prior)])
        lam["fit_seasons"] = prior
        fits[s] = lam
        print(f"FIT {s} <- {prior}  lam_reb {lam['reb']:+.4f}  "
              f"lam_ast {lam['ast']:+.4f}  n={lam['n_fit']}", flush=True)

    # ---- row selection: ALL ramp-active + stride sample of inactive ---------
    sel = []
    for s in SCORED:
        sub = F[F.season == s]
        act = sub[sub.delta > 0]
        ina = sub[sub.delta <= 0]
        if len(act) > max_active // len(SCORED):
            act = act.iloc[::max(1, len(act) // (max_active // len(SCORED)))]
        step = max(1, len(ina) // (MAX_INACTIVE // len(SCORED)))
        sel.append(pd.concat([act, ina.iloc[::step]]))
    S = pd.concat(sel)
    print(f"selected {len(S)} rows ({int((S.delta>0).sum())} active) "
          f"({time.time()-t0:.0f}s)", flush=True)

    rows = []
    n_ok = 0
    for r in S.itertuples():
        base = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
        if base is None or base["n_games"] < 8 or base["proj_min"] < 20:
            continue
        lam = fits[r.season]
        i = len(rows)
        rec = dict(player_id=int(r.player_id), season=r.season,
                   date=str(r.game_date)[:10], month=int(r.game_date.month),
                   gp=int(r.gp), miss10=int(r.miss10),
                   delta=float(base.get("ramp_delta", 0.0)),
                   proj=float(base["proj_min"]),
                   y_points=float(r.y_pts), y_rebounds=float(r.y_reb),
                   y_assists=float(r.y_ast),
                   lam_reb=float(lam["reb"]), lam_ast=float(lam["ast"]))
        prng = np.random.default_rng(70_000 + i)
        for a in ARMS:
            if a == "ctrl":
                os.environ.pop("PROPS_CHANNEL_RAMP", None)
                props.CHANNEL_LAM["reb"] = 1.0
                props.CHANNEL_LAM["ast"] = 1.0
            else:
                os.environ["PROPS_CHANNEL_RAMP"] = a
                props.CHANNEL_LAM["reb"] = float(lam["reb"])
                props.CHANNEL_LAM["ast"] = float(lam["ast"])
            sim = simulate_player(base, sims, seed=i)
            for mk in MARKETS:
                rec[f"crps_{mk}_{a}"] = crps(sim[mk], rec[f"y_{mk}"])
            rec[f"pit_points_{a}"] = pit(sim["points"], rec["y_points"], prng)
            rec[f"pit_rebounds_{a}"] = pit(sim["rebounds"], rec["y_rebounds"], prng)
            rec[f"pit_assists_{a}"] = pit(sim["assists"], rec["y_assists"], prng)
            rec[f"mean_reb_{a}"] = float(sim["rebounds"].mean())
            rec[f"mean_ast_{a}"] = float(sim["assists"].mean())
            rec[f"pts_hash_{a}"] = float(sim["points"].sum())
            rec[f"thr_hash_{a}"] = float(sim["threes"].sum())
        rows.append(rec)
        n_ok += 1
        if n_ok % 500 == 0:
            print(f"  {n_ok} rows ({time.time()-t0:.0f}s)", flush=True)
    os.environ.pop("PROPS_CHANNEL_RAMP", None)
    props.CHANNEL_LAM["reb"] = 1.0
    props.CHANNEL_LAM["ast"] = 1.0
    con.close()
    print(f"scored {len(rows)} rows ({time.time()-t0:.0f}s)", flush=True)

    D = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    np.savez_compressed("data/qg_channel_gate_rows.npz", **D)
    players = D["player_id"]
    seas = D["season"]
    month = D["month"]
    delta = D["delta"]
    active = delta > 0
    out = {"prereg_sha256": sha, "n": len(rows), "sims": sims,
           "boot_seed": BOOT_SEED, "nboot": NBOOT, "seasons": list(SCORED),
           "fits": fits, "n_active": int(active.sum()), "strata": {}}

    # ---- VETO: points and threes must be BITWISE unchanged -----------------
    out["points_bitwise"] = {
        a: {"max_abs_dcrps": float(np.abs(D["crps_points_ctrl"] - D[f"crps_points_{a}"]).max()),
            "max_abs_dsum_points": float(np.abs(D["pts_hash_ctrl"] - D[f"pts_hash_{a}"]).max()),
            "max_abs_dsum_threes": float(np.abs(D["thr_hash_ctrl"] - D[f"thr_hash_{a}"]).max())}
        for a in ("A", "B")}
    print("\nPOINTS/THREES BITWISE VETO:", json.dumps(out["points_bitwise"], indent=1))

    inact = ~active
    out["zero_outside_window"] = {
        a: float(np.abs(D["crps_rebounds_ctrl"][inact] - D[f"crps_rebounds_{a}"][inact]).max())
        for a in ("A", "B")} if inact.sum() else {}
    print("zero-outside (ramp-inactive, rebounds):",
          json.dumps(out["zero_outside_window"]))

    def report(mask, label):
        if mask.sum() < 30:
            return
        blk = {"n": int(mask.sum()),
               "players": int(len(set(players[mask].tolist())))}
        print(f"\n--- {label} (n={mask.sum()}, {blk['players']} players) ---")
        for mk in MARKETS:
            key = f"crps_{mk}"
            blk[key] = {"ctrl_mean": float(D[f"{key}_ctrl"][mask].mean())}
            for a in ("A", "B"):
                d = (D[f"{key}_ctrl"] - D[f"{key}_{a}"])[mask]
                pt_, lo, hi, se = cluster_boot(d, players[mask])
                sig = "SIG" if (lo > 0 or hi < 0) else "ns"
                blk[key][a] = dict(delta=pt_, lo=lo, hi=hi, se=se, sig=sig,
                                   mde80=2.802 * se)
                print(f"  {mk:9s} CRPS {a} {pt_:+.5f} CI[{lo:+.5f},{hi:+.5f}] "
                      f"se {se:.5f} MDE80 {2.802*se:.5f} {sig}")
        for mk in MARKETS:
            blk[f"pit_{mk}"] = {a: float(D[f"pit_{mk}_{a}"][mask].mean())
                                for a in ARMS}
        print("  PIT reb: " + "  ".join(f"{a} {blk['pit_rebounds'][a]:.4f}" for a in ARMS)
              + " | ast: " + "  ".join(f"{a} {blk['pit_assists'][a]:.4f}" for a in ARMS))
        out["strata"][label] = blk

    report(active, "PRIMARY — ramp-active (delta>0)")
    report(active & (delta >= 2), "delta>=2 (concentration)")
    report(active & np.isin(month, (10, 11)), "PRIMARY Oct-Nov")
    report(active & ~np.isin(month, (10, 11)), "PRIMARY Dec-Jun")
    report(np.ones(len(rows), bool), "ALL scored rows pooled")
    report(inact, "ramp-inactive (must be 0)")
    for s in SCORED:
        report(active & (seas == s), f"PRIMARY {s}")

    Path("data/qg_channel_gate.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote data/qg_channel_gate.json ({time.time()-t0:.0f}s)")
    print("QG_CHANNEL_GATE_DONE", flush=True)


if __name__ == "__main__":
    main()
