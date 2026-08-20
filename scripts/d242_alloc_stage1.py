#!/usr/bin/env python3
"""D242 stage 1 — do minute-conserving arms beat the incumbent at the
COMPOSITION-MARGIN level, once scale is removed? Prereg sha256 4d7e98a4...

WHY NOT GO STRAIGHT TO THE FULL STACK. Under any 240-conserving allocation
`sum_i m_i / 48 = 5` exactly, so team strength stops being a sum of
contributions and becomes 5 x (minutes-weighted mean talent). Level and
dispersion change completely, which would break the fixed 0.5 blend weight for
reasons unrelated to the hypothesis. Every arm is therefore AFFINE-RECALIBRATED
per fold (`a + b*cm`, fitted on prior seasons only) before scoring, which
isolates the SHAPE change -- mass conservation -- from a pure scale artefact.

Endpoint: RMSE of the recalibrated composition margin against the realised
margin, season-clustered. Not log loss: at this layer we are asking whether the
availability channel is a better POINT estimate, and the probability conversion
belongs to stage 2.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from scipy import stats                                           # noqa: E402

SCR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp")


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def load(tag):
    p = SCR / f"ch_{tag}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d["game_id"] = zf(d["game_id"])
    return d[["game_id", "season", "game_date", "m_comp"]].rename(
        columns={"m_comp": tag})


def clus(v):
    v = np.asarray(v, float); k = len(v)
    se = v.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return v.mean(), v.mean() - tc * se, v.mean() + tc * se, v.mean() / se, k


def main():
    # THE PRE-REGISTERED GRID IS REQUIRED, NOT OPTIONAL. The first version
    # silently skipped absent arms, which is how the pre-registered PLACEBO --
    # declared as a HARNESS-VALIDITY gate -- vanished from a run that then
    # reported results as if the grid were complete. Missing arms now fail.
    # The FULL pre-registered grid is required: six arms x four constructions
    # plus the placebo and the oracle. The first fix listed only five names, so
    # B_N3 -- the arm that actually moved -- could still have gone missing while
    # the run called itself complete.
    REQUIRED = {"ctrl", "A",
                "B_N1", "C_N1", "B_N2", "C_N2",
                "B_N3", "C_N3", "B_N4", "C_N4",
                "PLACEBO", "ORACLE"}
    FORENSIC_DIAGNOSTICS = ["ORACLE_MIN", "HARD_RAW", "HARD_N3", "ORACLE_N3"]
    have = {t for t in list(REQUIRED) + FORENSIC_DIAGNOSTICS
            if (SCR / f"ch_{t}.csv").exists()}
    missing = REQUIRED - have
    # EXPLICIT "1", not truthiness: os.environ.get returns the STRING "0",
    # which is truthy, so `if not os.environ.get(...)` would have let
    # D242_ALLOW_PARTIAL=0 silently ENABLE partial runs.
    if missing and os.environ.get("D242_ALLOW_PARTIAL") != "1":
        raise RuntimeError(
            f"INCOMPLETE PRE-REGISTERED GRID: {sorted(missing)}. "
            f"Set D242_ALLOW_PARTIAL=1 only for an explicitly partial diagnostic.")
    tags = [t for t in ["ctrl", "A", "B_N1", "C_N1", "B_N2", "C_N2", "B_N3",
                        "C_N3", "B_N4", "C_N4", "PLACEBO", "ORACLE",
                        "ORACLE_MIN", "HARD_RAW", "HARD_N3", "ORACLE_N3"]
            if t in have]
    print("arms present:", tags)
    if missing:
        print(f"  *** PARTIAL RUN, missing required: {sorted(missing)}")
    base = load("ctrl")
    if base is None:
        raise SystemExit("control run missing")
    f = base
    for t in tags:
        if t == "ctrl":
            continue
        f = f.merge(load(t), on=["game_id", "season", "game_date"], how="inner")
    pit = pd.read_csv(ROOT / "data" / "pit_frame.csv.gz")
    pit["game_id"] = zf(pit["game_id"])
    f = f.merge(pit[["game_id", "margin_actual"]], on="game_id", how="inner")
    f = f.dropna(subset=["margin_actual"]).sort_values(["season", "game_date"])
    print(f"frame {len(f):,} games, {f.season.nunique()} seasons")

    # dispersion, to show WHY recalibration is required
    print("\n  per-arm composition-margin sd (raw, before recalibration):")
    for t in tags:
        print(f"    {t:8} sd {f[t].std():7.3f}   mean {f[t].mean():+7.3f}")

    seasons = sorted(f.season.unique())
    rows = []
    for i, s in enumerate(seasons):
        if i == 0:
            continue
        tr, te = f[f.season.isin(seasons[:i])], f[f.season == s]
        r = {"season": s, "n": len(te)}
        for t in tags:
            b, a = np.polyfit(tr[t], tr["margin_actual"], 1)   # affine, per fold
            pred = a + b * te[t]
            r[t] = float(np.sqrt(((te["margin_actual"] - pred) ** 2).mean()))
        rows.append(r)
    d = pd.DataFrame(rows)
    print("\n--- per-season RMSE of the recalibrated composition margin ---")
    print(d.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    print("\n--- vs CONTROL (negative = better) ---")
    out = {}
    for t in tags:
        if t == "ctrl":
            continue
        delta = (d[t] - d["ctrl"]).to_numpy()
        m, lo, hi, tt, k = clus(delta)
        flag = "BETTER" if hi < 0 else ("WORSE" if lo > 0 else "ns")
        print(f"  {t:8} mean {m:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
              f"t {tt:+6.2f}  better {int((delta<0).sum())}/{k}  {flag}")
        out[t] = dict(mean=float(m), ci=[float(lo), float(hi)], t=float(tt),
                      better=int((delta < 0).sum()), k=k, flag=flag)

    if "C_N1" in out and "A" in out:
        ca = (d["C_N1"] - d["A"]).to_numpy()
        m, lo, hi, tt, k = clus(ca)
        print(f"\n  T4 (arm C vs arm A — does allocation rescue broad "
              f"participation?)\n    mean {m:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
              f"t {tt:+.2f}  {'C BETTER' if hi < 0 else ('C WORSE' if lo > 0 else 'ns')}")
        out["T4_C_vs_A"] = dict(mean=float(m), ci=[float(lo), float(hi)])
    if "C_N1" in out and "B_N1" in out:
        cb = (d["C_N1"] - d["B_N1"]).to_numpy()
        m, lo, hi, tt, k = clus(cb)
        print(f"  C vs B (incremental value of the broad participation model, "
              f"allocation held fixed)\n    mean {m:+.4f}  CI [{lo:+.4f}, "
              f"{hi:+.4f}]  t {tt:+.2f}")
        out["C_vs_B"] = dict(mean=float(m), ci=[float(lo), float(hi)])

    json.dump({"per_season": rows, "vs_control": out},
              open(ROOT / "data" / "d242_stage1.json", "w"), default=float)
    print("\nwrote data/d242_stage1.json")


if __name__ == "__main__":
    main()
