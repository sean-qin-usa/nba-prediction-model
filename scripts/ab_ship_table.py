#!/usr/bin/env python3
"""Fit the FULL-CORPUS b_miss table that production would ship (the D133
convention: the shipped MINUTES_RAMP is the all-seasons fit, while the GATE
scored strictly-prior-season fits). Also reports the per-cutoff table so the
stationarity claim is auditable. Read-only. Prints a paste-ready tuple."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from ab_props_gate import MISS_B, WINDOW_LO, build_index2, fit_tables
from nbapred.db import connect
from pr_ramp_gate import load_corpus

if __name__ == "__main__":
    con = connect(read_only=True)
    df, tg = load_corpus(con)
    con.close()
    byp, tsched = build_index2(df, tg)
    allseas = sorted(set(df["season"]))
    meta, bA, bC, bk = fit_tables(df, byp, tsched, allseas)
    print("fit seasons:", allseas, "n =", meta["n_fit"])
    print("per-bucket n:", meta["n_bucket"])
    print("bA_fit  :", [round(meta["bA_fit"][i], 4) for i in range(len(MISS_B))])
    print("bA_APPLY:", [round(bA[i], 4) for i in range(len(MISS_B))])
    print("b_const :", round(bk, 4))
    tbl = tuple((MISS_B[i][1] + 1, round(bA[i], 3)) for i in range(len(MISS_B))
                if bA[i] != 0.0)
    print("\nSHIP TABLE (miss_exclusive_upper, bias):")
    print("MISS_RAMP =", tbl)
    json.dump({"n_fit": meta["n_fit"], "bA_fit": meta["bA_fit"],
               "bA_applied": bA, "b_const": bk, "n_bucket": meta["n_bucket"],
               "ship_table": [list(x) for x in tbl]},
              open("data/ab_ship_table.json", "w"), indent=1, default=float)
    print("wrote data/ab_ship_table.json")
