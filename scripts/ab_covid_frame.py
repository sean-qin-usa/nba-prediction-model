#!/usr/bin/env python3
"""GATE_POLICY_V2 §10.4 COVID-FRAME CHECK for the D145 absence ramp.

The shipped b table follows the D133 convention and is fit on 2019-20..2025-26,
which INCLUDES E0/E1 (2019-20) and E2 (2020-21). §10.4 requires the estimate to
be reported with and without those frames (precedent: D136's two "SIG MATCH"
travel coefficients existed ONLY in the COVID frame).
"""
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from ab_props_gate import MISS_B, build_index2, fit_tables
from nbapred.db import connect
from pr_ramp_gate import load_corpus

con = connect(read_only=True); df, tg = load_corpus(con); con.close()
byp, tsched = build_index2(df, tg)
frames = {
    "FULL 2019-20..2025-26 (shipped, D133 convention)":
        ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"],
    "E3-E6 only 2021-22..2025-26 (no COVID frame)":
        ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"],
    "COVID frames only 2019-20+2020-21 (E0/E1/E2)": ["2019-20", "2020-21"],
}
out = {}
for lab, ss in frames.items():
    meta, bA, bC, bk = fit_tables(df, byp, tsched, ss)
    out[lab] = {"n_fit": meta["n_fit"],
                "bA_fit": [round(meta["bA_fit"][i], 4) for i in range(len(MISS_B))],
                "applied": [round(bA[i], 4) for i in range(len(MISS_B))],
                "b_const": round(bk, 4)}
    print(f"{lab}\n   n={meta['n_fit']}  bA_fit={out[lab]['bA_fit']}  "
          f"APPLIED={out[lab]['applied']}")
json.dump(out, open("data/ab_covid_frame.json", "w"), indent=1)
print("wrote data/ab_covid_frame.json")
