#!/usr/bin/env python3
"""D145 x D133 SWITCH COMPOSITION — the two ramps subtract from the SAME
proj_min. Measures the JOINT distribution of (minutes_ramp(gp),
absence_ramp(miss10)) over the ENTIRE props eval universe and the worst-case
combined subtraction, and proves proj_min cannot go negative or reach the
simulator floor for a real rotation player. Read-only."""
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from ab_props_gate import build_index2, row_feats
from nbapred.db import connect
from nbapred.engine.props import absence_ramp, minutes_ramp
from pr_ramp_gate import load_corpus

con = connect(read_only=True); df, tg = load_corpus(con); con.close()
byp, tsched = build_index2(df, tg)
rows = []
for r in df.itertuples():
    f = row_feats(byp, tsched, int(r.player_id), r.ord, r.season)
    if f is None:
        continue
    proj_raw, nh, gp, mi, lt = f
    if nh < 8 or proj_raw < 20:
        continue
    a, b = minutes_ramp(gp), absence_ramp(mi)
    rows.append((gp, mi, a, b, proj_raw, max(max(proj_raw - a, 0.0) - b, 0.0)))
a = np.array(rows, float)
gp, mi, ra, rb, praw, pfin = a.T
tot = ra + rb
out = {
  "n_universe": int(len(a)),
  "max_gp_ramp": float(ra.max()), "max_absence_ramp": float(rb.max()),
  "theoretical_max_if_independent": float(ra.max() + rb.max()),
  "OBSERVED_max_combined": float(tot.max()),
  "n_both_terms_active": int(((ra != 0) & (rb != 0)).sum()),
  "share_both_active": float(((ra != 0) & (rb != 0)).mean()),
  "n_absence_only": int(((ra == 0) & (rb != 0)).sum()),
  "n_gp_only": int(((ra != 0) & (rb == 0)).sum()),
  "n_neither": int(((ra == 0) & (rb == 0)).sum()),
  "min_final_proj_min": float(pfin.min()),
  "n_final_below_20": int((pfin < 20).sum()),
  "n_final_below_15": int((pfin < 15).sum()),
  "n_final_below_10_sim_floor": int((pfin < 10).sum()),
  "n_final_clipped_at_zero": int((pfin <= 0).sum()),
  "min_praw": float(praw.min()),
}
# can gp==0 co-occur with a live absence term? (own-set is empty by construction)
m0 = gp == 0
out["gp0_rows"] = int(m0.sum())
out["gp0_with_absence_term"] = int((rb[m0] != 0).sum())
for lo, hi in ((0, 0), (1, 2), (3, 5), (6, 9), (10, 19), (20, 10**6)):
    m = (gp >= lo) & (gp <= hi)
    if m.sum():
        out[f"gp[{lo},{hi}] max_combined"] = round(float(tot[m].max()), 4)
print(json.dumps(out, indent=1))
json.dump(out, open("data/ab_switch_compose.json", "w"), indent=1)
