"""D171 TASK 4 — build the k19 chart's model-panel stats from the D171
19-season T2 frame, in the exact shape scripts/k19_chart.py consumes.

Deliberately does NOT re-run scripts/k19_analyze.py: that script's registered
output (data/k19_model_stats.json) is D161's BLIND arm and stays at its
vintage. This writes a separate file.
"""
import sys, csv, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
LN2 = 0.6931471805599453

src = json.load(open(ROOT/"data/k19_d171_t2.json"))["seasons"]
rows = list(csv.DictReader(open(ROOT/"data/k19_d171_t2_pergame.csv")))
def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-15, 1-1e-15); y = np.asarray(y, float)
    return -(y*np.log(p) + (1-y)*np.log(1-p))
y  = np.array([int(r["y"]) for r in rows]);  se = np.array([r["season"] for r in rows])
du = ll(y, [float(r["p_us"]) for r in rows]) - ll(y, [float(r["p_mkt"]) for r in rows])

per = [{"season": r["season"], "norm_gap_pct": r["norm_gap_pct"],
        "tier": r["tier_label"],
        "darko_cov": r["darko_frac_roster_nonzero"],
        "outs": r["mean_outs_per_team"]} for r in src]
n = np.array([r["n"] for r in src], float)
U = float((n*np.array([r["ll_us"] for r in src])).sum()/n.sum())
M = float((n*np.array([r["ll_mkt"] for r in src])).sum()/n.sum())
# K=19 season-cluster-mean t interval on the raw paired gap (GATE_POLICY_V2 §9.1(4))
cl = np.array([du[se == s].mean() for s in sorted(set(se))])
K = len(cl); m = cl.mean(); s_ = cl.std(ddof=1)/np.sqrt(K)
from scipy import stats as st
tcrit = float(st.t.ppf(0.975, K-1))
blocks = [{"label": "ALL 19 (T2/T2i, tier labelled per season)",
           "n": int(n.sum()), "norm_gap_pct": round(100*(U-M)/(LN2-M), 2),
           "raw_gap": round(U-M, 5),
           "t_lo": round(m - tcrit*s_, 5), "t_hi": round(m + tcrit*s_, 5)}]
out = {"tier": "t2-per-season-labelled", "lower_bound": False,
       "per_season": per, "blocks": blocks,
       "tier_cost": {"t2_pooled_3": 12.87}}   # D171 certified 5-season pooled
json.dump(out, open(ROOT/"data/d171_k19_model_stats.json", "w"), indent=1)
print("pooled norm %.2f%%  raw %+.5f  K=19 t [%+.5f, %+.5f]  n=%d"
      % (blocks[0]["norm_gap_pct"], blocks[0]["raw_gap"],
         blocks[0]["t_lo"], blocks[0]["t_hi"], blocks[0]["n"]))
print("seasons positive (market wins): %d/19" % sum(1 for r in per if r["norm_gap_pct"] > 0))
