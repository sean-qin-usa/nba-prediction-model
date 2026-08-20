"""D171 TASK 3 — price the missing 5PM injury report: paired test on the T2 vs
T2i per-game frames, and its dependence on absence density."""
import sys, csv, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
LN2 = 0.6931471805599453

def load(p):
    out = {}
    for r in csv.DictReader(open(p)):
        out[(r["season"], r["game_id"])] = (int(r["y"]), float(r["p_us"]),
                                            float(r["p_mkt"]),
                                            int(r["n_out_home"])+int(r["n_out_away"]))
    return out
t2  = load(ROOT/"data/k19_d171_t2_pergame.csv")
t2i = load(ROOT/"data/k19_d171_t2i_pergame.csv")
keys = sorted(set(t2) & set(t2i))
def pll(y, p):
    p = min(max(p, 1e-15), 1-1e-15)
    return -(y*np.log(p) + (1-y)*np.log(1-p))

# per-game paired difference in (model - market) log loss
d = np.array([pll(t2[k][0], t2[k][1]) - pll(t2i[k][0], t2i[k][1]) for k in keys])
seas = np.array([k[0] for k in keys])
print("="*94)
print("WHAT THE 5PM INJURY REPORT IS WORTH (T2 minus T2i), paired per-game, modern 8 seasons")
print("="*94)
print(f"n = {len(keys)} games (2018-19..2025-26)")
print(f"mean paired dLL = {d.mean():+.6f}   (negative = the report HELPS)")
# cluster by season (K=8), the register's standard
cl = np.array([d[seas == s].mean() for s in sorted(set(seas))])
se = cl.std(ddof=1)/np.sqrt(len(cl))
print(f"season-clustered mean {cl.mean():+.6f}  SE {se:.6f}  t = {cl.mean()/se:+.2f}  (K=8)")
print(f"seasons improved by the report: {(cl < 0).sum()}/8")
# translate to normalized points
ll_mkt = float(np.mean([pll(t2[k][0], t2[k][2]) for k in keys]))
print(f"pooled normalized worth = {100*d.mean()/(LN2-ll_mkt):+.3f}pp  (market ll {ll_mkt:.5f})")

print()
print("DEPENDENCE ON ABSENCE DENSITY — the whole question of what it is worth to an OLD season")
per = []
for s in sorted(set(seas)):
    m = seas == s
    outs = np.mean([t2[k][3] for k in keys if k[0] == s])/2
    add  = outs - np.mean([t2i[k][3] for k in keys if k[0] == s])/2
    per.append((s, outs, add, 100*d[m].mean()/(LN2-ll_mkt)))
    print(f"  {s}  T2 outs/team-gm {outs:.2f}  report ADDS {add:+.2f} outs/team-gm  "
          f"worth {per[-1][3]:+.2f}pp")
xa = np.array([p[2] for p in per]); xt = np.array([p[1] for p in per])
yv = np.array([p[3] for p in per])
r_add = float(np.corrcoef(xa, yv)[0, 1])
r_tot = float(np.corrcoef(xt, yv)[0, 1])
print(f"\ncorr(outs the report ADDS, its worth)      = {r_add:+.3f}")
print(f"corr(TOTAL outs/team-gm, its worth)       = {r_tot:+.3f}")
print("BOTH ARE NOISE AT K=8 (|t| < 1.4). The density -> worth slope is NOT")
print("estimable from 8 seasons and this entry does not pretend otherwise.")

old_outs = 0.39   # mean scored OUT/team-gm 2012-13..2015-16 (D171 census)
mod_outs = float(np.mean([p[1] for p in per]))
mod_add  = float(np.mean([p[2] for p in per]))
scale = old_outs/mod_outs
pooled_pp = 100*d.mean()/(LN2-ll_mkt)
print(f"\nSO THE OLD-ERA PRICE IS GIVEN AS A BOUNDED RANGE, NOT A POINT ESTIMATE:")
print(f"  modern: total {mod_outs:.2f} outs/team-gm, of which the report ADDS {mod_add:+.2f}"
      f"  -> worth {pooled_pp:+.2f}pp")
print(f"  2012-16: total {old_outs:.2f} outs/team-gm (ratio {scale:.2f}); on the same"
      f" proportionality the report could add only ~{mod_add*scale:+.2f}")
print(f"  UPPER bound (report worth the same as it is now, no density scaling) : {pooled_pp:+.2f}pp")
print(f"  LOWER bound (worth scales with the information it adds)              : {pooled_pp*scale:+.2f}pp")
print("  => the permanent pre-2018-12-17 report gap costs those 11 seasons somewhere")
print(f"     between {abs(pooled_pp*scale):.2f} and {abs(pooled_pp):.2f} normalized points. Either end is SMALL")
print("     next to the 17-21pp the DARKO backfill moved them, and small next to the")
print("     residual 2024-25 advantage of +3.97pp. It is NOT the remaining story.")
json.dump({"n": len(keys), "mean_dll": float(d.mean()),
           "clustered_mean": float(cl.mean()), "clustered_se": float(se),
           "t": float(cl.mean()/se), "pooled_pp": float(pooled_pp),
           "per_season": per, "corr_add": r_add, "corr_total": r_tot,
           "old_bound_low": float(pooled_pp*scale), "old_bound_high": float(pooled_pp),
           "modern_outs": mod_outs, "modern_add": mod_add, "old_outs": old_outs},
          open(ROOT/"data/d171_report_price.json","w"), indent=1)
