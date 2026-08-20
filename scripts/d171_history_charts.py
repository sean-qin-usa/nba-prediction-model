"""D171 TASK 4 — regenerate the two historical charts from the NEW 19-season
frame (data/k19_d171_t2.json), which is scored at the best availability tier
each season can reach rather than availability-blind.

Reuses scripts/make_history_charts.py's chart functions UNMODIFIED except for
the D171 title/ylim pass made in that file, so the house style cannot drift.
The third chart there (history_feature_by_era.png) needs the D153 ablation
battery, which this entry does NOT re-run — it is left at its D153 vintage.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
import numpy as np
import make_history_charts as H

LN2 = 0.6931471805599453
ERA = {}
for s in ["2007-08","2008-09","2009-10","2010-11"]: ERA[s] = "E-3"
for s in ["2011-12","2012-13","2013-14"]:           ERA[s] = "E-2"
for s in ["2014-15","2015-16","2016-17","2017-18","2018-19"]: ERA[s] = "E-1"
ERA["2019-20"]="E0"; ERA["2020-21"]="E2"; ERA["2021-22"]="E3"
ERA["2022-23"]="E4"; ERA["2023-24"]="E5"
ERA["2024-25"]="E6"; ERA["2025-26"]="E6"
STRATA = {"2011-12", "2019-20", "2020-21"}          # lockout, bubble, no-crowd
CERT   = {"2021-22","2022-23","2023-24","2024-25","2025-26"}

src = json.load(open(ROOT/"data/k19_d171_t2.json"))["seasons"]
per = {}
for r in src:
    s = r["season"]
    per[s] = {"season": s, "era": ERA[s], "n": r["n"],
              "ll_us": r["ll_us"], "ll_mkt": r["ll_mkt"],
              "raw_gap": round(r["ll_us"]-r["ll_mkt"], 5),
              "mkt_skill": round(LN2-r["ll_mkt"], 5),
              "norm_gap_pct": r["norm_gap_pct"],
              "in_cert_corpus": s in CERT, "stratum": s in STRATA,
              "tier": r["tier_label"]}

def pool(seasons, label):
    n = np.array([per[s]["n"] for s in seasons], float)
    u = float((n*np.array([per[s]["ll_us"] for s in seasons])).sum()/n.sum())
    m = float((n*np.array([per[s]["ll_mkt"] for s in seasons])).sum()/n.sum())
    return {"label": label, "seasons": list(seasons), "n": int(n.sum()),
            "ll_us": round(u,5), "ll_mkt": round(m,5),
            "raw_gap": round(u-m,5), "norm_gap_pct": round(100*(u-m)/(LN2-m),2)}

allp = [s for s in per if s not in STRATA]
res = {"per_season": per,
       "pooled": {"certified_5": pool(sorted(CERT), "certified corpus 2021-26 (T2)"),
                  "historical_new": pool([s for s in sorted(per) if s not in CERT and s not in STRATA],
                                         "pre-certified seasons (T2/T2i)"),
                  "all_poolable": pool(sorted(allp), "all poolable seasons")}}
json.dump(res, open(ROOT/"data/d171_history_analysis.json","w"), indent=1)

print("per-season tiers:", {s: per[s]["tier"] for s in sorted(per)})
for k, v in res["pooled"].items():
    print(f"  {k:<16} n={v['n']:>6}  ll_us={v['ll_us']:.5f} ll_mkt={v['ll_mkt']:.5f} "
          f"norm={v['norm_gap_pct']:+.2f}%")
print("wrote", H.chart_logloss(res))
print("wrote", H.chart_normgap(res))
