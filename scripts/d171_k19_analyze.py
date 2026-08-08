"""D171 TASK 3 — 19-season table with per-season tiers, the ADV metric, and the
2024-25 residual, on the D171 (Clippers-fixed) numbers.  Read-only."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
LN2 = 0.6931471805599453

t2 = {s["season"]: s for s in json.load(open(ROOT/"data/k19_d171_t2.json"))["seasons"]}
try:
    t2i = {s["season"]: s for s in json.load(open(ROOT/"data/k19_d171_t2i.json"))["seasons"]}
except Exception:
    t2i = {}
# D170's registered arms, quoted (never recomputed)
D170_A  = {"2007-08":26.87,"2008-09":17.85,"2009-10":25.69,"2010-11":23.61,"2011-12":27.61,
           "2012-13":16.67,"2013-14":23.90,"2014-15":20.16,"2015-16":13.05,"2016-17":9.71,
           "2017-18":27.48,"2018-19":22.60,"2019-20":12.12,"2020-21":36.76,"2021-22":29.52,
           "2022-23":22.34,"2023-24":20.16,"2024-25":11.72,"2025-26":15.01}
D170_B2 = {"2007-08":7.18,"2008-09":-2.44,"2009-10":4.24,"2010-11":4.22,"2011-12":9.74,
           "2012-13":7.33,"2013-14":11.67,"2014-15":11.18,"2015-16":5.78,"2016-17":9.15,
           "2017-18":25.58,"2018-19":17.27,"2019-20":11.75,"2020-21":35.14,"2021-22":27.10,
           "2022-23":23.69,"2023-24":20.20,"2024-25":11.59,"2025-26":15.12}
D170_C  = {"2007-08":6.54,"2008-09":-2.01,"2009-10":3.78,"2010-11":1.58,"2011-12":8.55,
           "2012-13":7.06,"2013-14":9.82,"2014-15":9.80,"2015-16":5.07,"2016-17":7.91,
           "2017-18":22.17,"2018-19":14.98,"2019-20":6.26,"2020-21":26.85,"2021-22":16.79,
           "2022-23":13.21,"2023-24":16.35,"2024-25":6.22,"2025-26":12.21}
S = list(t2)

print("="*118)
print("D171 — 19 SEASONS AT THE BEST TIER EACH SEASON CAN REACH (tier labelled per season, never pooled silently)")
print("="*118)
print(f"{'season':<9}{'tier':<14}{'n':>6}{'ll_us':>9}{'ll_mkt':>9}{'raw':>10}{'norm':>9}"
      f" | {'D170 C':>8}{'d':>7} | {'T2i':>8}{'report worth':>14}{'outs/tm':>9}")
tot = {}
for s in S:
    r = t2[s]; i = t2i.get(s)
    rep = (r["norm_gap_pct"] - i["norm_gap_pct"]) if i else None
    print(f"{s:<9}{r['tier_label']:<14}{r['n']:>6}{r['ll_us']:>9.5f}{r['ll_mkt']:>9.5f}"
          f"{r['raw_gap']:>+10.5f}{r['norm_gap_pct']:>8.2f}% | {D170_C[s]:>7.2f}%"
          f"{r['norm_gap_pct']-D170_C[s]:>+7.2f} | "
          f"{(('%8.2f%%'%i['norm_gap_pct']) if i else '       -')}"
          f"{(('%+13.2fpp'%rep) if rep is not None else '            -')}"
          f"{r['mean_outs_per_team']:>9.2f}")

# pooled, weighted by n (log loss is a mean, so weight by games)
def pooled(d, key="norm_gap"):
    n = np.array([d[s]["n"] for s in d]); u = np.array([d[s]["ll_us"] for s in d])
    m = np.array([d[s]["ll_mkt"] for s in d])
    U = float((n*u).sum()/n.sum()); M = float((n*m).sum()/n.sum())
    return U, M, 100*(U-M)/(LN2-M), int(n.sum())
U, M, P, N = pooled(t2)
print(f"{'POOLED':<9}{'mixed(labelled)':<14}{N:>6}{U:>9.5f}{M:>9.5f}{U-M:>+10.5f}{P:>8.2f}%"
      f" | {9.50:>7.2f}%{P-9.50:>+7.2f} |")
if t2i:
    Ui, Mi, Pi, Ni = pooled(t2i)
    print(f"   modern-8 pooled: T2 {pooled({s:t2[s] for s in t2i})[2]:.2f}%  "
          f"T2i {Pi:.2f}%  -> the 5PM report is worth {pooled({s:t2[s] for s in t2i})[2]-Pi:+.2f}pp "
          f"on n={Ni} modern games")

# ---- ADV metric (D170's pre-registered definition) -------------------------
print()
print("="*118)
print("IS 2024-25 STILL SPECIAL?   ADV(s) = mean(norm gap of the OTHER 18) - norm gap(s)")
print("="*118)
def advtab(d, label):
    g = {s: d[s] for s in d}
    adv = {s: (sum(g[x] for x in g if x != s)/(len(g)-1)) - g[s] for s in g}
    rank = {s: 1+sorted(g.values()).index(g[s]) for s in g}
    return adv, rank, label
arms = [(D170_A, "D161 arm A (blind, pre-backfill DARKO)"),
        (D170_B2, "D170 B2 (blind, full DARKO+reports)"),
        (D170_C, "D170 C  (T2, pre-Clippers-fix)"),
        ({s: t2[s]["norm_gap_pct"] for s in S}, "D171   (T2, Clippers-fixed)")]
print(f"{'arm':<40}{'2024-25 gap':>13}{'ADV':>10}{'rank':>8}{'best season':>16}")
for g, lab in arms:
    adv, rank, _ = advtab(g, lab)
    best = min(g, key=lambda k: g[k])
    print(f"{lab:<40}{g['2024-25']:>12.2f}%{adv['2024-25']:>+10.2f}"
          f"{rank['2024-25']:>5}/19{best:>16}")
a0 = advtab(D170_A, "")[0]["2024-25"]
a1 = advtab({s: t2[s]["norm_gap_pct"] for s in S}, "")[0]["2024-25"]
print(f"\nshare of 2024-25's ORIGINAL apparent advantage that was data completeness: "
      f"{100*(1-a1/a0):.1f}%   residual genuinely the season: {100*a1/a0:.1f}% ({a1:+.2f}pp of {a0:+.2f}pp)")

json.dump({"t2": {s: t2[s] for s in S}, "pooled": [U, M, P, N],
           "adv": {lab: advtab(g, lab)[0]["2024-25"] for g, lab in arms},
           "rank": {lab: advtab(g, lab)[1]["2024-25"] for g, lab in arms}},
          open(ROOT/"data/d171_k19_analyze.json","w"), indent=1)
