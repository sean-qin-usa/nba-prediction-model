#!/usr/bin/env python3
"""D182 — "should we not trade in April or December then?"

The monthly table in docs/SIM_REPORT.md shows December -$127,073 and April
-$13,756. The question is whether to drop them. That is a SUBGROUP SELECTION
made after seeing the endpoint, which is the exact procedure D165/D166 measured
as manufacturing +16.92 ROI points from pure noise. So it does not get answered
by looking at the table; it gets answered by these four tests, all declared here
before scoring:

  T1  IN-SAMPLE GAIN. What does dropping Dec+Apr buy, in-sample? (Expected to
      be positive by construction — this is the number that tempts you.)
  T2  IS THE MONTH EFFECT EVEN REAL? Season-clustered t on each month's ROI,
      i.e. is December reliably bad across seasons, or is it one bad December?
  T3  MANUFACTURING NULL. Shuffle the month label within each season (preserving
      season and bet composition) and re-run "drop the 2 worst months" 2,000
      times. If the real gain sits inside that distribution, the gain is search.
  T4  WALK-FORWARD. The only test that matters. At each step choose which months
      to drop using ONLY seasons 1..k, freeze, score season k+1. This is what
      the rule would actually have earned.

DECISION RULE, stated before scoring: ship the month filter only if T4 beats the
unfiltered incumbent AND T3's null does not explain T1. Net-of-null alone is not
sufficient (D176).

Read-only. Nothing ships. No default changed.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                                # noqa: E402

TIER = "k=9 raw"
NDRAW = 2000
SEED = 20260805
MN = {"10": "Oct", "11": "Nov", "12": "Dec", "01": "Jan", "02": "Feb",
      "03": "Mar", "04": "Apr", "05": "May", "07": "Jul", "08": "Aug"}


def load():
    pb = json.load(open(ROOT / "data" / "wf_perbet_D181.json"))
    b = [dict(season=x["season"], mo=x["date"][5:7], date=x["date"],
              ev=x["ev"] * x["keep"]) for x in pb[TIER]]
    return sorted(b, key=lambda z: z["date"])


def roi(rows):
    return 100 * sum(r["ev"] for r in rows) / len(rows) if rows else float("nan")


def clustered_t(per_season):
    v = np.array([x for x in per_season if np.isfinite(x)], float)
    if len(v) < 3:
        return float("nan"), float("nan"), len(v)
    se = v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() / se if se > 0 else float("nan"), len(v)


def main():
    bets = load()
    seasons = sorted({b["season"] for b in bets})
    months = sorted({b["mo"] for b in bets}, key=lambda m: (int(m) < 7, m))
    base = roi(bets)
    print(f"TIER {TIER}   n={len(bets)}  seasons={len(seasons)}  "
          f"baseline ROI {base:+.3f}%\n")

    # ---------------------------------------------------------------- T1
    print("=" * 74)
    print("T1  IN-SAMPLE: what does dropping months buy?")
    print("=" * 74)
    permo = {m: [b for b in bets if b["mo"] == m] for m in months}
    print(f"  {'month':6} {'n':>5} {'ROI%':>8} {'u':>8}  {'seasons':>7}")
    for m in months:
        r = permo[m]
        print(f"  {MN[m]:6} {len(r):5d} {roi(r):+8.2f} "
              f"{sum(x['ev'] for x in r):+8.1f}  {len({x['season'] for x in r}):7d}")
    drop = {"12", "04"}
    kept = [b for b in bets if b["mo"] not in drop]
    print(f"\n  baseline                 ROI {base:+.3f}%  n={len(bets)}")
    print(f"  drop Dec+Apr             ROI {roi(kept):+.3f}%  n={len(kept)}  "
          f"-> GAIN {roi(kept)-base:+.3f} ROI points")

    # ---------------------------------------------------------------- T2
    print("\n" + "=" * 74)
    print("T2  IS THE MONTH EFFECT REAL? season-clustered t on each month")
    print("=" * 74)
    print(f"  {'month':6} {'K':>3} {'mean season ROI':>16} {'t':>7}  verdict")
    for m in months:
        ps = [roi([b for b in permo[m] if b["season"] == s])
              for s in seasons if any(b["season"] == s for b in permo[m])]
        mu, t, K = clustered_t(ps)
        vv = ("SIG" if abs(t) > 2.16 else "ns") if np.isfinite(t) else "K<3"
        star = "  <-- the two in question" if m in drop else ""
        print(f"  {MN[m]:6} {K:3d} {mu:+16.2f} {t:+7.2f}  {vv}{star}")
    # is December one bad season or many?
    dec = [(s, roi([b for b in permo["12"] if b["season"] == s]))
           for s in seasons if any(b["season"] == s for b in permo["12"])]
    dec_sorted = sorted(dec, key=lambda z: z[1])
    print(f"\n  December by season (worst first): "
          f"{', '.join(f'{s} {v:+.1f}%' for s, v in dec_sorted)}")
    neg = sum(1 for _, v in dec if v < 0)
    print(f"  December negative in {neg}/{len(dec)} seasons")
    worst_s, worst_v = dec_sorted[0]
    ex = [b for b in permo["12"] if b["season"] != worst_s]
    print(f"  December excluding its worst season ({worst_s}): {roi(ex):+.2f}% "
          f"on n={len(ex)}")

    # ---------------------------------------------------------------- T3
    print("\n" + "=" * 74)
    print(f"T3  MANUFACTURING NULL — shuffle month labels within season, "
          f"{NDRAW} draws")
    print("=" * 74)
    rng = np.random.default_rng(SEED)
    by_season = defaultdict(list)
    for b in bets:
        by_season[b["season"]].append(b)
    real_gain = roi(kept) - base

    def best_drop_gain(rows_by_season):
        allr = [b for v in rows_by_season.values() for b in v]
        bs = roi(allr)
        gains = []
        for m in months:
            for m2 in months:
                if m >= m2:
                    continue
                k = [b for b in allr if b["mo"] not in (m, m2)]
                if len(k) > 50:
                    gains.append(roi(k) - bs)
        return max(gains) if gains else 0.0

    null = np.empty(NDRAW)
    for i in range(NDRAW):
        shuf = {}
        for s, rows in by_season.items():
            mos = [r["mo"] for r in rows]
            rng.shuffle(mos)
            shuf[s] = [dict(r, mo=m) for r, m in zip(rows, mos)]
        null[i] = best_drop_gain(shuf)
    p = float((null >= real_gain).mean())
    print(f"  real gain from dropping the 2 worst months : {real_gain:+.3f} pts")
    print(f"  null 'best 2-month drop' mean              : {null.mean():+.3f} pts")
    print(f"  null 95th pct                              : "
          f"{np.percentile(null, 95):+.3f} pts")
    print(f"  p(null >= real)                            : {p:.4f}")
    print(f"  NET OF NULL                                : "
          f"{real_gain - null.mean():+.3f} pts")
    print(f"  -> {'gain is INSIDE what search manufactures' if p > 0.05 else 'gain exceeds the search null'}")

    # ---------------------------------------------------------------- T4
    print("\n" + "=" * 74)
    print("T4  WALK-FORWARD — pick the months to drop on seasons 1..k, score k+1")
    print("=" * 74)
    print(f"  {'score season':13} {'dropped (chosen on prior)':34} "
          f"{'filtered':>9} {'unfiltered':>11} {'delta':>8}")
    f_ev, u_ev, per_delta = [], [], []
    for i in range(1, len(seasons)):
        prior = [b for b in bets if b["season"] in seasons[:i]]
        test = [b for b in bets if b["season"] == seasons[i]]
        if not test:
            continue
        pm = defaultdict(list)
        for b in prior:
            pm[b["mo"]].append(b)
        # rule declared above: drop every month with negative ROI on 1..k
        bad = {m for m, r in pm.items() if len(r) >= 20 and roi(r) < 0}
        ft = [b for b in test if b["mo"] not in bad]
        if not ft:
            print(f"  {seasons[i]:13} {'(filter would drop all)':34}")
            continue
        f_ev += [b["ev"] for b in ft]
        u_ev += [b["ev"] for b in test]
        d = roi(ft) - roi(test)
        per_delta.append(d)
        print(f"  {seasons[i]:13} "
              f"{','.join(MN[m] for m in sorted(bad)) or '(none)':34} "
              f"{roi(ft):+9.2f} {roi(test):+11.2f} {d:+8.2f}")
    fr = 100 * np.sum(f_ev) / len(f_ev)
    ur = 100 * np.sum(u_ev) / len(u_ev)
    mu, t, K = clustered_t(per_delta)
    print(f"\n  WALK-FORWARD filtered   ROI {fr:+.3f}%  n={len(f_ev)}")
    print(f"  WALK-FORWARD unfiltered ROI {ur:+.3f}%  n={len(u_ev)}  (incumbent)")
    print(f"  DELTA vs incumbent      {fr-ur:+.3f} ROI points")
    print(f"  season-clustered mean delta {mu:+.3f} pts, t={t:+.2f} (K={K}) "
          f"-> {'SIG' if abs(t) > 2.16 else 'ns'}")

    verdict = ("SHIP" if (fr > ur and p <= 0.05) else "DO NOT SHIP")
    print("\n" + "=" * 74)
    print(f"VERDICT: {verdict}")
    print("=" * 74)
    json.dump(dict(base=base, t1_gain=real_gain, t3_p=p,
                   t3_null_mean=float(null.mean()),
                   t4_filtered=fr, t4_unfiltered=ur, t4_delta=fr - ur,
                   t4_t=float(t), verdict=verdict),
              open(ROOT / "data" / "d182_month_filter.json", "w"), indent=1)
    print("wrote data/d182_month_filter.json")


if __name__ == "__main__":
    main()
