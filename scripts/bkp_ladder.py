#!/usr/bin/env python3
"""BOOK PANEL, PART 2 — the CONTROLLED ladder, the contemporaneous dispersion,
and the analytic re-pricing of D166's equity path.

The naive season-over-season ladder comparison is CONFOUNDED: 2023-24's panel
has 10-11 operators and 2024-25's has 3-4, so a falling ladder could be a
shrinking MARKET or a shrinking PANEL.  Everything here holds the operator set
fixed so the two can be told apart.

Reads data/bkp_panel_rows.csv.gz (written by scripts/bkp_panel.py).  No network,
no DB access at all (the DB is touched only by bkp_panel.py, read_only=True).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                      # noqa: E402
nbapred.threads.pin(1)

import collections                          # noqa: E402
import csv                                  # noqa: E402
import gzip                                 # noqa: E402
import itertools                            # noqa: E402
import json                                 # noqa: E402
import math                                 # noqa: E402

import numpy as np                          # noqa: E402

LOG: list[str] = []
R: dict = {}
KS = (1, 2, 3, 5, 8)
# D162 §6 / D166 §3: 1 spread point of shop gain -> 0.0317276 of cover rate.
DCOVER_PER_PT = 0.0317276
DP_PER_PT = 0.3989422804014327 / 12.574
BE110 = 110.0 / 210.0
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
       14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101}


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def juice_pts(american):
    if american is None:
        return None
    a = float(american)
    b = (-a) / (-a + 100.0) if a < 0 else 100.0 / (a + 100.0)
    return (b - BE110) / DP_PER_PT


def amax_normal(k):
    if k <= 1:
        return 0.0
    rng = np.random.default_rng(11)
    return float(rng.standard_normal((200000, k)).max(axis=1).mean())


def clustered(vals, clusters):
    """Season-clustered mean + 95% CI on the cluster means (D166's statistic)."""
    g = collections.defaultdict(list)
    for v, c in zip(vals, clusters):
        g[c].append(v)
    means = np.array([np.mean(v) for v in g.values()])
    K = len(means)
    if K < 2:
        return (float(means.mean()) if K else float("nan")), None, None, K
    m = float(means.mean())
    se = float(means.std(ddof=1) / math.sqrt(K))
    t = T95.get(K - 1, 1.96)
    return m, m - t * se, m + t * se, K


def load():
    pan = collections.defaultdict(dict)
    with gzip.open(ROOT / "data" / "bkp_panel_rows.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            r["home_margin"] = float(r["home_margin"])
            for k in ("juice_home", "juice_away"):
                r[k] = float(r[k]) if r[k] not in ("", "None") else None
            pan[(r["season"], r["phase"], r["game_date"], r["home"],
                 r["away"])][r["operator"]] = r
    return pan


def ladder_vals(panel, ks=KS, use_juice=False, haircut=None, nsub=200):
    """gain_k = E[range of a random k-subset]/2, in spread points (D163)."""
    rng = np.random.default_rng(20260804)
    per_k = collections.defaultdict(list)
    keys = collections.defaultdict(list)
    for key, d in panel.items():
        ops = sorted(d)
        if use_juice:
            vh, va, ok = [], [], True
            for o in ops:
                pj = juice_pts(d[o]["juice_home"])
                pa = juice_pts(d[o]["juice_away"])
                if pj is None or pa is None:
                    ok = False
                    break
                vh.append(d[o]["home_margin"] + pj)
                va.append(d[o]["home_margin"] - pa)
            if not ok:
                continue
            vh, va = np.array(vh), np.array(va)
        else:
            vh = va = np.array([d[o]["home_margin"] for o in ops])
        n = len(vh)
        if n < 2:
            continue
        if haircut == "outlier":
            med = float(np.median(vh))
            keep = np.abs(vh - med) <= 1.5
            if keep.sum() >= 2:
                vh, va, n = vh[keep], va[keep], int(keep.sum())
        for k in ks:
            if k == 1:
                per_k[1].append(0.0); keys[1].append(key); continue
            kk = min(k, n)
            if kk < 2:
                per_k[k].append(0.0); keys[k].append(key); continue
            if math.comb(n, kk) <= nsub:
                subs = list(itertools.combinations(range(n), kk))
            else:
                subs = [tuple(rng.choice(n, kk, replace=False)) for _ in range(nsub)]
            gh = np.mean([vh[list(s)].mean() - vh[list(s)].min() for s in subs])
            ga = np.mean([va[list(s)].max() - va[list(s)].mean() for s in subs])
            per_k[k].append(0.5 * (gh + ga)); keys[k].append(key)
    return per_k, keys


def show(per_k, keys, label, seasons=True):
    say(f"\n{label}")
    say(f"{'k':>3s} {'n':>6s} {'gain':>8s} {'ceiling':>8s} {'m/c':>6s}  "
        f"{'season-clustered 95% CI':>26s}")
    tab = []
    for k in sorted(per_k):
        v = np.array(per_k[k])
        ceil = 0.586 * amax_normal(k)
        ci, lo, hi = "", None, None
        if seasons and k > 1:
            m, lo, hi, K = clustered(v, [x[0] for x in keys[k]])
            ci = f"[{lo:.4f},{hi:.4f}] K={K}" if lo is not None else f"K={K} (no CI)"
        say(f"{k:3d} {len(v):6d} {v.mean():8.4f} {ceil:8.4f} "
            f"{(v.mean()/ceil if ceil else float('nan')):6.3f}  {ci:>26s}")
        tab.append(dict(k=k, n=len(v), gain=float(v.mean()), ceiling=ceil,
                        ratio=(v.mean() / ceil if ceil else None),
                        ci=([lo, hi] if lo is not None else None)))
    return tab


def main():
    pan = load()
    MOD = ["2023-24", "2024-25", "2025-26"]

    say("=" * 78)
    say("ARM 1 — THE CONFOUND, AND THE FIX. IS THE LADDER FALLING BECAUSE THE")
    say("MARKET CHANGED, OR BECAUSE THE PANEL SHRANK?")
    say("=" * 78)
    say("\nStep 1: the NAIVE comparison — every operator each season offers, at")
    say("the CLOSE (the only phase with a real panel after 2023-24).")
    naive = {}
    for s in MOD:
        p = {k: v for k, v in pan.items()
             if k[0] == s and k[1] == "close" and len(v) >= 2}
        modal = collections.Counter(len(v) for v in p.values()).most_common(1)[0][0]
        naive[s] = show(*ladder_vals(p),
                        f"  ALL operators, {s} CLOSE (modal {modal} ops/game)",
                        seasons=False)
    R["naive_close"] = naive

    say("\nStep 2: THE CONTROLLED COMPARISON. The three books Action Network")
    say("carries DENSELY in ALL THREE seasons — DraftKings, FanDuel, BetRivers")
    say("— and nothing else. Same operators, same feed, same instant of")
    say("capture. Any movement that survives IS the market.")
    FIXED = {"draftkings", "fanduel", "betrivers"}
    fixed = {}
    for s in MOD:
        p = {}
        for k, v in pan.items():
            if k[0] != s or k[1] != "close":
                continue
            sub = {o: q for o, q in v.items() if o in FIXED and q["source"] == "an"}
            if len(sub) == 3:
                p[k] = sub
        fixed[s] = show(*ladder_vals(p, ks=(1, 2, 3)),
                        f"  FIXED {{DK, FanDuel, BetRivers}}, {s} CLOSE, n={len(p)}",
                        seasons=False)
    R["fixed_close"] = fixed
    say("\n  VERDICT — fixed-basket gain by season:")
    for k in (2, 3):
        vals = [next(r["gain"] for r in fixed[s] if r["k"] == k) for s in MOD]
        say(f"    k={k}: " + "  ".join(f"{s} {v:.4f}" for s, v in zip(MOD, vals))
            + f"   spread {max(vals)-min(vals):.4f} pts "
              f"({100*(max(vals)-min(vals))/np.mean(vals):.1f}% of mean)")

    say("\n" + "=" * 78)
    say("ARM 2 — OPEN vs CLOSE, MATCHED OPERATORS AND MATCHED GAMES (2023-24)")
    say("The ONLY season with a multi-operator panel at BOTH phases. This ratio")
    say("is the only defensible bridge from a measured CLOSE ladder to the OPEN")
    say("tier that D167 actually bets.")
    say("=" * 78)
    po, pc = {}, {}
    for k, v in pan.items():
        if k[0] != "2023-24":
            continue
        (po if k[1] == "open" else pc)[k[2:]] = v
    both = set(po) & set(pc)
    say(f"  games with a panel at BOTH phases: {len(both)}")
    mo, mc = {}, {}
    for g in both:
        ops = set(po[g]) & set(pc[g])
        if len(ops) >= 2:
            mo[g] = {o: po[g][o] for o in ops}
            mc[g] = {o: pc[g][o] for o in ops}
    modal = collections.Counter(len(v) for v in mo.values()).most_common(1)[0][0]
    say(f"  after intersecting the OPERATOR sets too: {len(mo)} games, modal {modal} ops")
    lo_ = show(*ladder_vals(mo), "  MATCHED 2023-24 OPEN", seasons=False)
    lc_ = show(*ladder_vals(mc), "  MATCHED 2023-24 CLOSE", seasons=False)
    say("\n  OPEN/CLOSE RATIO (matched games AND matched operators):")
    ratios = {}
    for a, b in zip(lo_, lc_):
        if a["k"] == 1:
            continue
        ratios[a["k"]] = a["gain"] / b["gain"]
        say(f"    k={a['k']}: open {a['gain']:.4f} / close {b['gain']:.4f} = "
            f"{ratios[a['k']]:.4f}")
    R["open_close_ratio"] = ratios
    R["matched_open"], R["matched_close"] = lo_, lc_
    say("\n  NOTE: this ratio cannot be re-measured on the AN basket — Action")
    say("  Network's payload carries NO per-book opening price (book_id 30 is a")
    say("  single CONSENSUS open). The ratio is ESPN's, on ESPN's operators.")

    say("\n" + "=" * 78)
    say("ARM 3 — CONTEMPORANEOUS DISPERSION: THE NUMBER THAT GOVERNS SHOPPING.")
    say("D163 §8 could not test this at the open because ESPN's `open` block")
    say("carries NO timestamp. The Action Network panel settles it BY")
    say("CONSTRUCTION: every book in a game comes from ONE HTTP response, so")
    say("the AN cross-section is SIMULTANEOUS to the millisecond.")
    say("=" * 78)
    say("\n  First, WHICH PHASE is the AN snapshot? Same-operator cross-feed tie")
    say("  rates (bkp_panel.py PART C):")
    say("    AN[DraftKings] vs ESPN[DraftKings] at ESPN's CLOSE:")
    say("      2023-24 tie 81.54%  mean|d| 0.1474 | 2025-26 tie 96.45%  mean|d| 0.0396")
    say("    AN[DraftKings] vs ESPN[DraftKings] at ESPN's OPEN:")
    say("      2023-24 tie 13.99%  mean|d| 1.5897 | 2025-26 tie 25.71%  mean|d| 1.4839")
    say("    => the AN snapshot IS the CLOSE. And 81-96% agreement is neither")
    say("       one resold feed (a true duplicate ties at 100.00%, mean|d|")
    say("       exactly 0.0000 — D163 §4) nor two independent operators (36.5%).")
    contemp = {}
    for s in MOD:
        p = {}
        for k, v in pan.items():
            if k[0] != s or k[1] != "close":
                continue
            sub = {o: q for o, q in v.items() if q["source"] == "an"}
            if len(sub) >= 2:
                p[k] = sub
        contemp[s] = show(*ladder_vals(p),
                          f"  CONTEMPORANEOUS (AN one-response cross-section), {s}",
                          seasons=False)
    R["contemporaneous"] = contemp

    say("\n  THE COMPARISON D163 ASKED FOR — how much of a 2-book 'opening")
    say("  dispersion' is TIME rather than disagreement:")
    say("    D142/D163 TeamRankings 2-book OPEN, books a median 2.9h apart:")
    say("      mean|b1-b2| 0.6494  =>  gain_2 = 0.3247 pts   NON-SIMULTANEOUS")
    for s in MOD:
        g2 = next(r["gain"] for r in contemp[s] if r["k"] == 2)
        say(f"    AN {s} CONTEMPORANEOUS gain_2 = {g2:.4f} pts  "
            f"({100*g2/0.3247:.1f}% of the non-simultaneous 2-book number)")
    say(f"    ESPN 2023-24 OPEN  (untimestamped)     gain_2 = "
        f"{next(r['gain'] for r in lo_ if r['k'] == 2):.4f} pts (matched ops)")
    say(f"    ESPN 2023-24 CLOSE (pinned to tip-off) gain_2 = "
        f"{next(r['gain'] for r in lc_ if r['k'] == 2):.4f} pts (matched ops)")

    say("\n" + "=" * 78)
    say("ARM 4 — ERA: OFFSHORE (erichqiu 2012-19) vs US RETAIL, FIXED BASIS")
    say("=" * 78)
    ERS = ["2012-13", "2013-14", "2014-15", "2015-16", "2016-17", "2017-18",
           "2018-19"]
    erp = {k: v for k, v in pan.items()
           if k[0] in ERS and k[1] == "close" and len(v) >= 2}
    R["erich_pooled"] = show(*ladder_vals(erp, ks=(1, 2, 3, 5)),
                             "  erichqiu offshore, 7 SEASONS POOLED (Pinnacle/"
                             "5dimes/Heritage/Bovada/BetOnline)")
    say("\n  per season (5-book panel, so k=5 is the whole panel):")
    per = {}
    for s in ERS:
        p = {k: v for k, v in erp.items() if k[0] == s}
        pkx, _ = ladder_vals(p, ks=(2, 5))
        per[s] = {k: float(np.mean(v)) for k, v in pkx.items()}
        say(f"    {s}  k2 {per[s][2]:.4f}   k5 {per[s][5]:.4f}   n={len(p)}")
    R["erich_per_season"] = per
    say("\n  D163's KAG (ehallmar, 9 offshore operators, 2006-18) at the CLOSE:")
    say("    k=2 0.1002   k=3 0.1527   k=5 0.2254   k=8 0.2920")
    say("  erichqiu is an INDEPENDENT offshore scrape and lands in the same")
    say("  band => the offshore law replicates on a second source, and 2018-19")
    say("  — previously EXTRAPOLATED — is now MEASURED.")

    say("\n" + "=" * 78)
    say("ARM 5 — RE-PRICING D166's EQUITY PATH WITH THE MEASURED GAINS")
    say("=" * 78)
    D166 = [   # season, n, 1-book cover%, k5+haircut cover%, D166's label
        ("2012-13", 66, 53.12, 54.11, "MEASURED"),
        ("2013-14", 68, 64.18, 64.69, "MEASURED"),
        ("2014-15", 89, 51.14, 51.74, "MEASURED"),
        ("2015-16", 69, 49.28, 49.41, "MEASURED"),
        ("2016-17", 84, 56.10, 56.75, "MEASURED"),
        ("2017-18", 111, 53.77, 55.14, "MEASURED"),
        ("2018-19", 101, 44.00, 44.41, "EXTRAPOLATED"),
        ("2019-20", 87, 59.52, 60.45, "EXTRAPOLATED"),
        ("2020-21", 58, 51.72, 51.72, "EXTRAPOLATED"),
        ("2021-22", 113, 47.79, 47.85, "EXTRAPOLATED"),
        ("2022-23", 102, 50.00, 50.27, "EXTRAPOLATED"),
        ("2023-24", 188, 54.89, 56.21, "MEASURED"),
        ("2024-25", 243, 60.08, 61.43, "EXTRAPOLATED"),
        ("2025-26", 174, 47.13, 49.40, "EXTRAPOLATED"),
    ]

    def roi(c):
        return (100.0 / 110.0) * c - (1.0 - c)

    CONV = DCOVER_PER_PT * 1.084     # D166 §3's realised conversion at k=5+HC
    say("\n  Step 1 — recover the gain D166 APPLIED, from its own dcover.")
    say("  D166 §3 verified realised dcover/(0.0317276*g) = 1.084 at k=5+HC,")
    say("  so applied_g = dcover / (0.0317276 * 1.084).")
    applied = {s: (c5 - c1) / 100.0 / CONV for s, n, c1, c5, lab in D166}
    for s, n, c1, c5, lab in D166:
        say(f"    {s}  n={n:4d}  dcover {c5-c1:+.2f}pp  ->  applied gain "
            f"{applied[s]:.4f} pts   [{lab}]")

    say("\n  Step 2 — the MEASURED k=5 + outlier-haircut gain now available.")
    meas = {}
    for s in MOD + ERS:
        p = {k: v for k, v in pan.items()
             if k[0] == s and k[1] == "close" and len(v) >= 2}
        pk, _ = ladder_vals(p, ks=(5,), haircut="outlier")
        meas[s] = float(np.mean(pk[5]))
        say(f"    {s}  MEASURED k=5+haircut gain {meas[s]:.4f} pts  (n={len(p)})")
    R["measured_k5_haircut"] = meas

    say("\n  Step 3 — the re-priced path. Only seasons whose LABEL CHANGES move.")
    say("  2018-19 gains a measured offshore panel (erichqiu); 2024-25 and")
    say("  2025-26 gain a measured US-retail CLOSE panel (AN+ESPN).")
    say(f"  {'season':9s} {'n':>4s} {'D166 g':>7s} {'MEAS g':>7s} {'D166cov':>8s} "
        f"{'NEWcov':>7s} {'D166ROI':>8s} {'NEWROI':>7s}  label")
    newrows, cum_old, cum_new = [], 0.0, 0.0
    for s, n, c1, c5, lab in D166:
        g_old = applied[s]
        if s in ("2018-19", "2024-25", "2025-26"):
            g_new, newlab = meas[s], "MEASURED*"
        else:
            g_new, newlab = g_old, lab
        c_new = c1 + 100.0 * g_new * CONV
        r_old, r_new = roi(c5 / 100.0), roi(c_new / 100.0)
        cum_old += n * r_old
        cum_new += n * r_new
        say(f"  {s:9s} {n:4d} {g_old:7.4f} {g_new:7.4f} {c5:8.2f} {c_new:7.2f} "
            f"{100*r_old:+8.2f} {100*r_new:+7.2f}  {newlab}")
        newrows.append(dict(season=s, n=n, g_d166=g_old, g_meas=g_new,
                            cover_d166=c5, cover_new=c_new,
                            roi_d166=100 * r_old, roi_new=100 * r_new,
                            label=newlab))
    N = sum(n for _, n, _, _, _ in D166)
    say(f"\n  POOLED (bet-weighted): D166 {100*cum_old/N:+.2f}% ({cum_old:+.1f}u)"
        f"   RE-PRICED {100*cum_new/N:+.2f}% ({cum_new:+.1f}u)")
    m_o, lo_o, hi_o, K = clustered([r["roi_d166"] for r in newrows],
                                   [r["season"] for r in newrows])
    m_n, lo_n, hi_n, _ = clustered([r["roi_new"] for r in newrows],
                                   [r["season"] for r in newrows])
    say(f"  season-clustered mean ROI  D166 {m_o:+.2f}% [{lo_o:+.2f},{hi_o:+.2f}] K={K}")
    say(f"  season-clustered mean ROI  NEW  {m_n:+.2f}% [{lo_n:+.2f},{hi_n:+.2f}] K={K}")
    R["repriced"] = dict(rows=newrows, pooled_d166=100 * cum_old / N,
                         pooled_new=100 * cum_new / N, units_d166=cum_old,
                         units_new=cum_new, clustered_d166=[m_o, lo_o, hi_o],
                         clustered_new=[m_n, lo_n, hi_n], K=K)

    say("\n  Step 4 — THE OWNER'S QUESTION: IS THE POST-2024 JUMP REAL?")
    r = next(x for x in newrows if x["season"] == "2024-25")
    say(f"    2024-25, 243 bets, ONE BOOK (no shop at all): cover 60.08%  ROI +14.70%")
    say(f"    2024-25, D166 EXTRAPOLATED k=5+HC:            cover {r['cover_d166']:.2f}%"
        f"  ROI {r['roi_d166']:+.2f}%")
    say(f"    2024-25, MEASURED k=5+HC (this entry):        cover {r['cover_new']:.2f}%"
        f"  ROI {r['roi_new']:+.2f}%")
    say(f"    the measured shop is worth {r['roi_new']-14.70:+.2f} ROI points, "
        f"not the {r['roi_d166']-14.70:+.2f} that was extrapolated.")
    say(f"    => of 2024-25's headline {r['roi_d166']:+.2f}%, {14.70:.2f} points are")
    say(f"       the MODEL at one book and only {r['roi_new']-14.70:.2f} is the shop.")
    say("       THE SEASON IS NOT A SHOPPING ARTEFACT.")
    say("\n    DROP-2024-25 test on the re-priced path:")
    sub = [x for x in newrows if x["season"] != "2024-25"]
    ns = sum(x["n"] for x in sub)
    say(f"      re-priced pooled without 2024-25: "
        f"{sum(x['n']*x['roi_new'] for x in sub)/ns:+.2f}% "
        f"({sum(x['n']*x['roi_new']/100 for x in sub):+.1f}u)")
    say("      D166's own figure without 2024-25 (k=5+HC): +1.00% / +13.0u")
    say("      D166's ONE-BOOK figure without 2024-25:     -0.76% / -10.0u")

    say("\n" + "=" * 78)
    say("ARM 6 — THE HONEST BAND. THE MEASURED NUMBER IS A LOWER BOUND AND THE")
    say("EXTRAPOLATED ONE AN UPPER BOUND, AND ARM 1 SAYS WHY.")
    say("=" * 78)
    say("Two corrections run in OPPOSITE directions and both are measured:")
    say("  (a) PHASE. The 24-25/25-26 panel exists only at the CLOSE; the bet is")
    say("      at the OPEN. ARM 2: open/close = 1.11 at k=5. Pushes the")
    say("      measured number UP by ~11%.")
    say("  (b) PANEL SIZE. Only 3-4 operators are observable after 2023-24, so")
    say("      the k=5 and k=8 tiers SATURATE. A bettor holding 5+ real")
    say("      accounts is not capped that way. Pushes UP by more.")
    say("  (c) Against both: ARM 1 shows the per-book dispersion LAW is flat")
    say("      (3.0% across three seasons on identical books), so nothing")
    say("      justifies scaling the law itself.")
    r2324_full = next(x["gain"] for x in naive["2023-24"] if x["k"] == 5)
    fx = {s: next(x["gain"] for x in fixed[s] if x["k"] == 2) for s in MOD}
    say(f"\n  2023-24 FULL 10-op panel, k=5 CLOSE, raw           {r2324_full:.4f}")
    say(f"  fixed-3-basket k=2, 2023-24 / 2024-25 / 2025-26   "
        f"{fx['2023-24']:.4f} / {fx['2024-25']:.4f} / {fx['2025-26']:.4f}")
    for s in ("2024-25", "2025-26"):
        scale = fx[s] / fx["2023-24"]
        say(f"  => IF {s} had 2023-24's book COUNT, its k=5 CLOSE ladder would be")
        say(f"     {r2324_full:.4f} x {scale:.4f} = {r2324_full*scale:.4f} pts "
            f"(law transferred, count assumed)")
    say("\n  SO THE BAND ON 2024-25's k=5 SHOP GAIN, ALL THREE LABELLED:")
    m = meas["2024-25"]
    say(f"    LOWER  MEASURED, 3-4 books, CLOSE, +haircut      {m:.4f} pts")
    say(f"    MID    MEASURED + ARM 2 phase bridge (x1.11)     {m*1.1105:.4f} pts  PART-MEASURED")
    say(f"    UPPER  D166's EXTRAPOLATION (9-book law forward) {applied['2024-25']:.4f} pts  EXTRAPOLATED")
    say("  The gap between LOWER and UPPER is a BOOK-COUNT question, not a")
    say("  market question: ARM 1 proves the law is the same, and the only")
    say("  thing in dispute is how many books a 2024-25 bettor could reach.")
    say("  NO FREE SOURCE ANSWERS THAT — ESPN stopped syndicating rival books")
    say("  when ESPN BET launched, and Action Network's scoreboard shows a")
    say("  FIXED FEATURED SET. Feed policy and market truth are not")
    say("  distinguishable from these files, and this entry does not pretend")
    say("  otherwise.")
    R["band"] = dict(lower=m, mid=m * 1.1105, upper=applied["2024-25"],
                     fixed_basket_k2=fx, full_2324_k5=r2324_full)

    say("\n  UNPRICED BIAS, RESTATED ONCE (D163 §17f): best-of-k ALWAYS")
    say("  transacts at the most offside book in the panel, and NOTHING here")
    say("  or in D163/D166 charges for being limited, restricted or voided.")
    say("  Every number above is gross of that cost.")

    (ROOT / "data" / "bkp_ladder.json").write_text(json.dumps(R, indent=1, default=str))
    (ROOT / "data" / "logs" / "bkp_ladder.log").write_text("\n".join(LOG))
    say("\nwrote data/bkp_ladder.json, data/logs/bkp_ladder.log")


if __name__ == "__main__":
    main()
