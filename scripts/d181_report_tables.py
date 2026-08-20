#!/usr/bin/env python3
"""D181 — compute every table for the simulation-performance report, in the
format of the US Treasury basis-RV sim report the owner supplied.

Tier: k=9 raw = MAX BOOKS OBSERVED AT THE OPEN, no outlier-realism haircut
(both per owner instruction).

Mapping from the Treasury report's schema to this one:
    tenor            -> season          (the independently-scored books)
    trading session  -> betting day
    traded notional  -> staked notional (both sides of a bet do not both trade,
                        so notional = stake, NOT stake x 2)
    edge (bps)       -> net PnL per dollar staked, in bps  (= ROI x 100)

Read-only.  Nothing ships.  No default changed.
"""
from __future__ import annotations

import json, os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                                # noqa: E402

TIER = "k=9 raw"
STAKE = 10_000.0        # nominal $/bet, flat.  Stated, not implied.
TAG = os.environ.get("RPT_TAG", "_HONEST")
# D207: the trading frame must respect the same coverage rule as the model frame.
# The daily injury report begins 2018-12-17, so the availability leg — half the
# production margin — is EMPTY before 2019-20. Seasons before that measure a
# different, crippled model, exactly as D186 established for model accuracy.
FROM = os.environ.get("RPT_FROM", "2019-20")


def load():
    pb = json.load(open(ROOT / "data" / f"wf_perbet{TAG}.json"))
    R = json.load(open(ROOT / "data" / f"wf_equity{TAG}.json"))
    bets = sorted(pb[TIER], key=lambda b: (b["date"], b["gid"]))
    return bets, R


def daily(bets):
    d = defaultdict(float)
    c = defaultdict(int)
    for b in bets:
        d[b["date"]] += b["ev"] * b["keep"] * STAKE
        c[b["date"]] += 1
    days = sorted(d)
    return days, np.array([d[x] for x in days]), np.array([c[x] for x in days])


def stats(bets, label):
    days, pnl, cnt = daily(bets)
    n_bets = int(cnt.sum())
    net = pnl.sum()
    cum = np.cumsum(pnl)
    dd = float((cum - np.maximum.accumulate(cum)).min())
    sd = pnl.std(ddof=1)
    # annualised two ways: the report convention (sqrt 252) and the honest one
    # (this strategy only trades ~120 days a year, so sqrt252 overstates it)
    sh252 = (pnl.mean() / sd) * np.sqrt(252) if sd > 0 else float("nan")
    per_yr = len(days) / (len(set(b["season"] for b in bets)) or 1)
    sh_real = (pnl.mean() / sd) * np.sqrt(per_yr) if sd > 0 else float("nan")
    ntl = n_bets * STAKE
    return dict(label=label, days=len(days), n=n_bets, net=net,
                per_day=net / len(days), sharpe252=sh252, sharpe_real=sh_real,
                win=float((pnl > 0).mean()), dd=dd,
                trades_day=n_bets / len(days), ntl_day=ntl / len(days),
                edge_bps=1e4 * net / ntl, roi=100 * net / ntl,
                days_per_season=per_yr)


def row(s):
    return (f"| {s['label']} | {s['days']} | ${s['net']:,.0f} | "
            f"${s['per_day']:,.0f} | {s['sharpe_real']:.1f} | "
            f"{100*s['win']:.0f}% | −${abs(s['dd']):,.0f} | "
            f"{s['trades_day']:.1f} | ${s['ntl_day']/1e6:.2f}M | "
            f"{s['edge_bps']:.0f} |")


def main():
    bets, R = load()
    bets = [b for b in bets if b["season"] >= FROM]
    seasons = sorted({b["season"] for b in bets})
    print(f"FRAME: {FROM} onward — the fully injury-covered seasons only")
    print(f"TIER {TIER}   stake ${STAKE:,.0f}/bet   seasons {len(seasons)}")
    print(f"seasons: {seasons}\n")

    # ---- headline: first half / second half / full, split by SEASON count
    half = len(seasons) // 2
    A = [b for b in bets if b["season"] in seasons[:half]]
    B = [b for b in bets if b["season"] in seasons[half:]]
    sA = stats(A, f"First half ({seasons[0]} – {seasons[half-1]})")
    sB = stats(B, f"Second half ({seasons[half]} – {seasons[-1]})")
    sF = stats(bets, "**Full window**")
    print("HEADLINE")
    print("| window | days | net PnL | PnL/day | Sharpe (ann.) | win days | "
          "max drawdown | trades/day | staked ntl/day | edge (bps) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s in (sA, sB, sF):
        print(row(s))

    # ---- per-season attribution  (the report's per-tenor table)
    print("\nPER-SEASON ATTRIBUTION")
    print("| season | net PnL | PnL/day | Sharpe (ann.) | win days | "
          "trades/day | staked ntl/day | edge (bps) |")
    print("|---|---|---|---|---|---|---|---|")
    per = {}
    for s in seasons:
        ss = stats([b for b in bets if b["season"] == s], s)
        per[s] = ss
        print(f"| {s} | ${ss['net']:,.0f} | ${ss['per_day']:,.0f} | "
              f"{ss['sharpe_real']:.1f} | {100*ss['win']:.0f}% | "
              f"{ss['trades_day']:.1f} | ${ss['ntl_day']/1e6:.2f}M | "
              f"{ss['edge_bps']:+.0f} |")
    print(f"| **ALL** | **${sF['net']:,.0f}** | **${sF['per_day']:,.0f}** | "
          f"**{sF['sharpe_real']:.1f}** | **{100*sF['win']:.0f}%** | "
          f"**{sF['trades_day']:.1f}** | **${sF['ntl_day']/1e6:.2f}M** | "
          f"**{sF['edge_bps']:+.0f}** |")

    npos = sum(1 for s in seasons if per[s]["net"] > 0)
    print(f"\nseasons profitable: {npos}/{len(seasons)}")

    # season-clustered CI on ROI
    # BUGFIX: the season-clustered t interval must be CENTRED ON THE SAME
    # QUANTITY it takes its sd from — the UNWEIGHTED mean of per-season ROIs
    # (oc.cluster_mean_t, which is what GATE_POLICY_V2 and wf_equity use).
    # Centring on the n-WEIGHTED pooled ROI while using the unweighted sd is
    # inconsistent and inflated the interval away from zero, briefly making the
    # offset arm look significant when the production pipeline said otherwise.
    import sys as _s
    _s.path.insert(0, str(ROOT / "scripts"))
    import oc_capacity as _oc
    roi = np.array([per[s]["roi"] for s in seasons])
    ci = _oc.cluster_mean_t(roi)
    print(f"pooled ROI (n-weighted) {sF['roi']:+.2f}%")
    print(f"season-clustered mean   {ci['mean']:+.2f}%  95% CI "
          f"[{ci['lo']:+.2f}, {ci['hi']:+.2f}]  (K={ci['K']})  "
          f"-> {'EXCLUDES' if ci['sig'] else 'CONTAINS'} zero")

    # ---- correlation of daily PnL across seasons is undefined (disjoint dates);
    #      the analogue is season-to-season dispersion
    print(f"season ROI dispersion: sd {roi.std(ddof=1):.2f}pp  "
          f"min {roi.min():+.2f}%  max {roi.max():+.2f}%")

    # ---- monthly breakdown (calendar month across all seasons)
    print("\nMONTHLY BREAKDOWN")
    print("| month | days | net PnL | PnL/day | win days |")
    print("|---|---|---|---|---|")
    bym = defaultdict(list)
    for b in bets:
        bym[b["date"][5:7]].append(b)
    MN = {"10": "October", "11": "November", "12": "December", "01": "January",
          "02": "February", "03": "March", "04": "April"}
    for mo in sorted(bym, key=lambda x: (int(x) < 7, x)):
        ss = stats(bym[mo], MN.get(mo, mo))
        print(f"| {MN.get(mo, mo)} | {ss['days']} | ${ss['net']:,.0f} | "
              f"${ss['per_day']:,.0f} | {100*ss['win']:.0f}% |")

    # ---- tier ladder for the caveats section
    print("\nTIER LADDER (same bets, execution varied)")
    for t in ("k=1 raw", "k=2 raw", "k=5 raw", "k=8 raw", "k=9 raw",
              "k=9 +haircut", "exchange c=2%"):
        if t in R["tiers"]:
            r = R["tiers"][t]
            print(f"  {t:16s} ROI {100*r['roi']:+6.2f}%  cum {r['cum'][-1]:+7.1f}u  "
                  f"CI [{100*r['ci']['lo']:+6.2f},{100*r['ci']['hi']:+6.2f}]")

    json.dump(dict(headline=[sA, sB, sF], per_season=per,
                   stake=STAKE, tier=TIER),
              open(ROOT / "data" / "d203_report_honest.json", "w"), indent=1,
              default=float)
    print("\nwrote data/d203_report_honest.json")


if __name__ == "__main__":
    main()
