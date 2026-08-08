#!/usr/bin/env python3
"""D183 — what is the correct annualisation factor for this strategy's Sharpe?

Owner: "i dont see why not sharpe sqrt(252) - we should do on amount of trades,
not sessions traded, right?"

The principle: annualising means scaling a per-period Sharpe by sqrt(periods per
YEAR). It is not a house convention, it is an accounting identity for how much
return variance a year accumulates. So the question has a checkable answer:

  route A   per-SESSION Sharpe x sqrt(sessions per year)
  route B   per-TRADE   Sharpe x sqrt(trades   per year)

A and B must AGREE, because both estimate the same annual quantity. They agree
exactly when trades within a session are uncorrelated; they diverge by exactly
the intra-session correlation. So this script measures that correlation instead
of asserting a convention.

sqrt(252) is a third thing: it is route A with the WRONG period count
substituted (252 sessions per year when the strategy trades ~54).

Read-only. Nothing ships.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                                # noqa: E402

TIER = "k=9 raw"
NSEAS = 14


def main():
    pb = json.load(open(ROOT / "data" / f"wf_perbet_D181.json"))
    bets = sorted(pb[TIER], key=lambda b: (b["date"], b["gid"]))
    ev = np.array([b["ev"] * b["keep"] for b in bets], float)      # units/bet

    day = defaultdict(list)
    for b in bets:
        day[b["date"]].append(b["ev"] * b["keep"])
    days = sorted(day)
    dpnl = np.array([sum(day[d]) for d in days], float)
    dn = np.array([len(day[d]) for d in days], float)

    n_tr, n_se = len(ev), len(days)
    tr_yr, se_yr = n_tr / NSEAS, n_se / NSEAS
    print(f"tier {TIER}")
    print(f"  trades   {n_tr:5d}   -> {tr_yr:6.1f} per season")
    print(f"  sessions {n_se:5d}   -> {se_yr:6.1f} per season")
    print(f"  mean bets per session {dn.mean():.2f}\n")

    sh_tr = ev.mean() / ev.std(ddof=1)
    sh_se = dpnl.mean() / dpnl.std(ddof=1)
    print(f"  per-TRADE   mean {ev.mean():+.5f} u  sd {ev.std(ddof=1):.5f}  "
          f"Sharpe {sh_tr:+.5f}")
    print(f"  per-SESSION mean {dpnl.mean():+.5f} u  sd {dpnl.std(ddof=1):.5f}  "
          f"Sharpe {sh_se:+.5f}\n")

    print("=" * 70)
    print("THE TWO ROUTES")
    print("=" * 70)
    A = sh_se * np.sqrt(se_yr)
    B = sh_tr * np.sqrt(tr_yr)
    W = sh_se * np.sqrt(252)
    W2 = sh_tr * np.sqrt(252)
    print(f"  A  per-session x sqrt({se_yr:.0f} sessions/yr) = {A:6.3f}")
    print(f"  B  per-trade   x sqrt({tr_yr:.0f} trades/yr)   = {B:6.3f}")
    print(f"     -> routes agree to {abs(A-B):.4f}  "
          f"({100*abs(A-B)/max(abs(A),abs(B)):.1f}% apart)")
    print(f"\n  per-session x sqrt(252)                  = {W:6.3f}   "
          f"<- overstates A by {W/A:.2f}x")
    print(f"  per-trade   x sqrt(252)                  = {W2:6.3f}   "
          f"<- overstates B by {W2/B:.2f}x")

    print("\n" + "=" * 70)
    print("WHY THEY (DIS)AGREE: intra-session correlation")
    print("=" * 70)
    # if trades in a session were independent: var(session) = E[n] * var(trade)
    pred = dn.mean() * ev.var(ddof=1)
    obs = dpnl.var(ddof=1)
    print(f"  var(session) if trades independent : {pred:.5f}")
    print(f"  var(session) observed              : {obs:.5f}")
    print(f"  ratio observed/independent         : {obs/pred:.4f}")
    # average pairwise corr implied
    nb = dn.mean()
    rho = (obs / pred - 1) / (nb - 1) if nb > 1 else float("nan")
    print(f"  implied mean intra-session corr    : {rho:+.4f}")

    multi = [d for d in days if len(day[d]) > 1]
    print(f"  sessions with >1 bet: {len(multi)}/{n_se} "
          f"({100*len(multi)/n_se:.0f}%)")

    print("\n" + "=" * 70)
    print("DOES A YEAR EVEN CONTAIN A FULL SEASON OF OPPORTUNITY?")
    print("=" * 70)
    span = defaultdict(list)
    for b in bets:
        span[b["season"]].append(b["date"])
    lens = []
    for s, ds in span.items():
        lo, hi = min(ds), max(ds)
        import datetime as dt
        lens.append((dt.date.fromisoformat(hi) - dt.date.fromisoformat(lo)).days)
    print(f"  mean calendar span of a season's betting window: "
          f"{np.mean(lens):.0f} days of 365")
    print(f"  capital is idle {100*(1-np.mean(lens)/365):.0f}% of the year; an "
          f"'annualised' Sharpe\n  computed only on active days does not charge "
          f"for that idleness.")

    json.dump(dict(route_a=float(A), route_b=float(B), sqrt252=float(W),
                   sh_trade=float(sh_tr), sh_session=float(sh_se),
                   trades_yr=float(tr_yr), sessions_yr=float(se_yr),
                   rho=float(rho)),
              open(ROOT / "data" / "d183_sharpe.json", "w"), indent=1)
    print("\nwrote data/d183_sharpe.json")


if __name__ == "__main__":
    main()
