#!/usr/bin/env python3
"""D181 — the equity figure for the simulation-performance report, laid out to
match the US Treasury basis-RV report the owner supplied: cumulative net PnL on
top, daily net PnL bars beneath, shared date axis.

Tier k=9 raw (MAX BOOKS at the open, no outlier-realism haircut) per owner.

Sign is carried by bar DIRECTION as well as colour, so the green/red pair is
never the only encoding.

Read-only.  Nothing ships.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import matplotlib                                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.dates as mdates                                 # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

NAVY, GREEN, RED = "#1f3864", "#2e7d32", "#c62828"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e7e6e2"
plt.rcParams.update({
    "axes.formatter.use_mathtext": False, "mathtext.default": "regular",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlecolor": INK,
})
TIER, STAKE = "k=9 raw", 10_000.0
MEASURED = {"2012-13", "2013-14", "2014-15", "2015-16", "2016-17", "2017-18",
            "2023-24"}


def main():
    pb = json.load(open(ROOT / "data" / "wf_perbet_HONEST.json"))
    FROM = "2019-20"          # D207: match the document frame
    bets = sorted((b for b in pb[TIER] if b["season"] >= FROM),
                  key=lambda b: (b["date"], b["gid"]))
    d, seas = defaultdict(float), {}
    for b in bets:
        d[b["date"]] += b["ev"] * b["keep"] * STAKE
        seas[b["date"]] = b["season"]
    days = sorted(d)
    x = [dt.date.fromisoformat(s) for s in days]
    p = np.array([d[s] for s in days])
    cum = np.cumsum(p)

    fig = plt.figure(figsize=(13.6, 7.6), dpi=150)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.05, 1.0], hspace=0.13,
                          left=0.075, right=0.985, top=0.845, bottom=0.115)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x, cum / 1e6, color=NAVY, lw=1.7, zorder=5)
    ax.axhline(0, color=INK2, lw=0.9, zorder=3)
    # shade the seasons whose multi-book price is MEASURED rather than modelled
    for s in sorted({b["season"] for b in bets}):
        ds = [dt.date.fromisoformat(k) for k, v in seas.items() if v == s]
        if s in MEASURED and ds:
            ax.axvspan(min(ds), max(ds), color=NAVY, alpha=0.10, zorder=1,
                       lw=0)
    peak = np.maximum.accumulate(cum)
    i_dd = int(np.argmin(cum - peak))
    ax.plot([x[i_dd]], [cum[i_dd] / 1e6], marker="v", ms=8, color=RED, zorder=6)
    ax.annotate(f"max drawdown  −\\${abs((cum-peak).min()):,.0f}",
                xy=(x[i_dd], cum[i_dd] / 1e6), xytext=(12, -26),
                textcoords="offset points", color=RED, fontsize=8.8,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.7, alpha=0.7))
    ax.set_ylabel("cumulative net PnL ($MM)")
    ax.set_xticklabels([])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(0.004, 0.955, "shaded = seasons whose multi-book price is MEASURED; "
            "unshaded = modelled uplift from a single book",
            transform=ax.transAxes, fontsize=8.4, color=INK2, va="top")

    axb = fig.add_subplot(gs[1, 0], sharex=ax)
    axb.bar(x, p / 1e3, color=[GREEN if v >= 0 else RED for v in p], width=3.0,
            zorder=3)
    axb.axhline(0, color=INK2, lw=0.9, zorder=4)
    axb.set_ylabel("daily net PnL ($k)")
    axb.xaxis.set_major_locator(mdates.YearLocator())
    axb.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    plt.setp(axb.get_xticklabels(), rotation=0, ha="center")

    net, nb = cum[-1], len(bets)
    fig.suptitle("Simulation performance — NBA opening-spread relative value",
                 fontsize=13.4, x=0.075, ha="left", y=0.972, color=NAVY,
                 fontweight="bold")
    fig.text(0.075, 0.935,
             f"{len(days)} betting sessions across 7 seasons (2019-20 … "
             f"2025-26)  ·  {nb:,} bets  ·  net \\${net:,.0f} at a flat "
             f"\\${STAKE:,.0f} stake  ·  edge {1e4*net/(nb*STAKE):.0f} bps  ·  "
             f"tier {TIER} (max books, no haircut)",
             fontsize=9.2, color=INK2, ha="left", va="top")
    out = ROOT / "charts" / "sim_report_equity.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print("wrote", out)
    print(f"  net ${net:,.0f}  maxDD ${(cum-peak).min():,.0f}  days {len(days)}")


if __name__ == "__main__":
    main()
