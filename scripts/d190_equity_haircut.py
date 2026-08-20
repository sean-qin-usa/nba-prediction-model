#!/usr/bin/env python3
"""D190 — charts/equity_2023_26_k8_haircut.png

Owner: "push this graph with haircut version as well."

The walk-forward equity path over the 500 bets of 2023-24..2025-26 at k=8 (best
of the 8 books held), drawn TWICE on one axes:

  raw          +43.62u  /  +8.72%   the headline
  +haircut     +29.82u  /  +5.96%   after the outlier-realism charge

The haircut charges for the 8.1% of best-of-N prices that sit more than 1.5
points off the next book — precisely the prices a book limits, lowers or voids.
Both are on one axes deliberately: the gap between the lines IS the execution
assumption, and separating them into two figures would let either be quoted
alone.

Style matches the review document: navy path, brown dashed season boundaries,
flat break-even rule, x = sequential bet in date order.

Read-only.  Nothing ships.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import matplotlib                                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

NAVY, HAIR = "#1f3864", "#7d9ac2"
INK, INK2, GRID, BROWN = "#0b0b0b", "#52514e", "#e7e6e2", "#8a6d3b"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlecolor": INK,
})
WINDOW = ("2023-24", "2024-25", "2025-26")
ARMS = [("k=8 raw", NAVY, 2.2, "-", "best of 8 books, raw"),
        ("k=8 +haircut", HAIR, 2.2, "--", "after outlier-realism haircut")]


def main():
    pb = json.load(open(ROOT / "data" / "wf_perbet_HONEST.json"))
    series = {}
    for nm, *_ in ARMS:
        rows = [b for b in pb[nm] if b["season"] in WINDOW]
        rows.sort(key=lambda b: (b["date"], b["gid"]))
        series[nm] = rows
    ref = series["k=8 raw"]
    n = len(ref)
    seas = [b["season"] for b in ref]

    fig, ax = plt.subplots(figsize=(13.4, 6.6), dpi=150)
    ax.axhline(0, color=INK2, lw=1.1, ls="--", zorder=3)

    ends = []
    for nm, col, lw, ls, lab in ARMS:
        ev = np.array([b["ev"] * b["keep"] for b in series[nm]], float)
        cum = np.cumsum(ev)
        ax.plot(np.arange(len(cum)), cum, color=col, lw=lw, ls=ls,
                dashes=(5, 2.4) if ls == "--" else (None, None), zorder=5,
                label=f"{lab}   {cum[-1]:+.2f}u  ({100*ev.sum()/len(ev):+.2f}% ROI)")
        ends.append((cum[-1], col, lab, 100 * ev.sum() / len(ev)))

    # season boundaries
    for s in WINDOW[1:]:
        i = seas.index(s)
        ax.axvline(i - 0.5, color=BROWN, lw=1.2, ls=(0, (5, 2.5)), zorder=4)
    ylo, yhi = ax.get_ylim()
    for s in WINDOW:
        i = seas.index(s)
        ax.text(i + 3, ylo + 0.035 * (yhi - ylo), s, color=BROWN, fontsize=9,
                ha="left", va="bottom")

    for v, col, lab, roi in ends:
        ax.annotate(f"{v:+.2f}u\n{roi:+.2f}%", xy=(n - 1, v),
                    xytext=(n + 9, v), color=col, fontsize=9.6, va="center",
                    ha="left", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.8,
                                    alpha=0.6))
    # the gap between the lines is the execution assumption
    lo_v, hi_v = min(e[0] for e in ends), max(e[0] for e in ends)
    ax.annotate("", xy=(n - 12, hi_v), xytext=(n - 12, lo_v),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.1))
    ax.text(n - 20, (hi_v + lo_v) / 2, f"{hi_v-lo_v:.2f}u\ncharged by\nthe haircut",
            color=INK2, fontsize=8.6, ha="right", va="center")

    ax.set_xlim(-5, n * 1.13)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel(f"sequential bet, in date order  (n = {n} across the three "
                  f"seasons)")
    ax.set_ylabel("cumulative P&L, units  (flat 1u per bet)")
    ax.text(n * 0.985, 0, "break-even (flat)", ha="right", va="bottom",
            fontsize=8.8, color=INK2)
    ax.legend(frameon=False, fontsize=9.4, loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.suptitle("NBA opening-spread walk-forward — equity at k=8, raw and after "
                 "the outlier-realism haircut",
                 fontsize=13.2, x=0.062, ha="left", y=0.972, color=NAVY,
                 fontweight="bold")
    fig.text(0.062, 0.928,
             "500 bets, 2023-24 … 2025-26, priced at the OPENING spread at −110, "
             "flat 1u, no compounding, no calendar filter.  The configuration is "
             "chosen on seasons 1..k and scored on k+1 only.\n"
             "The haircut charges for the 8.1% of best-of-N prices sitting more "
             "than 1.5 points off the next book — the ones a book limits, "
             "lowers or voids.\n"
             "Season-clustered 95% CI on the raw path is [−6.72%, +24.17%]: it "
             "contains zero, and 2024-25 alone supplies 65% of the P&L.",
             fontsize=9.2, color=INK2, ha="left", va="top", linespacing=1.5)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.795, bottom=0.105)
    out = ROOT / "charts" / "equity_2023_26_k8_haircut.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print("wrote", out)
    for v, _, lab, roi in ends:
        print(f"  {lab:34s} {v:+7.2f}u  {roi:+6.2f}%")
    print(f"  haircut charge: {hi_v-lo_v:.2f}u")


if __name__ == "__main__":
    main()
