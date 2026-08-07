#!/usr/bin/env python3
"""D210 — the review's equity figure, in the ORIGINAL single-panel layout.

The owner's original report used ONE panel: cumulative P&L in units against the
SEQUENTIAL BET INDEX (not a date axis), with dashed season dividers and a
break-even rule. The two-panel date-axis version that replaced it was rejected.
This restores the original shape with current numbers.

charts/review_equity.png
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

NAVY, BROWN, RED = "#1f3864", "#8a6d3b", "#c62828"
INK2, GRID = "#52514e", "#e7e6e2"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 9,
})
# D215: the headline table reports the recent three seasons, so the equity path
# must show the same block or the figure and the table disagree.
FROM, TIER = "2023-24", "k=9 raw"


def main():
    pb = json.load(open(ROOT / "data" / "wf_perbet_OFFSET.json"))[TIER]
    bets = sorted((b for b in pb if b["season"] >= FROM),
                  key=lambda b: (b["date"], b["gid"]))
    ev = np.array([b["ev"] * b["keep"] for b in bets], float)
    cum = np.cumsum(ev)
    seas = [b["season"] for b in bets]
    n = len(ev)

    fig, ax = plt.subplots(figsize=(10.6, 2.85), dpi=200)
    ax.axhline(0, color=INK2, lw=0.9, ls="--", zorder=3)
    ax.plot(np.arange(n), cum, color=NAVY, lw=1.5, zorder=5)

    starts = []
    seen = set()
    for i, s in enumerate(seas):
        if s not in seen:
            seen.add(s)
            starts.append((i, s))
    ylo, yhi = ax.get_ylim()
    for i, s in starts[1:]:
        ax.axvline(i - 0.5, color=BROWN, lw=0.9, ls=(0, (5, 2.5)), zorder=4)
    for i, s in starts:
        ax.text(i + 4, ylo + 0.035 * (yhi - ylo), s, color=BROWN, fontsize=7.4,
                ha="left", va="bottom")

    # Max DRAWDOWN (peak-to-trough), not the raw minimum: on a block that rises
    # from bet 1 the raw minimum is ~0 at bet 2 and says nothing.
    dd = cum - np.maximum.accumulate(cum)
    j = int(np.argmin(dd))
    ax.plot([j], [cum[j]], marker="v", ms=5, color=RED, zorder=6)
    ax.annotate(f"max drawdown {dd.min():+.1f}u", xy=(j, cum[j]),
                xytext=(10, -12), textcoords="offset points", color=RED,
                fontsize=7.2)

    ax.set_ylim(ylo, yhi)
    ax.set_xlim(-8, n * 1.02)
    ax.set_xlabel(f"sequential bet, in date order  (n = {n} across the three "
                  f"scored seasons 2023-24 … 2025-26)", fontsize=8.4)
    ax.set_ylabel("cumulative P&L, units  (flat 1u per bet)", fontsize=8.4)
    ax.text(n * 0.995, 0.6, "break-even (flat)", ha="right", va="bottom",
            fontsize=7.4, color=INK2)
    ax.annotate(f"{cum[-1]:+.1f}u", xy=(n - 1, cum[-1]), xytext=(-4, 6),
                textcoords="offset points", color=NAVY, fontsize=8.4,
                fontweight="bold", ha="right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.965, bottom=0.165)
    out = ROOT / "charts" / "review_equity.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print("wrote", out)
    print(f"  n={n}  final {cum[-1]:+.2f}u  maxDD {dd.min():+.2f}u at bet {j+1}")


if __name__ == "__main__":
    main()
