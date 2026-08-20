#!/usr/bin/env python3
"""charts/recent_equity_perbet.png — the owner's request: "another chart with
results only from 24-26.  can you make it continuous, per bet?"

BET-BY-BET cumulative equity across 2024-25 and 2025-26 ONLY, x = sequential
bet index in date order (NOT season), y = cumulative P&L in units at flat
1u/bet.  Built on the D173 re-run of the walk-forward loop on the D170/D171
backfilled data — not on the stale pre-backfill inputs.

House style per scripts/make_status_charts.py: model blue / market orange, thin
marks, recessive grid, negative bar labels inside bars, no dual axes, 150 dpi.

Read-only.  Nothing ships.  No default changed.
"""
from __future__ import annotations

import json
import os
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

C_MODEL, C_MKT = "#2a78d6", "#eb6834"
C_GOOD, C_BAD = "#1baf7a", "#d1495b"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e7e6e2"
BROWN = "#8a6d3b"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlecolor": INK,
})

WINDOW = ("2024-25", "2025-26")
ROLL = 50                       # bets; stated on the chart
TIERS = [("k=1 raw", "#a9bdd0", "1 book (retail reference)", 1.6, "-"),
         ("k=8 raw", "#12406f", "k=8 raw (optimistic bound)", 1.6, "-"),
         ("k=5 +haircut", C_MODEL, "k=5 measured + haircut  (FIRM DEFAULT)",
          3.0, "--")]


def main():
    tag = os.environ.get("WF_TAG", "_HONEST")
    pb = json.load(open(ROOT / "data" / f"wf_perbet{tag}.json"))
    R = json.load(open(ROOT / "data" / f"wf_equity{tag}.json"))
    D14 = R["tiers"]["k=5 +haircut"]

    series = {}
    for nm, *_ in TIERS:
        rows = [b for b in pb[nm] if b["season"] in WINDOW]
        rows.sort(key=lambda b: (b["date"], b["gid"]))
        series[nm] = rows
    n_bets = len(series["k=5 +haircut"])
    dates = [b["date"] for b in series["k=5 +haircut"]]
    seas = [b["season"] for b in series["k=5 +haircut"]]
    tags = sorted({b["tag"] for r in series.values() for b in r})
    assert tags == ["EXTRAPOLATED"], f"expected all EXTRAPOLATED, got {tags}"
    bnd = next(i for i, s in enumerate(seas) if s == "2025-26")

    fig = plt.figure(figsize=(15.6, 9.6), dpi=150)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.85, 1.0], hspace=0.42,
                          left=0.062, right=0.985, top=0.845, bottom=0.135)

    # ------------------------------------------------------------- PANEL A
    ax = fig.add_subplot(gs[0, 0])
    ax.axhline(0, color=INK2, lw=1.1, ls="--", zorder=2)
    ends = []
    for nm, col, lab, lw, ls in TIERS:
        ev = np.array([b["ev"] * b["keep"] for b in series[nm]], float)
        cum = np.cumsum(ev)
        ax.plot(np.arange(len(cum)), cum, color=col, lw=lw, ls=ls,
                dashes=(4, 2.2) if ls == "--" else (None, None), zorder=5)
        ends.append((cum[-1], lab, col, len(ev), 100 * ev.sum() / len(ev)))

    # de-collide the end labels
    ends.sort(key=lambda t: -t[0])
    ylo0, yhi0 = ax.get_ylim()
    gap = 0.085 * (yhi0 - ylo0)
    yp, prev = [], None
    for v, *_ in ends:
        y = v if prev is None else min(v, prev - gap)
        prev = y
        yp.append(y)
    for (v, lab, col, nn, roi), y in zip(ends, yp):
        ax.annotate(f"{lab}\n{v:+.1f}u over {nn} bets   ({roi:+.2f}% ROI)",
                    xy=(n_bets - 1, v), xytext=(n_bets + 6, y), color=col,
                    fontsize=9.2, va="center", ha="left",
                    fontweight="bold" if "DEFAULT" in lab else "normal",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.7,
                                    shrinkA=0, shrinkB=1, alpha=0.6))

    ax.axvline(bnd - 0.5, color=BROWN, lw=1.2, ls=(0, (5, 2.5)), zorder=4)
    ax.set_xlim(-4, n_bets * 1.30)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi)                    # no clipping: limits stay as drawn
    ax.text(bnd + 2, ylo + 0.055 * (yhi - ylo), "2025-26 season starts",
            color=BROWN, fontsize=8.6, ha="left", va="bottom")
    ax.text(2, ylo + 0.055 * (yhi - ylo), "2024-25", color=BROWN,
            fontsize=8.6, ha="left", va="bottom")

    # a date axis under the bet index: FIRST bet of each distinct month, so a
    # label never repeats (the configs are phase-restricted, so the two seasons
    # occupy Feb-Apr 2025 and Oct-Dec 2025 rather than full seasons)
    ticks, tlabs, seen = [], [], set()
    for i, dt in enumerate(dates):
        if dt[:7] not in seen:
            seen.add(dt[:7])
            ticks.append(i)
            tlabs.append(dt[:7])
    ax.set_xticks(ticks, tlabs, fontsize=8.8)
    ax.set_xlabel(f"sequential bet, in date order  (n = {n_bets} across the two "
                  f"seasons; tick labels are the month of that bet)")
    ax.set_ylabel("cumulative P&L, units  (flat 1u per bet)")
    ax.text(n_bets * 0.985, 0, "break-even (flat)", ha="right", va="bottom",
            fontsize=8.6, color=INK2)
    ax.set_title("(a)  CONTINUOUS, PER BET — every bet the walk-forward strategy "
                 "placed in 2024-25 and 2025-26, in the order it placed them",
                 fontsize=11.8, pad=11, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ------------------------------------------------------------- PANEL B
    axb = fig.add_subplot(gs[1, 0])
    ev = np.array([b["ev"] * b["keep"] for b in series["k=5 +haircut"]], float)
    x = np.arange(len(ev))
    axb.scatter(x, ev, s=8, color=C_MODEL, alpha=0.30, zorder=3,
                linewidths=0,
                label="raw per-bet P&L, units (+0.909 win / -1.000 loss) — never replaced")
    roll = np.convolve(ev, np.ones(ROLL) / ROLL, mode="valid")
    axb.plot(np.arange(ROLL - 1, len(ev)), roll, color=C_MODEL, lw=2.2,
             zorder=5, label=f"{ROLL}-bet rolling mean (units/bet = ROI as a fraction)")
    axb.axhline(0, color=INK2, lw=1.0, zorder=4)
    pooled_u = ev.sum() / len(ev)
    pooled = 100 * pooled_u
    axb.axhline(pooled_u, color=BROWN, lw=1.3, ls=(0, (5, 2.5)), zorder=4)
    axb.axvline(bnd - 0.5, color=BROWN, lw=1.2, ls=(0, (5, 2.5)), zorder=4)
    axb.set_xlim(-4, n_bets * 1.30)
    axb.text(n_bets + 6, pooled_u + 0.30,
             f"2-season pooled\n{pooled_u:+.4f} u/bet  =  {pooled:+.2f}% ROI",
             color=BROWN, fontsize=8.8, va="center", ha="left")
    axb.annotate("", xy=(n_bets + 2, pooled_u), xytext=(n_bets + 6, pooled_u + 0.24),
                 arrowprops=dict(arrowstyle="-", color=BROWN, lw=0.7, alpha=0.6))
    axb.set_xticks(ticks, tlabs, fontsize=8.8)
    axb.set_ylabel("P&L per bet, units")
    axb.set_xlabel("sequential bet, in date order")
    axb.legend(frameon=False, fontsize=8.4, loc="upper left", ncol=2)
    axb.set_title(f"(b)  THE SAME BETS, UNSMOOTHED, WITH A {ROLL}-BET ROLLING "
                  "MEAN OVER THE TOP — the raw path is the evidence; the line "
                  "is only an aid to the eye",
                  fontsize=10.6, pad=10, loc="left")
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)

    # ------------------------------------------------------------- titles
    per = {s: [b["ev"] * b["keep"] for b in series["k=5 +haircut"]
               if b["season"] == s] for s in WINDOW}
    roi_s = {s: 100 * sum(v) / len(v) for s, v in per.items()}
    sd = np.std([roi_s[s] for s in WINDOW], ddof=1)
    mde80 = (12.706 + 0.8416) * sd / np.sqrt(2)      # K=2 -> 1 dof
    tot14 = sum(r["pay"] for r in D14["rows"])
    sh2425 = 100 * next(r["pay"] for r in D14["rows"]
                        if r["season"] == "2024-25") / tot14

    fig.suptitle("THE RECENT WINDOW, BET BY BET — 2024-25 and 2025-26 only, "
                 "walk-forward strategy at the opening spread",
                 fontsize=13.8, x=0.062, ha="left", y=0.982, color=INK)
    fig.text(0.062, 0.955,
             "RE-RUN ON THE D170/D171 BACKFILLED DATA (D173).  No-lookahead: the "
             "configuration is chosen on seasons 1..k and scored on k+1 only; "
             "nothing here is re-selected.  Priced at the OPENING spread at -110.",
             fontsize=9.4, color=INK2, ha="left", va="top")

    out = ROOT / "charts" / "recent_equity_perbet.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print("wrote", out)
    print(f"  n={n_bets}  MDE80(K=2)={mde80:.1f}pp  "
          f"2024-25 share of 14-season total = {sh2425:.1f}%")


if __name__ == "__main__":
    main()
