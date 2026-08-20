"""charts/adaptive_selection.png — does selecting from RECENT data help?

House style per scripts/make_status_charts.py. 150 dpi.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import nbapred.threads  # noqa: E402
nbapred.threads.pin(1)

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

C_MODEL, C_MKT = "#2a78d6", "#eb6834"
C_GOOD, C_BAD = "#1baf7a", "#d1495b"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e7e6e2"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlecolor": INK,
})
BB = dict(facecolor="white", edgecolor="none", alpha=0.92,
          boxstyle="square,pad=0.12")

ARMS = ["A_REC1", "B_REC2", "B_REC3", "B_REC5", "C_ALL", "D_ONLINE", "D_HYBRID"]
LAB = {
    "A_REC1": "A  RECENCY-1\nseason k only\n(PRIMARY)",
    "B_REC2": "B  RECENCY-2\nlast 2 seasons",
    "B_REC3": "B  RECENCY-3\nlast 3 seasons",
    "B_REC5": "B  RECENCY-5\nlast 5 seasons",
    "C_ALL": "C  ALL-HISTORY\nseasons 1..k\n(D164 benchmark)",
    "D_ONLINE": "D  ONLINE\nseason-to-date\nonly",
    "D_HYBRID": "D  HYBRID\nall history +\nseason-to-date",
}


def main():
    r = json.load(open(ROOT / "data" / "as_adaptive.json"))
    A = r["arms"]
    seasons = r["seasons"]

    fig = plt.figure(figsize=(15.6, 9.6), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.28, 1.0],
                          width_ratios=[1.06, 1.0], hspace=0.50, wspace=0.17)

    # ============================================ PANEL A: the seven arms
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(ARMS))
    roi = np.array([A[a]["pooled_roi"] for a in ARMS]) * 100
    lo = np.array([A[a]["ci"]["lo"] for a in ARMS]) * 100
    hi = np.array([A[a]["ci"]["hi"] for a in ARMS]) * 100
    nmean = np.array([A[a]["null"]["mean"] for a in ARMS]) * 100
    n05 = np.array([A[a]["null"]["p05"] for a in ARMS]) * 100
    n95 = np.array([A[a]["null"]["p95"] for a in ARMS]) * 100

    emax = r["null"]["family_max"]["mean"] * 100
    emax95 = r["null"]["family_max"]["p95"] * 100

    # each arm's OWN noise floor, drawn alongside its bar
    for xi, a, m, p5, p95 in zip(x, ARMS, nmean, n05, n95):
        ax.add_patch(Rectangle((xi + 0.06, p5), 0.30, p95 - p5, facecolor=C_BAD,
                               alpha=0.16, edgecolor="none", zorder=1))
        ax.plot([xi + 0.06, xi + 0.36], [m, m], color=C_BAD, lw=1.6, zorder=3)

    bw = 0.36
    cols = [C_GOOD if v > 0 else C_BAD for v in roi]
    ax.bar(x - 0.21, roi, bw, color=cols, zorder=3)
    ax.errorbar(x - 0.21, roi, yerr=[roi - lo, hi - roi], fmt="none",
                ecolor=INK2, elinewidth=1.3, capsize=4.5, capthick=1.3, zorder=5)
    ax.axhline(0, color=INK, lw=1.2, zorder=4)
    ax.axhline(emax, color=C_MKT, ls=(0, (5, 3)), lw=1.4, zorder=2)
    ax.axhline(emax95, color=C_MKT, ls=(0, (1, 2.5)), lw=1.2, zorder=2)

    for xi, v, l, h, a in zip(x, roi, lo, hi, ARMS):
        ax.text(xi - 0.21, h + 0.55, f"{v:+.2f}%", ha="center", va="bottom",
                fontsize=10.2, color=INK, fontweight="bold", zorder=7, bbox=BB)
        ax.text(xi - 0.21, l - 0.55, f"net of own null {100*A[a]['net_of_null']:+.2f}\n"
                f"p {A[a]['p_value']:.3f}   MDE80 {100*A[a]['mde80']:.1f}",
                ha="center", va="top", fontsize=7.9, color=INK2, zorder=7,
                linespacing=1.35, bbox=BB)

    ax.set_xticks(x, [LAB[a] for a in ARMS], fontsize=9.0, linespacing=1.5)
    ax.set_ylim(-14.5, 16.2)
    ax.set_ylabel("pooled ATS ROI at -110, 14 unseen seasons (%)")
    ax.set_title("Seven selection procedures, one scored track (2012-13..2025-26, "
                 "the same 600-cell space): NOT ONE INTERVAL EXCLUDES ZERO",
                 fontsize=12.8, pad=34, loc="left")
    ax.text(0.0, 1.048,
            "the owner's proposal (A, select on the last season only) is the WORST "
            "season-window arm on the board and is 4.31 points below the "
            "all-history benchmark it was meant to beat",
            transform=ax.transAxes, fontsize=9.5, color=INK2, ha="left")
    hd = [Patch(fc=C_GOOD, label="pooled ROI, positive"),
          Patch(fc=C_BAD, alpha=0.3,
                label="that arm's OWN noise floor: the identical procedure on "
                      "permuted\npredictions, 200 draws (bar = mean, band = 5-95%)"),
          Patch(fc=C_BAD, label="pooled ROI, negative"),
          plt.Line2D([], [], color=C_MKT, ls=(0, (5, 3)), lw=1.4,
                     label=f"best of these 7 arms on PURE NOISE: E[max] = {emax:+.2f}%"),
          plt.Line2D([], [], color=INK2, lw=1.3, marker="_", ms=9,
                     label="13-dof season-cluster t interval"),
          plt.Line2D([], [], color=C_MKT, ls=(0, (1, 2.5)), lw=1.2,
                     label=f"...and its 95th percentile, {emax95:+.2f}%   "
                           f"(max of 200: {100*r['null']['family_max']['max']:+.2f}%)"),
          plt.Line2D([], [], color=INK, lw=1.2,
                     label="breakeven at -110  (ROI = 0, cover 52.381%)")]
    ax.legend(handles=hd, loc="upper left", frameon=False, fontsize=8.5, ncol=2,
              labelspacing=0.5, columnspacing=1.6, handlelength=2.4,
              bbox_to_anchor=(0.002, 1.002))

    # ================================ PANEL B: the manufacturing ladder
    ax2 = fig.add_subplot(gs[1, 0])
    ladder = ["A_REC1", "B_REC2", "B_REC3", "B_REC5", "C_ALL"]
    xl = np.arange(len(ladder))
    sel = np.array([A[a]["mean_sel_roi"] for a in ladder]) * 100
    tst = np.array([A[a]["pooled_roi"] for a in ladder]) * 100
    seln = [A[a]["mean_sel_n"] for a in ladder]

    ax2.fill_between(xl, tst, sel, color=C_MKT, alpha=0.14, zorder=1)
    ax2.plot(xl, sel, color=C_MKT, lw=2.0, marker="o", ms=5.5, zorder=4,
             label="ROI on the window it was SELECTED on (in sample)")
    ax2.plot(xl, tst, color=C_MODEL, lw=2.0, marker="o", ms=5.5, zorder=4,
             label="ROI on the next UNSEEN season")
    ax2.axhline(0, color=INK, lw=1.1, zorder=3)
    for xi, s, t in zip(xl, sel, tst):
        ax2.annotate("", xy=(xi, t), xytext=(xi, s), zorder=5,
                     arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9,
                                     shrinkA=3, shrinkB=3))
        ax2.text(xi + 0.10, (s + t) / 2, f"{s-t:+.1f}", ha="left", va="center",
                 fontsize=9.0, color=INK, fontweight="bold", zorder=7, bbox=BB)
    ax2.set_xticks(xl, ["1 season\nn~123", "2 seasons\nn~239", "3 seasons\nn~357",
                        "5 seasons\nn~590", "ALL history\nn~1405"],
                   fontsize=8.8, linespacing=1.5)
    ax2.set_xlim(-0.35, 4.42)
    ax2.set_ylim(-6.5, 18.5)
    ax2.set_ylabel("ATS ROI at -110 (%)")
    ax2.set_xlabel("length of the selection window   (mean bets it was chosen on)",
                   fontsize=9.0, labelpad=6)
    ax2.set_title("THE MANUFACTURING LADDER: a shorter window buys capacity, not edge",
                  fontsize=11.4, loc="left", pad=22)
    ax2.text(0.0, 1.035, "the gap is the DECAY — what the search invented and the "
             "next season took back. D164's measured capacity: +16.92",
             transform=ax2.transAxes, fontsize=8.8, color=INK2, ha="left")
    ax2.legend(loc="upper right", frameon=False, fontsize=8.5, labelspacing=0.4)

    # ================================ PANEL C: the adjacency effect
    ax3 = fig.add_subplot(gs[1, 1])
    ad = r["adjacency"]
    rows = [z for z in ad["rows"] if np.isfinite(z["prem1"])]
    xp = np.arange(len(rows))
    prem = np.array([z["prem1"] for z in rows]) * 100
    nl = r["null"]["premium_lag1"]
    p1 = ad["premium_lag1"]

    ax3.axhspan(100 * nl["p05"], 100 * nl["p95"], color=INK2, alpha=0.13,
                zorder=0, lw=0)
    ax3.bar(xp, prem, 0.66, color=[C_GOOD if v > 0 else C_BAD for v in prem],
            zorder=3)
    ax3.axhline(0, color=INK, lw=1.1, zorder=4)
    m = 100 * p1["mean"]
    ax3.axhline(m, color=C_MODEL, lw=2.0, zorder=5)
    ax3.add_patch(Rectangle((-0.7, 100 * p1["ci"]["lo"]), len(rows) + 0.4,
                            100 * (p1["ci"]["hi"] - p1["ci"]["lo"]),
                            facecolor=C_MODEL, alpha=0.10, edgecolor="none",
                            zorder=1))
    ax3.set_xticks(xp, [seasons[z["k"]][2:] for z in rows], fontsize=8.0,
                   rotation=90)
    ax3.set_xlim(-0.75, len(rows) - 0.25)
    ax3.set_ylim(-23, 22)
    ax3.set_ylabel("ROI on season k+1 minus ROI on all other seasons (pts)")
    ax3.set_xlabel("season k the configuration was tuned on", fontsize=9.0,
                   labelpad=4)
    ax3.set_title("THE MECHANISM TEST: is the NEXT season special? No.",
                  fontsize=11.4, loc="left", pad=22)
    ax3.text(0.0, 1.035, "regime continuity is what the owner's proposal rests on; "
             "this is the quantity it predicts should be positive",
             transform=ax3.transAxes, fontsize=8.8, color=INK2, ha="left")
    ax3.text(len(rows) - 0.4, 21.2,
             f"ADJACENCY PREMIUM  {m:+.3f} pts   {p1['positive']}/{p1['n']} positive\n"
             f"17-dof CI [{100*p1['ci']['lo']:+.2f},{100*p1['ci']['hi']:+.2f}]  ns\n"
             f"permutation reference {100*nl['mean']:+.3f}   p = {p1['p_value']:.3f}",
             ha="right", va="top", fontsize=8.8, color=INK, zorder=8,
             linespacing=1.5, bbox=dict(facecolor="white", edgecolor=GRID,
                                        boxstyle="round,pad=0.35"))
    ax3.text(0.02, 0.055, "grey band = permutation reference, 5-95%",
             transform=ax3.transAxes, fontsize=8.2, color=INK2, zorder=8, bbox=BB)

    fig.suptitle("DOES SELECTING FROM RECENT DATA BEAT SELECTING FROM ALL "
                 "HISTORY?   No — and the mechanism it needs does not exist",
                 fontsize=14.2, x=0.008, ha="left", y=0.988, color=INK)
    fig.text(0.008, 0.951,
             "600-cell search space, D162's 19-season ATS frame (22,742 games, "
             "opening spreads, -110, pushes in the ROI denominator). "
             "Every arm scored on the SAME 14 unseen seasons. Pre-registered and "
             "hashed before scoring. DIAGNOSTIC \u2014 nothing ships.",
             fontsize=8.2, color=INK2, ha="left")

    fig.subplots_adjust(left=0.052, right=0.988, top=0.858, bottom=0.075)
    p = ROOT / "charts" / "adaptive_selection.png"
    fig.savefig(p, dpi=150, facecolor="white")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
