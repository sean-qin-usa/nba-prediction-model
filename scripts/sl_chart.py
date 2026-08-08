#!/usr/bin/env python3
"""charts/structural_lookahead.png — house style, 150 dpi.

  LEFT   D166's cumulative equity curve re-drawn for the FULL shipped model and
         for progressively more primitive structures, firm-default execution.
  RIGHT  leave-one-season-out influence on the pooled ROI, all 14 scored
         seasons, at the retail 1-book tier and at the firm default.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
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

EXTRAP = {"2018-19", "2019-20", "2020-21", "2021-22", "2022-23",
          "2024-25", "2025-26"}


def main():
    R = json.load(open(ROOT / "data" / "sl_score.json"))
    V = R["Q1_variants"]
    seas = V["V0_FULL"]["seasons"]
    x = np.arange(len(seas))

    fig = plt.figure(figsize=(15.6, 6.9), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.06, 1.0], wspace=0.20)

    # ------------------------------------------------------------- LEFT
    ax = fig.add_subplot(gs[0, 0])
    for s in EXTRAP:
        i = seas.index(s)
        ax.axvspan(i - 0.5, i + 0.5, color="#f4f3ef", zorder=0)
    ax.axhline(0, color=INK2, lw=1.0, ls="--", zorder=1)
    series = [
        ("V0_FULL", "FULL shipped stack  +3.54%", C_MODEL, 2.8, "-"),
        ("V1_noTANK", "- D73 tank  +1.68%", "#7aa8dd", 1.6, "-"),
        ("V2_noTANK_noBRIDGE", "- tank - D84A bridge  +1.35%", "#9fbfe6", 1.5, "-"),
        ("V3_noTANK_noBRIDGE_noCARRY",
         "- tank - bridge - D62 carry  -0.16%", "#b48ead", 2.0, "--"),
        ("V5_FF_ONLY", "four-factors + home only  -1.79%", "#e0a35c", 1.7, "-."),
        ("V4_STRIPPED", "STRIPPED: 4F + composition + home  -3.70%",
         C_BAD, 2.8, "-"),
    ]
    for key, lab, col, lw, ls in series:
        y = np.array(V[key]["cum_path"], float)
        ax.plot(x, y, color=col, lw=lw, ls=ls, marker="o", ms=3.0,
                label=lab, zorder=3)
        # pass 3: the -2.2u end label sat on the break-even dashed line; end
        # labels within 4u of zero are nudged clear of it.
        dy = 3.0 if 0 > y[-1] > -4.0 else (-3.0 if 0 <= y[-1] < 4.0 else 0.0)
        ax.text(x[-1] + 0.14, y[-1] - dy, f"{y[-1]:+.0f}u", color=col,
                fontsize=8.5, va="center", ha="left", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s[2:] for s in seas], fontsize=8.5, rotation=0)
    ax.set_xlim(-0.6, len(seas) - 0.05 + 1.05)
    ax.set_ylabel("cumulative P&L, units (flat 1u)", fontsize=9.5)
    ax.set_title("(a) THE SAME WALK-FORWARD SELECTION PROCEDURE, ON\n"
                 "PROGRESSIVELY LESS FUTURE-INFORMED MODEL STRUCTURES",
                 fontsize=10.5, fontweight="bold", loc="left", pad=9)
    ax.legend(fontsize=8.2, loc="lower left", frameon=True, framealpha=0.95,
              edgecolor=GRID, ncol=1, borderpad=0.6)
    ax.text(0.5, -0.135, "firm-default execution: D163 k=5 measured panels + "
            "outlier-realism haircut.  shaded = execution EXTRAPOLATED "
            "(D166 §2).\nModel STRUCTURE is fixed within each line; only the "
            "betting config is re-selected walk-forward (D164/D165/D166's own "
            "loop, unchanged).",
            transform=ax.transAxes, fontsize=7.8, color=INK2, ha="center",
            va="top")

    # ------------------------------------------------------------ RIGHT
    ax2 = fig.add_subplot(gs[0, 1])
    l1 = R["Q2_loso"]["rows_1book"]
    l5 = R["Q2_loso"]["rows_default"]
    p1, p5 = R["Q2_loso"]["pooled_1book"], R["Q2_loso"]["pooled_default"]
    w = 0.38
    d1 = np.array([100 * r["roi_drop"] for r in l1])
    d5 = np.array([100 * r["roi_drop"] for r in l5])
    ax2.axhline(0, color=INK, lw=1.1, zorder=2)
    ax2.bar(x - w / 2, d1, width=w, color=C_MKT, zorder=3,
            label=f"1 book (retail): pooled {100*p1:+.2f}%")
    ax2.bar(x + w / 2, d5, width=w, color=C_MODEL, zorder=3,
            label=f"firm default k=5+haircut: pooled {100*p5:+.2f}%")
    ax2.axhline(100 * p1, color=C_MKT, lw=1.1, ls=":", zorder=2)
    ax2.axhline(100 * p5, color=C_MODEL, lw=1.1, ls=":", zorder=2)
    i25 = seas.index("2024-25")
    ax2.annotate(f"drop 2024-25\n1 book {d1[i25]:+.2f}%  FLIPS NEGATIVE\n"
                 f"firm {d5[i25]:+.2f}% stays positive",
                 xy=(i25 - w / 2, d1[i25]), xytext=(i25 - 5.9, -3.15),
                 fontsize=8.0, color=C_BAD, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_BAD, lw=1.1),
                 ha="left", va="center", zorder=5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([s[2:] for s in seas], fontsize=8.5)
    ax2.set_xlabel("season LEFT OUT", fontsize=9.5, labelpad=2)
    ax2.set_ylabel("pooled ROI with that season removed, %", fontsize=9.5)
    ax2.set_ylim(-4.2, 8.4)
    ax2.set_title("(b) LEAVE-ONE-SEASON-OUT INFLUENCE ON THE POOLED ROI,\n"
                  "ALL 14 SCORED SEASONS — DOES THE RESULT REST ON ONE YEAR?",
                  fontsize=10.5, fontweight="bold", loc="left", pad=9)
    ax2.legend(fontsize=8.4, loc="upper left", frameon=True, framealpha=0.95,
               edgecolor=GRID)
    # influence labels sit just above each firm-tier bar, never at a fixed y
    # (pass 2: a fixed y put them through the legend box and the caption)
    for i in range(len(x)):
        ax2.text(i + w / 2, max(d5[i], 0.0) + 0.18,
                 f"{100*(p5-l5[i]['roi_drop']):+.2f}", fontsize=7.0,
                 color=INK2, ha="center", va="bottom", rotation=90)
    ax2.text(len(x) - 0.55, 8.15, "value above each blue bar =\ninfluence in pp "
             "on the firm tier", fontsize=7.4, color=INK2, ha="right", va="top")
    ax2.text(0.5, -0.135,
             "dotted = the full-sample pooled ROI.  Every bar is the pooled ROI "
             "over the OTHER 13 seasons.\nNo interval here excludes zero: MDE80 "
             "is 8.10 ROI points at K=14 (D166 §8), and this is a STABILITY "
             "diagnostic, not 14 proofs.",
             transform=ax2.transAxes, fontsize=7.8, color=INK2, ha="center",
             va="top")

    fig.suptitle("STRUCTURAL LOOKAHEAD — HOW MUCH OF D166's +3.54% IS THE "
                 "SELECTION PROCEDURE, AND HOW MUCH IS A MODEL BUILT WITH "
                 "2021-26 IN HAND?",
                 fontsize=12.4, fontweight="bold", color=INK, y=0.985)
    fig.subplots_adjust(top=0.855, bottom=0.175, left=0.052, right=0.985)
    out = ROOT / "charts" / "structural_lookahead.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
