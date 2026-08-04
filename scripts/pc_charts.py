#!/usr/bin/env python3
"""charts/pca_spectrum_and_cost.png — the eigenvalue spectrum of the rejected
pile beside the cost-vs-components curve, with D154's raw-column ladder and the
pure-noise benchmark overlaid.

House style (scripts/make_status_charts.py): white ground, recessive grid,
thin marks, direct labels, no dual axes, 150 dpi.

  python scripts/pc_charts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import nbapred.threads as _t  # noqa: E402

_t.pin(1)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C_MODEL, C_MKT = "#2a78d6", "#eb6834"
C_GOOD, C_BAD = "#1baf7a", "#d1495b"
C_PURPLE, C_TEAL = "#7b5ea7", "#0f8a8a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e7e6e2"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlecolor": INK,
})


def despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    spec = json.load(open(REPO / "data" / "pca_spectrum.json"))
    pca = json.load(open(REPO / "data" / "pca_ladder.json"))
    ca = json.load(open(REPO / "data" / "carryall_ladder.json"))
    A = pca["arms"]

    fig = plt.figure(figsize=(15.2, 9.4), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.22],
                          width_ratios=[1.0, 1.0], hspace=0.42, wspace=0.22)

    # ================= PANEL A — scree ==================================
    ax = fig.add_subplot(gs[0, 0])
    p15 = spec["blocks"]["PILE15"]
    ev = np.array(p15["ev"])
    lo, hi = p15["mp_edges"]
    ax.axhspan(lo, hi, color=GRID, alpha=0.9, zorder=0)
    ax.bar(np.arange(1, 16), ev, color=C_MODEL, width=0.62, zorder=3)
    th = np.array(spec["blocks"]["TEAMHOME"]["ev"])[:29]
    ax.plot(np.linspace(1, 15, 29), th, color=C_MKT, lw=2.0, zorder=4)
    ax.text(10.2, 1.44, "the 30 TEAM-HOME dummies: 29 eigenvalues of 1.035\n"
            "— orthogonal by construction, NOTHING TO COMPRESS",
            color=C_MKT, fontsize=8.5, ha="center")
    ax.text(15.4, 2.38, "shaded = Marchenko-Pastur null band for 15\n"
            "independent columns at n=2460.\n"
            "5 eigenvalues sit above it, 6 below: real\n"
            "structure, but spread over many directions",
            fontsize=8, color=INK2, va="top", ha="right")
    ax.axhline(1.0, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("component  (15-column rejected pile; team-home rescaled to fit)")
    ax.set_ylabel("eigenvalue of the correlation matrix")
    ax.set_title("A. The pile is NOT low-rank: the top component holds only "
                 "13.6% of the variance", fontsize=10.5, loc="left")
    ax.set_xlim(0.3, 15.7)
    ax.set_ylim(0, 2.42)
    for i, v in enumerate(ev[:6]):
        ax.text(i + 1, v + 0.05, f"{100*v/ev.sum():.1f}%", ha="center",
                fontsize=7.6, color=INK2)
    despine(ax)

    # ================= PANEL B — cumulative variance ====================
    ax = fig.add_subplot(gs[0, 1])
    for key, col, lab, dx, dy in (
            ("PILE13", C_MODEL, "rotated pile (13 cols)", -60, 4),
            ("JOINT43", C_PURPLE, "pile + team-home (43 cols)", -49, -28),
            ("TEAMHOME", C_MKT, "team-home only (30 cols)", -22, -56)):
        b = spec["blocks"][key]
        c = np.array(b["cum"]) * 100
        x = np.arange(1, len(c) + 1) / b["p"] * 100
        ax.plot(x, c, color=col, lw=2.0, label=lab)
        r = b["r"]["90"]
        ax.plot([r / b["p"] * 100], [c[r - 1]], "o", ms=6, color=col, zorder=5)
        ax.annotate(f"90% of the variance\nneeds {r} of {b['p']}",
                    xy=(r / b["p"] * 100, c[r - 1]),
                    xytext=(r / b["p"] * 100 + dx, c[r - 1] + dy),
                    color=col, fontsize=8.4,
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.9))
    ax.plot([0, 100], [0, 100], color=INK2, lw=0.9, ls="--")
    ax.text(56, 46, "no compression at all  (y = x)", color=INK2, fontsize=8.2,
            rotation=31, ha="center", va="top")
    ax.axhline(90, color=GRID, lw=1.4)
    ax.set_xlabel("% of components kept, variance-ordered")
    ax.set_ylabel("% of variance explained")
    ax.set_title("B. Every block hugs the no-compression diagonal\n"
                 "(80/90/95% of the 15 raw columns needs 10/12/14)",
                 fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.6, loc="lower right")
    ax.set_xlim(0, 101)
    ax.set_ylim(0, 104)
    despine(ax)

    # ================= PANEL C — cost vs coefficients, the pile =========
    ax = fig.add_subplot(gs[1, 0])
    xs = np.arange(0, 16.5, 0.5)
    ax.plot(xs, -1.0e-4 * xs, color=INK2, lw=1.6, ls="--", zorder=2,
            label="pure-noise benchmark, -1.0e-4 per coefficient (D154)")
    nk = [(ca["arms"][f"noise:k{k}"]["ncol"], ca["arms"][f"noise:k{k}"]["delta"])
          for k in (1, 2, 3, 5, 8, 10, 15)]
    ax.plot([a for a, _ in nk], [b for _, b in nk], "s", ms=4.5,
            color=INK2, alpha=0.5, zorder=3,
            label="D154 measured pure-noise arms")
    rk = [(ca["arms"][f"pile:k{k}"]["ncol"], ca["arms"][f"pile:k{k}"]["delta"])
          for k in (1, 2, 3, 5, 8, 10, 15)]
    ax.plot([a for a, _ in rk], [b for _, b in rk], "o-", ms=5, lw=1.7,
            color=C_BAD, zorder=4,
            label="D154 RAW columns, cumulative  (all 15 = -0.00571)")
    rr = list(range(0, 14))
    ax.plot([r + 2 for r in rr], [A[f"pca:r{r}"]["delta"] for r in rr],
            "o-", ms=4.5, lw=2.2, color=C_MODEL, zorder=5,
            label="PCA, variance-ordered   ARM A (r=11) = -0.00552, "
                  "paired +0.00019 ns")
    ax.plot([r + 2 for r in rr], [A[f"pcat:r{r}"]["delta"] for r in rr],
            "^--", ms=4, lw=1.4, color=C_PURPLE, zorder=5,
            label="PCA, |t|-ordered — WORSE at every r (keeping what fits "
                  "in-window hurts)")
    edf = pca["ridge_edf"]
    lam = [0.0, 12.5, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
    ax.plot([edf[f"ridgeD:L{l:g}"] for l in lam],
            [A[f"ridgeD:L{l:g}"]["delta"] for l in lam],
            "d-", ms=5, lw=2.4, color=C_GOOD, zorder=6,
            label="RIDGE on the SAME 15 terms — reaches -0.00009 ns")
    for nm, col, mk in (("A:PCA90", C_MODEL, "*"), ("C:EB_ALL15", C_MKT, "*")):
        a = A[nm]
        ax.errorbar([a["ncol"]], [a["delta"]],
                    yerr=[[a["delta"] - a["cl_lo"]], [a["cl_hi"] - a["delta"]]],
                    fmt=mk, ms=17, color=col, ecolor=col, elinewidth=1.3,
                    capsize=3, zorder=8)
    ax.text(12.55, A["A:PCA90"]["delta"], "A", fontsize=13,
            color=C_MODEL, ha="right", va="center", weight="bold")
    ax.text(15.45, A["C:EB_ALL15"]["delta"], "C", fontsize=13,
            color=C_MKT, ha="left", va="center", weight="bold")
    ax.text(15.4, -0.0075, "ARM C = per-term empirical-Bayes shrinkage:\n"
            "-0.00634, paired -0.00063 vs raw — WORSE, hypothesis REFUTED",
            fontsize=8.4, color=C_MKT, ha="right", va="top")
    ax.axhline(0, color=INK2, lw=0.9)
    ax.set_xlabel("coefficients carried   (PCA: r components + the 2 dead terms;"
                  "  ridge: effective degrees of freedom)")
    ax.set_ylabel("pooled log-loss delta vs the D132 control")
    ax.set_title("C. Does compression beat carrying the raw pile?  Barely.  "
                 "Better shrinkage does.\nn = 6,148 certified games; "
                 "whiskers = season-clustered 95% CI", fontsize=10.5, loc="left")
    ax.set_xlim(0, 16.6)
    ax.set_ylim(-0.0084, 0.0031)
    ax.legend(frameon=False, fontsize=8.0, loc="upper right",
              handlelength=2.4, borderaxespad=0.2, labelspacing=0.35)
    despine(ax)

    # ================= PANEL D — the expensive case =====================
    ax = fig.add_subplot(gs[1, 1])
    xs = np.arange(0, 46.5, 0.5)
    ax.plot(xs, -1.0e-4 * xs, color=INK2, lw=1.6, ls="--", zorder=2,
            label="pure-noise benchmark, -1.0e-4 per coefficient")
    ax.plot([45], [ca["arms"]["noise:k45"]["delta"]], "s", ms=6, color=INK2,
            alpha=0.55, zorder=3, label="45 pure-noise columns: -0.00361")
    rb = [5, 10, 20, 30, 35, 40, 43]
    ax.plot([r + 2 for r in rb], [A[f"pcaTH:r{r}"]["delta"] for r in rb],
            "o-", ms=5, lw=2.2, color=C_MODEL, zorder=5,
            label="PCA on pile + 30 team-home dummies")
    a = A["raw:ALL15+TEAMHOME"]
    ax.errorbar([45], [a["delta"]],
                yerr=[[a["delta"] - a["cl_lo"]], [a["cl_hi"] - a["delta"]]],
                fmt="o", ms=7, color=C_BAD, ecolor=C_BAD, elinewidth=1.4,
                capsize=3, zorder=7,
                label="carry EVERYTHING raw: -0.01803 = 161% of the gap")
    b = A["B:PCA90+TH"]
    ax.errorbar([b["ncol"]], [b["delta"]],
                yerr=[[b["delta"] - b["cl_lo"]], [b["cl_hi"] - b["delta"]]],
                fmt="*", ms=18, color=C_MODEL, ecolor=C_MODEL, elinewidth=1.3,
                capsize=3, zorder=8,
                label="ARM B  PCA @ 90% var (r=35 +2): -0.01639, "
                      "paired +0.00164 ns")
    # the 19x penalty swing on the SAME 30-dummy block (D154 §13)
    ax.plot([30, 30], [-0.012525, -0.000655], color=C_GOOD, lw=1.3, ls=":",
            zorder=6)
    ax.plot([30], [-0.012525], "x", ms=9, mew=2, color=C_BAD, zorder=8)
    ax.plot([30], [-0.000655], "P", ms=11, color=C_GOOD, zorder=8,
            label="the SAME 30-dummy block at RIDGE 200: -0.00066 ns")
    ax.annotate("", xy=(30, -0.00090), xytext=(30, -0.01220),
                arrowprops=dict(arrowstyle="<->", color=C_GOOD, lw=1.5))
    ax.text(28.6, -0.0068, "19x on the\nPENALTY alone\n(D154 §13)",
            fontsize=8.6, color=C_GOOD, ha="right", va="center")
    ax.axhline(0, color=INK2, lw=0.9)
    ax.set_xlabel("coefficients carried")
    ax.set_ylabel("pooled log-loss delta vs the D132 control")
    ax.set_title("D. The expensive case: 30 team-home dummies.\n"
                 "Nothing to compress — the fix is the PENALTY, not the basis",
                 fontsize=10.5, loc="left")
    ax.set_xlim(0, 47.5)
    ax.set_ylim(-0.0215, 0.0055)
    ax.legend(frameon=False, fontsize=8.0, loc="lower left",
              handlelength=2.2, borderaxespad=0.2, labelspacing=0.35)
    despine(ax)

    fig.suptitle("PCA on the rejected feature pile — spectrum and cost   "
                 "(prereg 0c3720ba…, DB read-only, no default changed)",
                 fontsize=12.5, color=INK, x=0.012, ha="left", y=0.985)
    out = REPO / "charts" / "pca_spectrum_and_cost.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
