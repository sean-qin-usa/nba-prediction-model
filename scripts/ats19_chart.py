#!/usr/bin/env python3
"""ATS19 CHART — charts/ats19_open.png

LEFT   cover rate vs the -110 breakeven (52.381%) for every one of the 19
       seasons, with the DEVELOPMENT era (2021-22..2025-26) demarcated and the
       out-of-sample block (2007-08..2020-21) called out.
RIGHT  TOP    pooled ROI by window with the K-1 season-cluster-mean t interval
              (the shipping statistic), OOS14 highlighted.
       BOTTOM ROI by the pre-declared |edge| threshold, POOL19 vs OOS14, same
              interval.

House style per scripts/make_status_charts.py: model blue #2a78d6 / market
orange #eb6834, thin marks, recessive grid, direct labels, no dual axes,
150 dpi.
"""
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
C_MODEL, C_MKT = "#2a78d6", "#eb6834"
C_GOOD, C_BAD = "#1baf7a", "#d1495b"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e7e6e2"
DEVBG = "#f3f6fb"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlecolor": INK,
})
BE = 100.0 / (1.0 + 100.0 / 110.0)          # 52.3810
BB = dict(facecolor="white", edgecolor="none", alpha=0.93,
          boxstyle="square,pad=0.13")


def main():
    r = json.load(open(ROOT / "data" /
                       f"ats19{os.environ.get('ATS19_TAG', '')}.json"))
    per = r["per_season_allgames"]
    seasons = [p["season"] for p in per]
    cover = [100 * p["hit"] for p in per]
    dev = set(r["constants"].get("dev5", [])) or {
        "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"}

    fig = plt.figure(figsize=(17.2, 9.4), dpi=150)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.06, 1.0],
                          height_ratios=[1.0, 1.0], wspace=0.20, hspace=0.40)

    # ==================================================== LEFT: by season ====
    ax = fig.add_subplot(gs[:, 0])
    n = len(seasons)
    ypos = np.arange(n)
    # DEV band behind everything
    dev_idx = [i for i, s in enumerate(seasons) if s in dev]
    ax.axhspan(min(dev_idx) - 0.5, max(dev_idx) + 0.5, color=DEVBG, zorder=0)
    cols = [C_GOOD if c > BE else (C_MODEL if c > 50.0 else C_BAD)
            for c in cover]
    ax.barh(ypos, [c - 50.0 for c in cover], left=50.0, color=cols,
            height=0.58, zorder=3)
    ax.axvline(50.0, color=INK2, lw=1.0, zorder=4)
    ax.axvline(BE, color=C_MKT, ls="--", lw=1.6, zorder=5)
    pooled_hit = 100 * r["allgames_windows"]["POOL19"]["hit"]
    ax.axvline(pooled_hit, color=INK2, ls=":", lw=1.4, zorder=4)
    lo = min(min(cover), 47.5) - 1.2
    hi = max(max(cover), BE) + 2.9          # right gutter for the band labels
    ax.set_xlim(lo, hi)
    ax.set_ylim(n - 0.45, -2.35)            # headroom INSIDE the axes for the
    #                                         reference labels (pass 1: they
    #                                         were drawn above the frame and
    #                                         collided with the panel title)
    for i, c in enumerate(cover):
        off = 0.16 if c >= 50 else -0.16
        ax.text(c + off, i, f"{c:.2f}", va="center",
                ha="left" if c >= 50 else "right", fontsize=8.5, color=INK2,
                zorder=6, bbox=BB)
    ax.set_yticks(ypos, seasons, fontsize=9.0)
    ax.set_xlabel("cover rate % against the OPENING spread (pushes excluded)"
                  "   —   green clears -110, blue beats 50%, red below 50%")
    ax.set_title("COVER RATE BY SEASON — our expected margin vs the opening "
                 "spread, every game", fontsize=12.0, pad=8, loc="left")
    # reference labels parked in the headroom rows, on opaque bboxes
    ax.text(BE + 0.07, -1.95, f"-110 breakeven {BE:.3f}%", ha="left",
            va="center", fontsize=9.0, color=C_MKT, zorder=7, bbox=BB)
    ax.text(50.0 - 0.07, -1.95, "50.000% coin flip", ha="right", va="center",
            fontsize=9.0, color=INK2, zorder=7, bbox=BB)
    ax.text(pooled_hit + 0.07, -1.15, f"pooled 19 seasons {pooled_hit:.3f}%",
            ha="left", va="center", fontsize=9.0, color=INK2, zorder=7,
            bbox=BB)
    # band labels live in the RIGHT GUTTER, which no bar or value label reaches
    ax.text(hi - 0.18, min(dev_idx) + 0.15, "DEVELOPMENT ERA  2021-26",
            ha="right", va="top", fontsize=8.6, color="#7d8aa0", zorder=6)
    ax.text(hi - 0.18, 0.15, "OUT OF SAMPLE\n2007-08 .. 2020-21\n14 seasons",
            ha="right", va="top", fontsize=8.6, color="#8d8b86", zorder=6,
            linespacing=1.35)
    ax.grid(axis="y", visible=False)

    # ============================================ RIGHT TOP: pooled ROI ======
    ax2 = fig.add_subplot(gs[0, 1])
    AW = r["allgames_windows"]
    order = [("POOL19", "POOL 19 seasons\nn=%s" % f"{AW['POOL19']['n']:,}"),
             ("OOS14", "OOS 2007-08\n..2020-21\n14 sns, n=%s"
              % f"{AW['OOS14']['n']:,}"),
             ("DEV5", "DEV\n2021-26\n5 sns, n=%s" % f"{AW['DEV5']['n']:,}"),
             ("NOCOVID17", "ex-COVID\n17 seasons\nn=%s"
              % f"{AW['NOCOVID17']['n']:,}")]
    # D173: the owner's PRIMARY frame — the report era, when injury reports
    # exist (they begin 2018-12-17).  Rendered only if the run produced it.
    if "REPORT8" in AW:
        order.insert(1, ("REPORT8", "PRIMARY\nREPORT ERA, T2\n8 sns, n=%s"
                         % f"{AW['REPORT8']['n']:,}"))
    xs = np.arange(len(order))
    est = [100 * r["allgames_windows"][k]["roi"]["mean"] for k, _ in order]
    tlo = [100 * r["allgames_windows"][k]["roi"]["tlo"] for k, _ in order]
    thi = [100 * r["allgames_windows"][k]["roi"]["thi"] for k, _ in order]
    for i, (k, _) in enumerate(order):
        c = C_BAD if thi[i] < 0 else ("#a9a7a2" if tlo[i] < 0 else C_GOOD)
        lw = 3.4 if k == "REPORT8" else 3.0 if k == "OOS14" else 2.0
        ax2.plot([i, i], [tlo[i], thi[i]], color=c, lw=lw, solid_capstyle="butt",
                 zorder=3)
        ax2.plot([i], [est[i]], "o",
                 ms=10 if k == "REPORT8" else 9 if k == "OOS14" else 7,
                 color=c, zorder=4)
        ax2.text(i + 0.16, est[i], f"{est[i]:+.2f}%", ha="left", va="center",
                 fontsize=9.4, color=INK, zorder=6, bbox=BB)
        ax2.text(i + 0.16, thi[i], f"{thi[i]:+.2f}", ha="left", va="center",
                 fontsize=8.0, color=INK2, zorder=6)
        ax2.text(i + 0.16, tlo[i], f"{tlo[i]:+.2f}", ha="left", va="center",
                 fontsize=8.0, color=INK2, zorder=6)
    ax2.axhline(0.0, color=C_MKT, ls="--", lw=1.5, zorder=2)
    ax2.text(len(order) - 0.42, 0.0, "breakeven at -110", ha="right",
             va="bottom", fontsize=9.0, color=C_MKT, zorder=6, bbox=BB)
    ax2.set_xlim(-0.55, len(order) - 0.10)
    ax2.set_xticks(xs, [t for _, t in order], fontsize=8.6)
    ax2.set_ylabel("ROI % per unit staked")
    _o = AW["OOS14"]["roi"]
    _verd = ("SIGNIFICANTLY NEGATIVE" if _o["thi"] < 0 else
             "SIGNIFICANTLY POSITIVE" if _o["tlo"] > 0 else
             "NOT SIGNIFICANT (D162's rejection is GONE)")
    ax2.set_title("POOLED ROI, K−1 SEASON-CLUSTER-MEAN t INTERVAL\n"
                  f"the out-of-sample block is the real evidence —\n"
                  f"and it is {_verd}", fontsize=11.0, pad=9, loc="left")
    ax2.grid(axis="x", visible=False)
    _oi = [k for k, _ in order].index("OOS14")
    ax2.annotate("OUT OF SAMPLE", xy=(_oi, thi[_oi]),
                 xytext=(_oi, thi[_oi] + 1.55),
                 ha="center", fontsize=9.2, color=C_BAD, fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color=C_BAD, lw=1.2))
    ax2.set_ylim(min(tlo) - 1.3, max(max(thi), 0) + 2.6)

    # ======================================= RIGHT BOTTOM: thresholds ========
    ax3 = fig.add_subplot(gs[1, 1])
    T = ["0.0", "1.0", "2.0", "3.0"]
    w = 0.34
    for j, (win, col, lab) in enumerate(
            (("POOL19", C_MODEL, "POOL 19 seasons"),
             ("OOS14", C_BAD, "OOS 14 seasons (2007-08..2020-21)"))):
        xx = np.arange(len(T)) + (j - 0.5) * w
        e = [100 * r["thresholds"][win][t]["roi"]["mean"] for t in T]
        lo_ = [100 * r["thresholds"][win][t]["roi"]["tlo"] for t in T]
        hi_ = [100 * r["thresholds"][win][t]["roi"]["thi"] for t in T]
        ax3.errorbar(xx, e, yerr=[np.array(e) - np.array(lo_),
                                  np.array(hi_) - np.array(e)],
                     fmt="o", ms=6.5, lw=1.9, capsize=4, color=col,
                     label=lab, zorder=3)
        for k, t in enumerate(T):
            ax3.text(xx[k], hi_[k] + 0.30, f"{e[k]:+.2f}", ha="center",
                     va="bottom", fontsize=8.4, color=col, zorder=6)
    ax3.axhline(0.0, color=C_MKT, ls="--", lw=1.5, zorder=2)
    ax3.set_xticks(np.arange(len(T)),
                   [f"|edge| ≥ {t} pt\nn={r['thresholds']['POOL19'][t]['n']:,}"
                    for t in T], fontsize=8.8)
    ax3.set_ylabel("ROI % per unit staked")
    ax3.set_xlabel("pre-declared, untuned threshold on "
                   "(model margin − opening spread)", labelpad=6)
    ax3.set_title("ROI BY THE PRE-DECLARED THRESHOLD — no cell clears -110 on "
                  "either block", fontsize=11.4, pad=10, loc="left")
    ax3.legend(loc="upper left", frameon=False, fontsize=8.8, ncol=1)
    ax3.grid(axis="x", visible=False)
    ax3.set_xlim(-0.55, len(T) - 0.45)

    clv = r["clv_points"]["by_threshold"]["0.0"]
    hit19 = 100 * r["allgames_windows"]["POOL19"]["hit"]
    fig.text(0.006, 0.977,
             "ATS AT THE OPEN, K = 19 SEASONS — the first open-price test in "
             "this project with real statistical power",
             fontsize=15.0, ha="left", va="top", color=INK)
    # pass 2: this line overran into the right panel's title — shortened and
    # confined to the left half of the figure.
    fig.text(0.006, 0.947,
             f"BEATS the spread: {hit19:.3f}% vs 50.000%.   "
             f"DOES NOT CLEAR -110: {hit19:.3f}% vs 52.381%, ROI "
             f"{100*AW['POOL19']['roi']['mean']:+.2f}% "
             f"[{100*AW['POOL19']['roi']['tlo']:+.2f},"
             f"{100*AW['POOL19']['roi']['thi']:+.2f}] at 18 dof "
             f"({'SIG NEG' if AW['POOL19']['roi']['thi'] < 0 else 'ns'}).",
             fontsize=10.8, ha="left", va="top", color=C_BAD)
    fig.text(0.006, 0.036,
             f"D173 RE-RUN on the D170/D171 backfilled data.  Availability tier "
             f"is BEST-AVAILABLE and LABELLED, never silently pooled: T2i "
             f"(official inactives) 2007-08..2017-18, full T2 (5PM injury "
             f"report UNION inactives) 2018-19..2025-26.  Pushes "
             f"{r['push']['n_push_open']}/{r['n_frame']:,} = "
             f"{r['push']['pct_open']:.3f}%: excluded from the cover rate, "
             f"included in the ROI denominator.",
             fontsize=8.6, color=INK2, ha="left", va="bottom")
    fig.text(0.006, 0.014,
             f"Pre-registered before scoring: data/ats19_prereg.md sha256 "
             f"{r['prereg_sha256'][:24]}…   |   spread-point CLV "
             f"+{clv['mean']:.4f} pts/bet [{clv['tlo']:+.4f},"
             f"{clv['thi']:+.4f}] at 18 dof — SIG POSITIVE, and worth far "
             f"less than the vig.",
             fontsize=8.6, color=INK2, ha="left", va="bottom")
    fig.subplots_adjust(left=0.052, right=0.985, top=0.868, bottom=0.118)
    out = ROOT / "charts" / "ats19_open.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
