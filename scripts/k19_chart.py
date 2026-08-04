#!/usr/bin/env python3
"""K19 CHART — charts/k19_model_and_rules.png

LEFT   normalized model gap by season across all 19, with the pooled blind
       level and the CERTIFIED FULL-FEED estimate (D158, 11.45%) marked.
RIGHT  the frozen rules by era — ROI vs breakeven (top) and CLV (bottom) —
       with K-1 season-cluster-mean t intervals, so "do the rules survive 19
       seasons" is answerable at a glance.

House style: model blue #2a78d6 / market orange #eb6834, thin marks, recessive
grid, negative bar labels INSIDE bars, no dual axes, 150 dpi.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
ERA_ORDER = ["K-A", "K-B", "K-C", "K-D", "K-E"]
ERA_X = {"K-A": "K-A\n2008-11\n4 seasons", "K-B": "K-B\n2012-14\n3 seasons",
         "K-C": "K-C\n2015-19\n5 seasons", "K-D": "K-D\nCOVID\n2 seasons",
         "K-E": "K-E\n2022-26\n5 seasons"}
SETS = [("R4_LOWT", "R4_LOWT"), ("T20_D03_10_W", "T20_D03_10_W"),
        ("T20_D03_10", "T20_D03_10"), ("STAR_FAV_SHARPER", "STAR_FAV"),
        ("UNION", "UNION")]
SETCOL = {"R4_LOWT": "#7fb2e8", "T20_D03_10_W": "#a9a7a2",
          "T20_D03_10": C_MKT, "STAR_FAV_SHARPER": "#c9a227",
          "UNION": C_MODEL}


def main():
    ms = json.load(open(ROOT / "data" / "k19_model_stats.json"))
    rs = json.load(open(ROOT / "data" / "k19_rules.json"))
    per = ms["per_season"]
    pooled = [b for b in ms["blocks"] if b["label"] == "ALL 19 (blind)"][0]
    t2 = ms["tier_cost"]["t2_pooled_3"]

    fig = plt.figure(figsize=(17.6, 9.8), dpi=150)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.28],
                          height_ratios=[1, 1], wspace=0.19, hspace=0.46)

    # ================================================= LEFT: model by season
    ax = fig.add_subplot(gs[:, 0])
    labs = [r["season"] for r in per]
    gaps = [r["norm_gap_pct"] for r in per]
    cov = [100 * r["darko_cov"] for r in per]
    ax.barh(range(len(gaps)), gaps, color=C_MODEL, height=0.58, zorder=3)
    ax.axvline(pooled["norm_gap_pct"], color=INK2, ls="--", lw=1.2, zorder=4)
    ax.axvline(t2, color=C_MKT, ls="-.", lw=1.5, zorder=4)
    xmax = max(gaps) * 1.44
    ax.set_xlim(0, xmax)
    _bb = dict(facecolor="white", edgecolor="none", alpha=0.92,
               boxstyle="square,pad=0.14")
    for i, (g, c) in enumerate(zip(gaps, cov)):
        ax.text(g + 0.45, i, f"{g:.1f}%", va="center", ha="left", fontsize=8.6,
                color=INK2, zorder=6, bbox=_bb)
        ax.text(xmax - 0.6, i, f"DARKO {c:.0f}%", va="center", ha="right",
                fontsize=7.6, color="#8d8b86", zorder=6)
    # reference labels: parked in row gaps, on opaque bboxes, so they never
    # ride on a bar value label (checked by rendering)
    ax.text(pooled["norm_gap_pct"] + 0.5, -1.75,
            f"pooled BLIND {pooled['norm_gap_pct']:.2f}%", ha="left",
            va="center", fontsize=8.8, color=INK2, zorder=7, bbox=_bb)
    ax.text(t2 + 0.5, -0.95, f"certified FULL-FEED (D158) {t2:.2f}%",
            ha="left", va="center", fontsize=8.8, color=C_MKT, zorder=7,
            bbox=_bb)
    ax.set_yticks(range(len(labs)), labs, fontsize=8.8)
    ax.set_ylim(19.4, -2.4)
    ax.set_xlabel("% of the market's skill-above-coinflip that we MISS "
                  "(lower = closer)")
    ax.set_title("MODEL — 19 seasons, ONE constant availability tier (BLIND)\n"
                 "19/19 seasons positive: the market beats us in every one",
                 fontsize=10.5, pad=10)
    ax.text(0.985, 0.012,
            f"pooled raw gap {pooled['raw_gap']:+.5f}   n={pooled['n']:,}\n"
            f"K=19 season-cluster-mean t "
            f"[{pooled['t_lo']:+.5f}, {pooled['t_hi']:+.5f}]  SIG",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.4,
            color=INK2, zorder=8,
            bbox=dict(facecolor="white", edgecolor=GRID, alpha=0.96,
                      boxstyle="round,pad=0.34"))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ============================================ RIGHT TOP: ROI by era =====
    axR = fig.add_subplot(gs[0, 1])
    era = rs["era"]["CLOSE|ML|19"]
    xs = np.arange(len(ERA_ORDER))
    YR = (-30.0, 24.0)          # K-D (2 seasons, 1 dof) has a +-165% CI; it is
    off_top = []                # CLIPPED, and every clipped whisker is flagged
    for j, (k, nm) in enumerate(SETS):
        off = (j - 2) * 0.155
        for i, e in enumerate(ERA_ORDER):
            c = era.get(k, {}).get(e, {})
            if "roi" not in c or not np.isfinite(c["roi"].get("tlo", np.nan)):
                continue
            pt = 100 * c["roi"]["mean"]
            lo = max(YR[0], 100 * c["roi"]["tlo"])
            hi = min(YR[1], 100 * c["roi"]["thi"])
            if (100 * c["roi"]["tlo"] < YR[0]) or (100 * c["roi"]["thi"] > YR[1]):
                off_top.append(e)
            axR.errorbar([i + off], [pt], yerr=[[pt - lo], [hi - pt]],
                         fmt="o", ms=5.4, color=SETCOL[k], lw=0,
                         elinewidth=1.3, capsize=2.4, zorder=4,
                         label=nm if i == 0 or (k == "STAR_FAV_SHARPER"
                                                and e == "K-E") else None,
                         alpha=1.0 if k == "UNION" else 0.85)
    axR.axhline(0, color=INK, lw=1.5, zorder=3)
    axR.set_ylim(*YR)
    axR.text(-0.47, 1.1, "BREAKEVEN — above this line the bet clears the vig",
             fontsize=8.4, color=INK, ha="left", va="bottom", zorder=6,
             bbox=_bb)
    axR.text(0.995, 0.015, "K-D whiskers clipped (K=2 -> 1 dof)",
             transform=axR.transAxes, fontsize=7.6, color="#8d8b86",
             ha="right", va="bottom", zorder=7, bbox=_bb)
    axR.set_xticks(xs, [ERA_X[e] for e in ERA_ORDER], fontsize=8.4)
    axR.set_ylabel("ROI per bet, %  (real closing moneylines)")
    axR.set_title("RULES — ROI by era, REAL closing moneylines, 19 seasons\n"
                  "every era below breakeven  |  0 of 92 cells SIG positive, "
                  "23 SIG negative",
                  fontsize=10.5, pad=26)
    axR.legend(frameon=False, fontsize=8.2, ncol=5,
               loc="lower center", bbox_to_anchor=(0.5, 1.005),
               columnspacing=1.4, handletextpad=0.3)
    for sp in ("top", "right"):
        axR.spines[sp].set_visible(False)

    # ============================================ RIGHT BOTTOM: CLV by era ==
    axC = fig.add_subplot(gs[1, 1])
    eraC = rs["era"]["OPEN|SP|19"]
    YC = (-0.026, 0.050)
    for j, (k, nm) in enumerate(SETS):
        off = (j - 2) * 0.155
        for i, e in enumerate(ERA_ORDER):
            c = eraC.get(k, {}).get(e, {})
            if "clv" not in c or not np.isfinite(c["clv"].get("tlo", np.nan)):
                continue
            pt = c["clv"]["mean"]
            lo = max(YC[0], c["clv"]["tlo"])
            hi = min(YC[1], c["clv"]["thi"])
            axC.errorbar([i + off], [pt], yerr=[[pt - lo], [hi - pt]],
                         fmt="o", ms=5.4, color=SETCOL[k], lw=0,
                         elinewidth=1.3, capsize=2.4, zorder=4,
                         alpha=1.0 if k == "UNION" else 0.85)
    axC.axhline(0, color=INK, lw=1.5, zorder=3)
    u = rs["table"]["OPEN|SP|19"]["POOL19"]["UNION"]["clv"]
    axC.axhline(u["mean"], color=C_MODEL, ls="--", lw=1.2, zorder=3)
    axC.set_ylim(*YC)
    axC.set_xlim(-0.55, 4.55)
    axC.text(-0.47, 0.0012, "ZERO — no line movement toward our side",
             fontsize=8.4, color=INK, ha="left", va="bottom", zorder=6,
             bbox=_bb)
    axC.text(0.5, 0.975,
             f"UNION pooled {u['mean']:+.5f}   K=19 cluster-mean t "
             f"[{u['tlo']:+.5f}, {u['thi']:+.5f}]  SIG",
             transform=axC.transAxes, fontsize=8.4, color=C_MODEL,
             ha="center", va="top", zorder=8,
             bbox=dict(facecolor="white", edgecolor=GRID, alpha=0.97,
                       boxstyle="round,pad=0.3"))
    axC.text(0.995, 0.015, "K-D whiskers clipped (K=2 -> 1 dof)",
             transform=axC.transAxes, fontsize=7.6, color="#8d8b86",
             ha="right", va="bottom", zorder=7, bbox=_bb)
    axC.set_xticks(xs, [ERA_X[e] for e in ERA_ORDER], fontsize=8.4)
    axC.set_ylabel("CLV per bet  (p_close - p_open, our side)")
    axC.set_title("RULES — CLV by era, bets fired at the OPEN (SP convention, "
                  "19 seasons)\npositive in ALL 5 eras, I2=0% ERA-STABLE, "
                  "within-date placebo p=0.000  |  THE ASSET IS CLV, NOT ROI",
                  fontsize=10.5, pad=10)
    for sp in ("top", "right"):
        axC.spines[sp].set_visible(False)

    fig.suptitle("K19 — MODEL AND THE FOUR FROZEN RULES ACROSS 19 CONTIGUOUS "
                 "SEASONS (2007-08..2025-26), AVAILABILITY-BLIND = LOWER BOUND",
                 fontsize=13, y=0.985)
    fig.text(0.5, 0.010,
             "Availability-BLIND on EVERY season (empty OUT sets): injury "
             "reports start 2023-10, inactives 2022-23 — one constant, honest, "
             "deliberately WEAK tier, so every level here is a LOWER BOUND.  "
             "No played-set oracle.  TANK_SEASON_FLOOR=2020-21 pinned.\n"
             "Intervals are K-1 season-cluster-mean t (GATE_POLICY_V2 "
             "§9.1(4)).   Sources: data/k19_pergame.csv (22,804 games), "
             "data/k19_model_stats.json, data/k19_rules.json.",
             ha="center", va="bottom", fontsize=8.0, color="#8d8b86")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.905,
                        bottom=0.115)
    out = ROOT / "charts" / "k19_model_and_rules.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
