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
    # D171: the MODEL panel is rebuilt on the re-scored 19-season frame (each
    # season at the best availability tier it can reach). K19_STATS lets the
    # D161 blind vintage still be rendered by pointing at the old file.
    import os as _os
    ms = json.load(open(ROOT / "data" /
                        _os.environ.get("K19_STATS", "k19_model_stats.json")))
    rs = json.load(open(ROOT / "data" /
                        f"k19_rules{os.environ.get('K19_RULES_TAG', '')}.json"))
    per = ms["per_season"]
    pooled = ms["blocks"][0] if len(ms["blocks"]) == 1 else \
        [b for b in ms["blocks"] if b["label"].startswith("ALL 19")][0]
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
    # D171: 2008-09 is NEGATIVE at T2 (the model beats the market); a left
    # limit of 0 hid the bar entirely.
    xmin = min(0.0, min(gaps) * 1.30)
    ax.set_xlim(xmin, xmax)
    if min(gaps) < 0:
        ax.axvline(0.0, color=C_GOOD, lw=1.3, zorder=5)
    _bb = dict(facecolor="white", edgecolor="none", alpha=0.92,
               boxstyle="square,pad=0.14")
    for i, (g, c) in enumerate(zip(gaps, cov)):
        # D171 label-collision pass: a NEGATIVE bar's label placed to the left
        # of its own end runs straight into the season tick label. Park it in
        # the empty space to the RIGHT of zero instead.
        ax.text(g + 0.45 if g >= 0 else 0.45, i, f"{g:.1f}%", va="center",
                ha="left", fontsize=8.6,
                color=C_GOOD if g < 0 else INK2, zorder=6, bbox=_bb,
                fontweight="bold" if g < 0 else "normal")
        ax.text(xmax - 0.6, i, f"DARKO {c:.0f}%", va="center", ha="right",
                fontsize=7.6, color="#8d8b86", zorder=6)
    # reference labels: parked in row gaps, on opaque bboxes, so they never
    # ride on a bar value label (checked by rendering)
    ax.text(pooled["norm_gap_pct"] + 0.5, -1.75,
            f"pooled (tier labelled per season) {pooled['norm_gap_pct']:.2f}%", ha="left",
            va="center", fontsize=8.8, color=INK2, zorder=7, bbox=_bb)
    ax.text(t2 + 0.5, -0.95, f"CERTIFIED 5-season (D171) {t2:.2f}%",
            ha="left", va="center", fontsize=8.8, color=C_MKT, zorder=7,
            bbox=_bb)
    ax.set_yticks(range(len(labs)), labs, fontsize=8.8)
    ax.set_ylim(19.4, -2.4)
    ax.set_xlabel("% of the market's skill-above-coinflip that we MISS "
                  "(lower = closer)")
    ax.set_title("MODEL — 19 seasons, each at the BEST availability tier it can "
                 "reach (D171)\nT2 from 2018-19, T2i before; 18/19 positive — "
                 "2008-09 now BEATS the market",
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
            # D173 (D171's charting bug class): whiskers may be CLAMPED and are
            # flagged when they are, but a POINT ESTIMATE must never be clipped.
            assert YR[0] < pt < YR[1], f"ROI point {pt:.2f} outside ylim {YR}"
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
    _fw = list(rs["familywise"].values())[0]
    axR.set_title("RULES — ROI by era, REAL closing moneylines, 19 seasons "
                  "(D173, backfilled data)\n"
                  f"{_fw['observed_pos']} of {_fw['cells']} pre-specified ROI "
                  f"cells SIG positive against {_fw['expected']:.2f} expected;  "
                  f"{_fw['observed_neg']} SIG negative "
                  f"(D161 on the old data: 0 / 92 / 23)",
                  fontsize=10.2, pad=40)
    # dedupe: STAR_FAV was emitted twice by the i==0 / K-E label rule
    _h, _l = axR.get_legend_handles_labels()
    _seen, _hh, _ll = set(), [], []
    for _hi, _li in zip(_h, _l):
        if _li in _seen:
            continue
        _seen.add(_li)
        _hh.append(_hi)
        _ll.append(_li)
    axR.legend(_hh, _ll, frameon=False, fontsize=8.2, ncol=5,
               loc="lower center", bbox_to_anchor=(0.5, 1.004),
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
            assert YC[0] < pt < YC[1], f"CLV point {pt:.5f} outside ylim {YC}"
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

    fig.suptitle("K19 — MODEL (D171) AND THE FOUR FROZEN RULES (D173), BOTH "
                 "RE-SCORED ON THE BACKFILLED DATA, 19 SEASONS (2007-08..2025-26)",
                 fontsize=13, y=0.992)
    fig.text(0.5, 0.010,
             "MODEL panel (left): D171 — every season at the best tier it can reach; "
             "T2 (5PM report UNION official inactives) from 2018-19, T2i (inactives only) "
             "before, LABELLED per season, never silently pooled.\n"
             "No played-set oracle.  TANK_SEASON_FLOOR=2020-21 pinned.  "
             "RULES panels (right): D173 — RE-RUN on the same backfilled data "
             "(scripts/k19_rules.py, K19_PERGAME=k19_d171_t2_pergame.csv). "
             "Nothing re-selected: the four frozen rules, thresholds and price "
             "conventions are carried verbatim.\n"
             "Intervals are K-1 season-cluster-mean t (GATE_POLICY_V2 §9.1(4)).  "
             "Sources: data/k19_d171_t2_pergame.csv (22,804 games), "
             "data/d171_k19_model_stats.json, data/k19_rules_D172.json.",
             ha="center", va="bottom", fontsize=7.6, color="#8d8b86")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.845,
                        bottom=0.135)
    out = ROOT / "charts" / "k19_model_and_rules.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
