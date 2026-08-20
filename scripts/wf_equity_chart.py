#!/usr/bin/env python3
"""charts/walkforward_equity.png — the equity path the owner asked for.

House style per scripts/make_status_charts.py: model blue / market orange, thin
marks, recessive grid, no dual axes, negative bar labels inside the bars, 150 dpi.
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

TIER_C = {"k=1 raw": "#a9bdd0", "k=2 raw": "#7aa5d2", "k=5 raw": C_MODEL,
          "k=8 raw": "#12406f"}

# D173: the owner's PRIMARY frame.  Injury reports begin 2018-12-17, so the
# report era is 2018-19..2025-26.  Scored track starts 2012-13, so 8 of the
# 14 scored seasons are report-era.
REPORT_ERA0 = "2018-19"
TQ = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
      8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
      14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101}


def frame_stats(rows, idx):
    """Pooled ROI and the K-1 dof cluster-mean t interval on a season subset."""
    n = sum(rows[i]["n"] for i in idx)
    pay = sum(rows[i]["pay"] for i in idx)
    per = np.array([rows[i]["roi"] for i in idx], float)
    K = len(per)
    sd = per.std(ddof=1) if K > 1 else np.nan
    se = sd / np.sqrt(K) if K > 1 else np.nan
    h = TQ.get(K - 1, 2.101) * se if K > 1 else np.nan
    return {"n": n, "pay": pay, "roi": pay / n if n else np.nan, "K": K,
            "per": per, "mean": float(per.mean()), "sd": float(sd),
            "lo": float(per.mean() - h), "hi": float(per.mean() + h),
            "sig": bool((per.mean() - h) * (per.mean() + h) > 0)}


def main():
    R = json.load(open(ROOT / "data" /
                        f"wf_equity{os.environ.get('WF_TAG', '')}.json"))
    # the spread-point CLV of the frame this equity path is priced on, so the
    # footer can never quote a superseded vintage (it said +0.166 after the
    # D173 re-run made it +0.320)
    _a = json.load(open(ROOT / "data" /
                        f"ats19{os.environ.get('ATS19_TAG', '_D173')}.json"))
    _clv = _a["clv_points"]["by_window"]["POOL19"]["mean"]
    seas = R["scored_seasons"]
    x = np.arange(len(seas))
    T = R["tiers"]
    F = R["friction"]
    D = T["k=5 +haircut"]

    # report-era index set (the owner's PRIMARY frame)
    ridx = [i for i, sname in enumerate(seas) if sname >= REPORT_ERA0]
    aidx = list(range(len(seas)))

    fig = plt.figure(figsize=(16.2, 15.6), dpi=150)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.28, 1.0, 1.0],
                          hspace=0.80, wspace=0.26,
                          left=0.052, right=0.988, top=0.912, bottom=0.093)

    # ================================================================ PANEL A
    ax = fig.add_subplot(gs[0, :])
    ax.axhline(0, color=INK2, lw=1.1, ls="--", zorder=2)
    ax.text(len(seas) - 0.62, 0, "break-even (flat)", ha="right", va="bottom",
            fontsize=8.5, color=INK2)

    order = ["k=1 raw", "k=2 raw", "k=5 raw", "k=8 raw"]
    ends = []
    for nm in order:
        c = TIER_C[nm]
        cum = np.array(T[nm]["cum"])
        lw = 1.5
        ax.plot(x, cum, color=c, lw=lw, marker="o", ms=3.1, zorder=4)
        ends.append((cum[-1], nm.replace(" raw", "") + " raw", c, "-"))
        hn = nm.replace(" raw", " +haircut")
        if hn in T:
            cumh = np.array(T[hn]["cum"])
            is_def = (hn == "k=5 +haircut")
            ax.plot(x, cumh, color=c, lw=3.0 if is_def else 1.3, ls="--",
                    dashes=(4, 2.2), marker="o" if is_def else None, ms=4.2,
                    zorder=6 if is_def else 3)
            ends.append((cumh[-1], nm.replace(" raw", "") + " +haircut", c, "--"))

    exc = np.array(T["exchange c=2%"]["cum"])
    ax.plot(x, exc, color=C_MKT, lw=1.5, ls=":", zorder=4)
    ends.append((exc[-1], "exchange c=2%", C_MKT, ":"))
    fr = np.array(F["both"]["cum"])
    ax.plot(x, fr, color="#7a6a5f", lw=1.5, ls="-.", zorder=4)
    ends.append((fr[-1], "k=5 +haircut, limits+50% fill", "#7a6a5f", "-."))

    # direct end-labels, de-collided by vertical spreading
    ends.sort(key=lambda t: -t[0])
    ypos, MINGAP = [], 4.6
    for v, *_ in ends:
        y = v
        if ypos and y > ypos[-1] - MINGAP:
            y = ypos[-1] - MINGAP
        ypos.append(y)
    for (v, lab, c, ls), y in zip(ends, ypos):
        w = "bold" if lab == "k=5 +haircut" else "normal"
        ax.annotate(f"{lab}   {v:+.0f}u", xy=(len(seas) - 1, v),
                    xytext=(len(seas) - 0.72, y), color=c, fontsize=9,
                    va="center", ha="left", fontweight=w,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.7,
                                    shrinkA=0, shrinkB=1, alpha=0.6))
    ax.set_xlim(-0.55, len(seas) + 4.35)
    ax.set_xticks(x, seas, fontsize=9, rotation=35, ha="right")
    ax.set_ylabel("cumulative P&L, units (flat 1u/bet)")
    ax.set_title(
        "(a)  CUMULATIVE P&L OF THE WALK-FORWARD ALL-HISTORY STRATEGY AT THE OPENING SPREAD, BY EXECUTION TIER\n"
        "solid = raw measured shopping ladder   dashed = + D163's outlier-realism haircut   "
        "THICK DASHED = the firm default (k=5, measured, haircut)",
        fontsize=11.5, pad=11, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # measured / extrapolated era shading
    tags = [r["tag"] for r in D["rows"]]
    for i, t in enumerate(tags):
        if t == "EXTRAPOLATED":
            ax.axvspan(i - 0.5, i + 0.5, color="#f6f5f2", zorder=0)
    # D173: the report-era boundary (injury reports begin 2018-12-17)
    xb = ridx[0] - 0.5
    ax.axvline(xb, color="#8a6d3b", lw=1.15, ls=(0, (5, 2.5)), zorder=5)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + 6)          # headroom only; never clips a datum
    ax.text(xb + 0.12, -0.9, "injury reports begin 2018-12-17"
            "   ->   PRIMARY FRAME, panel (d)",
            fontsize=8.2, color="#8a6d3b", ha="left", va="top")
    ax.text(0.0, yhi + 2.4, "MEASURED  offshore panel, 9 books", fontsize=8.4,
            color=INK2, ha="left", va="center")
    ax.text(6.0, yhi + 2.4, "EXTRAPOLATED  no panel exists\noffshore ladder (conservative)",
            fontsize=8.4, color=INK2, ha="left", va="center")
    ax.text(11.05, yhi + 2.4, "MEASURED\nUS retail", fontsize=8.4, color=INK2,
            ha="center", va="center")
    ax.text(12.5, yhi + 2.4, "EXTRAP.\nUS retail", fontsize=8.4, color=INK2,
            ha="center", va="center")
    ax.text(0.0, -0.245,
            f"n = {D['n']:.0f} bets over 14 seasons (58-243 per season); 7 seasons priced on a MEASURED book panel, 7 EXTRAPOLATED (shaded).\n"
            f"FINAL UNITS:   1 book {T['k=1 raw']['cum'][-1]:+.1f}    |    FIRM DEFAULT k=5 measured+haircut {D['cum'][-1]:+.1f}    |    "
            f"k=8 raw, the optimistic bound {T['k=8 raw']['cum'][-1]:+.1f}\n"
            f"firm default after limits + 50% fill {F['both']['cum'][-1]:+.1f}    |    "
            f"exchange c=2%, ARITHMETIC ONLY — we hold zero exchange data {T['exchange c=2%']['cum'][-1]:+.1f}",
            transform=ax.transAxes, fontsize=8.7, color=INK2, va="top", linespacing=1.6)

    # ================================================================ PANEL B
    ax = fig.add_subplot(gs[1, :2])
    r1 = np.array([r["roi"] for r in T["k=1 raw"]["rows"]]) * 100
    r5 = np.array([r["roi"] for r in D["rows"]]) * 100
    FR8 = frame_stats(D["rows"], ridx)          # PRIMARY: report era, K=8
    FR14 = frame_stats(D["rows"], aidx)         # the full scored track, K=14
    w = 0.38
    ax.bar(x - w / 2, r1, w, color="#a9bdd0", zorder=3, label="1 book (retail ref)")
    ax.bar(x + w / 2, r5, w, color=C_MODEL, zorder=3,
           label="k=5 measured + haircut (firm default)")
    m5, sd5 = r5.mean(), r5.std(ddof=1)
    ax.axhspan(m5 - sd5, m5 + sd5, color=C_MODEL, alpha=0.09, zorder=1)
    ax.axhline(m5, color=C_MODEL, lw=1.3, zorder=4)
    ax.axhline(0, color=INK2, lw=1.0, zorder=4)
    # D173: the PRIMARY (report-era) frame, drawn on the same bars
    ax.axvspan(ridx[0] - 0.5, len(seas) - 0.35, color="#8a6d3b", alpha=0.055,
               zorder=0)
    ax.axvline(ridx[0] - 0.5, color="#8a6d3b", lw=1.15, ls=(0, (5, 2.5)),
               zorder=4)
    mR = 100.0 * FR8["mean"]      # per-season ROIs are fractions; r5 is already x100
    ax.axhline(mR, color="#8a6d3b", lw=1.3, ls=(0, (5, 2.5)), zorder=4,
               xmin=(ridx[0] - 0.5 + 0.7) / (len(seas) + 0.35), xmax=1.0)
    ax.set_xlim(-0.7, len(seas) - 0.35)
    # de-collide the four right-hand annotations by vertical spreading
    marks = [(m5 + sd5, f"+1 sd   {m5+sd5:+.1f}%", C_MODEL, "normal"),
             (m5, f"14-season clustered mean  {m5:+.2f}%"
                  f"   (pooled {100*D['roi']:+.2f}%)", C_MODEL, "normal"),
             (mR, f"REPORT-ERA clustered mean  {mR:+.2f}%"
                  f"   (pooled {100*FR8['roi']:+.2f}%)   <- PRIMARY",
              "#8a6d3b", "bold"),
             (m5 - sd5, f"-1 sd   {m5-sd5:+.1f}%", C_MODEL, "normal")]
    # headroom for the key, then the key INSIDE the axes so it cannot run into
    # panel (c).  Headroom is added ABOVE the data only — no datum is clipped.
    blo, bhi = ax.get_ylim()
    need = max(r5.max(), r1.max(), m5 + sd5)
    ax.set_ylim(min(blo, r5.min(), r1.min(), m5 - sd5) * 1.06, max(bhi, need * 1.62))
    for j, (v, lab, col, fw) in enumerate(marks):
        ax.text(0.435, 0.985 - 0.062 * j, lab, transform=ax.transAxes,
                color=col, fontsize=8.4, va="top", ha="left", fontweight=fw,
                zorder=6)
    for xi, v in zip(x + w / 2, r5):
        if v <= -3.0:
            ax.text(xi, v + 0.7, f"{v:.0f}", ha="center", va="bottom",
                    fontsize=7.6, color="white", zorder=5)
    ax.set_xticks(x, seas, fontsize=8.6, rotation=40, ha="right")
    ax.set_ylabel("ROI %, flat stake")
    ax.legend(frameon=False, fontsize=8.6, loc="lower left",
              bbox_to_anchor=(0.0, -0.02))
    ax.set_title(
        "(b)  ROI BY SEASON — THIS SPREAD IS WHERE THE CONFIDENCE INTERVAL COMES FROM\n"
        f"14 seasons, {D['n']:.0f} bets, all MEASURED-or-EXTRAPOLATED as panel (a).  Seasons run {r5.min():+.1f}% to "
        f"{r5.max():+.1f}% around a {100*D['roi']:+.2f}% mean;\n"
        f"sd = {sd5:.1f}pp, so the 13-dof interval is [{100*D['ci']['lo']:+.2f},{100*D['ci']['hi']:+.2f}] and it takes "
        f"~{D['diag']['k_resolve']:.0f} seasons to resolve a {100*D['roi']:+.2f}% effect.",
        fontsize=10.2, pad=10, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ================================================================ PANEL C
    ax = fig.add_subplot(gs[1, 2])
    rows = R["window_choice"]["rows"]
    labs = [r["name"].replace("RECENCY-", "").replace("ALL-HISTORY", "all")
            for r in rows]
    nets = np.array([100 * r["net"] for r in rows])
    chosen = R["chosen"]
    cols = [C_MODEL if r["name"] == chosen else "#c9d6e2" for r in rows]
    b = ax.barh(np.arange(len(rows)), nets, color=cols, height=0.62, zorder=3)
    top = nets.max()
    ax.axvline(top, color=INK2, lw=1.0, ls="--", zorder=4)
    ax.axvspan(top - 2.13, top, color=INK2, alpha=0.06, zorder=1)
    for i, v in enumerate(nets):
        ax.text(v + 0.22, i, f"{v:+.2f}", va="center", fontsize=8.4, color=INK2,
                zorder=5)
    ax.set_yticks(np.arange(len(rows)), labs, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, top * 1.62)
    ax.set_xlabel("ROI net of that window's OWN permutation null (pp)")
    ax.set_ylabel("calibration window (seasons)")
    ax.set_ylim(4.72, -1.52)
    ax.annotate("", xy=(top - 2.13, -0.72), xytext=(top, -0.72),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.9))
    ax.text(0.0, -1.24,
            "shaded = the 2.13pp tie band, D165's measured cost of having had\n"
            "7 procedures to choose from.",
            fontsize=7.7, color=INK2, va="center", ha="left")
    ax.set_title("(c)  IS IT 5 SEASONS, 3, OR MORE?\n"
                 "Rule declared first: best net-of-own-null; ties inside\n"
                 "2.13pp broken toward MORE history, because capacity\n"
                 "falls with window length.   ANSWER: MORE — use all.",
                 fontsize=10.2, pad=10, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ================================================================ PANEL D
    # D173: THE OWNER'S PRIMARY FRAME, ON ITS OWN AXES.
    axd = fig.add_subplot(gs[2, :])
    xr = np.arange(len(ridx))
    axd.axhline(0, color=INK2, lw=1.1, ls="--", zorder=2)
    dends = []
    for nm in ["k=1 raw", "k=5 raw", "k=8 raw"]:
        cum = np.array(T[nm]["cum"], float)
        base = cum[ridx[0] - 1] if ridx[0] > 0 else 0.0
        reb = cum[ridx] - base
        axd.plot(xr, reb, color=TIER_C[nm], lw=1.5, marker="o", ms=3.1, zorder=4)
        dends.append((reb[-1], nm, TIER_C[nm]))
    cumh = np.array(T["k=5 +haircut"]["cum"], float)
    baseh = cumh[ridx[0] - 1] if ridx[0] > 0 else 0.0
    rebh = cumh[ridx] - baseh
    axd.plot(xr, rebh, color=C_MODEL, lw=3.0, ls="--", dashes=(4, 2.2),
             marker="o", ms=4.2, zorder=6)
    dends.append((rebh[-1], "k=5 +haircut  (FIRM DEFAULT)", C_MODEL))
    dends.sort(key=lambda t: -t[0])
    yp, MG = [], max(1.2, 0.055 * (max(d[0] for d in dends)
                                   - min(d[0] for d in dends) + 1e-9))
    for v, *_ in dends:
        yy = v
        if yp and yy > yp[-1] - MG:
            yy = yp[-1] - MG
        yp.append(yy)
    for (v, lab, c), yy in zip(dends, yp):
        axd.annotate(f"{lab}   {v:+.1f}u", xy=(len(ridx) - 1, v),
                     xytext=(len(ridx) - 0.80, yy), color=c, fontsize=9,
                     va="center", ha="left",
                     fontweight="bold" if "DEFAULT" in lab else "normal",
                     arrowprops=dict(arrowstyle="-", color=c, lw=0.7,
                                     shrinkA=0, shrinkB=1, alpha=0.6))
    axd.set_xlim(-0.45, len(ridx) + 3.9)
    axd.set_xticks(xr, [seas[i] for i in ridx], fontsize=9)
    axd.set_ylabel("cumulative P&L, units\n(re-based to 0 at 2018-19)")
    sg = "SIG" if FR8["sig"] else "ns"
    axd.set_title(
        "(d)  PRIMARY FRAME — THE REPORT ERA ONLY, 2018-19..2025-26 (8 seasons, tier T2).  "
        "This is the frame that matches how the model will actually run live.\n"
        f"k=5 measured+haircut:  pooled ROI {100*FR8['roi']:+.2f}%  on {FR8['n']:.0f} bets;  "
        f"season-clustered mean {100*FR8['mean']:+.2f}% "
        f"[{100*FR8['lo']:+.2f},{100*FR8['hi']:+.2f}] at K=8 -> 7 dof ({sg});  "
        f"sd {100*FR8['sd']:.1f}pp.\n"
        f"14-season track, for contrast: pooled {100*FR14['roi']:+.2f}%, "
        f"clustered mean {100*FR14['mean']:+.2f}% "
        f"[{100*FR14['lo']:+.2f},{100*FR14['hi']:+.2f}] at 13 dof.",
        fontsize=10.2, pad=10, loc="left")
    for sp in ("top", "right"):
        axd.spines[sp].set_visible(False)
    axd.text(0.0, -0.20,
             "2018-19 report coverage is 788/1230 games (the season starts before 2018-12-17), so the first bar of this panel is T2 but not uniformly so — stated, not smoothed.  "
             "Tiers are never pooled silently (D158).",
             transform=axd.transAxes, fontsize=8.2, color=INK2, va="top")

    fig.suptitle(
        "WHAT A FIRM WOULD ACTUALLY HAVE MADE:  walk-forward configuration selection on the opening spread, 2012-13..2025-26",
        fontsize=13.5, x=0.052, ha="left", y=0.991, color=INK)
    fig.text(0.052, 0.977,
             "NO-LOOKAHEAD: the configuration is chosen on seasons 1..k and scored on season k+1 only.  Priced at the OPENING spread at -110.  Nothing is re-selected here — the frozen rule set is carried verbatim.\n"
             "RE-RUN ON THE D170/D171 BACKFILLED DATA (D173).  Availability tier is now BEST-AVAILABLE, not blind: T2i (official inactives) 2007-08..2017-18, full T2 (5PM injury report UNION inactives) 2018-19..2025-26.\n"
             "Panel (d) is the owner's PRIMARY frame.  Every interval below is season-clustered at K-1 dof.  D164: a blind search of this 600-cell space manufactures +16.92 ROI points from pure noise; D165: 7 procedures cost +2.13pp.",
             fontsize=9.3, color=INK2, ha="left", va="top", linespacing=1.55)
    fig.text(0.052, 0.018,
             "UNPRICED, AND IT IS THE LARGEST KNOWN BIAS IN THE OPTIMISTIC DIRECTION — LIMITS.  Best-of-k always transacts at whichever book is most offside; D163 measured 11.6% of games with a >3pt best-worst range and 8.1% of best prices\n"
             "more than 1.5pts off the other book.  Nothing above charges for being limited, restricted or voided.  SECOND UNPRICED BIAS, SAME DIRECTION: a firm betting size into a soft opening number is itself part of the flow that closes\n"
             f"the gap, so some of the measured {_clv:+.3f}pt CLV is mechanically unavailable at scale — direction named, magnitude unknown, deliberately not modelled.  EVERY INTERVAL ABOVE SPANS ZERO: this is a candidate, not a finding.",
             fontsize=8.0, color=C_BAD, ha="left", va="bottom", linespacing=1.5)

    out = ROOT / "charts" / "walkforward_equity.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
