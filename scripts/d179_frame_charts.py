#!/usr/bin/env python3
"""D179 — the two REPORTING-FRAME charts, on the frames actually reported.

charts/frame_model_post2018.png    model accuracy, 2018-19 onward (injury-report
                                   era), against the pre-feed seasons as context
charts/frame_betting_k8_2023_26.png  betting at k=8 (MAX BOOKS) on the measured
                                   multi-book panel, 2023-24..2025-26

Both frames are set by DATA AVAILABILITY, not by which window scored best:
  * the daily injury report the availability leg depends on begins 2018-12-17
  * the measured multi-book price panel begins 2023-24 (earlier seasons infer
    the multi-book price from a shopping law — D177 showed it ~1.25x generous)

House style per scripts/make_status_charts.py: model blue / market orange, thin
marks, recessive grid, no dual axes, 150 dpi.  Palette validated (categorical,
light): CVD deutan worst adjacent dE 6.6 -> EVERY mark carries a direct value
label as the required secondary encoding, which also clears the contrast WARN on
the green.

Read-only.  Nothing ships.  No default changed.
"""
from __future__ import annotations

import json
import math
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

LN2 = math.log(2)
POST = ["2018-19", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
PRE = ["2007-08", "2008-09", "2009-10", "2010-11", "2012-13", "2013-14",
       "2014-15", "2015-16", "2016-17", "2017-18"]
BET = ["2023-24", "2024-25", "2025-26"]
TIER = "k=8 raw"            # MAX BOOKS — the owner's reporting tier
TIER_HC = "k=8 +haircut"


def pool(d, seasons):
    S = [d[s] for s in seasons if s in d]
    n = np.array([r["n"] for r in S], float)
    us = np.average([r["ll_us"] for r in S], weights=n)
    mk = np.average([r["ll_mkt"] for r in S], weights=n)
    return dict(K=len(S), n=int(n.sum()), ll_us=us, ll_mkt=mk,
                gap=100 * (us - mk) / (LN2 - mk))


# ------------------------------------------------------------------ CHART 1
def chart_model():
    d = json.load(open(ROOT / "data" / "d171_history_analysis.json"))["per_season"]
    post, pre = pool(d, POST), pool(d, PRE)
    gaps = [d[s]["norm_gap_pct"] for s in POST]

    fig = plt.figure(figsize=(14.2, 8.4), dpi=150)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.0], hspace=0.52,
                          left=0.175, right=0.975, top=0.825, bottom=0.10)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(POST))
    bars = ax.bar(x, gaps, width=0.58, color=C_MODEL, zorder=3)
    for xi, g in zip(x, gaps):
        # a label within 1.3pp of the pooled rule would sit on top of it, so
        # those drop inside the bar in white instead
        if abs(g - post["gap"]) < 1.3:
            ax.text(xi, g - 0.55, f"{g:.2f}%", ha="center", va="top",
                    fontsize=9.6, color="white", fontweight="bold", zorder=6)
        else:
            ax.text(xi, g + 0.35, f"{g:.2f}%", ha="center", va="bottom",
                    fontsize=9.6, color=INK, fontweight="bold")
    ax.axhline(post["gap"], color=BROWN, lw=1.6, ls=(0, (5, 2.5)), zorder=4)
    ax.text(len(POST) - 0.35, post["gap"], f"  pooled {post['gap']:.2f}%\n"
            f"  (n={post['n']:,})",
            color=BROWN, fontsize=9.4, ha="left", va="center",
            fontweight="bold")
    ax.axhline(0, color=INK2, lw=1.1, zorder=4)
    ax.text(len(POST) - 0.35, 0.35, "  0% = we match\n  the market",
            color=INK2, fontsize=8.8, ha="left", va="bottom")
    ax.set_xticks(x, POST)
    ax.set_xlim(-0.62, len(POST) + 0.35)      # room at right for the pooled label
    ax.set_ylim(0, max(gaps) * 1.22)
    ax.set_ylabel("normalized gap behind the market\n"
                  "(% of market skill-above-coinflip we miss)")
    ax.set_title("(a)  THE REPORTED MODEL FRAME — 2018-19 onward, the seasons in "
                 "which the daily injury report exists.\n"
                 "     Lower is better; every bar is above zero, so the market "
                 "wins every season of this frame.",
                 fontsize=11.0, pad=10, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    axb = fig.add_subplot(gs[1, 0])
    labs = ["pre-feed seasons\n2007-08 … 2017-18",
            "REPORTED FRAME\n2018-19 onward",
            "all poolable\n(blended headline)"]
    allp = pool(d, PRE + POST)
    vals = [pre["gap"], post["gap"], allp["gap"]]
    ks = [pre["K"], post["K"], allp["K"]]
    cols = ["#a9bdd0", C_MODEL, "#a9bdd0"]
    y = np.arange(len(labs))[::-1]
    axb.barh(y, vals, height=0.5, color=cols, zorder=3)
    for yi, v, k in zip(y, vals, ks):
        axb.text(v + 0.18, yi, f"{v:.2f}%   (K={k} seasons)", va="center",
                 ha="left", fontsize=9.6, color=INK,
                 fontweight="bold" if abs(v - post["gap"]) < 1e-9 else "normal")
    axb.set_yticks(y, labs, fontsize=9.2)
    axb.set_xlim(0, max(vals) * 1.42)
    axb.set_xlabel("normalized gap behind the market (%)")
    axb.set_title("(b)  WHY THIS FRAME, AND WHY IT IS THE WORSE NUMBER — before "
                  "the feed exists the availability leg runs on inputs it\n"
                  "     was never designed to have, so those seasons score a "
                  "crippled variant.  The blended headline averages two\n"
                  "     different models and flatters the one we would deploy.",
                  fontsize=10.2, pad=10, loc="left")
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)

    fig.suptitle("MODEL ACCURACY ON THE REPORTED FRAME — the injury-report era, "
                 "2018-19 onward", fontsize=13.6, x=0.052, ha="left", y=0.978,
                 color=INK)
    fig.text(0.052, 0.945,
             "Frame chosen by data availability (the daily injury report begins "
             "2018-12-17), not by which window scored best —\n"
             "and it scores WORSE than the window it replaces.  "
             "COVID seasons (2019-20, 2020-21) excluded throughout.",
             fontsize=9.3, color=INK2, ha="left", va="top", linespacing=1.5)
    out = ROOT / "charts" / "frame_model_post2018.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  post-2018 pooled gap {post['gap']:.2f}%  K={post['K']}  n={post['n']}")
    print(f"  pre-feed {pre['gap']:.2f}%   all {allp['gap']:.2f}%")


# ------------------------------------------------------------------ CHART 2
def chart_betting():
    R = json.load(open(ROOT / "data" / "wf_equity_HONEST.json"))
    pb = json.load(open(ROOT / "data" / "wf_perbet_HONEST.json"))
    rows = {r["season"]: r for r in R["tiers"][TIER]["rows"]}
    rows_hc = {r["season"]: r for r in R["tiers"][TIER_HC]["rows"]}

    roi = np.array([100 * rows[s]["pay"] / rows[s]["n"] for s in BET])
    n_tot = sum(rows[s]["n"] for s in BET)
    pay_tot = sum(rows[s]["pay"] for s in BET)
    pooled = 100 * pay_tot / n_tot
    se = roi.std(ddof=1) / np.sqrt(len(BET))
    tcrit = 4.303                                   # K=3 -> 2 dof, two-sided .05
    lo, hi = pooled - tcrit * se, pooled + tcrit * se
    roi_hc = 100 * sum(rows_hc[s]["pay"] for s in BET) / n_tot

    bets = [b for b in pb[TIER] if b["season"] in BET]
    bets.sort(key=lambda b: (b["date"], b["gid"]))
    ev = np.array([b["ev"] * b["keep"] for b in bets], float)
    cum = np.cumsum(ev)

    fig = plt.figure(figsize=(14.8, 9.0), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0],
                          width_ratios=[1.85, 1.0], hspace=0.46, wspace=0.24,
                          left=0.065, right=0.978, top=0.845, bottom=0.105)

    ax = fig.add_subplot(gs[0, :])
    ax.axhline(0, color=INK2, lw=1.1, ls="--", zorder=2)
    ax.plot(np.arange(len(cum)), cum, color=C_MODEL, lw=2.4, zorder=5)
    bnds, seen = [], set()
    for i, b in enumerate(bets):
        if b["season"] not in seen:
            seen.add(b["season"])
            bnds.append((i, b["season"]))
    ylo, yhi = ax.get_ylim()
    for i, s in bnds[1:]:
        ax.axvline(i - 0.5, color=BROWN, lw=1.1, ls=(0, (5, 2.5)), zorder=4)
    for i, s in bnds:
        ax.text(i + 3, ylo + 0.045 * (yhi - ylo), s, color=BROWN, fontsize=8.8,
                ha="left", va="bottom")
    ax.set_ylim(ylo, yhi)
    ax.annotate(f"{cum[-1]:+.1f}u over {len(ev)} bets\n{pooled:+.2f}% ROI",
                xy=(len(cum) - 1, cum[-1]), xytext=(len(cum) * 1.02, cum[-1]),
                color=C_MODEL, fontsize=10.0, va="center", ha="left",
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=C_MODEL, lw=0.8, alpha=0.6))
    ax.set_xlim(-4, len(cum) * 1.16)
    ax.set_xlabel(f"sequential bet, in date order  (n = {len(ev)})")
    ax.set_ylabel("cumulative P&L, units\n(flat 1u per bet)")
    ax.set_title(f"(a)  EVERY BET ON THE MEASURED PANEL, IN ORDER — walk-forward "
                 f"selection at the opening spread, priced at {TIER.upper()} "
                 f"(MAX BOOKS)",
                 fontsize=11.2, pad=11, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    axb = fig.add_subplot(gs[1, 0])
    x = np.arange(len(BET))
    cols = [C_GOOD if v > 0 else C_BAD for v in roi]
    axb.bar(x, roi, width=0.5, color=cols, zorder=3)
    for xi, s in zip(x, BET):
        v = 100 * rows[s]["pay"] / rows[s]["n"]
        axb.text(xi, v + (0.35 if v >= 0 else -0.35), f"{v:+.2f}%",
                 ha="center", va="bottom" if v >= 0 else "top",
                 fontsize=9.8, color=INK, fontweight="bold")
        axb.text(xi, 0.28, f"n={rows[s]['n']:.0f}", ha="center", va="bottom",
                 fontsize=8.4, color="white")
    axb.axhline(0, color=INK2, lw=1.1, zorder=4)
    axb.axhline(pooled, color=BROWN, lw=1.5, ls=(0, (5, 2.5)), zorder=4)
    axb.text(len(BET) - 0.55, pooled + 0.3, f"pooled {pooled:+.2f}%",
             color=BROWN, fontsize=9.2, ha="right", va="bottom",
             fontweight="bold")
    axb.set_xticks(x, BET)
    axb.set_ylim(min(0, roi.min()) - 1.4, roi.max() * 1.26)
    axb.set_ylabel("ROI (%)")
    axb.set_title("(b)  BY SEASON — 2024-25 carries it", fontsize=10.4, pad=9,
                  loc="left")
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)

    axc = fig.add_subplot(gs[1, 1])
    axc.axvline(0, color=C_BAD, lw=1.6, zorder=5)
    axc.plot([lo, hi], [0, 0], color=C_MODEL, lw=3.4, solid_capstyle="butt",
             zorder=4)
    axc.plot([pooled], [0], marker="o", ms=10, color=C_MODEL, zorder=6)
    axc.text(pooled, 0.22, f"{pooled:+.2f}%", ha="center", va="bottom",
             fontsize=10.4, color=INK, fontweight="bold")
    axc.text(lo, -0.26, f"{lo:+.1f}%", ha="center", va="top", fontsize=9.0,
             color=INK2)
    axc.text(hi, -0.26, f"{hi:+.1f}%", ha="center", va="top", fontsize=9.0,
             color=INK2)
    axc.text(0, 0.52, "zero", ha="center", va="bottom", fontsize=9.0,
             color=C_BAD, fontweight="bold")
    axc.set_ylim(-1.0, 1.0)
    axc.set_yticks([])
    axc.set_xlabel("ROI (%)")
    axc.set_title("(c)  95% CI — it contains zero", fontsize=10.4, pad=9,
                  loc="left")
    for sp in ("top", "right", "left"):
        axc.spines[sp].set_visible(False)

    fig.suptitle(f"THE REPORTED BETTING FRAME — 2023-24 … 2025-26, priced at "
                 f"{TIER} (MAX BOOKS)", fontsize=13.6, x=0.065, ha="left",
                 y=0.975, color=INK)
    fig.text(0.065, 0.938,
             f"The only seasons with a MEASURED multi-book price panel; earlier "
             f"seasons infer the multi-book price from a shopping law.  "
             f"No-lookahead: config chosen on seasons 1..k, scored on k+1.\n"
             f"After the outlier-realism haircut this is {roi_hc:+.2f}%.  "
             f"K=3 seasons, so the 95% interval is [{lo:+.2f}%, {hi:+.2f}%] — it "
             f"contains zero — and 2024-25 alone supplies "
             f"{100*rows['2024-25']['pay']/pay_tot:.0f}% of the P&L.  "
             f"A candidate, not a result.",
             fontsize=9.3, color=INK2, ha="left", va="top", linespacing=1.5)
    out = ROOT / "charts" / "frame_betting_k8_2023_26.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  {TIER}: pooled {pooled:+.2f}%  {pay_tot:+.2f}u on {n_tot} bets  "
          f"95%CI [{lo:+.2f},{hi:+.2f}]")
    print(f"  {TIER_HC}: {roi_hc:+.2f}%   2024-25 share {100*rows['2024-25']['pay']/pay_tot:.1f}%")


if __name__ == "__main__":
    chart_model()
    chart_betting()
