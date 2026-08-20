#!/usr/bin/env python3
"""The two carry-all charts.  House style per scripts/make_status_charts.py:
model blue / market orange, thin marks, recessive grid, negative bar labels
INSIDE the bars, no dual axes, 150 dpi.

  charts/carryall_cost_ladder.png   what it COSTS to carry k terms, against the
                                    pure-noise benchmark  ("can we keep it all?")
  charts/carryall_era_tracking.png  league home edge, truth vs our walk-forward
                                    estimate, per adaptation config (the LAG)
"""
from __future__ import annotations

import datetime as dt
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
GAP = 0.01120          # D132 remaining raw log-loss gap to the closing line
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlecolor": INK,
})


def despine(ax, keep=("left", "bottom")):
    for sp in ("top", "right", "left", "bottom"):
        if sp not in keep:
            ax.spines[sp].set_visible(False)


# ===========================================================  CHART 1
def cost_ladder():
    L = json.load(open(ROOT / "data" / "carryall_ladder.json"))
    A = L["arms"]
    real = [("pile:k1", 1), ("pile:k2", 2), ("pile:k3", 3), ("pile:k5", 5),
            ("pile:k8", 8), ("pile:k10", 10), ("pile:k15", 15),
            ("pile:ALL15+TEAMHOME", 45)]
    noise = [(f"noise:k{k}", k) for k in (1, 2, 3, 5, 8, 10, 15, 45)]

    fig = plt.figure(figsize=(15.2, 10.4), dpi=150)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.0],
                          height_ratios=[1.0, 0.92], wspace=0.26, hspace=0.34)

    # ---- panel A: the ladder
    ax = fig.add_subplot(gs[0, 0])
    for series, col, lab, mk in ((real, C_MODEL, "the REAL rejected pile", "o"),
                                 (noise, C_MKT, "pure NOISE columns "
                                                "(same shrinkage)", "s")):
        x = [k for _, k in series]
        y = [A[n]["delta"] for n, _ in series]
        lo = [A[n]["cl_lo"] for n, _ in series]
        hi = [A[n]["cl_hi"] for n, _ in series]
        ax.errorbar(x, y, yerr=[np.array(y) - np.array(lo),
                                np.array(hi) - np.array(y)],
                    color=col, marker=mk, ms=4.5, lw=1.4, elinewidth=1.0,
                    capsize=2.5, label=lab, zorder=3)
    xt = np.linspace(0, 46, 100)
    ax.plot(xt, -9.3e-5 * xt, color=INK2, ls=":", lw=1.1, zorder=2,
            label="pre-registered theory  $-9.3\\times10^{-5}\\,k$")
    ax.axhline(0, color=INK2, lw=0.9)
    ax.axhline(-GAP, color=C_BAD, ls="--", lw=1.1)
    ax.text(46, -GAP + 0.0004, "the ENTIRE remaining gap to the closing line\n"
            "(D132, −0.01120)", color=C_BAD, fontsize=8.2, ha="right", va="bottom")
    ax.set_xlabel("number of coefficients carried in the shrunk schedule layer")
    ax.set_ylabel("paired log loss vs the D132 control\n(negative = carrying COSTS us)")
    ax.set_title("Carrying everything is NOT free — and the real pile\n"
                 "costs 5× what the same number of noise columns costs",
                 fontsize=11)
    d6 = A["pile:DENSE6"]
    ax.plot([6], [d6["delta"]], marker="*", ms=13, color=C_GOOD, zorder=5,
            ls="none", label="the 6 DENSE schedule terms — FREE")
    ax.annotate(f"the 6 dense continuous terms\ncost {d6['delta']:+.5f}  "
                f"CI({d6['cl_lo']:+.5f},{d6['cl_hi']:+.5f}) ns\n"
                "— cheaper than 6 noise columns",
                xy=(6, d6["delta"]), xytext=(19.0, -0.0062),
                fontsize=8.2, color=C_GOOD,
                arrowprops=dict(arrowstyle="->", color=C_GOOD, lw=0.9))
    ax.legend(loc="lower left", frameon=False, fontsize=8.4)
    ax.set_xlim(-1, 47)
    despine(ax)

    # ---- panel B: per-term solo cost
    ax = fig.add_subplot(gs[0, 1])
    # TEAMHOME is 30 columns and -0.01253 -- an order of magnitude off this
    # scale, and a different animal.  It is shown in panel A and called out
    # in the footnote rather than crushing the 15 single-column terms.
    solos = sorted([(n.split(":", 1)[1], A[n]["delta"], A[n]["cl_lo"],
                     A[n]["cl_hi"], A[n]["ncol"])
                    for n in A if n.startswith("solo:")
                    and n != "solo:TEAMHOME"], key=lambda r: r[1])
    names = [r[0] for r in solos]
    vals = [r[1] for r in solos]
    cols = [C_BAD if v < -0.0002 else C_GOOD for v in vals]
    ax.barh(range(len(vals)), vals, color=cols, height=0.62, zorder=3)
    span = max(abs(min(vals)), 1e-9)
    for i, (nm, v, lo, hi, nc) in enumerate(solos):
        if v < 0 and abs(v) > 0.45 * span:          # long enough: label INSIDE
            ax.text(v + 0.03 * span, i, f"{v:+.5f}", va="center", ha="left",
                    fontsize=7.8, color="white", zorder=4)
        else:                                        # too short: label outside
            ax.text(v + (0.03 * span if v >= 0 else -0.03 * span), i,
                    f"{v:+.5f}", va="center",
                    ha="left" if v >= 0 else "right",
                    fontsize=7.8, color=INK2, zorder=4)
    ax.set_yticks(range(len(names)),
                  [f"{n}" + (f"  [{c} cols]" if c > 1 else "") for n, c in
                   zip(names, [r[4] for r in solos])], fontsize=8.2)
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_xlim(min(vals) * 1.55, max(vals) * 3.4)
    ax.set_xlabel("paired log loss, term carried ALONE")
    ax.set_title("Each term alone: nine of fifteen are free,\n"
                 "and the two D47 dead-team flags are not", fontsize=11)
    ax.text(min(vals) * 1.52, len(vals) - 0.35,
            "off-scale: TEAMHOME (D70, 30 cols) = −0.01253 — see panel A",
            fontsize=7.8, color=C_BAD, va="center")
    despine(ax)

    # ---- panel C: injected margin noise (deterministic, no outcome noise)
    ax = fig.add_subplot(gs[1, 0])
    for series, col, lab, mk in ((real, C_MODEL, "real pile", "o"),
                                 (noise, C_MKT, "pure noise", "s")):
        x = [k for _, k in series]
        ax.plot(x, [A[n]["rms_dm"] for n, _ in series], color=col, marker=mk,
                ms=4.5, lw=1.4, label=lab, zorder=3)
    ax.plot(xt, 0.216 * np.sqrt(xt), color=INK2, ls=":", lw=1.1,
            label="theory  $0.216\\sqrt{k}$")
    ax.axhline(1.485, color=C_BAD, ls="--", lw=1.1)
    ax.text(1.0, 1.56, "break-even for a TRUE term (1.485 pts)", color=C_BAD,
            fontsize=8.2, ha="left", va="bottom")
    ax.set_xlabel("number of coefficients carried")
    ax.set_ylabel("rms margin perturbation injected (points)")
    ax.set_title("The same story with no outcome noise:\n"
                 "how many points of junk we inject", fontsize=11)
    ax.legend(loc="upper left", frameon=False, fontsize=8.6)
    ax.set_xlim(-1, 47)
    despine(ax)

    # ---- panel D: THE RECONCILIATION.  Same D70 channel, same corpus, same
    # layer, same control -- only the shrinkage on the block changes.
    ax = fig.add_subplot(gs[1, 1])
    R = json.load(open(ROOT / "data" / "carryall_ridge_reconcile.json"))
    order = ["TEAMHOME_ridge200", "TEAMHOME_ridge50", "TEAMHOME_unpenalised"]
    labs = ["ridge 200\n(the shipped `team_home_ridge`,\nand D153's carry test)",
            "ridge 50", "UNPENALISED\n(the layer's global n/(n+600) only)"]
    vals = [R[k]["delta"] for k in order]
    los = [R[k]["cl_lo"] for k in order]
    his = [R[k]["cl_hi"] for k in order]
    colr = [C_GOOD, "#d98b2b", C_BAD]
    ax.barh(range(3), vals, color=colr, height=0.5, zorder=3)
    ax.errorbar(vals, range(3), xerr=[np.array(vals) - np.array(los),
                                      np.array(his) - np.array(vals)],
                fmt="none", ecolor=INK2, elinewidth=1.0, capsize=3, zorder=4)
    for i, (k, v, lo) in enumerate(zip(order, vals, los)):
        if abs(v) > 0.004:                       # long bar: label INSIDE
            ax.text(v + 0.0004, i,
                    f"{v:+.5f}   (coef rms {R[k]['th_rms']:.2f} pts)",
                    va="center", ha="left", fontsize=8.4, color="white",
                    zorder=5)
        else:                                     # short bar: clear the whisker
            ax.text(lo - 0.0004, i,
                    f"{v:+.5f}   (coef rms {R[k]['th_rms']:.2f} pts)",
                    va="center", ha="right", fontsize=8.4, color=INK2, zorder=5)
    ax.set_yticks(range(3), labs, fontsize=8.4)
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_xlim(min(vals) * 1.42, 0.0014)
    ax.set_xlabel("paired log loss, D70 team-home block carried alone")
    ax.set_title("THE ANSWER: identical channel, identical corpus —\n"
                 "cost swings 19x on the SHRINKAGE alone", fontsize=11)
    despine(ax)

    fig.suptitle("CAN WE JUST KEEP EVERYTHING?  Carrying the rejected pile in "
                 "the shrunk schedule layer — D132 control, n=6,148.\n"
                 "Not a question about which terms: a question about how hard "
                 "each one is shrunk.",
                 fontsize=12.5, y=0.985)
    fig.tight_layout(rect=(0, 0.01, 1, 0.945))
    p = ROOT / "charts" / "carryall_cost_ladder.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p)


# ===========================================================  CHART 2
def era_tracking():
    E = json.load(open(ROOT / "data" / "carryall_era.json"))
    tru = E["truth_season"]
    tr = E["track"]

    def xy(cfg):
        """Series with a NaN break at every season boundary, so the offseason
        is not drawn as a straight line through data that does not exist."""
        x, y, ss = [], [], []
        prev = None
        for a in tr[cfg]:
            if prev is not None and a["season"] != prev:
                x.append(dt.date.fromisoformat(a["date"]) - dt.timedelta(days=1))
                y.append(np.nan)
                ss.append(a["season"])
            x.append(dt.date.fromisoformat(a["date"]))
            y.append(a["he"])
            ss.append(a["season"])
            prev = a["season"]
        return x, y, ss

    fig = plt.figure(figsize=(15.2, 8.2), dpi=150)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.55, 1.0], hspace=0.22)

    # ---- truth step series
    seasons = sorted(tru)
    xs, ys = [], []
    for s in seasons:
        d = [dt.date.fromisoformat(a["date"]) for a in tr["BASE"]
             if a["season"] == s]
        if not d:
            continue
        xs += [d[0], d[-1], d[-1] + dt.timedelta(days=1)]
        ys += [tru[s], tru[s], np.nan]

    ax = fig.add_subplot(gs[0])
    ax.axvspan(dt.date(2020, 12, 22), dt.date(2021, 5, 16),
               color="#f4efe6", zorder=0)
    ax.annotate("2020-21 — NO CROWD\ntruth steps to +0.944",
                xy=(dt.date(2021, 2, 15), 0.98),
                xytext=(dt.date(2021, 9, 1), 0.72), fontsize=8.6, color=INK2,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
    ax.plot(xs, ys, color=INK, lw=1.6, zorder=4,
            label="TRUTH — realised season home margin")
    for cfg, col, lw, ls in (("BASE", C_MODEL, 1.6, "-"),
                             ("C1", C_MKT, 1.2, "-"),
                             ("C2+C3", C_GOOD, 1.2, "-")):
        x, y, _ = xy(cfg)
        ax.plot(x, y, color=col, lw=lw, ls=ls, zorder=3,
                alpha=0.85 if cfg == "C1" else 1.0,
                label={"BASE": "SHIPPED estimator (730d boxcar, n/(n+600) → 2.3)",
                       "C1": "C1 trend-aware (local-linear)",
                       "C2+C3": "C2+C3 data-driven prior + change-point"}[cfg])
    # 2010s decline annotation
    ax.annotate("the 2010s decline: −0.053 ± 0.010 pts/season\n"
                "(t=−5.34) — a DRIFT, not a random walk",
                xy=(dt.date(2015, 6, 1), 2.95), xytext=(dt.date(2012, 2, 1), 3.62),
                fontsize=8.6, color=INK2,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
    ax.annotate("SHIPPED estimate is still +0.75 pts too high at the\n"
                "END of 2020-21 — tracking-error half-life 219 days",
                xy=(dt.date(2021, 4, 25), 1.72),
                xytext=(dt.date(2022, 3, 1), 3.42),
                fontsize=8.8, color=C_MODEL,
                arrowprops=dict(arrowstyle="->", color=C_MODEL, lw=0.9))
    ax.set_ylabel("league home advantage (points)")
    ax.set_title("How fast do we actually update through eras?  Walk-forward "
                 "home edge vs truth, weekly refits, 2010-11 … 2025-26",
                 fontsize=12)
    ax.legend(loc="lower center", frameon=False, fontsize=8.8, ncol=2,
              bbox_to_anchor=(0.42, -0.02))
    ax.set_ylim(0.35, 4.05)
    despine(ax)

    # ---- error panel
    ax = fig.add_subplot(gs[1])
    ax.axvspan(dt.date(2020, 12, 22), dt.date(2021, 5, 16),
               color="#f4efe6", zorder=0)
    ax.axhline(0, color=INK, lw=1.0, zorder=2)
    for cfg, col, lw in (("BASE", C_MODEL, 1.6), ("C1", C_MKT, 1.2),
                         ("C2+C3", C_GOOD, 1.2)):
        x, y, ss = xy(cfg)
        e = [v - tru[s] for v, s in zip(y, ss)]
        ax.plot(x, e, color=col, lw=lw, zorder=3, label=cfg)
    lg = E["lag"]
    ax.text(dt.date(2010, 11, 1), 1.34,
            "mean |error| over 384 refits:   "
            + "   ".join(f"{c} {lg[c]['mae_season']:.3f}"
                         for c in ("BASE", "C1", "C2", "C3", "C2+C3")),
            fontsize=8.8, color=INK2)
    ax.set_ylabel("tracking error\n(estimate − truth, points)")
    ax.set_ylim(-1.45, 1.75)
    ax.legend(loc="lower right", frameon=False, fontsize=8.8, ncol=3)
    despine(ax)

    fig.suptitle("ANSWER: the era parameter DRIFTS slowly and STEPS once — a "
                 "1.19-pt step needs ~1,025 games (123 days) to detect at 2.5σ",
                 fontsize=12.5, y=0.985)
    fig.tight_layout(rect=(0, 0.01, 1, 0.945))
    p = ROOT / "charts" / "carryall_era_tracking.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    cost_ladder()
    era_tracking()
