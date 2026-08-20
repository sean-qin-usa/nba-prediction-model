"""charts/overfit_capacity.png — the decay, made self-evident.

House style per scripts/make_status_charts.py. 150 dpi.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nbapred.threads  # noqa: E402
nbapred.threads.pin(1)

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

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


def main():
    r = json.load(open(ROOT / "data" / "oc_capacity.json"))
    A, B, C = r["arm_a"], r["arm_b"], r["arm_c"]
    cap = r["capacity"]
    seasons = r["seasons"]
    lab = [s[2:] for s in seasons]           # 07-08, 08-09, ...

    IS = np.array([a["is_roi"] for a in A]) * 100
    OOS = np.array([a["oos_roi_pooled"] for a in A]) * 100

    fig = plt.figure(figsize=(15.5, 8.6), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.42, 1.0],
                          width_ratios=[1.0, 1.0], hspace=0.62, wspace=0.19)

    # ------------------------------------------------ PANEL A: the decay
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(seasons))
    w = 0.38

    # noise-floor band: the null distribution of the IN-SAMPLE tuned ROI
    n_is = C["is_roi"]
    ax.axhspan(100 * n_is["p05"], 100 * n_is["p95"], color=C_BAD, alpha=0.10,
               zorder=0, lw=0)
    ax.axhline(100 * n_is["mean"], color=C_BAD, ls=(0, (5, 3)), lw=1.2, zorder=1)
    n_oos = C["oos_roi"]
    ax.axhspan(100 * n_oos["p05"], 100 * n_oos["p95"], color=INK2, alpha=0.13,
               zorder=0, lw=0)

    ax.bar(x - w / 2, IS, w, color=C_MODEL, zorder=3,
           label="IN-SAMPLE: ROI on the season it was tuned to")
    ax.bar(x + w / 2, OOS, w, color=C_MKT, zorder=3,
           label="OUT-OF-SAMPLE: same frozen config, other 18 seasons")
    # decay connectors
    for xi, a, b in zip(x, IS, OOS):
        ax.plot([xi - w / 2, xi + w / 2], [a, b], color=INK2, lw=0.8,
                alpha=0.55, zorder=4)
    ax.axhline(0, color=INK, lw=1.0, zorder=2)

    # white bboxes: several value labels land on the shaded noise-floor bands
    bb = dict(facecolor="white", edgecolor="none", alpha=0.9,
              boxstyle="square,pad=0.10")
    for xi, v in zip(x, IS):
        ax.text(xi - w / 2, v + 0.7, f"{v:.0f}", ha="center", va="bottom",
                fontsize=8.0, color=C_MODEL, zorder=6, bbox=bb)
    for xi, v in zip(x, OOS):
        ax.text(xi + w / 2, v + (0.7 if v >= 0 else -0.7), f"{v:+.1f}",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8.0, color=C_MKT, zorder=6, bbox=bb)

    ax.set_xticks(x, lab, fontsize=9)
    ax.set_ylim(-15, 45)
    ax.set_ylabel("ATS ROI at -110  (%)")
    ax.set_title("Tune to ONE season, then look elsewhere: 19 of 19 tuning targets "
                 "are positive in sample, 5 of 19 survive out of it",
                 fontsize=12.5, pad=24, loc="left")
    ax.text(0.0, 1.018, "each pair = the best of 600 pre-registered configurations "
            "chosen on that season alone", transform=ax.transAxes,
            fontsize=9.4, color=INK2, ha="left")

    hd = [Patch(fc=C_MODEL, label="IN-SAMPLE: ROI on the season it was tuned to"),
          Patch(fc=C_MKT, label="OUT-OF-SAMPLE: same frozen config, other 18 seasons"),
          Patch(fc=C_BAD, alpha=0.25,
                label=f"noise floor, IN sample {100*n_is['mean']:+.1f}%   "
                      f"(5-95%: {100*n_is['p05']:+.1f} to {100*n_is['p95']:+.1f})"),
          Patch(fc=INK2, alpha=0.28,
                label=f"noise floor, OUT of sample {100*n_oos['mean']:+.1f}%   "
                      f"(5-95%: {100*n_oos['p05']:+.1f} to {100*n_oos['p95']:+.1f})")]
    ax.legend(handles=hd, loc="upper left", frameon=False, fontsize=8.9,
              ncol=1, bbox_to_anchor=(0.004, 0.995), labelspacing=0.42)

    # --------------------------------------- PANEL B: walk-forward vs the null
    ax2 = fig.add_subplot(gs[1, 0])
    steps = B["steps"]
    xs = np.arange(len(steps))
    wf = np.array([s["test_roi"] for s in steps]) * 100
    nullstep = np.array(C["wf_per_step_mean"]) * 100
    cols = [C_GOOD if v > 0 else C_BAD for v in wf]
    ax2.bar(xs, wf, 0.62, color=cols, zorder=3)
    ax2.plot(xs, nullstep, color=INK2, lw=1.3, ls=(0, (4, 2)), marker="o",
             ms=3.0, zorder=4, label="same loop on permuted predictions")
    ax2.axhline(0, color=INK, lw=1.0, zorder=2)
    ax2.axhline(B["pooled_roi"] * 100, color=C_MODEL, lw=1.5, zorder=5)
    ax2.set_xticks(xs, [s["test_season"][2:] for s in steps], fontsize=8.6)
    ax2.set_ylabel("ROI on the next unseen season (%)")
    ax2.set_ylim(-19, 26)
    ax2.set_title("WALK-FORWARD selection: tune on seasons 1..k, score on k+1",
                  fontsize=11.4, loc="left", pad=7)
    ax2.text(13.45, 25.2,
             f"pooled {B['pooled_roi']*100:+.2f}%   CI(13 dof) "
             f"[{B['ci']['lo']*100:+.2f},{B['ci']['hi']*100:+.2f}] ns",
             fontsize=8.8, color=C_MODEL, va="top", ha="right", zorder=7,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                       boxstyle="square,pad=0.14"))
    ax2.text(13.45, 20.6,
             f"noise floor {C['wf_roi']['mean']*100:+.2f}%  |  net "
             f"{B['net_of_null']*100:+.2f} pts, p={C['wf_p_value']:.3f}",
             fontsize=8.8, color=INK2, ha="right", va="top", zorder=7,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                       boxstyle="square,pad=0.14"))
    ax2.legend(loc="lower left", frameon=False, fontsize=8.6)

    # ------------------------------ PANEL C: capacity vs the register's numbers
    ax3 = fig.add_subplot(gs[1, 1])
    names = ["capacity\n(600 cells)", "capacity\n(6 cells, T only)",
             "D161 rules\nDEV-OOS gap", "2024-25\nATS ROI",
             "D162 DEV5-OOS14\ngap"]
    vals = [cap["mean_decay"] * 100, 3.08, 3.46, 3.22, 1.48]
    cols3 = [C_BAD, C_BAD, INK2, C_MODEL, INK2]
    bars = ax3.barh(np.arange(len(vals))[::-1], vals, 0.55, color=cols3, zorder=3)
    for b, v in zip(bars, vals):
        ax3.text(v + 0.35, b.get_y() + b.get_height() / 2, f"{v:+.2f}",
                 va="center", fontsize=9.2, color=INK2)
    ax3.set_yticks(np.arange(len(names))[::-1], names, fontsize=8.8)
    ax3.set_xlim(0, 20.5)
    ax3.set_xlabel("ROI points")
    ax3.set_title("What tuning can manufacture, vs every gap the register holds",
                  fontsize=11.4, loc="left", pad=7)
    ax3.text(19.9, 0.10,
             "every measured gap sits INSIDE\nwhat a one-season grid search\n"
             "produces from nothing",
             ha="right", va="bottom", fontsize=9.0, color=C_BAD)

    # the headline, as a full-width band in the gap between the two rows
    fig.text(0.5, 0.462,
             f"CAPACITY  =  mean in-sample {cap['mean_is']*100:+.2f}%   -   "
             f"mean out-of-sample {cap['mean_oos_pooled']*100:+.2f}%   =   "
             f"{cap['mean_decay']*100:+.2f} ROI points",
             ha="center", va="center", fontsize=12.4, color=INK, zorder=8,
             bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.42"))
    fig.text(0.5, 0.428,
             f"CI(18 dof) [{cap['ci_decay']['lo']*100:+.2f}, "
             f"{cap['ci_decay']['hi']*100:+.2f}]        "
             f"the SAME loop on pure noise manufactures "
             f"{C['capacity']['mean']*100:+.2f}   ->   "
             f"NET OF NOISE {cap['net_of_null']*100:+.2f} pts  (p={C['capacity_p_value']:.2f})",
             ha="center", va="center", fontsize=10.2, color=C_BAD, zorder=8)

    fig.suptitle("OVERFITTING CAPACITY — 19 seasons, 22,742 ATS bets at the "
                 "opening spread, -110  (D164, diagnostic; nothing shipped)",
                 fontsize=13.6, y=0.982, x=0.008, ha="left", color=INK)
    fig.text(0.008, 0.008,
             "data/ats19_frame.csv.gz (D162 frame, reproduced to the digit) | "
             "prereg data/overfit_capacity_prereg.md sha256 c0ec86df… | "
             "search space 600 cells, declared before scoring | "
             "null = 200 within-date permutations of (m_us, p_us)",
             fontsize=7.9, color=INK2, ha="left")

    fig.subplots_adjust(left=0.052, right=0.988, top=0.905, bottom=0.062)
    out = ROOT / "charts" / "overfit_capacity.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
