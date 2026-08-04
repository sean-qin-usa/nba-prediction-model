"""charts/longshot_bias.png — the favourite-longshot answer in one picture.

  A  calibration: implied vs realised by probability bin, season-clustered CIs,
     with the line the bias would have to REACH to clear the vig, and our
     frozen rules' bet distribution shaded underneath.
  B  ROI of blindly backing each bin at the real closing moneyline, against
     breakeven.
  C  the summary slope under every devig convention — because the choice of
     devig IS part of what is being tested.

House palette + marks per scripts/make_status_charts.py.
Run: python scripts/lb_chart.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import nbapred.threads                                           # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                               # noqa: E402
import matplotlib                                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

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


def rule_bet_probs():
    """Where our FROZEN rules' bets actually sit on the price axis."""
    import bo_openbacktest as bo
    res = {}
    m = bo.build(bo.RT1, "p_full", "PRIMARY rt1 p_full 4-season", res)
    out = {}
    for when in ("close", "open"):
        p_side, dec, ok = bo.price_cols(m, when, "ML")
        masks, _, _ = bo.registry_masks(m, p_side, when)
        u = np.zeros(len(m), bool)
        for v in masks.values():
            u |= v
        out[when] = p_side[u & ok]
    return out


def main():
    lb = json.load(open(os.path.join(ROOT, "data", "lb_longshot.json")))
    ex = json.load(open(os.path.join(ROOT, "data", "lb_exploit.json")))
    cal = lb["calib"]["close|prop"]
    slopes = lb["slopes"]
    ov = lb["overround"]["close"]["mean"]
    bets = rule_bet_probs()

    x = np.array([r["implied"] for r in cal])
    err = np.array([r["err"] for r in cal]) * 100
    elo = np.array([r["err_lo"] for r in cal]) * 100
    ehi = np.array([r["err_hi"] for r in cal]) * 100
    n = np.array([r["n"] for r in cal])
    be = np.array([r["breakeven"] for r in cal])
    roi = np.array([r["roi"] for r in cal]) * 100
    rlo = np.array([r["roi_lo"] for r in cal]) * 100
    rhi = np.array([r["roi_hi"] for r in cal]) * 100
    need = (be - x) * 100           # calibration error needed to break even

    fig = plt.figure(figsize=(16.2, 6.0), dpi=150)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.05, 0.78], wspace=0.26)

    # ---------------------------------------------------------- PANEL A ----
    ax = fig.add_subplot(gs[0, 0])
    axb = ax.twinx()
    axb.hist(bets["close"], bins=np.arange(0.0, 1.001, 0.05),
             color=INK2, alpha=0.10, zorder=0)
    axb.set_ylim(0, max(np.histogram(bets["close"],
                                     bins=np.arange(0, 1.001, 0.05))[0]) * 3.4)
    axb.set_yticks([])
    axb.grid(False)
    for sp in ("top", "right", "left"):
        axb.spines[sp].set_visible(False)

    ax.axhline(0, color=INK2, lw=1.0, zorder=1)
    gx = np.linspace(0.0, 1.0, 200)
    ax.plot(gx, 100 * gx * (ov - 1.0), color=C_MKT, ls="--", lw=1.6, zorder=3,
            label="bias needed to CLEAR THE VIG  (= p x %.2f%% overround)"
                  % (100 * (ov - 1)))
    ax.plot([0.02, 0.98], [-3.2, 3.2], color=C_BAD, ls=":", lw=1.2, zorder=2,
            label="what a textbook favourite-longshot bias would look like")
    ax.errorbar(x, err, yerr=[err - elo, ehi - err], fmt="o", ms=4.5,
                color=C_MODEL, lw=0, elinewidth=1.3, capsize=2.5, zorder=4,
                label="realised - implied, 95% season-clustered")
    med = float(np.median(bets["close"]))
    ax.axvline(med, color=INK2, lw=0.9, ls=":", zorder=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-6.2, 6.2)
    ax.set_xlabel("implied probability of the side (proportional de-vig)")
    ax.set_ylabel("calibration error, percentage points")
    ax.set_title("The favourite-longshot bias is NOT THERE\n"
                 "19 seasons, 23,769 games, 47,538 sides, real closing "
                 "moneylines", fontsize=11)
    ax.legend(frameon=False, fontsize=8.4, loc="upper left")
    ax.text(0.50, -5.55, "our frozen rules' bets (shaded, n=1,466) sit at "
            "p = 0.50-0.95, median %.2f — 100%% favourite side" % med,
            fontsize=8.4, color=INK2, ha="center")
    ax.text(0.945, -3.05, "NOT ONE bin is\nsignificant", fontsize=8.4,
            color=C_MODEL, ha="right", va="center")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ---------------------------------------------------------- PANEL B ----
    ax = fig.add_subplot(gs[0, 1])
    bars = ax.bar(np.arange(len(roi)), roi, color=C_MODEL, width=0.72)
    for i, (b, v, nn) in enumerate(zip(bars, roi, n)):
        if v <= -7.0:                       # negative labels INSIDE the bar
            ax.text(i, v * 0.5, f"{v:.0f}", ha="center", va="center",
                    fontsize=7.6, color="white", rotation=90)
        elif v <= -3.5:
            ax.text(i, v * 0.55, f"{v:.0f}", ha="center", va="center",
                    fontsize=7.2, color="white")
        elif v < 0:
            ax.text(i, v - 0.9, f"{v:.0f}", ha="center", va="top",
                    fontsize=7.2, color=INK2)
        else:
            ax.text(i, v + 0.9, f"+{v:.0f}", ha="center", va="bottom",
                    fontsize=7.4, color=INK2)
    ax.axhline(0, color=C_MKT, lw=1.6)
    ax.text(len(roi) - 0.4, 1.6, "BREAKEVEN after vig", fontsize=8.6,
            color=C_MKT, ha="right")
    lab = [r["bin"].split(",")[0][1:] for r in cal]
    ax.set_xticks(np.arange(len(roi))[::2], lab[::2], fontsize=8)
    ax.set_ylim(-58, 14)
    ax.set_xlabel("implied probability bin (lower edge)")
    ax.set_ylabel("ROI of blindly backing that bin, %")
    ax.set_title("0 of 38 bins clear breakeven\n"
                 "the only real bias is the extreme-dog tail, and it is a "
                 "COST", fontsize=11)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ---------------------------------------------------------- PANEL C ----
    ax = fig.add_subplot(gs[0, 2])
    order = ["prop", "add", "shin", "or", "power", "goto"]
    nice = {"prop": "proportional", "add": "additive", "shin": "Shin",
            "or": "odds-ratio", "power": "power", "goto": "goto_conversion"}
    v = [slopes[f"close|{k}"]["lin_b"] for k in order]
    lo = [slopes[f"close|{k}"]["lin_lo"] for k in order]
    hi = [slopes[f"close|{k}"]["lin_hi"] for k in order]
    ys = np.arange(len(order))
    cols = [C_MODEL if k == "prop" else C_MKT for k in order]
    ax.axvline(0, color=INK2, lw=1.0)
    for i in range(len(order)):
        ax.plot([lo[i], hi[i]], [ys[i], ys[i]], color=cols[i], lw=1.6,
                solid_capstyle="butt")
        ax.plot([v[i]], [ys[i]], "o", ms=6, color=cols[i])
        ax.text(hi[i] + 0.004, ys[i], f"{v[i]:+.4f}", fontsize=8.2,
                va="center", color=INK2)
    ax.set_yticks(ys, [nice[k] for k in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(-0.115, 0.075)
    ax.set_xlabel("FLB slope  (>0 = longshot bias)")
    ax.set_title("The de-vig IS the test\n"
                 "every FLB-aware method OVER-corrects", fontsize=11)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.suptitle("FAVOURITE-LONGSHOT BIAS IN OUR NBA MARKET — real, tiny, "
                 "confined to the tails, and nowhere near the vig",
                 fontsize=13, y=1.005)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(ROOT, "charts", "longshot_bias.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    print("rule bets close n=%d  p range %.3f..%.3f  median %.3f"
          % (len(bets["close"]), bets["close"].min(), bets["close"].max(),
             np.median(bets["close"])))


if __name__ == "__main__":
    main()
