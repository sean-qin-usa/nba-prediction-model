"""charts/bigplayer_ladder.png — the BIGPLAYER capstone in one look.

LEFT   the INFORMATION ladder T0..T5 as normalized gap vs the closing line,
       with the close marked as the target (0%) and the CLAIRVOYANT bound
       (perfect availability + minutes + talent) as a separate dashed
       reference that is NOT part of the buyable stack.
RIGHT  ROI by EXECUTION tier E0..E3 at model tier T0 and T5, breakeven marked.

The owner's question the figure has to answer at a glance: is MONEY or
INFORMATION the binding constraint?

House style per scripts/make_status_charts.py.  150 dpi.
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
C_T0, C_T5 = "#a9c8ec", "#2a78d6"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlecolor": INK,
})


def main():
    bat = json.load(open(ROOT / "data" / "bp_battery.json"))
    ex = json.load(open(ROOT / "data" / "bp_exec.json"))

    fig = plt.figure(figsize=(15.4, 6.6), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.06, 1.0], wspace=0.30)

    # ------------------------------------------------- LEFT: information ---
    ax = fig.add_subplot(gs[0, 0])
    tiers = ["T0", "T1", "T2", "T3", "T4", "T5"]
    names = ["T0  availability-BLIND floor",
             "T1  + 5PM injury report",
             "T2  + official inactives (T-30)",
             "T3  + purchased minutes (MAE 4)",
             "T4  + tracking, PRIOR season",
             "T5  + DARKO+EPM, prior day"]
    gap = [bat["gaps"][t]["POOLED"]["norm"] for t in tiers]
    inc = [None] + [bat["increments"][f"{a}->{b}"]
                    for a, b in zip(tiers, tiers[1:])]
    ypos = np.arange(len(tiers))[::-1]
    XMAX = max(gap) * 1.52
    ax.barh(ypos, gap, color=C_T5, height=0.56, zorder=3)
    for i, (yp, g) in enumerate(zip(ypos, gap)):
        ax.text(g + 0.28, yp, f"{g:.2f}%", va="center", fontsize=9.5,
                color=INK, zorder=4)
        if inc[i] is not None:
            e = inc[i]["est"]
            sig = "SIG" if inc[i]["clustered_sig"] else "ns"
            ax.text(XMAX - 0.25, yp, f"this rung buys {e:+.5f}  ({sig})",
                    va="center", ha="right", fontsize=8.8,
                    color=C_GOOD if inc[i]["clustered_sig"] else INK2, zorder=4)
    # the target and the unattainable reference
    ax.axvline(0.0, color=C_MKT, lw=2.0, zorder=5)
    ax.text(0.28, -0.72, "THE CLOSING LINE\n(target)", color=C_MKT,
            fontsize=9.2, va="center", ha="left", fontweight="bold")
    c3 = bat["gaps"]["C3"]["POOLED"]["norm"]
    ax.axvline(c3, color=INK2, ls="--", lw=1.4, zorder=5)
    ax.text(c3, -1.30, f"CLAIRVOYANT BOUND  {c3:.2f}%\n"
            "perfect availability + minutes + talent\nUNATTAINABLE — not for sale",
            color=INK2, fontsize=8.6, ha="center", va="center", zorder=7,
            bbox=dict(fc="white", ec="none", pad=1.5))
    c1 = bat["gaps"]["C1"]["POOLED"]["norm"]
    ax.axvline(c1, color=INK2, ls=":", lw=1.2, zorder=5)
    ax.text(c1 + 0.30, -0.60, f"D132 certified {c1:.2f}%\n(clairvoyant OUT sets)",
            color=INK2, fontsize=8.4, ha="left", va="center", zorder=7,
            bbox=dict(fc="white", ec="none", pad=1.5))
    ax.set_yticks(ypos, names, fontsize=9.5)
    ax.set_ylim(-1.95, len(tiers) - 0.30)
    ax.set_xlim(0, XMAX)
    ax.set_xlabel("normalized gap to the closing line   "
                  "(ll_us - ll_mkt) / (ln2 - ll_mkt),  0% = the close")
    ax.set_title("INFORMATION — everything money can buy closes 32% of the gap.\n"
                 f"T5 - T0 = +0.00604 season-clustered [+0.00507,+0.00876] SIG;\n"
                 f"residual {gap[-1]:.2f}% STILL BEHIND the close.",
                 fontsize=10.5, loc="left", pad=8)
    ax.grid(axis="y", visible=False)

    # --------------------------------------------------- RIGHT: execution --
    ax2 = fig.add_subplot(gs[0, 1])
    labs = ["E0\nour access\n1-2 retail books",
            "E1\n5-book shop\n(EXTRAPOLATED)",
            "E2\nexchange\n2% on net win",
            "E3\nE2 + middle\n(both sides)"]
    keys = ["E0", "E1_N5", "E2_c2"]
    vals, cis = {}, {}
    for mt in ("T0", "T5"):
        arm = ex["arms"][f"{mt}@open"]["UNION"]
        v = [arm[k]["roi"] * 100 for k in keys]
        c = [(arm[k]["clustered_lo"] * 100, arm[k]["clustered_hi"] * 100)
             for k in keys]
        mid = ex["E3"][f"{mt}/UNION"]["exchange c=2%"]
        v.append(mid["roi"] * 100)
        c.append((mid["clustered_lo"] * 100, mid["clustered_hi"] * 100))
        vals[mt], cis[mt] = v, c
    x = np.arange(len(labs))
    w = 0.36
    for off, mt, col in ((-w / 2, "T0", C_T0), (w / 2, "T5", C_T5)):
        v = np.array(vals[mt])
        lo = np.array([a for a, _ in cis[mt]])
        hi = np.array([b for _, b in cis[mt]])
        ax2.bar(x + off, v, width=w, color=col, zorder=3,
                label=f"model {mt}" + (" (availability-blind)" if mt == "T0"
                                       else " (fully equipped)"))
        ax2.errorbar(x + off, v, yerr=[v - lo, hi - v], fmt="none",
                     ecolor=INK2, elinewidth=1.0, capsize=3, zorder=4)
        for xi, vi, hh, ll_ in zip(x + off, v, hi, lo):
            top = max(vi, hh)
            ax2.text(xi, top + 0.30, f"{vi:+.2f}", ha="center", fontsize=8.8,
                     color=C_GOOD if vi > 0 else C_BAD, zorder=6,
                     bbox=dict(fc="white", ec="none", pad=1.0))
    ax2.axhline(0.0, color=C_MKT, lw=2.0, zorder=5)
    ax2.text(len(labs) - 0.52, 0.22, "BREAKEVEN", color=C_MKT, fontsize=9,
             fontweight="bold", va="bottom", ha="right", zorder=7,
             bbox=dict(fc="white", ec="none", pad=1.0))
    ax2.set_xticks(x, labs, fontsize=9)
    ax2.set_ylabel("ROI per unit staked, %   (registered rule UNION, bet at the OPEN)")
    ax2.set_title("EXECUTION — capital is worth +3.6pp of ROI, information +0.7pp.\n"
                  "The fully-equipped, fully-capitalised cell is +2.69%, and it is\n"
                  "NOT significant on the K-1 cluster-mean t bound [-5.16,+9.59].",
                  fontsize=10.5, loc="left", pad=8)
    ax2.legend(frameon=False, fontsize=9, loc="upper left")
    ax2.grid(axis="x", visible=False)
    ymin = min(min(vals["T0"]), min(vals["T5"]),
               min(a for a, _ in cis["T0"] + cis["T5"]))
    ymax = max(max(vals["T0"]), max(vals["T5"]),
               max(b for _, b in cis["T0"] + cis["T5"]))
    ax2.set_ylim(ymin - 1.6, ymax + 2.4)

    fig.suptitle("BIGPLAYER — a paid injury wire, bought minutes, a tracking feed, "
                 "a 5-book shop and an exchange account, on OUR model  "
                 "(2,889 games, 2023-24..2025-26p, K=3)",
                 fontsize=12.5, x=0.008, ha="left", y=0.985, color=INK)
    fig.text(0.007, 0.055,
             "Error bars = season-clustered bootstrap (K=3; GATE_POLICY_V2 §9.3 warns this is unreliable in BOTH directions at small K).  "
             "NO positive execution cell survives the K-1 cluster-mean t interval.",
             fontsize=8.0, color=INK2, ha="left")
    fig.text(0.007, 0.020,
             "E1 N=5 is D142's Gaussian EXTRAPOLATION from 2 real books — a CEILING, not a forecast.  The clairvoyant reference is "
             "UNATTAINABLE and is shown only to bound how much of the residual is information at all.",
             fontsize=8.0, color=INK2, ha="left")
    fig.subplots_adjust(left=0.165, right=0.988, top=0.815, bottom=0.165)
    out = ROOT / "charts" / "bigplayer_ladder.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
