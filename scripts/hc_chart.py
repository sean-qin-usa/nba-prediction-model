#!/usr/bin/env python3
"""charts/honest_clv.png — how much of our CLV was the availability leak?

House style from scripts/make_status_charts.py (model blue / market orange,
good green / bad red, recessive grid, direct labels, no dual axes), 150dpi.

Reads data/hc_honestclv.json (written by scripts/hc_honestclv.py).
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
BB = dict(facecolor="white", edgecolor="none", alpha=0.94,
          boxstyle="square,pad=0.15")

ORDER = ["R4_LOWT", "T20_D03_10", "T20_D03_10_W", "STAR_FAV_SHARPER", "UNION"]
LAB = {"R4_LOWT": "R4_LOWT", "T20_D03_10": "T20_D03_10",
       "T20_D03_10_W": "T20_D03_10_W", "STAR_FAV_SHARPER": "STAR_FAV",
       "UNION": "UNION"}


def main():
    d = json.load(open(ROOT / "data" / "hc_honestclv.json"))
    A = d["clv"]["ML_open"]
    B = d["clv"]["SP_open_policies"]
    dec = d["clv_decomposition"]
    bands = d["bands"]
    u = dec["ML@open|UNION"]

    fig = plt.figure(figsize=(17.0, 7.0), dpi=150)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.32, 0.95, 0.86], wspace=0.36)

    # ------------------------------------------------- panel A: per rule ---
    ax = fig.add_subplot(gs[0, 0])
    xs = np.arange(len(ORDER))
    for j, (key, col, nm) in enumerate(
            (("leaky", C_MKT, "LEAKY  (played-set availability oracle)"),
             ("honest", C_MODEL, "HONEST  (T2 — what October ships)"))):
        v = np.array([A[r][key]["mean"] for r in ORDER])
        lo = np.array([A[r][key]["tlo"] for r in ORDER])
        hi = np.array([A[r][key]["thi"] for r in ORDER])
        off = (j - 0.5) * 0.30
        ax.errorbar(xs + off, v, yerr=[v - lo, hi - v], fmt="o", ms=7.5,
                    color=col, lw=0, elinewidth=1.7, capsize=3, label=nm,
                    zorder=4)
    for i, r in enumerate(ORDER):
        hm, lm = A[r]["honest"]["mean"], A[r]["leaky"]["mean"]
        ax.annotate("", xy=(i + 0.15, hm), xytext=(i - 0.15, lm),
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9,
                                    alpha=0.55), zorder=3)
        ax.text(i + 0.20, hm - 0.0004, f"{100*(hm-lm)/lm:+.0f}%", ha="left",
                va="top", fontsize=8.5, color=INK2, zorder=6, bbox=BB)
    ax.axhline(0.0200, color=C_GOOD, ls=":", lw=1.4)
    ax.axhline(-0.0131, color=C_BAD, ls=":", lw=1.4)
    ax.axhline(0, color=INK2, lw=1.0)
    ax.text(4.42, 0.02045, "D121 GOOD  +0.0200", fontsize=8.5, color=C_GOOD,
            ha="right", va="bottom", zorder=6, bbox=BB)
    ax.text(4.42, -0.01265, "D121 RED FLAG  -0.0131", fontsize=8.5,
            color=C_BAD, ha="right", va="bottom", zorder=6, bbox=BB)
    ax.set_xticks(xs, [LAB[r] for r in ORDER], fontsize=8.5, rotation=14,
                  ha="right")
    ax.set_xlim(-0.55, 4.55)
    ax.set_ylim(-0.017, 0.0255)
    ax.set_ylabel("CLV per bet at the open   (p_close - p_open, our side)")
    ax.legend(frameon=False, fontsize=8.8, loc="lower left",
              bbox_to_anchor=(0.02, 0.09))
    ax.set_title("CLV per rule — HONEST vs LEAKY\n"
                 "real opening moneylines, 3 full-T2 seasons",
                 fontsize=10.5, pad=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # -------------------------------------- panel B: execution policies ----
    ax = fig.add_subplot(gs[0, 1])
    pols = ["WORST2", "ONEBOOK", "BEST2"]
    w = 0.36
    for j, (key, col, nm) in enumerate((("leaky", C_MKT, "LEAKY"),
                                        ("honest", C_MODEL, "HONEST"))):
        v = np.array([B["UNION"][p][key]["mean"] for p in pols])
        lo = np.array([B["UNION"][p][key]["lo"] for p in pols])
        hi = np.array([B["UNION"][p][key]["hi"] for p in pols])
        x = np.arange(len(pols)) + (j - 0.5) * w
        ax.bar(x, v, width=w * 0.90, color=col, alpha=0.90, label=nm, zorder=3)
        ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt="none", ecolor=INK2,
                    elinewidth=1.2, capsize=3, zorder=5)
        for xi, vi, hh in zip(x, v, hi):
            ax.text(xi, hh + 0.0011, f"{vi:+.4f}", ha="center", va="bottom",
                    fontsize=8.5, color=INK2, zorder=6)
    ax.axhline(0.0200, color=C_GOOD, ls=":", lw=1.4)
    ax.text(-0.48, 0.02045, "D121 GOOD  +0.0200", fontsize=8.5, color=C_GOOD,
            ha="left", va="bottom", zorder=6, bbox=BB)
    ax.set_xticks(np.arange(len(pols)),
                  ["worst of 2\nbooks", "one book\n(avg)", "best of 2\nbooks"],
                  fontsize=9)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, 0.0405)
    ax.set_ylabel("UNION CLV per bet at the open")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("D142's line shopping, re-priced honestly\n"
                 "spread frame, 4 seasons, 2-book panel",
                 fontsize=10.5, pad=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ------------------------------------------- panel C: the waterfall ----
    ax = fig.add_subplot(gs[0, 2])
    steps = [("REGISTERED\n(D155)", u["registered"], C_MKT),
             ("corpus\ndrift", u["frame_drift"], "#b9b7b1"),
             ("THE\nLEAK", u["leak"], C_BAD),
             ("HONEST\n(T2)", u["honest"], C_MODEL)]
    run = 0.0
    for i, (nm, val, col) in enumerate(steps):
        if i in (0, len(steps) - 1):
            ax.bar(i, val, width=0.60, color=col, zorder=3)
            ax.text(i, val + 0.0005, f"{val:+.5f}", ha="center", va="bottom",
                    fontsize=9, color=INK, zorder=6)
            run = val
        else:
            bot = run + val
            ax.bar(i, -val, bottom=bot, width=0.60, color=col, zorder=3)
            ax.text(i, bot - 0.0005, f"{val:+.5f}", ha="center", va="top",
                    fontsize=9, color=INK, zorder=6, bbox=BB)
            ax.plot([i - 0.30, i + 0.30], [run, run], color=INK2, lw=0.8,
                    ls="--", zorder=4)
            run = bot
    ax.axhline(0, color=INK2, lw=1.0)
    ax.set_xticks(range(len(steps)), [s[0] for s in steps], fontsize=8.8)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(0, 0.0205)
    ax.set_ylabel("UNION CLV per bet at the open")
    ax.set_title("Where the registered CLV went\n"
                 f"{100*u['leak']/u['total']:.0f}% leak / "
                 f"{100-100*u['leak']/u['total']:.0f}% corpus drift",
                 fontsize=10.5, pad=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.suptitle("DOES THE CLV SURVIVE THE D158 AVAILABILITY FIX?   YES — the "
                 "union keeps +0.0120 of CLV per bet at the open, 87% of the "
                 "same-corpus leaky level and 75% of the registered one",
                 fontsize=12.5, y=0.985)
    fig.text(0.5, 0.005,
             "LEAKY = data/capstone_pergame_oracle_ceiling.csv (played-set "
             "OUT sets, D158 §9); HONEST = data/capstone_pergame.csv "
             "(T2, report UNION inactives).  Same script, same corpus, same "
             "weekly refit — availability is the only difference.  "
             "Whiskers: panel A K-1 cluster-mean t at K=3, panels B/C "
             "season-clustered bootstrap.   D159, 2026-08-03.",
             ha="center", va="bottom", fontsize=8.2, color=INK2)
    fig.savefig(ROOT / "charts" / "honest_clv.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    hb = bands["ML|HONEST"]
    print("wrote", ROOT / "charts" / "honest_clv.png")
    print("union honest", A["UNION"]["honest"]["mean"],
          "leaky", A["UNION"]["leaky"]["mean"], "registered", u["registered"])
    print("honest union-centred ML band", hb["union_centred_red"],
          hb["union_centred_good"])


if __name__ == "__main__":
    main()
