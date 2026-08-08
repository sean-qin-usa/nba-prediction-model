#!/usr/bin/env python3
"""D189 — continuous log-loss charts on the CORRECTED frame, 2019-20 … 2025-26.

Owner: "update continuous log loss graphs from 2019" / "from 19-26".

The prior continuous charts covered either the certified 5-season corpus
(2021-22..2025-26) or all 19 seasons.  Neither is the frame this project now
reports on: D186 established that the daily injury report begins 2018-12-17, so
2019-20 is the FIRST FULLY COVERED SEASON and 2019-20..2025-26 (K=7, n=8,286) is
the only window in which the model runs as designed.

Source: data/k19_d171_t2_pergame.csv (the D171 re-certified 19-season run),
sliced to 2019-20+.  Reproduces the registered pooled gap 13.59% EXACTLY.

Outputs (both NEW files; nothing overwritten in place):
  charts/logloss_continuous_2019_26.png   rolling-100 log loss, one panel/season
  charts/frame_model_2019_26.png          per-season normalized gap on the frame

House style per scripts/make_charts_cert.py: model blue / market orange, thin
marks, recessive grid, no dual axes, 150 dpi.

Read-only.  Nothing ships.
"""
from __future__ import annotations

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
import pandas as pd                                               # noqa: E402

C_MODEL, C_MKT = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e7e6e2"
NAVY, BROWN, RED = "#1f3864", "#8a6d3b", "#c62828"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlecolor": INK,
})
LN2 = math.log(2)
ROLL = 100
COVID = {"2019-20", "2020-21"}


def ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    d = pd.read_csv(ROOT / "data" / "k19_d171_t2_pergame.csv")
    d = d[d["season"] >= "2019-20"].sort_values(["game_date", "game_id"])
    seasons = sorted(d["season"].unique())
    p_us = ll(d["p_us"], d["y"]).mean()
    p_mk = ll(d["p_mkt"], d["y"]).mean()
    gap = 100 * (p_us - p_mk) / (LN2 - p_mk)
    print(f"frame {seasons[0]}..{seasons[-1]}  K={len(seasons)}  n={len(d)}  "
          f"ll_us {p_us:.5f}  ll_mkt {p_mk:.5f}  gap {gap:.2f}%")

    # ------------------------------------------------ CHART 1: continuous
    fig, axes = plt.subplots(1, len(seasons), figsize=(21.0, 5.0), dpi=150,
                             sharey=True)
    for ax, s in zip(axes, seasons):
        g = d[d["season"] == s]
        lu = ll(g["p_us"], g["y"])
        lm = ll(g["p_mkt"], g["y"])
        x = np.arange(ROLL - 1, len(g))
        ru = np.convolve(lu, np.ones(ROLL) / ROLL, mode="valid")
        rm = np.convolve(lm, np.ones(ROLL) / ROLL, mode="valid")
        ax.plot(x, ru, color=C_MODEL, lw=1.5, label="Our model", zorder=5)
        ax.plot(x, rm, color=C_MKT, lw=1.5, label="Market close", zorder=4)
        u, m = lu.mean(), lm.mean()
        gp = 100 * (u - m) / (LN2 - m)
        ax.set_title(s, fontsize=12.4, pad=8,
                     color=RED if s in COVID else INK)
        ax.text(0.03, 0.975, f"LL {u:.4f} vs mkt {m:.4f}\nnorm gap {gp:.1f}%",
                transform=ax.transAxes, fontsize=8.6, va="top", color=INK2)
        if s in COVID:
            ax.text(0.03, 0.055, "COVID season\n(bubble / compressed)",
                    transform=ax.transAxes, fontsize=8.0, va="bottom",
                    color=RED, fontweight="bold")
        ax.set_xlabel("game # in season", fontsize=9)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel(f"log loss (rolling {ROLL})")
    # legend goes in the first NON-COVID panel: panel 0 already carries the
    # COVID annotation in its lower-left and the two collided.
    i_leg = next(i for i, s in enumerate(seasons) if s not in COVID)
    axes[i_leg].legend(frameon=False, fontsize=9, loc="lower left")

    fig.suptitle(f"Rolling log loss by season — THE CORRECTED FRAME, "
                 f"{seasons[0]} … {seasons[-1]}   |   pooled LL {p_us:.4f} vs "
                 f"market {p_mk:.4f}, normalized gap {gap:.2f}% of market skill "
                 f"(n={len(d):,})",
                 fontsize=12.6, y=0.975, color=INK)
    fig.text(0.5, 0.925,
             "Every season in which the daily injury report exists in full — the "
             "only window where the model runs as designed (D186). Lower is "
             "better; the market line is below ours in all seven.",
             fontsize=9.2, color=INK2, ha="center", va="top")
    fig.subplots_adjust(left=0.042, right=0.995, top=0.815, bottom=0.115,
                        wspace=0.10)
    out1 = ROOT / "charts" / "logloss_continuous_2019_26.png"
    fig.savefig(out1, dpi=150, facecolor="white")
    plt.close(fig)
    print("wrote", out1)

    # ------------------------------------------------ CHART 2: per-season gap
    gaps, ns = [], []
    for s in seasons:
        g = d[d["season"] == s]
        u = ll(g["p_us"], g["y"]).mean()
        m = ll(g["p_mkt"], g["y"]).mean()
        gaps.append(100 * (u - m) / (LN2 - m))
        ns.append(len(g))

    fig2, ax = plt.subplots(figsize=(12.6, 6.2), dpi=150)
    x = np.arange(len(seasons))
    cols = [RED if s in COVID else C_MODEL for s in seasons]
    ax.bar(x, gaps, width=0.58, color=cols, zorder=3)
    for xi, gp in zip(x, gaps):
        if abs(gp - gap) < 1.6:
            ax.text(xi, gp - 0.7, f"{gp:.2f}%", ha="center", va="top",
                    fontsize=9.6, color="white", fontweight="bold", zorder=6)
        else:
            ax.text(xi, gp + 0.45, f"{gp:.2f}%", ha="center", va="bottom",
                    fontsize=9.6, color=INK, fontweight="bold")
    ax.axhline(gap, color=BROWN, lw=1.6, ls=(0, (5, 2.5)), zorder=4)
    ax.text(len(seasons) - 0.35, gap, f"  pooled {gap:.2f}%\n  (n={len(d):,})",
            color=BROWN, fontsize=9.4, ha="left", va="center",
            fontweight="bold")
    ax.axhline(0, color=INK2, lw=1.1, zorder=4)
    ax.text(len(seasons) - 0.35, 0.4, "  0% = we\n  match the market",
            color=INK2, fontsize=8.8, ha="left", va="bottom")
    ax.set_xticks(x, seasons)
    ax.set_xlim(-0.62, len(seasons) + 0.55)
    ax.set_ylim(0, max(gaps) * 1.18)
    ax.set_ylabel("normalized gap behind the market\n"
                  "(% of market skill-above-coinflip we miss)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig2.suptitle("MODEL ACCURACY ON THE CORRECTED FRAME — 2019-20 … 2025-26, "
                  "every fully injury-covered season",
                  fontsize=13.2, x=0.055, ha="left", y=0.972, color=NAVY,
                  fontweight="bold")
    fig2.text(0.055, 0.928,
              "Lower is better; every bar is above zero, so the market wins all "
              "seven. Red bars are the COVID seasons (bubble / compressed "
              "calendars), kept in the frame\nbecause they are fully covered, "
              "but they are the two most extreme cells in either direction.",
              fontsize=9.2, color=INK2, ha="left", va="top", linespacing=1.5)
    fig2.subplots_adjust(left=0.105, right=0.985, top=0.815, bottom=0.09)
    out2 = ROOT / "charts" / "frame_model_2019_26.png"
    fig2.savefig(out2, dpi=150, facecolor="white")
    plt.close(fig2)
    print("wrote", out2)
    for s, gp, n in zip(seasons, gaps, ns):
        print(f"  {s}  n={n:5d}  gap {gp:6.2f}%")


if __name__ == "__main__":
    main()
