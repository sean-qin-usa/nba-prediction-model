#!/usr/bin/env python3
"""D188 — charts/data_coverage.png : what data actually exists, by season.

Owner: "make sure that it is clear on the github that we are working with
limited data until 2018."

One row per input feed, one column per season, cell shaded by measured coverage.
Sequential encoding (magnitude) so it is ONE hue light->dark, with an explicit
distinct 'none' state — never a rainbow. Every cell carries its number, so the
reading never depends on colour alone.

Read-only. Nothing ships.
"""
from __future__ import annotations

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

NAVY, INK, INK2, GRID = "#1f3864", "#0b0b0b", "#52514e", "#e7e6e2"
NONE_C = "#f2f1ee"
RED = "#c62828"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 10, "axes.titlecolor": INK,
})

ROWS = [
    ("injury", "Daily injury report\n(the availability leg)", True),
    ("darko", "DARKO player talent", False),
    ("odds_open", "Opening spread\n(single price)", False),
    ("books_pct", "Multi-book panel AT THE OPEN\n(best-price execution)", True),
]


def main():
    df = pd.read_csv(ROOT / "data" / "d188_coverage.csv")
    df = df.sort_values("season").reset_index(drop=True)
    # express observed books/game as a share of the 9-book ceiling
    df["books_pct"] = 100.0 * df["books_open"] / 9.0
    seasons = df["season"].tolist()
    n = len(seasons)

    fig, ax = plt.subplots(figsize=(15.0, 5.6), dpi=150)
    cmap = plt.get_cmap("Blues")

    for r, (key, label, critical) in enumerate(ROWS):
        y = len(ROWS) - 1 - r
        for j, s in enumerate(seasons):
            v = float(df.loc[j, key])
            if v <= 0.05:
                fc, tc, txt = NONE_C, RED, "none"
            else:
                fc = cmap(0.18 + 0.72 * min(v, 100) / 100)
                tc = "white" if v > 55 else INK
                txt = f"{v:.0f}%" if v >= 9.5 else f"{v:.1f}%"
            ax.add_patch(plt.Rectangle((j, y), 0.94, 0.88, facecolor=fc,
                                       edgecolor="white", lw=1.4, zorder=2))
            ax.text(j + 0.47, y + 0.44, txt, ha="center", va="center",
                    fontsize=7.6, color=tc,
                    fontweight="bold" if v <= 0.05 else "normal", zorder=3)

    ax.set_xlim(-0.05, n)
    ax.set_ylim(-0.62, len(ROWS))
    ax.set_xticks(np.arange(n) + 0.47)
    ax.set_xticklabels(seasons, rotation=45, ha="right", fontsize=8.4)
    ax.set_yticks([len(ROWS) - 1 - r + 0.44 for r in range(len(ROWS))])
    ax.set_yticklabels([lab for _, lab, _ in ROWS], fontsize=9.0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)

    # the boundary that matters
    i19 = seasons.index("2019-20")
    ax.axvline(i19 - 0.03, color=RED, lw=2.0, zorder=6)
    ax.text(i19 + 0.1, -0.34, "◀ first fully injury-covered season",
            color=RED, fontsize=9.0, ha="left", va="center", fontweight="bold")
    i18 = seasons.index("2018-19")
    ax.text(i18 + 0.47, len(ROWS) - 0.04, "partial\n(starts 2018-12-17)",
            color=RED, fontsize=7.4, ha="center", va="bottom")

    fig.suptitle("WHAT DATA ACTUALLY EXISTS — measured coverage of every model "
                 "input, by season", fontsize=13.4, x=0.012, ha="left", y=0.985,
                 color=NAVY, fontweight="bold")
    fig.text(0.012, 0.925,
             "The model has FOUR inputs. Three of them go back to 2007-08. The "
             "fourth — the daily injury report the availability leg is built on "
             "— does not exist at all before 2018-12-17.\n"
             "Eleven full seasons (2007-08 … 2017-18) therefore score a "
             "CRIPPLED variant of the model, not the shipped one. Any figure in "
             "this repository spanning those seasons blends two different "
             "systems.",
             fontsize=9.3, color=INK2, ha="left", va="top", linespacing=1.55)
    fig.subplots_adjust(left=0.145, right=0.995, top=0.775, bottom=0.145)
    out = ROOT / "charts" / "data_coverage.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print("wrote", out)
    print(df[["season", "injury", "darko", "odds_open", "books_pct"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
