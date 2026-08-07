#!/usr/bin/env python3
"""D211 — rolling log loss, FOUR series, plotted as a DIFFERENCE FROM THE OPENER.

WHY NOT LEVELS. The first build plotted the four log-loss levels directly and was
unreadable: the series differ by ~0.01 nats while the rolling-100 path of any one
of them swings ~0.30, so the signal is 3% of the noise and all four lines
overlap. Differencing against a common baseline cancels the shared game-to-game
difficulty and leaves only the quantity of interest.

Baseline is the OPENING LINE, at zero. Below the line is better than the opener,
above it is worse. That makes the whole result readable at a glance: the closing
line sits below, the market-blind model sits ABOVE, and the offset construction
sits between the opener and the close.

charts/logloss_4way_2019_26.png

The prior chart carried two series (model vs market close) and could not show the
thing that actually matters: where the offset construction sits relative to BOTH
market prices and to the model it is built from.

Frame 2019-20..2025-26. Each source is converted to a probability with its OWN
walk-forward logistic scale, fitted on prior seasons only, so no source is
handicapped by another's calibration (D193's rule).

Palette validated (categorical, light surface): all six checks pass, worst
adjacent CVD dE 22.9. Every series is also DIRECTLY LABELLED at its right edge,
so identity never depends on colour alone.
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
from scipy.optimize import minimize_scalar                        # noqa: E402

C = {"open": "#eb6834", "close": "#8845c7",
     "blind": "#2a78d6", "offset": "#1a9e5f"}
LBL = {"open": "Opening line", "close": "Closing line",
       "blind": "Market-blind model", "offset": "Offset construction"}
INK, INK2, GRID, NAVY = "#0b0b0b", "#52514e", "#e7e6e2", "#1f3864"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 9,
})
LN2, ROLL, FROM = math.log(2), 100, "2019-20"


def nll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_scale(m, y):
    return float(minimize_scalar(
        lambda s: nll(1 / (1 + np.exp(-m / s)), y).mean(),
        bounds=(2, 25), method="bounded").x)


def main():
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
    f = f[f["season"] >= FROM].copy()
    f["game_date"] = pd.to_datetime(f["game_date"])
    f = f.sort_values(["season", "game_date", "game_id"]).reset_index(drop=True)
    f["y"] = (f["margin_actual"] > 0).astype(float)
    SRC = {"blind": "m_us_blind", "offset": "m_us",
           "open": "open_margin", "close": "close_margin"}
    f = f.dropna(subset=list(SRC.values()) + ["margin_actual"])
    seasons = sorted(f["season"].unique())

    # per-source per-game loss, each with its own walk-forward scale
    loss = {k: np.full(len(f), np.nan) for k in SRC}
    for i, s in enumerate(seasons):
        te = f["season"] == s
        tr = f["season"].isin(seasons[:i]) if i else te   # season 0 self-scales
        for k, col in SRC.items():
            sc = fit_scale(f.loc[tr, col].to_numpy(float),
                           f.loc[tr, "y"].to_numpy(float))
            p = 1 / (1 + np.exp(-f.loc[te, col].to_numpy(float) / sc))
            loss[k][te.to_numpy()] = nll(p, f.loc[te, "y"].to_numpy(float))

    fig, axes = plt.subplots(1, len(seasons), figsize=(19.0, 4.6), dpi=150,
                             sharey=True)
    R = 200          # wider window: we are now plotting a small difference
    for ax, s in zip(axes, seasons):
        m = (f["season"] == s).to_numpy()
        base = loss["open"][m]
        x = np.arange(R - 1, int(m.sum()))
        ax.axhline(0, color=C["open"], lw=1.6, zorder=5)
        for k in ("close", "blind", "offset"):
            d = loss[k][m] - base
            ax.plot(x, np.convolve(d, np.ones(R) / R, mode="valid"),
                    color=C[k], lw=1.5, label=LBL[k],
                    zorder=7 if k == "offset" else 4)
        pooled = {k: loss[k][m].mean() - base.mean() for k in SRC}
        ax.set_title(s, fontsize=11.5, pad=7, color=INK)
        # vertical, one line per series, so it cannot run into the next panel
        # (the horizontal version truncated and overflowed on every panel)
        for j, k in enumerate(("close", "offset", "blind")):
            ax.text(0.04, 0.035 + 0.062 * (2 - j),
                    f"{pooled[k]:+.4f}", transform=ax.transAxes, fontsize=7.0,
                    va="bottom", color=C[k], family="monospace",
                    fontweight="bold",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))
        ax.text(0.04, 0.035 + 0.062 * 3, "vs open", transform=ax.transAxes,
                fontsize=6.6, va="bottom", color=INK2, family="monospace",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))
        ax.set_xlabel("game # in season", fontsize=8.2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel(f"log loss vs the OPENING line\n"
                       f"(rolling {R}; below 0 = better than the opener)")

    # direct labels, de-collided, in the right margin of the last panel
    last = (f["season"] == seasons[-1]).to_numpy()
    bl = loss["open"][last]
    xr = int(last.sum()) - 1
    ends = []
    for k in ("close", "blind", "offset"):
        d = loss[k][last] - bl
        ends.append((np.convolve(d, np.ones(R) / R, mode="valid")[-1], k))
    ends.append((0.0, "open"))
    ends.sort(key=lambda z: -z[0])
    ylo, yhi = axes[-1].get_ylim()
    gap = 0.085 * (yhi - ylo)
    prev = None
    for yv, k in ends:
        yy = yv if prev is None else min(yv, prev - gap)
        prev = yy
        axes[-1].annotate(LBL[k], xy=(xr, yv), xytext=(xr * 1.06, yy),
                          color=C[k], fontsize=7.8, va="center",
                          fontweight="bold",
                          arrowprops=dict(arrowstyle="-", color=C[k], lw=0.6,
                                          alpha=0.6, shrinkA=0, shrinkB=1))
    axes[-1].set_xlim(0, xr * 1.42)

    tot = {k: loss[k].mean() for k in SRC}
    cap = ((tot["open"] - tot["offset"]) / (tot["open"] - tot["close"]))
    capb = ((tot["open"] - tot["blind"]) / (tot["open"] - tot["close"]))
    fig.suptitle("Rolling log loss, measured AGAINST THE OPENING LINE — where "
                 "each forecast sits relative to the price you can transact at"
                 "  ·  2019-20 … 2025-26",
                 fontsize=12.6, x=0.045, ha="left", y=0.975, color=NAVY,
                 fontweight="bold")
    fig.text(0.045, 0.928,
             f"The opening line is the zero line. BELOW it is better than the "
             f"opener, above it is worse. Levels differ by ~0.01 nats while each "
             f"series' own rolling path swings ~0.30, so the levels are plotted "
             f"as differences —\notherwise all four overlap and nothing is "
             f"legible.   Pooled vs open: close {tot['close']-tot['open']:+.5f}  "
             f"·  offset {tot['offset']-tot['open']:+.5f}  ·  blind "
             f"{tot['blind']-tot['open']:+.5f}.   The market-blind model is the "
             f"only series ABOVE the opener.",
             fontsize=8.6, color=INK2, ha="left", va="top")
    fig.subplots_adjust(left=0.062, right=0.965, top=0.775, bottom=0.115,
                        wspace=0.09)
    out = ROOT / "charts" / "logloss_4way_2019_26.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print("wrote", out)
    for k in ("open", "close", "blind", "offset"):
        print(f"  {LBL[k]:22} {tot[k]:.5f}")
    print(f"  capture: blind {capb:+.3f}   offset {cap:+.3f}")


if __name__ == "__main__":
    main()
