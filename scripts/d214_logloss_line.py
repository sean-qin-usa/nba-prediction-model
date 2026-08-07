#!/usr/bin/env python3
"""D214 — the 4-way log-loss comparison as a single continuous LINE chart.

D211 (7 panels) is correct but illegible at report width; D212 replaced it with
bars, which the owner did not want. This is the line form that survives the
column: ONE panel, continuous across 2019-26, with dashed season dividers.

Still plotted as a DIFFERENCE FROM THE OPENING LINE (the zero axis) because the
four levels differ by ~0.01 nats against ~0.30 of rolling swing, and a levels
chart overlaps into one tangle. Below zero is better than the opener.

charts/logloss_line_2019_26.png
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import nbapred.threads; nbapred.threads.pin(1)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

C = {"close": "#8845c7", "blind": "#2a78d6", "offset": "#1a9e5f"}
LBL = {"close": "Closing line", "blind": "Market-blind model",
       "offset": "Offset construction"}
INK, INK2, GRID, ORANGE, BROWN = "#0b0b0b", "#52514e", "#e7e6e2", "#eb6834", "#8a6d3b"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 8.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
})
R = 400   # wide window: we are plotting a ~0.01 signal


def nll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit(m, y):
    return float(minimize_scalar(lambda s: nll(1/(1+np.exp(-m/s)), y).mean(),
                                 bounds=(2, 25), method="bounded").x)


f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
f = f[f["season"] >= "2019-20"].copy()
f["game_date"] = pd.to_datetime(f["game_date"])
f = f.sort_values(["season", "game_date", "game_id"]).reset_index(drop=True)
f["y"] = (f["margin_actual"] > 0).astype(float)
SRC = {"blind": "m_us_blind", "offset": "m_us",
       "open": "open_margin", "close": "close_margin"}
f = f.dropna(subset=list(SRC.values()) + ["margin_actual"]).reset_index(drop=True)
seasons = sorted(f["season"].unique())
loss = {k: np.full(len(f), np.nan) for k in SRC}
for i, s in enumerate(seasons):
    te = (f["season"] == s).to_numpy()
    tr = (f["season"].isin(seasons[:i])).to_numpy() if i else te
    for k, col in SRC.items():
        sc = fit(f[col].to_numpy(float)[tr], f["y"].to_numpy(float)[tr])
        loss[k][te] = nll(1/(1+np.exp(-f[col].to_numpy(float)[te]/sc)),
                          f["y"].to_numpy(float)[te])

fig, ax = plt.subplots(figsize=(9.6, 3.4), dpi=200)
base = loss["open"]
x = np.arange(R - 1, len(f))
ax.axhline(0, color=ORANGE, lw=2.0, zorder=6)
ends = []
for k in ("close", "offset", "blind"):
    d = np.convolve(loss[k] - base, np.ones(R) / R, mode="valid")
    ax.plot(x, d, color=C[k], lw=1.6, zorder=7 if k == "offset" else 5)
    ends.append((d[-1], k))
starts = []
seen = set()
for i, s in enumerate(f["season"]):
    if s not in seen:
        seen.add(s); starts.append((i, s))
ylo, yhi = ax.get_ylim()
for i, s in starts[1:]:
    ax.axvline(i, color=BROWN, lw=0.8, ls=(0, (4, 3)), zorder=3)
for i, s in starts:
    ax.text(i + 45, yhi - 0.045 * (yhi - ylo), s, color=BROWN, fontsize=7.0,
            ha="left", va="top")
ends.append((0.0, "open"))
ends.sort(key=lambda z: -z[0])
gap = 0.10 * (yhi - ylo)
prev = None
for yv, k in ends:
    yy = yv if prev is None else min(yv, prev - gap)
    prev = yy
    lab = "Opening line" if k == "open" else LBL[k]
    col = ORANGE if k == "open" else C[k]
    ax.annotate(lab, xy=(len(f) - 1, yv), xytext=(len(f) * 1.012, yy),
                color=col, fontsize=7.8, va="center", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=col, lw=0.6, alpha=0.6,
                                shrinkA=0, shrinkB=1))
ax.set_ylim(ylo, yhi)
ax.set_xlim(0, len(f) * 1.24)
ax.set_xlabel("game, in date order across the seven scored seasons", fontsize=8.2)
ax.set_ylabel(f"log loss vs the OPENING line\n(rolling {R}; below 0 is better)",
              fontsize=8.2)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.subplots_adjust(left=0.105, right=0.995, top=0.965, bottom=0.145)
out = ROOT / "charts" / "logloss_line_2019_26.png"
fig.savefig(out, dpi=200, facecolor="white")
print("wrote", out)
tot = {k: loss[k].mean() for k in SRC}
for k in ("open", "close", "offset", "blind"):
    print(f"  {k:8} {tot[k]:.5f}  vs open {tot[k]-tot['open']:+.5f}")
