#!/usr/bin/env python3
"""D212 — compact form of the 4-way log-loss comparison, for the PDF column.

The 7-panel version (D211) is right at full width but is reduced 2.75x to fit a
6.9-inch report column, which takes its 7pt annotations to ~2.5pt and makes it
decorative rather than informative. This is the same measurement as a per-season
bar chart: log loss vs the OPENING line, so the opener is the zero axis.

charts/logloss_compact_2019_26.png
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
INK, INK2, GRID, NAVY, ORANGE = "#0b0b0b", "#52514e", "#e7e6e2", "#1f3864", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 8.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
})

def nll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def fit(m, y):
    return float(minimize_scalar(lambda s: nll(1/(1+np.exp(-m/s)), y).mean(),
                                 bounds=(2, 25), method="bounded").x)

f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
f = f[f["season"] >= "2019-20"].copy()
f["y"] = (f["margin_actual"] > 0).astype(float)
SRC = {"blind": "m_us_blind", "offset": "m_us",
       "open": "open_margin", "close": "close_margin"}
f = f.dropna(subset=list(SRC.values()) + ["margin_actual"])
seasons = sorted(f["season"].unique())
loss = {k: np.full(len(f), np.nan) for k in SRC}
for i, s in enumerate(seasons):
    te = (f["season"] == s).to_numpy()
    tr = (f["season"].isin(seasons[:i])).to_numpy() if i else te
    for k, col in SRC.items():
        sc = fit(f[col].to_numpy(float)[tr], f["y"].to_numpy(float)[tr])
        loss[k][te] = nll(1/(1+np.exp(-f[col].to_numpy(float)[te]/sc)),
                          f["y"].to_numpy(float)[te])

fig, ax = plt.subplots(figsize=(9.6, 3.0), dpi=200)
x = np.arange(len(seasons)); w = 0.26
ax.axhline(0, color=ORANGE, lw=2.0, zorder=5)
for j, k in enumerate(("close", "offset", "blind")):
    v = [loss[k][(f["season"] == s).to_numpy()].mean()
         - loss["open"][(f["season"] == s).to_numpy()].mean() for s in seasons]
    ax.bar(x + (j - 1) * w, v, width=w * 0.92, color=C[k], label=LBL[k], zorder=3)
tot = {k: loss[k].mean() for k in SRC}
ax.set_xticks(x, seasons, fontsize=8)
ax.set_ylabel("log loss vs the OPENING line\n(below 0 = better than the opener)",
              fontsize=8)
ax.text(-0.55, 0.0012, "opening line", color=ORANGE, fontsize=7.4,
        va="bottom", fontweight="bold")
ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.subplots_adjust(left=0.105, right=0.99, top=0.96, bottom=0.14)
out = ROOT / "charts" / "logloss_compact_2019_26.png"
fig.savefig(out, dpi=200, facecolor="white")
print("wrote", out)
for k in ("open", "close", "offset", "blind"):
    print(f"  {k:8} {tot[k]:.5f}  vs open {tot[k]-tot['open']:+.5f}")
