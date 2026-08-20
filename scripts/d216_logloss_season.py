#!/usr/bin/env python3
"""D216 — log loss by season, four series, LEVELS (not differences).

The owner's original figure plotted per-season means with the values labelled.
That is why it worked: at 7 points per series the reader reads numbers, not
pixels. My rolling-path versions failed because 8,000 points of a ~0.30 swing
hide a ~0.01 difference, which forced the difference-from-opener workaround.
Going back to per-season means removes the need for it — levels are legible
again.

Four series differentiated by line STYLE as well as colour (solid / dashed /
dotted / dash-dot) and by marker, so the chart survives greyscale printing and
colour-vision deficiency without relying on hue.

charts/logloss_season_4way.png
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

STYLE = {
    "open":   ("#eb6834", "-",  "o", "Opening line"),
    "close":  ("#8845c7", "--", "s", "Closing line"),
    "offset": ("#1a9e5f", "-.", "^", "Offset construction"),
    "blind":  ("#2a78d6", ":",  "D", "Market-blind model"),
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e7e6e2"
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

per = {k: [loss[k][(f["season"] == s).to_numpy()].mean() for s in seasons]
       for k in SRC}
fig, ax = plt.subplots(figsize=(10.6, 3.0), dpi=200)
x = np.arange(len(seasons))
for k in ("open", "close", "offset", "blind"):
    c, ls, mk, lab = STYLE[k]
    ax.plot(x, per[k], color=c, ls=ls, marker=mk, ms=3.6, lw=1.15, label=lab,
            zorder=5, markeredgewidth=0)
ax.set_xticks(x, seasons, fontsize=8)
ax.set_ylabel("log loss  (lower is better)", fontsize=8.4)
ax.legend(frameon=False, fontsize=7.8, ncol=4, loc="upper center",
          bbox_to_anchor=(0.5, 1.14), handlelength=3.0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
tot = {k: loss[k].mean() for k in SRC}
# pooled figures deliberately NOT annotated here: the table directly beneath the
# figure in the report carries them exactly, and the annotation collided with the
# 2025-26 closing-line marker.
fig.subplots_adjust(left=0.085, right=0.995, top=0.875, bottom=0.115)
out = ROOT / "charts" / "logloss_season_4way.png"
fig.savefig(out, dpi=200, facecolor="white")
print("wrote", out)
print(f"{'season':10}" + "".join(f"{STYLE[k][3][:9]:>11}" for k in
                                  ("open", "close", "offset", "blind")))
for i, s in enumerate(seasons):
    print(f"{s:10}" + "".join(f"{per[k][i]:11.4f}" for k in
                              ("open", "close", "offset", "blind")))
print(f"{'pooled':10}" + "".join(f"{tot[k]:11.4f}" for k in
                                 ("open", "close", "offset", "blind")))
