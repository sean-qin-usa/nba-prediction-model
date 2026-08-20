#!/usr/bin/env python3
"""D172 — charts/coach_effects.png.  House style, 150 dpi.

Four panels, arranged as the argument runs:
  A  the coach-change event, RAW vs MATCHED — the whole apparent effect is
     mean reversion.
  B  what a coach demonstrably DOES move (rotation shape), and what it does
     not (pace, concentration), each against its permutation null.
  C  persistence — the D137 test.  The memory belongs to the team, and it
     does not travel with the man.
  D  ownership: the only survivor, with its levels and its matched placebo.
"""
from __future__ import annotations

import json
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

DATA = ROOT / "data"
CH = ROOT / "charts"
C_MODEL = "#2a78d6"
C_MKT = "#eb6834"
C_BEAT = "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e7e6e2"
C_NULL = "#9a9895"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 9.5, "axes.titlecolor": INK,
})

meas = json.loads((DATA / "d172_measure.json").read_text())
dec = json.loads((DATA / "d172_decompose.json").read_text())
nul = json.loads((DATA / "d172_nulls.json").read_text())
stz = json.loads((DATA / "d172_stress.json").read_text())

fig, axes = plt.subplots(2, 2, figsize=(15.6, 10.0), dpi=150)
(axA, axB), (axC, axD) = axes

# ------------------------------------------------------------------ PANEL A
did = dec["matched_did"]
labels = ["post − pre\n(raw)", "matched control\n(equally bad run,\nkept the coach)",
          "DIFF-IN-DIFF\n(the coach effect)"]
mk = did["vs MARKET (open)"]
vals = [mk["event_delta"], mk["ctrl_delta"], mk["did"]["mean"]]
errs = [None, None, (mk["did"]["mean"] - mk["did"]["lo"])]
cols = [C_MKT, C_NULL, C_BEAT if mk["did"]["p"] < 0.05 else INK2]
x = np.arange(3)
axA.bar(x, vals, 0.6, color=cols, edgecolor="white", linewidth=1.2)
axA.errorbar([2], [mk["did"]["mean"]], yerr=[[errs[2]], [errs[2]]],
             fmt="none", ecolor=INK, elinewidth=1.6, capsize=6, zorder=5)
for i, v in enumerate(vals):
    off = 0.10 if v >= 0 else -(errs[2] + 0.16)
    axA.text(i, v + off, f"{v:+.3f}", ha="center",
             va="bottom" if v >= 0 else "top", fontsize=11, color=INK,
             fontweight="bold")
axA.axhline(0, color=INK, lw=1.0)
axA.set_xticks(x)
axA.set_xticklabels(labels, fontsize=8.5)
axA.set_ylabel("change in team margin residual vs the MARKET (pts/game)")
axA.set_ylim(-2.0, 2.6)
axA.set_title("A  THE COACH-CHANGE EFFECT IS MEAN REVERSION\n"
              f"n={mk['did']['n']} changes, 19 seasons; DiD "
              f"{mk['did']['mean']:+.3f} "
              f"[{mk['did']['lo']:+.3f},{mk['did']['hi']:+.3f}] "
              f"p={mk['did']['p']:.3f} — NULL", fontsize=10, loc="left")
axA.text(0.5, -1.72, "teams that fired the coach and teams that did not\n"
                     "recovered by the SAME amount",
         ha="center", fontsize=8.5, color=INK2, style="italic")

# ------------------------------------------------------------------ PANEL B
bj = meas["behaviour_at_change"]
cn = nul["coach_spell_variance_null"]
names = ["n_used", "t5_turnover", "top8_share", "hhi", "poss"]
pretty = {"n_used": "players used\n(per game)",
          "t5_turnover": "top-5 churn\n(changes/game)",
          "top8_share": "top-8 minute\nshare",
          "hhi": "minutes\nHerfindahl",
          "poss": "pace\n(possessions)"}
# standardise each jump by that behaviour's own sd so they share an axis
sd = {k: v["sd_total"] for k, v in meas["variance_decomp"].items()}
y = np.arange(len(names))[::-1]
for i, n in zip(y, names):
    m = bj[n]["delta"] / sd[n]
    lo = bj[n]["lo"] / sd[n]
    hi = bj[n]["hi"] / sd[n]
    sig = bj[n]["p"] < 0.05
    axB.plot([lo, hi], [i, i], color=C_MODEL if sig else C_NULL, lw=3,
             solid_capstyle="round")
    axB.plot([m], [i], "o", color=C_MODEL if sig else C_NULL, ms=8, zorder=5)
    axB.text(max(hi, 0.0) + 0.018, i, f"{bj[n]['delta']:+.3f}  "
             f"({'p=%.4f' % bj[n]['p']})", va="center", fontsize=8.5,
             color=INK if sig else INK2)
axB.axvline(0, color=INK, lw=1.0)
axB.set_yticks(y)
axB.set_yticklabels([pretty[n] for n in names], fontsize=8.5)
axB.set_xlabel("jump at the coach change, in sd of the behaviour "
               "(season-clustered 95% CI)")
axB.set_xlim(-0.34, 0.60)
axB.set_title("B  A NEW COACH CHANGES THE ROTATION — AND NOTHING ELSE\n"
              "same roster, same season: the rotation shortens and the "
              "starting five churns;\npace and minute concentration do not move",
              fontsize=10, loc="left")

# ------------------------------------------------------------------ PANEL C
pd_ = dec["persistence_decomposed"]
tv = dec["coach_travels"]
rows = [
    ("same coach, same team", pd_["SAME coach, same team — vs our model"]["r"],
     pd_["SAME coach, same team — vs the MARKET"]["r"]),
    ("DIFFERENT coach, same team",
     pd_["DIFFERENT coach, same team — vs our model"]["r"],
     pd_["DIFFERENT coach, same team — vs the MARKET"]["r"]),
    ("same coach, DIFFERENT team\n(the only coach-specific test)",
     tv["vs our model"]["r"], tv["vs the MARKET"]["r"]),
]
yy = np.arange(len(rows))[::-1]
h = 0.32
for i, (lbl, rm, rk) in zip(yy, rows):
    axC.barh(i + h / 2, rm, h, color=C_MODEL, edgecolor="white")
    axC.barh(i - h / 2, rk, h, color=C_MKT, edgecolor="white")
    for val, yo in ((rm, i + h / 2), (rk, i - h / 2)):
        # a near-zero bar has no room on its own side: park the label right
        pos = val >= 0
        axC.text(val + (0.014 if pos else -0.014), yo, f"{val:+.3f}",
                 va="center", ha="left" if pos else "right", fontsize=8.5,
                 color=INK)
axC.axvline(0, color=INK, lw=1.0)
axC.set_yticks(yy)
axC.set_yticklabels([r[0] for r in rows], fontsize=8.5)
axC.set_xlabel("lag-1 correlation of the season effect")
axC.set_xlim(-0.26, 0.56)
axC.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_MODEL),
                    plt.Rectangle((0, 0), 1, 1, color=C_MKT)],
           labels=["vs OUR MODEL (coach-blind)", "vs the MARKET"],
           loc="lower right", fontsize=8.5, framealpha=1)
axC.set_title("C  PERSISTENCE BELONGS TO THE TEAM, NOT THE MAN  (D137's test)\n"
              "vs the market the effect does NOT travel with the man:\n"
              f"r={tv['vs the MARKET']['r']:+.4f} "
              f"(n={tv['vs the MARKET']['n']} coach-changes-team pairs)",
              fontsize=10, loc="left")

# ------------------------------------------------------------------ PANEL D
lv = stz["levels"]["vs the MARKET"]
pl = stz["placebo_did"]["vs the MARKET"]
labs = ["season BEFORE\nthe sale", "first FULL season\nunder the new owner",
        "matched placebo\nDIFF-IN-DIFF"]
vs = [lv["pre"]["mean"], lv["post"]["mean"], pl["mean"]]
es = [(lv["pre"]["mean"] - lv["pre"]["lo"]),
      (lv["post"]["mean"] - lv["post"]["lo"]),
      (pl["mean"] - pl["lo"])]
ps = [lv["pre"]["p"], lv["post"]["p"], pl["p"]]
cc = [C_NULL if p >= 0.05 else C_MKT for p in ps]
xx = np.arange(3)
axD.bar(xx, vs, 0.6, color=cc, edgecolor="white", linewidth=1.2)
axD.errorbar(xx, vs, yerr=[es, es], fmt="none", ecolor=INK, elinewidth=1.6,
             capsize=6, zorder=5)
for i, (v, p) in enumerate(zip(vs, ps)):
    axD.text(i, v - es[i] - 0.13 if v < 0 else v + es[i] + 0.06,
             f"{v:+.3f}\np={p:.4f}", ha="center",
             va="top" if v < 0 else "bottom", fontsize=9, color=INK,
             fontweight="bold")
axD.axhline(0, color=INK, lw=1.0)
axD.set_xticks(xx)
axD.set_xticklabels(labs, fontsize=8.5)
axD.set_ylabel("team margin residual vs the MARKET (pts/game)")
axD.set_ylim(-2.9, 2.0)
axD.set_title("D  OWNERSHIP — THE ONLY SURVIVOR, AND ITS DATA IS HAND-BUILT\n"
              f"n={lv['post']['n']} control sales, 19 seasons; survives "
              "BH at m=31.\nNOT GATED — the dates are hand-curated, not "
              "sourced", fontsize=10, loc="left")

fig.suptitle("D172  DO COACH AND OWNERSHIP CARRY SIGNAL?  "
             "coach: NO — a confounded shadow of the roster.  "
             "ownership: ONE unverified candidate.",
             fontsize=12.5, y=0.985, color=INK)
fig.text(0.5, 0.026,
         "1996-97..2025-26 corpus | coach panel 1,013 coach-team-seasons / "
         "191 coaches / 121 in-season changes, from Basketball-Reference "
         "(nba_api's coach feed is NOT point-in-time and was rejected)",
         ha="center", fontsize=8, color=INK2)
fig.text(0.5, 0.008,
         "residual frame 19 seasons / 22,742 games | season-clustered 95% CI "
         "throughout | Benjamini-Hochberg FDR 0.05 over a family of m=31",
         ha="center", fontsize=8, color=INK2)
fig.tight_layout(rect=[0, 0.050, 1, 0.952])
out = CH / "coach_effects.png"
fig.savefig(out, dpi=150)
print(f"WROTE {out}")
