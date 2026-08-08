#!/usr/bin/env python3
"""charts/era_local.png — era-local vs global selection, held out WITHIN era.

(a) per era and pooled: held-out ROI for the era-local and global arms, each
    drawn ALONGSIDE its own permutation null (grey), pooled with its 5-dof CI
(b) the primary statistic: the paired EL-minus-GLOBAL delta against the
    distribution the IDENTICAL procedure manufactures on permuted predictions
(c) the mechanism: manufacturing capacity by SELECTION-WINDOW length, measured
    on the same six held-out seasons

House palette / marks per scripts/make_status_charts.py.  150 dpi.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
C_EL, C_EC, C_GL = "#2a78d6", "#8fbaea", "#eb6834"
C_NULL, C_GOOD, C_BAD = "#8a8886", "#1baf7a", "#d1495b"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e7e6e2"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlecolor": INK,
})
BB = dict(facecolor="white", edgecolor="none", alpha=0.92,
          boxstyle="square,pad=0.12")
TAG = "t5"          # firm default k=5 measured + haircut


def main():
    d = json.load(open(ROOT / "data" / "el_eralocal.json"))
    fig = plt.figure(figsize=(15.2, 10.6), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.02, 1.0],
                          width_ratios=[1.28, 1.0], hspace=0.40, wspace=0.20)

    # ------------------------------------------------------------- panel (a)
    ax = fig.add_subplot(gs[0, :])
    groups = [("E_OLD", "E_OLD  2007-08..2013-14\nheld out 2012-13, 2013-14"),
              ("E_MID", "E_MID  2014-15..2018-19\nheld out 2017-18, 2018-19"),
              ("E_MOD", "E_MOD  2021-22..2025-26\nheld out 2024-25, 2025-26"),
              ("POOLED", "POOLED\nall 6 held-out seasons")]
    arms = [("EL", "ERA-LOCAL  model + config", C_EL),
            ("EC", "era-local config only (V0)", C_EC),
            ("GF", "GLOBAL  all-history config (D166)", C_GL)]
    w = 0.24
    for ai, (arm, lab, col) in enumerate(arms):
        xs, vals, nl, nlo, nhi = [], [], [], [], []
        for gi, (g, _t) in enumerate(groups):
            if g == "POOLED":
                r = d["pooled_real"][arm][TAG]["roi"]
                n = d["pooled_null"][arm][TAG]
            else:
                r = d["real"][arm][g][TAG]["roi"]
                n = d["null_summary"][arm][g][TAG]
            xs.append(gi + (ai - 1) * w)
            vals.append(100 * r)
            nl.append(100 * n["mean"])
            nlo.append(100 * n["p05"])
            nhi.append(100 * n["p95"])
        ax.bar(xs, vals, width=w * 0.88, color=col, label=lab, zorder=3)
        # each arm's OWN null: grey diamond at the mean, thin p05-p95 whisker
        # nulls offset slightly right of their bar so the bar value labels
        # never sit on a whisker cap (collision pass 3)
        xn = np.array(xs) + w * 0.32
        ax.errorbar(xn, nl, yerr=[np.array(nl) - np.array(nlo),
                                  np.array(nhi) - np.array(nl)],
                    fmt="D", ms=4.5, color=C_NULL, lw=0, elinewidth=1.3,
                    capsize=2.5, zorder=5,
                    label="own permutation null (mean, p05-p95)" if ai == 0
                    else None)
        for x, v in zip(xs, vals):
            ax.text(x, v + (0.7 if v >= 0 else -0.7), f"{v:+.2f}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8.6, color=INK2, zorder=6, bbox=BB)
    # pooled 5-dof cluster-mean CI, drawn only where it is claimable (K=6)
    for ai, (arm, _l, col) in enumerate(arms):
        ci = d["pooled_real"][arm][TAG]["ci"]
        x = 3 + (ai - 1) * w
        ax.plot([x, x], [100 * ci["lo"], 100 * ci["hi"]], color=col, lw=1.8,
                solid_capstyle="butt", zorder=4)
        ax.plot([x - w * 0.22, x + w * 0.22], [100 * ci["lo"]] * 2, color=col,
                lw=1.8, zorder=4)
        ax.plot([x - w * 0.22, x + w * 0.22], [100 * ci["hi"]] * 2, color=col,
                lw=1.8, zorder=4)
    ax.axhline(0, color=INK2, lw=1.1, zorder=2)
    ax.set_xticks(range(4), [t for _g, t in groups], fontsize=9.2)
    ax.set_ylim(-21, 30)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("held-out ROI %  (firm default: k=5 measured books + haircut)")
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2,
              handlelength=1.4, columnspacing=1.4)
    ax.set_title("(a)  ERA-LOCAL vs GLOBAL, scored ONLY on seasons held out inside "
                 "the same era — every arm beside its own null\n"
                 "K=2 per era: the per-era bars carry NO claimable interval "
                 "(1 dof).  Whiskers on POOLED are the 5-dof cluster-mean t.",
                 fontsize=11, pad=12, loc="left")
    ax.text(-0.46, -18.8, "on E_OLD the era's selection block IS all prior "
            "history, so era-local and global are the SAME arm by construction "
            "(delta exactly 0.00)",
            fontsize=8.6, color=INK2, ha="left", va="center", zorder=6, bbox=BB)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ------------------------------------------------------------- panel (b)
    ax = fig.add_subplot(gs[1, 0])
    pj = d["paired"][TAG]["EL_minus_GF"]
    real = 100 * pj["ci"]["mean"]
    nu = pj["null"]
    # rebuild the null draw vector from the stored summary is not possible;
    # draw the distribution from its quantiles as a smooth reference instead
    rng = np.random.default_rng(0)
    ax.axvspan(100 * nu["p05"], 100 * nu["p95"], color=C_NULL, alpha=0.16,
               lw=0, label="null p05-p95 (200 permutation draws)")
    ax.axvline(100 * nu["mean"], color=C_NULL, lw=1.6, ls="--",
               label=f"null mean {100*nu['mean']:+.2f}")
    ax.axvline(100 * nu["p95"], color=C_BAD, lw=1.4, ls=":")
    ax.axvline(real, color=C_EL, lw=2.6, label=f"REAL {real:+.2f}")
    lo, hi = 100 * pj["ci"]["lo"], 100 * pj["ci"]["hi"]
    ax.annotate("", xy=(lo, 0.30), xytext=(hi, 0.30),
                arrowprops=dict(arrowstyle="<->", color=C_EL, lw=1.5))
    ax.text((lo + hi) / 2, 0.345, f"5-dof CI [{lo:+.2f}, {hi:+.2f}]  — spans zero",
            ha="center", fontsize=9, color=C_EL, bbox=BB)
    ax.axvline(0, color=INK2, lw=1.0)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([])
    ax.set_xlim(-12, 17)
    ax.text(100 * nu["p95"] - 0.45, 0.50,
            f"null p95 {100*nu['p95']:+.2f}  →", fontsize=9, color=C_BAD,
            ha="right", va="center", bbox=BB)
    ax.text(real + 0.40, 0.66,
            f"p = {pj['p_own_null']:.3f}\n{100*pj['p_own_null']:.0f}% of "
            f"NO-INFORMATION draws\nbeat it — the real gap does\nNOT clear its "
            f"own null p95", fontsize=9, color=C_EL, va="center", bbox=BB)
    ax.text(-11.4, 0.16, f"null max {100*nu['max']:+.2f}: on pure noise this\n"
            f"procedure can look {100*nu['max']:.1f} points better\nthan global",
            fontsize=8.6, color=INK2, va="center", bbox=BB)
    ax.set_xlabel("paired per-season delta, ERA-LOCAL minus GLOBAL (ROI points)")
    ax.legend(frameon=False, fontsize=8.8, loc="upper left")
    ax.set_title("(b)  THE PRIMARY STATISTIC — and what the identical procedure\n"
                 "manufactures from predictions carrying NO information",
                 fontsize=11, pad=10, loc="left")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    # ------------------------------------------------------------- panel (c)
    ax = fig.add_subplot(gs[1, 1])
    Ws = ["1", "2", "3", "5", "None"]
    lab = ["1", "2", "3", "5", "ALL\n(5-17)"]
    xs = np.arange(len(Ws))
    ndec, nheld, rheld = [], [], []
    for W in Ws:
        rn = d["null_summary"]["WLADDER"][W]
        rr = d["real"]["WLADDER"][W]
        ndec.append(100 * (rn["sel_roi"]["mean"] - rn["t1"]["mean"]))
        nheld.append(100 * rn[TAG]["mean"])
        rheld.append(100 * rr[TAG]["roi"])
    ax.bar(xs, ndec, width=0.56, color=C_NULL, alpha=0.32, zorder=2)
    for x, v in zip(xs, ndec):
        ax.text(x, v + 0.5, f"{v:+.1f}", ha="center", fontsize=9,
                color=INK2, zorder=6, bbox=BB)
    ax.plot(xs, rheld, "o-", color=C_EL, lw=1.8, ms=6, zorder=4)
    ax.plot(xs, nheld, "s--", color=C_NULL, lw=1.4, ms=5, zorder=4)
    for x, v, b in zip(xs, rheld, ndec):
        above = v > b
        ax.text(x, v + (0.95 if above else -1.05), f"{v:+.1f}", ha="center",
                va="bottom" if above else "top", fontsize=8.8, color=C_EL,
                zorder=6, bbox=BB)
    ax.axhline(0, color=INK2, lw=1.0, zorder=3)
    # DIRECT LABELS instead of a legend (the legend collided with all 3 series)
    ax.text(0.28, 21.2, "MANUFACTURING CAPACITY on pure noise\n"
            "(null in-window ROI minus null held-out ROI)",
            fontsize=8.8, color=INK2, ha="left", va="center", zorder=6, bbox=BB)
    ax.text(4.30, rheld[-1] + 2.4, "REAL\nheld-out ROI", fontsize=9, color=C_EL,
            ha="left", va="center", zorder=6)
    ax.text(4.30, nheld[-1] - 0.4, "null\nheld-out ROI", fontsize=9,
            color=C_NULL, ha="left", va="top", zorder=6)
    ax.set_xticks(xs, lab, fontsize=9.5)
    for t, c in zip(ax.get_xticklabels(), [INK2, INK2, C_EL, C_EL, C_GL]):
        t.set_color(c)
        t.set_fontweight("bold" if c != INK2 else "normal")
    ax.set_xlim(-0.55, 5.35)
    ax.set_xlabel("SELECTION-WINDOW LENGTH (seasons) — same 6 held-out seasons,"
                  " V0, 600 cells\nera-local windows are 3 and 5 (blue); "
                  "the global arm is ALL (orange)")
    ax.set_ylabel("ROI points")
    ax.set_ylim(-9, 25)
    ax.set_title("(c)  THE MECHANISM — less data per decision buys\n"
                 "MANUFACTURING, not edge (D165's ladder, re-measured here)",
                 fontsize=11, pad=10, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.suptitle("DOES AN ERA-LOCAL MODEL BEAT THE GLOBAL ONE ON SEASONS HELD OUT "
                 "INSIDE ITS OWN ERA?   NOT ESTABLISHED — and the model half of "
                 "it is worth exactly nothing",
                 fontsize=13.5, y=0.985)
    fig.text(0.5, 0.945,
             "era-local model choice (6 D168 rungs) buys -0.03 ROI points "
             "[-0.92,+0.87] ns;  every point of the apparent advantage is the "
             "shorter CONFIG window, which is not an era effect — a fixed "
             "5-season window that ignores era boundaries returns +10.13% on "
             "the same seasons",
             ha="center", fontsize=9.6, color=INK2)
    fig.savefig(ROOT / "charts" / "era_local.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote charts/era_local.png")


if __name__ == "__main__":
    main()
