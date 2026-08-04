"""Two head-to-head status charts for the owner (2026-08-02).

  charts/status_logloss_h2h.png   model vs market: where we stand, by season and
                                  across the season, on the CERTIFIED D158 stack (honest availability)
  charts/status_trading_h2h.png   trading: ROI vs breakeven per execution policy,
                                  and CLV per rule against the D121 monthly bands

Palette + marks follow the house dataviz rules (model blue / market orange,
thin marks, recessive grid, direct labels, no dual axes).
"""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
LN2 = float(np.log(2.0))
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


def ll(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    charts = ROOT / "charts"
    rows = list(csv.DictReader(open(ROOT / "data" / "capstone_pergame.csv")))
    rows.sort(key=lambda r: (r["game_date"], r["game_id"]))
    seasons = sorted({r["season"] for r in rows})

    # ---------------------------------------------------------- CHART 1
    fig = plt.figure(figsize=(15, 6.2), dpi=150)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.28)

    # panel A: normalized gap by season (share of market skill we MISS)
    ax = fig.add_subplot(gs[0, 0])
    gaps, labs = [], []
    for s in seasons:
        rr = [r for r in rows if r["season"] == s]
        y = [float(r["y"]) for r in rr]
        us, mk = ll(y, [float(r["p_us"]) for r in rr]).mean(), ll(y, [float(r["p_mkt"]) for r in rr]).mean()
        gaps.append(100 * (us - mk) / (LN2 - mk))
        labs.append(s)
    y_all = [float(r["y"]) for r in rows]
    us_p = ll(y_all, [float(r["p_us"]) for r in rows]).mean()
    mk_p = ll(y_all, [float(r["p_mkt"]) for r in rows]).mean()
    pooled = 100 * (us_p - mk_p) / (LN2 - mk_p)
    bars = ax.barh(range(len(gaps)), gaps, color=C_MODEL, height=0.55)
    ax.axvline(pooled, color=INK2, ls="--", lw=1.2)
    # pooled marker labelled ABOVE the axis (D158: the pooled line moved to
    # 14.95% and collided with both the bar value labels and the x ticks)
    # place it in the row-gap the dashed line passes through, on a white
    # bbox — above the axis it hits the suptitle, at bar height it hits the
    # value labels and the x ticks (D158 label-collision pass)
    ax.text(pooled + 0.4, 1.5, f"pooled {pooled:.2f}%", ha="left", va="center",
            color=INK2, fontsize=9, zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9,
                      boxstyle="square,pad=0.15"))
    for i, (b, g) in enumerate(zip(bars, gaps)):
        # nudge the value label clear of the dashed pooled line
        off = 0.9 if abs(g - pooled) < 1.6 else 0.3
        ax.text(g + off, i, f"{g:.1f}%", va="center", fontsize=9, color=INK2,
                zorder=5, bbox=dict(facecolor="white", edgecolor="none",
                                    alpha=0.85, boxstyle="square,pad=0.12"))
    ax.set_yticks(range(len(labs)), labs, fontsize=9.5)
    ax.set_xlim(0, max(gaps) * 1.28)
    ax.invert_yaxis()
    ax.set_xlabel("% of the market's skill-above-coinflip that we MISS")
    ax.set_title("How far behind the closing line we are\n(lower = closer to the market)",
                 fontsize=11, pad=14)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # panel B: rolling head-to-head across all seasons
    ax = fig.add_subplot(gs[0, 1:])
    l_us = ll(y_all, [float(r["p_us"]) for r in rows])
    l_mk = ll(y_all, [float(r["p_mkt"]) for r in rows])
    w = 200
    k = np.ones(w) / w
    r_us, r_mk = np.convolve(l_us, k, "valid"), np.convolve(l_mk, k, "valid")
    ix = np.arange(w - 1, len(l_us))
    ax.plot(ix, r_us, color=C_MODEL, lw=1.5, label="Our model (certified)")
    ax.plot(ix, r_mk, color=C_MKT, lw=1.5, label="Market close")
    ax.fill_between(ix, r_us, r_mk, where=(r_us <= r_mk), color=C_GOOD,
                    alpha=0.30, lw=0, label="we beat the close")
    # season boundaries
    b = 0
    for s in seasons[:-1]:
        b += sum(1 for r in rows if r["season"] == s)
        ax.axvline(b, color=GRID, lw=1.1)
    b = 0
    for s in seasons:
        n = sum(1 for r in rows if r["season"] == s)
        ax.text(b + n / 2, ax.get_ylim()[1], s, ha="center", va="top",
                fontsize=8.5, color=INK2)
        b += n
    _lo, _hi = ax.get_ylim()
    ax.set_ylim(_lo, _hi + 0.06 * (_hi - _lo))
    beat = float((l_us < l_mk).mean()) * 100
    ax.set_xlabel(f"game index, all {len(seasons)} certified seasons (rolling {w})")
    ax.set_ylabel("log loss (lower = better)")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title(f"Head to head, game by game — we are better on {beat:.0f}% of individual "
                 f"games,\nbut worse on average: {us_p:.4f} vs {mk_p:.4f}",
                 fontsize=11, pad=14)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.suptitle("MODEL vs MARKET — prediction accuracy (certified stack D158, honest availability, n=6,148)",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(charts / "status_logloss_h2h.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------- CHART 2
    d = json.load(open(ROOT / "data" / "bo_lineshop.json"))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.2), dpi=150,
                                   gridspec_kw={"wspace": 0.26})

    # left: edge vs breakeven per execution policy (UNION of the frozen rules)
    pol = [("WORST2", "worst book"), ("B1", "one book"),
           ("BEST2", "best of 2"), ("BEST4", "best of 4*")]
    u = d["headline"]["UNION"]["scores"]
    edges = [100 * u[p]["edge_pp"] for p, _ in pol if p in u]
    names = [n for p, n in pol if p in u]
    cols = [C_BAD if e < 0 else C_GOOD for e in edges]
    bars = axL.bar(range(len(edges)), edges, color=cols, width=0.58)
    axL.axhline(0, color=INK2, lw=1.2)
    for i, e in enumerate(edges):
        # negative labels sit INSIDE the bar so they never collide with the
        # category tick underneath it
        # a near-zero bar has no inside to sit in, so its label would ride up
        # into the panel title — drop it BELOW the zero line instead (D158)
        if e > -0.4:
            axL.text(i, e - 0.10, f"{e:+.2f}pp", ha="center", fontsize=9.5,
                     va="top", color=INK2)
        else:
            axL.text(i, e + 0.12, f"{e:+.2f}pp", ha="center", fontsize=9.5,
                     va="bottom", color="white")
    axL.set_xticks(range(len(names)), names, fontsize=9.5)
    axL.set_ylabel("hit rate minus breakeven (pp)  —  above 0 = profitable")
    axL.set_title("TRADING at the OPEN: execution matters, but not enough\n"
                  "(frozen rules, union; *best-of-4 mixes vendors = upper bound)",
                  fontsize=11)
    for sp in ("top", "right"):
        axL.spines[sp].set_visible(False)

    # right: CLV per rule with the D121 monthly bands
    clv = d["clv"]
    order = ["R4_LOWT", "T20_D03_10", "T20_D03_10_W", "STAR_FAV_SHARPER", "UNION"]
    lab = {"R4_LOWT": "R4_LOWT\n(primary)", "T20_D03_10": "T20_D03_10",
           "T20_D03_10_W": "T20_D03_10_W", "STAR_FAV_SHARPER": "STAR_FAV",
           "UNION": "UNION"}
    xs = np.arange(len(order))
    # baseline is ONEBOOK (the average of a B1-only and a B2-only bettor), NOT
    # B1 — B1 happens to be the better of the two books, so using it as the
    # baseline understates the shopping gain.
    for j, (key, col, nm) in enumerate((("WORST2", C_BAD, "worst of 2"),
                                        ("ONEBOOK", C_MODEL, "one book (avg)"),
                                        ("BEST2", C_GOOD, "best of 2"))):
        vals = [clv[r][key]["clv"] for r in order]
        los = [clv[r][key]["lo"] for r in order]
        his = [clv[r][key]["hi"] for r in order]
        off = (j - 1) * 0.26
        axR.errorbar(xs + off, vals,
                     yerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)],
                     fmt="o", ms=7, color=col, lw=0, elinewidth=1.6,
                     capsize=3, label=nm)
    axR.axhline(0.0200, color=C_GOOD, ls=":", lw=1.4)
    axR.axhline(-0.0131, color=C_BAD, ls=":", lw=1.4)
    axR.axhline(0, color=INK2, lw=1.0)
    # reference-line labels ride over the leftmost error bars; a white bbox is
    # the only placement that survives every value of the underlying data
    _bb = dict(facecolor="white", edgecolor="none", alpha=0.9,
               boxstyle="square,pad=0.15")
    axR.text(-0.42, 0.0205, "GOOD month band (+0.020)", fontsize=8.5,
             color=C_GOOD, ha="left", va="bottom", zorder=6, bbox=_bb)
    axR.text(-0.42, -0.0126, "RED FLAG (-0.0131)", fontsize=8.5,
             color=C_BAD, ha="left", va="bottom", zorder=6, bbox=_bb)
    axR.set_xticks(xs, [lab[r] for r in order], fontsize=9)
    axR.set_ylabel("closing line value per bet")
    axR.legend(frameon=False, fontsize=9, loc="upper left")
    axR.set_title("CLV — the asset we actually own\n"
                  "every rule positive, and shopping 2 books lifts it ~49%",
                  fontsize=11)
    for sp in ("top", "right"):
        axR.spines[sp].set_visible(False)
    fig.suptitle("MODEL vs MARKET — trading (frozen rules, real opening prices)",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(charts / "status_trading_h2h.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"pooled norm gap {pooled:.2f}%  beat-rate {beat:.1f}%")
    print("union edge by policy:", dict(zip(names, [round(e, 3) for e in edges])))
    print("union CLV B1", clv["UNION"]["B1"]["clv"], "BEST2", clv["UNION"]["BEST2"]["clv"])


if __name__ == "__main__":
    main()
