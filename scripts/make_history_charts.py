#!/usr/bin/env python3
"""D153 charts — the historical out-of-sample picture.

House style follows scripts/make_charts_cert.py and make_status_charts.py:
model blue #2a78d6 / market orange #eb6834, thin marks, recessive grid, direct
labels, no dual axes, negative bar labels INSIDE the bar.  150 dpi into charts/.

  charts/history_logloss_by_season.png  model vs market LL, every scorable
        season, normalized gap annotated, certified corpus demarcated
  charts/history_normalized_gap.png     normalized gap as ONE series over 15
        years, era bands shaded and labelled
  charts/history_feature_by_era.png     per-shipped-term x per-era effect
        matrix with sign and significance

  python scripts/make_history_charts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

C_MODEL = "#2a78d6"
C_MKT = "#eb6834"
C_BEAT = "#1baf7a"
C_BAD = "#c2413a"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8781"
GRID = "#e7e6e2"
BAND = "#f2f1ed"
CERT_BAND = "#dfeafa"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlecolor": INK,
})

ERA_LABEL = {"E-3": "E-3\npre-lockout CBA", "E-2": "E-2\npost-lockout,\npre-3PT boom",
             "E-1": "E-1\n3PT ramp", "E0": "E0\npre-COVID", "E2": "E2\nno crowd",
             "E3": "E3\nre-entry", "E4": "E4\npost-COVID", "E5": "E5\nPPP+IST+CBA",
             "E6": "E6\napron"}
ERA_ORDER = ["E-3", "E-2", "E-1", "E0", "E1", "E2", "E3", "E4", "E5", "E6"]


def load():
    return json.load(open(ROOT / "data" / "history_analysis.json"))


# ---------------------------------------------------------------- chart 1
def chart_logloss(res):
    tab = res["per_season"]
    seasons = sorted(tab)
    x = np.arange(len(seasons))
    us = np.array([tab[s]["ll_us"] for s in seasons])
    mk = np.array([tab[s]["ll_mkt"] for s in seasons])
    ng = np.array([tab[s]["norm_gap_pct"] for s in seasons])
    cert = np.array([tab[s]["in_cert_corpus"] for s in seasons])
    strat = np.array([tab[s]["stratum"] for s in seasons])

    fig, ax = plt.subplots(figsize=(14.5, 7.0), dpi=150)
    # certified-corpus demarcation
    ci = np.where(cert)[0]
    if len(ci):
        ax.axvspan(ci[0] - 0.5, ci[-1] + 0.5, color=CERT_BAND, zorder=0)
        ax.text((ci[0] + ci[-1]) / 2, 0.985,
                "CERTIFIED EVAL CORPUS (D171)\nevery gate in the campaign was "
                "denominated here",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8.8, color="#2f5f96", linespacing=1.35)
    hi = np.where(~cert)[0]
    if len(hi):
        ax.text(hi[: len(hi) // 2 + 1].mean(), 0.985,
                "SCORED AT THE BEST TIER EACH SEASON CAN REACH (D171) — "
                "never touched ANY decision",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8.8, color=INK2)

    ax.fill_between(x, us, mk, color=C_MODEL, alpha=0.10, linewidth=0, zorder=1)
    ax.plot(x, mk, color=C_MKT, lw=1.7, zorder=3)
    ax.plot(x, us, color=C_MODEL, lw=1.7, zorder=3)
    for i, s in enumerate(seasons):
        m = "o" if not strat[i] else "s"
        fc = "white" if strat[i] else None
        ax.plot([x[i]], [mk[i]], m, color=C_MKT, markersize=5.5, zorder=4,
                markerfacecolor=fc if strat[i] else C_MKT, markeredgewidth=1.3)
        ax.plot([x[i]], [us[i]], m, color=C_MODEL, markersize=5.5, zorder=4,
                markerfacecolor=fc if strat[i] else C_MODEL, markeredgewidth=1.3)
        ax.annotate(f"{ng[i]:.1f}%", (x[i], max(us[i], mk[i])),
                    textcoords="offset points", xytext=(0, 13), ha="center",
                    fontsize=8.6,
                    color=(C_BAD if ng[i] > 15 else (C_BEAT if ng[i] < 8 else INK2)),
                    fontweight="bold" if (ng[i] > 20 or ng[i] < 5) else "normal")
    # direct labels instead of a legend box
    ax.annotate("our model", (x[-1], us[-1]), textcoords="offset points",
                xytext=(10, -2), color=C_MODEL, fontsize=10.5, va="center",
                fontweight="bold")
    ax.annotate("market close", (x[-1], mk[-1]), textcoords="offset points",
                xytext=(10, 0), color=C_MKT, fontsize=10.5, va="center",
                fontweight="bold")
    ax.set_xticks(x, seasons, rotation=45, ha="right", fontsize=9.2)
    for i, s in enumerate(seasons):
        if strat[i]:
            ax.get_xticklabels()[i].set_color(INK3)
    ax.set_xlim(-0.7, len(seasons) - 0.15)
    ax.set_ylabel("season log loss (lower = better)")
    ax.set_ylim(min(us.min(), mk.min()) - 0.008, max(us.max(), mk.max()) + 0.020)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    p = res["pooled"]
    sub = ("% above each season = normalized gap (ll_us − ll_mkt)/(ln2 − ll_mkt):"
           " the share of the market's own skill we still give away\n"
           "hollow squares = separate strata (2011-12 lockout, 2019-20 E0+bubble,"
           " 2020-21 no-crowd) — plotted, never pooled")
    ax.set_title(
        "The CERTIFIED D171 stack scored on every scorable season, each at the "
        "best availability tier it can reach — "
        f"{len(seasons)} seasons, {sum(tab[s]['n'] for s in seasons):,} games\n"
        + sub, fontsize=10.6, linespacing=1.55)
    txt = []
    for k in ("certified_5", "historical_new", "all_poolable"):
        if k in p:
            v = p[k]
            txt.append(f"{v['label']}: n={v['n']:,}  LL {v['ll_us']:.5f} vs "
                       f"mkt {v['ll_mkt']:.5f}  raw {v['raw_gap']:+.5f}  "
                       f"norm {v['norm_gap_pct']:+.2f}%")
    ax.text(0.005, -0.235, "\n".join(txt), transform=ax.transAxes,
            fontsize=9.0, color=INK2, va="top", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    out = ROOT / "charts" / "history_logloss_by_season.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- chart 2
def chart_normgap(res):
    tab = res["per_season"]
    seasons = sorted(tab)
    x = np.arange(len(seasons))
    ng = np.array([tab[s]["norm_gap_pct"] for s in seasons])
    era = [tab[s]["era"] for s in seasons]
    cert = np.array([tab[s]["in_cert_corpus"] for s in seasons])
    strat = np.array([tab[s]["stratum"] for s in seasons])

    fig, ax = plt.subplots(figsize=(14.5, 6.6), dpi=150)
    # era bands
    i = 0
    shade = True
    narrow = 0          # D171: single-season era bands (E3/E4/E5) are narrower
    while i < len(seasons):     # than their own labels — stagger them vertically
        j = i                   # instead of letting "post-COVID" and "PPP+IST+CBA"
        while j + 1 < len(seasons) and era[j + 1] == era[i]:   # run together.
            j += 1
        if shade:
            ax.axvspan(i - 0.5, j + 0.5, color=BAND, zorder=0)
        if j - i < 1:
            ytxt = 0.995 if narrow % 2 == 0 else 0.905
            narrow += 1
        else:
            ytxt = 0.995
        ax.text((i + j) / 2, ytxt, ERA_LABEL.get(era[i], era[i]),
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8.4, color=INK3, linespacing=1.25)
        ax.axvline(j + 0.5, color=GRID, lw=1.0, zorder=0)
        shade = not shade
        i = j + 1
    ax.plot(x, ng, color=C_MODEL, lw=1.8, zorder=3)
    for i in range(len(seasons)):
        if strat[i]:
            ax.plot([x[i]], [ng[i]], "s", color=C_MODEL, markersize=6.5,
                    markerfacecolor="white", markeredgewidth=1.5, zorder=4)
        else:
            ax.plot([x[i]], [ng[i]], "o", color=C_MODEL, markersize=6.5, zorder=4)
        # label placement: local minima below, local maxima above, and on a
        # monotone run nudge into the empty quadrant so the label never lands
        # on the line
        pv = ng[i - 1] if i else ng[i]
        nv = ng[i + 1] if i + 1 < len(ng) else ng[i]
        if ng[i] <= pv and ng[i] <= nv:
            dx, dy, va = 0, -10.5, "top"
        elif ng[i] >= pv and ng[i] >= nv:
            dx, dy, va = 0, 8.5, "bottom"
        elif pv > ng[i] > nv:            # descending
            dx, dy, va = 15, 6.0, "bottom"
        else:                            # ascending
            dx, dy, va = -15, 6.0, "bottom"
        ax.annotate(f"{ng[i]:.1f}", (x[i], ng[i]), textcoords="offset points",
                    xytext=(dx, dy), ha="center", va=va, fontsize=8.7,
                    color=INK2)
    ci = np.where(cert)[0]
    if len(ci):
        ax.axvspan(ci[0] - 0.5, ci[-1] + 0.5, color=CERT_BAND, alpha=0.55,
                   zorder=1)
    pooled = res["pooled"]
    for k, c, lab in (("all_poolable", INK3, "pooled, all poolable seasons"),
                      ("certified_5", "#2f5f96", "pooled, certified corpus")):
        if k in pooled:
            v = pooled[k]["norm_gap_pct"]
            ax.axhline(v, color=c, lw=1.1, ls=":", zorder=2)
            ax.annotate(f"{lab}  {v:.2f}%", (-0.4, v),
                        textcoords="offset points", xytext=(0, 4), ha="left",
                        fontsize=8.6, color=c)
    ax.set_xticks(x, seasons, rotation=45, ha="right", fontsize=9.2)
    ax.set_xlim(-0.5, len(seasons) - 0.5)
    # D171: the gap is no longer non-negative — 2008-09 at T2 BEATS the market
    # (-2.01%), and a floor of 0 silently clipped the single most important
    # point on the chart. Floor below the minimum and draw the zero line.
    lo = min(0.0, ng.min())
    ax.set_ylim(lo - 0.10 * (max(ng) - lo), max(ng) * 1.28)
    if ng.min() < 0:
        ax.axhline(0.0, color=C_BEAT, lw=1.2, zorder=2)
        ax.annotate("we BEAT the market below this line", (len(seasons) - 0.6, 0.0),
                    textcoords="offset points", xytext=(0, -13), ha="right",
                    fontsize=8.8, color=C_BEAT, fontweight="bold")
    ax.set_ylabel("normalized gap: share of the market's skill we still give away")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(
        "Are we getting closer to the market across 19 years?  CERTIFIED D171, "
        "each season at the best availability tier it can reach\n"
        "normalized gap = (ll_us − ll_mkt)/(ln2 − ll_mkt), lower = better;  "
        "era bands from docs/ERAS.md;  blue band = the certified corpus;  "
        "hollow squares = strata, plotted but never pooled",
        fontsize=10.8, linespacing=1.55)
    fig.tight_layout()
    out = ROOT / "charts" / "history_normalized_gap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- chart 3
def chart_feature_by_era(res):
    ab = res["ablation"]
    arms = [a for a in ("no_sched", "no_carry", "no_tank", "no_bridge",
                        "no_ff", "no_comp", "add_late") if a in ab]
    names = {"no_sched": "D46 schedule layer", "no_carry": "D62 carry",
             "no_tank": "D73 tank term", "no_bridge": "D91 October bridge",
             "no_ff": "D21 four-factors leg", "no_comp": "D19 composition leg",
             "add_late": "D90 late-state (add back)"}
    eras = [e for e in ERA_ORDER
            if any(e in ab[a]["ALL"]["per_era"] for a in arms)]
    M = np.full((len(arms), len(eras)), np.nan)
    SIG = np.zeros_like(M, dtype=bool)
    N = np.zeros_like(M)
    for i, a in enumerate(arms):
        pe = ab[a]["ALL"]["per_era"]
        for j, e in enumerate(eras):
            if e in pe:
                M[i, j] = pe[e]["est"] * 1000.0     # millinats
                SIG[i, j] = pe[e]["sig"]
                N[i, j] = pe[e]["n"]
    lim = np.nanmax(np.abs(M)) or 1.0
    fig, ax = plt.subplots(figsize=(14.6, 7.4), dpi=150)
    ax.grid(False)
    im = ax.imshow(M, cmap="PuOr_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(arms)):
        for j in range(len(eras)):
            if np.isnan(M[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8.5,
                        color=INK3)
                continue
            v = M[i, j]
            col = "white" if abs(v) > 0.62 * lim else INK
            ax.text(j, i, f"{v:+.2f}" + ("*" if SIG[i, j] else ""),
                    ha="center", va="center", fontsize=9.6, color=col,
                    fontweight="bold" if SIG[i, j] else "normal")
    short = {"E-3": "pre-lockout", "E-2": "post-lockout\npre-3PT boom",
             "E-1": "3PT ramp", "E0": "pre-COVID", "E2": "no crowd",
             "E3": "re-entry", "E4": "post-COVID", "E5": "PPP+IST+CBA",
             "E6": "apron"}
    xt = []
    for j, e in enumerate(eras):
        ss = ab[arms[0]]["ALL"]["per_era"].get(e, {}).get("seasons", [])
        yrs = f"{ss[0]}..{ss[-1]}" if len(ss) > 1 else (ss[0] if ss else "")
        xt.append(f"{e}  {short.get(e, '')}\n{yrs}\nn={int(N[0, j]):,}")
    ax.set_xticks(np.arange(len(eras)), xt, fontsize=8.5, linespacing=1.4)
    ax.set_yticks(np.arange(len(arms)), [names[a] for a in arms], fontsize=10)
    ax.set_xticks(np.arange(-0.5, len(eras), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(arms), 1), minor=True)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="minor", color="white", lw=2.0)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.045,
                     pad=0.20, aspect=55)
    cb.set_label("effect on log loss, millinats per game  "
                 "(+ = the term HELPS, − = the term HURTS)", fontsize=9)
    cb.outline.set_visible(False)
    # per-row verdict, direct-labelled to the right of the matrix
    for i, a in enumerate(arms):
        c = ab[a]["ALL"]["classification"]
        colr = {"ERA-STABLE": C_BEAT, "ERA-CONDITIONAL": "#b8860b",
                "ERA-SPECIFIC": C_BAD}.get(c["verdict"], INK2)
        lab = c["verdict"] + (" — SIGN FLIP" if c["significant_sign_flip"] else "")
        ax.annotate(lab, (len(eras) - 0.42, i), xycoords="data",
                    textcoords="offset points", xytext=(6, 0), va="center",
                    fontsize=8.9, color=colr, fontweight="bold",
                    annotation_clip=False)
    ax.set_title(
        "Would any feature design change over different seasons?  "
        "Per-term effect by era, all 11 poolable seasons\n"
        "* = 95% paired-bootstrap CI excludes zero;  verdicts per "
        "GATE_POLICY_V2 §10.3;  +0.00 with no star = structurally inactive",
        fontsize=10.8, linespacing=1.55)
    fig.text(0.012, 0.012,
             "CAUTION on the two blend rows. The D19 composition leg consumes "
             "DARKO, whose MINUTE COVERAGE ramps 11% (2010-11) → 19% (2012-13)"
             " → 44% (2016-17) → 62% (2018-19) → 100% (2023-24 on).\n"
             "corr(coverage, D19 effect) = +0.79 and corr(coverage, D21 "
             "effect) = −0.61 across these 11 seasons, so this pair reads as a "
             "FEATURE-AVAILABILITY ramp — the four-factors leg carrying the "
             "load\nwhile the talent input is starved — NOT as an era effect. "
             "That is the docs/ERAS.md §5 trap at corpus scale.",
             fontsize=8.5, color=INK2, va="bottom", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.085, 0.80, 1))
    out = ROOT / "charts" / "history_feature_by_era.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    res = load()
    for f in (chart_logloss, chart_normgap, chart_feature_by_era):
        print("wrote", f(res))


if __name__ == "__main__":
    main()
