"""Certification charts from the certified capstone per-game CSV
(data/capstone_pergame.csv, written by scripts/prod_by_season.py with current
defaults: LATE_STATE=0, TANK_TERM=1, T2-HONEST availability OUT tier —
official 5PM injury report UNION official inactives, per D158).

Produces (150 dpi, charts/):
  - logloss_by_season_normalized.png   one panel per season, rolling-100 LL,
                                     model vs market, normalized gap annotated
  - progress_by_ship.png             season LL by shipped stage (structure of
                                     the original chart, extended through the
                                     D90 revert + D112 floor fix)
  - logloss_continuous_current.png   continuous rolling-100 LL across all
                                     certified seasons, boundaries marked

Normalized gap = (ll_us - ll_mkt) / (ln 2 - ll_mkt)   (the D106 measure).
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
LN2 = float(np.log(2.0))

# palette (dataviz reference, validated): model blue / market orange
C_MODEL = "#2a78d6"
C_MKT = "#eb6834"
C_BEAT = "#1baf7a"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e7e6e2"
# ordinal blues for season lines on the progress chart (steps 300/500/700)
SEASON_BLUES = {"2023-24": "#6da7ec", "2024-25": "#256abf", "2025-26": "#0d366b"}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlecolor": INK,
})


def pergame_ll(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def rolling(x, w=100):
    x = np.asarray(x, float)
    if len(x) < w:
        return np.array([]), np.array([])
    kern = np.ones(w) / w
    r = np.convolve(x, kern, mode="valid")
    idx = np.arange(w - 1, len(x))
    return idx, r


def load(path):
    rows = list(csv.DictReader(open(path)))
    rows.sort(key=lambda r: (r["game_date"], r["game_id"]))
    return rows


def season_lls(rows, season=None):
    rr = [r for r in rows if season is None or r["season"] == season]
    y = [float(r["y"]) for r in rr]
    us = pergame_ll(y, [float(r["p_us"]) for r in rr]).mean()
    mk = pergame_ll(y, [float(r["p_mkt"]) for r in rr]).mean()
    return float(us), float(mk), len(rr)


def norm_gap(us, mk):
    return (us - mk) / (LN2 - mk)


def main():
    cert = load(ROOT / "data" / "capstone_pergame.csv")
    seasons = sorted({r["season"] for r in cert})
    charts = ROOT / "charts"

    # ---------------- chart 1: one panel per season, rolling 100 ----------
    fig, axes = plt.subplots(1, len(seasons), figsize=(16, 4.6),
                             sharey=True, dpi=150)
    for ax, s in zip(np.atleast_1d(axes), seasons):
        rr = [r for r in cert if r["season"] == s]
        y = [float(r["y"]) for r in rr]
        l_us = pergame_ll(y, [float(r["p_us"]) for r in rr])
        l_mk = pergame_ll(y, [float(r["p_mkt"]) for r in rr])
        ix, r_us = rolling(l_us)
        _, r_mk = rolling(l_mk)
        ax.plot(ix, r_us, color=C_MODEL, lw=1.7, label="Our model")
        ax.plot(ix, r_mk, color=C_MKT, lw=1.7, label="Market close")
        us, mk, n = season_lls(cert, s)
        ax.set_title(f"{s}", fontsize=11)
        # opaque bbox: on the honest tier the 2025-26 rolling series runs up
        # into this corner, so the annotation must sit ON TOP of the lines
        # rather than behind them (label-collision check, D158).
        ax.text(0.03, 0.965,
                f"LL {us:.4f} vs mkt {mk:.4f}\nnorm gap {100*norm_gap(us, mk):.1f}%",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=8.5, color=INK2, zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                          boxstyle="square,pad=0.25"))
        ax.set_xlabel("game # in season", fontsize=8.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    # headroom so the annotation band never overlaps the series
    ax0 = np.atleast_1d(axes)[0]
    lo, hi = ax0.get_ylim()
    ax0.set_ylim(lo, hi + 0.13 * (hi - lo))
    ax0.set_ylabel("log loss (rolling 100)")
    np.atleast_1d(axes)[0].legend(loc="lower left", fontsize=8.5,
                                  frameon=False)
    p_us, p_mk, n = season_lls(cert)
    fig.suptitle(
        "Rolling log loss by season — certified production stack "
        f"(LATE_STATE=0, TANK_TERM=1, availability T2-HONEST)  |  pooled LL "
        f"{p_us:.4f} vs mkt {p_mk:.4f}, "
        f"normalized gap {100*norm_gap(p_us, p_mk):.2f}% of market skill (n={n})",
        fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(charts / "logloss_by_season_normalized.png", dpi=150)
    plt.close(fig)

    # ---------------- chart 2: progress by shipped improvement ------------
    # stage sources: D41 register literals (pre-PIT), then the archived
    # per-stage capstone CSVs, then the certified run (this script's input).
    # Labels are numbered so the CHRONOLOGY is unambiguous: step number first,
    # then what shipped, then the register D-number and the date the stage was
    # measured (D-numbers are assigned in register order, so step order, D order
    # and date order all agree — the numbering just makes that visible).
    stage_files = [
        ("2. PIT DARKO\nD44 · Jul 29", "capstone_pergame_legacy.csv"),
        ("3. sched layer\nD46 · Jul 29", "capstone_pergame_sched.csv"),
        ("4. cold-start fix\nD55 · Jul 30", "capstone_pergame_csfix.csv"),
        ("5. carry\nD63 · Jul 30", "capstone_pergame_carry.csv"),
        ("6. tank term\nD73 · Jul 30", "capstone_pergame_tank.csv"),
        ("7. late-state\nD90 · Jul 31\n(later REVERTED)", "capstone_pergame_late.csv"),
    ]
    mod3 = ["2023-24", "2024-25", "2025-26"]
    stages = ["1. pre-PIT\n(stale DARKO)\nD41 · Jul 29"]
    vals = {s: [v] for s, v in zip(mod3, (0.6234, 0.6092, 0.5875))}  # D41
    for label, fn in stage_files:
        rows = load(ROOT / "data" / fn)
        stages.append(label)
        for s in mod3:
            vals[s].append(season_lls(rows, s)[0])
    stages.append("8. CERTIFIED NOW\nhonest availability\nD158 · Aug 3")
    for s in mod3:
        vals[s].append(season_lls(cert, s)[0])
    mkts = {s: season_lls(cert, s)[1] for s in mod3}

    fig, ax = plt.subplots(figsize=(13.2, 6.6), dpi=150)
    x = np.arange(len(stages))
    for s in mod3:
        c = SEASON_BLUES[s]
        ax.plot(x, vals[s], color=c, lw=2.0, marker="o", markersize=6.5,
                label=f"{s} (mkt {mkts[s]:.4f})")
        ax.axhline(mkts[s], color=c, lw=1.1, ls=":", alpha=0.75)
    ax.set_xticks(x, stages, fontsize=8.2)
    ax.set_ylabel("season log loss (dotted = market close)")
    ax.legend(frameon=False, fontsize=9.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    hold = []
    for s in ("2021-22", "2022-23"):
        if s in seasons:
            us, mk, _ = season_lls(cert, s)
            hold.append(f"{s} LL {us:.4f} vs mkt {mk:.4f} "
                        f"(norm gap {100*norm_gap(us, mk):.1f}%)")
    if hold:
        ax.text(0.01, -0.16,
                "certified stack on the pre-2023 holdout seasons (not part of the ship "
                "sequence above):  " + "   |   ".join(hold),
                transform=ax.transAxes, fontsize=8.5, color=INK2)
    ax.set_title("Model progress in chronological ship order (step 1 = oldest, "
                 "step 8 = certified today)", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(charts / "progress_by_ship.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---------------- chart 3: continuous rolling LL, one axis ------------
    y = [float(r["y"]) for r in cert]
    l_us = pergame_ll(y, [float(r["p_us"]) for r in cert])
    l_mk = pergame_ll(y, [float(r["p_mkt"]) for r in cert])
    ix, r_us = rolling(l_us)
    _, r_mk = rolling(l_mk)
    fig, ax = plt.subplots(figsize=(15, 5.2), dpi=150)
    ax.plot(ix, r_us, color=C_MODEL, lw=1.5, label="Our model (certified stack)")
    ax.plot(ix, r_mk, color=C_MKT, lw=1.5, label="Market close")
    ax.fill_between(ix, r_us, r_mk, where=(r_us <= r_mk), color=C_BEAT,
                    alpha=0.25, linewidth=0, label="we beat the close")
    ax.fill_between(ix, r_us, r_mk, where=(r_us > r_mk), color=C_MODEL,
                    alpha=0.07, linewidth=0)
    # season boundaries
    start = 0
    for s in seasons:
        n_s = sum(1 for r in cert if r["season"] == s)
        if start > 0:
            ax.axvline(start, color=INK2, lw=1.0, ls="--", alpha=0.55)
        ax.text(start + n_s / 2, 0.995, s, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color=INK2)
        start += n_s
    ax.set_xlabel(f"game index, continuous ({len(seasons)} seasons; rolling 100)")
    ax.set_ylabel("log loss (lower = better)")
    ax.legend(loc="lower left", frameon=False, fontsize=9.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Continuous log loss — certified production "
                 f"(late-state OFF, tank ON, availability T2-HONEST)  |  pooled norm gap "
                 f"{100*norm_gap(p_us, p_mk):.2f}%", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(charts / "logloss_continuous_current.png", dpi=150)
    plt.close(fig)

    # ---------------- console summary -------------------------------------
    print(f"{'season':<9}{'ll_us':>9}{'ll_mkt':>9}{'gap':>10}{'norm':>8}{'n':>7}")
    for s in seasons:
        us, mk, n = season_lls(cert, s)
        print(f"{s:<9}{us:>9.5f}{mk:>9.5f}{us-mk:>+10.5f}"
              f"{100*norm_gap(us, mk):>7.2f}%{n:>7}")
    us, mk, n = season_lls(cert)
    print(f"{'POOLED':<9}{us:>9.5f}{mk:>9.5f}{us-mk:>+10.5f}"
          f"{100*norm_gap(us, mk):>7.2f}%{n:>7}")


if __name__ == "__main__":
    main()
