"""charts/why_lines_move.png  (D167)

  LEFT   share of total open->close movement completed vs hours-to-tip, with the
         5PM ET report and the T-30 inactive filing marked, split by tip slot so
         the clock-landmark vs tip-landmark question is answerable by eye.
  RIGHT  our CLV split by news / no-news games, with season-clustered intervals.

House palette + marks (scripts/make_status_charts.py): model blue / market orange,
thin marks, recessive grid, direct labels, no dual axes. 150 dpi.
"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SP = Path('/tmp/claude-1004/-hdd-steveqin-sean-dev/4912ce58-69cd-488d-a5f4-0d3ed2eed30c/scratchpad')
C_MODEL, C_MKT = "#2a78d6", "#eb6834"
C_GOOD, C_BAD = "#1baf7a", "#d1495b"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e7e6e2"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlecolor": INK})
try:
    from scipy import stats; tppf = lambda k: stats.t.ppf(0.975, k)
except Exception:
    tppf = lambda k: {2: 4.303, 4: 2.776}.get(k, 2.0)


def clus(s, v):
    d = pd.DataFrame({"s": s, "v": v}); ms = d.groupby("s").v.mean().values
    K = len(ms); mu = ms.mean(); sd = ms.std(ddof=1)
    return mu, tppf(K - 1) * sd / math.sqrt(K), K


def build_share_curve():
    ev = pd.read_csv(ROOT / "data/wlm_events.csv.gz", parse_dates=["ts"], dtype={"game_id": str}).dropna(subset=["cons"])
    gm = pd.read_csv(SP / "wlm_gm1.csv", parse_dates=["game_date", "tip_ts", "first_ts"], dtype={"game_id": str})
    gm = gm[gm.tip_ts.notna()].copy()
    tipmin = gm.tip_ts.dt.hour * 60 + gm.tip_ts.dt.minute
    gm["sched_min"] = (np.floor((tipmin - 3) / 30) * 30).astype(int)
    ev = ev.merge(gm[["game_id", "tip_ts", "sched_min"]], on="game_id", how="inner")
    ev = ev.sort_values(["game_id", "ts"], kind="stable")
    ev["h2tip"] = (ev.tip_ts - ev.ts).dt.total_seconds() / 3600.0
    ev = ev[ev.h2tip >= 0]
    g = ev.groupby("game_id")
    gg = gm.set_index("game_id"); gg["po"] = g.cons.first(); gg["pc"] = g.cons.last()
    gg = gg.dropna(subset=["po", "pc"]); gg["D"] = gg.pc - gg.po
    HR = np.array([24, 18, 12, 10, 8, 7, 6, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.75, 1.5, 1.25, 1, .83, .67, .5, .33, .17, 0])
    out = {}
    for lab, idx in [("all", gg.index),
                     ("early", gg.index[gg.sched_min <= 19 * 60 + 30]),
                     ("late", gg.index[gg.sched_min >= 21 * 60])]:
        sg = gg.loc[idx]; Dt_tot = sg.D.values; den = (Dt_tot ** 2).sum()
        es = ev[ev.game_id.isin(set(idx))]
        ys = []
        for h in HR:
            last = es[es.h2tip >= h].groupby("game_id").cons.last()
            cur = sg.po.copy(); cur.update(last)
            ys.append(((cur - sg.po).values * Dt_tot).sum() / den)
        out[lab] = (HR, np.array(ys), len(sg))
    return out, gg


def main():
    share, gg = build_share_curve()
    m = pd.read_csv(SP / "wlm_seg.csv", dtype={"game_id": str, "gid8": str})
    ch = pd.read_csv(SP / "wlm_churn.csv", dtype={"gid8": str, "game_id": str})
    m3 = m.merge(ch[["gid8", "churn_any", "quest", "qd", "late_scratch"]], on="gid8", how="inner")
    m3["clv"] = np.sign(m3.edge.values) * (m3.h_pc.values - m3.h_po.values)

    fig = plt.figure(figsize=(14.6, 6.4), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0], wspace=0.24)

    # ---------------------------------------------------------------- LEFT
    ax = fig.add_subplot(gs[0, 0])
    for lab, col, lw, z in [("late", C_MKT, 1.6, 3), ("early", C_MODEL, 1.6, 3), ("all", INK, 2.1, 4)]:
        H, Y, n = share[lab]
        ax.plot(H, 100 * Y, "-", color=col, lw=lw, zorder=z)
    ax.axvspan(0, 2.0, color=C_GOOD, alpha=0.07, zorder=0, lw=0)

    # direct labels, parked in the empty triangle below the curve
    for yl, col, txt in [(31.5, C_MKT, "tips >=21:00 ET   n=%d" % share["late"][2]),
                         (24.5, INK,   "ALL games         n=%d" % share["all"][2]),
                         (17.5, C_MODEL, "tips <=19:30 ET   n=%d" % share["early"][2])]:
        ax.plot([15.4, 13.2], [yl, yl], "-", color=col, lw=2.0)
        ax.text(12.6, yl, txt, color=col, fontsize=9, va="center", ha="left")

    # 5PM ET is a CLOCK landmark -> a different lag for each tip slot
    for lab, col, slot, tx, ty in [("late", C_MKT, 21.18, 6.4, 52.0),
                                   ("early", C_MODEL, 19.18, 2.55, 62.0)]:
        lag = slot - 17.0
        H, Y, n = share[lab]
        yv = 100 * np.interp(-lag, -H, Y)
        ax.plot([lag], [yv], "o", color=col, ms=8, mfc="white", mew=2.0, zorder=6)
        ax.annotate("5PM ET report lands here\n%.0f%% of the move already gone" % yv,
                    xy=(lag, yv), xytext=(tx, ty), color=col, fontsize=8.8, ha="left", va="top",
                    arrowprops=dict(arrowstyle="-", color=col, lw=1.0, shrinkA=4, shrinkB=4))
    H, Y, n = share["all"]
    y30 = 100 * np.interp(-0.68, -H, Y)       # T-30 vs SCHEDULED tip == T-41 vs actual tip
    ax.plot([0.68], [y30], "s", color=C_BAD, ms=7, mfc="white", mew=2.0, zorder=6)
    ax.annotate("T-30 inactive list\n%.0f%% gone" % y30, xy=(0.68, y30), xytext=(1.28, 68.0),
                color=C_BAD, fontsize=8.8, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=C_BAD, lw=1.0, shrinkA=4, shrinkB=4))
    ax.text(0.42, 41.0, "the only burst:\nT-2h -> tip\nactivity rate x3.5", color=C_GOOD,
            fontsize=8.8, ha="center", va="center")
    for h in (4, 2, 1):
        yv = 100 * np.interp(-h, -H, Y)
        ax.plot([h], [yv], ".", color=INK2, ms=7, zorder=5)
    ax.text(22.0, 97.5, "share completed:   76% at T-4h      80% at T-2h      91% at T-1h",
            color=INK2, fontsize=9.0, ha="left", va="center")
    ax.set_xlim(24, -0.35); ax.set_ylim(13, 103)
    ax.set_xscale("symlog", linthresh=1.0, linscale=1.4)
    ax.set_xticks([24, 12, 8, 6, 4, 3, 2, 1, 0.5, 0])
    ax.set_xticklabels(["24h", "12h", "8h", "6h", "4h", "3h", "2h", "1h", "30m", "tip"])
    ax.set_xlabel("time before tip")
    ax.set_ylabel("% of total open->close movement already completed")
    ax.set_title("The line does not wait for the report.\nBy 5PM ET three quarters of the move is gone "
                 "\u2014 for early AND late games.", fontsize=11.5, loc="left", pad=10)

    # --------------------------------------------------------------- RIGHT
    ax2 = fig.add_subplot(gs[0, 1])
    cells = [("nothing unresolved\nat 5PM  (quest=0)", m3[m3.quest == 0], C_GOOD),
             ("1-2 unresolved", m3[(m3.quest >= 1) & (m3.quest <= 2)], INK2),
             (">=3 unresolved", m3[m3.quest >= 3], C_MKT),
             ("no D-1->D status\nchange (churn=0)", m3[m3.churn_any == 0], C_GOOD),
             ("heavy churn\n(>=12 changes)", m3[m3.churn_any >= 12], C_MKT),
             ("no late scratch", m3[m3.late_scratch == 0], C_GOOD),
             (">=3 late scratches", m3[m3.late_scratch >= 3], C_MKT)]
    ys = np.arange(len(cells))[::-1]
    XHI = 1.75
    for y, (lab, s_, col) in zip(ys, cells):
        if len(s_) < 20: continue
        mu, h, K = clus(s_.season.values, s_.clv.values)
        ax2.plot([mu - h, mu + h], [y, y], "-", color=col, lw=1.6, solid_capstyle="butt", zorder=3)
        ax2.plot([mu], [y], "o", color=col, ms=7, zorder=4)
        ax2.text(mu + h + 0.06, y, "%+.3f" % mu, color=col, fontsize=8.8, ha="left", va="center")
        ax2.text(XHI - 0.03, y, "n=%s" % format(len(s_), ","), color=INK2, fontsize=8.4,
                 ha="right", va="center")
    mu_all, h_all, _ = clus(m3.season.values, m3.clv.values)
    ax2.axvline(mu_all, color=INK, lw=1.1, ls="--", zorder=2)
    ax2.text(mu_all + 0.04, -0.62, "all report-covered games %+.3f" % mu_all, color=INK,
             fontsize=8.8, ha="left", va="center")
    ax2.axvline(0, color=INK2, lw=1.0, zorder=1)
    ax2.set_yticks(ys); ax2.set_yticklabels([c[0] for c in cells], fontsize=9)
    ax2.set_xlim(-1.15, XHI); ax2.set_ylim(-1.05, len(cells) - 0.35)
    ax2.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    ax2.set_xlabel("our CLV, spread points per bet   (season-clustered 95% interval, K=3)")
    ax2.set_title("And the CLV is not anticipation of that report.\nIt is the same size where nothing broke.",
                  fontsize=11.5, loc="left", pad=10)
    ax2.grid(axis="y", visible=False)

    fig.text(0.008, 0.012,
             "left: n=5,851 games with an intraday path, 2021-22..2025-26; share = weighted price contribution "
             "sum D(t)D(T)/sum D(T)^2.   right: n=2,602 report-covered games, 2023-24..2025-26 "
             "(2025-26 PARTIAL, feed ends 2025-12-21).   Read-only; nothing shipped.  D167",
             fontsize=7.6, color=INK2)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.855, bottom=0.115)
    out = ROOT / "charts" / "why_lines_move.png"
    fig.savefig(out, dpi=150)
    print("WROTE", out)


if __name__ == "__main__":
    main()
