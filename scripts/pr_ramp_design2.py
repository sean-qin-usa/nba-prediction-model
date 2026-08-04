#!/usr/bin/env python3
"""PROPS EARLY-MINUTES RAMP — DESIGN DIAGNOSTIC 2 (minutes only, NO scoring).

design1 found the proj_min bias decays in PLAYER games-played (gp) far faster
than linear-to-20 and that the prior-season FULL-season mean is itself +2.48 min
too high at gp=0 — so "recency inflation of the prior season" is NOT the
mechanism. This script separates the two candidate mechanisms that remain:

  (M1) CALENDAR: October/November minutes are structurally lower than any
       prior-season-based estimate (rotation expansion, load management,
       minutes restrictions) -> bias should track TEAM games played (tgp) and
       be present even for players who are fully available.
  (M2) ROLE-UNESTABLISHED: a player with few games played RELATIVE to his
       team's games played is injured/demoted -> low gp is itself bad news,
       and the bias should track the availability share gp/tgp at any date.

Read-only DB. Writes data/pr_ramp_design2.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from nbapred.db import connect

DEV = ("2023-24", "2024-25", "2025-26")
HL = 10.0
OUT = ROOT / "data" / "pr_ramp_design2.json"


def prev_season(s):
    y = int(s[:4])
    return f"{y-1}-{str(y)[-2:]}"


def main():
    con = connect(read_only=True)
    df = con.execute("""
        SELECT s.player_id, s.team_id, g.season, g.game_date, s.seconds/60.0 AS mins
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '002%' AND s.seconds >= 720
        ORDER BY s.player_id, g.game_date
    """).fetchdf()
    # team schedule: team games played strictly before a date, per season
    tg = con.execute("""
        SELECT season, team_id, game_date FROM nba_games
        WHERE game_id LIKE '002%' ORDER BY season, team_id, game_date
    """).fetchdf()
    con.close()
    df["game_date"] = df["game_date"].astype("datetime64[ns]")
    df["ord"] = df["game_date"].values.astype("datetime64[D]").astype(int)
    tg["ord"] = tg["game_date"].astype("datetime64[ns]").values.astype("datetime64[D]").astype(int)
    tsched = {}
    for (s, t), sub in tg.groupby(["season", "team_id"], sort=False):
        tsched[(s, int(t))] = np.sort(sub["ord"].to_numpy())
    print(f"loaded {len(df)} rows, {len(tsched)} team-seasons", flush=True)

    byp = {}
    for pid, sub in df.groupby("player_id", sort=False):
        byp[int(pid)] = (sub["ord"].to_numpy(), sub["mins"].to_numpy(float),
                         sub["season"].to_numpy(object))
    ps_mean = df.groupby(["player_id", "season"])["mins"].mean().to_dict()

    recs = []
    for season in DEV:
        prev = prev_season(season)
        sel = df[df["season"] == season]
        for r in sel.itertuples():
            pid = int(r.player_id)
            dates, mins, seas = byp[pid]
            i = int(np.searchsorted(dates, r.ord))
            if i < 8:
                continue
            h = mins[:i]
            w = 0.5 ** (np.arange(i)[::-1] / HL)
            proj = float(np.sum(w * h) / np.sum(w))
            if proj < 20:
                continue
            gp = int((seas[:i] == season).sum())
            sch = tsched.get((season, int(r.team_id)))
            tgp = int(np.searchsorted(sch, r.ord)) if sch is not None else -1
            pm = ps_mean.get((pid, prev), np.nan)
            recs.append((season, int(r.game_date.month), gp, tgp, proj,
                         float(r.mins), float(pm) if pm == pm else np.nan))
    print(f"{len(recs)} conditioned rows", flush=True)

    seas = np.array([x[0] for x in recs])
    mon = np.array([x[1] for x in recs])
    gp = np.array([x[2] for x in recs], float)
    tgp = np.array([x[3] for x in recs], float)
    proj = np.array([x[4] for x in recs])
    y = np.array([x[5] for x in recs])
    prior = np.array([x[6] for x in recs])
    bias = proj - y
    out = {"n": len(recs)}

    def blk(m, label):
        if m.sum() < 30:
            return
        d = dict(n=int(m.sum()), bias=float(bias[m].mean()),
                 se=float(bias[m].std(ddof=1) / np.sqrt(m.sum())),
                 gp=float(gp[m].mean()), tgp=float(tgp[m].mean()),
                 proj=float(proj[m].mean()), y=float(y[m].mean()))
        hp = m & np.isfinite(prior)
        d["prior_bias"] = float((prior[hp] - y[hp]).mean()) if hp.sum() > 20 else None
        out[label] = d
        pb = f"{d['prior_bias']:+.3f}" if d["prior_bias"] is not None else "  n/a"
        print(f"{label:26s} n={d['n']:6d} bias {d['bias']:+.3f} (se {d['se']:.3f})"
              f"  prior-y {pb}  gp {d['gp']:5.1f} tgp {d['tgp']:5.1f}", flush=True)

    print("\n== A. bias by TEAM games played (calendar-early axis, M1) ==")
    for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 15), (15, 20), (20, 30),
                   (30, 45), (45, 90)):
        blk((tgp >= lo) & (tgp < hi), f"tgp[{lo},{hi})")

    print("\n== B. bias by PLAYER gp WITHIN tgp<20 (early season) ==")
    for lo, hi in ((0, 1), (1, 3), (3, 6), (6, 10), (10, 20)):
        blk((tgp < 20) & (gp >= lo) & (gp < hi), f"tgp<20 gp[{lo},{hi})")

    print("\n== C. bias by PLAYER gp WITHIN tgp>=30 (mid/late season, M2 test) ==")
    for lo, hi in ((0, 1), (1, 3), (3, 6), (6, 10), (10, 20), (20, 90)):
        blk((tgp >= 30) & (gp >= lo) & (gp < hi), f"tgp>=30 gp[{lo},{hi})")

    print("\n== D. bias by availability share gp/max(tgp,1), tgp>=10 ==")
    share = gp / np.maximum(tgp, 1)
    for lo, hi in ((0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.01)):
        blk((tgp >= 10) & (share >= lo) & (share < hi), f"share[{lo},{hi})")

    print("\n== E. FULLY AVAILABLE ONLY (gp == tgp): pure calendar effect ==")
    full = gp >= tgp - 0.5
    for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 15), (15, 20), (20, 30),
                   (30, 45), (45, 90)):
        blk(full & (tgp >= lo) & (tgp < hi), f"FULL tgp[{lo},{hi})")

    print("\n== F. per-season, fully available, tgp<15 (stability of the level) ==")
    for s in DEV:
        blk(full & (tgp < 15) & (seas == s), f"FULL tgp<15 {s}")
    print("\n== G. per-season, all rows, tgp<15 ==")
    for s in DEV:
        blk((tgp < 15) & (seas == s), f"tgp<15 {s}")
    print("\n== H. month x tgp sanity ==")
    for mm in (10, 11, 12, 1, 2, 3, 4):
        blk(mon == mm, f"month {mm}")

    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
