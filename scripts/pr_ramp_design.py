#!/usr/bin/env python3
"""PROPS EARLY-MINUTES RAMP — DESIGN DIAGNOSTIC (minutes only, NO scoring).

Runs BEFORE the pre-registration. Touches only the MINUTES level (proj_min vs
realized minutes vs candidate priors) — it never evaluates the props endpoint
(points CRPS/PIT), so it cannot select on the gated endpoint. Disclosed in
data/props_ramp_prereg.md as design input.

Answers, on the DEV seasons only:
  1. Is proj_min - realized_min > 0 in Oct-Nov, and how does it decay in gp?
  2. Is the candidate shrink target (prior-season FULL-season mean minutes)
     BELOW the recency-weighted proj_min, i.e. is the shrink direction right?
  3. What does the Oct-Nov eval universe look like (gp distribution, movers,
     prior-season availability)?

Read-only DB. Writes data/pr_ramp_design.json.
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
OUT = ROOT / "data" / "pr_ramp_design.json"


def prev_season(s):
    y = int(s[:4])
    return f"{y-1}-{str(y)[-2:]}"


def load(con):
    df = con.execute("""
        SELECT s.player_id, s.team_id, g.season, g.game_date, s.seconds/60.0 AS mins
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '002%' AND s.seconds >= 720
        ORDER BY s.player_id, g.game_date
    """).fetchdf()
    df["game_date"] = df["game_date"].astype("datetime64[ns]")
    df["ord"] = df["game_date"].values.astype("datetime64[D]").astype(int)
    return df


def main():
    con = connect(read_only=True)
    df = load(con)
    con.close()
    print(f"loaded {len(df)} 002/>=720 player-games", flush=True)

    # per-player arrays (chronological)
    byp = {}
    for pid, sub in df.groupby("player_id", sort=False):
        byp[int(pid)] = (sub["ord"].to_numpy(), sub["mins"].to_numpy(float),
                         sub["season"].to_numpy(object), sub["team_id"].to_numpy())

    # prior-season full-season mean minutes per (player, season)
    ps_mean = (df.groupby(["player_id", "season"])["mins"].mean().to_dict())
    ps_n = (df.groupby(["player_id", "season"])["mins"].size().to_dict())
    # prior-season primary team = team of the LAST prior-season game (D103 arg_max)
    ps_team = {}
    for (pid, seas), sub in df.groupby(["player_id", "season"], sort=False):
        ps_team[(int(pid), seas)] = int(sub["team_id"].to_numpy()[-1])

    recs = []
    for season in DEV:
        prev = prev_season(season)
        sel = df[(df["season"] == season)
                 & (df["game_date"].dt.month.isin((10, 11, 12, 1, 2, 3, 4)))]
        for r in sel.itertuples():
            pid = int(r.player_id)
            dates, mins, seas, teams = byp[pid]
            i = int(np.searchsorted(dates, r.ord))   # strictly-before cutoff
            if i < 3:
                continue
            h = mins[:i]
            age = np.arange(i)[::-1]
            w = 0.5 ** (age / HL)
            proj = float(np.sum(w * h) / np.sum(w))
            gp = int((seas[:i] == season).sum())
            pm = ps_mean.get((pid, prev))
            pn = ps_n.get((pid, prev), 0)
            recs.append(dict(season=season, month=int(r.game_date.month),
                             gp=gp, n_hist=i, proj=proj, y=float(r.mins),
                             prior=(float(pm) if pm is not None else np.nan),
                             prior_n=int(pn),
                             mover=int(ps_team.get((pid, prev), -1) != int(r.team_id)
                                       if pm is not None else -1)))
    print(f"{len(recs)} candidate rows", flush=True)

    a = {k: np.array([x[k] for x in recs], float) for k in
         ("gp", "n_hist", "proj", "y", "prior", "prior_n", "month", "mover")}
    seas = np.array([x["season"] for x in recs])
    # the eval conditioning used by the D128 harness
    cond = (a["n_hist"] >= 8) & (a["proj"] >= 20)
    octnov = np.isin(a["month"], (10, 11))
    out = {"n_all": len(recs), "n_cond": int(cond.sum())}

    def blk(mask, label):
        m = mask & cond
        if m.sum() < 20:
            return
        bias = a["proj"][m] - a["y"][m]
        has = m & np.isfinite(a["prior"])
        shrink = a["prior"][has] - a["proj"][has]
        d = dict(n=int(m.sum()), bias_mean=float(bias.mean()),
                 bias_se=float(bias.std(ddof=1) / np.sqrt(m.sum())),
                 proj_mean=float(a["proj"][m].mean()), y_mean=float(a["y"][m].mean()),
                 prior_avail=float(np.isfinite(a["prior"][m]).mean()),
                 shrink_mean=float(shrink.mean()) if has.sum() else None,
                 prior_bias=float((a["prior"][has] - a["y"][has]).mean()) if has.sum() else None,
                 mover_share=float((a["mover"][m] == 1).mean()),
                 gp_mean=float(a["gp"][m].mean()))
        out[label] = d
        print(f"{label:22s} n={d['n']:6d} proj-y {d['bias_mean']:+.3f}"
              f" (se {d['bias_se']:.3f})  prior-proj {d['shrink_mean']:+.3f}"
              f"  prior-y {d['prior_bias']:+.3f}  gp {d['gp_mean']:.1f}"
              f"  prior_avail {d['prior_avail']:.2f}  movers {d['mover_share']:.2f}",
              flush=True)

    blk(np.ones(len(recs), bool), "ALL")
    blk(octnov, "OCT+NOV")
    blk(a["month"] == 10, "OCT")
    blk(a["month"] == 11, "NOV")
    blk(~octnov, "DEC-APR")
    for s in DEV:
        blk(octnov & (seas == s), f"OCTNOV {s}")
    print()
    for lo, hi in ((0, 1), (1, 3), (3, 6), (6, 10), (10, 15), (15, 20), (20, 30),
                   (30, 45), (45, 200)):
        blk((a["gp"] >= lo) & (a["gp"] < hi), f"gp[{lo},{hi})")

    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
