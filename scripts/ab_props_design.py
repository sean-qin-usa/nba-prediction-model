#!/usr/bin/env python3
"""ABSENCE AUDIT — PROPS DESIGN DIAGNOSTIC (minutes only, NO endpoint scoring).

D133 shipped `proj_min -= b(gp)` with b == 0 at gp >= 20, on the argument that
the estimator's own memory (20-long minutes_hist, EWMA hl=10) has warmed by
then. But the MECHANISM D133 itself identified is ABSENCE, and gp is only a
PROXY for absence: early in a season everyone has low gp, so b(gp) picks up the
absence of a whole universe. A player at gp = 40 who has just missed 8 of his
team's last 10 games gets a correction of EXACTLY ZERO.

This script measures the RESIDUAL minutes bias of the SHIPPED production
estimator (D133 ramp ON, `props.player_rates_from_stats` semantics reproduced
exactly) as a function of the player's OWN recent absence, separately in the
ramp-active region (gp < 20) and the ramp-INERT region (gp >= 20), and at both
October-November and mid-season (team-gp >= 30) so a composition artefact
cannot masquerade as a calendar effect.

Universe = the D128/D133 eval universe verbatim: 002 targets, seconds >= 720,
n_hist >= 8, proj_min >= 20.

MINUTES LEVEL ONLY. The points/CRPS endpoint is never touched here.
Read-only DB. Writes data/ab_props_design.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from nbapred.db import connect
from nbapred.engine.props import minutes_ramp

SEASONS = ("2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
           "2025-26")
DEV = ("2023-24", "2024-25", "2025-26")
HOLDOUT = ("2021-22", "2022-23")
HL = 10.0


def main():
    con = connect(read_only=True)
    df = con.execute("""
        SELECT s.player_id, s.team_id, g.season, g.game_date, s.seconds/60.0 AS mins
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '002%' AND s.seconds >= 720
        ORDER BY s.player_id, g.game_date""").fetchdf()
    df["ord"] = df["game_date"].values.astype("datetime64[D]").astype(int)
    # team schedules + the player's own >=12-min dates, for the absence axis
    sched = defaultdict(list)
    for s, t, d in con.execute("""
        SELECT season, team_id, game_date FROM nba_games
        WHERE game_id LIKE '002%' AND wl IS NOT NULL""").fetchall():
        sched[(s, int(t))].append(
            np.datetime64(d).astype("datetime64[D]").astype(int))
    for k in sched:
        sched[k] = np.array(sorted(set(sched[k])))
    con.close()
    print("loaded", len(df), "002/>=720 player-games", flush=True)

    byp = {}
    for pid, sub in df.groupby("player_id", sort=False):
        byp[int(pid)] = (sub["ord"].to_numpy(), sub["mins"].to_numpy(float),
                         sub["season"].to_numpy(object),
                         sub["team_id"].to_numpy())
    bigdates = defaultdict(set)          # (season, tid, pid) -> {ord}
    for r in df.itertuples():
        bigdates[(r.season, int(r.team_id), int(r.player_id))].add(int(r.ord))

    recs = []
    for r in df.itertuples():
        pid = int(r.player_id)
        dates, mins, seas, teams = byp[pid]
        i = int(np.searchsorted(dates, r.ord))
        if i < 3:
            continue
        h = mins[:i]
        w = 0.5 ** (np.arange(i)[::-1] / HL)
        proj_raw = float(np.sum(w * h) / np.sum(w))
        gp = int((seas[:i] == r.season).sum())
        proj = max(proj_raw - minutes_ramp(gp), 0.0)     # SHIPPED estimator
        if i < 8 or proj_raw < 20:                       # D128 eval conditioning
            continue
        sc = sched.get((r.season, int(r.team_id)))
        if sc is None:
            continue
        prior = sc[sc < r.ord]
        tgp = int(len(prior))
        w10 = prior[-10:]
        pset = bigdates[(r.season, int(r.team_id), pid)]
        if not w10.size:
            continue
        first = min(pset) if pset else r.ord
        miss10 = int(sum(1 for d in w10 if d >= first and d not in pset))
        recs.append((r.season, int(r.game_date.month), gp, tgp, miss10,
                     proj_raw, proj, float(r.mins)))
    print("rows in eval universe:", len(recs), flush=True)

    import pandas as pd
    d = pd.DataFrame(recs, columns=["season", "month", "gp", "tgp", "miss10",
                                    "proj_raw", "proj", "y"])
    d["res"] = d["proj"] - d["y"]          # residual bias AFTER the D133 ramp
    d["res_raw"] = d["proj_raw"] - d["y"]  # bias BEFORE the ramp
    d["octnov"] = d.month.isin((10, 11)).astype(int)
    d.to_csv("data/ab_props_rows.csv.gz", index=False, compression="gzip")

    out = {"n_rows": int(len(d))}

    def curve(sub, label, col="res"):
        res = []
        for lo, hi in [(0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, 10)]:
            s = sub[(sub.miss10 >= lo) & (sub.miss10 <= hi)]
            res.append((f"{lo}-{hi}", int(len(s)),
                        round(float(s[col].mean()), 4) if len(s) else None,
                        round(float(s[col].std(ddof=1) / np.sqrt(len(s))), 4)
                        if len(s) > 1 else None))
        out[label] = res
        print(f"\n-- {label} [{col}] (miss10, n, mean min bias, se) --")
        for x in res:
            print("   ", x)

    dv = d[d.season.isin(DEV + HOLDOUT)]
    curve(dv, "res_all")
    curve(dv[dv.gp >= 20], "res_gp20plus")          # ramp EXACTLY ZERO here
    curve(dv[(dv.gp >= 20) & (dv.tgp >= 30)], "res_gp20_tgp30")
    curve(dv[(dv.gp >= 20) & (dv.octnov == 0)], "res_gp20_not_octnov")
    curve(dv[dv.gp < 20], "res_gp_lt20")
    curve(dv[dv.octnov == 1], "res_octnov")
    curve(dv, "res_raw_all", col="res_raw")

    # share of the eval universe by (gp>=20, miss10>=1)
    tab = {}
    for g20 in (0, 1):
        for m1 in (0, 1):
            s = dv[((dv.gp >= 20).astype(int) == g20)
                   & ((dv.miss10 >= 1).astype(int) == m1)]
            tab[f"gp20={g20},miss={m1}"] = [int(len(s)),
                                            round(float(len(s) / len(dv)), 4),
                                            round(float(s.res.mean()), 4) if len(s) else None]
    out["universe_split"] = tab
    print("\nuniverse split [n, share, mean residual]:", json.dumps(tab, indent=1))

    # per-season stability of the gp>=20 absence contrast
    ss = {}
    for s_ in SEASONS:
        s = dv[(dv.season == s_) & (dv.gp >= 20)]
        if len(s) == 0:
            continue
        a0 = s[s.miss10 == 0].res.mean()
        a1 = s[s.miss10 >= 3].res.mean()
        ss[s_] = [int((s.miss10 == 0).sum()), round(float(a0), 4),
                  int((s.miss10 >= 3).sum()), round(float(a1), 4),
                  round(float(a1 - a0), 4)]
    out["by_season_gp20_contrast"] = ss
    print("\nby season, gp>=20 [n0, res0, n3+, res3+, contrast]:")
    for k, v in ss.items():
        print("   ", k, v)

    json.dump(out, open("data/ab_props_design.json", "w"), indent=1)
    print("\nwrote data/ab_props_design.json")


if __name__ == "__main__":
    main()
