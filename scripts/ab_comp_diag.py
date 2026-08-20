"""ABSENCE AUDIT — composition.py diagnostic (D133 analogue at the comp leg).

Measures, PRODUCTION-FAITHFULLY (weekly refit schedule of prod_by_season.py,
CompositionModel verbatim, oracle OUT sets), whether `trail_min` — the
per-player minutes LEVEL that composition.strength() weights DARKO talent by —
is biased as a function of the PLAYER'S OWN RECENT ABSENCE.

Bias is reported in the estimator's own units (minutes) and in points of comp
strength (talent x dmin / 48), at BOTH October and mid-season (team-gp>=30) so
a composition artefact cannot masquerade as a calendar effect (D133 method).

READ-ONLY on data/nba.duckdb. Writes data/ab_comp_diag.json.
"""
import datetime as _dt
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect
from nbapred.model.composition import ROSTER_DAYS, CompositionModel

SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
PLAY_SEC = 720          # composition's own participation threshold


def _d(x):
    return x.date() if hasattr(x, "date") else x


def main():
    con = connect(read_only=True)
    # ---- corpus ------------------------------------------------------------
    pm = con.execute("""
        SELECT s.game_id, s.team_id, s.player_id, s.seconds, g.game_date, g.season
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '002%'""").fetchdf()
    pm["game_date"] = [_d(x) for x in pm["game_date"]]
    played = defaultdict(set)              # (gid, tid) -> {pid} seconds>0 (oracle)
    mins = {}                              # (gid, pid) -> minutes
    for r in pm.itertuples():
        if r.seconds and r.seconds > 0:
            played[(r.game_id, int(r.team_id))].add(int(r.player_id))
        mins[(r.game_id, int(r.player_id))] = float(r.seconds or 0) / 60.0
    # per (season, team) schedule dates; per (season, team, player) played dates
    sched = defaultdict(list)
    pdates = defaultdict(set)              # (season, tid, pid) -> {dates with >=720}
    pdates_any = defaultdict(set)          # (season, tid, pid) -> {dates with >0}
    for r in pm.itertuples():
        k = (r.season, int(r.team_id))
        if r.seconds and r.seconds >= PLAY_SEC:
            pdates[(r.season, int(r.team_id), int(r.player_id))].add(r.game_date)
        if r.seconds and r.seconds > 0:
            pdates_any[(r.season, int(r.team_id), int(r.player_id))].add(r.game_date)
    tg = con.execute("""
        SELECT season, team_id, game_date FROM nba_games
        WHERE game_id LIKE '002%' AND wl IS NOT NULL""").fetchall()
    for s, t, d in tg:
        sched[(s, int(t))].append(_d(d))
    for k in sched:
        sched[k] = sorted(set(sched[k]))

    rows = []
    for season in SEASONS:
        meta = con.execute("""
            SELECT game_id, team_id, team_abbrev, matchup, wl, game_date
            FROM nba_games WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL
            ORDER BY game_date""", [season]).fetchdf()
        by, order = {}, []
        for x in meta.itertuples():
            if x.game_id not in by:
                order.append(x.game_id)
            by.setdefault(x.game_id, []).append(x)
        comp = None
        last = None
        for gid in order:
            recs = by[gid]
            if len(recs) != 2:
                continue
            gd = _d(recs[0].game_date)
            if last is None or (gd - last).days >= 7:
                comp = CompositionModel(con, before=gd)
                last = gd
            for x in recs:
                t = int(x.team_id)
                pl = played.get((gid, t), set())
                sc = sched[(season, t)]
                # team games strictly before gd this season
                prior = [d for d in sc if d < gd]
                tgp = len(prior)
                w10 = prior[-10:]
                for pid, p in comp.players.items():
                    if p["team_id"] != t:
                        continue
                    if (gd - p["last_played"]).days > ROSTER_DAYS:
                        continue
                    if pid not in pl:
                        continue      # oracle OUT -> dropped by production too
                    pset = pdates.get((season, t, pid), set())
                    aset = pdates_any.get((season, t, pid), set())
                    first = min(pset) if pset else gd
                    miss10 = sum(1 for d in w10 if d not in pset)
                    miss10s = sum(1 for d in w10 if d >= first and d not in pset)
                    miss10a = sum(1 for d in w10 if d >= first and d not in aset)
                    gp = sum(1 for d in pset if d < gd)
                    rows.append((season, gid, str(gd), t, pid,
                                 float(p["trail_min"]), float(p["talent"]),
                                 float(mins.get((gid, pid), 0.0)),
                                 (gd - p["last_played"]).days,
                                 miss10, miss10s, miss10a, gp, tgp,
                                 1 if str(gd)[5:7] in ("10", "11") else 0))
        print(f"{season}: cumulative rows {len(rows)}", flush=True)
    con.close()

    import pandas as pd
    df = pd.DataFrame(rows, columns=[
        "season", "game_id", "game_date", "team_id", "player_id",
        "trail_min", "talent", "real_min", "dsl", "miss10", "miss10s",
        "miss10a", "gp", "tgp", "octnov"])
    df["bias"] = df["trail_min"] - df["real_min"]
    df["pts_bias"] = df["talent"] * df["bias"] / 48.0
    df.to_csv("data/ab_comp_rows.csv.gz", index=False, compression="gzip")
    print("rows", len(df))

    out = {"n_rows": int(len(df))}

    def curve(sub, key, buckets, label):
        res = []
        for lo, hi in buckets:
            m = (sub[key] >= lo) & (sub[key] <= hi)
            s = sub[m]
            if len(s) == 0:
                res.append((f"{lo}-{hi}", 0, None, None, None))
                continue
            res.append((f"{lo}-{hi}", int(len(s)),
                        round(float(s["bias"].mean()), 4),
                        round(float(s["pts_bias"].mean()), 5),
                        round(float(s["bias"].std(ddof=1) / np.sqrt(len(s))), 4)))
        out[label] = res
        print(f"\n-- {label} (bucket, n, mean minutes bias, mean pts bias, se) --")
        for r in res:
            print("   ", r)

    B = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, 10)]
    curve(df, "miss10s", B, "miss10s_all")
    curve(df[df.octnov == 1], "miss10s", B, "miss10s_octnov")
    curve(df[df.tgp >= 30], "miss10s", B, "miss10s_tgp30")
    curve(df[(df.tgp >= 30) & (df.octnov == 0)], "miss10s", B, "miss10s_tgp30_notoctnov")
    curve(df, "miss10a", B, "miss10a_all")
    curve(df[df.tgp >= 30], "miss10a", B, "miss10a_tgp30")
    GB = [(0, 0), (1, 2), (3, 5), (6, 9), (10, 14), (15, 19), (20, 99)]
    curve(df, "gp", GB, "gp_all")
    curve(df[df.tgp >= 30], "gp", GB, "gp_tgp30")
    DB = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 12)]
    curve(df, "dsl", DB, "dsl_all")
    curve(df[df.tgp >= 30], "dsl", DB, "dsl_tgp30")

    # players who missed NOTHING in the window (D133 (c) control)
    clean = df[df.miss10s == 0]
    out["clean_overall"] = [int(len(clean)), round(float(clean["bias"].mean()), 4),
                            round(float(clean["pts_bias"].mean()), 5)]
    print("\nclean (miss10s==0):", out["clean_overall"])

    # per-season stability of the miss>=1 vs miss==0 contrast
    ss = {}
    for s in SEASONS:
        d0 = df[(df.season == s) & (df.miss10s == 0)]
        d1 = df[(df.season == s) & (df.miss10s >= 1)]
        ss[s] = [int(len(d0)), round(float(d0["bias"].mean()), 4),
                 int(len(d1)), round(float(d1["bias"].mean()), 4),
                 round(float(d1["bias"].mean() - d0["bias"].mean()), 4)]
    out["by_season_miss_contrast"] = ss
    print("\nby season [n0, bias0, n1, bias1, contrast]:")
    for s, v in ss.items():
        print("   ", s, v)

    # TEAM-LEVEL: how much comp strength error per team-game, and the
    # home-away margin error, split by the team's absence load
    g = df.groupby(["season", "game_id", "team_id"]).agg(
        pts_bias=("pts_bias", "sum"), min_bias=("bias", "sum"),
        n_pl=("player_id", "size"), sum_trail=("trail_min", "sum"),
        sum_real=("real_min", "sum"),
        n_miss=("miss10s", lambda x: int((x >= 1).sum())),
        tgp=("tgp", "max")).reset_index()
    out["team_level"] = {
        "n_team_games": int(len(g)),
        "mean_players_counted": round(float(g.n_pl.mean()), 2),
        "mean_sum_trail_min": round(float(g.sum_trail.mean()), 2),
        "mean_sum_real_min": round(float(g.sum_real.mean()), 2),
        "mean_pts_bias": round(float(g.pts_bias.mean()), 5),
        "sd_pts_bias": round(float(g.pts_bias.std(ddof=1)), 5),
    }
    print("\nTEAM LEVEL:", json.dumps(out["team_level"], indent=1))
    tb = []
    for lo, hi in [(0, 0), (1, 1), (2, 2), (3, 3), (4, 9)]:
        s = g[(g.n_miss >= lo) & (g.n_miss <= hi)]
        tb.append((f"{lo}-{hi}", int(len(s)),
                   round(float(s.pts_bias.mean()), 5) if len(s) else None,
                   round(float(s.sum_trail.mean() - s.sum_real.mean()), 3) if len(s) else None))
    out["team_by_nmiss"] = tb
    print("team pts_bias by n players with a recent miss:", tb)
    g.to_csv("data/ab_comp_team.csv", index=False)

    json.dump(out, open("data/ab_comp_diag.json", "w"), indent=1)
    print("\nwrote data/ab_comp_diag.json")


if __name__ == "__main__":
    main()
