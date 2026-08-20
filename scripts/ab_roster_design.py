"""ABSENCE AUDIT — DESIGN diagnostic for the ROSTER-MEMBERSHIP arm (site #3).

`composition.strength()` skips any player whose `last_played` is more than
ROSTER_DAYS=12 days old. `last_played` is a PARTICIPATION observation, so the
rule evicts a player whose absence exceeds 12 days and does not re-admit him
until he plays again — the D133 mechanism applied to roster MEMBERSHIP.

ARM R: widen the window to N days CONSISTENTLY in `strength()` and in the
OUT-set construction (`prod_by_season.py` builds outs from the same 12-day
test). Exact margin identity:
    dstrength_N(t) = SUM over comp players p with team_id==t, 12 < dsl <= N,
                     who are NOT in the OUT set, of talent*trail_min/48
    dmargin = 0.5 * (dstrength_N(home) - dstrength_N(away))
With the oracle OUT sets of the certified capstone, "not in the OUT set" is
exactly "played tonight".

Reports MARGIN-scale footprints and the §5.5 power arithmetic for a range of N.
Outcomes (y) are never touched. Writes data/ab_roster_design.json.
"""
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from nbapred.db import connect
from nbapred.model.composition import CompositionModel

SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
SCALE, W_COMP = 7.2, 0.5
NS = (14, 18, 21, 25, 30, 45, 9999)


def _d(x):
    return x.date() if hasattr(x, "date") else x


def main():
    con = connect(read_only=True)
    pm = con.execute("""
        SELECT s.game_id, s.team_id, s.player_id, s.seconds
        FROM player_game_stats s WHERE s.game_id LIKE '002%'""").fetchdf()
    played = defaultdict(set)
    for r in pm.itertuples():
        if r.seconds and r.seconds > 0:
            played[(r.game_id, int(r.team_id))].add(int(r.player_id))

    rows = []
    gapstat = []
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
        comp = last = None
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
                host = (x.matchup.split("@")[-1].strip() if "@" in x.matchup
                        else x.matchup.split("vs.")[0].strip())
                sgn = 1.0 if x.team_abbrev == host else -1.0
                pl = played.get((gid, t), set())
                for pid, p in comp.players.items():
                    if p["team_id"] != t:
                        continue
                    dsl = (gd - p["last_played"]).days
                    if dsl <= 12:
                        continue          # already counted by the control
                    gapstat.append((dsl, pid in pl))
                    if pid not in pl:
                        continue          # OUT tonight -> excluded at every N
                    rows.append((season, gid, sgn, dsl,
                                 float(p["talent"]) * float(p["trail_min"]) / 48.0))
        print(season, "cum", len(rows), flush=True)
    con.close()

    gp = pd.DataFrame(gapstat, columns=["dsl", "played"])
    out = {"n_admitted_rows": int(len(rows))}
    print("\nP(player with dsl>12 actually PLAYS tonight), by dsl:")
    gb = []
    for lo, hi in [(13, 14), (15, 17), (18, 21), (22, 25), (26, 30),
                   (31, 45), (46, 10**6)]:
        s = gp[(gp.dsl >= lo) & (gp.dsl <= hi)]
        gb.append((f"{lo}-{hi}", int(len(s)),
                   round(float(s.played.mean()), 4) if len(s) else None))
        print("   ", gb[-1])
    out["p_play_by_dsl"] = gb

    df = pd.DataFrame(rows, columns=["season", "game_id", "sgn", "dsl", "v"])
    caps = pd.read_csv("data/capstone_pergame_d132.csv", dtype={"game_id": str})
    p = caps.p_us.to_numpy(float)
    res = {}
    for N in NS:
        s = df[df.dsl <= N]
        gm = s.assign(sv=s.sgn * s.v).groupby(
            ["season", "game_id"])["sv"].sum().rename("dmargin").reset_index()
        j = caps.merge(gm, on=["season", "game_id"], how="left")
        dm = W_COMP * j.dmargin.fillna(0.0).to_numpy()
        d = dm / SCALE
        lg = np.log(p / (1 - p)) + d
        pn = 1.0 / (1.0 + np.exp(-lg))
        # sd of the paired per-game log-loss delta under BOTH outcomes,
        # averaged with weight p -- a pure VARIANCE quantity (no y used)
        d1 = -np.log(pn / p)
        d0 = -np.log((1 - pn) / (1 - p))
        m = p * d1 + (1 - p) * d0
        v = p * d1 ** 2 + (1 - p) * d0 ** 2 - m ** 2
        sd = float(np.sqrt(v.mean() + m.var()))
        res[str(N)] = {
            "n_players_admitted": int(len(s)),
            "rms_dmargin_pts": round(float(np.sqrt((dm ** 2).mean())), 4),
            "frac_games_moved": round(float((np.abs(dm) > 1e-9).mean()), 4),
            "max_abs_dmargin": round(float(np.abs(dm).max()), 4),
            "mean_dmargin": round(float(dm.mean()), 5),
            "sd_paired_delta": round(sd, 5),
            "MDE80": round(2.802 * sd / np.sqrt(len(dm)), 5),
            "best_case_dlogloss": round(float(0.5 * np.mean(d ** 2 * p * (1 - p))), 6),
        }
        res[str(N)]["ratio_MDE80_over_best"] = round(
            res[str(N)]["MDE80"] / max(res[str(N)]["best_case_dlogloss"], 1e-12), 2)
        print("N=%-5s %s" % (N, json.dumps(res[str(N)])))
        np.save(f"data/ab_dmargin_R{N}.npy", dm)
    out["arms"] = res
    json.dump(out, open("data/ab_roster_design.json", "w"), indent=1)
    print("wrote data/ab_roster_design.json")


if __name__ == "__main__":
    main()
