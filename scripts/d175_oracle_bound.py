#!/usr/bin/env python3
"""D175: does ANY availability information close the old-era gap?

The oracle-cleaned CBS arm still LOSES (+0.440pp). Before blaming the source,
test the ceiling: a PERFECT pregame availability oracle on the same season —
out-set = every player who logged 0 minutes in that game. That is maximal
leakage and is not shippable; it exists to separate two very different worlds:

  * oracle WINS big  -> the old-era gap IS an availability-data gap, and a
                        better unofficial source is worth hunting.
  * oracle is small  -> the gap is NOT about who is out, and no injury feed
                        (official, unofficial, or perfect) can close it.

READ-ONLY. Ships nothing, changes no default.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("TANK_SEASON_FLOOR", "2020-21")
import nbapred.threads              # noqa: E402
nbapred.threads.pin(1)
import numpy as np, pandas as pd    # noqa: E402
from nbapred.db import connect      # noqa: E402
import k19_t2                       # noqa: E402

con = connect(read_only=True, retry_s=60)
inact = {}
for g, p in con.execute("SELECT game_id, player_id FROM game_inactives").fetchall():
    inact.setdefault(g, set()).add(int(p))

res = {}
for season in ("2015-16", "2012-13", "2023-24"):
    base = k19_t2.season_run(con, season, "t2i", {}, set(), inact)
    # ---- PERFECT ORACLE: everyone who logged no minutes in this exact game ----
    g = con.execute("""SELECT game_id, game_date, team_id, team_abbrev FROM nba_games
        WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL""", [season]).fetchdf()
    g["ds"] = g.game_date.astype(str).str[:10]
    pl = con.execute("""SELECT game_id, player_id FROM player_game_stats
                        WHERE seconds>0""").fetchdf()
    playedby = {}
    for r in pl.itertuples():
        playedby.setdefault(r.game_id, set()).add(int(r.player_id))
    # candidate pool per team = anyone who played for that team this season
    roster = con.execute("""SELECT DISTINCT p.team_id, p.player_id
        FROM player_game_stats p JOIN (SELECT game_id FROM nba_games WHERE season=?
        AND game_id LIKE '002%' AND wl IS NOT NULL) s USING(game_id)""", [season]).fetchdf()
    rst = {}
    for r in roster.itertuples():
        rst.setdefault(int(r.team_id), set()).add(int(r.player_id))
    orout = {}
    for r in g.itertuples():
        did = playedby.get(r.game_id, set())
        out = rst.get(int(r.team_id), set()) - did
        if out:
            orout[(r.ds, k19_t2.fx(r.team_abbrev))] = out
    rcov = set(g.ds)
    o = k19_t2.season_run(con, season, "t2", orout, rcov, inact)
    d = o["norm_gap_pct"] - base["norm_gap_pct"]
    print(f"{season}:  t2i {base['norm_gap_pct']:+6.2f}% (outs/tm {base['mean_outs_per_team']:.3f})"
          f"   PERFECT-ORACLE {o['norm_gap_pct']:+6.2f}% (outs/tm {o['mean_outs_per_team']:.3f})"
          f"   DELTA {d:+.3f}pp")
    res[season] = {"t2i": base["norm_gap_pct"], "oracle": o["norm_gap_pct"],
                   "delta_pp": round(float(d), 4),
                   "t2i_outs": base["mean_outs_per_team"],
                   "oracle_outs": o["mean_outs_per_team"]}
json.dump(res, open(ROOT / "data" / "d175_oracle_bound.json", "w"), indent=1)
print("\nwrote data/d175_oracle_bound.json")
