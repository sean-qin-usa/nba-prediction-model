#!/usr/bin/env python3
"""D175: the HEADROOM bound. Not shippable — deliberately uses outcomes.

The CBS arm loses -> +0.490pp. The diagnosis is marginal-set PRECISION (78.6%
vs the official report's 96.6-98.8%). This script bounds how much of that is
recoverable: it re-runs the CBS arm with the false positives REMOVED BY ORACLE
(any CBS-marginal name who actually logged minutes is dropped).

That is leakage BY CONSTRUCTION and the number is an UPPER BOUND, not a result.
It answers exactly one question: if a perfect-precision archival feed existed
for an old season, would it be worth the -0.25..-0.74pp benchmark, or is there
simply no headroom in an old season at all?

READ-ONLY on the DB. Changes no default. Ships nothing.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("TANK_SEASON_FLOOR", "2020-21")
import nbapred.threads              # noqa: E402
nbapred.threads.pin(1)
import numpy as np, pandas as pd    # noqa: E402
from nbapred.db import connect      # noqa: E402
import k19_t2                       # noqa: E402
from d175_price_cbs import build_cbs_rout   # noqa: E402

season = sys.argv[1] if len(sys.argv) > 1 else "2015-16"
tag = sys.argv[2] if len(sys.argv) > 2 else "cbs_2015_16"
con = connect(read_only=True, retry_s=60)
inact = {}
for g, p in con.execute("SELECT game_id, player_id FROM game_inactives").fetchall():
    inact.setdefault(g, set()).add(int(p))

base = k19_t2.season_run(con, season, "t2i", {}, set(), inact)
print(f"BASELINE t2i           norm={base['norm_gap_pct']:+.2f}%  outs/tm={base['mean_outs_per_team']:.3f}")

rout, rcov, _ = build_cbs_rout(con, tag, season, ("OUT", "OUT_SEASON"))
r_raw = k19_t2.season_run(con, season, "t2", rout, rcov, inact)
print(f"CBS as-is              norm={r_raw['norm_gap_pct']:+.2f}%  "
      f"DELTA={r_raw['norm_gap_pct']-base['norm_gap_pct']:+.3f}pp  "
      f"outs/tm={r_raw['mean_outs_per_team']:.3f}")

# ---- ORACLE CLEAN: drop any (game_date, team, player) who actually played ----
g = con.execute("""SELECT game_id, game_date, team_abbrev FROM nba_games
    WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL""", [season]).fetchdf()
g["ds"] = g.game_date.astype(str).str[:10]
played = con.execute("""SELECT game_id, player_id FROM player_game_stats
    WHERE seconds>0""").fetchdf()
pl = {(r.game_id, int(r.player_id)) for r in played.itertuples()}
gk = {(r.ds, k19_t2.fx(r.team_abbrev)): r.game_id for r in g.itertuples()}

clean, dropped = {}, 0
for (ds, ab), s in rout.items():
    gid = gk.get((ds, ab))
    if gid is None:
        continue
    keep = {p for p in s if (gid, p) not in pl}
    dropped += len(s) - len(keep)
    if keep:
        clean[(ds, ab)] = keep
r_cl = k19_t2.season_run(con, season, "t2", clean, rcov, inact)
d_cl = r_cl["norm_gap_pct"] - base["norm_gap_pct"]
print(f"CBS ORACLE-CLEANED     norm={r_cl['norm_gap_pct']:+.2f}%  "
      f"DELTA={d_cl:+.3f}pp  outs/tm={r_cl['mean_outs_per_team']:.3f}  "
      f"(dropped {dropped} false-positive names)")

# ---- and the FULL oracle: every player who did not play, from the box score ----
# bounds what a PERFECT pregame availability oracle is worth on this season
full = {}
for r in g.itertuples():
    pass
allp = con.execute("""SELECT gi.game_id, gi.player_id FROM game_inactives gi""").fetchdf()
print()
print(f"benchmark to beat: official report on modern seasons = -0.741pp "
      f"(honest old-era range -0.25..-0.74pp)")
json.dump({"baseline_norm": base["norm_gap_pct"],
           "cbs_raw_norm": r_raw["norm_gap_pct"],
           "cbs_raw_delta_pp": round(r_raw["norm_gap_pct"] - base["norm_gap_pct"], 4),
           "cbs_oracleclean_norm": r_cl["norm_gap_pct"],
           "cbs_oracleclean_delta_pp": round(d_cl, 4),
           "false_positive_names_dropped": dropped,
           "season": season},
          open(ROOT / "data" / f"d175_headroom_{tag}.json", "w"), indent=1)
print(f"wrote data/d175_headroom_{tag}.json")
