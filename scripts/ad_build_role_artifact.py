#!/usr/bin/env python3
"""Freeze the GameRotation ROLE artifact that production reads.

`data/ad_role_flags.npz` : three parallel int arrays (player_id, ord, starter)
sorted by (player_id, ord), where `ord` is the game date as days since epoch.
This is the ONLY thing `nbapred/engine/props.py` needs — it keys on
(player_id, game_date), which is unique (a player plays at most one game a day),
so props.py does not need game_id or team_id.

CLEANLINESS GUARD: a team-game is admitted only if it has EXACTLY 5 starters
(`first_in <= 0.5` tenths). 99.682% of the 10,052 cached team-games qualify;
the 32 that do not come from truncated result sets and are dropped rather than
silently mis-flagged.

Also prints the frozen b(bucket) coefficients for props.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from nbapred.db import connect


def main():
    pg = pd.read_csv(ROOT / "data" / "ad_rotation_pg.csv.gz", dtype={"game_id": str})
    ok = pg.groupby(["game_id", "team_id"]).is_starter.transform("sum") == 5
    print(f"team-game 5-starter guard: keeping {ok.mean():.5f} of {len(pg)} rows")
    pg = pg[ok]

    con = connect(read_only=True)
    g = con.execute("""
        SELECT DISTINCT game_id, game_date FROM nba_games WHERE game_id LIKE '002%'
    """).fetchdf()
    con.close()
    g["game_id"] = g["game_id"].astype(str)
    g["ord"] = g["game_date"].astype("datetime64[ns]").values.astype(
        "datetime64[D]").astype(int)
    m = pg.merge(g[["game_id", "ord"]], on="game_id", how="inner")
    m = m[["player_id", "ord", "is_starter"]].drop_duplicates(
        ["player_id", "ord"]).sort_values(["player_id", "ord"])
    print(f"artifact rows {len(m)}  players {m.player_id.nunique()}  "
          f"ord range {m.ord.min()}..{m.ord.max()}")
    np.savez_compressed(ROOT / "data" / "ad_role_flags.npz",
                        player_id=m.player_id.to_numpy(np.int64),
                        ord=m.ord.to_numpy(np.int64),
                        starter=m.is_starter.to_numpy(np.int8))
    print("wrote data/ad_role_flags.npz")

    # ---- frozen coefficients: full-history fit (D133 MINUTES_RAMP precedent)
    r = pd.read_csv(ROOT / "data" / "ad_design2_rows.csv.gz")

    def bk(sl, s5, n5, gap):
        if not np.isfinite(sl) or n5 < 5 or gap > 0:
            return "NA"
        if sl == 1.0 and s5 < 0.5:
            return "PROMOTED"
        if sl == 0.0 and s5 > 0.5:
            return "DEMOTED"
        return "STABLE"

    r["bk"] = [bk(a, b, c, d) for a, b, c, d in
               zip(r.sr_last, r.sr5, r.n5, r.gap)]
    print("\nFULL-HISTORY b(bucket) = mean(proj_min - realized):")
    for lab in ("PROMOTED", "DEMOTED", "STABLE", "NA"):
        s = r.resid[r.bk == lab]
        print(f"  {lab:9s} n={len(s):6d}  b={s.mean():+.4f}  (round3 {round(s.mean(),3):+.3f})")


if __name__ == "__main__":
    main()
