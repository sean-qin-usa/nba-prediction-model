"""DATA-STARVATION FIX (Step 2): ingest the MISSING nba_games schedules.

Context: `nba_games` held only 2022-23..2025-26. Every cross-season trailing
consumer therefore ran STARVED on the earliest eval season:
  * fit_schedule_layer() takes a 730-DAY trailing window — for 2023-24 refits
    that window reaches into 2021-22, which simply did not exist, so n was
    ~half of design and the n/(n+600) shrink pulled hard toward SCHED_PRIOR.
  * continuity_map()/carry needs prior-season 002 rows joined to nba_games —
    absent for 2021-22, so 2022-23 could not be a carry-enabled eval season.
  * tanking.py / latestate.py floor at season >= '2022-23' (a literal chosen
    when 2022-23 WAS the floor).

player_game_stats already holds 002 rows for 2021-22 (1230 games) and 2020-21
(780 of 1080); only the schedule table was missing. leaguegamefinder is the
same source load_season_games() already uses for the loaded seasons, so this is
a pure backfill of an existing loader — no new construction.

Usage:  python scripts/ds_ingest_schedules.py [--seasons 2021-22,2020-21,2019-20]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nbapred.config import DB_PATH  # noqa: E402
from nbapred.db import connect  # noqa: E402
from nbapred.ingest.nba_stats import load_season_games, pull_season_games  # noqa: E402

DEFAULT_SEASONS = ["2021-22", "2020-21", "2019-20"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=",".join(DEFAULT_SEASONS))
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]

    if not args.no_backup:
        bak = Path(str(DB_PATH) + ".pre_ds_ingest.bak")
        if not bak.exists():
            shutil.copy2(DB_PATH, bak)
            print(f"backup -> {bak}")

    # pull first (network), then take the write lock for as short as possible
    frames = {}
    for s in seasons:
        df = pull_season_games(s)
        frames[s] = df
        print(f"pulled {s}: {len(df)} rows, {df.GAME_ID.nunique()} games, "
              f"{df.GAME_DATE.min()}..{df.GAME_DATE.max()}")

    con = connect()
    try:
        for s, df in frames.items():
            n = load_season_games(con, s, df=df)
            print(f"loaded {s}: {n} rows")
        print("\n-- nba_games coverage after load --")
        for r in con.execute("""
            SELECT season, count(DISTINCT game_id) games,
                   count(DISTINCT CASE WHEN game_id LIKE '002%' THEN game_id END) reg,
                   min(game_date), max(game_date)
            FROM nba_games GROUP BY 1 ORDER BY 1""").fetchall():
            print("  ", r)
    finally:
        con.close()


if __name__ == "__main__":
    main()
