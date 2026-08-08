#!/usr/bin/env python3
"""D170 verification: per-season coverage of all three availability feeds, plus
the DIRECT CORRECTNESS TEST on game_inactives.

CORRECTNESS TEST: a player listed inactive for a game must not appear in
`player_game_stats` with minutes for that game. This is an independent check —
`game_inactives` comes from BoxScoreSummaryV2's InactivePlayers result set and
`player_game_stats` from BoxScoreTraditionalV3, so agreement is not tautological.

DARKO MINUTE COVERAGE is computed POINT-IN-TIME (a player counts as rated only
if he has a darko_history row STRICTLY BEFORE the game date), which is the
condition CompositionModel actually applies — not "has a row somewhere".

READ-ONLY.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import nbapred.threads  # noqa: E402
nbapred.threads.pin(1)

import duckdb  # noqa: E402

DB = REPO / "data" / "nba.duckdb"


def _ro():
    for i in range(60):
        try:
            return duckdb.connect(str(DB), read_only=True)
        except duckdb.IOException as e:
            if "lock" not in str(e).lower():
                raise
            print(f"write lock held; yielding 60s ({i+1}/60)", flush=True)
            time.sleep(60)
    raise RuntimeError("lock held too long")


def main() -> None:
    con = _ro()
    print("=" * 118)
    print("GAME_INACTIVES — coverage, density, and the inactive-vs-minutes "
          "correctness test")
    print("=" * 118)
    q = """
    WITH g AS (SELECT DISTINCT season, game_id, game_date FROM nba_games
               WHERE game_id LIKE '002%'),
    cov AS (SELECT g.season, count(DISTINCT g.game_id) tot,
                   count(DISTINCT i.game_id) cov
            FROM g LEFT JOIN (SELECT DISTINCT game_id FROM game_inactives) i
              USING(game_id) GROUP BY 1),
    dens AS (SELECT g.season, count(*) n_inact,
                    count(DISTINCT i.game_id) gi
             FROM game_inactives i JOIN g USING(game_id) GROUP BY 1),
    viol AS (SELECT g.season,
                    sum(CASE WHEN p.seconds > 0 THEN 1 ELSE 0 END) v_min,
                    sum(CASE WHEN p.player_id IS NOT NULL THEN 1 ELSE 0 END) v_any,
                    count(*) n
             FROM game_inactives i JOIN g USING(game_id)
             LEFT JOIN player_game_stats p
               ON p.game_id = i.game_id AND p.player_id = i.player_id
             GROUP BY 1)
    SELECT cov.season, cov.tot, cov.cov,
           round(100.0*cov.cov/cov.tot,1) pct,
           coalesce(dens.n_inact,0) rows_,
           round(coalesce(dens.n_inact,0)/nullif(2.0*dens.gi,0),2) per_team_game,
           coalesce(viol.v_min,0) viol_minutes, coalesce(viol.v_any,0) viol_any,
           round(100.0*coalesce(viol.v_min,0)/nullif(viol.n,0),4) viol_pct
    FROM cov LEFT JOIN dens USING(season) LEFT JOIN viol USING(season)
    ORDER BY 1"""
    df = con.execute(q).fetchdf()
    print(df.to_string(index=False))
    tot = con.execute("""
      SELECT count(*) n, sum(CASE WHEN p.seconds>0 THEN 1 ELSE 0 END) v
      FROM game_inactives i LEFT JOIN player_game_stats p
        ON p.game_id=i.game_id AND p.player_id=i.player_id""").fetchone()
    print(f"\nPOOLED: {tot[0]} inactive rows, {tot[1]} of them ALSO have "
          f"minutes>0 in player_game_stats => VIOLATION RATE "
          f"{100.0*tot[1]/tot[0]:.4f}%")

    print("\n" + "=" * 118)
    print("INJURY_REPORTS_PIT — report-days per season")
    print("=" * 118)
    print(con.execute("""
      SELECT CASE WHEN month(game_date)>=10
              THEN year(game_date)||'-'||substr(cast(year(game_date)+1 as varchar),3,2)
              ELSE cast(year(game_date)-1 as varchar)||'-'||substr(cast(year(game_date) as varchar),3,2)
             END season,
             count(DISTINCT report_date) report_days, count(*) n_rows,
             min(game_date) first_gd, max(game_date) last_gd
      FROM injury_reports_pit GROUP BY 1 ORDER BY 1""").fetchdf().to_string(index=False))

    print("\n" + "=" * 118)
    print("DARKO — distinct rated players and PIT MINUTE COVERAGE per season")
    print("  (minute coverage = share of minutes actually played by players who")
    print("   already had a darko_history row BEFORE that game — the condition")
    print("   CompositionModel applies)")
    print("=" * 118)
    print(con.execute("""
      WITH g AS (SELECT DISTINCT season, game_id, game_date FROM nba_games
                 WHERE game_id LIKE '002%'),
      pg AS (SELECT g.season, g.game_date, p.player_id, p.seconds
             FROM player_game_stats p JOIN g USING(game_id)
             WHERE p.seconds > 0),
      fd AS (SELECT player_id, min(date) first_date FROM darko_history GROUP BY 1)
      SELECT pg.season,
             count(DISTINCT pg.player_id) players,
             count(DISTINCT CASE WHEN fd.first_date < pg.game_date
                                 THEN pg.player_id END) rated_players,
             round(sum(pg.seconds)/60.0, 0) tot_min,
             round(100.0*sum(CASE WHEN fd.first_date < pg.game_date
                                  THEN pg.seconds ELSE 0 END)/sum(pg.seconds), 2)
               AS pit_minute_cov_pct
      FROM pg LEFT JOIN fd USING(player_id)
      GROUP BY 1 ORDER BY 1""").fetchdf().to_string(index=False))
    print(con.execute(
        "SELECT count(*) rows_, count(DISTINCT player_id) players_, "
        "min(date) first_, max(date) last_ FROM darko_history").fetchdf()
        .to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
