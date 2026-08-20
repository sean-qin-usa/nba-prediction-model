#!/usr/bin/env python3
"""SCORABLE-SET RESOLVER for the historical evaluation (D153).

`scripts/scorability_probe.py` answers "does fit_production RETURN?".  That is
not the same question as "can this season honestly be scored", and on the
current DB it gives the wrong answer for 2007-08 and 2008-09: the D123 guard
only checks that carry rows EXIST, and a single placeholder game in the prior
season (24 player_game_stats rows) satisfies it.

This module resolves the set on DATA SUFFICIENCY instead:

  * prior season has >= MIN_PRIOR_PGS player_game_stats rows  (D62 carry input,
    and the FourFactors seed on opening night);
  * this season's own 002 box-score coverage is >= MIN_OWN_COVER of its
    completed games (CompositionModel needs it all season long);
  * odds_market has closes for the season (market comparison).

Everything is derived AT RUN TIME so the answer grows as the backfill lands.

READ-ONLY.  python scripts/history_scorable.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nbapred import threads  # noqa: E402
threads.pin(1)

from nbapred.db import connect  # noqa: E402

MIN_PRIOR_PGS = 20000     # a full season is 25k-32k rows; 20k admits the
                          # lockout season (25,455) and rejects placeholders
MIN_OWN_COVER = 0.99      # same bar tanking.season_floor uses

# Regime strata that are scorable but NOT poolable, with the register reason.
STRATA = {
    "2011-12": "lockout, 66 games, densest schedule in the corpus (ERAS.md §1 §)",
    "2019-20": "E0+E1 pooled: 88 of 1,059 games are the one-site bubble",
    "2020-21": "E2 no-crowd compressed season, out of the eval corpus (D131)",
}


def resolve(con) -> dict:
    rows = con.execute("""
        SELECT g.season,
               count(DISTINCT g.game_id) AS sched,
               count(DISTINCT CASE WHEN s.game_id IS NOT NULL
                                   THEN g.game_id END) AS boxed
        FROM (SELECT DISTINCT game_id, season FROM nba_games
              WHERE game_id LIKE '002%' AND wl IS NOT NULL) g
        LEFT JOIN (SELECT DISTINCT game_id FROM player_game_stats
                   WHERE seconds > 0) s USING (game_id)
        GROUP BY 1 ORDER BY 1""").fetchall()
    pgs = dict(con.execute("""
        SELECT g.season, count(*) FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, season FROM nba_games
              WHERE game_id LIKE '002%') g USING (game_id)
        GROUP BY 1""").fetchall())
    odds = dict(con.execute(
        "SELECT season_end, count(*) FROM odds_market "
        "WHERE p_home_spread IS NOT NULL GROUP BY 1").fetchall())
    out = {}
    for season, sched, boxed in rows:
        y = int(season[:4])
        prev = f"{y - 1}-{(y) % 100:02d}"
        cover = boxed / sched if sched else 0.0
        prior = pgs.get(prev, 0)
        nod = odds.get(y + 1, 0)
        ok = (prior >= MIN_PRIOR_PGS and cover >= MIN_OWN_COVER and nod > 0)
        why = []
        if prior < MIN_PRIOR_PGS:
            why.append(f"prior {prev} has {prior} pgs rows < {MIN_PRIOR_PGS}")
        if cover < MIN_OWN_COVER:
            why.append(f"own box coverage {boxed}/{sched}={cover:.3f}")
        if not nod:
            why.append("no odds_market closes")
        out[season] = {"season": season, "sched": sched, "boxed": boxed,
                       "own_cover": round(cover, 4), "prior_season": prev,
                       "prior_pgs_rows": prior, "odds_rows": nod,
                       "scorable": ok, "why_not": "; ".join(why),
                       "separate_stratum": STRATA.get(season)}
    return out


def sets(con):
    """(poolable_seasons, stratum_seasons) in chronological order."""
    r = resolve(con)
    ok = [s for s in sorted(r) if r[s]["scorable"]]
    pool = [s for s in ok if not r[s]["separate_stratum"]]
    strat = [s for s in ok if r[s]["separate_stratum"]]
    return pool, strat, r


if __name__ == "__main__":
    con = connect(read_only=True)
    pool, strat, r = sets(con)
    from nbapred.model.tanking import season_floor
    tf = season_floor(con)
    con.close()
    print(f"{'season':9s} {'sched':>5s} {'boxed':>5s} {'cover':>6s} "
          f"{'prior_pgs':>9s} {'odds':>5s}  scorable  note")
    for s in sorted(r):
        d = r[s]
        note = d["separate_stratum"] or d["why_not"]
        print(f"{s:9s} {d['sched']:5d} {d['boxed']:5d} {d['own_cover']:6.3f} "
              f"{d['prior_pgs_rows']:9d} {d['odds_rows']:5d}  "
              f"{'YES' if d['scorable'] else 'no ':8s}  {note[:70]}")
    print(f"\nPOOLABLE ({len(pool)}): {', '.join(pool)}")
    print(f"SEPARATE STRATA ({len(strat)}): {', '.join(strat)}")
    print(f"tanking.season_floor() = {tf}")
    json.dump({"poolable": pool, "strata": strat, "detail": r,
               "tank_season_floor": tf},
              open(ROOT / "data" / "history_scorable.json", "w"), indent=1)
