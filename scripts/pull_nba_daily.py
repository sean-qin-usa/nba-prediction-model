#!/usr/bin/env python3
"""Daily NBA-stats pull: static players, season game log, and per-game artifacts
(play-by-play, boxscore, rotations) for any game not yet cached.

Usage: python scripts/pull_nba_daily.py [--season 2025-26] [--max-games N]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nbapred.config import CURRENT_SEASON, RAW_NBA
from nbapred.db import connect
from nbapred.ingest import nba_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pull_nba_daily")


def seasons_to_pull(explicit: str | None) -> list[str]:
    """Season labels whose games can be occurring now. LeagueGameFinder is
    per-season and INCLUDES preseason (001) games — required by the D84-A
    October bridge / ps-continuity carry, which need 001 rosters BEFORE the
    opener. Gap fixed here: preseason can start in late September, when
    current_season() still resolves to the just-finished season (by design —
    trailing-data consumers want that) — so a September run must ALSO pull
    the UPCOMING season or the first 001 games would be missed until Oct 1."""
    import datetime as dt
    if explicit:
        return [explicit]
    seasons = [CURRENT_SEASON]
    today = dt.date.today()
    if today.month == 9:
        seasons.append(f"{today.year}-{(today.year + 1) % 100:02d}")
    return seasons


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=None,
                    help=f"explicit season (default: {CURRENT_SEASON}, plus "
                         "the upcoming season during September preseason)")
    ap.add_argument("--max-games", type=int, default=None,
                    help="cap per-game pulls this run (rate-limit hygiene)")
    args = ap.parse_args()

    seasons = seasons_to_pull(args.season)
    # NETWORK BEFORE LOCK (standing rule): fetch the season game log BEFORE
    # opening the DB — stats.nba.com trickle-stalls previously froze the write
    # lock for the whole stall (starved every reader).
    dfs = {s: nba_stats.pull_season_games(s) for s in seasons}  # network, no lock
    con = connect()                                       # short write window
    nba_stats.sync_static_players_teams(con)
    game_ids = []
    for s in seasons:
        n = nba_stats.load_season_games(con, s, df=dfs[s])  # no network in lock
        log.info("season %s: %d team-game rows", s, n)
        game_ids += [r[0] for r in con.execute(
            "SELECT DISTINCT game_id FROM nba_games WHERE season=? ORDER BY game_id",
            [s]).fetchall()]
    con.close()  # per-game pulls only write raw files; no DB lock held

    import hashlib, json

    def cached(bucket: str, gid: str) -> bool:
        key = hashlib.sha1(json.dumps({"game_id": gid}, sort_keys=True).encode()).hexdigest()[:16]
        return (RAW_NBA / bucket / f"{key}.json").exists()

    def complete(g: str) -> bool:
        rot_ok = cached("gamerotation", g) or not g.startswith(("002", "004"))
        return cached("playbyplayv3", g) and cached("boxscoretraditionalv3", g) and rot_ok

    todo = [g for g in sorted(set(game_ids)) if not complete(g)]
    if args.max_games:
        todo = todo[: args.max_games]
    log.info("%d games need artifacts", len(todo))
    for i, gid in enumerate(todo):
        try:
            nba_stats.pull_play_by_play(gid)
            nba_stats.pull_boxscore(gid)
            # gamerotation 404s on preseason (001*)/all-star; only regular+playoffs
            if gid.startswith(("002", "004")):
                nba_stats.pull_rotations(gid)
        except Exception:
            log.exception("game %s failed; continuing", gid)
        if (i + 1) % 25 == 0:
            log.info("progress %d/%d", i + 1, len(todo))


if __name__ == "__main__":
    main()
