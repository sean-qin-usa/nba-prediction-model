"""Point-in-time guards against look-ahead leakage.

The cardinal rule: a feature used to predict a game may depend ONLY on
information that existed strictly before that game's tip. Two traps are easy to
hit because the source is a single "current" snapshot:

  * DARKO (darko_dpm) is TODAY's talent estimate. Joining it to a past game as
    a prior leaks the rest of that season (and beyond) into the backtest.
  * 2K ratings (ratings_2k) are the CURRENT edition. Same problem, plus an
    era-scale mismatch (see docs/LEAKAGE.md).

These helpers force an as-of join: return only the latest snapshot dated
strictly before a cutoff. With just one (current) snapshot stored, they
correctly return EMPTY for any past date — which is the point: you cannot
silently backtest on future ratings.
"""
from __future__ import annotations

import datetime as dt


def darko_asof(con, cutoff: dt.date):
    """DARKO rows from the latest snapshot strictly before `cutoff` (or None)."""
    row = con.execute(
        "SELECT max(snapshot_date) FROM darko_dpm WHERE snapshot_date < ?", [cutoff]
    ).fetchone()
    if not row or row[0] is None:
        return con.execute("SELECT * FROM darko_dpm WHERE false").fetchdf()
    return con.execute("SELECT * FROM darko_dpm WHERE snapshot_date = ?", [row[0]]).fetchdf()


def ratings_2k_asof(con, cutoff: dt.date):
    """2K rows from the latest scrape strictly before `cutoff` (or empty)."""
    row = con.execute(
        "SELECT max(scrape_date) FROM ratings_2k WHERE scrape_date < ?", [cutoff]
    ).fetchone()
    if not row or row[0] is None:
        return con.execute("SELECT * FROM ratings_2k WHERE false").fetchdf()
    return con.execute("SELECT * FROM ratings_2k WHERE scrape_date = ?", [row[0]]).fetchdf()


def trailing_player_stats(con, player_id: int, before: dt.date, seasons: list[str] | None = None):
    """A player's per-game sufficient stats for games strictly before `before`.
    Join through nba_games for the date. Use this (never a full-season mean) to
    build a pregame skill feature."""
    q = """
        SELECT s.* FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g USING (game_id)
        WHERE s.player_id = ? AND g.game_date < ? AND s.game_id LIKE '002%'
    """
    params: list = [player_id, before]
    if seasons:
        q += f" AND g.season IN ({','.join('?' * len(seasons))})"
        params += list(seasons)
    return con.execute(q, params).fetchdf()
