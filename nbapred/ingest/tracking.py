"""Player-tracking ingest (v1.5): individual defended-FG% by shot category.

LeagueDashPtDefend gives, per defender, opponents' FG% at the rim / on 3s / etc.
vs their normal rate (PCT_PLUSMINUS) — the INDIVIDUAL matchup signal def-RAPM
approximated. This is what unlocks 'who guards the shooter' in props (the one
place def-RAPM lost to raw allowance was team-aggregation; individual data fixes
that). Cached + rate-limited like all nba_api pulls.
"""
from __future__ import annotations

import logging

from .nba_stats import _frames, cached_endpoint

log = logging.getLogger("tracking")

CATEGORIES = ["Overall", "3 Pointers", "2 Pointers", "Less Than 6Ft", "Greater Than 15Ft"]


def pull_defended_fg(season: str) -> dict:
    """{category: DataFrame of per-defender defended-FG%} for a season."""
    from nba_api.stats.endpoints import leaguedashptdefend
    out = {}
    for cat in CATEGORIES:
        raw = cached_endpoint(
            leaguedashptdefend.LeagueDashPtDefend, "ptdefend", immutable=True,
            season=season, defense_category=cat, per_mode_simple="PerGame")
        frames = _frames(raw)
        if frames:
            out[cat] = next(iter(frames.values()))
    return out


SCHEMA = """
CREATE TABLE IF NOT EXISTS defended_fg (
    season      VARCHAR, category VARCHAR,
    defender_id BIGINT, defender_name VARCHAR,
    freq DOUBLE, d_fga DOUBLE, d_fg_pct DOUBLE, normal_fg_pct DOUBLE, pct_plusminus DOUBLE,
    ingest_ts TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (season, category, defender_id)
);
"""


def load(connect_fn, seasons: list[str]) -> int:
    import datetime as dt
    data = {s: pull_defended_fg(s) for s in seasons}  # network first
    now = dt.datetime.now(dt.timezone.utc)
    con = connect_fn(); con.execute(SCHEMA)
    n = 0
    for season, cats in data.items():
        for cat, df in cats.items():
            con.execute("DELETE FROM defended_fg WHERE season=? AND category=?", [season, cat])
            # non-Overall categories use stat-specific column names (D_FG3A etc.)
            cols = {c.upper(): c for c in df.columns}

            def pick(row, *names, default=None):
                for nm in names:
                    if nm in cols:
                        return getattr(row, cols[nm])
                return default

            for r in df.itertuples():
                dfga = pick(r, "D_FGA", "D_FG3A", "D_FG2A", "LT_06_FGA", "GT_15_FGA")
                dfgp = pick(r, "D_FG_PCT", "D_FG3_PCT", "D_FG2_PCT", "LT_06_PCT", "GT_15_PCT")
                nfgp = pick(r, "NORMAL_FG_PCT", "NORMAL_FG3_PCT", "NORMAL_FG2_PCT")
                pm = pick(r, "PCT_PLUSMINUS", "FG3_PCT_PLUSMINUS", "FG2_PCT_PLUSMINUS",
                          "LT_06_PCT_PLUSMINUS", "GT_15_PCT_PLUSMINUS", default=0.0)
                if dfga is None:
                    continue
                con.execute("INSERT INTO defended_fg VALUES (?,?,?,?,?,?,?,?,?,?)", [
                    season, cat, int(r.CLOSE_DEF_PERSON_ID), r.PLAYER_NAME,
                    float(r.FREQ), float(dfga), float(dfgp or 0),
                    float(nfgp or 0), float(pm or 0), now])
                n += 1
    con.close()
    log.info("defended_fg: %d rows over %d seasons", n, len(seasons))
    return n
