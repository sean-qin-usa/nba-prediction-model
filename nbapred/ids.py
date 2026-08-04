"""Player-ID crosswalk: NBA (stats.nba.com player_id) <-> 2K (2kratings name)
<-> odds (The Odds API prop outcome description). Build early, test hard.

Strategy: normalized-name join (casefold, strip accents/punct/suffixes), then
manual overrides from data/xwalk_overrides.csv (nba_player_id,name_2k,name_odds).
Odds names are filled lazily as they appear in logged prop quotes.
"""
from __future__ import annotations

import csv
import datetime as dt
import logging
import re
import unicodedata

from .config import DATA

log = logging.getLogger("ids")

OVERRIDES_CSV = DATA / "xwalk_overrides.csv"

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z ]", " ", s.casefold())
    parts = [p for p in s.split() if p not in _SUFFIXES]
    return " ".join(parts)


def _load_overrides() -> dict[int, dict]:
    if not OVERRIDES_CSV.exists():
        return {}
    out = {}
    with OVERRIDES_CSV.open() as f:
        for row in csv.DictReader(f):
            out[int(row["nba_player_id"])] = row
    return out


def build_crosswalk(con) -> dict:
    """Match active NBA players to the latest 2K scrape. Returns match stats."""
    nba = con.execute(
        "SELECT player_id, full_name FROM nba_players WHERE is_active").fetchall()
    twok = con.execute("""
        SELECT DISTINCT player_name FROM ratings_2k
        WHERE scrape_date = (SELECT max(scrape_date) FROM ratings_2k)
    """).fetchall()
    twok_by_norm: dict[str, str] = {}
    for (name,) in twok:
        twok_by_norm.setdefault(norm_name(name), name)

    overrides = _load_overrides()
    now = dt.datetime.now(dt.timezone.utc)
    stats = {"exact": 0, "normalized": 0, "manual": 0, "unmatched": 0}
    twok_exact = {name for (name,) in twok}

    con.execute("DELETE FROM player_xwalk")
    for pid, full_name in nba:
        nn = norm_name(full_name)
        if pid in overrides and overrides[pid].get("name_2k"):
            name_2k, method = overrides[pid]["name_2k"], "manual"
        elif full_name in twok_exact:
            name_2k, method = full_name, "exact"
        elif nn in twok_by_norm:
            name_2k, method = twok_by_norm[nn], "normalized"
        else:
            name_2k, method = None, "unmatched"
        stats[method] += 1
        name_odds = overrides.get(pid, {}).get("name_odds") or None
        con.execute("INSERT INTO player_xwalk VALUES (?,?,?,?,?,?,?)",
                    [pid, full_name, nn, name_2k, method, name_odds, now])
    return stats


def unmatched_2k(con) -> list[str]:
    """2K names (latest scrape) not claimed by any NBA player — inspection aid."""
    rows = con.execute("""
        SELECT DISTINCT r.player_name FROM ratings_2k r
        WHERE r.scrape_date = (SELECT max(scrape_date) FROM ratings_2k)
          AND r.player_name NOT IN (
            SELECT name_2k FROM player_xwalk WHERE name_2k IS NOT NULL)
        ORDER BY 1
    """).fetchall()
    return [r[0] for r in rows]
