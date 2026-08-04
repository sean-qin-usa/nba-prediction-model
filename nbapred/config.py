"""Central config: paths, env, constants. No credentials in code — .env only."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_ODDS = RAW / "odds"            # append-only JSONL, one file per UTC day
RAW_NBA = RAW / "nba_api"          # cached raw endpoint responses
RAW_2K = RAW / "ratings_2k"        # archived raw HTML, versioned by scrape date
DB_PATH = DATA / "nba.duckdb"

for p in (RAW_ODDS, RAW_NBA, RAW_2K):
    p.mkdir(parents=True, exist_ok=True)


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Featured markets logged on every snapshot; props are per-event calls (credit-priced).
ODDS_SPORT = "basketball_nba"
ODDS_MARKETS_MAIN = "h2h,spreads,totals"
ODDS_REGIONS = os.environ.get("ODDS_REGIONS", "us")
ODDS_PROP_MARKETS = os.environ.get(
    "ODDS_PROP_MARKETS",
    "player_points,player_rebounds,player_assists,player_threes,player_points_rebounds_assists",
)

def current_season(today=None) -> str:
    """NBA season label for a date (default: today). Oct-Dec -> the season
    STARTING that year ('2026-27' for Oct 2026); Jan-Sep -> the season that
    started the prior calendar year (a July offseason date maps to the
    just-finished season, which is what trailing-data consumers want).
    Replaces the hardcoded literal that would have silently broken every
    default-season code path at the October 2026 opener (codex round 6 #4/#8)."""
    import datetime as _dt
    d = today or _dt.date.today()
    start = d.year if d.month >= 10 else d.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def prev_season(season: str) -> str:
    """'2025-26' -> '2024-25'."""
    y = int(season[:4])
    return f"{y - 1}-{y % 100:02d}"


CURRENT_SEASON = current_season()
STATS_TIMEOUT = 45
STATS_MIN_INTERVAL = 0.65  # seconds between stats.nba.com calls (plus jitter)
