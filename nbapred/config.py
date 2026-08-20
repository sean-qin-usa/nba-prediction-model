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

# MARKETS ARE TIERED BECAUSE CREDITS ARE PRICED PER MARKET PER REGION (D228).
# The old single "h2h,spreads,totals" cost 3 credits on EVERY poll, which is what
# made the four-snapshot ladder unaffordable on the free tier. The sides strategy
# transacts on SPREADS ONLY -- nothing in nbapred/ or bet_engine reads h2h or
# totals from the live feed -- so the core poll is 1 credit and the extras ride
# along once a day for the record. That is a 3x cadence increase for no money.
ODDS_SPORT = "basketball_nba"
ODDS_MARKETS_CORE = os.environ.get("ODDS_MARKETS_CORE", "spreads")
ODDS_MARKETS_EXTRA = os.environ.get("ODDS_MARKETS_EXTRA", "h2h,totals")
ODDS_MARKETS_MAIN = ",".join(x for x in (ODDS_MARKETS_CORE, ODDS_MARKETS_EXTRA) if x)
ODDS_REGIONS = os.environ.get("ODDS_REGIONS", "us")
# Props are per-EVENT, so they are the only unbounded cost in the logger. They
# are rationed by a nightly event cap against genuine leftover allowance rather
# than switched off wholesale -- the wholesale switch is why no prop price has
# ever been logged, and why "market-offset props" is still unexplored.
# ONE market by default, because a prop event costs markets x regions: at three
# markets an event is 3 credits and props lose every leftover contest against the
# sides ladder, which is how they end up at zero again. player_points is the most
# liquid line and the one the offset-on-props test needs first; widen only when
# the budget is observed to carry it.
ODDS_PROP_MARKETS = os.environ.get("ODDS_PROP_MARKETS", "player_points")
ODDS_PROP_EVENTS_PER_NIGHT = int(os.environ.get("ODDS_PROP_EVENTS_PER_NIGHT", 2))

def current_season(today=None) -> str:
    """NBA season label for a date (default: today). Oct-Dec -> the season
    STARTING that year ('2026-27' for Oct 2026); Jan-Sep -> the season that
    started the prior calendar year (a July offseason date maps to the
    just-finished season, which is what trailing-data consumers want).
    Replaces the hardcoded literal that would have silently broken every
    default-season code path at the October 2026 opener (external review round 6 #4/#8)."""
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
