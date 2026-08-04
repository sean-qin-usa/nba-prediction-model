"""Free supplementary line capture: ESPN's public scoreboard API embeds ESPN BET
odds (spread, total, moneylines) on upcoming/live events in-season. Zero auth,
zero credits — cron it as often as we like. One book only; complements the
budget-paced The Odds API logger (see docs/PAID_OPTIONS.md).

Raw scoreboard JSON archived per poll; odds blocks additionally appended to
lines.jsonl for cheap loading. Offseason scoreboards simply have no odds —
the poller is safe to run year-round. NOTE: the odds-bearing payload shape is
unverified until preseason (offseason events carry no `odds` key); the raw
archive guarantees nothing is lost if the flatten misses fields.
"""
from __future__ import annotations

import datetime as dt
import json
import logging

import requests

from ..config import RAW

log = logging.getLogger("espn_lines")

RAW_ESPN = RAW / "espn"
RAW_ESPN.mkdir(parents=True, exist_ok=True)

URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0"}


def poll_once(date: str | None = None) -> int:
    """Fetch scoreboard (optionally YYYYMMDD), archive raw, append odds lines.
    Returns number of events with odds attached."""
    now = dt.datetime.now(dt.timezone.utc)
    params = {"dates": date} if date else {}
    r = requests.get(URL, headers=UA, params=params, timeout=30)
    r.raise_for_status()
    body = r.json()

    day_dir = RAW_ESPN / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%H%M%S")
    (day_dir / f"scoreboard_{stamp}.json").write_text(json.dumps(body))

    n_odds = 0
    lines_path = RAW_ESPN / "lines.jsonl"
    for ev in body.get("events", []):
        for comp in ev.get("competitions", []):
            odds = comp.get("odds")
            if not odds:
                continue
            n_odds += 1
            with lines_path.open("a") as f:
                f.write(json.dumps({
                    "snapshot_ts": now.isoformat(),
                    "event_id": ev.get("id"),
                    "date": ev.get("date"),
                    "name": ev.get("shortName"),
                    "status": (ev.get("status") or {}).get("type", {}).get("name"),
                    "odds": odds,
                }, separators=(",", ":")) + "\n")
    log.info("scoreboard: %d events, %d with odds", len(body.get("events", [])), n_odds)
    return n_odds
