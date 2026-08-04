"""Soft-book prop-line capture: PrizePicks + Underdog keyless public endpoints.

These DFS-style books post player prop lines via open JSON APIs, are free to
log, and are the most realistic first edge target (softer pricing than
sportsbooks). Lines are NON-RECONSTRUCTIBLE — logging forward from day one
builds the archive that the H-B edge measurement needs.

Same discipline as the odds logger: append-only JSONL raw capture, never
touches DuckDB. Offseason: NBA sections are empty — poller logs a heartbeat
and exits cleanly. Cron every 30-60 min in-season.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os

import requests

from ..config import RAW

log = logging.getLogger("softbook")

RAW_SOFT = RAW / "softbook_props"
RAW_SOFT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0",
      "Accept": "application/json"}

PRIZEPICKS_URL = "https://api.prizepicks.com/projections"
PRIZEPICKS_NBA_LEAGUE = 7
UNDERDOG_URL = "https://api.underdogfantasy.com/beta/v5/over_under_lines"


def _append(kind: str, payload) -> int:
    day = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    path = RAW_SOFT / f"{day}.jsonl"
    rec = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(), "kind": kind,
           "data": payload}
    with path.open("a") as f:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        f.flush(); os.fsync(f.fileno())
    return 1


def poll_prizepicks() -> int:
    try:
        import cloudscraper   # PrizePicks sits behind Cloudflare
        s = cloudscraper.create_scraper()
        r = s.get(PRIZEPICKS_URL, timeout=30,
                  params={"league_id": PRIZEPICKS_NBA_LEAGUE,
                          "per_page": 500, "single_stat": "true"})
        r.raise_for_status()
        body = r.json()
        n = len(body.get("data", []))
        _append("prizepicks", body)
        log.info("prizepicks: %d NBA projections", n)
        return n
    except Exception as e:  # noqa: BLE001
        log.warning("prizepicks poll failed: %s", e)
        _append("prizepicks_error", str(e))
        return -1


def poll_underdog() -> int:
    try:
        r = requests.get(UNDERDOG_URL, headers=UA, timeout=30)
        r.raise_for_status()
        body = r.json()
        lines = body.get("over_under_lines", [])
        # compact extraction (full raw is ~8MB/poll = untenable; keep the fields
        # the H-B comparison needs: who/what/line/odds/status/timestamps)
        compact = []
        for l in lines:
            ou = l.get("over_under") or {}
            opts = [{"choice": o.get("choice"), "payout": o.get("payout_multiplier"),
                     "american": o.get("american_price")} for o in (l.get("options") or [])]
            compact.append({
                "id": l.get("id"), "title": ou.get("title"),
                "stat_value": l.get("stat_value"), "status": l.get("status"),
                "appearance_stat": (ou.get("appearance_stat") or {}).get("display_stat"),
                "options": opts,
            })
        _append("underdog", {"n": len(lines), "lines": compact})
        log.info("underdog: %d lines captured (compact)", len(lines))
        return len(lines)
    except Exception as e:  # noqa: BLE001
        log.warning("underdog poll failed: %s", e)
        _append("underdog_error", str(e))
        return -1


def poll_once() -> dict:
    return {"prizepicks": poll_prizepicks(), "underdog": poll_underdog()}
