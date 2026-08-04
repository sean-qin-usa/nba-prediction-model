"""Always-on odds logger (The Odds API v4).

Uptime-critical component: it does exactly one thing — poll and append raw JSON
to data/raw/odds/YYYY-MM-DD.jsonl. It never opens DuckDB (single-writer rule);
scripts/load_odds.py batch-loads the JSONL later. Crash-safe by construction:
every poll is one appended line, partial lines impossible (single write+flush).

Cadence policy (H-A needs open, moves, close):
  - main lines (h2h/spreads/totals): every MAIN_INTERVAL_MIN minutes, tightened
    to CLOSE_INTERVAL_MIN when any event starts within CLOSE_WINDOW_MIN.
  - player props: per-event calls (credit-priced) only within PROP_WINDOW_HRS
    of tip, at PROP_INTERVAL_MIN cadence.
  - credit guard: if x-requests-remaining falls below CREDIT_FLOOR, props stop
    first, then main polling degrades to hourly. Never silently dies.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import signal
import time
from pathlib import Path

import requests

from ..config import (
    ODDS_API_BASE,
    ODDS_API_KEY,
    ODDS_MARKETS_MAIN,
    ODDS_PROP_MARKETS,
    ODDS_REGIONS,
    ODDS_SPORT,
    RAW_ODDS,
)

log = logging.getLogger("odds_logger")

MAIN_INTERVAL_MIN = float(os.environ.get("ODDS_MAIN_INTERVAL_MIN", 30))
CLOSE_INTERVAL_MIN = float(os.environ.get("ODDS_CLOSE_INTERVAL_MIN", 5))
CLOSE_WINDOW_MIN = float(os.environ.get("ODDS_CLOSE_WINDOW_MIN", 90))
PROP_WINDOW_HRS = float(os.environ.get("ODDS_PROP_WINDOW_HRS", 24))
PROP_INTERVAL_MIN = float(os.environ.get("ODDS_PROP_INTERVAL_MIN", 60))
CREDIT_FLOOR = int(os.environ.get("ODDS_CREDIT_FLOOR", 500))
# Free-tier pacing: total credits per calendar month (0 = paid tier, no pacing).
# When set, the sleep between main polls is stretched so the month's budget
# lasts: polls/day = (remaining/days_left)/cost_per_poll. Props are disabled
# whenever the monthly budget is set (they'd eat it in one evening).
MONTHLY_BUDGET = int(os.environ.get("ODDS_MONTHLY_BUDGET", 0))
_MAIN_POLL_COST = len(ODDS_MARKETS_MAIN.split(",")) * len(ODDS_REGIONS.split(","))

_stop = False


def _handle_sig(signum, frame):  # noqa: ARG001
    global _stop
    _stop = True
    log.info("signal %s received, finishing current poll then exiting", signum)


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _append(record: dict) -> Path:
    day = _utcnow().strftime("%Y-%m-%d")
    path = RAW_ODDS / f"{day}.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return path


def _get(endpoint: str, params: dict) -> tuple[object, dict]:
    """One API call. Returns (json_body, quota_headers). Raises on HTTP error."""
    params = {"apiKey": ODDS_API_KEY, **params}
    r = requests.get(f"{ODDS_API_BASE}{endpoint}", params=params, timeout=30)
    quota = {
        "requests_remaining": r.headers.get("x-requests-remaining"),
        "requests_used": r.headers.get("x-requests-used"),
        "requests_last": r.headers.get("x-requests-last"),
    }
    r.raise_for_status()
    return r.json(), quota


def poll_main() -> tuple[list, dict]:
    """Snapshot featured markets for all upcoming NBA events (1 credit/market-region)."""
    body, quota = _get(
        f"/sports/{ODDS_SPORT}/odds",
        {"regions": ODDS_REGIONS, "markets": ODDS_MARKETS_MAIN, "oddsFormat": "decimal"},
    )
    _append({
        "snapshot_ts": _utcnow().isoformat(),
        "kind": "main",
        "sport": ODDS_SPORT,
        "regions": ODDS_REGIONS,
        "markets": ODDS_MARKETS_MAIN,
        "quota": quota,
        "data": body,
    })
    return body, quota


def poll_props(event_id: str) -> dict:
    body, quota = _get(
        f"/sports/{ODDS_SPORT}/events/{event_id}/odds",
        {"regions": ODDS_REGIONS, "markets": ODDS_PROP_MARKETS, "oddsFormat": "decimal"},
    )
    _append({
        "snapshot_ts": _utcnow().isoformat(),
        "kind": "props",
        "event_id": event_id,
        "regions": ODDS_REGIONS,
        "markets": ODDS_PROP_MARKETS,
        "quota": quota,
        "data": body,
    })
    return quota


def _budget_paced_sleep_min(remaining: int) -> float:
    """Minutes between main polls so `remaining` credits last through the month."""
    now = _utcnow()
    if now.month == 12:
        month_end = now.replace(year=now.year + 1, month=1, day=1)
    else:
        month_end = now.replace(month=now.month + 1, day=1)
    month_end = month_end.replace(hour=0, minute=0, second=0, microsecond=0)
    days_left = max((month_end - now).total_seconds() / 86400, 0.25)
    polls_per_day = max((remaining / days_left) / _MAIN_POLL_COST, 1.0)
    return max(1440.0 / polls_per_day, CLOSE_INTERVAL_MIN)


def _minutes_to_next_tip(events: list) -> float | None:
    now = _utcnow()
    best = None
    for ev in events:
        try:
            t = dt.datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        mins = (t - now).total_seconds() / 60
        if mins > -240 and (best is None or mins < best):  # ignore long-finished games
            best = mins
    return best


def run_forever() -> None:
    if not ODDS_API_KEY:
        raise SystemExit("ODDS_API_KEY not set (put it in .env) — refusing to start.")
    signal.signal(signal.SIGTERM, _handle_sig)
    signal.signal(signal.SIGINT, _handle_sig)
    last_props: dict[str, float] = {}  # event_id -> monotonic ts of last prop poll
    remaining = None

    while not _stop:
        try:
            events, quota = poll_main()
            remaining = int(quota["requests_remaining"] or 0)
            log.info("main snapshot: %d events, %s credits left", len(events), remaining)

            if MONTHLY_BUDGET == 0 and remaining > CREDIT_FLOOR:
                now_mono = time.monotonic()
                for ev in events:
                    t = dt.datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
                    hrs = (t - _utcnow()).total_seconds() / 3600
                    if 0 <= hrs <= PROP_WINDOW_HRS:
                        if now_mono - last_props.get(ev["id"], 0) >= PROP_INTERVAL_MIN * 60:
                            q = poll_props(ev["id"])
                            last_props[ev["id"]] = now_mono
                            remaining = int(q["requests_remaining"] or 0)
                            if remaining <= CREDIT_FLOOR:
                                log.warning("credit floor hit (%s) — props paused", remaining)
                                break
            mins_next = _minutes_to_next_tip(events)
            if MONTHLY_BUDGET:
                sleep_min = _budget_paced_sleep_min(remaining)
                # spend a burst near tip (close capture) even on a tight budget,
                # by taking the paced sleep only when no close window is near
                if mins_next is not None and 0 <= mins_next <= CLOSE_WINDOW_MIN:
                    sleep_min = min(sleep_min, CLOSE_WINDOW_MIN / 3)
            elif remaining is not None and remaining <= CREDIT_FLOOR:
                sleep_min = 60.0
            elif mins_next is not None and mins_next <= CLOSE_WINDOW_MIN:
                sleep_min = CLOSE_INTERVAL_MIN
            else:
                sleep_min = MAIN_INTERVAL_MIN
        except Exception:
            log.exception("poll failed; retrying in 2 min")
            sleep_min = 2.0

        for _ in range(int(sleep_min * 60)):
            if _stop:
                break
            time.sleep(1)

    log.info("odds logger stopped cleanly")
