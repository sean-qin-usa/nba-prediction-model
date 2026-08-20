"""Always-on odds logger (The Odds API v4).

Uptime-critical component: it does exactly one thing — poll and append raw JSON
to data/raw/odds/YYYY-MM-DD.jsonl. It never opens DuckDB (single-writer rule);
scripts/load_odds.py batch-loads the JSONL later. Crash-safe by construction:
every poll is one appended line, partial lines impossible (write+flush+fsync).

CADENCE (D228). Poll times come from `odds_sched.plan()`, a pure function of the
slate's tip times and the credits actually left. See that module for why the
ladder is tip-relative rather than anchored to the 5PM ET report clock.

WHAT THIS REPLACES, AND WHY IT MATTERED.  The previous policy computed a
budget-paced sleep and then did

    if a tip is within CLOSE_WINDOW_MIN:
        sleep_min = min(sleep_min, CLOSE_WINDOW_MIN / 3)

so the evening burst OVERRODE the pacer.  At 3 credits a poll and ~9 polls a
night that is 27 credits/night, or 675 across 25 game nights against a 500
budget — a 35% overdraw.  When the credits ran out `_get` raised, the handler
retried every 2 minutes forever, and the unit stayed `active` while capturing
nothing.  Now the budget bounds the plan and the plan bounds the polling, so
the burst cannot escape it; when credits get tight the ladder degrades to
open + close and stops there rather than running the month dry.

PROPS.  Previously gated on `MONTHLY_BUDGET == 0`, i.e. unreachable in the
shipped configuration and the reason no prop price has ever been logged. Now
rationed against leftover daily allowance under a nightly event cap, so they
degrade first and never starve the sides ladder.
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
    ODDS_MARKETS_CORE,
    ODDS_MARKETS_MAIN,
    ODDS_PROP_EVENTS_PER_NIGHT,
    ODDS_PROP_MARKETS,
    ODDS_REGIONS,
    ODDS_SPORT,
    RAW_ODDS,
)
from .odds_sched import (
    ET,
    UTC,
    daily_allowance,
    next_target,
    plan,
    prop_candidates,
    sleep_minutes,
)

log = logging.getLogger("odds_logger")

PROP_WINDOW_HRS = float(os.environ.get("ODDS_PROP_WINDOW_HRS", 6))
IDLE_SLEEP_MIN = float(os.environ.get("ODDS_IDLE_SLEEP_MIN", 720))  # offseason
# Free-tier pacing: total credits per calendar month (0 = paid tier, no pacing).
MONTHLY_BUDGET = int(os.environ.get("ODDS_MONTHLY_BUDGET", 0))

_N_REGIONS = len(ODDS_REGIONS.split(","))
COST_CORE = len(ODDS_MARKETS_CORE.split(",")) * _N_REGIONS
COST_EXTRA = (len(ODDS_MARKETS_MAIN.split(",")) * _N_REGIONS) - COST_CORE
COST_PROP_EVENT = len(ODDS_PROP_MARKETS.split(",")) * _N_REGIONS

_stop = False


def _handle_sig(signum, frame):  # noqa: ARG001
    global _stop
    _stop = True
    log.info("signal %s received, finishing current poll then exiting", signum)


def _utcnow() -> dt.datetime:
    return dt.datetime.now(UTC)


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


def poll_main(markets: str, target_kind: str = "main") -> tuple[list, dict]:
    """Snapshot featured markets for all upcoming NBA events.

    `target_kind` is stamped into the record so the ladder position (open /
    t4h / t2h / t1h / close) survives into the JSONL and does not have to be
    re-derived against tip times at load time.
    """
    body, quota = _get(
        f"/sports/{ODDS_SPORT}/odds",
        {"regions": ODDS_REGIONS, "markets": markets, "oddsFormat": "decimal"},
    )
    _append({
        "snapshot_ts": _utcnow().isoformat(),
        "kind": "main",
        "target_kind": target_kind,
        "sport": ODDS_SPORT,
        "regions": ODDS_REGIONS,
        "markets": markets,
        "quota": quota,
        "data": body,
    })
    return body, quota


def poll_props(event_id: str) -> dict:
    body, quota = _get(
        f"/sports/{ODDS_SPORT}/events/{event_id}/odds",
        {"regions": ODDS_REGIONS, "markets": ODDS_PROP_MARKETS,
         "oddsFormat": "decimal"},
    )
    _append({
        "snapshot_ts": _utcnow().isoformat(),
        "kind": "props",
        "target_kind": "props",
        "event_id": event_id,
        "regions": ODDS_REGIONS,
        "markets": ODDS_PROP_MARKETS,
        "quota": quota,
        "data": body,
    })
    return quota


def _tip(ev: dict) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
    except (KeyError, ValueError, AttributeError):
        return None


def next_slate(events: list, now: dt.datetime) -> tuple[list, dict]:
    """Tips of the earliest ET game-date that still has a future game.

    Planning one slate at a time is what keeps the daily allowance meaningful:
    generating targets for every event the API returns (often days ahead) would
    inflate the day's cost and over-trim tonight's ladder.
    """
    fut = [(t, ev) for ev in events
           if (t := _tip(ev)) is not None and t > now - dt.timedelta(hours=4)]
    if not fut:
        return [], {}
    date0 = min(t for t, _ in fut).astimezone(ET).date()
    sel = [(t, ev) for t, ev in fut if t.astimezone(ET).date() == date0]
    return [t for t, _ in sel], {ev["id"]: t for t, ev in sel if ev.get("id")}


def run_forever() -> None:
    if not ODDS_API_KEY:
        raise SystemExit("ODDS_API_KEY not set (put it in .env) — refusing to start.")
    signal.signal(signal.SIGTERM, _handle_sig)
    signal.signal(signal.SIGINT, _handle_sig)
    log.info("cadence: core=%s (%d cr) extra=+%d cr props=%s (%d cr/event) "
             "budget=%s", ODDS_MARKETS_CORE, COST_CORE, COST_EXTRA,
             ODDS_PROP_MARKETS, COST_PROP_EVENT, MONTHLY_BUDGET or "none")

    kind, want_extra = "open", False        # the first poll of a run
    props_done: dict[str, set] = {}         # ET date -> event ids sampled
    plan_cache: list = []
    need_poll = True

    while not _stop:
        try:
            now = _utcnow()
            if not need_poll:
                # A CAPPED WAKE, NOT AN ARRIVAL. `sleep_minutes` bounds a sleep at
                # MAX_SLEEP_MIN so the loop re-plans on a long quiet stretch, but
                # waking is not the same as a target falling due -- polling here
                # would spend a credit for nothing and break the invariant the
                # budget rests on, that polls == plan targets.
                sleep_min, why = sleep_minutes(now, plan_cache)
                nxt = next_target(now, plan_cache)
                need_poll = nxt is not None and \
                    sleep_min >= (nxt.when - now).total_seconds() / 60 - 1e-9
                if nxt is None:
                    need_poll = True
                else:
                    kind, want_extra = nxt.kind, bool(nxt.extra)
                log.info("waiting %.0fm — %s", sleep_min, why)
                for _ in range(int(sleep_min * 60)):
                    if _stop:
                        break
                    time.sleep(1)
                continue

            markets = ODDS_MARKETS_MAIN if want_extra else ODDS_MARKETS_CORE
            events, quota = poll_main(markets, target_kind=kind)
            remaining = int(quota["requests_remaining"] or 0)
            now = _utcnow()
            tips, tips_by_event = next_slate(events, now)
            log.info("%s snapshot: %d events, %d on the next slate, %s credits left",
                     kind, len(events), len(tips), remaining)

            if not events:
                plan_cache = []
                sleep_min, why = IDLE_SLEEP_MIN, "no events listed (offseason)"
                kind, want_extra, need_poll = "open", False, True
            else:
                allowance = daily_allowance(remaining, now, MONTHLY_BUDGET)
                p = plan(tips, allowance=allowance,
                         cost_core=COST_CORE, cost_extra=COST_EXTRA)

                # Props ride on GENUINE leftovers: whatever the sides ladder did
                # not claim. They therefore vanish first as credits tighten and
                # can never starve open/close.
                if allowance is not None:
                    spent = sum(COST_CORE + (COST_EXTRA if t.extra else 0)
                                for t in p)
                    left = allowance - spent
                else:
                    left = float("inf")
                day = now.astimezone(ET).date().isoformat()
                done = props_done.setdefault(day, set())
                # NIGHTLY cap, so it must be net of what tonight already spent:
                # `prop_candidates` only excludes events already sampled, so a
                # raw cap here would sample that many MORE on every main poll --
                # 2/night became 10/night and a 2.4x budget overdraw.
                budget_events = (ODDS_PROP_EVENTS_PER_NIGHT
                                 if left == float("inf")
                                 else int(left // COST_PROP_EVENT))
                cap = min(ODDS_PROP_EVENTS_PER_NIGHT - len(done), budget_events)
                for eid in prop_candidates(tips_by_event, now,
                                           window_hrs=PROP_WINDOW_HRS,
                                           cap=cap, already=done):
                    q = poll_props(eid)
                    done.add(eid)
                    remaining = int(q["requests_remaining"] or 0)
                    log.info("props: event %s sampled, %s credits left",
                             eid, remaining)

                plan_cache = p
                sleep_min, why = sleep_minutes(now, p)
                nxt = next_target(now, p)
                kind = nxt.kind if nxt else "open"
                want_extra = bool(nxt.extra) if nxt else False
                # only the NEXT wake that actually reaches the target may poll
                need_poll = nxt is None or \
                    sleep_min >= (nxt.when - now).total_seconds() / 60 - 1e-9
                if len(props_done) > 3:      # keep the dict from growing forever
                    for k in sorted(props_done)[:-3]:
                        props_done.pop(k, None)

            log.info("sleeping %.0fm — %s", sleep_min, why)
        except Exception:
            log.exception("poll failed; retrying in 2 min")
            sleep_min = 2.0
            kind, want_extra, need_poll = "open", False, True

        for _ in range(int(sleep_min * 60)):
            if _stop:
                break
            time.sleep(1)

    log.info("odds logger stopped cleanly")
