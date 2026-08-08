"""Poll scheduler for the odds logger — PURE (no network, no clock, no DB), so
every decision below is unit-testable without waiting for October.

WHY THE LADDER IS TIP-RELATIVE AND NOT CLOCK-RELATIVE.  The obvious spec is
"snapshot before and after the 5PM ET injury report".  `scripts/wlm_chart.py`
measured the within-day move and that spec is anchored to the wrong clock:

    share of open->close movement already completed
        T-4h  76%      T-2h  80%      T-1h  91%
        at the 5PM ET report:  ~75% ALREADY GONE, for early AND late games
        the only burst: T-2h -> tip, activity rate x3.5

5PM is a fixed clock landmark but tips are staggered 19:00-22:30 ET, so a
clock-anchored pair straddles a DIFFERENT completion fraction for every game and
lands after most of the information for all of them.  Movement is a function of
TIME TO TIP, so the ladder is too.

WHY ONE POLL SERVES THE WHOLE SLATE.  `/sports/{sport}/odds` returns every
upcoming event in one response, so a single wall-clock poll sits at a different
tip-relative offset for each game.  Targets are therefore generated per game and
then MERGED: a 10-game night's ~40 marks collapse to a handful of actual calls
because tips cluster.  This is what makes the ladder affordable.

WHAT GETS DROPPED FIRST WHEN THE BUDGET IS TIGHT.  OPEN is the price the strategy
transacts at and CLOSE is the CLV reference; every published number depends on
those two, and the intraday path is a research luxury.  So trimming runs from
EXTRA down and never touches OPEN or CLOSE.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = dt.timezone.utc

# Priority ladder. LOWER survives trimming; OPEN and CLOSE are never dropped.
OPEN, CLOSE, T1H, T2H, T4H, EXTRA = range(6)
KIND = {OPEN: "open", CLOSE: "close", T1H: "t1h",
        T2H: "t2h", T4H: "t4h", EXTRA: "extra"}

OPEN_HOUR_ET = 10        # first look of the day, once lines are posted
MIN_OPEN_LEAD_HRS = 4    # ...but never closer than this to the first tip
CLOSE_LEAD_MIN = 15      # last look before tip
MERGE_TOL_MIN = 20       # two marks this close share one call
MAX_SLEEP_MIN = 240      # re-plan at least this often, even on an empty day


@dataclass
class Target:
    when: dt.datetime          # UTC
    prio: int
    extra: bool = False        # poll the extra markets too (costs more)

    @property
    def kind(self) -> str:
        return KIND[self.prio]


def _open_time(tips: list[dt.datetime]) -> dt.datetime:
    """OPEN_HOUR_ET on the game date, pulled earlier if the first tip is early.

    A fixed 10:00 ET is only 2h ahead of a Christmas noon tip, which is inside
    the window where the line has already moved (T-2h = 80% complete), so the
    minimum-lead clamp is what keeps `open` meaning "before the move" rather
    than "10 o'clock".
    """
    first_et = min(tips).astimezone(ET)
    o = first_et.replace(hour=OPEN_HOUR_ET, minute=0, second=0, microsecond=0)
    o = min(o, first_et - dt.timedelta(hours=MIN_OPEN_LEAD_HRS))
    return o.astimezone(UTC)


def _merge(ts: list[Target], tol_min: int = MERGE_TOL_MIN) -> list[Target]:
    """Collapse marks within `tol_min` into one call.

    The cluster keeps the EARLIER time when priorities tie, which matters for
    CLOSE: anchoring a close cluster late would put the call after the earlier
    game's tip and lose that close entirely.  Early is recoverable, late is not.
    """
    out: list[Target] = []
    for t in sorted(ts, key=lambda x: (x.when, x.prio)):
        hit = next((o for o in out
                    if abs((t.when - o.when).total_seconds()) <= tol_min * 60),
                   None)
        if hit is None:
            out.append(t)
        else:
            hit.extra = hit.extra or t.extra
            if t.prio < hit.prio:
                hit.when, hit.prio = t.when, t.prio
    return sorted(out, key=lambda x: x.when)


def targets(tips: list[dt.datetime], *, want_extra: bool = True) -> list[Target]:
    """The full tip-relative ladder for one slate, merged."""
    if not tips:
        return []
    ts = [Target(_open_time(tips), OPEN, extra=want_extra)]
    for t in tips:
        ts.append(Target(t - dt.timedelta(minutes=CLOSE_LEAD_MIN), CLOSE))
        ts.append(Target(t - dt.timedelta(hours=1), T1H))
        ts.append(Target(t - dt.timedelta(hours=2), T2H))
        ts.append(Target(t - dt.timedelta(hours=4), T4H))
    return _merge(ts)


def cost(ts: list[Target], cost_core: int, cost_extra: int) -> int:
    return sum(cost_core + (cost_extra if t.extra else 0) for t in ts)


def plan(tips: list[dt.datetime], *, allowance: float | None,
         cost_core: int = 1, cost_extra: int = 2) -> list[Target]:
    """The ladder trimmed to fit `allowance` credits (None = no budget).

    Trimming drops the highest prio NUMBER first (EXTRA, then T4H, T2H, T1H) and
    stops at CLOSE.  If even OPEN+CLOSE exceeds the allowance they are kept
    anyway and the overrun is the caller's to report: losing the transaction
    price to save credits is never the right trade.
    """
    ts = targets(tips, want_extra=True)
    if allowance is None:
        return ts
    # extras are a market upgrade, not a call: drop them before dropping calls
    if cost(ts, cost_core, cost_extra) > allowance:
        for t in ts:
            t.extra = False
    for prio in (T4H, T2H, T1H):
        while cost(ts, cost_core, cost_extra) > allowance and \
                any(t.prio == prio for t in ts):
            ts.remove(next(t for t in reversed(ts) if t.prio == prio))
    return ts


def next_target(now: dt.datetime, ts: list[Target]) -> Target | None:
    return next((t for t in ts if t.when > now), None)


def sleep_minutes(now: dt.datetime, ts: list[Target]) -> tuple[float, str]:
    """Minutes until the next mark, and why. Capped so the loop re-plans."""
    nxt = next_target(now, ts)
    if nxt is None:
        return MAX_SLEEP_MIN, "no further marks today"
    mins = (nxt.when - now).total_seconds() / 60
    if mins <= 0:
        return 0.0, f"{nxt.kind} due now"
    return min(mins, MAX_SLEEP_MIN), f"next {nxt.kind} in {mins:.0f}m"


def daily_allowance(remaining: int | None, now: dt.datetime,
                    monthly_budget: int) -> float | None:
    """Credits this calendar day may spend, from what is ACTUALLY left.

    Driven by the live `x-requests-remaining` header rather than the nominal
    budget, so an overspend yesterday tightens today automatically instead of
    silently running the month dry.
    """
    if not monthly_budget or remaining is None:
        return None
    nxt_month = (now.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    month_end = nxt_month.replace(hour=0, minute=0, second=0, microsecond=0)
    days_left = max((month_end - now).total_seconds() / 86400, 0.25)
    return max(remaining / days_left, 1.0)


def prop_candidates(tips_by_event: dict[str, dt.datetime], now: dt.datetime,
                    *, window_hrs: float, cap: int,
                    already: set[str]) -> list[str]:
    """Up to `cap` events tipping soonest inside the window, not yet sampled.

    Props are per-event and therefore the only unbounded cost in the logger.  A
    hard nightly cap is what makes them affordable at all: the register's props
    gap exists because they were switched off wholesale rather than rationed.
    """
    live = [(t, e) for e, t in tips_by_event.items()
            if e not in already and 0 <= (t - now).total_seconds() / 3600 <= window_hrs]
    return [e for _, e in sorted(live)[:max(cap, 0)]]
