"""Scheduler tests for the odds logger (D228).

The property that actually matters is the one the old cadence violated: the
budget must bound what a game night can spend.  The previous policy applied
`sleep = min(paced, CLOSE_WINDOW/3)` inside the close window, which let the
evening burst ignore the pacer entirely and overdraw the month by ~35%.  So the
central test here is `test_a_real_night_fits_the_real_budget`, and beside it the
guarantee that trimming can never buy that headroom by dropping the two prices
every published number depends on.
"""
from __future__ import annotations

import datetime as dt

import pytest

from nbapred.ingest.odds_sched import (
    CLOSE, EXTRA, ET, OPEN, T1H, T2H, T4H, UTC, Target, cost,
    daily_allowance, next_target, plan, prop_candidates, sleep_minutes,
    targets)


def _tip(y, m, d, hh, mm=0):
    """A tip at ET wall-clock, returned in UTC (what the API gives us)."""
    return dt.datetime(y, m, d, hh, mm, tzinfo=ET).astimezone(UTC)


def _night(*hours):
    return [_tip(2026, 11, 12, int(h), int(round((h % 1) * 60))) for h in hours]


# --------------------------------------------------------------- the ladder

def test_ladder_is_tip_relative_not_clock_relative():
    """The marks must track tip time. A 5PM-anchored spec would put the same
    wall-clock time at a different completion fraction for every game."""
    tip = _tip(2026, 11, 12, 22)          # a late game
    ts = targets([tip], want_extra=False)
    offsets = sorted(round((tip - t.when).total_seconds() / 60) for t in ts
                     if t.prio != OPEN)
    assert offsets == [15, 60, 120, 240]


def test_open_is_ten_am_et_and_falls_back_for_an_early_tip():
    ts = targets(_night(19), want_extra=False)
    op = next(t for t in ts if t.prio == OPEN)
    assert op.when.astimezone(ET).hour == 10

    noon = _tip(2026, 12, 25, 12)          # Christmas noon tip
    op2 = next(t for t in targets([noon], want_extra=False) if t.prio == OPEN)
    assert (noon - op2.when) == dt.timedelta(hours=4)


def test_clustered_tips_merge_into_shared_calls():
    """One call serves the whole slate, so 5 games must not cost 5x."""
    solo = targets(_night(19), want_extra=False)
    five = targets(_night(19, 19, 19, 19, 19), want_extra=False)
    assert len(five) == len(solo)


def test_a_close_cluster_anchors_early_never_after_a_tip():
    """Anchoring a merged close late would fire after the earlier game tipped
    and lose that close for good. Early is recoverable; late is not."""
    tips = _night(19, 19 + 10 / 60)        # tips 10 minutes apart
    ts = targets(tips, want_extra=False)
    closes = [t for t in ts if t.prio == CLOSE]
    assert len(closes) == 1
    assert closes[0].when <= min(tips) - dt.timedelta(minutes=15)


# --------------------------------------------------------------- trimming

def test_trimming_never_drops_open_or_close():
    tips = _night(19, 19.5, 20, 22, 22.5)
    for allowance in (1, 2, 3, 5, 8, 13, 21):
        ts = plan(tips, allowance=allowance)
        kinds = {t.prio for t in ts}
        assert OPEN in kinds, allowance
        assert sum(t.prio == CLOSE for t in ts) == \
            sum(t.prio == CLOSE for t in targets(tips)), allowance


def test_trimming_order_is_extra_then_the_shallowest_marks():
    tips = _night(19, 22)
    full = plan(tips, allowance=None)
    assert any(t.extra for t in full)

    # extras are a market upgrade, so they go before any call is dropped
    tight = plan(tips, allowance=cost(full, 1, 2) - 1)
    assert not any(t.extra for t in tight)
    assert len(tight) == len(full)

    # then T4H (the 76% mark) before T2H, and T2H before T1H (the 91% mark)
    order = []
    for a in range(cost(full, 1, 2), 0, -1):
        got = {t.prio for t in plan(tips, allowance=a)}
        for p in (T4H, T2H, T1H):
            if p not in got and p not in order:
                order.append(p)
    assert order == [T4H, T2H, T1H]


def test_a_real_night_fits_the_real_budget():
    """THE REGRESSION TEST. Ten staggered games on the shipped config must cost
    no more than the day's allowance -- the failure this replaces was an evening
    burst that ignored the pacer and overdrew a 500-credit month by ~35%."""
    tips = _night(19, 19, 19.5, 19.5, 20, 20, 22, 22, 22.5, 22.5)
    now = _tip(2026, 11, 12, 9)
    allowance = daily_allowance(480, now, monthly_budget=500)
    spend = cost(plan(tips, allowance=allowance), cost_core=1, cost_extra=2)
    assert spend <= allowance
    # and 25 such nights must not exhaust the month
    assert spend * 25 <= 500


def test_open_and_close_survive_even_an_impossible_allowance():
    tips = _night(19, 22)
    ts = plan(tips, allowance=1)
    assert {OPEN, CLOSE} <= {t.prio for t in ts}


# --------------------------------------------------------------- budget

def test_allowance_tracks_credits_left_not_the_nominal_budget():
    now = dt.datetime(2026, 11, 15, 12, tzinfo=UTC)     # ~15 days left
    rich = daily_allowance(450, now, monthly_budget=500)
    poor = daily_allowance(45, now, monthly_budget=500)
    assert rich > poor * 5
    assert daily_allowance(450, now, monthly_budget=0) is None


def test_allowance_handles_december_rollover():
    now = dt.datetime(2026, 12, 20, 12, tzinfo=UTC)
    assert daily_allowance(300, now, monthly_budget=500) > 0


# --------------------------------------------------------------- sleeping

def test_sleep_is_capped_so_the_loop_replans():
    from nbapred.ingest.odds_sched import MAX_SLEEP_MIN
    mins, why = sleep_minutes(dt.datetime(2026, 8, 1, 12, tzinfo=UTC), [])
    assert mins == MAX_SLEEP_MIN and "no further" in why


def test_sleep_lands_on_the_next_mark():
    tips = _night(19, 22)
    ts = plan(tips, allowance=None)
    now = _tip(2026, 11, 12, 9)
    mins, _ = sleep_minutes(now, ts)
    assert abs((now + dt.timedelta(minutes=mins)) - next_target(now, ts).when) \
        < dt.timedelta(seconds=1)


# --------------------------------------------------------------- props

def test_props_are_rationed_not_switched_off():
    """The register's props gap exists because props were disabled wholesale
    whenever a budget was set. A cap makes them affordable instead."""
    now = _tip(2026, 11, 12, 12)
    tips = {f"g{i}": t for i, t in enumerate(_night(19, 19.5, 20, 22, 22.5))}
    got = prop_candidates(tips, now, window_hrs=24, cap=2, already=set())
    assert len(got) == 2
    assert got == ["g0", "g1"]                     # soonest first
    assert prop_candidates(tips, now, window_hrs=24, cap=2,
                           already={"g0"}) == ["g1", "g2"]
    assert prop_candidates(tips, now, window_hrs=1, cap=2, already=set()) == []
    assert prop_candidates(tips, now, window_hrs=24, cap=0, already=set()) == []


def test_props_never_look_backwards():
    now = _tip(2026, 11, 12, 21)
    tips = {"early": _tip(2026, 11, 12, 19), "late": _tip(2026, 11, 12, 22)}
    assert prop_candidates(tips, now, window_hrs=24, cap=5,
                           already=set()) == ["late"]
