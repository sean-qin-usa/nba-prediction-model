"""Identity tests for SOFT availability (D202) — the ones that were missing.

WHY THIS FILE EXISTS.  D202 shipped soft availability and the register recorded
that the 0 / 0.5 / 1.0 identities had been checked, but the checks were run
interactively and never committed.  The public mirror of this repository then
went out with `prod_by_season.py` passing `{player_id: p_out}` while its copy of
`composition.py` still did a membership test — so every player named in the dict
was dropped whole, at any probability.  That combination is worse than either
policy: it is not the pre-D202 hard rule (which received a set of ACTUAL outs)
and it is not soft availability.  Nothing failed, because nothing was asserting.

These are exact identities, not tolerances, and they are the cheapest possible
guard on the boundary between the two `out` forms:

    strength(dict{p: 0.0})  ==  strength(nobody out)       a 0% player is present
    strength(dict{p: 1.0})  ==  strength(set{p})           a 100% player is gone
    strength(dict{p: 0.5})  ==  midpoint of the two        and it is LINEAR

The third is the one that catches a membership test: a `pid in out` check passes
the first two by accident and can only fail on an interior probability.
"""
from __future__ import annotations

import datetime as dt

import pytest

from nbapred.model.composition import CompositionModel

TEAM, OTHER = 10, 20


def _model(n=3, talent=30.0, mins=24.0):
    m = CompositionModel.__new__(CompositionModel)
    m.asof = dt.date(2026, 1, 1)
    m.players = {}
    for pid in range(1, n + 1):
        m.players[pid] = {"team_id": TEAM, "talent": talent, "trail_min": mins,
                          "last_played": dt.date(2026, 1, 1)}
    m.players[99] = {"team_id": OTHER, "talent": talent, "trail_min": mins,
                     "last_played": dt.date(2026, 1, 1)}
    return m


def test_zero_probability_player_is_fully_present():
    m = _model()
    assert m.strength(TEAM, {1: 0.0}) == pytest.approx(m.strength(TEAM, None), abs=1e-12)


def test_certain_out_is_byte_identical_to_the_hard_set():
    """The set form must be preserved exactly, or D202's incumbent arm and every
    pre-D202 number stop being comparable."""
    m = _model()
    assert m.strength(TEAM, {1: 1.0}) == pytest.approx(m.strength(TEAM, {1}), abs=1e-12)
    assert m.strength(TEAM, {1: 1.0, 2: 1.0}) == pytest.approx(
        m.strength(TEAM, {1, 2}), abs=1e-12)


def test_half_probability_is_exactly_halfway():
    """THE TEST THAT CATCHES A MEMBERSHIP CHECK. `pid in out` passes the 0.0 and
    1.0 cases by accident; only an interior probability can expose it."""
    m = _model()
    full, gone = m.strength(TEAM, None), m.strength(TEAM, {1})
    assert m.strength(TEAM, {1: 0.5}) == pytest.approx((full + gone) / 2, abs=1e-12)


@pytest.mark.parametrize("p", [0.0, 0.1, 0.289, 0.5, 0.75, 0.9, 1.0])
def test_strength_is_linear_and_monotone_in_p_out(p):
    m = _model()
    full, gone = m.strength(TEAM, None), m.strength(TEAM, {1})
    assert m.strength(TEAM, {1: p}) == pytest.approx(full - p * (full - gone), abs=1e-12)


def test_a_dict_of_all_zeros_removes_nobody():
    """The public-mirror failure mode, stated as an assertion: prod_by_season
    hands over EVERY player carrying any probability, so a membership test wipes
    the roster rather than the out-set."""
    m = _model()
    everyone = {pid: 0.0 for pid in (1, 2, 3)}
    assert m.strength(TEAM, everyone) == pytest.approx(m.strength(TEAM, None), abs=1e-12)
    assert m.strength(TEAM, everyone) > 0.0


def test_margin_carries_the_soft_weighting_through():
    """Soft availability must survive the call the production margin actually
    makes, not just the leg it is unit-tested on."""
    m = _model()
    hard = m.margin(TEAM, OTHER, {1}, None)
    soft = m.margin(TEAM, OTHER, {1: 0.5}, None)
    none = m.margin(TEAM, OTHER, None, None)
    assert hard < soft < none
    assert soft == pytest.approx((hard + none) / 2, abs=1e-12)


def test_out_set_for_the_other_team_is_independent():
    m = _model()
    base = m.strength(OTHER, None)
    assert m.strength(OTHER, {1: 1.0}) == pytest.approx(base, abs=1e-12)
