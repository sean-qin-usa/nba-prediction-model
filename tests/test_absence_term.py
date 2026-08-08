"""Identity tests for the D232 absence-response term.

The bug this guards against already happened once, in this project, twice. The
soft out-set is a DICT {player_id: P(out)}, and `len(dict)` counts everyone
carrying any doubt while `sum(dict.values())` is the expected number absent.
`prod_by_season.py` used len() while its own comment claimed it was the
expectation, and the first D232 gate run was materially weaker for it (t -2.82
against the pre-registered -4.96).

So: assert that `expected_out` sums a dict and counts a set, that the two forms
agree exactly where they should, and that the term is exactly zero on a
symmetric slate — the last one is what makes "inert when off" checkable without
a full pipeline run.
"""
from __future__ import annotations

import importlib

import pytest

from nbapred.model import absence


def test_expected_out_sums_a_dict_and_does_not_count_it():
    """THE CENTRAL ONE. len() would give 3 here; the expectation is 0.9."""
    d = {1: 0.5, 2: 0.3, 3: 0.1}
    assert absence.expected_out(d) == pytest.approx(0.9, abs=1e-12)
    assert absence.expected_out(d) != len(d)


def test_expected_out_counts_a_hard_set():
    assert absence.expected_out({1, 2, 3}) == pytest.approx(3.0, abs=1e-12)


def test_the_two_forms_agree_when_every_probability_is_one():
    """A hard set is the soft dict with all mass at 1.0; anything else means the
    pre-D201 and post-D201 paths are not comparable."""
    assert absence.expected_out({1: 1.0, 2: 1.0}) == \
        pytest.approx(absence.expected_out({1, 2}), abs=1e-12)


def test_a_dict_of_zeros_is_zero_expected_absences():
    """The failure mode from D229 restated for this term: a roster where nobody
    is likely out must contribute nothing, even though the dict is non-empty."""
    assert absence.expected_out({1: 0.0, 2: 0.0, 3: 0.0}) == \
        pytest.approx(0.0, abs=1e-12)


def test_empty_and_none_are_zero():
    for v in (None, {}, set()):
        assert absence.expected_out(v) == 0.0


def test_term_is_zero_on_a_symmetric_slate():
    """Equal expected absences must leave the margin untouched.

    "Exactly zero" holds when the two sums are BIT-equal, which `term()`
    short-circuits. It does NOT hold when they are only arithmetically equal:
    0.4 + 0.2 is 0.6000000000000001, not 0.6, so that case lands ~1e-16 off.
    Both are asserted, at the strength each actually has -- claiming bit-exact
    zero for the second would be a false invariant, and this test caught me
    writing one."""
    assert absence.term({1: 1.0}, {2: 1.0}) == 0.0          # bit-equal sums
    assert absence.term(None, None) == 0.0
    assert absence.term({1: 0.5}, {7: 0.5}) == 0.0
    # arithmetically equal, not bit-equal: negligible, not exact
    v = absence.term({1: 0.4, 2: 0.2}, {7: 0.6})
    assert v != 0.0
    assert abs(v) < 1e-13, v            # far under the D230b margin floor


def test_sign_is_negative_for_the_side_with_more_absences():
    """More HOME absences must LOWER the home margin. A sign flip here silently
    inverts the whole correction."""
    t = absence.term({1: 1.0, 2: 1.0}, {7: 0.0})
    assert t < 0.0
    # and it is linear in the differential
    t2 = absence.term({1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}, {7: 0.0})
    assert t2 == pytest.approx(2 * t, abs=1e-12)


def test_away_absences_raise_the_home_margin():
    assert absence.term({1: 0.0}, {7: 1.0, 8: 1.0}) > 0.0


def test_disabled_switch_returns_exact_zero(monkeypatch):
    monkeypatch.setenv("ABSENCE_TERM", "0")
    mod = importlib.reload(absence)
    try:
        assert mod.ENABLED is False
        assert mod.term({1: 1.0, 2: 1.0}, {}) == 0.0
    finally:
        monkeypatch.delenv("ABSENCE_TERM", raising=False)
        importlib.reload(absence)


def test_shipped_coefficient_is_negative_and_sane():
    c = absence.coefficients()
    beta = c["coefs"][0]
    assert -2.0 < beta < 0.0, beta
    # the walk-forward folds must bracket it in sign, or the frozen constant is
    # not the same quantity the gate validated
    assert all(b < 0 for b in c["walk_forward_betas"])
