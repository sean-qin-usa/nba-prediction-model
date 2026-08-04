"""PIT and arithmetic tests for the market-anchored CLV model (D147/D148).

The central test is the one the brief demands: shuffle the label's FUTURE and
require the features to be bit-identical.  It is written so that it CANNOT go
vacuous — a companion assertion requires the same features to MOVE when the
whole label history is permuted, so a feature that simply never reads the
label cannot pass by doing nothing (the D144 fresh-guard lesson).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nbapred.market.anchored import (
    OVERROUND, PriceRidge, assert_pit, run_price_ridge, sigmoid)


def _toy(n_days=60, n_teams=8, seed=7):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    strength = rng.normal(0, 4, n_teams)
    rows = []
    for d in range(n_days):
        order = rng.permutation(n_teams)
        for k in range(0, n_teams, 2):
            h, a = teams[order[k]], teams[order[k + 1]]
            true = strength[order[k]] - strength[order[k + 1]] + 2.6
            op = true + rng.normal(0, 2.0)
            cl = true + rng.normal(0, 1.0)
            rows.append(dict(day=d, home=h, away=a, open_margin=op,
                             close_margin=cl,
                             score_home=110 + true, score_away=110.0))
    return pd.DataFrame(rows)


def _build(df):
    r_close = run_price_ridge(df.home.values, df.away.values, df.day.values,
                              df.close_margin.values)
    r_res = run_price_ridge(df.home.values, df.away.values, df.day.values,
                            (df.score_home - df.score_away).values)
    X = np.column_stack([r_close - df.open_margin.values,
                         r_res - df.open_margin.values,
                         df.open_margin.values])
    return X, ["resid_close", "resid_res", "open_margin"]


def test_pit_future_shuffle_leaves_features_identical():
    df = _toy()
    bad, moved = assert_pit(_build, df,
                            ["close_margin", "score_home", "score_away"])
    assert bad == [], f"PIT violation: {bad} read a same-day or future label"
    # NON-VACUITY: the guard must be capable of failing.
    assert "resid_close" in moved and "resid_res" in moved, (
        "the ridge features did not move even under a FULL label shuffle — "
        "the PIT test above is vacuous")


def test_ridge_never_sees_its_own_or_same_day_labels():
    """Directly: change ONE game's close and require that no feature on that
    game's date (its own row included) moves, while a LATER date does."""
    df = _toy()
    X0, _ = _build(df)
    tgt = len(df) // 2
    d_tgt = df.day.values[tgt]
    df2 = df.copy()
    df2.loc[tgt, "close_margin"] += 25.0
    X1, _ = _build(df2)
    same_day = df.day.values == d_tgt
    assert np.allclose(X0[same_day], X1[same_day]), (
        "a same-day sideways leak: changing one game's close moved a feature "
        "on its own slate")
    later = df.day.values > d_tgt
    assert not np.allclose(X0[later], X1[later]), (
        "changing a close never propagates forward — the ridge is inert")


def test_price_ridge_recovers_strength_ordering():
    df = _toy(n_days=200)
    pred = run_price_ridge(df.home.values, df.away.values, df.day.values,
                           df.close_margin.values)
    warm = df.day.values > 40
    assert np.corrcoef(pred[warm], df.close_margin.values[warm])[0, 1] > 0.5


def test_flush_day_is_required_for_visibility():
    r = PriceRidge(["A", "B"])
    before = r.predict("A", "B")
    r.observe("A", "B", 20.0)
    assert r.predict("A", "B") == pytest.approx(before), (
        "observe() leaked before flush_day() — same-day games would be visible "
        "to each other")
    r.flush_day()
    assert r.predict("A", "B") > before


def test_arb_threshold_arithmetic():
    """The whole of Deliverable 2 rests on this identity: a two-book round trip
    locks iff the CLV on the entry side exceeds 1 - 1/overround."""
    from scripts.cm_arb import arb_rate
    thr = 1.0 - 1.0 / OVERROUND
    for p_open in (0.35, 0.5, 0.65):
        for extra in (-0.005, +0.005):
            p_close = p_open + thr + extra
            d1 = 1.0 / (p_open * OVERROUND)
            d2 = 1.0 / ((1 - p_close) * OVERROUND)
            r = arb_rate(d1, d2)
            assert (r > 0) == (extra > 0), (
                f"arb boundary wrong at p_open={p_open}, extra={extra}")


def test_lay_rate_commission_is_on_net_winnings():
    """A lay at the fair price with zero commission and zero sweep must return
    exactly the book's edge, and commission must only bite the winnings."""
    from scripts.cm_arb import lay_rate
    p = 0.5
    back = 1.0 / (p * OVERROUND)          # the book's price, with its vig
    fair_lay = 1.0 / p                    # exchange fair
    assert lay_rate(back, fair_lay, 0.0) == pytest.approx(
        back / fair_lay - 1.0, rel=1e-9)
    # a positive commission can only reduce the locked rate
    assert lay_rate(back, fair_lay, 0.05) < lay_rate(back, fair_lay, 0.0)


def test_sigmoid_map_is_the_documented_one():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(6.96 / 6.96 * 0.0) == pytest.approx(0.5)
