"""Switches introduced by the three queued props / star-out gates.

The gates are adjudicated in DECISIONS.md; what these tests protect is the
INVARIANT that every one of the switches is a NO-OP at its default, plus the
arithmetic each mode is supposed to implement. A gate's same-run control is
only trustworthy if the default path is bitwise what it was before the switch
existed (GATE_POLICY_V2 §6.6), and a switch that silently changes production
when unset would invalidate every control in the register.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nbapred.engine import props, starout  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env():
    keep = {k: os.environ.pop(k, None)
            for k in ("STAROUT_TRAIL", "STAROUT_USAGE", "PROPS_CHANNEL_RAMP")}
    props.CHANNEL_LAM["reb"] = 1.0
    props.CHANNEL_LAM["ast"] = 1.0
    yield
    for k, v in keep.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    props.CHANNEL_LAM["reb"] = 1.0
    props.CHANNEL_LAM["ast"] = 1.0


def _rates(delta=0.0):
    return {
        "proj_min": 30.0, "sd_min": 4.0,
        "minutes_hist": [24.0, 28.0, 30.0, 33.0, 35.0, 31.0],
        "rate_rim": 0.20, "rate_mid": 0.10, "rate_thr": 0.15,
        "fg_rim": 0.62, "fg_mid": 0.44, "fg_thr": 0.36,
        "fta_per_min": 0.12, "ft_pct": 0.80,
        "reb_per_min": 0.18, "ast_per_min": 0.14, "n_games": 20,
        "ramp_delta": delta,
    }


# --------------------------------------------------------------- mode parsing
def test_modes_default_to_shipped_behaviour():
    assert starout.trail_mode() == "current"
    assert starout.usage_mode() == "softmax"
    assert props.channel_ramp_mode() == "0"


def test_unknown_mode_values_fall_back_to_shipped():
    """An operator typo must never silently select an ungated construction."""
    os.environ["STAROUT_TRAIL"] = "PLAYED_ONLY_MAYBE"
    os.environ["STAROUT_USAGE"] = "nullu"          # missing underscore
    os.environ["PROPS_CHANNEL_RAMP"] = "C"
    assert starout.trail_mode() == "current"
    assert starout.usage_mode() == "softmax"
    assert props.channel_ramp_mode() == "0"


def test_modes_are_reachable_when_set_exactly():
    os.environ["STAROUT_TRAIL"] = "played"
    assert starout.trail_mode() == "played"
    os.environ["STAROUT_TRAIL"] = "floor"
    assert starout.trail_mode() == "floor"
    os.environ["STAROUT_USAGE"] = "null_u"
    assert starout.usage_mode() == "null_u"
    os.environ["PROPS_CHANNEL_RAMP"] = "A"
    assert props.channel_ramp_mode() == "A"
    os.environ["PROPS_CHANNEL_RAMP"] = "b"          # case-insensitive
    assert props.channel_ramp_mode() == "B"


# ------------------------------------------------- channel ramp: no-op at rest
def test_channel_ramp_is_bitwise_noop_when_unset():
    r = _rates(delta=3.0)
    a = props.simulate_player(r, 3000, seed=11)
    props.CHANNEL_LAM["reb"] = 0.5                  # would move rebounds if ON
    props.CHANNEL_LAM["ast"] = 2.0
    b = props.simulate_player(r, 3000, seed=11)
    for k in ("points", "threes", "rebounds", "assists"):
        assert np.array_equal(a[k], b[k]), k


def test_channel_ramp_is_bitwise_noop_when_delta_is_zero():
    """The term must be exactly zero outside its window (delta == 0 rows)."""
    r = _rates(delta=0.0)
    a = props.simulate_player(r, 3000, seed=12)
    os.environ["PROPS_CHANNEL_RAMP"] = "A"
    props.CHANNEL_LAM["reb"] = 0.5
    props.CHANNEL_LAM["ast"] = 2.0
    b = props.simulate_player(r, 3000, seed=12)
    for k in ("points", "threes", "rebounds", "assists"):
        assert np.array_equal(a[k], b[k]), k


def test_points_and_threes_are_bitwise_fixed_in_every_channel_mode():
    """The GATE-3 veto, as an invariant: points/threes are read off the SAME
    zone-attempt draws, which are consumed BEFORE the channel block, so no
    channel mode may perturb them."""
    r = _rates(delta=3.0)
    base = props.simulate_player(r, 3000, seed=13)
    for mode in ("A", "B"):
        os.environ["PROPS_CHANNEL_RAMP"] = mode
        props.CHANNEL_LAM["reb"] = 0.60
        props.CHANNEL_LAM["ast"] = 1.40
        got = props.simulate_player(r, 3000, seed=13)
        assert np.array_equal(base["points"], got["points"]), mode
        assert np.array_equal(base["threes"], got["threes"]), mode
        # ... and rebounds MUST move, or the switch is not implementing the arm
        assert not np.array_equal(base["rebounds"], got["rebounds"]), mode


def test_channel_modes_A_and_B_are_first_moment_matched():
    """ARM B is specified as ARM A's magnitude-matched dispersion control, so
    their means must agree while their draws differ (D133's location-vs-spread
    ablation, applied per channel)."""
    r = _rates(delta=4.0)
    props.CHANNEL_LAM["reb"] = 0.70
    props.CHANNEL_LAM["ast"] = 1.30
    os.environ["PROPS_CHANNEL_RAMP"] = "A"
    a = props.simulate_player(r, 200_000, seed=14)
    os.environ["PROPS_CHANNEL_RAMP"] = "B"
    b = props.simulate_player(r, 200_000, seed=14)
    assert abs(a["rebounds"].mean() - b["rebounds"].mean()) < 0.02
    # assists exposure is a SCALAR, so A and B are the SAME Poisson in both
    # modes; the draws still differ because `rng.poisson` consumes a
    # lambda-dependent amount of randomness in the rebound call above, so the
    # stream position differs. Equality is therefore in distribution, not
    # bitwise — which is why the gate's assists deltas for A and B agree to
    # 1e-5 rather than exactly.
    assert abs(a["assists"].mean() - b["assists"].mean()) < 0.02
    assert abs(a["assists"].std() - b["assists"].std()) < 0.02


def test_channel_ramp_direction():
    """lam < 1 gives the channel minutes BACK (higher mean); lam > 1 takes more."""
    r = _rates(delta=4.0)
    os.environ["PROPS_CHANNEL_RAMP"] = "A"
    props.CHANNEL_LAM["reb"] = 1.0
    props.CHANNEL_LAM["ast"] = 1.0
    mid = props.simulate_player(r, 100_000, seed=15)
    props.CHANNEL_LAM["reb"] = 0.5
    props.CHANNEL_LAM["ast"] = 1.5
    moved = props.simulate_player(r, 100_000, seed=15)
    assert moved["rebounds"].mean() > mid["rebounds"].mean()
    assert moved["assists"].mean() < mid["assists"].mean()


def test_ramp_delta_is_exported_by_the_rate_builder():
    """`ramp_delta` is what the channel block consumes; if the key silently
    disappeared the switch would become a no-op that still looks enabled."""
    import inspect
    src = inspect.getsource(props.player_rates_from_stats)
    assert '"ramp_delta"' in src
    assert "proj_min_raw" in src


# ------------------------------------------------------- starout trail arms
def test_played_floor_constant_cannot_bind_on_the_star_side():
    """GATE 1 ARM B is a NULL BY ARITHMETIC and this test pins the arithmetic:
    trail_min >= STAR_TRAILING_MIN over TRAIL_GAMES rows forces at least
    ceil(STAR*TRAIL/max_minutes) played rows, and with a 48-minute regulation
    game that already meets PLAYED_FLOOR. If a future edit lowers the star
    threshold or raises the floor, this test fails and the arm stops being a
    registered null."""
    import math
    max_minutes = 60.0            # generous: longest observed row is 56.52
    implied = math.ceil(starout.STAR_TRAILING_MIN * starout.TRAIL_GAMES / max_minutes)
    assert implied >= starout.PLAYED_FLOOR


def test_null_u_lift_is_pure_pool_arithmetic():
    """GATE 2 ARM U: uniform weights must reduce the D33 softmax exactly to
    N/(N-1) over pool+star, with data/v2_usage.npz never consulted."""
    for n_pool in (4, 6, 8, 10):
        pool = set(range(1, n_pool + 1))
        lift = starout.compute_lift({}, pool, 0, default=1.0)
        assert abs(lift - (n_pool + 1) / n_pool) < 1e-12


def test_production_lift_applies_the_D83_residual_scale():
    """Whatever the usage mode, the LIVE magnitude is 1 + 0.16*(L-1) — the D83
    residual calibration. A mode that changed the scale would be a different
    hypothesis than the one gated."""
    pool = set(range(1, 8))
    full = starout.compute_lift({}, pool, 0, default=1.0)
    prod = starout.production_lift({}, pool, 0, default=1.0)
    assert abs(prod - (1.0 + starout.RESID_ATT_SCALE * (full - 1.0))) < 1e-12
    assert 1.0 <= prod < full
