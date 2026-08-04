"""Star-out redistribution adjustment (nbapred/engine/starout.py, D82 lean-in).

Pure-function tests: lift bounds, no-star no-op, volume-only adjustment
(efficiency untouched per D34), D39 positional tilt selection.
"""
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nbapred.engine import starout  # noqa: E402


def _rates():
    return {
        "proj_min": 30.0, "sd_min": 4.0, "minutes_hist": [28, 30, 32],
        "rate_rim": 0.20, "rate_mid": 0.10, "rate_thr": 0.15,
        "fg_rim": 0.62, "fg_mid": 0.44, "fg_thr": 0.36,
        "fta_per_min": 0.12, "ft_pct": 0.80,
        "reb_per_min": 0.18, "ast_per_min": 0.14, "n_games": 20,
    }


def test_lift_bounds():
    # dominant star -> raw S/(S-w) explodes -> capped at 1.6
    w = {1: 100.0, 2: 1.0, 3: 1.0}
    assert starout.compute_lift(w, {2, 3}, 1) == starout.LIFT_HI
    # negligible star -> lift floored at 1.0, never shrinks attempts
    w = {1: 1e-9, 2: 1.0, 3: 1.0}
    assert starout.compute_lift(w, {2, 3}, 1) >= starout.LIFT_LO
    # equal weights, 8 remaining + star: S/(S-w) = 9/8 = 1.125 (uncapped zone)
    w = {p: 2.0 for p in range(9)}
    lift = starout.compute_lift(w, set(range(1, 9)), 0)
    assert abs(lift - 9.0 / 8.0) < 1e-9
    # unknown players fill with `default` (gate's u.get(p, 0) -> exp = 1.0)
    lift = starout.compute_lift({}, {1, 2, 3, 4}, 9, default=1.0)
    assert abs(lift - 5.0 / 4.0) < 1e-9
    assert starout.LIFT_LO <= lift <= starout.LIFT_HI
    # production lift = residual fraction of the softmax excess, same bounds
    w = {1: 100.0, 2: 1.0, 3: 1.0}
    pl = starout.production_lift(w, {2, 3}, 1)
    assert abs(pl - (1 + starout.RESID_ATT_SCALE * (starout.LIFT_HI - 1))) < 1e-12
    assert starout.production_lift({1: 1e-9, 2: 1.0}, {2}, 1) >= 1.0


def test_no_star_no_op():
    r = _rates()
    # no context (no qualifying star-out) -> the exact same object back
    assert starout.adjust_player_rates(r, 123, None) is r
    # empty OUT set -> no context, before any DB work (con unused)
    assert starout.team_context(None, 1610612737, set(), dt.date(2026, 1, 1)) is None
    # the out star himself is never adjusted
    ctx = {"star": 123, "lift": 1.3, "star_pos": "G", "usage_source": "x", "n_pool": 8}
    assert starout.adjust_player_rates(r, 123, ctx) is r


def test_adjustment_volume_only():
    r = _rates()
    ctx = {"star": 99, "lift": 1.25, "lift_softmax": 1.4, "star_pos": "G",
           "usage_source": "v2_usage.npz", "n_pool": 8}
    out = starout.adjust_player_rates(r, 1, ctx, positions={1: "G"})
    # attempts scaled by the context lift (team_context already residual-scales)
    for k in starout.ATTEMPT_KEYS:
        assert abs(out[k] - r[k] * 1.25) < 1e-12
    # efficiency UNCHANGED (D34: attempts move, efficiency drops; conservative)
    for k in ("fg_rim", "fg_mid", "fg_thr", "ft_pct"):
        assert out[k] == r[k]
    # D39 same-position tilt at the residual-calibrated production magnitude
    expect = 30.0 + starout.RESID_MIN_SCALE * starout.TILT_SAME_POS
    assert abs(out["proj_min"] - expect) < 1e-12
    # reb/ast held VOLUME-NEUTRAL under the tilt (tail-only residual finding):
    # per-min rate x proj_min is invariant
    for k in ("reb_per_min", "ast_per_min"):
        assert abs(out[k] * out["proj_min"] - r[k] * r["proj_min"]) < 1e-9
    # input dict not mutated
    assert r == _rates()


def test_roster_delta_factors():
    # D85 arrival shape: regime-B roll residuals; k1-3 zero (arriver ramp)
    assert starout.arr_att_shape(2) == 0.0
    assert starout.arr_att_shape(5) == -0.12
    assert starout.arr_att_shape(10) == -0.07
    assert starout.arr_att_shape(20) == -0.04
    assert starout.arr_att_shape(45) == 0.0          # outside the window
    # bottom-usage tercile SHIELDED (b89f4a: compression not uniform)
    assert starout.arrival_att_factor(6, 0) == 1.0
    f = starout.arrival_att_factor(6, 2)
    assert starout.ARR_FACTOR_LO <= f < 1.0
    assert abs(f - (1.0 + starout.ARR_ATT_SCALE * -0.12)) < 1e-12
    # deeper compression at k4-7 than k13-30
    assert starout.arrival_att_factor(6, 1) < starout.arrival_att_factor(20, 1)
    # tilt: negative, same-pos loses MOST (D39 mirror), residual-scaled
    assert starout.arrival_tilt("G", "G") < starout.arrival_tilt("C", "G") < 0
    assert abs(starout.arrival_tilt("F", "F") -
               starout.ARR_MIN_SCALE * starout.ARR_TILT_SAME) < 1e-12
    assert starout.arrival_tilt(None, "G") == starout.ARR_MIN_SCALE * starout.ARR_TILT_FLAT
    # departure attenuation: mild, within the sanity clip
    d = starout.departure_att_factor()
    assert starout.DEP_FACTOR_LO <= d < 1.0
    assert abs(d - (1.0 + starout.DEP_ATT_SCALE * starout.DEP_ATT_SHAPE)) < 1e-12


def test_roster_delta_adjust():
    r = _rates()
    # no context / player outside pools -> the exact same object back
    assert starout.adjust_player_rates_rd(r, 1, None) is r
    rd = {"dep": {"star": 9, "pool": {2, 3}, "att_factor": 0.973, "k": 4},
          "arr": None}
    assert starout.adjust_player_rates_rd(r, 1, rd) is r          # not in pool
    out = starout.adjust_player_rates_rd(r, 2, rd)
    for k in starout.ATTEMPT_KEYS:
        assert abs(out[k] - r[k] * 0.973) < 1e-12
    assert out["proj_min"] == r["proj_min"]                       # DEP: no tilt
    for k in ("fg_rim", "fg_mid", "fg_thr", "ft_pct", "reb_per_min", "ast_per_min"):
        assert out[k] == r[k]                                     # efficiency+reb/ast rates untouched
    # arrival: compression + NEGATIVE tilt; reb/ast per-min rates untouched
    # (per-game reb/ast FLOW DOWN with the minutes — measured, unlike D83)
    rd = {"dep": None,
          "arr": {"arriver": 9, "k": 6, "pool": {1, 2}, "tercile": {1: 2, 2: 0},
                  "arr_pos": "G", "star_trail": 30.0}}
    out = starout.adjust_player_rates_rd(r, 1, rd, positions={1: "G"})
    f = starout.arrival_att_factor(6, 2)
    for k in starout.ATTEMPT_KEYS:
        assert abs(out[k] - r[k] * f) < 1e-12
    assert abs(out["proj_min"] - (30.0 + starout.arrival_tilt("G", "G"))) < 1e-12
    assert out["reb_per_min"] == r["reb_per_min"]
    # bottom-tercile incumbent: attempts shielded, tilt still applies
    out = starout.adjust_player_rates_rd(r, 2, rd, positions={2: "C"})
    for k in starout.ATTEMPT_KEYS:
        assert out[k] == r[k]
    assert out["proj_min"] < r["proj_min"]
    # floor holds
    low = dict(r, proj_min=8.5)
    out = starout.adjust_player_rates_rd(low, 1, rd, positions={1: "G"})
    assert out["proj_min"] == starout.ARR_MIN_FLOOR
    # input never mutated
    assert r == _rates()


def test_positional_tilt():
    assert starout.minutes_tilt("G", "G") == starout.TILT_SAME_POS
    assert starout.minutes_tilt("C", "G") == starout.TILT_DIFF_POS
    # hybrid listings share a letter -> same-position (G-F vs F)
    assert starout.minutes_tilt("G-F", "F") == starout.TILT_SAME_POS
    assert starout.minutes_tilt("F-C", "G") == starout.TILT_DIFF_POS
    # unknown position on either side -> flat +2.4 fallback
    assert starout.minutes_tilt(None, "G") == starout.TILT_FLAT
    assert starout.minutes_tilt("F", None) == starout.TILT_FLAT
    # production tilt keeps the D39 ordering at residual magnitude
    assert starout.production_tilt("G", "G") == starout.RESID_MIN_SCALE * starout.TILT_SAME_POS
    assert starout.production_tilt("C", "G") < starout.production_tilt("G", "G")
    # proj_min sanity cap holds even for iron-man projections
    out = starout.adjust_rates({"proj_min": 45.5, "rate_rim": 0.2, "rate_mid": 0.1,
                                "rate_thr": 0.1, "fta_per_min": 0.1}, 1.2, 2.91)
    assert out["proj_min"] == starout.PROJ_MIN_CAP


def test_fresh_guard_requires_actually_played():
    """D146: a DNP row (seconds=0) must NOT refresh a player's last-played
    date. player_game_stats carries 38,311 such rows; before the fix a benched
    player kept his own freshness alive indefinitely and FRESH_DAYS was inert."""
    import datetime as dt
    from nbapred.db import connect
    from nbapred.engine import starout

    con = connect(read_only=True)
    before = dt.date(2025, 3, 1)
    rows = con.execute("""
        WITH tg AS (
          SELECT s.player_id, s.team_id, s.seconds/60.0 m, g.game_date
          FROM player_game_stats s
          JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g
            USING (game_id)
          WHERE s.game_id LIKE '002%' AND g.game_date < ?)
        SELECT team_id, player_id, max(game_date) last_any,
               max(CASE WHEN m > 0 THEN game_date END) last_played
        FROM tg GROUP BY 1, 2
        HAVING last_played IS NOT NULL AND max(game_date) > last_played
        LIMIT 50
    """, [before]).fetchall()
    con.close()
    # the corpus must actually contain the pathology, else this test is vacuous
    assert rows, "no DNP-after-last-played pairs found; test would be vacuous"
    # for every such pair the played-date is strictly older, so any guard built
    # on max(game_date) is measuring the wrong thing
    for _t, _p, last_any, last_played in rows:
        assert last_any > last_played

    # and the shipped guard must use the played date: a player whose only
    # recent rows are DNPs is NOT fresh
    assert starout.FRESH_DAYS == 12
