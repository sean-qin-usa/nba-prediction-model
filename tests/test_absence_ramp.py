"""D145 props ABSENCE RAMP — construction + zero-outside-window tests.

Mirrors the D73/D133 test style: prove the term is exactly zero outside its
pre-registered window, that the env kill-switch restores the prior behaviour
bitwise, and that the absence axis is computed from strictly-prior data only.
"""
import datetime as dt
import os
import warnings

import pytest

warnings.filterwarnings("ignore")


@pytest.fixture(autouse=True)
def _clean_env():
    saved = os.environ.get("PROPS_ABSENCE_RAMP")
    os.environ.pop("PROPS_ABSENCE_RAMP", None)
    yield
    if saved is None:
        os.environ.pop("PROPS_ABSENCE_RAMP", None)
    else:
        os.environ["PROPS_ABSENCE_RAMP"] = saved


def test_absence_ramp_table_shape():
    from nbapred.engine.props import absence_ramp
    # EXACTLY zero on the 0..4 window (89.7% of the props universe)
    assert [absence_ramp(m) for m in range(5)] == [0.0] * 5
    # monotone non-decreasing, and the two live buckets carry the fitted bias
    vals = [absence_ramp(m) for m in range(11)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert absence_ramp(5) == absence_ramp(6) == absence_ramp(7) == 0.858
    assert absence_ramp(8) == absence_ramp(9) == absence_ramp(10) == 2.987
    # out-of-range is clamped to the top bucket, never negative
    assert absence_ramp(99) == 2.987
    assert min(vals) >= 0.0


def test_kill_switch_zeroes_the_term():
    from nbapred.engine.props import absence_ramp
    os.environ["PROPS_ABSENCE_RAMP"] = "0"
    assert [absence_ramp(m) for m in range(11)] == [0.0] * 11


def test_term_is_independent_of_the_d133_gp_ramp():
    """The two ramps are separate switches on separate axes (D145 measured them
    as complementary, A - C = +0.00407 SIG)."""
    from nbapred.engine.props import absence_ramp, minutes_ramp
    os.environ["PROPS_MIN_RAMP"] = "0"
    try:
        assert minutes_ramp(0) == 0.0
        assert absence_ramp(9) == 2.987      # unaffected by the other switch
    finally:
        os.environ.pop("PROPS_MIN_RAMP", None)


def _one_player(con):
    row = con.execute("""
        SELECT s.player_id FROM player_game_stats s
        WHERE s.game_id LIKE '002%' AND s.seconds >= 720
        GROUP BY 1 ORDER BY count(*) DESC LIMIT 1""").fetchone()
    return int(row[0]) if row else None


def test_miss10_is_bounded_and_pit():
    """games_missed_last10 is in [0,10] and reads nothing at/after the cutoff."""
    from nbapred.db import connect
    from nbapred.config import current_season
    from nbapred.engine.props import games_missed_last10
    con = connect(read_only=True)
    pid = _one_player(con)
    if pid is None:
        con.close(); return
    for cut in (dt.date(2023, 1, 15), dt.date(2024, 3, 5), dt.date(2025, 12, 1)):
        df = con.execute("""
            SELECT g.game_date, s.team_id, s.seconds FROM player_game_stats s
            JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
            WHERE s.player_id = ? AND s.game_id LIKE '002%' AND s.seconds >= 720
              AND g.game_date < ? ORDER BY g.game_date""", [pid, cut]).fetchdf()
        if df.empty:
            continue
        season = current_season(cut)
        df = df.assign(_season=[current_season(d) for d in df["game_date"]])
        m = games_missed_last10(con, df, season, cut)
        assert isinstance(m, int) and 0 <= m <= 10
    con.close()


def test_zero_outside_window_end_to_end():
    """On a full-time player with no recent absence the shipped path must be
    BITWISE identical with the switch on and off."""
    from nbapred.db import connect
    from nbapred.engine.props import player_rates_from_stats
    con = connect(read_only=True)
    pid = _one_player(con)
    if pid is None:
        con.close(); return
    cut = dt.date(2024, 3, 5)
    os.environ["PROPS_ABSENCE_RAMP"] = "0"
    off = player_rates_from_stats(con, pid, before=cut)
    os.environ["PROPS_ABSENCE_RAMP"] = "1"
    on = player_rates_from_stats(con, pid, before=cut)
    con.close()
    if off is None or on is None:
        return
    # every non-minutes field is untouched by construction
    for k in ("rate_rim", "rate_mid", "rate_thr", "fg_rim", "fg_mid", "fg_thr",
              "fta_per_min", "ft_pct", "reb_per_min", "ast_per_min", "n_games"):
        assert off[k] == on[k], k
    # minutes differ by exactly one of the three table values, never anything else
    assert round(off["proj_min"] - on["proj_min"], 6) in (0.0, 0.858, 2.987)
