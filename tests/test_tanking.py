"""D73 tank term: construction parity, window gating, k estimation.

The heavy test is live-parity: build a truncated in-memory DB (everything
strictly before a cutoff date, as live would see it), request VIRTUAL rows
for the teams that actually played on the cutoff date, and require the
virtual composite to match the full-history gate table
(data/apr_tank_stats.csv) at those team-dates. This is the D68 discipline:
the number predict_today computes tonight must equal what the backtest
will later compute for the same game.

CORPUS FLOOR (D112). The composite's pooled expanding z and the k fit frame
both start at tanking.season_floor(), which is now DERIVED from box-score
coverage (2021-22 on the current corpus) rather than hardcoded to '2022-23'.
Two fixtures in this file were generated under the old floor, so they are
pinned to it with TANK_SEASON_FLOOR=2022-23 — which doubles as the refactor's
no-op proof: at the old floor the new code must reproduce the old numbers
bitwise. The derived-floor values are pinned separately.
"""
import datetime as dt
import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CUTOFF = dt.date(2026, 4, 3)          # late 2025-26: tank window active
OLD_FLOOR = "2022-23"                 # pre-D112 hardcoded literal


@contextmanager
def floor(season: str | None):
    """Pin (or clear) TANK_SEASON_FLOOR and drop the heavy module caches."""
    from nbapred.model import latestate as ls
    from nbapred.model import tanking as tk
    prev = os.environ.get("TANK_SEASON_FLOOR")
    if season is None:
        os.environ.pop("TANK_SEASON_FLOOR", None)
    else:
        os.environ["TANK_SEASON_FLOOR"] = season
    tk._CACHE.clear(); ls._CACHE.clear()
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("TANK_SEASON_FLOOR", None)
        else:
            os.environ["TANK_SEASON_FLOOR"] = prev
        tk._CACHE.clear(); ls._CACHE.clear()


def _truncated_mem_db():
    """In-memory DuckDB holding only what live would see before CUTOFF."""
    import duckdb
    from nbapred.config import DB_PATH
    mem = duckdb.connect()
    mem.execute(f"ATTACH '{DB_PATH}' AS src (READ_ONLY)")
    mem.execute("CREATE TABLE nba_games AS SELECT * FROM src.nba_games "
                "WHERE game_date < ?", [CUTOFF])
    mem.execute("CREATE TABLE player_game_stats AS SELECT * FROM "
                "src.player_game_stats WHERE game_id IN "
                "(SELECT game_id FROM nba_games)")
    mem.execute("CREATE TABLE darko_history AS SELECT * FROM "
                "src.darko_history WHERE date < ?", [CUTOFF])
    mem.execute("CREATE TABLE nba_players AS SELECT * FROM src.nba_players")
    mem.execute("CREATE TABLE injury_reports_pit AS SELECT * FROM "
                "src.injury_reports_pit WHERE report_date < ?", [CUTOFF])
    return mem


def _slate():
    from nbapred.db import connect
    con = connect(read_only=True)
    slate = con.execute("""SELECT DISTINCT season, team_id FROM nba_games
        WHERE game_id LIKE '002%' AND game_date = ? AND wl IS NOT NULL""",
        [CUTOFF]).fetchall()
    con.close()
    assert len(slate) >= 4, "cutoff date has no games"
    return slate


def test_live_virtual_rows_match_gate_table():
    """D68 live parity AND the D112 refactor no-op: pinned to the old floor
    because data/apr_tank_stats.csv was generated under it."""
    import pandas as pd
    from nbapred.model.tanking import TankModel
    slate = _slate()
    apr = pd.read_csv(ROOT / "data/apr_tank_stats.csv", dtype={"game_id": str})
    apr["game_date"] = pd.to_datetime(apr["game_date"])
    ref = apr[apr.game_date == pd.Timestamp(CUTOFF)]
    with floor(OLD_FLOOR):
        mem = _truncated_mem_db()
        tm = TankModel(mem, virtual_games=[(s, t, CUTOFF) for s, t in slate])
        mem.close()
    assert tm.floor == OLD_FLOOR
    checked = 0
    for s, tid in slate:
        row = ref[ref.team_id == tid]
        if row.empty:
            continue
        got_t, got_gp = tm.score(int(tid), CUTOFF)
        exp_t, exp_gp = float(row.tank_score.iloc[0]), int(row.gp_before.iloc[0])
        assert got_gp == exp_gp, (tid, got_gp, exp_gp)
        assert abs(got_t - exp_t) < 1e-9, (tid, got_t, exp_t)
        checked += 1
    assert checked >= 4


def test_live_virtual_rows_match_full_history_at_derived_floor():
    """Floor-agnostic D68 parity: at the DERIVED floor the virtual-primed
    live build must still equal the full-history build at the same team-dates
    (no external fixture — the property, not a pinned vintage)."""
    from nbapred.db import connect
    from nbapred.model.tanking import TankModel, get_tank_model, season_floor
    slate = _slate()
    with floor(None):
        con = connect(read_only=True)
        derived = season_floor(con)
        full = get_tank_model(con)
        con.close()
        mem = _truncated_mem_db()
        tm = TankModel(mem, virtual_games=[(s, t, CUTOFF) for s, t in slate])
        mem.close()
    assert derived == tm.floor == full.floor
    for s, tid in slate:
        got_t, got_gp = tm.score(int(tid), CUTOFF)
        exp_t, exp_gp = full.score(int(tid), CUTOFF)
        assert got_gp == exp_gp, (tid, got_gp, exp_gp)
        assert abs(got_t - exp_t) < 1e-9, (tid, got_t, exp_t)


def test_season_floor_is_derived_and_overridable():
    """D155 changed this contract. The RESOLVED floor is now PINNED, because a
    coverage-derived floor moves whenever a backfill lands and silently
    invalidated the certified table twice (D131, D153). The DERIVATION is kept
    intact for drift detection via derived_season_floor()/floor_audit(), and
    the env override still works for same-run controls.

    So this test now asserts three things: the derivation is still correct on
    its own terms, the resolved value is the pin (not the derivation), and the
    drift between them is reported rather than hidden.
    """
    from nbapred.db import connect
    from nbapred.model.tanking import (FLOOR_MIN_BOX_COVERAGE,
                                       PINNED_SEASON_FLOOR,
                                       derived_season_floor, floor_audit,
                                       season_floor)
    con = connect(read_only=True)
    with floor(None):
        derived = derived_season_floor(con)
        # the pin is what production and every gate resolve to
        assert season_floor(con) == PINNED_SEASON_FLOOR
        audit = floor_audit(con)
        assert audit["pinned"] == PINNED_SEASON_FLOOR
        assert audit["derived"] == derived
        assert audit["drifted"] == (derived != PINNED_SEASON_FLOOR)
    cov = dict(con.execute("""
        SELECT g.season, count(DISTINCT CASE WHEN s.game_id IS NOT NULL
                                             THEN g.game_id END) * 1.0
                         / count(DISTINCT g.game_id)
        FROM (SELECT DISTINCT game_id, season FROM nba_games
              WHERE game_id LIKE '002%' AND wl IS NOT NULL) g
        LEFT JOIN (SELECT DISTINCT game_id FROM player_game_stats
                   WHERE seconds > 0) s USING (game_id)
        GROUP BY 1""").fetchall())
    with floor(OLD_FLOOR):
        assert season_floor(con) == OLD_FLOOR
    con.close()
    # every season at or after the floor clears the bar ...
    assert all(c >= FLOOR_MIN_BOX_COVERAGE
               for s, c in cov.items() if s >= derived)
    # ... and the season immediately below it does not (else the floor is
    # leaving usable data on the table)
    below = [s for s in cov if s < derived]
    if below:
        assert cov[max(below)] < FLOOR_MIN_BOX_COVERAGE


def test_gp55_window_gating_and_lookup_miss():
    from nbapred.db import connect
    from nbapred.model.tanking import GP_ACTIVE, get_tank_model
    with floor(None):
        con = connect(read_only=True)
        tm = get_tank_model(con)
        con.close()
    df = tm.df
    below = df[df.gp_before < GP_ACTIVE]
    some = below.sample(50, random_state=7) if len(below) > 50 else below
    for r in some.itertuples():
        assert tm.active(int(r.team_id), r.game_date) == 0.0
    on = df[(df.gp_before >= GP_ACTIVE) & (df.tank_score != 0)]
    r = on.iloc[0]
    assert tm.active(int(r.team_id), r.game_date) == r.tank_score
    # unknown (team, date) -> exactly 0.0
    assert tm.active(0, dt.date(2031, 1, 1)) == 0.0
    assert tm.diff(0, 1, None if False else dt.date(2031, 1, 1)) == 0.0


def test_fit_k_walkforward_old_floor_is_unchanged():
    """D112 refactor no-op proof: pinned to the old floor, fit_k must still
    return the pre-D112 registered ship value."""
    from nbapred.db import connect
    from nbapred.model.tanking import K_CLIP, get_tank_model
    with floor(OLD_FLOOR):
        con = connect(read_only=True)
        tm = get_tank_model(con)
        con.close()
        # 2022-23 is the floor season -> its window has not opened by Jan 2023
        assert tm.fit_k(dt.date(2023, 1, 1)) == 0.0    # <20 active rows yet
        k_24 = tm.fit_k(dt.date(2024, 10, 1))
        k_26 = tm.fit_k(dt.date(2026, 4, 9))
    assert -K_CLIP <= k_24 < 0 and -K_CLIP <= k_26 < 0  # negative, clipped
    assert abs(k_26 - (-2.2699)) < 0.01                 # pre-D112 ship value


def test_fit_k_walkforward_derived_floor():
    """At the derived floor the estimator is WARM a season earlier than at the
    old hardcoded floor: the burn-in zero sits inside the DERIVED floor season
    and every later k is bigger in magnitude (D112 — the whole point of
    relaxing the floor).

    D131: every anchor here is derived from tanking.season_floor(con) and the
    fit frame, never hardcoded. The old literals ("2022-01-01 is inside the
    floor season", "k_26 == -2.8156") silently encoded a 5-season corpus; when
    it grew to 7 the floor moved 2021-22 -> 2020-21 and both went stale, i.e.
    the test failed on a DATA change rather than a code change. The absolute-
    value regression pin lives in test_fit_k_walkforward_old_floor_is_unchanged
    above, which pins TANK_SEASON_FLOOR and so cannot drift with the corpus.
    """
    from nbapred.db import connect
    from nbapred.model.tanking import (K_CLIP, K_MIN_ACTIVE, get_tank_model,
                                       season_floor)
    with floor(None):
        con = connect(read_only=True)
        derived = season_floor(con)
        tm = get_tank_model(con)
        lo, hi = con.execute(
            """SELECT min(game_date), max(game_date) FROM nba_games
               WHERE season = ? AND game_id LIKE '002%'""", [derived]).fetchone()
        con.close()
        assert tm.floor == derived
        act = np.sort(tm._act_dates)
        first_active = act[0].astype("datetime64[D]").astype(dt.date)
        kth_active = act[K_MIN_ACTIVE - 1].astype("datetime64[D]").astype(dt.date)
        # the burn-in must sit INSIDE the derived floor season (the content the
        # old 2022-01-01 literal carried), and k must be EXACTLY 0 across it
        assert lo <= first_active <= hi, (first_active, derived)
        assert lo <= kth_active <= hi, (kth_active, derived)
        assert tm.fit_k(lo) == 0.0              # floor season opening night
        assert tm.fit_k(first_active) == 0.0    # gp>=55 window just opened
        assert tm.fit_k(kth_active) == 0.0      # <K_MIN_ACTIVE rows still
        k_23 = tm.fit_k(dt.date(2023, 1, 1))    # was EXACTLY 0 pre-D112
        k_24 = tm.fit_k(dt.date(2024, 10, 1))
        k_26 = tm.fit_k(dt.date(2026, 4, 9))
    with floor(OLD_FLOOR):
        con = connect(read_only=True)
        tm_old = get_tank_model(con)
        con.close()
        k_24_old = tm_old.fit_k(dt.date(2024, 10, 1))
        k_26_old = tm_old.fit_k(dt.date(2026, 4, 9))
    assert -K_CLIP <= k_23 < 0                          # warm on the holdout
    assert -K_CLIP <= k_24 < 0 and -K_CLIP <= k_26 < 0
    # "bigger in magnitude" stated against the old floor computed in the same
    # run, so corpus growth moves both sides and the claim stays exact
    assert abs(k_24) > abs(k_24_old) and abs(k_26) > abs(k_26_old)
