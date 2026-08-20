"""D90 late-state layer: window gating, sign guards, PIT, live parity.

Live-parity heavy test follows tests/test_tanking.py (D68 discipline): a
truncated in-memory DB (strictly before a cutoff) + virtual-primed tank must
yield the same coefficients and per-team features the full-history build
computes for that date.
"""
import datetime as dt
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CUTOFF = dt.date(2026, 4, 3)          # late 2025-26: window active
EARLY = dt.date(2025, 12, 1)          # both-gp<55 everywhere


def _model():
    from nbapred.db import connect
    from nbapred.model.latestate import get_latestate_model
    con = connect(read_only=True)
    m = get_latestate_model(con)
    con.close()
    return m


def _d(x) -> dt.date:
    """numpy datetime64 -> datetime.date."""
    return x.astype("datetime64[D]").astype(dt.date)


def _season_span(season: str) -> tuple:
    """(first, last) 002 game date of a season."""
    from nbapred.db import connect
    con = connect(read_only=True)
    lo, hi = con.execute(
        """SELECT min(game_date), max(game_date) FROM nba_games
           WHERE season = ? AND game_id LIKE '002%'""", [season]).fetchone()
    con.close()
    return lo, hi


def test_zero_outside_window_and_no_date():
    m = _model()
    from nbapred.db import connect
    con = connect(read_only=True)
    teams = [t for (t,) in con.execute(
        """SELECT DISTINCT team_id FROM nba_games
           WHERE season='2025-26' AND game_id LIKE '002%'""").fetchall()]
    con.close()
    h, a = int(teams[0]), int(teams[1])
    assert m.term(h, a, {1, 2, 3}, set(), None) == 0.0
    # EARLY: every team gp<55 -> exactly 0.0 regardless of features
    assert m.tank.score(h, EARLY)[1] < 55 and m.tank.score(a, EARLY)[1] < 55
    assert m.term(h, a, {1, 2, 3, 4, 5}, set(), EARLY) == 0.0


def test_burn_in_tracks_the_derived_corpus_floor():
    """D112: the burn-in season is no longer the hardcoded '2022-23' — it is
    the tank model's derived floor, and the layer is ALIVE on the seasons the
    old literal zeroed out (2021-22 identically 0.0, 2022-23 cold)."""
    m = _model()
    assert m.burn_in == m.tank.floor
    assert m.burn_in <= "2021-22", (
        "corpus floor did not relax — D73/D90 stay untestable on the holdout")
    # active fit rows now exist inside the holdout seasons
    n_2122 = int((m._act_dates < np.datetime64(dt.date(2022, 7, 1))).sum())
    n_2223 = int((m._act_dates < np.datetime64(dt.date(2023, 7, 1))).sum())
    assert n_2122 > 0 and n_2223 > n_2122
    # ... and the coefficients are warm inside 2022-23 (they were 0.0 there
    # under the old floor for c_o and near-0 for c_f)
    c_f, c_o = m.coefs(dt.date(2023, 4, 8))
    assert c_f > 0.0 and c_o < 0.0


def test_sign_guards_and_min_active():
    """Burn-in is COLD at the floor, plus the sign guards.

    D131: the burn-in anchor is DERIVED from the fit frame and the tank model's
    floor instead of hardcoded. The old literal (2022-01-15) silently encoded
    "the floor season is 2021-22"; when the corpus grew to 7 seasons the floor
    moved to 2020-21 and the literal landed a whole season AFTER the burn-in,
    so the test failed on a data change rather than a code change.
    """
    from nbapred.model.latestate import C_MIN_ACTIVE
    m = _model()
    act = np.sort(m._act_dates)
    first_active = _d(act[0])                 # first gp>=55 fit row in existence
    kth_active = _d(act[C_MIN_ACTIVE - 1])    # the C_MIN_ACTIVE'th such row
    lo, hi = _season_span(m.burn_in)
    # The whole burn-in must sit INSIDE the floor season — this is the content
    # the old literal carried (2022-01-15 was inside the then-floor 2021-22),
    # and it is what stops the layer going warm before the derived floor.
    assert lo <= first_active <= hi, (first_active, m.burn_in)
    assert lo <= kth_active <= hi, (kth_active, m.burn_in)
    # coefs are exactly (0, 0) at the floor season's opening night ...
    assert m.coefs(lo) == (0.0, 0.0)
    # ... still exactly (0, 0) the instant the gp>=55 window opens ...
    assert m.coefs(first_active) == (0.0, 0.0)
    # ... and still exactly (0, 0) until C_MIN_ACTIVE rows have accrued.
    assert m.coefs(kth_active) == (0.0, 0.0)
    for d in (dt.date(2023, 3, 1), dt.date(2024, 3, 1), dt.date(2025, 3, 1),
              dt.date(2026, 3, 1), CUTOFF):
        c_f, c_o = m.coefs(d)
        assert c_f >= 0.0 and c_o <= 0.0
    # late 2025-26 the outs coef should be alive (regime-D expectation)
    assert m.coefs(CUTOFF)[1] < 0.0


def test_term_is_pit_and_deterministic():
    m = _model()
    t1 = m.coefs(CUTOFF)
    t2 = m.coefs(CUTOFF)
    assert t1 == t2
    # active fit frame only uses rows strictly before the date
    n_before = int((m._act_dates < np.datetime64(CUTOFF)).sum())
    n_total = len(m._act_dates)
    assert 0 < n_before < n_total


def test_outdiff_direction():
    """More outs on one side must never RAISE that side's margin."""
    m = _model()
    from nbapred.db import connect
    con = connect(read_only=True)
    row = con.execute("""SELECT h.team_id, a.team_id FROM
        (SELECT game_id, team_id, is_home FROM nba_games
         WHERE season='2025-26' AND game_id LIKE '002%' AND is_home) h
        JOIN (SELECT game_id, team_id, is_home FROM nba_games
         WHERE season='2025-26' AND game_id LIKE '002%' AND NOT is_home) a
        USING (game_id) LIMIT 1""").fetchone()
    con.close()
    h, a = int(row[0]), int(row[1])
    base = m.term(h, a, set(), set(), CUTOFF)
    more_h_out = m.term(h, a, {1, 2, 3}, set(), CUTOFF)
    assert more_h_out <= base + 1e-12


def test_live_truncated_parity():
    """Truncated-DB + virtual-primed tank == full-history layer at CUTOFF."""
    import duckdb
    from nbapred.config import DB_PATH
    from nbapred.db import connect
    from nbapred.model import latestate as ls
    from nbapred.model import tanking as tk
    full = _model()
    c_full = full.coefs(CUTOFF)

    con = connect(read_only=True)
    slate = con.execute("""SELECT DISTINCT season, team_id FROM nba_games
        WHERE game_id LIKE '002%' AND game_date = ? AND wl IS NOT NULL""",
        [CUTOFF]).fetchall()
    con.close()
    assert len(slate) >= 4

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
    tank_mem = tk.TankModel(mem, virtual_games=[(s, t, CUTOFF)
                                                for s, t in slate])
    layer_mem = ls.LateStateModel(mem, tank_mem)
    mem.close()

    c_mem = layer_mem.coefs(CUTOFF)
    assert abs(c_mem[0] - c_full[0]) < 1e-9
    assert abs(c_mem[1] - c_full[1]) < 1e-9
    for s, t in slate:
        assert abs(layer_mem.form5(int(t), CUTOFF)
                   - full.form5(int(t), CUTOFF)) < 1e-9
        # virtual-primed gp must match the full-history gp for the slate
        assert tank_mem.score(int(t), CUTOFF)[1] \
            == full.tank.score(int(t), CUTOFF)[1]
