"""Paper-trade bet engine: the pre-registered F4 registry (exact operators vs
the D75/D78/D82 sims + the D112 confidence-excess cap and edge shrinkage),
consensus/best-price odds view, the parallel sizing arms (incl. the frozen
open_shrunk diagnostic), the three snapshot views (OPEN/POST_REPORT/PRETIP),
the in-place bet_paper migration, and the emit/settle round trip on a
scratch DB."""
import datetime as dt
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import bet_engine  # noqa: E402
import f4_shrinkage  # noqa: E402

INF = float("inf")


# ---- rule registry ----------------------------------------------------------

def test_opposite_side_never_bet():
    assert bet_engine.rules_fired(0.60, 0.45, 70, 70, True) == []
    assert bet_engine.rules_fired(0.40, 0.55, 70, 70, True) == []


def test_r4_lowt_exact_operators():
    # edge .05 > .02, late via either team gp>=55; tails NOT cleared (.10)
    assert bet_engine.rules_fired(0.60, 0.55, 60, 20, False) == ["R4_LOWT"]
    # threshold is strict: edge exactly .02 does not fire
    assert bet_engine.rules_fired(0.57, 0.55, 60, 20, False) == []
    # early season (both gp < 55): no late rules
    assert bet_engine.rules_fired(0.60, 0.55, 10, 10, False) == []


def test_tail_band_rules_and_cap():
    # tails .25 > .20, edge .05 in [.03,.10], late -> R4 + both T20 rules
    assert bet_engine.rules_fired(0.75, 0.70, 60, 58, False) == [
        "R4_LOWT", "T20_D03_10_W", "T20_D03_10"]
    # same without late window -> only the base T20 rule
    assert bet_engine.rules_fired(0.75, 0.70, 10, 10, False) == ["T20_D03_10"]
    # D13 divergence CAP (band upper bound .10): with the D112 registry cap
    # LIFTED, edge .18 kills the T20 band and R4 still fires
    assert bet_engine.rules_fired(0.80, 0.62, 60, 20, False, cap=INF) == [
        "R4_LOWT"]
    # tail tier is strict: |p_us-.5| = .20 exactly does not clear
    assert bet_engine.rules_fired(0.70, 0.65, 10, 10, False) == []
    # away-side pick works symmetrically
    assert bet_engine.rules_fired(0.25, 0.30, 10, 10, False) == ["T20_D03_10"]


# ---- D112 upper confidence-excess cap ---------------------------------------

def test_conf_excess_cap_is_registry_level():
    """conf_us - conf_mkt > 0.08 -> NO rule fires, including R4_LOWT and
    STAR_FAV_SHARPER which carried no cap at all before D112."""
    assert bet_engine.CONF_EXCESS_CAP == 0.08
    # conf_us .30, conf_mkt .12 -> excess .18: everything is vetoed
    assert bet_engine.rules_fired(0.80, 0.62, 60, 20, True) == []
    # ... and the SAME game fires the old registry with the cap lifted
    assert bet_engine.rules_fired(0.80, 0.62, 60, 20, True, cap=INF) == [
        "R4_LOWT", "STAR_FAV_SHARPER"]
    # the skip is STRICT: excess == cap is kept, a hair over is dropped.
    # 0.75/0.625 are binary-exact so excess is exactly 0.125.
    assert bet_engine.rules_fired(0.75, 0.625, 60, 20, False,
                                  cap=0.125) == ["R4_LOWT"]
    assert bet_engine.rules_fired(0.75, 0.625, 60, 20, False, cap=0.124) == []
    # at the registered cap, .079 fires and .081 does not
    assert bet_engine.rules_fired(0.679, 0.600, 60, 20, False) == ["R4_LOWT"]
    assert bet_engine.rules_fired(0.681, 0.600, 60, 20, False) == []
    # away-side: conf is |p-0.5| on both, so the cap is side-symmetric
    assert bet_engine.rules_fired(0.321, 0.400, 60, 20, False) == ["R4_LOWT"]
    assert bet_engine.rules_fired(0.319, 0.400, 60, 20, False) == []
    # the cap subsumes the old T20 .10 band cap: edge .09 was inside the band
    # and is now vetoed outright
    assert bet_engine.rules_fired(0.79, 0.70, 10, 10, False) == []
    assert bet_engine.rules_fired(0.79, 0.70, 10, 10, False,
                                  cap=INF) == ["T20_D03_10"]


# ---- D112 edge shrinkage / sizing arms --------------------------------------

def test_shrink_edge_floor_and_slope():
    a, b = -0.0140, 0.184
    assert f4_shrinkage.shrink_edge(0.02, a, b) == 0.0      # floored at 0
    assert f4_shrinkage.shrink_edge(-0.5, a, b) == 0.0
    assert math.isclose(f4_shrinkage.shrink_edge(0.20, a, b), -0.0140 + 0.184 * 0.20)
    # break-even claimed edge = -a/b; just above it the calibrated edge is > 0
    be = -a / b
    assert f4_shrinkage.shrink_edge(be * 0.999, a, b) == 0.0
    assert f4_shrinkage.shrink_edge(be * 1.001, a, b) > 0.0


def test_registered_coeffs_are_the_d112_fit():
    c = f4_shrinkage.load_coeffs()
    assert math.isclose(c["a"], -0.0140, abs_tol=5e-4)
    assert math.isclose(c["b"], 0.184, abs_tol=5e-4)
    assert 0.0 < c["b"] < 1.0            # shrinkage can only shrink


def test_three_sizing_arms():
    coeffs = {"a": -0.0140, "b": 0.184}
    p_us, p_mkt = 0.62, 0.56
    dec = 1.0 / (p_mkt * 1.045)         # the sims' 4.5%-overround price
    flat = f4_shrinkage.stake_units("flat", p_us, p_mkt, dec, coeffs)
    raw = f4_shrinkage.stake_units("raw_kelly", p_us, p_mkt, dec, coeffs)
    shr = f4_shrinkage.stake_units("shrunk_kelly", p_us, p_mkt, dec, coeffs)
    assert flat == 1.0
    # raw Kelly on the claimed edge is live; the calibrated edge is 0 here
    # (claimed .06 < break-even .076), so at a VIGGED price the shrunk arm
    # stakes NOTHING — the D112 result, in one assertion.
    f = (p_us * dec - 1) / (dec - 1)
    assert math.isclose(raw, min(0.25 * f * 100.0, 10.0))
    assert shr == 0.0
    assert raw > shr                    # the ship can only ever shrink stakes
    # a shopped price BETTER than consensus fair revives the shrunk arm —
    # this is the only channel through which it ever deploys capital
    assert f4_shrinkage.stake_units(
        "shrunk_kelly", p_us, p_mkt, 1.0 / p_mkt + 0.05, coeffs) > 0.0
    # ... and at exactly consensus fair it is still 0 (f* == 0)
    assert f4_shrinkage.stake_units(
        "shrunk_kelly", p_us, p_mkt, 1.0 / p_mkt, coeffs) == 0.0
    # non-positive Kelly fraction -> 0, never negative
    assert f4_shrinkage.stake_units("raw_kelly", 0.51, 0.50, 1.20, coeffs) == 0.0
    assert f4_shrinkage.stake_units("raw_kelly", 0.99, 0.50, 50.0, coeffs) == 10.0


def test_star_fav_sharper():
    # sharper (edge>0) + favorite star-out; not late, not tails
    assert bet_engine.rules_fired(0.60, 0.58, 10, 10, True) == [
        "STAR_FAV_SHARPER"]
    # flat/negative edge: never
    assert bet_engine.rules_fired(0.56, 0.58, 10, 10, True) == []
    # all four can co-fire
    assert bet_engine.rules_fired(0.75, 0.70, 60, 58, True) == [
        "R4_LOWT", "T20_D03_10_W", "T20_D03_10", "STAR_FAV_SHARPER"]


# ---- market snapshot --------------------------------------------------------

def _q(snap, book, name, price, home="Home HH", away="Away AA", ev="ev1",
       commence=None):
    return (snap, None, "the-odds-api", ev, commence, home, away, book,
            None, "h2h", name, None, price, None, "f.jsonl")


def test_market_snapshot_consensus_and_best():
    t0 = dt.datetime(2026, 10, 20, 18, 0)
    t1 = dt.datetime(2026, 10, 20, 22, 0)
    quotes = [
        # book A stale snapshot (must be ignored in favor of t1)
        _q(t0, "bookA", "Home HH", 2.20), _q(t0, "bookA", "Away AA", 1.70),
        _q(t1, "bookA", "Home HH", 1.90), _q(t1, "bookA", "Away AA", 1.95),
        _q(t1, "bookB", "Home HH", 2.00), _q(t1, "bookB", "Away AA", 1.85),
        # other game must be excluded
        _q(t1, "bookA", "X", 1.50, home="X", away="Y", ev="ev2"),
    ]
    m = bet_engine.market_snapshot(quotes, "Home HH", "Away AA")
    assert m is not None and m["n_books"] == 2
    pA = (1 / 1.90) / (1 / 1.90 + 1 / 1.95)
    pB = (1 / 2.00) / (1 / 2.00 + 1 / 1.85)
    assert abs(m["p_home"] - (pA + pB) / 2) < 1e-12
    assert m["best"]["home"] == (2.00, "bookB")
    assert m["best"]["away"] == (1.95, "bookA")
    # panel telemetry: every two-sided book contributes both sides; the
    # consensus is the per-side MEDIAN decimal; ts is the freshest quote used
    assert len(m["panel"]) == 4                      # 2 books x 2 sides at t1
    assert {(b, s) for _, b, s, _ in m["panel"]} == {
        ("bookA", "home"), ("bookA", "away"),
        ("bookB", "home"), ("bookB", "away")}
    assert abs(m["consensus"]["home"] - 1.95) < 1e-12    # median(1.90, 2.00)
    assert abs(m["consensus"]["away"] - 1.90) < 1e-12    # median(1.95, 1.85)
    assert m["ts"] == t1
    # before_ts filter (close reconstruction): only t0 remains
    m0 = bet_engine.market_snapshot(quotes, "Home HH", "Away AA",
                                    before_ts=t0)
    assert m0["best"]["home"] == (2.20, "bookA")
    assert bet_engine.market_snapshot(quotes, "No", "Game") is None


# ---- bet_paper round trip ---------------------------------------------------

PRE_D112_DDL = """CREATE TABLE bet_paper (
    candidate_ts  TIMESTAMPTZ NOT NULL, game_date DATE NOT NULL,
    game_id VARCHAR NOT NULL, home VARCHAR, away VARCHAR,
    side VARCHAR NOT NULL, rule VARCHAR NOT NULL,
    p_us DOUBLE, p_us_side DOUBLE, p_mkt_side DOUBLE, edge DOUBLE,
    price_decimal DOUBLE, book VARCHAR, implied_p DOUBLE,
    event_id VARCHAR, commence_ts TIMESTAMPTZ, stake_units DOUBLE DEFAULT 1.0,
    close_price DOUBLE, close_implied DOUBLE, clv DOUBLE, outcome INTEGER,
    pnl_units DOUBLE, settled_ts TIMESTAMPTZ, detail VARCHAR,
    PRIMARY KEY (game_date, game_id, side, rule))"""

AUX_DDL = ["""CREATE TABLE nba_games (season VARCHAR, game_id VARCHAR,
    game_date DATE, team_id BIGINT, team_abbrev VARCHAR, matchup VARCHAR,
    is_home BOOLEAN, wl VARCHAR, pts INTEGER, ingest_ts TIMESTAMPTZ)""",
    """CREATE TABLE odds_quotes (snapshot_ts TIMESTAMPTZ,
    ingest_ts TIMESTAMPTZ, source VARCHAR, event_id VARCHAR,
    commence_time TIMESTAMPTZ, home_team VARCHAR, away_team VARCHAR,
    bookmaker VARCHAR, book_last_update TIMESTAMPTZ, market VARCHAR,
    outcome_name VARCHAR, outcome_desc VARCHAR, price_decimal DOUBLE,
    point DOUBLE, raw_file VARCHAR)"""]


def test_migration_from_pre_d112_table(tmp_path, monkeypatch):
    """A legacy 24-column bet_paper (4-col PK, no snapshot_kind) migrates in
    place: every new column is ADDed with its default, the PK is rebuilt to
    include snapshot_kind, and the legacy row survives labelled POST_REPORT."""
    import duckdb
    db = tmp_path / "mig.duckdb"
    con = duckdb.connect(str(db))
    con.execute(PRE_D112_DDL)
    gd = dt.date(2026, 10, 24)
    now = dt.datetime.now(dt.timezone.utc)
    con.execute("INSERT INTO bet_paper VALUES (" + ",".join("?" * 24) + ")",
                [now, gd, "0022600042", "BOS", "NYK", "home", "R4_LOWT",
                 0.61, 0.61, 0.56, 0.05, 2.05, "bookB", 0.56, "ev1", None,
                 1.0, None, None, None, None, None, None, "legacy"])
    bet_engine._ensure_schema(con)
    cols = {r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='bet_paper'").fetchall()}
    for c, _, _ in bet_engine.MIGRATION_COLUMNS:
        assert c in cols, c
    kind, price = con.execute(
        "SELECT snapshot_kind, price_decimal FROM bet_paper").fetchone()
    assert kind == "POST_REPORT" and price == 2.05   # legacy row intact
    pk = con.execute("""SELECT constraint_column_names
        FROM duckdb_constraints() WHERE table_name='bet_paper'
        AND constraint_type='PRIMARY KEY'""").fetchone()[0]
    assert "snapshot_kind" in pk
    # the SAME (game, side, rule) now books once per snapshot view ...
    con.execute("INSERT INTO bet_paper (candidate_ts, snapshot_kind, "
                "game_date, game_id, side, rule) VALUES (?,?,?,?,?,?)",
                [now, "OPEN", gd, "0022600042", "home", "R4_LOWT"])
    assert con.execute("SELECT count(*) FROM bet_paper").fetchone()[0] == 2
    # ... and re-emission of a view is still first-write-wins
    con.execute("INSERT OR IGNORE INTO bet_paper (candidate_ts, "
                "snapshot_kind, game_date, game_id, side, rule, "
                "price_decimal) VALUES (?,?,?,?,?,?,?)",
                [now, "OPEN", gd, "0022600042", "home", "R4_LOWT", 9.99])
    assert con.execute("SELECT count(*) FROM bet_paper").fetchone()[0] == 2
    # idempotent: a second _ensure_schema is a no-op
    bet_engine._ensure_schema(con)
    assert con.execute("SELECT count(*) FROM bet_paper").fetchone()[0] == 2
    con.close()


def test_emit_insert_ignore_and_settle(tmp_path, monkeypatch):
    import duckdb
    db = tmp_path / "scratch.duckdb"

    def fake_connect(read_only=False):
        return duckdb.connect(str(db), read_only=read_only)

    monkeypatch.setattr(bet_engine, "_connect", fake_connect)
    con = fake_connect()
    con.execute(bet_engine.SCHEMA)
    for ddl in AUX_DDL:
        con.execute(ddl)
    gd = dt.date(2026, 10, 24)
    now = dt.datetime.now(dt.timezone.utc)
    base = {"candidate_ts": now, "snapshot_kind": "OPEN", "game_date": gd,
            "game_id": "0022600042", "home": "BOS", "away": "NYK",
            "side": "home", "rule": "R4_LOWT", "p_us": 0.61,
            "p_us_side": 0.61, "p_mkt_side": 0.56, "edge": 0.05,
            "price_decimal": 2.05, "book": "bookB", "best_price": 2.05,
            "consensus_price": 2.00, "implied_p": 0.56, "event_id": "ev1",
            "stake_units": 1.0, "detail": "test"}

    def ins(d):
        cols = list(d)
        con.execute("INSERT OR IGNORE INTO bet_paper (" + ",".join(cols) +
                    ") VALUES (" + ",".join("?" * len(cols)) + ")",
                    [d[c] for c in cols])

    ins(base)
    # re-emission of the SAME view with a different price must NOT overwrite
    ins({**base, "price_decimal": 1.80})
    assert con.execute("SELECT count(*), max(price_decimal) FROM bet_paper"
                       ).fetchone() == (1, 2.05)
    # the same bet under the OTHER views books separately (3-view design)
    ins({**base, "snapshot_kind": "POST_REPORT", "price_decimal": 2.00})
    ins({**base, "snapshot_kind": "PRETIP", "price_decimal": 1.95})
    assert con.execute("SELECT count(*) FROM bet_paper").fetchone()[0] == 3
    # game result lands -> settle fills outcome/pnl (no close quotes -> CLV NULL)
    con.execute("INSERT INTO nba_games VALUES ('2026-27','0022600042',?,1,"
                "'BOS','BOS vs. NYK',true,'W',110,?)", [gd, now])
    con.execute("INSERT INTO nba_games VALUES ('2026-27','0022600042',?,2,"
                "'NYK','NYK @ BOS',false,'L',100,?)", [gd, now])
    con.close()
    n = bet_engine.settle(today=gd + dt.timedelta(days=1))
    assert n == 3                                 # one per snapshot view
    con = fake_connect(read_only=True)
    rows = con.execute(
        "SELECT snapshot_kind, outcome, pnl_units, clv, settled_ts, "
        "pnl_raw_kelly, pnl_shrunk_kelly, pnl_open_shrunk "
        "FROM bet_paper ORDER BY price_decimal DESC").fetchall()
    con.close()
    assert [r[0] for r in rows] == ["OPEN", "POST_REPORT", "PRETIP"]
    for kind, outcome, pnl, clv, sts, prk, psk, pos in rows:
        assert outcome == 1 and sts is not None and clv is None
        # absent (0-stake) Kelly arms score exactly 0 — never a loss
        assert prk == 0.0 and psk == 0.0 and pos == 0.0
    assert abs(rows[0][2] - 1.05) < 1e-12         # OPEN: 1u at 2.05, won
    assert abs(rows[2][2] - 0.95) < 1e-12         # PRETIP: 1u at 1.95, won
    # second settle run: nothing left open
    assert bet_engine.settle(today=gd + dt.timedelta(days=1)) == 0


def test_settle_scores_all_four_arms(tmp_path, monkeypatch):
    """A row carrying all four stakes settles to four PnLs off ONE price."""
    import duckdb
    db = tmp_path / "scratch2.duckdb"

    def fake_connect(read_only=False):
        return duckdb.connect(str(db), read_only=read_only)

    monkeypatch.setattr(bet_engine, "_connect", fake_connect)
    con = fake_connect()
    bet_engine._ensure_schema(con)
    for ddl in AUX_DDL:
        con.execute(ddl)
    gd = dt.date(2026, 10, 25)
    now = dt.datetime.now(dt.timezone.utc)
    vals = {"candidate_ts": now, "game_date": gd, "game_id": "0022600043",
            "home": "BOS", "away": "NYK", "side": "away", "rule": "R4_LOWT",
            "p_us": 0.39, "p_us_side": 0.61, "p_mkt_side": 0.56, "edge": 0.05,
            "price_decimal": 2.05, "book": "bookB", "implied_p": 0.56,
            "stake_units": 1.0, "stake_raw_kelly": 4.0,
            "stake_shrunk_kelly": 0.0, "stake_open_shrunk": 2.0,
            "conf_excess": 0.05, "cap_in_force": 0.08, "shrunk_edge": 0.0,
            "open_shrunk_edge": 0.0247}
    cols = list(vals)
    con.execute("INSERT INTO bet_paper (" + ",".join(cols) + ") VALUES (" +
                ",".join("?" * len(cols)) + ")", [vals[c] for c in cols])
    # away side loses when the home team wins
    con.execute("INSERT INTO nba_games VALUES ('2026-27','0022600043',?,1,"
                "'BOS','BOS vs. NYK',true,'W',110,?)", [gd, now])
    con.close()
    assert bet_engine.settle(today=gd + dt.timedelta(days=1)) == 1
    con = fake_connect(read_only=True)
    o, kind, pf, pr, ps, po = con.execute(
        "SELECT outcome, snapshot_kind, pnl_units, pnl_raw_kelly, "
        "pnl_shrunk_kelly, pnl_open_shrunk FROM bet_paper").fetchone()
    con.close()
    assert o == 0
    assert kind == "POST_REPORT"            # schema default (legacy label)
    assert pf == -1.0                       # flat arm loses its 1u
    assert pr == -4.0                       # raw-Kelly loses its 4u
    assert ps == 0.0                        # shrunk arm never deployed capital
    assert po == -2.0                       # open_shrunk diagnostic loses its 2u


# ---- open_shrunk diagnostic arm (frozen D120/D121 open calibration) ---------

def test_open_shrunk_frozen_constants():
    """The frozen open-arm calibration is the D120/D121 registered fit — the
    values in data/bo_openbacktest.json kelly_slope PRIMARY|SP|OPEN — and it
    must never silently drift (FROZEN: refit = new D-line)."""
    c = bet_engine.OPEN_SHRUNK
    assert math.isclose(c["a"], -0.0037733442091709493, rel_tol=1e-12)
    assert math.isclose(c["b"], 0.5684830302091815, rel_tol=1e-12)
    assert c["n"] == 3848 and c["frozen"] is True
    # break-even claimed edge at the open ~ 0.0066 (vs 0.0758 at the close)
    assert math.isclose(-c["a"] / c["b"], 0.006637567, abs_tol=1e-6)


def test_open_shrunk_arm_stakes():
    """open_shrunk = shrunk-Kelly construction under the OPEN coefficients:
    at a 4.5%-overround consensus price it needs claimed edge > ~5.1% to
    stake; the close-fit arm needs > 27%+ (i.e. never, under the cap)."""
    st = bet_engine._stakes(0.60, 0.56, 1.0 / (0.56 * 1.045),
                            {"a": -0.0140, "b": 0.184})
    assert set(st) == set(bet_engine.ALL_ARMS)
    assert st["flat"] == 1.0
    assert st["shrunk_kelly"] == 0.0        # D117: calibrated close edge <= vig
    assert st["open_shrunk"] == 0.0         # edge .04 -> p+.019 still < 1/dec
    st2 = bet_engine._stakes(0.62, 0.56, 1.0 / (0.56 * 1.045),
                             {"a": -0.0140, "b": 0.184})
    assert st2["open_shrunk"] > 0.0         # edge .06 clears the open calib
                                            # (breakeven claimed edge ~.051)
    assert st2["shrunk_kelly"] == 0.0       # ... while the close calib never
    # no price -> flat books 1u, every Kelly arm stands down
    st3 = bet_engine._stakes(0.62, 0.56, None, {"a": -0.0140, "b": 0.184})
    assert st3 == {"flat": 1.0, "raw_kelly": 0.0, "shrunk_kelly": 0.0,
                   "open_shrunk": 0.0}
