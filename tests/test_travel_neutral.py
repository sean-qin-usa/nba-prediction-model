"""D140: neutral-site travel. Pins the bug D139 found and the fix's contract.

The defect: `nbapred/model/travel.py` derived the venue from `matchup`, so the
2020 Orlando bubble — 88 games at ONE complex, TRUE travel 0 km/team-game —
was assigned 1,505.5 km/team-game, and the nominal host was then carried
forward as the ORIGIN of each team's next game. D136's two "SIG" margin
coefficients existed only in the frame containing those games.

Most tests here run against a fake connection so they pin the LOGIC without a
DB; the corpus-level assertions are skipped when data/nba.duckdb is absent.
"""
import datetime as dt

import pytest

from nbapred.model import travel as TV


# ------------------------------------------------------------------ fake DB

class _Res:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class FakeCon:
    """Minimal stand-in: dispatches build_state's three queries by SQL text."""

    def __init__(self, rows, bubble_ids=(), both_away_ids=()):
        self.rows = rows                      # (season, gid, date, tid, ab, matchup)
        self.bubble_ids = list(bubble_ids)
        self.both_away_ids = list(both_away_ids)

    def execute(self, q, params=None):
        if "SELECT DISTINCT game_id" in q:
            return _Res([(g,) for g in self.bubble_ids])
        if "GROUP BY game_id" in q:
            return _Res([(g,) for g in self.both_away_ids])
        return _Res(sorted(self.rows, key=lambda r: (r[2], r[1])))


def _g(gid, date, home, away, season="2019-20"):
    """One game -> two nba_games rows, matchup written from the AWAY side."""
    mu_a, mu_h = f"{away} @ {home}", f"{home} vs. {away}"
    return [(season, gid, date, hash(home) % 10**6, home, mu_h),
            (season, gid, date, hash(away) % 10**6, away, mu_a)]


def _tid(ab):
    return hash(ab) % 10**6


# ------------------------------------------------------------ venue plumbing

def test_neutral_venue_keys_cannot_collide_with_team_abbrevs():
    assert not (set(TV.NEUTRAL_VENUES) & set(TV.arenas()))


def test_venues_is_arenas_plus_neutral_venues():
    v = TV.venues()
    assert set(v) == set(TV.arenas()) | set(TV.NEUTRAL_VENUES)
    assert v[TV.BUBBLE_VENUE]["arena"].startswith("ESPN Wide World")


def test_haversine_resolves_neutral_venue_keys():
    # the bubble complex really is near Orlando's arena, and nowhere near LAL
    assert TV.haversine_km(TV.BUBBLE_VENUE, "ORL") < 40
    assert TV.haversine_km(TV.BUBBLE_VENUE, "LAL") > 3000
    assert TV.haversine_km(TV.BUBBLE_VENUE, TV.BUBBLE_VENUE) == 0.0


def test_unknown_venue_key_is_zero_distance_not_an_exception():
    assert TV.haversine_km("NEUTRAL:00224001", "BOS") == 0.0
    assert TV.utc_offset_h("NEUTRAL:00224001", dt.date(2025, 1, 23)) == 0.0


# ------------------------------------------- the bug, on a synthetic schedule

def test_same_neutral_venue_twice_gives_zero_travel_both_teams():
    """THE BUG. Two 'road' games at one neutral site, nominal hosts 2,000+ km
    apart. Pre-fix the second game charged both teams the host-to-host
    distance; post-fix it is 0 because the venue never changed."""
    rows = (_g("A1", dt.date(2020, 7, 30), "ORL", "LAL")
            + _g("A2", dt.date(2020, 8, 1), "BOS", "LAL"))
    con = FakeCon(rows, bubble_ids=["A1", "A2"])
    st = TV.build_state(con)
    for ab in ("LAL",):
        s2 = st[(_tid(ab), dt.date(2020, 8, 1))]
        assert s2["travel_km"] == 0.0
        assert s2["tz_east"] == 0.0
        assert s2["elev_gain_m"] == 0.0
        assert s2["travel_valid"] is True
    # ORL's nominal "home" game is NOT a home game
    assert st[(_tid("ORL"), dt.date(2020, 7, 30))]["at_home"] is False
    assert st[(_tid("ORL"), dt.date(2020, 7, 30))]["neutral"] is True


def test_neutral_game_does_not_poison_the_next_games_origin():
    """The second half of the bug: after a neutral game the ORIGIN of the next
    game must be the neutral VENUE, never the nominal host."""
    rows = (_g("B1", dt.date(2020, 7, 30), "LAL", "BOS")   # neutral, at WWoS
            + _g("B2", dt.date(2020, 8, 1), "ORL", "BOS"))  # still at WWoS
    con = FakeCon(rows, bubble_ids=["B1", "B2"])
    st = TV.build_state(con)
    # pre-fix BOS would be charged LAL->ORL (~3,700 km); the truth is 0
    assert st[(_tid("BOS"), dt.date(2020, 8, 1))]["travel_km"] == 0.0


def test_travel_out_of_a_neutral_venue_uses_that_venue_as_origin():
    """Leaving the bubble for a real arena is real travel measured FROM the
    bubble, not from the nominal host of the bubble game."""
    rows = (_g("C1", dt.date(2020, 7, 30), "LAL", "BOS")
            + _g("C2", dt.date(2020, 8, 5), "BOS", "NYK"))
    con = FakeCon(rows, bubble_ids=["C1"])
    st = TV.build_state(con)
    got = st[(_tid("BOS"), dt.date(2020, 8, 5))]["travel_km"]
    assert got == pytest.approx(TV.haversine_km(TV.BUBBLE_VENUE, "BOS"), rel=1e-9)
    # and emphatically NOT the LAL->BOS distance the old code would have used
    assert abs(got - TV.haversine_km("LAL", "BOS")) > 1000


def test_flagged_neutral_with_unknown_geo_is_invalid_not_zero():
    """A feed-flagged neutral court we have no coordinates for must report
    travel_valid=False — for the game itself AND for the next game, whose
    origin is equally unknown — instead of a fabricated distance."""
    rows = (_g("D0", dt.date(2025, 1, 21), "SAS", "DAL", season="2024-25")
            + _g("D1", dt.date(2025, 1, 23), "IND", "SAS", season="2024-25")
            + _g("D2", dt.date(2025, 1, 27), "SAS", "DAL", season="2024-25"))
    con = FakeCon(rows, both_away_ids=["D1"])
    st = TV.build_state(con)
    s1 = st[(_tid("SAS"), dt.date(2025, 1, 23))]
    assert s1["neutral"] is True and s1["at_home"] is False
    assert s1["travel_valid"] is False and s1["travel_km"] == 0.0
    s2 = st[(_tid("SAS"), dt.date(2025, 1, 27))]
    assert s2["neutral"] is False          # a normal home game...
    assert s2["travel_valid"] is False     # ...but from an unknown origin
    assert s2["travel_km"] == 0.0


def test_ordinary_games_are_untouched_by_the_fix():
    rows = (_g("E1", dt.date(2024, 11, 1), "BOS", "LAL", season="2024-25")
            + _g("E2", dt.date(2024, 11, 3), "NYK", "LAL", season="2024-25"))
    st = TV.build_state(FakeCon(rows))
    s = st[(_tid("LAL"), dt.date(2024, 11, 3))]
    assert s["travel_km"] == pytest.approx(TV.haversine_km("BOS", "NYK"), rel=1e-9)
    assert s["travel_valid"] is True and s["neutral"] is False
    assert st[(_tid("NYK"), dt.date(2024, 11, 3))]["at_home"] is True


def test_hiatus_longer_than_the_reset_window_zeroes_acute_travel():
    """The 2020 restart followed a 141-day shutdown; acute travel load does not
    survive that, so the chain restarts exactly as it does at a season opener."""
    rows = (_g("F1", dt.date(2020, 3, 11), "LAL", "BOS")
            + _g("F2", dt.date(2020, 7, 30), "ORL", "BOS"))
    st = TV.build_state(FakeCon(rows, bubble_ids=["F2"]))
    assert st[(_tid("BOS"), dt.date(2020, 7, 30))]["travel_km"] == 0.0
    # a normal gap still charges travel
    rows2 = (_g("G1", dt.date(2020, 3, 1), "LAL", "BOS")
             + _g("G2", dt.date(2020, 3, 1 + TV.HIATUS_RESET_DAYS), "ORL", "BOS"))
    st2 = TV.build_state(FakeCon(rows2))
    assert st2[(_tid("BOS"), dt.date(2020, 3, 1 + TV.HIATUS_RESET_DAYS))]["travel_km"] > 3000


# ------------------------------------------------------- against the corpus

def _con():
    try:
        from nbapred.db import connect
        return connect(read_only=True)
    except Exception:                                        # pragma: no cover
        pytest.skip("data/nba.duckdb unavailable")


def test_corpus_bubble_has_exactly_zero_travel_per_team_game():
    con = _con()
    try:
        st = TV.build_state(con)
    finally:
        con.close()
    bub = [v for (t, d), v in st.items()
           if TV.BUBBLE_FROM <= d <= TV.BUBBLE_TO and v["neutral"]]
    assert len(bub) == 176, "88 bubble games x 2 team-rows"
    assert all(v["venue"] == TV.BUBBLE_VENUE for v in bub)
    # the registered defect was 1,505.5 km/team-game
    assert max(v["travel_km"] for v in bub) == 0.0
    assert max(abs(v["tz_east"]) for v in bub) == 0.0
    assert max(abs(v["elev_gain_m"]) for v in bub) == 0.0
    assert all(v["at_home"] is False for v in bub)
    assert all(v["travel_valid"] is True for v in bub)


def test_corpus_feed_flagged_neutral_games_are_detected():
    con = _con()
    try:
        neutral = TV.neutral_game_venues(con)
    finally:
        con.close()
    flagged = [g for g, v in neutral.items() if v != TV.BUBBLE_VENUE]
    bubble = [g for g, v in neutral.items() if v == TV.BUBBLE_VENUE]
    assert len(bubble) == 88
    assert len(flagged) == 10, "D137: 10 is_home-FALSE-on-both games"


def test_corpus_scorable_seasons_have_no_hiatus_reset():
    """The reset must be structurally bubble-only: if a scorable season ever
    developed a >14-day gap this test fails loudly rather than silently
    changing a gated feature."""
    con = _con()
    try:
        rows = con.execute(
            """SELECT season, team_id, game_date FROM nba_games
               WHERE game_id LIKE '002%' AND season <> ?
               ORDER BY game_date""", [TV.BUBBLE_SEASON]).fetchall()
    finally:
        con.close()
    last = {}
    worst = 0
    for season, tid, gd in rows:
        gd = gd.date() if hasattr(gd, "date") else gd
        k = (tid, season)
        if k in last:
            worst = max(worst, (gd - last[k]).days)
        last[k] = gd
    assert worst <= TV.HIATUS_RESET_DAYS, f"max gap {worst}d outside 2019-20"
