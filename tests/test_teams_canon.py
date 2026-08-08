"""D172 — the canonical team vocabularies must not silently diverge.

The era/franchise crosswalk lived as three byte-identical inline copies in
scripts that produce CERTIFIED artifacts (k19_model.py, k19_t2.py,
ats19_score.py).  D172 canonicalised it as `nbapred.teams.FRANCHISE` but did
NOT edit those scripts, because editing a scored script is how a certified
number moves without anyone noticing.  These tests are the alternative
guarantee: if any copy drifts from the canonical map, this goes red.

Same bug class as D119 / D161 / D171 — a name map that nobody owned.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from nbapred.teams import (BBREF_TO_US, FRANCHISE, abbrev_for, modern,
                           resolve_map)

ROOT = Path(__file__).resolve().parent.parent
INLINE_COPIES = ["scripts/k19_model.py", "scripts/k19_t2.py",
                 "scripts/ats19_score.py"]


def _extract_dict(path: Path, name: str) -> dict:
    src = path.read_text()
    m = re.search(rf"^{name} = (\{{.*?\}})", src, re.S | re.M)
    assert m, f"{name} not found in {path}"
    return ast.literal_eval(m.group(1))


@pytest.mark.parametrize("rel", INLINE_COPIES)
def test_inline_franchise_copies_match_canonical(rel):
    """Every inline FRANCHISE dict equals nbapred.teams.FRANCHISE."""
    got = _extract_dict(ROOT / rel, "FRANCHISE")
    assert got == FRANCHISE, (
        f"{rel} FRANCHISE has drifted from nbapred.teams.FRANCHISE.\n"
        f"  only in {rel}: {set(got.items()) - set(FRANCHISE.items())}\n"
        f"  only in teams.py: {set(FRANCHISE.items()) - set(got.items())}")


def test_franchise_covers_every_relocation_in_corpus():
    """The six era codes nba_games emits over 1996-97..2025-26."""
    assert set(FRANCHISE) == {"SEA", "NJN", "NOH", "NOK", "VAN", "CHH"}
    assert set(FRANCHISE.values()) <= {"OKC", "BKN", "NOP", "MEM", "CHA"}


def test_modern_is_idempotent_and_total():
    """modern() must be a fixed point on its own output — otherwise a
    two-hop relocation (NOK -> NOH -> NOP) resolves differently by call order."""
    for k, v in FRANCHISE.items():
        assert modern(v) == v, f"{k}->{v} is not a fixed point"
        assert modern(modern(k)) == modern(k)
    assert modern("BOS") == "BOS"
    assert modern(None) is None


def test_bbref_vocabulary():
    assert BBREF_TO_US == {"BRK": "BKN", "CHO": "CHA", "PHO": "PHX"}
    for v in BBREF_TO_US.values():
        assert abbrev_for_is_modern(v)


def abbrev_for_is_modern(ab: str) -> bool:
    from nba_api.stats.static import teams as _t
    return ab in {t["abbreviation"] for t in _t.get_teams()}


def test_clippers_alias_still_resolves():
    """D171's regression: the injury PDFs say 'LA Clippers'."""
    assert abbrev_for("LA Clippers") == "LAC"
    assert abbrev_for("Los Angeles Clippers") == "LAC"


def test_resolve_map_reports_rather_than_drops():
    """The whole design point: an unresolvable name comes back, loudly."""
    ok, bad = resolve_map(["LA Clippers", "Boston Celtics", "da Silva, Tristan"])
    assert ok == {"LA Clippers": "LAC", "Boston Celtics": "BOS"}
    assert bad == ["da Silva, Tristan"]


# --------------------------------------------------------------------------
# D178: the CITY rule. Yahoo/BetMGM names every side by bare city, and before
# this rule 28 of 30 franchises were unresolvable through `nbapred.teams`.
# The rule is LAST in resolve(), so it must be strictly additive.
# --------------------------------------------------------------------------
MGM_CITY_STRINGS = [
    "Atlanta", "Boston", "Brooklyn", "Charlotte", "Chicago", "Cleveland",
    "Dallas", "Denver", "Detroit", "Golden State", "Houston", "Indiana",
    "LA Clippers", "LA Lakers", "Memphis", "Miami", "Milwaukee", "Minnesota",
    "New Orleans", "New York", "Oklahoma City", "Orlando", "Philadelphia",
    "Phoenix", "Portland", "Sacramento", "San Antonio", "Toronto", "Utah",
    "Washington",
]


def test_city_rule_resolves_every_yahoo_mgm_spelling():
    """All 30 sides of caseydurfee/mgm-grand-nba-betting-data resolve, and to
    30 DISTINCT franchises (a rule that collapsed two cities would still
    'resolve' 30 strings, so distinctness is the real assertion)."""
    abbrs = [abbrev_for(s) for s in MGM_CITY_STRINGS]
    assert all(a is not None for a in abbrs), \
        f"unresolved: {[s for s, a in zip(MGM_CITY_STRINGS, abbrs) if a is None]}"
    assert len(set(abbrs)) == 30, f"collapsed franchises: {sorted(abbrs)}"


def test_ambiguous_city_stays_unresolvable():
    """'Los Angeles' is TWO franchises. The whole point of D171 is that an
    ambiguous name must stay LOUD rather than silently pick one."""
    assert abbrev_for("Los Angeles") is None


def test_city_rule_is_strictly_additive():
    """Names that resolved before D178 must resolve to exactly the same code,
    and non-team strings must still return None."""
    for name, want in [("LA Clippers", "LAC"), ("Los Angeles Clippers", "LAC"),
                       ("Los Angeles Lakers", "LAL"),
                       ("Portland Trail Blazers", "POR"),
                       ("Portland Trailblazers", "POR"),
                       ("Golden State Warriors", "GSW")]:
        assert abbrev_for(name) == want, name
    for junk in ["Team Chuck", "Team Stars", "Team Stripes", "EAST", "WEST",
                 "World", "Melbourne United", "", None]:
        assert abbrev_for(junk) is None, junk


# ---- D178: the `da Silva, Tristan` parser artefact ---------------------------
# 30 rows of injury_reports_pit.team held a PLAYER NAME. Cause: the modern
# injury-PDF heuristic classified "Last, First" with a regex requiring an
# initial capital, so a lowercase nobiliary particle fell through to an
# unguarded `else: team = f` and then FORWARD-FILLED down the team's block.

def test_player_regex_accepts_lowercase_particles_and_never_a_team():
    """The classifier that decides player-vs-team, in isolation."""
    from nbapred.ingest.injury_pdf import PLAYER_RX as PLAYER
    for name in ("da Silva, Tristan",              # the artefact
                 "van Gundy, Stan", "de Colo, Nando", "di Vincenzo, Donte",
                 "Jackson Jr., Jaren", "Williams Jr., Vince",
                 "Tucker, P.J.", "O'Neale, Royce", "McClung, Mac"):
        assert PLAYER.match(name), name
    from nbapred.teams import known_report_names
    for team in known_report_names():              # 30 full names + LA Clippers
        assert not PLAYER.match(team), team


def test_da_silva_parses_as_an_orlando_player(tmp_path):
    """End-to-end on the real archived PDF: the row that produced the artefact
    must now yield player='da Silva, Tristan', team='Orlando Magic', and no
    later row in the block may inherit a player name as its team."""
    import pathlib
    from nbapred.ingest.injury_pdf import parse_pdf
    from nbapred.teams import known_report_names
    p = (pathlib.Path(__file__).resolve().parent.parent / "data" / "raw" /
         "injury_reports" / "Injury-Report_2024-10-26_05PM.pdf")
    if not p.exists():                     # archive optional in a bare checkout
        return
    rows = parse_pdf(p)
    ds = [r for r in rows if r["player"] == "da Silva, Tristan"]
    assert ds, "the player row must exist at all (it used to be dropped)"
    assert {r["team"] for r in ds} == {"Orlando Magic"}
    valid = known_report_names()
    bad = sorted({r["team"] for r in rows
                  if r["team"] is not None and r["team"] not in valid})
    assert bad == [], bad
