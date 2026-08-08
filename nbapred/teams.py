"""Canonical NBA team-name normalisation — ONE place, so the spelling of a
franchise can never again be a silent data loss.

WHY THIS EXISTS (D171).  The official injury-report PDFs spell the Clippers
**"LA Clippers"**.  `nba_api`'s `teams.get_teams()[i]["full_name"]` spells them
**"Los Angeles Clippers"**.  Five separate consumers built the map
`{full_name: abbreviation}` inline and looked the PDF string up in it; the
lookup returned `None` and the row was dropped **without a word**:

    name2ab = {t["full_name"]: t["abbreviation"] for t in _t.get_teams()}
    ab = name2ab.get(team)
    if ab: ...                      # <-- everything else silently vanishes

Consequence, measured in D171: **all 2,514 Clippers injury-report rows (2,119
of them status='Out', 1,919 of those same-day) had never entered a T1 or T2
out-set in any season, including the certified ones**, and the Clippers'
rest/management shutdown signal in `model/tanking.py` was identically blank.
One franchise in thirty was scored report-blind for the life of the project.
D170 §6 found it and left it for the owner; this module fixes it.

This is the third instance of the same bug class in this repo (D119's
"63% scrape failure" that was a mapping bug; D161's 938 games lost to era
abbreviations), so the design goal here is not merely "handle LA Clippers" but
**make an unresolvable name loud instead of silent**:

  * `abbrev_for` / `team_id_for` resolve exact name, then explicit alias, then
    a unique-nickname suffix match, and return None only when all three miss.
  * `resolve_map` hands back the unresolved strings with their row counts so a
    caller can log, assert or raise instead of quietly dropping.

`nba_api` is imported lazily and its result cached: the static team table is a
bundled list, but the import is not free and several callers are hot loops.
"""
from __future__ import annotations

from functools import lru_cache

# Spellings that real feeds emit which are NOT nba_api's `full_name`.
# Keyed alias -> canonical nba_api full_name.  Extend here, nowhere else.
TEAM_NAME_ALIASES: dict[str, str] = {
    "LA Clippers": "Los Angeles Clippers",      # every injury-report PDF
    "Los Angeles Clippers": "Los Angeles Clippers",
    "LA Lakers": "Los Angeles Lakers",          # not seen in our feeds; cheap
    "Portland Trailblazers": "Portland Trail Blazers",
    "Golden State Warrios": "Golden State Warriors",   # observed typo class
    # D177: Yahoo/BetMGM spells every side as a BARE CITY ("Atlanta", "Miami").
    # The unique-city rule below handles 27 of them, but two cannot come from
    # nba_api's `city` field: GSW's city there is "San Francisco", and the two
    # Los Angeles franchises share a city and must stay ambiguous.
    "Golden State": "Golden State Warriors",
}


# ---------------------------------------------------------------------------
# ERA / FRANCHISE CROSSWALK (D161, canonicalised at D172).
#
# `nba_games.team_abbrev` carries the abbreviation IN FORCE THAT SEASON, while
# every odds feed carries the MODERN franchise code.  The market join is
# (game_date, home, away), so an unmapped era code silently deletes the whole
# franchise from the frame — D161 measured **938 games** lost that way.
#
# D172 verified this map against the data: over 1996-97..2025-26 the ONLY
# non-modern codes `nba_games` ever emits for a REAL franchise are the six
# below (4,270 team-game rows: NJN 1,396 / SEA 1,021 / NOH 802 / CHH 495 /
# VAN 378 / NOK 178), and after this map the set difference
# (nba_games codes − odds codes) is empty on every season.
#
# NOTE the one that is NOT here: the 1996-97 Washington Bullets.  `nba_games`
# already spells them WAS (nba_api back-stamps the modern code), so WSB never
# appears on our side — but `data/arenas.csv` carries a WSB row that therefore
# joins to nothing.  Harmless, and documented rather than deleted.
FRANCHISE: dict[str, str] = {
    "SEA": "OKC",   # Seattle SuperSonics    -> Oklahoma City, 2008-09
    "NJN": "BKN",   # New Jersey Nets        -> Brooklyn,      2012-13
    "NOH": "NOP",   # New Orleans Hornets    -> Pelicans,      2013-14
    "NOK": "NOP",   # NO/Oklahoma City Hornets (2005-06..2006-07, Katrina)
    "VAN": "MEM",   # Vancouver Grizzlies    -> Memphis,       2001-02
    "CHH": "CHA",   # original Charlotte Hornets -> (to NOH 2002-03); the
                    #   2004-05 expansion Bobcats inherit CHA/CHO
}

# Basketball-Reference uses a THIRD vocabulary for three live franchises.
BBREF_TO_US: dict[str, str] = {"BRK": "BKN", "CHO": "CHA", "PHO": "PHX"}


def modern(ab: str | None) -> str | None:
    """Season abbreviation -> modern franchise code.  Identity when already
    modern; unknown strings pass through unchanged (the caller audits)."""
    if ab is None:
        return None
    return FRANCHISE.get(ab, ab)


@lru_cache(maxsize=1)
def _static():
    """(full_name->row, nickname->row, city->row) for the current 30 franchises.

    `by_city` keeps ONLY cities that map to exactly one franchise, so the two
    Los Angeles clubs are deliberately absent and a bare "Los Angeles" stays
    unresolvable (loud) rather than silently becoming one of them.
    """
    from nba_api.stats.static import teams as _t
    rows = _t.get_teams()
    by_name = {r["full_name"]: r for r in rows}
    by_nick: dict[str, dict] = {}
    for r in rows:
        # only keep nicknames that are unambiguous across the league
        by_nick.setdefault(r["nickname"], r)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["city"]] = counts.get(r["city"], 0) + 1
    by_city = {r["city"]: r for r in rows if counts[r["city"]] == 1}
    return by_name, by_nick, by_city


@lru_cache(maxsize=4096)
def resolve(name: str | None) -> dict | None:
    """Full nba_api team row for a feed spelling, or None if it is not a team.

    Order: exact full_name -> explicit alias -> unique-nickname suffix ->
    exact unambiguous city.
    The suffix rule is what makes an unseen spelling drift ("LA Clippers",
    "L.A. Clippers", "Los Angeles Clippers") resolve instead of vanishing; it
    cannot fire on a non-team string, because every one of the 30 nicknames is
    a distinct final token group and none is a suffix of another.

    The city rule (D177) is LAST and is therefore strictly additive: it can
    only fire where the three older rules already returned None, so it cannot
    change any name this module resolved before. It exists because Yahoo/BetMGM
    names sides by bare city ("Atlanta", "Utah") and without it 28 of 30
    franchises were unresolvable - the D171 failure mode, caught loudly.
    """
    if not name:
        return None
    s = str(name).strip()
    if not s:
        return None
    by_name, by_nick, by_city = _static()
    row = by_name.get(s)
    if row is not None:
        return row
    alias = TEAM_NAME_ALIASES.get(s)
    if alias is not None:
        return by_name.get(alias)
    for nick, r in by_nick.items():
        if s.endswith(nick):
            return r
    return by_city.get(s)


def abbrev_for(name: str | None) -> str | None:
    """'LA Clippers' -> 'LAC'.  None when the string is not a franchise."""
    r = resolve(name)
    return r["abbreviation"] if r else None


def team_id_for(name: str | None) -> int | None:
    """'LA Clippers' -> 1610612746.  None when the string is not a franchise."""
    r = resolve(name)
    return int(r["id"]) if r else None


def known_report_names() -> set[str]:
    """Every string a parser should accept as a team heading in a report PDF."""
    by_name, _, _ = _static()
    return set(by_name) | set(TEAM_NAME_ALIASES)


def resolve_map(names) -> tuple[dict[str, str], list[str]]:
    """(resolved {name: abbrev}, unresolved [name]) — for audits and asserts.

    Use this instead of a bare dict lookup whenever dropping a row would be a
    silent correctness loss; log or raise on the second element.
    """
    ok: dict[str, str] = {}
    bad: list[str] = []
    for n in names:
        ab = abbrev_for(n)
        (ok.__setitem__(n, ab) if ab else bad.append(n))
    return ok, bad
