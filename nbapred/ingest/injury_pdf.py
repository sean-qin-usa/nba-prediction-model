"""Parse official NBA injury-report PDFs (data/raw/injury_reports/) into a
point-in-time injury_reports table. Free PIT injury feed back to 2023-10 —
unlocks report-based OUT-sets, W1/W3 windows, fresh-news redistribution.

Layout: pdftotext -layout preserves columns; header row per page gives column
offsets (Game Date | Game Time | Matchup | Team | Player Name | Current Status
| Reason). Game/matchup/team carry forward on blank cells; reason text wraps
onto lines above/below the player row.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
import subprocess
from pathlib import Path

log = logging.getLogger("injury_pdf")

STATUSES = {"Out", "Doubtful", "Questionable", "Probable", "Available"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS injury_reports_pit (
    report_date DATE NOT NULL, edition VARCHAR NOT NULL,
    game_date DATE, matchup VARCHAR, team VARCHAR,
    player VARCHAR NOT NULL, status VARCHAR NOT NULL, reason VARCHAR,
    PRIMARY KEY (report_date, edition, game_date, team, player)
);
"""

HDR = re.compile(r"Game Date\s+Game Time\s+Matchup\s+Team\s+Player Name\s+Current Status\s+Reason")

# --- modern-layout fragment classifiers (module level since D178 so the one
# --- that carried the bug is directly unit-testable) ------------------------
_DATE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
_TIME = re.compile(r"^\d{2}:\d{2}\s*\(ET\)$")
_MATCH = re.compile(r"^[A-Z]{3}@[A-Z]{3}$")
_STATRX = re.compile(r"\s{2,}(Out|Doubtful|Questionable|Probable|Available)"
                     r"(?:\s{2,}(.*))?$")
# "Last, First".  D178: the old pattern REQUIRED an initial capital, so a
# lowercase nobiliary particle ("da Silva, Tristan", and the same class of name:
# van/de/di/del/le/ter/von) failed to match, fell through to an unguarded
# `else: team = f` in parse_pdf and was written into `team`.  TWO consequences,
# both measured on the archive:
#   * the player's OWN row was dropped entirely (no `player` -> no row appended)
#     — Tristan da Silva had no injury status at all, ever;
#   * the poisoned value FORWARD-FILLED to every later row in that team's block,
#     leaving 30 rows of injury_reports_pit.team holding the string
#     "da Silva, Tristan", which `slate_context`'s `team_id_for` resolves to
#     None — silently emptying those Orlando out-sets.
# The particle is bounded to 1-3 lowercase letters (plus an optional apostrophe
# form) so no team name can match: team names carry no comma, and none begins
# with a short lowercase word.
PLAYER_RX = re.compile(
    r"^(?:[a-z]{1,3}(?:'[a-z]+)? )?[A-Z][\w'.-]+(?: [A-Z][\w'.-]+)*, ")


def _cols(header: str) -> list[int]:
    return [header.index(k) for k in
            ("Game Date", "Game Time", "Matchup", "Team", "Player Name",
             "Current Status", "Reason")]


# --- D170: LEGACY COLUMN LAYOUTS -------------------------------------------
# The 2018-12-17..2019-12-16 reports do not have the modern column set, and the
# heuristic parser below silently mangles them: it takes the LAST status-like
# token on the line, which in those layouts is **Previous Status**, and it
# dumps the Reason text into `team` (measured: 45-56 distinct "teams" on a
# 30-team league). Four header variants exist in the archive:
#
#   Game Date Game Time Matchup Team Player Name Current Status Reason
#        (2019-12-23 .. 2026-04-10)  <- MODERN, parsed by parse_pdf below
#   Game Date Game Time Matchup Team Player Name Category Reason Current Status Previous Status
#        (2018-12-17 .. 2019-11-11)
#   Game Date Game Time Matchup Team Player Name Reason Current Status Previous Status
#        (2019-11-18)
#   Game Date Game Time Matchup Team Player Name Current Status Reason Previous Status
#        (2019-11-25 .. 2019-12-16)
#
# These are parsed by COLUMN POSITION off the header row instead of by
# heuristic, which is unambiguous. The modern path is left byte-identical so no
# certified row moves.
_KNOWN_COLS = ("Game Date", "Game Time", "Matchup", "Team", "Player Name",
               "Category", "Reason", "Current Status", "Previous Status")
_MODERN_COLS = ("Game Date", "Game Time", "Matchup", "Team", "Player Name",
                "Current Status", "Reason")
_FRAG = re.compile(r"\S(?:.*?\S)?(?=\s{2,}|$)")


_TEAMS_CACHE: set[str] | None = None


def _team_names() -> set[str]:
    """Real NBA team full names, plus the strings the PDFs actually use.

    D171: sourced from `nbapred.teams` so the alias list lives in exactly one
    place. The set is identical to the previous inline literal (30 full names
    + "LA Clippers"), so parsing behaviour is unchanged — it was only the
    CONSUMERS of the parsed rows that dropped the alias."""
    global _TEAMS_CACHE
    if _TEAMS_CACHE is None:
        from ..teams import known_report_names
        _TEAMS_CACHE = known_report_names()
    return _TEAMS_CACHE


def _header_line(txt: str) -> str | None:
    for line in txt.splitlines():
        if "Game Date" in line and "Player Name" in line:
            return line
    return None


def _is_modern(header: str) -> bool:
    present = [c for c in _KNOWN_COLS if c in header]
    return present == [c for c in _KNOWN_COLS if c in _MODERN_COLS]


def _spans(header: str):
    cols = sorted((header.index(k), k) for k in _KNOWN_COLS if k in header)
    out = []
    for i, (pos, name) in enumerate(cols):
        end = cols[i + 1][0] if i + 1 < len(cols) else 10 ** 6
        out.append((name, pos, end))
    return out


def _parse_columnar(txt: str, rdate: str, ed: str) -> list[dict]:
    # pdftotext -layout lays each PAGE out independently, so column offsets
    # drift page to page. Re-read the header every time one appears instead of
    # pinning the first page's offsets for the whole document.
    spans = _spans(_header_line(txt))

    rows: list[dict] = []
    gdate = matchup = team = None
    for line in txt.splitlines():
        if not line.strip():
            continue
        s = line.strip()
        if "Game Date" in line and "Player Name" in line:
            # carry gdate/matchup/team ACROSS the page break: a team's block
            # routinely spans one
            spans = _spans(line)
            continue
        if s.startswith("Injury Report:") or s.startswith("Page ") or \
                "NOT YET SUBMITTED" in line:
            continue
        # assign each whitespace-delimited fragment to the column whose span
        # contains its start offset (tolerant of values overflowing a column)
        cell = {name: "" for name, _a, _b in spans}
        for m in _FRAG.finditer(line):
            st = m.start()
            for name, a, b in spans:
                if a - 2 <= st < b:
                    cell[name] = (cell[name] + " " + m.group()).strip()
                    break
        if cell.get("Game Date"):
            gdate = cell["Game Date"]
        if cell.get("Matchup"):
            matchup = cell["Matchup"]
        player = cell.get("Player Name", "")
        status = cell.get("Current Status", "")
        if cell.get("Team"):
            # A two-word team name is routinely WRAPPED onto its own line
            # ("Minnesota" on the row, "Timberwolves" underneath). Such a line
            # has ONLY the Team cell filled; append it and retro-fix the rows
            # already emitted under the partial name.
            if not player and status not in STATUSES and \
                    not any(cell.get(c) for c in ("Game Date", "Matchup",
                                                  "Category", "Reason")):
                frag = cell["Team"]
                valid = _team_names()
                if team and f"{team} {frag}" in valid:
                    partial, team = team, f"{team} {frag}"
                else:
                    # the wrap may arrive when `team` still holds the PREVIOUS
                    # block's name; resolve the fragment on its own if it is
                    # the unambiguous tail of exactly one real team
                    cands = [v for v in valid if v.endswith(" " + frag)]
                    if len(cands) != 1:
                        continue          # not a team wrap; ignore the line
                    partial, team = team, cands[0]
                for r in rows:
                    if r["team"] == partial:
                        r["team"] = team
                continue
            team = cell["Team"]
            team_rows_start = len(rows)  # noqa: F841
        if not player or status not in STATUSES:
            # continuation line: reason text wrapped under its own row
            frag = " ".join(x for x in (cell.get("Category", ""),
                                        cell.get("Reason", "")) if x and x != "-")
            if rows and frag and not player:
                rows[-1]["reason"] = (rows[-1]["reason"] + " " + frag).strip()
            continue
        cat = cell.get("Category", "").strip()
        rsn = cell.get("Reason", "").strip()
        # compose to the MODERN reason shape ("Injury/Illness - Left Ankle Sprain")
        if cat and cat != "-":
            reason = f"{cat} - {rsn}" if rsn and rsn != "-" else cat
        else:
            reason = rsn
        rows.append(dict(report_date=rdate, edition=ed, game_date=gdate,
                         matchup=matchup, team=team, player=player,
                         status=status, reason=reason))
    return rows


def parse_pdf(path: Path) -> list[dict]:
    txt = subprocess.run(["pdftotext", "-layout", str(path), "-"],
                         capture_output=True, text=True, timeout=60).stdout
    # D170: the league renamed the file on 2026-01-01 from
    # `Injury-Report_2025-12-21_05PM.pdf` to `Injury-Report_2026-01-01_05_00PM.pdf`
    # (HH_MM instead of HH). The old pattern raised AttributeError on every new
    # file, load_all() swallowed it as a parse failure, and 97 archived
    # report-days (2026-01-01..2026-04-12, the whole back half of 2025-26)
    # silently never reached injury_reports_pit. The minute field is normalised
    # away so the edition key stays '05PM' and joins to the pre-2026 rows.
    m = re.search(r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})(?:_(\d{2}))?([AP]M)",
                  path.name)
    if m is None:
        raise ValueError(f"unrecognised injury-report filename: {path.name}")
    rdate = m.group(1)
    ed = f"{m.group(2)}{m.group(4)}"
    # D170: legacy column layouts (2018-12-17..2019-12-16) go through the
    # header-position parser; the heuristic below is only valid for the modern
    # column set and silently mis-reads the others.
    _h = _header_line(txt)
    if _h is not None and not _is_modern(_h):
        return _parse_columnar(txt, rdate, ed)
    DATE, TIME, MATCH, PLAYER, STATRX = (_DATE, _TIME, _MATCH, PLAYER_RX,
                                         _STATRX)
    rows = []
    gdate = matchup = team = None
    pending = ""             # reason fragment wrapped ABOVE its player row
    last_was_player = False  # wrap-BELOW follows its player row directly
    for line in txt.splitlines():
        if not line.strip():
            last_was_player = False   # blank breaks below-adjacency, keeps pending
            continue
        s = line.strip()
        if (HDR.search(line) or s.startswith("Injury Report:")
                or s.startswith("Page ") or "NOT YET SUBMITTED" in line):
            pending = ""; last_was_player = False
            continue
        m = STATRX.search(line.rstrip())
        if m:
            status, reason = m.group(1), (m.group(2) or "").strip()
            left = [f for f in re.split(r"\s{2,}", line[:m.start()].strip()) if f]
            player = ""
            for f in left:
                if DATE.match(f):
                    gdate = f
                elif TIME.match(f):
                    pass
                elif MATCH.match(f):
                    matchup = f
                elif PLAYER.match(f):
                    player = f
                elif f in _team_names():
                    team = f
                else:
                    # D178 BACKSTOP.  Previously this branch was an unguarded
                    # `team = f`, so ANY fragment the classifiers above failed
                    # to recognise became the team and then forward-filled
                    # through the rest of the block.  A string that is neither
                    # a known team nor a parseable player is a PARSER MISS:
                    # REPORT it and leave `team` alone (D171's teams.py law —
                    # report, never silently corrupt).  Carrying the previous
                    # team forward is also the correct guess, because the Team
                    # cell is blank on continuation rows by design.
                    log.warning("injury_pdf: unrecognised Team/Player fragment "
                                "%r (keeping team=%r) in %s %s",
                                f, team, rdate, ed)
            if player:
                rows.append(dict(report_date=rdate, edition=ed, game_date=gdate,
                                 matchup=matchup, team=team, player=player,
                                 status=status,
                                 reason=(pending + " " + reason).strip()))
                pending = ""; last_was_player = True
                continue
        # no player+status on this line: reason fragment (wrap above/below)
        frags = [f for f in re.split(r"\s{2,}", s) if f]
        if len(frags) == 1 and not DATE.match(frags[0]) and not MATCH.match(frags[0]):
            if last_was_player and rows:
                rows[-1]["reason"] = (rows[-1]["reason"] + " " + frags[0]).strip()
            else:
                pending = (pending + " " + frags[0]).strip()
        last_was_player = False
    return rows


def load_all(connect_fn, raw_dir: Path) -> dict:
    import pandas as pd
    files = sorted(raw_dir.glob("Injury-Report_*.pdf"))
    all_rows, bad = [], 0
    for f in files:
        try:
            all_rows.extend(parse_pdf(f))
        except Exception:
            bad += 1
            log.exception("parse failed %s", f.name)
    df = pd.DataFrame(all_rows)
    df["game_date"] = pd.to_datetime(df.game_date, format="%m/%d/%Y",
                                     errors="coerce").dt.date
    df["report_date"] = pd.to_datetime(df.report_date).dt.date
    df = df.dropna(subset=["game_date"]).drop_duplicates(
        subset=["report_date", "edition", "game_date", "team", "player"])
    con = connect_fn()
    con.execute(SCHEMA)
    con.execute("DELETE FROM injury_reports_pit")
    con.execute("""INSERT INTO injury_reports_pit
        SELECT report_date, edition, game_date, matchup, team, player, status, reason
        FROM df""")
    con.close()
    return {"files": len(files), "bad": bad, "rows": len(df)}
