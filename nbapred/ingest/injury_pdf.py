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


def _cols(header: str) -> list[int]:
    return [header.index(k) for k in
            ("Game Date", "Game Time", "Matchup", "Team", "Player Name",
             "Current Status", "Reason")]


def parse_pdf(path: Path) -> list[dict]:
    txt = subprocess.run(["pdftotext", "-layout", str(path), "-"],
                         capture_output=True, text=True, timeout=60).stdout
    m = re.search(r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2}[AP]M)", path.name)
    rdate, ed = m.group(1), m.group(2)
    DATE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
    TIME = re.compile(r"^\d{2}:\d{2}\s*\(ET\)$")
    MATCH = re.compile(r"^[A-Z]{3}@[A-Z]{3}$")
    PLAYER = re.compile(r"^[A-Z][\w'.-]+(?: [A-Z][\w'.-]+)*, ")   # "Last, First"
    STATRX = re.compile(r"\s{2,}(Out|Doubtful|Questionable|Probable|Available)"
                        r"(?:\s{2,}(.*))?$")
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
                else:
                    team = f
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
