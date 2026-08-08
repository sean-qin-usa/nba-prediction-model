#!/usr/bin/env python3
"""D175 pilot: parse archived CBS Sports NBA injury pages -> tidy rows.

Server-rendered `<table class="data">`, one per team:
  title row  -> /nba/teams/page/<ABBR>/<team-slug>
  label row  -> Updated | Player | Pos | Injury | Expected Return
  data rows  -> 12/08/15 | <a .../playerpage/<id>/<slug>>Name</a> | C | Hip |
                Probable for Dec. 9 at Dallas

Writes a CSV only. NOTHING touches DuckDB.
"""
from __future__ import annotations
import csv, html, re, sys
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
RAW = REPO / "data" / "raw" / "unofficial" / "wayback"

TBL = re.compile(r'<table[^>]*class="data"[^>]*>(.*?)</table>', re.S | re.I)
TITLE = re.compile(r'<tr class="title">.*?/nba/teams/page/([A-Z]{2,3})/([a-z0-9\-]+)', re.S)
ROW = re.compile(r'<tr class="row\d"[^>]*>(.*?)</tr>', re.S | re.I)
CELL = re.compile(r'<td[^>]*>(.*?)</td>', re.S | re.I)
PLAYER = re.compile(r'/nba/players/playerpage/(\d+)/([a-z0-9\-\.]+)')
TAGS = re.compile(r'<[^>]+>')


def txt(s: str) -> str:
    s = TAGS.sub(" ", s)
    s = html.unescape(s).replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
          "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
VAGUE = {"early": 5, "mid": 15, "late": 25}
UNTIL = re.compile(
    r"until at least\s+(?:(early|mid|late)[\s\-]+)?"
    r"([a-z]{3})[a-z]*\.?\s*(\d{1,2})?", re.I)


def until_date(expected: str, season: str):
    """Parse CBS's 'out until at least <date>' into a real date.

    CRITICAL FOR PRECISION: a row saying 'out until at least Nov 25', carried
    forward and read on Dec 5, is NOT a statement that the player is out. Until
    D175 added this, such rows were counted as OUT for the rest of the season.
    `season` is 'YYYY-YY'; Oct-Dec belong to the first year, Jan-Sep the second.
    """
    m = UNTIL.search(expected)
    if not m:
        return None
    mon = MONTHS.get(m.group(2).lower())
    if not mon:
        return None
    day = int(m.group(3)) if m.group(3) else VAGUE.get(
        (m.group(1) or "mid").lower(), 15)
    y0 = int(season[:4])
    yr = y0 if mon >= 10 else y0 + 1
    try:
        return dt.date(yr, mon, min(day, 28 if mon == 2 else 30))
    except ValueError:
        return None


def classify(expected: str) -> str:
    """Map CBS's free-text `Expected Return` to a status token.

    D175: CBS changed wording between seasons — 2015-16 writes 'Out until at
    least Feb 15', 2016-17 writes 'Expected to be out until at least Feb 1'.
    Missing the second form silently classified 3,209 of 6,001 2016-17 rows as
    OTHER and made the season look like it had no OUT listings at all.
    """
    e = expected.lower().strip()
    if "out for season" in e or "out for the season" in e:
        return "OUT_SEASON"
    if "game time decision" in e or "game-time decision" in e:
        return "GTD"
    if e.startswith("questionable") or " questionable" in e:
        return "QUESTIONABLE"
    if e.startswith("doubtful") or " doubtful" in e:
        return "DOUBTFUL"
    if e.startswith("probable") or " probable" in e:
        return "PROBABLE"
    if ("out until" in e or "out indefinitely" in e or e.startswith("out ")
            or e == "out" or "expected to miss" in e or "expected to be out" in e):
        return "OUT"
    if "day to day" in e or "day-to-day" in e:
        return "DTD"
    return "OTHER"


def main() -> None:
    global SEASON
    SEASON = sys.argv[2]
    src = RAW / sys.argv[1]
    out = RAW / f"{sys.argv[1]}_rows.csv"
    n_files = n_rows = 0
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["snap_ts_utc", "snap_et_date", "snap_et_hour", "team_abbr",
                    "team_slug", "cbs_player_id", "player_slug", "player_name",
                    "pos", "injury", "expected_return", "status", "updated",
                    "until_date"])
        for f in sorted(src.glob("*.html")):
            ts = f.stem
            u = dt.datetime.strptime(ts, "%Y%m%d%H%M%S").replace(
                tzinfo=dt.timezone.utc)
            et = u.astimezone(ZoneInfo("America/New_York"))
            body = f.read_text(encoding="utf-8", errors="replace")
            n_files += 1
            for tb in TBL.finditer(body):
                blk = tb.group(0)
                tm = TITLE.search(blk)
                if not tm:
                    continue
                abbr, slug = tm.group(1), tm.group(2)
                for rm in ROW.finditer(blk):
                    cells = CELL.findall(rm.group(1))
                    if len(cells) < 5:
                        continue
                    pm = PLAYER.search(cells[1])
                    if not pm:
                        continue
                    exp = txt(cells[4])
                    w.writerow([ts, et.date().isoformat(), et.hour, abbr, slug,
                                pm.group(1), pm.group(2), txt(cells[1]),
                                txt(cells[2]), txt(cells[3]), exp,
                                classify(exp), txt(cells[0]),
                                until_date(exp, SEASON) or ""])
                    n_rows += 1
    print(f"parsed {n_files} snapshots -> {n_rows} rows -> {out}")


SEASON = "2015-16"

if __name__ == "__main__":
    main()
