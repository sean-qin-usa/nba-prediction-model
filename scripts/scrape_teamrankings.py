#!/usr/bin/env python
"""Scrape TeamRankings NBA point-spread movement pages -> line-delimited JSON.

Source page: https://www.teamrankings.com/nba/matchup/<slug>/spread-movement
(NOTE: /point-spread-movement is a valid-looking URL that silently returns a page
with no data. Always use /spread-movement.)

Page layout (verified 2026-05-17 CLE/DET and ~a dozen others 2011-12..2025-26):
  * two <table class="tr-table movement-table"> summary tables, favourite first:
        ['DET -4.5', 'Open', '-3.5', 'High', '-4.5', 'Last', '-3.5', 'Low', '-3.5']
    the mirrored underdog table follows.
  * one <table class="tr-table scrollable"> history table:
        header  : ['', 'Book 1', 'Book 2', 'Book 3']
        Current : one spread per book ('--' when the book has no quote)
        Open    : ditto
        History : separator row
        then rows keyed 'MM/DD HH:MM AM/PM' -> one spread per book.
    Timestamps carry no year; it is inferred from the game date (a stamp whose
    month is later than the game's month belongs to the previous calendar year).
  * pages with no data instead render a single table containing
    'The spread is not yet available.'

Output: one JSON object per game appended to --out. Resumable: game_ids already
present in the output file are skipped, so an interrupted run is cheap to resume.

Usage:
    python scripts/scrape_teamrankings.py --seasons 2024,2025,2026
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36"
)

DEFAULT_GAMES = (
    "/tmp/claude-1004/-hdd-steveqin-sean-dev/"
    "4912ce58-69cd-488d-a5f4-0d3ed2eed30c/scratchpad/tr_games.csv"
)
DEFAULT_OUT = (
    "/hdd/steveqin/sean_dev/nba_model/data/raw/teamrankings/spread_movement.jsonl"
)

TS_RE = re.compile(r"^(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2})\s*(AM|PM)$", re.I)
NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


# ---------------------------------------------------------------- parsing ---
def _num(txt):
    """'-4.5' -> -4.5 ; '+3.5' -> 3.5 ; '--'/'PK'/'' -> None."""
    if txt is None:
        return None
    t = txt.strip().replace("−", "-").replace("+", "")
    if not t or t in {"--", "-", "N/A", "NL"}:
        return None
    if t.upper() in {"PK", "PICK", "EV"}:
        return 0.0
    if NUM_RE.match(t):
        return float(t)
    return None


def _cells(row):
    return [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]


def _parse_summary(table):
    """['DET -4.5','Open','-3.5','High','-4.5','Last','-3.5','Low','-3.5'] -> dict."""
    flat = []
    for r in table.find_all("tr"):
        flat.extend(_cells(r))
    if not flat:
        return None
    head = flat[0].split()
    out = {"team": head[0] if head else None,
           "line": _num(head[-1]) if len(head) > 1 else None}
    for i, cell in enumerate(flat):
        key = cell.strip().lower()
        if key in {"open", "high", "last", "low"} and i + 1 < len(flat):
            out[key] = _num(flat[i + 1])
    for k in ("open", "high", "last", "low"):
        out.setdefault(k, None)
    return out


def _infer_year(month, game_date):
    """Timestamps have no year; a month later than the game's is the prior year."""
    gy, gm = game_date.year, game_date.month
    return gy - 1 if month > gm else gy


def parse_page(html, game_date):
    """Return dict of parsed fields (no_data=True when the page carries no lines)."""
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")

    summaries, hist_table = [], None
    for t in tables:
        cls = t.get("class") or []
        if "movement-table" in cls:
            summaries.append(t)
        elif "Book 1" in t.get_text():
            hist_table = t
    if hist_table is None:
        for t in tables:
            if "Book 1" in t.get_text():
                hist_table = t
                break

    rec = {
        "fav_team": None, "fav_line": None, "fav_open": None, "fav_last": None,
        "fav_high": None, "fav_low": None,
        "dog_team": None, "dog_line": None, "dog_open": None, "dog_last": None,
        "dog_high": None, "dog_low": None,
        "current_books": None, "open_books": None,
        "history": [], "n_history": 0, "no_data": True,
    }

    parsed = [p for p in (_parse_summary(t) for t in summaries[:2]) if p]
    # Tables render favourite-first; fall back to the sign of the current line.
    if len(parsed) == 2:
        fav, dog = parsed
        if fav.get("line") is not None and dog.get("line") is not None \
                and fav["line"] > 0 and dog["line"] < 0:
            fav, dog = dog, fav
        for pre, side in (("fav", fav), ("dog", dog)):
            rec[f"{pre}_team"] = side["team"]
            for k in ("line", "open", "high", "last", "low"):
                rec[f"{pre}_{k}"] = side[k]

    if hist_table is not None:
        rows = hist_table.find_all("tr")
        for r in rows:
            c = _cells(r)
            if not c:
                continue
            label = c[0].strip().lower()
            vals = [_num(x) for x in c[1:4]]
            vals += [None] * (3 - len(vals))
            if label == "current":
                rec["current_books"] = vals
            elif label == "open":
                rec["open_books"] = vals
            else:
                m = TS_RE.match(c[0].strip())
                if not m:
                    continue
                mo, day, hh, mm, ap = int(m.group(1)), int(m.group(2)), \
                    int(m.group(3)), int(m.group(4)), m.group(5).upper()
                hh = 0 if (hh == 12 and ap == "AM") else \
                    (hh + 12 if (ap == "PM" and hh != 12) else hh)
                yr = _infer_year(mo, game_date)
                try:
                    iso = datetime(yr, mo, day, hh, mm).isoformat()
                except ValueError:
                    iso = None
                rec["history"].append({
                    "ts": c[0].strip(), "ts_iso": iso,
                    "book1": vals[0], "book2": vals[1], "book3": vals[2],
                })

    rec["n_history"] = len(rec["history"])
    rec["no_data"] = not (rec["fav_open"] is not None or rec["n_history"] > 0
                          or any(v is not None for v in (rec["open_books"] or [])))
    return rec


# ---------------------------------------------------------------- fetching ---
class Throttle:
    """Global minimum gap between request starts (on top of per-worker sleeps)."""

    def __init__(self, min_gap):
        self.min_gap = min_gap
        self.lock = threading.Lock()
        self.last = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            gap = self.min_gap - (now - self.last)
            if gap > 0:
                time.sleep(gap)
            self.last = time.time()


def fetch(session, url, throttle, retries=3, timeout=25):
    last_status, last_err = None, None
    for attempt in range(retries):
        throttle.wait()
        try:
            r = session.get(url, timeout=timeout)
            last_status = r.status_code
            if r.status_code == 200:
                return 200, r.text, None
            if r.status_code in (404, 410):
                return r.status_code, None, f"http {r.status_code}"
            last_err = f"http {r.status_code}"
        except Exception as exc:  # noqa: BLE001 - network layer, keep going
            last_err = f"{type(exc).__name__}: {exc}"
        time.sleep((2 ** attempt) * 1.5 + random.random())
    return last_status, None, last_err


# ------------------------------------------------------------------- main ---
def load_games(path, seasons):
    import csv
    out = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row["season"]) in seasons:
                url = row["source_url"].replace("/money-line-movement",
                                                "/spread-movement")
                out.append({
                    "game_id": row["game_id"],
                    "game_date": row["gd"],
                    "season_end": int(row["season"]),
                    "url": url,
                })
    out.sort(key=lambda r: (r["game_date"], r["game_id"]))
    return out


def load_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(str(json.loads(line)["game_id"]))
            except Exception:  # noqa: BLE001 - tolerate a torn final line
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=DEFAULT_GAMES)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seasons", default="2024,2025,2026")
    ap.add_argument("--workers", type=int, default=2, help="max 2 (politeness)")
    ap.add_argument("--min-gap", type=float, default=0.45,
                    help="global min seconds between request starts")
    ap.add_argument("--sleep-lo", type=float, default=0.7)
    ap.add_argument("--sleep-hi", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    workers = min(args.workers, 2)
    seasons = {int(s) for s in args.seasons.split(",") if s.strip()}
    games = load_games(args.games, seasons)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = load_done(args.out)
    todo = [g for g in games if g["game_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(f"[tr] seasons={sorted(seasons)} total={len(games)} "
          f"already_done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return

    throttle = Throttle(args.min_gap)
    write_lock = threading.Lock()
    counter = {"n": 0, "ok": 0, "nodata": 0, "err": 0}
    t0 = time.time()
    out_fh = open(args.out, "a", buffering=1)

    local = threading.local()

    def session():
        if not hasattr(local, "s"):
            local.s = requests.Session()
            local.s.headers.update({"User-Agent": UA,
                                    "Accept-Language": "en-US,en;q=0.9"})
        return local.s

    def work(g):
        status, html, err = fetch(session(), g["url"], throttle)
        rec = dict(g)
        rec["http_status"] = status
        rec["fetched_ts"] = datetime.now(timezone.utc).isoformat()
        if html is None:
            rec.update({"no_data": True, "error": err, "history": [],
                        "n_history": 0})
        else:
            gd = datetime.strptime(g["game_date"], "%Y-%m-%d")
            try:
                rec.update(parse_page(html, gd))
            except Exception as exc:  # noqa: BLE001 - never lose a row to a parse bug
                rec.update({"no_data": True, "error": f"parse: {exc}",
                            "history": [], "n_history": 0})
        with write_lock:
            out_fh.write(json.dumps(rec) + "\n")
            counter["n"] += 1
            if rec.get("error"):
                counter["err"] += 1
            elif rec.get("no_data"):
                counter["nodata"] += 1
            else:
                counter["ok"] += 1
            if counter["n"] % 100 == 0:
                el = time.time() - t0
                rate = counter["n"] / el
                eta = (len(todo) - counter["n"]) / rate / 60 if rate else 0
                print(f"[tr] {counter['n']}/{len(todo)} ok={counter['ok']} "
                      f"nodata={counter['nodata']} err={counter['err']} "
                      f"{rate:.2f}/s eta={eta:.1f}m", flush=True)
        time.sleep(random.uniform(args.sleep_lo, args.sleep_hi))

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, todo))
    finally:
        out_fh.close()

    print(f"[tr] DONE {counter} in {(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    sys.exit(main())
