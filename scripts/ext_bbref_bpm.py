#!/usr/bin/env python3
"""PROBE: Basketball-Reference BPM/VORP — free scrape now, Wayback for PIT.

What it is: box-only Box Plus/Minus 2.0 (+ VORP) on the league "advanced"
page — full history back to 1974, market-free, methodology fully published.

Two access paths, both demonstrated here:
1. CURRENT: https://www.basketball-reference.com/leagues/NBA_{yyyy}_advanced.html
   fetches fine with a browser UA (verified 200/2.2 MB). Sports-Reference
   rate limit: <=20 req/min; we do ONE request and cache the HTML.
2. PIT: the same page is in the Wayback Machine at weekly-or-better density
   in-season (measured: 72 distinct days in 2023-24, 30 in 2024-25, 27 in
   2025-26). This probe asks the CDX API for the snapshot closest to a
   target date and parses BPM out of the archived bytes (id_ URL).

BUILD-VS-TAKE (recommendation: BUILD for the ensemble, TAKE for validation):
BPM 2.0 is a deterministic published formula on box-score rates + team
adjustment, and our DuckDB already holds every box score — computing BPM
as-of any date in-house gives DAILY exact PIT with zero fetches. The scrape
paths here are for validating that in-house implementation (and VORP), not
for feeding the backtest at weekly granularity.

Raw rule: every page cached byte-for-byte to data/raw/ext_bbref/ before
parsing. No DB writes in probe mode.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nbapred.config import RAW  # noqa: E402

RAW_BBR = RAW / "ext_bbref"
RAW_BBR.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) "
                    "Gecko/20100101 Firefox/127.0"}

PAGE = "https://www.basketball-reference.com/leagues/NBA_{yyyy}_advanced.html"
CDX = "http://web.archive.org/cdx/search/cdx"

_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_CELL = re.compile(r'data-stat="([a-z0-9_]+)"[^>]*>(?:<[^>]+>)*([^<]*)')


def parse_advanced(html: str) -> list[dict]:
    """Pull player/team/mp/bpm/vorp rows; tolerant of comment-wrapped tables."""
    html = html.replace("<!--", "").replace("-->", "")
    out = []
    for m in _ROW.finditer(html):
        cells = dict(_CELL.findall(m.group(1)))
        name = cells.get("player") or cells.get("name_display")
        if name and (cells.get("bpm") or "").strip() not in ("", "-"):
            try:
                out.append({"player": name,
                            "team": cells.get("team_id") or
                                    cells.get("team_name_abbr"),
                            "mp": float(cells.get("mp") or 0),
                            "bpm": float(cells["bpm"]),
                            "vorp": float(cells.get("vorp") or 0)})
            except ValueError:
                pass
    return out


def fetch_current(yyyy: int, force: bool = False) -> list[dict]:
    path = RAW_BBR / f"NBA_{yyyy}_advanced.html"
    if not path.exists() or force:
        resp = requests.get(PAGE.format(yyyy=yyyy), headers=UA, timeout=60)
        resp.raise_for_status()
        path.write_bytes(resp.content)
        time.sleep(3.5)  # sports-reference courtesy (<20 req/min hard limit)
    return parse_advanced(path.read_text(errors="replace"))


def fetch_pit(yyyy: int, asof: str, force: bool = False) -> tuple[str, list[dict]]:
    """Closest Wayback snapshot to `asof` (YYYYMMDD) of that season's page."""
    tag = RAW_BBR / f"NBA_{yyyy}_advanced_wb_{asof}.html"
    meta = tag.with_suffix(".meta.json")
    if tag.exists() and meta.exists() and not force:
        return json.loads(meta.read_text())["timestamp"], \
            parse_advanced(tag.read_text(errors="replace"))
    q = {"url": f"basketball-reference.com/leagues/NBA_{yyyy}_advanced.html",
         "output": "json", "filter": "statuscode:200", "limit": "1",
         "sort": "closest", "closest": asof}
    rows = requests.get(CDX, params=q, headers=UA, timeout=60).json()
    ts = rows[1][1]
    url = (f"https://web.archive.org/web/{ts}id_/"
           f"https://www.basketball-reference.com/leagues/"
           f"NBA_{yyyy}_advanced.html")
    resp = requests.get(url, headers=UA, timeout=120)
    resp.raise_for_status()
    tag.write_bytes(resp.content)
    meta.write_text(json.dumps({"timestamp": ts, "url": url}))
    return ts, parse_advanced(tag.read_text(errors="replace"))


if __name__ == "__main__":
    cur = fetch_current(2026)
    top = sorted(cur, key=lambda r: -r["vorp"])[:3]
    print(f"current 2025-26: {len(cur)} player rows | top VORP:",
          [(r["player"], r["bpm"], r["vorp"]) for r in top])
    ts, pit = fetch_pit(2024, "20240115")
    tp = sorted(pit, key=lambda r: -r["vorp"])[:3]
    print(f"PIT snapshot {ts}: {len(pit)} rows | top VORP:",
          [(r["player"], r["bpm"], r["vorp"]) for r in tp])
