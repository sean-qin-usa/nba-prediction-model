#!/usr/bin/env python3
"""D85 step 1a-REVISED — pull ALL in-season Wayback captures of the LIVE
dunksandthrees /epm page for the 3 backtest seasons.

WHY (the lock finding, 2026-07-31): the `/epm/__data.json?date=` endpoint
serves VALUES for any date but MASKS IDENTITY beyond the top 5 players
("Locked Player", player_id 4) for anonymous users — on historical dates AND
today. `has_access:0` is NOT UI-only; the D85 probe's verification counted
non-null `tot` and never checked identity fields. The identity-bearing free
PIT path is the LIVE page (fully named, server-rendered, today) and its
Wayback captures (~weekly 2023-24, ~biweekly 2024-25, sparser 2025-26).

Two capture eras, both parseable:
  A (2023-10 .. ~2024-11): full table embedded as positional JS arrays in the
    __sveltekit_* script: [..., team_id, alias, alias, player_id, "Name", ...
    ..., off, def, tot(=off+def), ...]. No embedded as-of date -> effective
    date derived from the capture timestamp (UTC-5, minus 1 day: the table
    holds games through the previous completed slate).
  B (2024-12 ..): unquoted JS literal `{date:"YYYY-MM-DD",stats:[{player_id:
    ...,player_name:"...",off:...,def:...,tot:...}]}` with the true as-of
    game date embedded — used directly.

Raw HTML cached under data/raw/ext_epm/wayback/epm_{ts}.html; parsed rows to
data/raw/ext_epm/wayback/parsed_{ts}.json. NO DB writes here.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ext_epm import RAW_EPM, UA  # noqa: E402

WB_DIR = RAW_EPM / "wayback"
WB_DIR.mkdir(parents=True, exist_ok=True)
CDX = "http://web.archive.org/cdx/search/cdx"

ERA_B_RE = re.compile(
    r'\{season:(\d+),game_dt:"([\d-]+)",player_id:(\d+),'
    r'player_name:"([^"]*)",team_id:(\d+),team_alias:"([^"]*)"'
    r'(.*?)(?=\},?\{season:|\]\})', re.S)
ERA_B_DATE = re.compile(r'data:\{date:"([\d-]+)",stats:\[\{season:')
FIELD = {k: re.compile(k + r":(-?[\d.]+|null)") for k in ("off", "def", "tot")}
# era A row: ...,team_id,"AAA","AAA",player_id,"Name",...
ERA_A_ROW = re.compile(
    r'\[(\d{4}),\d+,\d+,\d+,(\d{10}),"([A-Z]{2,3})","[A-Z]{2,3}",(\d+),'
    r'"([^"]+)",(.*?)\](?=,\[|\]\})', re.S)


def in_season(day: str) -> bool:
    m = int(day[4:6])
    return m >= 10 or m <= 4


def capture_list() -> list[str]:
    cache = WB_DIR / "cdx_2023_2026.json"
    if not cache.exists():
        r = requests.get(CDX, params={
            "url": "dunksandthrees.com/epm", "output": "json",
            "filter": "statuscode:200", "from": "20231001", "to": "20260430",
            "collapse": "timestamp:8", "limit": "2000"},
            headers=UA, timeout=90)
        r.raise_for_status()
        cache.write_text(r.text)
    rows = json.loads(cache.read_text())
    return [x[1] for x in rows[1:] if in_season(x[1][:8])]


def fetch(ts: str) -> str | None:
    f = WB_DIR / f"epm_{ts}.html"
    if f.exists():
        return f.read_text(errors="replace")
    for attempt in range(3):
        try:
            r = requests.get(
                f"https://web.archive.org/web/{ts}id_/"
                f"https://dunksandthrees.com/epm",
                headers=UA, timeout=180)
            if r.status_code == 200:
                f.write_bytes(r.content)
                time.sleep(2.0 + 0.5 * attempt)
                return f.read_text(errors="replace")
            print(f"  {ts}: HTTP {r.status_code}", flush=True)
            time.sleep(8)
        except Exception as e:                          # noqa: BLE001
            print(f"  {ts}: {e}", flush=True)
            time.sleep(10)
    return None


def effective_date(ts: str) -> str:
    """Era A: table holds games through the previous completed ET slate."""
    utc = dt.datetime.strptime(ts, "%Y%m%d%H%M%S")
    et = utc - dt.timedelta(hours=5)
    return str(et.date() - dt.timedelta(days=1))


_NUM = re.compile(r'^-?(?:\d+\.\d+|\.\d+)$')     # decimals only (JS may omit
                                                 # the leading zero: ".69")
_OBJ = re.compile(r'\{[^{}]*\}')


def _era_a_rows(html: str) -> list[dict]:
    # restrict to the FIRST stats:[[...]] block — later arrays on the page
    # (career/leader lists) produce junk rows with accidental value triples
    i = html.find("stats:[[")
    if i >= 0:
        j = html.find("]]", i)
        if j > i:
            html = html[i:j + 2]
    rows = []
    for r in ERA_A_ROW.finditer(html):
        # structure: ..., mpg, {z-obj}, off, def, tot(=off+def), ewins, ...
        # -> the three decimals right after the FIRST stripped object; rows
        # with null EPM (no z-objects / null slots) are skipped by design
        toks = [t.strip() for t in _OBJ.sub("#", r.group(6)).split(",")]
        try:
            k = toks.index("#")
        except ValueError:
            continue
        nxt = toks[k + 1:k + 4]
        if len(nxt) < 3 or not all(_NUM.match(t) for t in nxt):
            continue
        trip = tuple(float(t) for t in nxt)
        if abs(trip[0] + trip[1] - trip[2]) > 5e-3 or trip[2] == 0:
            continue
        rows.append(dict(player_id=int(r.group(4)), player_name=r.group(5),
                         team_id=int(r.group(2)), team_alias=r.group(3),
                         off=trip[0], def_=trip[1], tot=trip[2]))
    return rows


def parse(ts: str, html: str) -> dict | None:
    rows, dates = [], {}
    for r in ERA_B_RE.finditer(html):        # era B: self-describing rows
        body = r.group(7)
        vals = {}
        for k, rx in FIELD.items():
            v = rx.search(body)
            vals[k] = None if (not v or v.group(1) == "null") \
                else float(v.group(1))
        if vals["tot"] is None:
            continue
        dates[r.group(2)] = dates.get(r.group(2), 0) + 1
        rows.append(dict(player_id=int(r.group(3)), player_name=r.group(4),
                         team_id=int(r.group(5)), team_alias=r.group(6),
                         off=vals["off"], def_=vals["def"], tot=vals["tot"]))
    if len(rows) >= 100:
        asof, era = max(dates, key=dates.get), "B"
    else:
        rows = _era_a_rows(html)
        era = "A"
        # era-A pages embed the true as-of date as the dominant YYYY-MM-DD
        # string (immune to Wayback serving a nearby-but-later capture, which
        # would make the capture-timestamp rule UNDERSTATE the as-of — a
        # lookahead hazard). Fallback: capture-date-1 (ET).
        cnt: dict[str, int] = {}
        for m2 in re.finditer(r"20\d\d-\d\d-\d\d", html):
            cnt[m2.group(0)] = cnt.get(m2.group(0), 0) + 1
        asof = effective_date(ts)
        if cnt:
            modal = max(cnt, key=cnt.get)
            cap = dt.date(int(ts[:4]), int(ts[4:6]), int(ts[6:8]))
            if abs((dt.date.fromisoformat(modal) - cap).days) <= 30:
                asof = modal
    if len(rows) < 100:
        return None
    # dedupe on player_id keeping first occurrence (page may repeat blocks)
    seen, uniq = set(), []
    for r in rows:
        if r["player_id"] in seen:
            continue
        seen.add(r["player_id"])
        uniq.append(r)
    return dict(ts=ts, era=era, asof=asof, n=len(uniq), rows=uniq)


def main() -> None:
    caps = capture_list()
    print(f"in-season captures: {len(caps)}", flush=True)
    ok = bad = 0
    for i, ts in enumerate(caps, 1):
        out = WB_DIR / f"parsed_{ts}.json"
        if out.exists():
            ok += 1
            continue
        html = fetch(ts)
        if html is None:
            bad += 1
            continue
        p = parse(ts, html)
        if p is None:
            print(f"[{i}/{len(caps)}] {ts}: PARSE FAIL", flush=True)
            bad += 1
            continue
        out.write_text(json.dumps(p))
        ok += 1
        if i % 10 == 0 or i == len(caps):
            print(f"[{i}/{len(caps)}] {ts}: era {p['era']} asof {p['asof']} "
                  f"n={p['n']}", flush=True)
    print(f"DONE ok={ok} bad={bad}", flush=True)


if __name__ == "__main__":
    main()
