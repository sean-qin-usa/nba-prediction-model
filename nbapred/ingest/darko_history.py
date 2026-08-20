"""Point-in-time DARKO history from darko.app player pages.

Sean's Wayback idea, upgraded: the current site server-renders each player's
FULL daily DPM time series (back to 2017-10-17) into /player/{nba_id}. One
fetch per player yields every historical as-of date — solving the
DARKO-staleness handicap on historical backtests (capstone: gap +0.0425 in
2023-24 vs +0.0164 in 2025-26 with a current-day snapshot) and giving
leakage-free PIT talent priors + DARKO's own minutes projection (x_minutes).

Raw rule: extracted per-player series JSON is cached to data/raw/darko_history/
{player_id}.json before any DB write. Rate limit ~1.5s/page (one-time bulk).
"""
from __future__ import annotations

import json
import logging
import random
import re
import time

import requests

from ..config import RAW

log = logging.getLogger("darko_history")

RAW_DH = RAW / "darko_history"
RAW_DH.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS darko_history (
    player_id BIGINT NOT NULL,
    date      DATE   NOT NULL,
    dpm DOUBLE, o_dpm DOUBLE, d_dpm DOUBLE,
    box_dpm DOUBLE, box_odpm DOUBLE, box_ddpm DOUBLE,
    x_minutes DOUBLE, age DOUBLE,
    PRIMARY KEY (player_id, date)
);
"""

_REC = re.compile(r'date:"(\d{4}-\d{2}-\d{2})",([^{}]*)')
_KV = re.compile(r'([a-z_]+):(-?\.?\d[\d.eE+-]*|null)')

FIELDS = ("dpm", "o_dpm", "d_dpm", "box_dpm", "box_odpm", "box_ddpm",
          "x_minutes", "age")


def parse_series(html: str) -> list[dict]:
    out = []
    for m in _REC.finditer(html):
        rec = {"date": m.group(1)}
        for k, v in _KV.findall(m.group(2)):
            if k in FIELDS:
                rec[k] = None if v == "null" else float(v)
        if len(rec) > 1:
            out.append(rec)
    # de-dup dates (page may render the series in more than one component)
    seen: dict[str, dict] = {}
    for r in out:
        seen[r["date"]] = r
    return sorted(seen.values(), key=lambda r: r["date"])


def fetch_player(player_id: int, session: requests.Session | None = None,
                 force: bool = False) -> list[dict]:
    path = RAW_DH / f"{player_id}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())["series"]
    s = session or requests
    resp = s.get(f"https://www.darko.app/player/{player_id}",
                 headers=UA, timeout=60)
    series = parse_series(resp.text) if resp.status_code == 200 else []
    path.write_text(json.dumps({"player_id": player_id,
                                "status": resp.status_code,
                                "series": series}))
    return series


def load_players(connect_fn, player_ids: list[int],
                 min_interval: float = 1.5) -> dict:
    """Fetch-then-load. NETWORK BEFORE LOCK: all pages are fetched to the raw
    cache with no DB connection open; the DB write is one short bulk insert."""
    import pandas as pd
    sess = requests.Session()
    rows, ok, empty = [], 0, 0
    for i, pid in enumerate(player_ids):
        cached = (RAW_DH / f"{pid}.json").exists()
        try:
            series = fetch_player(pid, sess)
        except Exception:
            log.exception("player %s failed; continuing", pid)
            continue
        if series:
            ok += 1
            for r in series:
                rows.append([pid, r["date"]] + [r.get(f) for f in FIELDS])
        else:
            empty += 1
        if not cached:
            time.sleep(min_interval + random.uniform(0, 0.6))
        if (i + 1) % 25 == 0:
            log.info("progress %d/%d (ok %d, empty %d, rows %d)",
                     i + 1, len(player_ids), ok, empty, len(rows))
    if rows:
        df = pd.DataFrame(rows, columns=["player_id", "date", *FIELDS])
        con = connect_fn()
        con.execute(SCHEMA)
        con.execute("INSERT OR REPLACE INTO darko_history SELECT * FROM df")
        con.close()
    return {"players": ok, "empty": empty, "rows": len(rows)}
