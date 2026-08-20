#!/usr/bin/env python3
"""PROBE: dunksandthrees EPM — daily PIT history, free, the DARKO pattern.

Discovery (2026-07-31): the current site serves the FULL EPM table for ANY
historical date via its SvelteKit data endpoint

    https://dunksandthrees.com/epm/__data.json?date=YYYY-MM-DD

anonymously. The `has_access` flag in the payload is 0 for historical dates
but the stat rows are server-rendered regardless (verified: 552/552 non-null
`tot` on 2024-01-15). Seasons go back to 2002; per D&T's methodology page the
history is ~4,700 per-date cumulative decayed-RAPM runs, i.e. each date's
value uses only possessions through that date (data-PIT by construction —
same epistemic status as the accepted D43 DARKO history: PIT-in-data,
current model version). MARKET-FREE: SPM prior + RAPM, no Vegas anywhere.

One fetch per DATE returns every player (~550 rows: off/def/tot EPM plus
projected minutes p_mp_48, usage, shooting profile). A 3-season backtest grid
is ~650 fetches at 1.5 s — one evening, one time.

Raw rule: every response is cached to data/raw/ext_epm/{date}.json before
any parsing/DB use. Probe mode (default) fetches 3 era-spanning dates and
prints a sample; `--date YYYY-MM-DD` fetches one date; no DB writes here.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nbapred.config import RAW  # noqa: E402

RAW_EPM = RAW / "ext_epm"
RAW_EPM.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) "
                    "Gecko/20100101 Firefox/127.0"}
URL = "https://dunksandthrees.com/epm/__data.json"

# fields worth keeping (payload has ~60 per-player fields; these are the core)
KEEP = ("season", "game_dt", "player_id", "player_name", "team_id",
        "team_alias", "age", "off", "def", "tot", "tot_change",
        "p_mp_48", "p_pct_start", "p_usg", "p_tspct")


def _decode(text: str) -> list[dict]:
    """Decode the SvelteKit `devalue` payload: newline-delimited JSON chunks;
    the data node is a flat list where dict values are INDICES into the list."""
    for line in text.splitlines():
        if not line.strip():
            continue
        chunk = json.loads(line)
        for node in chunk.get("nodes") or []:
            if not (node and node.get("type") == "data"):
                continue
            data = node["data"]
            root = data[0]
            if not (isinstance(root, dict) and "stats" in root):
                continue
            rows = []
            for si in data[root["stats"]]:
                ref = data[si]
                rows.append({k: data[v] for k, v in ref.items() if k in KEEP})
            return rows
    return []


def fetch_date(date: str, session: requests.Session | None = None,
               force: bool = False, min_interval: float = 1.5) -> list[dict]:
    """Cache-first fetch of one as-of date. Returns decoded player rows."""
    path = RAW_EPM / f"{date}.json"
    if path.exists() and not force:
        return _decode(path.read_text())
    s = session or requests
    resp = s.get(URL, params={"date": date}, headers=UA, timeout=60)
    resp.raise_for_status()
    path.write_text(resp.text)
    time.sleep(min_interval + random.uniform(0, 0.6))
    return _decode(resp.text)


if __name__ == "__main__":
    dates = sys.argv[2:3] if len(sys.argv) > 2 and sys.argv[1] == "--date" \
        else ["2024-01-15", "2025-11-15", "2003-01-15"]
    sess = requests.Session()
    for d in dates:
        rows = fetch_date(d, sess)
        nn = sum(1 for r in rows if r.get("tot") is not None)
        top = sorted((r for r in rows if r.get("tot") is not None),
                     key=lambda r: -r["tot"])[:3]
        print(f"{d}: {len(rows)} players, {nn} non-null tot | top3:",
              [(r.get("player_name"), round(r["tot"], 2)) for r in top])
