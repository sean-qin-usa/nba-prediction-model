"""Official NBA injury report capture (PIT-critical, non-reconstructible).

The league publishes PDFs at
  https://ak-static.cms.nba.com/referee/injury/Injury-Report_{YYYY-MM-DD}_{HH}{AM|PM}.pdf
periodically on game days (ET timestamps). Off-season: nothing exists (the CDN
403s on missing keys). This poller probes plausible stamps for today/yesterday
(ET), archives any PDF it hasn't seen, and records a capture manifest line.

Parsing PDF -> injury_reports rows is deferred until we hold a real 2026-27
sample (layout changes season to season); capture must not wait on parsing.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
from zoneinfo import ZoneInfo

import requests

from ..config import RAW

log = logging.getLogger("injury_report")

RAW_INJ = RAW / "injury_reports"
RAW_INJ.mkdir(parents=True, exist_ok=True)

URL_TMPL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date}_{stamp}.pdf"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0"}
ET = ZoneInfo("America/New_York")
# Stamps observed across seasons; probing a miss costs one cheap 403.
STAMPS = [f"{h:02d}AM" for h in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)] + ["12PM"] + \
         [f"{h:02d}PM" for h in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)] + ["12AM"]


def poll_once() -> int:
    """Probe today + yesterday (ET); archive new PDFs. Returns #captured."""
    captured = 0
    now_et = dt.datetime.now(ET)
    manifest = RAW_INJ / "manifest.jsonl"
    for day in (now_et.date(), now_et.date() - dt.timedelta(days=1)):
        for stamp in STAMPS:
            fname = f"Injury-Report_{day.isoformat()}_{stamp}.pdf"
            out = RAW_INJ / fname
            if out.exists():
                continue
            url = URL_TMPL.format(date=day.isoformat(), stamp=stamp)
            try:
                r = requests.get(url, headers=UA, timeout=20)
            except requests.RequestException as e:
                log.warning("probe failed %s: %s", url, e)
                continue
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                tmp = out.with_suffix(".tmp")
                tmp.write_bytes(r.content)
                os.replace(tmp, out)
                with manifest.open("a") as f:
                    f.write(json.dumps({
                        "file": fname, "url": url, "bytes": len(r.content),
                        "ingest_ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                    }) + "\n")
                captured += 1
                log.info("captured %s (%d bytes)", fname, len(r.content))
    return captured
