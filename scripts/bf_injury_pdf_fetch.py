#!/usr/bin/env python3
"""D170: download historical official NBA injury-report PDFs (05PM edition).

Writes nothing to DuckDB — PDFs land in data/raw/injury_reports/ and
scripts/bf_injury_load.py does the one batched write.

ONLY THE 05PM EDITION. A `01PM` edition also exists, but `report_out_map`
(prod_by_season.py, the T2 tier definition) filters on
`status='Out' AND report_date=game_date` with NO edition predicate, so loading
a second edition would silently union two different information sets into the
T2 out-set. That is a tier change, not a backfill, and it is out of scope here.

THROTTLE DISCIPLINE. ak-static.cms.nba.com IP-blocks bursts: a ~250-request
sweep with no spacing produced 403s on dates that had returned 200 seconds
earlier, and the block cleared after ~4 minutes. A 403 is therefore AMBIGUOUS
(no report that day / temporarily blocked). This script never accepts a 403 on
one look: every miss is retried on a later pass, and each pass re-verifies a
KNOWN-GOOD control URL first. A date is only declared genuinely absent after it
403s on two separate passes with a passing control.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import requests  # noqa: E402

RAW_INJ = REPO / "data" / "raw" / "injury_reports"
RAW_INJ.mkdir(parents=True, exist_ok=True)
BASE = "https://ak-static.cms.nba.com/referee/injury/"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) "
                    "Gecko/20100101 Firefox/127.0"}
CONTROL = "Injury-Report_2024-01-15_05PM.pdf"   # known to exist


def fname(d: dt.date) -> str:
    return f"Injury-Report_{d.isoformat()}_05PM.pdf"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--interval", type=float, default=1.15)
    ap.add_argument("--passes", type=int, default=3)
    a = ap.parse_args()

    s = requests.Session()

    def sl():
        time.sleep(a.interval + random.uniform(0, 0.4))

    def get(name):
        try:
            r = s.get(BASE + name, headers=UA, timeout=30)
            return r.status_code, r.content
        except Exception as e:  # noqa: BLE001
            return f"ERR:{type(e).__name__}", b""

    def control_ok():
        for _ in range(6):
            c, _b = get(CONTROL)
            sl()
            if c == 200:
                return True
            print(f"  control -> {c}; cooling 300s", flush=True)
            time.sleep(300)
        return False

    d0 = dt.date.fromisoformat(a.start)
    d1 = dt.date.fromisoformat(a.end)
    todo = []
    d = d0
    while d <= d1:
        if not (RAW_INJ / fname(d)).exists():
            todo.append(d)
        d += dt.timedelta(days=1)
    print(f"range {d0}..{d1}: {len(todo)} dates not already on disk", flush=True)

    manifest = RAW_INJ / "manifest.jsonl"
    got, missing = 0, list(todo)
    for p in range(1, a.passes + 1):
        if not missing:
            break
        print(f"=== PASS {p}: {len(missing)} dates ===", flush=True)
        if not control_ok():
            print("CONTROL NEVER RECOVERED — aborting rather than recording "
                  "false absences", flush=True)
            sys.exit(2)
        still = []
        consec403 = 0
        for i, d in enumerate(missing):
            code, body = get(fname(d))
            sl()
            if code == 200 and body[:4] == b"%PDF":
                out = RAW_INJ / fname(d)
                tmp = out.with_suffix(".tmp")
                tmp.write_bytes(body)
                os.replace(tmp, out)
                with manifest.open("a") as f:
                    f.write(json.dumps({
                        "file": fname(d), "url": BASE + fname(d),
                        "bytes": len(body), "source": "bf_injury_pdf_fetch (D170)",
                        "ingest_ts": dt.datetime.now(dt.timezone.utc).isoformat(),
                    }) + "\n")
                got += 1
                consec403 = 0
            else:
                still.append(d)
                consec403 += 1
                # a run of misses is either the off-season or a block; check
                if consec403 == 15:
                    if not control_ok():
                        print("  blocked mid-pass; aborting pass", flush=True)
                        still.extend(missing[i + 1:])
                        break
                    consec403 = 0
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(missing)} got={got} still_missing={len(still)}",
                      flush=True)
        missing = still
        print(f"PASS {p} end: got={got} still_missing={len(missing)}", flush=True)

    print(f"DOWNLOADED {got} PDFs; {len(missing)} dates absent after "
          f"{a.passes} passes (control verified each pass)", flush=True)
    if missing:
        print("absent (first 40): " + ", ".join(str(x) for x in missing[:40]),
              flush=True)
    print("INJURY_PDF_FETCH_DONE", flush=True)


if __name__ == "__main__":
    main()
