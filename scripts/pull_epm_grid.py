#!/usr/bin/env python3
"""D85 EXECUTION step 1 — bulk-pull the EPM backtest grid (risk hedge: pull
EARLY, access could tighten). DAILY grid Oct 1 .. Apr 30 for each backtest
season (2023-24, 2024-25, 2025-26) + today: ~640 dates. This SUPERSETS the
registered minimum (weekly 2023-10..2026-04 + daily Octobers) at the same
~650-fetch budget the inventory doc costed. Cache-first (reruns are free),
polite pacing lives in ext_epm.fetch_date (1.5s + U(0,0.6) jitter after each
network hit). Raw responses under data/raw/ext_epm/{date}.json. NO DB writes
here — the epm_history load is a separate single-writer step.
"""
from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ext_epm import RAW_EPM, fetch_date  # noqa: E402

SEASON_STARTS = (2023, 2024, 2025)


def grid() -> list[str]:
    dates = []
    for y in SEASON_STARTS:
        d = dt.date(y, 10, 1)
        end = dt.date(y + 1, 4, 30)
        while d <= end:
            dates.append(str(d))
            d += dt.timedelta(days=1)
    today = str(dt.date.today())
    if today not in dates:
        dates.append(today)
    return dates


def main() -> None:
    dates = grid()
    cached = {p.stem for p in RAW_EPM.glob("*.json")}
    todo = [d for d in dates if d not in cached]
    print(f"grid={len(dates)} cached={len(dates) - len(todo)} todo={len(todo)}",
          flush=True)
    sess = requests.Session()
    t0 = time.time()
    ok = empty = err = 0
    for i, d in enumerate(todo, 1):
        try:
            rows = fetch_date(d, sess)
            nn = sum(1 for r in rows if r.get("tot") is not None)
            if nn == 0:
                empty += 1
                print(f"[{i}/{len(todo)}] {d}: EMPTY ({len(rows)} rows)",
                      flush=True)
            else:
                ok += 1
                if i % 25 == 0 or i == len(todo):
                    rate = i / max(time.time() - t0, 1e-9)
                    print(f"[{i}/{len(todo)}] {d}: {nn} non-null | "
                          f"{rate:.2f}/s eta {int((len(todo)-i)/max(rate,1e-9))}s",
                          flush=True)
        except Exception as e:                      # noqa: BLE001
            err += 1
            print(f"[{i}/{len(todo)}] {d}: ERROR {e}", flush=True)
            time.sleep(10)                          # back off on failures
    print(f"DONE ok={ok} empty={empty} err={err} "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
