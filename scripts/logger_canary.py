#!/usr/bin/env python3
"""D228 — is the odds logger ACTUALLY capturing? Exits non-zero on any FAIL.

`systemctl is-active` is not the question.  The unit stays `active` through
every failure mode that matters: exhausted credits, a revoked key, a wrong sport
key, or a season that started while the poller sat in its offseason idle.  In
all of those the process is healthy, the log fills, and nothing lands on disk.
The prior logger died on 2026-07-27 and sat dead until it was noticed by hand.

So this checks the OUTPUT, not the process.  Run it from cron daily in season;
the one check that cannot be deferred is OPEN CAPTURE, because an opening price
missed on the night is missed permanently.

    python scripts/logger_canary.py [--date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nbapred.config import RAW_ODDS                                # noqa: E402
from nbapred.ingest.odds_sched import ET, UTC                      # noqa: E402

STALE_MIN = 8 * 60          # in season nothing should be quiet this long
MIN_MONTH_CREDITS = 40      # below this the ladder is already open+close only

rows: list[tuple[str, str, str]] = []


def chk(name: str, ok: bool | None, detail: str) -> None:
    rows.append(("PASS" if ok else ("WARN" if ok is None else "FAIL"), name, detail))


def _records(day: dt.date) -> list[dict]:
    p = RAW_ODDS / f"{day:%Y-%m-%d}.jsonl"
    if not p.exists():
        return []
    out = []
    for ln in p.read_text().splitlines():
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    a = ap.parse_args()
    today = dt.date.fromisoformat(a.date) if a.date else dt.datetime.now(UTC).date()
    now = dt.datetime.now(UTC)

    # 1. the unit is up (necessary, nowhere near sufficient)
    try:
        st = subprocess.run(["systemctl", "--user", "is-active",
                             "nba-odds-logger.service"], capture_output=True,
                            text=True, timeout=10).stdout.strip()
    except Exception as e:                                   # noqa: BLE001
        st = f"unknown ({e})"
    chk("service active", st == "active", st)

    recs = _records(today) or _records(today - dt.timedelta(days=1))
    if not recs:
        chk("any capture in 48h", False, "no JSONL records — logger is not writing")
        return _report()

    # 2. freshness — a live process that stopped landing data looks identical
    #    to a healthy one in `systemctl`, so age is measured off the DATA.
    last = max(dt.datetime.fromisoformat(r["snapshot_ts"]) for r in recs)
    age = (now - last).total_seconds() / 60
    chk("capture is fresh", age <= STALE_MIN, f"last snapshot {age:.0f}m ago")

    # 3. credits — the failure the old cadence walked into: burn the month on
    #    evening bursts, then retry-loop for the remainder capturing nothing.
    q = [r["quota"].get("requests_remaining") for r in recs if r.get("quota")]
    rem = int([x for x in q if x][-1]) if any(q) else None
    chk("credits remain", None if rem is None else rem > MIN_MONTH_CREDITS,
        "unknown" if rem is None else f"{rem} left")

    # 4. events listed — a wrong sport key or a missed season start reads
    #    EXACTLY like a healthy offseason in the log, so it needs its own check.
    n_ev = max((len(r["data"]) for r in recs if r.get("kind") == "main"
                and isinstance(r.get("data"), list)), default=0)
    in_season = today.month in (10, 11, 12, 1, 2, 3, 4, 5, 6)
    chk("events listed", None if not in_season else n_ev > 0,
        f"{n_ev} events, {'in' if in_season else 'off'}-season")

    # 5. THE ONE THAT CANNOT BE DEFERRED. An opening price not captured tonight
    #    is not recoverable tomorrow, which is what makes this the whole point.
    kinds = {}
    for r in recs:
        kinds[r.get("target_kind", "?")] = kinds.get(r.get("target_kind", "?"), 0) + 1
    if n_ev:
        chk("open captured", kinds.get("open", 0) > 0,
            f"target_kind counts: {kinds}")
        chk("close captured", None if kinds.get("close", 0) else False,
            f"{kinds.get('close', 0)} close snapshots")
    else:
        chk("open captured", None, "no events to open on")
        chk("close captured", None, "no events to close on")

    # 6. books actually quoting — best-of-N execution is the headline assumption
    bks = set()
    for r in recs:
        for ev in (r.get("data") or []) if isinstance(r.get("data"), list) else []:
            for b in ev.get("bookmakers", []):
                bks.add(b.get("key"))
    chk("multi-book panel", None if not n_ev else len(bks) >= 2,
        f"{len(bks)} distinct books")

    # 7. props — rationed, not disabled (the pre-D228 state logged zero forever)
    chk("props sampled", None if not in_season or not n_ev
        else kinds.get("props", 0) > 0, f"{kinds.get('props', 0)} prop snapshots")
    return _report()


def _report() -> int:
    w = max(len(n) for _, n, _ in rows)
    print(f"{'':4}  {'check':{w}}  detail")
    for s, n, d in rows:
        print(f"{s:4}  {n:{w}}  {d}")
    nf = sum(s == "FAIL" for s, _, _ in rows)
    nw = sum(s == "WARN" for s, _, _ in rows)
    print(f"\n{sum(s == 'PASS' for s, _, _ in rows)} pass / {nw} warn / {nf} fail")
    return 1 if nf else 0


if __name__ == "__main__":
    raise SystemExit(main())
