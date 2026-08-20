#!/usr/bin/env python3
"""D85 step 1b — pre-registered EPM Wayback-stability falsification test.

Question (docs/EXTERNAL_MODELS.md, EPM caveat 1): the ?date= endpoint serves
PIT-in-data values under the CURRENT model version. Does dunksandthrees
REWRITE history on model updates? Cheap test: the Wayback Machine archived
the live /epm page ~weekly through 2021-22; those captures embed the full
table AS RENDERED IN 2022 (`var data = {"player_stats": [...]}`). Diff the
archived `tot` values against today's endpoint for the matching as-of date.

As-of alignment: an archive captured at timestamp T shows data through the
last completed slate before T, which we don't know exactly — so for each
archive we score endpoint dates {capture_date - 2 .. capture_date} using the
archived per-player `mp` (minutes are monotone in date and pin the slate
exactly when they match) and report drift on the best-aligned date.

All fetches raw-cached (archives: data/raw/ext_epm/wayback/, endpoint dates:
data/raw/ext_epm/ via ext_epm.fetch_date's cache-first path + pacing).
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ext_epm import RAW_EPM, UA, fetch_date  # noqa: E402

WB_DIR = RAW_EPM / "wayback"
WB_DIR.mkdir(parents=True, exist_ok=True)
CDX = "http://web.archive.org/cdx/search/cdx"
TARGETS = ("20220115", "20220215", "20220315", "20220405")
DATA_RE = re.compile(r"var data = `(\{.*?\})`", re.S)


def cdx_2022() -> list[str]:
    cache = WB_DIR / "cdx_2022.json"
    if cache.exists():
        rows = json.loads(cache.read_text())
    else:
        r = requests.get(CDX, params={
            "url": "dunksandthrees.com/epm", "output": "json",
            "filter": "statuscode:200", "from": "2022", "to": "20220601",
            "limit": "300"}, headers=UA, timeout=90)
        r.raise_for_status()
        rows = r.json()
        cache.write_text(json.dumps(rows))
    return [x[1] for x in rows[1:]]


def fetch_archive(ts: str) -> dict:
    f = WB_DIR / f"epm_{ts}.html"
    if not f.exists():
        url = (f"https://web.archive.org/web/{ts}id_/"
               f"https://dunksandthrees.com/epm")
        r = requests.get(url, headers=UA, timeout=180)
        r.raise_for_status()
        f.write_bytes(r.content)
        time.sleep(2.0)
    m = DATA_RE.search(f.read_text(errors="replace"))
    if not m:
        raise RuntimeError(f"no embedded data block in {f}")
    rows = json.loads(m.group(1))["player_stats"]
    return {int(r["player_id"]): (float(r["tot"]), float(r.get("mp") or 0),
                                  int(r.get("gp") or 0))
            for r in rows if r.get("tot") is not None}


def endpoint_full(date: str) -> tuple[dict, list]:
    """Endpoint rows for `date`: ({named player_id: tot}, [all tot values]).
    LOCK FINDING: masked rows share player_id=4, so a pid-keyed dict
    collapses — identity exists only for the top 5; the full table is
    returned as a VALUE LIST for rank-based comparison."""
    fetch_date(date)                       # ensures raw cache exists (paced)
    text = (RAW_EPM / f"{date}.json").read_text()
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
            named, vals = {}, []
            for si in data[root["stats"]]:
                ref = data[si]
                tot = data[ref["tot"]] if "tot" in ref else None
                if tot is None:
                    continue
                vals.append(float(tot))
                pid = data[ref["player_id"]] if "player_id" in ref else None
                if pid and pid > 100:
                    named[int(pid)] = float(tot)
            return named, vals
    return {}, []


def main() -> None:
    """LOCK CAVEAT (2026-07-31 finding): the endpoint MASKS identity beyond
    the top 5 players on every date (anonymous access), so the pointwise
    identity diff is only possible on those 5; the full-table comparison is
    done BY RANK (both sides sorted by tot desc, top-300 ranks) — valid
    because a history rewrite would move the whole value distribution."""
    snaps = cdx_2022()
    picked = []
    for tgt in TARGETS:
        best = min(snaps, key=lambda s: abs(int(s[:8]) - int(tgt)))
        if best not in picked:
            picked.append(best)
    print("snapshots:", picked)
    report = []
    for ts in picked:
        try:
            arch = fetch_archive(ts)
        except Exception as e:                          # noqa: BLE001
            print(f"{ts}: archive fetch failed ({e}) — skipped this pass")
            continue
        cap = dt.date(int(ts[:4]), int(ts[4:6]), int(ts[6:8]))
        best = None
        for k in (0, 1, 2, 3):
            d = str(cap - dt.timedelta(days=k))
            named_map, vals = endpoint_full(d)
            named = sorted(set(arch) & set(named_map))
            if len(named) < 4:
                continue
            gap = float(np.mean([abs(arch[p][0] - named_map[p])
                                 for p in named]))
            if best is None or gap < best[0]:
                best = (gap, d, named_map, vals, named)
        gap, d, named_map, vals, named = best
        a_rank = np.sort([v[0] for v in arch.values()])[::-1][:300]
        e_rank = np.sort(vals)[::-1][:300]
        n = min(len(a_rank), len(e_rank))
        dv = a_rank[:n] - e_rank[:n]
        r = dict(snapshot=ts, endpoint_date=d,
                 named_top5=[(p, round(arch[p][0] - named_map[p], 4))
                             for p in named],
                 named_mean_abs=round(gap, 4),
                 n_rank=n,
                 rank_corr=round(float(np.corrcoef(a_rank[:n],
                                                   e_rank[:n])[0, 1]), 5),
                 rank_mean_abs=round(float(np.mean(np.abs(dv))), 4),
                 rank_median_abs=round(float(np.median(np.abs(dv))), 4),
                 rank_p95_abs=round(float(np.percentile(np.abs(dv), 95)), 4),
                 rank_max_abs=round(float(np.max(np.abs(dv))), 4),
                 rank_mean_signed=round(float(np.mean(dv)), 4))
        report.append(r)
        print(json.dumps(r))
    out = Path("data/logs/epm_wayback_diff.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print("saved ->", out)


if __name__ == "__main__":
    main()
