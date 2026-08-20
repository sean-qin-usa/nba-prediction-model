#!/usr/bin/env python3
"""D175 pilot: fetch archived CBS Sports NBA injury pages for one old season.

POLITENESS / PERMISSION. web.archive.org publishes no robots.txt (404 =
unrestricted, RFC 9309) and the CDX index is a documented public API. We use an
honest User-Agent, ONE request at a time, >=1.2s apart, and every response is
cached to disk so a re-run costs the archive nothing.

Writes NOTHING to DuckDB. Raw bytes only -> data/raw/unofficial/wayback/<dir>/.
"""
from __future__ import annotations
import argparse, json, sys, time, urllib.parse, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw" / "unofficial" / "wayback"
UA = "nba_model-research/1.0 (personal research; github.com/sean-qin-usa)"
CDX = "http://web.archive.org/cdx/search/cdx"


def get(url: str, tries: int = 4) -> bytes:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read()
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"  retry {i+1}: {e}", flush=True)
            time.sleep(5 * (i + 1))
    raise RuntimeError


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="www.cbssports.com/nba/injuries")
    ap.add_argument("--frm", required=True)          # YYYYMMDD
    ap.add_argument("--to", required=True)
    ap.add_argument("--out", required=True)          # subdir under wayback/
    ap.add_argument("--interval", type=float, default=1.2)
    a = ap.parse_args()

    outdir = RAW / a.out
    outdir.mkdir(parents=True, exist_ok=True)

    q = (f"{CDX}?url={urllib.parse.quote(a.url)}&from={a.frm}&to={a.to}"
         f"&output=json&fl=timestamp,original,statuscode,digest"
         f"&filter=statuscode:200&collapse=digest&limit=5000")
    idx_f = outdir / "_cdx.json"
    if idx_f.exists():
        rows = json.loads(idx_f.read_text())
    else:
        rows = json.loads(get(q).decode())
        idx_f.write_text(json.dumps(rows))
        time.sleep(a.interval)
    body = rows[1:] if rows and rows[0][0] == "timestamp" else rows
    print(f"CDX: {len(body)} distinct-content 200 snapshots for {a.url}", flush=True)

    n_new = 0
    for i, r in enumerate(body):
        ts = r[0]
        f = outdir / f"{ts}.html"
        if f.exists() and f.stat().st_size > 2000:
            continue
        # `id_` = the ORIGINAL archived bytes, no Wayback banner/rewriting
        u = f"https://web.archive.org/web/{ts}id_/http://{a.url}"
        try:
            f.write_bytes(get(u))
            n_new += 1
        except Exception as e:
            print(f"  FAIL {ts}: {e}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(body)} (new={n_new})", flush=True)
        time.sleep(a.interval)
    print(f"done: {n_new} new, {len(list(outdir.glob('*.html')))} cached", flush=True)


if __name__ == "__main__":
    main()
