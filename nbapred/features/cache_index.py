"""Persistent game_id -> cache-file index, so feature builds don't re-open
every raw JSON just to learn which game each file holds.

The nba_api cache names files by a params hash (not game_id), so mapping
file -> game_id requires reading the file. We do that ONCE per file and persist
{filename: game_id} in a sidecar manifest; later runs only open files new since
the last index. Turns the O(all files) scan into O(new files).
"""
from __future__ import annotations

import glob
import os
from concurrent.futures import ThreadPoolExecutor

import orjson

from ..config import RAW_NBA


def _gid_from(bucket: str, raw: dict):
    if bucket == "boxscoretraditionalv3":
        return raw["boxScoreTraditional"]["gameId"]
    if bucket == "playbyplayv3":
        return raw["game"]["gameId"]
    if bucket == "gamerotation":
        return raw["resultSets"][0]["rowSet"][0][0]
    raise ValueError(bucket)


def game_index(bucket: str) -> dict[str, str]:
    """{game_id: filepath} for a cache bucket, using an incremental manifest."""
    bdir = RAW_NBA / bucket
    manifest_path = bdir / "_index.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = orjson.loads(manifest_path.read_bytes())  # {basename: game_id}
        except Exception:  # noqa: BLE001
            manifest = {}

    files = [p for p in glob.glob(str(bdir / "*.json")) if not p.endswith("_index.json")]
    todo = [p for p in files if os.path.basename(p) not in manifest]

    def _read_gid(path):
        try:
            raw = orjson.loads(open(path, "rb").read())["response"]
            return os.path.basename(path), _gid_from(bucket, raw)
        except Exception:  # noqa: BLE001
            return os.path.basename(path), None

    if todo:
        # file reads are I/O-bound -> threads parallelize well (no pickling)
        with ThreadPoolExecutor(max_workers=12) as ex:
            for base, gid in ex.map(_read_gid, todo):
                manifest[base] = gid
        tmp = manifest_path.with_suffix(".tmp")
        tmp.write_bytes(orjson.dumps(manifest))
        os.replace(tmp, manifest_path)

    return {gid: str(bdir / base) for base, gid in manifest.items() if gid}
