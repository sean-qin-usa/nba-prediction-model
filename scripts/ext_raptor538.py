#!/usr/bin/env python3
"""PROBE: FiveThirtyEight RAPTOR + game forecasts (nba_elo.csv) — free, PIT.

Two assets, both market-free and both dead-but-frozen (538 shut 2023):

1. RAPTOR player-season CSVs — live on GitHub (fivethirtyeight/data,
   nba-raptor/): modern 2014-2023 w/ box + on/off components, historical
   1977-2023. Season-level, retrodictive within season -> usable only as
   PRIOR-SEASON talent input or method benchmark, never as-of mid-season.

2. nba_elo.csv — game-by-game rows 1946-2023 with PREGAME win probabilities
   from three independent systems (elo_prob1, carm-elo_prob1, raptor_prob1)
   plus quality/importance. PIT BY CONSTRUCTION: Elo-family forecasts are
   deterministic walk-forward functions of results through the game date.
   projects.fivethirtyeight.com now 302s to abcnews -> we take the final
   file from the Wayback Machine (snapshot 20250306125344, 3.85 MB, verified
   200). A full independent model's pregame numbers for ~70k games, free.

   LIMIT (rank-relevant): coverage ENDS June 2023 — zero overlap with our
   2023-26 backtest seasons and no live feed. Value = independent
   calibration/benchmark oracle on older eras + prior-season RAPTOR for the
   2023-24 cold start, NOT an ensemble leg going forward.

Raw rule: files cached byte-for-byte to data/raw/ext_538/ before any use.
Probe prints season coverage + forecast-column sanity. No DB writes.
"""
from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nbapred.config import RAW  # noqa: E402

RAW_538 = RAW / "ext_538"
RAW_538.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) "
                    "Gecko/20100101 Firefox/127.0"}

GH = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-raptor"
RAPTOR_FILES = ("modern_RAPTOR_by_player.csv", "modern_RAPTOR_by_team.csv",
                "historical_RAPTOR_by_player.csv")
# final full nba_elo.csv via Wayback (id_ = original bytes, no archive banner)
ELO_WB = ("https://web.archive.org/web/20250306125344id_/"
          "https://projects.fivethirtyeight.com/nba-model/nba_elo.csv")


def fetch(url: str, name: str, force: bool = False) -> Path:
    path = RAW_538 / name
    if path.exists() and not force:
        return path
    resp = requests.get(url, headers=UA, timeout=180)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def summarize_raptor(path: Path) -> None:
    rows = list(csv.DictReader(io.StringIO(path.read_text())))
    seasons = sorted({r["season"] for r in rows})
    print(f"{path.name}: {len(rows)} rows, seasons {seasons[0]}-{seasons[-1]},"
          f" cols={len(rows[0])}")


def summarize_elo(path: Path) -> None:
    rows = list(csv.DictReader(io.StringIO(path.read_text())))
    dates = sorted(r["date"] for r in rows)
    n_raptor = sum(1 for r in rows if r.get("raptor_prob1"))
    n_carm = sum(1 for r in rows if r.get("carm-elo_prob1"))
    print(f"{path.name}: {len(rows)} games {dates[0]}..{dates[-1]} | "
          f"raptor_prob rows={n_raptor}, carm-elo rows={n_carm}")
    last = [r for r in rows if r.get("raptor_prob1")][-1]
    print("  sample:", {k: last[k] for k in
                        ("date", "team1", "team2", "raptor_prob1", "elo_prob1",
                         "score1", "score2") if k in last})


if __name__ == "__main__":
    for f in RAPTOR_FILES:
        summarize_raptor(fetch(f"{GH}/{f}", f))
    summarize_elo(fetch(ELO_WB, "nba_elo_final_wb20250306.csv"))
