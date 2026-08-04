#!/usr/bin/env python3
"""Rebuild all derived feature tables from the raw cache (idempotent).
Run after each nba_api backfill/daily pull. Safe to re-run any time.

  player_game_stats  (sufficient statistics; possessions.py)
  lineup_stints      (RAPM stint margins; stints.py)
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nbapred.db import connect
from nbapred.features import possessions, schedule, stints

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    print("player_game_stats:", possessions.load_corpus(connect))
    print("lineup_stints:", stints.load_corpus(connect))
    print("schedule_features:", schedule.build(connect))
