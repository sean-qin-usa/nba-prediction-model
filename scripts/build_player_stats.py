#!/usr/bin/env python3
"""Build player_game_stats (sufficient statistics) from cached box + PBP.

Usage: python scripts/build_player_stats.py [--limit N]
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nbapred.db import connect
from nbapred.features.possessions import load_corpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    print(load_corpus(connect, limit=args.limit))
