#!/usr/bin/env python3
"""Snapshot DARKO player-talent priors into darko_dpm (daily, free)."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nbapred.db import connect
from nbapred.ingest.darko import load

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    print("darko players:", load(connect))
