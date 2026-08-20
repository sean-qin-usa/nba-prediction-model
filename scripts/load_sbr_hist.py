#!/usr/bin/env python3
"""Fetch + load SBR historical NBA odds (free, 2007-08..2022-23).

Usage: python scripts/load_sbr_hist.py [season ...]   e.g. 2022-23
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nbapred.db import connect
from nbapred.ingest.sbr_hist import load_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    print(load_all(connect, seasons=sys.argv[1:] or None))
