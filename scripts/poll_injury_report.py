#!/usr/bin/env python3
"""One injury-report poll (cron this every 15 min in-season)."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nbapred.ingest.injury_report import poll_once

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    print("captured:", poll_once())
