#!/usr/bin/env python3
"""One ESPN scoreboard/odds poll (cron every 30 min in-season, free)."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nbapred.ingest.espn_lines import poll_once

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    print("events with odds:", poll_once(sys.argv[1] if len(sys.argv) > 1 else None))
