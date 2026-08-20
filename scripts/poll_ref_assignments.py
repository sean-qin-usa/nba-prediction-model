#!/usr/bin/env python3
"""Poll the NBA pregame referee-assignments feed (free; cron on game days)."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nbapred.ingest.referees import poll_assignments

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    print("assignment rows:", poll_assignments(sys.argv[1] if len(sys.argv) > 1 else "2025-26"))
