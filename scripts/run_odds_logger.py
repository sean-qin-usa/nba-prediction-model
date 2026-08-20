#!/usr/bin/env python3
"""Run the always-on odds logger (foreground; systemd/nohup handles daemonizing)."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nbapred.ingest.odds_logger import run_forever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

if __name__ == "__main__":
    run_forever()
