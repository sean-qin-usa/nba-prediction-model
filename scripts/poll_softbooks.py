#!/usr/bin/env python3
"""One soft-book prop poll (cron 30-60 min in-season)."""
import logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nbapred.ingest.softbook_props import poll_once
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
if __name__ == "__main__":
    print(poll_once())
