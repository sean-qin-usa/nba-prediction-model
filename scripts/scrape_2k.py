#!/usr/bin/env python3
"""Scrape 2kratings.com current rosters into ratings_2k (HTML archived).

Usage: python scripts/scrape_2k.py [team-slug ...]   (default: all 30 teams)
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nbapred.db import connect
from nbapred.ingest.ratings_2k import scrape_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    con = connect()
    n = scrape_all(con, teams=sys.argv[1:] or None)
    print(f"scraped {n} player ratings")
    con.close()
