#!/usr/bin/env python3
"""Build the NBA <-> 2K <-> odds player crosswalk and report match quality."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nbapred.db import connect
from nbapred.ids import build_crosswalk, unmatched_2k

if __name__ == "__main__":
    con = connect()
    stats = build_crosswalk(con)
    print("crosswalk:", stats)
    orphans = unmatched_2k(con)
    if orphans:
        print(f"\n2K names with no NBA match ({len(orphans)}):")
        for name in orphans:
            print("  ", name)
    con.close()
