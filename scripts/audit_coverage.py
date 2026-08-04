#!/usr/bin/env python3
"""Data-quality / coverage audit of the nba_api cache (docs/DATA_QUALITY.md).

Poor or missing recording biases a fit if 'not recorded' is silently treated as
'zero'. This reports, per game, whether each artifact exists and whether its key
fields are populated, so gaps are known BEFORE fitting — never discovered as a
mystery in the posteriors. (Same spirit as the BrokerTec missing-L1 diagnostics.)
"""
import glob
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nbapred.config import RAW_NBA


def _load(path):
    try:
        return json.loads(open(path).read())["response"]
    except Exception:
        return None


def main():
    box = {}
    for p in glob.glob(str(RAW_NBA / "boxscoretraditionalv3" / "*.json")):
        r = _load(p)
        if r:
            box[r["boxScoreTraditional"]["gameId"]] = r
    pbp = {}
    for p in glob.glob(str(RAW_NBA / "playbyplayv3" / "*.json")):
        r = _load(p)
        if r:
            pbp[r["game"]["gameId"]] = r
    rot = set()
    for p in glob.glob(str(RAW_NBA / "gamerotation" / "*.json")):
        r = _load(p)
        if r and r["resultSets"][0]["rowSet"]:
            rot.add(r["resultSets"][0]["rowSet"][0][0])

    by_type = Counter(g[:3] for g in box)
    print("=== cache coverage ===")
    print("boxscores:", len(box), "| pbp:", len(pbp), "| rotation:", len(rot))
    print("by game-type prefix (002=reg, 004=playoff, 001=pre, 003/005=allstar):", dict(by_type))

    # field-population checks on the games we have
    missing_pbp = [g for g in box if g not in pbp]
    reg_playoff = [g for g in box if g[:3] in ("002", "004")]
    missing_rot = [g for g in reg_playoff if g not in rot]

    empty_coords = shots = 0
    empty_minutes = box_players = 0
    for gid, r in list(pbp.items())[:500]:
        for a in r["game"]["actions"]:
            if a.get("actionType") in ("Made Shot", "Missed Shot"):
                shots += 1
                if a.get("xLegacy") in (None, "") and a.get("shotDistance") in (None, ""):
                    empty_coords += 1
    for gid, r in list(box.items())[:500]:
        for side in ("homeTeam", "awayTeam"):
            for pl in r["boxScoreTraditional"][side]["players"]:
                box_players += 1
                if not pl["statistics"].get("minutes"):
                    empty_minutes += 1

    print("\n=== gaps ===")
    print(f"boxscores missing PBP: {len(missing_pbp)}")
    print(f"reg/playoff games missing rotation: {len(missing_rot)}"
          f" ({'expected 0 once backfill done' if missing_rot else 'none'})")
    print(f"shot events w/o coords or distance: {empty_coords}/{shots}"
          f" ({100*empty_coords/max(shots,1):.2f}%)")
    print(f"box player-rows w/ empty minutes (DNP/bench, expected nonzero): "
          f"{empty_minutes}/{box_players} ({100*empty_minutes/max(box_players,1):.1f}%)")
    print("\nNote: empty minutes = DNPs (legitimately 0), NOT a recording gap.")


if __name__ == "__main__":
    main()
