#!/usr/bin/env python3
"""D100 JOB-2b: rebuild the per-player zone-defense ratings CLEAN, and prove the
two defects the D99 audit left standing are gone.

Defect 1 (D81, patched in defense_zone.py 2026-07-31 but never re-run): the
cached playbyplayv3 `game` object has no `homeTeamId`, so `accumulate` charged
100% of shots to the HOME five and 49.9% of those attributions were the wrong
five.
Defect 2 (D99 standing hazard #3, fixed here): `build_zone_defense` scanned the
ENTIRE cache -- preseason + playoffs, and every game ever played, with no date
cutoff -- so any backtest that called it used future information.

Arms:
  BUGGY   home_id = None  (the shipped pre-D81 attribution), all games
  ALL     fixed attribution, no 002 filter, no cutoff   (D81-only fix)
  CLEAN   fixed attribution + only_002=True             (the new default)
  PIT     CLEAN + before=<cutoff>                       (cutoff support proof)

Writes data/zone_defense.json (CLEAN ratings + league baselines + the arm
comparison). Read-only against DuckDB.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                                # noqa: E402
import orjson                                                     # noqa: E402

from nbapred.db import connect                                    # noqa: E402
from nbapred.features import defense_zone as dz                   # noqa: E402
from nbapred.features.cache_index import game_index               # noqa: E402

CUT = "2026-02-01"          # same PIT cut the D99 usage re-runs used
ZONES = ("rim", "mid", "thr")


def build_buggy(limit=None):
    """The shipped pre-D81 behaviour: home_id is always None, so
    `defenders = a5 if shooter_team == home_id else h5` -> always h5."""
    rots = game_index("gamerotation")
    pbps = game_index("playbyplayv3")
    gids = sorted(set(rots) & set(pbps))
    if limit:
        gids = gids[:limit]
    allowed, league = {}, {z: [0, 0] for z in ZONES}
    for gid in gids:
        try:
            rot = orjson.loads(open(rots[gid], "rb").read())["response"]
            pbp = orjson.loads(open(pbps[gid], "rb").read())["response"]
            dz.accumulate(rot, pbp, None, allowed)     # <- the bug, verbatim
        except Exception:
            continue
    for cells in allowed.values():
        for z, (att, made) in cells.items():
            league[z][0] += att
            league[z][1] += made
    lg = {z: (league[z][1] / league[z][0] if league[z][0] else 0.5) for z in league}
    K = {"rim": 60, "mid": 80, "thr": 100}
    out = {}
    for pid, cells in allowed.items():
        r = {}
        for z, (att, made) in cells.items():
            if att < 10:
                r[z] = 0.0
                continue
            r[z] = float((lg[z] - (made + K[z] * lg[z]) / (att + K[z])) * 100)
            r[z + "_att"] = att
        out[pid] = r
    return out, lg


def stats(rat, lg, label):
    att = {z: sum(v.get(z + "_att", 0) for v in rat.values()) for z in ZONES}
    return {"label": label, "players": len(rat), "league_fg": {z: round(lg[z], 5) for z in ZONES},
            "defender_slots": {z: int(att[z]) for z in ZONES},
            "shots": {z: int(att[z] / 5) for z in ZONES}}


def corr(a, b, z):
    common = sorted(set(a) & set(b))
    x = np.array([a[p].get(z, 0.0) for p in common])
    y = np.array([b[p].get(z, 0.0) for p in common])
    m = (x != 0) | (y != 0)
    if m.sum() < 10:
        return None, 0
    return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())


def main():
    t0 = time.time()
    con = connect(read_only=True)
    res = {"cut": CUT}

    print("BUGGY  (pre-D81 attribution, whole cache)...", flush=True)
    r_bug, lg_bug = build_buggy()
    res["buggy"] = stats(r_bug, lg_bug, "BUGGY all-shots-to-home-five, unfiltered")

    print("ALL    (D81-fixed, unfiltered)...", flush=True)
    r_all, lg_all = dz.build_zone_defense(only_002=False, con=con)
    res["all"] = stats(r_all, lg_all, "FIXED, no 002 filter, no cutoff")

    print("CLEAN  (D81-fixed + 002 filter)...", flush=True)
    r_cln, lg_cln = dz.build_zone_defense(only_002=True, con=con)
    res["clean"] = stats(r_cln, lg_cln, "FIXED + only_002 (new default)")

    print(f"PIT    (CLEAN + before={CUT})...", flush=True)
    r_pit, lg_pit = dz.build_zone_defense(only_002=True, before=CUT, con=con)
    res["pit"] = stats(r_pit, lg_pit, f"CLEAN + before={CUT}")
    con.close()

    res["corr_clean_vs_buggy"] = {}
    res["corr_clean_vs_all"] = {}
    res["corr_clean_vs_pit"] = {}
    for z in ZONES:
        c, n = corr(r_cln, r_bug, z); res["corr_clean_vs_buggy"][z] = {"corr": c, "n": n}
        c, n = corr(r_cln, r_all, z); res["corr_clean_vs_all"][z] = {"corr": c, "n": n}
        c, n = corr(r_cln, r_pit, z); res["corr_clean_vs_pit"][z] = {"corr": c, "n": n}

    for k in ("buggy", "all", "clean", "pit"):
        s = res[k]
        print(f"\n{s['label']}")
        print(f"  players {s['players']}  shots {s['shots']}  league FG% {s['league_fg']}")
    print("\nfixed-vs-buggy player-rating correlation (the size of the D81 error):")
    for z in ZONES:
        c = res["corr_clean_vs_buggy"][z]
        print(f"  {z}: corr {c['corr']:.3f}  (n={c['n']})")
    print("002 filter effect (CLEAN vs ALL):")
    for z in ZONES:
        c = res["corr_clean_vs_all"][z]
        print(f"  {z}: corr {c['corr']:.4f}  (n={c['n']})")
    print(f"PIT (before={CUT}) vs full-season CLEAN:")
    for z in ZONES:
        c = res["corr_clean_vs_pit"][z]
        print(f"  {z}: corr {c['corr']:.4f}  (n={c['n']})")

    out = ROOT / "data/zone_defense.json"
    out.write_text(json.dumps(
        {"built": time.strftime("%Y-%m-%dT%H:%M:%S"), "arms": res,
         "league_fg_clean": lg_cln,
         "ratings_clean": {str(k): v for k, v in r_cln.items()}}, indent=1))
    print(f"\nwrote {out}  ({time.time()-t0:.0f}s)\nZONE_REBUILD_DONE")


if __name__ == "__main__":
    main()
