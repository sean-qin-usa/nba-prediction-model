"""SIDE-EFFECT AUDIT: does the D101 corpus fix perturb the F6 October-bridge
pre-registered construction, and if so by how much?

tests/test_october_bridge.py::test_bridge_matches_gate_construction started
failing after scripts/ds_ingest_schedules.py landed. The bridge's trailing-
minutes leg is "last 10 games with seconds>=720 in 002 data strictly before
the cutoff, spanning the prior season", and it reaches player_game_stats
THROUGH a join on nba_games — so rows for 2021-22 / 2020-21 that had always
existed in player_game_stats were previously DROPPED by that join and are now
visible. That is a genuine widening of the bridge's history, not a bug, but it
moves a construction that is registered for a one-shot live test (F6).

This script measures the perturbation on the exact 2025-26 week-1 window the
gate table covers: per-game |cm_ps_new - cm_ps_registered|, how many games
move, and which players changed contribution.

Read-only. Output: data/ds_bridge_impact.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402

from nbapred.config import DB_PATH  # noqa: E402
from nbapred.model.october_bridge import OctoberBridge  # noqa: E402

SEASON = "2025-26"


def mem_db(opener, min_season=None):
    mem = duckdb.connect()
    mem.execute(f"ATTACH '{DB_PATH}' AS src (READ_ONLY)")
    flt = f" AND season >= '{min_season}'" if min_season else ""
    mem.execute(f"CREATE TABLE nba_games AS SELECT * FROM src.nba_games "
                f"WHERE game_date < ?{flt}", [opener])
    mem.execute("CREATE TABLE player_game_stats AS SELECT * FROM "
                "src.player_game_stats WHERE game_id IN "
                "(SELECT game_id FROM nba_games)")
    mem.execute("CREATE TABLE darko_history AS SELECT * FROM "
                "src.darko_history WHERE date < ?", [opener])
    return mem


def main():
    import test_october_bridge as t
    opener = t.OPENER
    sub = t._week1_frame()
    gids = tuple(g.zfill(10) for g in sub.game_id)
    ab2id, played = t._oracle_outs_and_ids(gids)

    arms = {}
    for label, floor in (("registered_corpus", "2022-23"), ("fixed_corpus", None)):
        br = OctoberBridge(mem_db(opener, floor), SEASON, before=opener)
        roster = {}
        for p, (tm, _c) in br.contrib.items():
            roster.setdefault(tm, set()).add(p)
        vals = {}
        for r in sub.itertuples():
            gid = r.game_id.zfill(10)
            hid, aid = ab2id[r.home], ab2id[r.away]
            vals[gid] = br.margin(hid, aid,
                                  roster.get(hid, set()) - played.get((gid, hid), set()),
                                  roster.get(aid, set()) - played.get((gid, aid), set()))
        arms[label] = {"vals": vals, "contrib": dict(br.contrib)}

    reg = {r.game_id.zfill(10): float(r.cm_ps_o) for r in sub.itertuples()}
    d_reg = {g: arms["registered_corpus"]["vals"][g] - reg[g] for g in reg}
    d_new = {g: arms["fixed_corpus"]["vals"][g] - reg[g] for g in reg}

    cr, cn = arms["registered_corpus"]["contrib"], arms["fixed_corpus"]["contrib"]
    moved = {p: (round(cr.get(p, (None, 0.0))[1], 4), round(cn[p][1], 4))
             for p in cn
             if abs(cn[p][1] - cr.get(p, (None, 0.0))[1]) > 1e-9}
    added = sorted(set(cn) - set(cr))
    dropped = sorted(set(cr) - set(cn))

    res = {
        "window": f"{SEASON} week-1 gate games (cm==0)", "n_games": len(reg),
        "registered_corpus_vs_gate_table_max_abs": max(abs(v) for v in d_reg.values()),
        "fixed_corpus_vs_gate_table": {
            "n_games_moved": sum(abs(v) > 1e-9 for v in d_new.values()),
            "max_abs": max(abs(v) for v in d_new.values()),
            "mean_abs": sum(abs(v) for v in d_new.values()) / len(d_new),
            "worst": sorted(((round(v, 5), g) for g, v in d_new.items()),
                            key=lambda x: -abs(x[0]))[:6]},
        "players_with_changed_contribution": len(moved),
        "players_added_to_roster": len(added),
        "players_dropped_from_roster": len(dropped),
        "largest_player_shifts": sorted(
            ((p, a, b) for p, (a, b) in moved.items()),
            key=lambda x: -abs(x[2] - x[1]))[:10],
    }
    print(json.dumps(res, indent=1))
    json.dump(res, open(ROOT / "data" / "ds_bridge_impact.json", "w"), indent=1)


if __name__ == "__main__":
    main()
