"""STEP-1 QUANTIFIER: how starved was each trailing consumer, per eval season?

Measures, at each weekly refit date of every eval season, the corpus each
cross-season consumer actually saw under the PRE-FIX nba_games (season >=
'2022-23') vs the POST-FIX corpus:

  * fit_schedule_layer  — n games in the 730d window and the resulting
    n/(n+600) shrink weight (1.0 = fully data-driven, 0.0 = pure SCHED_PRIOR)
  * D62 carry           — whether continuity_map() returns a map at all
  * tanking / latestate — rows available under their season >= '2022-23' floor

Read-only. Output: data/ds_starvation_diag.json
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ds_corpus import arm_connection  # noqa: E402
from nbapred.model.production import SCHED_SHRINK, continuity_map  # noqa: E402

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def probe(con, season):
    dates = [r[0] for r in con.execute(
        """SELECT DISTINCT game_date FROM nba_games WHERE season=?
           AND game_id LIKE '002%' ORDER BY 1""", [season]).fetchall()]
    if not dates:
        return None
    refits, last = [], None
    for d in dates:
        if last is None or (d - last).days >= 7:
            refits.append(d)
            last = d
    out = []
    for d in refits:
        n = con.execute("""
            WITH t AS (SELECT game_id, team_id, is_home, pts FROM nba_games
                       WHERE game_id LIKE '002%' AND pts IS NOT NULL
                       AND game_date < ? AND game_date >= ?)
            SELECT count(*) FROM t h JOIN t a USING (game_id)
            WHERE h.is_home AND NOT a.is_home""",
            [d, d - dt.timedelta(days=730)]).fetchone()[0]
        out.append({"date": str(d), "sched_n": int(n),
                    "shrink_w": round(n / (n + SCHED_SHRINK), 4)})
    cm = continuity_map(con, season, before=refits[0])
    tank_rows = con.execute("""SELECT count(*) FROM nba_games
        WHERE game_id LIKE '002%' AND season >= '2022-23' AND game_date < ?""",
        [refits[0]]).fetchone()[0]
    return {"n_refits": len(out), "first_refit": str(refits[0]),
            "sched_n_first": out[0]["sched_n"], "shrink_w_first": out[0]["shrink_w"],
            "sched_n_median": sorted(x["sched_n"] for x in out)[len(out) // 2],
            "shrink_w_median": sorted(x["shrink_w"] for x in out)[len(out) // 2],
            "carry_available": cm is not None and len(cm) > 0,
            "carry_teams": len(cm) if cm else 0,
            "tank_floor_rows_at_first_refit": int(tank_rows),
            "trajectory": out}


def main():
    res = {"note": "sched_n = team-game pairs in fit_schedule_layer's 730d window; "
                   "shrink_w = n/(n+600), the weight on DATA vs the SCHED_PRIOR "
                   "constants (2.3, -1.5, +1.5)"}
    for label, floor in (("STARVED_pre_fix", "2022-23"), ("FULL_post_fix", None)):
        con = arm_connection(floor)
        res[label] = {}
        for s in SEASONS:
            p = probe(con, s)
            if p:
                res[label][s] = p
                print(f"[{label}] {s}: first-refit sched_n={p['sched_n_first']} "
                      f"w={p['shrink_w_first']} | median n={p['sched_n_median']} "
                      f"w={p['shrink_w_median']} | carry={p['carry_available']} "
                      f"({p['carry_teams']} teams)")
            else:
                print(f"[{label}] {s}: NO GAMES")
        con.close()
    json.dump(res, open(ROOT / "data" / "ds_starvation_diag.json", "w"), indent=1)
    print("wrote data/ds_starvation_diag.json")


if __name__ == "__main__":
    main()
