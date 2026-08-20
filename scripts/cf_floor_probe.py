"""CF-0 — corpus-floor probe (D112 JOB 1a).

Builds the TankModel + LateStateModel under BOTH floors in one process and
reports:
  (i)  REFACTOR PARITY: at TANK_SEASON_FLOOR=2022-23 the new code must
       reproduce the pre-D112 build exactly — tank_score vs the pinned gate
       table data/apr_tank_stats.csv, and fit_k(2026-04-09) vs the pinned
       ship value -2.2699.
  (ii) K TRAJECTORY: fit_k at each season's tank-window opening under both
       floors, plus the coldness measure the D110 audit used (mean |term|).
  (iii) LATE-STATE COEFS: c_f / c_o at the same dates under both floors.

Read-only. Usage: python scripts/cf_floor_probe.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nbapred.db import connect  # noqa: E402

# tank window (gp>=55) opens ~Feb; sample k through each season's late window
PROBE_DATES = [dt.date(2022, 2, 15), dt.date(2022, 4, 9),
               dt.date(2023, 2, 15), dt.date(2023, 4, 8),
               dt.date(2024, 2, 15), dt.date(2024, 4, 13),
               dt.date(2025, 2, 15), dt.date(2025, 4, 12),
               dt.date(2026, 2, 15), dt.date(2026, 4, 9)]
SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def build(con, floor):
    import nbapred.model.latestate as ls
    import nbapred.model.tanking as tk
    tk._CACHE.clear()
    ls._CACHE.clear()
    tank = tk.TankModel(con, floor=floor)
    late = ls.LateStateModel(con, tank)
    return tank, late


def summarize(tank, late, con):
    out = {"floor": tank.floor, "burn_in": late.burn_in,
           "n_active_k_rows": int(len(tank._act_dates)),
           "n_active_late_rows": int(len(late._act_dates)),
           "k": {}, "late_coefs": {}, "per_season": {}}
    for d in PROBE_DATES:
        out["k"][str(d)] = round(float(tank.fit_k(d)), 5)
        cf, co = late.coefs(d)
        out["late_coefs"][str(d)] = [round(cf, 5), round(co, 5)]

    # per-season applied-term strength on the REAL game list
    g = con.execute("""
        WITH t AS (SELECT season, game_id, game_date, team_id, is_home
                   FROM nba_games WHERE game_id LIKE '002%' AND wl IS NOT NULL)
        SELECT h.season, h.game_id, h.game_date, h.team_id ht, a.team_id awt
        FROM t h JOIN t a USING (game_id)
        WHERE h.is_home AND NOT a.is_home ORDER BY h.game_date""").fetchdf()
    for s in SEASONS:
        sub = g[g.season == s]
        tds, lts, ks = [], [], []
        for r in sub.itertuples():
            d = r.game_date.date() if hasattr(r.game_date, "date") else r.game_date
            td = tank.diff(int(r.ht), int(r.awt), d)
            k = tank.fit_k(d)
            tds.append(abs(k * td))
            ks.append(k)
            lts.append(abs(late.term(int(r.ht), int(r.awt), set(), set(), d)))
        nz = [x for x in tds if x != 0]
        nzl = [x for x in lts if x != 0]
        out["per_season"][s] = {
            "n": len(sub),
            "tank_active_share": round(len(nz) / max(len(sub), 1), 4),
            "tank_mean_abs_term_active": round(float(np.mean(nz)) if nz else 0.0, 4),
            "k_min": round(float(min(ks)), 4), "k_max": round(float(max(ks)), 4),
            "late_active_share": round(len(nzl) / max(len(sub), 1), 4),
            "late_mean_abs_term_active": round(float(np.mean(nzl)) if nzl else 0.0, 4),
        }
    return out


def main():
    con = connect(read_only=True)
    res = {}

    # ---- (i) refactor parity at the OLD floor ----------------------------
    os.environ["TANK_SEASON_FLOOR"] = "2022-23"
    tank_old, late_old = build(con, "2022-23")
    apr = pd.read_csv(ROOT / "data/apr_tank_stats.csv", dtype={"game_id": str})
    apr["game_date"] = pd.to_datetime(apr["game_date"])
    ref = {(int(r.team_id), r.game_date.date()): float(r.tank_score)
           for r in apr.itertuples()}
    diffs = [abs(tank_old.score(t, d)[0] - v) for (t, d), v in ref.items()
             if (t, d) in tank_old.map]
    res["refactor_parity_old_floor"] = {
        "n_compared": len(diffs),
        "max_abs_tank_score_diff": float(max(diffs)) if diffs else None,
        "fit_k_2026_04_09": round(float(tank_old.fit_k(dt.date(2026, 4, 9))), 6),
        "pinned_ship_k": -2.2699}
    res["OLD_2223"] = summarize(tank_old, late_old, con)

    # ---- (ii) the NEW derived floor --------------------------------------
    del os.environ["TANK_SEASON_FLOOR"]
    from nbapred.model.tanking import season_floor
    new_floor = season_floor(con)
    tank_new, late_new = build(con, new_floor)
    res["NEW_derived"] = summarize(tank_new, late_new, con)

    con.close()
    json.dump(res, open(ROOT / "data" / "cf_floor_probe.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("\nwrote data/cf_floor_probe.json")


if __name__ == "__main__":
    main()
