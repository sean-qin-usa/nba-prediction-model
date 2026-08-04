#!/usr/bin/env python3
"""CARRY-ALL identity check (D45/D134 control discipline applied to IDENTITY).

Proves that `ca_bank.Layer.fit(before, cols=())` reproduces
`production.fit_schedule_layer(con, before)` -- same trailing frame, same
design matrix, same lstsq, same n/(n+600) shrinkage -- at a battery of refit
dates spanning every scorable era.  If this does not hold to lstsq round-off,
every number downstream is about the replica and not about the terms
(hall-of-shame #15).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import nbapred.threads as _t  # noqa: E402

_t.pin(1)
import datetime as dt  # noqa: E402

import numpy as np  # noqa: E402

from ca_bank import Layer, load_bank  # noqa: E402
from nbapred.db import connect  # noqa: E402
from nbapred.model.production import fit_schedule_layer  # noqa: E402

DATES = [dt.date(y, m, d) for (y, m, d) in [
    (2012, 1, 15), (2013, 3, 1), (2017, 11, 20), (2018, 12, 25),
    (2019, 2, 14), (2020, 1, 5), (2021, 3, 10), (2021, 10, 19),
    (2022, 1, 3), (2022, 12, 25), (2023, 10, 24), (2024, 2, 1),
    (2025, 4, 10), (2025, 10, 21), (2026, 3, 1)]]

if __name__ == "__main__":
    con = connect(read_only=True)
    b = load_bank(con)
    L = Layer(b)
    worst = 0.0
    print(f"{'date':12s} {'n_ship':>7s} {'n_bank':>7s} {'max|dbeta|':>12s}  shipped 5-tuple")
    for d in DATES:
        ship = fit_schedule_layer(con, d)
        mine = L.fit(d, cols=())
        base5, _, _, n, w, _ = mine
        nship = con.execute("""
            WITH t AS (SELECT game_id, game_date, team_id, is_home, pts
                       FROM nba_games WHERE game_id LIKE '002%' AND pts IS NOT NULL
                       AND game_date < ? AND game_date >= ?)
            SELECT count(*) FROM t h JOIN t a USING (game_id)
            WHERE h.is_home AND NOT a.is_home""",
            [d, d - dt.timedelta(days=730)]).fetchone()[0]
        dmax = max(abs(a - c) for a, c in zip(ship, base5))
        worst = max(worst, dmax)
        print(f"{str(d):12s} {nship:7d} {n:7d} {dmax:12.3e}  "
              + " ".join(f"{x:+.5f}" for x in ship))
    print(f"\nWORST |dbeta| over {len(DATES)} refit dates: {worst:.3e}")
    assert worst < 1e-9, "REPLICA DOES NOT MATCH THE SHIPPED LAYER"
    print("IDENTITY OK — Layer(cols=()) == fit_schedule_layer")
    con.close()
