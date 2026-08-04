"""Shared helper for the DATA-STARVATION re-tests: arm connections that expose
the SAME DuckDB with a TRUNCATED `nba_games` corpus.

Every cross-season trailing consumer in production reads `nba_games`
unqualified:
  * fit_schedule_layer()  — 730-DAY trailing window (the only cross-season
    consumer that is season-agnostic; it was silently half-fed on 2023-24
    because 2021-22 rows did not exist)
  * continuity_map()/ps_continuity()/carry — prior-season 002 rows
  * tanking.TankModel / latestate.LateStateModel — floor at season >= literal
  * game_rows()/factor_game_rows()/games_played — season-scoped (unaffected)

To build a SAME-RUN control we attach the real DB READ-ONLY into an in-memory
catalog and shadow every table with a view; `nba_games` gets a season filter.
Unqualified names resolve to the in-memory catalog first, so production code is
untouched and the DB is never opened for writing.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import duckdb  # noqa: E402

from nbapred.config import DB_PATH  # noqa: E402


def arm_connection(min_season: str | None = None) -> duckdb.DuckDBPyConnection:
    """In-memory catalog of views over the read-only real DB.

    min_season=None  -> FULL corpus (every nba_games row).
    min_season='2022-23' -> the pre-fix STARVED corpus (the state every gate
                            from D46 through D94 was measured in).
    """
    con = duckdb.connect(":memory:")
    con.execute(f"ATTACH '{DB_PATH}' AS src (READ_ONLY)")
    tables = [r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_catalog = 'src' AND table_schema = 'main'").fetchall()]
    for t in tables:
        if t == "nba_games" and min_season is not None:
            con.execute(f"CREATE VIEW {t} AS SELECT * FROM src.main.{t} "
                        f"WHERE season >= '{min_season}'")
        else:
            con.execute(f"CREATE VIEW {t} AS SELECT * FROM src.main.{t}")
    return con


def paired_bootstrap(delta, n_boot: int = 2000, seed: int = 20260731):
    """Paired bootstrap mean + 95% CI on a per-game delta vector (positive =
    variant better, matching the repo's log-loss-improvement convention)."""
    import numpy as np
    d = np.asarray(delta, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(d)
    if n == 0:
        return {"n": 0, "mean": 0.0, "lo": 0.0, "hi": 0.0, "verdict": "EMPTY"}
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"n": n, "mean": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "verdict": "PASS" if lo > 0 or hi < 0 else "NS"}
