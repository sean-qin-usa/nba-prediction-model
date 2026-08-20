#!/usr/bin/env python3
"""SCORABILITY MATRIX (D152) — which seasons can `fit_production` actually fit?

The D123 fail-loud guard makes `fit_production` raise when FourFactors is not
ready, and D62's carry means readiness at OPENING NIGHT depends on the PRIOR
season's `player_game_stats` / `nba_games`. So "is season S scorable?" is
answered by calling fit_production at S's opening-night cutoff (the hardest
point in the season) and checking that it returns rather than raises.

READ-ONLY on data/nba.duckdb. Output: data/scorability.json + stdout table.

  python scripts/scorability_probe.py [--seasons 2017-18,2018-19,...]
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nbapred import threads  # noqa: E402
threads.pin(1)

from nbapred.db import connect  # noqa: E402

OUT = ROOT / "data" / "scorability.json"


def opening_night(con, season: str):
    r = con.execute(
        "SELECT min(game_date) FROM nba_games WHERE season=? AND game_id LIKE '002%'",
        [season]).fetchone()
    return r[0] if r else None


def prior_rows(con, season: str) -> tuple[int, int]:
    """(002 games, player_game_stats rows) for the PRIOR season — the carry input."""
    y = int(season[:4]) - 1
    prev = f"{y}-{(y + 1) % 100:02d}"
    g = con.execute("SELECT count(DISTINCT game_id) FROM nba_games "
                    "WHERE season=? AND game_id LIKE '002%'", [prev]).fetchone()[0]
    n = con.execute("""SELECT count(*) FROM player_game_stats s
                       WHERE s.game_id IN (SELECT game_id FROM nba_games
                         WHERE season=? AND game_id LIKE '002%')""", [prev]).fetchone()[0]
    return g, n


def probe(con, season: str) -> dict:
    """D160 — `scorable` is now `fit_returns AND sufficient`, not `fit_returns`.

    D153 caught this probe REPORTING 2007-08 AND 2008-09 AS SCORABLE WHEN THEY
    WERE NOT: the D123 fail-loud guard only checks that carry rows EXIST, and a
    single placeholder game in the prior season (24 `player_game_stats` rows)
    satisfied it. "Does `fit_production` return?" is a necessary condition, not
    a sufficient one. The sufficiency test is imported from
    `scripts/history_scorable.py` — ONE definition, so the two can no longer
    disagree — and both components are reported separately so the distinction
    stays visible instead of being collapsed into a single boolean.
    """
    from nbapred.model.production import fit_production
    out = {"season": season}
    on = opening_night(con, season)
    out["opening_night"] = str(on) if on else None
    pg, pn = prior_rows(con, season)
    out["prior_002_games"], out["prior_pgs_rows"] = pg, pn
    suf = _sufficiency(con).get(season)
    out["sufficient"] = bool(suf and suf["scorable"])
    out["insufficient_why"] = (suf or {}).get("why_not", "season absent from nba_games")
    out["own_cover"] = (suf or {}).get("own_cover")
    out["odds_rows"] = (suf or {}).get("odds_rows", 0)
    if on is None:
        out["fit_returns"] = False
        out["scorable"] = False
        out["error"] = "no 002 games in nba_games for this season"
        return out
    try:
        m = fit_production(con, season, before=on)
        out["fit_returns"] = True
        ff = getattr(m, "ff", None) if not isinstance(m, dict) else m.get("ff")
        out["ff_ready"] = bool(getattr(ff, "ready", True)) if ff is not None else True
    except Exception as e:  # noqa: BLE001
        out["fit_returns"] = False
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["trace_tail"] = traceback.format_exc().strip().splitlines()[-1][:200]
    out["scorable"] = bool(out["fit_returns"] and out["sufficient"])
    if out["fit_returns"] and not out["sufficient"]:
        out["error"] = ("fit_production RETURNS but the season is NOT scorable: "
                        + out["insufficient_why"])
    return out


_SUF_CACHE: dict = {}


def _sufficiency(con) -> dict:
    """D153's DATA-SUFFICIENCY resolver, reused verbatim (not re-implemented)."""
    if not _SUF_CACHE:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "history_scorable", Path(__file__).resolve().parent / "history_scorable.py")
        hs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hs)
        _SUF_CACHE.update(hs.resolve(con))
    return _SUF_CACHE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default=None)
    args = ap.parse_args()
    con = connect(read_only=True)
    if args.seasons:
        seasons = args.seasons.split(",")
    else:
        seasons = [r[0] for r in con.execute(
            "SELECT DISTINCT season FROM nba_games WHERE game_id LIKE '002%' "
            "ORDER BY 1").fetchall()]
    res = [probe(con, s) for s in seasons]
    con.close()
    OUT.write_text(json.dumps(res, indent=1))
    print(f"{'season':9} {'open':12} {'prior002':>9} {'priorPGS':>9} "
          f"{'fit':>5} {'suff':>5} {'scorable':>9}  note")
    for r in res:
        print(f"{r['season']:9} {str(r['opening_night']):12} "
              f"{r['prior_002_games']:9d} {r['prior_pgs_rows']:9d} "
              f"{str(r.get('fit_returns')):>5} {str(r['sufficient']):>5} "
              f"{str(r['scorable']):>9}  {r.get('error','')[:80]}")
    n = sum(1 for r in res if r["scorable"])
    print(f"\nSCORABLE: {n}/{len(res)} — "
          f"{', '.join(r['season'] for r in res if r['scorable'])}")


if __name__ == "__main__":
    main()
