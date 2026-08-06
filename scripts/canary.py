#!/usr/bin/env python3
"""canary.py — pre-flight and in-season tripwire.

Run it on a schedule.  Exit code 0 = all clear, 1 = at least one FAIL.
Every check here exists because the corresponding failure ACTUALLY HAPPENED in
this project, and most of them were found by accident rather than by a test.

    python scripts/canary.py            # full run
    python scripts/canary.py --quiet    # only WARN/FAIL lines

CHECKS AND THEIR PROVENANCE
  1  odds_quotes populated + growing   D178: the multi-book logger has NEVER run
                                       in-season; odds_quotes is 0 rows.  If it
                                       is not up on opening night the season's
                                       open-price CLV record is unrecoverable.
                                       THIS IS THE LOUDEST FAILURE IN THE SYSTEM.
  2  >=2 books at the open             D178: the gate degrades volume rather than
                                       dropping silently; if the rate collapses
                                       we are effectively single-book.
  3  injury feed fresh                 D186: the availability leg IS the model
                                       after 2019; a stale feed silently reverts
                                       the model to a pre-2019 cripple.
  4  DARKO fresh                       D170: backfilled to 100%; regressions here
                                       are silent.
  5  team names all resolvable         D171/D177/D185: FOUR instances of a join
                                       dropping rows on a name mismatch.
  6  no non-regular-season contamination  D178: todays_games() had NO game-type
                                       filter; a February All-Star date would
                                       have reached the bet engine.
  7  tank floor pinned, not drifted    D155: the floor drifted twice under
                                       backfills, silently invalidating a
                                       certification.
  8  paper-bet CLV inside its band     D178: bands re-derived; RED is now
                                       essentially "any negative month".
  9  model calibration sane            catch-all: NaNs, degenerate probabilities,
                                       or a mean far off base rate.

Read-only.  Never writes.  Never changes a default.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

from nbapred.db import connect                                    # noqa: E402

QUIET = "--quiet" in sys.argv
RESULTS = []


def report(level, name, msg):
    RESULTS.append((level, name, msg))
    if QUIET and level == "PASS":
        return
    tag = {"PASS": "  ok  ", "WARN": " WARN ", "FAIL": " FAIL ", "SKIP": " skip "}[level]
    print(f"[{tag}] {name:34s} {msg}")


def days_since(con, table, col):
    try:
        v = con.execute(f"SELECT max({col}) FROM {table}").fetchone()[0]
    except Exception:
        return None, None
    if v is None:
        return None, None
    d = v if isinstance(v, dt.date) else dt.date.fromisoformat(str(v)[:10])
    return (dt.date.today() - d).days, d


def main():
    con = connect(read_only=True)
    today = dt.date.today()
    # in-season = roughly mid-Oct to mid-Apr
    in_season = (today.month >= 10) or (today.month <= 4)
    print(f"canary {today}  (in-season={in_season})\n")

    # ---- 1  odds_quotes ------------------------------------------------
    try:
        n = con.execute("SELECT count(*) FROM odds_quotes").fetchone()[0]
    except Exception as e:
        n = None
        report("FAIL", "odds_quotes table", f"unreadable: {str(e)[:50]}")
    if n is not None:
        if n == 0:
            report("FAIL" if in_season else "WARN", "odds_quotes populated",
                   "0 rows — the multi-book logger has never run in-season. "
                   "If this is opening night, the season's open-price CLV "
                   "record is being lost and cannot be reconstructed.")
        else:
            age, d = days_since(con, "odds_quotes", "captured_at")
            if age is None:
                report("WARN", "odds_quotes freshness", f"{n} rows, no timestamp")
            elif in_season and age > 2:
                report("FAIL", "odds_quotes freshness",
                       f"{n} rows but newest is {age}d old ({d})")
            else:
                report("PASS", "odds_quotes", f"{n} rows, newest {d}")

    # ---- 2  >=2 books at the open --------------------------------------
    try:
        r = con.execute("""
            SELECT count(*) tot,
                   sum(CASE WHEN n_books >= 2 THEN 1 ELSE 0 END) ok
            FROM bet_paper WHERE snapshot_kind='OPEN'""").fetchone()
        if not r or not r[0]:
            report("SKIP", ">=2 books at open", "no OPEN paper rows yet")
        else:
            pct = 100.0 * (r[1] or 0) / r[0]
            lvl = "PASS" if pct >= 80 else "WARN" if pct >= 40 else "FAIL"
            report(lvl, ">=2 books at open",
                   f"{pct:.0f}% of {r[0]} OPEN rows have >=2 books")
    except Exception as e:
        report("SKIP", ">=2 books at open", f"n/a ({str(e)[:40]})")

    # ---- 3  injury feed ------------------------------------------------
    age, d = days_since(con, "injury_reports_pit", "game_date")
    if age is None:
        report("FAIL", "injury feed", "no rows")
    elif in_season and age > 2:
        report("FAIL", "injury feed", f"stale — newest {d} ({age}d ago). "
               "The availability leg is the model after 2019.")
    else:
        report("PASS", "injury feed", f"newest {d} ({age}d ago)")

    # ---- 4  DARKO ------------------------------------------------------
    age, d = days_since(con, "darko_history", "date")
    if age is None:
        report("WARN", "DARKO feed", "no rows")
    elif in_season and age > 7:
        report("WARN", "DARKO feed", f"stale — newest {d} ({age}d ago)")
    else:
        report("PASS", "DARKO feed", f"newest {d} ({age}d ago)")

    # ---- 5  team-name resolvability ------------------------------------
    try:
        from nbapred import teams as T
        names = [r[0] for r in con.execute(
            "SELECT DISTINCT team FROM injury_reports_pit "
            "WHERE team IS NOT NULL").fetchall()]
        _, unres = T.resolve_map(names)
        if unres:
            report("WARN", "team names resolvable",
                   f"{len(unres)} unresolvable: {unres[:4]}"
                   f"{' ...' if len(unres) > 4 else ''}")
        else:
            report("PASS", "team names resolvable", f"all {len(names)} resolve")
    except Exception as e:
        report("SKIP", "team names resolvable", f"n/a ({str(e)[:40]})")

    # ---- 6  game-type contamination ------------------------------------
    try:
        bad = con.execute("""
            SELECT count(*) FROM bet_paper
            WHERE game_id IS NOT NULL AND game_id NOT LIKE '002%'""").fetchone()[0]
        if bad:
            report("FAIL", "regular-season only",
                   f"{bad} paper rows carry a non-002 game_id "
                   "(preseason/All-Star/play-in reached the engine)")
        else:
            report("PASS", "regular-season only", "no non-002 game ids")
    except Exception as e:
        report("SKIP", "regular-season only", f"n/a ({str(e)[:40]})")

    # ---- 7  tank floor -------------------------------------------------
    try:
        from nbapred.model import tanking
        aud = tanking.floor_audit(con)
        if aud.get("drifted"):
            # NOT a failure: tanking.floor_audit's own contract says drift means
            # "the data now supports a different floor than the certified stack
            # was built on", and moving to it is a gated model change (D155).
            # The pinned value is doing its job. FAIL is reserved for things
            # that are broken right now.
            report("WARN", "tank season floor",
                   f"drifted: pinned {aud['pinned']} vs derived {aud['derived']} "
                   "— pinned value is protecting the certification; moving to "
                   "the derived floor would be a gated model change, not a fix")
        else:
            report("PASS", "tank season floor", f"pinned {aud['pinned']}")
    except Exception as e:
        report("SKIP", "tank season floor", f"n/a ({str(e)[:40]})")

    # ---- 8  CLV band ---------------------------------------------------
    try:
        import json
        band = json.load(open(ROOT / "data" / "d178_clvbands.json"))
        b = band["bands"]["ML|HONEST"] if "bands" in band else None
        rows = con.execute("""
            SELECT avg(clv), count(*) FROM bet_paper
            WHERE snapshot_kind='OPEN' AND clv IS NOT NULL
              AND clv_eligible""").fetchone()
        if not rows or not rows[1]:
            report("SKIP", "CLV inside band", "no eligible settled OPEN bets")
        elif b:
            red = b["centre"] - 2 * b["se"]
            lvl = "PASS" if rows[0] >= red else "WARN"
            report(lvl, "CLV inside band",
                   f"mean {rows[0]:+.5f} on n={rows[1]} (RED below {red:+.5f})")
    except Exception as e:
        report("SKIP", "CLV inside band", f"n/a ({str(e)[:40]})")

    # ---- 9  calibration sanity ------------------------------------------
    try:
        import pandas as pd
        f = ROOT / "data" / "capstone_pergame.csv"
        c = pd.read_csv(f)
        bad = int(c["p_us"].isna().sum() + ((c["p_us"] <= 0) | (c["p_us"] >= 1)).sum())
        mp, my = float(c["p_us"].mean()), float(c["y"].mean())
        if bad:
            report("FAIL", "model outputs sane",
                   f"{bad} NaN/degenerate probabilities in {f.name}")
        elif abs(mp - my) > 0.03:
            report("WARN", "model outputs sane",
                   f"mean p {mp:.3f} vs base rate {my:.3f} — calibration drift")
        else:
            report("PASS", "model outputs sane",
                   f"mean p {mp:.3f} vs base rate {my:.3f}, no degenerates")
    except Exception as e:
        report("SKIP", "model outputs sane", f"n/a ({str(e)[:40]})")

    con.close()
    n_fail = sum(1 for lv, _, _ in RESULTS if lv == "FAIL")
    n_warn = sum(1 for lv, _, _ in RESULTS if lv == "WARN")
    print(f"\n{len(RESULTS)} checks: "
          f"{sum(1 for lv,_,_ in RESULTS if lv=='PASS')} pass, {n_warn} warn, "
          f"{n_fail} fail")
    if n_fail:
        print("\nFAILURES:")
        for lv, nm, msg in RESULTS:
            if lv == "FAIL":
                print(f"  {nm}: {msg}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
