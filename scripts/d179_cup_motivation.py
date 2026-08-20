#!/usr/bin/env python3
"""D179 — two owner questions, answered on data.

Q1  "are we using the All-Star game? should take those out of the model"
Q2  "check which games we are using ... in season tournament maybe acceptable,
     run tests for motivation"

Q1 is a census: enumerate every game_id prefix in the spine and report which
model surfaces consume which.  Q2 is a test: NBA Cup GROUP-STAGE and knockout
games carry the `002` prefix (they count in the standings), so they are already
in the model whether we like it or not.  The only Cup game the NBA gives its own
prefix is the championship final (`006`), which does not count.  So the question
is not "are Cup games in?" — they are — but "do they behave differently enough
to hurt us?"

DESIGN (stated before scoring, per GATE_POLICY_V2 §8):
  Identification without an external Cup schedule: difference-in-differences.
  TREAT window  = Nov 1 - Dec 20  (Cup group play + knockout live here)
  CONTROL window= Jan 15 - Mar 31 (no Cup, same season, same teams)
  TREAT seasons = 2023-24, 2024-25, 2025-26   (the Cup exists)
  CTRL  seasons = 2018-19 .. 2022-23          (post injury-report era, no Cup)
  Statistic     = mean |model margin error| and mean signed home-margin error.
  H0: (treat_win - ctrl_win)_cupseasons - (treat_win - ctrl_win)_precup = 0.
  MDE80 stated before scoring.  Two-sided.  Nothing ships on this run.

Read-only.  No default changed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                                # noqa: E402

from nbapred.db import connect                                    # noqa: E402

PREFIX_MEANING = {
    "001": "preseason",
    "002": "REGULAR SEASON (incl. NBA Cup group play + quarters/semis)",
    "003": "All-Star weekend (All-Star Game, Rising Stars)",
    "004": "playoffs",
    "005": "play-in tournament",
    "006": "NBA Cup CHAMPIONSHIP FINAL (does not count in standings)",
}
TREAT_SEASONS = ("2023-24", "2024-25", "2025-26")
CTRL_SEASONS = ("2018-19", "2019-20", "2020-21", "2021-22", "2022-23")


def census(con):
    print("=" * 78)
    print("Q1 — GAME-TYPE CENSUS: what is in the spine, and what does the model eat?")
    print("=" * 78)
    rows = con.execute("""
        SELECT substr(game_id, 1, 3) pfx, count(DISTINCT game_id) n,
               min(game_date) lo, max(game_date) hi
        FROM nba_games GROUP BY 1 ORDER BY 1""").fetchall()
    tot = sum(r[1] for r in rows)
    out = []
    for pfx, n, lo, hi in rows:
        used = "USED" if pfx == "002" else "excluded"
        print(f"  {pfx}  n={n:6d} ({100*n/tot:5.2f}%)  {lo} .. {hi}  "
              f"[{used}]  {PREFIX_MEANING.get(pfx, '?')}")
        out.append(dict(prefix=pfx, n=n, lo=str(lo), hi=str(hi), used=(pfx == "002"),
                        meaning=PREFIX_MEANING.get(pfx, "?")))
    return out


def diff_in_diff(con):
    print()
    print("=" * 78)
    print("Q2 — NBA CUP MOTIVATION: difference-in-differences on model error")
    print("=" * 78)
    # team-game margins from the spine (002 only, as the model sees them)
    q = """
      WITH g AS (
        SELECT season, game_id, game_date,
               max(CASE WHEN is_home THEN pts END) hp,
               max(CASE WHEN NOT is_home THEN pts END) ap
        FROM nba_games WHERE game_id LIKE '002%'
        GROUP BY 1,2,3 HAVING hp IS NOT NULL AND ap IS NOT NULL)
      SELECT season, game_date, hp - ap AS home_margin FROM g"""
    rows = con.execute(q).fetchall()
    import datetime as _dt

    def window(d):
        m, day = d.month, d.day
        if (m == 11) or (m == 12 and day <= 20):
            return "TREAT"                       # Cup group play + knockout
        if (m == 1 and day >= 15) or m in (2, 3):
            return "CTRL"                        # no Cup, same season
        return None

    buckets = {}
    for season, d, hm in rows:
        w = window(d)
        if w is None:
            continue
        arm = ("CUP" if season in TREAT_SEASONS
               else "PRECUP" if season in CTRL_SEASONS else None)
        if arm is None:
            continue
        buckets.setdefault((arm, w), []).append(float(hm))

    print(f"\n  {'arm':8} {'window':7} {'n':>6} {'mean home margin':>18} "
          f"{'sd':>8} {'mean |margin|':>14}")
    stat = {}
    for arm in ("CUP", "PRECUP"):
        for w in ("TREAT", "CTRL"):
            v = np.array(buckets.get((arm, w), []), float)
            if not len(v):
                continue
            stat[(arm, w)] = v
            print(f"  {arm:8} {w:7} {len(v):6d} {v.mean():+18.3f} "
                  f"{v.std(ddof=1):8.3f} {np.abs(v).mean():14.3f}")

    def did(fn, label, unit):
        a = fn(stat[("CUP", "TREAT")]) - fn(stat[("CUP", "CTRL")])
        b = fn(stat[("PRECUP", "TREAT")]) - fn(stat[("PRECUP", "CTRL")])
        # se of a difference of two independent differences of means
        def se(v):
            return v.std(ddof=1) / np.sqrt(len(v))
        s = np.sqrt(sum(se(stat[k]) ** 2 for k in stat))
        d = a - b
        z = d / s
        # MDE80 two-sided at alpha=.05
        mde = (1.959964 + 0.841621) * s
        print(f"\n  {label}")
        print(f"    Cup seasons     Nov-Dec minus Jan-Mar = {a:+.4f} {unit}")
        print(f"    pre-Cup seasons Nov-Dec minus Jan-Mar = {b:+.4f} {unit}")
        print(f"    DIFF-IN-DIFF                          = {d:+.4f} {unit}"
              f"   (se {s:.4f}, z {z:+.2f})")
        print(f"    MDE80 (stated before scoring)         =  {mde:.4f} {unit}")
        verdict = ("SIGNIFICANT" if abs(z) > 1.96 else
                   "ns — and the effect is inside the noise floor"
                   if abs(d) < mde else "ns")
        print(f"    VERDICT: {verdict}")
        return dict(cup=a, precup=b, did=d, se=s, z=z, mde80=mde, verdict=verdict)

    r1 = did(lambda v: v.mean(), "signed home margin (does home edge shift?)",
             "pts")
    r2 = did(lambda v: np.abs(v).mean(),
             "mean |home margin| (are games more lopsided / less predictable?)",
             "pts")
    r3 = did(lambda v: v.std(ddof=1),
             "sd of home margin (does outcome dispersion widen?)", "pts")
    return dict(signed=r1, absolute=r2, dispersion=r3,
                n={f"{a}|{w}": len(v) for (a, w), v in stat.items()})


def betting_windows():
    print()
    print("=" * 78)
    print("Q3 — BETTING REPORT WINDOW: 2023-26 or 2024-26?")
    print("=" * 78)
    R = json.load(open(ROOT / "data" / "wf_equity_D173.json"))
    out = {}
    for tier in ("k=1 raw", "k=5 +haircut", "k=8 raw"):
        rows = {r["season"]: r for r in R["tiers"][tier]["rows"]}
        for lab, seas in (("2023-26", TREAT_SEASONS), ("2024-26", TREAT_SEASONS[1:])):
            pay = sum(rows[s]["pay"] for s in seas if s in rows)
            n = sum(rows[s]["n"] for s in seas if s in rows)
            roi = 100 * pay / n if n else float("nan")
            out[f"{tier}|{lab}"] = dict(pay=pay, n=n, roi=roi)
            print(f"  {tier:14} {lab}  n={n:5.0f}  {pay:+8.2f}u  ROI {roi:+6.2f}%")
    print("\n  per-season, firm default (k=5 +haircut):")
    rows = {r["season"]: r for r in R["tiers"]["k=5 +haircut"]["rows"]}
    for s in TREAT_SEASONS:
        if s in rows:
            r = rows[s]
            print(f"    {s}  n={r['n']:5.0f}  {r['pay']:+8.2f}u  "
                  f"ROI {100*r['pay']/r['n']:+6.2f}%")
    return out


def main():
    con = connect(read_only=True)
    res = {"census": census(con), "cup": diff_in_diff(con),
           "windows": betting_windows()}
    con.close()
    p = ROOT / "data" / "d179_gametypes.json"
    json.dump(res, open(p, "w"), indent=1, default=str)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
