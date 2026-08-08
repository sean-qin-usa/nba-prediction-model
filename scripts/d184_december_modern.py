#!/usr/bin/env python3
"""D184 — the owner's sharper version of the month question.

"if training to try to profit on post 2023 (or 2018?) markets, i think we should
not rely on the 2013-14 result, there must be something structurally wrong with
december and april in recent seasons if all negative"

This is a better argument than D182's and gets tested on its own terms:
  (1) era-restrict to the frame we actually report on (2018-19+), so a pre-feed
      season cannot rescue a month;
  (2) look for a MECHANISM rather than a cell that happens to be low.

Declared before scoring:
  H1  December is negative in every modern season -> sign test + pooled CI.
  H2  April likewise.
  H3  If the effect is real there should be a mechanism visible in the inputs,
      not just in the endpoint: December absence load (the availability leg is
      the model's injury-reactive half, so a month with anomalous absence volume
      is where it should misprice).
  H4  What would the filter cost/earn prospectively, and how much December
      exposure does the strategy even have?

Read-only. Nothing ships.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                                # noqa: E402

from nbapred.db import connect                                    # noqa: E402

MODERN = "2018-19"          # the reported model frame
TIER = "k=9 raw"
SEED = 20260805


def roi(r):
    return 100 * sum(x["ev"] for x in r) / len(r) if r else float("nan")


def main():
    pb = json.load(open(ROOT / "data" / "wf_perbet_D181.json"))
    b = [dict(s=x["season"], m=x["date"][5:7], d=x["date"],
              ev=x["ev"] * x["keep"]) for x in pb[TIER]]
    seasons = sorted({x["s"] for x in b})
    modern = [s for s in seasons if s >= MODERN]
    print(f"tier {TIER}   modern frame = {MODERN}+  ({len(modern)} seasons)\n")

    # -------------------------------------------------------------- H1 / H2
    print("=" * 74)
    print("H1/H2  DECEMBER AND APRIL, ERA-RESTRICTED TO THE REPORTED FRAME")
    print("=" * 74)
    rng = np.random.default_rng(SEED)
    out = {}
    for mo, nm in (("12", "DECEMBER"), ("04", "APRIL")):
        rows = [x for x in b if x["m"] == mo and x["s"] >= MODERN]
        bys = {s: [x for x in rows if x["s"] == s] for s in modern}
        bys = {s: v for s, v in bys.items() if v}
        print(f"\n  --- {nm} ({MODERN}+) ---")
        for s, v in bys.items():
            print(f"     {s}  n={len(v):4d}  ROI {roi(v):+8.2f}%")
        neg = sum(1 for v in bys.values() if roi(v) < 0)
        K = len(bys)
        # exact one-sided sign test
        from math import comb
        p_sign = sum(comb(K, i) for i in range(neg, K + 1)) / 2 ** K
        pool = roi(rows)
        ev = np.array([x["ev"] for x in rows])
        boot = np.array([rng.choice(ev, len(ev), replace=True).mean()
                         for _ in range(4000)]) * 100
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"     POOLED n={len(rows)}  ROI {pool:+.2f}%  "
              f"bootstrap 95% CI [{lo:+.2f}, {hi:+.2f}]")
        print(f"     negative in {neg}/{K} modern seasons  "
              f"-> exact sign-test p = {p_sign:.4f} "
              f"{'(SIG at .05)' if p_sign <= 0.05 else '(ns at .05)'}")
        out[nm] = dict(n=len(rows), roi=pool, neg=neg, K=K, p_sign=p_sign,
                       ci=[float(lo), float(hi)])

    # -------------------------------------------------------------- H4 first
    print("\n" + "=" * 74)
    print("H4  HOW MUCH DECEMBER EXPOSURE DOES THE STRATEGY EVEN HAVE?")
    print("=" * 74)
    ndec = {s: len([x for x in b if x["s"] == s and x["m"] == "12"])
            for s in seasons}
    zero = [s for s in seasons if ndec[s] == 0]
    print(f"  seasons with ZERO December bets: {len(zero)}/{len(seasons)} "
          f"-> {', '.join(zero)}")
    tot = sum(ndec.values())
    print(f"  December bets: {tot} of {len(b)} = {100*tot/len(b):.1f}% of all "
          f"volume across 14 seasons")
    mtot = sum(ndec[s] for s in modern)
    mall = len([x for x in b if x["s"] >= MODERN])
    print(f"  in the modern frame: {mtot} of {mall} = {100*mtot/mall:.1f}%")
    print("  -> the walk-forward selector ALREADY declines December in most "
          "seasons;\n     the question is whether to hard-code what it mostly "
          "does anyway.")

    # -------------------------------------------------------------- H3
    print("\n" + "=" * 74)
    print("H3  MECHANISM PROBE — is December anomalous in the model's INPUTS?")
    print("=" * 74)
    con = connect(read_only=True)
    q = """
      WITH g AS (SELECT DISTINCT game_id, game_date, season FROM nba_games
                 WHERE game_id LIKE '002%'),
      o AS (SELECT game_date, count(*) n_out
            FROM injury_reports_pit WHERE status IN ('Out','Doubtful')
            GROUP BY 1)
      SELECT g.season, strftime(g.game_date, '%m') mo,
             count(DISTINCT g.game_id) games,
             avg(COALESCE(o.n_out, 0)) mean_out
      FROM g LEFT JOIN o USING (game_date)
      WHERE g.season >= ? GROUP BY 1,2 ORDER BY 1,2"""
    try:
        rows = con.execute(q, [MODERN]).fetchall()
    except Exception as e:
        print(f"  (injury_reports_pit probe unavailable: {str(e)[:80]})")
        rows = []
    if rows:
        bym = defaultdict(list)
        for season, mo, games, mout in rows:
            if mo in ("10", "11", "12", "01", "02", "03", "04"):
                bym[mo].append(float(mout or 0))
        print(f"  {'month':7} {'mean daily Out+Doubtful listings':>34} "
              f"{'vs season mean':>15}")
        allm = np.mean([v for vs in bym.values() for v in vs])
        for mo in ("10", "11", "12", "01", "02", "03", "04"):
            if bym.get(mo):
                mu = np.mean(bym[mo])
                print(f"  {mo:7} {mu:34.1f} {100*(mu/allm-1):+14.1f}%")
        d = np.mean(bym.get("12", [np.nan]))
        print(f"\n  December absence load vs all-month mean: "
              f"{100*(d/allm-1):+.1f}%")
        print("  -> a mechanism story needs December to be ANOMALOUS here.")
    con.close()

    json.dump(out, open(ROOT / "data" / "d184_december.json", "w"), indent=1)
    print("\nwrote data/d184_december.json")


if __name__ == "__main__":
    main()
