#!/usr/bin/env python3
"""End-to-end harness demo/baseline on the free SBR historical outcomes
(2007-2023). Proves the eval stack works and lays down the baseline OOS
numbers every future model must beat (handoff I.5 / III.3).

Also demonstrates the ablation gate: Elo vs Elo+home-rest, judged by whether
the OOS log-loss improvement's 95% CI clears zero.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect
from nbapred.eval.ablate import ablation_report
from nbapred.eval.metrics import summary
from nbapred.eval.walkforward import Elo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_baselines")


def load_games():
    con = connect(read_only=True)
    rows = con.execute("""
        SELECT season, game_date, home, visitor, h_final, v_final,
               spread_close, spread_open
        FROM odds_hist_sbr
        WHERE h_final IS NOT NULL AND v_final IS NOT NULL AND h_final <> v_final
        ORDER BY season, game_date
    """).fetchall()
    con.close()
    return rows


def main():
    rows = load_games()
    if not rows:
        log.warning("no SBR games loaded; run scripts/load_sbr_hist.py first")
        return
    seasons = sorted({r[0] for r in rows})
    by_season = {s: [] for s in seasons}
    for season, d, home, away, hf, vf, sc, so in rows:
        by_season[season].append((d, home, away, int(hf > vf), sc))

    elo = Elo()
    y, p_elo, p_rest = [], [], []
    # a tiny, PRE-SPECIFIED home-rest bump: home teams are modestly favored
    # already in Elo; here we test whether nudging by the closing spread's sign
    # (a stand-in "extra feature") helps — expected to help a lot (it's market
    # info) and serves as a positive control that the ablation gate fires.
    for si, season in enumerate(seasons):
        if si > 0:
            elo.new_season()
        for d, home, away, hw, sc in by_season[season]:
            if si > 0:
                pe = elo.p_home(home, away)
                y.append(hw)
                p_elo.append(pe)
                # blend a hair toward the closing spread as a positive-control feature
                if sc is not None:
                    implied = 1.0 / (1.0 + 10 ** (sc / 8.0))  # home spread<0 -> >0.5
                    p_rest.append(0.5 * pe + 0.5 * implied)
                else:
                    p_rest.append(pe)
            elo.update(home, away, hw)

    y = np.array(y)
    print("\n=== Walk-forward baseline (SBR 2007-2023, first season burn-in) ===")
    print("games scored OOS:", len(y))
    print("coin flip     :", round(0.6931, 4), "(log loss reference)")
    print("Elo           :", {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in summary(y, p_elo).items()})
    print("Elo+market blend:", {k: round(v, 4) if isinstance(v, float) else v
                                for k, v in summary(y, p_rest).items()})

    print("\n=== Ablation gate (does the extra feature earn its place?) ===")
    rep = ablation_report(y, {"base": p_elo, "elo_plus_market": p_rest})
    for k, v in rep.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
