#!/usr/bin/env python3
"""How good are the books? De-vig the SBR closing (and opening) moneylines into
win probabilities and score them against outcomes. This is THE bar the model
must beat (I.5 kill criterion) and the answer to "how close are we to the
market" once the model produces probabilities.

Also scores opening vs closing to show how much the line sharpens intraday
(the space CLV lives in), and compares the Elo baseline on identical games.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect
from nbapred.eval.metrics import summary
from nbapred.eval.walkforward import Elo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def american_to_prob(ml: float) -> float:
    ml = float(ml)
    return (-ml) / (-ml + 100.0) if ml < 0 else 100.0 / (ml + 100.0)


def devig_home(h_ml, v_ml):
    ph, pv = american_to_prob(h_ml), american_to_prob(v_ml)
    return ph / (ph + pv)   # multiplicative de-vig


def main():
    con = connect(read_only=True)
    rows = con.execute("""
        SELECT season, game_date, home, visitor, h_final, v_final, h_ml, v_ml
        FROM odds_hist_sbr
        WHERE h_final IS NOT NULL AND h_final <> v_final
          AND h_ml IS NOT NULL AND v_ml IS NOT NULL AND h_ml <> 0 AND v_ml <> 0
        ORDER BY season, game_date
    """).fetchall()
    con.close()

    y, p_mkt = [], []
    by_season = {}
    for season, d, home, away, hf, vf, hml, vml in rows:
        hw = int(hf > vf)
        y.append(hw)
        p_mkt.append(devig_home(hml, vml))
        by_season.setdefault(season, []).append((d, home, away, hw))

    y = np.array(y)
    print("\n=== Market accuracy (SBR closing moneyline, de-vigged) ===")
    print("games:", len(y))
    m = summary(y, p_mkt)
    print("MARKET  :", {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})

    # Elo on the same games, walk-forward (first season burn-in)
    elo = Elo()
    ey, ep = [], []
    for si, season in enumerate(sorted(by_season)):
        if si > 0:
            elo.new_season()
        for d, home, away, hw in by_season[season]:
            if si > 0:
                ey.append(hw)
                ep.append(elo.p_home(home, away))
            elo.update(home, away, hw)
    print("ELO     :", {k: round(v, 4) if isinstance(v, float) else v
                        for k, v in summary(np.array(ey), ep).items()})
    print("\nInterpretation: MARKET log loss is the number the simulator must")
    print("beat WITHIN flagged windows (I.5). Elo>market gap = room a real model")
    print("has. A model between Elo and market is progress; below market = edge.")


if __name__ == "__main__":
    main()
