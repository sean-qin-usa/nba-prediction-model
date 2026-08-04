"""Cold-start vs warm-start (last-season prior, regress 0.75), like-for-like:
same games, same seasons. Scores ALL games from game 1 (both models score every
game) plus the early-season subset. Output JSON for the chart.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect
from nbapred.eval.metrics import log_loss
from scripts.eval_coldstart import MarginSRS, season_end_ratings

SCALE = 7.2
sig = lambda x: 1 / (1 + np.exp(-np.asarray(x)))


def main(regress=0.75, ridge=30.0, refit_every=10, early_n=15):
    con = connect(read_only=True)
    df = con.execute("""SELECT season_end, game_date, home, away, score_home, score_away,
        home_win, p_home_spread FROM odds_market WHERE season_end>=2016
        ORDER BY season_end, game_date""").fetchdf()
    con.close()

    prev_end = {}
    out = []
    for se, grp in df.groupby("season_end"):
        grp = grp.sort_values("game_date").reset_index(drop=True)
        prior = {t: regress * v for t, v in prev_end.items()}
        hist, cold, warm, since, tg = [], None, None, 10**9, {}
        rec = {k: ([], []) for k in ("cold_all", "warm_all", "cold_early", "warm_early",
                                     "mkt_all")}
        for i, g in grp.iterrows():
            if since >= refit_every:
                cold = MarginSRS(ridge).fit(hist)
                warm = MarginSRS(ridge, prior=prior).fit(hist)
                since = 0
            since += 1
            y = int(g.home_win)
            pc = sig((cold.margin(g.home, g.away) if cold and cold.r else 3.0) / SCALE)
            pw = sig((warm.margin(g.home, g.away) if warm and warm.r else
                      prior.get(g.home, 0) - prior.get(g.away, 0) + 3.0) / SCALE)
            rec["cold_all"][0].append(y); rec["cold_all"][1].append(float(pc))
            rec["warm_all"][0].append(y); rec["warm_all"][1].append(float(pw))
            rec["mkt_all"][0].append(y); rec["mkt_all"][1].append(float(g.p_home_spread))
            if max(tg.get(g.home, 0), tg.get(g.away, 0)) < early_n and prev_end:
                rec["cold_early"][0].append(y); rec["cold_early"][1].append(float(pc))
                rec["warm_early"][0].append(y); rec["warm_early"][1].append(float(pw))
            hist.append((g.home, g.away, g.score_home - g.score_away))
            tg[g.home] = tg.get(g.home, 0) + 1; tg[g.away] = tg.get(g.away, 0) + 1
        prev_end = season_end_ratings(hist, ridge)
        row = {"season": f"{se-1}-{str(se)[2:]}"}
        for k, (y, p) in rec.items():
            row[k] = round(log_loss(np.array(y), p), 4) if len(y) > 100 else None
        out.append(row)
        print(row)
    json.dump(out, open(Path(__file__).resolve().parent.parent / "data" / "coldwarm.json", "w"))


if __name__ == "__main__":
    main()
