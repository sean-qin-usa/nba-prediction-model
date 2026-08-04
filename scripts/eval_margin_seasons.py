"""Our win-prob model's prediction power for EVERY recent season, using only
final scores (which we have for 2008-2026 via odds_market). Opponent-adjusted
margin SRS (ridge), walk-forward within each season. This is a faithful proxy
for our team model where we lack play-by-play; on 2025-26 the efficiency version
scored ~0.601 full / 0.5815 mature, so this margin version should track it.
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

SCALE = 7.2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


class MarginSRS:
    def __init__(self, ridge=30.0):
        self.ridge = ridge; self.r = {}; self.home = 3.0; self.mu = 0.0

    def fit(self, rows):
        # rows: (home, away, home_margin)
        teams = sorted({t for h, a, _ in rows for t in (h, a)})
        idx = {t: i for i, t in enumerate(teams)}; T = len(teams)
        if T < 5 or len(rows) < 40:
            return self
        X = np.zeros((len(rows), T + 1)); y = np.zeros(len(rows))
        for k, (h, a, m) in enumerate(rows):
            X[k, idx[h]] += 1; X[k, idx[a]] -= 1; X[k, T] = 1.0; y[k] = m
        P = np.full(T + 1, self.ridge); P[T] = 0.0
        beta = np.linalg.solve(X.T @ X + np.diag(P), X.T @ y)
        rr = beta[:T] - beta[:T].mean()
        self.r = {t: float(rr[idx[t]]) for t in teams}; self.home = float(beta[T])
        return self

    def margin(self, h, a):
        return self.r.get(h, 0.0) - self.r.get(a, 0.0) + self.home


def main(min_train=120, refit_every=10, ridge=30.0):
    con = connect(read_only=True)
    rows = con.execute("""SELECT season_end, game_date, home, away, score_home, score_away, home_win
        FROM odds_market WHERE season_end>=2016 ORDER BY season_end, game_date""").fetchdf()
    con.close()
    out = []
    for se, grp in rows.groupby("season_end"):
        grp = grp.sort_values("game_date").reset_index(drop=True)
        hist, srs, since = [], None, 10**9
        y, p = [], []
        for i, g in grp.iterrows():
            if i >= min_train:
                if srs is None or since >= refit_every:
                    srs = MarginSRS(ridge).fit(hist); since = 0
                since += 1
                y.append(int(g.home_win))
                p.append(float(sigmoid(srs.margin(g.home, g.away) / SCALE)))
            hist.append((g.home, g.away, g.score_home - g.score_away))
        if len(y) > 200:
            out.append({"season": f"{se-1}-{str(se)[2:]}", "ours": round(log_loss(np.array(y), p), 4),
                        "n": len(y)})
    json.dump(out, open(str(Path(__file__).resolve().parent.parent /
              "data" / "ours_by_season.json"), "w"))
    for d in out:
        print(d["season"], "ours", d["ours"], f"(n={d['n']})")


if __name__ == "__main__":
    main()
