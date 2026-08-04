"""Sean's cold-start idea: carry last season's ratings (regressed toward mean
for roster turnover) as the prior, so early-season games aren't predicted from
scratch. Tests whether this fixes the early-season gap — the #1 reason our
full-walk-forward trails the market. Uses scores-only (all seasons available).
"""
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
    def __init__(self, ridge=30.0, prior=None):
        self.ridge = ridge; self.prior = prior or {}; self.r = {}; self.home = 3.0

    def fit(self, rows):
        teams = sorted({t for h, a, _ in rows for t in (h, a)})
        idx = {t: i for i, t in enumerate(teams)}; T = len(teams)
        if T < 5 or len(rows) < 10:
            self.r = dict(self.prior); return self
        X = np.zeros((len(rows), T + 1)); y = np.zeros(len(rows))
        for k, (h, a, m) in enumerate(rows):
            X[k, idx[h]] += 1; X[k, idx[a]] -= 1; X[k, T] = 1; y[k] = m
        p = np.array([self.prior.get(t, 0.0) for t in teams] + [3.0])
        yr = y - X @ p
        P = np.full(T + 1, self.ridge); P[T] = 0.0
        beta = p + np.linalg.solve(X.T @ X + np.diag(P), X.T @ yr)
        rr = beta[:T] - beta[:T].mean()
        self.r = {t: float(rr[idx[t]]) for t in teams}; self.home = float(beta[T])
        return self

    def margin(self, h, a):
        return self.r.get(h, self.prior.get(h, 0.0)) - self.r.get(a, self.prior.get(a, 0.0)) + self.home


def season_end_ratings(rows, ridge=30.0):
    return MarginSRS(ridge).fit(rows).r


def main(regress=0.70, early_n=15, refit_every=10, ridge=30.0, min_train_cold=120):
    con = connect(read_only=True)
    df = con.execute("""SELECT season_end, game_date, home, away, score_home, score_away, home_win
        FROM odds_market WHERE season_end>=2016 ORDER BY season_end, game_date""").fetchdf()
    con.close()

    prev_end = {}
    res = {"cold_all": ([], []), "warm_all": ([], []),
           "cold_early": ([], []), "warm_early": ([], [])}
    for se, grp in df.groupby("season_end"):
        grp = grp.sort_values("game_date").reset_index(drop=True)
        prior = {t: regress * v for t, v in prev_end.items()}   # regress to mean
        team_games = {}
        hist, cold, warm, since = [], None, None, 10**9
        for i, g in grp.iterrows():
            gc = team_games.get(g.home, 0) + team_games.get(g.away, 0)
            if since >= refit_every:
                cold = MarginSRS(ridge).fit(hist)                       # from scratch
                warm = MarginSRS(ridge, prior=prior).fit(hist)          # last-season prior
                since = 0
            since += 1
            y = int(g.home_win)
            # cold model only scores once it has its usual warm-up; warm scores from game 1
            if i >= min_train_cold:
                res["cold_all"][0].append(y); res["cold_all"][1].append(sigmoid(cold.margin(g.home, g.away)/SCALE))
            res["warm_all"][0].append(y); res["warm_all"][1].append(sigmoid((warm.margin(g.home, g.away) if warm else prior_margin(prior, g))/SCALE))
            # early-season subset (each team's first `early_n` games)
            if max(team_games.get(g.home, 0), team_games.get(g.away, 0)) < early_n and prev_end:
                res["warm_early"][0].append(y); res["warm_early"][1].append(sigmoid((warm.margin(g.home, g.away) if warm else 0)/SCALE))
                cm = cold.margin(g.home, g.away) if cold else 0.0
                res["cold_early"][0].append(y); res["cold_early"][1].append(sigmoid(cm/SCALE))
            hist.append((g.home, g.away, g.score_home - g.score_away))
            team_games[g.home] = team_games.get(g.home, 0)+1; team_games[g.away] = team_games.get(g.away, 0)+1
        prev_end = season_end_ratings(hist, ridge)

    def ll(k):
        y, p = res[k]
        return log_loss(np.array(y), p) if len(y) > 50 else None
    print(f"cold-start (from scratch)      all-season: {ll('cold_all'):.4f}")
    print(f"WARM-START (last-season prior)  all-season: {ll('warm_all'):.4f}")
    print(f"\nEARLY-SEASON games (each team's first {early_n}):")
    print(f"  cold-start: {ll('cold_early'):.4f}")
    print(f"  WARM-START: {ll('warm_early'):.4f}   <- the fix for early-season")


def prior_margin(prior, g):
    return prior.get(g.home, 0.0) - prior.get(g.away, 0.0) + 3.0


if __name__ == "__main__":
    main()
