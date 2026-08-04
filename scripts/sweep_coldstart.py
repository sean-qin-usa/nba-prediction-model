"""Play with cold-start: how much to regress last season's ratings (roster
turnover), and how long the prior should carry. Sweeps the regression factor and
reports EARLY-SEASON log loss (each team's first N games) — the target metric.
Hypothesis (not blind grid): ~30% roster turnover/yr => regress ~0.6-0.75 best;
too high (1.0) ignores turnover, too low (0.3) throws away real info.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from nbapred.db import connect
from nbapred.eval.metrics import log_loss
from scripts.eval_coldstart import MarginSRS, season_end_ratings

SCALE = 7.2
sig = lambda x: 1/(1+np.exp(-np.asarray(x)))


def run(df, regress, early_n=15, ridge=30.0, refit_every=10):
    prev_end = {}
    Y, P = [], []
    for se, grp in df.groupby("season_end"):
        grp = grp.sort_values("game_date").reset_index(drop=True)
        prior = {t: regress*v for t, v in prev_end.items()}
        hist, warm, since, tg = [], None, 10**9, {}
        for i, g in grp.iterrows():
            if since >= refit_every:
                warm = MarginSRS(ridge, prior=prior).fit(hist); since = 0
            since += 1
            if prev_end and max(tg.get(g.home,0), tg.get(g.away,0)) < early_n:
                Y.append(int(g.home_win))
                m = warm.margin(g.home,g.away) if warm else (prior.get(g.home,0)-prior.get(g.away,0)+3)
                P.append(float(sig(m/SCALE)))
            hist.append((g.home,g.away,g.score_home-g.score_away))
            tg[g.home]=tg.get(g.home,0)+1; tg[g.away]=tg.get(g.away,0)+1
        prev_end = season_end_ratings(hist, ridge)
    return log_loss(np.array(Y), P), len(Y)


def main():
    con = connect(read_only=True)
    df = con.execute("""SELECT season_end, game_date, home, away, score_home, score_away, home_win
        FROM odds_market WHERE season_end>=2016 ORDER BY season_end, game_date""").fetchdf()
    con.close()
    print("early-season (first 15/team) log loss by regression factor:")
    print("(cold-start baseline was 0.6686)")
    for reg in [0.0, 0.4, 0.55, 0.65, 0.75, 0.85, 1.0]:
        ll, n = run(df, reg)
        tag = " <- ignores turnover" if reg == 1.0 else (" <- no carryover(=cold)" if reg == 0.0 else "")
        print(f"  regress={reg:.2f}: {ll:.4f}  (n={n}){tag}")


if __name__ == "__main__":
    main()
