"""Does the Kalman form filter beat EWMA at one-step-ahead scoring prediction?
For each rotation player, walk their games chronologically; at each game predict
per-minute scoring rate from PRIOR games only, three ways: season-mean, EWMA
(current approach), Kalman filter. Score by weighted MAE of predicted points
(rate*minutes) vs actual. Kalman should win where form changes / time gaps.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect
from nbapred.model.form_filter import FormFilter


def main(min_games=20, min_min=15):
    con = connect(read_only=True)
    df = con.execute("""SELECT s.player_id, g.game_date, s.seconds, s.pts
        FROM player_game_stats s JOIN (SELECT DISTINCT game_id,game_date FROM nba_games) g USING(game_id)
        WHERE s.game_id LIKE '002%' AND s.seconds > 0 ORDER BY s.player_id, g.game_date""").fetchdf()
    con.close()

    err = {"mean": [], "ewma": [], "kalman": []}
    for pid, grp in df.groupby("player_id"):
        grp = grp[grp.seconds > min_min * 60]
        if len(grp) < min_games:
            continue
        mins = grp.seconds.to_numpy() / 60.0
        rate = grp.pts.to_numpy() / mins
        dates = np.array([d.toordinal() for d in grp.game_date])
        pmean = rate.mean()

        f = FormFilter(pmean)
        ewma = pmean
        last = None
        burn = 8  # warm-up before scoring
        for i in range(len(grp)):
            dt = 0.0 if last is None else dates[i] - last
            k_pred, _ = f.predict(dt)
            if i >= burn:
                actual_pts = rate[i] * mins[i]
                err["mean"].append(abs(pmean * mins[i] - actual_pts))
                err["ewma"].append(abs(ewma * mins[i] - actual_pts))
                err["kalman"].append(abs(k_pred * mins[i] - actual_pts))
            f.update(rate[i], mins[i])
            ewma = 0.85 * ewma + 0.15 * rate[i]      # ~10-game half-life EWMA
            last = dates[i]

    n = len(err["kalman"])
    print(f"scored predictions: {n}")
    for k in ("mean", "ewma", "kalman"):
        print(f"  {k:7} MAE (points): {np.mean(err[k]):.4f}")
    d = np.array(err["ewma"]) - np.array(err["kalman"])
    rng = np.random.default_rng(0)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\nKalman vs EWMA MAE improvement: {d.mean():+.4f}  95% CI ({lo:+.4f},{hi:+.4f})  "
          f"-> {'KEEP' if lo > 0 else 'not significant'}")


if __name__ == "__main__":
    main()
