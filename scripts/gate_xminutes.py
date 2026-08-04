#!/usr/bin/env python3
"""Gate: DARKO x_minutes vs trailing-minutes as the props minutes projector.

D12 established minutes as the prop-error bottleneck (11.2% oracle share);
every internal challenger (EWMA, empirical, Kalman-in-props) failed the gate.
x_minutes is DARKO's own daily minutes projection (now PIT via darko_history,
D43) — an EXTERNAL projector that already blends injury news, role change,
and blowout ecology. H: x_minutes (or a blend) beats trailing-20 on held-out
player-game minutes.

Universe matches props conditioning: 2025-26 regular season, played >= 10 min.
PIT: x_minutes as-of strictly before game date. Metric: MAE, paired bootstrap
CI on the per-observation |err| difference.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nbapred.db import connect  # noqa: E402

RNG = np.random.default_rng(11)


def main() -> None:
    con = connect(read_only=True)
    pg = con.execute("""
        WITH pg AS (
          SELECT s.player_id, s.game_id, g.game_date, s.seconds/60.0 AS mins
          FROM player_game_stats s
          JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g USING (game_id)
          WHERE g.season = '2025-26' AND s.game_id LIKE '002%' AND s.seconds >= 600
        )
        SELECT p.*,
               (SELECT avg(q.mins) FROM (
                    SELECT mins FROM pg q2
                    WHERE q2.player_id = p.player_id AND q2.game_date < p.game_date
                    ORDER BY q2.game_date DESC LIMIT 20) q)      AS trail20,
               (SELECT count(*) FROM pg q3
                 WHERE q3.player_id = p.player_id AND q3.game_date < p.game_date) AS n_prior,
               (SELECT h.x_minutes FROM darko_history h
                 WHERE h.player_id = p.player_id AND h.date < p.game_date
                 ORDER BY h.date DESC LIMIT 1)                   AS xmin
        FROM pg p
    """).fetchdf()
    con.close()

    df = pg.dropna(subset=["trail20", "xmin"])
    df = df[df.n_prior >= 5]
    y = df.mins.values
    e_tr = np.abs(y - df.trail20.values)
    e_xm = np.abs(y - df.xmin.values)
    e_bl = np.abs(y - 0.5 * (df.trail20.values + df.xmin.values))
    n = len(df)
    print(f"n={n}  MAE trail20 {e_tr.mean():.3f} | x_minutes {e_xm.mean():.3f} "
          f"| 50/50 blend {e_bl.mean():.3f}")

    for name, e in (("x_minutes", e_xm), ("blend", e_bl)):
        d = e_tr - e            # + = challenger better
        bs = [d[RNG.integers(0, n, n)].mean() for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"{name}: delta-MAE {d.mean():+.4f} CI[{lo:+.4f},{hi:+.4f}]  "
              f"{'PASS' if lo > 0 else 'FAIL'}")


if __name__ == "__main__":
    main()
