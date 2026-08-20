"""Two prop features through the gate (aligned >=720 eval):
  A) REST: players on zero days rest (B2B) — empirical efficiency check, then a
     fixed 0.97 scoring-rate multiplier on B2B games (prior-based, not fitted).
  B) OVERDISPERSION: per-sim lognormal multiplier (sigma=0.10) on shot rates —
     'hot/cold game' variance beyond binomial. PIT std 0.302>0.289 says points
     tails are slightly under-covered; this widens them.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect
from nbapred.engine.props import player_rates_from_stats, simulate_player


def crps(s, y):
    s = np.sort(s); n = len(s)
    return float(np.mean(np.abs(s - y)) - 0.5 * (2 * np.arange(1, n + 1) - n - 1) @ s / n**2)


def sim_overdisp(rates, n, seed, sigma=0.10):
    """simulate_player with a shared per-sim rate multiplier (game heat)."""
    rng = np.random.default_rng(seed)
    mult = np.exp(rng.normal(-sigma**2 / 2, sigma))  # mean-1 lognormal, scalar per batch...
    # need per-sim multipliers: run in K batches for tractability
    K = 8
    outs = []
    for k in range(K):
        m = float(np.exp(rng.normal(-sigma**2 / 2, sigma)))
        r2 = dict(rates)
        for key in ("rate_rim", "rate_mid", "rate_thr", "fta_per_min"):
            r2[key] = rates[key] * m
        outs.append(simulate_player(r2, n // K, seed=seed * K + k)["points"])
    return np.concatenate(outs)


def main(sims=3200, max_eval=1200):
    con = connect(read_only=True)
    pg = con.execute("""SELECT s.player_id, g.game_date, g.team_abbrev, s.pts, s.seconds,
        sf.days_rest
        FROM player_game_stats s
        JOIN nba_games g ON g.game_id=s.game_id AND g.team_id=s.team_id
        LEFT JOIN schedule_features sf ON sf.game_id=s.game_id AND sf.team=g.team_abbrev
        WHERE s.game_id LIKE '002%' AND s.seconds>=720 ORDER BY g.game_date""").fetchdf()

    # A0: empirical — points per minute on B2B vs rested
    ppm = pg.pts / (pg.seconds / 60.0)
    b2b = pg.days_rest == 1
    rested = pg.days_rest >= 2
    print(f"EMPIRICAL: pts/min on B2B {ppm[b2b].mean():.4f} (n={b2b.sum()}) "
          f"vs rested {ppm[rested].mean():.4f} (n={rested.sum()}) "
          f"ratio {ppm[b2b].mean()/ppm[rested].mean():.4f}")

    stride = max(1, len(pg) // (max_eval * 3))
    pg = pg.iloc[::stride]
    base_c, rest_c, od_c = [], [], []
    base_pit, od_pit = [], []
    n = 0
    for r in pg.itertuples():
        if n >= max_eval:
            break
        rates = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
        if not rates or rates["n_games"] < 8 or rates["proj_min"] < 20:
            continue
        y = r.pts
        rn = dict(rates); rn.pop("minutes_hist", None)
        sb = simulate_player(rn, sims, seed=n)["points"]
        base_c.append(crps(sb, y)); base_pit.append(float(np.mean(sb < y) + 0.5 * np.mean(sb == y)))
        # A: rest multiplier
        rr = dict(rn)
        import pandas as pd
        if pd.notna(r.days_rest) and r.days_rest == 1:
            for key in ("rate_rim", "rate_mid", "rate_thr"):
                rr[key] = rn[key] * 0.97
        rest_c.append(crps(simulate_player(rr, sims, seed=n)["points"], y))
        # B: overdispersion
        so = sim_overdisp(rn, sims, seed=n + 1)
        od_c.append(crps(so, y)); od_pit.append(float(np.mean(so < y) + 0.5 * np.mean(so == y)))
        n += 1
    con.close()

    def gate(name, d):
        d = np.array(d); rng = np.random.default_rng(0)
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {name}: delta {d.mean():+.4f}  CI ({lo:+.4f},{hi:+.4f}) "
              f"-> {'KEEP' if lo > 0 else 'no'}")

    print(f"\nn={n}  CRPS base {np.mean(base_c):.4f}")
    print(f"PIT base mean {np.mean(base_pit):.3f} std {np.std(base_pit):.3f} (want .500/.289)")
    gate("rest(B2B x0.97)", np.array(base_c) - np.array(rest_c))
    gate("overdispersion ", np.array(base_c) - np.array(od_c))
    print(f"PIT overdisp mean {np.mean(od_pit):.3f} std {np.std(od_pit):.3f}")


if __name__ == "__main__":
    main()
