"""REGIME D follow-up 4: LATE OUTS-MAGNITUDE term.

R3 eyeball (rw_regimeD_final.py) showed the April market-heavy losses are
final-week intent games with extreme out-counts (5-8) where our margin
under-moves despite oracle outs being IN the capstone inputs (D51: composition
channel flat-blended). Test: does a late-gated linear out-count-differential
margin term recover it?

  m' = m_us + c * (n_out_home - n_out_away) on gates:
    G1 active window (tsd!=0), G2 April only, G3 final 7 calendar days,
    G4 active & outdiff extreme (|diff|>=3).
  Report residual slope first (theory direction: negative — more outs than
  opponent => we OVER-predict that side => resid slope negative), then
  hindsight c* + paired delta vs shipped, per season.

PIT: n_out_* are the capstone's own oracle-tier inputs. DuckDB not needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CSV = "/hdd/steveqin/sean_dev/nba_model/data/capstone_pergame_tank.csv"
OM_DB = "/hdd/steveqin/sean_dev/nba_model/data/nba.duckdb"
SCALE = 7.2
RNG = np.random.default_rng(46)


def ll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def boot_ci(x, B=4000):
    x = np.asarray(x, float)
    idx = RNG.integers(0, len(x), (B, len(x)))
    return tuple(np.percentile(x[idx].mean(axis=1), [2.5, 97.5]))


def slope_ci(x, r):
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, r, rcond=None)
    e = r - X @ beta
    se = np.sqrt((e @ e) / (len(x) - 2) * np.linalg.inv(X.T @ X)[1, 1])
    return beta[1], 1.96 * se


def main():
    import duckdb
    con = duckdb.connect(OM_DB, read_only=True)
    om = con.execute("""SELECT game_date, home, away, score_home, score_away
        FROM odds_market WHERE season_end >= 2024""").df()
    con.close()
    om["game_date"] = om.game_date.astype(str)

    df = pd.read_csv(CSV, dtype={"game_id": str})
    df = df.merge(om, on=["game_date", "home", "away"], how="left")
    df["game_date"] = pd.to_datetime(df.game_date)
    df["month"] = df.game_date.dt.month
    df["m_us"] = SCALE * logit(df.p_us)
    df["margin_home"] = df.score_home - df.score_away
    df["resid_post"] = df.margin_home - df.m_us
    df["d"] = ll(df.p_us.values, df.y.values) - ll(df.p_mkt.values, df.y.values)
    df["outdiff"] = df.n_out_home - df.n_out_away
    # final 7 calendar days per season
    last = df.groupby("season").game_date.transform("max")
    df["final7"] = (last - df.game_date).dt.days < 7
    yv = df.y.values
    act = (df.tsd != 0).values

    gates = {
        "G1 active window": act,
        "G2 April active": act & (df.month == 4).values,
        "G3 final 7 days": df.final7.values,
        "G4 active & |outdiff|>=3": act & (df.outdiff.abs() >= 3).values,
        "G0 ALL games (control)": np.ones(len(df), bool),
    }
    for nm, m in gates.items():
        g = df[m]
        b, w = slope_ci(g.outdiff.values, g.resid_post.values)
        cs = np.linspace(-2, 2, 161)
        best, bc = 1e18, 0.0
        for c in cs:
            p = sigmoid((df.m_us.values
                         + np.where(m, c * df.outdiff.values, 0.0)) / SCALE)
            L = ll(p, yv)[m].sum()
            if L < best:
                best, bc = L, c
        p = sigmoid((df.m_us.values
                     + np.where(m, bc * df.outdiff.values, 0.0)) / SCALE)
        delta = ll(p, yv) - ll(df.p_us.values, yv)
        lo, hi = boot_ci(delta[m])
        by = " ".join(f"{s[-5:]}:{delta[m & (df.season == s).values].mean():+.4f}"
                      for s in sorted(df.season.unique()))
        print(f"{nm:26s} n={int(m.sum()):4d}  resid~outdiff {b:+.3f}+-{w:.3f}"
              f"  c*={bc:+.2f}  vs shipped {delta[m].mean():+.5f} "
              f"CI({lo:+.5f},{hi:+.5f}) nats {delta[m].sum():+5.1f}  {by}")


if __name__ == "__main__":
    main()
