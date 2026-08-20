"""IG probes D+E on capstone_pergame_carry2.csv (the shipped headline run).

D) Weekly-refit staleness: replay the exact refit schedule of prod_by_season.py
   (refit when (gd-last)>=7 over the full meta order, BEFORE the market filter),
   tag each scored game with days-since-refit, bucket log loss.
E) SCALE=7.2: recover margin = SCALE*logit(p_us); fit logistic (a + b*m) with
   leave-one-season-out; also carry-era vs post-carry era slopes. Reports the
   honest log-loss delta of an estimated scale vs the hardcoded one.
Read-only.
"""
import sys, warnings, datetime as dt
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from nbapred.db import connect

CSV = "/hdd/steveqin/sean_dev/nba_model/data/capstone_pergame_carry2.csv"

def refit_schedule(con, season):
    meta = con.execute("""SELECT game_id, team_abbrev, matchup, game_date FROM nba_games
        WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL ORDER BY game_date""",
        [season]).fetchdf()
    by, order = {}, []
    for x in meta.itertuples():
        if x.game_id not in by:
            order.append(x.game_id)
        by.setdefault(x.game_id, []).append(x)
    last = None
    stale = {}
    for gid in order:
        recs = by[gid]
        if len(recs) != 2:
            continue
        m = recs[0].matchup
        host = m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        h = next((x for x in recs if x.team_abbrev == host), None)
        a = next((x for x in recs if x.team_abbrev != host), None)
        if not h or not a:
            continue
        gd = h.game_date.date() if hasattr(h.game_date, "date") else h.game_date
        if last is None or (gd - last).days >= 7:
            last = gd
        stale[gid] = (gd - last).days
    return stale

def ll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def fit_logistic(m, y, l2=1e-6):
    # Newton, 2 params
    b = np.array([0.0, 1 / 7.2])
    X = np.c_[np.ones(len(m)), m]
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ b))
        g = X.T @ (p - y) + l2 * b
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + l2 * np.eye(2)
        step = np.linalg.solve(H, g)
        b -= step
        if np.abs(step).max() < 1e-10:
            break
    return b

def main():
    df = pd.read_csv(CSV, parse_dates=["game_date"])
    df["gid"] = df.game_id.apply(lambda g: f"00{g}" if len(str(g)) == 8 else str(g))
    con = connect(read_only=True)
    stmap = {}
    for s in df.season.unique():
        stmap.update(refit_schedule(con, s))
    con.close()
    df["stale"] = df.gid.map(stmap)
    print("unmatched staleness:", df.stale.isna().sum())
    df["ll"] = ll(df.y.values, df.p_us.values)
    df["llm"] = ll(df.y.values, df.p_mkt.values)
    print("\n== D: log loss by days-since-refit (pooled 3 seasons) ==")
    g = df.groupby("stale").agg(n=("ll", "size"), ll=("ll", "mean"), ll_mkt=("llm", "mean"))
    print(g.round(4))
    # regression slope of ll on staleness (with season FE crudely via demean)
    d2 = df.dropna(subset=["stale"]).copy()
    d2["ll_dm"] = d2.ll - d2.groupby("season").ll.transform("mean")
    A = np.c_[np.ones(len(d2)), d2.stale.values]
    coef = np.linalg.lstsq(A, d2.ll_dm.values, rcond=None)[0]
    # bootstrap the slope
    rng = np.random.default_rng(0)
    sl = []
    for _ in range(2000):
        i = rng.integers(0, len(d2), len(d2))
        sl.append(np.linalg.lstsq(A[i], d2.ll_dm.values[i], rcond=None)[0][1])
    lo, hi = np.percentile(sl, [2.5, 97.5])
    print(f" slope per staleness-day: {coef[1]:+.5f}  CI95 ({lo:+.5f},{hi:+.5f})"
          f"  mean staleness {d2.stale.mean():.2f}")
    print(f" implied daily-refit saving upper bound ~ {coef[1] * d2.stale.mean():+.5f}/game")

    print("\n== E: SCALE calibration ==")
    df["margin"] = 7.2 * np.log(df.p_us / (1 - df.p_us))
    for s in df.season.unique():
        tr = df[df.season != s]; te = df[df.season == s]
        b = fit_logistic(tr.margin.values, tr.y.values)
        p_new = 1 / (1 + np.exp(-(b[0] + b[1] * te.margin.values)))
        d = ll(te.y.values, te.p_us.values).mean() - ll(te.y.values, p_new).mean()
        print(f" LOSO {s}: fitted a={b[0]:+.4f} b={b[1]:.4f} (scale {1/b[1]:.2f} vs 7.2)"
              f"  delta_ll vs shipped {d:+.5f} ({'fitted better' if d > 0 else 'shipped better'})")
    # era split: carry era = before Nov 10 of season start year
    df["early"] = df.game_date.dt.month.isin([10]) | ((df.game_date.dt.month == 11) & (df.game_date.dt.day <= 8))
    for era, sub in df.groupby("early"):
        b = fit_logistic(sub.margin.values, sub.y.values)
        print(f" era early={era}: n={len(sub)} fitted scale {1/b[1]:.2f} a={b[0]:+.3f}"
              f"  in-sample delta {ll(sub.y.values, sub.p_us.values).mean() - ll(sub.y.values, 1/(1+np.exp(-(b[0]+b[1]*sub.margin.values)))).mean():+.5f}")
    # margin sd by era (is early-margin overdispersed?)
    print(" margin sd early:", round(df[df.early].margin.std(), 2),
          " late:", round(df[~df.early].margin.std(), 2))

if __name__ == "__main__":
    main()
