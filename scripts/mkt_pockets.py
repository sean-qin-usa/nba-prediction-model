"""MKT-POCKETS: firm up the marginal signals from mkt_worst.py's pooled logistic.

For each candidate pocket among market-confident games (q>=0.75):
  P1 fav-tanky:   tsd_fav > +0.5 (the FAVORITE is the tankier/deader side)
  P2 dog-tanky:   tsd_fav < -0.5 (favorite faces a big-tank dog)
  P3 early heavy: either team gp < 10
  P4 late:        both teams gp >= 55
  P5 fav rested:  rest_diff >= +2 days
  P6 dog b2b:     dog on back-to-back
report n, mean q (implied upset rate), empirical upset rate, exact binomial CI,
and the excess in LL terms: what a bettor fading/backing the favorite inside the
pocket at fair close prices would have collected (per-game LL edge of the
empirically-recalibrated q vs the market q -- an upper bound on harvestable).
Read-only DB.
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
Q_CONF = 0.75

df = pd.read_csv(ROOT / "data" / "capstone_pergame_tank.csv", dtype={"game_id": str})
df["game_id"] = df.game_id.str.zfill(10)
df["fav_home"] = df.p_mkt >= 0.5
df["q_mkt"] = np.where(df.fav_home, df.p_mkt, 1 - df.p_mkt)
df["y_fav"] = np.where(df.fav_home, df.y, 1 - df.y).astype(int)
df["tsd_fav"] = np.where(df.fav_home, df.tsd, -df.tsd)
df["fav"] = np.where(df.fav_home, df.home, df.away)
df["dog"] = np.where(df.fav_home, df.away, df.home)

con = duckdb.connect(str(ROOT / "data" / "nba.duckdb"), read_only=True)
tg = con.execute("""
    SELECT season, game_id, team_abbrev, game_date FROM nba_games
    WHERE game_id LIKE '002%'
""").fetchdf()
tg["game_date"] = pd.to_datetime(tg.game_date)
tg = tg.sort_values(["team_abbrev", "game_date"])
g = tg.groupby(["season", "team_abbrev"], sort=False)
tg["gp_before"] = g.cumcount()
tg["gap"] = (tg.game_date - g.game_date.shift(1)).dt.days
for side in ["fav", "dog"]:
    df = df.merge(tg[["season", "game_id", "team_abbrev", "gp_before", "gap"]],
                  left_on=["season", "game_id", side],
                  right_on=["season", "game_id", "team_abbrev"], how="left")
    df = df.rename(columns={"gp_before": f"{side}_gp", "gap": f"{side}_gap"}
                   ).drop(columns="team_abbrev")
df["rest_diff"] = df.fav_gap - df.dog_gap

conf = df[df.q_mkt >= Q_CONF].copy()
POCKETS = [
    ("P1 fav-tanky (tsd_fav>+0.5)",  conf.tsd_fav > 0.5),
    ("P2 dog-tanky (tsd_fav<-0.5)",  conf.tsd_fav < -0.5),
    ("P3 early heavy (min gp<10)",   conf[["fav_gp", "dog_gp"]].min(axis=1) < 10),
    ("P4 late (both gp>=55)",        conf[["fav_gp", "dog_gp"]].min(axis=1) >= 55),
    ("P5 fav rested (rest_diff>=2)", conf.rest_diff >= 2),
    ("P6 dog b2b (dog gap==1)",      conf.dog_gap == 1),
    ("P0 ALL confident",             pd.Series(True, index=conf.index)),
]
print(f"confident set q>={Q_CONF}: n={len(conf)}")
print(f"{'pocket':32s} {'n':>5s} {'impl_up':>8s} {'emp_up':>7s} "
      f"{'ci95':>15s} {'excess':>7s} {'LL_edge/gm':>10s}")
for name, m in POCKETS:
    d = conf[m.values]
    n = len(d)
    ups = int((d.y_fav == 0).sum())
    impl = 1 - d.q_mkt.mean()
    emp = ups / n
    lo, hi = stats.beta.ppf([0.025, 0.975], ups + 0.5, n - ups + 0.5)
    # upper bound on harvest: recalibrate q inside pocket by the empirical
    # excess (shift in logit space fit to match emp), LL edge vs market
    sh = np.log(emp / (1 - emp)) - np.log(impl / (1 - impl))
    qs = 1 / (1 + np.exp(-(np.log(d.q_mkt / (1 - d.q_mkt)) + (-sh))))
    L_m = -np.where(d.y_fav == 1, np.log(d.q_mkt), np.log(1 - d.q_mkt))
    L_s = -np.where(d.y_fav == 1, np.log(qs), np.log(1 - qs))
    print(f"{name:32s} {n:5d} {impl:8.3f} {emp:7.3f} "
          f"[{lo:.3f},{hi:.3f}] {emp-impl:+7.3f} {(L_m-L_s).mean():+10.4f}")

# season stability of the two live-looking pockets
for name, m in POCKETS[:3]:
    d = conf[m.values]
    parts = []
    for sn, dd in d.groupby("season"):
        parts.append(f"{sn}: {int((dd.y_fav==0).sum())}/{len(dd)} "
                     f"({(dd.y_fav==0).mean():.3f} vs impl {1-dd.q_mkt.mean():.3f})")
    print(f"  {name}: " + " | ".join(parts))
