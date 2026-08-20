"""Audit: does ignoring clustering flip the player-level (props) gates?

Reconstructs the D33-style star-out attempts universe (same construction as
scripts/gate_conditional_redis.py) and scores the uniform softmax lift vs
no-lift on per-row Poisson LL. Then compares:
  * iid row bootstrap        (what every props gate shipped with)
  * cluster by star-out team-game event (teammates share the same absence + game)
  * cluster by player        (repeated player-games)
Read-only DB. No production files touched.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from nbapred.db import connect

con = connect(read_only=True)
pg = con.execute("""SELECT s.game_id, s.player_id, s.team_id, g.game_date,
    s.seconds/60.0 mins, s.rima+s.mida+s.thra shots
    FROM player_game_stats s JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING(game_id)
    WHERE s.game_id LIKE '002%' ORDER BY g.game_date""").fetchdf()
con.close()
uz = np.load("data/v2_usage.npz")
u = dict(zip(uz["player_ids"].tolist(), uz["u"].tolist()))
pg = pg.sort_values(["player_id", "game_date"])
pg["avg_min"] = pg.groupby("player_id")["mins"].transform(
    lambda s: s.shift(1).rolling(10, min_periods=5).mean())
pg["avg_shots"] = pg.groupby("player_id")["shots"].transform(
    lambda s: s.shift(1).rolling(10, min_periods=5).mean())
played = pg[pg.mins >= 8].groupby(["game_id", "team_id"])["player_id"].apply(set)
stars = pg[pg.avg_min >= 28.0]
sbt = {}
for r in stars[["player_id", "team_id", "game_date"]].itertuples():
    sbt.setdefault(r.team_id, []).append((r.game_date, r.player_id))
rot = pg[(pg.avg_min >= 15) & (pg.mins >= 12) & pg.avg_shots.notna()].copy()

rows = []
for r in rot.itertuples():
    recent = {p for (d0, p) in sbt.get(r.team_id, []) if 0 < (r.game_date - d0).days <= 12}
    outs = (recent - played.get((r.game_id, r.team_id), set())) - {r.player_id}
    if not outs:
        continue
    star = max(outs, key=lambda p: u.get(p, 0.0))
    pool = {int(p) for p in rot[(rot.team_id == r.team_id) &
                                (rot.game_date == r.game_date)].player_id} | {star}
    S = sum(np.exp(u.get(p, 0.0)) for p in pool)
    L = float(min(S / max(S - np.exp(u.get(star, 0.0)), 1e-9), 1.5))
    rows.append((r.game_id, int(r.team_id), int(r.player_id), r.avg_shots, r.shots, L))
df = pd.DataFrame(rows, columns=["game_id", "team_id", "player_id", "a", "y", "L"])
print(f"star-out player-games n={len(df)} "
      f"(D33 registered n=14,581); events={df.groupby(['game_id','team_id']).ngroups}, "
      f"players={df.player_id.nunique()}")
print(f"rows per event: mean {len(df)/df.groupby(['game_id','team_id']).ngroups:.2f}")

def pois_ll(pred, y):
    pred = np.clip(pred, 0.2, None)
    return y * np.log(pred) - pred

d = pois_ll(df.a.values * df.L.values, df.y.values) - pois_ll(df.a.values, df.y.values)
n = len(d)
rng = np.random.default_rng(0)

def iid_ci(x, B=4000):
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(axis=1)
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5), bs.std()

def cluster_ci(x, labels, B=4000):
    lab = pd.Series(labels)
    sums = pd.Series(x).groupby(lab.values).sum().values
    cnts = pd.Series(x).groupby(lab.values).count().values
    K = len(sums)
    bs = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, K, K)
        bs[b] = sums[pick].sum() / cnts[pick].sum()
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5), bs.std()

lo, hi, se0 = iid_ci(d)
print(f"mean delta {d.mean():+.5f} (D33 registered +0.044 — same sign/scale expected)")
print(f"  iid rows           : CI({lo:+.5f},{hi:+.5f}) SE {se0:.5f}  {'PASS' if lo>0 else 'FAIL'}")
ev = (df.game_id.astype(str) + "_" + df.team_id.astype(str)).values
for name, lab in (("team-game event", ev), ("player", df.player_id.values),
                  ("game", df.game_id.values)):
    lo, hi, se = cluster_ci(d, lab)
    print(f"  cluster {name:11s}: CI({lo:+.5f},{hi:+.5f}) SE {se:.5f}  "
          f"SE-inflation x{se/se0:4.2f}  {'PASS' if lo>0 else 'FAIL'}")
# within-event ICC of the delta
g = pd.DataFrame({"d": d, "ev": ev}).groupby("ev")["d"]
wi = g.apply(lambda s: s.var(ddof=1) if len(s) > 1 else np.nan).mean()
tot = d.var(ddof=1)
print(f"  within-event var {wi:.5f} vs total var {tot:.5f} -> rough ICC {(tot-wi)/tot:.2f}")
print("AUDIT_CLUSTER_PROPS_DONE")
