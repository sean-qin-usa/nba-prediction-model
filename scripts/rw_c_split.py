#!/usr/bin/env python3
"""REGIME B refinement (read-only): split class-c (traded-away) eval rows into
PRE-TRADE holdout games (star already sitting, not yet traded — newcomers not
yet arrived) vs POST-TRADE games (roster delta realized). Sharp test of
whether the benched-style lift survives into the true permanent regime.
Reads the pickle from rw_star_transitions.py; re-derives game dates from DB.
"""
import sys
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nbapred.db import connect  # noqa: E402

SCRATCH = Path("data/scratch")

df = pd.read_pickle(SCRATCH / "rw_star_transitions_rows.pkl")
c = df[(df.cls == "c") & (~df.other_star_out)].copy()

con = connect(read_only=True)
pg = con.execute("""
    SELECT s.game_id, s.player_id, s.team_id, g.season, g.game_date,
           s.seconds/60.0 AS mins
    FROM player_game_stats s
    JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g USING (game_id)
    WHERE s.game_id LIKE '002%' ORDER BY g.game_date""").fetchdf()
tg = con.execute("""
    SELECT DISTINCT season, game_id, game_date, team_id FROM nba_games
    WHERE game_id LIKE '002%' ORDER BY game_date""").fetchdf()
names = dict(con.execute("SELECT player_id, full_name FROM nba_players").fetchall())
con.close()
pg["game_date"] = pd.to_datetime(pg["game_date"])
tg["game_date"] = pd.to_datetime(tg["game_date"])
played = pg[pg.mins >= 8.0]

# star's first played game for a NEW team after leaving old team, per season
sched = {}
for (season, team), g in tg.groupby(["season", "team_id"]):
    g = g.sort_values("game_date")
    sched[(season, team)] = list(zip(g.game_id, g.game_date))

# map each (eid) -> event k=1 date, and star's first new-team date
ev = c.groupby("eid").agg(star=("star", "first"), team=("team", "first"),
                          season=("season", "first"),
                          run_len=("run_len", "first")).reset_index()
pdate = defaultdict(list)
for r in played.itertuples():
    pdate[r.player_id].append((r.game_date, r.team_id))

rows = []
for e in ev.itertuples():
    sc = sched[(e.season, e.team)]
    # star's last played game for team
    star_dates = [d for (d, t) in pdate[e.star] if t == e.team
                  and sc[0][1] <= d <= sc[-1][1]]
    last = max(star_dates)
    trade_dt = min((d for (d, t) in pdate[e.star]
                    if d > last and t != e.team), default=None)
    post_games = [(k + 1, gid, gd) for k, (gid, gd) in
                  enumerate((g for g in sc if g[1] > last))]
    for (k, gid, gd) in post_games:
        rows.append(dict(eid=e.eid, k=k, phase="post" if trade_dt and gd >= trade_dt
                         else "pre", star=names.get(e.star, e.star)))
km = pd.DataFrame(rows)
c = c.merge(km[["eid", "k", "phase"]], on=["eid", "k"], how="left")
print("class-c eval rows by phase:", c.phase.value_counts().to_dict())
print("example traded stars:",
      sorted({names.get(s, s) for s in ev.star})[:25])

rng = np.random.default_rng(9)
def eboot(sub, col, iters=1500):
    g = sub.groupby("eid")[col].mean()
    if len(g) < 3:
        return (np.nan, np.nan, np.nan, 0)
    v = g.values
    bs = [np.mean(rng.choice(v, len(v))) for _ in range(iters)]
    return (float(np.mean(v)), *np.percentile(bs, [2.5, 97.5]), len(g))

p12 = c[c.mins >= 12].copy()
p12["att"] = p12.fga / p12.base_fga.clip(lower=0.5)
p12["roll"] = p12.fga / p12.roll_fga.clip(lower=0.5)
p1 = c[c.mins >= 1].copy()
p1["dmin"] = p1.mins - p1.base_min
for phase in ["pre", "post"]:
    for lo, hi in [(1, 3), (4, 7), (8, 15), (16, 30)]:
        m = p12[(p12.phase == phase) & (p12.k >= lo) & (p12.k <= hi)]
        m1 = p1[(p1.phase == phase) & (p1.k >= lo) & (p1.k <= hi)]
        a = eboot(m, "att"); r = eboot(m, "roll"); dm = eboot(m1, "dmin")
        nc = eboot(m.drop_duplicates(["eid", "k"]), "newcomer_share")
        print(f"{phase:4} k={lo}-{hi:>2} n={len(m):>4} ev={a[3]:>2} "
              f"att {a[0]:+.3f}[{a[1]:+.3f},{a[2]:+.3f}] "
              f"roll {r[0]:+.3f}[{r[1]:+.3f},{r[2]:+.3f}] "
              f"dmin {dm[0]:+.2f}[{dm[1]:+.2f},{dm[2]:+.2f}] "
              f"newc {nc[0]:+.3f}")
