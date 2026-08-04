"""IG probe B2: log-loss cost of the ROSTER_DAYS=12 exclusion of players who
actually PLAY, split (i) season-opener cluster (everyone stale at opener) vs
(ii) mid-season injury returns (gap 13-90d within season).

Counterfactual: add back 0.5 * (net signed excluded talent x trail_min/48) to
the recovered margin, rescore. Hindsight-flavored (we know they played), but a
live fix has the injury feed's available/probable status. Read-only.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from nbapred.db import connect
from nbapred.model.production import SCALE

Q = """
WITH q AS (
  SELECT s.player_id, g.game_date, s.seconds/60.0 m
  FROM player_game_stats s
  JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
  WHERE s.game_id LIKE '002%' AND s.seconds >= 720
),
qroll AS (
  SELECT player_id, game_date,
         avg(m) OVER (PARTITION BY player_id ORDER BY game_date
                      ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) tm10
  FROM q
),
sg AS (SELECT DISTINCT game_id, game_date FROM nba_games
       WHERE season = ? AND game_id LIKE '002%' AND wl IS NOT NULL),
played AS (
  SELECT s.player_id, s.team_id, s.game_id, sg.game_date
  FROM player_game_stats s JOIN sg USING (game_id) WHERE s.seconds > 0
)
SELECT p.player_id, p.team_id, p.game_id, p.game_date, qr.game_date last_q, qr.tm10,
       (p.game_date - qr.game_date) gap
FROM played p ASOF JOIN qroll qr
  ON qr.player_id = p.player_id AND qr.game_date < p.game_date
WHERE (p.game_date - qr.game_date) > 12
"""

def ll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def main():
    con = connect(read_only=True)
    darko = dict(con.execute("""SELECT player_id, dpm FROM (
        SELECT player_id, dpm, row_number() OVER (PARTITION BY player_id ORDER BY date DESC) rn
        FROM darko_history) WHERE rn=1""").fetchall())
    cap = pd.read_csv("/hdd/steveqin/sean_dev/nba_model/data/capstone_pergame_carry2.csv",
                      parse_dates=["game_date"])
    cap["gid"] = cap.game_id.apply(lambda g: f"00{g}" if len(str(g)) == 8 else str(g))
    # game -> home team_id
    h2 = {}
    for season in cap.season.unique():
        for gid, ab, tid in con.execute("""SELECT game_id, team_abbrev, team_id
            FROM nba_games WHERE season=? AND game_id LIKE '002%'""", [season]).fetchall():
            h2[(gid, ab)] = tid
    cap["home_tid"] = [h2.get((g, h)) for g, h in zip(cap.gid, cap.home)]
    cap["margin"] = SCALE * np.log(cap.p_us / (1 - cap.p_us))
    for mode, glo, ghi in (("ALL exclusions", 13, 10000), ("mid-season returns only", 13, 90)):
        net = {}
        for season in cap.season.unique():
            for pid, tid, gid, gd, lq, tm10, gap in con.execute(Q, [season]).fetchall():
                if not (glo <= gap <= ghi):
                    continue
                c = darko.get(pid, 0.0) * float(tm10) / 48.0
                net[gid] = net.get(gid, {})
                net[gid][tid] = net[gid].get(tid, 0.0) + c
        dm = []
        for r in cap.itertuples():
            d = net.get(r.gid, {})
            home_c = d.get(r.home_tid, 0.0)
            away_c = sum(v for k, v in d.items() if k != r.home_tid)
            dm.append(0.5 * (home_c - away_c))
        cap["dm"] = dm
        p_fix = 1 / (1 + np.exp(-(cap.margin + cap.dm) / SCALE))
        d = ll(cap.y.values, cap.p_us.values) - ll(cap.y.values, p_fix.values)
        aff = cap.dm.abs() > 1e-9
        rng = np.random.default_rng(0)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"\n== {mode} ==  affected scored games: {aff.sum()} ({aff.mean():.1%}), "
              f"mean|dm| on affected {cap.dm[aff].abs().mean():.2f} pts")
        print(f" pooled gain from restoring played-but-excluded talent: {d.mean():+.5f} "
              f"CI95({lo:+.5f},{hi:+.5f}); on affected games {d[aff].mean():+.5f}")
        for s, sub in cap.assign(d=d).groupby("season"):
            print(f"   {s}: {sub.d.mean():+.5f}")
    con.close()

if __name__ == "__main__":
    main()
