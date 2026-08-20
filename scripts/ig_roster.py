"""IG probe B: ROSTER_DAYS=12 cliff in composition.py.

In the shipped backtest, oracle OUT-sets already remove absent players inside
the 12-day window, so the cliff BINDS on the other side: a player who PLAYS
tonight but whose last >=12-min game was >12 days ago is silently excluded from
his team's strength (return-from-injury night), and a star easing back on <12
minute games never resets last_played. Measure: how many scored games have such
excluded-but-playing talent, and how large.
Read-only.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from nbapred.db import connect

Q = """
WITH q AS (
  SELECT s.player_id, s.team_id, g.game_date, s.seconds/60.0 m
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
  SELECT s.player_id, s.team_id, s.game_id, sg.game_date, s.seconds/60.0 mins
  FROM player_game_stats s JOIN sg USING (game_id) WHERE s.seconds > 0
)
SELECT p.player_id, p.team_id, p.game_id, p.game_date, p.mins,
       qr.game_date last_q, qr.tm10,
       (p.game_date - qr.game_date) gap
FROM played p ASOF JOIN qroll qr
  ON qr.player_id = p.player_id AND qr.game_date < p.game_date
WHERE (p.game_date - qr.game_date) > 12
"""

def main():
    con = connect(read_only=True)
    darko = dict(con.execute("""SELECT player_id, dpm FROM (
        SELECT player_id, dpm, row_number() OVER (PARTITION BY player_id ORDER BY date DESC) rn
        FROM darko_history) WHERE rn=1""").fetchall())
    names = dict(con.execute("SELECT player_id, full_name FROM nba_players").fetchall())
    cap = pd.read_csv("/hdd/steveqin/sean_dev/nba_model/data/capstone_pergame_carry2.csv")
    cap["gid"] = cap.game_id.apply(lambda g: f"00{g}" if len(str(g)) == 8 else str(g))
    scored = set(cap.gid)
    for season in ("2023-24", "2024-25", "2025-26"):
        rows = con.execute(Q, [season]).fetchall()
        events = []
        for pid, tid, gid, gd, mins, lq, tm10, gap in rows:
            contrib = abs(darko.get(pid, 0.0)) * float(tm10) / 48.0
            events.append((gid, pid, contrib, float(mins), int(gap)))
        ev = pd.DataFrame(events, columns=["gid", "pid", "contrib", "mins", "gap"])
        ev_sc = ev[ev.gid.isin(scored)]
        big = ev_sc[ev_sc.contrib > 0.5]
        ggames = big.groupby("gid").contrib.sum()
        print(f"\n== {season} ==")
        print(f" playing-but-roster-excluded player-games: {len(ev)} "
              f"(in scored games: {len(ev_sc)}), with contrib>0.5pts: {len(big)}")
        print(f" scored games affected by >0.5pt exclusion: {len(ggames)} "
              f"({100 * len(ggames) / cap[cap.season == season].shape[0]:.1f}% of scored)"
              f"  mean excluded mass {ggames.mean():.2f} pts  max {ggames.max():.2f}")
        top = big.nlargest(5, "contrib")
        for r in top.itertuples():
            print(f"   e.g. {names.get(r.pid, r.pid):24} contrib {r.contrib:.2f} pts "
                  f"(gap {r.gap}d, played {r.mins:.0f}min that night)")
    con.close()

if __name__ == "__main__":
    main()
