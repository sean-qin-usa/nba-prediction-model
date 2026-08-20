"""IG probe C: composition.py `any_value(team_id)` traded-player attribution.

CompositionModel groups the last-10 qualifying games per player and takes
any_value(team_id). For a recently traded player the rn<=10 window spans BOTH
teams; any_value is order-arbitrary, so the player can be attributed to his OLD
team while his last_played (new team) keeps him inside ROSTER_DAYS — old team
keeps his strength, new team never gets it.

Probe: at every Monday cutoff across each season, run the EXACT production CTE
and compare any_value(team_id) to the rn=1 (most recent) team. Count mismatches
and their talent x minutes mass (= margin points of error per affected team).
Read-only.
"""
import sys, warnings, datetime as dt
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nbapred.db import connect

Q = """
WITH pg AS (
  SELECT s.player_id, s.team_id, s.seconds/60.0 m, g.game_date,
         row_number() OVER (PARTITION BY s.player_id ORDER BY g.game_date DESC) rn
  FROM player_game_stats s
  JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
  WHERE s.game_id LIKE '002%' AND s.seconds >= 720 AND g.game_date < ?
),
agg AS (
  SELECT player_id, any_value(team_id) team_prod, avg(m) trail_min,
         max(game_date) last_played
  FROM pg WHERE rn <= 10 GROUP BY player_id
),
cur AS (SELECT player_id, team_id team_true FROM pg WHERE rn = 1),
span AS (SELECT player_id, count(DISTINCT team_id) nteams FROM pg WHERE rn <= 10 GROUP BY 1)
SELECT a.player_id, a.team_prod, c.team_true, a.trail_min, a.last_played, s.nteams
FROM agg a JOIN cur c USING (player_id) JOIN span s USING (player_id)
WHERE ? - a.last_played <= 12
"""

def main():
    con = connect(read_only=True)
    darko = dict(con.execute("""SELECT player_id, dpm FROM (
        SELECT player_id, dpm, row_number() OVER (PARTITION BY player_id ORDER BY date DESC) rn
        FROM darko_history) WHERE rn=1""").fetchall())
    names = dict(con.execute("SELECT player_id, full_name FROM nba_players").fetchall())
    for season in ("2023-24", "2024-25", "2025-26"):
        d0, d1 = con.execute("""SELECT min(game_date), max(game_date) FROM nba_games
            WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL""", [season]).fetchone()
        n_dates = mism_dates = 0
        tot_active = tot_span = tot_mism = 0
        worst = []
        d = d0 + dt.timedelta(days=14)
        while d <= d1:
            rows = con.execute(Q, [d, d]).fetchall()
            n_dates += 1
            tot_active += len(rows)
            for pid, tp, tt, tm, lp, nt in rows:
                if nt > 1:
                    tot_span += 1
                if tp != tt:
                    tot_mism += 1
                    pts = abs(darko.get(pid, 0.0)) * tm / 48.0
                    worst.append((round(pts, 2), names.get(pid, pid), str(d), int(tp), int(tt)))
            if any(tp != tt for _, tp, tt, *_ in rows):
                mism_dates += 1
            d += dt.timedelta(days=7)
        worst.sort(reverse=True)
        print(f"\n== {season} == cutoffs={n_dates} active-player rows={tot_active}")
        print(f" multi-team windows: {tot_span}  MISATTRIBUTED (any_value != most-recent team): {tot_mism}"
              f"  cutoffs affected: {mism_dates}")
        for w in worst[:8]:
            print("  worst:", w)
    con.close()

if __name__ == "__main__":
    main()
