#!/usr/bin/env python3
"""IG probe (read-only): hardcoded constants vs measured values.

1. fg fallback constants in props.py (0.60/0.42/0.35 rim/mid/thr, 0.77 FT)
   vs measured league averages per season.
2. team_zone_defense league priors (0.613/0.44/0.359) vs measured.
3. LEAGUE_PACE=99.5 (props.py apply_pace divisor) vs measured pace per season.
4. How often the fg fallbacks actually TRIGGER in the eval universe
   (weighted attempts den <= 5) and how much rate mass they carry.
5. possession.py LEAGUE seed rates vs measured (feeds adjusted.py calibrator).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from nbapred.db import connect

con = connect(read_only=True)

print("=== 1/2. League zone FG% + FT% by season (vs fallbacks 0.60/0.42/0.35, FT 0.77;"
      " zone-D priors 0.613/0.44/0.359) ===")
for tag, lab in (("00222%", "2022-23"), ("00223%", "2023-24"),
                 ("00224%", "2024-25"), ("00225%", "2025-26")):
    r = con.execute("""
        SELECT sum(rimm)/sum(rima), sum(midm)/sum(mida), sum(thrm)/sum(thra),
               sum(ftm)/sum(fta)
        FROM player_game_stats WHERE game_id LIKE ?""", [tag]).fetchone()
    print(f"  {lab}: rim {r[0]:.4f}  mid {r[1]:.4f}  thr {r[2]:.4f}  ft {r[3]:.4f}")

print("\n=== 3. Pace per team-game by season (vs LEAGUE_PACE=99.5) ===")
for tag, lab in (("00222%", "2022-23"), ("00223%", "2023-24"),
                 ("00224%", "2024-25"), ("00225%", "2025-26")):
    r = con.execute("""
        SELECT avg(poss) FROM (
          SELECT game_id, team_id, sum(fga + 0.44*fta - oreb + tov) poss
          FROM player_game_stats WHERE game_id LIKE ? GROUP BY 1,2)""", [tag]).fetchone()
    print(f"  {lab}: {r[0]:.2f}")

print("\n=== 4. fg fallback trigger frequency (eval-like universe, 2025-26) ===")
# players with >=8 games of >=720s (the prop eval filter), unweighted attempt sums
df = con.execute("""
    SELECT player_id, count(*) n, sum(rima) ra, sum(mida) ma, sum(thra) ta,
           sum(fta) fa, sum(mida)/(sum(seconds)/60.0) mid_rate
    FROM player_game_stats WHERE game_id LIKE '00225%' AND seconds >= 720
    GROUP BY 1 HAVING count(*) >= 8""").fetchdf()
n = len(df)
for col, lab in (("ra", "rim"), ("ma", "mid"), ("ta", "thr"), ("fa", "ft")):
    frac = float((df[col] <= 5).mean())
    print(f"  {lab}: den<=5 (fallback fires, full-season) {frac:.3f} of {n} players")
# EWMA-weighted den is smaller than full-season den -> early-season much worse.
# approximate early-season: first 10 games only
df10 = con.execute("""
    WITH r AS (SELECT s.player_id, s.rima, s.mida, s.thra, s.fta,
               row_number() OVER (PARTITION BY s.player_id ORDER BY g.game_date) rn
               FROM player_game_stats s
               JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
               WHERE s.game_id LIKE '00225%' AND s.seconds >= 720)
    SELECT player_id, sum(rima) ra, sum(mida) ma, sum(thra) ta, sum(fta) fa
    FROM r WHERE rn <= 10 GROUP BY 1 HAVING count(*) >= 8""").fetchdf()
for col, lab in (("ra", "rim"), ("ma", "mid"), ("ta", "thr"), ("fa", "ft")):
    print(f"  {lab}: den<=5 within first-10-games window {float((df10[col] <= 5).mean()):.3f}")

print("\n=== 5. possession.py LEAGUE seed rates vs measured 2025-26 ===")
r = con.execute("""
    SELECT sum(rima), sum(mida), sum(thra), sum(fga), sum(fta),
           sum(shooting_fouls), sum(oreb), sum(tov),
           sum(fga + 0.44*fta - oreb + tov)
    FROM player_game_stats WHERE game_id LIKE '00225%'""").fetchone()
rim, mid, thr, fga, fta, sf, oreb, tov, poss = [float(x) for x in r]
print(f"  zone_share measured: rim {rim/fga:.3f} mid {mid/fga:.3f} thr {thr/fga:.3f} "
      f"(code: .346/.235/.418)")
print(f"  tov_per_poss {tov/poss:.3f} (code .132)  fta/fga {fta/fga:.3f} (code implies ~0.29)")
con.close()
