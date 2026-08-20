"""D171 TASK 3 — is the low OUT/team-game of 2012-16 a DATA GAP or a REAL ERA
DIFFERENCE?  Measured three ways.  Read-only."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nbapred.threads; nbapred.threads.pin(1)
import numpy as np
from nbapred.db import connect
con = connect(read_only=True, retry_s=60.0)
SEASONS = ["%d-%02d" % (y, (y+1) % 100) for y in range(2007, 2026)]

# 1. RAW inactive-list density (the official list, before any roster filter)
raw = dict(con.execute("""
    SELECT g.season, count(*)*1.0/count(DISTINCT (g.game_id||'-'||g.team_id))
    FROM game_inactives i JOIN nba_games g ON g.game_id=i.game_id
    WHERE g.game_id LIKE '002%' GROUP BY g.season""").fetchall())
tg = dict(con.execute("""SELECT season, count(*) FROM nba_games
    WHERE game_id LIKE '002%' GROUP BY season""").fetchall())
inact_rows = dict(con.execute("""
    SELECT g.season, count(*) FROM game_inactives i
    JOIN nba_games g ON g.game_id=i.game_id AND g.team_id=i.team_id
    WHERE g.game_id LIKE '002%' GROUP BY g.season""").fetchall())

# 2. Correctness of the list per season (D170's test, re-run per season):
viol = dict(con.execute("""
    SELECT g.season, sum(CASE WHEN s.seconds>0 THEN 1 ELSE 0 END)
    FROM game_inactives i JOIN nba_games g ON g.game_id=i.game_id
    LEFT JOIN player_game_stats s ON s.game_id=i.game_id AND s.player_id=i.player_id
    WHERE g.game_id LIKE '002%' GROUP BY g.season""").fetchall())

# 3. Roster-size / rotation depth: how many distinct players a team actually
#    used per game, and how many were on the 12-day window. A thinner OUT set
#    could mean "fewer absences" OR "smaller rostered pool"; this separates them.
depth = dict(con.execute("""
    SELECT g.season, count(*)*1.0/count(DISTINCT (g.game_id||'-'||g.team_id))
    FROM player_game_stats s JOIN nba_games g
      ON g.game_id=s.game_id AND g.team_id=s.team_id
    WHERE g.game_id LIKE '002%' AND s.seconds>0 GROUP BY g.season""").fetchall())

# 4. The scored OUT/team-game actually used by the model, from the k19 arms
k19 = json.load(open(Path(__file__).resolve().parents[1]/"data/k19_d171_t2.json"))
outs = {s["season"]: s["mean_outs_per_team"] for s in k19["seasons"]}

print("="*100)
print("ERA DENSITY — is 2012-16's thin OUT set a DATA GAP or a REAL DIFFERENCE?")
print("="*100)
print(f"{'season':<9}{'inactive rows':>14}{'raw/team-gm':>13}{'scored OUT/tm':>15}"
      f"{'kept%':>8}{'played/team-gm':>16}{'viol':>7}")
for s in SEASONS:
    r = raw.get(s, 0.0); o = outs.get(s, 0.0)
    print(f"{s:<9}{inact_rows.get(s,0):>14,}{r:>13.2f}{o:>15.2f}"
          f"{(100*o/r if r else 0):>7.1f}%{depth.get(s,0):>16.2f}{viol.get(s,0):>7}")
print()
print("READING:")
print(" * `viol` is the count of inactive-listed players who nevertheless logged")
print("   minutes. 0 on EVERY season = the list is COMPLETE and CORRECT in both eras.")
print(" * raw/team-gm is the official list's own density — it is not something we")
print("   built, so it cannot be an ingest artefact once viol=0.")
print(" * kept% is what survives the 12-day roster window (the model only counts")
print("   an OUT if the player was plausibly available).")
con.close()
