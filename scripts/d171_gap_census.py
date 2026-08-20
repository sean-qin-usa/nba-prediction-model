"""D171 TASK 3 — per-season coverage census of EVERY feed, so "what is still
asymmetric between an old season and 2024-25" is answered from the DB rather
than from memory.  Read-only."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nbapred.threads; nbapred.threads.pin(1)
from nbapred.db import connect

con = connect(read_only=True, retry_s=60.0)
SEASONS = ["%d-%02d" % (y, (y+1) % 100) for y in range(2007, 2026)]

# games per season (the denominator)
games = dict(con.execute(
    "SELECT season, count(DISTINCT game_id) FROM nba_games WHERE game_id LIKE '002%' "
    "GROUP BY season").fetchall())

def by_game(tbl, extra=""):
    q = (f"SELECT g.season, count(DISTINCT t.game_id) FROM {tbl} t "
         f"JOIN nba_games g ON g.game_id = t.game_id "
         f"WHERE g.game_id LIKE '002%' {extra} GROUP BY g.season")
    return dict(con.execute(q).fetchall())

cov = {}
cov["game_inactives"]    = by_game("game_inactives")
cov["player_game_stats"] = by_game("player_game_stats")
cov["possessions_v2"]    = by_game("possessions_v2")
cov["lineup_stints"]     = by_game("lineup_stints")
cov["game_officials"]    = by_game("game_officials")
cov["schedule_features"] = dict(con.execute(
    "SELECT season, count(DISTINCT game_id) FROM schedule_features GROUP BY season").fetchall())
cov["odds_market"] = dict(con.execute(
    "SELECT season_end-1, count(*) FROM odds_market GROUP BY 1").fetchall())
cov["odds_open"] = dict(con.execute(
    "SELECT season, count(*) FROM odds_open GROUP BY 1").fetchall())
cov["odds_hist_sbr"] = dict(con.execute(
    "SELECT season, count(*) FROM odds_hist_sbr GROUP BY 1").fetchall())

# injury report: report-days that are also game-days, per season
irep = dict(con.execute("""
    SELECT g.season, count(DISTINCT i.game_date) FROM injury_reports_pit i
    JOIN (SELECT DISTINCT season, game_date FROM nba_games WHERE game_id LIKE '002%') g
      ON g.game_date = i.game_date
    WHERE i.report_date = i.game_date GROUP BY g.season""").fetchall())
gdays = dict(con.execute(
    "SELECT season, count(DISTINCT game_date) FROM nba_games WHERE game_id LIKE '002%' "
    "GROUP BY season").fetchall())

# DARKO PIT minute coverage per season (the D170 headline metric)
darko = dict(con.execute("""
    WITH pm AS (SELECT g.season, s.player_id, s.seconds/60.0 mins, g.game_date
                FROM player_game_stats s JOIN nba_games g
                  ON g.game_id=s.game_id AND g.team_id=s.team_id
                WHERE g.game_id LIKE '002%' AND s.seconds>0)
    SELECT pm.season,
           sum(CASE WHEN d.player_id IS NOT NULL THEN pm.mins ELSE 0 END)/sum(pm.mins)
    FROM pm LEFT JOIN (SELECT DISTINCT player_id FROM darko_history) d
      ON d.player_id = pm.player_id GROUP BY pm.season""").fetchall())

# defended_fg / hustle-class tables carry a season column directly
cov["defended_fg"] = dict(con.execute(
    "SELECT season, count(*) FROM defended_fg GROUP BY season").fetchall())

# odds panel depth: distinct books per season where the schema exposes it
try:
    cov["odds_books"] = dict(con.execute(
        "SELECT season, count(DISTINCT book) FROM odds_hist_sbr GROUP BY season").fetchall())
except Exception:
    cov["odds_books"] = {}

hdr = ("season   games  inact%  report_days  darko%  poss%   stints%  refs%  "
       "sched  odds_mkt  odds_open  sbr")
print("="*len(hdr)); print("D171 PER-SEASON COVERAGE CENSUS"); print("="*len(hdr))
print(hdr); print("-"*len(hdr))
rows = {}
for s in SEASONS:
    g = games.get(s, 0)
    def pc(d):
        return 100.0*d.get(s, 0)/g if g else 0.0
    r = dict(games=g, inact=pc(cov["game_inactives"]),
             report_days=irep.get(s, 0), game_days=gdays.get(s, 0),
             darko=100*darko.get(s, 0.0), poss=pc(cov["possessions_v2"]),
             stints=pc(cov["lineup_stints"]), refs=pc(cov["game_officials"]),
             sched=cov["schedule_features"].get(s, 0),
             odds_mkt=cov["odds_market"].get(int(s[:4]), 0),
             odds_open=cov["odds_open"].get(s, 0),
             sbr=cov["odds_hist_sbr"].get(s, 0))
    rows[s] = r
    print(f"{s}  {g:>5}  {r['inact']:5.1f}  {r['report_days']:>4}/{r['game_days']:<4}   "
          f"{r['darko']:5.1f}  {r['poss']:5.1f}  {r['stints']:6.1f}  {r['refs']:5.1f}  "
          f"{r['sched']:>5}  {r['odds_mkt']:>7}  {r['odds_open']:>8}  {r['sbr']:>5}")

json.dump(rows, open(Path(__file__).resolve().parents[1]/"data/d171_gap_census.json","w"), indent=1)
con.close()
