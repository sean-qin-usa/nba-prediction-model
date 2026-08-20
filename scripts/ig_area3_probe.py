"""IG Area-3 probe (read-only): market-filter universe bias, table liveness,
darko staleness, refit-staleness distribution, predict_today top-3 staleness."""
import sys, json, datetime as dt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nbapred.db import connect

con = connect(read_only=True)
out = {}

# --- P0: table inventory + liveness of zero-evidence tables
tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
out["tables"] = tables
for t in ("clv_log", "flagged_windows"):
    out[f"{t}_exists"] = t in tables
    if t in tables:
        out[f"{t}_rows"] = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]

# --- P1: darko staleness
try:
    out["darko_history_max_date"] = str(con.execute("SELECT max(date) FROM darko_history").fetchone()[0])
    out["darko_history_rows"] = con.execute("SELECT count(*) FROM darko_history").fetchone()[0]
except Exception as e:
    out["darko_history_err"] = str(e)
try:
    out["darko_dpm_max_snapshot"] = str(con.execute("SELECT max(snapshot_date) FROM darko_dpm").fetchone()[0])
    out["darko_dpm_n_snapshots"] = con.execute("SELECT count(DISTINCT snapshot_date) FROM darko_dpm").fetchone()[0]
except Exception as e:
    out["darko_dpm_err"] = str(e)

# --- P2: capstone market filter — how many games dropped, and are they biased?
res = {}
for season in ("2023-24", "2024-25", "2025-26"):
    end = int(season[:4]) + 1
    mkt = {(str(r[0])[:10], r[1], r[2]): r[3] for r in con.execute(
        "SELECT game_date, home, away, p_home_spread FROM odds_market WHERE season_end=?",
        [end]).fetchall()}
    meta = con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl, game_date
        FROM nba_games WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL
        ORDER BY game_date""", [season]).fetchdf()
    by = {}
    for x in meta.itertuples():
        by.setdefault(x.game_id, []).append(x)
    total = matched = malformed = 0
    dropped = []
    for gid, recs in by.items():
        if len(recs) != 2:
            malformed += 1
            continue
        m = recs[0].matchup
        host = m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        h = next((x for x in recs if x.team_abbrev == host), None)
        a = next((x for x in recs if x.team_abbrev != host), None)
        if not h or not a:
            malformed += 1
            continue
        total += 1
        gd = h.game_date.date() if hasattr(h.game_date, "date") else h.game_date
        pmv = mkt.get((str(gd)[:10], h.team_abbrev, a.team_abbrev))
        if pmv is None:
            dropped.append((gid, str(gd)[:10], h.team_abbrev, a.team_abbrev,
                            int(h.wl == "W")))
        else:
            matched += 1
    # profile dropped: by month, home-win rate of dropped vs kept
    bymonth = {}
    for _, d, _, _, _ in dropped:
        bymonth[d[:7]] = bymonth.get(d[:7], 0) + 1
    hw_drop = sum(x[4] for x in dropped) / len(dropped) if dropped else None
    res[season] = dict(total=total, matched=matched, dropped=len(dropped),
                       malformed=malformed, dropped_by_month=bymonth,
                       dropped_homewin=hw_drop,
                       dropped_sample=dropped[:8])
out["market_filter"] = res

# null p_home_spread inside odds_market (present row but null prob)
out["odds_market_null_spreadprob"] = con.execute("""
    SELECT season_end, count(*) FILTER (WHERE p_home_spread IS NULL), count(*)
    FROM odds_market WHERE season_end IN (2024,2025,2026) GROUP BY 1 ORDER BY 1""").fetchall()

# --- P3: weekly-refit staleness distribution (simulate prod_by_season refit rule)
stale = {}
for season in ("2023-24", "2024-25", "2025-26"):
    dates = [r[0] for r in con.execute("""SELECT DISTINCT game_date FROM nba_games
        WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL ORDER BY 1""",
        [season]).fetchall()]
    last = None
    ages = []
    ngames = dict(con.execute("""SELECT game_date, count(DISTINCT game_id) FROM nba_games
        WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL GROUP BY 1""",
        [season]).fetchall())
    for d in dates:
        if last is None or (d - last).days >= 7:
            last = d
        ages += [(d - last).days] * ngames[d]
    import numpy as np
    a = np.array(ages)
    stale[season] = dict(mean_staleness_days=float(a.mean()),
                         frac_ge3=float((a >= 3).mean()),
                         frac_ge5=float((a >= 5).mean()), n=int(a.size))
out["refit_staleness"] = stale

# --- P4: predict_today's "top-3 by all-time seconds" — departed players?
probe = {}
for ab in ("LAL", "DAL", "POR"):
    tid = con.execute("SELECT DISTINCT team_id FROM nba_games WHERE team_abbrev=? LIMIT 1",
                      [ab]).fetchone()[0]
    top = con.execute("""SELECT s.player_id, sum(s.seconds) sec FROM player_game_stats s
        WHERE s.team_id=? GROUP BY 1 ORDER BY 2 DESC LIMIT 3""", [tid]).fetchall()
    rows = []
    for pid, sec in top:
        nm = con.execute("SELECT full_name FROM nba_players WHERE player_id=?", [pid]).fetchone()
        last = con.execute("""SELECT max(g.game_date) FROM player_game_stats s
            JOIN nba_games g USING (game_id) WHERE s.player_id=? AND s.team_id=?""",
            [pid, tid]).fetchone()[0]
        cur = con.execute("""SELECT any_value(t.team_id) FROM (
            SELECT s.team_id, row_number() OVER (ORDER BY g.game_date DESC) rn
            FROM player_game_stats s JOIN nba_games g USING (game_id)
            WHERE s.player_id=?) t WHERE t.rn=1""", [pid]).fetchone()[0]
        rows.append(dict(pid=int(pid), name=nm[0] if nm else None,
                         last_played_for_team=str(last),
                         still_on_team=bool(cur == tid)))
    probe[ab] = rows
out["top3_alltime"] = probe

# --- P5: injury_reports_pit availability for live wiring
if "injury_reports_pit" in tables:
    out["injury_pit_rows"] = con.execute("SELECT count(*) FROM injury_reports_pit").fetchone()[0]
    out["injury_pit_maxdate"] = str(con.execute("SELECT max(game_date) FROM injury_reports_pit").fetchone()[0])

# --- P6: espn_lines (live market source) vs odds_market (backtest market source)
for t in ("espn_lines", "odds_quotes"):
    if t in tables:
        out[f"{t}_rows"] = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]

con.close()
print(json.dumps(out, indent=1, default=str))
