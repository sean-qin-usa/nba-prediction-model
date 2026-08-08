#!/usr/bin/env python3
"""D175 pilot validation: archived CBS injury rows vs what actually happened.

READ-ONLY on DuckDB (read_only=True, 60s retry). Writes JSON/CSV artifacts only.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import nbapred.threads          # noqa: E402
nbapred.threads.pin(1)
import duckdb, numpy as np, pandas as pd   # noqa: E402

RAW = REPO / "data" / "raw" / "unofficial" / "wayback"
SEASON = sys.argv[2] if len(sys.argv) > 2 else "2015-16"


def ro():
    for i in range(60):
        try:
            return duckdb.connect(str(REPO / "data" / "nba.duckdb"), read_only=True)
        except duckdb.IOException as e:
            if "lock" not in str(e).lower():
                raise
            print(f"write lock held; yielding 60s ({i+1}/60)", flush=True)
            time.sleep(60)
    raise RuntimeError("db locked")


def norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


con = ro()
d = pd.read_csv(RAW / f"{sys.argv[1]}_rows.csv")
d["snap_et_date"] = pd.to_datetime(d["snap_et_date"])
# target game date: before 5pm ET the same day's slate is still ahead of us
d["target_date"] = d["snap_et_date"] + pd.to_timedelta(
    (d["snap_et_hour"] >= 17).astype(int), unit="D")
d["pkey"] = d["player_slug"].map(norm)

# ---- player crosswalk -----------------------------------------------------
pl = con.execute("SELECT player_id, first_name, last_name FROM nba_players").fetchdf()
pl["pkey"] = (pl.first_name.fillna("") + pl.last_name.fillna("")).map(norm)
pl = pl.drop_duplicates("pkey")
d = d.merge(pl[["pkey", "player_id"]], on="pkey", how="left")
match_rate = d.player_id.notna().mean()
unmatched = sorted(d.loc[d.player_id.isna(), "player_name"].unique())

# ---- schedule + outcomes --------------------------------------------------
g = con.execute("""
    SELECT season, game_id, game_date, team_id, team_abbrev
    FROM nba_games WHERE game_id LIKE '002%' AND wl IS NOT NULL AND season = ?
""", [SEASON]).fetchdf()
g["game_date"] = pd.to_datetime(g["game_date"])
from nbapred.teams import FRANCHISE  # noqa: E402
CBS = {"PHO": "PHX", "CHO": "CHA", "GS": "GSW", "NO": "NOP", "NY": "NYK",
       "SA": "SAS", "UTAH": "UTA", "BRK": "BKN", "WSH": "WAS"}
d["team_ab"] = d["team_abbr"].map(lambda a: CBS.get(a, a))

m = d.merge(g, left_on=["team_ab", "target_date"],
            right_on=["team_abbrev", "game_date"], how="inner")
# keep only the MOST RECENT snapshot per (player, game)
m = m.sort_values("snap_ts_utc").drop_duplicates(
    subset=["player_id", "game_id"], keep="last")
m = m[m.player_id.notna()].copy()
m["player_id"] = m.player_id.astype("int64")

box = con.execute("""SELECT game_id, player_id, seconds FROM player_game_stats
                     WHERE seconds > 0""").fetchdf()
ina = con.execute("SELECT game_id, player_id, 1 AS inactive FROM game_inactives").fetchdf()
m = m.merge(box, on=["game_id", "player_id"], how="left") \
     .merge(ina, on=["game_id", "player_id"], how="left")
m["played"] = m.seconds.notna() & (m.seconds.fillna(0) > 0)
m["mins"] = m.seconds.fillna(0) / 60.0
m["inactive"] = m.inactive.fillna(0).astype(int)

# ---- advance notice -------------------------------------------------------
m["updated_dt"] = pd.to_datetime(m["updated"], format="%m/%d/%y", errors="coerce")
m["notice_days"] = (m.game_date - m.updated_dt).dt.days
snap = pd.to_datetime(m.snap_ts_utc.astype(str), format="%Y%m%d%H%M%S", utc=True)
m["snap_lead_h"] = ((m.game_date.dt.tz_localize("America/New_York")
                     + pd.Timedelta(hours=19)) - snap).dt.total_seconds() / 3600.0

# ---- report ---------------------------------------------------------------
out = {}
print(f"=== PILOT {SEASON}: {sys.argv[1]} ===")
print(f"parsed rows {len(d)}  name-match {match_rate:.4f}  "
      f"unmatched-names {len(unmatched)}")
print(f"joined (player,game) pairs, most-recent-snapshot-only: {len(m)}")
out["parsed_rows"] = int(len(d)); out["name_match_rate"] = float(match_rate)
out["unmatched_names"] = unmatched[:40]; out["pairs"] = int(len(m))

print("\n--- AGREEMENT: status vs what actually happened ---")
tab = m.groupby("status").agg(
    n=("played", "size"), played=("played", "sum"),
    pct_played=("played", "mean"), pct_inactive=("inactive", "mean"),
    mean_min_if_played=("mins", lambda s: s[s > 0].mean())).reset_index()
tab["pct_played"] = (100 * tab.pct_played).round(2)
tab["pct_inactive"] = (100 * tab.pct_inactive).round(2)
tab["mean_min_if_played"] = tab.mean_min_if_played.round(1)
print(tab.to_string(index=False))
out["agreement"] = tab.to_dict("records")

sure_out = m[m.status.isin(["OUT", "OUT_SEASON"])]
print(f"\nOUT+OUT_SEASON: n={len(sure_out)}  did-NOT-play "
      f"{100*(1-sure_out.played.mean()):.2f}%  in game_inactives "
      f"{100*sure_out.inactive.mean():.2f}%")
out["out_n"] = int(len(sure_out))
out["out_didnotplay_pct"] = float(100 * (1 - sure_out.played.mean()))
out["out_in_inactives_pct"] = float(100 * sure_out.inactive.mean())

print("\n--- ADVANCE NOTICE (days from CBS 'Updated' date to game date) ---")
nd = m.notice_days.dropna()
print(f"n={len(nd)}  >=1d {100*(nd>=1).mean():.2f}%  >=2d {100*(nd>=2).mean():.2f}%  "
      f"median {nd.median():.1f}  p25 {nd.quantile(.25):.1f}  p75 {nd.quantile(.75):.1f}")
sl = m.snap_lead_h.dropna()
print(f"snapshot lead vs 19:00 ET tip: median {sl.median():.1f}h  "
      f">=24h {100*(sl>=24).mean():.2f}%  >=5h(pre-report-equiv) {100*(sl>=5).mean():.2f}%")
out["notice_ge1d_pct"] = float(100 * (nd >= 1).mean())
out["notice_median_days"] = float(nd.median())
out["snap_lead_median_h"] = float(sl.median())

print("\n--- REASON COVERAGE of game_inactives (the deliverable-2 fraction) ---")
gi = con.execute("""
  SELECT gi.game_id, gi.player_id FROM game_inactives gi
  JOIN (SELECT DISTINCT game_id FROM nba_games WHERE season=? AND game_id LIKE '002%'
        AND wl IS NOT NULL) s USING (game_id)""", [SEASON]).fetchdf()
cov = gi.merge(m[["game_id", "player_id", "status", "injury"]],
               on=["game_id", "player_id"], how="left")
frac = cov.status.notna().mean()
print(f"game_inactives rows in {SEASON}: {len(gi)}")
print(f"  with a CBS status+reason attached: {cov.status.notna().sum()} = {100*frac:.2f}%")
out["inactive_rows"] = int(len(gi)); out["inactive_reason_pct"] = float(100 * frac)

# how much of the model-relevant absence mass is covered? weight by prior mpg
print("\n--- coverage weighted by player importance (season mpg) ---")
mpg = con.execute("""
  SELECT p.player_id, sum(p.seconds)/60.0/count(*) mpg
  FROM player_game_stats p JOIN (SELECT game_id FROM nba_games WHERE season=?
       AND game_id LIKE '002%' AND wl IS NOT NULL) s USING (game_id)
  WHERE p.seconds>0 GROUP BY 1""", [SEASON]).fetchdf()
cov = cov.merge(mpg, on="player_id", how="left")
cov["mpg"] = cov.mpg.fillna(0)
for thr in (0, 15, 20, 25, 30):
    sub = cov[cov.mpg >= thr]
    if len(sub):
        print(f"  mpg>={thr:>2}: n={len(sub):>5}  reason-covered {100*sub.status.notna().mean():>6.2f}%")
        out[f"inactive_reason_pct_mpg{thr}"] = float(100 * sub.status.notna().mean())

print("\n--- CBS reason vocabulary (top 25) ---")
print(m.injury.value_counts().head(25).to_string())
out["injury_vocab"] = m.injury.value_counts().head(40).to_dict()

json.dump(out, open(REPO / "data" / f"d175_{sys.argv[1]}_validate.json", "w"),
          indent=1, default=str)
m.to_csv(RAW / f"{sys.argv[1]}_joined.csv.gz", index=False, compression="gzip")
print(f"\nwrote data/d175_{sys.argv[1]}_validate.json")
