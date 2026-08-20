"""HA-PANEL — build the game-level home-advantage panel (DIAGNOSTIC, read-only).

One row per REGULAR-SEASON game (game_id LIKE '002%'), 2019-20..2025-26, with:
  home/away team (parsed from `matchup`, not the unreliable is_home flag),
  margin = home_pts - away_pts,
  schedule state for BOTH sides rebuilt from nba_games + the static ARENAS geo
  (schedule_features only covers 2025-26 in this DB),
  season-phase indices (team games played, calendar month, days into season),
  crowd stratum per D131.

Writes the panel to the scratchpad only. Nothing in nbapred/ or data/ is
modified. NEW file; no production import beyond the read-only ARENAS constant.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import duckdb
import numpy as np
import pandas as pd

from nbapred.features.schedule import ARENAS

DB = REPO / "data" / "nba.duckdb"
OUT = Path("data/scratch/ha_panel.csv")

# metres above sea level, arena elevation (public/geographic, fixed constants)
ALTITUDE_M = {
    "DEN": 1609, "UTA": 1288, "PHX": 331, "OKC": 366, "SAS": 198, "DAL": 131,
    "ATL": 320, "MIL": 188, "MIN": 250, "CLE": 199, "DET": 190, "CHI": 181,
    "IND": 218, "MEM": 78, "CHA": 229, "POR": 15, "SAC": 9, "LAL": 87,
    "LAC": 87, "GSW": 3, "HOU": 15, "NOP": 2, "MIA": 2, "ORL": 32, "BOS": 6,
    "BKN": 12, "NYK": 10, "PHI": 12, "TOR": 76, "WAS": 8,
}


def connect_ro(retries: int = 12, wait: float = 5.0):
    """Read-only connection, 60s of retry on lock."""
    import time
    last = None
    for _ in range(retries):
        try:
            return duckdb.connect(str(DB), read_only=True)
        except Exception as e:  # lock
            last = e
            time.sleep(wait)
    raise last


def _hav(a: str, b: str) -> float:
    if a not in ARENAS or b not in ARENAS:
        return np.nan
    (la1, lo1, _), (la2, lo2, _) = ARENAS[a], ARENAS[b]
    r = 6371.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dphi, dl = math.radians(la2 - la1), math.radians(lo2 - lo1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _host(matchup: str):
    if not matchup:
        return None
    if "@" in matchup:
        return matchup.split("@")[-1].strip()
    if "vs." in matchup:
        return matchup.split("vs.")[0].strip()
    return None


def stratum(season: str, d) -> str:
    if season == "2019-20":
        return "bubble" if d >= pd.Timestamp("2020-07-30").date() else "pre"
    if season == "2020-21":
        return "nocrowd"
    return "normal"


def build() -> pd.DataFrame:
    con = connect_ro()
    raw = con.execute("""
        SELECT season, game_id, game_date, team_abbrev AS team, matchup, pts, is_home
        FROM nba_games
        WHERE game_id LIKE '002%'
        ORDER BY game_date, game_id
    """).fetchdf()
    con.close()
    raw["game_date"] = pd.to_datetime(raw["game_date"]).dt.date
    # NEUTRAL SITE: the NBA feed sets is_home=False on BOTH team-rows for
    # neutral-court regular-season games (Mexico City / Paris / NBA Cup
    # semifinals in Las Vegas). 10 such games in the corpus (2024-25, 2025-26).
    # The flag is incomplete for earlier seasons (the 2023 Cup semifinals in
    # Las Vegas are not marked) -> a residual ~2-4 games/season contaminant
    # that cannot move a 1,230-game mean by more than ~0.01 pt.
    neu = raw.groupby("game_id")["is_home"].sum()
    neutral_ids = set(neu[neu == 0].index)

    # --- collapse to one row per game -------------------------------------
    games = []
    for gid, g in raw.groupby("game_id", sort=False):
        if len(g) != 2:
            continue
        host = None
        for m in g["matchup"]:
            host = _host(m)
            if host:
                break
        teams = list(g["team"])
        if host not in teams:
            continue
        hi = teams.index(host)
        ai = 1 - hi
        r = g.iloc[hi]
        a = g.iloc[ai]
        games.append(dict(season=r["season"], game_id=gid, game_date=r["game_date"],
                          home=r["team"], away=a["team"],
                          home_pts=r["pts"], away_pts=a["pts"],
                          neutral=int(gid in neutral_ids)))
    df = pd.DataFrame(games).sort_values(["game_date", "game_id"]).reset_index(drop=True)
    df["margin"] = df["home_pts"] - df["away_pts"]
    df["stratum"] = [stratum(s, d) for s, d in zip(df["season"], df["game_date"])]
    df.loc[df["stratum"] == "bubble", "neutral"] = 1   # bubble = physically neutral

    # --- per-team chronological schedule state, within season -------------
    hist: dict[tuple, list] = {}          # (season, team) -> [(date, host)]
    cols = {k: [] for k in
            ["h_rest", "a_rest", "h_b2b", "a_b2b", "h_3in4", "a_3in4",
             "h_travel", "a_travel", "h_tz", "a_tz", "h_gp", "a_gp",
             "h_road_streak", "a_road_streak", "h_g7", "a_g7"]}
    for row in df.itertuples():
        for side, team in (("h", row.home), ("a", row.away)):
            past = hist.get((row.season, team), [])
            gp = len(past)
            rest = (row.game_date - past[-1][0]).days if past else np.nan
            prev_host = past[-1][1] if past else None
            travel = _hav(prev_host, row.home) if prev_host else 0.0
            tz = (ARENAS[row.home][2] - ARENAS[prev_host][2]) if (
                prev_host in ARENAS and row.home in ARENAS) else 0
            last4 = [p for p, _ in past if 0 <= (row.game_date - p).days <= 3]
            g7 = sum(1 for p, _ in past if 0 <= (row.game_date - p).days <= 7)
            # consecutive prior games away from own arena
            streak = 0
            for p, hh in reversed(past):
                if hh != team:
                    streak += 1
                else:
                    break
            cols[f"{side}_rest"].append(rest)
            cols[f"{side}_b2b"].append(1.0 if rest == 1 else 0.0)
            cols[f"{side}_3in4"].append(1.0 if len(last4) >= 2 else 0.0)
            cols[f"{side}_travel"].append(travel)
            cols[f"{side}_tz"].append(tz)
            cols[f"{side}_gp"].append(gp)
            cols[f"{side}_road_streak"].append(streak)
            cols[f"{side}_g7"].append(g7)
        for team in (row.home, row.away):
            hist.setdefault((row.season, team), []).append((row.game_date, row.home))
    for k, v in cols.items():
        df[k] = v

    # season-phase indices
    df["month"] = pd.to_datetime(df["game_date"]).dt.month
    first = df.groupby("season")["game_date"].transform("min")
    df["days_into"] = (pd.to_datetime(df["game_date"]) - pd.to_datetime(first)).dt.days
    df["min_gp"] = df[["h_gp", "a_gp"]].min(axis=1)
    df["mean_gp"] = df[["h_gp", "a_gp"]].mean(axis=1)
    df["h_alt"] = df["home"].map(ALTITUDE_M)
    df["a_alt"] = df["away"].map(ALTITUDE_M)
    df["alt_gain"] = df["h_alt"] - df["a_alt"]
    return df


if __name__ == "__main__":
    d = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)
    print(f"wrote {OUT}  rows={len(d)}")
    print(d.groupby(["season", "stratum"]).agg(
        n=("margin", "size"), home_margin=("margin", "mean"),
        home_wr=("margin", lambda s: (s > 0).mean())).round(4).to_string())
