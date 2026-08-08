#!/usr/bin/env python3
"""D172 TASK 2 step 1b — coach-by-team-season AND mid-season coach changes,
$0, from Basketball-Reference's public per-season coaches page.

WHY NOT nba_api.  `commonteamroster`'s Coaches result set is NOT
point-in-time for historical seasons: asked for HOU 2015-16 it returns Mike
D'Antoni (hired 2016-17) and for BKN 2015-16 Kenny Atkinson (hired 2016-17);
the men who actually coached those teams that season (McHale/Bickerstaff,
Hollins/Brown) are absent.  Verified against BBRef in d172_coach_verify.
It is therefore used ONLY as a cross-check, never as the source of truth.

BBRef's `NBA_<end>_coaches.html` lists one row per (coach, team, season) with
that coach's REGULAR-SEASON GAMES IN THAT SEASON, in chronological order
within a team.  A team with k>1 rows had k-1 in-season changes, and the
cumulative game count locates each change EXACTLY in that team's schedule —
which is what turns "coach effect" into an event study with a date.

BBRef team codes are a FOURTH vocabulary (BRK/CHO/PHO/CHH/NOH/NOK/SEA/VAN/
NJN/WSB); they are crosswalked to our modern codes here and every unresolved
code is REPORTED, per nbapred/teams.py's design rule.

Writes data/d172_coach_bbref.csv.  No DB writes.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import pandas as pd                                               # noqa: E402
import requests                                                   # noqa: E402

OUT = ROOT / "data" / "d172_coach_bbref.csv"
CACHE = ROOT / "data" / "raw" / "ext_bbref" / "coaches"
CACHE.mkdir(parents=True, exist_ok=True)
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# BBRef franchise codes -> the code nba_games used THAT SEASON.  BBRef uses one
# code per franchise-era, we use nba_api's; they disagree on five live codes.
BB2US = {"BRK": "BKN", "CHO": "CHA", "PHO": "PHX",
         "NJN": "NJN", "CHH": "CHH", "NOH": "NOH", "NOK": "NOK",
         "SEA": "SEA", "VAN": "VAN", "WSB": "WAS",
         "CHA": "CHA", "NOP": "NOP"}
END_YEARS = list(range(1997, 2027))          # 1996-97 .. 2025-26


def season_str(end: int) -> str:
    return f"{end - 1}-{str(end)[-2:]}"


def fetch(end: int) -> str:
    p = CACHE / f"NBA_{end}_coaches.html"
    if p.exists() and p.stat().st_size > 20000:
        return p.read_text()
    url = f"https://www.basketball-reference.com/leagues/NBA_{end}_coaches.html"
    r = requests.get(url, headers={"User-Agent": UA}, timeout=45)
    r.raise_for_status()
    p.write_text(r.text)
    time.sleep(4.0)                     # BBRef asks for <=20 req/min; be polite
    return r.text


rows, unresolved = [], {}
for end in END_YEARS:
    try:
        h = fetch(end)
    except Exception as e:                                        # noqa: BLE001
        print(f"{season_str(end)}  FETCH FAILED: {e}")
        continue
    t = pd.read_html(io.StringIO(h))[0]
    t.columns = ["_".join([c for c in col if not str(c).startswith("Unnamed")])
                 .strip("_") for col in t.columns]
    t = t[t.Coach.notna() & (t.Coach != "Coach")].copy()
    g_col = "Regular Season_Current Season_G"
    w_col = "Regular Season_Current Season_W"
    l_col = "Regular Season_Current Season_L"
    t = t[["Coach", "Tm", g_col, w_col, l_col]].rename(
        columns={g_col: "g", w_col: "w", l_col: "l"})
    t["season"] = season_str(end)
    t["order"] = t.groupby("Tm").cumcount()          # chronological within team
    t["team_bb"] = t.Tm
    t["team"] = t.Tm.map(lambda x: BB2US.get(x, x))
    for x in set(t.Tm) - set(BB2US):
        if len(x) == 3 and x.isupper():
            pass                                     # matches our code already
    rows.append(t)
    nteams = t.Tm.nunique()
    nchg = int((t.groupby("Tm").size() > 1).sum())
    print(f"{season_str(end)}  coaches={len(t):>3}  teams={nteams:>2}  "
          f"teams_with_change={nchg:>2}  "
          f"G_sum_ok={(t.groupby('Tm').g.sum().between(48, 84)).all()}")

df = pd.concat(rows, ignore_index=True)
df["g"] = pd.to_numeric(df.g, errors="coerce")
df["w"] = pd.to_numeric(df.w, errors="coerce")
df["l"] = pd.to_numeric(df.l, errors="coerce")
df.to_csv(OUT, index=False)
print(f"\nWROTE {OUT}  rows={len(df):,}")

# ---- coverage + vocabulary report -------------------------------------------
print("\nCOVERAGE")
cov = df.groupby("season").agg(coach_rows=("Coach", "size"),
                               teams=("team", "nunique"),
                               distinct_coaches=("Coach", "nunique"))
cov["changes"] = df.groupby("season").apply(
    lambda d: int((d.groupby("team").size() - 1).clip(lower=0).sum()))
print(cov.to_string())
print(f"\nTOTAL coach-team-seasons {len(df):,}; distinct coaches "
      f"{df.Coach.nunique():,}; in-season changes "
      f"{int(cov.changes.sum())}")

print("\nBBREF TEAM VOCABULARY vs OURS")
import duckdb                                                     # noqa: E402
con = duckdb.connect(str(ROOT / "data" / "nba.duckdb"), read_only=True)
ng = {r[0] for r in con.execute(
    "SELECT DISTINCT team_abbrev FROM nba_games WHERE game_id LIKE '002%'"
).fetchall()}
con.close()
bb = set(df.team_bb)
mapped = set(df.team)
print(f"BBRef codes {len(bb)}: {sorted(bb)}")
print(f"mapped  -> {len(mapped)}: unresolved against nba_games = "
      f"{sorted(mapped - ng)}")
print(f"nba_games regular-season codes not produced by BBRef: "
      f"{sorted(ng - mapped)}")
