#!/usr/bin/env python3
"""D172 TASK 2 step 1 — pull coach-by-team-season, $0, from nba_api.

`commonteamroster` returns a second result set (Coaches) with COACH_TYPE
'Head Coach'.  It reports the coach(es) ATTACHED TO THE TEAM-SEASON but gives
no dates, so it identifies a SPELL, not a change point.  The change point comes
from Basketball-Reference's per-season coaches table (d172_coach_bbref.py),
which reports each coach's GAMES with that team that season; the cumulative
game count locates the switch exactly in the schedule.

Writes data/d172_coach_nbaapi.csv.  No DB writes.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import pandas as pd                                               # noqa: E402
from nba_api.stats.endpoints import commonteamroster              # noqa: E402
from nba_api.stats.static import teams as _t                      # noqa: E402

OUT = ROOT / "data" / "d172_coach_nbaapi.csv"
SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(1996, 2026)]
TEAMS = _t.get_teams()

rows, fails = [], []
t0 = time.time()
for si, s in enumerate(SEASONS):
    for tm in TEAMS:
        for attempt in range(3):
            try:
                d = commonteamroster.CommonTeamRoster(
                    team_id=tm["id"], season=s, timeout=45).get_data_frames()
                c = d[1]
                if len(c):
                    c = c.copy()
                    c["season"] = s
                    c["team_abbrev"] = tm["abbreviation"]
                    rows.append(c)
                break
            except Exception as e:                                # noqa: BLE001
                if attempt == 2:
                    fails.append((s, tm["abbreviation"], str(e)[:80]))
                else:
                    time.sleep(2.0 * (attempt + 1))
        time.sleep(0.45)
    done = (si + 1) * len(TEAMS)
    print(f"{s}  cum_rows={sum(len(x) for x in rows):>6}  "
          f"fails={len(fails):>3}  {time.time() - t0:.0f}s", flush=True)

df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
df.to_csv(OUT, index=False)
print(f"\nWROTE {OUT}  rows={len(df):,}  fails={len(fails)}")
if fails:
    print("FAILS (first 20):", fails[:20])
if len(df):
    hc = df[df.COACH_TYPE == "Head Coach"]
    print(f"head-coach rows={len(hc):,}  "
          f"team-seasons={hc.groupby(['season','team_abbrev']).ngroups:,}  "
          f"distinct coaches={hc.COACH_ID.nunique()}")
    cov = (hc.groupby("season")
             .agg(team_seasons=("team_abbrev", "nunique"),
                  hc_rows=("COACH_ID", "size"),
                  coaches=("COACH_ID", "nunique")))
    print(cov.to_string())
