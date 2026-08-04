#!/usr/bin/env python3
"""STALENESS DISCLOSURE (D131/D134 class): how much does the D144 ROLE window
overlap the absence-ramp window that landed in props.py DURING this run?

A concurrent agent shipped `absence_ramp` / `PROPS_ABSENCE_RAMP` (default ON)
into `player_rates_from_stats` at 2026-08-01 23:54:57 — 8 s AFTER the D144
holdout gate wrote its output, and ~9 min after the dev gate. Both D144 gate
runs therefore scored against the PRE-absence-ramp production control. The
gates are internally consistent (same control in both splits, replica assertion
bitwise) but the control is now STALE relative to today's production.

The two terms are both proj_min LOCATION corrections on populations that may
overlap heavily: "missed >=5 of the team's last 10" and "role flipped in the
immediately-prior game" are different events but plausibly the same players.
This script measures the overlap so the D144 entry can state it as a number
instead of a worry. Read-only; writes data/ad_overlap_absence.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from nbapred.db import connect
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("ad_role_gate", ROOT / "scripts" / "ad_role_gate.py")
G = _ilu.module_from_spec(_spec); _spec.loader.exec_module(G)


def main():
    con = connect(read_only=True)
    df = G.load_corpus(con)
    df["team_id"] = df["team_id"].astype(int)
    roles = G.load_roles()
    byp = G.build_index(df, roles)

    # team schedules, as ord arrays
    tg = con.execute("""
        SELECT season, team_id, game_date FROM nba_games
        WHERE game_id LIKE '002%' ORDER BY season, team_id, game_date
    """).fetchdf()
    tg["ord"] = tg["game_date"].astype("datetime64[ns]").values.astype(
        "datetime64[D]").astype(int)
    tsched = {(s, int(t)): np.sort(g["ord"].to_numpy())
              for (s, t), g in tg.groupby(["season", "team_id"], sort=False)}
    con.close()

    # a player's own played-ords per (season, team)
    own = {}
    for r in df.itertuples():
        own.setdefault((int(r.player_id), r.season, int(r.team_id)), []).append(int(r.ord))
    own = {k: np.sort(np.array(v)) for k, v in own.items()}

    def miss10(pid, season, team, day):
        sch = tsched.get((season, int(team)))
        if sch is None:
            return 0
        prior = sch[sch < day][-10:]
        o = own.get((pid, season, int(team)))
        if o is None or not len(o):
            return 0
        first = o.min()
        oset = set(o.tolist())
        return int(sum(1 for d in prior if d >= first and d not in oset))

    SEASONS = G.DEV + G.HOLDOUT
    rows = []
    for r in df[df["season"].isin(SEASONS)].itertuples():
        st = G.row_state(byp, int(r.player_id), r.ord, r.season)
        if st is None:
            continue
        proj, nh, gp, rb, mb = st
        if nh < 8 or proj < 20:
            continue
        rows.append((rb, miss10(int(r.player_id), r.season, int(r.team_id), int(r.ord)), gp))
    rb = np.array([x[0] for x in rows])
    m10 = np.array([x[1] for x in rows])
    gp = np.array([x[2] for x in rows])
    active = np.isin(rb, ("PROMOTED", "DEMOTED"))
    abs_on = m10 >= 5                     # the absence ramp's active window

    out = dict(
        n=len(rows),
        role_active=int(active.sum()), role_active_share=float(active.mean()),
        absence_active=int(abs_on.sum()), absence_active_share=float(abs_on.mean()),
        both=int((active & abs_on).sum()),
        share_of_role_rows_also_absence=float(abs_on[active].mean()),
        share_of_absence_rows_also_role=float(active[abs_on].mean()),
        jaccard=float((active & abs_on).sum() / (active | abs_on).sum()),
        by_bucket={b: dict(n=int((rb == b).sum()),
                           absence_share=float(abs_on[rb == b].mean()))
                   for b in ("PROMOTED", "DEMOTED", "STABLE", "NA")},
        role_active_and_gp_ge20=float((gp >= 20)[active].mean()),
    )
    print(json.dumps(out, indent=2))
    (ROOT / "data" / "ad_overlap_absence.json").write_text(json.dumps(out, indent=2))
    print("AD_OVERLAP_DONE")


if __name__ == "__main__":
    main()
