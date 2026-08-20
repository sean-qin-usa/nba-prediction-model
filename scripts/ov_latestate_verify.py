"""D90 ship-confirm: does the PRODUCTION PORT of the late-state layer
(nbapred/model/latestate.py, wired in production.fit_production) reproduce
the pre-registered gate arm (scripts/ov_latestate_gate.py)?

Registered check: per-season capstone logloss within 0.0005 of the gate
arm's variant (0.59745 / 0.58774 / 0.57901).

Run under OCT_BRIDGE=0: the D84-A October bridge + ps-continuity landed in
production BETWEEN the gate run and this port (independent thread); the gate
arm predates them, so the like-for-like port check disables the bridge. Any
residual inactive-window mismatch is attributed per-game (the only remaining
non-late-state delta is the non-env-gated refit-1 ps-continuity carry
weights, which only touch week-1 games — far outside the gp>=55 window).

Outputs data/ov_latestate_verify.json + per-game CSV. Read-only DB.
"""
import csv
import datetime as dt
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("OCT_BRIDGE", "0")   # like-for-like vs the gate arm
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from nbapred.db import connect
from nbapred.model.composition import CompositionModel
from nbapred.model.production import fit_production

OUT_DIR = REPO / "data"
SEASONS = ("2023-24", "2024-25", "2025-26")
TOL = 0.0005


def season_run(season):
    """prod_by_season.py loop verbatim; p_us now includes the D90 layer."""
    t0 = time.time()
    con = connect(read_only=True)
    pm = con.execute("""SELECT game_id, team_id, player_id, seconds/60.0 AS mins
        FROM player_game_stats WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    played = {(g, t): set(grp.player_id)
              for (g, t), grp in pm.groupby(["game_id", "team_id"])}
    meta = con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl,
        game_date FROM nba_games WHERE season=? AND game_id LIKE '002%'
        AND wl IS NOT NULL ORDER BY game_date""", [season]).fetchdf()
    mkt = {(str(r[0])[:10], r[1], r[2]): r[3] for r in con.execute(
        "SELECT game_date, home, away, p_home_spread FROM odds_market WHERE season_end=?",
        [int(season[:4]) + 1]).fetchall()}
    by, order = {}, []
    for x in meta.itertuples():
        if x.game_id not in by:
            order.append(x.game_id)
        by.setdefault(x.game_id, []).append(x)
    tdates = {}
    for x in meta.itertuples():
        d = x.game_date.date() if hasattr(x.game_date, "date") else x.game_date
        tdates.setdefault(x.team_id, set()).add(d)

    def b2b(tid, d):
        return (d - dt.timedelta(days=1)) in tdates.get(tid, set())

    rows = []
    model = comp = None
    last = None
    for gid in order:
        recs = by[gid]
        if len(recs) != 2:
            continue
        m = recs[0].matchup
        host = m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        h = next((x for x in recs if x.team_abbrev == host), None)
        a = next((x for x in recs if x.team_abbrev != host), None)
        if not h or not a:
            continue
        gd = h.game_date.date() if hasattr(h.game_date, "date") else h.game_date
        if last is None or (gd - last).days >= 7:
            model = fit_production(con, season, before=gd, w_comp=0.7)
            comp = CompositionModel(con, before=gd)
            last = gd
        pmv = mkt.get((str(gd)[:10], h.team_abbrev, a.team_abbrev))
        if pmv is None:
            continue
        outs = {}
        for t in (h.team_id, a.team_id):
            pl = played.get((gid, t), set())
            outs[t] = {p for p, d0 in comp.players.items()
                       if d0["team_id"] == t and (gd - d0["last_played"]).days <= 12
                       and p not in pl}
        p = model.p_home(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id],
                         gd, b2b_home=b2b(h.team_id, gd), b2b_away=b2b(a.team_id, gd))
        rows.append(dict(season=season, game_id=gid, game_date=str(gd)[:10],
                         y=int(h.wl == "W"), p_mkt=float(pmv), p_port=float(p)))
    con.close()
    print(f"[{season}] n={len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    return rows


def ll_vec(y, p):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    all_rows = []
    for s in SEASONS:
        all_rows += season_run(s)
    with open(OUT_DIR / "ov_latestate_verify_pergame.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(all_rows)

    # join to the gate arm
    gate = {}
    with open(OUT_DIR / "ov_latestate_pergame.csv") as f:
        for r in csv.DictReader(f):
            gate[(r["season"], r["game_id"])] = (
                float(r["p_var"]), float(r["p_ctrl"]), int(r["active"]))
    y = np.array([r["y"] for r in all_rows])
    seas = np.array([r["season"] for r in all_rows])
    p_port = np.array([r["p_port"] for r in all_rows])
    gv = np.array([gate[(r["season"], r["game_id"])][0] for r in all_rows])
    gc = np.array([gate[(r["season"], r["game_id"])][1] for r in all_rows])
    act = np.array([gate[(r["season"], r["game_id"])][2] for r in all_rows], bool)
    ll_port = ll_vec(y, p_port)
    ll_gate = ll_vec(y, gv)

    per = {}
    for s in SEASONS:
        m = seas == s
        per[s] = dict(
            ll_port=round(float(ll_port[m].mean()), 5),
            ll_gate_arm=round(float(ll_gate[m].mean()), 5),
            delta=round(float(ll_port[m].mean() - ll_gate[m].mean()), 5),
            within_tol=bool(abs(ll_port[m].mean() - ll_gate[m].mean()) <= TOL))
    inact = ~act
    res = dict(
        env=dict(OCT_BRIDGE=os.environ.get("OCT_BRIDGE"),
                 LATE_STATE=os.environ.get("LATE_STATE", "1")),
        tolerance_per_season=TOL,
        per_season=per,
        all_within_tol=bool(all(v["within_tol"] for v in per.values())),
        pergame_vs_gate=dict(
            n=len(all_rows),
            max_abs_diff_inactive=float(np.abs(p_port - gv)[inact].max()),
            max_abs_diff_active=float(np.abs(p_port - gv)[act].max()),
            mean_abs_diff_active=float(np.abs(p_port - gv)[act].mean()),
            n_inactive_offparity_1e6=int((np.abs(p_port - gv)[inact] > 1e-6).sum()),
            max_abs_diff_inactive_vs_ctrl=float(np.abs(p_port - gc)[inact].max())),
    )
    with open(OUT_DIR / "ov_latestate_verify.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
