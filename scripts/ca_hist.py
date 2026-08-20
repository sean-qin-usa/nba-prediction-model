#!/usr/bin/env python3
"""PART A4 — what the carried pile BUYS (or costs) on the HISTORICAL seasons.

Prereg data/carryall_prereg.md sha256 9a4a414d...59bd9b, §4.

The certified artifact covers only 2021-26, so the control here is a same-run
`fit_production` walk-forward at D132 environment defaults, weekly refit, the
`prod_by_season.py` loop verbatim.  Treatments are the same additive
schedule-layer arms as `ca_ladder.py`, so control and treatment share `m_base`
exactly and the pairing is exact.

The scorable set is re-derived AT RUN TIME from the sibling agent's
`data/history_scorable.json` when present (D152 chains are still landing
seasons), else from a local data-sufficiency probe.

  python scripts/ca_hist.py [season ...]
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import nbapred.threads as _t  # noqa: E402

_t.pin(1)
import numpy as np  # noqa: E402

from ca_bank import TERM_NAMES, Layer, load_bank  # noqa: E402
from ca_ladder import arms, ll, refit_dates  # noqa: E402
from nbapred.model.composition import CompositionModel  # noqa: E402
from nbapred.model.production import SCALE, fit_production, sigmoid  # noqa: E402

OUT = REPO / "data" / "carryall_hist.json"
ROWS = REPO / "data" / "carryall_hist_rows.npz"


def connect_retry(attempts=120, wait_s=60):
    from nbapred.db import connect as _c
    last = None
    for _ in range(attempts):
        try:
            return _c(read_only=True)
        except Exception as e:                                  # noqa: BLE001
            last = e
            print(f"DB connect failed ({e}); retry in {wait_s}s", flush=True)
            time.sleep(wait_s)
    raise last


def scorable_set(con):
    f = REPO / "data" / "history_scorable.json"
    if f.exists():
        d = json.load(open(f))
        return list(d["poolable"]), list(d.get("strata", []))
    # local fallback: prior-season box depth >= 20k rows and own coverage ~1.0
    rows = con.execute("""
        SELECT g.season, count(DISTINCT s.game_id) boxed
        FROM player_game_stats s
        JOIN nba_games g ON g.game_id=s.game_id AND g.team_id=s.team_id
        WHERE s.game_id LIKE '002%' GROUP BY 1""").fetchall()
    box = {r[0]: r[1] for r in rows}
    sched = {r[0]: r[1] for r in con.execute(
        """SELECT season, count(DISTINCT game_id) FROM nba_games
           WHERE game_id LIKE '002%' GROUP BY 1""").fetchall()}
    pool = []
    for s in sorted(sched):
        y0 = int(s[:4])
        prev = f"{y0-1}-{str(y0)[-2:]}"
        if box.get(prev, 0) >= 900 and box.get(s, 0) >= 0.99 * sched[s]:
            pool.append(s)
    return pool, []


def season_control(con, season):
    """`prod_by_season.py`'s walk-forward, returning p_ctrl per game_id."""
    pm = con.execute("""SELECT game_id, team_id, player_id FROM player_game_stats
        WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    played = {(g, t): set(grp.player_id)
              for (g, t), grp in pm.groupby(["game_id", "team_id"])}
    meta = con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl, game_date
        FROM nba_games WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL
        ORDER BY game_date""", [season]).fetchdf()
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
        return (d - _dt.timedelta(days=1)) in tdates.get(tid, set())

    out = {}
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
        outs = {}
        for t in (h.team_id, a.team_id):
            pl = played.get((gid, t), set())
            outs[t] = {p for p, d0 in comp.players.items()
                       if d0["team_id"] == t
                       and (gd - d0["last_played"]).days <= 12 and p not in pl}
        out[gid] = float(model.p_home(
            h.team_id, a.team_id, outs[h.team_id], outs[a.team_id], gd,
            b2b_home=b2b(h.team_id, gd), b2b_away=b2b(a.team_id, gd)))
    return out


if __name__ == "__main__":
    con = connect_retry()
    bank = load_bank(con)
    pool, strata = scorable_set(con)
    want = sys.argv[1:] or (pool + strata)
    want = [s for s in want if s in set(pool) | set(strata)]
    print(f"[hist] poolable {pool}\n[hist] strata {strata}\n[hist] running {want}",
          flush=True)

    A = arms()
    gid_all, seas_all, y_all, p0_all, dmm = [], [], [], [], {k: [] for k in A}
    per_season = {}
    for s in want:
        t0 = time.time()
        ctrl = season_control(con, s)
        order, dates, dm, coefs = ca_run = __import__("ca_ladder").run(
            bank, [s], verbose=False)
        gids = bank["gid"][order].astype(str)
        keep = np.array([g in ctrl for g in gids])
        idx = np.where(keep)[0]
        p0 = np.array([ctrl[g] for g in gids[idx]])
        y = bank["y"][order][idx]
        gid_all += list(gids[idx]); seas_all += [s] * len(idx)
        y_all += list(y); p0_all += list(p0)
        for k in A:
            dmm[k] += list(dm[k][idx])
        per_season[s] = dict(n=int(len(idx)), ll=float(ll(y, p0).mean()),
                             secs=round(time.time() - t0, 1),
                             home_edge_mean=float(np.mean(
                                 [c["base5"][0] for c in coefs["pile:k1"]])))
        print(f"[hist] {s} n={len(idx)} ll={per_season[s]['ll']:.5f} "
              f"({per_season[s]['secs']}s)", flush=True)

    y = np.array(y_all); p0 = np.array(p0_all)
    seas = np.array(seas_all); m_base = SCALE * np.log(p0 / (1 - p0))
    from nbapred.eval.splits import Panel, full_report
    res = {"prereg_sha256": "9a4a414db294ba44908b4a4ee5f0bd490e0b2d0094"
                            "293eb0f2103a101459bd9b",
           "poolable": pool, "strata": strata, "ran": want,
           "per_season": per_season, "n": int(len(y)), "arms": {}}
    poolmask = np.isin(seas, pool)
    print(f"\n{'arm':26s} {'n':>6s} {'rms(dm)':>8s} {'delta':>10s} "
          f"{'season-cluster CI':>24s} {'MDE80':>8s}")
    for nm in sorted(A):
        d = np.array(dmm[nm])
        p1 = 1.0 / (1.0 + np.exp(-(m_base + d) / SCALE))
        st = {}
        for tag, msk in (("poolable", poolmask),) + tuple(
                (f"stratum:{s}", seas == s) for s in strata if (seas == s).any()):
            if msk.sum() < 100:
                continue
            pan = Panel.from_logloss(seas[msk], y[msk], p0[msk], p1[msk],
                                     label=f"{nm}|{tag}")
            rep = full_report(pan, B=2000, seed=20260802)
            cl = rep["clustering"]["season_cluster_boot"]
            st[tag] = dict(n=int(msk.sum()),
                           rms_dm=float(np.sqrt((d[msk] ** 2).mean())),
                           delta=rep["pooled"]["est"],
                           iid_lo=rep["pooled"]["lo"], iid_hi=rep["pooled"]["hi"],
                           cl_lo=cl["lo"], cl_hi=cl["hi"],
                           t_lo=rep["clustering"]["season_mean_t"]["lo"],
                           t_hi=rep["clustering"]["season_mean_t"]["hi"],
                           mde80=rep["pooled_mde80"],
                           per_season={x["season"]: x["est"]
                                       for x in rep["per_season"]},
                           ro=[f["fold"]["est"]
                               for f in rep["rolling_origin"]["folds"]],
                           I2=rep["era"]["I2"],
                           era_stable=rep["era"]["era_stable"],
                           flags=rep["verdict"]["flags"])
        res["arms"][nm] = st
        if "poolable" in st:
            v = st["poolable"]
            print(f"{nm:26s} {v['n']:6d} {v['rms_dm']:8.4f} {v['delta']:+10.5f} "
                  f"({v['cl_lo']:+10.5f},{v['cl_hi']:+10.5f}) {v['mde80']:8.5f}")
    json.dump(res, open(OUT, "w"), indent=1)
    np.savez_compressed(ROWS, gid=np.array(gid_all), seas=seas, y=y, p0=p0,
                        **{("dm_" + k): np.array(dmm[k]) for k in A})
    print(f"\nwrote {OUT.name} + {ROWS.name}")
    con.close()
