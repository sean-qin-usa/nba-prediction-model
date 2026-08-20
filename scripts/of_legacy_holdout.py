"""OF-2 — WHOLE-MODEL TRANSFER: pre-campaign (D23-era) stack vs CURRENT
production, on the quasi-holdout seasons the campaign never scored.

PRE-CAMPAIGN CONTROL = scripts/_legacy_baseline_run.py's `fit_production_legacy`
imported VERBATIM (the D23 stack: margin = 0.5*FourFactors(home-in) +
0.5*Composition(hardcoded HOME_EDGE=3.0), ratings+cold-start prior as the
pre-ff-ready fallback; NO schedule layer, NO carry, NO tank, NO late-state, NO
October bridge). Talent is PIT DARKO (`CompositionModel(con, before=gd)`), the
same convention the D44-era legacy reproduction used — a literal "snapshot
DARKO" arm would carry end-of-season talent into October and would flatter the
legacy arm with leakage, so PIT is the honest control and is stated as such.

CURRENT = the `p_base` column of data/of_transfer_pergame.csv (the literal
production predictor, anchored at every refit in OF-1), joined on game_id, so
the comparison is per-game paired on an identical game universe.

Gate: paired bootstrap 2000x seed 20260801 on per-game log-loss delta
(legacy - current; POSITIVE = the campaign HELPS), reported per season and per
season group. If the campaign's gains are real they transfer to 2022-23 /
2021-22 at a similar magnitude; if they were selected on 2024-25/2025-26 they
shrink or vanish there.

Read-only DB. Usage: python scripts/of_legacy_holdout.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from _legacy_baseline_run import fit_production_legacy  # noqa: E402
from nbapred.db import connect  # noqa: E402
from nbapred.eval.metrics import log_loss  # noqa: E402
from nbapred.model.composition import CompositionModel  # noqa: E402
from of_transfer_ablation import (GROUPS, SEASONS, ll_pg,  # noqa: E402
                                  paired_bootstrap)


def season_run(con, season):
    pm = con.execute("""SELECT game_id, team_id, player_id FROM player_game_stats
        WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    played = {(g, t): set(grp.player_id)
              for (g, t), grp in pm.groupby(["game_id", "team_id"])}
    meta = con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl, game_date
        FROM nba_games WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL
        ORDER BY game_date""", [season]).fetchdf()
    mkt = {(str(r[0])[:10], r[1], r[2]): r[3] for r in con.execute(
        "SELECT game_date, home, away, p_home_spread FROM odds_market "
        "WHERE season_end=?", [int(season[:4]) + 1]).fetchall()}
    by, order = {}, []
    for x in meta.itertuples():
        if x.game_id not in by:
            order.append(x.game_id)
        by.setdefault(x.game_id, []).append(x)
    rows, model, comp, last = [], None, None, None
    for gid in order:
        recs = by[gid]
        if len(recs) != 2:
            continue
        mm = recs[0].matchup
        host = mm.split("@")[-1].strip() if "@" in mm else mm.split("vs.")[0].strip()
        h = next((x for x in recs if x.team_abbrev == host), None)
        a = next((x for x in recs if x.team_abbrev != host), None)
        if not h or not a:
            continue
        gd = h.game_date.date() if hasattr(h.game_date, "date") else h.game_date
        if last is None or (gd - last).days >= 7:
            model = fit_production_legacy(con, season, before=gd, w_comp=0.7)
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
        p = model.p_home(h.team_id, a.team_id, outs[h.team_id],
                         outs[a.team_id], gd)
        rows.append({"season": season, "game_id": gid, "y": int(h.wl == "W"),
                     "p_legacy": float(p), "p_mkt": float(pmv)})
    return rows


def main():
    cache = ROOT / "data" / "of_legacy_raw.csv"
    if cache.exists():
        allrows = [{**r, "y": int(r["y"]), "p_legacy": float(r["p_legacy"]),
                    "p_mkt": float(r["p_mkt"])}
                   for r in csv.DictReader(open(cache))]
        print(f"reusing {cache.name} n={len(allrows)}", flush=True)
    else:
        con = connect(read_only=True)
        allrows = []
        for s in SEASONS:
            t0 = time.time()
            rr = season_run(con, s)
            allrows += rr
            print(f"{s}: legacy n={len(rr)} ({time.time()-t0:.0f}s)", flush=True)
        con.close()
        with open(cache, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
            w.writeheader()
            w.writerows(allrows)
    cur = {}
    with open(ROOT / "data" / "of_transfer_pergame.csv") as f:
        for r in csv.DictReader(f):
            cur[r["game_id"]] = r

    paired = []
    for r in allrows:
        c = cur.get(r["game_id"])
        if c is None:
            continue
        paired.append({"season": r["season"], "game_id": r["game_id"],
                       "y": r["y"], "p_legacy": r["p_legacy"],
                       "p_cur": float(c["p_base"]), "p_mkt": r["p_mkt"],
                       "gp_home": int(c["gp_home"]), "gp_away": int(c["gp_away"])})
    print(f"paired n={len(paired)} of legacy {len(allrows)}", flush=True)

    with open(ROOT / "data" / "of_legacy_pergame.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0].keys()))
        w.writeheader()
        w.writerows(paired)

    LN2 = 0.6931471805599453
    res = {"design": "legacy D23 stack (PIT DARKO) vs current production, "
                     "per-game paired, identical universe",
           "seasons": {}, "groups": {}}
    for s in SEASONS:
        rs = [r for r in paired if r["season"] == s]
        y = np.array([r["y"] for r in rs])
        ll_l = float(log_loss(y, np.array([r["p_legacy"] for r in rs])))
        ll_c = float(log_loss(y, np.array([r["p_cur"] for r in rs])))
        ll_m = float(log_loss(y, np.array([r["p_mkt"] for r in rs])))
        d = np.array([ll_pg(r, "p_legacy") - ll_pg(r, "p_cur") for r in rs])
        res["seasons"][s] = {
            "n": len(rs), "ll_legacy": round(ll_l, 5), "ll_current": round(ll_c, 5),
            "ll_mkt": round(ll_m, 5),
            "gap_legacy": round(ll_l - ll_m, 5), "gap_current": round(ll_c - ll_m, 5),
            "norm_gap_legacy_pct": round(100 * (ll_l - ll_m) / (LN2 - ll_m), 2),
            "norm_gap_current_pct": round(100 * (ll_c - ll_m) / (LN2 - ll_m), 2),
            "campaign_gain": paired_bootstrap(d)}
        print(f"{s}: legacy {ll_l:.5f} cur {ll_c:.5f} mkt {ll_m:.5f} | "
              f"norm {res['seasons'][s]['norm_gap_legacy_pct']}% -> "
              f"{res['seasons'][s]['norm_gap_current_pct']}% | "
              f"gain {res['seasons'][s]['campaign_gain']}", flush=True)
    for g, ss in GROUPS.items():
        rs = [r for r in paired if r["season"] in ss]
        d = np.array([ll_pg(r, "p_legacy") - ll_pg(r, "p_cur") for r in rs])
        res["groups"][g] = paired_bootstrap(d)
        early = np.array([i for i, r in enumerate(rs)
                          if r["gp_home"] < 20 or r["gp_away"] < 20])
        res["groups"][g + "_early_gp_lt20"] = (paired_bootstrap(d[early])
                                               if len(early) else None)
        print(f"{g}: {res['groups'][g]}", flush=True)
    json.dump(res, open(ROOT / "data" / "of_legacy_results.json", "w"), indent=1)
    print("wrote data/of_legacy_results.json", flush=True)


if __name__ == "__main__":
    main()
