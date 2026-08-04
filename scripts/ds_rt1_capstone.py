"""RT1 (HEADLINE) — is "2023-24 is bad" a DATA-STARVATION ARTIFACT?

2023-24 was our FIRST eval season. Every published number for it was measured
with a trailing corpus that had NO 2021-22 rows in `nba_games`, so
fit_schedule_layer()'s 730-day window was half-empty for the whole first half
of the season and its n/(n+600) shrink pulled hard toward the SCHED_PRIOR
constants. 2022-23 could not be an eval season at all (continuity_map returns
None with no prior season -> no D62 carry).

DESIGN — two arms, ONE run, per-game paired:
  ARM_FULL     : the corpus after scripts/ds_ingest_schedules.py (2019-20..)
  ARM_STARVED  : nba_games truncated to season >= '2022-23' = the exact state
                 every gate from D46 through D94 was measured in.
Production code is IDENTICAL in both arms (nbapred/ untouched); only the view
under `nba_games` differs. Walk-forward, refit cadence, OUT-set convention and
market join are VERBATIM scripts/prod_by_season.py.

Eval seasons: 2022-23 (newly scorable), 2023-24, 2024-25, 2025-26.
2024-25/2025-26 are the NEGATIVE CONTROL: their 730d windows were already
fully covered, so ARM_FULL must reproduce ARM_STARVED there.

Validation anchors:
  * ARM_STARVED 2023-24..2025-26 must reproduce data/capstone_pergame.csv
    (the shipped bridge+layer capstone) to ~1e-14.
  * ARM_STARVED 2022-23 must reproduce the pg_* re-gates' control (ll 0.6326).

Gate (pre-registered here): paired bootstrap 2000x, seed 20260731, on
per-game log-loss delta (ARM_STARVED - ARM_FULL; positive = the corpus fix
HELPS). Reported pooled, per season, and on the early window (either gp<20)
where the schedule-layer shrink bites hardest.

Read-only DB. Usage: python scripts/ds_rt1_capstone.py
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ds_corpus import arm_connection, paired_bootstrap  # noqa: E402
from nbapred.eval.metrics import log_loss  # noqa: E402
from nbapred.model.production import DEAD_GP, DEAD_WPCT, fit_production  # noqa: E402

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]   # paired gate seasons
EXTRA_FULL = ["2021-22"]   # newly scorable; FULL arm only (STARVED has no rows)
STARVED_FLOOR = "2022-23"


def season_run(con, season):
    """VERBATIM prod_by_season.season_run, default (availability-oracle) tier,
    with the connection injected instead of opened."""
    pm = con.execute("""SELECT game_id, team_id, player_id, seconds/60.0 AS mins
        FROM player_game_stats WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    played = {(g, t): set(grp.player_id) for (g, t), grp in pm.groupby(["game_id", "team_id"])}
    meta = con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl, game_date
        FROM nba_games WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL
        ORDER BY game_date""", [season]).fetchdf()
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
        return (d - _dt.timedelta(days=1)) in tdates.get(tid, set())

    hist = {}
    for x in meta.itertuples():
        d = x.game_date.date() if hasattr(x.game_date, "date") else x.game_date
        hist.setdefault(x.team_id, []).append((d, x.wl == "W"))
    for t in hist:
        hist[t].sort()

    gp_hist = {}
    for t, lst in hist.items():
        gp_hist[t] = [d for d, _ in lst]

    rows = []
    model = None
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
            from nbapred.model.composition import CompositionModel
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
        y = int(h.wl == "W")
        p = model.p_home(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id], gd,
                         b2b_home=b2b(h.team_id, gd), b2b_away=b2b(a.team_id, gd))
        gp_h = sum(1 for d in gp_hist.get(h.team_id, []) if d < gd)
        gp_a = sum(1 for d in gp_hist.get(a.team_id, []) if d < gd)
        rows.append({"season": season, "game_id": gid, "game_date": str(gd)[:10],
                     "home": h.team_abbrev, "away": a.team_abbrev, "y": y,
                     "p_us": float(p), "p_mkt": float(pmv),
                     "gp_home": gp_h, "gp_away": gp_a,
                     "tsd": float(model.tank_diff(h.team_id, a.team_id, gd)),
                     "k": float(model.tank_k)})
    return rows


def run_arm(label, min_season, seasons=SEASONS):
    con = arm_connection(min_season)
    out = {}
    for s in seasons:
        t0 = time.time()
        out[s] = season_run(con, s)
        print(f"  [{label}] {s}: n={len(out[s])} ({time.time()-t0:.0f}s)", flush=True)
    con.close()
    return out


def ll_of(rows):
    y = np.array([r["y"] for r in rows])
    return (round(log_loss(y, np.array([r["p_us"] for r in rows])), 5),
            round(log_loss(y, np.array([r["p_mkt"] for r in rows])), 5))


def per_game_ll(r, key):
    p = min(max(r[key], 1e-12), 1 - 1e-12)
    return -(r["y"] * np.log(p) + (1 - r["y"]) * np.log(1 - p))


def main():
    print("ARM_STARVED (nba_games >= 2022-23 = the pre-fix state)", flush=True)
    starved = run_arm("STARVED", STARVED_FLOOR)
    print("ARM_FULL (corpus after ds_ingest_schedules)", flush=True)
    full = run_arm("FULL", None, SEASONS + EXTRA_FULL)

    res = {"design": "same-script paired arms; only the nba_games view differs",
           "gate": "paired bootstrap 2000x seed 20260731, delta = ll(STARVED) - ll(FULL)",
           "seasons": {}, "anchors": {}}

    # --- validation anchor vs the shipped capstone -------------------------
    base = {}
    cap = ROOT / "data" / "capstone_pergame.csv"
    if cap.exists():
        for r in csv.DictReader(open(cap)):
            base[(r["season"], r["game_id"])] = float(r["p_us"])
    maxd = {}
    for s in SEASONS:
        ds = [abs(r["p_us"] - base[(s, r["game_id"])])
              for r in starved[s] if (s, r["game_id"]) in base]
        maxd[s] = {"n": len(ds), "max_abs_dp": max(ds) if ds else None}
    res["anchors"]["starved_vs_capstone_pergame.csv"] = maxd

    pooled_d, pooled_rows = [], []
    for s in SEASONS:
        fs = {r["game_id"]: r for r in full[s]}
        pairs = [(r, fs[r["game_id"]]) for r in starved[s] if r["game_id"] in fs]
        d = np.array([per_game_ll(cs, "p_us") - per_game_ll(cf, "p_us")
                      for cs, cf in pairs])
        us_s, mk = ll_of([p[0] for p in pairs])
        us_f, _ = ll_of([p[1] for p in pairs])
        res["seasons"][s] = {
            "n": len(pairs),
            "ll_starved": us_s, "ll_full": us_f, "ll_mkt": mk,
            "gap_starved": round(us_s - mk, 5), "gap_full": round(us_f - mk, 5),
            "gap_shrink": round((us_s - mk) - (us_f - mk), 5),
            "gate": paired_bootstrap(d),
        }
        early = np.array([i for i, (cs, _) in enumerate(pairs)
                          if cs["gp_home"] < 20 or cs["gp_away"] < 20])
        if len(early):
            res["seasons"][s]["early_gp_lt20"] = paired_bootstrap(d[early])
        pooled_d.append(d)
        pooled_rows.extend(pairs)
        print(f"{s}: starved {us_s} / full {us_f} / mkt {mk} | "
              f"gap {us_s-mk:+.5f} -> {us_f-mk:+.5f} | "
              f"delta {res['seasons'][s]['gate']}", flush=True)

    D = np.concatenate(pooled_d)
    res["pooled"] = paired_bootstrap(D)
    res["pooled_2326"] = paired_bootstrap(
        np.concatenate([pooled_d[i] for i, s in enumerate(SEASONS) if s != "2022-23"]))
    early_idx = np.array([i for i, (cs, _) in enumerate(pooled_rows)
                          if cs["gp_home"] < 20 or cs["gp_away"] < 20])
    res["pooled_early_gp_lt20"] = paired_bootstrap(D[early_idx])
    print("POOLED", res["pooled"], flush=True)
    print("POOLED 23-26 only", res["pooled_2326"], flush=True)
    print("POOLED early gp<20", res["pooled_early_gp_lt20"], flush=True)

    # newly-scorable seasons (FULL arm only — STARVED has no rows there)
    for s in EXTRA_FULL:
        if full.get(s):
            us, mk = ll_of(full[s])
            res["seasons"][s] = {"n": len(full[s]), "ll_full": us, "ll_mkt": mk,
                                 "gap_full": round(us - mk, 5),
                                 "note": "FULL arm only — newly scorable season, "
                                         "no paired control (STARVED corpus has no rows); "
                                         "carry into it is degraded (2020-21 "
                                         "player_game_stats 780/1080)"}
            print(f"{s} (FULL only): {us} vs mkt {mk} | gap {us-mk:+.5f}", flush=True)

    with open(ROOT / "data" / "ds_rt1_pergame.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["season", "game_id", "game_date", "home", "away", "y",
                    "p_starved", "p_full", "p_mkt", "gp_home", "gp_away", "tsd"])
        for s in SEASONS:
            fs = {r["game_id"]: r for r in full[s]}
            for r in starved[s]:
                if r["game_id"] in fs:
                    w.writerow([s, r["game_id"], r["game_date"], r["home"], r["away"],
                                r["y"], r["p_us"], fs[r["game_id"]]["p_us"], r["p_mkt"],
                                r["gp_home"], r["gp_away"], r["tsd"]])
    json.dump(res, open(ROOT / "data" / "ds_rt1_results.json", "w"), indent=1)
    print("wrote data/ds_rt1_results.json", flush=True)


if __name__ == "__main__":
    main()
