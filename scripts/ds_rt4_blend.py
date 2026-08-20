"""RT4 — D22 FITTED BLEND WEIGHTS re-tested at 4-5 season power.

D22 (2026-07-28) rejected fitted blend weights in favour of the fixed
0.5*FourFactors + 0.5*Composition, and the D46 re-gate line records "fitted
blend weights still ns". BOTH were measured when the fittable corpus was ONE
season (n~500 games) — a textbook data-starvation rejection: a single scalar
fitted on 500 binary outcomes has a standard error large enough that a fixed
0.5 wins on variance alone, regardless of where the truth sits.

This re-test pools the whole post-fix corpus (2021-22..2025-26).

PASS 1 (dump): walk-forward per-game components on the FULL corpus —
  m_tot (the shipped production margin, recovered exactly from RT1's p_full),
  fm    = ff.margin_neutral(home, away)   [FourFactors leg]
  cm    = comp margin (incl. the D84-A October bridge substitution)
  rest  = m_tot - 0.5*fm - 0.5*cm         [sched + tank + late-state + fallback]
PASS 2 (gate, offline): variant margin = w_d*fm + (1-w_d)*cm + rest, with w_d
fitted WALK-FORWARD at each weekly refit by logistic MLE on every completed
game strictly before d, w constrained to [0,1] (the sign constraint: neither
leg may enter negatively) and shrunk toward the shipped 0.5 by n/(n+600) —
the same shrink convention as SCHED_SHRINK / FORM_SHRINK / K_SHRINK.
w = 0.5 until 200 fittable games (burn-in).

PRIMARY = the shrunk arm (one pre-registered config). The unshrunk MLE arm is
reported as a diagnostic only. Gate: paired bootstrap 2000x seed 20260731 vs
the same-run control (the shipped fixed-0.5 margin); ship only if the pooled
CI excludes 0.

Read-only DB. Usage:
  python scripts/ds_rt4_blend.py --dump     (pass 1, ~5 min)
  python scripts/ds_rt4_blend.py            (pass 1 if needed, then gate)
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from ds_corpus import arm_connection, paired_bootstrap  # noqa: E402
from nbapred.model.composition import CompositionModel  # noqa: E402
from nbapred.model.production import SCALE, fit_production, sigmoid  # noqa: E402

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
DUMP = ROOT / "data" / "ds_rt4_components.csv"
BLEND_SHRINK = 600.0
BURN_IN = 200


def dump_components():
    con = arm_connection(None)
    rows = []
    pm = con.execute("""SELECT game_id, team_id, player_id FROM player_game_stats
        WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    played = {(g, t): set(grp.player_id) for (g, t), grp in pm.groupby(["game_id", "team_id"])}
    for season in SEASONS:
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
        tdates = {}
        for x in meta.itertuples():
            d = x.game_date.date() if hasattr(x.game_date, "date") else x.game_date
            tdates.setdefault(x.team_id, set()).add(d)
        model = comp = ffobj = None
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
                ffobj = _ff_of(model)
                last = gd
            pmv = mkt.get((str(gd)[:10], h.team_abbrev, a.team_abbrev))
            if pmv is None:
                continue
            outs = {}
            for t in (h.team_id, a.team_id):
                pl = played.get((gid, t), set())
                outs[t] = {p for p, d0 in comp.players.items()
                           if d0["team_id"] == t
                           and (gd - d0["last_played"]).days <= 12 and p not in pl}
            b2h = (gd - _dt.timedelta(days=1)) in tdates.get(h.team_id, set())
            b2a = (gd - _dt.timedelta(days=1)) in tdates.get(a.team_id, set())
            m_tot = model.margin(h.team_id, a.team_id, outs[h.team_id],
                                 outs[a.team_id], gd, b2b_home=b2h, b2b_away=b2a)
            if not (ffobj is not None and ffobj.ready):
                continue          # fallback branch: no fm/cm blend to re-weight
            fm = ffobj.margin_neutral(h.team_id, a.team_id)
            cm = _cm_of(model, comp, h, a, outs, gd)
            rows.append({"season": season, "game_id": gid, "game_date": str(gd)[:10],
                         "y": int(h.wl == "W"), "p_mkt": float(pmv),
                         "m_tot": float(m_tot), "fm": float(fm), "cm": float(cm),
                         "rest": float(m_tot - 0.5 * fm - 0.5 * cm)})
        print(f"  dumped {season}: {sum(r['season']==season for r in rows)}", flush=True)
    con.close()
    with open(DUMP, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def _ff_of(model):
    """Read the fitted FourFactors instance out of the Predictor closure
    (read-only; nbapred/ untouched) — same technique as pg_eventrecency."""
    for cell in model.margin.__func__.__closure__ or ():
        v = cell.cell_contents
        if v.__class__.__name__ == "FourFactors":
            return v
    return None


def _cm_of(model, comp, h, a, outs, gd):
    """The comp leg exactly as Predictor.margin computes it, INCLUDING the
    D84-A October-bridge substitution."""
    from nbapred.model.october_bridge import rotation_empty as _rot_empty
    cm = comp.margin(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id], gd,
                     home_edge=0.0)
    for cell in model.margin.__func__.__closure__ or ():
        v = cell.cell_contents
        if v.__class__.__name__ == "OctoberBridge":
            if _rot_empty(comp, h.team_id, a.team_id, gd):
                cm = v.margin(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id])
            break
    return cm


def fit_w(fm, cm, rest, y, w_prev=0.5):
    """Logistic MLE for w in [0,1] on p = sigmoid((w*fm+(1-w)*cm+rest)/SCALE).
    w is a single bounded scalar, so a dense 0.001-grid scan is the exact
    minimiser to within the grid step and is deterministic."""
    if len(y) < BURN_IN:
        return 0.5, len(y), 0.5
    fm, cm, rest, y = map(np.asarray, (fm, cm, rest, y))

    def nll(w):
        m = w * fm + (1 - w) * cm + rest
        p = np.clip(sigmoid(m / SCALE), 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    grid = np.linspace(0.0, 1.0, 1001)
    vals = np.array([nll(w) for w in grid])
    w_mle = float(grid[int(np.argmin(vals))])
    n = len(y)
    w_shr = (n / (n + BLEND_SHRINK)) * w_mle + (1 - n / (n + BLEND_SHRINK)) * 0.5
    return w_shr, n, w_mle


def main():
    if "--dump" in sys.argv or not DUMP.exists():
        print("PASS 1: component dump", flush=True)
        rows = dump_components()
    else:
        rows = [dict(r, y=int(r["y"]), p_mkt=float(r["p_mkt"]),
                     m_tot=float(r["m_tot"]), fm=float(r["fm"]),
                     cm=float(r["cm"]), rest=float(r["rest"]))
                for r in csv.DictReader(open(DUMP))]
    if "--dump" in sys.argv:
        return

    rows.sort(key=lambda r: (r["game_date"], r["game_id"]))
    dates = sorted({r["game_date"] for r in rows})
    refit, last = {}, None
    for ds in dates:
        d = _dt.date.fromisoformat(ds)
        if last is None or (d - last).days >= 7:
            last = d
        refit[ds] = last

    wlog = []
    p_ctl, p_shr, p_mle, ys, seas, mkts = [], [], [], [], [], []
    cache = {}
    for r in rows:
        rd = refit[r["game_date"]]
        if rd not in cache:
            past = [q for q in rows if _dt.date.fromisoformat(q["game_date"]) < rd]
            cache[rd] = fit_w([q["fm"] for q in past], [q["cm"] for q in past],
                              [q["rest"] for q in past], [q["y"] for q in past])
            wlog.append({"date": str(rd), "w_shrunk": round(cache[rd][0], 4),
                         "n": cache[rd][1], "w_mle": round(cache[rd][2], 4)})
        w_shr, _, w_mle = cache[rd]
        p_ctl.append(float(sigmoid(r["m_tot"] / SCALE)))
        p_shr.append(float(sigmoid((w_shr * r["fm"] + (1 - w_shr) * r["cm"]
                                    + r["rest"]) / SCALE)))
        p_mle.append(float(sigmoid((w_mle * r["fm"] + (1 - w_mle) * r["cm"]
                                    + r["rest"]) / SCALE)))
        ys.append(r["y"])
        seas.append(r["season"])
        mkts.append(r["p_mkt"])

    y = np.array(ys)

    def ll(p):
        p = np.clip(np.asarray(p), 1e-12, 1 - 1e-12)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    l_ctl, l_shr, l_mle = ll(p_ctl), ll(p_shr), ll(p_mle)
    res = {"design": "walk-forward sign-constrained ([0,1]) logistic MLE for the "
                     "FF/comp blend weight, shrunk n/(n+600) toward the shipped 0.5",
           "primary": "shrunk arm", "burn_in_games": BURN_IN,
           "n": len(y),
           "ll_control_fixed_50_50": round(float(l_ctl.mean()), 5),
           "ll_shrunk": round(float(l_shr.mean()), 5),
           "ll_mle_unshrunk": round(float(l_mle.mean()), 5),
           "ll_market": round(float(ll(mkts).mean()), 5),
           "pooled_shrunk": paired_bootstrap(l_ctl - l_shr),
           "pooled_mle_diagnostic": paired_bootstrap(l_ctl - l_mle),
           "w_trajectory": wlog,
           "w_final": wlog[-1] if wlog else None,
           "per_season": {}}
    for s in sorted(set(seas)):
        idx = np.array([i for i, x in enumerate(seas) if x == s])
        res["per_season"][s] = {"n": len(idx),
                                "shrunk": paired_bootstrap((l_ctl - l_shr)[idx]),
                                "mle": paired_bootstrap((l_ctl - l_mle)[idx])}
    print(json.dumps({k: v for k, v in res.items() if k != "w_trajectory"}, indent=1))
    json.dump(res, open(ROOT / "data" / "ds_rt4_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
