"""RT2 — does the D71 LATE-GATED FORM term add ON TOP of the shipped D90
late-state layer, now that the corpus is 4-5 seasons instead of 3?

D71's discovery (isolation +0.00178 CI(+0.00076,+0.00275), late +0.0053) was
never shipped: it was a third look on spent data and got FROZEN as F1, then
D90's late-state layer shipped a term of the same functional family
(c_f*(form5_h-form5_a), either-gp>=55) and DECISIONS.md declares F1
"SUPERSEDED by this layer (do not run both)".

This is the pre-registered incremental question the directive asks: with F1's
own construction (730d ROLLING window, weekly k, wpct-control only) layered on
top of the shipped layer (EXPANDING frame since 2022-23, daily c_f, jointly
controlled for tank score + outs), is there residual signal or is it a pure
double-count?

CONSTRUCTION: fit_form_k is imported VERBATIM from scripts/ba_windowed.py
(the D71 gate script). Control margins are recovered EXACTLY from the RT1
FULL-arm probabilities (m = SCALE*logit(p); the production margin is linear
and p = sigmoid(m/SCALE), so this is an exact inverse, verified below), so
the control is bitwise the RT1 arm — no re-fit, no drift.

  m_exp = m_ctl + k_form * (form5_home - form5_away)   [gp>=55 gate]

Gate (pre-registered): paired bootstrap 2000x seed 20260731 on per-game
log-loss delta, pooled over the eval seasons; also late window (either
gp>=55, the term's own footprint) and per season. SHIP only if the pooled CI
excludes 0 (G1) — this is an ADD-ON to a shipped layer, so the bar is the
incremental one.

Read-only DB. Requires data/ds_rt1_pergame.csv.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from ba_windowed import FORM_GP, FORM_N, fit_form_k  # noqa: E402  (verbatim D71)
from ds_corpus import arm_connection, paired_bootstrap  # noqa: E402
from nbapred.model.production import SCALE, sigmoid  # noqa: E402

PERGAME = ROOT / "data" / "ds_rt1_pergame.csv"


def build_form_state(con):
    """(season, team_id) -> chronological signed margins, and a form5/gp
    lookup at a date — PIT by construction (strictly-before slicing)."""
    rows = con.execute("""
        SELECT season, game_id, game_date, team_id, is_home, pts
        FROM nba_games WHERE game_id LIKE '002%' AND pts IS NOT NULL
        ORDER BY game_date, game_id""").fetchall()
    byg = {}
    for season, gid, d, tid, ish, pts in rows:
        d = d.date() if hasattr(d, "date") else d
        g = byg.setdefault(gid, dict(season=season, date=d))
        g["h" if ish else "a"] = (int(tid), float(pts))
    hist = {}
    for g in sorted((x for x in byg.values() if "h" in x and "a" in x),
                    key=lambda x: (x["date"], x["season"])):
        ht, hp = g["h"]
        at, ap = g["a"]
        hist.setdefault((g["season"], ht), []).append((g["date"], hp - ap))
        hist.setdefault((g["season"], at), []).append((g["date"], ap - hp))
    return hist


def form5_gp(hist, season, tid, d):
    h = hist.get((season, tid), [])
    past = [m for (dd, m) in h if dd < d]
    f5 = float(np.mean(past[-FORM_N:])) if len(past) >= FORM_N else 0.0
    return f5, len(past)


def main():
    if not PERGAME.exists():
        raise SystemExit("run scripts/ds_rt1_capstone.py first")
    rows = list(csv.DictReader(open(PERGAME)))
    con = arm_connection(None)
    hist = build_form_state(con)

    # weekly walk-forward k (D71 cadence), cached per refit date
    kcache, klog = {}, []
    seasons = sorted({r["season"] for r in rows})
    refit = {}
    for s in seasons:
        ds = sorted({_dt.date.fromisoformat(r["game_date"])
                     for r in rows if r["season"] == s})
        last = None
        for d in ds:
            if last is None or (d - last).days >= 7:
                last = d
            refit[(s, d)] = last

    out = []
    for r in rows:
        s = r["season"]
        d = _dt.date.fromisoformat(r["game_date"])
        p_ctl = float(r["p_full"])
        m_ctl = SCALE * float(np.log(p_ctl / (1 - p_ctl)))
        rd = refit[(s, d)]
        if rd not in kcache:
            k, n, braw = fit_form_k(con, rd)
            kcache[rd] = k
            klog.append({"date": str(rd), "k": round(k, 5), "n": n,
                         "beta_raw": round(braw, 5)})
        k = kcache[rd]
        out.append((r, d, m_ctl, k, int(r["gp_home"]), int(r["gp_away"])))

    # resolve team ids once
    tid = {}
    for gid, t, ish in con.execute("""SELECT game_id, team_id, is_home
            FROM nba_games WHERE game_id LIKE '002%'""").fetchall():
        tid.setdefault(gid, {})["h" if ish else "a"] = int(t)

    recs = []
    for r, d, m_ctl, k, gph, gpa in out:
        ids = tid.get(r["game_id"])
        if not ids or "h" not in ids or "a" not in ids:
            continue
        term = 0.0
        if gph >= FORM_GP or gpa >= FORM_GP:
            f5h, _ = form5_gp(hist, r["season"], ids["h"], d)
            f5a, _ = form5_gp(hist, r["season"], ids["a"], d)
            term = k * (f5h - f5a)
        p_exp = float(sigmoid((m_ctl + term) / SCALE))
        y = int(r["y"])
        recs.append({"season": r["season"], "game_id": r["game_id"], "y": y,
                     "p_ctl": float(r["p_full"]), "p_exp": p_exp,
                     "p_mkt": float(r["p_mkt"]), "term": term,
                     "late": gph >= FORM_GP or gpa >= FORM_GP})

    def ll(p, y):
        p = min(max(p, 1e-12), 1 - 1e-12)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    d_all = np.array([ll(x["p_ctl"], x["y"]) - ll(x["p_exp"], x["y"]) for x in recs])
    res = {"design": "D71 fit_form_k imported verbatim from ba_windowed.py; "
                     "control = RT1 FULL arm (shipped production incl. D90 layer), "
                     "margins recovered exactly via m = SCALE*logit(p)",
           "gate": "paired bootstrap 2000x seed 20260731, positive = D71 add-on better",
           "n_touched": int(sum(x["term"] != 0.0 for x in recs)),
           "mean_abs_term": round(float(np.mean([abs(x["term"]) for x in recs
                                                 if x["term"] != 0.0]) or 0.0), 4),
           "k_trajectory_tail": klog[-6:],
           "pooled": paired_bootstrap(d_all)}
    late = np.array([i for i, x in enumerate(recs) if x["late"]])
    res["late_either_gp_ge55"] = paired_bootstrap(d_all[late])
    res["per_season"] = {}
    for s in seasons:
        idx = np.array([i for i, x in enumerate(recs) if x["season"] == s])
        res["per_season"][s] = paired_bootstrap(d_all[idx])
    # zero-outside-window check
    off = [x for x in recs if not x["late"]]
    res["zero_outside_window_max_abs_dp"] = max(
        (abs(x["p_exp"] - x["p_ctl"]) for x in off), default=0.0)
    print(json.dumps(res, indent=1))
    json.dump(res, open(ROOT / "data" / "ds_rt2_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
