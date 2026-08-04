#!/usr/bin/env python3
"""Precision-weighted (method-of-moments) ff/comp blend vs shipped fixed 50/50.

NOT outcome-fitted: weights come from per-component MARGIN-ERROR VARIANCES
estimated on strictly-past games (monthly walk-forward, 400-game burn-in),
then combined by the inverse-variance identity
    w_ff = (1/v_ff) / (1/v_ff + 1/v_comp).
No search over w, no objective evaluated on the weight's own eval games.

--- reconstruction algebra (verified against nbapred/model/production.py) ---
Production margin  P = 0.5*ff_neutral + 0.5*comp_neutral + sched
data/component_pergame.csv was produced by scripts/component_dump.py, which
  * called comp.margin(...) WITHOUT home_edge=0.0  -> m_comp = comp_neutral + 3.0
    (composition.HOME_EDGE = 3.0; production passes home_edge=0.0)
  * recovered m_ff = 2*P - m_comp                  -> m_ff   = ff_neutral + 2*sched - 3.0
  * never passed b2b/dead flags                    -> sched  = he (walk-forward home edge)
Therefore, with he_t the walk-forward home edge in force at game t:
    pred_ff   = ff_neutral + he = m_ff   + 3.0 - he
    pred_comp = comp_neutral+ he = m_comp - 3.0 + he
    blend(w)  = w*pred_ff + (1-w)*pred_comp
    blend(0.5)= 0.5*(m_ff + m_comp) == P exactly  (the same-run CONTROL)
he_t is recomputed in-run by importing the shipped fit_schedule_layer at the
same weekly refit cadence component_dump used -> control is a same-run control,
not a published baseline.

p = sigmoid(margin / 7.2) for every arm. Gate: variant-vs-control paired
bootstrap, 2000x, 95% percentile CI on mean log-loss delta.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nbapred.db import connect                                   # noqa: E402
from nbapred.model.production import SCALE, fit_schedule_layer   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "component_pergame.csv"
HOME_EDGE_DUMP = 3.0     # composition.HOME_EDGE baked into the CSV's m_comp
BURN_IN = 400            # games of strictly-past history before a weight is used
PHASE_MIN = 150          # min past games in a phase cell before phase-specific v
EARLY_GP = 20            # "first 20 team-games" phase boundary
NBOOT = 2000
SEED = 20260730


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def logloss_vec(y, p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ---------------------------------------------------------------- data ----
def load():
    d = pd.read_csv(CSV, dtype={"game_id": str})
    d["game_date"] = pd.to_datetime(d["game_date"]).dt.date

    con = connect(read_only=True)
    pts = con.execute("""
        SELECT game_id, team_abbrev, pts FROM nba_games
        WHERE game_id LIKE '002%' AND pts IS NOT NULL
    """).fetchdf()
    lut = {(r.game_id, r.team_abbrev): r.pts for r in pts.itertuples()}
    d["ph"] = [lut.get((g, t)) for g, t in zip(d.game_id, d.home)]
    d["pa"] = [lut.get((g, t)) for g, t in zip(d.game_id, d.away)]
    assert d.ph.notna().all() and d.pa.notna().all(), "score join incomplete"
    d["y_margin"] = (d.ph - d.pa).astype(float)
    assert ((d.y_margin > 0).astype(int) == d.y).all(), "y disagrees with scores"

    # ---- walk-forward schedule term, replicating component_dump's cadence:
    # refit when (game_date - last_refit).days >= 7, reset each season.
    he_by_date = {}
    for season, grp in d.groupby("season", sort=True):
        last = None
        for gd in sorted(grp.game_date.unique()):
            if last is None or (gd - last).days >= 7:
                last = gd
                he_by_date[(season, gd)] = fit_schedule_layer(con, gd)[0]
            he_by_date[(season, gd)] = he_by_date[(season, last)]
    con.close()
    d["he"] = [he_by_date[(s, g)] for s, g in zip(d.season, d.game_date)]

    # ---- PIT team-games-played within season (for the phase split)
    gp = {}
    d = d.sort_values(["game_date", "game_id"], kind="mergesort").reset_index(drop=True)
    gph, gpa = [], []
    for r in d.itertuples():
        kh, ka = (r.season, r.home), (r.season, r.away)
        gph.append(gp.get(kh, 0))
        gpa.append(gp.get(ka, 0))
        gp[kh] = gp.get(kh, 0) + 1
        gp[ka] = gp.get(ka, 0) + 1
    d["gp_home"], d["gp_away"] = gph, gpa
    d["early"] = (d.gp_home < EARLY_GP) & (d.gp_away < EARLY_GP)

    # ---- component predictions on a common full-margin scale
    d = d[d.m_ff.notna()].reset_index(drop=True)     # ff-not-ready rows: prod used
    d["pred_ff"] = d.m_ff + HOME_EDGE_DUMP - d.he    # a different (fallback) stack
    d["pred_comp"] = d.m_comp - HOME_EDGE_DUMP + d.he
    d["e_ff"] = d.y_margin - d.pred_ff
    d["e_comp"] = d.y_margin - d.pred_comp
    d["month"] = [dt.date(g.year, g.month, 1) for g in d.game_date]
    return d


# ------------------------------------------------------- weight schedule ----
def prec_w(vf, vc):
    return (1.0 / vf) / ((1.0 / vf) + (1.0 / vc))


def gls_w(vf, vc, cov):
    den = vf + vc - 2.0 * cov
    if den <= 1e-9:
        return 0.5
    return float(np.clip((vc - cov) / den, 0.0, 1.0))


def moments(sub, centered=False):
    ef, ec = sub.e_ff.values, sub.e_comp.values
    if centered:
        ef = ef - ef.mean()
        ec = ec - ec.mean()
    return float(np.mean(ef ** 2)), float(np.mean(ec ** 2)), float(np.mean(ef * ec))


def build_weights(d, centered=False):
    """Monthly walk-forward. For each month, moments come from games strictly
    before the first day of that month. Returns per-game weights + a log."""
    rows = []
    w_glob = np.full(len(d), np.nan)
    w_phase = np.full(len(d), np.nan)
    w_gls = np.full(len(d), np.nan)
    for m in sorted(d.month.unique()):
        past = d[d.game_date < m]
        idx = np.flatnonzero((d.month == m).values)
        if len(past) < BURN_IN:
            continue
        vf, vc, cov = moments(past, centered)
        wg = prec_w(vf, vc)
        wl = gls_w(vf, vc, cov)
        w_glob[idx] = wg
        w_gls[idx] = wl
        cell = {}
        for ph in (True, False):
            sub = past[past.early == ph]
            if len(sub) >= PHASE_MIN:
                f, c, _ = moments(sub, centered)
                cell[ph] = (prec_w(f, c), f, c, len(sub))
            else:
                cell[ph] = (wg, vf, vc, len(sub))
        for ph in (True, False):
            sel = idx[(d.early.values[idx] == ph)]
            w_phase[sel] = cell[ph][0]
        rows.append(dict(month=str(m), n_past=len(past), n_month=len(idx),
                         v_ff=vf, v_comp=vc, corr=cov / np.sqrt(vf * vc),
                         w_ff_global=wg, w_ff_gls=wl,
                         w_ff_early=cell[True][0], n_past_early=cell[True][3],
                         w_ff_late=cell[False][0], n_past_late=cell[False][3]))
    return w_glob, w_phase, w_gls, pd.DataFrame(rows)


# ---------------------------------------------------------------- gate ----
def paired_boot(dl, rng, nboot=NBOOT):
    n = len(dl)
    if n == 0:
        return (np.nan, np.nan)
    draws = rng.integers(0, n, size=(nboot, n))
    means = dl[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def verdict(lo, hi):
    if lo < 0 and hi < 0:
        return "PASS"
    if lo > 0 and hi > 0:
        return "FAIL"
    return "NS"


def main():
    d = load()
    out = {}
    for tag, centered in (("mse", False), ("centered", True)):
        wg, wp, wl, log = build_weights(d, centered)
        ok = ~np.isnan(wg)
        e = d[ok].reset_index(drop=True)
        wg, wp, wl = wg[ok], wp[ok], wl[ok]

        m_ctrl = 0.5 * e.pred_ff + 0.5 * e.pred_comp          # == shipped production
        arms = {
            "control_50_50": m_ctrl.values,
            "prec_global": (wg * e.pred_ff + (1 - wg) * e.pred_comp).values,
            "prec_phase": (wp * e.pred_ff + (1 - wp) * e.pred_comp).values,
            "gls_global": (wl * e.pred_ff + (1 - wl) * e.pred_comp).values,
        }
        ll = {k: logloss_vec(e.y.values, sigmoid(v / SCALE)) for k, v in arms.items()}

        rng = np.random.default_rng(SEED)
        res = {"n_eval": int(len(e)),
               "eval_start": str(e.game_date.min()), "eval_end": str(e.game_date.max()),
               "control_ll": float(ll["control_50_50"].mean()),
               "w_summary": {
                   "global_mean": float(np.mean(wg)),
                   "global_min": float(np.min(wg)), "global_max": float(np.max(wg)),
                   "phase_early_mean": float(np.mean(wp[e.early.values])),
                   "phase_late_mean": float(np.mean(wp[~e.early.values])),
                   "gls_mean": float(np.mean(wl)),
                   "n_early_eval": int(e.early.sum())},
               "arms": {}}
        for k in ("prec_global", "prec_phase", "gls_global"):
            dl = ll[k] - ll["control_50_50"]
            lo, hi = paired_boot(dl, rng)
            per = {}
            for s, g in e.groupby("season"):
                i = g.index.values
                slo, shi = paired_boot(dl[i], rng)
                per[s] = dict(n=int(len(i)), delta=float(dl[i].mean()),
                              ci=[round(slo, 5), round(shi, 5)],
                              ctrl_ll=float(ll["control_50_50"][i].mean()),
                              var_ll=float(ll[k][i].mean()),
                              verdict=verdict(slo, shi))
            res["arms"][k] = dict(ll=float(ll[k].mean()), delta=float(dl.mean()),
                                  ci=[round(lo, 6), round(hi, 6)],
                                  verdict=verdict(lo, hi), per_season=per)
        out[tag] = res
        out[tag + "_weight_log"] = log.round(4).to_dict("records")

    print(json.dumps(out, indent=1, default=str))
    (ROOT / "data" / "stat_precision_blend.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
