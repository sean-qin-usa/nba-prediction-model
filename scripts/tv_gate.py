"""TRAVEL / CIRCADIAN / DENSITY — endpoint gate.

Pre-registration: data/travel_prereg.md
sha256 d3d334b92665af13dae7914133af626c9d8c1993982a67023a40810c2fbb5a3e
(frozen BEFORE this file was run).

One walk-forward pass produces the SAME-RUN CONTROL (the shipped path, bitwise
what scripts/prod_by_season.py computes) and all five treatments (arms A, B, C,
D and the ABCD portfolio). Each treatment replaces the shipped D46 schedule
contribution with the JOINTLY REFIT extended layer, so nothing is double-counted
against the shipped b2b terms:

    p_ctrl = sigmoid( m_base / SCALE )
    p_arm  = sigmoid( (m_base - sched_shipped + sched_arm) / SCALE )

Usage:  python scripts/tv_gate.py dev|holdout
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import datetime as _dt  # noqa: E402

import numpy as np  # noqa: E402

from nbapred.db import connect  # noqa: E402
from nbapred.model.composition import CompositionModel  # noqa: E402
from nbapred.model.production import (  # noqa: E402
    SCALE, fit_production, fit_schedule_layer, fit_schedule_layer_ext, sigmoid)
from nbapred.model.travel import ARM_TERMS, TERM_PRED, build_state, term_value  # noqa: E402

DEV = ("2023-24", "2024-25", "2025-26")
HOLDOUT = ("2021-22", "2022-23")
ARMSETS = {"A": ("A",), "B": ("B",), "C": ("C",), "D": ("D",),
           "ABCD": ("A", "B", "C", "D")}
SEED = 20260801
NBOOT = 2000


def season_run(season, state):
    con = connect(read_only=True)
    pm = con.execute("""SELECT game_id, team_id, player_id FROM player_game_stats
        WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    played = {(g, t): set(grp.player_id)
              for (g, t), grp in pm.groupby(["game_id", "team_id"])}
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

    out = {"season": season, "gid": [], "date": [], "home": [], "away": [],
           "y": [], "p_ctrl": [], "p_mkt": [], "dm": {k: [] for k in ARMSETS}}
    for k in ARMSETS:
        out.setdefault("p_" + k, [])
    coefs = []
    model = comp = None
    last = None
    sh5 = None
    ext = {}
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
            sh5 = fit_schedule_layer(con, gd)
            ext = {k: fit_schedule_layer_ext(con, gd, arms=v, state=state)
                   for k, v in ARMSETS.items()}
            coefs.append({"date": str(gd), "shipped": [round(float(x), 5) for x in sh5],
                          **{k: {"base": [round(float(x), 5) for x in v[0]],
                                 "extra": {kk: round(float(vv), 5)
                                           for kk, vv in v[1].items()}}
                             for k, v in ext.items()}})
            last = gd
        pmv = mkt.get((str(gd)[:10], h.team_abbrev, a.team_abbrev))
        if pmv is None:
            continue
        sh_st, sa_st = state.get((h.team_id, gd)), state.get((a.team_id, gd))
        if sh_st is None or sa_st is None:
            continue
        outs = {}
        for t in (h.team_id, a.team_id):
            pl = played.get((gid, t), set())
            outs[t] = {p for p, d0 in comp.players.items()
                       if d0["team_id"] == t and (gd - d0["last_played"]).days <= 12
                       and p not in pl}
        bh, ba = b2b(h.team_id, gd), b2b(a.team_id, gd)
        m_base = model.margin(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id],
                              gd, b2b_home=bh, b2b_away=ba)
        sched_shipped = sh5[0] + (sh5[1] if bh else 0.0) + (sh5[2] if ba else 0.0)
        out["gid"].append(gid); out["date"].append(str(gd)[:10])
        out["home"].append(h.team_abbrev); out["away"].append(a.team_abbrev)
        out["y"].append(int(h.wl == "W")); out["p_mkt"].append(float(pmv))
        out["p_ctrl"].append(float(sigmoid(m_base / SCALE)))
        for k, arms in ARMSETS.items():
            b5, ex = ext[k]
            s_arm = b5[0] + (b5[1] if bh else 0.0) + (b5[2] if ba else 0.0)
            for aa in arms:
                for c, fn in ARM_TERMS[aa]:
                    s_arm += ex[c] * term_value(c, fn, sh_st, sa_st)
            dm = s_arm - sched_shipped
            out["dm"][k].append(float(dm))
            out["p_" + k].append(float(sigmoid((m_base + dm) / SCALE)))
    con.close()
    out["coefs"] = coefs
    return out


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired(y, p0, p1, seed=SEED, nboot=NBOOT):
    """delta = control loss - treatment loss (positive = treatment better)."""
    d = ll(y, p0) - ll(y, p1)
    n = len(d)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(nboot, n))
    bs = d[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pt = float(d.mean())
    pw = float(np.mean(bs <= 0)) if pt > 0 else float(np.mean(bs >= 0))
    return dict(n=n, delta=pt, lo=float(lo), hi=float(hi),
                se=float(d.std(ddof=1) / np.sqrt(n)), sd=float(d.std(ddof=1)),
                p_wrongside=pw, sig=bool(lo > 0 or hi < 0),
                mde80=float(2.802 * d.std(ddof=1) / np.sqrt(n)))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "dev"
    seasons = DEV if which == "dev" else HOLDOUT
    con = connect(read_only=True)
    state = build_state(con)
    con.close()
    runs = [season_run(s, state) for s in seasons]

    y = np.array([v for r in runs for v in r["y"]])
    p0 = np.array([v for r in runs for v in r["p_ctrl"]])
    pmk = np.array([v for r in runs for v in r["p_mkt"]])
    seas = np.array([r["season"] for r in runs for _ in r["y"]])
    print(f"\n{'='*74}\nUNIVERSE {which.upper()} {seasons}  n={len(y)}")
    print(f"control ll {ll(y,p0).mean():.5f}   market ll {ll(y,pmk).mean():.5f}")

    # ---- D134 control-hash rule: same-run control vs the certified artifact
    import csv
    cert = {r["game_id"]: float(r["p_us"]) for r in
            csv.DictReader(open(ROOT / "data" / "capstone_pergame.csv"))}
    gids = [g for r in runs for g in r["gid"]]
    dps = [abs(p0[i] - cert[g]) for i, g in enumerate(gids) if g in cert]
    print(f"CONTROL HASH (D134): matched {len(dps)}/{len(gids)} games vs "
          f"data/capstone_pergame.csv, max|dp| {max(dps):.3e}, "
          f"frac moved {np.mean(np.array(dps)>1e-12):.4f}")

    res = {"universe": which, "seasons": list(seasons), "n": int(len(y)),
           "control_ll": float(ll(y, p0).mean()), "mkt_ll": float(ll(y, pmk).mean()),
           "control_max_dp_vs_cert": float(max(dps)), "arms": {}}
    print(f"\n{'arm':6s} {'delta':>10s} {'95% CI':>22s} {'MDE80':>9s} "
          f"{'p_wrong':>8s} {'rms(dm)':>8s} {'max|dm|':>8s}  verdict")
    for k in ARMSETS:
        p1 = np.array([v for r in runs for v in r["p_" + k]])
        dm = np.array([v for r in runs for v in r["dm"][k]])
        st = paired(y, p0, p1)
        st["rms_dm"] = float(np.sqrt((dm ** 2).mean()))
        st["max_abs_dm"] = float(np.abs(dm).max())
        st["by_season"] = {s: paired(y[seas == s], p0[seas == s], p1[seas == s])
                           for s in seasons}
        res["arms"][k] = st
        print(f"{k:6s} {st['delta']:+10.5f} ({st['lo']:+9.5f},{st['hi']:+9.5f}) "
              f"{st['mde80']:9.5f} {st['p_wrongside']:8.3f} {st['rms_dm']:8.3f} "
              f"{st['max_abs_dm']:8.3f}  {'SIG' if st['sig'] else 'ns'}")
        for s in seasons:
            b = st["by_season"][s]
            print(f"       {s} {b['delta']:+10.5f} ({b['lo']:+9.5f},{b['hi']:+9.5f}) n={b['n']}")

    # ---- walk-forward coefficient stability + sign check vs pre-registration
    print("\nWALK-FORWARD COEFFICIENTS (mean over refits, POINTS) + sign vs prereg")
    allc = [c for r in runs for c in r["coefs"]]
    res["coef_summary"] = {}
    for k in ARMSETS:
        names = [c for aa in ARMSETS[k] for c, _ in ARM_TERMS[aa]]
        print(f"  armset {k}:")
        for nm in names:
            v = np.array([c[k]["extra"][nm] for c in allc])
            pred = TERM_PRED[nm]
            frac = float(np.mean(np.sign(v) == pred))
            ok = "MATCH" if np.sign(v.mean()) == pred else "**MISS**"
            print(f"    {nm:14s} mean {v.mean():+8.4f}  sd {v.std():7.4f}  "
                  f"min {v.min():+7.3f} max {v.max():+7.3f}  "
                  f"refits sign-correct {frac:5.1%}  {ok}")
            res["coef_summary"][f"{k}.{nm}"] = dict(
                mean=float(v.mean()), sd=float(v.std()), min=float(v.min()),
                max=float(v.max()), frac_sign_correct=frac, pred=pred,
                mean_sign_match=bool(np.sign(v.mean()) == pred))
        b = np.array([c[k]["base"] for c in allc])
        sh = np.array([c["shipped"] for c in allc])
        print(f"    b2b shift vs shipped: home {b[:,1].mean()-sh[:,1].mean():+.4f}  "
              f"away {b[:,2].mean()-sh[:,2].mean():+.4f}  "
              f"home_edge {b[:,0].mean()-sh[:,0].mean():+.4f}")
        res["coef_summary"][f"{k}.b2b_shift"] = dict(
            home=float(b[:, 1].mean() - sh[:, 1].mean()),
            away=float(b[:, 2].mean() - sh[:, 2].mean()),
            home_edge=float(b[:, 0].mean() - sh[:, 0].mean()))

    json.dump(res, open(f"data/tv_gate_{which}.json", "w"), indent=1)
    np.savez(f"data/tv_gate_{which}_rows.npz", y=y, p_ctrl=p0, p_mkt=pmk,
             seas=seas, gid=np.array(gids),
             **{("p_" + k): np.array([v for r in runs for v in r["p_" + k]])
                for k in ARMSETS},
             **{("dm_" + k): np.array([v for r in runs for v in r["dm"][k]])
                for k in ARMSETS})
    print(f"\nwrote data/tv_gate_{which}.json + data/tv_gate_{which}_rows.npz")
