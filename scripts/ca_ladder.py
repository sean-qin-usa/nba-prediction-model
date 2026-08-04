#!/usr/bin/env python3
"""PART A — THE CARRY-ALL COST LADDER.

Pre-registration: data/carryall_prereg.md
sha256 9a4a414db294ba44908b4a4ee5f0bd490e0b2d0094293eb0f2103a101459bd9b
(frozen BEFORE this file produced a single log-loss number).

Control = data/capstone_pergame.csv, the D132 certified artifact.  Every arm is
an ADDITIVE schedule-layer change, so
    p_arm = sigmoid((SCALE*logit(p_us) + dm) / SCALE)
and the control is bit-identical to the certified artifact by construction.

  python scripts/ca_ladder.py            # certified 5-season corpus
"""
from __future__ import annotations

import csv
import json
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
import datetime as dt  # noqa: E402

import numpy as np  # noqa: E402

from ca_bank import TERM_NAMES, Layer, load_bank  # noqa: E402
from nbapred.model.production import SCALE  # noqa: E402

CERT = REPO / "data" / "capstone_pergame.csv"
SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
NBOOT = 2000
SEED = 20260802

# ---- pre-registered arms (data/carryall_prereg.md §3) ----------------------
LADDER_K = [1, 2, 3, 5, 8, 10, 15]
NOISE_K = [1, 2, 3, 5, 8, 10, 15, 45]


def arms():
    """{name: (cols, teamhome, noise)}   cols = indices into bank['X']."""
    a = {}
    for i, nm in enumerate(TERM_NAMES):
        a[f"solo:{nm}"] = ([i], False, 0)
    a["solo:TEAMHOME"] = ([], True, 0)
    for k in LADDER_K:
        a[f"pile:k{k}"] = (list(range(k)), False, 0)
    # DENSE6 = the terms whose column is supported on >=30% of fit rows and
    # is a genuine continuous schedule quantity rather than an identity proxy.
    # The split is STRUCTURAL (column support, knowable without any endpoint)
    # and is a decomposition of the pre-registered pile, disclosed as added
    # after the k-ladder was scored.
    a["pile:DENSE6"] = ([TERM_NAMES.index(x) for x in
                         ("dtrav_kkm", "dtz_east", "rlen_extra_a", "d3in4",
                          "drest", "delev_km")], False, 0)
    a["pile:DENSE13"] = (list(range(2, len(TERM_NAMES))), False, 0)
    a["pile:ALL15"] = (list(range(len(TERM_NAMES))), False, 0)
    a["pile:ALL15+TEAMHOME"] = (list(range(len(TERM_NAMES))), True, 0)
    for k in NOISE_K:
        a[f"noise:k{k}"] = ([], False, k)
    return a


def refit_dates(dates_sorted):
    """The production weekly cadence: refit at the first game, then whenever
    >=7 days have elapsed (`prod_by_season.py` / `tv_gate.py` verbatim)."""
    out, last = [], None
    for d in dates_sorted:
        if last is None or (d - last).days >= 7:
            out.append(d)
            last = d
    return out


def run(bank, seasons, layer_kw=None, arm_set=None, verbose=True):
    """Returns dict arm -> per-game dm array (aligned to the returned index)."""
    L = Layer(bank, **(layer_kw or {}))
    A = arm_set if arm_set is not None else arms()
    seas = bank["season"].astype(str)
    sel = np.where(np.isin(seas, list(seasons)))[0]
    order = sel[np.argsort(bank["date"][sel], kind="stable")]
    dates = bank["date"][order].astype("datetime64[D]").astype(dt.date)

    dm = {k: np.zeros(len(order)) for k in A}
    dm["_shipped"] = np.zeros(len(order))
    coefs = {k: [] for k in A}
    rd_all = []
    for s in seasons:
        m = seas[order] == s
        if not m.any():
            continue
        sd = sorted(set(dates[m]))
        for d in refit_dates(sd):
            rd_all.append((s, d))
    if verbose:
        print(f"[ladder] {len(order)} games, {len(rd_all)} refit dates, "
              f"{len(A)} arms", flush=True)

    # per game -> its governing refit date
    gov = np.empty(len(order), dtype=object)
    for s in seasons:
        m = seas[order] == s
        if not m.any():
            continue
        sd = sorted(set(dates[m]))
        rds = refit_dates(sd)
        j = 0
        cur = rds[0]
        for i in np.where(m)[0]:
            while j + 1 < len(rds) and rds[j + 1] <= dates[i]:
                j += 1
            cur = rds[j]
            gov[i] = cur

    t0 = time.time()
    for n_, (s, d) in enumerate(rd_all):
        rows = np.where(gov == d)[0]
        rows = rows[seas[order][rows] == s]
        if len(rows) == 0:
            continue
        b5, _, _, nfit, w, _ = L.fit(d)                       # shipped
        for i in rows:
            dm["_shipped"][i] = L.sched_value(order[i], b5, {}, {}, [])
        for name, (cols, th, nz) in A.items():
            bb5, ex, thd, nfit2, w2, nb = L.fit(d, cols=cols, teamhome=th,
                                                noise=nz)
            apply_dead = tuple(c for c in cols if c in (0, 1))
            for i in rows:
                dm[name][i] = L.sched_value(order[i], bb5, ex, thd, nb,
                                            apply_dead=apply_dead) \
                    - dm["_shipped"][i]
            coefs[name].append(dict(date=str(d), n=nfit2, w=round(w2, 5),
                                    base5=[round(float(x), 5) for x in bb5],
                                    extra={TERM_NAMES[c]: round(v, 5)
                                           for c, v in ex.items()},
                                    th_rms=(round(float(np.std(list(thd.values()))), 5)
                                            if thd else 0.0),
                                    noise_rms=(round(float(np.std(nb)), 5)
                                               if nb else 0.0)))
        if verbose and (n_ + 1) % 20 == 0:
            print(f"   refit {n_+1}/{len(rd_all)}  {time.time()-t0:.0f}s",
                  flush=True)
    if verbose:
        print(f"[ladder] done in {time.time()-t0:.0f}s", flush=True)
    return order, dates, dm, coefs


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


if __name__ == "__main__":
    from nbapred.db import connect
    con = connect(read_only=True)
    bank = load_bank(con)
    con.close()

    cert = {r["game_id"]: (float(r["p_us"]), float(r["y"]), r["season"],
                           float(r["p_mkt"]))
            for r in csv.DictReader(open(CERT))}
    order, dates, dm, coefs = run(bank, SEASONS)
    gids = bank["gid"][order].astype(str)
    keep = np.array([g in cert for g in gids])
    print(f"[ladder] matched {keep.sum()}/{len(gids)} games against "
          f"{CERT.name} ({len(cert)} rows)")

    idx = np.where(keep)[0]
    p0 = np.array([cert[g][0] for g in gids[idx]])
    y = np.array([cert[g][1] for g in gids[idx]])
    seas = np.array([cert[g][2] for g in gids[idx]])
    pmk = np.array([cert[g][3] for g in gids[idx]])
    m_base = SCALE * np.log(p0 / (1 - p0))
    dts = np.array([dates[i] for i in idx])

    out = {"prereg_sha256": "9a4a414db294ba44908b4a4ee5f0bd490e0b2d0094"
                            "293eb0f2103a101459bd9b",
           "n": int(len(y)), "seasons": list(SEASONS),
           "control_ll": float(ll(y, p0).mean()),
           "mkt_ll": float(ll(y, pmk).mean()),
           "E_pq": float((p0 * (1 - p0)).mean()),
           "arms": {}}

    from nbapred.eval.splits import Panel, full_report
    names = sorted(dm.keys() - {"_shipped"},
                   key=lambda s: (s.split(":")[0], len(s), s))
    print(f"\n{'arm':26s} {'ncol':>5s} {'rms(dm)':>8s} {'delta':>10s} "
          f"{'season-cluster CI':>24s} {'MDE80':>8s}  verdict")
    for nm in names:
        d = dm[nm][idx]
        p1 = 1.0 / (1.0 + np.exp(-(m_base + d) / SCALE))
        pan = Panel.from_logloss(seas, y, p0, p1, date=dts, label=nm)
        rep = full_report(pan, B=NBOOT, seed=SEED)
        ncol = (len(arms()[nm][0])
                + (30 if arms()[nm][1] else 0) + arms()[nm][2])
        cl = rep["clustering"]["season_cluster_boot"]
        st = dict(ncol=int(ncol), rms_dm=float(np.sqrt((d ** 2).mean())),
                  max_abs_dm=float(np.abs(d).max()),
                  delta=rep["pooled"]["est"], iid_lo=rep["pooled"]["lo"],
                  iid_hi=rep["pooled"]["hi"],
                  cl_lo=cl["lo"], cl_hi=cl["hi"], cl_se=cl["se"],
                  t_lo=rep["clustering"]["season_mean_t"]["lo"],
                  t_hi=rep["clustering"]["season_mean_t"]["hi"],
                  mde80=rep["pooled_mde80"],
                  per_season={x["season"]: x["est"] for x in rep["per_season"]},
                  ro=[f["fold"]["est"] for f in rep["rolling_origin"]["folds"]],
                  I2=rep["era"]["I2"], era_stable=rep["era"]["era_stable"],
                  deff=rep["clustering"].get("design_effect_season"),
                  flags=rep["verdict"]["flags"])
        out["arms"][nm] = st
        sig = "SIG" if (cl["lo"] > 0 or cl["hi"] < 0) else "ns"
        print(f"{nm:26s} {ncol:5d} {st['rms_dm']:8.4f} {st['delta']:+10.5f} "
              f"({cl['lo']:+10.5f},{cl['hi']:+10.5f}) {st['mde80']:8.5f}  {sig}")

    out["coef_summary"] = {
        k: dict(n_refits=len(v),
                mean_extra={kk: round(float(np.mean([c["extra"].get(kk, 0.0)
                                                     for c in v])), 5)
                            for kk in (v[0]["extra"] if v else {})},
                mean_th_rms=round(float(np.mean([c["th_rms"] for c in v])), 5),
                mean_noise_rms=round(float(np.mean([c["noise_rms"] for c in v])), 5),
                mean_w=round(float(np.mean([c["w"] for c in v])), 5),
                mean_nfit=round(float(np.mean([c["n"] for c in v])), 1),
                home_edge_mean=round(float(np.mean([c["base5"][0] for c in v])), 5))
        for k, v in coefs.items() if v}
    json.dump(out, open(REPO / "data" / "carryall_ladder.json", "w"), indent=1)
    np.savez_compressed(REPO / "data" / "carryall_ladder_rows.npz",
                        gid=gids[idx], seas=seas, y=y, p0=p0, pmk=pmk,
                        date=np.array([str(x) for x in dts]),
                        **{("dm_" + k): dm[k][idx] for k in names})
    print("\nwrote data/carryall_ladder.json + carryall_ladder_rows.npz")
