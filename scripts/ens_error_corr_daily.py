#!/usr/bin/env python3
"""D94 — error-correlation diagnostic RERUN with the DAILY re-identified
EPM leg (epm_history_daily; scripts/epm_reid.py). Companion to the D86
run (scripts/ens_error_corr.py, weekly Wayback captures, offdiag mean
0.874): does a fresh daily EPM change rho? Same design otherwise: weekly
evaluation grid, 3 seasons, fwd-30d on-court net/48 target, universe =
players in all three legs + >=150 fwd minutes.

Output: data/logs/ens_error_corr_daily.json + stdout table.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nbapred.db import connect  # noqa: E402
from nbapred.features.bpm import bpm_asof  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from ens_error_corr import (FWD_DAYS, LEGS, SEASONS, darko_asof,  # noqa: E402
                            realized_fwd, zmap)


def daily_epm(con):
    rows = con.execute("""
        SELECT asof_date, player_id, tot_epm
        FROM epm_history_daily ORDER BY asof_date""").fetchall()
    by_date = {}
    for d, p, t in rows:
        by_date.setdefault(d, {})[int(p)] = float(t)
    return sorted(by_date), by_date


def main() -> None:
    con = connect(read_only=True)
    dates, by_date = daily_epm(con)
    print(f"daily EPM asof dates: {len(dates)} ({dates[0]} .. {dates[-1]})")
    sig_mats, err_mats, skills, ns, ages = [], [], [], [], []
    per_season = {s: dict(sig=[], err=[], skill=[], n=[]) for s in SEASONS}
    for season, (d0, d1) in SEASONS.items():
        d = d0
        while d <= d1 - dt.timedelta(days=FWD_DAYS):
            prior = [x for x in dates if x < d]
            if not prior:
                d += dt.timedelta(days=7)
                continue
            best = prior[-1]
            epm = dict(by_date[best])
            age = (d - best).days
            legs = dict(darko=darko_asof(con, d), epm=epm,
                        bpm={p: v["bpm"] for p, v in bpm_asof(con, d).items()})
            y = realized_fwd(con, d)
            keys = [p for p in y if all(p in legs[s] for s in LEGS)]
            if len(keys) < 60:
                d += dt.timedelta(days=7)
                continue
            zs = {s: zmap(legs[s], keys) for s in LEGS}
            zy = zmap(y, keys)
            zv = {s: np.array([zs[s][k] for k in keys]) for s in LEGS}
            yv = np.array([zy[k] for k in keys])
            sig = np.array([[np.corrcoef(zv[a], zv[b])[0, 1]
                             for b in LEGS] for a in LEGS])
            err = np.array([[np.corrcoef(zv[a] - yv, zv[b] - yv)[0, 1]
                             for b in LEGS] for a in LEGS])
            skill = np.array([np.corrcoef(zv[a], yv)[0, 1] for a in LEGS])
            sig_mats.append(sig)
            err_mats.append(err)
            skills.append(skill)
            ns.append(len(keys))
            ages.append(age)
            for tgt, val in (("sig", sig), ("err", err), ("skill", skill),
                             ("n", len(keys))):
                per_season[season][tgt].append(val)
            d += dt.timedelta(days=7)

    def fmt(m):
        return [[round(float(x), 3) for x in row] for row in np.asarray(m)]

    sig_bar = np.mean(sig_mats, axis=0)
    err_bar = np.mean(err_mats, axis=0)
    skill_bar = np.mean(skills, axis=0)
    offdiag = [err_bar[i][j] for i in range(3) for j in range(3) if i < j]
    res = dict(
        note="D94 second look: daily re-identified EPM leg "
             "(epm_history_daily); compare data/logs/ens_error_corr.json "
             "(D86, weekly Wayback captures)",
        n_dates=len(ns), mean_universe=round(float(np.mean(ns)), 1),
        mean_epm_asof_age_days=round(float(np.mean(ages)), 1),
        median_epm_asof_age_days=float(np.median(ages)),
        legs=list(LEGS),
        signal_rho_bar=fmt(sig_bar),
        error_rho_bar=fmt(err_bar),
        skill_bar=[round(float(x), 3) for x in skill_bar],
        error_rho_bar_offdiag_mean=round(float(np.mean(offdiag)), 3),
        per_season={s: dict(
            n_dates=len(v["n"]),
            signal_rho=fmt(np.mean(v["sig"], axis=0)),
            error_rho=fmt(np.mean(v["err"], axis=0)),
            skill=[round(float(x), 3) for x in np.mean(v["skill"], axis=0)])
            for s, v in per_season.items() if v["n"]})
    out = REPO / "data/logs/ens_error_corr_daily.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
