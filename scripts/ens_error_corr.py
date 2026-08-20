#!/usr/bin/env python3
"""D85 step 3 — PRE-REGISTERED error-correlation diagnostic (runs BEFORE the
ensemble gate; docs/EXTERNAL_MODELS.md).

Question: how much independent information do the three talent legs carry?
DARKO and EPM are cousin architectures (SPM prior + decayed RAPM on the same
possession stream) — expect signal rho 0.7-0.9; BPM (box-only) is the
mechanical diversifier. Expected ensemble gain scales with (1 - rho_bar) of
the ERRORS; if error rho_bar > 0.9 the ensemble is a measured no-op and the
negative is recorded WITHOUT spending a gate.

Design (registered):
 * dates: weekly grid Nov 1 .. (season end - 30d), 3 backtest seasons
 * legs as-of d (strict PIT, all < d): DARKO = darko_history latest date < d;
   EPM = latest parsed Wayback capture asof < d (identity-bearing path after
   the lock finding); BPM = nbapred.features.bpm.bpm_asof (365d rolling,
   published low-minutes regression)
 * realized target: on-court net per 48 = sum(plus_minus)/minutes*48 over
   games in [d, d+30), players with >= 150 forward minutes ("realized future
   30-day dpm")
 * universe per date: players present in ALL THREE legs and the target
 * per date: cross-sectional z-scores; SIGNAL rho matrix = corr(z_si, z_sj);
   ERROR rho matrix = corr(z_si - z_y, z_sj - z_y); skill = corr(z_si, z_y).
   Report the across-date means (rho-bar) and date counts.
Output: data/logs/ens_error_corr.json + stdout table.
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

WB_DIR = REPO / "data/raw/ext_epm/wayback"
SEASONS = {"2023-24": (dt.date(2023, 11, 1), dt.date(2024, 4, 14)),
           "2024-25": (dt.date(2024, 11, 1), dt.date(2025, 4, 13)),
           "2025-26": (dt.date(2025, 11, 1), dt.date(2026, 4, 12))}
FWD_DAYS = 30
MIN_FWD_MIN = 150.0
LEGS = ("darko", "epm", "bpm")


def epm_captures() -> list[tuple[str, Path]]:
    caps = []
    for f in sorted(WB_DIR.glob("parsed_*.json")):
        meta = json.loads(f.read_text())
        caps.append((meta["asof"], f))
    caps.sort()
    return caps


def epm_asof(caps, d: dt.date) -> tuple[dict, int] | tuple[None, None]:
    best = None
    for asof, f in caps:
        if dt.date.fromisoformat(asof) < d:
            best = (asof, f)
        else:
            break
    if best is None:
        return None, None
    meta = json.loads(best[1].read_text())
    age = (d - dt.date.fromisoformat(best[0])).days
    return {r["player_id"]: r["tot"] for r in meta["rows"]}, age


def darko_asof(con, d: dt.date) -> dict:
    return dict(con.execute("""
        SELECT player_id, dpm FROM (
          SELECT player_id, dpm, row_number() OVER (
            PARTITION BY player_id ORDER BY date DESC) rn
          FROM darko_history WHERE date < ?) WHERE rn = 1""", [d]).fetchall())


def realized_fwd(con, d: dt.date) -> dict:
    rows = con.execute("""
        SELECT s.player_id, sum(s.plus_minus) pm, sum(s.seconds)/60.0 mins
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g
          USING (game_id)
        WHERE s.game_id LIKE '002%' AND g.game_date >= ? AND g.game_date < ?
          AND s.seconds > 0
        GROUP BY 1""", [d, d + dt.timedelta(days=FWD_DAYS)]).fetchall()
    return {int(p): pm / m * 48.0 for p, pm, m in rows
            if m and m >= MIN_FWD_MIN}


def zmap(vals: dict, keys) -> dict:
    v = np.array([vals[k] for k in keys], float)
    mu, sd = v.mean(), v.std()
    if sd <= 0:
        sd = 1.0
    return {k: (vals[k] - mu) / sd for k in keys}


def main() -> None:
    con = connect(read_only=True)
    caps = epm_captures()
    print(f"EPM captures available: {len(caps)} "
          f"({caps[0][0]} .. {caps[-1][0]})" if caps else "NO EPM captures")
    sig_mats, err_mats, skills, ns, ages = [], [], [], [], []
    per_season = {s: dict(sig=[], err=[], skill=[], n=[]) for s in SEASONS}
    for season, (d0, d1) in SEASONS.items():
        d = d0
        while d <= d1 - dt.timedelta(days=FWD_DAYS):
            epm, age = epm_asof(caps, d)
            if epm is None:
                d += dt.timedelta(days=7)
                continue
            legs = dict(darko=darko_asof(con, d), epm=epm,
                        bpm={p: v["bpm"] for p, v in bpm_asof(con, d).items()})
            y = realized_fwd(con, d)
            keys = [p for p in y
                    if all(p in legs[s] for s in LEGS)]
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
        n_dates=len(ns), mean_universe=round(float(np.mean(ns)), 1),
        mean_epm_capture_age_days=round(float(np.mean(ages)), 1),
        median_epm_capture_age_days=float(np.median(ages)),
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
    out = REPO / "data/logs/ens_error_corr.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
