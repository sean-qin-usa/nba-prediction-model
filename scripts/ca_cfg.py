#!/usr/bin/env python3
"""PART B §8 — GATE THE THREE ADAPTATION CONFIGS ON THE FULL SCORABLE HISTORY.

Prereg data/carryall_prereg.md sha256 9a4a414d...59bd9b §§6-8.
V3 battery: rolling-origin primary, LOSO, era decomposition, clustered
inference (season cluster = shipping CI, cluster-mean t beside it).

Controls, both exact:
  * 2021-22..2025-26 -> `data/capstone_pergame.csv` (D132 certified);
  * every other scorable season -> the same-run `fit_production` walk-forward
    already computed by `ca_hist.py` (`data/carryall_hist_rows.npz`).
Each config changes ONLY the schedule layer, so the pairing is exact:
`p_cfg = sigmoid((SCALE*logit(p_ctrl) + dm) / SCALE)`.

  python scripts/ca_cfg.py
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import nbapred.threads as _t  # noqa: E402

_t.pin(1)
import numpy as np  # noqa: E402

from ca_bank import Layer, load_bank  # noqa: E402
from ca_era import CONFIGS  # noqa: E402
from ca_ladder import ll, refit_dates  # noqa: E402
from nbapred.model.production import SCALE  # noqa: E402

CERT = REPO / "data" / "capstone_pergame.csv"
HIST = REPO / "data" / "carryall_hist_rows.npz"
CERT_SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")


def config_dm(bank, seasons):
    """dm per game for each config, walk-forward at the production cadence.

    C3 is STATEFUL (its break persists), so each config is run once over the
    whole chronology rather than per season.
    """
    seas = bank["season"].astype(str)
    sel = np.where(np.isin(seas, list(seasons)))[0]
    order = sel[np.argsort(bank["date"][sel], kind="stable")]
    dates = bank["date"][order].astype("datetime64[D]").astype(dt.date)

    gov = np.empty(len(order), dtype=object)
    rds_all = []
    for s in seasons:
        m = seas[order] == s
        if not m.any():
            continue
        rds = refit_dates(sorted(set(dates[m])))
        rds_all += [(s, d) for d in rds]
        j = 0
        for i in np.where(m)[0]:
            while j + 1 < len(rds) and rds[j + 1] <= dates[i]:
                j += 1
            gov[i] = rds[j]
    rds_all.sort(key=lambda x: x[1])

    base = np.zeros(len(order))
    L0 = Layer(bank)
    for s, d in rds_all:
        rows = np.where((gov == d) & (seas[order] == s))[0]
        if not len(rows):
            continue
        b5, _, _, _, _, _ = L0.fit(d)
        for i in rows:
            base[i] = L0.sched_value(order[i], b5, {}, {}, [])

    dm, meta = {}, {}
    for cfg, kw in CONFIGS.items():
        if cfg == "BASE":
            continue
        L = Layer(bank, **kw)
        v = np.zeros(len(order))
        for s, d in rds_all:
            rows = np.where((gov == d) & (seas[order] == s))[0]
            if not len(rows):
                continue
            b5, _, _, _, _, _ = L.fit(d)
            for i in rows:
                v[i] = L.sched_value(order[i], b5, {}, {}, []) - base[i]
        dm[cfg] = v
        meta[cfg] = dict(breaks=L.breaks)
    return order, dates, dm, meta


if __name__ == "__main__":
    bank = load_bank()
    cert = {r["game_id"]: (float(r["p_us"]), float(r["y"]), r["season"])
            for r in csv.DictReader(open(CERT))}
    ctrl = {g: v[0] for g, v in cert.items()}
    ymap = {g: v[1] for g, v in cert.items()}
    seasons = list(CERT_SEASONS)
    pool = list(CERT_SEASONS)
    strata = []
    if HIST.exists():
        z = np.load(HIST, allow_pickle=True)
        hj = json.load(open(REPO / "data" / "carryall_hist.json"))
        pool, strata = hj["poolable"], hj["strata"]
        for g, p, yy in zip(z["gid"].astype(str), z["p0"], z["y"]):
            ctrl.setdefault(g, float(p))
            ymap.setdefault(g, float(yy))
        seasons = sorted(set(pool) | set(strata))
        print(f"[cfg] using historical control from {HIST.name}: "
              f"poolable {pool}\n      strata {strata}")
    else:
        print("[cfg] WARNING: no historical control yet; certified corpus only")

    order, dates, dm, meta = config_dm(bank, seasons)
    gids = bank["gid"][order].astype(str)
    keep = np.array([g in ctrl for g in gids])
    idx = np.where(keep)[0]
    p0 = np.array([ctrl[g] for g in gids[idx]])
    y = np.array([ymap[g] for g in gids[idx]])
    seas = bank["season"].astype(str)[order][idx]
    m_base = SCALE * np.log(p0 / (1 - p0))
    dts = np.array([dates[i] for i in idx])
    print(f"[cfg] scored {len(idx)} games over {len(set(seas))} seasons")

    from nbapred.eval.splits import Panel, full_report
    out = {"prereg_sha256": "9a4a414db294ba44908b4a4ee5f0bd490e0b2d0094"
                            "293eb0f2103a101459bd9b",
           "poolable": pool, "strata": strata, "n": int(len(idx)),
           "breaks": {k: v["breaks"] for k, v in meta.items()},
           "control_ll_pool": float(ll(y[np.isin(seas, pool)],
                                       p0[np.isin(seas, pool)]).mean()),
           "cfg": {}}
    universes = [("POOLABLE", np.isin(seas, pool)),
                 ("CERTIFIED-5", np.isin(seas, CERT_SEASONS)),
                 ("W3 drift 2010-19",
                  np.isin(seas, [s for s in pool if s <= "2018-19"]))]
    universes += [(f"stratum {s}", seas == s) for s in strata]
    for cfg in dm:
        d = dm[cfg][idx]
        p1 = 1.0 / (1.0 + np.exp(-(m_base + d) / SCALE))
        st = {}
        for tag, msk in universes:
            if msk.sum() < 300:
                continue
            pan = Panel.from_logloss(seas[msk], y[msk], p0[msk], p1[msk],
                                     date=dts[msk], label=f"{cfg}|{tag}")
            rep = full_report(pan, B=2000, seed=20260802)
            cl = rep["clustering"]["season_cluster_boot"]
            t = rep["clustering"]["season_mean_t"]
            ro = [f["fold"]["est"] for f in rep["rolling_origin"]["folds"]]
            st[tag] = dict(n=int(msk.sum()),
                           rms_dm=float(np.sqrt((d[msk] ** 2).mean())),
                           delta=rep["pooled"]["est"],
                           iid_lo=rep["pooled"]["lo"], iid_hi=rep["pooled"]["hi"],
                           cl_lo=cl["lo"], cl_hi=cl["hi"],
                           t_lo=t["lo"], t_hi=t["hi"], t_sig=bool(t["sig"]),
                           mde80=rep["pooled_mde80"],
                           ro=ro, ro_pos=int(sum(x > 0 for x in ro)),
                           ro_n=len(ro),
                           per_season={x["season"]: x["est"]
                                       for x in rep["per_season"]},
                           loso=[f["test_on"]["est"] for f in rep["loso"]["folds"]],
                           loso_sign=rep["loso"]["sign_consistency"],
                           I2=rep["era"]["I2"],
                           era_stable=bool(rep["era"]["era_stable"]),
                           deff=rep["clustering"].get("design_effect_season"),
                           flags=rep["verdict"]["flags"])
        out["cfg"][cfg] = st
    json.dump(out, open(REPO / "data" / "carryall_cfg.json", "w"), indent=1)

    print(f"\n{'cfg':7s} {'universe':20s} {'n':>6s} {'rms(dm)':>8s} "
          f"{'delta':>10s} {'season-cluster CI':>24s} "
          f"{'cluster-t CI':>22s} {'MDE80':>8s} {'RO':>5s}")
    for cfg, st in out["cfg"].items():
        for tag, v in st.items():
            print(f"{cfg:7s} {tag:20s} {v['n']:6d} {v['rms_dm']:8.4f} "
                  f"{v['delta']:+10.5f} ({v['cl_lo']:+10.5f},{v['cl_hi']:+10.5f}) "
                  f"({v['t_lo']:+9.5f},{v['t_hi']:+9.5f}) {v['mde80']:8.5f} "
                  f"{v['ro_pos']}/{v['ro_n']}")
    print("\nwrote data/carryall_cfg.json")
