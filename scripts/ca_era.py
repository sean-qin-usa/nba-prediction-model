#!/usr/bin/env python3
"""PART B — HOW FAST DO WE UPDATE THROUGH ERAS?

Prereg data/carryall_prereg.md sha256 9a4a414d...59bd9b, §§5-8.

Two deliverables, both MARGIN-scale (no log-loss endpoint here):
  1. the walk-forward home-edge track of the incumbent estimator against the
     truth, so the LAG is visible and quantifiable -- half-life of the
     tracking error after the COVID step, and the lag bias under the measured
     -0.053..-0.093 pts/season drift;
  2. the same track for each pre-registered adaptation config (C1 trend-aware,
     C2 data-driven shrinkage target, C3 change-point window), plus an ORACLE
     that knows the truth, which upper-bounds what any config can recover.

  python scripts/ca_era.py
"""
from __future__ import annotations

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
from nbapred.model.production import SCALE  # noqa: E402

CONFIGS = {
    "BASE":  dict(),
    "C1":    dict(trend=True),
    "C2":    dict(prior_mode="data5"),
    "C3":    dict(changepoint=True),
    "C2+C3": dict(prior_mode="data5", changepoint=True),
}
FIRST = "2010-11"


def weekly_dates(bank, seasons):
    seas = bank["season"].astype(str)
    out = []
    for s in seasons:
        d = sorted(set(bank["date"][seas == s].astype("datetime64[D]").astype(dt.date)))
        last = None
        for x in d:
            if last is None or (x - last).days >= 7:
                out.append((s, x))
                last = x
    return out


def truth_series(bank, seasons):
    """TRUTH = the season mean home margin (the era parameter's realised
    value), plus a 121-day CENTRED local mean for the within-season view."""
    seas = bank["season"].astype(str)
    by_season = {s: float(bank["margin"][seas == s].mean()) for s in seasons}
    di = bank["date"].astype("datetime64[D]").astype("int64")
    return by_season, di


def centred_local(bank, day, half=182):
    di = bank["date"].astype("datetime64[D]").astype("int64")
    m = (di >= day - half) & (di <= day + half)
    return float(bank["margin"][m].mean()) if m.sum() > 200 else np.nan


if __name__ == "__main__":
    bank = load_bank()
    seas = bank["season"].astype(str)
    seasons = [s for s in sorted(set(seas)) if s >= FIRST]
    wd = weekly_dates(bank, seasons)
    tru_season, di = truth_series(bank, seasons)
    print(f"[era] {len(wd)} weekly refits over {len(seasons)} seasons "
          f"{seasons[0]}..{seasons[-1]}", flush=True)

    track = {}
    for cfg, kw in CONFIGS.items():
        L = Layer(bank, **kw)
        rec = []
        for s, d in wd:
            b5, _, _, n, w, _ = L.fit(d)
            rec.append(dict(season=s, date=str(d), he=float(b5[0]),
                            n=int(n), w=round(float(w), 5),
                            lo=L.last["lo"], slope=round(L.last["slope"], 5),
                            prior0=round(L.last["prior0"], 4)))
        track[cfg] = rec
        print(f"[era] {cfg:6s} done; breaks={L.breaks}", flush=True)
        if L.breaks:
            track.setdefault("_breaks", {})[cfg] = L.breaks

    # ---- centred local truth at each refit date
    days = sorted({int(np.datetime64(dt.date.fromisoformat(r["date"]),
                                     "D").astype("int64"))
                   for r in track["BASE"]})
    loc = {d: centred_local(bank, d) for d in days}

    out = {"prereg_sha256": "9a4a414db294ba44908b4a4ee5f0bd490e0b2d0094"
                            "293eb0f2103a101459bd9b",
           "seasons": seasons, "truth_season": tru_season,
           "track": track, "local_truth": {str(k): v for k, v in loc.items()}}

    # ---- LAG METRICS -----------------------------------------------------
    print(f"\n{'cfg':7s} {'MAE vs season truth':>20s} {'bias':>8s} "
          f"{'MAE vs local':>13s}")
    lag = {}
    for cfg in CONFIGS:
        err_s, err_l = [], []
        for r in track[cfg]:
            d = int(np.datetime64(dt.date.fromisoformat(r["date"]),
                                  "D").astype("int64"))
            err_s.append(r["he"] - tru_season[r["season"]])
            if np.isfinite(loc[d]):
                err_l.append(r["he"] - loc[d])
        lag[cfg] = dict(mae_season=float(np.mean(np.abs(err_s))),
                        bias_season=float(np.mean(err_s)),
                        mae_local=float(np.mean(np.abs(err_l))),
                        bias_local=float(np.mean(err_l)))
        print(f"{cfg:7s} {lag[cfg]['mae_season']:20.4f} "
              f"{lag[cfg]['bias_season']:+8.4f} {lag[cfg]['mae_local']:13.4f}")

    # ---- COVID step: half-life of the tracking error ---------------------
    # The step DOWN is dated at the 2020-21 opener; the step UP at the
    # 2021-22 opener.  err(t) = err0 * 2^(-t/H) fitted by OLS on log|err|.
    print("\nCOVID STEP TRACKING (truth = the new season's own mean)")
    step = {}
    for cfg in CONFIGS:
        st = {}
        for tag, s_new in (("down", "2020-21"), ("up", "2021-22")):
            rs = [r for r in track[cfg] if r["season"] == s_new]
            if not rs:
                continue
            d0 = np.datetime64(dt.date.fromisoformat(rs[0]["date"]), "D").astype("int64")
            t = np.array([int(np.datetime64(dt.date.fromisoformat(r["date"]),
                                            "D").astype("int64")) - d0
                          for r in rs], float)
            e = np.array([r["he"] - tru_season[s_new] for r in rs])
            ok = np.abs(e) > 1e-6
            H = np.nan
            if ok.sum() >= 5 and np.sign(e[0]) != 0:
                sgn = np.sign(e[0])
                m = ok & (np.sign(e) == sgn)
                if m.sum() >= 5:
                    A = np.c_[np.ones(m.sum()), t[m]]
                    sol = np.linalg.lstsq(A, np.log(np.abs(e[m])), rcond=None)[0]
                    if sol[1] < 0:
                        H = float(np.log(0.5) / sol[1])
            st[tag] = dict(err0=float(e[0]), err_end=float(e[-1]),
                           n_refits=len(rs), half_life_days=H,
                           mean_abs_err=float(np.mean(np.abs(e))),
                           days_to_half=(float(
                               t[np.argmax(np.abs(e) <= abs(e[0]) / 2)])
                               if (np.abs(e) <= abs(e[0]) / 2).any() else np.nan))
        step[cfg] = st
        d_, u_ = st.get("down", {}), st.get("up", {})
        print(f"  {cfg:7s} DOWN err0 {d_.get('err0', float('nan')):+.3f} -> "
              f"end {d_.get('err_end', float('nan')):+.3f}  H="
              f"{d_.get('half_life_days', float('nan')):7.1f}d  "
              f"mean|err| {d_.get('mean_abs_err', float('nan')):.3f}   |   "
              f"UP err0 {u_.get('err0', float('nan')):+.3f} -> "
              f"{u_.get('err_end', float('nan')):+.3f}  H="
              f"{u_.get('half_life_days', float('nan')):7.1f}d  "
              f"mean|err| {u_.get('mean_abs_err', float('nan')):.3f}")
    out["lag"] = lag
    out["step"] = step

    # ---- what the lag COST, by the additive identity ---------------------
    # dm = (truth - estimate) applied as a home-edge correction; the ORACLE
    # gain is 0.5*E[p(1-p)]*E[dm^2]/SCALE^2 and upper-bounds any config.
    Epq = 0.20636
    print("\nORACLE UPPER BOUND on log loss recoverable from the tracking lag")
    cost = {}
    for cfg in CONFIGS:
        c = {}
        for tag, sub in (("2020-21 (W1)", ["2020-21"]),
                         ("2021-22 H1 (W2)", ["2021-22"]),
                         ("2010-11..2018-19 (W3)",
                          [s for s in seasons if s <= "2018-19"]),
                         ("all", seasons)):
            e = [r["he"] - tru_season[r["season"]] for r in track[cfg]
                 if r["season"] in sub
                 and not (tag.endswith("(W2)") and r["date"] > "2022-01-31")]
            if not e:
                continue
            e = np.array(e)
            c[tag] = dict(rms=float(np.sqrt((e ** 2).mean())),
                          bias=float(e.mean()),
                          oracle_ll=float(0.5 * Epq * (e ** 2).mean() / SCALE ** 2))
        cost[cfg] = c
        if cfg == "BASE":
            for k, v in c.items():
                print(f"  {k:24s} rms {v['rms']:.4f} pts  bias {v['bias']:+.4f}"
                      f"  oracle dLL {v['oracle_ll']:+.5f}")
    out["oracle_cost"] = cost
    json.dump(out, open(REPO / "data" / "carryall_era.json", "w"), indent=1)
    print("\nwrote data/carryall_era.json")
