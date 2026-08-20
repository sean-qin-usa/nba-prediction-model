#!/usr/bin/env python3
"""D246 — OPENER-TO-CLOSE MOVEMENT TEACHER, two heads. Prereg 661404b6...

    Y - O = (C - O) + (Y - C)
            head M     head F

head M predicts what the market will absorb by close; head F predicts what may
remain mispriced AT close. Both use OPENER-TIME FEATURES ONLY. The closing line
is a TRAINING TARGET and never an input — asserted by a leakage check that
permutes the close within season and requires every feature bit-identical.

Why this rather than more work on the composition channel: D245d measured a ~14x
two-stage attenuation (blend share 0.652 x offset edge 0.3413) on anything
entering through the blend. This predicts a market quantity directly. The target
is also 5.9x less noisy than the outcome residual (sd 2.303 vs 13.60).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from scipy import stats                                           # noqa: E402
from scipy.optimize import minimize_scalar                        # noqa: E402

FROM = "2019-20"
LAM = 50.0


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def nll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_scale(m, y):
    return float(minimize_scalar(
        lambda s: nll(1 / (1 + np.exp(-m / s)), y).mean(),
        bounds=(2, 25), method="bounded").x)


def clus(v):
    v = np.asarray(v, float); k = len(v)
    se = v.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return v.mean(), v.mean() - tc * se, v.mean() + tc * se, v.mean() / se, k


def features(f):
    """OPENER-TIME ONLY. Built before close_margin is ever referenced."""
    return np.column_stack([
        np.ones(len(f)),
        f["edge"].to_numpy(float),
        f["open_margin"].abs().to_numpy(float),
        f["rest_diff"].to_numpy(float),
        f["eo_diff"].to_numpy(float),
        f["tot"].to_numpy(float) - 225.0,
        f["days_in"].to_numpy(float) / 100.0,
        f["mkt_ll"].to_numpy(float) - 0.61,
    ])


def ridge(X, y, lam=LAM):
    P = lam * np.eye(X.shape[1]); P[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + P, X.T @ y)


def main():
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
    f = f[f.season >= FROM].copy()
    f["game_id"] = zf(f["game_id"])
    f["game_date"] = pd.to_datetime(f["game_date"])
    cap = pd.read_csv(ROOT / "data" / "capstone_2019_26.csv")
    cap["game_id"] = zf(cap["game_id"])
    pit = pd.read_csv(ROOT / "data" / "pit_frame.csv.gz")
    pit["game_id"] = zf(pit["game_id"])
    f = (f.merge(cap[["game_id", "eo_home", "eo_away"]], on="game_id", how="left")
           .merge(pit[["game_id", "rest_home", "rest_away"]], on="game_id",
                  how="left"))
    f = f.dropna(subset=["open_margin", "close_margin", "margin_actual",
                         "m_us_blind"])
    f["edge"] = f["m_us_blind"] - f["open_margin"]
    f["eo_diff"] = (f["eo_home"] - f["eo_away"]).fillna(0.0)
    f["rest_diff"] = (f["rest_home"].clip(upper=7).fillna(0)
                      - f["rest_away"].clip(upper=7).fillna(0))
    f["tot"] = pd.to_numeric(f["open_total"], errors="coerce").fillna(225.0)
    f.loc[(f.tot < 150) | (f.tot > 290), "tot"] = 225.0
    f["days_in"] = (f["game_date"]
                    - f.groupby("season")["game_date"].transform("min")).dt.days
    p = np.clip(1 / (1 + np.exp(-f["open_margin"] / 6.96)), 1e-9, 1 - 1e-9)
    yb = (f["margin_actual"] > 0).astype(float)
    f["llo"] = -(yb * np.log(p) + (1 - yb) * np.log(1 - p))
    per = f.groupby("game_date")["llo"].mean()
    roll = per.rolling(60, min_periods=20).mean().shift(1)
    f["mkt_ll"] = f["game_date"].map(roll).fillna(0.61)
    f["y"] = yb
    f = f.sort_values(["season", "game_date", "game_id"]).reset_index(drop=True)
    print(f"frame {len(f):,} games, {f.season.nunique()} seasons")

    # ---- LEAKAGE CHECK: permute the close, features must not move ------
    X0 = features(f)
    g = f.copy()
    rng = np.random.default_rng(246)
    for s in g.season.unique():
        m = (g.season == s).to_numpy()
        g.loc[m, "close_margin"] = rng.permutation(g.loc[m, "close_margin"].to_numpy())
    X1 = features(g)
    assert np.abs(X0 - X1).max() == 0.0, "FEATURES DEPEND ON THE CLOSE"
    print(f"leakage check: features bit-identical under a within-season "
          f"permutation of the close (max|d| {np.abs(X0-X1).max():.1e})")

    seasons = sorted(f.season.unique())
    rows = []
    for i, s in enumerate(seasons):
        if i == 0:
            continue
        tr, te = f[f.season.isin(seasons[:i])], f[f.season == s]
        Xtr, Xte = features(tr), features(te)
        dM_tr = (tr.close_margin - tr.open_margin).to_numpy(float)
        dF_tr = (tr.margin_actual - tr.close_margin).to_numpy(float)
        dM_te = (te.close_margin - te.open_margin).to_numpy(float)
        bM, bF = ridge(Xtr, dM_tr), ridge(Xtr, dF_tr)
        pM_te, pF_te = Xte @ bM, Xte @ bF
        # OOS R^2 of each head against a no-movement baseline
        r2M = 1 - ((dM_te - pM_te) ** 2).sum() / (dM_te ** 2).sum()
        dF_te = (te.margin_actual - te.close_margin).to_numpy(float)
        r2F = 1 - ((dF_te - pF_te) ** 2).sum() / (dF_te ** 2).sum()
        # combined forecast, gamma/eta fitted on the training block
        A = np.column_stack([Xtr @ bM, Xtr @ bF])
        ge = np.linalg.lstsq(A, (tr.margin_actual - tr.open_margin).to_numpy(float),
                             rcond=None)[0]
        m_te = te.open_margin.to_numpy(float) + np.column_stack([pM_te, pF_te]) @ ge
        m_tr = tr.open_margin.to_numpy(float) + A @ ge
        ytr, yte = tr.y.to_numpy(float), te.y.to_numpy(float)
        sc = fit_scale(m_tr, ytr)
        ll_new = float(nll(1 / (1 + np.exp(-m_te / sc)), yte).mean())
        sc0 = fit_scale(tr.m_us.to_numpy(float), ytr)
        ll_off = float(nll(1 / (1 + np.exp(-te.m_us.to_numpy(float) / sc0)),
                           yte).mean())
        # T4: does each head survive controlling for the other?
        aM = np.linalg.lstsq(np.column_stack([np.ones(len(tr)), Xtr @ bF]),
                             dM_tr, rcond=None)[0]
        resM = dM_tr - np.column_stack([np.ones(len(tr)), Xtr @ bF]) @ aM
        inc_M = float(np.corrcoef(Xtr @ bM, resM)[0, 1])
        rows.append(dict(season=s, n=len(te), r2_M=float(r2M), r2_F=float(r2F),
                         gamma=float(ge[0]), eta=float(ge[1]),
                         ll_offset=ll_off, ll_teacher=ll_new,
                         d_ll=ll_new - ll_off, inc_M=inc_M))
    d = pd.DataFrame(rows)
    print("\n" + d[["season", "n", "r2_M", "r2_F", "gamma", "eta",
                    "ll_offset", "ll_teacher", "d_ll"]].to_string(
        index=False, float_format=lambda v: f"{v:9.5f}"))

    print(f"\n  T1 head M mean OOS R^2 {d.r2_M.mean():+.4f}  "
          f"({'CONFIRMED' if d.r2_M.mean() > 0 else 'REFUTED'}; D147 got 0.171 "
          f"with richer live features)")
    print(f"  T2 head F mean OOS R^2 {d.r2_F.mean():+.4f}  "
          f"({'CONFIRMED — F weaker than M' if d.r2_F.mean() < d.r2_M.mean() else 'REFUTED'})")
    m, lo, hi, t, k = clus(d.d_ll)
    print(f"\n  T3 full-stack vs the SHIPPED OFFSET: {m:+.6f}  "
          f"CI [{lo:+.6f}, {hi:+.6f}]  t {t:+.2f}  better {int((d.d_ll<0).sum())}/{k}")
    print(f"     {'SHIP' if hi < 0 else 'NO SHIP — CI includes zero'}")
    print(f"  T4 corr(head M, head F residual) mean {d.inc_M.mean():+.3f} "
          f"({'heads are distinct' if abs(d.inc_M.mean()) > 0.1 else 'heads largely redundant'})")
    json.dump({"rows": rows}, open(ROOT / "data" / "d246_movement.json", "w"),
              default=float)
    print("\nwrote data/d246_movement.json")


if __name__ == "__main__":
    main()
