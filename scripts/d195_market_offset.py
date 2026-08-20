#!/usr/bin/env python3
"""D195 — THE MARKET-OFFSET (OUTCOME-RESIDUAL) MODEL.

D193 measured that the market-blind model does NOT beat the opening line as a
general forecaster (capture fraction -0.019 in probability space, -0.254 in
margin space).  The direct consequence is that a deployable model should take
the opening line as its PRIOR and predict its residual, rather than forecasting
the game independently and comparing afterwards.

`nbapred/market/anchored.py` already implements the MOVEMENT head (predict the
close from the open, D147).  This script builds the missing OUTCOME head:

    r = actual_margin - open_margin          (what the opener got wrong)
    m_final = open_margin + f_theta(x)       (the deployable forecast)

DECLARED BEFORE SCORING
  Baseline to beat: f = 0, i.e. THE OPENER ITSELF.  Per D176, beating a null is
  necessary but not sufficient — the incumbent here is the opener, and that is
  the comparison that decides.
  Features, all knowable at the OPEN (LEAKAGE.md):
    x1  model_edge   = m_us - open_margin   our market-blind disagreement
    x2  rest_diff                            schedule layer input
    x3  absence_diff  TRAILING out-load, home minus away (never tonight's 5pm
                      report, which does not exist when the opener is posted)
    x4  gidx          games into the season (cold-start / prior-dominated)
    x5  |open_margin| favourite magnitude
  Estimator: RIDGE with strong shrinkage toward ZERO (i.e. toward "the opener is
  right"), lambda chosen by GENERALISED CROSS-VALIDATION inside the training
  fold only — a Type-A constant per D192, no grid, no held-out data.
  Protocol: fit on seasons 1..k, score season k+1, roll forward.  Scored in BOTH
  margin space (RMSE) and probability space (log loss), against the opener, on
  identical games — the D193 frame.

Read-only.  Nothing ships.  No production default changed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from scipy.optimize import minimize_scalar                        # noqa: E402

import oc_capacity as oc                                          # noqa: E402
from nbapred.db import connect                                    # noqa: E402
from nbapred import teams as T                                    # noqa: E402

MODERN = "2019-20"
FEATS = ["model_edge", "rest_diff", "absence_diff", "gidx", "abs_open"]


def nll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_scale(m, y):
    return float(minimize_scalar(
        lambda s: nll(1 / (1 + np.exp(-m / s)), y),
        bounds=(2.0, 25.0), method="bounded").x)


def ridge_gcv(X, y, lams=None):
    """Ridge with lambda by generalised cross-validation, computed INSIDE the
    training fold. Type-A (D192): no grid over held-out performance, no DOF."""
    if lams is None:
        lams = np.logspace(0, 6, 40)
    n, p = X.shape
    Xc = X - X.mean(0)
    yc = y - y.mean()
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    Uty = U.T @ yc
    best, best_g = None, np.inf
    for lam in lams:
        d = s ** 2 / (s ** 2 + lam)
        fit = U @ (d * Uty)
        dof = d.sum()
        rss = float(((yc - fit) ** 2).sum())
        g = rss / max(n - dof, 1e-9) ** 2 * n
        if g < best_g:
            best_g, best = g, lam
    d = s / (s ** 2 + best)
    beta = Vt.T @ (d * Uty)
    return beta, float(y.mean() - X.mean(0) @ beta), float(best)


def build(df):
    d = df[df["season"] >= MODERN].copy()
    ok = (d["m_us"].notna() & d["open_margin"].notna() &
          d["close_margin"].notna() & d["margin_actual"].notna())
    d = d[ok].reset_index(drop=True)

    # x1, x5
    d["model_edge"] = d["m_us"] - d["open_margin"]
    d["abs_open"] = d["open_margin"].abs()

    # x4 games into season
    idx, cnt = np.zeros(len(d), int), {}
    for i, (s, h, a) in enumerate(zip(d["season"], d["home"], d["away"])):
        ih, ia = cnt.get((s, h), 0), cnt.get((s, a), 0)
        idx[i] = max(ih, ia)
        cnt[(s, h)], cnt[(s, a)] = ih + 1, ia + 1
    d["gidx"] = idx

    # x2 rest differential, from the schedule spine
    con = connect(read_only=True)
    g = pd.DataFrame(con.execute(
        "SELECT DISTINCT game_date, team_abbrev FROM nba_games "
        "WHERE game_id LIKE '002%'").fetchall(), columns=["d", "t"])
    g["d"] = pd.to_datetime(g["d"])
    g = g.sort_values("d")
    g["prev"] = g.groupby("t")["d"].shift(1)
    rest = {(r.t, r.d): (r.d - r.prev).days if pd.notna(r.prev) else 7
            for r in g.itertuples()}
    dd = pd.to_datetime(d["game_date"])
    d["rest_diff"] = [min(rest.get((h, x), 7), 7) - min(rest.get((a, x), 7), 7)
                      for h, a, x in zip(d["home"], d["away"], dd)]

    # x3 TRAILING absence load differential (never tonight's report)
    inj = pd.DataFrame(con.execute(
        "SELECT game_date, team, count(*) n FROM injury_reports_pit "
        "WHERE status IN ('Out','Doubtful') GROUP BY 1,2").fetchall(),
        columns=["d", "team", "n"])
    con.close()
    amap, unres = T.resolve_map(sorted(inj["team"].unique()))
    if unres:
        print(f"  [teams] {len(unres)} unresolvable, REPORTED: {unres[:5]}")
    inj["team"] = inj["team"].map(amap)
    inj = inj[inj["team"].notna()]
    inj["d"] = pd.to_datetime(inj["d"])
    inj = inj.sort_values("d")
    inj["tr"] = inj.groupby("team")["n"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).mean())
    m = dict(zip(zip(inj["team"], inj["d"]), inj["tr"]))
    hv = np.array([m.get((h, x), np.nan) for h, x in zip(d["home"], dd)], float)
    av = np.array([m.get((a, x), np.nan) for a, x in zip(d["away"], dd)], float)
    diff = hv - av
    d["absence_diff"] = np.where(np.isfinite(diff), diff, 0.0)
    return d


def main():
    df, _ = oc.load()
    d = build(df)
    seas = sorted(d["season"].unique())
    print(f"frame {seas[0]}..{seas[-1]}  n={len(d)}  features={FEATS}\n")

    rows = []
    for i in range(2, len(seas)):
        tr = d[d["season"].isin(seas[:i])]
        te = d[d["season"] == seas[i]]
        Xtr = tr[FEATS].to_numpy(float)
        ytr = (tr["margin_actual"] - tr["open_margin"]).to_numpy(float)
        beta, b0, lam = ridge_gcv(Xtr, ytr)
        pred = te[FEATS].to_numpy(float) @ beta + b0
        m_off = te["open_margin"].to_numpy(float) + pred
        a = te["margin_actual"].to_numpy(float)
        yb = (a > 0).astype(float)
        ytrb = (tr["margin_actual"].to_numpy(float) > 0).astype(float)

        r = {}
        for nm, mm, mmtr in (
                ("offset", m_off, tr["open_margin"].to_numpy(float) +
                 (Xtr @ beta + b0)),
                ("opener", te["open_margin"].to_numpy(float),
                 tr["open_margin"].to_numpy(float)),
                ("blind", te["m_us"].to_numpy(float),
                 tr["m_us"].to_numpy(float)),
                ("close", te["close_margin"].to_numpy(float),
                 tr["close_margin"].to_numpy(float))):
            s = fit_scale(mmtr, ytrb)
            r[nm] = dict(rmse=float(np.sqrt(((a - mm) ** 2).mean())),
                         ll=nll(1 / (1 + np.exp(-mm / s)), yb))
        rows.append((seas[i], r, lam, beta, len(te)))
        print(f"  {seas[i]}  lambda={lam:9.1f}  "
              f"RMSE offset {r['offset']['rmse']:.4f} vs opener "
              f"{r['opener']['rmse']:.4f}  |  LL offset {r['offset']['ll']:.5f} "
              f"vs opener {r['opener']['ll']:.5f}")

    w = np.array([x[4] for x in rows], float)
    print("\n" + "=" * 74)
    print("POOLED, walk-forward, identical games")
    print("=" * 74)
    print(f"  {'source':10} {'RMSE':>10} {'log loss':>11}")
    pooled = {}
    for nm in ("blind", "opener", "offset", "close"):
        rm = np.average([x[1][nm]["rmse"] for x in rows], weights=w)
        ll = np.average([x[1][nm]["ll"] for x in rows], weights=w)
        pooled[nm] = dict(rmse=float(rm), ll=float(ll))
        print(f"  {nm:10} {rm:10.4f} {ll:11.5f}")

    cap_ll = ((pooled["opener"]["ll"] - pooled["offset"]["ll"]) /
              (pooled["opener"]["ll"] - pooled["close"]["ll"]))
    cap_m = ((pooled["opener"]["rmse"] - pooled["offset"]["rmse"]) /
             (pooled["opener"]["rmse"] - pooled["close"]["rmse"]))
    print(f"\n  CAPTURE FRACTION (log loss) : {cap_ll:+.3f}   "
          f"(market-blind model was -0.019)")
    print(f"  CAPTURE FRACTION (margin)   : {cap_m:+.3f}   "
          f"(market-blind model was -0.254)")

    per = np.array([(x[1]["opener"]["ll"] - x[1]["offset"]["ll"]) for x in rows])
    K = len(per)
    se = per.std(ddof=1) / np.sqrt(K)
    print(f"\n  per-season LL gain over the opener: {np.round(per, 5).tolist()}")
    print(f"  mean {per.mean():+.5f}  t={per.mean()/se:+.2f} (K={K}) -> "
          f"{'SIG' if abs(per.mean()/se) > oc.t_crit(K-1) else 'ns'}")
    print(f"  positive in {(per > 0).sum()}/{K} seasons")

    print("\n  mean fitted coefficients (margin points per unit):")
    B = np.mean([x[3] for x in rows], axis=0)
    for f, b in zip(FEATS, B):
        print(f"    {f:14} {b:+.5f}")

    json.dump(dict(pooled=pooled, capture_ll=float(cap_ll),
                   capture_margin=float(cap_m),
                   per_season_gain=per.tolist(),
                   coefs=dict(zip(FEATS, B.tolist()))),
              open(ROOT / "data" / "d195_offset.json", "w"), indent=1)
    print("\nwrote data/d195_offset.json")


if __name__ == "__main__":
    main()
