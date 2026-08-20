#!/usr/bin/env python3
"""D254 — SHOULD THE OFFSET SPEND OUR EDGE CONVEXLY RATHER THAN LINEARLY?

The only shippable idea to come out of D253's sweep. Two independent findings
point the same way:

  * D252b, by slicing: the largest-|edge| quintile is where the shipped offset
    beats the opener MOST (-0.00577, the largest of any slice in the family).
  * D253b, by fitting: `our_edge_abs` carries a NEGATIVE coefficient in 100% of
    folds, and `our_edge` + `our_edge_abs` alone give OOS R^2 +0.00837 while the
    other 24 features sit on the null.

The production layer is LINEAR in the edge — `0.3564 * (m_blind - m_open)` — so
it spends the same fraction whether we disagree by half a point or by eight. If
the relationship is genuinely convex, that is a shape the current layer cannot
express, and it sits INSIDE the offset, so the 4.5x two-stage attenuation
(D245d, magnitude corrected in D252) does not apply to it.

ANTISYMMETRY IS A CONSTRAINT, NOT A CHOICE. The correction must be odd in the
edge: if our disagreement flips sign the correction must flip with it. So the
convex term is `edge * |edge|`, never `edge^2`, which would push the same
direction for both signs and is simply wrong here.

ARMS, fixed before any endpoint is read. All refit walk-forward per fold at the
production ridge, and the probability scale is refit per fold per arm so no arm
is judged through another's calibration:

  L  LINEAR      b1*edge + rest + |open|             (the incumbent)
  Q  QUADRATIC   b1*edge + b2*edge*|edge| + ...
  S  SPLINE      b1*edge + b2*max(|edge|-k,0)*sign(edge) + ...   k = 3.0 pts
  C  CLIPPED     b1*clip(edge, -k, k) + b2*(edge - clip) + ...   k = 3.0 pts

Q and S add one parameter; C reparameterises into an inner and an outer slope.
The knot k = 3.0 is fixed in advance at roughly the 60th percentile of |edge|,
NOT tuned — a tuned knot on 19 seasons is exactly the D239 failure mode.

PRIMARY ENDPOINT: season-clustered mean change in log loss against arm L.
MDE80 is computed from the arms' own season-level dispersion and REPORTED
ALONGSIDE the estimate, so an underpowered null is never read as a refutation.
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

LAM = 3000.0
KNOT = 3.0


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def nll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_scale(m, y):
    return float(minimize_scalar(
        lambda s: nll(1 / (1 + np.exp(-m / s)), y).mean(),
        bounds=(3, 15), method="bounded").x)


def clus(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    k = len(v)
    se = v.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return v.mean(), v.mean() - tc * se, v.mean() + tc * se, v.mean() / se, k


def mde80(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return 2.80 * v.std(ddof=1) / np.sqrt(len(v))


def design(arm, edge, rest, absopen):
    base = [rest, absopen]
    if arm == "L":
        cols = [edge]
    elif arm == "Q":
        cols = [edge, edge * np.abs(edge)]
    elif arm == "S":
        cols = [edge, np.maximum(np.abs(edge) - KNOT, 0.0) * np.sign(edge)]
    elif arm == "C":
        inner = np.clip(edge, -KNOT, KNOT)
        cols = [inner, edge - inner]
    else:
        raise ValueError(arm)
    return np.column_stack(cols + base)


def main():
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
    f["game_id"] = zf(f["game_id"])
    pit = pd.read_csv(ROOT / "data" / "pit_frame.csv.gz")
    pit["game_id"] = zf(pit["game_id"])
    f = f.merge(pit[["game_id", "rest_home", "rest_away"]], on="game_id",
                how="left")
    f = f.dropna(subset=["open_margin", "margin_actual", "m_us_blind"]).copy()
    f["rest_diff"] = (f.rest_home.clip(upper=7).fillna(0)
                      - f.rest_away.clip(upper=7).fillna(0))
    f["edge"] = f.m_us_blind - f.open_margin
    f["y"] = (f.margin_actual > 0).astype(float)
    f = f.sort_values(["season", "game_id"]).reset_index(drop=True)
    seasons = sorted(f.season.unique())
    q = np.percentile(f.edge.abs(), [50, 60, 70])
    print(f"{len(f):,} games, {len(seasons)} seasons")
    print(f"|edge| percentiles 50/60/70 = {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}; "
          f"knot fixed in advance at {KNOT}")

    arms = ("L", "Q", "S", "C")
    rows = []
    for i, s in enumerate(seasons):
        if i == 0:
            continue
        tr, te = f[f.season.isin(seasons[:i])], f[f.season == s]
        ytr, yte = tr.y.to_numpy(float), te.y.to_numpy(float)
        r = {"season": s, "n": len(te)}
        for a in arms:
            Xtr = design(a, tr.edge.to_numpy(float),
                         tr.rest_diff.to_numpy(float),
                         tr.open_margin.abs().to_numpy(float))
            Xte = design(a, te.edge.to_numpy(float),
                         te.rest_diff.to_numpy(float),
                         te.open_margin.abs().to_numpy(float))
            resid = (tr.margin_actual - tr.open_margin).to_numpy(float)
            b = np.linalg.solve(Xtr.T @ Xtr + LAM * np.eye(Xtr.shape[1]),
                                Xtr.T @ resid)
            m_tr = tr.open_margin.to_numpy(float) + Xtr @ b
            m_te = te.open_margin.to_numpy(float) + Xte @ b
            sc = fit_scale(m_tr, ytr)           # link refit per arm per fold
            r[f"ll_{a}"] = float(nll(1 / (1 + np.exp(-m_te / sc)), yte).mean())
            r[f"b1_{a}"] = float(b[0])
            if a != "L":
                r[f"b2_{a}"] = float(b[1])
        rows.append(r)
    d = pd.DataFrame(rows)

    print("\n=== per-season log loss by arm ===")
    print(d[["season", "n"] + [f"ll_{a}" for a in arms]].to_string(
        index=False, float_format=lambda v: f"{v:9.5f}"))

    print("\n=== fitted coefficients (mean over folds) ===")
    print(f"  L  edge {d.b1_L.mean():+.4f}")
    print(f"  Q  edge {d.b1_Q.mean():+.4f}   edge*|edge| {d.b2_Q.mean():+.5f}")
    print(f"  S  edge {d.b1_S.mean():+.4f}   beyond {KNOT}pt "
          f"{d.b2_S.mean():+.4f}   (outer slope "
          f"{d.b1_S.mean()+d.b2_S.mean():+.4f})")
    print(f"  C  inner {d.b1_C.mean():+.4f}  outer {d.b2_C.mean():+.4f}")
    print("\n  CONVEX means the outer slope EXCEEDS the inner one — we should")
    print("  spend MORE of a large disagreement, not the same fraction.")

    print("\n=== PRIMARY: change in log loss vs the LINEAR incumbent ===")
    out = {}
    for a in ("Q", "S", "C"):
        v = (d[f"ll_{a}"] - d.ll_L).to_numpy()
        m, lo, hi, t, k = clus(v)
        md = mde80(v)
        flag = ("SHIP" if hi < 0 else
                "WORSE" if lo > 0 else
                f"ns (MDE80 {md:.5f} = {md/abs(m) if m else float('inf'):.1f}x "
                f"the estimate)")
        print(f"  {a} vs L  {m:+.6f}  CI [{lo:+.6f}, {hi:+.6f}]  "
              f"t {t:+5.2f}  better {int((v<0).sum())}/{k}  {flag}")
        out[a] = dict(mean=float(m), ci=[float(lo), float(hi)],
                      better=int((v < 0).sum()), k=int(k), mde80=float(md))

    print("\n  Reference: the whole offset layer is worth -0.00548 vs the blind")
    print("  model (D252), and our edge over the opener is -0.00173. A convex")
    print("  term is a refinement of a small effect and should be read on that")
    print("  scale, not against zero.")
    json.dump({"rows": rows, "tests": out},
              open(ROOT / "data" / "d254_convex.json", "w"), default=float)
    print("\nwrote data/d254_convex.json")


if __name__ == "__main__":
    main()
