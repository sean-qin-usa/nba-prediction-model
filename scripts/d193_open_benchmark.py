#!/usr/bin/env python3
"""D193 — THE IDENTICAL-FRAME MODEL / OPEN / CLOSE BENCHMARK.

Priority 1 of the outside roadmap: the repository has been reporting model log
loss (0.6111) and CLOSING-market log loss (0.5982) on the 2019-26 frame, but
OPENING-market log loss only on a different 1,892-game moneyline frame (0.62615).
Those cannot be combined, and the comparison that actually matters — does the
model add information over the price you can actually transact at — was never
computed on identical games.

It is computable: the frame carries open_margin AND close_margin on 8,237 of
8,239 games.  This script computes all three on exactly the same games.

HEADLINE STATISTIC (the roadmap's):
    capture = (LL_open - LL_model) / (LL_open - LL_close)
  > 0  the model improves on information available AT THE OPEN
  < 0  the model does not beat the opener as a general forecaster, whatever CLV
       a selected subset may earn

TWO SPACES, because the probability conversion is a modelling choice and the
margin comparison is not:
  (A) MARGIN SPACE — RMSE/MAE against the actual margin. No devig, no link, no
      conversion of any kind. This is the assumption-free comparison.
  (B) PROBABILITY SPACE — log loss. Each source gets its OWN logistic scale,
      fitted walk-forward on prior seasons only, so no source is handicapped by
      another's calibration.

Read-only.  Nothing ships.
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
from scipy.optimize import minimize_scalar                        # noqa: E402

import oc_capacity as oc                                          # noqa: E402

MODERN = "2019-20"
SOURCES = [("model", "m_us"), ("open", "open_margin"), ("close", "close_margin")]


def nll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_scale(m, y):
    r = minimize_scalar(lambda s: nll(1 / (1 + np.exp(-m / s)), y),
                        bounds=(2.0, 25.0), method="bounded")
    return float(r.x)


def main():
    df, _ = oc.load()
    d = df[df["season"] >= MODERN].copy()
    ok = d["m_us"].notna() & d["open_margin"].notna() & \
        d["close_margin"].notna() & d["margin_actual"].notna()
    d = d[ok].reset_index(drop=True)
    seas = sorted(d["season"].unique())
    a = d["margin_actual"].to_numpy(float)
    y = (a > 0).astype(float)
    print(f"IDENTICAL FRAME: {seas[0]}..{seas[-1]}  K={len(seas)}  n={len(d)}")
    print("(games where model, opening AND closing prices all exist)\n")

    # ---------------------------------------------------------- (A) MARGIN
    print("=" * 74)
    print("(A) MARGIN SPACE — no probability conversion anywhere")
    print("=" * 74)
    print(f"  {'source':8} {'RMSE':>9} {'MAE':>9} {'bias':>8}   vs actual margin")
    marg = {}
    for nm, col in SOURCES:
        m = d[col].to_numpy(float)
        r = a - m
        marg[nm] = dict(rmse=float(np.sqrt((r ** 2).mean())),
                        mae=float(np.abs(r).mean()), bias=float(r.mean()))
        print(f"  {nm:8} {marg[nm]['rmse']:9.4f} {marg[nm]['mae']:9.4f} "
              f"{marg[nm]['bias']:+8.4f}")
    cap_m = ((marg["open"]["rmse"] - marg["model"]["rmse"]) /
             (marg["open"]["rmse"] - marg["close"]["rmse"]))
    print(f"\n  open->close RMSE improvement : "
          f"{marg['open']['rmse']-marg['close']['rmse']:+.4f} pts")
    print(f"  open->model RMSE improvement : "
          f"{marg['open']['rmse']-marg['model']['rmse']:+.4f} pts")
    print(f"  CAPTURE FRACTION (margin)    : {cap_m:+.3f}")

    # ---------------------------------------------------- (B) PROBABILITY
    print("\n" + "=" * 74)
    print("(B) PROBABILITY SPACE — each source keeps its OWN walk-forward scale")
    print("=" * 74)
    per = {nm: [] for nm, _ in SOURCES}
    scales = {nm: [] for nm, _ in SOURCES}
    ns = []
    for i in range(2, len(seas)):
        tr = d[d["season"].isin(seas[:i])]
        te = d[d["season"] == seas[i]]
        yt = (te["margin_actual"].to_numpy(float) > 0).astype(float)
        ytr = (tr["margin_actual"].to_numpy(float) > 0).astype(float)
        ns.append(len(te))
        for nm, col in SOURCES:
            s = fit_scale(tr[col].to_numpy(float), ytr)
            scales[nm].append(s)
            per[nm].append(nll(1 / (1 + np.exp(-te[col].to_numpy(float) / s)),
                               yt))
    w = np.array(ns, float)
    print(f"  {'season':10} " + "".join(f"{nm:>12}" for nm, _ in SOURCES))
    for j, s in enumerate(seas[2:]):
        print(f"  {s:10} " + "".join(f"{per[nm][j]:12.5f}" for nm, _ in SOURCES))
    pooled = {nm: float(np.average(per[nm], weights=w)) for nm, _ in SOURCES}
    print(f"  {'POOLED':10} " + "".join(f"{pooled[nm]:12.5f}" for nm, _ in SOURCES))
    print(f"  {'scale':10} " + "".join(f"{np.mean(scales[nm]):12.3f}"
                                        for nm, _ in SOURCES))

    cap = (pooled["open"] - pooled["model"]) / (pooled["open"] - pooled["close"])
    print(f"\n  LL(open) - LL(close) = {pooled['open']-pooled['close']:+.5f}  "
          f"(the whole open->close information)")
    print(f"  LL(open) - LL(model) = {pooled['open']-pooled['model']:+.5f}  "
          f"(what the model recovers)")
    print(f"\n  *** CAPTURE FRACTION = {cap:+.3f} ***")
    if cap > 0:
        print("  -> the model DOES improve on information available at the open")
    else:
        print("  -> the model does NOT beat the opener as a general forecaster")

    # per-season capture, season-clustered
    caps = [(per["open"][j] - per["model"][j]) /
            (per["open"][j] - per["close"][j])
            for j in range(len(per["open"]))
            if abs(per["open"][j] - per["close"][j]) > 1e-9]
    caps = np.array(caps)
    K = len(caps)
    se = caps.std(ddof=1) / np.sqrt(K)
    tc = oc.t_crit(K - 1)
    print(f"\n  per-season capture: {np.round(caps, 3).tolist()}")
    print(f"  mean {caps.mean():+.3f}  95% CI "
          f"[{caps.mean()-tc*se:+.3f}, {caps.mean()+tc*se:+.3f}]  (K={K})")
    print(f"  seasons with positive capture: {(caps > 0).sum()}/{K}")

    json.dump(dict(n=len(d), seasons=seas, margin=marg, capture_margin=cap_m,
                   pooled_ll=pooled, capture_ll=float(cap),
                   per_season_capture=caps.tolist()),
              open(ROOT / "data" / "d193_open_benchmark.json", "w"), indent=1)
    print("\nwrote data/d193_open_benchmark.json")


if __name__ == "__main__":
    main()
