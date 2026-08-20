#!/usr/bin/env python3
"""D253b — INTERROGATE THE ONE POSITIVE CELL.

D253 ran four cells. Three are flat. One is not: Model B (7 seasons, injury
report included) predicting `y_open` returns OOS R^2 +0.00170 against a null
whose maximum over 40 draws was -0.00072, reported as p = 0.000.

Two reasons not to accept that as written.

(1) FORTY DRAWS CANNOT RESOLVE p = 0.000. The finest p a 40-draw permutation can
    express is 1/40 = 0.025, and it was one of FOUR cells, so the family-adjusted
    floor is ~0.1. This rerun uses 400 draws on that cell alone.

(2) THE TWO FEATURES SELECTED IN 100% OF FOLDS ARE `our_edge` AND
    `our_edge_abs` — our own disagreement with the opener. That is not an
    external condition telling us when to trust ourselves; it is a restatement
    of the offset's own shrinkage rule, and D252b already found the
    largest-disagreement quintile is where we beat the opener most (-0.00577).
    If the whole result is those two columns, then nothing about injuries,
    schedule, season length or absences contributes, and the honest headline is
    "we already knew this and it is already in production".

    So: ablate. Fit (a) the full 26-feature model, (b) our_edge + our_edge_abs
    alone, (c) the full set with those two REMOVED. If (b) ~= (a) and (c) ~= 0,
    the differentiator is our own edge size and nothing else.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import importlib.util                                             # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from sklearn.linear_model import ElasticNet                       # noqa: E402

spec = importlib.util.spec_from_file_location(
    "d253", ROOT / "scripts" / "d253_differentiators.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)

ALPHAS = D.ALPHAS
L1S = D.L1S
EDGE = ["our_edge", "our_edge_abs"]
N_PERM = 400


def walk(d, FEATS, seasons, yy):
    preds = np.full(len(d), np.nan)
    for i, s in enumerate(seasons):
        if i < 3:
            continue
        tr = d.season.isin(seasons[:i]).to_numpy()
        te = (d.season == s).to_numpy()
        Xtr = d.loc[tr, FEATS].to_numpy(float)
        Xte = d.loc[te, FEATS].to_numpy(float)
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
        ytr = yy[tr]
        inner = d.loc[tr, "season"].to_numpy() == seasons[i - 1]
        best, bp = np.inf, (ALPHAS[0], L1S[0])
        for a in ALPHAS:
            for l1 in L1S:
                m = ElasticNet(alpha=a, l1_ratio=l1, max_iter=5000)
                m.fit(Ztr[~inner], ytr[~inner])
                e = ((m.predict(Ztr[inner]) - ytr[inner]) ** 2).mean()
                if e < best:
                    best, bp = e, (a, l1)
        m = ElasticNet(alpha=bp[0], l1_ratio=bp[1], max_iter=5000)
        m.fit(Ztr, ytr)
        preds[te] = m.predict(Zte)
    ok = np.isfinite(preds)
    r = yy[ok] - preds[ok]
    b = yy[ok] - yy[ok].mean()
    return float(1 - (r ** 2).sum() / (b ** 2).sum()), ok


def main():
    f, _ = D.assemble()
    y = (f.margin_actual > 0).astype(float).to_numpy()
    def ll(m):
        return D.nll(1 / (1 + np.exp(-np.asarray(m, float) / D.SCALE)), y)
    f["y_open"] = ll(f.m_us) - ll(f.open_margin)
    FULL = [c for c in D.TIER_A + D.TIER_B if c in f.columns]
    d = f.dropna(subset=FULL + ["y_open"]).copy()
    seasons = sorted(d.season.unique())
    yy = d.y_open.to_numpy(float)
    print(f"{len(d):,} games, {len(seasons)} seasons "
          f"({seasons[0]}..{seasons[-1]}), {len(FULL)} features\n")

    arms = {
        "FULL (26 features)": FULL,
        "our_edge + our_edge_abs ONLY": EDGE,
        "FULL minus the two edge columns": [c for c in FULL if c not in EDGE],
    }
    res = {}
    for lab, feats in arms.items():
        r2, ok = walk(d, feats, seasons, yy)
        res[lab] = r2
        print(f"  {lab:34} OOS R^2 {r2:+.5f}  ({len(feats)} features)")

    print(f"\n  permutation null, {N_PERM} draws, FULL model, "
          f"outcome shuffled within season")
    rng = np.random.default_rng(2531)
    scode = pd.factorize(d.season)[0]
    idx = [np.flatnonzero(scode == i) for i in range(scode.max() + 1)]
    null = np.empty(N_PERM)
    for it in range(N_PERM):
        perm = yy.copy()
        for ix in idx:
            perm[ix] = rng.permutation(yy[ix])
        null[it], _ = walk(d, FULL, seasons, perm)
    obs = res["FULL (26 features)"]
    p = float((null >= obs).mean())
    p_fam = 1 - (1 - p) ** 4          # D253 ran four cells
    print(f"    null OOS R^2: median {np.median(null):+.5f}, "
          f"95th {np.percentile(null,95):+.5f}, max {null.max():+.5f}")
    print(f"    observed {obs:+.5f}   p = {p:.4f}   "
          f"family-adjusted over D253's 4 cells p = {p_fam:.4f}")
    print(f"    {'SURVIVES' if p_fam < 0.05 else 'DOES NOT SURVIVE'} "
          f"the family adjustment")

    print("\n  READ: if the edge-only arm reproduces the full model and the")
    print("  no-edge arm is ~0, the only differentiator is the size of our own")
    print("  disagreement — which the offset layer already prices at 0.3564.")
    json.dump({"arms": res, "p": p, "p_family": p_fam,
               "null_median": float(np.median(null)),
               "null_p95": float(np.percentile(null, 95))},
              open(ROOT / "data" / "d253b_probe.json", "w"), default=float)
    print("\nwrote data/d253b_probe.json")


if __name__ == "__main__":
    main()
