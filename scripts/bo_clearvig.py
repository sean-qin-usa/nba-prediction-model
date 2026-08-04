#!/usr/bin/env python3
"""BO-CLEARVIG — does the CALIBRATED bettor stake anything at the OPEN, and
does he win?  The direct successor to D117.

D117 asked exactly this question at the CLOSE and the answer was NO: with the
Kelly-slope calibration applied, the break-even claimed edge (-a/b = 0.0758)
sat BELOW the registered 0.08 confidence-excess cap, so shrunk-Kelly staked
ZERO on every rule, every window, both frames.  "After calibrating our own
edge estimates, WE HAVE NO POSITIVE-EV BETS AGAINST THE CLOSE."

bo_openbacktest.py [5] shows the calibration is a DIFFERENT OBJECT at the
open: the slope roughly TREBLES (b = 0.57 vs 0.21 on the spread frame; 0.52 vs
0.19 on the real-moneyline frame).  So D117's verdict has to be re-run, not
assumed.

THE PRE-SPECIFIED DECISION RULE (written before the numbers were read; it is
D116's shipped live rule with the price swapped from close to open):
  1. WALK-FORWARD calibration.  For season s, fit realised_excess = a + b*edge
     on COMPLETED PRIOR SEASONS ONLY, same-side bets, at the OPEN price.  The
     first season has no prior -> stakes nothing (cold start), exactly as
     scripts/f4_resim.walkforward_coeffs does.
  2. shrunk_edge = max(0, a + b*edge)                     [f4_shrinkage]
  3. REGISTERED CAP: skip when conf_excess > 0.08         [D112/D116 ship]
  4. STAKE iff the ACTUAL transactable decimal beats the calibrated fair
     price:  dec > 1 / (p_side + shrunk_edge).
     On the ML frame this involves NO vig assumption whatsoever — `dec` is the
     real opening moneyline.
  5. Score flat 1u.  PRIMARY = pooled.  Both IS/OOS directions reported.
     Noise math: P(ROI >= observed | every bet is EXACTLY breakeven).

This is ONE derived test with a rule fixed in advance, not a sweep.  The
sensitivity sweep at the bottom is labelled and never selected from.

RULES HONORED: DuckDB read_only=True; new file scripts/bo_*.py; nbapred/
untouched.

Run:  python scripts/bo_clearvig.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402

from bo_openbacktest import (CF, DEV, NONDEV, N_BOOT, N_NULL,    # noqa: E402
                             RT1, SEED, build, price_cols,
                             registry_masks)
from f4_shrinkage import (CONF_EXCESS_CAP, fit_kelly_slope,      # noqa: E402
                          shrink_edge)

OUT = os.path.join(ROOT, "data", "bo_clearvig.json")


def walkforward(m: pd.DataFrame, p_side: np.ndarray, ok: np.ndarray) -> dict:
    """(a,b) per season from COMPLETED PRIOR seasons only, at this price."""
    edge = m.p_us_side.values - p_side
    same = ((m.p_us.values - 0.5) *
            (np.where(m.pick_home, p_side, 1 - p_side) - 0.5)) > 0
    seasons = sorted(m.season.unique())
    out = {}
    for i, s in enumerate(seasons):
        prior = ok & same & m.season.isin(seasons[:i]).values
        if i == 0 or prior.sum() < 50:
            out[s] = None
            continue
        out[s] = fit_kelly_slope(pd.Series(edge[prior]),
                                 pd.Series(m.hit.values[prior] - p_side[prior]))
    return out


def score(m, sel, dec, seed=SEED) -> dict:
    if sel.sum() == 0:
        return dict(n=0)
    hit = m.hit.values[sel].astype(bool)
    d = dec[sel]
    pnl = np.where(hit, d - 1.0, -1.0)
    be = 1.0 / d
    roi = float(pnl.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pnl), (N_BOOT, len(pnl)))
    lo, hi = np.percentile(pnl[idx].mean(axis=1), [2.5, 97.5])
    r2 = np.random.default_rng(seed + 1)
    win = r2.random((N_NULL, len(be))) < be
    nroi = np.where(win, d - 1.0, -1.0).mean(axis=1)
    return dict(n=int(sel.sum()), hit=float(hit.mean()),
                breakeven=float(be.mean()),
                edge_pp=float(hit.mean() - be.mean()), roi=roi,
                roi_lo=float(lo), roi_hi=float(hi), pnl=float(pnl.sum()),
                null_be=float((nroi >= roi).mean()),
                mean_dec=float(d.mean()))


def run(m: pd.DataFrame, src: str, label: str, res: dict) -> None:
    po, deco, oko = price_cols(m, "open", src)
    pc, decc, okc = price_cols(m, "close", src)
    ok = oko & okc
    print(f"\n{'#' * 112}\nFRAME {label}   price source {src}   "
          f"n priced = {int(ok.sum())}\n{'#' * 112}")

    for when, p_side, dec in (("OPEN", po, deco), ("CLOSE", pc, decc)):
        wf = walkforward(m, p_side, ok)
        edge = m.p_us_side.values - p_side
        same = ((m.p_us.values - 0.5) *
                (np.where(m.pick_home, p_side, 1 - p_side) - 0.5)) > 0
        conf_excess = edge                       # == conf_us - conf_mkt on same-side
        print(f"\n  ---- price = {when} "
              f"------------------------------------------------")
        shrunk = np.zeros(len(m))
        cold = np.zeros(len(m), bool)
        for s, c in wf.items():
            sel = (m.season.values == s)
            if c is None:
                cold |= sel
                print(f"    walk-forward {s}: COLD START (no completed prior "
                      f"season) -> stakes 0")
                continue
            shrunk[sel] = [shrink_edge(e, c["a"], c["b"]) for e in edge[sel]]
            be_claim = (-c["a"] / c["b"]) if c["b"] > 0 else float("nan")
            print(f"    walk-forward {s}: a={c['a']:+.4f} b={c['b']:+.4f} "
                  f"(se {c['se_b']:.4f}, t={c['t']:+.2f}, n={c['n']}) -> "
                  f"break-even claimed edge {be_claim:+.4f}")
        # THE PRE-SPECIFIED STAKE CONDITION
        fair_dec = np.where(p_side + shrunk > 0, 1.0 / (p_side + shrunk),
                            np.inf)
        stake = (ok & same & ~cold & (conf_excess <= CONF_EXCESS_CAP)
                 & (dec > fair_dec) & (shrunk > 0))
        elig = ok & same & ~cold & (conf_excess <= CONF_EXCESS_CAP)
        print(f"    eligible (priced, same-side, warm, under the 0.08 cap): "
              f"{int(elig.sum())}")
        print(f"    STAKED (actual decimal beats the calibrated fair price): "
              f"{int(stake.sum())}"
              f"  = {100 * stake.sum() / max(1, elig.sum()):.1f}% of eligible")
        if stake.sum():
            band = conf_excess[stake]
            print(f"    staked claimed-edge band: "
                  f"[{band.min():+.4f}, {band.max():+.4f}]  "
                  f"median {np.median(band):+.4f}")
        hdr = (f"      {'window':<10}{'n':>5}{'hit%':>7}{'be%':>8}"
               f"{'edge_pp':>9}{'ROI%':>8}{'      [95% CI]':>18}"
               f"{'null_be':>9}")
        print(hdr)
        rows = {}
        wins = [("POOL", None), ("DEV", DEV), ("NONDEV", NONDEV)] + \
               [(s, {s}) for s in sorted(m.season.unique())]
        for wn, ws in wins:
            sel = stake & (m.season.isin(ws).values if ws else True)
            r = score(m, sel, dec)
            if not r.get("n"):
                print(f"      {wn:<10}{0:>5}   (no bets)")
                rows[wn] = r
                continue
            print(f"      {wn:<10}{r['n']:>5}{100 * r['hit']:>7.1f}"
                  f"{100 * r['breakeven']:>8.2f}{100 * r['edge_pp']:>+9.2f}"
                  f"{100 * r['roi']:>8.2f}"
                  f"[{100 * r['roi_lo']:>+7.2f},{100 * r['roi_hi']:>+7.2f}]"
                  f"{r['null_be']:>9.3f}")
            rows[wn] = r
        res.setdefault("calibrated", {})[f"{label}|{src}|{when}"] = {
            "n_eligible": int(elig.sum()), "n_staked": int(stake.sum()),
            "windows": rows,
            "wf": {k: (v if v is None else {kk: float(vv) for kk, vv
                                            in v.items()})
                   for k, v in wf.items()}}

    # ---- SENSITIVITY (labelled diagnostic; nothing is selected from it) ----
    wf = walkforward(m, po, ok)
    edge = m.p_us_side.values - po
    same = ((m.p_us.values - 0.5) *
            (np.where(m.pick_home, po, 1 - po) - 0.5)) > 0
    shrunk = np.zeros(len(m))
    cold = np.zeros(len(m), bool)
    for s, c in wf.items():
        sel = (m.season.values == s)
        if c is None:
            cold |= sel
        else:
            shrunk[sel] = [shrink_edge(e, c["a"], c["b"]) for e in edge[sel]]
    print(f"\n  SENSITIVITY (diagnostic only — the shipped cap stays 0.08): "
          f"staked-bet ROI at the OPEN as the cap moves")
    print(f"      {'cap':>6}{'n':>6}{'hit%':>7}{'be%':>8}{'edge_pp':>9}"
          f"{'ROI%':>8}{'null_be':>9}")
    sens = []
    for cap in (0.04, 0.06, 0.08, 0.10, 0.15, 1.00):
        sel = (ok & same & ~cold & (edge <= cap)
               & (deco > 1.0 / (po + shrunk)) & (shrunk > 0))
        r = score(m, sel, deco)
        if r.get("n"):
            print(f"      {cap:>6.2f}{r['n']:>6}{100 * r['hit']:>7.1f}"
                  f"{100 * r['breakeven']:>8.2f}{100 * r['edge_pp']:>+9.2f}"
                  f"{100 * r['roi']:>8.2f}{r['null_be']:>9.3f}")
            sens.append({"cap": cap, **r})
    res.setdefault("sensitivity", {})[f"{label}|{src}"] = sens


def main() -> None:
    res: dict = {"cap": CONF_EXCESS_CAP, "seed": SEED}
    print("=" * 112)
    print("BO-CLEARVIG — does the CALIBRATED bettor stake at the OPEN, and "
          "does he win?  (successor to D117)")
    print("=" * 112)
    print("STAKE RULE (fixed in advance): warm walk-forward (a,b) at the OPEN "
          "-> shrunk_edge = max(0, a+b*edge)")
    print("                               -> bet iff conf_excess <= 0.08 AND "
          "the ACTUAL decimal > 1/(p_side + shrunk_edge)")
    for label, csv, pcol in (("PRIMARY rt1 p_full 4-season", RT1, "p_full"),
                             ("WIDE cf_holdout p_base 5-season", CF, "p_base")):
        sink: dict = {}
        m = build(csv, pcol, label, sink)
        for src in ("SP", "ML"):
            if src == "ML" and m.p_open_ml.notna().sum() < 200:
                continue
            run(m, src, label, res)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
