#!/usr/bin/env python3
"""F4 EDGE-SHRINKAGE — calibration of OUR OWN edge estimate (D112 ship).

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
D112 measured the Kelly slope on 4 seasons of same-side games:

    realised_excess = -0.0140 + 0.184 x claimed_excess      (se 0.097, n=4367)
    realised_excess = hit - p_mkt_side ;  claimed_excess = p_us_side - p_mkt_side

The slope is 0.184 (replication frame 0.106), not 1.0.  **82-89% of the edge
we claim is illusory.**  Kelly sizing consumes the CLAIMED edge linearly, so
our stakes have been 5-9x oversized — which is exactly the unexplained D75
signature ("quarter-Kelly NEGATIVE on flat-positive bets").  A Kelly bettor
whose stated edge is 5x too large is not a bettor with a small edge, he is a
bettor who is over-betting into an edge that is not there.

This module replaces the raw claimed edge in SIZING with

    shrunk_edge = max(0, a + b * claimed_edge)

and computes the Kelly fraction from `p_mkt_side + shrunk_edge` instead of
`p_us_side`.  This is a CALIBRATION of our own estimator against realised
outcomes — a one-dimensional linear correction with (a, b) read off a
regression, refit annually on ALL completed seasons and never chosen to
maximise PnL.  It is NOT curve-fitting: nothing about the rule set, the
threshold, the window or the cap is touched, no search is run, and the
correction can only ever SHRINK stakes towards zero (b < 1 and a < 0 on every
frame measured).  The honest test of a calibration is whether the corrected
estimate is closer to realised than the raw one; that is what `--refit`
reports, not ROI.

    max(0, .) is a floor, not a filter: it says "when the calibrated edge is
    negative, stake nothing", which is the only coherent Kelly answer.

COLD START.  The walk-forward estimator needs at least one COMPLETED season.
Before then `SHRUNK_KELLY` is undefined and the engine falls back to
`stake = 0` for that arm (never to raw Kelly).  For 2026-27 we have four
completed seasons, so the live engine ships with the registered 4-season fit
below and refits every October.

USAGE
    python scripts/f4_shrinkage.py --refit          # fit on completed seasons
    python scripts/f4_shrinkage.py --show           # print the live coeffs

    from f4_shrinkage import load_coeffs, shrink_edge, stake_units
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
COEFF_PATH = ROOT / "data" / "f4_shrinkage.json"

# ---- pre-registered constants (frozen 2026-08-01, BEFORE any 2026-27 data) --
CONF_EXCESS_CAP = 0.08   # D112: skip when conf_us - conf_mkt > CAP.
#                          TENTATIVE BY SWEEP (8 caps x 6 rules x 2 frames, no
#                          selection protection).  The DIRECTION is what the
#                          rule-free Kelly slope + reliability curve support.
KELLY_FRAC = 0.25        # quarter-Kelly (bet_sim3 convention, unchanged)
BANKROLL_REF = 100.0     # fixed reference bankroll, non-compounding
KELLY_CAP = 10.0         # hard per-bet cap in units of that bankroll

# Registered fallback = the D112 primary-frame fit (rt1 p_full, 4 seasons,
# 2022-23..2025-26, n=4367 same-side games).  data/f4_shrinkage.json, written
# by --refit, supersedes it once it exists.
REGISTERED = {
    "a": -0.01396309947824558,
    "b": 0.18418278015358797,
    "se_b": 0.09715256939851488,
    "n": 4367,
    "seasons": ["2022-23", "2023-24", "2024-25", "2025-26"],
    "frame": "data/ds_rt1_pergame.csv p_full",
    "fit_date": "2026-08-01",
    "source": "D112 kelly-slope, registered fallback",
}


# ---- estimator --------------------------------------------------------------

def fit_kelly_slope(claimed_edge, realised_excess) -> dict:
    """OLS realised_excess ~ a + b*claimed_edge.  Returns the D112 quantities.

    claimed_edge     = p_us_side - p_mkt_side   (same-side games only)
    realised_excess  = hit - p_mkt_side
    """
    x = np.asarray(claimed_edge, float)
    y = np.asarray(realised_excess, float)
    n = len(x)
    if n < 50:
        raise ValueError(f"refusing to fit a shrinkage on n={n} (<50)")
    b, a = np.polyfit(x, y, 1)
    r = y - (a + b * x)
    sxx = ((x - x.mean()) ** 2).sum()
    se_b = float(np.sqrt((r ** 2).sum() / (n - 2) / sxx))
    return {"a": float(a), "b": float(b), "se_b": se_b, "n": int(n),
            "t": float(b / se_b) if se_b else float("nan")}


def shrink_edge(edge: float, a: float, b: float) -> float:
    """The calibrated edge.  Floored at 0 — a negative calibrated edge means
    'stake nothing', which is what Kelly says when f* <= 0 anyway."""
    return max(0.0, a + b * float(edge))


def kelly_fraction(p_side: float, dec: float) -> float:
    """Full-Kelly fraction of bankroll on decimal odds `dec`.  <=0 -> 0."""
    if dec is None or dec <= 1.0:
        return 0.0
    f = (p_side * dec - 1.0) / (dec - 1.0)
    return max(0.0, float(f))


def load_coeffs(path: os.PathLike | None = None) -> dict:
    """Live (a, b).  data/f4_shrinkage.json if present, else the D112 fit."""
    p = Path(path) if path is not None else COEFF_PATH
    if p.exists():
        try:
            d = json.loads(p.read_text())
            if "a" in d and "b" in d:
                return d
        except (json.JSONDecodeError, OSError):
            pass
    return dict(REGISTERED)


# ---- the three pre-registered sizing arms -----------------------------------

SIZING_ARMS = ("flat", "raw_kelly", "shrunk_kelly")


def stake_units(arm: str, p_us_side: float, p_mkt_side: float, dec: float,
                coeffs: dict | None = None) -> float:
    """Stake for one bet under one arm, in units of a 100u reference bankroll
    (bet_sim3 convention, so the numbers line up with D75/D78/D112 tables).

      flat          1.0u                                    — the honest control
      raw_kelly     quarter-Kelly on p_us_side              — what D75 ran
      shrunk_kelly  quarter-Kelly on p_mkt_side + shrunk_edge  — D112 ship

    All three are logged for every candidate; October settles sizing on data.
    """
    if arm == "flat":
        return 1.0
    if arm == "raw_kelly":
        p = float(p_us_side)
    elif arm == "shrunk_kelly":
        c = coeffs or load_coeffs()
        p = float(p_mkt_side) + shrink_edge(
            float(p_us_side) - float(p_mkt_side), c["a"], c["b"])
    else:
        raise ValueError(f"unknown sizing arm {arm!r}")
    f = kelly_fraction(p, dec)
    if f <= 0:
        return 0.0
    return float(min(KELLY_FRAC * f * BANKROLL_REF, KELLY_CAP))


# ---- annual refit -----------------------------------------------------------

def _completed_frame():
    """Same-side games from every COMPLETED season, from the production sim
    frame (ds_rt1_pergame.csv, p_full).  Kept deliberately dumb: one file, one
    probability column, no rule logic — the fit must not see bet selection."""
    import pandas as pd
    csv = ROOT / "data" / "ds_rt1_pergame.csv"
    df = pd.read_csv(csv, dtype={"game_id": str})
    df["p_us"] = df["p_full"]
    pick_home = df.p_us > 0.5
    same = (df.p_us - 0.5) * (df.p_mkt - 0.5) > 0
    p_us_side = np.where(pick_home, df.p_us, 1 - df.p_us)
    p_mkt_side = np.where(pick_home, df.p_mkt, 1 - df.p_mkt)
    hit = np.where(pick_home, df.y == 1, df.y == 0).astype(float)
    out = df.loc[same, ["season", "game_id"]].copy()
    out["claimed"] = (p_us_side - p_mkt_side)[same.values]
    out["realised"] = (hit - p_mkt_side)[same.values]
    return out


def refit(write: bool = True) -> dict:
    df = _completed_frame()
    fit = fit_kelly_slope(df.claimed, df.realised)
    seasons = sorted(df.season.unique())
    # calibration check — is the corrected estimate closer to realised than raw?
    sh = np.maximum(0.0, fit["a"] + fit["b"] * df.claimed.values)
    mae_raw = float(np.abs(df.realised.values - df.claimed.values).mean())
    mae_shr = float(np.abs(df.realised.values - sh).mean())
    out = {**fit, "seasons": seasons,
           "frame": "data/ds_rt1_pergame.csv p_full",
           "fit_date": dt.date.today().isoformat(),
           "mae_raw": mae_raw, "mae_shrunk": mae_shr,
           "source": "f4_shrinkage.py --refit"}
    print(f"seasons     : {', '.join(seasons)}")
    print(f"n same-side : {fit['n']}")
    print(f"KELLY SLOPE : realised = {fit['a']:+.4f} {fit['b']:+.4f} x claimed"
          f"   (se {fit['se_b']:.4f}, t={fit['t']:+.2f})")
    print(f"shrinkage   : {100*(1-fit['b']):.0f}% of the claimed edge is "
          f"discarded; break-even claimed edge = "
          f"{(-fit['a']/fit['b']) if fit['b'] else float('nan'):.4f}")
    print(f"calibration : |realised - claimed| {mae_raw:.5f} -> "
          f"|realised - shrunk| {mae_shr:.5f} "
          f"({'BETTER' if mae_shr < mae_raw else 'WORSE'})")
    if write:
        COEFF_PATH.write_text(json.dumps(out, indent=2))
        print(f"wrote {COEFF_PATH}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true",
                    help="refit (a,b) on all completed seasons and write "
                         "data/f4_shrinkage.json")
    ap.add_argument("--show", action="store_true", help="print live coeffs")
    ap.add_argument("--dry-run", action="store_true", help="refit, do not write")
    args = ap.parse_args()
    if args.refit or args.dry_run:
        refit(write=not args.dry_run)
    if args.show or not (args.refit or args.dry_run):
        c = load_coeffs()
        print(json.dumps(c, indent=2))
        print(f"\nCONF_EXCESS_CAP = {CONF_EXCESS_CAP}  (pre-registered, "
              f"tentative by sweep)")
        for e in (0.02, 0.04, 0.06, 0.08):
            print(f"  claimed edge {e:+.3f} -> shrunk "
                  f"{shrink_edge(e, c['a'], c['b']):+.4f}")


if __name__ == "__main__":
    main()
