#!/usr/bin/env python3
"""LEAKAGE PROOF for the walk-forward PCA rotation (prereg §5), plus the
EBLayer == Layer identity check.

R2 — THE PERMUTATION PROOF.  For a battery of refit dates, every column of
every bank row dated >= `before` is randomly permuted AND resampled (including
the outcome-derived ones), the rotation is refit, and the result is required to
be BIT-IDENTICAL: mean, sd, eigenvalues, eigenvectors, and the r the variance
rule selects.  A rotation fitted on the full corpus CANNOT pass this — the
control arm below deliberately fits on all rows and is required to FAIL.

EB IDENTITY.  `EBLayer(eb=False)` must reproduce `Layer` exactly, so that
arm C's only difference from D154's harness is the shrinkage weight.

  TANK_SEASON_FLOOR=2020-21 python scripts/pc_leaktest.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import nbapred.threads as _t  # noqa: E402

_t.pin(1)
import datetime as dt  # noqa: E402

import numpy as np  # noqa: E402

from ca_bank import Layer, load_bank  # noqa: E402
from pc_layer import PILE13, EBLayer, Rotation, widen  # noqa: E402

DATES = [dt.date(*x) for x in [
    (2021, 10, 19), (2022, 1, 3), (2022, 12, 25), (2023, 10, 24),
    (2024, 2, 1), (2024, 11, 6), (2025, 4, 10), (2025, 10, 21),
    (2026, 1, 14), (2026, 3, 1)]]


def scramble_future(bank, before, rng):
    """A copy of the bank in which EVERY row dated >= `before` has had every
    column permuted and resampled.  Strictly-prior rows are untouched."""
    b = {k: (v.copy() if isinstance(v, np.ndarray) else v)
         for k, v in bank.items()}
    bb = np.datetime64(before, "D")
    fut = np.where(b["date"] >= bb)[0]
    pri = np.where(b["date"] < bb)[0]
    if len(fut) == 0:
        return b, 0
    for k in ("X", "Z"):
        M = b[k][fut]
        for j in range(M.shape[1]):
            M[:, j] = rng.permutation(M[:, j]) * rng.choice([-1.0, 1.0],
                                                            size=len(M)) \
                + rng.normal(0, 3.0, size=len(M))
        b[k][fut] = M
    for k in ("margin", "y", "hb2b", "ab2b", "qd"):
        b[k][fut] = rng.permutation(b[k][fut])
    b["home"][fut] = rng.permutation(b["home"][fut])
    b["away"][fut] = rng.permutation(b["away"][fut])
    return b, len(pri)


if __name__ == "__main__":
    os.environ.setdefault("TANK_SEASON_FLOOR", "2020-21")
    bank = widen(load_bank())
    rng = np.random.default_rng(20260803)
    out = {"dates": [], "control_full_corpus": [], "eb_identity": []}

    print("R2 PERMUTATION PROOF — rotation must be bit-identical after the "
          "future is scrambled\n")
    print(f"{'date':12s} {'n_fit':>6s} {'n_scrambled':>12s} "
          f"{'max|d ev|':>11s} {'max|d V|':>11s} {'max|d mu|':>11s} "
          f"{'r same':>7s}")
    worst = 0.0
    for d in DATES:
        L = Layer(bank)
        b_int = int(np.datetime64(d, "D").astype("int64"))
        fit_rows = L._rows(d, b_int - L.window)
        r0 = Rotation(0.90).fit(bank, fit_rows, d)
        r0B = Rotation(0.90, teamhome=True).fit(bank, fit_rows, d)

        b2, npri = scramble_future(bank, d, rng)
        nfut = int((bank["date"] >= np.datetime64(d, "D")).sum())
        L2 = Layer(b2)
        fr2 = L2._rows(d, b_int - L2.window)
        assert np.array_equal(fit_rows, fr2), "fit frame moved under scramble"
        r1 = Rotation(0.90).fit(b2, fr2, d)
        r1B = Rotation(0.90, teamhome=True).fit(b2, fr2, d)

        dev = float(np.abs(r0.ev - r1.ev).max())
        dV = float(np.abs(r0.V - r1.V).max())
        dmu = float(max(np.abs(r0.mu - r1.mu).max(), np.abs(r0.sd - r1.sd).max()))
        same = (r0.r_rule == r1.r_rule and r0B.r_rule == r1B.r_rule
                and r0.key() == r1.key() and r0B.key() == r1B.key())
        worst = max(worst, dev, dV, dmu)
        out["dates"].append(dict(date=str(d), n_fit=int(len(fit_rows)),
                                 n_future_scrambled=nfut, d_ev=dev, d_V=dV,
                                 d_mu=dmu, r_rule=r0.r_rule,
                                 r_rule_TH=r0B.r_rule, identical=bool(same)))
        print(f"{str(d):12s} {len(fit_rows):6d} {nfut:12d} {dev:11.3e} "
              f"{dV:11.3e} {dmu:11.3e} {str(same):>7s}")
        assert same, "ROTATION MOVED WHEN THE FUTURE MOVED — LEAKAGE"

        # ---- CONTROL ARM: a full-corpus rotation MUST fail the same test --
        allrows = np.where(bank["fit_ok"])[0]
        c0 = Rotation(0.90)
        c0.mu = None
        M0 = c0.block(bank, allrows)
        M1 = c0.block(b2, allrows)
        mu0, mu1 = M0.mean(0), M1.mean(0)
        C0 = np.corrcoef(M0, rowvar=False)
        C1 = np.corrcoef(M1, rowvar=False)
        e0 = np.sort(np.linalg.eigvalsh(np.nan_to_num(C0)))[::-1]
        e1 = np.sort(np.linalg.eigvalsh(np.nan_to_num(C1)))[::-1]
        out["control_full_corpus"].append(
            dict(date=str(d), d_ev=float(np.abs(e0 - e1).max()),
                 d_mu=float(np.abs(mu0 - mu1).max())))

    print(f"\nWORST deviation over {len(DATES)} dates: {worst:.3e}  "
          f"(required exactly 0.0)")
    assert worst == 0.0, "not bit-identical"
    cm = max(x["d_ev"] for x in out["control_full_corpus"])
    print(f"CONTROL ARM (full-corpus rotation, deliberately leaky): "
          f"max|d ev| = {cm:.4f}  -> the test HAS power "
          f"({'DETECTED' if cm > 1e-6 else 'NO POWER — TEST IS VACUOUS'})")
    assert cm > 1e-6, "the permutation test has no power"

    # ---- EBLayer(eb=False) == Layer -------------------------------------
    print("\nEB IDENTITY — EBLayer(eb=False) must equal Layer exactly\n")
    L = Layer(bank)
    E = EBLayer(bank, eb=False)
    wmax = 0.0
    for d in DATES:
        for cols, th in ((), False), (PILE13, False), (PILE13, True):
            a = L.fit(d, cols=cols, teamhome=th)
            b = E.fit(d, cols=cols, teamhome=th)
            m = max(max(abs(x - y) for x, y in zip(a[0], b[0])),
                    max([abs(a[1][k] - b[1][k]) for k in a[1]] or [0.0]),
                    max([abs(a[2][k] - b[2][k]) for k in a[2]] or [0.0]))
            wmax = max(wmax, m)
    print(f"WORST |dbeta| over {len(DATES)} dates x 3 designs: {wmax:.3e}")
    out["eb_identity"] = dict(worst=wmax)
    assert wmax == 0.0, "EBLayer(eb=False) is not bit-identical to Layer"
    print("EB IDENTITY OK")

    # ---- EBLayer(eb=True) really does change the weights -----------------
    E2 = EBLayer(bank, eb=True)
    a = L.fit(dt.date(2026, 3, 1), cols=PILE13)
    b = E2.fit(dt.date(2026, 3, 1), cols=PILE13)
    dg = E2.eb_diag[-1]
    out["eb_bite"] = dict(w_global=dg["w_global"], w_eb=dg["w_eb"],
                          t=dg["t_extras"], w_eb_dead=dg["w_eb_dead"],
                          t_dead=dg["t_dead"],
                          max_coef_shrink=float(max(
                              abs(a[1][k] - b[1][k]) for k in a[1])))
    print(f"\nEB BITES: global w = {dg['w_global']}, per-term w_eb = "
          f"{sorted(dg['w_eb'].values())}")

    json.dump(out, open(REPO / "data" / "pca_leaktest.json", "w"), indent=1)
    print("\nwrote data/pca_leaktest.json — ALL LEAKAGE ASSERTIONS PASS")
