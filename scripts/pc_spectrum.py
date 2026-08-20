#!/usr/bin/env python3
"""PCA STEP 1 — THE CORRELATION STRUCTURE OF THE REJECTED PILE.

No log loss is computed here.  This is a property of the DESIGN MATRIX only:
how many genuinely independent axes do the 15 class-(i) columns of D154 span,
and what are those axes in plain language?

Blocks measured
  PILE15    the 15 class-(i) carryable columns (D154 §2, `ca_bank.TERMS` order)
  PILE13    the same, minus dead_h/dead_a (which are ALREADY columns of the
            shipped design and are therefore never re-added as extras —
            `ca_bank.Layer.fit` drops indices 0/1 from `cols`)
  TEAMHOME  the 30 centred home-team dummies (D70/D20 block)
  JOINT43   PILE13 + TEAMHOME

Method: correlation-matrix PCA (each column z-scored on the fit window), which
is the defensible default for a design whose columns carry incompatible units
(1000 km, timezones, rest days, points of form, 0/1 indicators).  The rotation
is computed on the SAME rows `Layer.fit` uses at that date: `fit_ok`, inside
the trailing 730-day window, strictly BEFORE the refit date.

Read-only on data/nba.duckdb (in fact it touches only the cached bank).
Nothing under nbapred/ is imported except threads/production constants.

  python scripts/pc_spectrum.py
"""
from __future__ import annotations

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
import datetime as dt  # noqa: E402

import numpy as np  # noqa: E402

from ca_bank import TERM_NAMES, load_bank  # noqa: E402

OUT = REPO / "data" / "pca_spectrum.json"
VAR_RULES = (0.80, 0.90, 0.95)


def design(bank, rows, block):
    """The standardisable design block at `rows`.  Returns (M, names)."""
    X = bank["X"][rows]
    if block == "PILE15":
        return X, list(TERM_NAMES)
    if block == "PILE13":
        keep = [i for i in range(15) if i not in (0, 1)]
        return X[:, keep], [TERM_NAMES[i] for i in keep]
    hs = bank["home"][rows]
    ts = sorted(set(hs.tolist()))
    k = len(ts)
    D = np.column_stack([(hs == t).astype(float) - 1.0 / k for t in ts])
    dn = [f"th_{t}" for t in ts]
    if block == "TEAMHOME":
        return D, dn
    keep = [i for i in range(15) if i not in (0, 1)]
    return np.column_stack([X[:, keep], D]), [TERM_NAMES[i] for i in keep] + dn


def spectrum(M):
    """Correlation-matrix eigen-decomposition.  Returns (ev desc, V, mu, sd)."""
    mu = M.mean(0)
    sd = M.std(0)
    sdz = np.where(sd > 1e-12, sd, 1.0)
    Z = (M - mu) / sdz
    C = (Z.T @ Z) / len(Z)
    ev, V = np.linalg.eigh(C)
    o = np.argsort(-ev)
    return np.maximum(ev[o], 0.0), V[:, o], mu, sd


def r_for(ev, thr):
    cum = np.cumsum(ev) / ev.sum()
    return int(np.searchsorted(cum, thr - 1e-12) + 1)


def mp_edges(n, p):
    """Marchenko-Pastur support for a pure-noise correlation matrix."""
    q = p / n
    return (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2


def refit_dates(dates_sorted):
    out, last = [], None
    for d in dates_sorted:
        if last is None or (d - last).days >= 7:
            out.append(d)
            last = d
    return out


if __name__ == "__main__":
    bank = load_bank()
    di = bank["date"].astype("int64")
    seas = bank["season"].astype(str)
    dates = bank["date"].astype("datetime64[D]").astype(dt.date)
    CERT_SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")

    out = {"blocks": {}, "stability": {}, "note": "no log loss computed here"}

    # ---- headline: the spectrum at the LAST certified refit date ----------
    end = dt.date(2026, 3, 1)
    b = int(np.datetime64(end, "D").astype("int64"))
    rows = np.where((di < b) & (di >= b - 730) & bank["fit_ok"])[0]
    assert bank["date"][rows].max() < np.datetime64(end, "D"), "LEAK"
    print(f"reference window: rows < {end}, n={len(rows)}\n")

    for blk in ("PILE15", "PILE13", "TEAMHOME", "JOINT43"):
        M, names = design(bank, rows, blk)
        ev, V, mu, sd = spectrum(M)
        lo, hi = mp_edges(len(rows), M.shape[1])
        rec = dict(p=M.shape[1], n=int(len(rows)),
                   ev=[round(float(x), 5) for x in ev],
                   var_frac=[round(float(x / ev.sum()), 5) for x in ev],
                   cum=[round(float(x), 5) for x in np.cumsum(ev) / ev.sum()],
                   r={f"{int(100*t)}": r_for(ev, t) for t in VAR_RULES},
                   mp_edges=[round(lo, 4), round(hi, 4)],
                   n_above_mp=int((ev > hi).sum()),
                   cond_number=float(ev[0] / max(ev[ev > 1e-9].min(), 1e-12)),
                   effective_rank=float(np.exp(-(lambda q: (q * np.log(q + 1e-300)).sum())(ev / ev.sum()))),
                   names=names)
        rec["loadings"] = {
            f"PC{i+1}": sorted(
                [[names[j], round(float(V[j, i]), 3)]
                 for j in range(len(names)) if abs(V[j, i]) >= 0.25],
                key=lambda z: -abs(z[1]))
            for i in range(min(8, M.shape[1]))}
        out["blocks"][blk] = rec
        print(f"== {blk}  p={rec['p']}  n={rec['n']}")
        print(f"   ev[:6]={np.round(ev[:6],3)}  ev[-3:]={np.round(ev[-3:],4)}")
        print(f"   MP null support [{lo:.3f},{hi:.3f}]  "
              f"#ev above upper edge = {rec['n_above_mp']}")
        print(f"   r(80/90/95%) = {rec['r']['80']}/{rec['r']['90']}/{rec['r']['95']}"
              f"   effective rank (exp entropy) = {rec['effective_rank']:.2f}")
        if blk in ("PILE15", "TEAMHOME"):
            for i in range(min(6, M.shape[1])):
                s = ", ".join(f"{a}{v:+.2f}" for a, v in rec["loadings"][f"PC{i+1}"])
                print(f"     PC{i+1}: {s[:110]}")
        print()

    # ---- correlation matrix of PILE15, for the plain-language reading -----
    M, names = design(bank, rows, "PILE15")
    ev, V, mu, sd = spectrum(M)
    Z = (M - mu) / np.where(sd > 1e-12, sd, 1)
    C = (Z.T @ Z) / len(Z)
    out["pile15_corr"] = [[round(float(C[i, j]), 4) for j in range(15)]
                          for i in range(15)]
    pairs = sorted([(abs(C[i, j]), names[i], names[j], round(float(C[i, j]), 3))
                    for i in range(15) for j in range(i + 1, 15)], reverse=True)
    out["pile15_top_pairs"] = [[a, b, c] for _, a, b, c in pairs[:12]]
    print("strongest raw correlations in PILE15:")
    for _, a, bb, c in pairs[:10]:
        print(f"   {a:14s} {bb:14s} {c:+.3f}")

    # ---- stability of the rank rule across every certified refit date ----
    print("\nrank-rule stability across refit dates (PILE13, the rotated block):")
    stab = []
    for s in CERT_SEASONS:
        m = seas == s
        sd_ = sorted(set(dates[m]))
        for d in refit_dates(sd_):
            bb = int(np.datetime64(d, "D").astype("int64"))
            rr = np.where((di < bb) & (di >= bb - 730) & bank["fit_ok"])[0]
            if len(rr) < 300:
                continue
            Mx, _ = design(bank, rr, "PILE13")
            e, _, _, _ = spectrum(Mx)
            stab.append(dict(date=str(d), n=int(len(rr)),
                             r80=r_for(e, .80), r90=r_for(e, .90),
                             r95=r_for(e, .95), ev1=round(float(e[0]), 4)))
    out["stability"]["PILE13"] = stab
    for key in ("r80", "r90", "r95"):
        vals = [x[key] for x in stab]
        print(f"   {key}: min={min(vals)} max={max(vals)} "
              f"mode={max(set(vals), key=vals.count)} over {len(vals)} refits")

    # long-history version too (1996-2026), for the era statement
    stab2 = []
    for yr in range(2000, 2027):
        d = dt.date(yr, 3, 1)
        bb = int(np.datetime64(d, "D").astype("int64"))
        rr = np.where((di < bb) & (di >= bb - 730) & bank["fit_ok"])[0]
        if len(rr) < 300:
            continue
        Mx, _ = design(bank, rr, "PILE13")
        e, _, _, _ = spectrum(Mx)
        stab2.append(dict(date=str(d), n=int(len(rr)), r80=r_for(e, .80),
                          r90=r_for(e, .90), r95=r_for(e, .95)))
    out["stability"]["PILE13_history"] = stab2
    print("   history 2000-2026 r90: "
          f"{sorted(set(x['r90'] for x in stab2))}")

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT.relative_to(REPO)}")
