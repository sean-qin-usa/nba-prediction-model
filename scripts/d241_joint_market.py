#!/usr/bin/env python3
"""D241 — JOINT-MARKET DISTRIBUTION. Prereg sha256 6c4d7890...

Production converts margin to probability with a FIXED scale 7.2. The offset
layer reads the spread but never the TOTAL. A 240-total game has more margin
variance than a 205-total game, so the same corrected margin should imply a
probability nearer 0.5 in the high-total game.

D198 tested conditional variance from availability / early-season / rest /
favourite-size and found a precise null whose gain was entirely a global
rescale. It never used the market total — the one variance proxy priced by
someone with more information than we have.

Tested at the FULL STACK (m_offset), because D235 established that gating an
intermediate layer overstates the shipped effect. The total enters the SCALE
only; T4 checks it does not reprice sides.
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
from scipy.optimize import minimize                               # noqa: E402

FROM = "2019-20"


def nll_p(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_const(m, y):
    def f(th):
        return nll_p(1 / (1 + np.exp(-m / np.exp(th[0]))), y).mean()
    r = minimize(f, [np.log(7.2)], method="Nelder-Mead",
                 options=dict(xatol=1e-6, fatol=1e-10, maxiter=500))
    return np.exp(r.x[0])


def fit_linear(m, y, t):
    """s_i = exp(a) + b*(t_i - tbar), floored positive."""
    tb = t.mean()

    def f(th):
        s = np.exp(th[0]) + th[1] * (t - tb)
        s = np.clip(s, 2.0, 30.0)
        return nll_p(1 / (1 + np.exp(-m / s)), y).mean()
    r = minimize(f, [np.log(7.2), 0.0], method="Nelder-Mead",
                 options=dict(xatol=1e-6, fatol=1e-10, maxiter=2000))
    return r.x, tb


def fit_power(m, y, t):
    """s_i = exp(a) * (t_i / tbar)^g."""
    tb = t.mean()

    def f(th):
        s = np.exp(th[0]) * (t / tb) ** th[1]
        s = np.clip(s, 2.0, 30.0)
        return nll_p(1 / (1 + np.exp(-m / s)), y).mean()
    r = minimize(f, [np.log(7.2), 0.0], method="Nelder-Mead",
                 options=dict(xatol=1e-6, fatol=1e-10, maxiter=2000))
    return r.x, tb


def clus(v):
    v = np.asarray(v, float); k = len(v)
    se = v.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return v.mean(), v.mean() - tc * se, v.mean() + tc * se, v.mean() / se, k


def main():
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
    f = f[f.season >= FROM].copy()
    f = f.dropna(subset=["m_us", "margin_actual"])
    f["y"] = (f["margin_actual"] > 0).astype(float)
    f["tot"] = pd.to_numeric(f["open_total"], errors="coerce")
    f.loc[(f["tot"] <= 100) | (f["tot"] > 300), "tot"] = np.nan
    f = f.sort_values(["season", "game_date", "game_id"]).reset_index(drop=True)
    print(f"frame {len(f):,} games; usable open totals "
          f"{f.tot.notna().sum():,} ({100*f.tot.notna().mean():.1f}%)")
    seasons = sorted(f.season.unique())

    rows = []
    for i, s in enumerate(seasons):
        if i == 0:
            continue
        tr, te = f[f.season.isin(seasons[:i])], f[f.season == s]
        trm = tr.dropna(subset=["tot"])
        y_tr, y_te = tr.y.to_numpy(float), te.y.to_numpy(float)
        m_tr, m_te = tr.m_us.to_numpy(float), te.m_us.to_numpy(float)

        s0 = fit_const(m_tr, y_tr)
        p_inc = 1 / (1 + np.exp(-m_te / s0))

        thA, tbA = fit_linear(trm.m_us.to_numpy(float), trm.y.to_numpy(float),
                              trm.tot.to_numpy(float))
        thB, tbB = fit_power(trm.m_us.to_numpy(float), trm.y.to_numpy(float),
                             trm.tot.to_numpy(float))
        tt = te.tot.to_numpy(float)
        sA = np.where(np.isnan(tt), s0,
                      np.clip(np.exp(thA[0]) + thA[1] * (tt - tbA), 2, 30))
        sB = np.where(np.isnan(tt), s0,
                      np.clip(np.exp(thB[0]) * (np.nan_to_num(tt, nan=tbB) / tbB)
                              ** thB[1], 2, 30))
        pA = 1 / (1 + np.exp(-m_te / sA))
        pB = 1 / (1 + np.exp(-m_te / sB))
        rows.append(dict(
            season=s, n=len(te), s0=float(s0), s1=float(thA[1]),
            gamma=float(thB[1]),
            ll_inc=float(nll_p(p_inc, y_te).mean()),
            ll_A=float(nll_p(pA, y_te).mean()),
            ll_B=float(nll_p(pB, y_te).mean()),
            side_flip=float(np.mean((p_inc > 0.5) != (pA > 0.5))),
            p_inc=float(p_inc.mean()), p_A=float(pA.mean()),
            base=float(y_te.mean())))
        rows[-1]["dA"] = rows[-1]["ll_A"] - rows[-1]["ll_inc"]
        rows[-1]["dB"] = rows[-1]["ll_B"] - rows[-1]["ll_inc"]
    r = pd.DataFrame(rows)
    print("\n" + r[["season", "n", "s0", "s1", "gamma", "ll_inc", "ll_A",
                    "dA", "dB", "side_flip"]].to_string(
        index=False, float_format=lambda v: f"{v:9.5f}"))

    # MDE80 from a within-season permutation of the total
    nulls = []
    for sd in range(8):
        rng = np.random.default_rng(sd)
        g = f.copy()
        for s in seasons:
            m = (g.season == s).to_numpy()
            g.loc[m, "tot"] = rng.permutation(g.loc[m, "tot"].to_numpy())
        for i, s in enumerate(seasons):
            if i == 0:
                continue
            tr, te = g[g.season.isin(seasons[:i])], g[g.season == s]
            trm = tr.dropna(subset=["tot"])
            s0 = fit_const(tr.m_us.to_numpy(float), tr.y.to_numpy(float))
            th, tb = fit_linear(trm.m_us.to_numpy(float),
                                trm.y.to_numpy(float), trm.tot.to_numpy(float))
            tt = te.tot.to_numpy(float)
            sA = np.where(np.isnan(tt), s0,
                          np.clip(np.exp(th[0]) + th[1] * (tt - tb), 2, 30))
            y = te.y.to_numpy(float); m_ = te.m_us.to_numpy(float)
            nulls.append(nll_p(1 / (1 + np.exp(-m_ / sA)), y).mean()
                         - nll_p(1 / (1 + np.exp(-m_ / s0)), y).mean())
    k = len(seasons) - 1
    mde = ((stats.t.ppf(0.975, k - 1) + stats.t.ppf(0.80, k - 1))
           * np.std(nulls, ddof=1) / np.sqrt(k))
    print(f"\nMDE80 (permutation null, stated first): {mde:.5f} nats")

    for col, nm in (("dA", "ARM A linear"), ("dB", "ARM B power")):
        m, lo, hi, t, kk = clus(r[col])
        print(f"\n=== {nm} ===")
        print(f"  season-clustered mean delta {m:+.6f}")
        print(f"  95% CI ({kk-1} dof)          [{lo:+.6f}, {hi:+.6f}]")
        print(f"  t {t:+.2f}   better in {int((r[col]<0).sum())}/{kk}")
        di = (r.p_inc - r.base).abs().mean(); dc = (r.p_A - r.base).abs().mean()
        if col == "dA":
            print(f"  calibration drift: inc {di:.4f} chal {dc:.4f} -> "
                  f"{'PASS' if dc <= di + 1e-6 else 'VETO'}")
            print(f"  T4 side flips: {100*r.side_flip.mean():.2f}% of games")
        print(f"  VERDICT: {'SHIP' if hi < 0 else 'NO SHIP — CI includes zero'}")
    print(f"\n  T1: s1 sign — {'POSITIVE as predicted' if r.s1.mean() > 0 else 'NEGATIVE, contra T1'} "
          f"(mean {r.s1.mean():+.5f})")

    json.dump({"rows": rows, "mde80": float(mde)},
              open(ROOT / "data" / "d241_joint_market.json", "w"), default=float)
    print("\nwrote data/d241_joint_market.json")


if __name__ == "__main__":
    main()
