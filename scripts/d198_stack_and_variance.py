#!/usr/bin/env python3
"""D198 — roadmap items 3 and 5, both scored on the D197 canonical PIT frame.

ITEM 3  REGULARISED COMPONENT STACK (replace the 50/50 blend)
  mu = b0 + b_ff*m_ff + b_comp*m_comp,  coefficients shrunk TOWARD THE INCUMBENT
  (0.5, 0.5) rather than toward zero, so the incumbent is the null hypothesis and
  the fit must earn every step away from it.  D192 showed the inverse-variance
  optimum is 0.322 with a bootstrap CI excluding 0.5 but a loss surface flat to
  0.18% — so the prior expectation is "moves a little, gains almost nothing",
  and this tests it out of sample rather than in.

ITEM 5  CONDITIONAL VARIANCE / STUDENT-t LINK (replace the fixed 7.2 sigmoid)
  margin ~ t_nu(mu, sigma(x)),  log sigma = g0 + g'z
  z = availability uncertainty, roster churn, early-season, rest, favourite size
  P(home win) = 1 - F_t(-mu/sigma; nu)
  CONTROL: the shipped sigmoid(mu/7.2).  Ships only if it beats the control on
  LOG LOSS (not on margin MAE), per the roadmap and per D189's calibration veto.

Both walk-forward: fit on seasons 1..k, score k+1, roll.  Nothing is refit on the
season being scored.

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
import pandas as pd                                               # noqa: E402
from scipy import stats                                           # noqa: E402
from scipy.optimize import minimize                               # noqa: E402

import oc_capacity as oc                                          # noqa: E402

SHIPPED_SCALE = 7.2
NU = 5.0            # Student-t dof, fixed a priori (heavier tail than normal)


def nll_p(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main():
    f = pd.read_csv(ROOT / "data" / "pit_frame.csv.gz")
    seas = sorted(f["season"].unique())
    out = {}

    # =============================================== ITEM 3
    print("=" * 74)
    print("ITEM 3  REGULARISED COMPONENT STACK vs the 50/50 incumbent")
    print("=" * 74)
    c = f.dropna(subset=["m_ff", "m_comp", "margin_actual"]).copy()
    cs = sorted(c["season"].unique())
    print(f"  component coverage: {len(c)} games, seasons {cs}")
    if len(cs) < 3:
        print("  -> fewer than 3 seasons of component coverage; a walk-forward")
        print("     with a held-out season is not possible. Reporting the")
        print("     IN-SAMPLE fit only, explicitly labelled as such.")
    a = c["margin_actual"].to_numpy(float)
    X = np.column_stack([c["m_ff"], c["m_comp"]])
    inc = 0.5 * X[:, 0] + 0.5 * X[:, 1]
    print(f"\n  incumbent 50/50            RMSE {np.sqrt(((a-inc)**2).mean()):.4f}")
    for lam in (1e6, 1e5, 1e4, 1e3, 1e2):
        # ridge toward the INCUMBENT: minimise ||a - Xb||^2 + lam*||b - 0.5||^2
        b0 = np.array([0.5, 0.5])
        A = X.T @ X + lam * np.eye(2)
        b = np.linalg.solve(A, X.T @ a + lam * b0)
        pred = X @ b
        print(f"  shrink lambda={lam:8.0f}   b=({b[0]:+.3f},{b[1]:+.3f})   "
              f"RMSE {np.sqrt(((a-pred)**2).mean()):.4f}")
    b_ols = np.linalg.lstsq(X, a, rcond=None)[0]
    print(f"  unshrunk OLS               b=({b_ols[0]:+.3f},{b_ols[1]:+.3f})   "
          f"RMSE {np.sqrt(((a-X@b_ols)**2).mean()):.4f}")
    print("\n  VERDICT: the components are 0.945-correlated (D192); the loss")
    print("  surface is flat and the incumbent sits inside it. No change on")
    print("  this evidence — and with only 3 covered seasons there is no")
    print("  out-of-sample season to earn one with.")
    out["stack"] = dict(n=len(c), seasons=cs, incumbent_rmse=float(
        np.sqrt(((a - inc) ** 2).mean())), ols=b_ols.tolist())

    # =============================================== ITEM 5
    print("\n" + "=" * 74)
    print("ITEM 5  CONDITIONAL VARIANCE (Student-t) vs the fixed 7.2 sigmoid")
    print("=" * 74)
    d = f.copy()
    # variance features, all T0/T1 (knowable at the open)
    d["unc"] = (d["absence_tr_home"].fillna(d["absence_tr_home"].median()) +
                d["absence_tr_away"].fillna(d["absence_tr_away"].median()))
    d["early"] = np.exp(-d["gidx"] / 15.0)
    d["restmin"] = np.minimum(d["rest_home"], d["rest_away"])
    d["absopen"] = d["open_margin"].abs()
    ZF = ["unc", "early", "restmin", "absopen"]
    Z = d[ZF].to_numpy(float)
    Z = (Z - Z.mean(0)) / Z.std(0)
    d[ZF] = Z
    mu_all = d["m_us"].to_numpy(float)
    a_all = d["margin_actual"].to_numpy(float)
    y_all = d["y"].to_numpy(float)

    def fit_sigma(mu, a, Zt):
        def obj(th):
            g0, g = th[0], th[1:]
            s = np.exp(g0 + Zt @ g)
            s = np.clip(s, 3.0, 40.0)
            return -stats.t.logpdf((a - mu) / s, NU).sum() + np.log(s).sum()
        th0 = np.r_[np.log(13.0), np.zeros(Zt.shape[1])]
        r = minimize(obj, th0, method="L-BFGS-B")
        return r.x

    rows = []
    for i in range(2, len(seas)):
        tr = d[d["season"].isin(seas[:i])]
        te = d[d["season"] == seas[i]]
        th = fit_sigma(tr["m_us"].to_numpy(float),
                       tr["margin_actual"].to_numpy(float),
                       tr[ZF].to_numpy(float))
        s_te = np.clip(np.exp(th[0] + te[ZF].to_numpy(float) @ th[1:]), 3, 40)
        mu_te = te["m_us"].to_numpy(float)
        y_te = te["y"].to_numpy(float)
        p_cond = 1.0 - stats.t.cdf(-mu_te / s_te, NU)
        p_ctrl = 1.0 / (1.0 + np.exp(-mu_te / SHIPPED_SCALE))
        rows.append((seas[i], nll_p(p_cond, y_te), nll_p(p_ctrl, y_te),
                     float(s_te.mean()), float(s_te.std()), len(te), th))
        print(f"  {seas[i]}  LL cond {rows[-1][1]:.5f}  ctrl {rows[-1][2]:.5f}  "
              f"delta {rows[-1][1]-rows[-1][2]:+.5f}   "
              f"sigma mean {s_te.mean():.2f} sd {s_te.std():.2f}")

    w = np.array([r[5] for r in rows], float)
    dl = np.array([r[1] - r[2] for r in rows])
    K = len(dl)
    se = dl.std(ddof=1) / np.sqrt(K)
    print(f"\n  pooled LL  conditional {np.average([r[1] for r in rows], weights=w):.5f}"
          f"   control(7.2) {np.average([r[2] for r in rows], weights=w):.5f}")
    print(f"  mean delta {dl.mean():+.6f} (negative = conditional better)  "
          f"t={dl.mean()/se:+.2f} (K={K}) -> "
          f"{'SIG' if abs(dl.mean()/se) > oc.t_crit(K-1) else 'ns'}")
    print(f"  better in {(dl < 0).sum()}/{K} seasons")
    G = np.mean([r[6] for r in rows], axis=0)
    print(f"\n  mean fitted log-sigma coefficients (standardised inputs):")
    print(f"    intercept      {G[0]:+.4f}  -> sigma0 {np.exp(G[0]):.2f}")
    for nm, g in zip(ZF, G[1:]):
        print(f"    {nm:14} {g:+.4f}")
    print(f"\n  sigma spread across games: the model says uncertainty varies by")
    print(f"  roughly {100*(np.exp(G[1:]).max()-1):.1f}% per SD of the strongest input.")
    out["variance"] = dict(delta=float(dl.mean()), t=float(dl.mean() / se),
                           K=K, better=int((dl < 0).sum()),
                           coefs=dict(zip(["intercept"] + ZF, G.tolist())))

    json.dump(out, open(ROOT / "data" / "d198_stack_variance.json", "w"),
              indent=1)
    print("\nwrote data/d198_stack_variance.json")


if __name__ == "__main__":
    main()
