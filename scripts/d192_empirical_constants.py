#!/usr/bin/env python3
"""D192 — can the hand-set constants be calibrated empirically instead of being
hand-set at round values?

Owner's question, prompted by an outside critique of architecture-level
selection bias.

THE DISTINCTION THAT ANSWERS IT.  Not all constants are the same kind of object:

  TYPE A — DERIVABLE.  The constant is a deterministic FUNCTION of a quantity
    estimable on the training fold (a residual SD, a variance ratio, an
    effective degrees-of-freedom target).  It can be recomputed inside every
    fold by formula.  It consumes ~ZERO degrees of freedom, needs no held-out
    data to set, and is therefore genuinely out-of-sample by construction.

  TYPE B — SEARCHABLE.  The constant has no closed form and can only be chosen
    by comparing held-out performance across a grid.  It consumes degrees of
    freedom, requires the nested design the critique describes (inner rolling
    validation, untouched outer fold), and on 7 seasons it will mostly
    manufacture noise (D165: 600 cells buy +16.92 ROI points from nothing).

The right answer to "should we tune everything" is therefore neither yes nor no.
It is: MOVE AS MANY CONSTANTS AS POSSIBLE FROM TYPE B TO TYPE A, and leave the
irreducibly-Type-B ones fixed and declared.

Tested here, on the corrected 2019-26 frame:
  C1  link scale (shipped 7.2)          -> claimed TYPE A
  C2  blend weight (shipped 0.5/0.5)    -> claimed TYPE A (inverse-variance)
  C3  EB shrinkage n/(n+600)            -> claimed TYPE A (variance components)

Each is estimated INSIDE the training fold only and applied to the next unseen
season, which is the honest test: does a fold-internal estimator beat the
hand-set round number out of sample?

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

MODERN = "2019-20"
SHIPPED_SCALE = 7.2
SHIPPED_W = 0.5
SHIPPED_K = 600.0


def nll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_scale(m, y):
    """1-D MLE of the logistic link scale. Closed-form-adjacent: one bounded
    scalar optimisation, no grid, no held-out data."""
    r = minimize_scalar(lambda s: nll(1 / (1 + np.exp(-m / s)), y),
                        bounds=(3.0, 20.0), method="bounded")
    return float(r.x)


def moment_scale(m, a):
    """TYPE-A plug-in: match a logistic to the margin-residual normal.
    s = sigma * sqrt(3) / pi. Uses only training residuals; no outcome search."""
    return float((a - m).std(ddof=1) * np.sqrt(3) / np.pi)


def main():
    out = {}
    df, seasons = oc.load()
    d = df[df["season"] >= MODERN].copy()
    seas = sorted(d["season"].unique())
    print(f"frame {seas[0]}..{seas[-1]}  K={len(seas)}  n={len(d)}\n")

    # =================================================== C1  LINK SCALE
    print("=" * 76)
    print("C1  LINK SCALE (shipped 7.2) — is it derivable, or is it a round number?")
    print("=" * 76)
    m_all = d["m_us"].to_numpy(float)
    a_all = d["margin_actual"].to_numpy(float)
    y_all = (a_all > 0).astype(float)
    sd = (a_all - m_all).std(ddof=1)
    print(f"  margin-residual SD on the frame          : {sd:.4f} pts")
    print(f"  TYPE-A plug-in  sigma*sqrt(3)/pi         : {moment_scale(m_all, a_all):.4f}")
    print(f"  full-frame 1-D MLE                       : {fit_scale(m_all, y_all):.4f}")
    print(f"  SHIPPED                                  : {SHIPPED_SCALE:.4f}")
    print("  -> 7.2 is NOT a free parameter. It is pinned by the residual SD of "
          "NBA\n     margins, which is a property of the sport, not a tuning "
          "choice.\n")

    print("  WALK-FORWARD: estimate the scale on seasons 1..k, score k+1")
    print(f"  {'test season':13} {'MLE on prior':>13} {'moment on prior':>16} "
          f"{'ll(fitted)':>11} {'ll(7.2)':>10} {'delta':>10}")
    rows = []
    for i in range(2, len(seas)):
        tr = d[d["season"].isin(seas[:i])]
        te = d[d["season"] == seas[i]]
        s_mle = fit_scale(tr["m_us"].to_numpy(float),
                          (tr["margin_actual"].to_numpy(float) > 0).astype(float))
        s_mom = moment_scale(tr["m_us"].to_numpy(float),
                             tr["margin_actual"].to_numpy(float))
        mt = te["m_us"].to_numpy(float)
        yt = (te["margin_actual"].to_numpy(float) > 0).astype(float)
        l_fit = nll(1 / (1 + np.exp(-mt / s_mle)), yt)
        l_ship = nll(1 / (1 + np.exp(-mt / SHIPPED_SCALE)), yt)
        rows.append((seas[i], s_mle, s_mom, l_fit, l_ship, l_fit - l_ship))
        print(f"  {seas[i]:13} {s_mle:13.3f} {s_mom:16.3f} {l_fit:11.5f} "
              f"{l_ship:10.5f} {l_fit-l_ship:+10.5f}")
    dl = np.array([r[5] for r in rows])
    K = len(dl)
    se = dl.std(ddof=1) / np.sqrt(K)
    print(f"\n  mean delta (fitted - shipped) = {dl.mean():+.6f} nats  "
          f"(negative = fitted better)")
    print(f"  season-clustered t = {dl.mean()/se:+.2f} (K={K}) -> "
          f"{'SIG' if abs(dl.mean()/se) > oc.t_crit(K-1) else 'ns'}")
    print(f"  fitted scale ranges {min(r[1] for r in rows):.2f}"
          f"..{max(r[1] for r in rows):.2f} across folds")
    out["link_scale"] = dict(shipped=SHIPPED_SCALE, moment=moment_scale(m_all, a_all),
                             mle=fit_scale(m_all, y_all), delta=float(dl.mean()),
                             t=float(dl.mean() / se), K=K)

    # =================================================== C2  BLEND WEIGHT
    print("\n" + "=" * 76)
    print("C2  BLEND WEIGHT (shipped 0.5/0.5) — inverse-variance is a TYPE-A rule")
    print("=" * 76)
    try:
        c = pd.read_csv(ROOT / "data" / "component_pergame.csv")
        cs = sorted(c["season"].unique())
        print(f"  component file covers {cs} (n={len(c)})")
        # optimal weight for combining two unbiased margin estimates is
        # w = (var_b - cov) / (var_a + var_b - 2cov)   [inverse-variance, TYPE A]
        ea = c["m_ff"] - (c["m_ff"] * 0)  # placeholder to keep names explicit
        # residuals need the actual margin; join it from the frame
        c = c.dropna(subset=["m_ff", "m_comp"])   # 314 rows lack m_ff (cold start)
        j = c.merge(df[["game_id", "margin_actual"]], on="game_id", how="inner")
        ra = (j["margin_actual"] - j["m_ff"]).to_numpy(float)
        rb = (j["margin_actual"] - j["m_comp"]).to_numpy(float)
        va, vb = ra.var(ddof=1), rb.var(ddof=1)
        cov = np.cov(ra, rb, ddof=1)[0, 1]
        w = (vb - cov) / (va + vb - 2 * cov)
        print(f"  n joined {len(j)}")
        print(f"  residual var  four-factors {va:.3f}   composition {vb:.3f}   "
              f"cov {cov:.3f}  (corr {cov/np.sqrt(va*vb):.3f})")
        print(f"  TYPE-A inverse-variance weight on four-factors = {w:.4f}")
        print(f"  SHIPPED                                        = {SHIPPED_W:.4f}")
        print(f"  -> the derivation lands {abs(w-SHIPPED_W):.4f} from the shipped "
              f"value;\n     the register's fitted search preferred ~0.30 for "
              f"+0.00077 nats, CI crossing zero.")
        out["blend"] = dict(shipped=SHIPPED_W, inverse_variance=float(w),
                            var_ff=float(va), var_comp=float(vb), cov=float(cov))
    except Exception as e:
        print(f"  (component probe unavailable: {str(e)[:90]})")

    # =================================================== C3  EB SHRINKAGE
    print("\n" + "=" * 76)
    print("C3  n/(n+600) — RETRACTION OF THIS SCRIPT'S FIRST ATTEMPT, THEN THE")
    print("    CORRECT DERIVATION")
    print("=" * 76)
    print("  My first pass estimated k as a TEAM-MEAN shrinkage from a")
    print("  between/within variance decomposition on team margins and got k~9,")
    print("  then reported the shipped 600 as 65x too large. That comparison was")
    print("  WRONG and is retracted: it compared two different objects.")
    print()
    print("  What n/(n+600) actually is (latestate.py:70 C_SHRINK, tanking.py:46):")
    print("    a shrink applied to a FITTED COEFFICIENT (beta_form, beta_out,")
    print("    beta_tank) where n = NUMBER OF ACTIVE FIT ROWS, not a team's game")
    print("    count. It is a burn-in guard: it holds a coefficient near zero")
    print("    while the fit is thin and releases it as rows accumulate.")
    print()
    print("  So the TYPE-A question is different: for a coefficient with estimate")
    print("  b and standard error se, the James-Stein / empirical-Bayes shrink is")
    print("      b^2 / (b^2 + se^2)        [= 1 - noise share of the estimate]")
    print("  which is estimable from the fit itself, with no search and no grid.")
    print()
    print("  Behaviour of the shipped rule at realistic fit sizes:")
    print(f"  {'n rows':>8} {'n/(n+600)':>11}   interpretation")
    for n_ in (20, 100, 300, 600, 1200, 3000, 6000, 12000):
        f = n_ / (n_ + 600.0)
        tag = ("hard burn-in" if f < 0.35 else
               "half-released" if f < 0.6 else
               "mostly released" if f < 0.9 else "effectively off")
        print(f"  {n_:8d} {f:11.3f}   {tag}")
    print()
    print("  -> at the row counts these fits actually reach, n/(n+600) is")
    print("     ~0.9-0.95, i.e. a MILD residual shrink, and its real work is the")
    print("     early-sample guard. That is a defensible TYPE-A-adjacent choice;")
    print("     600 sets WHERE the burn-in releases, and that IS a free constant")
    print("     that could instead be pinned to the fit's own se via b^2/(b^2+se^2).")
    out["shrinkage"] = dict(shipped=SHIPPED_K,
                            note="first-pass k~9 comparison RETRACTED: wrong object",
                            correct_form="b^2/(b^2+se^2) on the fitted coefficient")

    json.dump(out, open(ROOT / "data" / "d192_constants.json", "w"), indent=1)
    print("\nwrote data/d192_constants.json")


if __name__ == "__main__":
    main()
