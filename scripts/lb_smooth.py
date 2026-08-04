"""Follow-up on the ONE football analogue that does NOT come back null here:
SMOOTHING the market across a season.

  football_exercise: LOO static fit to a season's own odds, -0.0031
  CI[-0.0069,+0.0008] ns — and that test RETRACTED an earlier "beats the
  market" claim when it was season-clustered.

  NBA first pass (lb_football_analogues.py): pure REPLACEMENT is harmful
  (+0.0015 ns), but a 45/55 BLEND of the closing price with the season ridge
  reconstruction is -0.00480 CI[-0.00602,-0.00352], 18/19 seasons.

That is a big enough number that it has to be attacked, in this order:
  (1) is the blend weight honestly chosen?      -> LOSO selection
  (2) is it only an ORACLE?                     -> WALK-FORWARD (past only)
  (3) is it just a rescale of the price?        -> joint scale re-optimisation
  (4) DOES IT CLEAR THE VIG?                    -> ROI at real closing MLs
Everything season-clustered, plus the K-1 cluster-mean t (GATE_POLICY_V2 §9).

Read-only on data/nba.duckdb.  Run: python scripts/lb_smooth.py
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

import nbapred.threads                                           # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402

from lb_longshot import (cluster_boot, cluster_mean_t, load_panel,  # noqa: E402
                         logit, sides, sigmoid)

OUT = os.path.join(ROOT, "data", "lb_smooth.json")
N_BOOT = 4000
LAM = 0.1


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def design(hs, aw, teams):
    ti = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    n = len(hs)
    X = np.zeros((n, T + 1))
    X[np.arange(n), [ti[t] for t in hs]] = 1.0
    X[np.arange(n), [ti[t] for t in aw]] = -1.0
    X[:, T] = 1.0
    return X


def oracle_zhat(gm, z, lam=LAM):
    """Leave-one-GAME-out ridge fit inside each season — the football project's
    construction.  Uses the whole season, so it sees the future: ORACLE."""
    cl = gm.season.values
    zhat = np.empty(len(z))
    for s0 in sorted(set(cl)):
        m = cl == s0
        zz = z[m]
        teams = sorted(set(gm.home.values[m]) | set(gm.away.values[m]))
        X = design(gm.home.values[m], gm.away.values[m], teams)
        A = X.T @ X + lam * np.eye(X.shape[1])
        Ai = np.linalg.inv(A)
        fit = X @ (Ai @ (X.T @ zz))
        h = np.einsum("ij,jk,ik->i", X, Ai, X)
        zhat[m] = zz - (zz - fit) / np.clip(1.0 - h, 1e-6, None)
    return zhat


def walkforward_zhat(gm, z, lam=LAM, decay=1.0):
    """PAST-ONLY: inside each season, accumulate the normal equations day by
    day and predict each day BEFORE its own prices are folded in.  This is the
    tradeable version.  The ridge term IS the cold-start prior (strengths 0,
    home edge free), so day 1 emits the prior home edge."""
    cl = gm.season.values
    dates = gm.game_date.values
    zhat = np.empty(len(z))
    nprior = np.zeros(len(z))
    for s0 in sorted(set(cl)):
        m = np.where(cl == s0)[0]
        order = m[np.argsort(dates[m], kind="stable")]
        teams = sorted(set(gm.home.values[m]) | set(gm.away.values[m]))
        ti = {t: i for i, t in enumerate(teams)}
        T = len(teams)
        A = lam * np.eye(T + 1)
        b = np.zeros(T + 1)
        seen = 0
        i = 0
        while i < len(order):
            d0 = dates[order[i]]
            j = i
            while j < len(order) and dates[order[j]] == d0:
                j += 1
            day = order[i:j]
            beta = np.linalg.solve(A, b)
            for g in day:
                x = np.zeros(T + 1)
                x[ti[gm.home.values[g]]] = 1.0
                x[ti[gm.away.values[g]]] = -1.0
                x[T] = 1.0
                zhat[g] = float(x @ beta)
                nprior[g] = seen
            if decay != 1.0:
                A = decay * A + (1 - decay) * lam * np.eye(T + 1)
                b = decay * b
            for g in day:
                x = np.zeros(T + 1)
                x[ti[gm.home.values[g]]] = 1.0
                x[ti[gm.away.values[g]]] = -1.0
                x[T] = 1.0
                A += np.outer(x, x)
                b += x * z[g]
            seen += len(day)
            i = j
    return zhat, nprior


def report(tag, y, cl, base, p, res, extra=None):
    d = ll(y, p) - base
    lo, hi, se = cluster_boot(d, cl, n_boot=N_BOOT)
    tlo, thi, K = cluster_mean_t(d, cl)
    nwin = sum(1 for s0 in sorted(set(cl)) if d[cl == s0].mean() < 0)
    row = dict(d=float(d.mean()), lo=lo, hi=hi, se=se, tlo=tlo, thi=thi, K=K,
               seasons_improved=nwin, n=int(len(y)))
    if extra:
        row.update(extra)
    res[tag] = row
    print(f"  {tag:<44s} d={d.mean():+.5f} CI[{lo:+.5f},{hi:+.5f}]  "
          f"t-CI[{tlo:+.5f},{thi:+.5f}]  {nwin}/{K} seasons")
    return row


def bet_test(gm, p_model, tag, res, thrs=(0.0, 0.01, 0.02, 0.03, 0.05)):
    """THE TEST THAT MATTERS.  Bet whichever side the model thinks is
    mispriced by more than `thr`, at the REAL closing decimal."""
    dh = gm.dec_home.values
    da = gm.dec_away.values
    y = gm.y.values
    cl = gm.season.values
    eh = p_model * dh - 1.0
    ea = (1.0 - p_model) * da - 1.0
    rows = []
    for thr in thrs:
        take_h = eh > thr
        take_a = ea > thr
        sel = take_h | take_a
        both = take_h & take_a
        take_h = take_h & (~both | (eh >= ea))
        take_a = take_a & ~take_h
        if sel.sum() < 30:
            continue
        pnl = np.where(take_h, np.where(y > 0.5, dh - 1.0, -1.0),
                       np.where(y < 0.5, da - 1.0, -1.0))[sel]
        c = cl[sel]
        lo, hi, _ = cluster_boot(pnl, c, n_boot=N_BOOT)
        tlo, thi, K = cluster_mean_t(pnl, c)
        rows.append(dict(thr=float(thr), n=int(sel.sum()),
                         pct_dog=float((take_a[sel] & (p_model[sel] < 0.5)
                                        ).mean()),
                         roi=float(pnl.mean()), lo=lo, hi=hi,
                         tlo=tlo, thi=thi, K=K,
                         seasons_pos=int(sum(1 for s0 in sorted(set(c))
                                             if pnl[c == s0].mean() > 0))))
        print(f"    thr={thr:+.2f}  n={sel.sum():6d}  ROI={100*pnl.mean():+7.2f}%"
              f"  CI[{100*lo:+.2f},{100*hi:+.2f}]  t-CI[{100*tlo:+.2f},"
              f"{100*thi:+.2f}]  {rows[-1]['seasons_pos']}/{K} seasons +")
    res[tag] = rows
    return rows


def main():
    res = {}
    g = load_panel()
    sc = sides(g, "close")
    gm = sc[sc.side == "home"].copy().reset_index(drop=True)
    gm["dec_home"] = gm.dec.values
    aw = sc[sc.side == "away"][["game_date", "home", "away", "dec"]].rename(
        columns={"dec": "dec_away"})
    n0 = len(gm)
    gm = gm.merge(aw, on=["game_date", "home", "away"], how="left")
    assert len(gm) == n0
    y = gm.y.values
    cl = gm.season.values
    z = logit(gm.p_prop.values)
    base = ll(y, sigmoid(z))
    print(f"n={len(gm)} games, K={gm.season.nunique()} seasons; "
          f"market close LL {base.mean():.5f}")
    res["base_ll"] = float(base.mean())
    res["n"] = int(len(gm))

    zo = oracle_zhat(gm, z)
    zw, npr = walkforward_zhat(gm, z)
    res["corr_z_zhat_oracle"] = float(np.corrcoef(z, zo)[0, 1])
    res["corr_z_zhat_walkfwd"] = float(np.corrcoef(z, zw)[0, 1])

    # ---- (1) blend weight, in-sample optimum vs LOSO-selected -----------
    print("\n(1) BLEND WEIGHT — in-sample optimum vs leave-one-SEASON-out pick")
    grid = np.round(np.arange(0.0, 0.81, 0.05), 2)
    for lab, zh in (("ORACLE(LOO within season)", zo),
                    ("WALK-FORWARD(past only)", zw)):
        curve = [(float(w), float((ll(y, sigmoid((1 - w) * z + w * zh))
                                   - base).mean())) for w in grid]
        wbest = float(min(curve, key=lambda t: t[1])[0])
        print(f"  {lab}: in-sample best w={wbest:.2f}")
        report(f"blend_insample|{lab}", y, cl, base,
               sigmoid((1 - wbest) * z + wbest * zh), res,
               extra=dict(w=wbest, curve=curve))
        # LOSO-selected w
        p_loso = np.empty(len(y))
        picks = {}
        for s0 in sorted(set(cl)):
            tr = cl != s0
            c2 = [(float(w), float((ll(y[tr], sigmoid((1 - w) * z[tr]
                                                      + w * zh[tr]))
                                    - base[tr]).mean())) for w in grid]
            wb = float(min(c2, key=lambda t: t[1])[0])
            picks[s0] = wb
            te = ~tr
            p_loso[te] = sigmoid((1 - wb) * z[te] + wb * zh[te])
        report(f"blend_LOSO|{lab}", y, cl, base, p_loso, res,
               extra=dict(picks=picks))

    # ---- (3) is it just a rescale? --------------------------------------
    print("\n(3) IS IT JUST A RESCALE OF THE PRICE?  best pure scale vs blend, "
          "and the blend after re-optimising the scale")
    sg = np.round(np.arange(0.80, 1.351, 0.01), 2)
    sc_curve = [(float(s), float((ll(y, sigmoid(s * z)) - base).mean()))
                for s in sg]
    sbest = float(min(sc_curve, key=lambda t: t[1])[0])
    report("pure_scale_best", y, cl, base, sigmoid(sbest * z), res,
           extra=dict(s=sbest))
    for lab, zh in (("ORACLE", zo), ("WALKFWD", zw)):
        best = (None, 1e9)
        for w in grid:
            for s in sg:
                v = float((ll(y, sigmoid(s * ((1 - w) * z + w * zh)))
                           - base).mean())
                if v < best[1]:
                    best = ((float(w), float(s)), v)
        (wb, sb) = best[0]
        report(f"blend+scale|{lab}", y, cl, base,
               sigmoid(sb * ((1 - wb) * z + wb * zh)), res,
               extra=dict(w=wb, s=sb))

    # ---- (2)/(4) the economic test --------------------------------------
    print("\n(4) DOES IT CLEAR THE VIG?  bet the disagreement at the REAL "
          "closing moneyline")
    for lab, zh in (("ORACLE", zo), ("WALKFWD", zw)):
        c2 = [(float(w), float((ll(y, sigmoid((1 - w) * z + w * zh))
                                - base).mean())) for w in grid]
        wb = float(min(c2, key=lambda t: t[1])[0])
        print(f"  [{lab}] w={wb:.2f}")
        bet_test(gm, sigmoid((1 - wb) * z + wb * zh), f"bets|{lab}", res)
    print("  [CONTROL: the market against itself — bet its own price, "
          "which must lose exactly the vig]")
    bet_test(gm, gm.p_prop.values, "bets|MARKET_ITSELF", res,
             thrs=(-1.0, 0.0))

    # ---- where does the oracle gain live? -------------------------------
    zh = zo
    c2 = [(float(w), float((ll(y, sigmoid((1 - w) * z + w * zh))
                            - base).mean())) for w in grid]
    wb = float(min(c2, key=lambda t: t[1])[0])
    d = ll(y, sigmoid((1 - wb) * z + wb * zh)) - base
    q = np.quantile(npr, [0.2, 0.4, 0.6, 0.8])
    res["oracle_gain_by_gameno"] = [
        [int(k), int((np.digitize(npr, q) == k).sum()),
         float(d[np.digitize(npr, q) == k].mean())] for k in range(5)]
    print(f"\n  ORACLE gain by point in season (quintiles of games already "
          f"played): " + "  ".join(
              f"Q{k+1} {v[2]:+.5f}" for k, v in
              enumerate(res["oracle_gain_by_gameno"])))
    per = [[s0, float(d[cl == s0].mean())] for s0 in sorted(set(cl))]
    res["oracle_per_season"] = per
    print("  per season: " + " ".join(f"{s}:{v:+.4f}" for s, v in per))

    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(f"\nwrote {OUT}")
    return res


if __name__ == "__main__":
    main()
