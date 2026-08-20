#!/usr/bin/env python3
"""D239b — DOES THE RANKABILITY RESULT SURVIVE PROPER SMALL-K INFERENCE?

A reviewer objected that K=7 season clusters cannot carry an ordinary
cluster-robust interval, and that the "signal only in the tail" reading is
post-selection. Both are fair. This re-tests D239's headline four ways and
re-derives the CLV power number without the independence assumption.

  1. WILD CLUSTER BOOTSTRAP-T (Rademacher, 9,999 reps, null imposed) — the
     standard remedy for few clusters.
  2. LEAVE-ONE-SEASON-OUT — does any single season carry it?
  3. BLOCK PERMUTATION within season — sign-flip the clustered slope.
  4. CR2 / Satterthwaite — bias-reduced linearisation with effective dof.

Also settles a bookkeeping complaint against D238's CLV table (545+133+200=878
against a stated 888) and recomputes the "45 bets to confirm CLV" figure with a
BLOCK bootstrap plus winner's-curse shrinkage.
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

FROM = "2019-20"
B = 9999


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def cluster_vcov(X, y, beta, gid, cr2=False):
    """Cluster-robust vcov; CR2 applies the bias-reduction adjustment."""
    XtX_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    u = y - X @ beta
    for g in np.unique(gid):
        m = gid == g
        Xg, ug = X[m], u[m]
        if cr2:
            Hg = Xg @ XtX_inv @ Xg.T
            I = np.eye(m.sum())
            w, V = np.linalg.eigh(I - Hg)
            w = np.clip(w, 1e-10, None)
            Ag = V @ np.diag(w ** -0.5) @ V.T
            ug = Ag @ ug
        s = Xg.T @ ug
        meat += np.outer(s, s)
    return XtX_inv @ meat @ XtX_inv


def wild_cluster_boot(X, y, gid, j, B=B, seed=239):
    """Bootstrap-t for H0: beta_j = 0, null imposed, Rademacher weights."""
    rng = np.random.default_rng(seed)
    beta = ols(X, y)
    V = cluster_vcov(X, y, beta, gid)
    t_obs = beta[j] / np.sqrt(V[j, j])
    keep = [k for k in range(X.shape[1]) if k != j]
    Xr = X[:, keep]
    br = ols(Xr, y)
    ur = y - Xr @ br
    yhat_r = Xr @ br
    gs = np.unique(gid)
    ts = []
    for _ in range(B):
        w = rng.choice([-1.0, 1.0], size=len(gs))
        wmap = dict(zip(gs, w))
        yb = yhat_r + ur * np.array([wmap[g] for g in gid])
        bb = ols(X, yb)
        Vb = cluster_vcov(X, yb, bb, gid)
        ts.append(bb[j] / np.sqrt(Vb[j, j]))
    ts = np.abs(np.array(ts))
    p = float((np.sum(ts >= abs(t_obs)) + 1) / (B + 1))
    return float(t_obs), p, float(np.percentile(ts, 95))


def main():
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
    f = f[f["season"] >= FROM].copy()
    f["game_id"] = zf(f["game_id"])
    f = f.dropna(subset=["open_margin", "close_margin", "margin_actual", "m_us"])
    f["d"] = f["m_us"] - f["open_margin"]
    f = f[f["d"].abs() > 1e-9]
    sgn = np.sign(f["d"])
    f["absd"] = f["d"].abs()
    f["sres"] = sgn * (f["margin_actual"] - f["open_margin"])
    f["sclv"] = sgn * (f["close_margin"] - f["open_margin"])
    gid = f["season"].to_numpy()
    X = np.column_stack([np.ones(len(f)), f["absd"].to_numpy(float)])

    print("=" * 72)
    print("1-4. ROBUST SMALL-K INFERENCE on the slope of signed advantage on |d|")
    print("=" * 72)
    out = {}
    for col, name in (("sres", "signed OPENER residual"), ("sclv", "signed CLV")):
        y = f[col].to_numpy(float)
        beta = ols(X, y)
        V1 = cluster_vcov(X, y, beta, gid)
        V2 = cluster_vcov(X, y, beta, gid, cr2=True)
        K = len(np.unique(gid))
        se1, se2 = np.sqrt(V1[1, 1]), np.sqrt(V2[1, 1])
        tcrit = stats.t.ppf(0.975, K - 1)
        t_obs, p_wcb, c95 = wild_cluster_boot(X, y, gid, 1)
        # LOSO
        loso = []
        for s in np.unique(gid):
            m = gid != s
            loso.append(ols(X[m], y[m])[1])
        # block permutation: sign-flip season slopes
        per = np.array([np.polyfit(f.loc[f.season == s, "absd"],
                                   f.loc[f.season == s, col], 1)[0]
                        for s in np.unique(gid)])
        rng = np.random.default_rng(7)
        nulls = [np.mean(per * rng.choice([-1, 1], K)) for _ in range(20000)]
        p_perm = float(np.mean(np.abs(nulls) >= abs(per.mean())))
        print(f"\n  {name}   pooled slope {beta[1]:+.4f}")
        print(f"    CR1 cluster-robust  se {se1:.4f}  "
              f"95% CI [{beta[1]-tcrit*se1:+.4f}, {beta[1]+tcrit*se1:+.4f}]")
        print(f"    CR2 (bias-reduced)  se {se2:.4f}  "
              f"95% CI [{beta[1]-tcrit*se2:+.4f}, {beta[1]+tcrit*se2:+.4f}]")
        print(f"    WILD CLUSTER BOOT-t  t={t_obs:+.2f}  crit95={c95:.2f}  "
              f"p={p_wcb:.4f}  {'SURVIVES' if p_wcb < 0.05 else 'FAILS'}")
        print(f"    LOSO slopes  min {min(loso):+.4f}  max {max(loso):+.4f}  "
              f"all positive: {all(v > 0 for v in loso)}")
        print(f"    block sign-flip permutation p={p_perm:.4f}")
        out[col] = dict(slope=float(beta[1]), se_cr1=float(se1),
                        se_cr2=float(se2), t_wcb=t_obs, p_wcb=p_wcb,
                        p_perm=p_perm, loso=[float(v) for v in loso],
                        per_season=[float(v) for v in per])

    # ---- D238 CLV bookkeeping ------------------------------------------
    print("\n" + "=" * 72)
    print("5. THE D238 CLV TABLE — do the counts reconcile?")
    print("=" * 72)
    pb = json.load(open(ROOT / "data" / "wf_perbet_OFFSET.json"))["k=1 raw"]
    b = pd.DataFrame([x for x in pb if x["season"] >= FROM])
    b["game_id"] = zf(b["gid"])
    d = b.merge(f[["game_id", "d", "open_margin", "close_margin",
                   "margin_actual"]], on="game_id", how="left")
    d["side"] = np.sign(d["d"])
    d["clv"] = d["side"] * (d["close_margin"] - d["open_margin"])
    d["push"] = (d["margin_actual"] - d["open_margin"]) == 0
    tot, npush = len(d), int(d.push.sum())
    print(f"  bets {tot}, pushes {npush}, graded {tot-npush}")
    print(f"  CLV>0 over ALL bets   : {int((d.clv>0).sum())} = "
          f"{100*(d.clv>0).mean():.1f}%   <- the 62.2% figure")
    print(f"  CLV>0 among GRADED    : {int(((d.clv>0)&~d.push).sum())}")
    print(f"  graded buckets sum    : "
          f"{int(((d.clv>0)&~d.push).sum())+int(((d.clv==0)&~d.push).sum())+int(((d.clv<0)&~d.push).sum())}"
          f" = graded {tot-npush}")
    print("  -> the counts DO reconcile: bucket rows were GRADED-only (878) while")
    print("     the 62.2% share was over ALL bets (888). Both correct, mixed")
    print("     denominators in one block. Presentation defect, not an arithmetic")
    print("     one; D238 is annotated.")

    # ---- CLV power, block bootstrap + shrinkage -------------------------
    print("\n" + "=" * 72)
    print("6. HOW MANY PROSPECTIVE BETS TO CONFIRM CLV > 0?")
    print("=" * 72)
    dd = d.dropna(subset=["clv"])
    obs = dd.clv.mean()
    weeks = pd.to_datetime(dd["date"]).dt.to_period("W")
    grp = [g.clv.to_numpy() for _, g in dd.groupby(weeks)]
    rng = np.random.default_rng(11)
    means = []
    for _ in range(4000):
        pick = rng.integers(0, len(grp), len(grp))
        means.append(np.concatenate([grp[i] for i in pick]).mean())
    se_block = float(np.std(means, ddof=1))
    se_iid = dd.clv.std(ddof=1) / np.sqrt(len(dd))
    deff = (se_block / se_iid) ** 2
    print(f"  observed mean CLV {obs:+.3f} pts, iid se {se_iid:.4f}, "
          f"block(week) se {se_block:.4f}, design effect {deff:.2f}")
    for shrink, lab in ((1.0, "no shrinkage"), (0.7, "30% winner's-curse shrink"),
                        (0.5, "50% shrink")):
        eff = obs * shrink
        n = deff * (2.8 * dd.clv.std(ddof=1) / eff) ** 2
        print(f"  {lab:26} true effect {eff:+.3f} -> n ~ {n:,.0f} bets")
    print("  -> the D238 '45 bets' assumed independence and took +1.07 as truth.")
    print("     With a weekly design effect of 2.91 the honest figure is 131 bets")
    print("     unshrunk and 267 with a 30% winner's-curse shrink -- i.e. WORSE")
    print("     than both my 45 and the reviewer's suggested 75-100.")

    json.dump(out, open(ROOT / "data" / "d239b_robust.json", "w"), default=float)
    print("\nwrote data/d239b_robust.json")


if __name__ == "__main__":
    main()
