"""Regime C: disciplined walk-forward tests for the two live candidates.

W1. Mid-range local recal, SEASON WALK-FORWARD (no pooled train->confirm reuse):
    fit local slope b on all PRIOR seasons' regime-C games (|l|<c), apply to the
    next season. 24-25 is a true pseudo-OOS year; 25-26 matches the main split.
    c in {3.5, 5.0} pts (flagged: c chosen post-hoc from sensitivity — this run
    measures whether the shape even generalizes season-to-season).
W2. Home-intercept correction, season walk-forward: c0 = mean resid_us on all
    PRIOR seasons' regime-C games, apply next season. Also scope check: mean
    resid_us by season globally and by phase (early gp<20 / core 20-54 / late).
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

OUTDIR = os.environ.get(
    "RW_OUT",
    "/tmp/claude-1004/-hdd-steveqin-sean-dev/4912ce58-69cd-488d-a5f4-0d3ed2eed30c/scratchpad",
)
SCALE = 7.2
rng = np.random.default_rng(31)
SEASONS = ["2023-24", "2024-25", "2025-26"]


def paired_delta(f, mask, p_new):
    sub = f[mask]
    eps = 1e-12
    p2 = p_new[mask]
    L2 = -(sub.y.values * np.log(p2 + eps) + (1 - sub.y.values) * np.log(1 - p2 + eps))
    dl = sub.L_us.values - L2
    n = len(dl)
    boot = [dl[rng.integers(0, n, n)].mean() for _ in range(10000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dl.mean(), lo, hi, n


def recal(lv, b, c):
    out = np.where(np.abs(lv) <= c, b * lv,
                   np.sign(lv) * (b * c + (np.abs(lv) - c)))
    return 1 / (1 + np.exp(-out))


def main():
    f = pd.read_pickle(os.path.join(OUTDIR, "regimec_features.pkl"))
    f["logit_us"] = np.log(f.p_us / (1 - f.p_us))
    l = f.logit_us.values

    print("=== W1. season walk-forward local recal ===")
    for cpts in (3.5, 5.0):
        c = cpts / SCALE
        for i in (1, 2):
            tgt = SEASONS[i]
            prior = SEASONS[:i]
            fit_m = (f.regimeC & f.season.isin(prior) & (np.abs(l) < c)).values
            apply_m = (f.regimeC & (f.season == tgt)).values
            r = sm.GLM(f.y.values[fit_m], sm.add_constant(l[fit_m]),
                       family=sm.families.Binomial()).fit()
            b = r.params[1]
            p_new = recal(l, b, c)
            d = paired_delta(f, apply_m, p_new)
            print(f"  c={cpts} -> {tgt}: b_fit={b:+.2f} (se {r.bse[1]:.2f}, n={fit_m.sum()}) "
                  f"dLL {d[0]:+.5f} ({d[1]:+.5f},{d[2]:+.5f}) n={d[3]}")

    print("\n=== W2. home-intercept: scope + walk-forward ===")
    f["phase"] = np.where(f.gn_min < 20, "early", np.where(f.gn_min <= 54, "core", "late"))
    t = f.groupby(["season", "phase"]).resid_us.agg(["size", "mean", "sem"])
    print(t.to_string())
    print(f"  ALL games pooled: mean={f.resid_us.mean():+.3f} sem={f.resid_us.sem():.3f} "
          f"t={f.resid_us.mean()/f.resid_us.sem():+.2f} n={len(f)}")
    print(f"  ALL core pooled : mean={f.resid_us[f.regimeC].mean():+.3f} "
          f"t={f.resid_us[f.regimeC].mean()/f.resid_us[f.regimeC].sem():+.2f}")
    for i in (1, 2):
        tgt = SEASONS[i]
        prior = SEASONS[:i]
        fit_m = (f.regimeC & f.season.isin(prior)).values
        apply_m = (f.regimeC & (f.season == tgt)).values
        c0 = f.resid_us.values[fit_m].mean()
        p_new = 1 / (1 + np.exp(-(f.m_us.values + c0) / SCALE))
        d = paired_delta(f, apply_m, p_new)
        print(f"  core wf -> {tgt}: c0={c0:+.3f} dLL {d[0]:+.5f} ({d[1]:+.5f},{d[2]:+.5f})")
    # global version (all games, not just core)
    for i in (1, 2):
        tgt = SEASONS[i]
        prior = SEASONS[:i]
        fit_m = f.season.isin(prior).values
        apply_m = (f.season == tgt).values
        c0 = f.resid_us.values[fit_m].mean()
        p_new = 1 / (1 + np.exp(-(f.m_us.values + c0) / SCALE))
        d = paired_delta(f, apply_m, p_new)
        print(f"  ALL  wf -> {tgt}: c0={c0:+.3f} dLL {d[0]:+.5f} ({d[1]:+.5f},{d[2]:+.5f})")

    print("\n=== W1+W2 combined (core, walk-forward, c=5) ===")
    for i in (1, 2):
        tgt = SEASONS[i]
        prior = SEASONS[:i]
        c = 5.0 / SCALE
        fit_m = (f.regimeC & f.season.isin(prior) & (np.abs(l) < c)).values
        apply_m = (f.regimeC & (f.season == tgt)).values
        r = sm.GLM(f.y.values[fit_m], sm.add_constant(l[fit_m]),
                   family=sm.families.Binomial()).fit()
        c0 = f.resid_us.values[(f.regimeC & f.season.isin(prior)).values].mean()
        l_adj = np.log(1 / (1 + np.exp(-(f.m_us.values + c0) / SCALE)) /
                       (1 - 1 / (1 + np.exp(-(f.m_us.values + c0) / SCALE))))
        p_new = recal(l_adj, r.params[1], c)
        d = paired_delta(f, apply_m, p_new)
        print(f"  -> {tgt}: dLL {d[0]:+.5f} ({d[1]:+.5f},{d[2]:+.5f})")


if __name__ == "__main__":
    main()
