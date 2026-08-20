"""Regime C construction candidates — final tests.

Candidates (market data only for subset DIAGNOSIS; all constructions market-free):
A. Mid-distribution local recalibration: l' = b*l for |l|<=c (continuous outside),
   b fit on TRAIN regime C, c pre-set at |m_us|=2.5 pts. Motivated by the
   replicated toss-up deficit pocket; D74 only killed UNIFORM recal.
B. Out-count over-penalty margin term: m' = m_us + k*(n_out_home - n_out_away),
   k fit on TRAIN (direction persistent in both splits, train-NS).
C. Form NS-portfolio: momentum_diff + blowout15_diff joint OLS term (train-fit).
D. Regime-local home-edge intercept: m' = m_us + c0, c0 = train mean resid.
Diagnostics: calibration slope in |l|<c on both splits; deficit by OUR confidence;
resid_us means.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

OUTDIR = os.environ.get(
    "RW_OUT",
    "data/scratch",
)
SCALE = 7.2
rng = np.random.default_rng(23)


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


def main():
    f = pd.read_pickle(os.path.join(OUTDIR, "regimec_features.pkl"))
    f["logit_us"] = np.log(f.p_us / (1 - f.p_us))
    train = (f.regimeC & f.season.isin(["2023-24", "2024-25"])).values
    conf = (f.regimeC & (f.season == "2025-26")).values
    l = f.logit_us.values

    print("=== diagnostics ===")
    print(f"mean resid_us: train {f.resid_us[train].mean():+.3f} pts "
          f"(t={f.resid_us[train].mean()/f.resid_us[train].sem():+.2f}), "
          f"confirm {f.resid_us[conf].mean():+.3f} "
          f"(t={f.resid_us[conf].mean()/f.resid_us[conf].sem():+.2f})")
    # deficit by OUR confidence
    f["us_conf"] = (f.p_us - 0.5).abs()
    for label, mask in (("train", train), ("confirm", conf)):
        sub = f[mask].copy()
        sub["bucket"] = pd.cut(sub.us_conf, [0, .1, .2, .3, .5])
        print(f"  d_excess by |p_us-0.5| ({label}):")
        print(sub.groupby("bucket", observed=True).d_excess.agg(["size", "mean"]).to_string())
    # local calibration slope inside |m_us| < 2.5 (|l| < 0.347)
    c = 2.5 / SCALE
    for label, mask in (("train", train), ("confirm", conf)):
        m2 = mask & (np.abs(l) < c)
        r = sm.GLM(f.y.values[m2], sm.add_constant(l[m2]),
                   family=sm.families.Binomial()).fit()
        print(f"  local slope |l|<{c:.3f} {label}: b={r.params[1]:+.2f} "
              f"(se {r.bse[1]:.2f}) a={r.params[0]:+.3f} n={m2.sum()}")
    # global regime-C slope for reference
    for label, mask in (("train", train), ("confirm", conf)):
        r = sm.GLM(f.y.values[mask], sm.add_constant(l[mask]),
                   family=sm.families.Binomial()).fit()
        print(f"  regime-C global slope {label}: b={r.params[1]:+.2f} (se {r.bse[1]:.2f})")

    print("\n=== A. mid-distribution local recal (c=2.5 pts) ===")
    m2 = train & (np.abs(l) < c)
    r = sm.GLM(f.y.values[m2], sm.add_constant(l[m2]), family=sm.families.Binomial()).fit()
    b = r.params[1]
    def recal(lv, b, c):
        out = np.where(np.abs(lv) <= c, b * lv,
                       np.sign(lv) * (b * c + (np.abs(lv) - c)))
        return 1 / (1 + np.exp(-out))
    for bb, tag in ((b, f"b_trainfit={b:.2f}"), (0.5, "b=0.5"), (0.0, "b=0 (flatten)")):
        p_new = recal(l, bb, c)
        dtr = paired_delta(f, train, p_new)
        dcf = paired_delta(f, conf, p_new)
        print(f"  {tag:18s}: train {dtr[0]:+.5f} ({dtr[1]:+.5f},{dtr[2]:+.5f}) | "
              f"confirm {dcf[0]:+.5f} ({dcf[1]:+.5f},{dcf[2]:+.5f})")

    print("\n=== B. out-count over-penalty term ===")
    x = (f.n_out_home - f.n_out_away).values.astype(float)
    k = sm.OLS(f.resid_us.values[train], sm.add_constant(x[train])).fit().params[1]
    p_new = 1 / (1 + np.exp(-(f.m_us.values + k * x) / SCALE))
    dtr = paired_delta(f, train, p_new)
    dcf = paired_delta(f, conf, p_new)
    print(f"  k={k:+.3f} pts/out: train {dtr[0]:+.5f} ({dtr[1]:+.5f},{dtr[2]:+.5f}) | "
          f"confirm {dcf[0]:+.5f} ({dcf[1]:+.5f},{dcf[2]:+.5f})")

    print("\n=== C. form NS-portfolio (momentum + blowout15) ===")
    F = pd.DataFrame({
        "mom": (f.momentum_H - f.momentum_A),
        "blow": (f.blowout15_15_H - f.blowout15_15_A)}).fillna(0.0)
    r = sm.OLS(f.resid_us.values[train], sm.add_constant(F.values[train])).fit()
    adj = F.values @ r.params[1:]
    p_new = 1 / (1 + np.exp(-(f.m_us.values + adj) / SCALE))
    dtr = paired_delta(f, train, p_new)
    dcf = paired_delta(f, conf, p_new)
    print(f"  betas mom={r.params[1]:+.3f} blow={r.params[2]:+.3f}: "
          f"train {dtr[0]:+.5f} ({dtr[1]:+.5f},{dtr[2]:+.5f}) | "
          f"confirm {dcf[0]:+.5f} ({dcf[1]:+.5f},{dcf[2]:+.5f})")

    print("\n=== D. regime-local home intercept ===")
    c0 = f.resid_us[train].mean()
    p_new = 1 / (1 + np.exp(-(f.m_us.values + c0) / SCALE))
    dtr = paired_delta(f, train, p_new)
    dcf = paired_delta(f, conf, p_new)
    print(f"  c0={c0:+.3f} pts: train {dtr[0]:+.5f} ({dtr[1]:+.5f},{dtr[2]:+.5f}) | "
          f"confirm {dcf[0]:+.5f} ({dcf[1]:+.5f},{dcf[2]:+.5f})")

    print("\n=== A' sensitivity: local recal thresholds (diagnosis, not selection) ===")
    for cpts in (1.5, 2.5, 3.5, 5.0):
        cc = cpts / SCALE
        m2 = train & (np.abs(l) < cc)
        r = sm.GLM(f.y.values[m2], sm.add_constant(l[m2]),
                   family=sm.families.Binomial()).fit()
        p_new = recal(l, r.params[1], cc)
        dcf = paired_delta(f, conf, p_new)
        m2c = conf & (np.abs(l) < cc)
        print(f"  c={cpts} pts: b_train={r.params[1]:+.2f} (n={m2.sum()}) "
              f"confirm dLL {dcf[0]:+.5f} ({dcf[1]:+.5f},{dcf[2]:+.5f}) "
              f"[conf games inside: {m2c.sum()}]")


if __name__ == "__main__":
    main()
