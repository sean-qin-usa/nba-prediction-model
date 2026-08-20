"""HA-(1g) IS THE WITHIN-SEASON "TRUE SPREAD" tau ACTUALLY HOME ADVANTAGE?

Deliverable (1)'s method-of-moments says ~26% of the observed spread in team
home deviations is real (tau ~1.8 pts) even though it does not persist across
seasons. Before that is called "real home advantage", two alternative
explanations have to be killed:

  (A) ANALYTIC SEs WRONG. If the homoskedastic SE understates the true sampling
      noise, tau is spurious by construction. Checked with a residual-bootstrap
      NULL: simulate seasons in which every team has EXACTLY the league home
      edge and no team-specific deviation, keeping the real schedule, the real
      team strengths and the real residual distribution. Under that null,
      how big is sd(d_hat)?

  (B) WITHIN-SEASON FORM x SCHEDULE TIMING. A season-constant team fixed effect
      cannot tell "this team is better at home" from "this team's home games
      happened to fall in its healthy stretch". Checked by letting team
      strength drift: team x half-season, and team x calendar-month fixed
      effects. If tau collapses, the within-season spread is form timing, not
      venue.

DESCRIPTIVE, FULL-SAMPLE.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ha_core import CONTROLS, boot_ci, design, eb_shrink, fit_season, load_panel

SEED = 20260801
OUT = Path("data/scratch")
SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
NORMAL = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def fit_drift(sub, split_col=None):
    """Team FE allowed to vary by `split_col` within the season; one home
    effect per team for the whole season."""
    teams = sorted(set(sub.home) | set(sub.away))
    ti = {t: i for i, t in enumerate(teams)}
    if split_col is None:
        keys = pd.Series(["_"] * len(sub), index=sub.index)
    else:
        keys = sub[split_col].astype(str)
    ks = sorted(keys.unique())
    ts = [(t, k) for k in ks for t in teams]
    tsi = {k: i for i, k in enumerate(ts)}
    n = len(sub); r = np.arange(n)
    Z = np.zeros((n, len(ts)))
    Z[r, [tsi[(h, k)] for h, k in zip(sub.home, keys)]] += 1
    Z[r, [tsi[(a, k)] for a, k in zip(sub.away, keys)]] -= 1
    H = np.zeros((n, len(teams)))
    H[r, sub.home.map(ti).to_numpy()] = 1
    C = sub[CONTROLS].to_numpy(float)
    X = np.hstack([Z, H, C])
    y = sub.margin.to_numpy(float)
    cf, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ cf
    rank = np.linalg.matrix_rank(X)
    s2 = res @ res / (n - rank)
    cov = s2 * np.linalg.pinv(X.T @ X)
    a = cf[len(ts):len(ts) + len(teams)]
    M = np.eye(len(teams)) - np.ones((len(teams), len(teams))) / len(teams)
    d = M @ a
    se = np.sqrt(np.clip(np.diag(M @ cov[len(ts):len(ts) + len(teams),
                                          len(ts):len(ts) + len(teams)] @ M.T), 0, None))
    return pd.Series(d, index=teams), pd.Series(se, index=teams)


def main():
    rng = np.random.default_rng(SEED)
    d = load_panel()
    d["half"] = np.where(d.mean_gp < 41, "A", "B")
    d["third"] = pd.cut(d.mean_gp, [-1, 27, 54, 200], labels=["1", "2", "3"])
    res = {}

    # ---------- (A) residual-bootstrap null --------------------------------
    print("=== (1g-A) NULL DISTRIBUTION OF sd(d_hat) WITH NO TEAM-SPECIFIC "
          "HOME ADVANTAGE ===")
    print("  (real schedule, real strengths, real residuals; only the "
          "team-specific home deviation is set to ZERO)")
    rows = []
    for s in SEASONS:
        sub = d[d.season == s].reset_index(drop=True)
        f = fit_season(sub, CONTROLS)
        teams = f["teams"]
        X, k = design(sub, teams, CONTROLS)
        cf = np.concatenate([f["s"], f["a"], [f["controls"][c] for c in CONTROLS]])
        a_flat = np.full(k, f["a"].mean())          # every team = league HFA
        cf0 = np.concatenate([f["s"], a_flat, [f["controls"][c] for c in CONTROLS]])
        mu0 = X @ cf0
        resid = sub.margin.to_numpy(float) - X @ cf
        sds = []
        for _ in range(300):
            ysim = mu0 + resid[rng.integers(0, len(resid), len(resid))]
            s2 = sub.copy(); s2["margin"] = ysim
            fs = fit_season(s2, CONTROLS, fast_rank=True)
            sds.append(fs["d"].std(ddof=1))
        sds = np.array(sds)
        obs = f["d"].std(ddof=1)
        p = float((sds >= obs).mean())
        rows.append(dict(season=s, obs_sd_d=obs, null_sd_d=float(sds.mean()),
                         null_p5=float(np.percentile(sds, 5)),
                         null_p95=float(np.percentile(sds, 95)),
                         analytic_rms_se=float(np.sqrt((f["se_d"] ** 2).mean())),
                         p_value=p))
    t = pd.DataFrame(rows)
    print(t.round(4).to_string(index=False))
    print("  (null_sd_d should match analytic_rms_se if the SEs are honest)")
    # pooled Fisher-style combination over the 5 normal seasons
    tn = t[t.season.isin(NORMAL)]
    print(f"  NORMAL 5: mean observed sd {tn.obs_sd_d.mean():.4f} vs mean null "
          f"{tn.null_sd_d.mean():.4f}; per-season p-values "
          f"{[round(v,3) for v in tn.p_value]}")
    res["null_check"] = t.to_dict("records")

    # ---------- (B) does tau survive within-season strength drift? ---------
    print("\n=== (1g-B) DOES THE WITHIN-SEASON SPREAD SURVIVE LETTING TEAM "
          "STRENGTH DRIFT? ===")
    out_rows = []
    for label, col in (("season-constant team FE", None),
                       ("team x HALF-season FE", "half"),
                       ("team x THIRD-of-season FE", "third"),
                       ("team x MONTH FE", "month")):
        ds, ses = [], []
        per = []
        for s in NORMAL:
            sub = d[d.season == s].reset_index(drop=True)
            dd, se = fit_drift(sub, col)
            ds.append(dd.to_numpy()); ses.append(se.to_numpy())
            t2, _, sh = eb_shrink(dd.to_numpy(), se.to_numpy())
            per.append(sh)
        ds = np.concatenate(ds); ses = np.concatenate(ses)
        tau2, _, share = eb_shrink(ds, ses)
        out_rows.append(dict(spec=label, sd_d=ds.std(ddof=1),
                             rms_se=np.sqrt((ses ** 2).mean()),
                             tau=np.sqrt(tau2), signal_share=share))
        print(f"  {label:28s} sd(d)={ds.std(ddof=1):.4f} rms_se="
              f"{np.sqrt((ses**2).mean()):.4f} tau={np.sqrt(tau2):.4f} "
              f"share={share*100:5.2f}%   per-season "
              f"{[round(v*100,1) for v in per]}")
    res["drift"] = out_rows

    # ---------- (C) does the lag-1 improve under any of these? -------------
    print("\n=== (1g-C) LAG-1 PERSISTENCE UNDER EACH SPECIFICATION ===")
    for label, col in (("season-constant team FE", None),
                       ("team x HALF-season FE", "half"),
                       ("team x MONTH FE", "month")):
        Dm = {}
        for s in SEASONS:
            sub = d[d.season == s].reset_index(drop=True)
            dd, _ = fit_drift(sub, col)
            Dm[s] = dd
        Dm = pd.DataFrame(Dm)
        xs, ys = [], []
        for a, b in zip(NORMAL, NORMAL[1:]):
            x, y = Dm[a], Dm[b]
            xs.append((x - x.mean()).to_numpy()); ys.append((y - y.mean()).to_numpy())
        r = float(np.corrcoef(np.concatenate(xs), np.concatenate(ys))[0, 1])
        print(f"  {label:28s} lag-1 pooled r = {r:+.4f}")
        res[f"lag1_{label}"] = r

    (OUT / "ha_tau_check.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {OUT/'ha_tau_check.json'}")


if __name__ == "__main__":
    main()
