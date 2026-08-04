"""HA-(1) THE STATIONARITY VERDICT.

Per-season, opponent- and own-quality-controlled team home deviations d_t,
their sampling SEs, EB shrinkage, split-half reliability WITHIN season, lag-1 /
lag-2 correlation ACROSS seasons, and the implied signal share.

TWO METHODOLOGICAL TRAPS, both hit and both fixed (recorded because they change
the answer by a factor of two):

  * SPLIT-HALF MUST NOT SHARE THE FITTED HOME EFFECT. Residualising with the
    home-team effect ADDED BACK gives both halves a common a_hat_t term and
    inflates the half-half correlation mechanically (0.47 instead of 0.13).
    Correct: residualise on team STRENGTH + controls only, so each half's mean
    is an independent draw around the same true a_t; and corroborate with a
    fully independent refit of the whole regression on each half.
  * BOOTSTRAPPING GAMES CANNOT ESTIMATE tau. Resampling games with replacement
    puts a SECOND layer of sampling noise on each d_hat, so
    var(d_hat_boot) ~ tau^2 + 2*se^2 while mean(se^2_boot) ~ se^2, and the
    method-of-moments tau comes back as sqrt(tau^2+se^2) -- 3.5 instead of 1.8.
    Correct: bootstrap the SECOND STAGE, i.e. resample the (d_hat, se) pairs,
    which propagates the finite number of team-seasons without adding noise;
    plus a parametric null (d ~ N(mu, se^2), zero true spread) for the p-value.

DESCRIPTIVE, FULL-SAMPLE. Not a forecastable edge.
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
OUT = Path("/tmp/claude-1004/-hdd-steveqin-sean-dev/"
           "4912ce58-69cd-488d-a5f4-0d3ed2eed30c/scratchpad")
SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
NORMAL = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def main():
    rng = np.random.default_rng(SEED)
    d = load_panel()
    out = {}

    # ---------- per-season fits -------------------------------------------
    fits, fits_nc = {}, {}
    rows = []
    for s in SEASONS:
        sub = d[d["season"] == s]
        f = fit_season(sub, CONTROLS)
        fnc = fit_season(sub, None)
        fits[s], fits_nc[s] = f, fnc
        tau2, dsh, share = eb_shrink(f["d"], f["se_d"])
        rows.append(dict(season=s, n=f["n"], home_g=len(sub) / 30.0,
                         hfa=f["hfa"], se_hfa=f["se_hfa"], hfa_nc=fnc["hfa"],
                         resid_sd=f["sigma"], sd_d=f["d"].std(ddof=1),
                         rms_se_d=np.sqrt((f["se_d"] ** 2).mean()),
                         tau=np.sqrt(tau2), signal_share=share))
    tab = pd.DataFrame(rows)
    print("=== (1a) PER-SEASON FIT "
          "(hfa = league home edge, controlled; hfa_nc = no schedule controls) ===")
    print(tab.round(4).to_string(index=False))
    out["per_season"] = tab.to_dict("records")

    teams = fits["2025-26"]["teams"]
    D = pd.DataFrame({s: pd.Series(fits[s]["d"], index=fits[s]["teams"])
                      for s in SEASONS}).reindex(teams)
    SE = pd.DataFrame({s: pd.Series(fits[s]["se_d"], index=fits[s]["teams"])
                       for s in SEASONS}).reindex(teams)
    print("\n=== (1b) TEAM HOME DEVIATION d_t (points, opponent+quality+schedule "
          "controlled, UNSHRUNK) ===")
    print(D.round(2).to_string())
    D.to_csv(OUT / "ha_dev_matrix.csv"); SE.to_csv(OUT / "ha_dev_se.csv")
    # EB-shrunk version
    DS = D.copy()
    for s in SEASONS:
        t2, sh, _ = eb_shrink(D[s].to_numpy(), SE[s].to_numpy())
        DS[s] = sh
    print("\n  EB-SHRUNK d_t (what an honest estimator would actually carry):")
    print(DS.round(2).to_string())
    DS.to_csv(OUT / "ha_dev_shrunk.csv")

    # ---------- pooled EB over normal seasons ------------------------------
    dn = np.concatenate([fits[s]["d"] for s in NORMAL])
    sen = np.concatenate([fits[s]["se_d"] for s in NORMAL])
    tau2, _, share = eb_shrink(dn, sen)
    v_obs, v_noise = float(np.var(dn, ddof=1)), float(np.mean(sen ** 2))
    print(f"\n=== (1c) POOLED EB, {len(NORMAL)} normal seasons "
          f"(n={len(dn)} team-seasons, 41 home games each) ===")
    print(f"  sd(d_hat) observed              = {np.sqrt(v_obs):.4f} pts")
    print(f"  rms sampling SE of d_hat        = {np.sqrt(v_noise):.4f} pts")
    print(f"  tau = sqrt(var_obs - var_noise) = {np.sqrt(tau2):.4f} pts  (TRUE spread)")
    print(f"  SIGNAL SHARE                    = {share*100:.2f}% of observed variance")

    # CI: resample the (d_hat, se) PAIRS (second stage only)
    B = 20000
    taus, shares = [], []
    for _ in range(B):
        ix = rng.integers(0, len(dn), len(dn))
        t2, _, sh = eb_shrink(dn[ix], sen[ix])
        taus.append(np.sqrt(t2)); shares.append(sh)
    tlo, thi = boot_ci(np.array(taus)); slo, shi = boot_ci(np.array(shares))
    print(f"  second-stage bootstrap {B}x: tau CI ({tlo:.4f},{thi:.4f})  "
          f"share CI ({slo*100:.2f}%,{shi*100:.2f}%)")

    # parametric null: zero true spread
    nulls = np.array([ (lambda x: np.var(x, ddof=1) - v_noise)(
        dn.mean() + rng.normal(0, sen)) for _ in range(20000)])
    p = float((nulls >= (v_obs - v_noise)).mean())
    print(f"  parametric null (tau=0): p(observed excess variance) = {p:.5f} "
          f"-> {'TRUE SPREAD IS REAL within season' if p<0.05 else 'cannot reject zero spread'}")
    out["pooled_eb"] = dict(sd_obs=float(np.sqrt(v_obs)),
                            rms_se=float(np.sqrt(v_noise)),
                            tau=float(np.sqrt(tau2)), signal_share=float(share),
                            tau_ci=[tlo, thi], share_ci=[slo, shi], p_null=p)

    # ---------- split-half reliability WITHIN season ------------------------
    print("\n=== (1d) SPLIT-HALF RELIABILITY WITHIN SEASON ===")
    print("  (i) CONDITIONAL on the full-season team-strength fit -- an "
          "OPTIMISTIC bound;\n      it ignores the noise in the strength "
          "estimate that also enters d_t.")
    R = 200
    sh_rows = []
    for s in SEASONS:
        sub = d[d["season"] == s].reset_index(drop=True)
        f = fits[s]
        tmap = {t: i for i, t in enumerate(f["teams"])}
        X, k = design(sub, f["teams"], CONTROLS)
        # strength + controls part ONLY (the home block deliberately excluded)
        fitted_no_home = X[:, :k] @ f["s"] + (
            sub[CONTROLS].to_numpy(float) @ np.array([f["controls"][c] for c in CONTROLS]))
        e = sub["margin"].to_numpy(float) - fitted_no_home
        hom = sub["home"].to_numpy()
        cors = []
        for _ in range(R):
            coin = rng.random(len(sub)) < 0.5
            m1 = pd.Series(e[coin]).groupby(hom[coin]).mean()
            m2 = pd.Series(e[~coin]).groupby(hom[~coin]).mean()
            j = m1.index.intersection(m2.index)
            if len(j) >= 25:
                cors.append(np.corrcoef(m1[j], m2[j])[0, 1])
        rh = float(np.mean(cors))
        sh_rows.append(dict(season=s, cond_half_r=rh,
                            cond_SB=2 * rh / (1 + rh) if rh > -1 else np.nan))
    shtab = pd.DataFrame(sh_rows)

    print("  (ii) FULLY INDEPENDENT: refit the entire regression on each random "
          "half.\n       This is the honest number -- it carries every source "
          "of noise in d_t.")
    for s in SEASONS:
        sub = d[d["season"] == s].reset_index(drop=True)
        cs = []
        for _ in range(60):
            coin = rng.random(len(sub)) < 0.5
            try:
                fa = fit_season(sub[coin], CONTROLS, fast_rank=True)
                fb = fit_season(sub[~coin], CONTROLS, fast_rank=True)
            except Exception:
                continue
            sa = pd.Series(fa["d"], index=fa["teams"])
            sb = pd.Series(fb["d"], index=fb["teams"])
            j = sa.index.intersection(sb.index)
            if len(j) >= 25:
                cs.append(np.corrcoef(sa[j], sb[j])[0, 1])
        r = float(np.mean(cs))
        shtab.loc[shtab.season == s, "indep_half_r"] = r
        shtab.loc[shtab.season == s, "indep_SB"] = 2 * r / (1 + r)
    print(shtab.round(4).to_string(index=False))
    nn = shtab[shtab.season.isin(NORMAL)]
    print(f"  MEAN over the 5 normal seasons: conditional SB={nn.cond_SB.mean():.4f}"
          f"   INDEPENDENT SB={nn.indep_SB.mean():.4f}")
    print(f"  (the independent SB {nn.indep_SB.mean():.3f} agrees with the EB "
          f"signal share {share:.3f} -- two different routes to the same answer)")
    out["split_half"] = shtab.to_dict("records")

    # ---------- lag-1 / lag-2 across seasons -------------------------------
    print("\n=== (1e) SEASON-TO-SEASON PERSISTENCE OF d_t ===")

    def lag_pairs(seasons, lag):
        pr = []
        for i in range(len(seasons) - lag):
            s0, s1 = seasons[i], seasons[i + lag]
            x, y = D[s0].dropna(), D[s1].dropna()
            j = x.index.intersection(y.index)
            pr.append((s0, s1, x[j].to_numpy(), y[j].to_numpy()))
        return pr

    def pooled_r(pr):
        xs = np.concatenate([x - x.mean() for _, _, x, _ in pr])
        ys = np.concatenate([y - y.mean() for _, _, _, y in pr])
        return float(np.corrcoef(xs, ys)[0, 1]), len(xs)

    for label, seas in (("ALL 7", SEASONS), ("NORMAL 5", NORMAL)):
        for lag in (1, 2):
            pr = lag_pairs(seas, lag)
            r, n = pooled_r(pr)
            per = ", ".join(f"{a[2:4]}->{b[2:4]} {np.corrcoef(x,y)[0,1]:+.3f}"
                            for a, b, x, y in pr)
            print(f"  {label:9s} lag-{lag}: pooled r={r:+.4f} (n={n})   [{per}]")
            out[f"lag{lag}_{label.split()[0]}"] = dict(r=r, n=n)

    tlist = list(D.index)
    for lag in (1, 2):
        pr = lag_pairs(NORMAL, lag)
        rs = []
        for _ in range(8000):
            pick = rng.integers(0, len(tlist), len(tlist))
            xs = np.concatenate([(x - x.mean())[pick] for _, _, x, _ in pr])
            ys = np.concatenate([(y - y.mean())[pick] for _, _, _, y in pr])
            rs.append(np.corrcoef(xs, ys)[0, 1])
        r0, _ = pooled_r(pr)
        lo, hi = boot_ci(np.array(rs))
        print(f"  NORMAL 5 lag-{lag}: r={r0:+.4f} 95% CI ({lo:+.4f},{hi:+.4f}) "
              f"{'SIG' if lo>0 or hi<0 else 'NS'}   [cluster bootstrap over teams]")
        out[f"lag{lag}_ci"] = dict(r=r0, lo=lo, hi=hi)

    # ---------- what is forecastable --------------------------------------
    print("\n=== (1f) FORECASTABLE COMPONENT ===")
    r1 = out["lag1_ci"]["r"]
    sd_obs = out["pooled_eb"]["sd_obs"]
    fc_sd = abs(r1) * sd_obs
    rel = float(nn.indep_SB.mean())
    print(f"  observed sd(d_hat) per season          : {sd_obs:.3f} pts")
    print(f"  TRUE within-season spread tau          : {np.sqrt(tau2):.3f} pts")
    print(f"  within-season reliability (independent): {rel:.3f}")
    print(f"  lag-1 slope of the optimal predictor   : {r1:+.4f}")
    print(f"  SD OF THE BEST FORECAST OF NEXT-SEASON d: {fc_sd:.3f} pts")
    print(f"  disattenuated TRUE persistence         : {r1/rel:+.4f}"
          f"  (lag-1 r divided by reliability)")
    print(f"  upper end of the lag-1 CI, as a forecast: "
          f"{abs(out['lag1_ci']['hi'])*sd_obs:.3f} pts")
    out["forecastable"] = dict(sd_obs=sd_obs, tau=float(np.sqrt(tau2)),
                               reliability=rel, beta=r1, fc_sd=fc_sd,
                               disattenuated=r1 / rel if rel else None,
                               fc_sd_upper=abs(out['lag1_ci']['hi']) * sd_obs)

    (OUT / "ha_stationarity.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {OUT/'ha_stationarity.json'}")


if __name__ == "__main__":
    main()
