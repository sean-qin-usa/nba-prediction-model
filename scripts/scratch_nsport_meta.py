"""SCRATCH (measurement only, no production imports) — meta-analysis of the
REJECTED PILE.

Population: every registered SIDES (win-probability) per-game log-loss gate in
docs/DECISIONS.md / docs/FEATURE_LEDGER.md for a candidate FEATURE that was
TESTED AND NOT SHIPPED.  One row per distinct hypothesis-channel.

Excluded by construction (documented, not silently dropped):
  * props CRPS/MAE gates (different units/endpoint)
  * possession-level log loss (D108/D113, different unit)
  * betting ROI / CLV (different endpoint)
  * ORACLE / CEILING bounds (D97 talent oracle, D72 tracking oracle, D112
    hindsight-k ceilings) — these are upper bounds, not candidate features
  * DATA/corpus fixes measured as deltas (D102 RT1, D104, D112 floor
    relaxation) — not features
  * ablation deltas of ALREADY-SHIPPED terms (D110/D112 transfer battery)
  * composites/bundles (D64 joint, D71 windowed, D136 ABCD) — their members
    are already rows; a bundle is not an independent hypothesis
  * ablation CONTROLS that exist only to interpret a primary arm
    (D127 S2/S3, D133 ARM A0/B)

est/lo/hi are pooled per-game log-loss deltas, + = candidate better.
SE = (hi-lo)/(2*1.96) when a CI is registered.
`first` = the estimate that put the hypothesis in the ledger.
`latest` = the most-powered later re-test of the SAME channel, if any.
`indep` = independence group id (rows sharing an id are re-tests/variants of
the same underlying channel and count ONCE in the primary meta-analysis).
"""
import json
import math
import numpy as np

# name, group, first_est, first_lo, first_hi, first_n, latest_est, latest_lo,
# latest_hi, latest_n, source
ROWS = [
    # ---- the NS-PORTFOLIO six ----
    dict(name="3P-luck defense-only", grp="ffluck", first=(+0.00041, None, None, 3690),
         latest=(+0.00026, -0.00035, +0.00085, 3690), first_se=0.00042,
         src="GATE_POLICY_V2 s2 / D64 same-run decomposition"),
    dict(name="event-recency window blend (F2)", grp="evrec", first=(+0.00138, None, None, 3690),
         latest=(+0.00002, -0.00152, +0.00166, 6148), first_se=0.00112,
         src="GATE_POLICY_V2 s2 -> D102 RT3 -> D124"),
    dict(name="comp-heavy 60/40 blend", grp="blend", first=(+0.00100, None, None, 3690),
         latest=(+0.00041, -0.00050, +0.00128, 3690), first_se=0.00051,
         src="D46-era re-gates / D64 same-run"),
    dict(name="dead-team FE at predict time", grp="dead", first=(+0.00038, None, None, 3690),
         latest=(+0.00054, -0.00223, +0.00333, 3690), first_se=0.00128,
         src="D47 / capstone_pergame_dead.csv / D64 same-run"),
    dict(name="continuity prior", grp="contin", first=(+0.00020, None, None, None),
         latest=None, first_se=0.00040, src="GATE_POLICY_V2 s2"),
    # carry ship-confirm is SHIPPED (D62/D63) -> excluded from the rejected pile
    # ---- other registered sides candidates, tested & not shipped ----
    dict(name="late-gated form term (F1)", grp="form", first=(+0.00178, +0.00076, +0.00275, 3690),
         latest=(-0.00012, -0.00111, +0.00085, 4910), first_se=None,
         src="D71 isolation -> D102 RT2 on top of D90"),
    dict(name="fitted FF/comp blend weight (D22)", grp="blend", first=None,
         latest=(+0.00077, -0.00036, +0.00187, 6148), first_se=None,
         src="D22 -> D102 RT4 (same channel as comp-heavy 60/40)"),
    dict(name="team-specific home advantage (D20 restoration)", grp="teamhome",
         first=(-0.00002, None, None, 6148), latest=None, first_se=None,
         src="D70 (pooled -0.00002 exact null; no CI registered)"),
    dict(name="altitude w/ physio prior", grp="altitude", first=(-0.00002, -0.00041, +0.00037, 6148),
         latest=None, first_se=None, src="D96"),
    dict(name="walk-forward (a,b) calibration layer (T1)", grp="calib",
         first=(+0.00010, -0.00130, +0.00150, 3690), latest=None, first_se=None, src="D74"),
    dict(name="3-axis gated variance inflation", grp="calib2",
         first=(+0.00017, -0.00062, +0.00100, 4920), latest=None, first_se=None, src="D112 W49"),
    dict(name="trade-arrival efficiency adjustment", grp="arrival",
         first=(-0.00006, None, None, None), latest=None, first_se=None,
         src="D80 (CI spans 0; width not registered)"),
    dict(name="late-season urgency differential", grp="urgency",
         first=(+0.00028, -0.00025, +0.00079, None),
         latest=(+0.00048, None, None, 3690), first_se=None,
         src="D80 F5 -> D130 ARM A (pooled veto; primary late window +0.00115)"),
    dict(name="talent ensemble (DARKO+EPM+BPM)", grp="ensemble",
         first=(-0.00006, -0.00144, +0.00131, 3690),
         latest=(+0.00021, -0.00106, +0.00142, 3690), first_se=None,
         src="D86-ENSEMBLE -> D94 second look (daily EPM)"),
    dict(name="rookie draft-slot prior", grp="rookie", first=(-0.00222, -0.00636, +0.00182, None),
         latest=None, first_se=None, src="D85 (active window)"),
    dict(name="v3 M1 team-DLM inside the blend", grp="dlm", first=(-0.00210, None, None, None),
         latest=None, first_se=None, src="D92 (no CI registered for the blend arm)"),
    dict(name="defence-conditioned possession margin (S1)", grp="possdef",
         first=(-0.00108, -0.00278, +0.00062, 3690), latest=None, first_se=None, src="D127"),
    dict(name="clinched/locked-seed letdown", grp="clinch",
         first=(-0.00388, -0.00708, -0.00060, 672), latest=None, first_se=None, src="D130 ARM B"),
    dict(name="quit x urgent interaction", grp="quiturg",
         first=(-0.00026, -0.00183, +0.00121, 1006), latest=None, first_se=None, src="D130 ARM C"),
    dict(name="COVID crowd-regime fit guard", grp="covid",
         first=(+0.000079, -0.000417, +0.000569, 6148), latest=None, first_se=None, src="D131"),
    dict(name="travel fatigue (great-circle km)", grp="travel",
         first=(+0.00011, -0.00028, +0.00050, 3690), latest=None, first_se=None, src="D136 ARM A"),
    dict(name="circadian / signed tz crossings", grp="circad",
         first=(-0.00009, -0.00068, +0.00054, 3690), latest=None, first_se=None, src="D136 ARM B"),
    dict(name="road-trip state", grp="roadtrip",
         first=(-0.00038, -0.00116, +0.00034, 3690), latest=None, first_se=None, src="D136 ARM C"),
    dict(name="schedule density (3in4 / 5in7)", grp="density",
         first=(-0.00020, -0.00088, +0.00043, 3690), latest=None, first_se=None, src="D136 ARM D"),
    dict(name="recency-weighted ratings (60d + rescaled ridge)", grp="recency",
         first=(+0.00160, -0.00100, +0.00400, None), latest=None, first_se=None,
         src="FEATURE_LEDGER retest row (0.5989 vs 0.6005)"),
    dict(name="team closing-ability residual (ahead-after-Q3)", grp="closing",
         first=(+0.00011, -0.00015, +0.00036, 7005), latest=None, first_se=None,
         src="D135 C-B (conditional universe, log loss)"),
]


def se_of(est, lo, hi, fallback=None):
    if lo is not None and hi is not None:
        return (hi - lo) / (2 * 1.959964)
    return fallback


def build(which="latest"):
    """One row per independence group. which='first' uses the ledgering
    estimate; which='latest' uses the most-powered re-test where one exists."""
    seen, out = {}, []
    for r in ROWS:
        pick = None
        if which == "latest" and r["latest"] is not None:
            pick = r["latest"]
        elif r["first"] is not None:
            pick = r["first"]
        elif r["latest"] is not None:
            pick = r["latest"]
        if pick is None:
            continue
        est, lo, hi, n = pick
        se = se_of(est, lo, hi, r["first_se"] if pick is r["first"] else None)
        era, seas = ERA.get((r["grp"], which if pick is r.get("latest") else "first"),
                            ERA.get((r["grp"], "first"), ("?", "?")))
        rec = dict(name=r["name"], grp=r["grp"], est=est, se=se, n=n, src=r["src"],
                   era=era, seasons=seas)
        # one row per group: keep the most-powered (smallest SE, else largest n)
        k = r["grp"]
        if k in seen:
            prev = seen[k]
            better = (se is not None and (prev["se"] is None or se < prev["se"]))
            if not better:
                continue
        seen[k] = rec
    return list(seen.values())


# ---- era tagging (coordinator addition) -------------------------------------
# "post" = measured on 2023-24..2025-26 only (the post-PPP / in-season-tournament
#          / new-CBA-apron era, which is exactly our DEV window)
# "both" = the measurement universe spans the 2022-23 / 2023-24 boundary
# Seasons listed are the SCORED universe of the cited estimate.
ERA = {
    # first-estimate era (D64-era gates were all dev-only)
    ("ffluck", "first"): ("post", "2023-26"),
    ("evrec", "first"): ("post", "2023-26"),
    ("blend", "first"): ("post", "2023-26"),
    ("dead", "first"): ("post", "2023-26"),
    ("contin", "first"): ("post", "2023-26"),
    ("form", "first"): ("post", "2023-26"),
    ("teamhome", "first"): ("both", "2021-26"),
    ("altitude", "first"): ("both", "2021-26"),
    ("calib", "first"): ("post", "2023-26"),
    ("calib2", "first"): ("both", "2022-26 (n=4920)"),
    ("arrival", "first"): ("post", "2023-26"),
    ("urgency", "first"): ("post", "2023-26"),
    ("ensemble", "first"): ("post", "2023-26"),
    ("rookie", "first"): ("post", "2023-26"),
    ("dlm", "first"): ("post", "2023-26"),
    ("possdef", "first"): ("post", "2023-26"),
    ("clinch", "first"): ("post", "2023-26"),
    ("quiturg", "first"): ("post", "2023-26"),
    ("covid", "first"): ("both", "2021-26"),
    ("travel", "first"): ("post", "2023-26"),
    ("circad", "first"): ("post", "2023-26"),
    ("roadtrip", "first"): ("post", "2023-26"),
    ("density", "first"): ("post", "2023-26"),
    ("recency", "first"): ("post", "2023-26"),
    ("closing", "first"): ("both", "2019-26 conditional"),
    # latest-estimate era
    ("ffluck", "latest"): ("post", "2023-26"),
    ("evrec", "latest"): ("both", "2021-26"),
    ("blend", "latest"): ("both", "2021-26"),
    ("dead", "latest"): ("post", "2023-26"),
    ("form", "latest"): ("both", "2022-26 (n=4910)"),
    ("urgency", "latest"): ("post", "2023-26"),
    ("ensemble", "latest"): ("post", "2023-26"),
}


def sign_test(x):
    pos = sum(1 for v in x if v > 0)
    n = len(x)
    # two-sided exact binomial at p=0.5
    from math import comb
    k = min(pos, n - pos)
    p = 2 * sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return pos, n, min(p, 1.0)


def report(rows, label):
    est = np.array([r["est"] for r in rows], float)
    have = [r for r in rows if r["se"] is not None]
    e = np.array([r["est"] for r in have], float)
    s = np.array([r["se"] for r in have], float)
    w = 1.0 / s ** 2
    fe = float((w * e).sum() / w.sum())
    fe_se = float(math.sqrt(1.0 / w.sum()))
    # DerSimonian-Laird random effects
    Q = float((w * (e - fe) ** 2).sum())
    dfree = len(e) - 1
    C = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max(0.0, (Q - dfree) / C) if C > 0 else 0.0
    wr = 1.0 / (s ** 2 + tau2)
    re = float((wr * e).sum() / wr.sum())
    re_se = float(math.sqrt(1.0 / wr.sum()))
    # Egger-style precision regression: est ~ a + b*se
    A = np.c_[np.ones(len(s)), s]
    coef, *_ = np.linalg.lstsq(A, e, rcond=None)
    resid = e - A @ coef
    dof = len(e) - 2
    sig2 = float(resid @ resid / dof)
    cov = sig2 * np.linalg.inv(A.T @ A)
    pos, n, p_sign = sign_test(est)
    z = e / s
    # --- excess positive mass -------------------------------------------------
    # Under a GLOBAL NULL (every true effect 0) with the registered SEs, each
    # estimate ~ N(0, SE) and E[X * 1{X>0}] = SE/sqrt(2*pi) = 0.39894*SE.
    exp_pos = float(0.3989422804 * s.sum())
    obs_pos = float(e[e > 0].sum())
    obs_neg = float(e[e < 0].sum())
    # --- era split (coordinator addition) ------------------------------------
    eras = {}
    for grp in ("post", "both"):
        sub = [r for r in rows if r.get("era") == grp and r["se"] is not None]
        if not sub:
            continue
        se_ = np.array([r["se"] for r in sub], float)
        ee_ = np.array([r["est"] for r in sub], float)
        ww_ = 1.0 / se_ ** 2
        eras[grp] = dict(k=len(sub), mean=float(ee_.mean()),
                         fixed_effect=float((ww_ * ee_).sum() / ww_.sum()),
                         fixed_effect_se=float(math.sqrt(1.0 / ww_.sum())),
                         n_pos=int((ee_ > 0).sum()))
    out = dict(
        label=label, k_total=len(rows), k_with_se=len(have),
        mean_unweighted=float(est.mean()),
        sd_unweighted=float(est.std(ddof=1)),
        se_of_mean=float(est.std(ddof=1) / math.sqrt(len(est))),
        median=float(np.median(est)),
        sum_of_estimates=float(est.sum()),
        sum_positive_only=float(est[est > 0].sum()),
        fixed_effect=fe, fixed_effect_se=fe_se, fixed_effect_z=fe / fe_se,
        Q=Q, df=dfree, I2=float(max(0.0, (Q - dfree) / Q)) if Q > 0 else 0.0,
        tau2=tau2, tau=math.sqrt(tau2),
        random_effect=re, random_effect_se=re_se, random_effect_z=re / re_se,
        egger_intercept=float(coef[0]), egger_intercept_se=float(math.sqrt(cov[0, 0])),
        egger_slope=float(coef[1]), egger_slope_se=float(math.sqrt(cov[1, 1])),
        sign_pos=pos, sign_n=n, sign_p=p_sign,
        z_mean=float(z.mean()), z_sd=float(z.std(ddof=1)),
        z_abs_gt_196=int((np.abs(z) > 1.96).sum()),
        expected_positive_mass_under_null=exp_pos,
        observed_positive_mass=obs_pos,
        excess_positive_mass=obs_pos - exp_pos,
        observed_negative_mass=obs_neg,
        net_asymmetry_obs=obs_pos + obs_neg,
        sum_se=float(s.sum()),
        by_era=eras,
    )
    return out


# --- coordinator item (1)/(4): between- vs within-era variance of per-season
# estimates.  ERAS: pre-PPP = 2021-22, 2022-23; post-PPP = 2023-24..2025-26.
PRE, POST = ("2021-22", "2022-23"), ("2023-24", "2024-25", "2025-26")
SEASONS5 = PRE + POST


def era_anova(per_season, label=""):
    """per_season = {season: estimate}. Returns the one-way between/within
    decomposition treating era as the factor."""
    vals = np.array([per_season[s] for s in SEASONS5], float)
    grand = float(vals.mean())
    groups = [np.array([per_season[s] for s in PRE], float),
              np.array([per_season[s] for s in POST], float)]
    ss_b = float(sum(len(g) * (g.mean() - grand) ** 2 for g in groups))
    ss_w = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    df_b, df_w = len(groups) - 1, len(vals) - len(groups)
    ms_b, ms_w = ss_b / df_b, ss_w / df_w
    F = ms_b / ms_w if ms_w > 0 else float("inf")
    from scipy import stats as _st
    p = float(1 - _st.f.cdf(F, df_b, df_w))
    return dict(label=label, per_season=per_season,
                pre_mean=float(groups[0].mean()), post_mean=float(groups[1].mean()),
                grand=grand, SS_between=ss_b, SS_within=ss_w,
                MS_between=ms_b, MS_within=ms_w, F=F, p=p,
                between_share_of_SS=ss_b / (ss_b + ss_w) if (ss_b + ss_w) > 0 else 0.0)


if __name__ == "__main__":
    res = {}
    for which in ("first", "latest"):
        rows = build(which)
        rows.sort(key=lambda r: r["est"])
        res[which] = dict(summary=report(rows, which), rows=rows)
        print(f"\n===== {which.upper()} REGISTERED ESTIMATE PER CHANNEL "
              f"(k={len(rows)}) =====")
        for r in rows:
            se = f"{r['se']:.6f}" if r["se"] is not None else "   n/a  "
            zz = f"{r['est']/r['se']:+.2f}" if r["se"] else "  n/a"
            print(f"  {r['est']:+.6f}  se={se}  z={zz}  {r['name']}")
        s = res[which]["summary"]
        for k, v in s.items():
            if isinstance(v, float):
                print(f"  {k:24s} {v:+.6f}")
            else:
                print(f"  {k:24s} {v}")
    # coordinator item (4): is D124's F2 per-season profile an ERA story?
    print("\n===== ERA DECOMPOSITION OF REGISTERED PER-SEASON PROFILES =====")
    profiles = {
        "D124 F2 event-recency": dict(zip(SEASONS5,
            [-0.00061, -0.00113, -0.00201, +0.00292, +0.00092])),
        "D102 RT4 fitted blend weight": dict(zip(SEASONS5,
            [+0.00106, +0.00163, +0.00223, +0.00012, -0.00120])),
        "D112 corpus-floor relaxation (data change, control)": dict(zip(SEASONS5,
            [+0.00221, +0.00104, -0.00080, +0.00021, +0.00103])),
    }
    res["era_anova"] = {}
    for k, v in profiles.items():
        a = era_anova(v, k)
        res["era_anova"][k] = a
        print(f"  {k}")
        print(f"    per-season {[f'{v[s]:+.5f}' for s in SEASONS5]}")
        print(f"    pre-PPP mean {a['pre_mean']:+.5f}  post-PPP mean {a['post_mean']:+.5f}")
        print(f"    SS_between {a['SS_between']:.3e}  SS_within {a['SS_within']:.3e}"
              f"  between share {a['between_share_of_SS']*100:.1f}%")
        print(f"    F(1,3) = {a['F']:.3f}  p = {a['p']:.3f}")

    json.dump(res, open("data/nsport_meta.json", "w"), indent=1)
    print("\nwrote data/nsport_meta.json")
