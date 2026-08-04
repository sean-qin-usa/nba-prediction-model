"""REGIME A analysis — decompose the early-season (gp<20) loss vs market.

Inputs: data/rw_early_decomp_pergame.csv (bitwise-verified component dump),
data/rw_early_signals.csv (PIT preseason/rookie/continuity team signals),
data/rw_early_decomp_refits.json.

Sections:
  S1 loss localization (ll_us - ll_mkt) by gp bucket / phase / season
  S2 disagreement attribution: (am - m_mkt) ~ d_ff + d_cm where
     d_ff = 0.5*fm - 0.5*(m_mkt - sched), d_cm likewise (d_ff+d_cm = m_us-m_mkt)
  S3 component-only counterfactual log-losses per phase
  S4 early margin/prob calibration diagnostics (hindsight, diagnostic only)
  S5 preseason/rookie signal regressions: am ~ m_us + X and am ~ m_mkt + X
  S6 continuity proxy quality: DEFAULT vs first5 vs preseason variants
  S9 rookie-share variance shrink (hindsight grid, diagnostic only)

All outputs printed as JSON; nothing here is a gate — construction candidates
get pre-registered specs separately.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent

import numpy as np
import pandas as pd
import statsmodels.api as sm

SCALE = 7.2


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def ll_vec(y, p):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_ci(d, B=2000, seed=42):
    d = np.asarray(d, float)
    if len(d) == 0:
        return dict(mean=0.0, lo=0.0, hi=0.0, n=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (B, len(d)))
    means = d[idx].mean(axis=1)
    return dict(mean=round(float(d.mean()), 5),
                lo=round(float(np.percentile(means, 2.5)), 5),
                hi=round(float(np.percentile(means, 97.5)), 5), n=int(len(d)))


def ols(yv, X, names):
    Xd = sm.add_constant(np.column_stack(X))
    m = sm.OLS(yv, Xd).fit(cov_type="HC1")
    return {nm: dict(b=round(float(m.params[i + 1]), 4),
                     t=round(float(m.tvalues[i + 1]), 2))
            for i, nm in enumerate(names)} | {
        "const": dict(b=round(float(m.params[0]), 4),
                      t=round(float(m.tvalues[0]), 2)),
        "r2": round(float(m.rsquared), 4), "n": int(len(yv))}


df = pd.read_csv(REPO / "data" / "rw_early_decomp_pergame.csv")
sig = pd.read_csv(REPO / "data" / "rw_early_signals.csv")
refits = json.load(open(REPO / "data" / "rw_early_decomp_refits.json"))

df["m_mkt"] = SCALE * logit(df.p_mkt)
df["ll_us"] = ll_vec(df.y, df.p_us)
df["ll_mkt"] = ll_vec(df.y, df.p_mkt)
df["dgap"] = df.ll_us - df.ll_mkt          # positive = we lose to market
df["gp_min"] = df[["gp_home", "gp_away"]].min(axis=1)
df["week1"] = (df.cm == 0).astype(int)
df["phase"] = np.where(df.week1 == 1, "P1_week1_cm_dead",
               np.where(df.carry_active == 1, "P2_carry_cm_alive",
                        "P3_postcarry_early"))

out = {}

# ---- S1 loss localization ------------------------------------------------
S1 = {"pooled_early": paired_ci(df.dgap)}
for k, g in df.groupby("phase"):
    S1[k] = paired_ci(g.dgap)
for lo, hi in ((0, 3), (3, 5), (5, 10), (10, 20)):
    S1[f"gp_min[{lo},{hi})"] = paired_ci(df[(df.gp_min >= lo) & (df.gp_min < hi)].dgap)
for s, g in df.groupby("season"):
    S1[s] = paired_ci(g.dgap)
for s, g in df.groupby("season"):
    S1[f"{s}_week1"] = paired_ci(g[g.week1 == 1].dgap)
S1["lls"] = {ph: dict(us=round(float(g.ll_us.mean()), 4),
                      mkt=round(float(g.ll_mkt.mean()), 4), n=len(g))
             for ph, g in df.groupby("phase")}
out["S1_loss_localization"] = S1

# ---- S2 disagreement attribution ----------------------------------------
df["d_ff"] = 0.5 * df.fm - 0.5 * (df.m_mkt - df.sched)
df["d_cm"] = 0.5 * df.cm - 0.5 * (df.m_mkt - df.sched)
df["resid_mkt"] = df.am - df.m_mkt
df["resid_us"] = df.am - df.m_us
S2 = {}
w1 = df[df.phase == "P1_week1_cm_dead"]
S2["P1_week1: resid_mkt ~ d_ff"] = ols(w1.resid_mkt, [w1.d_ff], ["d_ff"])
p2 = df[df.phase == "P2_carry_cm_alive"]
S2["P2_carry: resid_mkt ~ d_ff + d_cm"] = ols(
    p2.resid_mkt, [p2.d_ff, p2.d_cm], ["d_ff", "d_cm"])
p3 = df[df.phase == "P3_postcarry_early"]
S2["P3_post: resid_mkt ~ d_ff + d_cm"] = ols(
    p3.resid_mkt, [p3.d_ff, p3.d_cm], ["d_ff", "d_cm"])
S2["all_early: resid_mkt ~ (m_us - m_mkt)"] = ols(
    df.resid_mkt, [df.m_us - df.m_mkt], ["dis_total"])
# our own residual: which component magnitude correlates with our error
S2["P2+P3: resid_us ~ d_ff + d_cm"] = ols(
    df[df.week1 == 0].resid_us,
    [df[df.week1 == 0].d_ff, df[df.week1 == 0].d_cm], ["d_ff", "d_cm"])
out["S2_disagreement_attribution"] = S2

# ---- S3 component-only counterfactual lls -------------------------------
S3 = {}
for ph, g in df.groupby("phase"):
    e = dict(n=len(g), shipped=round(float(g.ll_us.mean()), 4),
             mkt=round(float(g.ll_mkt.mean()), 4))
    p_ff = 1 / (1 + np.exp(-(g.fm + g.sched) / SCALE))
    e["ff_only"] = round(float(ll_vec(g.y, p_ff).mean()), 4)
    if (g.cm != 0).any():
        p_cm = 1 / (1 + np.exp(-(g.cm + g.sched) / SCALE))
        e["cm_only"] = round(float(ll_vec(g.y, p_cm).mean()), 4)
        p_rp = 1 / (1 + np.exp(-(g.rm_core + g.prior_term + g.sched) / SCALE))
        e["ratings_prior_only"] = round(float(ll_vec(g.y, p_rp).mean()), 4)
    S3[ph] = e
out["S3_component_only_lls"] = S3

# ---- S4 calibration (hindsight diagnostics) -----------------------------
S4 = {}
for ph, g in df.groupby("phase"):
    bu = np.polyfit(g.m_us, g.am, 1)
    bm = np.polyfit(g.m_mkt, g.am, 1)
    lr = sm.Logit(g.y, sm.add_constant(logit(g.p_us))).fit(disp=0)
    lrm = sm.Logit(g.y, sm.add_constant(logit(g.p_mkt))).fit(disp=0)
    S4[ph] = dict(margin_slope_us=round(float(bu[0]), 3),
                  margin_slope_mkt=round(float(bm[0]), 3),
                  logit_slope_us=round(float(lr.params[1]), 3),
                  logit_slope_mkt=round(float(lrm.params[1]), 3),
                  sd_m_us=round(float(g.m_us.std()), 2),
                  sd_m_mkt=round(float(g.m_mkt.std()), 2),
                  resid_us_sd=round(float(g.resid_us.std()), 2),
                  resid_mkt_sd=round(float(g.resid_mkt.std()), 2))
out["S4_calibration"] = S4

# ---- S5 signal regressions ----------------------------------------------
skey = sig.set_index(["season", "team"])
sigcols = ["ps_pd", "ps_cont_any", "ps_cont_last2", "ps_cont_wt",
           "ps_top5_ret", "rookie_ps_share", "first5_cont", "open_cont"]
for c in sigcols:
    df[f"{c}_h"] = [skey.loc[(s, t), c] if (s, t) in skey.index else np.nan
                    for s, t in zip(df.season, df.home)]
    df[f"{c}_a"] = [skey.loc[(s, t), c] if (s, t) in skey.index else np.nan
                    for s, t in zip(df.season, df.away)]
    df[f"{c}_d"] = df[f"{c}_h"] - df[f"{c}_a"]
df["rookie_sum"] = df.rookie_ps_share_h + df.rookie_ps_share_a

S5 = {}
for wname, sub in (("carry_window", df[df.carry_active == 1]),
                   ("gp_min<10", df[df.gp_min < 10]),
                   ("all_early", df)):
    e = {}
    for c in ["ps_pd", "ps_cont_any", "ps_cont_last2", "ps_cont_wt",
              "ps_top5_ret", "rookie_ps_share"]:
        x = sub[f"{c}_d"]
        ok = x.notna() & sub.am.notna()
        g = sub[ok]
        e[f"am~m_us+{c}_d"] = ols(g.am, [g.m_us, g[f"{c}_d"]],
                                  ["m_us", f"{c}_d"])
        e[f"am~m_mkt+{c}_d"] = ols(g.am, [g.m_mkt, g[f"{c}_d"]],
                                   ["m_mkt", f"{c}_d"])
    # joint: preseason pd + continuity + rookie beyond our margin
    g = sub[sub[["ps_pd_d", "ps_cont_any_d", "rookie_ps_share_d"]].notna().all(axis=1)]
    e["am~m_us+ps_pd+cont+rookie"] = ols(
        g.am, [g.m_us, g.ps_pd_d, g.ps_cont_any_d, g.rookie_ps_share_d],
        ["m_us", "ps_pd_d", "cont_d", "rookie_d"])
    e["am~m_mkt+ps_pd+cont+rookie"] = ols(
        g.am, [g.m_mkt, g.ps_pd_d, g.ps_cont_any_d, g.rookie_ps_share_d],
        ["m_mkt", "ps_pd_d", "cont_d", "rookie_d"])
    S5[wname] = e
out["S5_signal_regressions"] = S5

# ---- S6 continuity proxy quality ----------------------------------------
S6 = {}
m = sig.dropna(subset=["first5_cont"])
for c in ["ps_cont_any", "ps_cont_last2", "ps_cont_wt", "open_cont"]:
    v = m[c].astype(float)
    S6[c] = dict(corr_vs_first5=round(float(np.corrcoef(v, m.first5_cont)[0, 1]), 3),
                 mae_vs_first5=round(float((v - m.first5_cont).abs().mean()), 4))
S6["DEFAULT_0.5556"] = dict(
    corr_vs_first5=0.0,
    mae_vs_first5=round(float((0.5556 - m.first5_cont).abs().mean()), 4))
r1 = [r for r in refits if r["n_cur_rows"] == 0]
S6["refit1_all_default"] = all(
    all(abs(v - 0.5556) < 1e-3 for v in r["cont"].values()) for r in r1)
S6["first5_cont_spread"] = dict(sd=round(float(m.first5_cont.std()), 3),
                                min=round(float(m.first5_cont.min()), 3),
                                max=round(float(m.first5_cont.max()), 3))
out["S6_continuity_proxies"] = S6

# ---- S9 rookie variance shrink (hindsight grid) -------------------------
S9 = {}
g = df[df.rookie_sum.notna()]
S9["dgap~rookie_sum"] = ols(g.dgap, [g.rookie_sum], ["rookie_sum"])
S9["abs_resid_us~rookie_sum"] = ols(g.resid_us.abs(), [g.rookie_sum], ["rookie_sum"])
S9["abs_resid_mkt~rookie_sum"] = ols(g.resid_mkt.abs(), [g.rookie_sum], ["rookie_sum"])
base = float(g.ll_us.mean())
for c in (0.25, 0.5, 1.0):
    p2v = 1 / (1 + np.exp(-(1 - c * g.rookie_sum) * logit(g.p_us)))
    S9[f"shrink_c={c}"] = dict(
        ll=round(float(ll_vec(g.y, p2v).mean()), 5),
        delta_vs_shipped=round(base - float(ll_vec(g.y, p2v).mean()), 5))
out["S9_rookie_variance"] = S9

print(json.dumps(out, indent=1))
json.dump(out, open(REPO / "data" / "rw_early_analysis.json", "w"), indent=1)
