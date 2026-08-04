"""Regime C residual mining: decomposition + feature battery + confirmation split.

PRE-REGISTERED PROTOCOL (fixed before looking at 2025-26):
- Universe: capstone games, regime C = min team game-number in [20,54].
- TRAIN (mine): 2023-24 + 2024-25 regime C. CONFIRM: 2025-26 regime C.
- Targets: T1 resid_us = actual_margin - m_us (OLS, construction-primary);
           T2 d_excess = L_us - L_mkt (OLS, KPI relevance);
           T3 logistic y ~ offset(logit p_us) + x.
- Survivor rule: TRAIN T1 two-sided p < 0.05 AND CONFIRM same-sign one-sided
  p < 0.10 on T1. (Stricter p<0.05 tier reported too.) BH-FDR q reported on the
  TRAIN family. Features with no 25-26 coverage (refs, star-out feed) are
  mine-only and CANNOT survive — reported separately.
- For survivors: k fit on TRAIN (OLS slope of resid_us on raw x), margin term
  m' = m_us + k*x, confirm delta logloss on 25-26 regime C with paired
  bootstrap CI (10k resamples).
- Decomposition of dm = m_mkt - m_us into talent / sched / form / outs blocks +
  residual, with "is the channel real" logistic (y ~ offset + block projections).
Market columns (m_mkt, p_mkt, d_excess) are used ONLY as benchmark/diagnosis
targets, never as inputs to any construction candidate.
"""
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

OUTDIR = os.environ.get(
    "RW_OUT",
    "/tmp/claude-1004/-hdd-steveqin-sean-dev/4912ce58-69cd-488d-a5f4-0d3ed2eed30c/scratchpad",
)
SCALE = 7.2
rng = np.random.default_rng(7)


def load():
    f = pd.read_pickle(os.path.join(OUTDIR, "regimec_features.pkl"))
    b = pd.read_pickle(os.path.join(OUTDIR, "regimec_bench.pkl"))
    f = f.merge(b, on="game_id", how="left")
    return f


def build_features(f):
    X = pd.DataFrame(index=f.index)
    d = lambda c: f[c + "_H"] - f[c + "_A"]
    # schedule / rest / travel
    X["rest_diff"] = d("days_rest")
    X["b2b_H"] = f.b2b_H
    X["b2b_A"] = f.b2b_A
    X["threein4_diff"] = d("is_3in4")
    X["games_last7_diff"] = d("games_last7")
    X["travel3d_diff"] = d("travel3d_km")
    X["tz_from_home_A"] = f.tz_from_home_A
    X["tz_change_diff"] = d("tz_change")
    X["road_trip_A"] = f.road_trip_A
    X["home_stand_H"] = f.home_stand_H
    # form
    X["form10_diff"] = d("form10")
    X["momentum_diff"] = d("momentum")
    X["streak_diff"] = d("streak")
    X["sos10_diff"] = d("sos10")
    X["blowout15_diff"] = d("blowout15_15")
    X["wpct_diff"] = d("wpct")
    # style / clash
    X["pace_diff"] = d("td_poss")
    X["pace_clash"] = (f.td_poss_H - f.td_poss_A).abs()
    X["pace_sum"] = f.td_poss_H + f.td_poss_A
    X["ftr_diff"] = d("td_ftr")
    X["ftr_matchup"] = f.td_ftr_H * f.td_opp_ftr_alwd_A - f.td_ftr_A * f.td_opp_ftr_alwd_H
    X["p3rate_diff"] = d("td_p3rate")
    X["p3rate_sum"] = f.td_p3rate_H + f.td_p3rate_A
    X["p3pct_std_diff"] = d("td_p3pct_std")
    X["p3pct_std_sum"] = f.td_p3pct_std_H + f.td_p3pct_std_A
    X["orbp_diff"] = d("td_orbp")
    X["drbp_diff"] = d("td_drbp")
    X["crash_matchup"] = (f.td_orbp_H * (1 - f.td_drbp_A)
                          - f.td_orbp_A * (1 - f.td_drbp_H))
    X["rim_matchup"] = (f.td_rimrate_H * f.td_opp_rim_alwd_A
                        - f.td_rimrate_A * f.td_opp_rim_alwd_H)
    X["astr_diff"] = d("td_astr")
    X["tovr_diff"] = d("td_tovr")
    X["sfl_diff"] = d("td_sfl")
    # h2h
    X["h2h_mean_margin"] = f.h2h_mean_margin.fillna(0.0)
    X["h2h_n"] = f.h2h_n.fillna(0.0)
    X["h2h_days_since"] = f.h2h_days_since
    # continuity
    X["lineup_cont_diff"] = d("lineup_cont")
    X["starter_churn_diff"] = d("starter_churn")
    # availability
    X["n_out_diff"] = f.n_out_home - f.n_out_away
    X["out_dpm_pos_diff"] = f.out_dpm_pos_H - f.out_dpm_pos_A       # feed-covered only
    X["star_out_diff"] = f.star_out_H - f.star_out_A                # feed-covered only
    # refs (23-24/24-25 only)
    X["ref_home_bias"] = f.ref_home_bias
    # asymmetry / regime position
    X["gn_min"] = f.gn_min
    X["gn_diff"] = f.game_no_H - f.game_no_A
    # bench split
    X["bs_bench_rel_diff"] = f.bs_bench_rel_H - f.bs_bench_rel_A
    X["bs_starter_rel_diff"] = f.bs_starter_rel_H - f.bs_starter_rel_A
    X["bs_sb_gap_diff"] = f.bs_sb_gap_H - f.bs_sb_gap_A
    X["bs_bench_share_diff"] = f.bs_bench_share_H - f.bs_bench_share_A
    X["bs_bench_net_abs_diff"] = f.bs_bench_net_abs_H - f.bs_bench_net_abs_A
    return X


MINE_ONLY = {"ref_home_bias", "out_dpm_pos_diff", "star_out_diff"}  # no 25-26 coverage


def ols_t(y, x):
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 30 or np.nanstd(x[m]) == 0:
        return np.nan, np.nan, np.nan, int(m.sum())
    X = sm.add_constant(x[m])
    r = sm.OLS(y[m], X).fit(cov_type="HC1")
    return r.params[1], r.tvalues[1], r.pvalues[1], int(m.sum())


def logit_offset(y, x, off):
    m = np.isfinite(y) & np.isfinite(x) & np.isfinite(off)
    if m.sum() < 30 or np.nanstd(x[m]) == 0:
        return np.nan, np.nan, np.nan
    X = sm.add_constant(x[m])
    try:
        r = sm.GLM(y[m], X, family=sm.families.Binomial(),
                   offset=off[m]).fit()
        return r.params[1], r.tvalues[1], r.pvalues[1]
    except Exception:
        return np.nan, np.nan, np.nan


def bh_q(pvals):
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    ps = p[ok]
    n = len(ps)
    order = np.argsort(ps)
    ranked = ps[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def delta_logloss(f, k, xname, mask, X):
    """Paired delta logloss (control - treated); positive = improvement."""
    sub = f[mask].copy()
    x = X.loc[sub.index, xname].fillna(0.0).values
    m2 = sub.m_us.values + k * x
    p2 = 1 / (1 + np.exp(-m2 / SCALE))
    eps = 1e-12
    L2 = -(sub.y.values * np.log(p2 + eps) + (1 - sub.y.values) * np.log(1 - p2 + eps))
    dl = sub.L_us.values - L2
    boot = []
    n = len(dl)
    for _ in range(10000):
        idx = rng.integers(0, n, n)
        boot.append(dl[idx].mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dl.mean(), lo, hi, n


def main():
    f = load()
    X = build_features(f)
    f["logit_us"] = np.log(f.p_us / (1 - f.p_us))
    train = (f.regimeC & f.season.isin(["2023-24", "2024-25"])).values
    conf = (f.regimeC & (f.season == "2025-26")).values
    print(f"TRAIN n={train.sum()}  CONFIRM n={conf.sum()}")
    print(f"deficit/gm train={f.d_excess[train].mean():+.5f}  confirm={f.d_excess[conf].mean():+.5f}")

    # ---------- PART A: decomposition of dm ----------
    # talent channel: PIT expanding mean of signed dm per team (all games, shifted)
    g = f[["season", "game_id", "game_date", "home", "away", "dm"]].copy()
    long = pd.concat([
        g.assign(team=g.home, s_dm=g.dm),
        g.assign(team=g.away, s_dm=-g.dm)], ignore_index=True)
    long = long.sort_values(["season", "team", "game_date", "game_id"])
    long["tal"] = long.groupby(["season", "team"], sort=False)["s_dm"].transform(
        lambda s: s.shift(1).expanding(5).mean())
    talH = long[long.team == long.home][["game_id", "tal"]].rename(columns={"tal": "tal_H"})
    talA = long[long.team == long.away][["game_id", "tal"]].rename(columns={"tal": "tal_A"})
    f = f.merge(talH, on="game_id", how="left").merge(talA, on="game_id", how="left")
    f["z_talent_raw"] = f.tal_H - f.tal_A

    blocks = {
        "talent": ["z_talent_raw"],
        "sched": ["rest_diff", "b2b_H", "b2b_A", "threein4_diff", "travel3d_diff",
                  "tz_from_home_A", "road_trip_A", "home_stand_H"],
        "form": ["form10_diff", "momentum_diff", "streak_diff", "sos10_diff"],
        "outs": ["n_out_diff"],
    }
    D = X.copy()
    D["z_talent_raw"] = f.z_talent_raw
    Dt = D.loc[train].fillna(D.loc[train].mean())
    dm_t = f.dm[train].values
    proj = {}
    fitted_betas = {}
    for bname, cols in blocks.items():
        Xb = sm.add_constant(Dt[cols].values)
        r = sm.OLS(dm_t, Xb).fit()
        fitted_betas[bname] = (cols, r.params)
        proj[bname] = r.fittedvalues - r.params[0]  # de-mean block projection
    # sequential residual: fit all blocks jointly for variance shares
    all_cols = sum(blocks.values(), [])
    Xall = sm.add_constant(Dt[all_cols].values)
    rall = sm.OLS(dm_t, Xall).fit()
    print("\n=== PART A: decomposition of dm = m_mkt - m_us (TRAIN regime C) ===")
    print(f"var(dm) = {dm_t.var():.3f} pts^2, mean(dm) = {dm_t.mean():+.3f} pts, joint R2 = {rall.rsquared:.3f}")
    for bname in blocks:
        cols = blocks[bname]
        Xb = sm.add_constant(Dt[cols].values)
        rb = sm.OLS(dm_t, Xb).fit()
        print(f"  block {bname:8s}: R2 = {rb.rsquared:.4f}")
    # is each channel real? y ~ offset + block projections (+ residual)
    zres = dm_t - sum(proj.values()) - dm_t.mean()
    Z = np.column_stack([proj[b] for b in blocks] + [zres]) / SCALE
    Zc = sm.add_constant(Z)
    ry = sm.GLM(f.y[train].values, Zc, family=sm.families.Binomial(),
                offset=f.logit_us[train].values).fit()
    names = list(blocks) + ["resid"]
    print("channel-real logistic (coef ~1 = market fully right, 0 = noise):")
    for i, nm in enumerate(names):
        print(f"  {nm:8s}: b = {ry.params[i+1]:+.3f}  (t = {ry.tvalues[i+1]:+.2f})")
    # confirm-season replication of the channel test (blocks projected w/ train betas)
    Dc = D.loc[conf].fillna(D.loc[train].mean())
    dm_c = f.dm[conf].values
    projc = {}
    for bname, (cols, params) in fitted_betas.items():
        projc[bname] = sm.add_constant(Dc[cols].values) @ params - params[0]
    zresc = dm_c - sum(projc.values()) - dm_t.mean()
    Zc2 = sm.add_constant(np.column_stack([projc[b] for b in blocks] + [zresc]) / SCALE)
    ryc = sm.GLM(f.y[conf].values, Zc2, family=sm.families.Binomial(),
                 offset=f.logit_us[conf].values).fit()
    print("confirm 25-26:")
    for i, nm in enumerate(names):
        print(f"  {nm:8s}: b = {ryc.params[i+1]:+.3f}  (t = {ryc.tvalues[i+1]:+.2f})")

    # overall: how much better would we be if we fully absorbed dm? (bench-mark bound)
    for label, mask in (("train", train), ("confirm", conf)):
        print(f"  {label}: mean d_excess = {f.d_excess[mask].mean():+.5f}, "
              f"share market-better = {(f.d_excess[mask] > 0).mean():.3f}")

    # ---------- PART B: feature battery ----------
    print("\n=== PART B: battery (targets: T1 resid_us OLS | T2 d_excess OLS | T3 logit y) ===")
    rows = []
    for c in X.columns:
        x = X[c].values.astype(float)
        cov_note = ""
        tr_mask, cf_mask = train.copy(), conf.copy()
        if c in ("out_dpm_pos_diff", "star_out_diff"):
            covered = (f.outs_feed_covered_H > 0).values
            tr_mask &= covered
            cf_mask &= covered
            cov_note = "feed<=2025-12-21"
        if c == "ref_home_bias":
            cov_note = "no 25-26 coverage"
        mu, sd = np.nanmean(x[tr_mask]), np.nanstd(x[tr_mask])
        if not np.isfinite(sd) or sd == 0:
            continue
        xs = (x - mu) / sd
        b1, t1, p1, n1 = ols_t(f.resid_us.values, xs) if False else ols_t(
            f.resid_us.values[tr_mask], xs[tr_mask])
        b2, t2, p2, _ = ols_t(f.d_excess.values[tr_mask], xs[tr_mask])
        b3, t3, p3 = logit_offset(f.y.values[tr_mask], xs[tr_mask],
                                  f.logit_us.values[tr_mask])
        # dm alignment: does market already price it?
        bd, td, pdv, _ = ols_t(f.dm.values[tr_mask], xs[tr_mask])
        # confirm
        c1, ct1, cp1, nc = ols_t(f.resid_us.values[cf_mask], xs[cf_mask])
        cb3, ct3, cp3 = logit_offset(f.y.values[cf_mask], xs[cf_mask],
                                     f.logit_us.values[cf_mask])
        rows.append(dict(feature=c, n_train=n1, b_resid=b1, t_resid=t1, p_resid=p1,
                         t_dexcess=t2, b_logit=b3, t_logit=t3,
                         b_dm=bd, t_dm=td,
                         n_conf=nc, b_resid_conf=c1, t_resid_conf=ct1,
                         t_logit_conf=ct3, note=cov_note))
    R = pd.DataFrame(rows)
    R["q_train"] = bh_q(R.p_resid.values)
    # one-sided confirm p (same sign as train)
    same = np.sign(R.b_resid) == np.sign(R.b_resid_conf)
    z = R.t_resid_conf.abs()
    R["p_conf_1s"] = np.where(same, 1 - stats.norm.cdf(z), 1.0)
    R["survive_p10"] = (R.p_resid < 0.05) & (R.p_conf_1s < 0.10) & (~R.feature.isin(MINE_ONLY))
    R["survive_p05"] = (R.p_resid < 0.05) & (R.p_conf_1s < 0.05) & (~R.feature.isin(MINE_ONLY))
    R = R.sort_values("p_resid")
    pd.set_option("display.width", 250)
    print(R[["feature", "n_train", "t_resid", "q_train", "t_dexcess", "t_logit", "t_dm",
             "t_resid_conf", "p_conf_1s", "survive_p10", "survive_p05", "note"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    R.to_csv(os.path.join(OUTDIR, "regimec_battery.csv"), index=False)

    # ---------- PART C: construction gates for survivors ----------
    print("\n=== PART C: k-fit + confirm delta logloss (paired, 10k bootstrap) ===")
    cand = R[(R.p_resid < 0.05) & (~R.feature.isin(MINE_ONLY))].feature.tolist()
    for c in cand:
        x = X[c].values.astype(float)
        mu, sd = np.nanmean(x[train]), np.nanstd(x[train])
        xs = (x - mu) / sd
        k, _, _, _ = ols_t(f.resid_us.values[train], xs[train])
        Xtmp = X.copy()
        Xtmp["_xs"] = xs
        dtr, lo_t, hi_t, ntr = delta_logloss(f, k, "_xs", train, Xtmp)
        dcf, lo_c, hi_c, ncf = delta_logloss(f, k, "_xs", conf, Xtmp)
        print(f"  {c:24s} k/sd={k:+.3f}pts  train {dtr:+.5f} ({lo_t:+.5f},{hi_t:+.5f}) "
              f"n={ntr} | confirm {dcf:+.5f} ({lo_c:+.5f},{hi_c:+.5f}) n={ncf}")


if __name__ == "__main__":
    main()
