"""Regime C follow-up: (1) market-free own-residual talent corrector (the
construction implied by Part A's talent channel), (2) block-level joint tests,
(3) bench-split reliability autopsy, (4) outs joint test, (5) deficit profile.

Same split protocol: mine/fit on 23-24+24-25 regime C, confirm on 25-26 regime C.
Own-residual corrector is walk-forward: for each (season, team), the expanding
mean of OUR OWN margin residual (actual - m_us) over strictly prior games
(all games, not just regime C), shrunk by n/(n+K). No market data enters it.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

OUTDIR = os.environ.get(
    "RW_OUT",
    "data/scratch",
)
SCALE = 7.2
rng = np.random.default_rng(11)


def paired_delta(f, mask, m_new):
    sub = f[mask]
    p2 = 1 / (1 + np.exp(-m_new[mask] / SCALE))
    eps = 1e-12
    L2 = -(sub.y.values * np.log(p2 + eps) + (1 - sub.y.values) * np.log(1 - p2 + eps))
    dl = sub.L_us.values - L2
    n = len(dl)
    boot = [dl[rng.integers(0, n, n)].mean() for _ in range(10000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dl.mean(), lo, hi, n


def main():
    f = pd.read_pickle(os.path.join(OUTDIR, "regimec_features.pkl"))
    b = pd.read_pickle(os.path.join(OUTDIR, "regimec_bench.pkl"))
    f = f.merge(b, on="game_id", how="left")
    f["logit_us"] = np.log(f.p_us / (1 - f.p_us))
    train = (f.regimeC & f.season.isin(["2023-24", "2024-25"])).values
    conf = (f.regimeC & (f.season == "2025-26")).values

    # ---------- 1. own-residual talent corrector ----------
    g = f[["season", "game_id", "game_date", "home", "away", "resid_us"]].copy()
    long = pd.concat([
        g.assign(team=g.home, r=g.resid_us),
        g.assign(team=g.away, r=-g.resid_us)], ignore_index=True)
    long = long.sort_values(["season", "team", "game_date", "game_id"])
    grp = long.groupby(["season", "team"], sort=False)["r"]
    long["own_mean"] = grp.transform(lambda s: s.shift(1).expanding(1).mean())
    long["own_n"] = grp.transform(lambda s: s.shift(1).expanding(1).count())
    print("=== 1. own-residual talent corrector (market-free) ===")
    for K in (10, 20, 40):
        long["own_shrunk"] = long.own_mean * long.own_n / (long.own_n + K)
        oh = long[long.team == long.home][["game_id", "own_shrunk"]].rename(
            columns={"own_shrunk": "own_H"})
        oa = long[long.team == long.away][["game_id", "own_shrunk"]].rename(
            columns={"own_shrunk": "own_A"})
        t = f.merge(oh, on="game_id", how="left").merge(oa, on="game_id", how="left")
        x = (t.own_H - t.own_A).fillna(0.0).values
        for label, mask in (("train", train), ("confirm", conf)):
            X = sm.add_constant(x[mask])
            r = sm.OLS(f.resid_us.values[mask], X).fit(cov_type="HC1")
            print(f"  K={K:3d} {label:8s}: slope={r.params[1]:+.3f} t={r.tvalues[1]:+.2f}")
        # gate: k fit on train, delta logloss both windows
        Xt = sm.add_constant(x[train])
        k = sm.OLS(f.resid_us.values[train], Xt).fit().params[1]
        m_new = f.m_us.values + k * x
        dtr = paired_delta(f, train, m_new)
        dcf = paired_delta(f, conf, m_new)
        print(f"      gate k={k:+.3f}: train {dtr[0]:+.5f} ({dtr[1]:+.5f},{dtr[2]:+.5f}) | "
              f"confirm {dcf[0]:+.5f} ({dcf[1]:+.5f},{dcf[2]:+.5f}) n={dcf[3]}")

    # also: does own-residual align with the market's talent view? (diagnosis only)
    long["own_shrunk"] = long.own_mean * long.own_n / (long.own_n + 20)
    oh = long[long.team == long.home][["game_id", "own_shrunk"]].rename(columns={"own_shrunk": "own_H"})
    oa = long[long.team == long.away][["game_id", "own_shrunk"]].rename(columns={"own_shrunk": "own_A"})
    t = f.merge(oh, on="game_id", how="left").merge(oa, on="game_id", how="left")
    xo = (t.own_H - t.own_A).fillna(0).values
    for label, mask in (("train", train), ("confirm", conf)):
        r = sm.OLS(f.dm.values[mask], sm.add_constant(xo[mask])).fit(cov_type="HC1")
        print(f"  dm ~ own_resid {label}: slope={r.params[1]:+.3f} t={r.tvalues[1]:+.2f} "
              f"(market already prices our persistent misses?)")

    # ---------- 2. block-level joint tests on resid_us ----------
    print("\n=== 2. block-level joint tests (train fit -> confirm projection) ===")
    dd = lambda c: (f[c + "_H"] - f[c + "_A"])
    blocks = {
        "sched": pd.DataFrame({"rest": dd("days_rest"), "b2bH": f.b2b_H, "b2bA": f.b2b_A,
                               "t34": dd("is_3in4"), "trav": dd("travel3d_km"),
                               "tzA": f.tz_from_home_A, "road": f.road_trip_A,
                               "stand": f.home_stand_H}),
        "form": pd.DataFrame({"f10": dd("form10"), "mom": dd("momentum"),
                              "streak": dd("streak"), "sos": dd("sos10"),
                              "blow": dd("blowout15_15")}),
        "style": pd.DataFrame({"pace": dd("td_poss"), "ftr": dd("td_ftr"),
                               "p3r": dd("td_p3rate"), "orb": dd("td_orbp"),
                               "drb": dd("td_drbp"), "ast": dd("td_astr"),
                               "tov": dd("td_tovr"), "rim": dd("td_rimrate")}),
        "outs": pd.DataFrame({"n_out": f.n_out_home - f.n_out_away}),
        "bench": pd.DataFrame({"brel": f.bs_bench_rel_H - f.bs_bench_rel_A,
                               "srel": f.bs_starter_rel_H - f.bs_starter_rel_A,
                               "share": f.bs_bench_share_H - f.bs_bench_share_A}),
        "contin": pd.DataFrame({"cont": dd("lineup_cont"), "churn": dd("starter_churn")}),
    }
    for name, B in blocks.items():
        Bt = B.loc[train].fillna(B.loc[train].mean())
        Xt = sm.add_constant(Bt.values)
        r = sm.OLS(f.resid_us.values[train], Xt).fit()
        fp = r.f_pvalue
        proj = B.fillna(B.loc[train].mean()).values @ r.params[1:]  # train betas
        pc = proj[conf]
        rc = sm.OLS(f.resid_us.values[conf], sm.add_constant(pc)).fit(cov_type="HC1")
        m_new = f.m_us.values + proj + r.params[0]
        dcf = paired_delta(f, conf, m_new)
        print(f"  {name:7s}: train R2={r.rsquared:.4f} F-p={fp:.3f} | confirm proj "
              f"slope={rc.params[1]:+.2f} t={rc.tvalues[1]:+.2f} | confirm dLL "
              f"{dcf[0]:+.5f} ({dcf[1]:+.5f},{dcf[2]:+.5f})")

    # ---------- 3. bench-split reliability autopsy ----------
    print("\n=== 3. bench_rel split-half reliability (is the measurement real?) ===")
    # first-half vs second-half season team bench_rel (contemporaneous halves)
    bb = pd.read_pickle(os.path.join(OUTDIR, "regimec_bench.pkl"))
    # reconstruct team-level trailing states at gm 41 and season end from H/A rows
    rows = []
    for side in ("H", "A"):
        cols = {f"bs_bench_rel_{side}": "bench_rel", f"bs_starter_rel_{side}": "starter_rel"}
        sub = f[["season", "game_id", "game_date", "home" if side == "H" else "away",
                 f"bs_bench_rel_{side}", f"bs_starter_rel_{side}",
                 "game_no_H" if side == "H" else "game_no_A"]].copy()
        sub.columns = ["season", "game_id", "game_date", "team", "bench_rel",
                       "starter_rel", "game_no"]
        rows.append(sub)
    ts = pd.concat(rows).dropna(subset=["bench_rel"])
    # state at game_no ~41 uses games 1-40; correlate with realized 2nd-half bench_rel
    # proxy: state at 41 vs final state minus... simpler: corr(state@41, state@end-state@41 scaled)
    at41 = ts[ts.game_no.between(39, 43)].groupby(["season", "team"]).bench_rel.mean()
    atend = ts[ts.game_no >= 78].groupby(["season", "team"]).bench_rel.mean()
    j = pd.concat([at41.rename("h1"), atend.rename("full")], axis=1).dropna()
    # 2nd-half-only signal: full = w*h1 + (1-w)*h2 approx; report corr(h1, full) and
    # the implied h1-h2 corr under equal weights
    r_h1full = np.corrcoef(j.h1, j.full)[0, 1]
    print(f"  corr(bench_rel@gm41, bench_rel@end) = {r_h1full:+.3f} (n={len(j)})"
          f"  [autocorrelated by construction; >0.7 expected if stable]")
    # cleaner: does bench_rel@41 predict OUR residual in 2nd half at team level?
    sh = ts[ts.game_no > 43].merge(at41.rename("b41"), on=["season", "team"])
    fl = f[["game_id", "resid_us", "home", "away"]]
    sh = sh.merge(fl, on="game_id")
    sh["r_team"] = np.where(sh.team == sh.home, sh.resid_us, -sh.resid_us)
    rr = sm.OLS(sh.r_team.values, sm.add_constant(sh.b41.values)).fit(cov_type="HC1")
    print(f"  2nd-half team residual ~ bench_rel@41: slope={rr.params[1]:+.3f} "
          f"t={rr.tvalues[1]:+.2f} n={len(sh)}")

    # ---------- 4. outs joint (train+confirm on feed-covered) ----------
    print("\n=== 4. outs joint test (n_out + out_dpm_pos, feed-covered only) ===")
    covered = (f.outs_feed_covered_H > 0).values
    O = pd.DataFrame({"n_out": f.n_out_home - f.n_out_away,
                      "dpm": f.out_dpm_pos_H - f.out_dpm_pos_A})
    for label, mask in (("train", train & covered), ("confirm", conf & covered)):
        r = sm.OLS(f.resid_us.values[mask], sm.add_constant(O[mask].values)).fit(cov_type="HC1")
        print(f"  {label}: n={mask.sum()} b_nout={r.params[1]:+.3f} (t={r.tvalues[1]:+.2f}) "
              f"b_dpm={r.params[2]:+.3f} (t={r.tvalues[2]:+.2f})")

    # ---------- 5. deficit profile ----------
    print("\n=== 5. where does the core deficit live? (d_excess by market confidence) ===")
    f["mkt_conf"] = (f.p_mkt - 0.5).abs()
    for label, mask in (("train", train), ("confirm", conf)):
        sub = f[mask].copy()
        sub["bucket"] = pd.cut(sub.mkt_conf, [0, .1, .2, .3, .5])
        t = sub.groupby("bucket", observed=True).agg(n=("d_excess", "size"),
                                                     d=("d_excess", "mean"))
        print(f"  {label}:")
        print(t.to_string())
    # disagreement magnitude
    for label, mask in (("train", train), ("confirm", conf)):
        sub = f[mask].copy()
        sub["adm"] = sub.dm.abs()
        sub["bucket"] = pd.qcut(sub.adm, 4)
        t = sub.groupby("bucket", observed=True).agg(n=("d_excess", "size"),
                                                     d=("d_excess", "mean"))
        print(f"  {label} by |dm| quartile:")
        print(t.to_string())


if __name__ == "__main__":
    main()
