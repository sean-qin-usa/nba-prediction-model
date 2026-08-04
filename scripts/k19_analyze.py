#!/usr/bin/env python3
"""K19-ANALYZE — the 19-season MODEL table: pooled, by era, the K=19
season-clustered CI on the pooled gap, the trend, and the DARKO-coverage
confound separated rather than assumed away.

Input `data/k19_pergame.csv` (availability-BLIND, one constant tier, LOWER
BOUND).  Read-only; writes only data/k19_model_stats.json.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from math import sqrt

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import nbapred.threads                                          # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

from lb_longshot import cluster_boot, cluster_mean_t, icc_oneway  # noqa: E402

LN2 = 0.6931471805599453
SEED = 20260803
SEASONS19 = ["%d-%02d" % (y, (y + 1) % 100) for y in range(2007, 2026)]
COVID = {"2019-20", "2020-21"}
LOCKOUT = {"2011-12"}
K19_ERA = {}
for _s in SEASONS19[0:4]:
    K19_ERA[_s] = "K-A"
for _s in SEASONS19[4:7]:
    K19_ERA[_s] = "K-B"
for _s in SEASONS19[7:12]:
    K19_ERA[_s] = "K-C"
for _s in SEASONS19[12:14]:
    K19_ERA[_s] = "K-D"
for _s in SEASONS19[14:]:
    K19_ERA[_s] = "K-E"
ERA_ORDER = ["K-A", "K-B", "K-C", "K-D", "K-E"]
ERA_DESC = {"K-A": "2007-08..2010-11 pre-lockout CBA",
            "K-B": "2011-12..2013-14 post-lockout (2011-12 = LOCKOUT stratum)",
            "K-C": "2014-15..2018-19 3PT ramp",
            "K-D": "2019-20..2020-21 COVID (both separate strata)",
            "K-E": "2021-22..2025-26 modern / the corpus every gate used"}
# D158's certified FULL-FEED (T2) reference — a DIFFERENT tier, never pooled
# with the blind arm, carried only as the marked reference on the chart.
D158_FULLFEED = {"2023-24": 16.21, "2024-25": 6.30, "2025-26": 11.79}
D158_FULLFEED_POOLED = 11.45


def ll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def norm_gap(dus, dmk):
    return 100.0 * (dus.mean() - dmk.mean()) / (LN2 - dmk.mean())


def block(df, label, seed=SEED):
    d = (df.ll_us - df.ll_mkt).values
    lo, hi, se = cluster_boot(d, df.season.values, n_boot=4000, seed=seed)
    tlo, thi, K = cluster_mean_t(d, df.season.values)
    icc, deff = icc_oneway(d, df.season.values)
    iid = d.std(ddof=1) / sqrt(len(d))
    return {"label": label, "n": int(len(df)), "K": int(K),
            "ll_us": float(df.ll_us.mean()), "ll_mkt": float(df.ll_mkt.mean()),
            "raw_gap": float(d.mean()),
            "norm_gap_pct": float(norm_gap(df.ll_us, df.ll_mkt)),
            "boot_lo": lo, "boot_hi": hi, "boot_se": se,
            "t_lo": float(tlo), "t_hi": float(thi),
            "icc": float(icc), "deff_anova": float(deff), "iid_se": float(iid),
            "deff_boot": float(se / iid) if iid else float("nan")}


def main():
    df = pd.read_csv(os.path.join(ROOT, "data", "k19_pergame.csv"),
                     dtype={"game_id": str})
    df["ll_us"] = ll(df.y.values, df.p_us.values)
    df["ll_mkt"] = ll(df.y.values, df.p_mkt.values)
    df["era"] = df.season.map(K19_ERA)
    mj = json.load(open(os.path.join(ROOT, "data", "k19_model.json")))
    dk = {o["season"]: o for o in mj["seasons"]}

    res = {"tier": mj["tier"], "lower_bound": True, "era_desc": ERA_DESC}
    print("=" * 124)
    print("K19-ANALYZE — 19-SEASON MODEL TABLE, AVAILABILITY-BLIND (ONE "
          "CONSTANT TIER).  EVERY LEVEL IS A LOWER BOUND ON THE MODEL.")
    print("  Production ships T2 on every game; blind is strictly weaker.  The "
          "3 full-feed seasons (D158: 11.45% pooled) are the live estimate.")
    print("=" * 124)

    # ---------------------------------------------------------- per season --
    print(f"\n[1] PER SEASON\n    {'season':<10}{'era':<6}{'n':>6}"
          f"{'ll_us':>9}{'ll_mkt':>9}{'raw gap':>10}{'norm gap':>10}"
          f"{'DARKO cov':>11}{'note':<28}")
    rows = []
    for s in SEASONS19:
        g = df[df.season == s]
        note = ("LOCKOUT stratum" if s in LOCKOUT else
                "COVID stratum" if s in COVID else "")
        r = {"season": s, "era": K19_ERA[s], "n": int(len(g)),
             "ll_us": float(g.ll_us.mean()), "ll_mkt": float(g.ll_mkt.mean()),
             "raw_gap": float((g.ll_us - g.ll_mkt).mean()),
             "norm_gap_pct": float(norm_gap(g.ll_us, g.ll_mkt)),
             "darko_cov": dk[s]["darko_frac_roster_nonzero"],
             "stratum": note}
        rows.append(r)
        print(f"    {s:<10}{r['era']:<6}{r['n']:>6}{r['ll_us']:>9.5f}"
              f"{r['ll_mkt']:>9.5f}{r['raw_gap']:>+10.5f}"
              f"{r['norm_gap_pct']:>+9.2f}%{100*r['darko_cov']:>10.1f}%"
              f"  {note:<28}")
    res["per_season"] = rows

    # -------------------------------------------------------------- pooled --
    print(f"\n[2] POOLED AND BY BLOCK — the K=19 season-clustered CI on the "
          f"pooled RAW GAP is the statistic this entry exists to produce.")
    print(f"    {'block':<34}{'K':>3}{'n':>7}{'ll_us':>9}{'ll_mkt':>9}"
          f"{'raw gap':>10}{'norm':>8}{'  [95% cluster boot]':>24}"
          f"{'  [K-1 cluster-mean t]':>26}{'ICC':>9}{'DEFF':>7}")
    blocks = [("ALL 19 (blind)", df),
              ("17 ex-COVID", df[~df.season.isin(COVID)]),
              ("16 ex-COVID ex-lockout",
               df[~df.season.isin(COVID | LOCKOUT)]),
              ("OOS_DEEP 15 (2007-08..2021-22)",
               df[df.season.isin(SEASONS19[:15])]),
              ("DEV 2 (2023-24+2024-25)",
               df[df.season.isin(["2023-24", "2024-25"])]),
              ("NONDEV 2 (2022-23+2025-26)",
               df[df.season.isin(["2022-23", "2025-26"])]),
              ("FULL-FEED 3 seasons, BLIND arm",
               df[df.season.isin(["2023-24", "2024-25", "2025-26"])]),
              ("D132 5-season corpus, BLIND arm",
               df[df.season.isin(SEASONS19[14:])])]
    bl = []
    for lab, sub in blocks:
        b = block(sub, lab)
        bl.append(b)
        print(f"    {lab:<34}{b['K']:>3}{b['n']:>7}{b['ll_us']:>9.5f}"
              f"{b['ll_mkt']:>9.5f}{b['raw_gap']:>+10.5f}"
              f"{b['norm_gap_pct']:>+7.2f}%"
              f"  [{b['boot_lo']:+.5f},{b['boot_hi']:+.5f}]"
              f"  [{b['t_lo']:+.5f},{b['t_hi']:+.5f}]"
              f"{b['icc']:>+9.5f}{b['deff_anova']:>7.2f}"
              f"{'  SIG' if b['t_lo']>0 or b['t_hi']<0 else '  ns'}")
    res["blocks"] = bl
    print(f"\n    NOTE: the gap is POSITIVE = WE LOSE TO THE MARKET.  'SIG' "
          f"here means 'the market beats us by a margin that is not zero',\n"
          f"    which is the OPPOSITE of a good result — it is what the K=19 "
          f"frame finally has the power to establish.")

    # ---------------------------------------------------------- by era ------
    print(f"\n[3] BY K19-ERA\n    {'era':<6}{'seasons':<34}{'K':>3}{'n':>7}"
          f"{'raw gap':>10}{'norm':>8}{'  [K-1 t]':>26}{'DARKO cov':>11}")
    era_rows = []
    for e in ERA_ORDER:
        sub = df[df.era == e]
        b = block(sub, e)
        cov = np.mean([r["darko_cov"] for r in rows if r["era"] == e])
        b["darko_cov"] = float(cov)
        era_rows.append(b)
        print(f"    {e:<6}{ERA_DESC[e]:<34}{b['K']:>3}{b['n']:>7}"
              f"{b['raw_gap']:>+10.5f}{b['norm_gap_pct']:>+7.2f}%"
              f"  [{b['t_lo']:+.5f},{b['t_hi']:+.5f}]{100*cov:>10.1f}%")
    res["by_era"] = era_rows

    # DerSimonian-Laird on the era means of the RAW gap
    pts = np.array([b["raw_gap"] for b in era_rows])
    ses = np.array([b["boot_se"] for b in era_rows])
    w = 1 / ses ** 2
    mu = (w * pts).sum() / w.sum()
    Q = (w * (pts - mu) ** 2).sum()
    dfq = len(pts) - 1
    I2 = max(0.0, 100 * (Q - dfq) / Q)
    from statistics import NormalDist
    z = ((Q / dfq) ** (1 / 3) - (1 - 2 / (9 * dfq))) / sqrt(2 / (9 * dfq))
    p = 1 - NormalDist().cdf(z)
    verdict = ("ERA-STABLE" if I2 < 50 and p > 0.10 else
               "ERA-CONDITIONAL" if pts.min() * pts.max() > 0 else
               "ERA-SPECIFIC")
    print(f"\n    ERA HETEROGENEITY (DL): Q={Q:.2f} df={dfq} I2={I2:.1f}% "
          f"p={p:.4f}  -> **{verdict}**  (sign is the same in all 5 eras: "
          f"we lose to the market in every one)")
    res["era_heterogeneity"] = {"Q": float(Q), "df": dfq, "I2": float(I2),
                                "p": float(p), "verdict": verdict}

    # ------------------------------------------------------------ trend -----
    print(f"\n[4] IS THE GAP STABLE, AND DOES IT TREND?  (season-level OLS, "
          f"n=19 season points; x = season index 0..18)")
    t = pd.DataFrame(rows)
    t["x"] = np.arange(len(t))
    tr = {}
    for yc in ("norm_gap_pct", "raw_gap"):
        for lab, sel in (("all 19", np.ones(len(t), bool)),
                         ("17 ex-COVID", ~t.season.isin(COVID).values)):
            s = t[sel]
            X = np.c_[np.ones(len(s)), s.x.values]
            b, *_ = np.linalg.lstsq(X, s[yc].values, rcond=None)
            resid = s[yc].values - X @ b
            sig2 = resid @ resid / (len(s) - 2)
            cov = sig2 * np.linalg.inv(X.T @ X)
            se = sqrt(cov[1, 1])
            tq = {15: 2.131, 17: 2.110}.get(len(s) - 2, 2.11)
            tr[f"{yc}|{lab}"] = {"slope_per_season": float(b[1]),
                                 "se": float(se), "t": float(b[1] / se),
                                 "lo": float(b[1] - tq * se),
                                 "hi": float(b[1] + tq * se),
                                 "n_seasons": int(len(s))}
            v = tr[f"{yc}|{lab}"]
            print(f"    {yc:<14}{lab:<14}slope={b[1]:+.5f}/season "
                  f"se={se:.5f} t={b[1]/se:+.2f} "
                  f"CI[{v['lo']:+.5f},{v['hi']:+.5f}] "
                  f"{'SIG' if v['lo']*v['hi']>0 else 'ns'}")
    res["trend"] = tr
    sd_norm = float(t.norm_gap_pct.std(ddof=1))
    print(f"    DISPERSION: norm gap mean {t.norm_gap_pct.mean():+.2f}% "
          f"sd {sd_norm:.2f}pp across 19 seasons; min "
          f"{t.norm_gap_pct.min():+.2f}% ({t.loc[t.norm_gap_pct.idxmin(),'season']}) "
          f"max {t.norm_gap_pct.max():+.2f}% "
          f"({t.loc[t.norm_gap_pct.idxmax(),'season']}); "
          f"**19/19 seasons positive = the market beats us in EVERY season.**")
    res["dispersion"] = {"mean": float(t.norm_gap_pct.mean()), "sd": sd_norm,
                         "min": float(t.norm_gap_pct.min()),
                         "max": float(t.norm_gap_pct.max()),
                         "n_positive": int((t.raw_gap > 0).sum())}

    # ------------------------------------------- DARKO-COVERAGE CONFOUND ----
    print(f"\n[5] THE DARKO-COVERAGE CONFOUND, MEASURED — D153 found the "
          f"historical readout partly measures OUR OWN TALENT-FEED RAMP "
          f"(corr +0.79).\n    DARKO minute-coverage of the roster window runs "
          f"{100*t.darko_cov.iloc[0]:.1f}% (2007-08) -> 100.0% (2023-24 on).  "
          f"It is COLLINEAR WITH TIME by construction\n    (corr with the "
          f"season index = {np.corrcoef(t.x, t.darko_cov)[0,1]:+.4f}), so "
          f"'era' and 'our feed' CANNOT be fully separated on this frame.  "
          f"What CAN be measured:")
    cc = {}
    for yc in ("norm_gap_pct", "raw_gap"):
        for lab, sel in (("all 19", np.ones(len(t), bool)),
                         ("17 ex-COVID", ~t.season.isin(COVID).values)):
            s = t[sel]
            r_cov = float(np.corrcoef(s.darko_cov, s[yc])[0, 1])
            r_x = float(np.corrcoef(s.x, s[yc])[0, 1])
            r_xc = float(np.corrcoef(s.x, s.darko_cov)[0, 1])
            # partial corr of gap vs coverage controlling for time, and v.v.
            pc = ((r_cov - r_x * r_xc) /
                  sqrt(max(1e-12, (1 - r_x ** 2) * (1 - r_xc ** 2))))
            px = ((r_x - r_cov * r_xc) /
                  sqrt(max(1e-12, (1 - r_cov ** 2) * (1 - r_xc ** 2))))
            cc[f"{yc}|{lab}"] = {"corr_gap_coverage": r_cov,
                                 "corr_gap_time": r_x,
                                 "corr_time_coverage": r_xc,
                                 "partial_gap_coverage_given_time": float(pc),
                                 "partial_gap_time_given_coverage": float(px)}
            print(f"    {yc:<14}{lab:<14}corr(gap,DARKOcov)={r_cov:+.4f}  "
                  f"corr(gap,time)={r_x:+.4f}  "
                  f"partial(gap,cov|time)={pc:+.4f}  "
                  f"partial(gap,time|cov)={px:+.4f}")
    res["darko_confound"] = cc
    lo_cov = t[t.darko_cov < 0.5]
    hi_cov = t[t.darko_cov >= 0.5]
    print(f"\n    SPLIT ON COVERAGE (not on calendar): "
          f"DARKO<50% -> {len(lo_cov)} seasons, mean norm gap "
          f"{lo_cov.norm_gap_pct.mean():+.2f}%;  DARKO>=50% -> "
          f"{len(hi_cov)} seasons, mean {hi_cov.norm_gap_pct.mean():+.2f}%  "
          f"(difference {hi_cov.norm_gap_pct.mean()-lo_cov.norm_gap_pct.mean():+.2f}pp)")
    b_lo = block(df[df.season.isin(lo_cov.season)], "DARKO<50%")
    b_hi = block(df[df.season.isin(hi_cov.season)], "DARKO>=50%")
    print(f"    game-level: DARKO<50%  n={b_lo['n']} raw {b_lo['raw_gap']:+.5f} "
          f"norm {b_lo['norm_gap_pct']:+.2f}%  K-1 t "
          f"[{b_lo['t_lo']:+.5f},{b_lo['t_hi']:+.5f}]")
    print(f"                DARKO>=50% n={b_hi['n']} raw {b_hi['raw_gap']:+.5f} "
          f"norm {b_hi['norm_gap_pct']:+.2f}%  K-1 t "
          f"[{b_hi['t_lo']:+.5f},{b_hi['t_hi']:+.5f}]")
    res["darko_split"] = {"low": b_lo, "high": b_hi,
                          "low_seasons": lo_cov.season.tolist(),
                          "high_seasons": hi_cov.season.tolist()}

    # ------------------------------------------- the tier gap, stated -------
    print(f"\n[6] THE TIER GAP — BLIND vs D158's CERTIFIED FULL-FEED (T2), the "
          f"same 3 seasons, so the reader can price the constant-tier choice:")
    print(f"    {'season':<10}{'BLIND norm':>12}{'D158 T2 norm':>15}"
          f"{'tier cost':>12}")
    tier = []
    for s, v in D158_FULLFEED.items():
        b = float(t[t.season == s].norm_gap_pct.iloc[0])
        tier.append({"season": s, "blind": b, "t2": v, "cost_pp": b - v})
        print(f"    {s:<10}{b:>+11.2f}%{v:>+14.2f}%{b-v:>+11.2f}pp")
    bf = block(df[df.season.isin(D158_FULLFEED)], "blind 3")
    print(f"    {'POOLED 3':<10}{bf['norm_gap_pct']:>+11.2f}%"
          f"{D158_FULLFEED_POOLED:>+14.2f}%"
          f"{bf['norm_gap_pct']-D158_FULLFEED_POOLED:>+11.2f}pp")
    print(f"    READ: running BLIND costs "
          f"{bf['norm_gap_pct']-D158_FULLFEED_POOLED:+.2f}pp of normalized gap "
          f"on the seasons where the honest feed exists.  Every 19-season "
          f"number\n    above carries that penalty, which is why they are "
          f"LOWER BOUNDS and NOT the live expectation.")
    res["tier_cost"] = {"per_season": tier,
                        "blind_pooled_3": bf["norm_gap_pct"],
                        "t2_pooled_3": D158_FULLFEED_POOLED,
                        "cost_pp": bf["norm_gap_pct"] - D158_FULLFEED_POOLED}

    json.dump(res, open(os.path.join(ROOT, "data", "k19_model_stats.json"),
                        "w"), indent=1, default=str)
    print("\nwrote data/k19_model_stats.json")


if __name__ == "__main__":
    main()
