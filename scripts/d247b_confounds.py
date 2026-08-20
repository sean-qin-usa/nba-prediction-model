#!/usr/bin/env python3
"""D247b — the two things D247 left open.

(1) WAS 2024-25 A SEASON WHEN THE MARKET WAS BEATABLE, OR A GOOD RUN OF 210
    BETS? The published +24.00% sits on 210 bets. The same season contains 1,230
    games, and every one of them carries a market-quality reading that does not
    depend on which bets the selector happened to take. If 2024-25 was a genuine
    mispricing regime -- the adaptation story -- the whole slate must look soft:
    bigger normalised gap, better CLV, larger market error. If the slate is
    ordinary and only the bets are good, the ROI was selection variance and the
    adaptation story is explaining a fact not in evidence.

    This is ~6x the sample of the bet ledger and it is not conditioned on the
    selector, so it is both more powerful and less circular.

(2) IS T3 THE FAVOURITE BIAS WEARING A COSTUME? T3 regressed a team's market
    residual on its PRIOR-season margin and found -0.024, CI just clear of zero.
    But prior-season margin is a strong proxy for CURRENT-season market
    expectation, and "big favourites underperform the spread" is one of the
    oldest documented effects in these markets. If that is what T3 found, it is
    not about adaptation, memory, or offseason training at all.

    The discriminating test: add the team's CURRENT-season mean closing line as
    a covariate. If prior-season margin survives, the market is anchoring on
    last year beyond where it has the team priced this year. If it collapses,
    T3 was the favourite bias.

Also reported: the Type M (magnitude) exaggeration. T3's |estimate| 0.024 is
BELOW its own MDE80 of 0.031. An estimate that reaches significance while the
study is underpowered to find it is upward-biased CONDITIONAL ON SIGNIFICANCE,
and the size of that bias is computable rather than a matter of opinion.
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

LN2 = np.log(2.0)


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def nll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def clus(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    k = len(v)
    if k < 2:
        return np.nan, np.nan, np.nan, np.nan, k
    se = v.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return v.mean(), v.mean() - tc * se, v.mean() + tc * se, v.mean() / se, k


def type_m(est, se):
    """Exaggeration ratio: E[|est| | significant] / true, evaluated at the
    hypothesis that the TRUE effect equals the observed estimate. Gelman/Carlin."""
    rng = np.random.default_rng(247)
    draws = rng.normal(est, se, 400000)
    sig = np.abs(draws) > 1.96 * se
    if sig.sum() < 100:
        return np.nan, np.nan
    exagg = np.abs(draws[sig]).mean() / abs(est)
    sign_err = float((np.sign(draws[sig]) != np.sign(est)).mean())
    return float(exagg), sign_err


def main():
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
    f["game_id"] = zf(f["game_id"])
    f["game_date"] = pd.to_datetime(f["game_date"])
    f = f.dropna(subset=["margin_actual", "close_margin", "open_margin",
                         "p_us", "y"]).copy()
    seasons = sorted(f.season.unique())
    out = {}

    # ================================================================
    # (1) FULL-SLATE MARKET QUALITY BY SEASON
    # ================================================================
    print("=" * 78)
    print("T5  WAS 2024-25 A SOFT MARKET? FULL SLATE, ALL 1,230 GAMES/SEASON")
    print("=" * 78)
    rows = []
    for s in seasons:
        g = f[f.season == s]
        y = g.y.to_numpy(float)
        p_mkt = np.clip(1 / (1 + np.exp(-g.close_margin.to_numpy(float) / 6.96)),
                        1e-9, 1 - 1e-9)
        p_us = np.clip(g.p_us.to_numpy(float), 1e-9, 1 - 1e-9)
        ll_mkt = nll(p_mkt, y).mean()
        ll_us = nll(p_us, y).mean()
        gap = (ll_us - ll_mkt) / (LN2 - ll_mkt)
        sd = np.sign(g.m_us.to_numpy(float) - g.open_margin.to_numpy(float))
        clv = (sd * (g.close_margin - g.open_margin).to_numpy(float)).mean()
        # model's ATS record vs the CLOSING line, whole slate
        cov = (np.sign(g.margin_actual.to_numpy(float)
                       - g.close_margin.to_numpy(float)) == sd)
        rows.append(dict(season=s, n=len(g),
                         mkt_ll=float(ll_mkt), us_ll=float(ll_us),
                         norm_gap=float(gap),
                         mkt_abs_err=float((g.margin_actual
                                            - g.close_margin).abs().mean()),
                         clv=float(clv), ats_vs_close=float(cov.mean())))
    d = pd.DataFrame(rows)
    print(d.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

    print("\n  Where does 2024-25 rank among the 19 seasons on each measure?")
    tgt = "2024-25"
    for c, lab, hi_good in (("norm_gap", "normalised gap (lower=better for us)", False),
                            ("clv", "CLV on the model's side", True),
                            ("mkt_abs_err", "market absolute error", True),
                            ("ats_vs_close", "model ATS vs the close", True)):
        v = d.set_index("season")[c]
        rank = int((v > v[tgt]).sum()) + 1 if hi_good else int((v < v[tgt]).sum()) + 1
        z = float((v[tgt] - v.mean()) / v.std(ddof=1))
        print(f"    {lab:38} {v[tgt]:+8.4f}   rank {rank:2d}/19   z {z:+5.2f}")
        out[f"rank_{c}"] = dict(value=float(v[tgt]), rank=rank, z=z)

    print("\n  Read: a genuine 'the market was soft in 2024-25' regime should put")
    print("  2024-25 at or near rank 1 on several of these at once.")

    # is the season-to-season variation in norm_gap even real?
    m, lo, hi, t, k = clus(d.norm_gap)
    print(f"\n  norm_gap across seasons: mean {m:+.4f} sd {d.norm_gap.std(ddof=1):.4f}")
    print(f"  2024-25 deviation from mean: "
          f"{d.set_index('season').norm_gap[tgt] - m:+.4f}")
    out["T5_rows"] = rows

    # ---- what a 210-bet season's ROI can resolve, analytically -------
    print("\n" + "-" * 78)
    print("  HOW MUCH ROI DISPERSION DOES A 210-BET SEASON PRODUCE BY ITSELF?")
    print("-" * 78)
    n_bets, roi_obs, roi_pool7 = 210, 24.00, 9.11
    for hit, dec in ((0.58, 1.909), (0.55, 1.909)):
        r_win, r_lose = dec - 1, -1.0
        mu = hit * r_win + (1 - hit) * r_lose
        sd = np.sqrt(hit * (r_win - mu) ** 2 + (1 - hit) * (r_lose - mu) ** 2)
        se = 100 * sd / np.sqrt(n_bets)
        print(f"  hit {hit:.0%} at {dec:.3f}: per-season ROI SE = {se:.2f} pts")
        print(f"    2024-25 (+{roi_obs:.2f}%) vs 7-season pooled (+{roi_pool7:.2f}%)"
              f" = {(roi_obs-roi_pool7)/se:+.2f} SE")
        # 2024-25 was the BEST of 7 seasons -- correct for selection
        p_one = 2 * (1 - stats.norm.cdf(abs(roi_obs - roi_pool7) / se))
        p_max = 1 - (1 - p_one) ** 7
        print(f"    unadjusted p {p_one:.3f};  as the MAX of 7 seasons "
              f"p {p_max:.3f}  "
              f"{'ANOMALY' if p_max < 0.05 else 'WITHIN NOISE'}")
        out[f"roi_noise_hit{int(hit*100)}"] = dict(se=float(se),
                                                   p_one=float(p_one),
                                                   p_max=float(p_max))

    # ================================================================
    # (2) IS T3 THE FAVOURITE BIAS?
    # ================================================================
    print("\n" + "=" * 78)
    print("T6  T3 WITH CONTROLS -- ANCHORING, OR THE FAVOURITE BIAS?")
    print("=" * 78)
    h = f.assign(team=f.home, resid=f.margin_actual - f.close_margin,
                 mkt=f.close_margin, act=f.margin_actual)
    a = f.assign(team=f.away, resid=-(f.margin_actual - f.close_margin),
                 mkt=-f.close_margin, act=-f.margin_actual)
    cols = ["season", "team", "resid", "mkt", "act"]
    L = pd.concat([h[cols], a[cols]], ignore_index=True)
    ts = (L.groupby(["season", "team"])
            .agg(act=("act", "mean"), resid=("resid", "mean"),
                 mkt=("mkt", "mean"), n=("act", "size")).reset_index())
    ts["prev_act"] = ts.groupby("team")["act"].shift(1)
    ts["prev_season"] = ts.groupby("team")["season"].shift(1)
    idx = {s: i for i, s in enumerate(seasons)}
    ok = ts.dropna(subset=["prev_act"]).copy()
    ok = ok[ok.apply(lambda r: idx[r.season] - idx[r.prev_season] == 1, axis=1)]
    print(f"  {len(ok)} team-seasons")
    print(f"  corr(prev_act, current mkt) = "
          f"{np.corrcoef(ok.prev_act, ok.mkt)[0,1]:+.3f}   "
          f"<- the confound, quantified")

    specs = {
        "prev only":            ["prev_act"],
        "mkt only":             ["mkt"],
        "prev + mkt (KEY)":     ["prev_act", "mkt"],
    }
    for lab, xs in specs.items():
        per = []
        for s, g in ok.groupby("season"):
            if len(g) < 12:
                continue
            X = np.column_stack([np.ones(len(g))]
                                + [g[x].to_numpy(float) for x in xs])
            b = np.linalg.lstsq(X, g.resid.to_numpy(float), rcond=None)[0]
            per.append({x: float(b[i + 1]) for i, x in enumerate(xs)}
                       | {"season": s})
        dp = pd.DataFrame(per)
        print(f"\n  --- {lab} ---")
        for x in xs:
            m, lo, hi, t, k = clus(dp[x])
            se = dp[x].std(ddof=1) / np.sqrt(k)
            mde = 2.80 * dp[x].std(ddof=1) / np.sqrt(k)
            flag = "SIG" if (hi < 0 or lo > 0) else "ns"
            print(f"    {x:10} {m:+.5f}  CI [{lo:+.5f}, {hi:+.5f}]  "
                  f"t {t:+5.2f}  k={k}  MDE80 {mde:.5f}  {flag}")
            if flag == "SIG":
                ex, serr = type_m(m, se)
                print(f"               Type M exaggeration {ex:.2f}x  "
                      f"(true effect likely ~{m/ex:+.5f}); "
                      f"sign-error risk {serr:.3f}")
            out[f"T6_{lab.split()[0]}_{x}"] = dict(
                mean=float(m), ci=[float(lo), float(hi)], k=int(k),
                mde80=float(mde), flag=flag)

    print("\n  DISCRIMINATION: if prev_act keeps its coefficient once mkt is in")
    print("  the model, the market is anchoring on last season beyond where it")
    print("  has the team priced now. If it collapses toward zero, T3 was the")
    print("  favourite bias and has nothing to do with offseason adaptation.")

    json.dump(out, open(ROOT / "data" / "d247b_confounds.json", "w"),
              default=float)
    print("\nwrote data/d247b_confounds.json")


if __name__ == "__main__":
    main()
