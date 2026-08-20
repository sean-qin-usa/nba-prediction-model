#!/usr/bin/env python3
"""D247 — BETWEEN-SEASON ADAPTATION: does the league adjust to the champion, and
does the market fail to price it?

Sean's hypothesis, in three limbs:
  (a) after DEN won 2022-23, opponents began altitude prep and DEN's altitude
      edge collapsed;
  (b) teams adjust training to counter whoever beat them / won the league;
  (c) teams fix last season's problems, and the market underprices the change --
      which is why 2024-25 was our best betting season.

WHY THIS NEEDS PREREGISTRATION. A 19-season x 30-team panel contains an enormous
number of subsets, and the register already carries two altitude nulls (D70, D96)
plus a documented 12-point season-to-season swing in DEN's home deviation. That
combination -- a noisy series and a story I already find plausible -- is the exact
configuration that manufactures findings. D239 measured the capacity directly:
best-of-N random subsets buy +2.54 ROI points from nothing.

So the arms below are fixed BEFORE any of them is read, the outcome is the same
for every arm, and the champion list is public record rather than anything chosen
from this data.

THE OUTCOME IS A MARKET RESIDUAL, NEVER RAW PERFORMANCE. The claim is not "the
champion declines" -- that is well known and priced. The claim is "the champion
declines and the market does not know." So every arm scores

    resid = signed(margin_actual - close_margin)      from the team's perspective

against the CLOSE, the most informed price available. A team can collapse
entirely and still produce resid = 0 if the market saw it coming; that is a
refutation of the hypothesis, not a confirmation of it.

T0 is the gate. If 2024-25's ROI is within sampling noise of the pooled ROI,
there is no anomaly to explain and limbs (b)/(c) are answering a question that
was never posed. T0 runs first and its verdict is reported whatever it says.
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

# Public record. The value is the team that won the title AT THE END of that
# season, so season s's champion is the DEFENDING champion during season s+1.
CHAMPION = {
    "2006-07": "SAS", "2007-08": "BOS", "2008-09": "LAL", "2009-10": "LAL",
    "2010-11": "DAL", "2011-12": "MIA", "2012-13": "MIA", "2013-14": "SAS",
    "2014-15": "GSW", "2015-16": "CLE", "2016-17": "GSW", "2017-18": "GSW",
    "2018-19": "TOR", "2019-20": "LAL", "2020-21": "MIL", "2021-22": "GSW",
    "2022-23": "DEN", "2023-24": "BOS", "2024-25": "OKC",
}
# Beaten finalist, same convention -- limb (b) says teams adapt to whoever beat
# THEM, so the runner-up is the team with the strongest motive to change.
RUNNER_UP = {
    "2006-07": "CLE", "2007-08": "LAL", "2008-09": "ORL", "2009-10": "BOS",
    "2010-11": "MIA", "2011-12": "OKC", "2012-13": "SAS", "2013-14": "MIA",
    "2014-15": "CLE", "2015-16": "GSW", "2016-17": "CLE", "2017-18": "CLE",
    "2018-19": "GSW", "2019-20": "MIA", "2020-21": "PHX", "2021-22": "BOS",
    "2022-23": "MIA", "2023-24": "DAL", "2024-25": "IND",
}
ALTITUDE = ("DEN", "UTA")


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def clus(v):
    """Season-clustered mean, CI and t. k = number of seasons, not games."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    k = len(v)
    if k < 2:
        return np.nan, np.nan, np.nan, np.nan, k
    se = v.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return v.mean(), v.mean() - tc * se, v.mean() + tc * se, v.mean() / se, k


def mde80(v):
    """Two-sided a=.05, 80% power, given the observed season-level spread."""
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    k = len(v)
    if k < 2:
        return np.nan
    return 2.80 * v.std(ddof=1) / np.sqrt(k)


def team_long(f):
    """One row per (game, team) so a team's games can be scored from its own
    perspective regardless of venue."""
    h = f.assign(team=f.home, opp=f.away, is_home=1,
                 resid=f.margin_actual - f.close_margin,
                 resid_open=f.margin_actual - f.open_margin,
                 mkt=f.close_margin, act=f.margin_actual)
    a = f.assign(team=f.away, opp=f.home, is_home=0,
                 resid=-(f.margin_actual - f.close_margin),
                 resid_open=-(f.margin_actual - f.open_margin),
                 mkt=-f.close_margin, act=-f.margin_actual)
    cols = ["season", "game_id", "game_date", "team", "opp", "is_home",
            "resid", "resid_open", "mkt", "act", "days_in"]
    return pd.concat([h[cols], a[cols]], ignore_index=True)


def main():
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
    f["game_id"] = zf(f["game_id"])
    f["game_date"] = pd.to_datetime(f["game_date"])
    f = f.dropna(subset=["margin_actual", "close_margin", "open_margin"]).copy()
    f["days_in"] = (f["game_date"]
                    - f.groupby("season")["game_date"].transform("min")).dt.days
    seasons = sorted(f.season.unique())
    print(f"frame {len(f):,} games, {len(seasons)} seasons "
          f"({seasons[0]} .. {seasons[-1]})")
    L = team_long(f)
    out = {}

    # ================================================================
    # T0  IS 2024-25 ANOMALOUS AT ALL?  Gate for the whole enquiry.
    # ================================================================
    print("\n" + "=" * 72)
    print("T0  IS 2024-25 ACTUALLY AN OUTLIER, OR IS IT SAMPLING NOISE?")
    print("=" * 72)
    led = json.load(open(ROOT / "data" / "wf_perbet_OFFSET.json"))["k=1 raw"]
    b = pd.DataFrame(led)
    bcol = [c for c in b.columns if c.lower() in
            ("season", "yr", "year")]
    print(f"  ledger {len(b)} bets, columns {list(b.columns)[:12]}")
    if bcol:
        sc = bcol[0]
        pcol = [c for c in b.columns
                if c.lower() in ("pnl", "profit", "ret", "roi", "units")]
        if pcol:
            pc = pcol[0]
            g = b.groupby(sc)[pc].agg(["mean", "count", "std"])
            g["roi_pct"] = 100 * g["mean"]
            print("\n  per-season ROI on the production ledger:")
            print(g.to_string(float_format=lambda v: f"{v:9.4f}"))
            pooled = b[pc].mean()
            print(f"\n  pooled ROI {100*pooled:+.2f}%")
            # Is the best season further from pooled than luck allows?
            # Per-season SE of the mean return, then a max-statistic
            # permutation that reshuffles the season label across bets.
            rng = np.random.default_rng(247)
            obs = g["roi_pct"].max()
            best_season = g["roi_pct"].idxmax()
            vals = b[pc].to_numpy(float)
            sizes = g["count"].to_numpy(int)
            null_max = np.empty(20000)
            for i in range(20000):
                perm = rng.permutation(vals)
                cuts = np.cumsum(sizes)[:-1]
                null_max[i] = max(100 * s.mean()
                                  for s in np.split(perm, cuts))
            p = float((null_max >= obs).mean())
            print(f"  best season {best_season} at {obs:+.2f}%")
            print(f"  max-statistic permutation p = {p:.4f} "
                  f"(20,000 reshuffles of the season label)")
            print(f"  null max-season ROI: median {np.median(null_max):+.2f}%, "
                  f"95th pct {np.percentile(null_max,95):+.2f}%")
            verdict = ("ANOMALY -- the best season exceeds what reshuffling "
                       "produces" if p < 0.05 else
                       "NO ANOMALY -- a season this good arises from "
                       "reshuffling alone; there is nothing to explain")
            print(f"  >>> {verdict}")
            out["T0"] = dict(best_season=str(best_season), best_roi=float(obs),
                             p=p, pooled_roi=float(100 * pooled),
                             null_p95=float(np.percentile(null_max, 95)))

    # ================================================================
    # T1  DEN's home deviation, rebuilt from data across all 19 seasons
    # ================================================================
    print("\n" + "=" * 72)
    print("T1  DENVER HOME: RAW EDGE vs MARKET RESIDUAL")
    print("=" * 72)
    print("  raw  = DEN home margin minus the league-average home margin")
    print("  res  = DEN home margin minus the CLOSING line (what we could bet)")
    lg = f.groupby("season")["margin_actual"].mean()
    for tm in ALTITUDE:
        r = []
        for s in seasons:
            hs = f[(f.season == s) & (f.home == tm)]
            if len(hs) < 10:
                continue
            r.append(dict(season=s, n=len(hs),
                          raw=float(hs.margin_actual.mean() - lg[s]),
                          res=float((hs.margin_actual - hs.close_margin).mean())))
        d = pd.DataFrame(r)
        print(f"\n  --- {tm} home ---")
        print(d.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
        m, lo, hi, t, k = clus(d.res)
        print(f"  {tm} home MARKET RESIDUAL pooled {m:+.3f} "
              f"CI [{lo:+.3f}, {hi:+.3f}] t {t:+.2f} k={k}  "
              f"{'SIG' if lo > 0 or hi < 0 else 'ns'}")
        out[f"T1_{tm}"] = dict(rows=r, pooled=float(m),
                               ci=[float(lo), float(hi)], k=int(k))
        # post-title split for DEN specifically
        if tm == "DEN":
            pre = d[d.season < "2022-23"]
            post = d[d.season >= "2022-23"]
            print(f"  DEN raw   pre-title {pre.raw.mean():+.3f}  "
                  f"title-onward {post.raw.mean():+.3f}  "
                  f"delta {post.raw.mean()-pre.raw.mean():+.3f}")
            print(f"  DEN resid pre-title {pre.res.mean():+.3f}  "
                  f"title-onward {post.res.mean():+.3f}  "
                  f"delta {post.res.mean()-pre.res.mean():+.3f}")
            out["T1_DEN_split"] = dict(
                raw_pre=float(pre.raw.mean()), raw_post=float(post.raw.mean()),
                res_pre=float(pre.res.mean()), res_post=float(post.res.mean()))

    # ================================================================
    # T2  DEFENDING CHAMPION: does the league adapt and the market miss it?
    # ================================================================
    print("\n" + "=" * 72)
    print("T2  DEFENDING CHAMPION vs THE CLOSING LINE")
    print("=" * 72)
    print("  H1: teams retool to beat the champion; the market prices the")
    print("      champion on last year's form -> champion resid < 0.")
    rows = []
    for i, s in enumerate(seasons):
        prev = seasons[i - 1] if i > 0 else None
        ch = CHAMPION.get(prev) if prev else CHAMPION.get("2006-07") \
            if s == "2007-08" else None
        if s == "2007-08":
            ch = CHAMPION["2006-07"]
        elif prev:
            ch = CHAMPION.get(prev)
        if not ch:
            continue
        g = L[(L.season == s) & (L.team == ch)]
        if len(g) < 20:
            continue
        rows.append(dict(season=s, champ=ch, n=len(g),
                         resid=float(g.resid.mean()),
                         resid_home=float(g[g.is_home == 1].resid.mean()),
                         resid_away=float(g[g.is_home == 0].resid.mean()),
                         mkt=float(g.mkt.mean()), act=float(g.act.mean())))
    dc = pd.DataFrame(rows)
    print(dc.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    m, lo, hi, t, k = clus(dc.resid)
    md = mde80(dc.resid)
    print(f"\n  defending-champion residual vs CLOSE: {m:+.4f} "
          f"CI [{lo:+.4f}, {hi:+.4f}] t {t:+.2f} k={k}")
    print(f"  seasons with resid<0: {int((dc.resid<0).sum())}/{k}")
    print(f"  MDE80 {md:.3f} pts/game -- an effect smaller than this is")
    print(f"        undetectable here regardless of whether it exists")
    v = ("CONFIRMED -- champions underperform the close" if hi < 0 else
         "REFUTED -- champions BEAT the close" if lo > 0 else
         "NULL / UNDERPOWERED -- CI spans zero")
    print(f"  >>> {v}")
    out["T2_champion"] = dict(rows=rows, mean=float(m),
                              ci=[float(lo), float(hi)], k=int(k),
                              mde80=float(md), verdict=v)

    # runner-up arm
    rows = []
    for i, s in enumerate(seasons):
        if i == 0:
            ru = RUNNER_UP.get("2006-07")
        else:
            ru = RUNNER_UP.get(seasons[i - 1])
        if not ru:
            continue
        g = L[(L.season == s) & (L.team == ru)]
        if len(g) < 20:
            continue
        rows.append(dict(season=s, ru=ru, n=len(g), resid=float(g.resid.mean())))
    dr = pd.DataFrame(rows)
    m2, lo2, hi2, t2, k2 = clus(dr.resid)
    print(f"\n  beaten finalist residual vs CLOSE: {m2:+.4f} "
          f"CI [{lo2:+.4f}, {hi2:+.4f}] t {t2:+.2f} k={k2}  "
          f"better {int((dr.resid>0).sum())}/{k2}")
    print("  (limb (b): the team with the strongest motive to retool)")
    out["T2_runnerup"] = dict(rows=rows, mean=float(m2),
                              ci=[float(lo2), float(hi2)], k=int(k2))

    # ================================================================
    # T3  SELF-CORRECTION: does last year's disappointment predict this
    #     year's market residual?
    # ================================================================
    print("\n" + "=" * 72)
    print("T3  DOES LAST SEASON'S RESULT PREDICT THIS SEASON'S MARKET RESIDUAL?")
    print("=" * 72)
    print("  H2: teams fix last year's problems; the market anchors on last")
    print("      year -> prior-season margin should predict resid NEGATIVELY.")
    ts = (L.groupby(["season", "team"])
            .agg(act=("act", "mean"), resid=("resid", "mean"),
                 n=("act", "size")).reset_index())
    ts["prev_act"] = ts.groupby("team")["act"].shift(1)
    ts["prev_season"] = ts.groupby("team")["season"].shift(1)
    ok = ts.dropna(subset=["prev_act"]).copy()
    # only consecutive seasons
    idx = {s: i for i, s in enumerate(seasons)}
    ok = ok[ok.apply(lambda r: idx[r.season] - idx[r.prev_season] == 1, axis=1)]
    print(f"  {len(ok)} team-seasons with a consecutive prior season")
    per = []
    for s, g in ok.groupby("season"):
        if len(g) < 10:
            continue
        b = np.polyfit(g.prev_act, g.resid, 1)[0]
        per.append(dict(season=s, n=len(g), slope=float(b),
                        corr=float(np.corrcoef(g.prev_act, g.resid)[0, 1])))
    dp = pd.DataFrame(per)
    print(dp.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    m3, lo3, hi3, t3, k3 = clus(dp.slope)
    print(f"\n  slope of resid on prior-season margin: {m3:+.5f} "
          f"CI [{lo3:+.5f}, {hi3:+.5f}] t {t3:+.2f} k={k3}")
    print(f"  MDE80 {mde80(dp.slope):.5f}")
    v3 = ("CONFIRMED -- market anchors on last year" if hi3 < 0 else
          "REFUTED -- runs the other way" if lo3 > 0 else
          "NULL -- the close already prices last season's result")
    print(f"  >>> {v3}")
    out["T3_reversion"] = dict(rows=per, slope=float(m3),
                               ci=[float(lo3), float(hi3)], k=int(k3),
                               mde80=float(mde80(dp.slope)), verdict=v3)

    # ================================================================
    # T4  IF adaptation is real and unpriced, the mispricing must be
    #     LARGEST EARLY and decay as the market learns.
    # ================================================================
    print("\n" + "=" * 72)
    print("T4  WITHIN-SEASON DECAY -- the sharpest implication of limb (c)")
    print("=" * 72)
    print("  An offseason change the market has not priced must show up in")
    print("  OCTOBER and shrink by JANUARY. A flat profile refutes the")
    print("  'market has not caught up yet' mechanism regardless of T2/T3.")
    bins = [(0, 14, "first 2 weeks"), (14, 30, "wks 3-4"),
            (30, 60, "month 2"), (60, 120, "months 3-4"),
            (120, 999, "months 5+")]
    f2 = f.copy()
    f2["abs_res"] = (f2.margin_actual - f2.close_margin).abs()
    f2["gap_open_close"] = (f2.close_margin - f2.open_margin).abs()
    rows = []
    for lo_, hi_, lab in bins:
        sub = f2[(f2.days_in >= lo_) & (f2.days_in < hi_)]
        per_s = sub.groupby("season").agg(
            abs_res=("abs_res", "mean"), mv=("gap_open_close", "mean"))
        m_, l_, h_, t_, k_ = clus(per_s.abs_res)
        mm, ml, mh, mt, mk = clus(per_s.mv)
        rows.append(dict(window=lab, n=len(sub),
                         mkt_abs_err=float(m_), ci=[float(l_), float(h_)],
                         open_to_close=float(mm)))
        print(f"  {lab:14} n={len(sub):6,}  market |error| {m_:6.3f} "
              f"CI [{l_:6.3f},{h_:6.3f}]   open->close move {mm:5.3f}")
    print("\n  If the market were uninformed early, its absolute error and the")
    print("  open->close move would both be LARGEST in the first two weeks.")
    out["T4_within_season"] = rows

    json.dump(out, open(ROOT / "data" / "d247_adaptation.json", "w"),
              default=float)
    print("\nwrote data/d247_adaptation.json")


if __name__ == "__main__":
    main()
