"""HA-(3) THE LAKERS / COACH-ERA QUESTION.

Sean's claim: team-specific home advantage is real, "the Lakers at home under
JJ Redick". Redick became LAL head coach for 2024-25.

Tests, all with opponent quality AND own quality controlled (season team FE,
so a good team's home wins are not counted as home advantage), plus schedule
controls:
  (3a) LAL raw home / road record and margin by season.
  (3b) LAL controlled home deviation d_t by season with SE.
  (3c) coach-era contrast: 2019-20..2023-24 (Vogel/Ham/Redick-pre) vs
       2024-25..2025-26 (Redick), difference with a bootstrap CI.
  (3d) the same era contrast for the 4 other teams with the largest apparent
       home edge, so LAL can be compared to the rest of the top of the draw.
  (3e) the honest control: how often does a RANDOM team, under the null of zero
       true team-specific home advantage, look at least as extreme as LAL?
       Answered by permutation.

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

from ha_core import CONTROLS, boot_ci, eb_shrink, fit_season, load_panel

SEED = 20260801
OUT = Path("/tmp/claude-1004/-hdd-steveqin-sean-dev/"
           "4912ce58-69cd-488d-a5f4-0d3ed2eed30c/scratchpad")
SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
NORMAL = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
REDICK = ["2024-25", "2025-26"]
PRE = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]


def era_dev(d, seasons, team_cols=None):
    """Fit each season separately, return the d_t matrix restricted to seasons."""
    out = {}
    for s in seasons:
        f = fit_season(d[d.season == s], CONTROLS)
        out[s] = pd.Series(f["d"], index=f["teams"])
    return pd.DataFrame(out)


def main():
    rng = np.random.default_rng(SEED)
    d = load_panel()

    # ---------- (3a) raw LAL --------------------------------------------
    print("=== (3a) LAL RAW, by season (regular season, neutral games dropped) ===")
    rows = []
    for s in SEASONS:
        sub = d[d.season == s]
        h = sub[sub.home == "LAL"]
        a = sub[sub.away == "LAL"]
        rows.append(dict(season=s, coach=("Redick" if s in REDICK else "pre"),
                         n_home=len(h), home_margin=h.margin.mean(),
                         home_wr=(h.margin > 0).mean(),
                         n_road=len(a), road_margin=-a.margin.mean(),
                         road_wr=(a.margin < 0).mean(),
                         overall=(h.margin.sum() - a.margin.sum()) / (len(h) + len(a)),
                         split=h.margin.mean() + a.margin.mean()))
    lal = pd.DataFrame(rows)
    print(lal.round(3).to_string(index=False))
    print("  'split' = home margin + away margin = 2x the raw uncontrolled HFA "
          "for LAL (opponent set NOT yet controlled)")

    # ---------- (3b) controlled deviation -------------------------------
    D = era_dev(d, SEASONS)
    SE = {}
    for s in SEASONS:
        f = fit_season(d[d.season == s], CONTROLS)
        SE[s] = pd.Series(f["se_d"], index=f["teams"])
    SE = pd.DataFrame(SE)
    print("\n=== (3b) CONTROLLED HOME DEVIATION d_t, LAL vs the league ===")
    lalrow = pd.DataFrame({"LAL d_t": D.loc["LAL"], "SE": SE.loc["LAL"],
                           "z": D.loc["LAL"] / SE.loc["LAL"],
                           "league rank (1=best)": [
                               int((D[s] > D.loc["LAL", s]).sum() + 1) for s in SEASONS]})
    print(lalrow.round(3).to_string())

    # ---------- (3c) coach-era contrast ---------------------------------
    print("\n=== (3c) COACH-ERA CONTRAST (Redick 2024-25,2025-26 vs prior) ===")
    # pooled fit within each era: era-specific team-strength FE by season,
    # era-specific home deviation. Use a single regression per era.
    def era_fit(seasons):
        sub = d[d.season.isin(seasons)].reset_index(drop=True)
        teams = sorted(set(sub.home) | set(sub.away))
        seas = sorted(sub.season.unique())
        ts = [(t, y) for y in seas for t in teams]
        tsi = {k: i for i, k in enumerate(ts)}
        n = len(sub)
        Z = np.zeros((n, len(ts)))
        r = np.arange(n)
        Z[r, [tsi[(h, y)] for h, y in zip(sub.home, sub.season)]] += 1
        Z[r, [tsi[(a, y)] for a, y in zip(sub.away, sub.season)]] -= 1
        H = np.zeros((n, len(teams)))
        ti = {t: i for i, t in enumerate(teams)}
        H[r, sub.home.map(ti).to_numpy()] = 1
        C = sub[CONTROLS].to_numpy(float)
        X = np.hstack([Z, H, C])
        y = sub.margin.to_numpy(float)
        cf, *_ = np.linalg.lstsq(X, y, rcond=None)
        a = cf[len(ts):len(ts) + len(teams)]
        return pd.Series(a - a.mean(), index=teams), sub

    d_red, _ = era_fit(REDICK)
    d_pre, _ = era_fit(PRE)
    d_pre_norm, _ = era_fit(["2021-22", "2022-23", "2023-24"])

    # bootstrap the LAL era difference
    def boot_era(team, B=2000):
        sr = d[d.season.isin(REDICK)].reset_index(drop=True)
        sp = d[d.season.isin(PRE)].reset_index(drop=True)
        outv = []
        for _ in range(B):
            a1 = _refit_one(sr.iloc[rng.integers(0, len(sr), len(sr))], team)
            a0 = _refit_one(sp.iloc[rng.integers(0, len(sp), len(sp))], team)
            if a1 is None or a0 is None:
                continue
            outv.append(a1 - a0)
        return np.array(outv)

    def _refit_one(sub, team):
        teams = sorted(set(sub.home) | set(sub.away))
        if team not in teams:
            return None
        seas = sorted(sub.season.unique())
        ts = [(t, y) for y in seas for t in teams]
        tsi = {k: i for i, k in enumerate(ts)}
        n = len(sub)
        Z = np.zeros((n, len(ts)))
        r = np.arange(n)
        try:
            Z[r, [tsi[(h, y)] for h, y in zip(sub.home, sub.season)]] += 1
            Z[r, [tsi[(a, y)] for a, y in zip(sub.away, sub.season)]] -= 1
        except KeyError:
            return None
        H = np.zeros((n, len(teams)))
        ti = {t: i for i, t in enumerate(teams)}
        H[r, sub.home.map(ti).to_numpy()] = 1
        X = np.hstack([Z, H, sub[CONTROLS].to_numpy(float)])
        cf, *_ = np.linalg.lstsq(X, sub.margin.to_numpy(float), rcond=None)
        a = cf[len(ts):len(ts) + len(teams)]
        a = a - a.mean()
        return a[ti[team]]

    # top-5 apparent home edge over the normal seasons
    d_norm_all, _ = era_fit(NORMAL)
    top = d_norm_all.sort_values(ascending=False)
    print("\n  TOP-8 apparent home deviation, pooled over 5 normal seasons (pts):")
    print(top.head(8).round(3).to_string())
    print("  BOTTOM-5:")
    print(top.tail(5).round(3).to_string())
    print(f"  LAL = {d_norm_all['LAL']:+.3f} pts, rank "
          f"{int((d_norm_all > d_norm_all['LAL']).sum()+1)}/30")

    focus = ["LAL"] + [t for t in top.head(5).index if t != "LAL"][:4]
    print(f"\n  ERA CONTRAST for LAL + the 4 largest apparent home edges: {focus}")
    era_rows = []
    for t in focus:
        dif = boot_era(t, B=1200)
        lo, hi = boot_ci(dif)
        era_rows.append(dict(team=t, pre=d_pre.get(t, np.nan),
                             redick_era=d_red.get(t, np.nan),
                             delta=d_red.get(t, np.nan) - d_pre.get(t, np.nan),
                             ci_lo=lo, ci_hi=hi,
                             sig=("SIG" if lo > 0 or hi < 0 else "NS")))
    et = pd.DataFrame(era_rows)
    print(et.round(3).to_string(index=False))

    # ---------- (3d) is LAL's level distinguishable from league average? ---
    print("\n=== (3d) IS LAL's HOME EDGE DISTINGUISHABLE FROM LEAGUE AVERAGE? ===")
    for label, seasons in (("Redick era (24-25,25-26)", REDICK),
                           ("normal 5 (21-22..25-26)", NORMAL),
                           ("all 7", SEASONS)):
        bs = []
        sub0 = d[d.season.isin(seasons)].reset_index(drop=True)
        for _ in range(1200):
            v = _refit_one(sub0.iloc[rng.integers(0, len(sub0), len(sub0))], "LAL")
            if v is not None:
                bs.append(v)
        bs = np.array(bs)
        pt = _refit_one(sub0, "LAL")
        lo, hi = boot_ci(bs)
        n_home = (sub0.home == "LAL").sum()
        print(f"  {label:26s} LAL d = {pt:+.3f} CI({lo:+.3f},{hi:+.3f}) "
              f"n_home={n_home}  {'SIG' if lo>0 or hi<0 else 'NS vs league average'}")

    # ---------- (3e) permutation: how extreme is the top of the draw? -----
    print("\n=== (3e) PERMUTATION NULL — under ZERO true team-specific home "
          "advantage, how big is the biggest apparent home edge? ===")
    # Null: shuffle which of a team's games are 'home' *within the same
    # opponent-adjusted residual pool*, preserving the schedule structure.
    sub = d[d.season.isin(NORMAL)].reset_index(drop=True)
    f_all, _ = era_fit(NORMAL)
    obs_max = float(f_all.max())
    obs_sd = float(f_all.std(ddof=1))
    # residual pool with the home effect removed
    teams = sorted(set(sub.home) | set(sub.away))
    seas = sorted(sub.season.unique())
    ts = [(t, y) for y in seas for t in teams]
    tsi = {k: i for i, k in enumerate(ts)}
    n = len(sub)
    Z = np.zeros((n, len(ts)))
    r = np.arange(n)
    Z[r, [tsi[(h, y)] for h, y in zip(sub.home, sub.season)]] += 1
    Z[r, [tsi[(a, y)] for a, y in zip(sub.away, sub.season)]] -= 1
    H = np.zeros((n, len(teams)))
    ti = {t: i for i, t in enumerate(teams)}
    H[r, sub.home.map(ti).to_numpy()] = 1
    C = sub[CONTROLS].to_numpy(float)
    X = np.hstack([Z, H, C])
    y = sub.margin.to_numpy(float)
    cf, *_ = np.linalg.lstsq(X, y, rcond=None)
    a_hat = cf[len(ts):len(ts) + len(teams)]
    fitted_nohome = X @ cf - H @ a_hat + a_hat.mean()   # league HFA kept, dev removed
    resid = y - X @ cf
    maxes, sds = [], []
    for _ in range(2000):
        ysim = fitted_nohome + resid[rng.integers(0, n, n)]
        cs, *_ = np.linalg.lstsq(X, ysim, rcond=None)
        aa = cs[len(ts):len(ts) + len(teams)]
        aa = aa - aa.mean()
        maxes.append(aa.max()); sds.append(aa.std(ddof=1))
    maxes = np.array(maxes); sds = np.array(sds)
    print(f"  OBSERVED  max d = {obs_max:+.3f}   sd(d) = {obs_sd:.3f}")
    print(f"  NULL      max d = {maxes.mean():+.3f} "
          f"[p5,p95]=({np.percentile(maxes,5):+.3f},{np.percentile(maxes,95):+.3f})"
          f"   sd(d) = {sds.mean():.3f} "
          f"[p5,p95]=({np.percentile(sds,5):.3f},{np.percentile(sds,95):.3f})")
    print(f"  p(null max >= observed max) = {(maxes >= obs_max).mean():.3f}")
    print(f"  p(null sd  >= observed sd ) = {(sds >= obs_sd).mean():.3f}")

    res = dict(lal_raw=lal.to_dict("records"),
               lal_dev=lalrow.reset_index().to_dict("records"),
               era=et.to_dict("records"),
               top=top.head(8).to_dict(), bottom=top.tail(5).to_dict(),
               perm=dict(obs_max=obs_max, obs_sd=obs_sd,
                         null_max=float(maxes.mean()), null_sd=float(sds.mean()),
                         p_max=float((maxes >= obs_max).mean()),
                         p_sd=float((sds >= obs_sd).mean())))
    (OUT / "ha_lakers.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {OUT/'ha_lakers.json'}")


if __name__ == "__main__":
    main()
