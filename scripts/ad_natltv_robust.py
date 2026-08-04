#!/usr/bin/env python3
"""NATIONAL-TV DNP SUPPRESSION — falsification and decomposition.

The headline (-0.0219 within player-season) is a MEASUREMENT, not a gate, so
what it needs is not more inference but attempts to break it:

  P1 PLACEBO — permute the national-TV flag WITHIN player-season. The estimator
     must return 0. If it does not, the FE machinery is manufacturing the effect.
  P2 WEEKDAY — national games cluster on particular weekdays, which carry
     different rest/travel patterns. Add weekday fixed effects.
  P3 BROADCASTER — ABC/ESPN/TNT are marquee; NBA TV is not. A behavioural
     "showcase" story predicts a gradient; a pure PPP story does not (the policy
     names national television, not the network).
  P4 STAR TIER — the Player Participation Policy applies to designated STARS.
     If the suppression is the policy, it must be CONCENTRATED in the top tier.
  P5 CALENDAR — the effect must not be a Christmas/opening-week artefact.

Writes data/ad_natltv_robust.json. DB read_only=True.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from nbapred.db import connect

NB = 2000
SEED = 20260801


def demean(v, key):
    s = pd.Series(np.asarray(v, float))
    return (s - s.groupby(np.asarray(key)).transform("mean")).to_numpy()


def fe_beta(y, x, key):
    yd, xd = demean(y, key), demean(x, key)
    den = float(np.dot(xd, xd))
    return float(np.dot(xd, yd) / den) if den > 0 else np.nan


def fe_beta_ci(y, x, key, groups, B=NB, seed=SEED):
    uniq, inv = np.unique(groups, return_inverse=True)
    idxby = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    pt = fe_beta(y, x, key)
    bs = np.empty(B)
    y = np.asarray(y, float); x = np.asarray(x, float); key = np.asarray(key)
    for b in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idxby[i] for i in pick])
        bs[b] = fe_beta(y[sel], x[sel], key[sel])
    bs = bs[np.isfinite(bs)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return pt, float(lo), float(hi), float(bs.std(ddof=1))


def sig(lo, hi):
    return "SIG" if (lo > 0 or hi < 0) else "ns"


def main():
    p = pd.read_csv(ROOT / "data" / "ad_natltv_panel.csv.gz")
    p["gd"] = pd.to_datetime(p.game_date)
    p["dow"] = p.gd.dt.dayofweek
    y = p.dnp.to_numpy(float)
    x = p.is_natl_tv.to_numpy(float)
    ps = p.ps.to_numpy()
    pid = p.player_id.to_numpy()
    out = {}

    base = fe_beta_ci(y, x, ps, pid)
    out["baseline_FE"] = dict(delta=base[0], lo=base[1], hi=base[2], se=base[3])
    print(f"baseline FE {base[0]:+.5f} CI[{base[1]:+.5f},{base[2]:+.5f}] "
          f"{sig(base[1],base[2])}")

    # ---- P1 PLACEBO: permute the flag within player-season
    rng = np.random.default_rng(SEED)
    ests = []
    for rep in range(200):
        xp = np.empty_like(x)
        for _, idx in pd.Series(range(len(p))).groupby(ps).groups.items():
            i = np.asarray(list(idx))
            xp[i] = rng.permutation(x[i])
        ests.append(fe_beta(y, xp, ps))
    ests = np.array(ests)
    out["P1_placebo"] = dict(n_rep=200, mean=float(ests.mean()),
                             sd=float(ests.std(ddof=1)),
                             q025=float(np.percentile(ests, 2.5)),
                             q975=float(np.percentile(ests, 97.5)),
                             p_more_extreme=float((ests <= base[0]).mean()))
    print(f"P1 PLACEBO (200 within-player-season permutations): mean "
          f"{ests.mean():+.6f} sd {ests.std(ddof=1):.6f}  null band "
          f"[{np.percentile(ests,2.5):+.5f},{np.percentile(ests,97.5):+.5f}]  "
          f"P(placebo <= observed) = {(ests<=base[0]).mean():.4f}")

    # ---- P2 WEEKDAY FE on top of player-season FE
    D = pd.get_dummies(p.dow, prefix="d", drop_first=True).to_numpy(float)
    yd, xd = demean(y, ps), demean(x, ps)
    Dd = np.column_stack([demean(D[:, j], ps) for j in range(D.shape[1])])
    X = np.column_stack([xd, Dd, np.ones(len(yd))])
    b = np.linalg.lstsq(X, yd, rcond=None)[0][0]
    # cluster bootstrap
    uniq, inv = np.unique(pid, return_inverse=True)
    idxby = [np.where(inv == i)[0] for i in range(len(uniq))]
    rg = np.random.default_rng(SEED)
    bs = np.empty(NB)
    for i in range(NB):
        pick = rg.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idxby[j] for j in pick])
        try:
            bs[i] = np.linalg.lstsq(X[sel], yd[sel], rcond=None)[0][0]
        except np.linalg.LinAlgError:
            bs[i] = np.nan
    bs = bs[np.isfinite(bs)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    out["P2_weekday_FE"] = dict(delta=float(b), lo=float(lo), hi=float(hi))
    print(f"P2 +weekday FE {b:+.5f} CI[{lo:+.5f},{hi:+.5f}] {sig(lo,hi)}")

    # ---- P3 BROADCASTER gradient
    print("P3 BROADCASTER:")
    out["P3_broadcaster"] = {}
    for net in sorted(p.natl_tv.dropna().unique()):
        m = (p.natl_tv.isna() | (p.natl_tv == net)).to_numpy()
        if m.sum() < 500 or (p.natl_tv[m] == net).sum() < 100:
            continue
        r = fe_beta_ci(y[m], x[m], ps[m], pid[m], B=800)
        out["P3_broadcaster"][net] = dict(n_natl=int((p.natl_tv == net).sum()),
                                          delta=r[0], lo=r[1], hi=r[2])
        print(f"   {net:6s} n_natl={int((p.natl_tv==net).sum()):5d} "
              f"{r[0]:+.5f} CI[{r[1]:+.5f},{r[2]:+.5f}] {sig(r[1],r[2])}")

    # ---- P4 STAR TIER (PPP applies to designated stars)
    con = connect(read_only=True)
    mpg = con.execute("""
        WITH ps AS (
          SELECT g.season, p.player_id, avg(p.seconds) sec, count(*) gp
          FROM player_game_stats p
          JOIN (SELECT DISTINCT game_id, season FROM nba_games
                WHERE game_id LIKE '002%') g USING (game_id)
          WHERE p.seconds > 0 GROUP BY 1,2)
        SELECT season, player_id, sec/60.0 mpg FROM ps
    """).fetchdf()
    con.close()
    p2 = p.merge(mpg, on=["season", "player_id"], how="left")
    p2["tier"] = pd.qcut(p2.mpg, 3, labels=["core-lo", "core-mid", "core-hi"])
    print("P4 STAR TIER (by season mpg tercile within the core cohort):")
    out["P4_tier"] = {}
    for t in ("core-lo", "core-mid", "core-hi"):
        m = (p2.tier == t).to_numpy()
        r = fe_beta_ci(y[m], x[m], ps[m], pid[m], B=800)
        out["P4_tier"][t] = dict(n=int(m.sum()), mpg=float(p2.mpg[m].mean()),
                                 delta=r[0], lo=r[1], hi=r[2])
        print(f"   {t:8s} n={m.sum():6d} mpg={p2.mpg[m].mean():.1f} "
              f"{r[0]:+.5f} CI[{r[1]:+.5f},{r[2]:+.5f}] {sig(r[1],r[2])}")

    # ---- P5 CALENDAR: drop Christmas + opening week + drop each month
    xmas = ((p.gd.dt.month == 12) & (p.gd.dt.day.between(24, 26))).to_numpy()
    early = (p.tgp < 5).to_numpy()
    for lab, keep in (("drop Christmas", ~xmas),
                      ("drop first 5 team games", ~early),
                      ("drop both", ~(xmas | early))):
        r = fe_beta_ci(y[keep], x[keep], ps[keep], pid[keep], B=800)
        out.setdefault("P5_calendar", {})[lab] = dict(n=int(keep.sum()),
                                                      delta=r[0], lo=r[1], hi=r[2])
        print(f"P5 {lab:24s} n={keep.sum():6d} {r[0]:+.5f} "
              f"CI[{r[1]:+.5f},{r[2]:+.5f}] {sig(r[1],r[2])}")

    (ROOT / "data" / "ad_natltv_robust.json").write_text(
        json.dumps(out, indent=2, default=float))
    print("AD_NATLTV_ROBUST_DONE")


if __name__ == "__main__":
    main()
