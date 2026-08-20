#!/usr/bin/env python3
"""D252b — D252 rerun on the SHIPPED OFFSET margin, and a faster family null.

D252 scored `p_us` from the frame. That column is exactly
`sigmoid(m_us696 / 6.96)` (verified to 2.2e-16) and equals
`sigmoid(m_us_blind / 7.2)` on 75.2% of games -- i.e. it is the MARKET-BLIND
model's probability, not the offset model's. So D252 measured where the BLIND
model stands against the close. The question asked was about the shipped
forecast, so this rescores every slice on `m_us`, the offset margin, at the same
6.96 scale used for the market. Same slices, same family null, fixed before
reading, no additions.

The permutation is vectorised (factorise + bincount) so the family null runs at
2,000 draws in seconds rather than minutes.
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

SCALE = 6.96


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def nll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def clus(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    k = len(v)
    if k < 2:
        return np.nan, np.nan, np.nan, k
    se = v.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return v.mean(), v.mean() - tc * se, v.mean() + tc * se, k


def season_then_group_mean(vals, gcode, scode, ng, ns):
    """mean over seasons of the per-(group,season) mean -- the clustered stat."""
    key = gcode * ns + scode
    s = np.bincount(key, weights=vals, minlength=ng * ns)
    c = np.bincount(key, minlength=ng * ns).astype(float)
    cell = np.where(c > 0, s / np.maximum(c, 1), np.nan).reshape(ng, ns)
    with np.errstate(invalid="ignore"):
        return np.nanmean(cell, axis=1)


def main():
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
    f["game_id"] = zf(f["game_id"])
    f["game_date"] = pd.to_datetime(f["game_date"])
    f = f.dropna(subset=["open_margin", "close_margin", "margin_actual",
                         "m_us", "m_us_blind"]).copy()
    st = pd.read_csv(ROOT / "data" / "d252_stars.csv.gz")
    st["game_id"] = zf(st["game_id"])
    hs = st.merge(f[["game_id", "home"]], on="game_id")
    hs = hs[hs.ab == hs.home].set_index("game_id").n_star_out
    as_ = st.merge(f[["game_id", "away"]], on="game_id")
    as_ = as_[as_.ab == as_.away].set_index("game_id").n_star_out
    f = f.set_index("game_id")
    f["stars_out"] = (hs.reindex(f.index) + as_.reindex(f.index))
    f = f.reset_index().dropna(subset=["stars_out"])

    y = (f.margin_actual > 0).astype(float).to_numpy()
    def ll(m):
        return nll(1 / (1 + np.exp(-np.asarray(m, float) / SCALE)), y)
    f["ll_us"] = ll(f.m_us)
    f["ll_close"] = ll(f.close_margin)
    f["ll_open"] = ll(f.open_margin)
    f["d_vs_close"] = f.ll_us - f.ll_close
    f["d_vs_open"] = f.ll_us - f.ll_open
    print(f"{len(f):,} games, offset margin at scale {SCALE}\n")

    print("=" * 74)
    print("Q1  STARS OUT — offset margin (negative = we beat the market)")
    print("=" * 74)
    print(f"  {'stars out':10} {'n':>6} {'ll_mkt':>8} {'vs close':>11} {'vs open':>11}")
    for k, g in f.groupby(f.stars_out.clip(upper=3)):
        pc = g.groupby("season").d_vs_close.mean()
        po = g.groupby("season").d_vs_open.mean()
        mc, *_ = clus(pc); mo, *_ = clus(po)
        lm = g.ll_close.mean()
        print(f"  {int(k)}{'+' if k==3 else '':9} {len(g):6} {lm:8.4f} "
              f"{mc:+11.5f} {mo:+11.5f}")
    per = (f.assign(grp=f.stars_out >= 1)
             .groupby(["season", "grp"]).d_vs_close.mean().unstack())
    m, lo, hi, k = clus(per[True] - per[False])
    print(f"\n  gap(>=1 out) - gap(none) vs CLOSE: {m:+.5f} "
          f"CI [{lo:+.5f}, {hi:+.5f}] k={k}  "
          f"{'SIG' if (hi<0 or lo>0) else 'ns'}")
    per = (f.assign(grp=f.stars_out >= 1)
             .groupby(["season", "grp"]).d_vs_open.mean().unstack())
    m2, lo2, hi2, _ = clus(per[True] - per[False])
    print(f"  gap(>=1 out) - gap(none) vs OPEN : {m2:+.5f} "
          f"CI [{lo2:+.5f}, {hi2:+.5f}]  "
          f"{'SIG' if (hi2<0 or lo2>0) else 'ns'}")

    print("\n" + "=" * 74)
    print("Q2  SLICES — offset margin vs the CLOSE, then vs the OPEN")
    print("=" * 74)
    f["absopen"] = f.open_margin.abs()
    f["absedge"] = (f.m_us_blind - f.open_margin).abs()
    f["days_in"] = (f.game_date - f.groupby("season")
                    .game_date.transform("min")).dt.days
    tot = pd.to_numeric(f.open_total, errors="coerce")
    f["totc"] = np.where((tot < 150) | (tot > 290), np.nan, tot)
    fam = {
        "|open| band": pd.cut(f.absopen, [-.01, 2, 5, 8, 12, 99],
                              labels=["0-2", "2-5", "5-8", "8-12", "12+"]),
        "stars out": f.stars_out.clip(upper=2).map({0: "none", 1: "one",
                                                    2: "two+"}),
        "|our edge| quintile": pd.qcut(f.absedge, 5,
                                       labels=[f"Q{i}" for i in range(1, 6)],
                                       duplicates="drop"),
        "total band": pd.qcut(f.totc, 4, labels=["low", "midlo", "midhi",
                                                 "high"], duplicates="drop"),
        "month of season": pd.cut(f.days_in, [-1, 30, 60, 120, 999],
                                  labels=["m1", "m2", "m3-4", "m5+"]),
    }
    obs_max, res = 0.0, []
    for name, lab in fam.items():
        dd = f.assign(_g=lab).dropna(subset=["_g"])
        print(f"\n  --- {name} ---")
        for g, sub in dd.groupby("_g", observed=True):
            mc, lc, hc, k = clus(sub.groupby("season").d_vs_close.mean())
            mo, lo_, ho, _ = clus(sub.groupby("season").d_vs_open.mean())
            fc = "SIG" if (hc < 0 or lc > 0) else ""
            fo = "SIG" if (ho < 0 or lo_ > 0) else ""
            print(f"    {str(g):10} n={len(sub):6}  vs close {mc:+.5f} "
                  f"[{lc:+.5f},{hc:+.5f}] {fc:3}   vs open {mo:+.5f} "
                  f"[{lo_:+.5f},{ho:+.5f}] {fo}")
            obs_max = max(obs_max, abs(mc))
            res.append(dict(family=name, level=str(g), n=len(sub),
                            vs_close=float(mc), vs_open=float(mo)))

    # -------- vectorised family-wide null --------
    scode, seas = pd.factorize(f.season)
    ns = len(seas)
    vals = f.d_vs_close.to_numpy(float)
    packs = []
    for name, lab in fam.items():
        s = pd.Series(lab).astype("object")
        ok = s.notna().to_numpy()
        gc, gl = pd.factorize(s[ok])
        packs.append((ok, gc, len(gl)))
    rng = np.random.default_rng(252)
    idx_by_season = [np.flatnonzero(scode == i) for i in range(ns)]
    null = np.empty(2000)
    for it in range(2000):
        perm = np.empty(len(vals), int)
        for ix in idx_by_season:
            perm[ix] = rng.permutation(ix)
        mx = 0.0
        for ok, gc, ng in packs:
            v = vals[perm][ok]
            gm = season_then_group_mean(v, gc, scode[ok], ng, ns)
            mx = max(mx, float(np.nanmax(np.abs(gm))))
        null[it] = mx
    p = float((null >= obs_max).mean())
    print("\n  --- FAMILY-WIDE NULL (2,000 within-season shuffles) ---")
    print(f"    observed max |gap vs close| {obs_max:.5f}")
    print(f"    null max: median {np.median(null):.5f}, "
          f"95th {np.percentile(null,95):.5f}")
    print(f"    p = {p:.4f}  "
          f"{'A REAL POCKET' if p < 0.05 else 'NO SLICE BEATS THE FAMILY NULL'}")

    json.dump({"slices": res, "family_p": p, "obs_max": float(obs_max)},
              open(ROOT / "data" / "d252b_edge.json", "w"), default=float)
    print("\nwrote data/d252b_edge.json")


if __name__ == "__main__":
    main()
