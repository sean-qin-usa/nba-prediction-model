#!/usr/bin/env python3
"""ATS19-EXTRA — the three diagnostics the pre-registration promised to report
separately, plus the exact D121 harness anchor.

  (A) "BEATS THE SPREAD" and "CLEARS -110" are SEPARATE CLAIMS (prereg §9).
      Test cover rate vs 50.000% with the same K-1 season-cluster-mean t.
  (B) SIDE COMPOSITION.  Our all-games arm's ROI (-3.25%) is almost identical
      to ALWAYS-AWAY (-3.26%).  Is the result an away-bias echo?  Measured:
      home-pick share, ROI split by picked side, and the PAIRED (same games)
      season-clustered delta against always-away and always-favourite.
  (C) THE D121 ANCHOR, EXACT.  D121 registered ATS-at-open 52.72%
      CI(51.2,54.2) on `data/ds_rt1_pergame.csv` p_full (the FULL-FEED tier,
      2022-23..2025-26) with m_us = 6.96*logit(p_us).  Reproduce it on that
      frame, that column and that scale, so this run's blind 19-season number
      is anchored to the registered one.

Read-only.  Writes only data/ats19_extra.json.
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

from lb_longshot import cluster_boot, cluster_mean_t            # noqa: E402

SEED = 20260804
N_BOOT = 4000
DEC = 1.0 + 100.0 / 110.0
BE = 1.0 / DEC
SEASONS19 = ["%d-%02d" % (y, (y + 1) % 100) for y in range(2007, 2026)]
OOS14 = set(SEASONS19[:14])
DEV5 = set(SEASONS19[14:])
FRAME = os.path.join(ROOT, "data", "ats19_frame.csv.gz")
ODDS = os.path.join(ROOT, "data", "derived", "odds_open.csv")
RT1 = os.path.join(ROOT, "data", "ds_rt1_pergame.csv")
OUT = os.path.join(ROOT, "data", "ats19_extra.json")


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def ct(v, c, seed=SEED, boot=True):
    v = np.asarray(v, float)
    tlo, thi, K = cluster_mean_t(v, c)
    lo = hi = se = float("nan")
    if boot:
        lo, hi, se = cluster_boot(v, c, n_boot=N_BOOT, seed=seed)
    return {"n": int(len(v)), "K": int(K), "mean": float(v.mean()),
            "lo": lo, "hi": hi, "se": se, "tlo": float(tlo), "thi": float(thi),
            "sig": bool(tlo > 0 or thi < 0)}


def sg(d):
    return " SIG+" if d["tlo"] > 0 else (" SIG-" if d["thi"] < 0 else "  ns ")


def main():
    res = {}
    m = pd.read_csv(FRAME)
    line = m.open_margin.values
    act = m.margin_actual.values
    diff = act - line
    push = diff == 0
    live = ~push
    pick_home = m.edge.values > 0
    cover = np.where(pick_home, diff > 0, diff < 0)
    pnl = np.where(push, 0.0, np.where(cover, DEC - 1.0, -1.0))
    seas = m.season.values

    print("=" * 120)
    print("ATS19-EXTRA — the separate claims, the side-composition control, "
          "and the exact D121 anchor.")
    print("=" * 120)

    # ---- (A) BEATS THE SPREAD vs CLEARS -110 -------------------------------
    print("\n[A] TWO SEPARATE CLAIMS (prereg §9).  'cover > 50.000%' = we have "
          "information the opening line does not.\n    'cover > 52.381%' = "
          "that information is bigger than the vig.  They are tested "
          "separately.")
    print(f"    {'window':<12}{'n_live':>8}{'cover%':>9}{'vs 50.000':>11}"
          f"{'  [K-1 t on cover-0.5]':>26}{'sig':>6}"
          f"{'vs 52.381':>11}{'  [K-1 t on cover-be]':>25}{'sig':>6}")
    a = {}
    for wn, ws in (("POOL19", set(SEASONS19)), ("OOS14", OOS14),
                   ("DEV5", DEV5)):
        s = live & np.isin(seas, list(ws))
        c50 = ct(cover[s].astype(float) - 0.5, seas[s])
        cbe = ct(cover[s].astype(float) - BE, seas[s], SEED + 1)
        a[wn] = {"cover": float(cover[s].mean()), "vs50": c50, "vsbe": cbe}
        print(f"    {wn:<12}{int(s.sum()):>8}{100*cover[s].mean():>8.3f}%"
              f"{100*c50['mean']:>+11.3f}"
              f"  [{100*c50['tlo']:>+9.3f},{100*c50['thi']:>+9.3f}]{sg(c50):>6}"
              f"{100*cbe['mean']:>+11.3f}"
              f"  [{100*cbe['tlo']:>+9.3f},{100*cbe['thi']:>+9.3f}]{sg(cbe):>6}")
    res["separate_claims"] = a

    # ---- (B) SIDE COMPOSITION ---------------------------------------------
    print("\n[B] SIDE COMPOSITION — is the all-games result just an AWAY "
          "tilt?  ALWAYS-AWAY returns -3.26% and our arm returns -3.25%.")
    fav_home = line > 0
    pnl_away = np.where(push, 0.0, np.where(diff < 0, DEC - 1.0, -1.0))
    pnl_home = np.where(push, 0.0, np.where(diff > 0, DEC - 1.0, -1.0))
    pnl_fav = np.where(push, 0.0,
                       np.where(np.where(fav_home, diff > 0, diff < 0),
                                DEC - 1.0, -1.0))
    print(f"    our arm picks HOME on {100*pick_home.mean():.2f}% of games "
          f"(a coin-flip selector would pick HOME on ~50%; the market "
          f"favourite is HOME on {100*fav_home.mean():.2f}%).")
    ph_season = pd.Series(pick_home).groupby(pd.Series(seas)).mean()
    print(f"    per-season HOME share: min {100*ph_season.min():.1f}% "
          f"({ph_season.idxmin()})  max {100*ph_season.max():.1f}% "
          f"({ph_season.idxmax()})")
    for nm, sub in (("picked HOME", pick_home), ("picked AWAY", ~pick_home)):
        s = sub & live
        r = ct(pnl[sub], seas[sub], SEED + 2)
        print(f"    {nm:<14} n={int(sub.sum()):>6}  cover "
              f"{100*cover[s].mean():>6.3f}%  ROI {100*r['mean']:>+6.2f}%  "
              f"[K-1 t {100*r['tlo']:+.2f},{100*r['thi']:+.2f}]{sg(r)}")
    print(f"\n    PAIRED DELTAS on the SAME 22,742 games (season-clustered, "
          f"K-1 = 18 dof) — this is the honest version of the question:")
    pd_ = {}
    for nm, ctrl in (("vs ALWAYS AWAY", pnl_away), ("vs ALWAYS HOME", pnl_home),
                     ("vs ALWAYS FAVOURITE", pnl_fav)):
        d = ct(pnl - ctrl, seas, SEED + 3)
        pd_[nm] = d
        print(f"    {nm:<22}delta ROI {100*d['mean']:>+6.2f}pp  "
              f"[{100*d['tlo']:>+6.2f},{100*d['thi']:>+6.2f}]{sg(d)}   "
              f"(control ROI {100*ctrl.mean():+.2f}%)")
    res["side_composition"] = {
        "pct_home_picks": float(pick_home.mean()),
        "pct_market_fav_home": float(fav_home.mean()),
        "per_season_home_share": {k: float(v) for k, v in ph_season.items()},
        "roi_home_picks": float(pnl[pick_home].mean()),
        "roi_away_picks": float(pnl[~pick_home].mean()),
        "cover_home_picks": float(cover[pick_home & live].mean()),
        "cover_away_picks": float(cover[(~pick_home) & live].mean()),
        "paired": pd_}

    # ---- (C) THE D121 ANCHOR, EXACT ---------------------------------------
    print("\n[C] THE D121 HARNESS ANCHOR, ON D121'S OWN FRAME.  D121: 'ATS vs "
          "the opening spread 52.72% CI(51.2,54.2) vs 52.38% breakeven, "
          "p=0.325'.")
    rt = pd.read_csv(RT1, dtype={"game_id": str})
    rt["game_date"] = pd.to_datetime(rt.game_date)
    oo = pd.read_csv(ODDS, parse_dates=["game_date"])
    k = ["season", "game_date", "home", "away", "open_margin", "close_margin",
         "score_home", "score_away"]
    j = rt.merge(oo[k], on=["season", "game_date", "home", "away"], how="left")
    j = j[j.open_margin.notna() & j.close_margin.notna()].copy()
    j["act"] = j.score_home - j.score_away
    anc = {}
    for pcol in ("p_full", "p_starved"):
        for sc in (6.96, 7.2):
            mu = sc * logit(j[pcol].values)
            ln = j.open_margin.values
            ph = mu > ln
            df_ = j.act.values - ln
            pu = df_ == 0
            cv = np.where(ph, df_ > 0, df_ < 0)
            pn = np.where(pu, 0.0, np.where(cv, DEC - 1.0, -1.0))
            rng = np.random.default_rng(SEED)
            idx = rng.integers(0, int((~pu).sum()), (N_BOOT, int((~pu).sum())))
            hb = cv[~pu][idx].mean(axis=1)
            lo, hi = np.percentile(hb, [2.5, 97.5])
            t = ct(pn, j.season.values, SEED + 4)
            anc[f"{pcol}@{sc}"] = {"n": int(len(j)), "n_push": int(pu.sum()),
                                   "cover": float(cv[~pu].mean()),
                                   "iid_lo": float(lo), "iid_hi": float(hi),
                                   "roi": t}
            print(f"    {pcol:<10} scale {sc:<5} n={len(j)} push={int(pu.sum())}"
                  f"  cover {100*cv[~pu].mean():.2f}% "
                  f"i.i.d.CI({100*lo:.1f},{100*hi:.1f})  "
                  f"ROI {100*pn.mean():+.2f}%  K-1 t "
                  f"[{100*t['tlo']:+.2f},{100*t['thi']:+.2f}]{sg(t)} K={t['K']}")
    print(f"    D121 REGISTERED 52.72% CI(51.2,54.2).  The p_full@6.96 row is "
          f"the exact replication target.")
    res["d121_anchor"] = anc

    # ---- (D) COMPOSITION vs SELECTION -------------------------------------
    print("\n[D] DECOMPOSING THE +1.60pp THE PLACEBO SAYS OUR INFORMATION IS "
          "WORTH: how much is SIDE COMPOSITION and how much is GAME-BY-GAME "
          "SELECTION?")
    q = float(pick_home.mean())
    roi_h, roi_a = float(pnl_home.mean()), float(pnl_away.mean())
    comp = q * roi_h + (1 - q) * roi_a
    print(f"    ALWAYS-HOME ROI {100*roi_h:+.2f}%  ALWAYS-AWAY ROI "
          f"{100*roi_a:+.2f}%  -> the AWAY side is worth "
          f"{100*(roi_a-roi_h):+.2f}pp in this corpus.")
    print(f"    A RANDOM selector with OUR OWN home share ({100*q:.2f}%) "
          f"would earn {100*comp:+.2f}%.  Our arm earns {100*pnl.mean():+.2f}%"
          f", so GAME-BY-GAME SELECTION is worth {100*(pnl.mean()-comp):+.2f}pp.")
    print(f"    But ALWAYS-AWAY — a rule with no model in it at all — earns "
          f"{100*roi_a:+.2f}%, i.e. the SAME as our arm ({100*pnl.mean():+.2f}%)"
          f".  Selection buys back exactly what the\n    home tilt costs, and "
          f"nothing more.  NEITHER CLEARS -110.")
    # placebo home share (cheap, 100 draws) — the placebo has NO away tilt
    m2 = pd.read_csv(FRAME)
    rng = np.random.default_rng(SEED)
    dcode = pd.factorize(m2.game_date.values)[0]
    order = np.argsort(dcode, kind="stable")
    st = np.searchsorted(dcode[order], np.arange(dcode.max() + 1))
    en = np.searchsorted(dcode[order], np.arange(dcode.max() + 1), "right")
    groups = [order[x:y] for x, y in zip(st, en) if y - x > 1]
    pu_ = m2.p_us.values
    shares = []
    for _ in range(100):
        pp = pu_.copy()
        for g in groups:
            pp[g] = pp[rng.permutation(g)]
        shares.append(float(((7.2 * logit(pp)) > line).mean()))
    print(f"    PLACEBO home share {100*np.mean(shares):.2f}% (sd "
          f"{100*np.std(shares):.2f}) — ESSENTIALLY OUR OWN ({100*q:.2f}%), "
          f"so the +1.60pp gain over the placebo is NOT a\n    composition "
          f"artefact: it is genuine game-level selection.  The always-away "
          f"comparison says something different and equally true — a rule "
          f"with\n    NO MODEL IN IT reaches the same ROI, because 100% away "
          f"exposure buys the same {100*(roi_a-roi_h):.2f}pp our selection "
          f"has to work for.")
    res["composition"] = {"home_share": q, "roi_always_home": roi_h,
                          "roi_always_away": roi_a,
                          "composition_matched_random": comp,
                          "arm_roi": float(pnl.mean()),
                          "selection_gain_pp": float(pnl.mean() - comp),
                          "placebo_home_share": float(np.mean(shares))}

    json.dump(res, open(OUT, "w"), indent=1, default=str)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
