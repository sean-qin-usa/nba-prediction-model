#!/usr/bin/env python3
"""BP-EXEC — PART B of the BIGPLAYER capstone: the EXECUTION ceiling.

Capital does not improve prediction, it improves EXECUTION.  The model is held
FIXED (at T0 and again at T5, the two ends of PART A's information ladder) and
only the PRICE varies:

  E0  our access          1-2 retail books, the REAL transacted moneyline
                          (D155: overround 3.84% close / 4.31% open)
  E1  a 5-8 book shop     transacted decimal x f_N, f_N = be_1bk / be_N from
                          D142's MEASURED quote-count ladder (k=1 be 73.03;
                          k=2 -0.97, k=3 -1.42, k=4 -1.70pp) and its Gaussian
                          EXTRAPOLATION (N=5 -1.94, N=8 -2.38pp).  N=2 is the
                          ONLY clean measurement; N=4 mixes two vendors; N=5
                          and N=8 are EXTRAPOLATIONS and are labelled CEILING.
  E2  exchange pricing    commission on NET WINNINGS ONLY:
                          dec_eff = 1 + (1/p_fair - 1)*(1 - c), c in {2%, 5%}
                          — structurally different from an overround baked
                          into the price, and shown to be.
  E3  E2 + both sides     the ex-ante MIDDLE, D148 s7 construction verbatim:
                          enter at the opening number, buy the other side at
                          the close only if the line moved >= W in the entry
                          side's favour.  W = 2.0 primary.

Bet sets: the four REGISTERED F4 rules + their union, operators imported
VERBATIM from scripts/bo_openbacktest.py (build / price_cols / registry_masks)
— the D126/D142/D155 precedent.  Nothing is re-chosen here.

REPORTING STATISTIC: season-clustered bootstrap CI, plus the cluster-mean t
interval at K-1 = 2 dof (s9.3's conservative bound).  The i.i.d. CI is
secondary.

Read-only.  Out: data/bp_exec.json, data/logs/bp_exec.log
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

from bo_openbacktest import (build, price_cols,                   # noqa: E402
                             registry_masks)

CSV = str(REPO / "data" / "bp_ladder_pergame.csv")
OUT = REPO / "data" / "bp_exec.json"
SEED = 20260803
B = 4000
ATS_DEC = 1.0 + 100.0 / 110.0            # -110

# --- D142 measured / extrapolated best-of-N breakeven ladder --------------
BE_1BK = 73.03
DBE = {2: 0.97, 3: 1.42, 4: 1.70, 5: 1.94, 8: 2.38}
SHOP_F = {n: BE_1BK / (BE_1BK - d) for n, d in DBE.items()}
SHOP_LABEL = {2: "MEASURED (2 real books)", 3: "measured subset",
              4: "measured but MIXES TWO VENDOR SNAPSHOTS — upper bound",
              5: "EXTRAPOLATION — CEILING, not a forecast",
              8: "EXTRAPOLATION — CEILING, not a forecast"}
COMMISSIONS = (0.02, 0.05)
W_MIDDLE = 2.0
TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


# ------------------------------------------------------------- statistics --
def cluster_stats(pnl, season, seed=SEED, Bn=B):
    """ROI with a SEASON-CLUSTERED bootstrap CI and the K-1 cluster-mean t."""
    pnl = np.asarray(pnl, float)
    season = np.asarray(season)
    n = len(pnl)
    if n == 0:
        return dict(n=0)
    keys = sorted(set(season.tolist()))
    K = len(keys)
    groups = [pnl[season == k] for k in keys]
    rng = np.random.default_rng(seed)
    # i.i.d. (secondary)
    idx = rng.integers(0, n, (Bn, n))
    bi = pnl[idx].mean(axis=1)
    # season-clustered
    bs = np.empty(Bn)
    for b in range(Bn):
        pick = rng.integers(0, K, K)
        bs[b] = np.concatenate([groups[i] for i in pick]).mean()
    means = np.array([g.mean() for g in groups])
    ns = np.array([len(g) for g in groups], float)
    se = float(means.std(ddof=1) / np.sqrt(K)) if K > 1 else float("nan")
    tc = TCRIT.get(K - 1, 1.96)
    tlo, thi = float(means.mean() - tc * se), float(means.mean() + tc * se)
    clo, chi = np.percentile(bs, [2.5, 97.5])
    ilo, ihi = np.percentile(bi, [2.5, 97.5])
    return dict(n=int(n), roi=float(pnl.mean()),
                clustered_lo=float(clo), clustered_hi=float(chi),
                clustered_sig=bool(clo > 0 or chi < 0),
                iid_lo=float(ilo), iid_hi=float(ihi),
                t_lo=tlo, t_hi=thi, t_sig=bool(tlo > 0 or thi < 0),
                K=K, per_season={k: float(m) for k, m in zip(keys, means)},
                n_per_season={k: int(x) for k, x in zip(keys, ns)})


# ------------------------------------------------------------- price tiers --
def exec_dec(dec, p_fair, tier):
    """Transactable decimal on OUR side under an execution tier."""
    if tier == "E0":
        return dec
    if tier.startswith("E1_N"):
        return dec * SHOP_F[int(tier[4:])]
    if tier.startswith("E2_c"):
        c = float(tier[4:]) / 100.0
        return 1.0 + (1.0 / p_fair - 1.0) * (1.0 - c)
    raise ValueError(tier)


TIERS = (["E0"] + [f"E1_N{n}" for n in (2, 4, 5, 8)]
         + [f"E2_c{int(c * 100)}" for c in COMMISSIONS])


# ---------------------------------------------------------------- middles --
def middles(m, sel, W, dec_leg, seed=SEED):
    """D148 s7 ex-ante MIDDLE, verbatim construction.  Enter our side at the
    OPENING number; buy the other side at the CLOSE only if the line moved
    >= W in the entry side's favour.  ROI is per ENTRY (leg-1 stake = 1u)."""
    mm = m[sel]
    if len(mm) == 0:
        return dict(n=0)
    Lo = mm.open_margin.values                 # market expected HOME margin
    Lc = mm.close_margin.values
    act = mm.margin_actual.values
    home = mm.pick_home.values.astype(bool)
    move = np.where(home, Lc - Lo, Lo - Lc)    # >0 = moved in our favour
    take2 = move >= W
    # leg 1: our side against the OPEN line
    d1 = np.where(home, act - Lo, Lo - act)
    # leg 2: the OTHER side against the CLOSE line
    d2 = np.where(home, Lc - act, act - Lc)
    w = dec_leg - 1.0
    pnl = np.where(d1 > 0, w, np.where(d1 == 0, 0.0, -1.0))
    p2 = np.where(d2 > 0, w, np.where(d2 == 0, 0.0, -1.0))
    pnl = np.where(take2, pnl + p2, pnl)
    mid = take2 & (d1 > 0) & (d2 > 0)
    r = cluster_stats(pnl, mm.season.values, seed)
    r.update(second_leg_pct=float(take2.mean()),
             p_middle_given_2leg=float(mid[take2].mean()) if take2.sum() else float("nan"),
             n_middle=int(mid.sum()), W=W, dec_leg=float(dec_leg))
    return r


# -------------------------------------------------------------------- main --
def run_arm(m, when, src, tag, res):
    """One (price-moment, source) arm: every rule x every execution tier."""
    p_fair, dec, ok = price_cols(m, when, src)
    masks, edge, same = registry_masks(m, p_fair, when)
    masks["UNION"] = np.logical_or.reduce(list(masks.values()))
    masks["ALL_UNIVERSE"] = same & ok
    print(f"\n{'=' * 112}\nARM {tag}  ({when} price, source {src})   "
          f"n_frame={len(m)}  priced={int(ok.sum())}")
    print(f"{'set':17}{'n':>6}  {'hit%':>6} " +
          "".join(f"{t:>16}" for t in TIERS))
    for name, msk in masks.items():
        sel = msk & ok
        if sel.sum() < 10:
            continue
        hit = m.hit.values[sel].astype(bool)
        row = {}
        line = f"{name:17}{int(sel.sum()):6d}  {100 * hit.mean():6.2f} "
        for tier in TIERS:
            d = exec_dec(dec[sel], p_fair[sel], tier)
            pnl = np.where(hit, d - 1.0, -1.0)
            st = cluster_stats(pnl, m.season.values[sel])
            st["mean_dec"] = float(d.mean())
            st["breakeven"] = float(np.mean(1.0 / d))
            row[tier] = st
            line += f"{100 * st['roi']:+7.2f}{'*' if st['clustered_sig'] else ' '}{'t' if st['t_sig'] else ' '}".rjust(16)
        print(line)
        res.setdefault(tag, {})[name] = row
    # detail for the union + the strongest rule
    for name in ("UNION", "T20_D03_10", "ALL_UNIVERSE"):
        if name not in res.get(tag, {}):
            continue
        print(f"\n  DETAIL {tag} / {name}")
        for tier in TIERS:
            st = res[tag][name][tier]
            print(f"    {tier:8} n={st['n']:5} ROI {100 * st['roi']:+7.3f}%  "
                  f"season-clustered [{100 * st['clustered_lo']:+7.3f},"
                  f"{100 * st['clustered_hi']:+7.3f}] "
                  f"{'SIG' if st['clustered_sig'] else 'ns '}  "
                  f"t(K-1)[{100 * st['t_lo']:+8.3f},{100 * st['t_hi']:+8.3f}] "
                  f"{'SIG' if st['t_sig'] else 'ns '}  "
                  f"mean_dec {st['mean_dec']:.4f} be {100 * st['breakeven']:.2f}%  "
                  f"per-season {[round(100 * v, 2) for v in st['per_season'].values()]}")
    return masks, ok


def main():
    res = {"shop_factors": {n: round(f, 6) for n, f in SHOP_F.items()},
           "shop_labels": SHOP_LABEL, "arms": {}}
    for model_tier, pcol in (("T0", "p_T0"), ("T5", "p_T5")):
        m = build(CSV, pcol, f"BIGPLAYER model tier {model_tier}", res)
        for when, src in (("open", "ML"), ("close", "ML")):
            tag = f"{model_tier}@{when}"
            masks, ok = run_arm(m, when, src, tag, res["arms"])
        # ---- E3: the ex-ante MIDDLE -------------------------------------
        p_fair, dec, okc = price_cols(m, "open", "ML")
        masks, _, _ = registry_masks(m, p_fair, "open")
        masks["UNION"] = np.logical_or.reduce(list(masks.values()))
        masks["ALL_UNIVERSE"] = np.ones(len(m), bool)
        have = m.open_margin.notna().values & m.close_margin.notna().values
        print(f"\n{'=' * 112}\nE3 — EX-ANTE MIDDLE (D148 s7), model tier "
              f"{model_tier}, W={W_MIDDLE}")
        for name, msk in masks.items():
            sel = msk & have
            if sel.sum() < 30:
                continue
            for lab, dleg in (("retail -110", ATS_DEC),
                              ("exchange c=2%", 1.0 + 1.0 * 0.98),
                              ("exchange c=5%", 1.0 + 1.0 * 0.95)):
                r = middles(m, sel, W_MIDDLE, dleg)
                res.setdefault("E3", {}).setdefault(f"{model_tier}/{name}",
                                                    {})[lab] = r
                print(f"  {name:17} {lab:14} n={r['n']:5} 2nd-leg "
                      f"{100 * r['second_leg_pct']:5.1f}%  P(mid|2leg) "
                      f"{100 * r['p_middle_given_2leg']:5.2f}%  ROI/entry "
                      f"{100 * r['roi']:+7.3f}%  clustered ["
                      f"{100 * r['clustered_lo']:+7.3f},{100 * r['clustered_hi']:+7.3f}] "
                      f"{'SIG' if r['clustered_sig'] else 'ns '}  "
                      f"t[{100 * r['t_lo']:+8.3f},{100 * r['t_hi']:+8.3f}] "
                      f"{'SIG' if r['t_sig'] else 'ns '}")
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
