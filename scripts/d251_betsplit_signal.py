#!/usr/bin/env python3
"""D251 — DOES THE PUBLIC BET SPLIT PREDICT ANYTHING? First test of a factor the
project has held on disk since August and never opened.

`data/an_market.csv.gz` carries, per game and book, the share of MONEY and the
share of TICKETS on each side. money% >> tickets% means a few large wagers;
tickets% >> money% means many small ones. The folk claim is that the first is
"sharp" and predicts covers.

WHAT THIS CAN AND CANNOT TEST. Action Network stores ONE snapshot per book, the
CLOSING state. So the split is known before tip and may legally predict the game
OUTCOME, but it cannot be used to predict line MOVEMENT that has already
happened. Any "reverse line movement" construction is unavailable here and would
be leakage if faked from a closing snapshot.

PRE-REGISTERED, in this order, and all three are reported whatever they say:

  T1  Back the side with the higher MONEY share than TICKET share, against the
      CLOSING line. Endpoint: ATS cover rate. Reference points 50.00% (no edge)
      and 52.38% (break-even at -110). A result between them is a real but
      unbettable effect, which is the outcome the register keeps finding.

  T2  Monotonicity. If the mechanism is real, the edge should grow with
      |money% - tickets%|. A flat profile across deciles refutes the mechanism
      even if T1 is positive, because it would mean the split is a proxy for
      something else.

  T3  Does it survive the model? Restricted to games the production model would
      bet, does the split add anything to the side the model already picks?

INFERENCE. Three seasons gives k=3 for season clustering, which cannot resolve
anything, so the primary interval is a WEEKLY BLOCK BOOTSTRAP (games within a
week share slate-level shocks). The per-season split is reported alongside
because D248/D249 showed 2024-25 is a feed outlier -- mean |money-tickets| runs
5.95 / 1.69 / 4.80 on the core basket -- so a pooled number that is driven by one
season must not be presented as three seasons of evidence.
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

# D248's core basket (15/68/69) was validated for SPREAD coverage. The bet
# split is a different field with a different population pattern, and it must be
# checked separately: books 68/69/71/75 carry a real money-vs-tickets split on
# only 3-31% of rows in 2023-24 and 2024-25 (the rest report the two shares as
# identical, i.e. a placeholder) and ~93% only in 2025-26. Book 15 is the ONLY
# one populated across all three seasons -- 83.3% / 91.4% / 92.8% on 1,221 /
# 1,205 / 1,174 rows -- so it is the whole usable panel for this factor.
CORE = ("15",)
BE = 0.5238


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def block_boot(df, val_col, week_col, n=20000, seed=251):
    """Weekly block bootstrap of a mean."""
    rng = np.random.default_rng(seed)
    wk = df[week_col].to_numpy()
    v = df[val_col].to_numpy(float)
    weeks, inv = np.unique(wk, return_inverse=True)
    buckets = [v[inv == i] for i in range(len(weeks))]
    out = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, len(buckets), len(buckets))
        out[i] = np.concatenate([buckets[j] for j in pick]).mean()
    return float(v.mean()), float(np.percentile(out, 2.5)), \
        float(np.percentile(out, 97.5))


def main():
    a = pd.read_csv(ROOT / "data" / "an_market.csv.gz")
    a["game_id"] = zf(a["game_id"])
    a = a[a.book_id.astype(str).isin(CORE)]
    a = a.dropna(subset=["spread_money_pct", "spread_tickets_pct"])
    # Compute the divergence PER BOOK and then aggregate. Differencing the
    # two medians is not the median of the differences, and doing it that way
    # collapsed the nonzero-divergence rate from ~90% to 36.9% by letting
    # independent medians coincide.
    a["bdiv"] = a.spread_money_pct - a.spread_tickets_pct
    g = (a.groupby(["game_id", "season"])
           .agg(money=("spread_money_pct", "median"),
                tickets=("spread_tickets_pct", "median"),
                bdiv=("bdiv", "median"),
                bdiv_mean=("bdiv", "mean"),
                nbook=("book_id", "nunique"),
                num_bets=("num_bets", "median")).reset_index())
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
    f["game_id"] = zf(f["game_id"])
    f["game_date"] = pd.to_datetime(f["game_date"])
    d = g.merge(f[["game_id", "game_date", "close_margin", "open_margin",
                   "margin_actual", "m_us", "m_us_blind"]], on="game_id")
    d = d.dropna(subset=["close_margin", "margin_actual"])
    d["week"] = d.game_date.dt.to_period("W").astype(str)
    d["divg"] = d.bdiv                         # >0 : money leans HOME
    d["ats_home"] = np.sign(d.margin_actual - d.close_margin)
    d = d[d.ats_home != 0]                     # drop pushes
    print(f"{len(d):,} games, {d.season.nunique()} seasons, "
          f"{d.week.nunique()} weeks; median books/game {d.nbook.median():.0f}")
    dv = d["divg"]
    print(f"div = money% - tickets% on the HOME spread side: "
          f"mean {dv.mean():+.2f}, sd {dv.std():.2f}, "
          f"nonzero {100*(dv != 0).mean():.1f}%")

    out = {}
    # ---------------- T1 --------------------------------------------
    print("\n" + "=" * 74)
    print("T1  BACK THE SIDE WITH MORE MONEY THAN TICKETS, vs THE CLOSE")
    print("=" * 74)
    t = d[d["divg"] != 0].copy()
    t["side"] = np.sign(t["divg"])                 # +1 home, -1 away
    t["win"] = (t.side == t.ats_home).astype(float)
    m, lo, hi = block_boot(t, "win", "week")
    print(f"  n={len(t):,}   cover {100*m:.2f}%   "
          f"weekly-block 95% CI [{100*lo:.2f}%, {100*hi:.2f}%]")
    print(f"  vs 50.00% : {100*(m-0.5):+.2f}pp     "
          f"vs 52.38% break-even : {100*(m-BE):+.2f}pp")
    verdict = ("PROFITABLE" if lo > BE else
               "REAL BUT UNBETTABLE" if lo > 0.5 else "NULL")
    print(f"  >>> {verdict}")
    out["T1"] = dict(n=len(t), cover=m, ci=[lo, hi], verdict=verdict)

    print("\n  per season (2024-25 is the feed outlier — D248/D249):")
    for s, gg in t.groupby("season"):
        mm, ll, hh = block_boot(gg, "win", "week", n=4000)
        md = gg["divg"].abs().mean()
        print(f"    {s}: n={len(gg):5}  cover {100*mm:6.2f}%  "
              f"CI [{100*ll:6.2f}%, {100*hh:6.2f}%]  mean|div| {md:5.2f}")
        out[f"T1_{s}"] = dict(n=len(gg), cover=mm, ci=[ll, hh])

    # ---------------- T2 --------------------------------------------
    print("\n" + "=" * 74)
    print("T2  MONOTONICITY IN |div| — required if the mechanism is real")
    print("=" * 74)
    t["absdiv"] = t["divg"].abs()
    try:
        t["dec"] = pd.qcut(t["absdiv"], 5, labels=False, duplicates="drop")
    except ValueError:
        t["dec"] = 0
    print(f"  {'quintile':10} {'n':>6} {'|div| range':>16} {'cover':>8}")
    prof = []
    for q, gg in t.groupby("dec"):
        cov = gg.win.mean()
        prof.append(float(cov))
        a0, a1 = gg["absdiv"].min(), gg["absdiv"].max()
        print(f"  {int(q)+1:<10} {len(gg):6} "
              f"{a0:7.1f}-{a1:<7.1f} {100*cov:7.2f}%")
    if len(prof) > 2:
        r = np.corrcoef(np.arange(len(prof)), prof)[0, 1]
        print(f"\n  rank correlation quintile -> cover: {r:+.3f}")
        print(f"  {'MONOTONE — consistent with the mechanism' if r > 0.5 else 'NOT MONOTONE — the split is proxying something else'}")
        out["T2_profile"] = prof
        out["T2_corr"] = float(r)

    # ---------------- T3 --------------------------------------------
    print("\n" + "=" * 74)
    print("T3  DOES IT ADD ANYTHING TO THE SIDE THE MODEL ALREADY PICKS?")
    print("=" * 74)
    t["model_side"] = np.sign(t.m_us - t.open_margin)
    t = t[t.model_side != 0]
    agree = t[t.model_side == t.side]
    disag = t[t.model_side != t.side]
    for lab, gg in (("model side, split AGREES", agree),
                    ("model side, split DISAGREES", disag)):
        w = (gg.model_side == gg.ats_home).astype(float)
        gg2 = gg.assign(w=w)
        mm, ll, hh = block_boot(gg2, "w", "week", n=6000)
        print(f"  {lab:30} n={len(gg):5}  model cover {100*mm:6.2f}%  "
              f"CI [{100*ll:6.2f}%, {100*hh:6.2f}%]")
        out[f"T3_{lab.split(',')[1].strip()}"] = dict(n=len(gg), cover=mm,
                                                      ci=[ll, hh])
    print("\n  If the model covers materially better when the money split")
    print("  agrees with it, the split is a usable FILTER even when T1 alone")
    print("  is unbettable. If the two are the same, it adds nothing.")

    json.dump(out, open(ROOT / "data" / "d251_betsplit.json", "w"), default=float)
    print("\nwrote data/d251_betsplit.json")


if __name__ == "__main__":
    main()
