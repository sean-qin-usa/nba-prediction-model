#!/usr/bin/env python3
"""D237 — IS THERE REGIME STRUCTURE IN THE BETTING RETURNS? Diagnostic only.

THE OWNER'S QUESTION: "in a period where all models are wonky, may be profitable
to capture there". That is a real and untested axis — every gate in this register
so far conditions the FORECAST; none conditions the BET DECISION on the state of
the market.

The theory is specific enough to be falsifiable: when the market's own opening
price is forecasting badly, prices are less informative, so a given disagreement
should be worth more. If that is true, bet returns should rise with trailing
market log loss.

WHY THIS SCRIPT DOES NOT TRADE ANYTHING. The register's own manufacturing-capacity
result is that best-of-N random subsets of this same book buy +2.54 ROI points
from nothing. A regime rule is EXACTLY a subset selector, so searching regimes
against realised returns on the seasons that built the strategy is the highest-
risk thing this project can do. This script therefore only MEASURES association,
reports it against a permutation null that respects the search, and ships
nothing. Any rule that looks good here has to be pre-registered and gated
separately, and the honest confirmation is 2026-27.

ALL REGIME VARIABLES ARE STRICTLY PRIOR. Each is computed from games that had
SETTLED before the bet's own game date, so the regime a bet is assigned to was
knowable when the bet was placed.
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

TIER = "k=1 raw"          # the OBSERVED single-book tier: real settled returns,
                          # not the modelled best-of-nine expectations (D236)
FROM = "2019-20"
WIN = 200                 # trailing games defining a regime


def market_frame() -> pd.DataFrame:
    """Every scored game with the opener's own loss — the market-skill series."""
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
    f = f[f["season"] >= "2017-18"].copy()          # lead-in for the trailing window
    f["game_date"] = pd.to_datetime(f["game_date"])
    f = f.dropna(subset=["open_margin", "margin_actual"])
    f["y"] = (f["margin_actual"] > 0).astype(float)
    p = 1 / (1 + np.exp(-f["open_margin"] / 6.96))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    f["mkt_ll"] = -(f["y"] * np.log(p) + (1 - f["y"]) * np.log(1 - p))
    f["mkt_abserr"] = (f["margin_actual"] - f["open_margin"]).abs()
    return f.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def regimes(mf: pd.DataFrame) -> pd.DataFrame:
    """Per-DATE regime state from strictly prior settled games."""
    g = mf.groupby("game_date").agg(ll=("mkt_ll", "mean"),
                                    ae=("mkt_abserr", "mean"),
                                    n=("mkt_ll", "size")).reset_index()
    g = g.sort_values("game_date")
    # expanding-then-trailing mean of the PREVIOUS WIN games, shifted so the
    # current date contributes nothing to its own regime
    ll, ae, dates = [], [], []
    buf_ll, buf_ae = [], []
    for r in g.itertuples():
        ll.append(np.mean(buf_ll[-WIN:]) if len(buf_ll) >= 50 else np.nan)
        ae.append(np.mean(buf_ae[-WIN:]) if len(buf_ae) >= 50 else np.nan)
        dates.append(r.game_date)
        buf_ll += [r.ll] * int(r.n)
        buf_ae += [r.ae] * int(r.n)
    out = pd.DataFrame({"game_date": dates, "mkt_ll_tr": ll, "mkt_ae_tr": ae})
    # days into the season — the other regime the owner named
    mf2 = mf[["game_date", "season"]].drop_duplicates()
    start = mf2.groupby("season")["game_date"].min().rename("season_start")
    out = out.merge(mf2, on="game_date", how="left").merge(start, on="season", how="left")
    out["days_in"] = (out["game_date"] - out["season_start"]).dt.days
    return out


def main():
    mf = market_frame()
    reg = regimes(mf)
    pb = json.load(open(ROOT / "data" / "wf_perbet_OFFSET.json"))[TIER]
    b = pd.DataFrame([x for x in pb if x["season"] >= FROM])
    b["game_date"] = pd.to_datetime(b["date"])
    d = b.merge(reg, on="game_date", how="left", suffixes=("", "_r"))
    d = d.dropna(subset=["mkt_ll_tr"])
    print(f"bets {len(d)} of {len(b)} carry a regime state, "
          f"seasons {d.season.min()}..{d.season.max()}")
    print(f"tier {TIER} (settled returns), pooled ROI "
          f"{100*d.ev.mean():+.2f}%\n")

    RV = [("mkt_ll_tr", "trailing market log loss (HIGH = market forecasting badly)"),
          ("mkt_ae_tr", "trailing |actual - open| (HIGH = outcomes less predictable)"),
          ("days_in", "days into the season (LOW = early, rosters unsettled)")]

    res = {}
    for col, desc in RV:
        print(f"=== {col} — {desc} ===")
        q = pd.qcut(d[col], 4, labels=False, duplicates="drop")
        rows = []
        for k in sorted(pd.Series(q).dropna().unique()):
            sub = d[q == k]
            roi = 100 * sub.ev.mean()
            se = 100 * sub.ev.std(ddof=1) / np.sqrt(len(sub))
            rows.append((int(k), len(sub), float(sub[col].median()), roi, se))
            print(f"   Q{int(k)+1}  n={len(sub):4}  {col}={sub[col].median():8.4f}  "
                  f"ROI {roi:+7.2f}%  +/-{se:5.2f}")
        # slope of per-bet return on the regime variable, season-clustered
        per = []
        for s, sub in d.groupby("season"):
            if len(sub) < 40:
                continue
            per.append(np.polyfit(sub[col], sub["ev"], 1)[0])
        per = np.array(per)
        k = len(per)
        se = per.std(ddof=1) / np.sqrt(k)
        tc = stats.t.ppf(0.975, k - 1)
        lo, hi = per.mean() - tc * se, per.mean() + tc * se
        sig = (lo > 0) or (hi < 0)
        print(f"   season-clustered slope {per.mean():+.5f} "
              f"95% CI [{lo:+.5f}, {hi:+.5f}]  K={k}  "
              f"{'SIGNIFICANT' if sig else 'ns'}")
        print(f"   Q4 minus Q1 ROI: {rows[-1][3]-rows[0][3]:+.2f} points\n")
        res[col] = dict(buckets=rows, slope=float(per.mean()),
                        ci=[float(lo), float(hi)], sig=bool(sig))

    # ---- the null that respects the search --------------------------------
    print("=== PERMUTATION NULL (what a regime split buys from nothing) ===")
    rng = np.random.default_rng(237)
    best = []
    for _ in range(2000):
        sh = d["ev"].to_numpy(float).copy()
        rng.shuffle(sh)
        gaps = []
        for col, _ in RV:
            q = pd.qcut(d[col], 4, labels=False, duplicates="drop").to_numpy()
            g = [sh[q == k].mean() for k in range(4)]
            gaps.append(100 * (max(g) - min(g)))
        best.append(max(gaps))          # best of the THREE candidates, as searched
    obs = max(100 * (max(r[3] for r in res[c]["buckets"])
                     - min(r[3] for r in res[c]["buckets"])) / 100
              for c in res)
    obs_pts = max(max(r[3] for r in res[c]["buckets"])
                  - min(r[3] for r in res[c]["buckets"]) for c in res)
    p = float(np.mean(np.array(best) >= obs_pts))
    print(f"   observed best spread across the 3 candidates: {obs_pts:.2f} ROI points")
    print(f"   null best-of-3 spread: median {np.median(best):.2f}, "
          f"95th pct {np.percentile(best,95):.2f}")
    print(f"   p(null >= observed) = {p:.3f}  -> "
          f"{'BEYOND the search null' if p < 0.05 else 'WITHIN what the search buys from noise'}")
    json.dump({"regimes": res, "null_p": p, "obs_spread": obs_pts,
               "null_95": float(np.percentile(best, 95))},
              open(ROOT / "data" / "d237_regime.json", "w"), default=float)


if __name__ == "__main__":
    main()
