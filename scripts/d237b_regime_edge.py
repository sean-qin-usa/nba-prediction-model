#!/usr/bin/env python3
"""D237b — THE POWERED VERSION: does our EDGE over the opener vary by regime?

D237 tested the owner's hypothesis on 888 BETS and could not resolve anything
below ~25 ROI points of bucket spread, because per-bet variance is enormous
(bucket SE +/-6.3 ROI points). A null result at that power says little.

The same hypothesis has a far better-powered form. "When the market is wonky our
edge is larger" is a claim about the FORECAST advantage, not about realised
betting returns, and the forecast advantage is observable on EVERY game rather
than only on the 12% of games that clear a betting threshold:

    edge_g = ll_open_g - ll_offset_g          (positive = we beat the opener)

8,286 games instead of 888 bets, and the per-game quantity is a log-loss
difference rather than a +0.91/-1.00 lottery ticket, so the variance is orders
of magnitude smaller. If the regime hypothesis is true at all it should show
here first; if it is absent here, the betting-level null is not merely a power
failure.

Regimes are strictly prior, identical in construction to D237.
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
from scipy.optimize import minimize_scalar                        # noqa: E402

WIN = 200
FROM = "2019-20"


def nll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_scale(m, y):
    return float(minimize_scalar(
        lambda s: nll(1 / (1 + np.exp(-m / s)), y).mean(),
        bounds=(2, 25), method="bounded").x)


def main():
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
    f = f[f["season"] >= "2017-18"].copy()
    f["game_date"] = pd.to_datetime(f["game_date"])
    f = f.dropna(subset=["open_margin", "margin_actual", "m_us"])
    f["y"] = (f["margin_actual"] > 0).astype(float)
    f = f.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    # per-game losses, each source on its OWN walk-forward scale (D193 rule)
    seasons = sorted(f["season"].unique())
    for col, out in (("open_margin", "ll_open"), ("m_us", "ll_off")):
        f[out] = np.nan
        for i, s in enumerate(seasons):
            te = (f["season"] == s).to_numpy()
            tr = f["season"].isin(seasons[:i]).to_numpy() if i else te
            sc = fit_scale(f.loc[tr, col].to_numpy(float), f.loc[tr, "y"].to_numpy(float))
            f.loc[te, out] = nll(1 / (1 + np.exp(-f.loc[te, col].to_numpy(float) / sc)),
                                 f.loc[te, "y"].to_numpy(float))
    f["edge"] = f["ll_open"] - f["ll_off"]          # positive = we beat the opener

    # strictly-prior regime state, same construction as D237
    g = f.groupby("game_date").agg(ll=("ll_open", "mean"),
                                   ae=("open_margin", "size")).reset_index()
    ae = f.assign(e=(f["margin_actual"] - f["open_margin"]).abs()) \
          .groupby("game_date")["e"].mean().rename("ae").reset_index()
    g = g.drop(columns="ae").merge(ae, on="game_date")
    buf_ll, buf_ae, rows = [], [], []
    for r in g.sort_values("game_date").itertuples():
        rows.append((r.game_date,
                     np.mean(buf_ll[-WIN:]) if len(buf_ll) >= 50 else np.nan,
                     np.mean(buf_ae[-WIN:]) if len(buf_ae) >= 50 else np.nan))
        buf_ll.append(r.ll); buf_ae.append(r.ae)
    reg = pd.DataFrame(rows, columns=["game_date", "mkt_ll_tr", "mkt_ae_tr"])
    d = f.merge(reg, on="game_date").dropna(subset=["mkt_ll_tr"])
    d = d[d["season"] >= FROM]
    start = d.groupby("season")["game_date"].transform("min")
    d["days_in"] = (d["game_date"] - start).dt.days

    print(f"games {len(d):,}, seasons {d.season.min()}..{d.season.max()}")
    print(f"pooled edge over the opener: {d.edge.mean():+.5f} nats "
          f"(positive = better than the open)\n")

    res = {}
    for col, desc in (("mkt_ll_tr", "trailing market log loss (HIGH = market worse)"),
                      ("mkt_ae_tr", "trailing |actual-open| (HIGH = less predictable)"),
                      ("days_in", "days into season (LOW = early)")):
        q = pd.qcut(d[col], 4, labels=False, duplicates="drop")
        print(f"=== {col} — {desc} ===")
        buckets = []
        for k in sorted(pd.Series(q).dropna().unique()):
            sub = d[q == k]
            m = sub.edge.mean()
            se = sub.edge.std(ddof=1) / np.sqrt(len(sub))
            buckets.append((int(k), len(sub), float(sub[col].median()), float(m), float(se)))
            print(f"   Q{int(k)+1}  n={len(sub):5}  {col}={sub[col].median():8.3f}  "
                  f"edge {m:+.5f} +/-{se:.5f}")
        per = []
        for s, sub in d.groupby("season"):
            per.append(np.polyfit(sub[col], sub["edge"], 1)[0])
        per = np.array(per); k = len(per)
        se = per.std(ddof=1) / np.sqrt(k)
        tc = stats.t.ppf(0.975, k - 1)
        lo, hi = per.mean() - tc * se, per.mean() + tc * se
        print(f"   season-clustered slope {per.mean():+.6f} "
              f"95% CI [{lo:+.6f}, {hi:+.6f}]  K={k}  "
              f"{'SIGNIFICANT' if (lo>0 or hi<0) else 'ns'}")
        print(f"   Q4-Q1 edge gap {buckets[-1][3]-buckets[0][3]:+.5f} nats\n")
        res[col] = dict(buckets=buckets, slope=float(per.mean()),
                        ci=[float(lo), float(hi)], sig=bool(lo > 0 or hi < 0))

    # MDE: what gap COULD this design resolve?
    sd = d.edge.std(ddof=1)
    n_b = len(d) // 4
    mde = 2.8 * sd * np.sqrt(2 / n_b)
    print(f"=== POWER ===\n   per-game edge sd {sd:.4f}, bucket n~{n_b}")
    print(f"   MDE80 on a Q4-Q1 gap: {mde:.5f} nats")
    print(f"   D237 (bets) could not resolve below ~25 ROI points; this design "
          f"resolves {mde:.5f} nats, i.e. {mde/abs(d.edge.mean()+1e-12):.1f}x the "
          f"pooled edge itself")
    json.dump(res, open(ROOT / "data" / "d237b_regime_edge.json", "w"), default=float)


if __name__ == "__main__":
    main()
