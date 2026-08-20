#!/usr/bin/env python3
"""D248b — RE-RUN D247's OUTLIER TEST ON THE CONVENTION-CONSISTENT FEED.

D247 found 2024-25 ranked 1/19 on CLV at z +2.80 and traced it to recorded line
travel (corr +0.917). D248 found the mechanism: in 2024-25 the frame's only book
with opens was ESPN BET, which posts 100% half-point spreads, so the line grid
is spaced 1.0 rather than 0.5 and the smallest observable move doubles.

`build_odds_unified.py` rebuilds 2023-24 and 2024-25 from TeamRankings book1,
which holds the historical ~50%-integer / ~9%-key-number convention, validated
against SBR on two overlapping seasons (bias -0.037 and +0.008, corr 0.97+).

THE SIDE HAS TO BE REBUILT, NOT REUSED. The production side is
`sign(m_us - open)` and `m_us` is the OFFSET model, which takes the opening line
as an input. Change the feed and that side changes. Reusing the frame's stored
`m_us` would score a new feed with a side computed from the old one.

`m_us_blind` is genuinely feed-invariant -- G2 makes the blind model
market-blind -- so it is reused, and only the offset layer is refitted:

    m_offset = open + ridge( m_blind - open , rest_diff , |open| ),  lam 3000

walk-forward, fitted on all prior seasons, exactly as production does it.

2025-26 is EXCLUDED from the cross-season comparison and reported separately:
every held book had moved to half-points-only by then, so no consistent price
exists and forcing one would reintroduce the artefact this script removes.
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

LAM = 3000.0


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def offset_side(tr, te, ocol, ccol):
    """Refit the offset ridge on `tr`, return the signed side for `te`."""
    def des(d):
        o = d[ocol].to_numpy(float)
        return np.column_stack([d.m_us_blind.to_numpy(float) - o,
                                d.rest_diff.to_numpy(float), np.abs(o)])
    X = des(tr)
    y = (tr.margin_actual - tr[ocol]).to_numpy(float)
    b = np.linalg.solve(X.T @ X + LAM * np.eye(3), X.T @ y)
    m_off = te[ocol].to_numpy(float) + des(te) @ b
    return np.sign(m_off - te[ocol].to_numpy(float)), float(b[0])


def main():
    u = pd.read_csv(ROOT / "data" / "odds_unified.csv.gz")
    u["game_id"] = zf(u["game_id"])
    pit = pd.read_csv(ROOT / "data" / "pit_frame.csv.gz")
    pit["game_id"] = zf(pit["game_id"])
    u = u.merge(pit[["game_id", "rest_home", "rest_away"]], on="game_id",
                how="left")
    u["rest_diff"] = (u.rest_home.clip(upper=7).fillna(0)
                      - u.rest_away.clip(upper=7).fillna(0))
    u = u.dropna(subset=["margin_actual", "m_us_blind"]).copy()

    rows = []
    for feed, ocol, ccol in (("OLD (stitched)", "open_margin", "close_margin"),
                             ("NEW (unified)", "u_open", "u_close")):
        d = u.dropna(subset=[ocol, ccol]).sort_values(["season", "game_id"])
        seasons = sorted(d.season.unique())
        for i, s in enumerate(seasons):
            if i == 0:
                continue
            tr, te = d[d.season.isin(seasons[:i])], d[d.season == s]
            if len(tr) < 500 or len(te) < 100:
                continue
            side, bedge = offset_side(tr, te, ocol, ccol)
            mv = (te[ccol] - te[ocol]).to_numpy(float)
            amv = np.abs(mv).mean()
            rows.append(dict(feed=feed, season=s, n=len(te),
                             abs_move=float(amv),
                             clv=float((side * mv).mean()),
                             capture=float((side * mv).mean() / amv)
                             if amv > 0 else np.nan,
                             edge_coef=bedge))
    r = pd.DataFrame(rows)

    print("=" * 84)
    print("SEASON DIAGNOSTICS, OLD FEED vs CONVENTION-CONSISTENT FEED")
    print("=" * 84)
    piv = r.pivot(index="season", columns="feed",
                  values=["abs_move", "clv", "capture"])
    print(piv.to_string(float_format=lambda v: f"{v:9.4f}"))

    print("\n" + "=" * 84)
    print("IS 2024-25 STILL AN OUTLIER?  (2025-26 EXCLUDED — convention break)")
    print("=" * 84)
    out = {}
    for feed in ("OLD (stitched)", "NEW (unified)"):
        sub = r[(r.feed == feed) & (r.season != "2025-26")].set_index("season")
        print(f"\n  --- {feed} --- ({len(sub)} seasons)")
        for c in ("clv", "capture", "abs_move"):
            v = sub[c].dropna()
            if "2024-25" not in v.index:
                continue
            z = float((v["2024-25"] - v.mean()) / v.std(ddof=1))
            rank = int((v > v["2024-25"]).sum()) + 1
            p1 = float(2 * (1 - stats.norm.cdf(abs(z))))
            pmax = float(1 - (1 - p1) ** len(v))
            print(f"    {c:9} {v['2024-25']:+8.4f}   rank {rank:2d}/{len(v)}   "
                  f"z {z:+5.2f}   p(max) {pmax:.4f}")
            out[f"{feed}|{c}"] = dict(value=float(v["2024-25"]), rank=rank,
                                      z=z, p_max=pmax, k=len(v))

    old = r[(r.feed == "OLD (stitched)") & (r.season != "2025-26")]
    new = r[(r.feed == "NEW (unified)") & (r.season != "2025-26")]
    print("\n" + "-" * 84)
    print(f"  corr(CLV, abs_move) across seasons:  "
          f"OLD {np.corrcoef(old.clv, old.abs_move)[0,1]:+.3f}   "
          f"NEW {np.corrcoef(new.clv, new.abs_move)[0,1]:+.3f}")
    print(f"  2024-25 mean |close-open|:  "
          f"OLD {old.set_index('season').abs_move.get('2024-25', np.nan):.4f}   "
          f"NEW {new.set_index('season').abs_move.get('2024-25', np.nan):.4f}")

    b = r[r.season == "2025-26"]
    if len(b):
        print("\n  2025-26, reported separately and NOT compared across seasons:")
        for _, x in b.iterrows():
            print(f"    {x.feed:16} n={x.n:5}  move {x.abs_move:.4f}  "
                  f"CLV {x.clv:+.4f}  capture {x.capture:+.4f}")

    json.dump({"rows": rows, "outlier": out},
              open(ROOT / "data" / "d248b_rescore.json", "w"), default=float)
    print("\nwrote data/d248b_rescore.json")


if __name__ == "__main__":
    main()
