#!/usr/bin/env python3
"""D247d — IS 2024-25's RANK-1 CLV A MARKET FACT OR A DATA-SOURCE FACT?

D247b found exactly one measure on which 2024-25 is a genuine outlier: CLV on
the model's side, +0.8626, rank 1/19, z +2.80. Everything else about the season
was ordinary -- normalised gap ranked 7th.

But CLV = side * (close - open) is MECHANICALLY PROPORTIONAL to how far the line
travels. A model with fixed directional skill earns more CLV in a season whose
lines move further, with no change in skill whatsoever. And 2024-25's movement
statistics are not those of a normal season:

    mean |close - open|   1.907   LARGEST of 19 seasons
    lines unchanged       22.3%   3rd HIGHEST of 19
    sd(open_margin)       7.713   2nd WIDEST of 19

More lines never moving, AND bigger moves when they do, AND a wider spread of
openers, is the signature of a different RECORDING SOURCE rather than a different
market. REVIEW.md already documents the change: books/game at open falls from
7.74 in 2023-24 to 1.00 in 2024-25.

So the comparison is rerun on CAPTURE FRACTION -- the share of the available
movement the model's side actually collects:

    capture = CLV / mean|close - open|

which is invariant to the scale of the recorded movement. If 2024-25 stays a
strong outlier on capture, the model really did read that season better. If it
regresses toward the pack, the rank-1 CLV was an artefact of the odds feed.

Selection is corrected explicitly: 2024-25 was picked BECAUSE it was the best
season, so the relevant question is not "how extreme is z" but "how often does
the MAXIMUM of 19 seasons reach this z", across the four measures tested.
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

N_MEASURES = 4          # pre-declared in D247b: norm_gap, CLV, mkt_err, ATS


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def main():
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
    f["game_id"] = zf(f["game_id"])
    f = f.dropna(subset=["margin_actual", "close_margin", "open_margin",
                         "m_us"]).copy()
    rows = []
    for s, g in f.groupby("season"):
        sd = np.sign(g.m_us.to_numpy(float) - g.open_margin.to_numpy(float))
        mv = (g.close_margin - g.open_margin).to_numpy(float)
        clv = (sd * mv).mean()
        amv = np.abs(mv).mean()
        moved = np.abs(mv) > 1e-9
        rows.append(dict(season=s, n=len(g), clv=float(clv),
                         abs_move=float(amv),
                         capture=float(clv / amv),
                         pct_moved=float(100 * moved.mean()),
                         sd_open=float(g.open_margin.std()),
                         # directional hit rate on games that DID move
                         dir_hit=float((sd[moved] * mv[moved] > 0).mean())))
    d = pd.DataFrame(rows).sort_values("season").reset_index(drop=True)
    print(d.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

    tgt = "2024-25"
    print(f"\n{'measure':28} {'2024-25':>9} {'rank':>6} {'z':>7} "
          f"{'p(one)':>8} {'p(max19)':>9} {'p(x4meas)':>10}")
    print("-" * 82)
    out = {}
    for c, lab in (("clv", "CLV (raw)"),
                   ("capture", "capture fraction"),
                   ("dir_hit", "direction hit | line moved"),
                   ("abs_move", "mean |close-open| (context)")):
        v = d.set_index("season")[c]
        z = float((v[tgt] - v.mean()) / v.std(ddof=1))
        rank = int((v > v[tgt]).sum()) + 1
        p1 = float(2 * (1 - stats.norm.cdf(abs(z))))
        pmax = float(1 - (1 - p1) ** len(v))
        pfam = float(1 - (1 - pmax) ** N_MEASURES)
        print(f"{lab:28} {v[tgt]:9.4f} {rank:5d}/19 {z:+7.2f} "
              f"{p1:8.4f} {pmax:9.4f} {pfam:10.4f}")
        out[c] = dict(value=float(v[tgt]), rank=rank, z=z,
                      p_one=p1, p_max19=pmax, p_family=pfam)

    print("\n  p(max19)  = chance the MOST extreme of 19 seasons reaches this z")
    print("  p(x4meas) = the same, across the four measures pre-declared in D247b")

    r = np.corrcoef(d.clv, d.abs_move)[0, 1]
    print(f"\n  corr(season CLV, season mean |close-open|) = {r:+.3f}")
    print("  -> CLV tracks how far the recorded line travels, which is a")
    print("     property of the ODDS FEED as much as of the market.")
    out["corr_clv_move"] = float(r)

    # the era split that matches the documented feed change
    pre = d[d.season <= "2023-24"]
    post = d[d.season >= "2024-25"]
    print(f"\n  feed era <=2023-24: CLV {pre.clv.mean():.4f}  "
          f"move {pre.abs_move.mean():.4f}  capture {pre.capture.mean():.4f}")
    print(f"  feed era >=2024-25: CLV {post.clv.mean():.4f}  "
          f"move {post.abs_move.mean():.4f}  capture {post.capture.mean():.4f}")
    out["era"] = dict(pre_clv=float(pre.clv.mean()),
                      post_clv=float(post.clv.mean()),
                      pre_capture=float(pre.capture.mean()),
                      post_capture=float(post.capture.mean()))

    json.dump({"rows": d.to_dict("records"), "tests": out},
              open(ROOT / "data" / "d247d_clv_norm.json", "w"), default=float)
    print("\nwrote data/d247d_clv_norm.json")


if __name__ == "__main__":
    main()
