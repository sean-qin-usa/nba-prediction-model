#!/usr/bin/env python3
"""D204 — build a trading frame whose m_us IS the offset construction.

m_offset = open_margin + f(x),  f a ridge shrunk hard toward zero, FITTED
WALK-FORWARD (season k+1 uses a model fitted on seasons 1..k only).  Written to
data/ats19_frame_offset.csv.gz so the SAME production pipeline (as_adaptive ->
wf_equity) that produced every published trading figure can score it.

This is the first time the offset construction is traded through production
machinery rather than a scratch harness.

Features (all knowable at the open): edge = m_us - open_margin, rest
differential, |open_margin|.  Base m_us is the HONEST frame (D203).
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import nbapred.threads; nbapred.threads.pin(1)
import numpy as np, pandas as pd
from nbapred.db import connect

f = pd.read_csv(ROOT / "data" / "ats19_frame_honest.csv.gz")
f["game_date"] = pd.to_datetime(f["game_date"])
con = connect(read_only=True)
g = pd.DataFrame(con.execute(
    "SELECT DISTINCT game_date, team_abbrev FROM nba_games "
    "WHERE game_id LIKE '002%'").fetchall(), columns=["d", "t"])
con.close()
g["d"] = pd.to_datetime(g["d"]); g = g.sort_values("d")
g["prev"] = g.groupby("t")["d"].shift(1)
rest = {(x.t, x.d): min((x.d - x.prev).days, 7) if pd.notna(x.prev) else 7
        for x in g.itertuples()}
f["rest_diff"] = [rest.get((h, d), 7) - rest.get((a, d), 7)
                  for h, a, d in zip(f["home"], f["away"], f["game_date"])]
f["edge"] = f["m_us"] - f["open_margin"]
f["abs_open"] = f["open_margin"].abs()
FE = ["edge", "rest_diff", "abs_open"]
LAM = 3000.0          # strong shrink toward "the opener is right"

seasons = sorted(f["season"].unique())
m_off = f["m_us"].to_numpy(float).copy()     # pre-window seasons keep m_us
coefs = {}
for i, s in enumerate(seasons):
    if i < 3:
        continue                              # need history to fit
    tr = f[f["season"].isin(seasons[:i])]
    te_i = f.index[f["season"] == s]
    X = tr[FE].to_numpy(float)
    r = (tr["margin_actual"] - tr["open_margin"]).to_numpy(float) \
        if "margin_actual" in tr else None
    if r is None:
        raise SystemExit("frame lacks margin_actual")
    b = np.linalg.solve(X.T @ X + LAM * np.eye(len(FE)), X.T @ r)
    coefs[s] = b.tolist()
    m_off[te_i] = (f.loc[te_i, "open_margin"].to_numpy(float)
                   + f.loc[te_i, FE].to_numpy(float) @ b)
    print(f"  {s}: fit on {len(tr):6d} prior games  b={np.round(b,4).tolist()}")

f["m_us_blind"] = f["m_us"]
f["m_us"] = m_off
f = f.drop(columns=["rest_diff", "edge", "abs_open"])
out = ROOT / "data" / "ats19_frame_offset.csv.gz"
f.to_csv(out, index=False, compression="gzip")
print(f"\nwrote {out}")
d = (f["m_us"] - f["m_us_blind"]).abs()
print(f"  |m_offset - m_blind|: mean {d.mean():.3f} pts, max {d.max():.3f}")
e0 = (f["m_us_blind"] - f["open_margin"]).abs()
e1 = (f["m_us"] - f["open_margin"]).abs()
print(f"  mean |edge| vs opener: blind {e0.mean():.3f} -> offset {e1.mean():.3f} "
      f"({100*e1.mean()/e0.mean():.0f}% — the ridge shrinks the disagreement)")
