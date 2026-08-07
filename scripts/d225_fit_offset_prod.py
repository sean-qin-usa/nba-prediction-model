#!/usr/bin/env python3
"""D225 — fit and freeze the PRODUCTION offset coefficients.

The backtest refits `f` inside every walk-forward fold. Production needs one
frozen coefficient vector, fitted on all history available before the season it
will be used in, and versioned so a later refit is a visible change rather than
a silent one.

    data/offset_coefs.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import nbapred.threads; nbapred.threads.pin(1)
import numpy as np, pandas as pd

FE = ["edge", "rest_diff", "abs_open"]
LAM = 3000.0                      # same shrink as the gated backtest arm

f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
f = f.dropna(subset=["m_us_blind", "open_margin", "margin_actual"]).copy()
f["game_date"] = pd.to_datetime(f["game_date"])
f["edge"] = f["m_us_blind"] - f["open_margin"]
f["abs_open"] = f["open_margin"].abs()

from nbapred.db import connect
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

X = f[FE].to_numpy(float)
r = (f["margin_actual"] - f["open_margin"]).to_numpy(float)
b = np.linalg.solve(X.T @ X + LAM * np.eye(len(FE)), X.T @ r)
out = dict(features=FE, coefs=b.tolist(), lam=LAM, n=int(len(f)),
           seasons=sorted(f["season"].unique()),
           fitted_through=str(f["game_date"].max().date()),
           gate="D224 (t=-3.68, CI [-0.0106,-0.0021], 6/7 seasons)",
           note="m_offset = m_open + coefs . (edge, rest_diff, |m_open|)")
json.dump(out, open(ROOT / "data" / "offset_coefs.json", "w"), indent=1)
print(json.dumps(out, indent=1))
