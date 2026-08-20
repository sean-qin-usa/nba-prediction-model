#!/usr/bin/env python3
"""IG probe (read-only, SQL only): dispersion family check for the sim's count
model, extended to assists and rebounds (candidates e/f).

Per player (>=30 games >=720s, 2025-26): minutes-scaled residual Fano factors
(Poisson=1) for zone attempts, FGA, FTA, AST, REB; cross-zone residual corrs
(independent Poisson => 0); FTA-rim coupling; and the assist question directly:
does AST dispersion exceed Poisson even GIVEN minutes (sim uses fixed exposure
=> Fano exactly 1, no minutes mixing at all)?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from nbapred.db import connect

con = connect(read_only=True)
df = con.execute("""
    SELECT player_id, seconds/60.0 mins, rima, mida, thra, fta, fga, ast,
           oreb + dreb reb
    FROM player_game_stats WHERE game_id LIKE '00225%' AND seconds >= 720""").fetchdf()
con.close()

cols = ("rima", "mida", "thra", "fga", "fta", "ast", "reb")
fano = {z: [] for z in cols}
fano_fixed = {z: [] for z in ("ast",)}   # residual vs FLAT mean (sim's fixed expo)
cors = {k: [] for k in ("rim_thr", "rim_mid", "mid_thr", "fta_rim")}
for pid, g in df.groupby("player_id"):
    if len(g) < 30:
        continue
    m = g["mins"].to_numpy()
    resid = {}
    for z in cols:
        c = g[z].to_numpy().astype(float)
        rate = c.sum() / m.sum()
        mu = rate * m
        if mu.mean() < 1.0:
            continue
        resid[z] = c - mu
        fano[z].append(np.var(c - mu) / mu.mean())
    # assists vs fixed exposure (what the sim actually does): mu = const
    c = g["ast"].to_numpy().astype(float)
    if c.mean() >= 1.0:
        fano_fixed["ast"].append(np.var(c - c.mean()) / c.mean())
    if all(k in resid for k in ("rima", "mida", "thra")):
        cors["rim_thr"].append(np.corrcoef(resid["rima"], resid["thra"])[0, 1])
        cors["rim_mid"].append(np.corrcoef(resid["rima"], resid["mida"])[0, 1])
        cors["mid_thr"].append(np.corrcoef(resid["mida"], resid["thra"])[0, 1])
    if "fta" in resid and "rima" in resid:
        cors["fta_rim"].append(np.corrcoef(resid["fta"], resid["rima"])[0, 1])

print("Fano of minutes-scaled residuals (Poisson=1; sim draws Poisson given mins):")
for z in cols:
    print(f"  {z:5s}: median {np.median(fano[z]):.2f}  (n={len(fano[z])})")
print(f"AST Fano vs FLAT exposure (the sim's actual construction => wants 1): "
      f"median {np.median(fano_fixed['ast']):.2f}")
print("cross-count residual corr (indep-Poisson sim => 0):")
for k, v in cors.items():
    print(f"  {k:8s}: median {np.median(v):+.3f}")
