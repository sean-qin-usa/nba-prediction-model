#!/usr/bin/env python3
"""Audit: re-derive the props-engine PIT calibration claim (mean~0.507, std~0.296,
FEATURE_LEDGER "Conditioning fix + truncated minutes") on 2025-26 data, and probe
the dispersion-by-construction questions:

  1. PIT (points) on 2025-26 regular season, same conditioning as validate_props
     (seconds>=720, >=8 prior games, proj_min>=20) — via the LIVE code path
     (empirical minutes hist) AND the truncated-normal path (hist stripped), since
     the ledger's ACCEPTED row is "truncated minutes" but player_rates_from_stats
     always returns minutes_hist, making the empirical path the de-facto default.
  2. Assists PIT — assists are decoupled from the minutes draw (D30) and use a
     FIXED exposure, i.e. pure Poisson with no mixing: expected under-dispersion.
  3. Structural battery (data, no sim): per-zone attempt dispersion vs Poisson
     (Fano on minutes-scaled residuals), cross-zone residual correlations
     (independent-Poisson vs multinomial-given-total), FTA-vs-rim-attempt
     residual correlation (FT/foul coupling ignored by the sim).

Read-only DB. New file per ground rules (never edits nbapred/).
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect
from nbapred.engine.props import player_rates_from_stats, simulate_player


def crps(samples, y):
    s = np.sort(samples); n = len(s)
    return float(np.mean(np.abs(s - y)) - 0.5 * (2 * np.arange(1, n + 1) - n - 1) @ s / n**2)


def pit_stats(pits, label):
    p = np.array(pits)
    h, _ = np.histogram(p, bins=10, range=(0, 1))
    tails = (h[0] + h[-1]) / len(p)          # mass in outer deciles; uniform = 0.20
    print(f"  {label:26s} mean {p.mean():.3f}  std {p.std():.3f}  "
          f"tail-mass(outer deciles) {tails:.3f} (uniform 0.200)")
    print(f"    hist: {(h / len(p) * 10).round(2).tolist()}")
    return p


def main(min_prior_games=8, min_proj_min=20, sims=2000, max_eval=1500):
    rng_master = np.random.default_rng(7)
    con = connect(read_only=True)
    pg = con.execute("""
        SELECT s.player_id, g.game_date, s.pts, s.ast, s.seconds, s.fga
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '00225%' AND s.seconds >= 720
        ORDER BY g.game_date, s.player_id
    """).fetchdf()
    print(f"2025-26 candidate player-games (>=12 min, reg season): {len(pg)}")

    # stride-sample across the WHOLE season (validate_props takes the first
    # max_eval in date order; striding removes the early-season skew)
    stride = max(1, len(pg) // (max_eval * 2))   # oversample; filters cut ~half
    rows = pg.iloc[::stride]

    res = {k: [] for k in ("pit_hist", "pit_trunc", "pit_rand_hist", "pit_ast",
                           "crps_hist", "crps_trunc", "z2_hist", "z2_trunc", "z2_ast")}
    cover80_hist = cover80_trunc = n = 0
    for r in rows.itertuples():
        if n >= max_eval:
            break
        rates = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
        if rates is None or rates["n_games"] < min_prior_games or rates["proj_min"] < min_proj_min:
            continue
        y, ya = float(r.pts), float(r.ast)

        # LIVE path (empirical minutes hist — what production actually runs)
        sim = simulate_player(rates, n=sims, seed=n)
        pts = sim["points"]; ast = sim["assists"]
        res["pit_hist"].append(np.mean(pts < y) + 0.5 * np.mean(pts == y))
        u = rng_master.random()
        res["pit_rand_hist"].append(np.mean(pts < y) + u * np.mean(pts == y))
        res["crps_hist"].append(crps(pts, y))
        res["z2_hist"].append(((y - pts.mean()) / max(pts.std(), 1e-6)) ** 2)
        lo, hi = np.percentile(pts, [10, 90])
        cover80_hist += int(lo <= y <= hi)
        res["pit_ast"].append(np.mean(ast < ya) + 0.5 * np.mean(ast == ya))
        res["z2_ast"].append(((ya - ast.mean()) / max(ast.std(), 1e-6)) ** 2)

        # Truncated-normal minutes path (the ledger's ACCEPTED construction)
        rt = dict(rates); rt.pop("minutes_hist", None)
        pts_t = simulate_player(rt, n=sims, seed=n)["points"]
        res["pit_trunc"].append(np.mean(pts_t < y) + 0.5 * np.mean(pts_t == y))
        res["crps_trunc"].append(crps(pts_t, y))
        res["z2_trunc"].append(((y - pts_t.mean()) / max(pts_t.std(), 1e-6)) ** 2)
        lo, hi = np.percentile(pts_t, [10, 90])
        cover80_trunc += int(lo <= y <= hi)
        n += 1

    print(f"\nevaluated player-games: {n}")
    print("\nPOINTS PIT (claim to replicate: mean 0.507 / std 0.296):")
    pit_stats(res["pit_hist"], "LIVE (empirical hist)")
    pit_stats(res["pit_rand_hist"], "LIVE randomized-PIT")
    pit_stats(res["pit_trunc"], "truncated-normal path")
    print(f"  80% coverage: hist {cover80_hist/n:.3f}  trunc {cover80_trunc/n:.3f} (want 0.800)")
    print(f"  CRPS: hist {np.mean(res['crps_hist']):.3f}  trunc {np.mean(res['crps_trunc']):.3f}")
    print(f"  mean z^2 (1=dispersion right, >1=sim too narrow): "
          f"hist {np.mean(res['z2_hist']):.3f}  trunc {np.mean(res['z2_trunc']):.3f}")

    print("\nASSISTS PIT (decoupled fixed-exposure Poisson, D30):")
    pit_stats(res["pit_ast"], "LIVE assists")
    print(f"  mean z^2 assists: {np.mean(res['z2_ast']):.3f}")

    # ---- structural battery: are independent Poisson attempts the right family?
    print("\nSTRUCTURAL (2025-26, players with >=30 qualifying games):")
    df = con.execute("""
        SELECT s.player_id, s.seconds/60.0 mins, s.rima, s.mida, s.thra, s.fta, s.fga
        FROM player_game_stats s
        WHERE s.game_id LIKE '00225%' AND s.seconds >= 720
    """).fetchdf()
    con.close()
    fano = {z: [] for z in ("rima", "mida", "thra", "fga", "fta")}
    cors = {k: [] for k in ("rim_thr", "rim_mid", "mid_thr", "fta_rim")}
    for pid, g in df.groupby("player_id"):
        if len(g) < 30:
            continue
        m = g["mins"].to_numpy()
        resid = {}
        for z in fano:
            c = g[z].to_numpy().astype(float)
            rate = c.sum() / m.sum()
            mu = rate * m
            if mu.mean() < 1.0:
                continue
            resid[z] = c - mu
            fano[z].append(np.var(c - mu) / mu.mean())   # Poisson => 1
        if all(k in resid for k in ("rima", "mida", "thra")):
            cors["rim_thr"].append(np.corrcoef(resid["rima"], resid["thra"])[0, 1])
            cors["rim_mid"].append(np.corrcoef(resid["rima"], resid["mida"])[0, 1])
            cors["mid_thr"].append(np.corrcoef(resid["mida"], resid["thra"])[0, 1])
        if "fta" in resid and "rima" in resid:
            cors["fta_rim"].append(np.corrcoef(resid["fta"], resid["rima"])[0, 1])
    print("  Fano factor of minutes-scaled residuals (Poisson=1.0, >1 over-dispersed):")
    for z in fano:
        if fano[z]:
            print(f"    {z:5s}: median {np.median(fano[z]):.2f}  (n={len(fano[z])} players)")
    print("  cross-count residual correlations (indep-Poisson sim => 0):")
    for k, v in cors.items():
        if v:
            print(f"    {k:8s}: median {np.median(v):+.3f}")


if __name__ == "__main__":
    main()
