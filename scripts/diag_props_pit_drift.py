#!/usr/bin/env python3
"""Diagnosis companion to audit_props_pit (THREAD-5a): WHERE does the post-D79
points-PIT decentering live? D79's journal (063f19) attributed the clean-universe
PIT drop (0.498 -> ~0.485) to a real ~+0.5-min early/mid-season minutes
over-projection that preseason contamination had been masking. Confirm by
splitting points-PIT and the minutes residual (actual - proj_min) by month on
2025-26, same conditioning as the audit (>=720s, >=8 prior games, proj_min>=20,
strided). Read-only.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect
from nbapred.engine.props import player_rates_from_stats, simulate_player


def main(min_prior_games=8, min_proj_min=20, sims=1000, max_eval=1500):
    con = connect(read_only=True)
    pg = con.execute("""
        SELECT s.player_id, g.game_date, s.pts, s.seconds
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '00225%' AND s.seconds >= 720
        ORDER BY g.game_date, s.player_id
    """).fetchdf()
    stride = max(1, len(pg) // (max_eval * 2))
    rows = pg.iloc[::stride]

    recs = []   # (month, pit, min_resid, pts_resid)
    n = 0
    for r in rows.itertuples():
        if n >= max_eval:
            break
        rates = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
        if rates is None or rates["n_games"] < min_prior_games or rates["proj_min"] < min_proj_min:
            continue
        y = float(r.pts)
        pts = simulate_player(rates, n=sims, seed=n)["points"]
        pit = float(np.mean(pts < y) + 0.5 * np.mean(pts == y))
        recs.append((r.game_date.strftime("%Y-%m"), pit,
                     r.seconds / 60.0 - rates["proj_min"], y - pts.mean()))
        n += 1
    con.close()

    recs = np.array(recs, dtype=object)
    print(f"evaluated player-games: {n}")
    print(f"\n{'month':>8s} {'n':>5s} {'PITmean':>8s} {'minResid':>9s} {'ptsResid':>9s}")
    months = sorted(set(recs[:, 0]))
    for m in months:
        sub = recs[recs[:, 0] == m]
        print(f"{m:>8s} {len(sub):5d} {np.mean(sub[:, 1].astype(float)):8.3f} "
              f"{np.mean(sub[:, 2].astype(float)):+9.2f} "
              f"{np.mean(sub[:, 3].astype(float)):+9.2f}")
    pit_all = recs[:, 1].astype(float)
    mr = recs[:, 2].astype(float)
    pr = recs[:, 3].astype(float)
    print(f"\n  ALL    {n:5d} {pit_all.mean():8.3f} {mr.mean():+9.2f} {pr.mean():+9.2f}")
    # split halves (early = Oct-Dec, late = Jan-Jun)
    early = np.array([m < "2026-01" for m in recs[:, 0]])
    for lab, mask in (("Oct-Dec", early), ("Jan-Jun", ~early)):
        print(f"  {lab:6s} {mask.sum():5d} {pit_all[mask].mean():8.3f} "
              f"{mr[mask].mean():+9.2f} {pr[mask].mean():+9.2f}")


if __name__ == "__main__":
    main()
