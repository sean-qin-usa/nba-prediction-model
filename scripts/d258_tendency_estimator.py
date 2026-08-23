#!/usr/bin/env python3
"""D258 — A PIT PLAYER-TENDENCY ESTIMATOR WITH PER-AXIS SHRINKAGE, FITTED.

D257 found tendencies are highly reliable (fg3_rate split-half +0.953) and that
how fast they switch differs sharply by axis: ~79% of a big three-point-rate
change is present within ten games, against ~57% for assist rate and ~47% for
turnover rate. The stated design consequence was that a PIT estimator must not
use one window for every axis.

Production does not do this. It carries exactly two things per player -- a DARKO
scalar and `trailing_min`, a fixed 10-game minutes average -- and no tendency at
all. So this builds the estimator, and it FITS the per-axis constant instead of
asserting it, which makes D257's reading falsifiable rather than decorative.

THE ESTIMATOR is empirical-Bayes shrinkage in the natural denominator, not a
window:

    tendency_hat = (numerator_to_date + k * base) / (denominator_to_date + k)

`k` is in units of the denominator -- field-goal attempts, made shots,
possessions -- and is exactly "how much prior evidence the base is worth". A
small k means the current season overwhelms the base quickly (a fast-switching
axis); a large k means the base persists (a slow one). This is strictly better
than a window: a window throws away everything older than its edge and weights
everything inside it equally, whereas shrinkage degrades smoothly and handles a
player with 4 attempts and one with 400 in the same expression.

`base` is the player's PRIOR-SEASON rate where it exists, else the league rate.
Both are point-in-time: nothing from the current game or later enters.

THE FALSIFIABLE PREDICTION. If D257 is right, the fitted k must be SMALL for
fg3_rate and rim_rate and LARGE for ast_rate and tov_rate. If they come back in
the wrong order, or all equal, the per-axis claim was wrong and one window would
have been fine.

Fitting is walk-forward: k is chosen on seasons strictly before s and scored on
season s, so no k is ever fitted on the games it is judged on. The endpoint is
attempt-weighted squared error against the next game's observed rate, since a
20-attempt game should count for more than a 2-attempt one.
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

# axis -> (numerator column, denominator column)
AXES = {
    "fg3_rate": ("fg3a", "fga"),
    "rim_rate": ("rima", "fga"),
    "ast_rate": ("ast", "fgm"),
    "ftr":      ("fta", "fga"),
    "tov_rate": ("tov", "poss"),
}
K_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
MIN_DEN = 3          # a game needs this much denominator to be scored


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def load():
    import duckdb
    con = duckdb.connect(str(ROOT / "data" / "nba.duckdb"), read_only=True)
    df = con.execute("""
        SELECT CAST(game_id AS VARCHAR) gid, player_id, seconds,
               fga, fgm, fg3a, fta, oreb, tov, ast,
               COALESCE(rima,0) rima
        FROM player_game_stats
        WHERE CAST(game_id AS VARCHAR) LIKE '002%' AND seconds > 0""").df()
    con.close()
    df["gid"] = df.gid.str.zfill(10)
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz",
                    usecols=["game_id", "season", "game_date"])
    f["game_id"] = zf(f["game_id"]); f["game_date"] = pd.to_datetime(f.game_date)
    df = df.merge(f, left_on="gid", right_on="game_id", how="inner")
    df["poss"] = df.fga + 0.44 * df.fta - df.oreb + df.tov
    return df.sort_values(["player_id", "game_date"]).reset_index(drop=True)


def main():
    d = load()
    seasons = sorted(d.season.unique())
    print(f"{len(d):,} player-games, {d.player_id.nunique():,} players, "
          f"{len(seasons)} seasons")

    results = {}
    for ax, (ncol, dcol) in AXES.items():
        g = d[(d[dcol] > 0)].copy()
        num, den = g[ncol].to_numpy(float), g[dcol].to_numpy(float)

        # --- PIT cumulative sums within (player, season), shifted by one game
        key = [g.player_id, g.season]
        cn = g.groupby(key)[ncol].cumsum().to_numpy(float) - num
        cd = g.groupby(key)[dcol].cumsum().to_numpy(float) - den

        # --- base: prior-season player rate, else league rate (both PIT)
        ps = (g.groupby(["player_id", "season"])[[ncol, dcol]].sum()
                .reset_index())
        ps["rate"] = ps[ncol] / ps[dcol].replace(0, np.nan)
        nxt = {seasons[i]: seasons[i + 1] for i in range(len(seasons) - 1)}
        ps["target"] = ps.season.map(nxt)
        prior = ps.dropna(subset=["target"]).set_index(
            ["target", "player_id"]).rate
        lg = g.groupby("season").apply(
            lambda x: x[ncol].sum() / max(x[dcol].sum(), 1e-9),
            include_groups=False)
        lg_prior = {s: lg.get({v: k for k, v in nxt.items()}.get(s), lg.mean())
                    for s in seasons}
        base = np.array([prior.get((s, p), np.nan)
                         for s, p in zip(g.season, g.player_id)])
        lgv = g.season.map(lg_prior).to_numpy(float)
        base = np.where(np.isfinite(base), base, lgv)

        obs = num / den
        scoreable = (den >= MIN_DEN) & np.isfinite(base) & (cd > 0)
        sc = g.season.to_numpy()

        # --- walk-forward: pick k on prior seasons, score on season s
        per_season, chosen = [], []
        for i, s in enumerate(seasons):
            if i < 3:
                continue
            tr = scoreable & (sc < s)
            te = scoreable & (sc == s)
            if tr.sum() < 5000 or te.sum() < 500:
                continue
            best_k, best_e = None, np.inf
            for k in K_GRID:
                pred = (cn[tr] + k * base[tr]) / (cd[tr] + k)
                e = float(np.average((pred - obs[tr]) ** 2, weights=den[tr]))
                if e < best_e:
                    best_e, best_k = e, k
            pred = (cn[te] + best_k * base[te]) / (cd[te] + best_k)
            err = float(np.average((pred - obs[te]) ** 2, weights=den[te]))
            # reference arms
            e_base = float(np.average((base[te] - obs[te]) ** 2, weights=den[te]))
            cur = np.where(cd[te] > 0, cn[te] / np.maximum(cd[te], 1e-9),
                           base[te])
            e_cur = float(np.average((cur - obs[te]) ** 2, weights=den[te]))
            per_season.append(dict(season=s, k=best_k, mse=err,
                                   mse_base=e_base, mse_current=e_cur))
            chosen.append(best_k)
        r = pd.DataFrame(per_season)
        med_k = int(np.median(chosen)) if chosen else None
        gain_b = 100 * (1 - r.mse.mean() / r.mse_base.mean())
        gain_c = 100 * (1 - r.mse.mean() / r.mse_current.mean())
        results[ax] = dict(median_k=med_k, k_by_season=chosen,
                           mse=float(r.mse.mean()),
                           mse_base=float(r.mse_base.mean()),
                           mse_current=float(r.mse_current.mean()),
                           gain_vs_base=float(gain_b),
                           gain_vs_current=float(gain_c),
                           denom=dcol)
        print(f"\n  --- {ax} (shrinking in {dcol}) ---")
        print(f"    fitted k per season: {chosen}")
        print(f"    median k = {med_k} {dcol}   "
              f"[k is the prior's weight in denominator units]")
        print(f"    weighted MSE  shrunk {r.mse.mean():.5f}  |  "
              f"prior-only {r.mse_base.mean():.5f}  |  "
              f"season-to-date-only {r.mse_current.mean():.5f}")
        print(f"    improvement: {gain_b:+.1f}% vs prior-only, "
              f"{gain_c:+.1f}% vs current-only")

    print("\n" + "=" * 76)
    print("THE FALSIFIABLE PREDICTION FROM D257")
    print("=" * 76)
    print("  D257 said fg3/rim switch FAST (small k) and ast/tov switch SLOW")
    print("  (large k). Fitted medians, in units of each axis's denominator:\n")
    order = sorted(results.items(), key=lambda kv: kv[1]["median_k"] or 0)
    for ax, v in order:
        print(f"    {ax:10} k = {v['median_k']:5}  ({v['denom']})")
    fast = {a for a, v in results.items()
            if v["median_k"] is not None and v["median_k"] <= 32}
    ok = "fg3_rate" in fast and "tov_rate" not in fast
    print(f"\n  >>> {'CONSISTENT with D257' if ok else 'INCONSISTENT — the per-axis claim does not hold'}")

    json.dump(results, open(ROOT / "data" / "d258_tendency_k.json", "w"),
              default=float)
    print("\nwrote data/d258_tendency_k.json")


if __name__ == "__main__":
    main()
