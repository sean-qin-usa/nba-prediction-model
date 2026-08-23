#!/usr/bin/env python3
"""D260 — THE SHIPPED MINUTES WINDOW WAS CHOSEN, NEVER FITTED. FIT IT.

`composition.py` weights every player by `trailing_min` — "avg of last 10 games
actually played (>=12 min)". Ten is a round number that appears nowhere in the
register as a measured choice. D258 showed that for tendencies the right amount
of history varies 16x by quantity, and that a flat window is the wrong shape
anyway: it discards everything past its edge and weights everything inside it
equally.

Minutes is the SECOND of the two things production carries per player, and it
multiplies talent directly, so an error here scales the whole composition
channel. This fits it.

ARMS, all strictly PIT, all predicting the next game's minutes GIVEN the player
plays (which is the shipped conditioning — availability is a separate leg):

  SHIPPED   flat mean of the last 10 games played with >= 12 minutes
  FLAT-w    flat mean of the last w games, w fitted
  EWMA-h    exponentially weighted, half-life h fitted
  SHRUNK-k  (sum_to_date + k * base) / (n_to_date + k), base = prior-season
            mean minutes, else league mean for that trailing-minutes band
  EWMA+SHR  EWMA for the recency signal, shrunk toward the prior-season base

Everything is fitted walk-forward: parameters chosen on seasons strictly before
s, scored on s. Endpoint is squared error on the next game's minutes.

WHY THIS MIGHT MATTER AND WHY IT MIGHT NOT. D245d is the precedent that governs
here: L1, L2 and L3 all improved significantly and L4 -- the layer production
actually emits -- was still null, because a composition gain reaches the emitted
margin at roughly 1/4.5 (blend share 0.652 x offset edge 0.3413, magnitude
corrected in D252). So a minutes improvement must be LARGE at L1 to survive.
This entry measures L1 only. If L1 is not large, there is nothing to propagate
and L4 is not worth running.
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

MIN_MIN = 12.0        # the shipped floor
WINDOWS = [3, 5, 8, 10, 15, 20, 30, 40]
HALFLIVES = [2, 3, 5, 8, 12, 20, 30]
KS = [1, 2, 4, 8, 16, 32, 64]


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def clus(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    k = len(v); se = v.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return v.mean(), v.mean() - tc * se, v.mean() + tc * se, k


def load():
    import duckdb
    con = duckdb.connect(str(ROOT / "data" / "nba.duckdb"), read_only=True)
    df = con.execute("""
        SELECT CAST(game_id AS VARCHAR) gid, player_id, seconds
        FROM player_game_stats
        WHERE CAST(game_id AS VARCHAR) LIKE '002%' AND seconds > 0""").df()
    con.close()
    df["gid"] = df.gid.str.zfill(10)
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz",
                    usecols=["game_id", "season", "game_date"])
    f["game_id"] = zf(f["game_id"]); f["game_date"] = pd.to_datetime(f.game_date)
    df = df.merge(f, left_on="gid", right_on="game_id", how="inner")
    df["mins"] = df.seconds / 60.0
    df = df[df.mins >= MIN_MIN]        # the shipped universe
    return df.sort_values(["player_id", "game_date"]).reset_index(drop=True)


def ewma_prev(s, halflife):
    """EWMA of prior values only (shifted), per group."""
    return s.shift(1).ewm(halflife=halflife, min_periods=2).mean()


def main():
    d = load()
    seasons = sorted(d.season.unique())
    print(f"{len(d):,} player-games at >= {MIN_MIN:.0f} min, "
          f"{d.player_id.nunique():,} players, {len(seasons)} seasons")

    g = d.groupby(["player_id", "season"])["mins"]
    # shipped: flat mean of last 10 PLAYED games
    d["ship"] = g.transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    for w in WINDOWS:
        d[f"flat{w}"] = g.transform(
            lambda s, w=w: s.shift(1).rolling(w, min_periods=2).mean())
    for h in HALFLIVES:
        d[f"ewma{h}"] = g.transform(lambda s, h=h: ewma_prev(s, h))
    # shrinkage toward prior-season mean
    ps = d.groupby(["player_id", "season"]).mins.mean().reset_index()
    nxt = {seasons[i]: seasons[i + 1] for i in range(len(seasons) - 1)}
    ps["tgt"] = ps.season.map(nxt)
    pri = ps.dropna(subset=["tgt"]).set_index(["tgt", "player_id"]).mins
    base = np.array([pri.get((s, p), np.nan)
                     for s, p in zip(d.season, d.player_id)])
    lg = d.groupby("season").mins.mean()
    prev = {v: k for k, v in nxt.items()}
    lgv = d.season.map({s: lg.get(prev.get(s), lg.mean())
                        for s in seasons}).to_numpy(float)
    base = np.where(np.isfinite(base), base, lgv)
    d["base"] = base
    csum = g.cumsum().to_numpy(float) - d.mins.to_numpy(float)
    cnt = g.cumcount().to_numpy(float)
    for k in KS:
        d[f"shr{k}"] = (csum + k * base) / (cnt + k)
    for h in (3, 5, 8):
        for k in (2, 4, 8):
            e = d[f"ewma{h}"].to_numpy(float)
            e = np.where(np.isfinite(e), e, base)
            wgt = cnt / (cnt + k)
            d[f"es{h}_{k}"] = wgt * e + (1 - wgt) * base

    arms = (["ship"] + [f"flat{w}" for w in WINDOWS]
            + [f"ewma{h}" for h in HALFLIVES] + [f"shr{k}" for k in KS]
            + [f"es{h}_{k}" for h in (3, 5, 8) for k in (2, 4, 8)])
    y = d.mins.to_numpy(float)
    ok_base = d.ship.notna().to_numpy() & np.isfinite(base)
    sc = d.season.to_numpy()

    # ---- walk-forward: pick the best arm on prior seasons, score on s ----
    rows = []
    for i, s in enumerate(seasons):
        if i < 3:
            continue
        tr = ok_base & (sc < s); te = ok_base & (sc == s)
        if tr.sum() < 20000 or te.sum() < 2000:
            continue
        errs = {}
        for a in arms:
            v = d[a].to_numpy(float)
            m = tr & np.isfinite(v)
            errs[a] = float(np.mean((v[m] - y[m]) ** 2))
        best = min(errs, key=errs.get)
        row = {"season": s, "best": best}
        for a in ("ship", best):
            v = d[a].to_numpy(float)
            m = te & np.isfinite(v)
            row[f"mse_{'ship' if a=='ship' else 'best'}"] = float(
                np.mean((v[m] - y[m]) ** 2))
        rows.append(row)
    r = pd.DataFrame(rows)
    print("\n=== walk-forward: chosen arm and test MSE vs the shipped window ===")
    print(r.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

    imp = 100 * (1 - r.mse_best / r.mse_ship)
    m, lo, hi, k = clus(imp)
    print(f"\n  arm chosen most often: {r.best.mode().iloc[0]} "
          f"({(r.best == r.best.mode().iloc[0]).sum()}/{len(r)} seasons)")
    print(f"  MSE reduction vs shipped: {m:+.2f}%  CI [{lo:+.2f}%, {hi:+.2f}%] "
          f"k={k}  better {int((imp > 0).sum())}/{k}")
    print(f"  shipped RMSE {np.sqrt(r.mse_ship.mean()):.3f} min  ->  "
          f"best {np.sqrt(r.mse_best.mean()):.3f} min")

    # ---- full arm table on the last-half seasons, for shape ----
    print("\n=== every arm, pooled MSE over the scored seasons ===")
    tab = []
    for a in arms:
        v = d[a].to_numpy(float)
        m = ok_base & np.isfinite(v) & (sc >= seasons[3])
        tab.append((a, float(np.mean((v[m] - y[m]) ** 2))))
    tab.sort(key=lambda x: x[1])
    ship_mse = dict(tab)["ship"]
    for a, e in tab[:10]:
        print(f"    {a:10} {e:8.4f}   {100*(1-e/ship_mse):+6.2f}% vs shipped"
              f"{'   <- SHIPPED' if a == 'ship' else ''}")
    print(f"    ... shipped ranks {[a for a,_ in tab].index('ship')+1} of {len(tab)}")

    def _num(x):
        a = np.asarray(x, dtype=float).ravel()
        return float(a[0]) if a.size == 1 else [float(v) for v in a]

    json.dump({"per_season": rows,
               "pooled": [(a, float(e)) for a, e in tab],
               "improvement_pct": _num(m), "ci": [_num(lo), _num(hi)]},
              open(ROOT / "data" / "d260_minutes.json", "w"), default=_num)
    print("\nwrote data/d260_minutes.json")


if __name__ == "__main__":
    main()
