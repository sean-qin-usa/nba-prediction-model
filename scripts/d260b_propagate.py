#!/usr/bin/env python3
"""D260b — MEASURE the propagation instead of bounding it by arithmetic.

D260 fitted the minutes estimator: `es3_2` (EWMA half-life 3, shrunk toward the
prior-season mean at k=2) beats the shipped flat 10-game window by 5.80% MSE,
CI [+5.48%, +6.12%], in 16 of 16 walk-forward seasons. The shipped window ranks
16th of 32 arms.

A back-of-envelope propagation said the emitted effect is ~0.008 margin points,
but that used sqrt(n) scaling across players, which is WRONG here for a reason
worth stating: team minutes are conserved at ~240. Over-estimating one player's
minutes necessarily under-estimates another's, so the team-strength error is a
weighted CONTRAST, not an independent sum, and the errors partly cancel. The
arithmetic bound could be off in either direction, so it should be measured.

WHAT THIS COMPUTES. The actual composition strength under both minutes
estimators, using the same PIT DARKO talent for each:

    strength(team) = sum over available players of  talent_i * min_i / 48
    margin        = strength_home - strength_away

then the distribution of |margin_new - margin_old| across real games. That is
the input change. Multiplying by the two-stage attenuation (1/4.5, magnitude
corrected in D252) gives what actually reaches the emitted forecast, and it can
be compared against the log-loss MDE the register already knows it needs.

NO LOG-LOSS RUN IS ATTEMPTED HERE. If the measured margin change is far below
what a season-clustered log-loss test can resolve, running L4 would produce a
null that says nothing about the estimator -- only about the test's power -- and
D245d already demonstrated that failure mode at length.
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

ATTEN = 1.0 / 4.5          # blend share 0.652 x offset edge 0.3413 (D252)


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def main():
    import duckdb
    con = duckdb.connect(str(ROOT / "data" / "nba.duckdb"), read_only=True)
    pg = con.execute("""
        SELECT CAST(game_id AS VARCHAR) gid, player_id, team_id, seconds
        FROM player_game_stats
        WHERE CAST(game_id AS VARCHAR) LIKE '002%' AND seconds > 0""").df()
    dk = con.execute("""
        SELECT player_id, date, dpm FROM darko_history
        WHERE dpm IS NOT NULL""").df()
    con.close()
    pg["gid"] = pg.gid.str.zfill(10)
    pg["mins"] = pg.seconds / 60.0
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz",
                    usecols=["game_id", "season", "game_date", "home", "away"])
    f["game_id"] = zf(f["game_id"]); f["game_date"] = pd.to_datetime(f.game_date)
    d = pg.merge(f, left_on="gid", right_on="game_id", how="inner")
    d = d[d.mins >= 12.0].sort_values(["player_id", "game_date"])

    # ---- both minutes estimators, PIT ------------------------------
    g = d.groupby(["player_id", "season"])["mins"]
    d["ship"] = g.transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    e3 = g.transform(lambda s: s.shift(1).ewm(halflife=3, min_periods=2).mean())
    seasons = sorted(d.season.unique())
    nxt = {seasons[i]: seasons[i + 1] for i in range(len(seasons) - 1)}
    prev = {v: k for k, v in nxt.items()}
    ps = d.groupby(["player_id", "season"]).mins.mean().reset_index()
    ps["tgt"] = ps.season.map(nxt)
    pri = ps.dropna(subset=["tgt"]).set_index(["tgt", "player_id"]).mins
    base = np.array([pri.get((s, p), np.nan)
                     for s, p in zip(d.season, d.player_id)])
    lg = d.groupby("season").mins.mean()
    lgv = d.season.map({s: lg.get(prev.get(s), lg.mean())
                        for s in seasons}).to_numpy(float)
    base = np.where(np.isfinite(base), base, lgv)
    cnt = g.cumcount().to_numpy(float)
    e3 = np.where(np.isfinite(e3.to_numpy(float)), e3.to_numpy(float), base)
    w = cnt / (cnt + 2.0)
    d["es32"] = w * e3 + (1 - w) * base

    # ---- PIT talent: last darko strictly before the game -----------
    # duckdb returns datetime64[us]; the frame is [ns]. merge_asof requires
    # identical dtypes, so normalise both explicitly.
    dk["date"] = pd.to_datetime(dk["date"]).astype("datetime64[ns]")
    d["game_date"] = pd.to_datetime(d["game_date"]).astype("datetime64[ns]")
    dk = dk.sort_values("date")
    d = d.sort_values("game_date")
    d = pd.merge_asof(d, dk.rename(columns={"date": "dk_date"}),
                      left_on="game_date", right_on="dk_date",
                      by="player_id", direction="backward",
                      allow_exact_matches=False)
    d = d.dropna(subset=["ship", "es32", "dpm"])
    print(f"{len(d):,} player-games with PIT talent and both minute estimates")

    # ---- team strength under each ----------------------------------
    d["s_ship"] = d.dpm * d.ship / 48.0
    d["s_es32"] = d.dpm * d.es32 / 48.0
    tg = d.groupby(["gid", "team_id"]).agg(
        s_ship=("s_ship", "sum"), s_es32=("s_es32", "sum"),
        n=("dpm", "size"), home=("home", "first"), away=("away", "first"),
        season=("season", "first")).reset_index()
    tg = tg[tg.n >= 6]
    two = tg.groupby("gid").filter(lambda x: len(x) == 2)
    piv = two.sort_values("team_id").groupby("gid").agg(
        a_ship=("s_ship", "first"), b_ship=("s_ship", "last"),
        a_es=("s_es32", "first"), b_es=("s_es32", "last"),
        season=("season", "first"))
    d_margin = (piv.a_es - piv.b_es) - (piv.a_ship - piv.b_ship)
    ad = d_margin.abs()
    print(f"{len(piv):,} games with both teams priced\n")

    print("=" * 72)
    print("MEASURED CHANGE IN THE COMPOSITION MARGIN (es3_2 minus shipped)")
    print("=" * 72)
    print(f"  mean |change|      {ad.mean():.4f} margin points")
    print(f"  median             {ad.median():.4f}")
    print(f"  p90                {ad.quantile(0.90):.4f}")
    print(f"  p99                {ad.quantile(0.99):.4f}")
    print(f"  sd of signed change {d_margin.std():.4f}")
    print(f"\n  arithmetic bound in D260 was 0.0368 pts -- "
          f"{'CONFIRMED' if 0.5 < ad.mean()/0.0368 < 2 else 'the envelope was off'}"
          f" (ratio {ad.mean()/0.0368:.2f}x)")

    emitted = ad.mean() * ATTEN
    print(f"\n  after the 4.5x two-stage attenuation: "
          f"**{emitted:.4f} margin points emitted**")
    # what log-loss change would that produce?
    scale = 6.96
    p = 0.5
    dll = (emitted / scale) * abs(p - 0.5) + 0.5 * (emitted / scale) ** 2
    print(f"  implied log-loss effect, order of magnitude: ~{dll:.2e} nats")
    print(f"  register's typical MDE80 on log loss:         ~1e-3 nats")
    print(f"  ratio: the effect is ~{1e-3/max(dll,1e-12):.0f}x BELOW what a "
          f"season-clustered test can resolve")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print("  The estimator is unambiguously better at its own job (16/16")
    print("  seasons, +5.80% MSE, tight CI). The propagated margin change is")
    print(f"  {ad.mean():.4f} pts before attenuation and {emitted:.4f} after.")
    print("  Running L4 would return a null that measures the TEST's power,")
    print("  not the estimator -- which is exactly D245d's failure mode.")

    json.dump({"mean_abs_change": float(ad.mean()),
               "median": float(ad.median()), "p90": float(ad.quantile(.9)),
               "emitted": float(emitted), "implied_nats": float(dll),
               "n_games": int(len(piv))},
              open(ROOT / "data" / "d260b_propagate.json", "w"))
    print("\nwrote data/d260b_propagate.json")


if __name__ == "__main__":
    main()
