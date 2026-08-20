#!/usr/bin/env python3
"""Measure open->close line movement on `odds_open`.

The question this exists to answer: our claimed edge requires betting at a
price BETTER than the close. Two things have to be true for that to be worth
anything:
  (1) the line actually MOVES between open and close (magnitude), and
  (2) the move is INFORMATIVE - the close is a better predictor than the open
      (otherwise an early price is not a better price, just a different one).

If (1) is small or (2) is false, "we bet at the open" buys nothing and the
whole better-than-close premise collapses.

Usage: python scripts/measure_line_movement.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nbapred.config import DATA  # noqa: E402
from nbapred.db import connect  # noqa: E402

PAD = 62
CSV = DATA / "derived" / "odds_open.csv"


def load_odds_open() -> pd.DataFrame:
    """Prefer the CSV snapshot - it is written by the same build step and
    stays readable while a loader holds the DuckDB write lock."""
    if CSV.exists():
        return pd.read_csv(CSV, parse_dates=["game_date"])
    con = connect(read_only=True)
    df = con.execute("SELECT * FROM odds_open").df()
    con.close()
    return df


def hdr(s: str) -> None:
    print("\n" + "=" * PAD + f"\n{s}\n" + "=" * PAD)


def boot_ci(x: np.ndarray, stat=np.mean, n: int = 4000, seed: int = 7):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    d = stat(x[idx], axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    df = load_odds_open()
    df = df[df.open_margin.notna() & df.close_margin.notna()].sort_values("game_date")

    df["actual"] = df.score_home - df.score_away
    df["move"] = df.close_margin - df.open_margin       # + = toward home
    df["err_open"] = df.actual - df.open_margin
    df["err_close"] = df.actual - df.close_margin

    # -------------------------------------------------- coverage
    hdr("COVERAGE BY SEASON (odds_open)")
    cov = df.groupby("season").agg(
        games=("actual", "size"),
        first=("game_date", "min"), last=("game_date", "max"),
        mean_abs_move=("move", lambda s: s.abs().mean()),
        pct_no_move=("move", lambda s: (s.abs() < 1e-9).mean() * 100),
    ).reset_index()
    cov["mean_abs_move"] = cov.mean_abs_move.round(3)
    cov["pct_no_move"] = cov.pct_no_move.round(1)
    print(cov.to_string(index=False))
    print(f"\nTOTAL games with BOTH open and close: {len(df)}")

    # -------------------------------------------------- magnitude
    hdr("(1) HOW MUCH DOES THE LINE MOVE?  |close - open|, points of spread")
    m = df.move.abs().values
    qs = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    print("  mean   %.3f   (95%% CI %.3f, %.3f)" % ((m.mean(),) + boot_ci(m)))
    print("  sd     %.3f" % m.std(ddof=1))
    print("  quantiles: " + "  ".join(
        f"p{q}={np.percentile(m, q):.1f}" for q in qs))
    for t in (0, 0.5, 1, 1.5, 2, 3, 4, 5):
        print(f"  P(|move| >  {t:>3}) = {(m > t).mean() * 100:5.1f}%"
              if t else f"  P(|move| == 0) = {(m == 0).mean() * 100:5.1f}%")
    print("\n  signed move (+ = toward home): mean %.4f  (95%% CI %.4f, %.4f)"
          % ((df.move.mean(),) + boot_ci(df.move.values)))
    tm = df.dropna(subset=["open_total", "close_total"])
    tmv = (tm.close_total - tm.open_total).abs().values
    print("  totals |move|: mean %.3f  median %.1f  n=%d"
          % (tmv.mean(), np.median(tmv), len(tmv)))

    # -------------------------------------------------- informativeness
    hdr("(2) IS THE MOVE INFORMATIVE?  open vs close as margin predictors")
    print("  MAE  open  %.4f   close  %.4f   delta %+.4f"
          % (df.err_open.abs().mean(), df.err_close.abs().mean(),
             df.err_close.abs().mean() - df.err_open.abs().mean()))
    print("  RMSE open  %.4f   close  %.4f   delta %+.4f"
          % (np.sqrt((df.err_open ** 2).mean()), np.sqrt((df.err_close ** 2).mean()),
             np.sqrt((df.err_close ** 2).mean()) - np.sqrt((df.err_open ** 2).mean())))
    d = (df.err_close.abs() - df.err_open.abs()).values
    lo, hi = boot_ci(d)
    print("  paired MAE(close)-MAE(open) = %+.4f  95%% CI (%+.4f, %+.4f)  %s"
          % (d.mean(), lo, hi, "close BETTER" if hi < 0 else
             ("close WORSE" if lo > 0 else "NOT SIGNIFICANT")))

    # Does the move point the right way?
    mv = df[df.move.abs() > 1e-9].copy()
    # residual at open: how far the eventual result was from the opening line
    mv["resid_open"] = mv.actual - mv.open_margin
    toward = np.sign(mv.move) == np.sign(mv.resid_open)
    tie = mv.resid_open.abs() < 1e-9
    t = toward[~tie]
    lo, hi = boot_ci(t.values.astype(float))
    print("\n  Of %d games where the line MOVED (excl. %d exact-push):" % (len(mv), tie.sum()))
    print("    moved TOWARD the eventual result: %.2f%%  95%% CI (%.2f%%, %.2f%%)"
          % (t.mean() * 100, lo * 100, hi * 100))
    print("    (50%% = movement is pure noise; >50%% = the move carries信息)"
          .replace("信息", " information"))

    # ATS: bet the OPEN line vs bet the CLOSE line, home side, graded on result
    hdr("(3) WHAT DOES THE OPEN PRICE BUY?  ATS at open vs at close")
    for lbl, col in (("open", "open_margin"), ("close", "close_margin")):
        r = df.actual - df[col]
        w, l, p = (r > 0).sum(), (r < 0).sum(), (r == 0).sum()
        print("  HOME side @ %-5s  W-L-P %4d-%4d-%3d   win%% %.2f%%"
              % (lbl, w, l, p, w / (w + l) * 100))
    # The move-favoured side: taking the open price on the side the market
    # later moved TOWARD means you beat the close by |move| points.
    mv2 = df[df.move.abs() > 1e-9].copy()
    side = np.sign(mv2.move)                      # +1 = market moved to home
    r_open = (mv2.actual - mv2.open_margin) * side
    r_close = (mv2.actual - mv2.close_margin) * side
    for lbl, r in (("open (beats close by |move|)", r_open),
                   ("close (no timing edge)", r_close)):
        w, l, p = (r > 0).sum(), (r < 0).sum(), (r == 0).sum()
        wr = w / (w + l)
        lo, hi = boot_ci((r[r != 0] > 0).values.astype(float))
        print("  STEAM side @ %-30s W-L-P %4d-%4d-%3d  win%% %.2f%% CI(%.2f,%.2f)"
              % (lbl, w, l, p, wr * 100, lo * 100, hi * 100))
    print("  (breakeven at -110 juice = 52.38%%)")

    # -------------------------------------------------- by season
    hdr("(4) STABILITY BY SEASON")
    rows = []
    for s, g in df.groupby("season"):
        gm = g[g.move.abs() > 1e-9]
        res = gm.actual - gm.open_margin
        tw = (np.sign(gm.move) == np.sign(res))[res.abs() > 1e-9]
        rows.append({
            "season": s, "n": len(g),
            "mean|move|": round(g.move.abs().mean(), 3),
            "MAE_open": round(g.err_open.abs().mean(), 3),
            "MAE_close": round(g.err_close.abs().mean(), 3),
            "close-open": round(g.err_close.abs().mean() - g.err_open.abs().mean(), 3),
            "move_toward%": round(tw.mean() * 100, 1),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # -------------------------------------------------- model-corpus overlap
    hdr("(5) OVERLAP WITH MODEL CORPUS (season_end 2022..2026)")
    ov = df[df.season_end >= 2022]
    con2 = connect(read_only=True)
    tot = con2.execute("""SELECT season_end, count(*) n FROM odds_market
                          WHERE season_end >= 2022 GROUP BY 1 ORDER BY 1""").df()
    con2.close()
    have = ov.groupby("season_end").size().rename("with_open").reset_index()
    j = tot.merge(have, on="season_end", how="left").fillna({"with_open": 0})
    j["with_open"] = j.with_open.astype(int)
    j["pct"] = (j.with_open / j.n * 100).round(1)
    print(j.to_string(index=False))
    print("\n  Games in the model corpus that have an opening line: "
          f"{j.with_open.sum()} / {j.n.sum()} ({j.with_open.sum()/j.n.sum()*100:.1f}%)")


if __name__ == "__main__":
    main()
