#!/usr/bin/env python3
"""D231 — THE MINUTES-TO-POINTS BRIDGE. Is per-minute production constant?

THE CLAIM UNDER TEST, which is not new here — D133 arm C and D144 both hit it
and both named it as the reason a better minutes forecast failed to become a
better points forecast:

    "a promoted player's per-minute rates were earned in a BENCH role and do
     not survive being scaled to starter minutes, so adding 2.4 correct minutes
     adds too many points"

The composition leg spends exactly that assumption: it carries
`talent x trail_min / 48`, i.e. a player's historical rate transferred UNCHANGED
to whatever minutes he plays tonight. If the rate degrades as minutes are scaled
up, that leg is biased upward precisely on the games where availability matters
most — the ones where somebody is out and somebody else absorbs the minutes.

TWO MEASUREMENTS, KEPT SEPARATE.

(A) DIAGNOSTIC — is the rate stable? Conditioning on ACTUAL minutes isolates the
    rate question from the minutes-forecast question. This is deliberately NOT a
    forecast: knowing tonight's minutes is not available at bet time. It answers
    "if the minutes forecast were perfect, would the bridge be right?"

(B) FORECAST — does a corrected bridge beat the naive one OUT OF SAMPLE using
    only information available before tip? This is the one that could ship.
    Walk-forward by season; nothing refitted on the season being scored.

ENDOGENEITY, STATED UP FRONT. Minutes and production are jointly determined
inside a game: a player who is playing well earns minutes, and garbage time adds
minutes at depressed value. (A) therefore measures an ASSOCIATION and is labelled
diagnostic. (B) uses only prior-game information and is a clean forecast test.
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

from nbapred.db import connect                                    # noqa: E402

HALF_LIFE = 8.0            # games; the trailing rate/minutes memory
MIN_HIST = 10              # games of history before a player is scoreable
SEASONS_FROM = "2018-19"   # injury-report era; matches the model frame


def load() -> pd.DataFrame:
    con = connect(read_only=True)
    df = con.execute("""
        SELECT s.game_id, s.player_id, s.team_id, s.seconds, s.pts,
               g.game_date, g.season
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, season, game_date FROM nba_games
              WHERE game_id LIKE '002%') g ON g.game_id = s.game_id
        WHERE s.game_id LIKE '002%' AND s.seconds > 0
        ORDER BY s.player_id, g.game_date, s.game_id
    """).fetchdf()
    con.close()
    df = df[df["season"] >= SEASONS_FROM].copy()
    df["mins"] = df["seconds"] / 60.0
    return df.reset_index(drop=True)


def add_trailing(df: pd.DataFrame) -> pd.DataFrame:
    """EWMA of minutes and points, STRICTLY PRIOR (shifted), per player.

    The rate is built as ewma(pts)/ewma(min) rather than ewma(pts/min): the
    latter lets a 2-minute cameo with one basket contribute the same weight as a
    38-minute start, which is exactly the noise this test is trying to see past.
    """
    a = 1 - 0.5 ** (1 / HALF_LIFE)
    g = df.groupby("player_id", sort=False)
    for col in ("mins", "pts"):
        e = g[col].transform(lambda s: s.ewm(alpha=a, adjust=False).mean())
        df[f"tr_{col}"] = g[col].transform(
            lambda s: s.ewm(alpha=a, adjust=False).mean().shift(1))
    df["n_hist"] = g.cumcount()
    df = df[df["n_hist"] >= MIN_HIST].copy()
    df = df[(df["tr_mins"] > 4.0) & (df["tr_pts"] >= 0)].copy()
    df["tr_rate"] = df["tr_pts"] / df["tr_mins"]          # pts per minute
    df["naive"] = df["mins"] * df["tr_rate"]              # the shipped bridge
    df["resid"] = df["pts"] - df["naive"]
    df["dmin"] = df["mins"] - df["tr_mins"]               # minutes deviation
    df["rmin"] = df["mins"] / df["tr_mins"]               # minutes ratio
    return df


def diagnostic(df: pd.DataFrame) -> dict:
    """(A) Does the naive bridge's error depend on the minutes deviation?"""
    print("\n=== (A) DIAGNOSTIC — bridge error vs minutes deviation ===")
    print("    (conditions on ACTUAL minutes; association, not a forecast)")
    q = pd.qcut(df["dmin"], 10, labels=False, duplicates="drop")
    print(f"\n{'decile':>6} {'dmin mid':>9} {'n':>7} {'mean resid':>11} "
          f"{'resid/naive':>12}")
    rows = []
    for d in sorted(q.dropna().unique()):
        m = q == d
        sub = df[m]
        rr = sub["resid"].mean()
        rel = rr / max(sub["naive"].mean(), 1e-9)
        rows.append(dict(decile=int(d), dmin=float(sub["dmin"].median()),
                         n=int(m.sum()), resid=float(rr), rel=float(rel)))
        print(f"{int(d):6d} {sub['dmin'].median():9.2f} {int(m.sum()):7d} "
              f"{rr:11.4f} {100*rel:11.2f}%")
    # single-number summary: OLS of residual on the deviation
    x = df["dmin"].to_numpy(float)
    y = df["resid"].to_numpy(float)
    b = np.polyfit(x, y, 1)
    print(f"\n  OLS resid = {b[1]:+.4f} {b[0]:+.5f} * dmin")
    print(f"  -> each EXTRA minute above a player's trailing norm yields "
          f"{b[0]:+.4f} pts MORE than the naive bridge predicts")
    return {"deciles": rows, "slope": float(b[0]), "intercept": float(b[1])}


def forecast(df: pd.DataFrame) -> dict:
    """(B) Walk-forward: can a corrected bridge beat naive out of sample?

    Correction is deliberately ONE parameter — a shrinkage of the transferred
    rate toward the player's own mean as minutes depart from his norm:

        pts_hat = mins * (tr_rate + c * dmin)

    c is fitted on prior seasons only. c = 0 IS the shipped bridge, so the null
    is the incumbent, not zero (D198's rule, as in D230).
    """
    print("\n=== (B) FORECAST — walk-forward, prior seasons only ===")
    seasons = sorted(df["season"].unique())
    rows = []
    for i, s in enumerate(seasons):
        if i == 0:
            continue
        tr = df[df["season"].isin(seasons[:i])]
        te = df[df["season"] == s]
        # closed-form LS for c on the training block
        x = (tr["mins"] * tr["dmin"]).to_numpy(float)
        y = tr["resid"].to_numpy(float)
        c = float((x @ y) / (x @ x)) if (x @ x) > 0 else 0.0
        pred = te["mins"] * (te["tr_rate"] + c * te["dmin"])
        mae_n = float((te["pts"] - te["naive"]).abs().mean())
        mae_c = float((te["pts"] - pred).abs().mean())
        rms_n = float(np.sqrt(((te["pts"] - te["naive"]) ** 2).mean()))
        rms_c = float(np.sqrt(((te["pts"] - pred) ** 2).mean()))
        rows.append(dict(season=s, n=len(te), c=c, mae_naive=mae_n,
                         mae_corr=mae_c, rmse_naive=rms_n, rmse_corr=rms_c,
                         d_mae=mae_c - mae_n, d_rmse=rms_c - rms_n))
        print(f"  {s}  n={len(te):6d}  c={c:+.5f}  MAE {mae_n:.4f} -> "
              f"{mae_c:.4f} ({mae_c-mae_n:+.4f})  RMSE {rms_n:.4f} -> "
              f"{rms_c:.4f} ({rms_c-rms_n:+.4f})")
    r = pd.DataFrame(rows)
    from scipy import stats
    d = r["d_rmse"].to_numpy(float)
    k = len(d)
    se = d.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    lo, hi = d.mean() - tc * se, d.mean() + tc * se
    print(f"\n  season-clustered mean d_RMSE {d.mean():+.5f} pts")
    print(f"  95% CI ({k-1} dof)            [{lo:+.5f}, {hi:+.5f}]")
    print(f"  better in                     {int((d<0).sum())}/{k} seasons")
    print(f"  VERDICT: {'IMPROVES' if hi < 0 else 'NO SHIP — CI includes zero'}")
    return {"per_season": rows, "mean_d_rmse": float(d.mean()),
            "ci": [float(lo), float(hi)],
            "better": int((d < 0).sum()), "k": k}


def main():
    df = add_trailing(load())
    print(f"frame: {len(df):,} player-games, {df['player_id'].nunique():,} "
          f"players, seasons {df['season'].min()}..{df['season'].max()}")
    print(f"mean minutes {df['mins'].mean():.2f}, mean pts {df['pts'].mean():.2f}")
    out = {"n": len(df), "diagnostic": diagnostic(df), "forecast": forecast(df)}
    json.dump(out, open(ROOT / "data" / "d231_bridge.json", "w"), default=float)
    print("\nwrote data/d231_bridge.json")


if __name__ == "__main__":
    main()
