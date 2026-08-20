#!/usr/bin/env python3
"""NEWSTRAT FEATURES — the two derived selectors that are not already on the frame.

Builds, PIT-safe, one row per game of `data/ats19_frame.csv.gz`:

  pred_dm   A1's selector core.  Walk-forward prediction of the OPEN->CLOSE
            spread movement (`close_margin - open_margin`), rebuilt on the full
            19-season frame from the D147 primitives in nbapred/market/anchored.py
            (D147's own artifact `data/cm_clvmodel_rows.csv.gz` carries pred_dm on
            only 4 seasons, 2022-23..2025-26, and the primary frame here needs 8).
            TIER A features only -- every one knowable when the opener is posted.
            Ridge refit every REFIT games on ALL strictly-prior games; the first
            REFIT games get pred_dm = 0.0, which IS the naive no-movement baseline
            D147 measures R^2 against, i.e. the honest cold start.

  retmin_h  A3's selector core, per side.  Minutes-weighted rotation minutes
  retmin_a  RETURNING to the lineup: players who (a) appeared for the team in its
            trailing 20 games this season with mpg >= ROT_MPG, (b) did NOT appear
            in the team's most recent game, and (c) are NOT on tonight's official
            inactive list.  Sum of their prior mpg.
            AVAILABILITY TIER: (c) reads `game_inactives`, i.e. the T-30 official
            filing -- the SAME tier the frame's own p_us consumes (T2).  It is
            NOT knowable at the open in the strict D147 Tier-A sense and is
            labelled T2/Tier-B wherever it is reported.  (a) and (b) are pure
            box-score history and are Tier A.

READ-ONLY on data/nba.duckdb (read_only=True, retry 60s).  No default changed.
Writes data/ns_features.csv.gz only.

  python3 scripts/ns_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import duckdb                                                     # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

from nbapred.market.anchored import (                             # noqa: E402
    run_price_ridge, ridge_fit, standardise, assert_pit,
)

DB = str(ROOT / "data" / "nba.duckdb")
FRAME = ROOT / "data" / "ats19_frame.csv.gz"
OUT = ROOT / "data" / "ns_features.csv.gz"

REFIT = 500            # refit cadence, games
LAM = 1.0              # ridge penalty on the movement model
ROT_MPG = 10.0         # rotation floor for A3
WIN_G = 20             # trailing team-games defining the rotation set
MIN_G = 5              # below this many prior team-games, retmin := 0


def gid10(v):
    """The frame stores game_id as an unpadded int (20700001); every DB table
    stores the canonical 10-char zero-padded string (0020700001).  Normalise."""
    return pd.Series(v).astype(str).str.zfill(10)


# ------------------------------------------------------------------ A1 ------
def build_movement_features(df):
    """TIER A design matrix for the open->close movement model.  PIT by
    construction: the three price ridges absorb a date only after every game on
    that date has been predicted (anchored.run_price_ridge)."""
    home = df["home"].to_numpy()
    away = df["away"].to_numpy()
    day = df["_day"].to_numpy()
    om = df["open_margin"].to_numpy(float)

    r_close = run_price_ridge(home, away, day, df["close_margin"].to_numpy(float))
    r_open = run_price_ridge(home, away, day, om)
    r_res = run_price_ridge(home, away, day, df["margin_actual"].to_numpy(float))

    n = len(df)
    X = np.column_stack([
        np.ones(n),                                   # intercept
        r_close - om,                                 # resid_close
        r_open - om,                                  # resid_open
        r_res - om,                                   # resid_res
        df["m_us"].to_numpy(float) - om,              # resid_us  (our edge)
        om,                                           # open_margin
        np.abs(om),                                   # abs_open
        np.abs(df["p_us"].to_numpy(float) - 0.5),     # conf_us
        (np.nan_to_num(df["open_total"].to_numpy(float), nan=215.0)
         - 215.0) / 10.0,                             # 535 games lack a total
    ])
    names = ["1", "resid_close", "resid_open", "resid_res", "resid_us",
             "open_margin", "abs_open", "conf_us", "open_total"]
    return X, names


def walk_forward_pred(X, y, refit=REFIT, lam=LAM):
    """pred[i] uses ONLY rows strictly before the refit boundary at or below i.
    Rows before the first boundary get 0.0 = the naive no-movement baseline.
    Rows whose label is missing are dropped from the FIT, never from prediction."""
    n = len(y)
    pred = np.zeros(n)
    beta = None
    mu = sd = None
    for start in range(0, n, refit):
        stop = min(start + refit, n)
        if beta is not None:
            pred[start:stop] = ((X[start:stop] - mu) / sd) @ beta
        if stop >= refit:
            good = np.isfinite(y[:stop]) & np.isfinite(X[:stop]).all(axis=1)
            Xtr, ytr = X[:stop][good], y[:stop][good]
            mu = Xtr.mean(axis=0)
            sd = Xtr.std(axis=0)
            mu[0], sd[0] = 0.0, 1.0
            sd = np.where(sd < 1e-12, 1.0, sd)
            beta = ridge_fit((Xtr - mu) / sd, ytr, lam=lam)
    return pred


# ------------------------------------------------------------------ A3 ------
def build_return_minutes(con, frame_gids):
    """Per (game_id, team_id): minutes of rotation players returning tonight."""
    box = con.execute("""
        SELECT s.game_id, s.player_id, s.team_id, s.seconds,
               g.game_date, g.season
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season, team_id
              FROM nba_games) g
          ON g.game_id = s.game_id AND g.team_id = s.team_id
        WHERE s.game_id LIKE '002%'
    """).fetchdf()
    inact = con.execute("SELECT game_id, player_id, team_id FROM game_inactives") \
              .fetchdf()
    inact_set = set(zip(inact.game_id.astype(str), inact.team_id.astype(np.int64),
                        inact.player_id.astype(np.int64)))

    box["game_id"] = box.game_id.astype(str)
    box["mins"] = box.seconds.astype(float) / 60.0
    box = box.sort_values(["season", "team_id", "game_date", "game_id"])

    rows = []
    for (season, team), g in box.groupby(["season", "team_id"], sort=False):
        # ordered distinct team-games
        gids, gdates = [], []
        for gid, gg in g.groupby("game_id", sort=False):
            gids.append(gid)
            gdates.append(gg.game_date.iloc[0])
        order = np.argsort(np.asarray(gdates), kind="stable")
        gids = [gids[i] for i in order]
        by_game = {gid: gg for gid, gg in g.groupby("game_id", sort=False)}

        hist: list[dict[int, float]] = []          # per past game: player -> mins
        for i, gid in enumerate(gids):
            if i < MIN_G:
                rows.append((gid, int(team), 0.0, 0))
            else:
                win = hist[-WIN_G:]
                tot: dict[int, float] = {}
                app: dict[int, int] = {}
                for d in win:
                    for pid, mn in d.items():
                        tot[pid] = tot.get(pid, 0.0) + mn
                        app[pid] = app.get(pid, 0) + 1
                last = hist[-1]
                rm, cnt = 0.0, 0
                for pid, tm in tot.items():
                    mpg = tm / app[pid]
                    if mpg < ROT_MPG:
                        continue
                    if pid in last and last[pid] > 0:
                        continue                    # played the most recent game
                    if (gid, int(team), int(pid)) in inact_set:
                        continue                    # out again tonight
                    rm += mpg
                    cnt += 1
                rows.append((gid, int(team), float(rm), int(cnt)))
            cur = by_game[gid]
            hist.append(dict(zip(cur.player_id.astype(np.int64).to_numpy(),
                                 cur.mins.to_numpy())))
    out = pd.DataFrame(rows, columns=["game_id", "team_id", "retmin", "retn"])
    return out[out.game_id.isin(frame_gids)].reset_index(drop=True)


# ----------------------------------------------------------------- main -----
def main():
    df = pd.read_csv(FRAME)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    df["_day"] = (df.game_date - df.game_date.min()).dt.days.to_numpy()

    # ---- A1 -----------------------------------------------------------------
    X, names = build_movement_features(df)
    y = (df.close_margin - df.open_margin).to_numpy(float)
    pred = walk_forward_pred(X, y, REFIT, LAM)
    df["pred_dm"] = pred

    ok = (np.arange(len(df)) >= REFIT) & np.isfinite(y)
    ss_res = float(((y[ok] - pred[ok]) ** 2).sum())
    ss_tot = float((y[ok] ** 2).sum())              # vs the NAIVE no-movement
    r2 = 1.0 - ss_res / ss_tot
    print(f"A1 movement model: n_pred={int(ok.sum())}  "
          f"R2_vs_naive={r2:+.5f}  sd(dm)={np.nanstd(y):.4f}  "
          f"corr={np.corrcoef(pred[ok], y[ok])[0,1]:+.4f}")
    for s_, gi in df.assign(_ok=ok, _y=y, _p=pred).groupby("season"):
        g = gi[gi._ok]
        if len(g) < 50:
            continue
        rr = 1.0 - ((g._y - g._p) ** 2).sum() / (g._y ** 2).sum()
        print(f"    {s_}  n={len(g):5d}  R2={rr:+.5f}  sd(dm)={g._y.std():.3f}")

    # PIT guard, D147's two-sided shuffle test
    viol, moved = assert_pit(build_movement_features, df,
                             label_cols=["close_margin", "margin_actual"],
                             day_col="_day")
    print(f"A1 PIT guard: violations={viol}   moved_under_full_shuffle={moved}")
    if viol:
        raise SystemExit(f"PIT VIOLATION in movement features: {viol}")
    if not moved:
        raise SystemExit("PIT guard is vacuous: no feature moved under shuffle")

    # ---- A3 -----------------------------------------------------------------
    con = duckdb.connect(DB, read_only=True, config={"lock_timeout": "60s"}) \
        if False else duckdb.connect(DB, read_only=True)
    try:
        df["_gid10"] = gid10(df.game_id.values)
        gids = set(df["_gid10"])
        rm = build_return_minutes(con, gids)
        teams = con.execute("""
            SELECT DISTINCT game_id, team_id, team_abbrev, is_home
            FROM nba_games WHERE game_id LIKE '002%'
        """).fetchdf()
    finally:
        con.close()
    teams["game_id"] = teams.game_id.astype(str)
    teams = teams.drop_duplicates(subset=["game_id", "team_id"])
    rm = rm.drop_duplicates(subset=["game_id", "team_id"])
    rm = rm.merge(teams, on=["game_id", "team_id"], how="left")
    h = rm[rm.is_home == 1][["game_id", "retmin", "retn"]] \
        .rename(columns={"retmin": "retmin_h", "retn": "retn_h"})
    a = rm[rm.is_home == 0][["game_id", "retmin", "retn"]] \
        .rename(columns={"retmin": "retmin_a", "retn": "retn_a"})
    h = h.rename(columns={"game_id": "_gid10"}).drop_duplicates(subset=["_gid10"])
    a = a.rename(columns={"game_id": "_gid10"}).drop_duplicates(subset=["_gid10"])
    df = df.merge(h, on="_gid10", how="left").merge(a, on="_gid10", how="left")
    for c in ["retmin_h", "retmin_a", "retn_h", "retn_a"]:
        df[c] = df[c].fillna(0.0)
    print(f"A3 return-minutes: mean_h={df.retmin_h.mean():.3f} "
          f"mean_a={df.retmin_a.mean():.3f}  "
          f"frac_nonzero={(df.retmin_h + df.retmin_a > 0).mean():.4f}  "
          f"mean_n_ret={(df.retn_h + df.retn_a).mean():.3f}")

    keep = ["game_id", "season", "game_date", "pred_dm",
            "retmin_h", "retmin_a", "retn_h", "retn_a"]
    df[keep].to_csv(OUT, index=False)
    print(f"wrote {OUT}  rows={len(df)}")


if __name__ == "__main__":
    main()
