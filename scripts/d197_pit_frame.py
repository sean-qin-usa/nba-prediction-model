#!/usr/bin/env python3
"""D197 — THE CANONICAL POINT-IN-TIME FRAME (roadmap item 1).

One immutable per-game artifact that every downstream study scores on, so that
"model vs opener vs close" can never again be computed on three different sets
of games (the defect D193 found).

    data/pit_frame.csv.gz     one row per regular-season game, 2019-20+

COLUMNS AND THEIR INFORMATION TIER — the tier is the point of the file.
  T0  known before the opener is posted
        game_id, season, game_date, home, away, gidx, rest_home, rest_away,
        absence_tr_home, absence_tr_away        (TRAILING out-load only)
  T1  the opener itself
        open_margin, open_spread, open_prob
  T2  known by the 5PM official injury report (NOT available at the open)
        outs_report_home, outs_report_away
  T3  known ~30 min pre-tip (NOT available at the open)
        outs_inactive_home, outs_inactive_away
  T4  the close
        close_margin, close_spread, close_prob
  T5  the outcome — never a feature
        margin_actual, y

  MODEL columns (market-blind, fit point-in-time):
        m_us, p_us, m_ff, m_comp   (m_ff/m_comp only where the component run
        covers the game; NULL elsewhere, never imputed)

RULES THIS FILE ENFORCES
  * every study must state which tiers it uses; a study that uses T2/T3 is NOT
    describing what is available at the open
  * no imputation: a missing column is NULL, never a fill
  * no game is ever removed retroactively

Read-only with respect to the DB.  Writes one artifact.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

import oc_capacity as oc                                          # noqa: E402
from nbapred import teams as T                                    # noqa: E402
from nbapred.db import connect                                    # noqa: E402

MODERN = "2019-20"
OUT = ROOT / "data" / "pit_frame.csv.gz"


def main():
    df, _ = oc.load()
    d = df[df["season"] >= MODERN].copy().reset_index(drop=True)
    print(f"base frame: {len(d)} games, {d['season'].nunique()} seasons")

    f = pd.DataFrame({
        # oc's frame stores game_id as an INT (20700001); the DB uses the
        # zero-padded VARCHAR (0022400211). Pad, or every DB join silently
        # returns nothing — which is exactly what the T3 inactives join did.
        "game_id": d["game_id"].astype(str).str.zfill(10),
        "season": d["season"], "game_date": pd.to_datetime(d["game_date"]),
        "home": d["home"], "away": d["away"],
        # T1
        "open_margin": d["open_margin"], "open_spread": d["open_spread"],
        # T4
        "close_margin": d["close_margin"], "close_spread": d["close_spread"],
        # T5
        "margin_actual": d["margin_actual"],
        "y": (d["margin_actual"] > 0).astype(int),
        # model
        "m_us": d["m_us"], "p_us": d["p_us"],
    })

    # ---- T0 gidx
    idx, cnt = np.zeros(len(f), int), {}
    for i, (s, h, a) in enumerate(zip(f["season"], f["home"], f["away"])):
        ih, ia = cnt.get((s, h), 0), cnt.get((s, a), 0)
        idx[i] = max(ih, ia)
        cnt[(s, h)], cnt[(s, a)] = ih + 1, ia + 1
    f["gidx"] = idx

    con = connect(read_only=True)

    # ---- T0 rest
    g = pd.DataFrame(con.execute(
        "SELECT DISTINCT game_date, team_abbrev FROM nba_games "
        "WHERE game_id LIKE '002%'").fetchall(), columns=["d", "t"])
    g["d"] = pd.to_datetime(g["d"])
    g = g.sort_values("d")
    g["prev"] = g.groupby("t")["d"].shift(1)
    rest = {(r.t, r.d): min((r.d - r.prev).days, 7) if pd.notna(r.prev) else 7
            for r in g.itertuples()}
    f["rest_home"] = [rest.get((h, x), 7) for h, x in zip(f["home"], f["game_date"])]
    f["rest_away"] = [rest.get((a, x), 7) for a, x in zip(f["away"], f["game_date"])]

    # ---- T0 trailing absence load  /  T2 same-day report counts
    inj = pd.DataFrame(con.execute(
        "SELECT game_date, team, count(*) n FROM injury_reports_pit "
        "WHERE status IN ('Out','Doubtful') GROUP BY 1,2").fetchall(),
        columns=["d", "team", "n"])
    amap, unres = T.resolve_map(sorted(inj["team"].unique()))
    if unres:
        print(f"  [teams] {len(unres)} unresolvable, REPORTED: {unres}")
    inj["team"] = inj["team"].map(amap)
    inj = inj[inj["team"].notna()].copy()
    inj["d"] = pd.to_datetime(inj["d"])
    inj = inj.sort_values("d")
    same = dict(zip(zip(inj["team"], inj["d"]), inj["n"]))          # T2
    inj["tr"] = inj.groupby("team")["n"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).mean())      # T0
    tr = dict(zip(zip(inj["team"], inj["d"]), inj["tr"]))
    for side, col in (("home", "home"), ("away", "away")):
        f[f"absence_tr_{side}"] = [tr.get((t, x), np.nan)
                                   for t, x in zip(f[col], f["game_date"])]
        f[f"outs_report_{side}"] = [same.get((t, x), np.nan)
                                    for t, x in zip(f[col], f["game_date"])]

    # ---- T3 official pregame inactives
    ina = pd.DataFrame(con.execute("""
        SELECT g.game_id, i.team_id, count(*) n
        FROM game_inactives i
        JOIN (SELECT DISTINCT game_id FROM nba_games
              WHERE game_id LIKE '002%') g USING (game_id)
        GROUP BY 1,2""").fetchall(), columns=["game_id", "team_id", "n"])
    tid = pd.DataFrame(con.execute(
        "SELECT DISTINCT game_id, team_id, is_home FROM nba_games "
        "WHERE game_id LIKE '002%'").fetchall(),
        columns=["game_id", "team_id", "is_home"])
    ina = ina.merge(tid, on=["game_id", "team_id"], how="left")
    ina["game_id"] = ina["game_id"].astype(str).str.zfill(10)
    n0 = len(f)
    for side, flag in (("home", True), ("away", False)):
        # drop_duplicates: a game_id must contribute AT MOST one row per side,
        # or the merge multiplies the frame (it added 10 rows before this).
        sub = (ina[ina["is_home"] == flag][["game_id", "n"]]
               .drop_duplicates("game_id")
               .rename(columns={"n": f"outs_inactive_{side}"}))
        f = f.merge(sub, on="game_id", how="left")
        assert len(f) == n0, f"inactives merge changed row count: {len(f)} != {n0}"
    con.close()

    # ---- component legs where the component run covers the game
    cp = ROOT / "data" / "component_pergame.csv"
    if cp.exists():
        c = pd.read_csv(cp)[["game_id", "m_ff", "m_comp"]]
        c["game_id"] = c["game_id"].astype(str).str.zfill(10)   # same padding
        c = c.drop_duplicates("game_id")
        f = f.merge(c, on="game_id", how="left")
        assert len(f) == n0, f"component merge changed row count: {len(f)} != {n0}"
    else:
        f["m_ff"] = np.nan
        f["m_comp"] = np.nan

    # ---- devigged market probabilities, one shared convention for open+close
    #      (a single logistic scale fitted on the frame; identical transform for
    #      both, so no source is advantaged — D193's rule)
    from scipy.optimize import minimize_scalar

    def nll(p, y):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    yv = f["y"].to_numpy(float)
    for src in ("open", "close"):
        m = f[f"{src}_margin"].to_numpy(float)
        okm = np.isfinite(m)
        s = float(minimize_scalar(
            lambda z: nll(1 / (1 + np.exp(-m[okm] / z)), yv[okm]),
            bounds=(2, 25), method="bounded").x)
        f[f"{src}_prob"] = np.where(okm, 1 / (1 + np.exp(-m / s)), np.nan)
        print(f"  {src}_prob: shared logistic scale {s:.4f}")

    f = f.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    f.to_csv(OUT, index=False, compression="gzip")
    print(f"\nwrote {OUT}  ({len(f)} rows, {len(f.columns)} cols)")

    print("\nTIER COVERAGE (non-null share)")
    tiers = {
        "T0 gidx/rest": ["gidx", "rest_home"],
        "T0 absence_tr": ["absence_tr_home", "absence_tr_away"],
        "T1 opener": ["open_margin", "open_prob"],
        "T2 report outs": ["outs_report_home", "outs_report_away"],
        "T3 inactives": ["outs_inactive_home", "outs_inactive_away"],
        "T4 close": ["close_margin", "close_prob"],
        "T5 outcome": ["margin_actual", "y"],
        "model": ["m_us", "p_us"],
        "components": ["m_ff", "m_comp"],
    }
    for k, cols in tiers.items():
        cov = np.mean([f[c].notna().mean() for c in cols if c in f])
        print(f"  {k:18} {100*cov:6.1f}%")

    print("\nGAMES USABLE FOR AN OPEN-TIME STUDY (T0+T1+T5, no T2/T3):")
    m = (f[["gidx", "rest_home", "rest_away", "open_margin",
            "margin_actual"]].notna().all(1))
    print(f"  {int(m.sum())} of {len(f)}  ({100*m.mean():.1f}%)")


if __name__ == "__main__":
    main()
