"""Load the Kaggle 'nba betting data' dataset (spreads/totals/ML, 2008-2026)
into odds_market. Moneylines only run through ~2022-23, but SPREADS+TOTALS
cover every season through 2025-26 — the free market benchmark for recent
seasons (moneyline gap otherwise). Spread -> win prob via a fixed logistic scale
calibrated on 2008-2023 (recovers the ML-devig 0.589, validating the conversion).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

TEAM_MAP = {"gs": "GSW", "no": "NOP", "ny": "NYK", "sa": "SAS", "phx": "PHX",
            "utah": "UTA", "wsh": "WAS"}
SPREAD_SCALE = 6.96  # calibrated on 2008-2023: sigmoid(home_exp_margin/scale)

SCHEMA = """
CREATE TABLE IF NOT EXISTS odds_market (
    season_end  INTEGER,           -- 2026 = 2025-26 season
    game_date   DATE,
    home        VARCHAR, away VARCHAR,   -- our abbreviations
    score_home  INTEGER, score_away INTEGER,
    home_win    BOOLEAN,
    home_exp_margin DOUBLE,         -- market expected home margin from spread
    total       DOUBLE,
    ml_home     INTEGER, ml_away INTEGER,   -- may be NULL for recent seasons
    p_home_spread DOUBLE,           -- spread-implied win prob (fixed calibration)
    ingest_ts   TIMESTAMPTZ NOT NULL
);
"""


def _abbr(code: str) -> str:
    return TEAM_MAP.get(code, code.upper())


def load(connect_fn, csv_path: str) -> int:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["spread", "whos_favored", "score_home", "score_away"])
    df = df[df.score_home != df.score_away].copy()
    df["home_exp"] = np.where(df.whos_favored == "home", df.spread, -df.spread)
    df["p_home"] = 1 / (1 + np.exp(-df.home_exp / SPREAD_SCALE))
    df["gdate"] = pd.to_datetime(df.date).dt.date          # vectorized (not per-row)
    df["home_ab"] = df.home.map(_abbr); df["away_ab"] = df.away.map(_abbr)
    now = dt.datetime.now(dt.timezone.utc)
    rows = []
    for r in df.itertuples():
        rows.append([int(r.season), r.gdate,
                     r.home_ab, r.away_ab, int(r.score_home), int(r.score_away),
                     bool(r.score_home > r.score_away), float(r.home_exp),
                     float(r.total) if pd.notna(r.total) else None,
                     int(r.moneyline_home) if pd.notna(r.moneyline_home) else None,
                     int(r.moneyline_away) if pd.notna(r.moneyline_away) else None,
                     float(r.p_home), now])
    out = pd.DataFrame(rows, columns=[
        "season_end", "game_date", "home", "away", "score_home", "score_away",
        "home_win", "home_exp_margin", "total", "ml_home", "ml_away",
        "p_home_spread", "ingest_ts"])
    con = connect_fn()
    con.execute(SCHEMA)
    con.execute("DELETE FROM odds_market")
    con.execute("INSERT INTO odds_market SELECT * FROM out")  # bulk (fast)
    con.close()
    return len(out)
