"""DARKO player-talent priors (free, daily-updated, non-lagged).

DARKO (github.com/andrewpatton/darko-public) is a public Bayesian player-skill
estimate — Daily Plus-Minus (DPM): overall / offensive / defensive, split into
box-score-only and on-off (RAPM-style) components, plus an age field. It is
keyed by NBA player_id, so it joins to nba_api with NO name matching.

Why this matters (Sean's Q, 2026-07-26): 2K ratings lag ~a full season. DARKO
updates every game day, so it is a current, stats-derived alternative prior
center for the skill model — and its on-off DPM is philosophically the same
object as the handoff's stint-margin (RAPM) likelihood target. See
docs/PRIORS.md for the design decision on how 2K vs DARKO vs raw stats combine.
"""
from __future__ import annotations

import datetime as dt
import io
import logging

import pandas as pd
import requests

from ..config import RAW

log = logging.getLogger("darko")

RAW_DARKO = RAW / "darko"
RAW_DARKO.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0"}
SHEET_ID = "1mhwOLqPu2F9026EQiVxFPIN1t9RGafGpl-dokaIsm9c"
CSV_URL = (f"https://docs.google.com/spreadsheets/d/{SHEET_ID}"
           "/gviz/tq?tqx=out:csv&sheet=Box%20DPM")

SCHEMA = """
CREATE TABLE IF NOT EXISTS darko_dpm (
    snapshot_date DATE NOT NULL,      -- version key (daily snapshots accumulate)
    nba_player_id BIGINT NOT NULL,
    player_name   VARCHAR,
    position      VARCHAR,
    age           DOUBLE,
    dpm           DOUBLE,
    o_dpm         DOUBLE,
    d_dpm         DOUBLE,
    box_o_dpm     DOUBLE,
    box_d_dpm     DOUBLE,
    onoff_o_dpm   DOUBLE,
    onoff_d_dpm   DOUBLE,
    ingest_ts     TIMESTAMPTZ NOT NULL
);
"""

_COLS = {
    "NBA ID": "nba_player_id", "Player Name": "player_name", "Position": "position",
    "Age": "age", "DPM": "dpm", "Offensive DPM": "o_dpm", "Defensive DPM": "d_dpm",
    "Box Only O-DPM": "box_o_dpm", "Box Only D-DPM": "box_d_dpm",
    "On Off O-DPM": "onoff_o_dpm", "On Off D-DPM": "onoff_d_dpm",
}


REQUIRED_COLS = ("nba_player_id", "dpm", "o_dpm", "d_dpm", "age")


def fetch() -> pd.DataFrame:
    # review6 #9: no raise_for_status / column validation meant an HTML error
    # page (rate limit, sheet moved) could be archived + ingested as garbage.
    r = requests.get(CSV_URL, headers=UA, timeout=30)
    r.raise_for_status()
    txt = r.text
    df = pd.read_csv(io.StringIO(txt)).rename(columns=_COLS)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"darko sheet schema changed (missing {missing}); "
                         "refusing to archive/ingest")
    snapshot = dt.date.today().isoformat()
    (RAW_DARKO / f"box_dpm_{snapshot}.csv").write_text(txt)   # archive only validated payloads
    df = df.dropna(subset=["nba_player_id"])
    df["nba_player_id"] = df["nba_player_id"].astype(int)
    return df


def load(connect_fn) -> int:
    df = fetch()  # network first, then short write (single-writer discipline)
    snapshot = dt.date.today().isoformat()
    now = dt.datetime.now(dt.timezone.utc)
    con = connect_fn()
    con.execute(SCHEMA)
    con.execute("DELETE FROM darko_dpm WHERE snapshot_date = ?", [snapshot])
    for _, r in df.iterrows():
        con.execute("INSERT INTO darko_dpm VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", [
            snapshot, int(r["nba_player_id"]), r.get("player_name"), r.get("position"),
            r.get("age"), r.get("dpm"), r.get("o_dpm"), r.get("d_dpm"),
            r.get("box_o_dpm"), r.get("box_d_dpm"), r.get("onoff_o_dpm"),
            r.get("onoff_d_dpm"), now])
    con.close()
    log.info("darko %s: %d players", snapshot, len(df))
    return len(df)
