#!/usr/bin/env python3
"""AVAILABILITY-DEPTH INGEST — the two cached, never-used sources.

  (1) NATL_TV_BROADCASTER_ABBREVIATION  <- data/raw/nba_api/boxscoresummaryv2/
  (2) GameRotation stints (USG_PCT, PT_DIFF, IN/OUT_TIME_REAL)
                                         <- data/raw/nba_api/gamerotation/

Writes:
  data/ad_natl_tv.csv        one row per game_id
  data/ad_rotation.csv.gz   one row per (game_id, team_id, player_id) STINT
  data/ad_rotation_pg.csv.gz one row per (game_id, team_id, player_id) AGGREGATE
  data/ad_ingest.json        coverage + validation report

Read-only: touches no DB (parses JSON cache only) except a read_only=True
connect for the nba_games join used in the coverage report.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from nbapred.db import connect

BSS = ROOT / "data" / "raw" / "nba_api" / "boxscoresummaryv2"
ROT = ROOT / "data" / "raw" / "nba_api" / "gamerotation"


def _rs(resp, name):
    for s in resp.get("resultSets", []):
        if s.get("name") == name:
            return s
    return None


# ------------------------------------------------------------------ natl TV
def ingest_natl_tv():
    rows = []
    bad = 0
    for f in sorted(BSS.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            bad += 1
            continue
        s = _rs(d.get("response", {}), "GameSummary")
        if s is None or not s["rowSet"]:
            bad += 1
            continue
        h = {k: i for i, k in enumerate(s["headers"])}
        r = s["rowSet"][0]
        rows.append(dict(
            game_id=str(r[h["GAME_ID"]]),
            game_date_est=str(r[h["GAME_DATE_EST"]])[:10],
            home_team_id=int(r[h["HOME_TEAM_ID"]] or 0),
            away_team_id=int(r[h["VISITOR_TEAM_ID"]] or 0),
            natl_tv=(r[h["NATL_TV_BROADCASTER_ABBREVIATION"]] or "").strip() or None,
            game_seq=r[h["GAME_SEQUENCE"]],
            ingest_ts=d.get("ingest_ts"),
        ))
    df = pd.DataFrame(rows).drop_duplicates("game_id")
    df["is_natl_tv"] = df["natl_tv"].notna().astype(int)
    return df, bad


# ---------------------------------------------------------------- rotation
def ingest_rotation():
    stints, bad = [], 0
    for f in sorted(ROT.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            bad += 1
            continue
        resp = d.get("response", {})
        got = False
        for side in ("HomeTeam", "AwayTeam"):
            s = _rs(resp, side)
            if s is None:
                continue
            h = {k: i for i, k in enumerate(s["headers"])}
            for r in s["rowSet"]:
                stints.append((
                    str(r[h["GAME_ID"]]), int(r[h["TEAM_ID"]]), int(r[h["PERSON_ID"]]),
                    1 if side == "HomeTeam" else 0,
                    float(r[h["IN_TIME_REAL"]]), float(r[h["OUT_TIME_REAL"]]),
                    float(r[h["PLAYER_PTS"]] or 0), float(r[h["PT_DIFF"]] or 0),
                    float(r[h["USG_PCT"]] or 0),
                ))
                got = True
        if not got:
            bad += 1
    df = pd.DataFrame(stints, columns=[
        "game_id", "team_id", "player_id", "is_home",
        "in_t", "out_t", "stint_pts", "pt_diff", "usg_pct"])
    # times are tenths of a second of elapsed real game clock
    df["stint_sec"] = (df["out_t"] - df["in_t"]) / 10.0
    return df, bad


def aggregate_rotation(st: pd.DataFrame) -> pd.DataFrame:
    """Per player-game aggregate of the stint table."""
    g = st.groupby(["game_id", "team_id", "player_id"], sort=False)
    out = g.agg(
        n_stints=("stint_sec", "size"),
        rot_sec=("stint_sec", "sum"),
        rot_pts=("stint_pts", "sum"),
        first_in=("in_t", "min"),
        last_out=("out_t", "max"),
        is_home=("is_home", "first"),
    ).reset_index()
    # duration-weighted USG_PCT and PT_DIFF; PT_DIFF is per-stint plus/minus so
    # it SUMS, usage is a rate so it is time-weighted.
    st = st.copy()
    st["_wu"] = st["usg_pct"] * st["stint_sec"]
    w = st.groupby(["game_id", "team_id", "player_id"], sort=False).agg(
        wu=("_wu", "sum"), ptdiff_sum=("pt_diff", "sum")).reset_index()
    out = out.merge(w, on=["game_id", "team_id", "player_id"], how="left")
    out["usg_w"] = np.where(out["rot_sec"] > 0, out["wu"] / out["rot_sec"], 0.0)
    out = out.drop(columns=["wu"])
    # stint fragmentation + starter flag
    out["is_starter"] = (out["first_in"] <= 0.5).astype(int)
    return out


def main():
    rep = {}
    print("--- natl TV ---", flush=True)
    tv, bad_tv = ingest_natl_tv()
    print(f"games {len(tv)}  unparsable {bad_tv}  natl-tv {tv.is_natl_tv.sum()} "
          f"({100*tv.is_natl_tv.mean():.2f}%)", flush=True)
    rep["natl_tv"] = dict(n_games=len(tv), unparsable=bad_tv,
                          n_natl=int(tv.is_natl_tv.sum()),
                          share=float(tv.is_natl_tv.mean()),
                          broadcasters=tv.natl_tv.value_counts().to_dict())

    print("--- rotation ---", flush=True)
    st, bad_rot = ingest_rotation()
    print(f"stints {len(st)}  files-bad {bad_rot}  games {st.game_id.nunique()}",
          flush=True)
    pg = aggregate_rotation(st)
    print(f"player-games {len(pg)}", flush=True)

    con = connect(read_only=True)
    games = con.execute("""
        SELECT DISTINCT game_id, season, game_date FROM nba_games
        WHERE game_id LIKE '002%'
    """).fetchdf()
    games["game_id"] = games["game_id"].astype(str)

    tv2 = tv.merge(games, on="game_id", how="inner")
    rep["natl_tv"]["by_season"] = (
        tv2.groupby("season").agg(n=("game_id", "size"),
                                  natl=("is_natl_tv", "sum"),
                                  share=("is_natl_tv", "mean"))
        .round(4).to_dict("index"))
    print(tv2.groupby("season").agg(n=("game_id", "size"),
                                    natl=("is_natl_tv", "sum"),
                                    share=("is_natl_tv", "mean")).round(4))

    pg2 = pg.merge(games, on="game_id", how="inner")
    rep["rotation"] = dict(n_stints=len(st), n_files_bad=bad_rot,
                           n_games=int(st.game_id.nunique()),
                           n_player_games=len(pg),
                           n_player_games_reg=len(pg2))
    rs = pg2.groupby("season").agg(games=("game_id", "nunique"),
                                   pg=("player_id", "size")).to_dict("index")
    rep["rotation"]["by_season"] = rs
    print(pg2.groupby("season").agg(games=("game_id", "nunique"),
                                    pg=("player_id", "size")))

    # ------------------------------------------------------ VALIDATION
    # rotation minutes vs box-score seconds
    box = con.execute("""
        SELECT game_id, team_id, player_id, seconds, pts FROM player_game_stats
        WHERE game_id LIKE '002%'
    """).fetchdf()
    box["game_id"] = box["game_id"].astype(str)
    m = pg2.merge(box, on=["game_id", "team_id", "player_id"], how="inner")
    m["d_sec"] = m["rot_sec"] - m["seconds"]
    m["d_pts"] = m["rot_pts"] - m["pts"]
    rep["validation"] = dict(
        n_matched=len(m),
        sec_mae=float(np.abs(m.d_sec).mean()),
        sec_p50=float(np.abs(m.d_sec).median()),
        sec_within_5s=float((np.abs(m.d_sec) <= 5).mean()),
        sec_within_30s=float((np.abs(m.d_sec) <= 30).mean()),
        pts_exact=float((m.d_pts == 0).mean()),
        pts_mae=float(np.abs(m.d_pts).mean()),
        rot_only=int(len(pg2) - len(m)),
        box_only=int(len(box.merge(pg2[["game_id", "team_id", "player_id"]],
                                   on=["game_id", "team_id", "player_id"],
                                   how="left", indicator=True)
                       .query('_merge=="left_only"'))),
    )
    print("VALIDATION", json.dumps(rep["validation"], indent=2))

    # PT_DIFF sanity: sum over a team's stints should relate to the team margin
    con.close()
    st.to_csv(ROOT / "data" / "ad_rotation.csv.gz", index=False, compression="gzip")
    pg.to_csv(ROOT / "data" / "ad_rotation_pg.csv.gz", index=False, compression="gzip")
    tv.to_csv(ROOT / "data" / "ad_natl_tv.csv", index=False)
    (ROOT / "data" / "ad_ingest.json").write_text(json.dumps(rep, indent=2, default=str))
    print("AD_INGEST_DONE")


if __name__ == "__main__":
    main()
