#!/usr/bin/env python3
"""Merge ESPN + Action Network into one per-game opening/closing line file.

Output: nba_open_close_<season>.csv, one row per game, with an `open_*` column
sourced from whichever feed has it (ESPN median-across-books preferred, Action
Network 'Open' book as fallback) plus both raw sources kept side by side so the
choice is auditable.
"""
import os
import pandas as pd

OUT = "/hdd/steveqin/sean_dev/nba_model/data/raw/sbr_ext"
SEASONS = ["2023-24", "2024-25", "2025-26"]
ABBR = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}


def norm(x):
    return ABBR.get(str(x).upper(), str(x).upper()) if pd.notna(x) else x


allrows = []
for s in SEASONS:
    e = pd.read_csv(f"{OUT}/espn_nba_open_close_{s}.csv")
    a = pd.read_csv(f"{OUT}/an_nba_open_close_{s}.csv")

    e = e[~e.provider_name.astype(str).str.contains("Live Odds", na=False)]
    eg = (e.groupby(["event_id", "game_date_et", "start_time_utc", "season_type",
                     "home_abbr", "away_abbr", "home_team", "away_team",
                     "home_score", "away_score"], as_index=False, dropna=False)
            .agg(espn_open_spread=("open_home_spread", "median"),
                 espn_close_spread=("close_home_spread", "median"),
                 espn_open_total=("open_total", "median"),
                 espn_close_total=("close_total", "median"),
                 espn_open_home_ml=("open_home_ml", "median"),
                 espn_close_home_ml=("close_home_ml", "median"),
                 espn_open_away_ml=("open_away_ml", "median"),
                 espn_close_away_ml=("close_away_ml", "median"),
                 espn_n_books_open=("open_home_spread", "count")))
    eg["k"] = eg.game_date_et + "|" + eg.home_abbr.map(norm) + "|" + eg.away_abbr.map(norm)

    ao = a[a.book_id == 30].rename(columns={
        "spread_home": "an_open_spread", "total": "an_open_total",
        "ml_home": "an_open_home_ml", "ml_away": "an_open_away_ml",
        "spread_home_odds": "an_open_spread_odds"})
    ac = a[a.book_id == 15].rename(columns={
        "spread_home": "an_close_spread", "total": "an_close_total",
        "ml_home": "an_close_home_ml", "ml_away": "an_close_away_ml"})
    # game-level spine first, so a game with a close but no open keeps its key
    spine = (a[["an_game_id", "game_date", "home_abbr", "away_abbr", "game_type",
                "home_team", "away_team", "home_score", "away_score"]]
             .drop_duplicates("an_game_id"))
    keep_o = ["an_game_id", "an_open_spread", "an_open_spread_odds", "an_open_total",
              "an_open_home_ml", "an_open_away_ml"]
    keep_c = ["an_game_id", "an_close_spread", "an_close_total",
              "an_close_home_ml", "an_close_away_ml"]
    ag = (spine.merge(ao[keep_o], on="an_game_id", how="left")
               .merge(ac[keep_c], on="an_game_id", how="left"))
    ag["k"] = ag.game_date + "|" + ag.home_abbr.map(norm) + "|" + ag.away_abbr.map(norm)
    ag = ag.rename(columns={"game_date": "an_game_date", "home_abbr": "an_home_abbr",
                            "away_abbr": "an_away_abbr", "home_team": "an_home_team",
                            "away_team": "an_away_team", "home_score": "an_home_score",
                            "away_score": "an_away_score", "game_type": "an_game_type"})

    m = eg.merge(ag, on="k", how="outer")
    m["season"] = s
    for c, ac_ in [("game_date_et", "an_game_date"), ("home_abbr", "an_home_abbr"),
                   ("away_abbr", "an_away_abbr"), ("home_team", "an_home_team"),
                   ("away_team", "an_away_team"), ("home_score", "an_home_score"),
                   ("away_score", "an_away_score")]:
        m[c] = m[c].combine_first(m[ac_])
    m["season_type"] = m.season_type.combine_first(
        m.an_game_type.map({"reg": "regular-season", "post": "post-season",
                            "pre": "preseason"}))
    m = m.drop(columns=["an_game_type"])
    m = m.drop(columns=["an_game_date", "an_home_abbr", "an_away_abbr", "an_home_team",
                        "an_away_team", "an_home_score", "an_away_score"])
    for f in ("spread", "total", "home_ml", "away_ml"):
        m[f"open_{f}"] = m[f"espn_open_{f}"].combine_first(m[f"an_open_{f}"])
        m[f"close_{f}"] = m[f"espn_close_{f}"].combine_first(m[f"an_close_{f}"])
        m[f"open_{f}_src"] = m[f"espn_open_{f}"].notna().map({True: "espn", False: ""})
        m.loc[m[f"espn_open_{f}"].isna() & m[f"an_open_{f}"].notna(), f"open_{f}_src"] = "actionnetwork"
    m["open_close_move"] = m.open_spread - m.close_spread
    allrows.append(m)

    n = len(m)
    print(f"[{s}] games={n} | open_spread={m.open_spread.notna().sum()} "
          f"({m.open_spread.notna().mean():.1%}) | close_spread={m.close_spread.notna().sum()} "
          f"| open_total={m.open_total.notna().sum()} | open_home_ml={m.open_home_ml.notna().sum()} "
          f"| src: {m[m.open_spread.notna()].open_spread_src.value_counts().to_dict()}")

    cols = (["season", "game_date_et", "start_time_utc", "season_type", "event_id", "an_game_id",
             "away_team", "home_team", "away_abbr", "home_abbr", "away_score", "home_score",
             "open_spread", "close_spread", "open_close_move", "open_total", "close_total",
             "open_home_ml", "close_home_ml", "open_away_ml", "close_away_ml",
             "open_spread_src", "open_total_src", "open_home_ml_src", "espn_n_books_open"]
            + [c for c in m.columns if c.startswith(("espn_", "an_")) and c != "espn_n_books_open"])
    cols = list(dict.fromkeys(c for c in cols if c in m.columns))
    p = f"{OUT}/nba_open_close_{s}.csv"
    m.sort_values(["game_date_et", "home_abbr"])[cols].to_csv(p, index=False)
    print(f"        -> {p}")

full = pd.concat(allrows, ignore_index=True)
print(f"\nTOTAL {len(full)} games 2023-24..2025-26, open_spread on "
      f"{full.open_spread.notna().sum()} ({full.open_spread.notna().mean():.1%})")
