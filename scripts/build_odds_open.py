#!/usr/bin/env python3
"""Build `odds_open`: opening AND closing lines on OUR team keys.

Why this table exists: `odds_market` carries only ONE price per game (the
close). Any strategy claiming edge must bet at a price better than the close,
which is unbacktestable without a pre-close price. `odds_open` supplies the
open, so open->close movement is measurable.

Sources (both free, stitched because neither covers the whole span):
  sbr       sportsbookreviewsonline.com season archives   2007-08 .. 2022-23
            (the site stopped publishing on 2023-01-16 mid-season)
  teamrankings  per-game spread-movement pages           2023-24 .. 2025-26
            (their spread history does not reach back before 2023-24)
See docs/OPENING_LINES.md for every avenue tried.

Raw files on disk are the ground truth, so this re-parses the archives rather
than reading the staging table. Output goes to BOTH DuckDB (`odds_open`) and
data/derived/odds_open.parquet - the parquet keeps analysis runnable while a
loader holds the single write lock.

Sign conventions (both stored, because mixing them up is the obvious failure):
  open_spread / close_spread   betting line ON THE HOME TEAM (neg = home favored)
  open_margin / close_margin   expected HOME margin (pos = home favored)
                               == -spread, matches odds_market.home_exp_margin
  line_move = close_margin - open_margin        (positive = moved TOWARD home)

SBR's ML column is the CLOSING moneyline only - the source carries no opening
ML, so open_ml_* is NULL for SBR rows. Opening moneylines for 2023-24 onward do
exist (Kaggle chevronronson/nba-stats-dataset, teamrankings-derived) and are
merged in where available.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nbapred.config import DATA, RAW  # noqa: E402
from nbapred.ingest.sbr_hist import SEASONS, fetch_season, fetch_season_xlsx, parse_season  # noqa: E402

log = logging.getLogger("odds_open")

SBR_URL = "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba-odds-{s}"
TR_JSONL = RAW / "teamrankings" / "spread_movement.jsonl"
EXT_DIR = RAW / "sbr_ext"          # ESPN + Action Network merged per-game csvs
KAGGLE_ML = (RAW / "kaggle" / "chevronronson__nba-stats-dataset" / "game_odds.csv")
OUT_CSV = DATA / "derived" / "odds_open.csv"   # no parquet engine installed

# SBR / teamrankings spellings -> our abbreviations. Relocated franchises map to
# the CURRENT abbreviation so joins against odds_market/nba_games work across
# the whole span (Seattle SuperSonics -> OKC, New Jersey Nets -> BKN).
TEAMS = {
    "Atlanta": "ATL", "Boston": "BOS", "Brooklyn": "BKN", "Charlotte": "CHA",
    "Chicago": "CHI", "Cleveland": "CLE", "Dallas": "DAL", "Denver": "DEN",
    "Detroit": "DET", "GoldenState": "GSW", "Golden State": "GSW",
    "Houston": "HOU", "Indiana": "IND", "LAClippers": "LAC",
    "LA Clippers": "LAC", "LALakers": "LAL", "LA Lakers": "LAL",
    "Memphis": "MEM", "Miami": "MIA", "Milwaukee": "MIL", "Minnesota": "MIN",
    "NewJersey": "BKN", "New Jersey": "BKN", "NewOrleans": "NOP",
    "New Orleans": "NOP", "NewYork": "NYK", "New York": "NYK",
    "OklahomaCity": "OKC", "Oklahoma City": "OKC", "Orlando": "ORL",
    "Philadelphia": "PHI", "Phoenix": "PHX", "Portland": "POR",
    "Sacramento": "SAC", "SanAntonio": "SAS", "San Antonio": "SAS",
    "Seattle": "OKC", "Toronto": "TOR", "Utah": "UTA", "Washington": "WAS",
}

# teamrankings/ESPN use shorter abbreviations than we do. Leaving these
# unmapped silently drops every game involving one of these teams (a 63% match
# rate that looks like a scrape failure rather than a mapping bug).
TR_TEAMS = {"BK": "BKN", "GS": "GSW", "NO": "NOP", "NY": "NYK",
            "PHO": "PHX", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}

COLS = ["season_end", "season", "game_date", "home", "away", "score_home",
        "score_away", "home_win", "open_spread", "close_spread", "open_margin",
        "close_margin", "open_total", "close_total", "open_ml_home",
        "open_ml_away", "close_ml_home", "close_ml_away", "line_move",
        "total_move", "source", "source_url", "ingest_ts"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS odds_open (
    season_end    INTEGER,      -- 2022 = 2021-22 season
    season        VARCHAR,
    game_date     DATE,
    home          VARCHAR, away VARCHAR,
    score_home    INTEGER, score_away INTEGER,
    home_win      BOOLEAN,
    open_spread   DOUBLE,       -- line on HOME (negative = home favored)
    close_spread  DOUBLE,
    open_margin   DOUBLE,       -- expected HOME margin (positive = home favored)
    close_margin  DOUBLE,
    open_total    DOUBLE, close_total DOUBLE,
    open_ml_home  INTEGER, open_ml_away INTEGER,
    close_ml_home INTEGER, close_ml_away INTEGER,
    line_move     DOUBLE,       -- close_margin - open_margin (+ = toward home)
    total_move    DOUBLE,
    source        VARCHAR,
    source_url    VARCHAR,
    ingest_ts     TIMESTAMPTZ NOT NULL
);
"""


def _from_sbr() -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        src = fetch_season_xlsx(season)
        if src is None:
            src = fetch_season(season)
        g = parse_season(src, season)
        if len(g):
            g["season"] = season
            frames.append(g)
        log.info("sbr %s: %d games", season, len(g))
    src = pd.concat(frames, ignore_index=True)

    src["home_ab"] = src.home.str.strip().map(TEAMS)
    src["away_ab"] = src.visitor.str.strip().map(TEAMS)
    bad = set(src[src.home_ab.isna()].home) | set(src[src.away_ab.isna()].visitor)
    if bad:
        raise SystemExit(f"unmapped SBR team names: {sorted(bad)}")

    out = pd.DataFrame({
        "season_end": src.season.str[:4].astype(int) + 1,
        "season": src.season,
        "game_date": pd.to_datetime(src.game_date).dt.date,
        "home": src.home_ab, "away": src.away_ab,
        "score_home": src.h_final, "score_away": src.v_final,
        "home_win": src.h_final > src.v_final,
        "open_spread": src.spread_open, "close_spread": src.spread_close,
        "open_margin": -src.spread_open, "close_margin": -src.spread_close,
        "open_total": src.total_open, "close_total": src.total_close,
        "open_ml_home": None, "open_ml_away": None,
        "close_ml_home": src.h_ml, "close_ml_away": src.v_ml,
        "source": "sbr",
        "source_url": [SBR_URL.format(s=s) for s in src.season],
    })
    return out


def _pair_join(df: pd.DataFrame, con, a: str, b: str) -> pd.DataFrame:
    """Resolve home/away, season and final score by joining the UNORDERED team
    pair + date against odds_market. Doing it this way also drops preseason,
    All-Star and exhibition rows for free (Melbourne United, Team Chuck, ...),
    which the raw feeds happily include."""
    mk = con.execute("""SELECT season_end, game_date, home, away, score_home,
                               score_away, total close_total_mkt
                        FROM odds_market""").df()
    mk["game_date"] = pd.to_datetime(mk.game_date).dt.date
    mk["pair"] = [frozenset(p) for p in zip(mk.home, mk.away)]
    df = df.copy()
    df[a] = df[a].replace(TR_TEAMS)
    df[b] = df[b].replace(TR_TEAMS)
    df["pair"] = [frozenset(p) for p in zip(df[a], df[b])]
    return df.merge(mk, on=["game_date", "pair"], how="inner")


def _from_espn_actionnetwork(con) -> pd.DataFrame:
    """ESPN core API + Action Network scoreboard, merged per game by
    scripts/build_nba_open_close.py. Both are keyless and public. This is the
    PRIMARY source for 2023-24..2025-26: it carries opening spread, total AND
    moneyline (teamrankings has spread only), and the two independent feeds
    cross-validate at corr ~0.97.

    Trap on record: ESPN's open.pointSpread.value holds the DECIMAL PRICE, not
    the handicap, in some vintages (all of 2024-25) - the handicap lives in the
    display string. And ESPN dates are UTC while Action Network's are ET."""
    frames = []
    for path in sorted(EXT_DIR.glob("nba_open_close_*.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        log.warning("no ESPN/ActionNetwork csvs in %s", EXT_DIR)
        return pd.DataFrame(columns=COLS)
    e = pd.concat(frames, ignore_index=True)
    e = e[e.open_spread.notna() & e.close_spread.notna()].copy()
    e["game_date"] = pd.to_datetime(e.game_date_et).dt.date
    m = _pair_join(e, con, "home_abbr", "away_abbr")
    log.info("espn+an: %d rows with open -> %d matched to odds_market", len(e), len(m))
    if not len(m):
        return pd.DataFrame(columns=COLS)

    # feed spreads are the HOME handicap already (negative = home favored), but
    # the pair join may have flipped which side we call home - re-anchor.
    flip = m.home_abbr.replace(TR_TEAMS) != m.home
    open_spread = m.open_spread.where(~flip, -m.open_spread)
    close_spread = m.close_spread.where(~flip, -m.close_spread)
    oh, oa = m.open_home_ml, m.open_away_ml
    ch, ca = m.close_home_ml, m.close_away_ml
    return pd.DataFrame({
        "season_end": m.season_end,
        "season": m.season_end.map(lambda y: f"{y - 1}-{str(y)[-2:]}"),
        "game_date": m.game_date,
        "home": m.home, "away": m.away,
        "score_home": m.score_home, "score_away": m.score_away,
        "home_win": m.score_home > m.score_away,
        "open_spread": open_spread, "close_spread": close_spread,
        "open_margin": -open_spread, "close_margin": -close_spread,
        "open_total": m.open_total, "close_total": m.close_total,
        "open_ml_home": oh.where(~flip, oa), "open_ml_away": oa.where(~flip, oh),
        "close_ml_home": ch.where(~flip, ca), "close_ml_away": ca.where(~flip, ch),
        "source": "espn+actionnetwork",
        "source_url": "sports.core.api.espn.com + api.actionnetwork.com",
    })


def _newest_quote(history: list) -> float | None:
    """Closing line = the most recent timestamped quote (history is newest
    first). The page's own 'Last' field disagrees with the history table on
    some games, so trust the timestamped path, not the summary cell."""
    for row in history or []:
        for b in ("book1", "book2", "book3"):
            if row.get(b) is not None:
                return float(row[b])
    return None


def _from_teamrankings(con) -> pd.DataFrame:
    """Per-game spread-movement pages, scraped by scripts/scrape_teamrankings.py.
    Covers the seasons SBR abandoned (2023-24 onward).

    The pages give lines from the FAVORITE's perspective and never say which
    side is home, so home/away and the final score are resolved by joining the
    unordered team pair + date against odds_market."""
    if not TR_JSONL.exists():
        log.warning("no teamrankings jsonl at %s - 2023-24..2025-26 will have "
                    "NO opening line. Run scripts/scrape_teamrankings.py.", TR_JSONL)
        return pd.DataFrame(columns=COLS)
    recs = [json.loads(x) for x in TR_JSONL.read_text().splitlines() if x.strip()]
    tr = pd.DataFrame(recs)
    tr = tr[~tr.get("no_data", pd.Series(False, index=tr.index)).fillna(False)]
    tr = tr[tr.fav_open.notna() & tr.fav_team.notna()].copy()
    if not len(tr):
        return pd.DataFrame(columns=COLS)
    tr["game_date"] = pd.to_datetime(tr.game_date).dt.date
    tr["close_fav"] = tr.history.map(_newest_quote)
    tr["close_fav"] = tr.close_fav.fillna(tr.fav_last)
    tr = tr.drop(columns=["season_end"], errors="ignore")   # odds_market owns it
    m = _pair_join(tr, con, "fav_team", "dog_team")
    log.info("teamrankings: %d scraped -> %d matched to odds_market", len(tr), len(m))

    fav_is_home = m.fav_team == m.home
    # line ON THE HOME TEAM: the favorite's number if home is favored, else the
    # dog's (which is the favorite's number negated).
    open_spread = pd.Series(
        [f if h else -f for f, h in zip(m.fav_open, fav_is_home)], index=m.index)
    close_spread = pd.Series(
        [f if h else -f for f, h in zip(m.close_fav, fav_is_home)], index=m.index)

    return pd.DataFrame({
        "season_end": m.season_end,
        "season": m.season_end.map(lambda y: f"{y - 1}-{str(y)[-2:]}"),
        "game_date": m.game_date,
        "home": m.home, "away": m.away,
        "score_home": m.score_home, "score_away": m.score_away,
        "home_win": m.score_home > m.score_away,
        "open_spread": open_spread, "close_spread": close_spread,
        "open_margin": -open_spread, "close_margin": -close_spread,
        "open_total": None, "close_total": m.close_total_mkt,
        "open_ml_home": None, "open_ml_away": None,
        "close_ml_home": None, "close_ml_away": None,
        "source": "teamrankings", "source_url": m.url,
    })


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    from nbapred.db import connect
    ro = connect(read_only=True)
    # Priority order matters: the de-dup below keeps the FIRST row per game.
    # SBR owns 2007-08..2022-23; for the gap seasons ESPN+ActionNetwork wins
    # over teamrankings because it carries totals and moneylines too, and is
    # cross-validated across two independent feeds.
    parts = [_from_sbr(), _from_espn_actionnetwork(ro), _from_teamrankings(ro)]
    ro.close()
    out = pd.concat([p for p in parts if len(p)], ignore_index=True)

    out["line_move"] = out.close_margin - out.open_margin
    out["total_move"] = out.close_total - out.open_total
    out["ingest_ts"] = dt.datetime.now(dt.timezone.utc)
    out = out.reindex(columns=COLS)

    # Drop rows with no usable final score (0-0 placeholders, ties).
    n0 = len(out)
    out = out[(out.score_home > 0) & (out.score_away > 0)
              & (out.score_home != out.score_away)].copy()
    log.info("dropped %d rows with no usable final score", n0 - len(out))
    out = out.drop_duplicates(subset=["game_date", "home", "away"], keep="first")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    log.info("wrote %s (%d rows)", OUT_CSV, len(out))

    try:
        con = connect()
    except Exception as e:                       # loader holds the write lock
        log.warning("DuckDB write skipped (%s). The CSV is current; rerun to "
                    "land the table.", e)
        print(out.groupby("season").size().to_string())
        return
    con.execute(SCHEMA)
    con.execute("DELETE FROM odds_open")
    con.execute("INSERT INTO odds_open SELECT * FROM out")
    cov = con.execute("""
        SELECT season, source, count(*) games, count(open_spread) with_open,
               min(game_date) first_game, max(game_date) last_game
        FROM odds_open GROUP BY 1, 2 ORDER BY 1
    """).df()
    gap = con.execute("""
        SELECT m.season_end, count(*) corpus_games,
               count(o.game_date) with_open
        FROM odds_market m
        LEFT JOIN odds_open o
          ON o.game_date = m.game_date AND o.home = m.home AND o.away = m.away
        WHERE m.season_end >= 2022 GROUP BY 1 ORDER BY 1
    """).df()
    con.close()
    print(cov.to_string(index=False))
    # Say the absence out loud. A left join against a season with no opening
    # line returns NULLs that downstream code will happily average as zero.
    gap["pct"] = (gap.with_open / gap.corpus_games * 100).round(1)
    print("\nMODEL-CORPUS COVERAGE (join on game_date+home+away):")
    print(gap.to_string(index=False))
    thin = gap[gap.pct < 99]
    if len(thin):
        print("\n!! PARTIAL COVERAGE - do NOT join odds_open across these "
              "seasons without filtering on open_spread IS NOT NULL:")
        for r in thin.itertuples():
            print(f"     season_end {r.season_end}: {r.with_open}/"
                  f"{r.corpus_games} ({r.pct}%)")
    print(f"\nodds_open rows: {len(out)}")


if __name__ == "__main__":
    main()
