"""Free historical odds: sportsbookreviewsonline.com season archives
(2007-08 .. 2022-23). Open/Close spread + total, ML, 2H — one composite book,
two rows per game (V then H). This is the $0 substitute for paid historical
snapshots (docs/PAID_OPTIONS.md): good enough for H-A open->close movement
backtests, no intraday path, no props.

SBR column quirk: 'Open'/'Close' mix spreads and totals — per game, the larger
of (V,H) values is the TOTAL and the smaller is the SPREAD (favorite's side
holds the spread). 'pk'/'PK' = 0. We store both raw values and the resolved
spread/total. Raw HTML archived under data/raw/sbr/.
"""
from __future__ import annotations

import datetime as dt
import logging
import re

import pandas as pd
import requests

from ..config import RAW

log = logging.getLogger("sbr_hist")

RAW_SBR = RAW / "sbr"
RAW_SBR.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0"}
BASE = "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba-odds-{season}"
SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2007, 2023)]  # 2007-08 .. 2022-23

SCHEMA = """
CREATE TABLE IF NOT EXISTS odds_hist_sbr (
    season       VARCHAR NOT NULL,
    game_date    DATE,
    visitor      VARCHAR,
    home         VARCHAR,
    v_final      INTEGER,
    h_final      INTEGER,
    spread_open  DOUBLE,     -- home-perspective (negative = home favored)
    spread_close DOUBLE,
    total_open   DOUBLE,
    total_close  DOUBLE,
    v_ml         INTEGER,
    h_ml         INTEGER,
    ingest_ts    TIMESTAMPTZ NOT NULL
);
"""


def _num(s: str) -> float | None:
    s = s.strip().lower().replace("½", ".5")
    if s in ("pk", "p", "", "nan", "nl"):
        return 0.0 if s in ("pk", "p") else None
    try:
        v = float(s)
        return None if v != v else v
    except ValueError:
        return None


def fetch_season(season: str) -> str:
    path = RAW_SBR / f"nba-odds-{season}.html"
    if path.exists():
        return path.read_text()
    r = requests.get(BASE.format(season=season), headers=UA, timeout=30)
    r.raise_for_status()
    path.write_text(r.text)
    return r.text


def fetch_season_xlsx(season: str) -> pd.DataFrame | None:
    """Full-season Excel via the Wayback Machine (live site dropped the files).
    Raw xlsx archived to RAW_SBR. Returns None when no snapshot exists."""
    path = RAW_SBR / f"nba odds {season}.xlsx"
    # Negative cache: 2008-09 and 2009-10 have no Wayback xlsx, and re-probing
    # them costs ~150s of rate-limited retries on EVERY rebuild. Record the miss.
    miss = RAW_SBR / f".no-xlsx-{season}"
    if not path.exists():
        if miss.exists():
            return None
        import time
        src = f"sportsbookreviewsonline.com/scoresoddsarchives/nba/nba%20odds%20{season}.xlsx"
        snap = None
        for attempt in range(5):  # availability API rate-limits aggressively
            r = requests.get(f"http://archive.org/wayback/available?url={src}", timeout=30)
            try:
                snap = r.json().get("archived_snapshots", {}).get("closest")
                break
            except ValueError:
                time.sleep(10 * (attempt + 1))
        if not snap or snap.get("status") != "200":
            miss.touch()
            return None
        # 'if_' suffix serves the original bytes without the wayback banner
        u = snap["url"].replace("/https://", "if_/https://", 1)
        x = requests.get(u, timeout=120)
        x.raise_for_status()
        if x.content[:2] != b"PK":   # snapshot exists but is not an xlsx
            miss.touch()
            return None
        path.write_bytes(x.content)
    import io
    return pd.read_excel(io.BytesIO(path.read_bytes()))


def parse_season(source: str | pd.DataFrame, season: str) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        df = source
    else:
        import io
        tables = pd.read_html(io.StringIO(source))
        df = max(tables, key=len)
    # header row may be data row 0 or actual header; normalize
    if "Date" not in df.columns:
        df.columns = df.iloc[0]
        df = df.iloc[1:]
    df = df.reset_index(drop=True)
    start_year = int(season[:4])
    # Year inference: SBR lists games chronologically, so walk the month
    # sequence and bump the year only on a 12->1 wrap. A naive "month >= 9 ->
    # start_year" rule silently misdates the 2019-20 bubble (Sep/Oct 2020
    # games landed in 2019, 109 games a full year early).
    year, prev_month = start_year, None
    games = []
    for i in range(0, len(df) - 1, 2):
        v, h = df.iloc[i], df.iloc[i + 1]
        if str(v.get("VH", "")).strip() not in ("V", "N") or str(h.get("VH", "")).strip() not in ("H", "N"):
            continue  # malformed pair; skip
        raw_date = str(v["Date"]).strip().split(".")[0]
        if not raw_date.isdigit():
            continue
        mmdd = raw_date.zfill(4)
        month, day = int(mmdd[:-2]), int(mmdd[-2:])
        if prev_month is not None and month < prev_month - 6:
            year += 1          # calendar wrap (Dec -> Jan), once per season
        prev_month = month
        vo, ho = _num(str(v["Open"])), _num(str(h["Open"]))
        vc, hc = _num(str(v["Close"])), _num(str(h["Close"]))

        def resolve(vx, hx):
            """(spread_home_perspective, total). Per game one cell is the total
            (large) and the other the spread, listed on the favorite's row.

            The split is inferred from magnitude, so a corrupt source row where
            BOTH cells are large silently fabricates a spread of ~200 and a
            total of ~1100 (real example: 2019-12-09 IND/LAC). Refuse to guess
            when either resolved value is outside its plausible range - a NULL
            is recoverable, a fabricated 216-point spread poisons every
            downstream mean."""
            if vx is None or hx is None:
                return None, None
            spread, total = (-hx, vx) if vx > hx else (vx, hx)
            if abs(spread) > 40 or not (120 <= total <= 300):
                return None, None
            return spread, total

        so, to = resolve(vo, ho)
        sc, tc = resolve(vc, hc)
        games.append({
            "season": season,
            "game_date": dt.date(year, month, day),
            "visitor": str(v["Team"]).strip(), "home": str(h["Team"]).strip(),
            "v_final": int(_num(str(v["Final"])) or 0), "h_final": int(_num(str(h["Final"])) or 0),
            "spread_open": so, "spread_close": sc,
            "total_open": to, "total_close": tc,
            "v_ml": int(_num(str(v.get("ML", ""))) or 0) or None,
            "h_ml": int(_num(str(h.get("ML", ""))) or 0) or None,
        })
    return pd.DataFrame(games)


def load_all(connect_fn, seasons: list[str] | None = None) -> dict[str, int]:
    """Fetch+parse everything BEFORE touching DuckDB, then write in one short
    transaction — network work must never sit on the single writer lock.
    `connect_fn` is called only after all fetching is done."""
    out, parsed = {}, {}
    for season in seasons or SEASONS:
        try:
            src = fetch_season_xlsx(season)
            if src is None:
                log.warning("%s: no wayback xlsx, falling back to (possibly "
                            "truncated) live HTML page", season)
                src = fetch_season(season)
            games = parse_season(src, season)
            parsed[season] = games.astype(object).where(pd.notna(games), None)
            out[season] = len(games)
            log.info("%s: %d games", season, len(games))
        except Exception:
            log.exception("season %s failed", season)
            out[season] = -1

    now = dt.datetime.now(dt.timezone.utc)
    if not parsed:
        return out
    # One bulk INSERT ... SELECT, not 17k single-row statements: each execute()
    # is its own transaction, so the row-at-a-time version sat on the single
    # write lock for minutes and blocked every other reader.
    frame = pd.concat(parsed.values(), ignore_index=True)
    frame["ingest_ts"] = now
    frame = frame[["season", "game_date", "visitor", "home", "v_final",
                   "h_final", "spread_open", "spread_close", "total_open",
                   "total_close", "v_ml", "h_ml", "ingest_ts"]]
    con = connect_fn()
    con.execute(SCHEMA)
    con.execute("DELETE FROM odds_hist_sbr WHERE season IN "
                f"({','.join('?' * len(parsed))})", list(parsed))
    con.execute("INSERT INTO odds_hist_sbr SELECT * FROM frame")
    con.close()
    return out
