#!/usr/bin/env python3
"""Pull NBA odds from Action Network's public scoreboard API (no auth, no key).

  https://api.actionnetwork.com/web/v2/scoreboard/nba?date=YYYYMMDD

Each game carries a `markets` dict keyed by book_id. book_id 30 == "Open"
(the opening line), book_id 15 == "Consensus" (the current/closing line);
the rest are individual sportsbooks. Verified against /web/v1/books.

Writes per season:
  an_nba_odds_raw_<season>.jsonl     one line per date (full API payload)
  an_nba_open_close_<season>.csv     one row per (game, book)
"""
import json
import csv
import os
import sys
import time
import datetime as dt
import urllib.request
import urllib.error

OUT = "/hdd/steveqin/sean_dev/nba_model/data/raw/sbr_ext"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/122 Safari/537.36"}
SLEEP = 0.25

SEASONS = {
    "2022-23": ("2022-10-01", "2023-06-30"),
    "2023-24": ("2023-10-01", "2024-06-30"),
    "2024-25": ("2024-10-01", "2025-06-30"),
    "2025-26": ("2025-10-01", "2026-06-30"),
}

BOOKS = {}  # id -> display name, filled at runtime


def get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=40) as f:
                return json.load(f)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 ** i)
        except Exception:
            time.sleep(1.5 * (i + 1))
    return None


def side(entries, want, field):
    if not entries:
        return None
    for e in entries:
        if e.get("side") == want and e.get("period") in (None, "event"):
            return e.get(field)
    return None


COLS = [
    "season", "game_date", "start_time_utc", "an_game_id", "game_type", "status",
    "away_team", "home_team", "away_abbr", "home_abbr",
    "away_score", "home_score",
    "book_id", "book_name",
    "spread_home", "spread_home_odds", "spread_away", "spread_away_odds",
    "ml_home", "ml_away",
    "total", "total_over_odds", "total_under_odds",
]


def main():
    os.makedirs(OUT, exist_ok=True)
    bk = get("https://api.actionnetwork.com/web/v1/books")
    if bk:
        for b in bk.get("books", []):
            BOOKS[str(b["id"])] = b.get("display_name")
    print("books loaded:", len(BOOKS), "| 30 =", BOOKS.get("30"), "| 15 =", BOOKS.get("15"), flush=True)

    for season in (sys.argv[1:] or list(SEASONS)):
        d0, d1 = SEASONS[season]
        d = dt.date.fromisoformat(d0)
        end = dt.date.fromisoformat(d1)
        raw_p = os.path.join(OUT, f"an_nba_odds_raw_{season}.jsonl")
        csv_p = os.path.join(OUT, f"an_nba_open_close_{season}.csv")
        ngame = nrow = nopen = 0
        with open(raw_p, "w") as rf, open(csv_p, "w", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=COLS)
            w.writeheader()
            while d <= end:
                ds = d.strftime("%Y%m%d")
                j = get(f"https://api.actionnetwork.com/web/v2/scoreboard/nba?date={ds}")
                time.sleep(SLEEP)
                games = (j or {}).get("games", [])
                if games:
                    rf.write(json.dumps({"season": season, "date": d.isoformat(),
                                         "games": games}) + "\n")
                for g in games:
                    ngame += 1
                    teams = {t["id"]: t for t in (g.get("teams") or [])}
                    at = teams.get(g.get("away_team_id"), {})
                    ht = teams.get(g.get("home_team_id"), {})
                    bx = g.get("boxscore") or {}
                    base = dict(
                        season=season, game_date=d.isoformat(),
                        start_time_utc=g.get("start_time"), an_game_id=g.get("id"),
                        game_type=g.get("type"), status=g.get("status"),
                        away_team=at.get("full_name"), home_team=ht.get("full_name"),
                        away_abbr=at.get("abbr"), home_abbr=ht.get("abbr"),
                        away_score=(bx.get("total_away_points")
                                    if bx.get("total_away_points") is not None
                                    else (bx.get("away") or {}).get("points")),
                        home_score=(bx.get("total_home_points")
                                    if bx.get("total_home_points") is not None
                                    else (bx.get("home") or {}).get("points")),
                    )
                    mk = g.get("markets") or {}
                    has_open = False
                    for bid, mv in mk.items():
                        ev = (mv or {}).get("event") or {}
                        sp, ml, to = ev.get("spread"), ev.get("moneyline"), ev.get("total")
                        if not (sp or ml or to):
                            continue
                        if bid == "30" and sp:
                            has_open = True
                        row = dict(base)
                        row["book_id"] = bid
                        row["book_name"] = BOOKS.get(bid)
                        row["spread_home"] = side(sp, "home", "value")
                        row["spread_home_odds"] = side(sp, "home", "odds")
                        row["spread_away"] = side(sp, "away", "value")
                        row["spread_away_odds"] = side(sp, "away", "odds")
                        row["ml_home"] = side(ml, "home", "odds")
                        row["ml_away"] = side(ml, "away", "odds")
                        row["total"] = side(to, "over", "value")
                        row["total_over_odds"] = side(to, "over", "odds")
                        row["total_under_odds"] = side(to, "under", "odds")
                        w.writerow({k: row.get(k) for k in COLS})
                        nrow += 1
                    if has_open:
                        nopen += 1
                if d.day == 1:
                    print(f"  [{season}] {d} games={ngame} rows={nrow} w/open={nopen}", flush=True)
                    cf.flush()
                    rf.flush()
                d += dt.timedelta(days=1)
        print(f"[{season}] DONE games={ngame} games_with_open={nopen} rows={nrow}\n"
              f"          {csv_p}\n          {raw_p}", flush=True)


if __name__ == "__main__":
    main()
