#!/usr/bin/env python3
"""Pull NBA open/close/current odds from ESPN's public core API (no auth).

NOTE on parsing: `open.pointSpread.value` carries the DECIMAL PRICE in some
payload vintages (all of 2024-25), not the handicap -- the handicap only
appears in `american`/`alternateDisplayValue`. Always parse the display string.

Endpoint: sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{id}/competitions/{id}/odds
Each provider item carries open{}, close{}, current{} blocks for pointSpread,
spread(odds), moneyLine, and a top-level open/close for the total.

Writes, per season:
  espn_nba_odds_raw_<season>.jsonl   one line per event (full provider items)
  espn_nba_open_close_<season>.csv   one row per (event, provider), flattened
"""
import json
import csv
import os
import sys
import time
import urllib.request
import urllib.error
import datetime as dt
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

OUT = "/hdd/steveqin/sean_dev/nba_model/data/raw/sbr_ext"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/122 Safari/537.36"}
SLEEP = 0.12

SEASONS = {
    "2023-24": [(2023, m) for m in (10, 11, 12)] + [(2024, m) for m in (1, 2, 3, 4, 5, 6, 7)],
    "2024-25": [(2024, m) for m in (10, 11, 12)] + [(2025, m) for m in (1, 2, 3, 4, 5, 6, 7)],
    "2025-26": [(2025, m) for m in (10, 11, 12)] + [(2026, m) for m in (1, 2, 3, 4, 5, 6, 7)],
}
# optional extra seasons for cross-validation against SBR
if os.environ.get("EXTRA_SEASONS"):
    SEASONS["2022-23"] = [(2022, m) for m in (10, 11, 12)] + [(2023, m) for m in (1, 2, 3, 4, 5, 6)]
    SEASONS["2021-22"] = [(2021, m) for m in (10, 11, 12)] + [(2022, m) for m in (1, 2, 3, 4, 5, 6)]


def get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=40) as f:
                return json.load(f)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (429, 500, 502, 503):
                time.sleep(2 ** i)
                continue
            return None
        except Exception:
            time.sleep(1.5 * (i + 1))
    return None


def month_days(y, m):
    if m == 12:
        return 31
    import calendar
    return calendar.monthrange(y, m)[1]


def parse_american(s):
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip().replace("+", "")
    if s.upper() in ("PK", "EVEN", "EV"):
        return 0.0
    if s in ("", "-", "OFF", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def point_spread(d, phase):
    """Handicap for one side. Display string is authoritative."""
    try:
        v = d[phase]["pointSpread"]
    except (KeyError, TypeError):
        return None
    if not isinstance(v, dict):
        return None
    for key in ("american", "alternateDisplayValue", "displayValue"):
        r = parse_american(v.get(key))
        if r is not None:
            return r
    val = v.get("value")
    # a bare `value` that equals `decimal` and sits in the decimal-price band is a price
    if val is not None and v.get("decimal") == val and 1.0 < float(val) < 2.6:
        return None
    return parse_american(val)


def price(d, phase, field):
    """American price (spread juice / moneyline)."""
    try:
        v = d[phase][field]
    except (KeyError, TypeError):
        return None
    if not isinstance(v, dict):
        return None
    for key in ("american", "alternateDisplayValue"):
        r = parse_american(v.get(key))
        if r is not None:
            return r
    return None


def total_val(item, phase):
    try:
        v = item[phase]["total"]
    except (KeyError, TypeError):
        return None
    if not isinstance(v, dict):
        return None
    for key in ("american", "alternateDisplayValue", "displayValue"):
        r = parse_american(v.get(key))
        if r is not None:
            return r
    return parse_american(v.get("value"))


def total_price(item, phase, side):
    try:
        v = item[phase][side]
    except (KeyError, TypeError):
        return None
    if not isinstance(v, dict):
        return None
    for key in ("american", "alternateDisplayValue"):
        r = parse_american(v.get(key))
        if r is not None:
            return r
    return None


COLS = [
    "season", "game_date", "game_date_et", "start_time_utc", "event_id", "season_type", "away_team", "home_team",
    "away_abbr", "home_abbr", "away_score", "home_score", "completed",
    "provider_id", "provider_name", "details",
    "open_home_spread", "close_home_spread", "current_home_spread", "final_spread",
    "open_away_spread", "close_away_spread", "current_away_spread",
    "open_home_spread_odds", "close_home_spread_odds", "current_home_spread_odds",
    "open_away_spread_odds", "close_away_spread_odds", "current_away_spread_odds",
    "open_home_ml", "close_home_ml", "current_home_ml", "final_home_ml",
    "open_away_ml", "close_away_ml", "current_away_ml", "final_away_ml",
    "open_total", "close_total", "current_total", "final_total",
    "open_over_odds", "close_over_odds", "open_under_odds", "close_under_odds",
]


def main():
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[1:] or list(SEASONS)
    for season in only:
        months = SEASONS[season]
        events = {}
        for (y, m) in months:
            d1 = f"{y}{m:02d}01"
            d2 = f"{y}{m:02d}{month_days(y, m)}"
            sb = get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
                     f"?dates={d1}-{d2}&limit=1000")
            time.sleep(SLEEP)
            if not sb:
                print(f"  [{season}] {y}-{m:02d} scoreboard FAILED", flush=True)
                continue
            for e in sb.get("events", []):
                if e["date"][:7] != f"{y}-{m:02d}":
                    continue
                comp = e["competitions"][0]
                cs = {c.get("homeAway"): c for c in comp.get("competitors", [])}
                st = (e.get("season") or {}).get("slug") or comp.get("type", {}).get("abbreviation")
                _u = dt.datetime.strptime(e["date"], "%Y-%m-%dT%H:%MZ").replace(
                    tzinfo=dt.timezone.utc)
                events[e["id"]] = dict(
                    game_date=e["date"][:10],
                    game_date_et=_u.astimezone(ET).date().isoformat(),
                    start_time_utc=e["date"],
                    event_id=e["id"], season_type=st,
                    away_team=(cs.get("away", {}).get("team", {}) or {}).get("displayName"),
                    home_team=(cs.get("home", {}).get("team", {}) or {}).get("displayName"),
                    away_abbr=(cs.get("away", {}).get("team", {}) or {}).get("abbreviation"),
                    home_abbr=(cs.get("home", {}).get("team", {}) or {}).get("abbreviation"),
                    away_score=cs.get("away", {}).get("score"),
                    home_score=cs.get("home", {}).get("score"),
                    completed=comp.get("status", {}).get("type", {}).get("completed"),
                )
            print(f"  [{season}] {y}-{m:02d}: {len(sb.get('events', []))} events "
                  f"(cum {len(events)})", flush=True)

        raw_p = os.path.join(OUT, f"espn_nba_odds_raw_{season}.jsonl")
        csv_p = os.path.join(OUT, f"espn_nba_open_close_{season}.csv")
        nrow = 0
        n_ev_odds = 0
        with open(raw_p, "w") as rf, open(csv_p, "w", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=COLS)
            w.writeheader()
            for i, (eid, meta) in enumerate(sorted(events.items(), key=lambda kv: kv[1]["game_date"])):
                od = get(f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/"
                         f"events/{eid}/competitions/{eid}/odds?limit=50")
                time.sleep(SLEEP)
                items = (od or {}).get("items", [])
                rf.write(json.dumps({"season": season, **meta, "odds_items": items}) + "\n")
                if items:
                    n_ev_odds += 1
                for it in items:
                    ho = it.get("homeTeamOdds") or {}
                    ao = it.get("awayTeamOdds") or {}
                    row = dict(season=season, **meta)
                    row["provider_id"] = (it.get("provider") or {}).get("id")
                    row["provider_name"] = (it.get("provider") or {}).get("name")
                    row["details"] = it.get("details")
                    row["final_spread"] = it.get("spread")
                    row["final_total"] = it.get("overUnder")
                    row["final_home_ml"] = ho.get("moneyLine")
                    row["final_away_ml"] = ao.get("moneyLine")
                    for ph in ("open", "close", "current"):
                        row[f"{ph}_home_spread"] = point_spread(ho, ph)
                        row[f"{ph}_away_spread"] = point_spread(ao, ph)
                        row[f"{ph}_home_spread_odds"] = price(ho, ph, "spread")
                        row[f"{ph}_away_spread_odds"] = price(ao, ph, "spread")
                        row[f"{ph}_home_ml"] = price(ho, ph, "moneyLine")
                        row[f"{ph}_away_ml"] = price(ao, ph, "moneyLine")
                        row[f"{ph}_total"] = total_val(it, ph)
                    for ph in ("open", "close"):
                        row[f"{ph}_over_odds"] = total_price(it, ph, "over")
                        row[f"{ph}_under_odds"] = total_price(it, ph, "under")
                    w.writerow({k: row.get(k) for k in COLS})
                    nrow += 1
                if (i + 1) % 100 == 0:
                    print(f"  [{season}] odds {i+1}/{len(events)} events, {nrow} rows", flush=True)
                    cf.flush()
                    rf.flush()
        print(f"[{season}] DONE events={len(events)} events_with_odds={n_ev_odds} rows={nrow}\n"
              f"          {csv_p}\n          {raw_p}", flush=True)


if __name__ == "__main__":
    main()
