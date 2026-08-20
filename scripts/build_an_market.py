#!/usr/bin/env python3
"""Extract the Action Network market panel that has been on disk, unused, since
August: per-book moneyline, totals, closing spread, and the public bet split.

D249 established these exist at 95.9-99.8% coverage for 2023-24..2025-26 and
that `docs/FAILURES.md` wrongly recorded post-2023 moneyline as DATA_BLOCKED --
an entry written after checking two DuckDB tables rather than the files on disk.

WHAT THIS IS AND IS NOT. Action Network stores ONE snapshot per book per game,
which is the CLOSING state. There is no opening price and no line history here,
so this cannot be used for CLV, timing, or anything requiring a pre-close price.
It is a closing-market panel. Treating its spread as an opener would be the same
class of error D248 caught.

BOOK IDS ARE LEFT OPAQUE. The raw carries numeric `book_id` with no name mapping
in the payload, so they are emitted as ids. Books 15, 68 and 69 are the three
present at >=95.6% in all three seasons (D248) and are the defensible basket for
anything cross-season.

THE BET SPLIT. Each market entry carries `bet_info` with `money.percent` and
`tickets.percent`. money% >> tickets% means a few large wagers on that side;
tickets% >> money% means many small ones. The absolute `value` fields are 0
throughout, so only proportions are available, never handle.

  CROSS-SEASON WARNING, MEASURED: mean |money% - tickets%| is 15.67 in 2023-24
  against 4.38 and 4.85 in the two later seasons. That is a 3x shift with no
  plausible market cause, so it carries the same suspicion D248 attached to CLV.
  Anything pooled across these seasons must show that it survives the split.

JOIN. AN timestamps are UTC, so a 02:00Z tip is the previous evening in the US.
Matching is on (home, away) with the game date allowed to differ by one day, and
every joined row is asserted unique.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

SEASONS = ("2023-24", "2024-25", "2025-26")
CORE_BOOKS = ("15", "68", "69")          # >=95.6% in all three seasons (D248)
RAW = ROOT / "data" / "raw" / "sbr_ext"


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def side_of(entries, want):
    for e in entries or []:
        if e.get("side") == want and not e.get("is_live"):
            return e
    return None


def pull(e):
    if not e:
        return {}
    bi = e.get("bet_info") or {}
    mo, ti = (bi.get("money") or {}), (bi.get("tickets") or {})
    return dict(value=e.get("value"), odds=e.get("odds"),
                money_pct=mo.get("percent"), tickets_pct=ti.get("percent"))


def main():
    rows = []
    for season in SEASONS:
        p = RAW / f"an_nba_odds_raw_{season}.jsonl"
        if not p.exists():
            print(f"  missing {p.name}"); continue
        n = 0
        with open(p) as fh:
            for line in fh:
                for g in (json.loads(line).get("games") or []):
                    if g.get("type") != "reg":
                        continue
                    tm = {t["id"]: t.get("abbr") for t in (g.get("teams") or [])}
                    home = tm.get(g.get("home_team_id"))
                    away = tm.get(g.get("away_team_id"))
                    if not home or not away:
                        continue
                    n += 1
                    ts = pd.to_datetime(g.get("start_time"), utc=True,
                                        errors="coerce")
                    mk = g.get("markets") or {}
                    for bid, m in (mk.items() if isinstance(mk, dict) else []):
                        ev = (m or {}).get("event") or {}
                        sp = pull(side_of(ev.get("spread"), "home"))
                        ml_h = pull(side_of(ev.get("moneyline"), "home"))
                        ml_a = pull(side_of(ev.get("moneyline"), "away"))
                        to = pull(side_of(ev.get("total"), "over"))
                        if not (sp or ml_h or to):
                            continue
                        rows.append(dict(
                            season=season, an_ts=ts, home=home, away=away,
                            book_id=str(bid), num_bets=g.get("num_bets"),
                            spread_home=sp.get("value"),
                            spread_odds=sp.get("odds"),
                            spread_money_pct=sp.get("money_pct"),
                            spread_tickets_pct=sp.get("tickets_pct"),
                            ml_home=ml_h.get("odds"), ml_away=ml_a.get("odds"),
                            ml_money_pct=ml_h.get("money_pct"),
                            ml_tickets_pct=ml_h.get("tickets_pct"),
                            total=to.get("value"), total_odds=to.get("odds"),
                            total_money_pct=to.get("money_pct"),
                            total_tickets_pct=to.get("tickets_pct")))
        print(f"  {season}: {n:,} regular-season games parsed")
    d = pd.DataFrame(rows)
    print(f"\nper-book rows: {len(d):,}  books {sorted(d.book_id.unique())}")

    # ---- join to our game ids -----------------------------------------
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
    f["game_id"] = zf(f["game_id"])
    f["game_date"] = pd.to_datetime(f["game_date"])
    f = f[f.season.isin(SEASONS)][["game_id", "season", "game_date",
                                   "home", "away"]]
    d["an_date"] = d.an_ts.dt.tz_convert(None).dt.normalize()
    best = None
    for off in (0, -1, 1):
        cand = d.copy()
        cand["game_date"] = cand.an_date + pd.Timedelta(days=off)
        j = cand.merge(f, on=["season", "game_date", "home", "away"],
                       how="inner")
        j["_off"] = off
        best = j if best is None else pd.concat([best, j])
    # keep one row per (game, book): the smallest |offset| that matched
    best["_a"] = best._off.abs()
    best = (best.sort_values("_a")
                .drop_duplicates(["game_id", "book_id"], keep="first")
                .drop(columns=["_a", "_off", "an_date"]))
    matched = best.game_id.nunique()
    total = f.game_id.nunique()
    print(f"joined {matched:,}/{total:,} games ({100*matched/total:.1f}%), "
          f"{len(best):,} game-book rows")
    assert not best.duplicated(["game_id", "book_id"]).any(), "duplicate rows"

    print("\ncoverage by season (share of frame games with >=1 book):")
    for s, g in best.groupby("season"):
        tot = f[f.season == s].game_id.nunique()
        core = g[g.book_id.isin(CORE_BOOKS)]
        print(f"  {s}: any book {100*g.game_id.nunique()/tot:5.1f}%   "
              f"core basket {100*core.game_id.nunique()/tot:5.1f}%   "
              f"ML {100*g.dropna(subset=['ml_home']).game_id.nunique()/tot:5.1f}%   "
              f"total {100*g.dropna(subset=['total']).game_id.nunique()/tot:5.1f}%   "
              f"bet-split {100*g.dropna(subset=['spread_money_pct']).game_id.nunique()/tot:5.1f}%")

    best = best.sort_values(["game_date", "game_id", "book_id"])
    best.to_csv(ROOT / "data" / "an_market.csv.gz", index=False,
                compression="gzip")
    print(f"\nwrote data/an_market.csv.gz  ({len(best):,} rows)")

    # ---- the cross-season warning, recomputed on the joined panel ------
    core = best[best.book_id.isin(CORE_BOOKS)].dropna(
        subset=["spread_money_pct", "spread_tickets_pct"])
    core = core.assign(div=core.spread_money_pct - core.spread_tickets_pct)
    print("\nmoney% - tickets% on the core basket, BY SEASON:")
    print(core.groupby("season").div.agg(
        n="size", mean="mean", mean_abs=lambda s: s.abs().mean(),
        sd="std").to_string(float_format=lambda v: f"{v:9.3f}"))
    print("  A 3x shift in mean_abs across seasons is a feed property until")
    print("  proven otherwise. Do not pool without showing it survives split.")


if __name__ == "__main__":
    main()
