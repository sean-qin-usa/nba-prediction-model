#!/usr/bin/env python3
"""Build a CONVENTION-CONSISTENT odds table across as many seasons as the held
data allows, with per-game provenance.

WHY. D248 located three recording regimes in `ats19_frame_offset`, not one:

    2007-08 .. 2022-23   SBR composite      ~50% integer lines, ~10% on 3/7
    2023-24              ESPN, 15-book mean ~43% integer, 2.8% quarter-points
    2024-25 .. 2025-26   ESPN BET alone     ~1% integer, 99% half-points

The third regime is not a market change. ESPN BET posts 100% half-point spreads
in EVERY season it appears -- verified directly in the raw feed -- and in
2024-25 it was the only book carrying opens, so the frame inherited its house
convention wholesale. The consequence is mechanical: a half-points-only grid is
spaced 1.0 apart rather than 0.5, so the smallest observable move DOUBLES. That
inflates mean |close-open| and, proportionally, CLV. It is the whole of D247's
"2024-25 read the market best" result.

THE FIX. TeamRankings book1 holds the historical convention:

    season    TR book1 integer share    TR key-number rate
    2021-22            52.3%                  10.59%
    2022-23            51.0%                   9.22%
    2023-24            50.9%                   9.74%
    2024-25            49.7%                   8.67%
    2025-26             0.0%                   3.36%     <- breaks

and it OVERLAPS the SBR era on 2021-22 and 2022-23, so the join can be verified
rather than assumed. That yields one consistent convention for 2007-08..2024-25.

2025-26 CANNOT BE UNIFIED and is not forced. By that season every book in every
held source posts half-points only. That is an industry shift, not a collection
failure, and pretending otherwise would hide the one thing a reader most needs
to know. It is emitted with `convention_break=True` and every cross-season
comparison must either exclude it or say that it did not.

Output: data/odds_unified.csv.gz, one row per game, with `feed_src` naming the
source of every value. No silent fallbacks -- a game with no consistent price
is emitted with nulls and counted, never quietly filled from another regime.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

SE2SEASON = {2022: "2021-22", 2023: "2022-23", 2024: "2023-24",
             2025: "2024-25", 2026: "2025-26"}
TR_ABBR = {"BK": "BKN", "GS": "GSW", "NO": "NOP", "NY": "NYK",
           "PHO": "PHX", "SA": "SAS"}
SBR_LAST = "2022-23"          # SBR's hard stop is 2023-01-16
TR_SEASONS = ("2023-24", "2024-25")
BREAK_SEASONS = ("2025-26",)


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def gran(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if not len(v):
        return dict(n=0)
    fr = np.abs(v - np.round(v))
    return dict(n=len(v),
                pct_int=100 * float((fr < 1e-9).mean()),
                pct_key=100 * float(np.isin(np.abs(v), [3.0, 7.0]).mean()))


def load_tr():
    rows = []
    with open(ROOT / "data" / "raw" / "teamrankings"
              / "spread_movement.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("no_data"):
                continue
            s = SE2SEASON.get(r.get("season_end"))
            if not s:
                continue
            ob = r.get("open_books") or []
            cb = r.get("current_books") or []
            fav = r.get("fav_team")
            rows.append(dict(
                game_id=str(r["game_id"]).zfill(10), season_tr=s,
                fav=TR_ABBR.get(fav, fav),
                b1_open=ob[0] if len(ob) > 0 and ob[0] is not None else None,
                b1_close=cb[0] if len(cb) > 0 and cb[0] is not None else None,
                cons_open=r.get("fav_open"), cons_close=r.get("fav_last")))
    return pd.DataFrame(rows)


def main():
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
    f["game_id"] = zf(f["game_id"])
    t = load_tr()
    print(f"frame {len(f):,} games | TeamRankings {len(t):,} rows")

    m = f.merge(t, on="game_id", how="left")
    known = m.fav.notna()
    bad = known & ~((m.fav == m.home) | (m.fav == m.away))
    assert bad.sum() == 0, (
        f"{bad.sum()} rows whose fav_team matches neither side — the sign of "
        f"those lines would be a coin flip: {sorted(m.loc[bad,'fav'].unique())}")
    ishome = (m.fav == m.home)
    for src, oc, cc in (("b1", "b1_open", "b1_close"),
                        ("cons", "cons_open", "cons_close")):
        m[f"tr_{src}_open"] = np.where(ishome, -m[oc], m[oc])
        m[f"tr_{src}_close"] = np.where(ishome, -m[cc], m[cc])
        m.loc[m[oc].isna(), f"tr_{src}_open"] = np.nan
        m.loc[m[cc].isna(), f"tr_{src}_close"] = np.nan

    # ---- OVERLAP VALIDATION: does TR book1 reproduce SBR where both exist? --
    print("\n=== OVERLAP: TR book1 vs the SBR-era frame (2021-22, 2022-23) ===")
    ov = m[m.season.isin(["2021-22", "2022-23"])].dropna(
        subset=["tr_b1_open", "open_margin"])
    for s, g in ov.groupby("season"):
        d = g.tr_b1_open - g.open_margin
        print(f"  {s}: n={len(g):5}  identical {100*(d.abs()<1e-9).mean():5.1f}%"
              f"  within 0.5 {100*(d.abs()<=0.5).mean():5.1f}%"
              f"  bias {d.mean():+.3f}  sd {d.std():.3f}"
              f"  corr {np.corrcoef(g.tr_b1_open, g.open_margin)[0,1]:.4f}")
    print("  Two books quoting the same game agree to within a half point most")
    print("  of the time. A low agreement rate here would mean the splice is")
    print("  NOT safe and the unified feed should not be built.")

    # ---- ASSEMBLE ---------------------------------------------------------
    m["u_open"] = np.nan
    m["u_close"] = np.nan
    m["feed_src"] = "none"
    sbr = m.season <= SBR_LAST
    m.loc[sbr, "u_open"] = m.loc[sbr, "open_margin"]
    m.loc[sbr, "u_close"] = m.loc[sbr, "close_margin"]
    m.loc[sbr, "feed_src"] = "sbr_composite"
    for s in TR_SEASONS + BREAK_SEASONS:
        sel = (m.season == s) & m.tr_b1_open.notna() & m.tr_b1_close.notna()
        m.loc[sel, "u_open"] = m.loc[sel, "tr_b1_open"]
        m.loc[sel, "u_close"] = m.loc[sel, "tr_b1_close"]
        m.loc[sel, "feed_src"] = "tr_book1"
    m["convention_break"] = m.season.isin(BREAK_SEASONS)

    print("\n=== COVERAGE AND GRANULARITY OF THE UNIFIED FEED ===")
    print(f"  {'season':9} {'n':>5} {'have':>6} {'src':>14} "
          f"{'int%':>6} {'key%':>6}  {'was int%':>9} {'was key%':>9}")
    rows = []
    for s, g in m.groupby("season"):
        have = g.u_open.notna()
        gu, go = gran(g.u_open[have]), gran(g.open_margin)
        src = g.loc[have, "feed_src"].mode()
        src = src.iloc[0] if len(src) else "none"
        flag = "  <- CONVENTION BREAK" if s in BREAK_SEASONS else ""
        print(f"  {s:9} {len(g):5} {100*have.mean():5.1f}% {src:>14} "
              f"{gu.get('pct_int',float('nan')):6.1f} "
              f"{gu.get('pct_key',float('nan')):6.2f}  "
              f"{go['pct_int']:9.1f} {go['pct_key']:9.2f}{flag}")
        rows.append(dict(season=s, n=len(g), have=float(have.mean()), src=src,
                         u_int=gu.get("pct_int"), u_key=gu.get("pct_key"),
                         old_int=go["pct_int"], old_key=go["pct_key"]))

    out = m[["game_id", "season", "game_date", "home", "away",
             "u_open", "u_close", "feed_src", "convention_break",
             "open_margin", "close_margin", "margin_actual",
             "m_us_blind", "m_us", "y"]].copy()
    out.to_csv(ROOT / "data" / "odds_unified.csv.gz", index=False,
               compression="gzip")
    json.dump(rows, open(ROOT / "data" / "odds_unified_coverage.json", "w"),
              default=float)
    n_ok = int(out.u_open.notna().sum())
    print(f"\n  wrote data/odds_unified.csv.gz — {n_ok:,}/{len(out):,} games "
          f"priced ({100*n_ok/len(out):.1f}%)")
    print(f"  consistent-convention span 2007-08..2024-25: "
          f"{int(out[~out.convention_break].u_open.notna().sum()):,} games")


if __name__ == "__main__":
    main()
