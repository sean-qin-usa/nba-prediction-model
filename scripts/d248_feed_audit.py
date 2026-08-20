#!/usr/bin/env python3
"""D248 — FEED AUDIT: find every cross-season discontinuity in the odds data,
then build the most consistent feed the held data allows.

D247 established that 2024-25's rank-1 CLV tracks the RECORDED line's travel
distance (corr +0.917) rather than the market. That was one discontinuity found
by following one anomaly. This asks the general question: WHERE ELSE does the
feed change, and how much of what the register reads as market history is
actually feed history?

The frame is stitched (docs/OPENING_LINES.md):

    SBR composite            2007-08 .. 2022-23     one composite book
    ESPN + Action Network    2023-24 .. 2025-26     15 books -> 2 -> 4

so there are at least TWO joins, not one: the source splice at 2022-23/2023-24,
and a within-ESPN collapse in book count at 2023-24/2024-25.

PART 1 measures fingerprints that are properties of the RECORDING, not of the
market, and locates changepoints in each by exhaustive split with a permutation
null over season orderings. Granularity and key-number clustering are the
sharpest: a single book posts -3 and -7 far more often than a consensus, which
smooths them, and no market mechanism changes the RATE OF HALF-POINT USE.

PART 2 uses TeamRankings, which covers 2021-22 .. 2025-26 and therefore OBSERVES
BOTH SIDES OF THE SPLICE, to measure the offset between sources directly instead
of assuming it.

PART 3 rebuilds the five modern seasons from TeamRankings alone -- one source,
one book -- and re-runs D247's outlier test. If 2024-25's CLV advantage
survives a consistent feed it was a market fact after all; if it collapses, the
register's cross-season CLV history has to be rewritten.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from scipy import stats                                           # noqa: E402

SE2SEASON = {2022: "2021-22", 2023: "2022-23", 2024: "2023-24",
             2025: "2024-25", 2026: "2025-26"}

# TeamRankings uses its own abbreviations. Six differ from ours, and because
# `fav_team` decides the SIGN of the spread, an unmapped abbreviation silently
# flips that game's line instead of dropping it -- the failure mode that looks
# like two feeds disagreeing by 5 points when they actually agree.
TR_ABBR = {"BK": "BKN", "GS": "GSW", "NO": "NOP", "NY": "NYK",
           "PHO": "PHX", "SA": "SAS"}


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def changepoint(vals, seasons, n_perm=20000, seed=248):
    """Best single split of an ordered season series, with a permutation null
    over season ORDER. Answers: is any split better than shuffling explains?"""
    v = np.asarray(vals, float)
    k = len(v)
    if k < 6:
        return None
    def best(x):
        bs, bi = -np.inf, None
        for i in range(2, len(x) - 1):
            a, b = x[:i], x[i:]
            sp = abs(a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a)
                                                    + b.var(ddof=1) / len(b))
            if np.isfinite(sp) and sp > bs:
                bs, bi = sp, i
        return bs, bi
    obs, idx = best(v)
    rng = np.random.default_rng(seed)
    null = np.array([best(rng.permutation(v))[0] for _ in range(n_perm)])
    p = float((null >= obs).mean())
    return dict(stat=float(obs), split_after=seasons[idx - 1],
                split_before=seasons[idx], p=p,
                pre=float(v[:idx].mean()), post=float(v[idx:].mean()))


def main():
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
    f["game_id"] = zf(f["game_id"])
    f = f.dropna(subset=["open_margin", "close_margin", "margin_actual"]).copy()
    seasons = sorted(f.season.unique())

    # ================================================================
    # PART 1 — FEED FINGERPRINTS
    # ================================================================
    print("=" * 92)
    print("PART 1  FEED FINGERPRINTS BY SEASON")
    print("=" * 92)
    rows = []
    for s, g in f.groupby("season"):
        o = g.open_margin.to_numpy(float)
        c = g.close_margin.to_numpy(float)
        mv = c - o
        # granularity: how are line values quantised?
        frac_o = np.abs(o - np.round(o))
        rows.append(dict(
            season=s, n=len(g),
            abs_move=float(np.abs(mv).mean()),
            pct_unmoved=float(100 * (np.abs(mv) < 1e-9).mean()),
            sd_open=float(o.std(ddof=1)),
            # granularity
            pct_int_open=float(100 * (frac_o < 1e-9).mean()),
            pct_half_open=float(100 * (np.abs(frac_o - 0.5) < 1e-9).mean()),
            pct_other_open=float(100 * ((frac_o > 1e-9)
                                        & (np.abs(frac_o - .5) > 1e-9)).mean()),
            # key numbers: a single book clusters on 3 and 7, a consensus does not
            pct_key=float(100 * np.isin(np.abs(o), [3.0, 7.0]).mean()),
            n_distinct_open=int(len(np.unique(o))),
            pct_total=float(100 * g.open_total.notna().mean())
            if "open_total" in g else np.nan,
        ))
    d = pd.DataFrame(rows)
    show = ["season", "n", "abs_move", "pct_unmoved", "sd_open",
            "pct_int_open", "pct_half_open", "pct_other_open", "pct_key",
            "n_distinct_open"]
    print(d[show].to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    print("\n" + "-" * 92)
    print("  CHANGEPOINTS — best single split, p from 20,000 season-order shuffles")
    print("-" * 92)
    print(f"  {'fingerprint':18} {'split':>22} {'pre':>9} {'post':>9} "
          f"{'stat':>7} {'p':>8}")
    cps = {}
    for c in ("abs_move", "pct_unmoved", "sd_open", "pct_int_open",
              "pct_half_open", "pct_key", "n_distinct_open"):
        r = changepoint(d[c].to_numpy(), list(d.season))
        if r:
            cps[c] = r
            flag = "***" if r["p"] < 0.01 else ("*" if r["p"] < 0.05 else "")
            print(f"  {c:18} {r['split_after']}|{r['split_before']:>9} "
                  f"{r['pre']:9.3f} {r['post']:9.3f} {r['stat']:7.2f} "
                  f"{r['p']:8.4f} {flag}")

    print("\n  A market can change how far lines move. It cannot change the RATE")
    print("  OF HALF-POINT USE or how many distinct line values exist. Those are")
    print("  recording conventions, so a changepoint there is a FEED change.")

    # ================================================================
    # PART 2 — TEAMRANKINGS vs THE FRAME, IN THE OVERLAP
    # ================================================================
    print("\n" + "=" * 92)
    print("PART 2  TEAMRANKINGS AS AN INDEPENDENT OBSERVER OF BOTH ERAS")
    print("=" * 92)
    tr = []
    with open(ROOT / "data" / "raw" / "teamrankings" / "spread_movement.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("no_data") or r.get("fav_open") is None:
                continue
            ob = r.get("open_books") or []
            cb = r.get("current_books") or []
            fav = r.get("fav_team")
            tr.append(dict(
                game_id=str(r["game_id"]).zfill(10),
                season=SE2SEASON.get(r.get("season_end")),
                fav=TR_ABBR.get(fav, fav),
                fav_open=r.get("fav_open"), fav_last=r.get("fav_last"),
                b1_open=ob[0] if len(ob) > 0 else None,
                b1_last=cb[0] if len(cb) > 0 else None,
                n_history=r.get("n_history") or 0))
    t = pd.DataFrame(tr).dropna(subset=["season"])
    print(f"  TeamRankings: {len(t):,} games with an open, "
          f"{t.season.nunique()} seasons")
    print(f"  book1 open present on {100*t.b1_open.notna().mean():.1f}%")

    m = f.merge(t, on="game_id", how="inner", suffixes=("", "_tr"))
    print(f"  joined to the frame on game_id: {len(m):,} games")
    if len(m):
        # every joined row must have an identifiable favourite, or the sign of
        # that game's line is a coin flip
        bad = ~((m.fav == m.home) | (m.fav == m.away))
        assert bad.mean() < 0.01, (
            f"{100*bad.mean():.1f}% of rows have fav_team matching neither "
            f"side: {sorted(m.loc[bad, 'fav'].unique())[:10]}")
        m = m[~bad].copy()
        # TR sign -> home margin
        ishome = m.fav == m.home
        m["tr_open_margin"] = np.where(ishome, -m.fav_open, m.fav_open)
        m["tr_close_margin"] = np.where(ishome, -m.fav_last, m.fav_last)
        m["d_open"] = m.tr_open_margin - m.open_margin
        m["d_close"] = m.tr_close_margin - m.close_margin
        agg = m.groupby("season").agg(
            n=("d_open", "size"),
            open_agree=("d_open", lambda x: 100 * (x.abs() < 1e-9).mean()),
            open_bias=("d_open", "mean"), open_sd=("d_open", "std"),
            close_agree=("d_close", lambda x: 100 * (x.abs() < 1e-9).mean()),
            close_bias=("d_close", "mean"), close_sd=("d_close", "std"))
        print("\n  TR minus FRAME, by season "
              "(agree% = identical to the 0.01 pt):")
        print(agg.to_string(float_format=lambda v: f"{v:9.3f}"))
        print("\n  A source splice shows up as a JUMP IN agree% or sd across")
        print("  2022-23|2023-24. TR itself does not change, so any jump is the")
        print("  frame's source changing under it.")

    # ================================================================
    # PART 3 — REBUILD ON ONE SOURCE AND RE-RUN THE D247 TEST
    # ================================================================
    print("\n" + "=" * 92)
    print("PART 3  THE SAME SEASONS, MEASURED ON ONE CONSISTENT FEED")
    print("=" * 92)
    if len(m):
        out = []
        for s, g in m.groupby("season"):
            for lab, o, c in (("FRAME (stitched)", g.open_margin,
                               g.close_margin),
                              ("TR consensus", g.tr_open_margin,
                               g.tr_close_margin)):
                o = o.to_numpy(float); c = c.to_numpy(float)
                ok = np.isfinite(o) & np.isfinite(c)
                o, c = o[ok], c[ok]
                sd = np.sign(g.m_us.to_numpy(float)[ok] - o)
                mv = c - o
                amv = np.abs(mv).mean()
                out.append(dict(season=s, feed=lab, n=int(ok.sum()),
                                abs_move=float(amv),
                                clv=float((sd * mv).mean()),
                                capture=float((sd * mv).mean() / amv)
                                if amv > 0 else np.nan))
        o = pd.DataFrame(out)
        piv = o.pivot(index="season", columns="feed",
                      values=["abs_move", "clv", "capture"])
        print(piv.to_string(float_format=lambda v: f"{v:9.4f}"))

        print("\n  2024-25 rank within these five seasons:")
        for feed in ("FRAME (stitched)", "TR consensus"):
            sub = o[o.feed == feed].set_index("season")
            for c in ("clv", "capture", "abs_move"):
                v = sub[c]
                if "2024-25" not in v.index:
                    continue
                rank = int((v > v["2024-25"]).sum()) + 1
                z = (v["2024-25"] - v.mean()) / v.std(ddof=1)
                print(f"    {feed:18} {c:9} {v['2024-25']:+8.4f}  "
                      f"rank {rank}/{len(v)}  z {z:+5.2f}")
        json.dump({"fingerprints": rows, "changepoints": cps,
                   "feed_compare": out},
                  open(ROOT / "data" / "d248_feed_audit.json", "w"), default=float)
        print("\nwrote data/d248_feed_audit.json")


if __name__ == "__main__":
    main()
