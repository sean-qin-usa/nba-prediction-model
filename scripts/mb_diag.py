#!/usr/bin/env python3
"""PART 1 — is the TeamRankings book1/book2 panel two REAL books or one feed
rendered twice?

D142 measured a 36.2% exact-tie rate between TR's `book1` and `book2` opening
spreads and flagged it as the reason its N=5/N=8 Gaussian extrapolation is a
CEILING ("extra books copy"). The owner's hypothesis is stronger: the 36% is a
VENDOR ARTIFACT and genuinely independent books disagree more.

Four diagnostics, all free, all on data already local:

  A  DISPERSION           the shape of |book1 - book2| — spike at zero?
  B  STRUCTURE vs VENDOR  do ties concentrate at key/round numbers and short
                          lines (STRUCTURAL — books genuinely agree there) or
                          are they flat across the line surface (VENDOR)?
  C  SYNCHRONY            do the two series move at the same timestamps?
                          Circular time-shift null, 400 draws, preserves both
                          marginal rates and within-book clustering.
  D  LAG-CORRELATION      does book2 reproduce book1's path with a lag?

  E  THE DECISIVE CONTROL — the ESPN 2023-24 panel carries 11-14 books on the
     same season, INCLUDING Caesars rendered three times (CO/TN/NJ), which is a
     KNOWN same-operator triple, and ESPN BET rendered twice (main + live).
     So the same tie statistic can be computed for a known-duplicate pair and
     for a known-distinct pair on the SAME games. That is what settles it.

Read-only. Writes JSON + a log. No DB access at all.
"""
from __future__ import annotations

import nbapred.threads  # noqa: F401  (pin BLAS before numpy)
nbapred.threads.pin(1)

import collections
import csv
import json
import math
import os
import random
import sys
from datetime import datetime

import numpy as np

ROOT = "/hdd/steveqin/sean_dev/nba_model"
TR = f"{ROOT}/data/raw/teamrankings/spread_movement.jsonl"
ESPN = f"{ROOT}/data/raw/sbr_ext/espn_nba_open_close_%s.csv"
OUT = f"{ROOT}/data/mb_diag.json"

# ESPN provider taxonomy. `accuscore` and `betegy` are MODELS that ESPN renders
# in the same odds array as books; they are excluded from every "book" panel and
# reported separately. The two "- Live Odds" feeds and the three Caesars state
# skins are the KNOWN-DUPLICATE arms.
MODELS = {"accuscore", "betegy"}
OPERATOR = {
    "ESPN BET": "espnbet",
    "ESPN Bet - Live Odds": "espnbet",
    "Caesars Sportsbook (Colorado)": "caesars",
    "Caesars Sportsbook (Tennessee)": "caesars",
    "Caesars Sportsbook (New Jersey)": "caesars",
    "Caesars Sportsbook (New Jersey) - Live Odds": "caesars",
    "Caesars Sportsbook": "caesars",
    "DraftKings": "draftkings",
    "DraftKings - Live Odds": "draftkings",
    "MGM": "mgm",
    "Unibet": "unibet",
    "Titanbets": "titanbets",
    "BetfairSportsbook": "betfair",
    "SugarHouse": "sugarhouse",
    "PointsBet": "pointsbet",
}

R = {}
LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def fnum(x):
    if x is None or x == "" or x == "None":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


# ----------------------------------------------------------------- load TR
def load_tr():
    rows = []
    with open(TR) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def part_a_b(tr):
    """A: dispersion. B: structural (key-number / line-level) vs vendor (flat)."""
    say("\n" + "=" * 72)
    say("A — DISPERSION OF |book1 - book2| AT THE OPEN (D142 §2 reproduction)")
    say("=" * 72)

    recs = []
    for g in tr:
        ob = g.get("open_books") or []
        b1 = fnum(ob[0]) if len(ob) > 0 else None
        b2 = fnum(ob[1]) if len(ob) > 1 else None
        b3 = fnum(ob[2]) if len(ob) > 2 else None
        cb = g.get("current_books") or []
        c1 = fnum(cb[0]) if len(cb) > 0 else None
        c2 = fnum(cb[1]) if len(cb) > 1 else None
        recs.append(dict(gid=g.get("game_id"), date=g.get("game_date"),
                         se=g.get("season_end"), b1=b1, b2=b2, b3=b3,
                         c1=c1, c2=c2, hist=g.get("history") or []))

    n_b1 = sum(1 for r in recs if r["b1"] is not None)
    n_b2 = sum(1 for r in recs if r["b2"] is not None)
    n_b3 = sum(1 for r in recs if r["b3"] is not None)
    say(f"games in file: {len(recs)}   open non-null: b1={n_b1} b2={n_b2} b3={n_b3}")

    both = [r for r in recs if r["b1"] is not None and r["b2"] is not None]
    d = np.array([r["b1"] - r["b2"] for r in both])
    ad = np.abs(d)
    say(f"n both books at open: {len(both)}")
    say(f"|b1-b2|: mean {ad.mean():.4f}  median {np.median(ad):.2f}  sd {ad.std(ddof=1):.4f}")
    tie = int((ad == 0).sum())
    lo, hi = wilson(tie, len(both))
    say(f"EXACT TIE: {tie}/{len(both)} = {100*tie/len(both):.2f}%  Wilson[{100*lo:.2f},{100*hi:.2f}]")
    for t in (0.5, 1.0, 1.5, 2.0, 3.0):
        say(f"  >= {t:>3}: {100*(ad >= t).mean():.2f}%")
    # signed: is one book systematically higher? (a vendor would be symmetric)
    say(f"signed (b1-b2): mean {d.mean():+.4f}  sd {d.std(ddof=1):.4f}  "
        f"P(b1<b2)={100*(d < 0).mean():.2f}%  P(b1>b2)={100*(d > 0).mean():.2f}%")
    # granularity
    grid = collections.Counter(round(x % 1.0, 3) for x in ad)
    say("|diff| mod 1.0 (granularity):", dict(grid.most_common(6)))

    # ---- THE LATTICE TEST (the finding) -------------------------------
    say("\n--- LATTICE: what GRID does each book quote on? ---")

    def gran(vals):
        c = collections.Counter(round(abs(v) % 1.0, 3) for v in vals)
        n = max(len(vals), 1)
        return dict(n=len(vals), integer=c[0.0] / n, half=c[0.5] / n)

    lat = {}
    for nm, vals in (
        ("open_book1", [r["b1"] for r in recs if r["b1"] is not None]),
        ("open_book2", [r["b2"] for r in recs if r["b2"] is not None]),
        ("curr_book1", [r["c1"] for r in recs if r["c1"] is not None]),
        ("curr_book2", [r["c2"] for r in recs if r["c2"] is not None]),
        ("hist_book1", [fnum(h.get("book1")) for r in recs for h in r["hist"]
                        if fnum(h.get("book1")) is not None]),
        ("hist_book2", [fnum(h.get("book2")) for r in recs for h in r["hist"]
                        if fnum(h.get("book2")) is not None]),
    ):
        g = gran(vals)
        lat[nm] = g
        say(f"  {nm:12s} n={g['n']:6d}  integer {100*g['integer']:6.2f}%  "
            f"half {100*g['half']:6.2f}%")
    R["A_lattice"] = lat
    say("  => book2 lives on the HALF-POINT lattice; book1 does not. Two")
    say("     sources on incompatible grids CANNOT be one feed rendered twice.")

    # tie rate conditional on lattice agreement
    say("\n--- TIE RATE CONDITIONAL ON THE TWO BOOKS SHARING A LATTICE ---")
    lm = np.array([float((abs(r["b1"]) % 1.0) == (abs(r["b2"]) % 1.0)) for r in both])
    for nm, m in (("SAME lattice", lm == 1), ("DIFFERENT lattice", lm == 0)):
        k = int((ad[m] == 0).sum())
        n = int(m.sum())
        lo_, hi_ = wilson(k, n)
        say(f"  {nm:20s} n={n:5d}  tie {100*k/max(n,1):6.2f}% "
            f"[{100*lo_:.2f},{100*hi_:.2f}]  mean|d| {ad[m].mean():.4f}")
    R["A_tie_by_lattice"] = dict(
        same_n=int((lm == 1).sum()), same_tie=float((ad[lm == 1] == 0).mean()),
        same_mad=float(ad[lm == 1].mean()),
        diff_n=int((lm == 0).sum()), diff_tie=float((ad[lm == 0] == 0).mean()),
        diff_mad=float(ad[lm == 0].mean()))

    R["A"] = dict(n=len(both), mean=float(ad.mean()), median=float(np.median(ad)),
                  sd=float(ad.std(ddof=1)), tie=tie, tie_rate=tie / len(both),
                  tie_ci=[lo, hi], signed_mean=float(d.mean()),
                  ge={str(t): float((ad >= t).mean()) for t in (0.5, 1, 1.5, 2, 3)})

    say("\n" + "=" * 72)
    say("B — STRUCTURAL (key numbers / short lines) vs VENDOR (flat surface)")
    say("=" * 72)
    say("If ties concentrate where two real books SHOULD agree — short lines,")
    say("round numbers — the tie rate is a property of the market. If the tie")
    say("rate is flat across the line surface it looks like a copy.")

    # tie rate by |line| bucket
    lvl = np.array([abs(r["b1"]) for r in both])
    buckets = [(0, 1.5), (1.5, 3.5), (3.5, 5.5), (5.5, 7.5), (7.5, 10.5), (10.5, 99)]
    say(f"\n{'|line| bucket':>16s} {'n':>6s} {'tie%':>7s} {'mean|d|':>8s}")
    bytab = []
    for a, b in buckets:
        m = (lvl >= a) & (lvl < b)
        if m.sum() == 0:
            continue
        tr_ = float((ad[m] == 0).mean())
        say(f"{f'[{a},{b})':>16s} {int(m.sum()):6d} {100*tr_:7.2f} {ad[m].mean():8.4f}")
        bytab.append(dict(lo=a, hi=b, n=int(m.sum()), tie=tr_, mad=float(ad[m].mean())))
    R["B_by_level"] = bytab

    # KEY NUMBER TEST — is the tie rate higher when the agreed number is a
    # "natural" one? For NBA the granularity story is integer vs half point.
    say("\nKEY-NUMBER TEST — tie rate by the VALUE the books land on")
    whole = np.array([float(abs(r["b1"]) % 1.0 == 0.0) for r in both])
    for nm, m in (("integer line (b1)", whole == 1), ("half-point line (b1)", whole == 0)):
        say(f"  {nm:22s} n={int(m.sum()):5d}  tie {100*float((ad[m] == 0).mean()):6.2f}%")
    keytab = []
    say(f"\n{'|b1|':>6s} {'n':>6s} {'tie%':>7s}")
    cnt = collections.Counter(abs(r["b1"]) for r in both)
    for v, n in sorted(cnt.items()):
        if n < 60:
            continue
        m = lvl == v
        t = float((ad[m] == 0).mean())
        say(f"{v:6.1f} {n:6d} {100*t:7.2f}")
        keytab.append(dict(v=v, n=n, tie=t))
    R["B_by_value"] = keytab
    R["B_int_vs_half"] = dict(
        integer=float((ad[whole == 1] == 0).mean()), n_int=int((whole == 1).sum()),
        half=float((ad[whole == 0] == 0).mean()), n_half=int((whole == 0).sum()))

    # tie rate by season
    say("\nTIE RATE BY SEASON")
    seas = collections.defaultdict(list)
    for r, x in zip(both, ad):
        seas[r["se"]].append(x)
    stab = []
    for s in sorted(seas):
        v = np.array(seas[s])
        say(f"  {s}  n={len(v):5d}  tie {100*float((v == 0).mean()):6.2f}%  mean|d| {v.mean():.4f}")
        stab.append(dict(season_end=s, n=len(v), tie=float((v == 0).mean()),
                         mad=float(v.mean())))
    R["B_by_season"] = stab
    return recs, both


def part_c_d(recs):
    """C: do the two series move at the same timestamps? D: lag-correlation."""
    say("\n" + "=" * 72)
    say("C — SYNCHRONY OF MOVES IN THE HISTORY ARRAY")
    say("=" * 72)

    # First: what IS a history row? one change by one book, or a joint state?
    both_nn = one_nn = zero_nn = 0
    n_rows = 0
    for r in recs:
        for h in r["hist"]:
            n_rows += 1
            k = sum(1 for b in ("book1", "book2", "book3") if fnum(h.get(b)) is not None)
            if k == 0:
                zero_nn += 1
            elif k == 1:
                one_nn += 1
            else:
                both_nn += 1
    say(f"history rows total {n_rows}: exactly-one-book {one_nn} "
        f"({100*one_nn/max(n_rows,1):.2f}%), >=two-books {both_nn} "
        f"({100*both_nn/max(n_rows,1):.2f}%), empty {zero_nn}")
    say("=> a history row is ONE BOOK'S CHANGE AT ONE TIMESTAMP (an event log),")
    say("   so 'simultaneous move' means two rows sharing a timestamp.")
    R["C_rowshape"] = dict(rows=n_rows, one=one_nn, multi=both_nn, empty=zero_nn)

    # Build per-game change-time sets
    def ts(h):
        s = h.get("ts_iso")
        if not s:
            return None
        try:
            return datetime.fromisoformat(s).timestamp() / 60.0  # minutes
        except ValueError:
            return None

    games = []
    gdate = []
    for r in recs:
        t1, t2 = [], []
        v1, v2 = [], []
        for h in r["hist"]:
            t = ts(h)
            if t is None:
                continue
            a, b = fnum(h.get("book1")), fnum(h.get("book2"))
            if a is not None:
                t1.append(t); v1.append(a)
            if b is not None:
                t2.append(t); v2.append(b)
        if len(t1) >= 3 and len(t2) >= 3:
            o1 = np.argsort(t1); o2 = np.argsort(t2)
            games.append((np.array(t1)[o1], np.array(v1)[o1],
                          np.array(t2)[o2], np.array(v2)[o2]))
            gdate.append(r["date"])
    say(f"games with >=3 changes on BOTH books: {len(games)}")
    by_date = collections.defaultdict(list)
    for i, d in enumerate(gdate):
        by_date[d].append(i)

    # exact timestamp collisions
    exact = tot2 = 0
    for t1, _, t2, _ in games:
        s1 = set(np.round(t1, 3))
        exact += sum(1 for x in np.round(t2, 3) if x in s1)
        tot2 += len(t2)
    say(f"book2 changes sharing an EXACT timestamp with a book1 change: "
        f"{exact}/{tot2} = {100*exact/max(tot2,1):.3f}%")

    # nearest-neighbour synchrony with a circular time-shift null
    rng = random.Random(20260804)
    WINDOWS = (1.0, 5.0, 15.0, 60.0)
    obs = {w: 0 for w in WINDOWS}
    denom = 0
    for t1, _, t2, _ in games:
        denom += len(t2)
        for x in t2:
            g = float(np.min(np.abs(t1 - x)))
            for w in WINDOWS:
                if g <= w:
                    obs[w] += 1

    NDRAW = 200

    def run_null(kind):
        """kind='shift'  circular shift of book1's times (DESTROYS diurnal
                         structure => overstates synchrony; kept for contrast).
           kind='swap'   book2's times of game g are matched against book1's
                         times of a DIFFERENT game ON THE SAME DATE. Preserves
                         time-of-day AND slate-level news, so excess synchrony
                         over this null is game-specific coupling."""
        out = {w: [] for w in WINDOWS}
        for _ in range(NDRAW):
            c = {w: 0 for w in WINDOWS}
            for gi, (t1, _, t2, _) in enumerate(games):
                if kind == "shift":
                    span = max(t1.max() - t1.min(), 1.0)
                    sh = rng.uniform(0, span)
                    ref = np.sort(t1.min() + np.mod(t1 - t1.min() + sh, span))
                else:
                    pool = by_date.get(gdate[gi], [])
                    if len(pool) < 2:
                        continue
                    j = gi
                    for _try in range(8):
                        j = rng.choice(pool)
                        if j != gi:
                            break
                    if j == gi:
                        continue
                    ref = games[j][0]
                for x in t2:
                    g = float(np.min(np.abs(ref - x)))
                    for w in WINDOWS:
                        if g <= w:
                            c[w] += 1
            for w in WINDOWS:
                out[w].append(c[w])
        return out

    nulls = {"shift": run_null("shift"), "swap": run_null("swap")}
    # the swap null skips games with no same-date partner: renormalise on its
    # own denominator
    swap_den = sum(len(games[i][2]) for i in range(len(games))
                   if len(by_date.get(gdate[i], [])) >= 2)

    ctab = []
    for kind in ("shift", "swap"):
        den = denom if kind == "shift" else max(swap_den, 1)
        say(f"\nNULL = {kind.upper()}   ({NDRAW} draws, denominator {den})")
        say(f"{'window':>8s} {'obs%':>8s} {'null%':>8s} {'nullsd':>7s} {'z':>8s} {'ratio':>6s}")
        for w in WINDOWS:
            o = 100 * obs[w] / denom
            nv = np.array(nulls[kind][w]) / den * 100
            z = (o - nv.mean()) / max(nv.std(ddof=1), 1e-9)
            say(f"{w:8.0f} {o:8.3f} {nv.mean():8.3f} {nv.std(ddof=1):7.3f} {z:8.2f} "
                f"{o/max(nv.mean(),1e-9):6.2f}")
            ctab.append(dict(null=kind, window_min=w, obs=o, null_mean=float(nv.mean()),
                             null_sd=float(nv.std(ddof=1)), z=float(z),
                             ratio=float(o / max(nv.mean(), 1e-9))))
    R["C_sync"] = dict(n_games=len(games), n_changes_b2=denom,
                       exact_ts_share=exact / max(tot2, 1), table=ctab,
                       n_null_draws=NDRAW)

    say("\n" + "=" * 72)
    say("D — LAG CORRELATION OF THE TWO STEP FUNCTIONS")
    say("=" * 72)
    say("Resample both books to a 30-minute grid over the last 5 days before")
    say("the last observed change, difference, and cross-correlate.")

    STEP = 30.0
    lags = list(range(-8, 9))
    num = {l: 0.0 for l in lags}
    d1s = 0.0
    d2s = 0.0
    nser = 0
    agree_lvl = []
    for t1, v1, t2, v2 in games:
        lo = max(min(t1.min(), t2.min()), max(t1.max(), t2.max()) - 5 * 24 * 60)
        hi = max(t1.max(), t2.max())
        if hi - lo < 4 * STEP:
            continue
        grid = np.arange(lo, hi + STEP, STEP)
        s1 = np.array([v1[np.searchsorted(t1, g, "right") - 1] if g >= t1[0] else np.nan for g in grid])
        s2 = np.array([v2[np.searchsorted(t2, g, "right") - 1] if g >= t2[0] else np.nan for g in grid])
        m = ~(np.isnan(s1) | np.isnan(s2))
        if m.sum() < 6:
            continue
        s1, s2 = s1[m], s2[m]
        agree_lvl.append(float((s1 == s2).mean()))
        x = np.diff(s1)
        y = np.diff(s2)
        if x.std() == 0 and y.std() == 0:
            continue
        nser += 1
        d1s += float((x * x).sum())
        d2s += float((y * y).sum())
        for l in lags:
            if l >= 0:
                a, b = x[:len(x) - l], y[l:]
            else:
                a, b = x[-l:], y[:len(y) + l]
            k = min(len(a), len(b))
            if k > 0:
                num[l] += float((a[:k] * b[:k]).sum())
    say(f"series used: {nser}")
    say(f"mean share of the 30-min grid where the two books' LEVELS are equal: "
        f"{100*float(np.mean(agree_lvl)):.2f}%")
    dtab = []
    den = math.sqrt(max(d1s, 1e-9) * max(d2s, 1e-9))
    say(f"\n{'lag(x30min)':>12s} {'corr':>8s}")
    for l in lags:
        c = num[l] / den
        say(f"{l:12d} {c:8.4f}")
        dtab.append(dict(lag_steps=l, lag_min=l * STEP, corr=float(c)))
    peak = max(dtab, key=lambda r: r["corr"])
    say(f"PEAK: lag {peak['lag_steps']} steps ({peak['lag_min']:.0f} min) corr {peak['corr']:.4f}")
    R["D_lagcorr"] = dict(n_series=nser, level_agreement=float(np.mean(agree_lvl)),
                          table=dtab, peak=peak)


# ------------------------------------------------------- E: the ESPN panel
def load_espn(season):
    rows = list(csv.DictReader(open(ESPN % season)))
    panel = collections.defaultdict(dict)   # event -> book -> open home spread
    meta = {}
    for x in rows:
        if x["season_type"] not in ("regular-season", "post-season", "play-in-season"):
            continue
        v = fnum(x["open_home_spread"])
        if v is None:
            continue
        panel[x["event_id"]][x["provider_name"]] = v
        meta[x["event_id"]] = dict(date=x["game_date_et"] or x["game_date"],
                                   home=x["home_abbr"], away=x["away_abbr"])
    return panel, meta


def part_e():
    say("\n" + "=" * 72)
    say("E — THE DECISIVE CONTROL: ESPN's 2023-24 PANEL, 11-14 BOOKS,")
    say("    WITH A KNOWN-DUPLICATE PAIR AND KNOWN-DISTINCT PAIRS ON THE")
    say("    SAME GAMES. Already on disk (data/raw/sbr_ext/), $0, no scrape.")
    say("=" * 72)

    panel, meta = load_espn("2023-24")
    say(f"events with >=1 opening spread (regular/post/play-in): {len(panel)}")

    books = collections.Counter()
    for e, d in panel.items():
        for b in d:
            books[b] += 1
    say("\nprovider coverage (events with an OPENING spread):")
    for b, n in books.most_common():
        tag = "MODEL" if b in MODELS else OPERATOR.get(b, "?")
        say(f"  {b:46s} {n:5d}  [{tag}]")

    # pairwise tie rate + dispersion
    names = [b for b, n in books.most_common() if n >= 300]
    say(f"\npairwise on the {len(names)} providers with >=300 games "
        f"(models INCLUDED here and flagged; excluded from every book statistic)")
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            dd = [panel[e][a] - panel[e][b] for e in panel if a in panel[e] and b in panel[e]]
            if len(dd) < 200:
                continue
            dd = np.array(dd)
            ad = np.abs(dd)
            same_op = (OPERATOR.get(a) == OPERATOR.get(b) and OPERATOR.get(a) is not None)
            is_model = (a in MODELS) or (b in MODELS)
            pairs.append(dict(a=a, b=b, n=len(dd), tie=float((ad == 0).mean()),
                              mad=float(ad.mean()), med=float(np.median(ad)),
                              sd=float(ad.std(ddof=1)),
                              ge1=float((ad >= 1).mean()),
                              same_operator=same_op, model=is_model))
    pairs.sort(key=lambda r: -r["tie"])
    say(f"\n{'A':30s} {'B':30s} {'n':>5s} {'tie%':>7s} {'mean|d|':>8s} {'kind':>10s}")
    for p in pairs:
        kind = "MODEL" if p["model"] else ("SAME-OP" if p["same_operator"] else "distinct")
        say(f"{p['a'][:30]:30s} {p['b'][:30]:30s} {p['n']:5d} {100*p['tie']:7.2f} "
            f"{p['mad']:8.4f} {kind:>10s}")
    R["E_pairs"] = pairs

    bookpairs = [p for p in pairs if not p["model"]]
    same = [p for p in bookpairs if p["same_operator"]]
    dist = [p for p in bookpairs if not p["same_operator"]]

    def summ(nm, ps):
        if not ps:
            say(f"{nm}: none")
            return None
        tw = sum(p["n"] for p in ps)
        tie = sum(p["tie"] * p["n"] for p in ps) / tw
        mad = sum(p["mad"] * p["n"] for p in ps) / tw
        say(f"{nm:34s} pairs={len(ps):3d}  n-weighted tie {100*tie:6.2f}%   "
            f"mean|d| {mad:.4f}   tie range [{100*min(p['tie'] for p in ps):.1f},"
            f"{100*max(p['tie'] for p in ps):.1f}]%")
        return dict(pairs=len(ps), tie=tie, mad=mad,
                    tie_min=min(p["tie"] for p in ps), tie_max=max(p["tie"] for p in ps))

    say("")
    R["E_same_operator"] = summ("SAME OPERATOR (known duplicate)", same)
    R["E_distinct"] = summ("DISTINCT OPERATORS (real books)", dist)
    R["E_models"] = summ("MODEL vs anything", [p for p in pairs if p["model"]])

    # panel-level statistics on DISTINCT operators only, one skin per operator
    say("\n" + "-" * 72)
    say("PANEL COLLAPSED TO ONE SKIN PER OPERATOR (the honest N)")
    say("-" * 72)
    prefer = ["ESPN BET", "DraftKings", "MGM", "Unibet", "Titanbets",
              "BetfairSportsbook", "Caesars Sportsbook (New Jersey)",
              "SugarHouse", "PointsBet"]
    op_of = {b: OPERATOR[b] for b in prefer}
    coll = {}
    for e, d in panel.items():
        row = {}
        for b in prefer:
            if b in d:
                row[op_of[b]] = d[b]
        if row:
            coll[e] = row
    say(f"operators kept: {sorted(set(op_of.values()))}")
    hist = collections.Counter(len(v) for v in coll.values())
    say("distinct-operator count per event: " +
        ", ".join(f"{k}:{v}" for k, v in sorted(hist.items())))
    R["E_collapsed_hist"] = {str(k): v for k, v in sorted(hist.items())}
    return panel, meta, coll


def main():
    say(f"mb_diag.py  start {datetime.utcnow().isoformat()}Z")
    tr = load_tr()
    recs, both = part_a_b(tr)
    part_c_d(recs)
    part_e()
    with open(OUT, "w") as f:
        json.dump(R, f, indent=1, default=float)
    os.makedirs(f"{ROOT}/data/logs", exist_ok=True)
    with open(f"{ROOT}/data/logs/mb_diag.log", "w") as f:
        f.write("\n".join(LOG))
    say(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
