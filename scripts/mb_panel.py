#!/usr/bin/env python3
"""PART 3 (a) — BUILD THE REAL MULTI-BOOK PANELS AND MEASURE THE LADDER.

Two panels, both already on disk, both $0, neither ever used as a panel:

  ESPN23   data/raw/sbr_ext/espn_nba_open_close_2023-24.csv
           one row per (event, provider). 16 provider names, of which 11 are
           distinct real OPERATORS. Carries the OPENING handicap AND the
           OPENING spread juice per book — the juice is the thing D142 never
           had and could not price.
           Known-duplicate arms for the vendor test: Caesars rendered as
           CO / TN / NJ / NJ-Live, ESPN BET rendered as main / Live.
           MODELS (accuscore, betegy) are excluded from every book statistic.

  KAG      data/raw/kaggle/ehallmar__.../nba_betting_spread.csv
           one row per (game, book), 131,690 rows, 14,914 games,
           2006-07..2017-18, up to 10 OFFSHORE books (Pinnacle, 5Dimes,
           Bookmaker, Bovada, BetOnline, Sportsbetting, JustBet, Intertops,
           YouWager, Heritage). No timestamp: the snapshot is UNLABELLED and
           is validated against our own open and close before use.
           Known-duplicate arm: BetOnline / Sportsbetting are one operator.

THE LADDER STATISTIC, and why it is the right one.
D142 measured "a shopper captures HALF the dispersion: +0.331 pts" on N=2.
That is exactly E[range]/2: a HOME bettor wants the smallest handicap, an AWAY
bettor the largest, a one-book bettor draws uniformly, so for a 50/50 side
    E[gain over one book] = ( E[mean-min] + E[max-mean] ) / 2 = E[range_k]/2.
It is model-free and side-free. For k books it generalises directly, and under
D142's own Gaussian (iid per-book idiosyncratic sd sigma = 0.586 pts) it equals
sigma * a_k with a_k = E[max of k standard normals]:
    N=2 0.331  N=3 0.496  N=4 0.603  N=5 0.682  N=6 0.743  N=8 0.834 pts
which reproduces D142's own +0.331 at N=2 and its pp ladder at a constant
2.85 pp/pt. So the measured ladder can be laid directly against the ceiling.

Read-only (DuckDB read_only=True, for the team-id crosswalk only).
"""
from __future__ import annotations

import nbapred.threads  # noqa: F401
nbapred.threads.pin(1)

import collections
import csv
import gzip
import itertools
import json
import math
import os
from datetime import datetime

import numpy as np

ROOT = "/hdd/steveqin/sean_dev/nba_model"
OUT = f"{ROOT}/data/mb_panel.json"

ESPN_ABBR = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS",
             "UTAH": "UTA", "WSH": "WAS", "PHO": "PHX", "BK": "BKN"}

MODELS = {"accuscore", "betegy"}
# provider -> operator. Skins of one operator must never be counted as two books.
OPERATOR = {
    "ESPN BET": "espnbet", "ESPN Bet - Live Odds": "espnbet",
    "Caesars Sportsbook (Colorado)": "caesars",
    "Caesars Sportsbook (Tennessee)": "caesars",
    "Caesars Sportsbook (New Jersey)": "caesars",
    "Caesars Sportsbook (New Jersey) - Live Odds": "caesars",
    "Caesars Sportsbook": "caesars",
    "DraftKings": "draftkings", "DraftKings - Live Odds": "draftkings",
    "MGM": "mgm", "Unibet": "unibet", "Titanbets": "titanbets",
    "BetfairSportsbook": "betfair", "SugarHouse": "sugarhouse",
    "PointsBet": "pointsbet",
}
# one canonical skin per operator, preferred order
PREFER = ["ESPN BET", "DraftKings", "MGM", "Unibet", "Titanbets",
          "BetfairSportsbook", "Caesars Sportsbook (New Jersey)",
          "SugarHouse", "PointsBet"]

# Kaggle offshore operators. BetOnline and Sportsbetting are ONE operator
# (same parent, same line feed) — the known-duplicate arm of this panel.
KAG_OPERATOR = {
    "Pinnacle Sports": "pinnacle", "5Dimes": "5dimes", "Bookmaker": "bookmaker",
    "Bovada": "bovada", "BetOnline": "betonline", "Sportsbetting": "betonline",
    "JustBet": "justbet", "Intertops": "intertops", "YouWager": "youwager",
    "Heritage": "heritage",
}
KAG_PREFER = ["Pinnacle Sports", "5Dimes", "Bookmaker", "Bovada", "BetOnline",
              "JustBet", "Intertops", "YouWager", "Heritage"]

# D162 §6: sd(actual margin - opening line) = 12.574 pts => dP/dpoint at the
# median is phi(0)/12.574. Used ONLY to price juice differences in points.
DP_PER_PT = 0.3989422804014327 / 12.574
BE110 = 110.0 / 210.0

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


def breakeven(american):
    """Win probability a bet at this American price must clear."""
    if american is None:
        return None
    a = float(american)
    if a < 0:
        return (-a) / (-a + 100.0)
    return 100.0 / (a + 100.0)


def juice_pts(american):
    """How many spread points worse than -110 this price is. -115 -> +0.350."""
    b = breakeven(american)
    if b is None:
        return None
    return (b - BE110) / DP_PER_PT


def amax_normal(k):
    """E[max of k iid standard normals], Monte Carlo, fixed seed."""
    if k <= 1:
        return 0.0
    rng = np.random.default_rng(11)
    return float(rng.standard_normal((200000, k)).max(axis=1).mean())


# ------------------------------------------------------------------ frames
def load_ats19():
    with gzip.open(f"{ROOT}/data/ats19_frame.csv.gz", "rt") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("p_us", "m_us", "m_us696", "open_margin", "close_margin",
                  "margin_actual", "edge"):
            r[k] = fnum(r[k])
    return rows


def load_espn_panel():
    """event -> {operator: dict(open_margin, home_juice, away_juice, close_margin)}
    plus the RAW provider-level open handicap for the duplicate-skin test."""
    rows = list(csv.DictReader(open(f"{ROOT}/data/raw/sbr_ext/espn_nba_open_close_2023-24.csv")))
    raw = collections.defaultdict(dict)     # event -> provider -> open home margin
    panel = collections.defaultdict(dict)   # event -> operator -> quote
    meta = {}
    for x in rows:
        if x["season_type"] not in ("regular-season", "post-season", "play-in-season"):
            continue
        sp = fnum(x["open_home_spread"])
        if sp is None:
            continue
        m = -sp                              # expected HOME margin
        raw[x["event_id"]][x["provider_name"]] = m
        meta[x["event_id"]] = dict(
            date=x["game_date_et"] or x["game_date"],
            home=ESPN_ABBR.get(x["home_abbr"], x["home_abbr"]),
            away=ESPN_ABBR.get(x["away_abbr"], x["away_abbr"]))
        if x["provider_name"] not in PREFER:
            continue
        op = OPERATOR[x["provider_name"]]
        panel[x["event_id"]][op] = dict(
            m=m,
            jh=fnum(x["open_home_spread_odds"]),
            ja=fnum(x["open_away_spread_odds"]),
            close=(-fnum(x["close_home_spread"])
                   if fnum(x["close_home_spread"]) is not None else None))
    return raw, panel, meta


def load_kaggle_panel():
    import duckdb
    con = duckdb.connect(f"{ROOT}/data/nba.duckdb", read_only=True)
    tm = con.execute(
        "select game_id, team_id, team_abbrev, is_home from nba_games").fetchall()
    con.close()
    home_of = {}
    for gid, tid, ab, ish in tm:
        home_of[(str(gid), int(tid))] = bool(ish)
    p = (f"{ROOT}/data/raw/kaggle/ehallmar__nba-historical-stats-and-betting-data/"
         "nba_betting_spread.csv")
    raw = collections.defaultdict(dict)
    panel = collections.defaultdict(dict)
    miss = 0
    for x in csv.DictReader(open(p)):
        gid = x["game_id"]
        tid = int(x["team_id"])
        s1 = fnum(x["spread1"])
        if s1 is None:
            continue
        ish = home_of.get((gid, tid))
        if ish is None:
            miss += 1
            continue
        # spread1 is the handicap ON team_id. home margin = -spread(home)
        m = -s1 if ish else s1
        bk = x["book_name"]
        raw[gid][bk] = m
        if bk in KAG_PREFER:
            panel[gid][KAG_OPERATOR[bk]] = dict(
                m=m, jh=fnum(x["price1"]) if ish else fnum(x["price2"]),
                ja=fnum(x["price2"]) if ish else fnum(x["price1"]), close=None)
    return raw, panel, miss


# -------------------------------------------------------------- statistics
def pairwise(raw, opmap, models, label):
    say(f"\n--- PAIRWISE TIE / DISPERSION — {label} ---")
    cnt = collections.Counter()
    for e, d in raw.items():
        for b in d:
            cnt[b] += 1
    names = [b for b, n in cnt.most_common() if n >= 300]
    pairs = []
    for a, b in itertools.combinations(names, 2):
        dd = [raw[e][a] - raw[e][b] for e in raw if a in raw[e] and b in raw[e]]
        if len(dd) < 200:
            continue
        ad = np.abs(np.array(dd))
        same = (opmap.get(a) is not None and opmap.get(a) == opmap.get(b))
        ism = (a in models) or (b in models)
        pairs.append(dict(a=a, b=b, n=len(ad), tie=float((ad == 0).mean()),
                          mad=float(ad.mean()), sd=float(ad.std(ddof=1)),
                          ge1=float((ad >= 1).mean()),
                          same_operator=same, model=ism))
    pairs.sort(key=lambda r: -r["tie"])
    say(f"{'A':32s} {'B':32s} {'n':>5s} {'tie%':>7s} {'mean|d|':>8s} {'kind':>9s}")
    for p in pairs:
        kind = "MODEL" if p["model"] else ("SAME-OP" if p["same_operator"] else "distinct")
        say(f"{p['a'][:32]:32s} {p['b'][:32]:32s} {p['n']:5d} {100*p['tie']:7.2f} "
            f"{p['mad']:8.4f} {kind:>9s}")

    def summ(nm, ps):
        if not ps:
            say(f"  {nm}: none")
            return None
        w = sum(p["n"] for p in ps)
        t = sum(p["tie"] * p["n"] for p in ps) / w
        m = sum(p["mad"] * p["n"] for p in ps) / w
        say(f"  {nm:36s} pairs={len(ps):3d} n-wtd tie {100*t:6.2f}%  mean|d| {m:.4f}"
            f"  range[{100*min(p['tie'] for p in ps):.1f},{100*max(p['tie'] for p in ps):.1f}]%")
        return dict(pairs=len(ps), n=w, tie=t, mad=m,
                    tie_min=min(p["tie"] for p in ps),
                    tie_max=max(p["tie"] for p in ps))
    say("")
    bp = [p for p in pairs if not p["model"]]
    out = dict(pairs=pairs)
    out["same_operator"] = summ("SAME OPERATOR (known duplicate)",
                                [p for p in bp if p["same_operator"]])
    out["distinct"] = summ("DISTINCT OPERATORS (real books)",
                           [p for p in bp if not p["same_operator"]])
    out["models"] = summ("MODEL vs anything", [p for p in pairs if p["model"]])
    return out


def ladder(panel, label, use_juice=False, side_key=None, nsub=200):
    """Measured best-of-k ladder, in SPREAD POINTS, side-agnostic:
       gain_k = E[ range of a random k-subset ] / 2.
       With use_juice the handicap is first converted to an effective handicap
       at -110 using D162's dP/dpoint, separately per side, and the gain is the
       average of the HOME-side and AWAY-side gains (identical by symmetry when
       juice is ignored)."""
    say(f"\n--- MEASURED BEST-OF-k LADDER — {label}"
        f"{' (JUICE-ADJUSTED)' if use_juice else ' (handicap only)'} ---")
    rng = np.random.default_rng(20260804)
    per_k = collections.defaultdict(list)
    nmax = 0
    for e, d in panel.items():
        ops = sorted(d)
        if use_juice:
            vh, va = [], []
            for o in ops:
                q = d[o]
                pj, pa = juice_pts(q["jh"]), juice_pts(q["ja"])
                if pj is None or pa is None:
                    vh = va = []
                    break
                vh.append(q["m"] + pj)      # HOME: smaller is better
                va.append(q["m"] - pa)      # AWAY: larger is better
            if not vh:
                continue
            vh = np.array(vh); va = np.array(va)
        else:
            vh = va = np.array([d[o]["m"] for o in ops])
        n = len(vh)
        if n < 2:
            continue
        nmax = max(nmax, n)
        for k in range(1, n + 1):
            if k == 1:
                per_k[k].append(0.0)
                continue
            idx = list(range(n))
            if math.comb(n, k) <= nsub:
                subs = list(itertools.combinations(idx, k))
            else:
                subs = [tuple(rng.choice(n, k, replace=False)) for _ in range(nsub)]
            gh = np.mean([vh[list(s)].mean() - vh[list(s)].min() for s in subs])
            ga = np.mean([va[list(s)].max() - va[list(s)].mean() for s in subs])
            per_k[k].append(0.5 * (gh + ga))
    say(f"{'k':>3s} {'n games':>8s} {'gain pts':>9s} {'D142 Gaussian':>14s} {'meas/ceiling':>13s}")
    tab = []
    for k in sorted(per_k):
        v = np.array(per_k[k])
        ceil = 0.586 * amax_normal(k)
        say(f"{k:3d} {len(v):8d} {v.mean():9.4f} {ceil:14.4f} "
            f"{(v.mean()/ceil if ceil > 0 else float('nan')):13.3f}")
        tab.append(dict(k=k, n=len(v), gain=float(v.mean()), sd=float(v.std(ddof=1)),
                        gaussian_ceiling=ceil,
                        ratio=(v.mean() / ceil if ceil > 0 else None)))
    return dict(table=tab, nmax=nmax, use_juice=use_juice)


def main():
    say(f"mb_panel.py  start {datetime.utcnow().isoformat()}Z")

    # ---------------- ESPN 2023-24
    say("\n" + "=" * 72)
    say("PANEL ESPN23 — 2023-24, ESPN core API, per-provider OPENING spread")
    say("=" * 72)
    eraw, epan, emeta = load_espn_panel()
    say(f"events with >=1 opening quote: {len(eraw)}")
    hist = collections.Counter(len(v) for v in epan.values())
    say("DISTINCT-OPERATOR count per event: " +
        ", ".join(f"{k}:{v}" for k, v in sorted(hist.items())))
    R["espn_operator_hist"] = {str(k): v for k, v in sorted(hist.items())}
    R["espn_pairwise"] = pairwise(eraw, OPERATOR, MODELS, "ESPN23 (all providers)")
    R["espn_ladder"] = ladder(epan, "ESPN23", use_juice=False)
    R["espn_ladder_juice"] = ladder(epan, "ESPN23", use_juice=True)

    # best-worst dispersion on the full operator panel
    rng_ = [max(q["m"] for q in d.values()) - min(q["m"] for q in d.values())
            for d in epan.values() if len(d) >= 2]
    rng_ = np.array(rng_)
    say(f"\nESPN23 best-worst dispersion over ALL available operators "
        f"(n={len(rng_)}): mean {rng_.mean():.4f} median {np.median(rng_):.2f} "
        f"sd {rng_.std(ddof=1):.4f}  ==0 on {100*(rng_ == 0).mean():.2f}%")
    R["espn_range"] = dict(n=len(rng_), mean=float(rng_.mean()),
                           median=float(np.median(rng_)),
                           sd=float(rng_.std(ddof=1)),
                           allsame=float((rng_ == 0).mean()))

    # unanimity: how often do ALL operators quote the same number?
    for k in (2, 5, 8, 9):
        sub = [d for d in epan.values() if len(d) >= k]
        if not sub:
            continue
        u = np.mean([max(q["m"] for q in d.values()) == min(q["m"] for q in d.values())
                     for d in sub])
        say(f"  events with >={k} operators: {len(sub):5d}   ALL AGREE {100*u:6.2f}%")

    # ---------------- Kaggle offshore
    say("\n" + "=" * 72)
    say("PANEL KAG — 2006-07..2017-18, Kaggle ehallmar, up to 10 offshore books")
    say("=" * 72)
    kraw, kpan, kmiss = load_kaggle_panel()
    say(f"games: {len(kraw)}  rows with no team-id crosswalk: {kmiss}")
    hist = collections.Counter(len(v) for v in kpan.values())
    say("DISTINCT-OPERATOR count per game: " +
        ", ".join(f"{k}:{v}" for k, v in sorted(hist.items())))
    R["kag_operator_hist"] = {str(k): v for k, v in sorted(hist.items())}
    R["kag_pairwise"] = pairwise(kraw, KAG_OPERATOR, set(), "KAG (all books)")
    R["kag_ladder"] = ladder(kpan, "KAG", use_juice=False)
    R["kag_ladder_juice"] = ladder(kpan, "KAG", use_juice=True)

    rng_ = [max(q["m"] for q in d.values()) - min(q["m"] for q in d.values())
            for d in kpan.values() if len(d) >= 2]
    rng_ = np.array(rng_)
    say(f"\nKAG best-worst dispersion over ALL available operators "
        f"(n={len(rng_)}): mean {rng_.mean():.4f} median {np.median(rng_):.2f} "
        f"sd {rng_.std(ddof=1):.4f}  ==0 on {100*(rng_ == 0).mean():.2f}%")
    R["kag_range"] = dict(n=len(rng_), mean=float(rng_.mean()),
                          median=float(np.median(rng_)),
                          sd=float(rng_.std(ddof=1)),
                          allsame=float((rng_ == 0).mean()))
    for k in (2, 5, 8, 9):
        sub = [d for d in kpan.values() if len(d) >= k]
        if not sub:
            continue
        u = np.mean([max(q["m"] for q in d.values()) == min(q["m"] for q in d.values())
                     for d in sub])
        say(f"  games with >={k} operators: {len(sub):5d}   ALL AGREE {100*u:6.2f}%")

    # ---------------- join coverage against the D162 frame
    say("\n" + "=" * 72)
    say("JOIN COVERAGE AGAINST THE D162 ATS FRAME (22,742 scored games)")
    say("=" * 72)
    ats = load_ats19()
    say(f"ats19 frame rows: {len(ats)}")
    bykey = {}
    for r in ats:
        bykey[(r["game_date"], frozenset((r["home"], r["away"])))] = r
    byid = {r["game_id"]: r for r in ats}

    hit = 0
    espn_join = {}
    for e, d in epan.items():
        m = emeta[e]
        r = bykey.get((m["date"], frozenset((m["home"], m["away"]))))
        if r is None:
            continue
        hit += 1
        espn_join[r["game_id"]] = d
    say(f"ESPN23 events joined to the ATS frame: {hit} / {len(epan)}")
    kj = {g: d for g, d in kpan.items() if g in byid}
    say(f"KAG games joined to the ATS frame:     {len(kj)} / {len(kpan)}")
    kseas = collections.Counter(byid[g]["season"] for g in kj)
    say("KAG joined by season: " + ", ".join(f"{k}:{v}" for k, v in sorted(kseas.items())))
    R["join"] = dict(espn=hit, espn_total=len(epan), kag=len(kj), kag_total=len(kpan),
                     kag_by_season={k: v for k, v in sorted(kseas.items())})

    # persist the joined panels for the scoring step
    with open(f"{ROOT}/data/mb_panel_espn.json", "w") as f:
        json.dump(espn_join, f)
    with open(f"{ROOT}/data/mb_panel_kag.json", "w") as f:
        json.dump(kj, f)

    with open(OUT, "w") as f:
        json.dump(R, f, indent=1, default=float)
    os.makedirs(f"{ROOT}/data/logs", exist_ok=True)
    with open(f"{ROOT}/data/logs/mb_panel.log", "w") as f:
        f.write("\n".join(LOG))
    say(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
