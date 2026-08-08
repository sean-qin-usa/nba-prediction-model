#!/usr/bin/env python3
"""BOOK PANEL — build MEASURED per-(game, book) OPEN and CLOSE panels for the
three modern seasons from raw files ALREADY ON DISK, plus the erichqiu 2018-19
offshore panel.  Successor to D163's scripts/mb_panel.py; nothing here is
extrapolated.

Sources, all local, all $0, no network:
  data/raw/sbr_ext/an_nba_odds_raw_{2023-24,2024-25,2025-26}.jsonl
  data/raw/sbr_ext/espn_nba_odds_raw_{2023-24,2024-25,2025-26}.jsonl
  data/raw/kaggle/erichqiu__nba-odds-and-scores/<season>/vegas.txt

Team names are routed through nbapred.teams (D171), which REPORTS unresolvable
names instead of dropping them silently.

Outputs data/bkp_panel.json + data/bkp_panel_rows.csv.gz and logs to stdout.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                      # noqa: E402
nbapred.threads.pin(1)                      # BEFORE numpy

import collections                          # noqa: E402
import csv                                  # noqa: E402
import gzip                                 # noqa: E402
import json                                 # noqa: E402
import math                                 # noqa: E402
from datetime import datetime, timezone     # noqa: E402

import numpy as np                          # noqa: E402

from nbapred.teams import abbrev_for, modern  # noqa: E402

RAWX = ROOT / "data" / "raw" / "sbr_ext"
ERICH = ROOT / "data" / "raw" / "kaggle" / "erichqiu__nba-odds-and-scores"
SEASONS = ["2023-24", "2024-25", "2025-26"]
REAL_TYPES = {"regular-season", "post-season", "play-in-season"}   # ESPN
AN_REAL_TYPES = {"reg", "post"}                                    # Action Network

# ---------------------------------------------------------------- operator map
# D163's rule, extended for the AN book ids.  A SKIN IS NOT A BOOK.
ESPN_OPERATOR = {
    "ESPN BET": "espnbet",
    "DraftKings": "draftkings",
    "MGM": "mgm", "Unibet": "unibet", "Titanbets": "titanbets",
    "BetfairSportsbook": "betfair", "SugarHouse": "sugarhouse",
    "PointsBet": "pointsbet",
    "Caesars Sportsbook (Colorado)": "caesars",
    "Caesars Sportsbook (Tennessee)": "caesars",
    "Caesars Sportsbook (New Jersey)": "caesars",
    "Caesars Sportsbook": "caesars",
}
# one canonical skin per operator, preferred order (D163 PREFER)
ESPN_PREFER = ["ESPN BET", "DraftKings", "MGM", "Unibet", "Titanbets",
               "BetfairSportsbook", "Caesars Sportsbook (New Jersey)",
               "SugarHouse", "PointsBet"]
ESPN_MODELS = {"accuscore", "betegy"}
# every "- Live Odds" provider is an IN-GAME market (D163 §4) -> excluded by
# never appearing in ESPN_PREFER.

AN_OPERATOR = {68: "draftkings", 69: "fanduel", 71: "betrivers",
               75: "mgm", 76: "caesars", 49: "caesars"}
AN_PSEUDO = {15: "an_consensus", 30: "an_open"}

ERICH_BOOKS = ["Pinnacle", "5dimes", "Heritage", "Bovada", "Betonline"]

# D162 §6 conversion, used ONLY to price juice differences in points (D163)
DP_PER_PT = 0.3989422804014327 / 12.574
BE110 = 110.0 / 210.0

LOG: list[str] = []
R: dict = {}


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def fnum(x):
    if x is None or x == "" or x == "None":
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def breakeven(american):
    if american is None:
        return None
    a = float(american)
    return (-a) / (-a + 100.0) if a < 0 else 100.0 / (a + 100.0)


def juice_pts(american):
    """How many spread points worse than -110 this price is. -115 -> +0.350."""
    b = breakeven(american)
    return None if b is None else (b - BE110) / DP_PER_PT


def amax_normal(k):
    """E[max of k iid standard normals], Monte Carlo, fixed seed (D163)."""
    if k <= 1:
        return 0.0
    rng = np.random.default_rng(11)
    return float(rng.standard_normal((200000, k)).max(axis=1).mean())


# --------------------------------------------------------------- team keys
_UNRESOLVED: collections.Counter = collections.Counter()
# feed abbreviation -> ours (build_odds_open.TR_TEAMS, kept for parity)
TR_TEAMS = {"BK": "BKN", "GS": "GSW", "NO": "NOP", "NY": "NYK",
            "PHO": "PHX", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}
# erichqiu spells teams by CITY
ERICH_CITY = {
    "Atlanta": "ATL", "Boston": "BOS", "Brooklyn": "BKN", "Charlotte": "CHA",
    "Chicago": "CHI", "Cleveland": "CLE", "Dallas": "DAL", "Denver": "DEN",
    "Detroit": "DET", "Golden State": "GSW", "Houston": "HOU",
    "Indiana": "IND", "LA Clippers": "LAC", "Los Angeles Clippers": "LAC",
    "LA Lakers": "LAL", "Los Angeles Lakers": "LAL", "L.A. Clippers": "LAC",
    "L.A. Lakers": "LAL", "Memphis": "MEM", "Miami": "MIA",
    "Milwaukee": "MIL", "Minnesota": "MIN", "New Orleans": "NOP",
    "New York": "NYK", "Oklahoma City": "OKC", "Orlando": "ORL",
    "Philadelphia": "PHI", "Phoenix": "PHX", "Portland": "POR",
    "Sacramento": "SAC", "San Antonio": "SAS", "Toronto": "TOR",
    "Utah": "UTA", "Washington": "WAS",
}


_MKT: dict = {}
_MISS: collections.Counter = collections.Counter()


def load_market():
    """Canonical game list from odds_market.  READ ONLY (read_only=True,
    retry 60s).  Index is (game_date, unordered team pair) -> canonical key,
    which is `build_odds_open._pair_join`'s key; it also drops preseason,
    All-Star and exhibition rows for free."""
    from nbapred.db import connect
    con = connect(read_only=True, retry_s=60)
    rows = con.execute("""SELECT season_end, game_date, home, away, score_home,
                                 score_away FROM odds_market""").fetchall()
    con.close()
    for se, gd, h, a, sh, sa in rows:
        _MKT[(gd.isoformat(), frozenset((h, a)))] = dict(
            season_end=se, date=gd.isoformat(), home=h, away=a,
            sh=sh, sa=sa)
    return len(rows)


def canon(date_iso: str, h: str, a: str):
    """Resolve a feed row to the canonical odds_market game.  ESPN dates are
    UTC and Action Network's are ET (build_odds_open.py records the trap), so
    +/-1 day is tried before giving up -- WITHOUT this, 75% of the ESPN/AN
    cross-source join silently fails and the two feeds look independent
    because they never meet."""
    import datetime as _dt
    pair = frozenset((h, a))
    d0 = _dt.date.fromisoformat(date_iso[:10])
    for off in (0, -1, 1):
        g = _MKT.get(((d0 + _dt.timedelta(days=off)).isoformat(), pair))
        if g is not None:
            return (g["date"], g["home"], g["away"])
    _MISS[f"{date_iso[:10]} {h}-{a}"] += 1
    return None


def team_key(full_name: str | None, abbr: str | None) -> str | None:
    """Resolve a feed team to OUR modern abbreviation.  D171 rule: try the
    full name through nbapred.teams first (it handles 'LA Clippers'), fall
    back to the feed abbreviation through TR_TEAMS + the franchise crosswalk,
    and RECORD anything that resolves to nothing."""
    ab = abbrev_for(full_name)
    if ab:
        return modern(ab)
    if abbr:
        a = str(abbr).strip().upper()
        a = TR_TEAMS.get(a, a)
        a = modern(a)
        if a and len(a) == 3:
            return a
    _UNRESOLVED[f"{full_name!r}/{abbr!r}"] += 1
    return None


# ============================================================ ESPN PANEL
def load_espn(season: str):
    """event -> dict with per-provider OPEN and CLOSE home margins.

    ESPN sign convention (verified against build_odds_open.py): the csv's
    `open_home_spread` is the HOME HANDICAP, so home margin = -handicap.  Here
    we parse the RAW jsonl instead, where the handicap lives in
    `awayTeamOdds.open.pointSpread.american` as the AWAY handicap -- D163's
    trap ('ESPN's open.pointSpread.value holds the DECIMAL PRICE, not the
    handicap, in some vintages') is why we read the *american* display string
    and never `.value`.  away handicap == home margin, directly.
    """
    path = RAWX / f"espn_nba_odds_raw_{season}.jsonl"
    ev = {}
    for line in open(path):
        d = json.loads(line)
        if d.get("season_type") not in REAL_TYPES:
            continue
        h = team_key(d.get("home_team"), d.get("home_abbr"))
        a = team_key(d.get("away_team"), d.get("away_abbr"))
        if not h or not a:
            continue
        key = canon(d["game_date"], h, a)
        if key is None:
            continue
        rec = dict(key=key, date=key[0], home=key[1], away=key[2],
                   sh=fnum(d.get("home_score")), sa=fnum(d.get("away_score")),
                   raw_open={}, raw_close={}, op_open={}, op_close={})
        for it in d.get("odds_items") or []:
            p = (it.get("provider") or {}).get("name")
            if not p:
                continue
            ao = it.get("awayTeamOdds") or {}
            for phase, bucket, rawb in (("open", rec["op_open"], rec["raw_open"]),
                                        ("close", rec["op_close"], rec["raw_close"])):
                blk = ao.get(phase) or {}
                ps = (blk.get("pointSpread") or {}).get("american")
                m = fnum(ps)
                if m is None:
                    continue
                rawb[p] = m                       # away handicap == home margin
                if p not in ESPN_PREFER:
                    continue
                sp = (blk.get("spread") or {}).get("american")
                bucket[ESPN_OPERATOR[p]] = dict(m=m, j=fnum(sp))
        if rec["op_open"] or rec["op_close"]:
            ev[key] = rec
    return ev


# ============================================================ ACTION NETWORK
def load_an(season: str):
    """game -> per-book quote.  The AN scoreboard payload is a SINGLE SNAPSHOT
    taken at scrape time (2026-08), so for a completed game each real book's
    number is its LAST price = the CLOSE.  book_id 30 ('Open') is Action
    Network's own CONSENSUS opening line -- one number, not a panel.
    """
    path = RAWX / f"an_nba_odds_raw_{season}.jsonl"
    ev = {}
    for line in open(path):
        day = json.loads(line)
        for g in day.get("games") or []:
            if g.get("type") not in AN_REAL_TYPES:
                continue
            tm = {t["id"]: t for t in (g.get("teams") or [])}
            ht, at = tm.get(g.get("home_team_id")), tm.get(g.get("away_team_id"))
            if not ht or not at:
                continue
            h = team_key(ht.get("full_name"), ht.get("abbr"))
            a = team_key(at.get("full_name"), at.get("abbr"))
            if not h or not a:
                continue
            key = canon(day["date"], h, a)
            if key is None:
                continue
            rec = dict(key=key, date=key[0], home=key[1], away=key[2],
                       start=g.get("start_time"), sh=None, sa=None,
                       books={}, pseudo={})
            bs = g.get("boxscore") or {}
            rec["sh"] = fnum(bs.get("total_home_points"))
            rec["sa"] = fnum(bs.get("total_away_points"))
            for bid_s, mk in (g.get("markets") or {}).items():
                bid = int(bid_s)
                sp = ((mk.get("event") or {}).get("spread")) or []
                away = next((e for e in sp if e.get("side") == "away"
                             and e.get("period") in (None, "event")), None)
                home = next((e for e in sp if e.get("side") == "home"
                             and e.get("period") in (None, "event")), None)
                if away is None or home is None:
                    continue
                m = fnum(away.get("value"))       # away handicap == home margin
                if m is None:
                    continue
                q = dict(m=m, jh=fnum(home.get("odds")), ja=fnum(away.get("odds")))
                if bid in AN_OPERATOR:
                    rec["books"][AN_OPERATOR[bid]] = q
                elif bid in AN_PSEUDO:
                    rec["pseudo"][AN_PSEUDO[bid]] = q
            if rec["books"] or rec["pseudo"]:
                ev[key] = rec
    return ev


# ============================================================ ERICHQIU 2018-19
def load_erich(season: str):
    """Offshore 5-book panel, one row per TEAM-game.  Home rows carry the
    home-side handicap, so home margin = -Line.  Returns game -> operator ->
    quote, plus the consensus open."""
    out = {}
    for fn in ("vegas.txt", "vegas_playoff.txt"):
        p = ERICH / season / fn
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            if r.get("Location") != "home":
                continue
            h = ERICH_CITY.get((r.get("Team") or "").strip())
            a = ERICH_CITY.get((r.get("OppTeam") or "").strip())
            if not h or not a:
                _UNRESOLVED[f"erich {r.get('Team')!r}/{r.get('OppTeam')!r}"] += 1
                continue
            books = {}
            for b in ERICH_BOOKS:
                ln = fnum(r.get(f"{b}_Line_Spread"))
                if ln is None:
                    continue
                books[b.lower()] = dict(m=-ln, j=fnum(r.get(f"{b}_Odds_Spread")))
            if len(books) < 2:
                continue
            key = canon(r["Date"], h, a)
            if key is None:
                continue
            op = fnum(r.get("Open_Line_Spread"))
            out[key] = dict(
                date=r["Date"], home=h, away=a, books=books,
                open_consensus=(-op if op is not None else None),
                pts=fnum(r.get("Pts")), spread_res=fnum(r.get("Spread")))
    return out


# ============================================================ STATISTICS
def pairwise(raw_by_event, opmap, models, label):
    """D163's shared-feed detector: tie rate between every pair of providers
    that quote the same game.  A true duplicate ties at 100.00%."""
    prov = collections.Counter()
    for d in raw_by_event.values():
        prov.update(d)
    names = [p for p, n in prov.most_common() if n >= 100]
    say(f"\n--- PAIRWISE TIE RATES — {label} ---")
    say(f"{'A':30s} {'B':30s} {'n':>5s} {'tie%':>7s} {'mean|d|':>8s} {'kind':>9s}")
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            va, vb = [], []
            for d in raw_by_event.values():
                if a in d and b in d:
                    va.append(d[a]); vb.append(d[b])
            if len(va) < 100:
                continue
            ad = np.abs(np.array(va) - np.array(vb))
            same = (opmap.get(a) is not None and opmap.get(a) == opmap.get(b))
            ism = (a in models) or (b in models)
            pairs.append(dict(a=a, b=b, n=len(ad), tie=float((ad == 0).mean()),
                              mad=float(ad.mean()), same_operator=same, model=ism))
    pairs.sort(key=lambda r: -r["tie"])
    for p in pairs:
        kind = "MODEL" if p["model"] else ("SAME-OP" if p["same_operator"] else "distinct")
        say(f"{p['a'][:30]:30s} {p['b'][:30]:30s} {p['n']:5d} {100*p['tie']:7.2f} "
            f"{p['mad']:8.4f} {kind:>9s}")
    return pairs


def dispersion(panel, label):
    """best-worst spread over ALL available operators, per game."""
    rng_ = np.array([max(q["m"] for q in d.values()) - min(q["m"] for q in d.values())
                     for d in panel.values() if len(d) >= 2])
    if not len(rng_):
        say(f"  {label}: no game with >=2 operators")
        return None
    say(f"  {label:38s} n={len(rng_):5d}  mean {rng_.mean():.4f}  "
        f"median {np.median(rng_):.2f}  sd {rng_.std(ddof=1):.4f}  "
        f"==0 on {100*(rng_ == 0).mean():6.2f}%")
    return dict(n=len(rng_), mean=float(rng_.mean()), median=float(np.median(rng_)),
                sd=float(rng_.std(ddof=1)), allsame=float((rng_ == 0).mean()))


def ladder(panel, label, use_juice=False, haircut=None, nsub=200, ks=(1, 2, 3, 5, 8)):
    """D163's statistic EXACTLY: gain_k = E[range of a random k-subset]/2, in
    spread points, side-agnostic and model-free.

    haircut='outlier' applies D142 §5(ii) as tightened by D163: a quote more
    than 1.5 pts from the panel MEDIAN is not realistically transactable, so it
    is removed from the subset before the best is taken.  No game is dropped.
    """
    import itertools
    rng = np.random.default_rng(20260804)
    per_k = collections.defaultdict(list)
    for d in panel.values():
        ops = sorted(d)
        if use_juice:
            vh, va = [], []
            ok = True
            for o in ops:
                q = d[o]
                jh = q.get("jh", q.get("j"))
                ja = q.get("ja", q.get("j"))
                pj, pa = juice_pts(jh), juice_pts(ja)
                if pj is None or pa is None:
                    ok = False
                    break
                vh.append(q["m"] + pj)
                va.append(q["m"] - pa)
            if not ok:
                continue
            vh = np.array(vh); va = np.array(va)
        else:
            vh = va = np.array([d[o]["m"] for o in ops])
        n = len(vh)
        if n < 2:
            continue
        if haircut == "outlier":
            med = float(np.median(vh))
            keep = np.abs(vh - med) <= 1.5
            if keep.sum() >= 2:
                vh, va = vh[keep], va[keep]
                n = len(vh)
            elif keep.sum() == 1:
                vh = va = np.array([vh[keep][0]] * 2)
                n = 2
        for k in ks:
            if k == 1:
                per_k[1].append(0.0)
                continue
            kk = min(k, n)
            if kk < 2:
                per_k[k].append(0.0)
                continue
            idx = list(range(n))
            if math.comb(n, kk) <= nsub:
                subs = list(itertools.combinations(idx, kk))
            else:
                subs = [tuple(rng.choice(n, kk, replace=False)) for _ in range(nsub)]
            gh = np.mean([vh[list(s)].mean() - vh[list(s)].min() for s in subs])
            ga = np.mean([va[list(s)].max() - va[list(s)].mean() for s in subs])
            per_k[k].append(0.5 * (gh + ga))
    tag = (" +JUICE" if use_juice else "") + (" +OUTLIER" if haircut else "")
    say(f"\n--- LADDER — {label}{tag} ---")
    say(f"{'k':>3s} {'n games':>8s} {'gain pts':>9s} {'D142 ceiling':>13s} {'meas/ceil':>10s}")
    tab = []
    for k in sorted(per_k):
        v = np.array(per_k[k])
        ceil = 0.586 * amax_normal(k)
        say(f"{k:3d} {len(v):8d} {v.mean():9.4f} {ceil:13.4f} "
            f"{(v.mean()/ceil if ceil > 0 else float('nan')):10.3f}")
        tab.append(dict(k=k, n=len(v), gain=float(v.mean()),
                        sd=float(v.std(ddof=1)) if len(v) > 1 else None,
                        ceiling=ceil, ratio=(v.mean() / ceil if ceil > 0 else None)))
    return tab


# ============================================================ MAIN
def main():
    say(f"bkp_panel.py  start {datetime.now(timezone.utc).isoformat()}")
    say(f"odds_market rows loaded (read_only=True, retry 60s): {load_market()}")
    say("=" * 78)
    say("PART A — WHAT EACH LOCAL SOURCE ACTUALLY CONTAINS  (MEASURED)")
    say("=" * 78)

    espn, an = {}, {}
    for s in SEASONS:
        espn[s] = load_espn(s)
        an[s] = load_an(s)
        eo = {k: v["op_open"] for k, v in espn[s].items() if len(v["op_open"]) >= 1}
        ec = {k: v["op_close"] for k, v in espn[s].items() if len(v["op_close"]) >= 1}
        ab = {k: v["books"] for k, v in an[s].items() if v["books"]}
        say(f"\n### {s}")
        say(f"  ESPN  games(real season types) {len(espn[s]):5d}   "
            f"with >=1 operator OPEN {len(eo):5d}   CLOSE {len(ec):5d}")
        ho = collections.Counter(len(v) for v in eo.values())
        hc = collections.Counter(len(v) for v in ec.values())
        say(f"        operators/game OPEN  " + ", ".join(f"{k}:{v}" for k, v in sorted(ho.items())))
        say(f"        operators/game CLOSE " + ", ".join(f"{k}:{v}" for k, v in sorted(hc.items())))
        say(f"  AN    games(reg+post)        {len(an[s]):5d}   with >=1 real book {len(ab):5d}")
        ha = collections.Counter(len(v) for v in ab.values())
        say(f"        books/game            " + ", ".join(f"{k}:{v}" for k, v in sorted(ha.items())))
        nopen = sum(1 for v in an[s].values() if "an_open" in v["pseudo"])
        ncons = sum(1 for v in an[s].values() if "an_consensus" in v["pseudo"])
        say(f"        pseudo-books: an_open {nopen}   an_consensus {ncons}")
        R.setdefault("inventory", {})[s] = dict(
            espn_games=len(espn[s]), espn_open=len(eo), espn_close=len(ec),
            espn_ops_open={str(k): v for k, v in sorted(ho.items())},
            espn_ops_close={str(k): v for k, v in sorted(hc.items())},
            an_games=len(an[s]), an_with_books=len(ab),
            an_books_per_game={str(k): v for k, v in sorted(ha.items())},
            an_open=nopen, an_consensus=ncons)

    say("\n" + "=" * 78)
    say("PART B — MEASURED BEST-WORST DISPERSION, PER SOURCE, OPEN AND CLOSE")
    say("=" * 78)
    for s in SEASONS:
        say(f"\n### {s}")
        R.setdefault("dispersion", {}).setdefault(s, {})
        R["dispersion"][s]["espn_open"] = dispersion(
            {k: v["op_open"] for k, v in espn[s].items() if len(v["op_open"]) >= 2},
            "ESPN operators, OPEN")
        R["dispersion"][s]["espn_close"] = dispersion(
            {k: v["op_close"] for k, v in espn[s].items() if len(v["op_close"]) >= 2},
            "ESPN operators, CLOSE")
        R["dispersion"][s]["an_close"] = dispersion(
            {k: v["books"] for k, v in an[s].items() if len(v["books"]) >= 2},
            "AN real books, SNAPSHOT(=close)")

    say("\n" + "=" * 78)
    say("PART C — ARE AN AND ESPN THE SAME FEED?  (D163's tie-rate method)")
    say("=" * 78)
    # ESPN internal pairwise, per season, on RAW provider names
    for s in SEASONS:
        raw = {k: v["raw_open"] for k, v in espn[s].items() if len(v["raw_open"]) >= 2}
        if len(raw) > 50:
            R.setdefault("espn_pairwise", {})[s] = pairwise(
                raw, ESPN_OPERATOR, ESPN_MODELS, f"ESPN {s} OPEN (raw providers)")
    # AN internal pairwise
    for s in SEASONS:
        raw = {k: {b: q["m"] for b, q in v["books"].items()}
               for k, v in an[s].items() if len(v["books"]) >= 2}
        if len(raw) > 50:
            R.setdefault("an_pairwise", {})[s] = pairwise(
                raw, {}, set(), f"AN {s} SNAPSHOT (real books)")

    # CROSS-SOURCE: where the SAME operator appears in both feeds
    say("\n--- CROSS-SOURCE: SAME OPERATOR IN BOTH FEEDS ---")
    say("If AN and ESPN resold ONE feed, a shared operator would tie at 100.00%")
    say("(D163's control: Kaggle BetOnline/Sportsbetting = one operator = 100.00%).")
    cross = {}
    for s in SEASONS:
        ekey = espn[s]
        for phase in ("op_open", "op_close"):
            acc = collections.defaultdict(lambda: ([], []))
            for key, av in an[s].items():
                ev = ekey.get(key)
                if ev is None:
                    continue
                for op, q in av["books"].items():
                    if op in ev[phase]:
                        acc[op][0].append(q["m"])
                        acc[op][1].append(ev[phase][op]["m"])
            for op, (x, y) in acc.items():
                if len(x) < 50:
                    continue
                d = np.abs(np.array(x) - np.array(y))
                say(f"  {s}  AN[{op}] vs ESPN[{op}] {phase:8s}  n={len(d):4d}  "
                    f"tie {100*(d == 0).mean():6.2f}%  mean|d| {d.mean():.4f}")
                cross[f"{s}|{op}|{phase}"] = dict(n=len(d), tie=float((d == 0).mean()),
                                                  mad=float(d.mean()))
    R["cross_source"] = cross
    # date-join diagnostic
    for s in SEASONS:
        ek = set(espn[s])
        ak = set(an[s])
        say(f"  {s} key overlap (date,home,away): ESPN {len(ek)}  AN {len(ak)}  "
            f"both {len(ek & ak)}")

    say("\n" + "=" * 78)
    say("PART D — MERGED PANEL, ONE PER SEASON")
    say("=" * 78)
    merged = {}
    for s in SEASONS:
        ekey = espn[s]
        keys = set(ekey) | set(an[s])
        mo, mc = {}, {}
        prov = collections.Counter()
        for key in keys:
            ev, av = ekey.get(key), an[s].get(key)
            o, c = {}, {}
            if ev:
                for op, q in ev["op_open"].items():
                    o[op] = dict(m=q["m"], jh=q["j"], ja=q["j"], src="espn")
                for op, q in ev["op_close"].items():
                    c[op] = dict(m=q["m"], jh=q["j"], ja=q["j"], src="espn")
            if av:
                # DEDUP RULE: on a shared operator ESPN wins at the OPEN
                # (it is a true per-book open); AN wins at the CLOSE (it is a
                # real book snapshot at final state).  See notes §3.
                for op, q in av["books"].items():
                    c[op] = dict(m=q["m"], jh=q["jh"], ja=q["ja"], src="an")
            for op in set(o) | set(c):
                prov[op] += 1
            if len(o) >= 1:
                mo[key] = o
            if len(c) >= 1:
                mc[key] = c
        merged[s] = dict(open=mo, close=mc)
        say(f"\n### {s}  merged games OPEN {len(mo)}  CLOSE {len(mc)}")
        say("  operators present: " + ", ".join(f"{k}({v})" for k, v in prov.most_common()))
        ho = collections.Counter(len(v) for v in mo.values())
        hc = collections.Counter(len(v) for v in mc.values())
        say("  operators/game OPEN  " + ", ".join(f"{k}:{v}" for k, v in sorted(ho.items())))
        say("  operators/game CLOSE " + ", ".join(f"{k}:{v}" for k, v in sorted(hc.items())))
        R.setdefault("merged", {})[s] = dict(
            n_open=len(mo), n_close=len(mc),
            operators={k: v for k, v in prov.most_common()},
            ops_per_game_open={str(k): v for k, v in sorted(ho.items())},
            ops_per_game_close={str(k): v for k, v in sorted(hc.items())})
        R["dispersion"][s]["merged_open"] = dispersion(
            {k: v for k, v in mo.items() if len(v) >= 2}, "MERGED, OPEN")
        R["dispersion"][s]["merged_close"] = dispersion(
            {k: v for k, v in mc.items() if len(v) >= 2}, "MERGED, CLOSE")

    say("\n" + "=" * 78)
    say("PART E — THE MEASURED LADDER")
    say("=" * 78)
    for s in SEASONS:
        for phase in ("open", "close"):
            pan = {k: v for k, v in merged[s][phase].items() if len(v) >= 2}
            if len(pan) < 50:
                say(f"\n--- {s} {phase}: only {len(pan)} games with >=2 operators — "
                    f"NO LADDER POSSIBLE ---")
                R.setdefault("ladder", {})[f"{s}|{phase}"] = None
                continue
            R.setdefault("ladder", {})[f"{s}|{phase}|raw"] = ladder(pan, f"{s} {phase}")
            R["ladder"][f"{s}|{phase}|juice"] = ladder(pan, f"{s} {phase}", use_juice=True)
            R["ladder"][f"{s}|{phase}|outlier"] = ladder(pan, f"{s} {phase}", haircut="outlier")
            R["ladder"][f"{s}|{phase}|juice_outlier"] = ladder(
                pan, f"{s} {phase}", use_juice=True, haircut="outlier")

    say("\n" + "=" * 78)
    say("PART F — ERICHQIU OFFSHORE PANEL (2012-13..2018-19), the 2018-19 cell")
    say("=" * 78)
    er = {}
    for s in ["2012-13", "2013-14", "2014-15", "2015-16", "2016-17", "2017-18", "2018-19"]:
        g = load_erich(s)
        er[s] = g
        pan = {k: v["books"] for k, v in g.items() if len(v["books"]) >= 2}
        h = collections.Counter(len(v) for v in pan.values())
        say(f"\n### {s}  games {len(g)}  with >=2 books {len(pan)}   "
            f"books/game " + ", ".join(f"{k}:{v}" for k, v in sorted(h.items())))
        d = dispersion(pan, f"erichqiu {s} (offshore, snapshot)")
        R.setdefault("erich", {})[s] = dict(n=len(g), n2=len(pan), disp=d,
                                            hist={str(k): v for k, v in sorted(h.items())})
        if len(pan) > 200:
            R["erich"][s]["ladder"] = ladder(pan, f"erichqiu {s}")
            R["erich"][s]["ladder_outlier"] = ladder(pan, f"erichqiu {s}", haircut="outlier")

    say("\n" + "=" * 78)
    say("UNRESOLVED TEAM NAMES (D171 rule: report, never silently drop)")
    say("=" * 78)
    if _UNRESOLVED:
        for k, v in _UNRESOLVED.most_common(40):
            say(f"  {k}  x{v}")
    else:
        say("  NONE — every team string in every source resolved.")
    R["unresolved"] = {k: v for k, v in _UNRESOLVED.most_common()}
    say(f"\nFEED ROWS WITH NO odds_market MATCH (+/-1 day pair join): {sum(_MISS.values())}")
    for k, v in _MISS.most_common(15):
        say(f"  {k}  x{v}")
    R["unmatched_market"] = dict(total=sum(_MISS.values()),
                                 top={k: v for k, v in _MISS.most_common(40)})

    (ROOT / "data" / "bkp_panel.json").write_text(json.dumps(R, indent=1, default=str))
    (ROOT / "data" / "logs").mkdir(exist_ok=True)
    (ROOT / "data" / "logs" / "bkp_panel.log").write_text("\n".join(LOG))

    # dump the merged modern panels + erichqiu as a flat csv for downstream use
    with gzip.open(ROOT / "data" / "bkp_panel_rows.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["season", "phase", "game_date", "home", "away", "operator",
                    "home_margin", "juice_home", "juice_away", "source"])
        for s in SEASONS:
            for phase in ("open", "close"):
                for (dt_, h, a), d in merged[s][phase].items():
                    for op, q in d.items():
                        w.writerow([s, phase, dt_, h, a, op, q["m"],
                                    q.get("jh"), q.get("ja"), q.get("src")])
        for s, g in er.items():
            for (dt_, h, a), v in g.items():
                for op, q in v["books"].items():
                    w.writerow([s, "close", dt_, h, a, op, q["m"], q.get("j"),
                                q.get("j"), "erichqiu"])
    say(f"\nwrote data/bkp_panel.json, data/bkp_panel_rows.csv.gz, "
        f"data/logs/bkp_panel.log")


if __name__ == "__main__":
    main()
