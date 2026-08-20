#!/usr/bin/env python3
"""D250 — CAN AVAILABILITY BE RECONSTRUCTED FOR ALL 19 SEASONS WITHOUT SCRAPING?

D186 bounds the availability leg to 2019-20+ because `injury_reports_pit` starts
2018-12-17. D249 established that `player_game_stats` covers 2000-01..2025-26 but
holds only DRESSED players (25.5-26.4 rows/game, 17-19% of them zero-minute), so
inactive players are absent from it entirely.

That absence is itself the signal. A player who is on a team's active roster in
that stretch of the season but carries NO ROW for a given game did not dress. So
per-game inactive sets should be derivable from data already on disk, for every
season in the frame, with no network access at all.

"Should be" is doing a lot of work in that sentence, so this validates it against
ground truth instead of asserting it.

GROUND TRUTH. basketball-reference box scores publish a per-team `Inactive:`
block. It is reachable (verified: HTTP 200, 469 KB) where stats.nba.com,
cdn.nba.com, site.api.espn.com and prosportstransactions are not — the first
three blocked at the egress proxy, the last behind a Cloudflare challenge.
b-ref asks for a maximum of 20 requests per minute; this script sleeps 4.0s
between fetches and touches a few dozen games, once, and caches every response
so a rerun costs nothing.

WHY VALIDATE RATHER THAN JUST SCRAPE. Backfilling 2007-08..2018-19 from b-ref is
~13,500 box scores, which at a polite rate is ~11 hours of someone else's
bandwidth. If the on-disk derivation reproduces the published inactive list, that
scrape is unnecessary and the data exists today for all 19 seasons. If it does
not, the failure mode tells us exactly what the derivation is missing.

THE ESTIMAND. Per team-game, comparing derived vs published inactive sets:
    precision = |derived AND published| / |derived|
    recall    = |derived AND published| / |published|
Both are reported per era, because the roster-window rule can plausibly behave
differently in the 3-inactive era than the modern load-management era.
"""
from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

CACHE = ROOT / "data" / "raw" / "bref_boxscores"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/120.0 Safari/537.36")
SLEEP = 4.0                      # b-ref asks <=20 req/min; this is 15
# b-ref team codes that differ from ours, and the codes are ERA-DEPENDENT:
# Charlotte is CHA through 2013-14 and CHO after; Brooklyn is NJN before the
# 2012-13 move. Getting this wrong produces a 404, not a wrong answer, which is
# why the first pilot lost 200901030CHO.
BREF = {"BKN": "BRK", "PHX": "PHO"}


def bref_code(ab, season):
    if ab == "CHA":
        return "CHO" if season >= "2014-15" else "CHA"
    if ab == "BKN":
        return "BRK" if season >= "2012-13" else "NJN"
    return BREF.get(ab, ab)


WINDOW = 10                      # team-games either side (ARM A only)


def norm(n):
    """Fold names to a comparable key: accents, punctuation, suffixes."""
    n = unicodedata.normalize("NFKD", str(n))
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.lower().replace(".", "").replace("'", "").replace("-", " ")
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n)
    return re.sub(r"\s+", " ", n).strip()


def fetch(code):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{code}.html"
    if f.exists() and f.stat().st_size > 5000:
        return f.read_text(encoding="utf-8", errors="replace"), True
    import urllib.request
    req = urllib.request.Request(
        f"https://www.basketball-reference.com/boxscores/{code}.html",
        headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        h = r.read().decode("utf-8", errors="replace")
    f.write_text(h, encoding="utf-8")
    time.sleep(SLEEP)
    return h, False


def parse_inactive(h):
    """-> {team_code: [names]} from the `Inactive:` block."""
    m = re.search(r"Inactive:(.{0,3000}?)(?:Officials:|Attendance:|</div>)",
                  h, re.S)
    if not m:
        return {}
    frag = m.group(1)
    frag = re.sub(r"<[^>]+>", "\x00", frag)
    frag = frag.replace("&nbsp;", " ")
    out, cur = {}, None
    for tok in (t.strip() for t in frag.split("\x00")):
        if not tok or tok in {",", "-"}:
            continue
        if re.fullmatch(r"[A-Z]{3}", tok):
            cur = tok; out[cur] = []
        elif cur and re.search(r"[A-Za-z]{2,}\s+[A-Za-z]", tok):
            for nm in tok.split(","):
                nm = nm.strip()
                if nm:
                    out[cur].append(nm)
    return {k: v for k, v in out.items() if v}


def main():
    import duckdb
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
    f["game_id"] = (f.game_id.astype(str).str.replace(r"\.0$", "", regex=True)
                    .str.zfill(10))
    f["game_date"] = pd.to_datetime(f.game_date)

    # sample across eras, deterministically
    rng = np.random.default_rng(250)
    picks = []
    for s in ("2008-09", "2012-13", "2016-17", "2018-19", "2021-22", "2024-25"):
        g = f[f.season == s]
        if len(g):
            picks.append(g.iloc[rng.choice(len(g), min(n_games // 6, len(g)),
                                           replace=False)])
    smp = pd.concat(picks).sort_values("game_date")
    print(f"validation sample: {len(smp)} games across "
          f"{smp.season.nunique()} seasons")

    con = duckdb.connect(str(ROOT / "data" / "nba.duckdb"), read_only=True)
    names = dict(con.execute(
        "SELECT player_id, full_name FROM nba_players").fetchall())
    pgs = con.execute("""
        SELECT CAST(game_id AS VARCHAR) gid, player_id, team_id, seconds
        FROM player_game_stats""").df()
    con.close()
    pgs["gid"] = pgs.gid.str.zfill(10)
    # team_id -> the frame's abbreviation, learned from the data itself
    tid_map = {}
    j = pgs.merge(f[["game_id", "home", "away"]], left_on="gid",
                  right_on="game_id", how="inner")
    order = pgs.groupby("gid").team_id.apply(lambda s: list(dict.fromkeys(s)))
    for gid, tids in order.items():
        row = f[f.game_id == gid]
        if len(row) != 1 or len(tids) != 2:
            continue
        # the first team_id listed is the AWAY side in NBA box order
        for t, ab in zip(tids, [row.away.iloc[0], row.home.iloc[0]]):
            tid_map.setdefault(t, {}).setdefault(ab, 0)
            tid_map[t][ab] += 1
    tid2ab = {t: max(d, key=d.get) for t, d in tid_map.items()}
    print(f"resolved {len(tid2ab)} team_id -> abbreviation mappings")

    gdate = dict(zip(f.game_id, f.game_date))
    pgs["gdate"] = pgs.gid.map(gdate)
    pgs["season"] = pgs.gid.map(dict(zip(f.game_id, f.season)))
    pgs = pgs.dropna(subset=["gdate", "season"])
    pgs["ab"] = pgs.team_id.map(tid2ab)

    rows, misses = [], []
    for _, gm in smp.iterrows():
        code = (gm.game_date.strftime("%Y%m%d") + "0"
                + bref_code(gm.home, gm.season))
        try:
            h, cached = fetch(code)
        except Exception as e:
            misses.append((code, str(e)[:60])); continue
        pub = parse_inactive(h)
        if not pub:
            misses.append((code, "no Inactive block")); continue
        for ab in (gm.home, gm.away):
            published = {norm(x) for x in pub.get(bref_code(ab, gm.season), [])}
            tg = pgs[(pgs.ab == ab) & (pgs.season == gm.season)]
            if tg.empty:
                continue
            dressed = set(tg[tg.gid == gm.game_id].player_id)
            if not dressed:
                continue
            # ---- ARM A: +-WINDOW team-games (the first pilot's rule)
            dates = np.sort(tg.gdate.unique())
            k = int(np.searchsorted(dates, np.datetime64(gm.game_date)))
            lo, hi = dates[max(0, k - WINDOW)], dates[min(len(dates) - 1,
                                                          k + WINDOW)]
            roster_a = set(tg[(tg.gdate >= lo) & (tg.gdate <= hi)].player_id)
            # ---- ARM B: TENURE. A player is on the roster for this game if
            # their first and last appearance FOR THIS TEAM THIS SEASON bracket
            # the date. A traded-away player's tenure ends before the game, so
            # he drops out; a player returning from a long absence has a later
            # appearance, so he stays in. This targets both observed failure
            # modes directly rather than retuning the window.
            ten = tg.groupby("player_id").gdate.agg(["min", "max"])
            roster_b = set(ten[(ten["min"] <= gm.game_date)
                               & (ten["max"] >= gm.game_date)].index)
            for arm, roster in (("A_window", roster_a), ("B_tenure", roster_b)):
                derived = {norm(names.get(p, "")) for p in (roster - dressed)}
                derived.discard("")
                inter = derived & published
                rows.append(dict(
                    arm=arm, season=gm.season, gid=gm.game_id, team=ab,
                    n_pub=len(published), n_der=len(derived), n_hit=len(inter),
                    prec=len(inter) / len(derived) if derived else np.nan,
                    rec=len(inter) / len(published) if published else np.nan,
                    only_der=";".join(sorted(derived - published))[:120],
                    only_pub=";".join(sorted(published - derived))[:120]))
    d = pd.DataFrame(rows)
    if d.empty:
        print("no comparable team-games; misses:", misses[:5]); return

    print(f"\ncompared {d.gid.nunique()} games x 2 teams x 2 arms "
          f"= {len(d)} rows; fetch misses {len(misses)}")
    for arm, da in d.groupby("arm"):
        agg = da.groupby("season").agg(
            team_games=("prec", "size"), pub=("n_pub", "mean"),
            der=("n_der", "mean"), precision=("prec", "mean"),
            recall=("rec", "mean"))
        print(f"\n=== ARM {arm} — derived vs published inactive ===")
        print(agg.to_string(float_format=lambda v: f"{v:8.3f}"))
        f1 = (2 * da.prec.mean() * da.rec.mean()
              / max(da.prec.mean() + da.rec.mean(), 1e-9))
        print(f"  pooled precision {da.prec.mean():.3f}   "
              f"recall {da.rec.mean():.3f}   F1 {f1:.3f}   "
              f"exact-set match {100*(da.n_hit==da.n_pub).mean():.1f}%")
    b = d[d.arm == "B_tenure"]
    print("\n  ARM B, derived-but-not-published:")
    for e in [x for x in b.only_der if x][:5]:
        print("   ", e)
    print("  ARM B, published-but-not-derived:")
    for e in [x for x in b.only_pub if x][:5]:
        print("   ", e)
    if misses:
        print("\n  fetch misses:", misses[:5])

    d.to_csv(ROOT / "data" / "d250_availability_validation.csv", index=False)
    json.dump({"pooled_precision": float(d.prec.mean()),
               "pooled_recall": float(d.rec.mean()),
               "by_season": agg.reset_index().to_dict("records")},
              open(ROOT / "data" / "d250_availability.json", "w"), default=float)
    print("\nwrote data/d250_availability_validation.csv")


if __name__ == "__main__":
    main()
