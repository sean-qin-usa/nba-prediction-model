#!/usr/bin/env python3
"""EV STUDY 1 - FAMILIARITY RAMP (descriptive, full hindsight - NOT PIT).

Sean's Luka-trade hypothesis: points travel instantly, passing chemistry does
not. Teammates' finishing on an arriving player's passes should ramp with
familiarity (cumulative shared floor minutes), while the arriver's own scoring
efficiency should not.

Design
------
1. Assist dyads from the PBP v3 cache: every made FG with an "(<name> N AST)"
   description tail -> (passer, scorer, game, date). assistPersonId is never
   populated in this cache, so passer is resolved from the description name via
   a per-(game, team) playerName->personId map built from all PBP actions, with
   a roster last-name fallback.
2. Co-minutes per player pair per game from lineup_stints (same-team pairs
   within each stint). Games with box stats but no stints (~38% of 2024-25 RS
   plus play-ins) are imputed with the overlap proxy min_i*min_j/48. Cumulative
   pair co-minutes as-of each date = cumsum before the current game.
3. Arrival events:
   - midseason: first game for a team with 0 prior games there that season and
     >=15 games for other team(s) earlier that season, season-to-date >=25
     min/game.
   - offseason (2024-25, 2025-26 only; 2022-23 player stats absent): first game
     of a season with a team different from last season's final team, prior
     season >=1500 min and >=25 min/game.
   Measurement window: the arriver's games with the new team from arrival until
   he next appears for a different team (window may span seasons).
4. Measures vs familiarity, bucketed 0-100 / 100-300 / 300-600 / 600+ shared
   minutes. Game-level familiarity = co-minute-weighted mean cumulative pair
   co-minutes with the teammates actually shared with that game. Within-player
   (within-pair for the dyad metric) deltas vs the same unit's own 600+
   steady-state bucket kill talent confounds. Cluster bootstrap by player
   (pair for the dyad metric).

HONESTY NOTES (say-so per repo rules):
- Fully descriptive/hindsight measurement; output parameterizes a future PIT
  adjustment. Nothing here is point-in-time.
- "Teammate eFG% on shots assisted BY the arriver" is not directly measurable:
  assists exist only on MADE shots (no denominator on misses). The chemistry
  channel is therefore measured as (i) teammate eFG% on all their FGA while the
  arriver is on the floor, (ii) the share of teammate makes (arriver on floor)
  assisted by the arriver, and (iii) dyad assists per 36 shared minutes -
  the direct volume ramp Sean hypothesizes.
- Co-minutes only accumulate within DB coverage (2023-24 on): veteran duos'
  true familiarity is right-censored at DB start, which biases the measured
  ramp toward zero (some "0-100" pairs are actually old friends only for
  2023-24 offseason arrivals; midseason and 24/25-26 offseason events are
  clean).

Usage: python scripts/ev_familiarity.py [--rebuild-cache] [--boot 1000]
DB is opened read_only=True. All outputs go to the scratchpad dir + stdout.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import unicodedata
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import duckdb
import numpy as np
import pandas as pd

try:
    import orjson as _json

    def _loads(b):
        return _json.loads(b)
except ImportError:  # pragma: no cover
    def _loads(b):
        return json.loads(b)

DB = "/hdd/steveqin/sean_dev/nba_model/data/nba.duckdb"
PBP_DIR = "/hdd/steveqin/sean_dev/nba_model/data/raw/nba_api/playbyplayv3"
SCRATCH = os.environ.get(
    "EV_SCRATCH",
    "data/scratch",
)
SHOTS_CACHE = os.path.join(SCRATCH, "ev_familiarity_shots.pkl")
RESULTS_JSON = os.path.join(SCRATCH, "ev_familiarity_results.json")

SEASONS = ["2023-24", "2024-25", "2025-26"]
GOOD_TYPES = ("002", "004", "005")  # regular season, playoffs, play-in
BUCKETS = [(0.0, 100.0), (100.0, 300.0), (300.0, 600.0), (600.0, float("inf"))]
BUCKET_LABELS = ["0-100", "100-300", "300-600", "600+"]
AST_RE = re.compile(r"\(([^()]+?) +(\d+) AST\)\s*$")
CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")


def _norm(s: str) -> str:
    """ASCII-fold + casefold: PBP descriptions strip diacritics ('Jokic') while
    playerName keeps them ('Jokić') - without this, EU-star assists are lost."""
    return (
        unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().casefold()
    )


# ---------------------------------------------------------------- PBP parsing
def _elapsed(period: int, clock: str) -> float:
    m = CLOCK_RE.match(clock or "")
    rem = (int(m.group(1)) * 60 + float(m.group(2))) if m else 0.0
    if period <= 4:
        return (period - 1) * 720.0 + (720.0 - rem)
    return 2880.0 + (period - 5) * 300.0 + (300.0 - rem)


def parse_game(args):
    """Return per-shot rows for one game.

    Row: (game_id, team_id, shooter, made, val, dist, t, passer)
    passer: personId of assister, 0 = unassisted, -1 = AST in description but
    name unresolvable.
    """
    path, gid, roster_lastname = args
    try:
        raw = _loads(open(path, "rb").read())
        acts = raw["response"]["game"]["actions"]
    except Exception:
        return gid, [], 0, 0
    name_map: dict[tuple[int, str], set] = defaultdict(set)
    for a in acts:
        pid = a.get("personId") or 0
        tid = a.get("teamId") or 0
        nm = a.get("playerName")
        if pid and tid and nm:
            name_map[(tid, _norm(nm))].add(pid)
            nmi = a.get("playerNameI")
            if nmi:
                name_map[(tid, _norm(nmi))].add(pid)
    rows, n_ast, n_unres = [], 0, 0
    for a in acts:
        if not a.get("isFieldGoal"):
            continue
        at = a.get("actionType")
        if at not in ("Made Shot", "Missed Shot"):
            continue
        made = 1 if at == "Made Shot" else 0
        tid = a.get("teamId") or 0
        shooter = a.get("personId") or 0
        val = a.get("shotValue") or 2
        dist = a.get("shotDistance")
        dist = -1 if dist is None else int(dist)
        t = _elapsed(a.get("period") or 1, a.get("clock"))
        passer = 0
        if made:
            apid = a.get("assistPersonId") or 0
            if apid:
                passer, n_ast = apid, n_ast + 1
            else:
                m = AST_RE.search(a.get("description") or "")
                if m:
                    n_ast += 1
                    cands = name_map.get((tid, _norm(m.group(1))), set())
                    if len(cands) == 1:
                        passer = next(iter(cands))
                    else:
                        fb = roster_lastname.get((tid, _norm(m.group(1))), set())
                        if len(fb) == 1:
                            passer = next(iter(fb))
                        else:
                            passer, n_unres = -1, n_unres + 1
        rows.append((gid, tid, shooter, made, val, dist, t, passer))
    return gid, rows, n_ast, n_unres


def build_shots(con, games: pd.DataFrame, rebuild: bool) -> pd.DataFrame:
    if os.path.exists(SHOTS_CACHE) and not rebuild:
        df = pd.read_pickle(SHOTS_CACHE)
        print(f"[shots] cache hit: {len(df):,} rows, {df.game_id.nunique():,} games")
        return df
    idx_path = os.path.join(PBP_DIR, "_index.json")
    manifest = _loads(open(idx_path, "rb").read())  # {basename: game_id}
    gid2file = {g: os.path.join(PBP_DIR, b) for b, g in manifest.items() if g}
    want = [g for g in games.game_id if g in gid2file]
    print(f"[shots] parsing {len(want):,} PBP files ({len(games) - len(want)} missing)")
    # roster last-name fallback: (game, team, last_name)->pids, unique-only used
    ros = con.execute(
        """
        select s.game_id, s.team_id, p.last_name, s.player_id
        from player_game_stats s join nba_players p using(player_id)
        """
    ).df()
    ros_by_game: dict[str, dict] = defaultdict(lambda: defaultdict(set))
    for gid, tid, ln, pid in ros.itertuples(index=False):
        if ln:
            ros_by_game[gid][(tid, _norm(ln))].add(pid)
    tasks = [(gid2file[g], g, dict(ros_by_game.get(g, {}))) for g in want]
    all_rows, tot_ast, tot_unres = [], 0, 0
    with ProcessPoolExecutor(max_workers=10) as ex:
        for _, rows, n_ast, n_unres in ex.map(parse_game, tasks, chunksize=25):
            all_rows.extend(rows)
            tot_ast += n_ast
            tot_unres += n_unres
    df = pd.DataFrame(
        all_rows,
        columns=["game_id", "team_id", "shooter", "made", "val", "dist", "t", "passer"],
    )
    os.makedirs(SCRATCH, exist_ok=True)
    df.to_pickle(SHOTS_CACHE)
    print(
        f"[shots] {len(df):,} FGA rows, {tot_ast:,} assisted makes, "
        f"{tot_unres:,} unresolved assister names ({tot_unres / max(tot_ast, 1):.2%})"
    )
    return df


# ------------------------------------------------------------ base tables
def load_base(con):
    games = con.execute(
        f"""
        select distinct season, game_id, game_date
        from nba_games
        where season in ({','.join("'" + s + "'" for s in SEASONS)})
          and substr(game_id,1,3) in {GOOD_TYPES}
        order by game_date, game_id
        """
    ).df()
    pg = con.execute(
        """
        select s.*, g.season, g.game_date
        from player_game_stats s
        join (select distinct season, game_id, game_date from nba_games) g using(game_id)
        where s.seconds > 0
        """
    ).df()
    pg = pg[pg.game_id.str[:3].isin(GOOD_TYPES) & pg.season.isin(SEASONS)].copy()
    pg["minutes"] = pg.seconds / 60.0
    pg = pg.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)
    return games, pg


def detect_events(pg: pd.DataFrame) -> pd.DataFrame:
    """Arrival events. kind in {midseason, offseason}."""
    events = []
    last_season_team, last_season_stats = {}, {}
    for season in SEASONS:
        sub = pg[pg.season == season]
        for pid, grp in sub.groupby("player_id", sort=False):
            teams = grp.team_id.to_numpy()
            mins = grp.minutes.to_numpy()
            dates = grp.game_date.to_numpy()
            gids = grp.game_id.to_numpy()
            seen: set = set()
            for i in range(len(grp)):
                t = teams[i]
                if t in seen:
                    continue
                if i > 0:  # midseason candidate: first game for t, i prior games
                    prior_min = mins[:i]
                    if i >= 15 and prior_min.mean() >= 25.0:
                        events.append(
                            dict(player_id=pid, team_id=t, season=season,
                                 arrival_date=dates[i], arrival_game=gids[i],
                                 kind="midseason", prior_mpg=prior_min.mean(),
                                 prior_gp=i)
                        )
                elif season != SEASONS[0]:  # offseason candidate (first game)
                    prev = last_season_team.get(pid)
                    st = last_season_stats.get(pid)
                    if (
                        prev is not None and prev != t and st is not None
                        and st["min"] >= 1500.0 and st["mpg"] >= 25.0
                    ):
                        events.append(
                            dict(player_id=pid, team_id=t, season=season,
                                 arrival_date=dates[i], arrival_game=gids[i],
                                 kind="offseason", prior_mpg=st["mpg"],
                                 prior_gp=st["gp"])
                        )
                seen.add(t)
            last_season_team[pid] = teams[-1]
            last_season_stats[pid] = dict(
                min=mins.sum(), mpg=mins.mean(), gp=len(grp)
            )
    ev = pd.DataFrame(events).sort_values(["arrival_date", "player_id"])
    ev["event_id"] = np.arange(len(ev))
    return ev.reset_index(drop=True)


def event_windows(pg: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """(event_id, game rows) for the arriver with the new team, arrival ->
    until he next plays for a different team (spans seasons)."""
    out = []
    pg_by_player = dict(tuple(pg.groupby("player_id", sort=False)))
    for e in ev.itertuples(index=False):
        g = pg_by_player[e.player_id]
        g = g[(g.game_date >= e.arrival_date)]
        stop = g[g.team_id != e.team_id]
        if len(stop):
            g = g[g.game_date < stop.game_date.iloc[0]]
        g = g[g.team_id == e.team_id].copy()
        g["event_id"] = e.event_id
        out.append(g)
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------ co-minutes
def pair_cominutes(con, pg, games, focus_players: set) -> pd.DataFrame:
    """Per-(game, lo, hi) same-team co-minutes for pairs touching focus set,
    stints when available else box overlap proxy; plus cumulative-before."""
    st = con.execute(
        "select game_id, seconds, home_lineup, away_lineup from lineup_stints "
        "where seconds > 0"
    ).df()
    st = st[st.game_id.isin(set(games.game_id))]
    stint_games = set(st.game_id)
    acc: dict = defaultdict(float)
    for gid, sec, hl, al in st.itertuples(index=False):
        for lu in (hl, al):
            ids = [int(x) for x in lu.split(",") if x]
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = ids[i], ids[j]
                    if a in focus_players or b in focus_players:
                        k = (gid, a, b) if a < b else (gid, b, a)
                        acc[k] += sec
    rows = [(g, a, b, s / 60.0, "stint") for (g, a, b), s in acc.items()]
    # proxy for games without stints
    miss = pg[~pg.game_id.isin(stint_games)]
    n_proxy_games = miss.game_id.nunique()
    for (gid, _tid), grp in miss.groupby(["game_id", "team_id"], sort=False):
        pids = grp.player_id.to_numpy()
        m = grp.minutes.to_numpy()
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                a, b = pids[i], pids[j]
                if a in focus_players or b in focus_players:
                    lo, hi = (a, b) if a < b else (b, a)
                    rows.append((gid, lo, hi, m[i] * m[j] / 48.0, "proxy"))
    co = pd.DataFrame(rows, columns=["game_id", "lo", "hi", "co_min", "src"])
    co = co.merge(games[["game_id", "game_date"]], on="game_id")
    co = co.sort_values(["lo", "hi", "game_date", "game_id"])
    co["cum_before"] = co.groupby(["lo", "hi"])["co_min"].cumsum() - co["co_min"]
    print(
        f"[co-min] {len(co):,} pair-games ({(co.src == 'proxy').mean():.1%} proxy "
        f"rows; {n_proxy_games} games w/o stints)"
    )
    return co


# ------------------------------------------------------------ on-floor
def onfloor_tables(con, games, win_pg):
    """Per (game, arriver): stint intervals + on-floor pts for/against/secs."""
    st = con.execute(
        "select game_id, home_team_id, away_team_id, t_start, t_end, seconds, "
        "home_lineup, away_lineup, home_pts, away_pts from lineup_stints "
        "where seconds > 0 order by game_id, t_start"
    ).df()
    need = win_pg.groupby("game_id").player_id.apply(set).to_dict()
    st = st[st.game_id.isin(need.keys())]
    intervals: dict = defaultdict(list)  # (gid,pid) -> [(t0,t1)]
    floor_pts: dict = defaultdict(lambda: [0.0, 0.0, 0.0])  # for, against, sec
    for r in st.itertuples(index=False):
        want = need[r.game_id]
        hset = {int(x) for x in r.home_lineup.split(",") if x}
        aset = {int(x) for x in r.away_lineup.split(",") if x}
        for pid in want:
            if pid in hset:
                pf, pa = r.home_pts, r.away_pts
            elif pid in aset:
                pf, pa = r.away_pts, r.home_pts
            else:
                continue
            intervals[(r.game_id, pid)].append((r.t_start, r.t_end))
            fp = floor_pts[(r.game_id, pid)]
            fp[0] += pf
            fp[1] += pa
            fp[2] += r.seconds
    return intervals, floor_pts


def on_floor(intervals, gid, pid, t):
    iv = intervals.get((gid, pid))
    if not iv:
        return False
    for t0, t1 in iv:
        if t0 <= t <= t1:
            return True
    return False


# ------------------------------------------------------------ aggregation
def bucket_of(x):
    for k, (lo, hi) in enumerate(BUCKETS):
        if lo <= x < hi:
            return k
    return len(BUCKETS) - 1


def pooled_deltas(stats, min_steady_den):
    """stats: {cluster: {bucket: [num, den]}} -> per-bucket within-cluster
    deltas vs the cluster's own 600+ bucket. Returns arrays for bootstrap."""
    clusters = []
    for c, bd in stats.items():
        sd = bd.get(3)
        if sd is None or sd[1] < min_steady_den:
            continue
        r3 = sd[0] / sd[1]
        row = {}
        for b in range(3):
            if b in bd and bd[b][1] > 0:
                nb, db = bd[b]
                w = db * sd[1] / (db + sd[1])  # harmonic exposure weight
                row[b] = (nb / db - r3, w)
        if row:
            clusters.append((c, row))
    return clusters


def combine(clusters, idx=None):
    """Weighted mean delta per bucket over (a bootstrap sample of) clusters."""
    out = {}
    use = clusters if idx is None else [clusters[i] for i in idx]
    for b in range(3):
        num = den = 0.0
        for _, row in use:
            if b in row:
                d, w = row[b]
                num += w * d
                den += w
        out[b] = (num / den) if den > 0 else np.nan
    return out


def fit_ramp(mids, deltas, weights):
    """delta(m) = A * 2^(-m/h); grid h, closed-form A. Returns (A, h)."""
    m = np.asarray(mids, float)
    d = np.asarray(deltas, float)
    w = np.asarray(weights, float)
    ok = np.isfinite(d) & np.isfinite(m) & (w > 0)
    if ok.sum() < 2:
        return np.nan, np.nan
    m, d, w = m[ok], d[ok], w[ok]
    best = (np.inf, np.nan, np.nan)
    for h in np.geomspace(15, 3000, 240):
        x = np.power(2.0, -m / h)
        A = (w * x * d).sum() / (w * x * x).sum()
        sse = (w * (d - A * x) ** 2).sum()
        if sse < best[0]:
            best = (sse, A, h)
    return best[1], best[2]


def boot_ci(clusters, mids_fn, n_boot, rng):
    """Bootstrap bucket deltas + ramp params over clusters."""
    n = len(clusters)
    d_samp = {b: [] for b in range(3)}
    a_samp, h_samp = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        cb = combine(clusters, idx)
        for b in range(3):
            d_samp[b].append(cb[b])
        mids, ds, ws = mids_fn(idx)
        A, h = fit_ramp(mids, ds, ws)
        a_samp.append(A)
        h_samp.append(h)
    def pct(a):
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        if len(a) == 0:
            return [np.nan, np.nan]
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {b: pct(d_samp[b]) for b in range(3)}, pct(a_samp), pct(h_samp)


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    os.makedirs(SCRATCH, exist_ok=True)

    con = duckdb.connect(DB, read_only=True)
    games, pg = load_base(con)
    ev = detect_events(pg)
    print(f"[events] {len(ev)} arrivals: {ev.kind.value_counts().to_dict()}")
    win = event_windows(pg, ev)
    print(f"[events] {len(win):,} arriver-games in measurement windows")

    focus = set(ev.player_id)
    shots = build_shots(con, games, args.rebuild_cache)
    co = pair_cominutes(con, pg, games, focus)
    intervals, floor_pts = onfloor_tables(con, games, win)

    # directed pair view for arrivers
    co_a = pd.concat(
        [
            co.rename(columns={"lo": "p", "hi": "q"}),
            co.rename(columns={"hi": "p", "lo": "q"}),
        ],
        ignore_index=True,
    )
    co_a = co_a[co_a.p.isin(focus)]
    co_by_game = dict(tuple(co_a.groupby(["game_id", "p"], sort=False)))

    shots_by_game = dict(tuple(shots.groupby("game_id", sort=False)))
    ev_by_id = {e.event_id: e for e in ev.itertuples(index=False)}
    ev_player = {e.event_id: e.player_id for e in ev.itertuples(index=False)}

    # league monthly baselines (descriptive detrend for seasonal drift: the
    # 0-100 bucket clusters in October / right after the trade deadline)
    pg_m = pg.copy()
    pg_m["month"] = pg_m.game_date.astype("datetime64[ns]").dt.strftime("%Y-%m")
    lg = pg_m.groupby("month").agg(pts=("pts", "sum"), fga=("fga", "sum"), fta=("fta", "sum"))
    league_ts = (lg.pts / (2.0 * (lg.fga + 0.44 * lg.fta))).to_dict()
    sh_m = shots.merge(games[["game_id", "game_date"]], on="game_id")
    sh_m["month"] = sh_m.game_date.astype("datetime64[ns]").dt.strftime("%Y-%m")
    sh_m["eff"] = sh_m.made * (1.0 + 0.5 * (sh_m.val == 3))
    league_efg = sh_m.groupby("month").eff.mean().to_dict()

    # metric accumulators: {metric: {cluster: {bucket: [num, den]}}}
    game_metrics = [
        "ast36", "passprox", "ts", "ts_ladj", "pts36", "tm_efg_onfloor",
        "tm_efg_ladj", "ast_share_tm_fgm", "ortg48", "net48", "ast3share",
    ]
    acc = {m: defaultdict(lambda: defaultdict(lambda: [0.0, 0.0])) for m in game_metrics}
    fam_expo = {m: defaultdict(lambda: defaultdict(lambda: [0.0, 0.0])) for m in game_metrics}
    dyad = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))  # pair cluster
    dyad_fam = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    n_games_used = n_games_nofam = 0

    for row in win.itertuples(index=False):
        eid, pid, gid = row.event_id, row.player_id, row.game_id
        pl = ev_player[eid]
        pairs = co_by_game.get((gid, pid))
        if pairs is None or pairs.co_min.sum() <= 0:
            n_games_nofam += 1
            continue
        n_games_used += 1
        wsum = pairs.co_min.sum()
        fam = float((pairs.co_min * pairs.cum_before).sum() / wsum)
        b = bucket_of(fam)
        cum_by_q = dict(zip(pairs.q, pairs.cum_before))
        com_by_q = dict(zip(pairs.q, pairs.co_min))

        def add(metric, num, den, cluster=pl, bb=b, f=fam):
            if den > 0:
                acc[metric][cluster][bb][0] += num
                acc[metric][cluster][bb][1] += den
                fam_expo[metric][cluster][bb][0] += f * den
                fam_expo[metric][cluster][bb][1] += den

        mkey = pd.Timestamp(row.game_date).strftime("%Y-%m")
        add("ast36", row.ast, row.minutes / 36.0)
        add("passprox", row.ast, row.ast + row.fga + row.tov + 0.44 * row.fta)
        tsa = 2.0 * (row.fga + 0.44 * row.fta)
        add("ts", row.pts, tsa)
        add("ts_ladj", row.pts - league_ts.get(mkey, np.nan) * tsa, tsa)
        add("pts36", row.pts, row.minutes / 36.0)
        fp = floor_pts.get((gid, pid))
        if fp is not None and fp[2] > 0:
            add("ortg48", fp[0], fp[2] / 2880.0)
            add("net48", fp[0] - fp[1], fp[2] / 2880.0)

        sg = shots_by_game.get(gid)
        if sg is not None:
            mine = sg[(sg.team_id == row.team_id)]
            # arriver's assists (dyads + mix)
            past = mine[mine.passer == pid]
            n_ast = len(past)
            if n_ast:
                add("ast3share", float((past.val == 3).sum()), float(n_ast))
            for q, cnt in past.groupby("shooter").size().items():
                cb = cum_by_q.get(q)
                cm = com_by_q.get(q, 0.0)
                if cb is None:
                    continue
                pb = bucket_of(cb)
                key = (pid, q)
                dyad[key][pb][0] += cnt
                dyad[key][pb][1] += max(cm, 1.0) / 36.0
                dyad_fam[key][pb][0] += cb * cm
                dyad_fam[key][pb][1] += cm
            # dyad denominators for teammate-games with zero assists
            for q, cm in com_by_q.items():
                if cm > 0 and q not in set(past.shooter):
                    cb = cum_by_q[q]
                    pb = bucket_of(cb)
                    key = (pid, q)
                    dyad[key][pb][1] += cm / 36.0
                    dyad_fam[key][pb][0] += cb * cm
                    dyad_fam[key][pb][1] += cm
            # teammate eFG while arriver on floor
            if (gid, pid) in intervals:
                tm = mine[mine.shooter != pid]
                if len(tm):
                    onf = np.fromiter(
                        (on_floor(intervals, gid, pid, t) for t in tm.t),
                        bool, len(tm),
                    )
                    tm_on = tm[onf]
                    if len(tm_on):
                        eff = (tm_on.made * (1.0 + 0.5 * (tm_on.val == 3))).sum()
                        add("tm_efg_onfloor", float(eff), float(len(tm_on)))
                        add(
                            "tm_efg_ladj",
                            float(eff) - league_efg.get(mkey, np.nan) * len(tm_on),
                            float(len(tm_on)),
                        )
                        fgm_on = int(tm_on.made.sum())
                        ast_by_p = int((tm_on.passer == pid).sum())
                        if fgm_on:
                            add("ast_share_tm_fgm", ast_by_p, fgm_on)

    print(f"[measure] {n_games_used:,} arriver-games measured, {n_games_nofam} lacked co-min rows")

    # thresholds on steady-state denominator per metric
    thr = dict(
        ast36=8.0, passprox=200.0, ts=150.0, ts_ladj=150.0, pts36=8.0,
        tm_efg_onfloor=150.0, tm_efg_ladj=150.0, ast_share_tm_fgm=60.0,
        ortg48=0.1, net48=0.1, ast3share=30.0,
    )
    results = {}
    for m in game_metrics:
        clusters = pooled_deltas(acc[m], thr[m])
        if not clusters:
            continue
        point = combine(clusters)
        # exposure-weighted mean familiarity midpoint per bucket (pooled)
        def mids_fn(idx=None, mm=m, cl=clusters):
            use = cl if idx is None else [cl[i] for i in idx]
            mids, ds, ws = [], [], []
            cb = combine(cl, idx)
            for b in range(3):
                fnum = fden = 0.0
                for c, row in use:
                    fe = fam_expo[mm].get(c, {}).get(b)
                    if fe and b in row:
                        fnum += fe[0]
                        fden += fe[1]
                if fden > 0 and np.isfinite(cb[b]):
                    mids.append(fnum / fden)
                    ds.append(cb[b])
                    ws.append(sum(row[b][1] for _, row in use if b in row))
            return mids, ds, ws
        mids, ds, ws = mids_fn()
        A, h = fit_ramp(mids, ds, ws)
        dci, aci, hci = boot_ci(clusters, mids_fn, args.boot, rng)
        # steady-state pooled level for context
        s_num = sum(bd[3][0] for _, bdrow in [(c, acc[m][c]) for c, _ in clusters] for bd in [bdrow] if 3 in bd)
        s_den = sum(bd[3][1] for _, bdrow in [(c, acc[m][c]) for c, _ in clusters] for bd in [bdrow] if 3 in bd)
        results[m] = dict(
            n_clusters=len(clusters),
            steady_level=s_num / s_den if s_den else np.nan,
            deltas={BUCKET_LABELS[b]: dict(point=point[b], ci=dci[b]) for b in range(3)},
            bucket_mid_fam=mids,
            ramp_A=A, ramp_A_ci=aci, half_life_min=h, half_life_ci=hci,
        )

    # dyad metric (cluster = pair)
    dy_clusters = pooled_deltas(dyad, 100.0 / 36.0)  # >=100 steady co-min
    if dy_clusters:
        point = combine(dy_clusters)
        def dy_mids(idx=None, cl=dy_clusters):
            use = cl if idx is None else [cl[i] for i in idx]
            cb = combine(cl, idx)
            mids, ds, ws = [], [], []
            for b in range(3):
                fnum = fden = 0.0
                for c, row in use:
                    fe = dyad_fam.get(c, {}).get(b)
                    if fe and b in row:
                        fnum += fe[0]
                        fden += fe[1]
                if fden > 0 and np.isfinite(cb[b]):
                    mids.append(fnum / fden)
                    ds.append(cb[b])
                    ws.append(sum(row[b][1] for _, row in use if b in row))
            return mids, ds, ws
        mids, ds, ws = dy_mids()
        A, h = fit_ramp(mids, ds, ws)
        dci, aci, hci = boot_ci(dy_clusters, dy_mids, args.boot, rng)
        s_num = sum(dyad[c][3][0] for c, _ in dy_clusters if 3 in dyad[c])
        s_den = sum(dyad[c][3][1] for c, _ in dy_clusters if 3 in dyad[c])
        results["dyad_ast_per36co"] = dict(
            n_clusters=len(dy_clusters),
            steady_level=s_num / s_den if s_den else np.nan,
            deltas={BUCKET_LABELS[b]: dict(point=point[b], ci=dci[b]) for b in range(3)},
            bucket_mid_fam=mids,
            ramp_A=A, ramp_A_ci=aci, half_life_min=h, half_life_ci=hci,
        )

    meta = dict(
        n_events=int(len(ev)),
        by_kind=ev.kind.value_counts().to_dict(),
        n_arriver_games=int(n_games_used),
        seasons=SEASONS,
        note=(
            "DESCRIPTIVE with full hindsight (not PIT). Chemistry channel is "
            "teammate on-floor eFG / arriver-assisted share / dyad assist "
            "volume; true 'eFG on shots assisted by arriver' is unobservable "
            "(assists only exist on makes). Within-player (dyad: within-pair) "
            "deltas vs own 600+ co-min steady state; cluster bootstrap by "
            "player (pair). Co-min imputed via min_i*min_j/48 for games "
            "without stints. Familiarity is right-censored at 2023-10 DB start."
        ),
    )
    out = dict(meta=meta, results=results)

    def _clean(o):
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else round(float(o), 6)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o

    with open(RESULTS_JSON, "w") as f:
        json.dump(_clean(out), f, indent=1)
    print(json.dumps(_clean(out), indent=1))
    print(f"[done] results -> {RESULTS_JSON}")


if __name__ == "__main__":
    main()
