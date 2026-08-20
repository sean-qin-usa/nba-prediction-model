#!/usr/bin/env python3
"""REGIME B part 1 (read-only): THREE-WAY star-removal comparison.

Sean's symmetry question: does the D33/D39 star-out science (measured on
1-game benched/injured absences) hold for (b) multi-game absence runs and
(c) PERMANENT removals (traded away mid-season), and how fast does each class
re-equilibrate?

Universe (all 4 seasons in DB, 002 games only, PIT trailing windows):
  star   = player with trailing-10 (min 5, team-scoped, strictly-before) mean
           minutes >= 28 at the event date (starout.py STAR_TRAILING_MIN).
  events per (star, team, season):
    a  : 1-game absence, star plays the team games immediately before+after
    b  : absence run of >= 2 consecutive team games, star returns to SAME team
    c  : PERMANENT-TRADE — star's last played game for T; his next played game
         this season is for a DIFFERENT team (traded/waived+signed)
    c2 : PERMANENT-SHUTDOWN — star never plays again this season for anyone,
         >= 10 team games remain (season-ending injury / shutdown)
  frozen pool = teammates with trailing-10 (min 5) team minutes >= 15 at the
         EVENT date (so post-event arrivals are excluded by construction).
  eval row = frozen-pool player appearing in post-event game k.

Metrics per row (k = games since event start, 1-based):
  frozen-baseline lifts : fga / pre-event mean fga (TOTAL redistribution),
                          per-min rate lift, minutes delta (D39 channel)
  rolling-baseline lift : fga / trailing-10-before-game-k mean fga (what the
                          LIVE trailing baseline still misses at k = residual)
  predicted lift        : D33 softmax S/(S - w_star) over frozen pool + star
                          (v2_usage.npz weights, cap [1, 1.6])
  newcomer share        : fraction of team FGA in game k by NON-frozen-pool,
                          non-star players (new equilibrium structure)
Primary rows exclude games where ANOTHER >=28-trailing star of the same team
is also absent (fresh <= 12d) — single-star science, contamination reported.

Output: printed tables + JSON to the scratchpad. Read-only DB. No prod edits.
"""
import json
import sys
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nbapred.db import connect  # noqa: E402
from nbapred.engine import starout  # noqa: E402

SCRATCH = Path("data/scratch")
PLAYED_MIN = 8.0
EVAL_MIN = 12.0           # gate-parity eval floor for attempt lifts
STAR_MIN = 28.0
ROT_MIN = 15.0
TRAIL_N = 10
TRAIL_MINGAMES = 5
FRESH_DAYS = 12
KMAX = 30


def load(con):
    pg = con.execute("""
        SELECT s.game_id, s.player_id, s.team_id, g.season, g.game_date,
               s.seconds/60.0 AS mins, s.fga
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '002%'
        ORDER BY g.game_date, s.game_id""").fetchdf()
    tg = con.execute("""
        SELECT DISTINCT season, game_id, game_date, team_id
        FROM nba_games WHERE game_id LIKE '002%'
        ORDER BY game_date, game_id""").fetchdf()
    return pg, tg


def main():
    con = connect(read_only=True)
    pg, tg = load(con)
    con.close()
    pg["game_date"] = pd.to_datetime(pg["game_date"])
    tg["game_date"] = pd.to_datetime(tg["game_date"])

    weights = starout.load_usage_weights() or {}
    positions = starout.load_positions()

    # ---- per (player, team) played-game history (mins >= PLAYED_MIN)
    hist = defaultdict(lambda: ([], [], []))  # (dates, mins, fga)
    played = pg[pg.mins >= PLAYED_MIN]
    for r in played.sort_values("game_date").itertuples():
        h = hist[(r.player_id, r.team_id)]
        h[0].append(r.game_date); h[1].append(r.mins); h[2].append(r.fga)
    # any-row lookup for eval games (mins may be < 8)
    rowmap = {(r.game_id, r.player_id): (r.mins, r.fga) for r in pg.itertuples()}
    played_set = defaultdict(set)
    for r in played.itertuples():
        played_set[(r.game_id, r.team_id)].add(r.player_id)
    team_fga = pg.groupby(["game_id", "team_id"])["fga"].sum().to_dict()
    # players per (team): all who ever appear for that team (for newcomer calc)
    # player's played dates for ANY team, for freshness / next-team lookup
    pdates = defaultdict(list)  # player -> [(date, team)]
    for r in played.sort_values("game_date").itertuples():
        pdates[r.player_id].append((r.game_date, r.team_id))

    def trail(player, team, date, n=TRAIL_N):
        d, m, f = hist[(player, team)]
        i = bisect_left(d, date)
        if i < TRAIL_MINGAMES:
            return None
        return (float(np.mean(m[max(0, i - n):i])),
                float(np.mean(f[max(0, i - n):i])), i)

    def played_within(player, team, date, days=FRESH_DAYS):
        d, _, _ = hist[(player, team)]
        i = bisect_left(d, date)
        return i > 0 and (date - d[i - 1]).days <= days

    # ---- team schedules
    sched = {}
    for (season, team), g in tg.groupby(["season", "team_id"]):
        g = g.sort_values("game_date")
        sched[(season, team)] = list(zip(g.game_id, g.game_date))

    # ---- detect events
    events = []  # dict: cls, star, team, season, event_date, games [(k, gid, gdate)], run_len
    star_pt = set()  # (player, team, season) with any >=28 trailing game
    pg_pt = played.merge(tg[["game_id", "season"]].drop_duplicates(), on="game_id",
                         suffixes=("", "_t"))
    for (p, t, s), g in played.merge(
            tg[["game_id"]].assign(sea=tg.season.values), on="game_id"
            ).groupby(["player_id", "team_id", "sea"]):
        if len(g) >= TRAIL_MINGAMES:
            star_pt.add((p, t, s))

    for (season, team), sc in sched.items():
        gid2idx = {gid: i for i, (gid, _) in enumerate(sc)}
        # candidate stars: players with >= TRAIL_MINGAMES played games for team
        cands = [p for (p, t, s) in star_pt if t == team and s == season]
        for star in set(cands):
            d, m, f = hist[(star, team)]
            idxs = sorted(gid2idx[gid] for gid, gd in sc
                          if star in played_set[(gid, team)])
            if len(idxs) < TRAIL_MINGAMES:
                continue
            # absence runs between consecutive played games
            for a_i, b_i in zip(idxs, idxs[1:]):
                if b_i == a_i + 1:
                    continue
                run = list(range(a_i + 1, b_i))
                ev_date = sc[run[0]][1]
                tr = trail(star, team, ev_date)
                if not tr or tr[0] < STAR_MIN:
                    continue
                events.append(dict(
                    cls="a" if len(run) == 1 else "b", star=star, team=team,
                    season=season, event_date=ev_date, run_len=len(run),
                    games=[(k + 1, sc[i][0], sc[i][1])
                           for k, i in enumerate(run[:KMAX])]))
            # permanent: after last played game
            last = idxs[-1]
            post = list(range(last + 1, len(sc)))
            if not post:
                continue
            ev_date = sc[post[0]][1]
            tr = trail(star, team, ev_date)
            if not tr or tr[0] < STAR_MIN:
                continue
            later = [(dd, tt) for (dd, tt) in pdates[star]
                     if dd > sc[last][1] and dd <= sc[-1][1] + pd.Timedelta(days=200)]
            # same-season different-team appearance?
            season_end = max(gd for _, gd in sc)
            later_same_season = [tt for (dd, tt) in later
                                 if dd <= season_end and tt != team]
            if later_same_season:
                cls = "c"
            elif len(post) >= 10 and not any(dd <= season_end for dd, _ in later):
                cls = "c2"
            else:
                continue
            events.append(dict(
                cls=cls, star=star, team=team, season=season,
                event_date=ev_date, run_len=len(post),
                games=[(k + 1, sc[i][0], sc[i][1])
                       for k, i in enumerate(post[:KMAX])]))

    print(f"events: {pd.Series([e['cls'] for e in events]).value_counts().to_dict()}")

    # ---- team star lists (for other-star-out flag): trailing at each event is
    # frozen; approximate team stars by any (p,t,season) reaching >=28 trailing
    # at any point (checked properly per game below via trail()).

    # ---- build eval rows
    rows = []
    for eid, ev in enumerate(events):
        team, season, star = ev["team"], ev["season"], ev["star"]
        ev_date = ev["event_date"]
        # frozen pool
        pool = []
        for (p, t), _ in list(hist.items()):
            if t != team or p == star:
                continue
            tr = trail(p, team, ev_date)
            if tr and tr[0] >= ROT_MIN:
                pool.append((p, tr[0], tr[1]))
        if len(pool) < 3:
            continue
        pool_ids = {p for p, _, _ in pool}
        # predicted softmax lift (uniform-proportional model)
        S = sum(weights.get(p, 1.0) for p in pool_ids) + weights.get(star, 1.0)
        pred_lift = float(np.clip(S / max(S - weights.get(star, 1.0), 1e-9),
                                  1.0, 1.6))
        s_pos = positions.get(int(star))
        # other >=28 stars on the team as of event date
        other_stars = []
        for (p, t), _ in list(hist.items()):
            if t != team or p == star:
                continue
            tr = trail(p, team, ev_date)
            if tr and tr[0] >= STAR_MIN:
                other_stars.append(p)
        for (k, gid, gdate) in ev["games"]:
            ps = played_set[(gid, team)]
            if star in ps:      # safety (shouldn't happen)
                continue
            oso = any(o not in ps and played_within(o, team, gdate)
                      for o in other_stars)
            tfga = team_fga.get((gid, team), np.nan)
            pool_fga = sum(rowmap.get((gid, p), (0, 0))[1] for p in pool_ids)
            newc = 1.0 - pool_fga / tfga if tfga and tfga > 0 else np.nan
            for (p, bmin, bfga) in pool:
                mn, fg = rowmap.get((gid, p), (0.0, 0.0))
                trl = trail(p, team, gdate)
                p_pos = positions.get(int(p))
                same = (None if not (p_pos and s_pos) else
                        bool({c for c in p_pos if c in "GFC"} &
                             {c for c in s_pos if c in "GFC"}))
                rows.append(dict(
                    eid=eid, cls=ev["cls"], season=season, k=k,
                    run_len=ev["run_len"], player=p, star=star, team=team,
                    mins=mn, fga=fg, base_min=bmin, base_fga=bfga,
                    roll_fga=(trl[1] if trl else np.nan),
                    pred_lift=pred_lift, same_pos=same, other_star_out=oso,
                    newcomer_share=newc, appeared=(gid, p) in rowmap))
    df = pd.DataFrame(rows)
    df.to_pickle(SCRATCH / "rw_star_transitions_rows.pkl")
    print(f"eval rows: {len(df)}, events kept: {df.eid.nunique()}")
    print(df.groupby('cls').eid.nunique().to_string())

    # ================= aggregation =================
    rng = np.random.default_rng(7)

    def eboot(sub, col, iters=1500):
        """event-clustered bootstrap mean CI"""
        g = sub.groupby("eid")[col].mean()
        if len(g) < 3:
            return (np.nan, np.nan, np.nan)
        v = g.values
        bs = [np.mean(rng.choice(v, len(v))) for _ in range(iters)]
        return (float(np.mean(v)), *np.percentile(bs, [2.5, 97.5]))

    clean = df[~df.other_star_out].copy()
    played12 = clean[clean.mins >= EVAL_MIN].copy()
    played12["att_lift"] = played12.fga / played12.base_fga.clip(lower=0.5)
    played12["rate_lift"] = (played12.fga / played12.mins) / \
        (played12.base_fga.clip(lower=0.5) / played12.base_min)
    played12["roll_lift"] = played12.fga / played12.roll_fga.clip(lower=0.5)
    p1 = clean[clean.mins >= 1].copy()
    p1["dmin"] = p1.mins - p1.base_min

    out = {"events": pd.Series([e['cls'] for e in events]).value_counts().to_dict(),
           "rows": len(df), "classes": {}}

    KB = {"a": [(1, 1)], "b": [(1, 1), (2, 2), (3, 3), (4, 5), (6, 30)],
          "c": [(1, 3), (4, 7), (8, 12), (13, 20), (21, 30)],
          "c2": [(1, 3), (4, 7), (8, 12), (13, 20), (21, 30)]}
    for cls in ["a", "b", "c", "c2"]:
        sub12, sub1 = played12[played12.cls == cls], p1[p1.cls == cls]
        cinfo = {"n_events": int(clean[clean.cls == cls].eid.nunique()),
                 "pred_lift_mean": float(sub12.groupby('eid').pred_lift.first().mean())
                 if len(sub12) else np.nan, "by_k": []}
        for (lo, hi) in KB[cls]:
            m12 = sub12[(sub12.k >= lo) & (sub12.k <= hi)]
            m1 = sub1[(sub1.k >= lo) & (sub1.k <= hi)]
            e = dict(k=f"{lo}-{hi}", n=len(m12),
                     att_lift=eboot(m12, "att_lift"),
                     rate_lift=eboot(m12, "rate_lift"),
                     roll_lift=eboot(m12, "roll_lift"),
                     dmin=eboot(m1, "dmin"),
                     dmin_same=eboot(m1[m1.same_pos == True], "dmin"),   # noqa: E712
                     dmin_diff=eboot(m1[m1.same_pos == False], "dmin"),  # noqa: E712
                     newcomer=eboot(m12.drop_duplicates(["eid", "k"]), "newcomer_share"))
            cinfo["by_k"].append(e)
        # predicted-vs-actual across events (k window: a:1, b: within-run, c: 1-7)
        if cls == "a":
            w = sub12[sub12.k == 1]
        elif cls == "b":
            w = sub12
        else:
            w = sub12[sub12.k <= 7]
        ge = w.groupby("eid").agg(al=("att_lift", "mean"), pl=("pred_lift", "first"))
        if len(ge) >= 10:
            cinfo["pred_vs_actual_corr"] = float(np.corrcoef(ge.pl, ge.al)[0, 1])
            cinfo["pred_vs_actual_slope"] = float(np.polyfit(ge.pl, ge.al, 1)[0])
        out["classes"][cls] = cinfo
        print(f"\n=== class {cls} (events={cinfo['n_events']}, "
              f"pred_lift={cinfo['pred_lift_mean']:.3f}) ===")
        for e in cinfo["by_k"]:
            def fmt(t):
                return (f"{t[0]:+.3f}[{t[1]:+.3f},{t[2]:+.3f}]"
                        if np.isfinite(t[0]) else "   n/a")
            print(f" k={e['k']:>5} n={e['n']:>5} att {fmt(e['att_lift'])} "
                  f"rate {fmt(e['rate_lift'])} roll {fmt(e['roll_lift'])} "
                  f"dmin {fmt(e['dmin'])} same {fmt(e['dmin_same'])} "
                  f"diff {fmt(e['dmin_diff'])} newc {fmt(e['newcomer'])}")
        if "pred_vs_actual_corr" in cinfo:
            print(f" pred-vs-actual: corr {cinfo['pred_vs_actual_corr']:+.3f} "
                  f"slope {cinfo['pred_vs_actual_slope']:+.3f}")

    # contamination share
    out["other_star_out_share"] = float(df.other_star_out.mean())
    print(f"\nother-star-out contaminated rows: {df.other_star_out.mean():.1%} (excluded)")
    with open(SCRATCH / "rw_star_transitions.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote", SCRATCH / "rw_star_transitions.json")


if __name__ == "__main__":
    main()
