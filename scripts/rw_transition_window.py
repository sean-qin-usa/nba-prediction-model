#!/usr/bin/env python3
"""REGIME B part 3 (read-only): TEAM-LEVEL re-convergence to market accuracy
after star trades (OUT-team and IN-team).

Events (re-derived, same construction as rw_star_transitions/rw_star_joins):
  OUT : team T's first game after a >=28-trailing-min star's last played game
        for T, star's next same-season game is for a different team (class c)
  IN  : team B's game where a star (>=28 trailing min at old team) plays his
        first game for B mid-season (arrival)
Benchmark: data/capstone_pergame_tank.csv (production carry+tank+sched,
2023-24..2025-26; game_id zfill). Per-game excess log loss vs de-vig close:
d = L_us - L_mkt, assigned to every capstone game the affected team plays.

Windows by games-since-event k: pre(-10..-1), 1-5, 6-10, 11-15, 16-25 with
event-clustered bootstrap; pooled all-games d as global reference. Market data
used ONLY as benchmark (G2 compliant). Read-only DB; outputs to scratchpad.
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

SCRATCH = Path("data/scratch")
PLAYED_MIN, TRAIL_N, TRAIL_MINGAMES, STAR_MIN = 8.0, 10, 5, 28.0
WINDOWS = [("pre -10..-1", -10, -1), ("k 1-5", 1, 5), ("k 6-10", 6, 10),
           ("k 11-15", 11, 15), ("k 16-25", 16, 25)]


def main():
    con = connect(read_only=True)
    pg = con.execute("""
        SELECT s.game_id, s.player_id, s.team_id, g.season, g.game_date,
               s.seconds/60.0 AS mins
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '002%' ORDER BY g.game_date, s.game_id""").fetchdf()
    tg = con.execute("""
        SELECT DISTINCT season, game_id, game_date, team_id
        FROM nba_games WHERE game_id LIKE '002%'
        ORDER BY game_date, game_id""").fetchdf()
    abbrev = con.execute("""
        SELECT DISTINCT team_id, team_abbrev FROM nba_games""").fetchdf()
    con.close()
    pg["game_date"] = pd.to_datetime(pg["game_date"])
    tg["game_date"] = pd.to_datetime(tg["game_date"])
    ab = dict(zip(abbrev.team_id, abbrev.team_abbrev))

    hist = defaultdict(lambda: ([], []))
    played = pg[pg.mins >= PLAYED_MIN]
    for r in played.sort_values("game_date").itertuples():
        h = hist[(r.player_id, r.team_id)]
        h[0].append(r.game_date); h[1].append(r.mins)
    pdates = defaultdict(list)
    for r in played.sort_values("game_date").itertuples():
        pdates[r.player_id].append((r.game_date, r.team_id))
    played_set = defaultdict(set)
    for r in played.itertuples():
        played_set[(r.game_id, r.team_id)].add(r.player_id)

    def trail_min(player, team, date):
        d, m = hist[(player, team)]
        i = bisect_left(d, date)
        if i < TRAIL_MINGAMES:
            return None
        return float(np.mean(m[max(0, i - TRAIL_N):i]))

    sched = {}
    for (season, team), g in tg.groupby(["season", "team_id"]):
        g = g.sort_values("game_date")
        sched[(season, team)] = list(zip(g.game_id, g.game_date))

    # ---- events
    events = []  # (kind, team, season, event_idx in sched)
    for (p, team), (dts, mns) in list(hist.items()):
        # OUT candidates: last played game for team, next game other team same season
        sea_rows = tg[(tg.team_id == team) & (tg.game_date == dts[0])]
        # arrivals: first game for this team with a prior same-season other-team game
        first_b = dts[0]
        sea = tg[(tg.team_id == team) & (tg.game_date == first_b)]["season"]
        if len(sea):
            season = sea.iloc[0]
            sc = sched[(season, team)]
            prior = [(dd, tt) for (dd, tt) in pdates[p]
                     if dd < first_b and tt != team]
            if prior and prior[-1][0] >= sc[0][1]:
                trm = trail_min(p, prior[-1][1], prior[-1][0] + pd.Timedelta(days=1))
                if trm and trm >= STAR_MIN:
                    idx = next(i for i, (gid, gd) in enumerate(sc) if gd == first_b)
                    events.append(dict(kind="IN", star=p, team=team,
                                       season=season, idx=idx))
        # OUT: last game
        last = dts[-1]
        sea = tg[(tg.team_id == team) & (tg.game_date == last)]["season"]
        if not len(sea):
            continue
        season = sea.iloc[0]
        sc = sched[(season, team)]
        season_end = sc[-1][1]
        later = [tt for (dd, tt) in pdates[p]
                 if dd > last and dd <= season_end and tt != team]
        if not later:
            continue
        trm = trail_min(p, team, last + pd.Timedelta(days=1))
        if not trm or trm < STAR_MIN:
            continue
        lidx = next(i for i, (gid, gd) in enumerate(sc) if gd == last)
        if lidx + 1 < len(sc):
            events.append(dict(kind="OUT", star=p, team=team,
                               season=season, idx=lidx + 1))

    cap = pd.read_csv(Path(__file__).resolve().parent.parent /
                      "data/capstone_pergame_tank.csv",
                      dtype={"game_id": str})
    cap["game_id"] = cap.game_id.str.zfill(10)
    eps = 1e-12
    cap["d"] = (-(np.where(cap.y == 1, np.log(cap.p_us + eps),
                           np.log(1 - cap.p_us + eps))) +
                (np.where(cap.y == 1, np.log(cap.p_mkt + eps),
                          np.log(1 - cap.p_mkt + eps))))
    dmap = dict(zip(cap.game_id, cap.d))
    cap_seasons = set(cap.season)
    print(f"pooled all-games d (L_us - L_mkt): {cap.d.mean():+.5f} (n={len(cap)})")

    ev_kept = [e for e in events if e["season"] in cap_seasons]
    print("events in capstone seasons:",
          pd.Series([e['kind'] for e in ev_kept]).value_counts().to_dict())
    rows = []
    for eid, e in enumerate(ev_kept):
        sc = sched[(e["season"], e["team"])]
        for i, (gid, gd) in enumerate(sc):
            k = i - e["idx"] + 1 if i >= e["idx"] else i - e["idx"]
            if gid in dmap and -10 <= k <= 25 and k != 0:
                rows.append(dict(eid=eid, kind=e["kind"], k=k, d=dmap[gid],
                                 team=ab.get(e["team"]), season=e["season"],
                                 star=e["star"]))
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(5)

    def eboot(sub, iters=2000):
        g = sub.groupby("eid")["d"].mean()
        if len(g) < 3:
            return (np.nan, np.nan, np.nan, 0)
        v = g.values
        bs = [np.mean(rng.choice(v, len(v))) for _ in range(iters)]
        return (float(np.mean(v)), *np.percentile(bs, [2.5, 97.5]), len(g))

    out = {"pooled_d": float(cap.d.mean()),
           "events": pd.Series([e['kind'] for e in ev_kept]).value_counts().to_dict(),
           "windows": {}}
    for kind in ["OUT", "IN", "BOTH"]:
        sub = df if kind == "BOTH" else df[df.kind == kind]
        print(f"\n=== {kind} (events={sub.eid.nunique()}) — d = L_us - L_mkt, "
              f"+ = market better ===")
        res = []
        for (lab, lo, hi) in WINDOWS:
            m = sub[(sub.k >= lo) & (sub.k <= hi)]
            mu, cl, ch, ne = eboot(m)
            res.append(dict(window=lab, n=len(m), events=ne, d=mu,
                            ci=[cl, ch]))
            print(f" {lab:>11} n={len(m):>4} ev={ne:>3} d {mu:+.5f} "
                  f"[{cl:+.5f},{ch:+.5f}]")
        out["windows"][kind] = res
    # per-k fine shape for BOTH
    fine = df.groupby("k")["d"].agg(["mean", "count"])
    out["fine_shape_both"] = {int(k): [float(m), int(c)]
                              for k, (m, c) in fine.iterrows()}
    with open(SCRATCH / "rw_transition_window.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\nwrote", SCRATCH / "rw_transition_window.json")


if __name__ == "__main__":
    main()
