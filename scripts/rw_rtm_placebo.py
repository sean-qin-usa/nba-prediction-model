#!/usr/bin/env python3
"""REGIME B calibration (read-only): PLACEBO null for the ratio metrics used in
rw_star_transitions.py / rw_star_joins.py.

E[fga_game / trailing-mean] > 1 under the null (mean reversion + ratio
convexity + within-season drift), so the event "lifts" must be read AGAINST
this placebo, not against 1.0. Same pool construction, same frozen-baseline
and metrics, but events are ARBITRARY team-game indices (20, 35, 50, 62) with
no star condition. Placebos overlapping real star events are NOT excluded
(makes placebo slightly event-contaminated -> differences are conservative).

Outputs: att factor / rate factor / dmin by k-bucket and by usage tercile,
split early (idx 20/35) vs late (idx 50/62) to calibrate seasonal drift.
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
PLAYED_MIN, EVAL_MIN, ROT_MIN = 8.0, 12.0, 15.0
TRAIL_N, TRAIL_MINGAMES, KMAX = 10, 5, 30
PLACEBO_IDX = [20, 35, 50, 62]
KB = [(1, 1), (1, 3), (4, 7), (8, 12), (13, 20), (21, 30)]


def main():
    con = connect(read_only=True)
    pg = con.execute("""
        SELECT s.game_id, s.player_id, s.team_id, g.season, g.game_date,
               s.seconds/60.0 AS mins, s.fga
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '002%' ORDER BY g.game_date, s.game_id""").fetchdf()
    tg = con.execute("""
        SELECT DISTINCT season, game_id, game_date, team_id
        FROM nba_games WHERE game_id LIKE '002%'
        ORDER BY game_date, game_id""").fetchdf()
    con.close()
    pg["game_date"] = pd.to_datetime(pg["game_date"])
    tg["game_date"] = pd.to_datetime(tg["game_date"])

    hist = defaultdict(lambda: ([], [], []))
    played = pg[pg.mins >= PLAYED_MIN]
    for r in played.sort_values("game_date").itertuples():
        h = hist[(r.player_id, r.team_id)]
        h[0].append(r.game_date); h[1].append(r.mins); h[2].append(r.fga)
    rowmap = {(r.game_id, r.player_id): (r.mins, r.fga) for r in pg.itertuples()}

    def trail(player, team, date, n=TRAIL_N):
        d, m, f = hist[(player, team)]
        i = bisect_left(d, date)
        if i < TRAIL_MINGAMES:
            return None
        return (float(np.mean(m[max(0, i - n):i])),
                float(np.mean(f[max(0, i - n):i])), i)

    rows = []
    eid = 0
    for (season, team), g in tg.groupby(["season", "team_id"]):
        sc = list(zip(*[iter([])]))  # placeholder
        g = g.sort_values("game_date")
        sc = list(zip(g.game_id, g.game_date))
        for idx in PLACEBO_IDX:
            if idx >= len(sc):
                continue
            ev_date = sc[idx][1]
            pool = []
            for (p, t), _ in list(hist.items()):
                if t != team:
                    continue
                tr = trail(p, team, ev_date)
                if tr and tr[0] >= ROT_MIN:
                    pool.append((p, tr[0], tr[1]))
            if len(pool) < 3:
                continue
            bf = np.array([b for _, _, b in pool])
            terc = np.searchsorted(np.quantile(bf, [1 / 3, 2 / 3]), bf, side="right")
            tmap = {p: int(tt) for (p, _, _), tt in zip(pool, terc)}
            eid += 1
            for k, i in enumerate(range(idx, min(idx + KMAX, len(sc)))):
                gid, gd = sc[i]
                for (p, bmin, bfga) in pool:
                    mn, fg = rowmap.get((gid, p), (0.0, 0.0))
                    rows.append(dict(eid=eid, phase="early" if idx <= 35 else "late",
                                     k=k + 1, player=p, mins=mn, fga=fg,
                                     base_min=bmin, base_fga=bfga, terc=tmap[p]))
    df = pd.DataFrame(rows)
    print(f"placebo rows: {len(df)}, placebo events: {df.eid.nunique()}")
    rng = np.random.default_rng(3)

    def eboot(sub, col, iters=800):
        g = sub.groupby("eid")[col].mean()
        if len(g) < 3:
            return (np.nan, np.nan, np.nan)
        v = g.values
        bs = [np.mean(rng.choice(v, len(v))) for _ in range(iters)]
        return (float(np.mean(v)), *np.percentile(bs, [2.5, 97.5]))

    def fmt(t):
        return (f"{t[0]:+.3f}[{t[1]:+.3f},{t[2]:+.3f}]"
                if np.isfinite(t[0]) else "   n/a")

    out = {}
    for phase in ["early", "late"]:
        sub = df[df.phase == phase]
        p12 = sub[sub.mins >= EVAL_MIN].copy()
        p12["att_f"] = p12.fga / p12.base_fga.clip(lower=0.5)
        p12["rate_f"] = (p12.fga / p12.mins) / (p12.base_fga.clip(lower=0.5) / p12.base_min)
        p1 = sub[sub.mins >= 1].copy()
        p1["dmin"] = p1.mins - p1.base_min
        ph = {"by_k": [], "by_tercile": {}}
        print(f"\n=== placebo {phase} (events={sub.eid.nunique()}) ===")
        for (lo, hi) in KB:
            m12 = p12[(p12.k >= lo) & (p12.k <= hi)]
            m1 = p1[(p1.k >= lo) & (p1.k <= hi)]
            e = dict(k=f"{lo}-{hi}", n=len(m12), att_f=eboot(m12, "att_f"),
                     rate_f=eboot(m12, "rate_f"), dmin=eboot(m1, "dmin"))
            ph["by_k"].append(e)
            print(f" k={e['k']:>5} n={e['n']:>6} att {fmt(e['att_f'])} "
                  f"rate {fmt(e['rate_f'])} dmin {fmt(e['dmin'])}")
        w = p12[p12.k <= 12]
        print(" terciles (k<=12 att factor):")
        for tc, lab in ((0, "low"), (1, "mid"), (2, "top")):
            r = eboot(w[w.terc == tc], "att_f")
            ph["by_tercile"][lab] = r
            print(f"   {lab:4} {fmt(r)}")
        out[phase] = ph
    with open(SCRATCH / "rw_rtm_placebo.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote", SCRATCH / "rw_rtm_placebo.json")


if __name__ == "__main__":
    main()
