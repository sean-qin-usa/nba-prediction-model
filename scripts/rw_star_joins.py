#!/usr/bin/env python3
"""REGIME B part 2 (read-only): STAR JOINS — incumbent usage COMPRESSION.

Mirror of rw_star_transitions.py: when a >=28-trailing-min star ARRIVES
mid-season (his previous played game this season was for another team), what
happens to the RECEIVING team's incumbents? D60 measured the ARRIVER; this
measures the INCUMBENTS.

  event   = star's first played game (mins>=8) for new team B, having played
            for a different team earlier the same season; trailing-10 minutes
            at the OLD team >= 28 (min 5 games) as of the move.
  pool    = B players with trailing-10 (min 5) B-minutes >= 15 at arrival,
            excluding the arriver and any same-window departures.
  departed= frozen-pool players whose last B game falls within [-1, +5] days
            of arrival (other side of the trade) — dropped from eval rows,
            used in the net inverse-softmax prediction.
  eval    = pool player appearing in B game k (k=1.. games since arrival,
            counting only games the ARRIVER PLAYS >= 8 min — compression is
            conditional on the star being on the floor).

Metrics: frozen-baseline attempt factor (fga / pre-arrival mean), per-min rate
factor, minutes delta (same-pos vs diff-pos vs arriver — D39 mirror: same-pos
should LOSE more), rolling-baseline factor (residual the live trailing rates
miss), predicted compression = S_old / (S_old - W_departed + w_star) from
v2_usage.npz (uniform-proportional inverse softmax), uniformity test by
incumbent usage tercile (D38 mirror).

Secondary tier: STAR_MIN=24 arrivals (sample-size sensitivity, pre-specified
single alternative — not a sweep).

Read-only DB; outputs to scratchpad.
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
EVAL_MIN = 12.0
ROT_MIN = 15.0
TRAIL_N, TRAIL_MINGAMES = 10, 5
KMAX = 30
KB = [(1, 3), (4, 7), (8, 12), (13, 20), (21, 30)]


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
    weights = starout.load_usage_weights() or {}
    positions = starout.load_positions()

    hist = defaultdict(lambda: ([], [], []))
    played = pg[pg.mins >= PLAYED_MIN]
    for r in played.sort_values("game_date").itertuples():
        h = hist[(r.player_id, r.team_id)]
        h[0].append(r.game_date); h[1].append(r.mins); h[2].append(r.fga)
    rowmap = {(r.game_id, r.player_id): (r.mins, r.fga) for r in pg.itertuples()}
    played_set = defaultdict(set)
    for r in played.itertuples():
        played_set[(r.game_id, r.team_id)].add(r.player_id)
    pdates = defaultdict(list)
    for r in played.sort_values("game_date").itertuples():
        pdates[r.player_id].append((r.game_date, r.team_id))

    def trail(player, team, date, n=TRAIL_N):
        d, m, f = hist[(player, team)]
        i = bisect_left(d, date)
        if i < TRAIL_MINGAMES:
            return None
        return (float(np.mean(m[max(0, i - n):i])),
                float(np.mean(f[max(0, i - n):i])), i)

    sched = {}
    for (season, team), g in tg.groupby(["season", "team_id"]):
        g = g.sort_values("game_date")
        sched[(season, team)] = list(zip(g.game_id, g.game_date))

    # ---- detect arrival events at both star tiers
    events = []
    for (p, team_b), (dts, mns, fgs) in list(hist.items()):
        first_b = dts[0]
        prior = [(dd, tt) for (dd, tt) in pdates[p] if dd < first_b and tt != team_b]
        if not prior:
            continue
        d_prev, team_a = prior[-1]
        # same season? find season of first_b
        sea = tg[(tg.team_id == team_b) &
                 (tg.game_date == first_b)]["season"]
        if not len(sea):
            continue
        season = sea.iloc[0]
        season_games = sched[(season, team_b)]
        if not (season_games[0][1] <= d_prev):   # prev game must be same season
            continue
        tr_a = trail(p, team_a, d_prev + pd.Timedelta(days=1))
        if not tr_a or tr_a[0] < 24.0:
            continue
        events.append(dict(star=p, team=team_b, old_team=team_a, season=season,
                           arrive=first_b, star_trail=tr_a[0],
                           tier="28" if tr_a[0] >= 28 else "24"))
    ec = pd.Series([e["tier"] for e in events]).value_counts().to_dict()
    print(f"arrival events: {ec}")

    rows = []
    for eid, ev in enumerate(events):
        team, season, star, arrive = ev["team"], ev["season"], ev["star"], ev["arrive"]
        sc = sched[(season, team)]
        pool = []
        for (p, t), _ in list(hist.items()):
            if t != team or p == star:
                continue
            tr = trail(p, team, arrive)
            if tr and tr[0] >= ROT_MIN:
                pool.append((p, tr[0], tr[1]))
        if len(pool) < 3:
            continue
        # departed pool players (other side of trade window)
        departed = set()
        for (p, bmin, bfga) in pool:
            d, _, _ = hist[(p, team)]
            last_b = d[-1] if d else None
            if last_b is not None and -1 <= (last_b - arrive).days <= 5:
                # check they actually stop playing for B after that
                later_b = [dd for dd in d if dd > arrive + pd.Timedelta(days=5)]
                if not later_b:
                    departed.add(p)
        pool_keep = [(p, bm, bf) for (p, bm, bf) in pool if p not in departed]
        S_old = sum(weights.get(p, 1.0) for p, _, _ in pool)
        W_dep = sum(weights.get(p, 1.0) for p in departed)
        w_star = weights.get(star, 1.0)
        pred = float(np.clip(S_old / max(S_old - W_dep + w_star, 1e-9), 0.6, 1.0))
        pred_simple = float(np.clip(S_old / (S_old + w_star), 0.6, 1.0))
        s_pos = positions.get(int(star))
        # usage terciles within event by baseline fga
        bf = np.array([b for _, _, b in pool_keep])
        terc = np.searchsorted(np.quantile(bf, [1 / 3, 2 / 3]), bf, side="right")
        tmap = {p: int(t) for (p, _, _), t in zip(pool_keep, terc)}
        # arriver's played B games after arrival
        star_games = set()
        d, m, _ = hist[(star, team)]
        for dd in d:
            star_games.add(dd)
        post = [(i, gid, gd) for i, (gid, gd) in enumerate(sc) if gd >= arrive]
        k = 0
        for (_, gid, gd) in post:
            if star not in played_set[(gid, team)]:
                continue    # only games the arriver plays
            k += 1
            if k > KMAX:
                break
            for (p, bmin, bfga) in pool_keep:
                mn, fg = rowmap.get((gid, p), (0.0, 0.0))
                trl = trail(p, team, gd)
                p_pos = positions.get(int(p))
                same = (None if not (p_pos and s_pos) else
                        bool({c for c in p_pos if c in "GFC"} &
                             {c for c in s_pos if c in "GFC"}))
                rows.append(dict(
                    eid=eid, tier=ev["tier"], k=k, player=p, star=star,
                    team=team, season=season, mins=mn, fga=fg,
                    base_min=bmin, base_fga=bfga,
                    roll_fga=(trl[1] if trl else np.nan),
                    pred=pred, pred_simple=pred_simple, same_pos=same,
                    terc=tmap[p]))
    df = pd.DataFrame(rows)
    df.to_pickle(SCRATCH / "rw_star_joins_rows.pkl")
    print(f"eval rows: {len(df)}, events: {df.eid.nunique()}")

    rng = np.random.default_rng(11)

    def eboot(sub, col, iters=1500):
        g = sub.groupby("eid")[col].mean()
        if len(g) < 3:
            return (np.nan, np.nan, np.nan)
        v = g.values
        bs = [np.mean(rng.choice(v, len(v))) for _ in range(iters)]
        return (float(np.mean(v)), *np.percentile(bs, [2.5, 97.5]))

    def fmt(t):
        return (f"{t[0]:+.3f}[{t[1]:+.3f},{t[2]:+.3f}]"
                if np.isfinite(t[0]) else "   n/a")

    out = {"events": ec, "tiers": {}}
    for tier_lab, tsub in (("28", df[df.tier == "28"]),
                           ("24+ (both tiers)", df)):
        p12 = tsub[tsub.mins >= EVAL_MIN].copy()
        p12["att_f"] = p12.fga / p12.base_fga.clip(lower=0.5)
        p12["rate_f"] = (p12.fga / p12.mins) / (p12.base_fga.clip(lower=0.5) / p12.base_min)
        p12["roll_f"] = p12.fga / p12.roll_fga.clip(lower=0.5)
        p1 = tsub[tsub.mins >= 1].copy()
        p1["dmin"] = p1.mins - p1.base_min
        t = {"n_events": int(tsub.eid.nunique()),
             "pred_net": float(tsub.groupby('eid').pred.first().mean()),
             "pred_simple": float(tsub.groupby('eid').pred_simple.first().mean()),
             "by_k": []}
        print(f"\n=== arrivals tier {tier_lab} (events={t['n_events']}, "
              f"pred_net={t['pred_net']:.3f}, pred_simple={t['pred_simple']:.3f}) ===")
        for (lo, hi) in KB:
            m12 = p12[(p12.k >= lo) & (p12.k <= hi)]
            m1 = p1[(p1.k >= lo) & (p1.k <= hi)]
            e = dict(k=f"{lo}-{hi}", n=len(m12),
                     att_f=eboot(m12, "att_f"), rate_f=eboot(m12, "rate_f"),
                     roll_f=eboot(m12, "roll_f"), dmin=eboot(m1, "dmin"),
                     dmin_same=eboot(m1[m1.same_pos == True], "dmin"),   # noqa: E712
                     dmin_diff=eboot(m1[m1.same_pos == False], "dmin"))  # noqa: E712
            t["by_k"].append(e)
            print(f" k={e['k']:>5} n={e['n']:>5} att {fmt(e['att_f'])} "
                  f"rate {fmt(e['rate_f'])} roll {fmt(e['roll_f'])} "
                  f"dmin {fmt(e['dmin'])} same {fmt(e['dmin_same'])} "
                  f"diff {fmt(e['dmin_diff'])}")
        # uniformity: attempt factor by usage tercile, k<=12
        w = p12[p12.k <= 12]
        t["by_tercile"] = {}
        print(" uniformity (k<=12 att factor by baseline-usage tercile):")
        for tc, lab in ((0, "low"), (1, "mid"), (2, "top")):
            r = eboot(w[w.terc == tc], "att_f")
            t["by_tercile"][lab] = r
            print(f"   {lab:4} {fmt(r)}")
        # predicted vs actual across events (k<=12)
        ge = w.groupby("eid").agg(af=("att_f", "mean"), pl=("pred", "first"))
        if len(ge) >= 10:
            t["pred_vs_actual_corr"] = float(np.corrcoef(ge.pl, ge.af)[0, 1])
            t["pred_vs_actual_slope"] = float(np.polyfit(ge.pl, ge.af, 1)[0])
            print(f" pred-vs-actual (k<=12): corr {t['pred_vs_actual_corr']:+.3f} "
                  f"slope {t['pred_vs_actual_slope']:+.3f}")
        out["tiers"][tier_lab] = t

    with open(SCRATCH / "rw_star_joins.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote", SCRATCH / "rw_star_joins.json")


if __name__ == "__main__":
    main()
