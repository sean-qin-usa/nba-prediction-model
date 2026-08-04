#!/usr/bin/env python3
"""CALIBRATION + VALIDATION (read-only) for the D85 roster-delta term
(nbapred/engine/starout.py trade-aware extension; regime-B science, journal
nba-star-transitions bd0208 / b89f4a / f3286e).

Universes (all 4 DB seasons, 002 only, PIT trailing windows, other-star-out
contaminated rows excluded):

  DEP : seller-team incumbents after a PERMANENT departure (>=28-trailing-min
        star's last team game, he then plays for ANOTHER team the same season
        = trade EXECUTED, live-observable). Rows = frozen pool (trailing >=15
        at event) in POST-EXECUTION team games k<=30. The shipped D83 module
        keeps lifting these rows while the absence is fresh (<=12d) — regime-B
        showed the lift collapses to placebo at execution and live trailing
        rates OVER-project incumbents ~6-10% persistently.
  ARR : buyer-team incumbents after a mid-season star arrival (>=28 trailing
        at the old team as of the move). Rows = frozen pool in games the
        ARRIVER PLAYS, k = arriver-played game index <= 30. Regime-B residual
        vs the trailing baseline: ~0 k1-3, -12% k4-7, -7% k8-12, -3..-5%
        k13-30; bottom-usage tercile shielded; minutes -2.2 adj (same-pos
        -3.6 vs diff -2.5).

Row filters (gate parity with validate_starout.py): eval player >=12 realized
min, rates profile n_games>=8 & proj_min>=15 from player_rates_from_stats
(the LIVE baseline, before=game_date).

WALK-FORWARD RESIDUAL CALIBRATION (the D83 double-count lesson): shapes come
from the regime-B science; MAGNITUDE scales are fit on TRAIN events (event
date <= q0.6 per universe) as live-baseline residuals NETTED against the same
pools' PRE-EVENT windows (k -10..-1; kills any global baseline bias), then
FROZEN (rounded) and scored on HELD-OUT events only.

Arms per row (same seed -> paired MC draws):
  base : the SHIPPED D83 behavior (star-out lift applied to DEP rows while
         the departed star is fresh <=12d; nothing on ARR rows)
  supp : suppression only (no lift post-execution; ARR untouched)
  rd   : the full roster-delta term (suppression + DEP attenuation + ARR
         compression w/ tercile shield + ARR minutes tilt)

Metrics: CRPS points/rebounds/assists + attempts-side shots Poisson LL,
paired per-row deltas vs base, PLAYER-CLUSTERED bootstrap (2000x, 95% CI),
delta > 0 = arm better. Held-out tables are the honest read.

DECLARED CAVEATS: event set is box-score-derived (execution = first game for
the new team; the same signal the live module uses — no oracle here); pool
and star definitions identical across arms so contrasts are fair; v2_usage
weights are pooled-season (D57: softmax science survives PIT refit).
"""
import sys
import warnings
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from nbapred.db import connect
from nbapred.engine import starout
from nbapred.engine.props import player_rates_from_stats, simulate_player

SIMS = 2000
BOOT = 2000
MARKETS = ("points", "rebounds", "assists")
ARMS = ("supp", "rd")          # deltas reported vs base
TRAIN_Q = 0.6
PLAYED_MIN, EVAL_MIN, STAR_MIN, ROT_MIN = 8.0, 12.0, 28.0, 15.0
TRAIL_N, TRAIL_MINGAMES, FRESH_DAYS, KMAX = 10, 5, 12, 30
ARR_BUCKETS = ((1, 3), (4, 7), (8, 12), (13, 30))
DEP_BUCKETS = ((1, 7), (8, 15), (16, 30))


def crps(s, y):
    s = np.sort(s); n = len(s)
    return float(np.mean(np.abs(s - y)) -
                 0.5 * (2 * np.arange(1, n + 1) - n - 1) @ s / n ** 2)


def cluster_boot(deltas, players, iters=BOOT, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(players)
    per = {p: deltas[players == p] for p in uniq}
    means = []
    for _ in range(iters):
        pick = rng.choice(uniq, len(uniq), replace=True)
        means.append(np.concatenate([per[p] for p in pick]).mean())
    return np.percentile(means, [2.5, 97.5])


def main(max_events=None):
    con = connect(read_only=True)
    pg = con.execute("""
        SELECT s.game_id, s.player_id, s.team_id, g.season, g.game_date,
               s.seconds/60.0 AS mins, s.pts, s.oreb + s.dreb AS reb, s.ast,
               s.rima + s.mida + s.thra AS shots
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '002%' ORDER BY g.game_date, s.game_id""").fetchdf()
    tg = con.execute("""
        SELECT DISTINCT season, game_id, game_date, team_id FROM nba_games
        WHERE game_id LIKE '002%' ORDER BY game_date, game_id""").fetchdf()
    pg["game_date"] = pd.to_datetime(pg["game_date"]).dt.date
    tg["game_date"] = pd.to_datetime(tg["game_date"]).dt.date
    weights = starout.load_usage_weights() or {}
    positions = starout.load_positions()

    played = pg[pg.mins >= PLAYED_MIN]
    hist = defaultdict(lambda: ([], [], []))          # (player, team) -> dates, mins, shots
    for r in played.itertuples():
        h = hist[(r.player_id, r.team_id)]
        h[0].append(r.game_date); h[1].append(r.mins); h[2].append(r.shots)
    rowmap = {(r.game_id, r.player_id): r for r in pg.itertuples()}
    played_set = defaultdict(set)
    for r in played.itertuples():
        played_set[(r.game_id, r.team_id)].add(r.player_id)
    pdates = defaultdict(list)                         # player -> [(date, team)]
    for r in played.itertuples():
        pdates[r.player_id].append((r.game_date, r.team_id))

    def trail(player, team, date, n=TRAIL_N):
        d, m, f = hist[(player, team)]
        i = bisect_left(d, date)
        if i < TRAIL_MINGAMES:
            return None
        return (float(np.mean(m[max(0, i - n):i])),
                float(np.mean(f[max(0, i - n):i])))

    sched = {}
    for (season, team), g in tg.groupby(["season", "team_id"]):
        g = g.sort_values("game_date")
        sched[(season, team)] = list(zip(g.game_id, g.game_date))

    # ---------------- event detection ----------------
    dep_events, arr_events = [], []
    for (p, team), (dts, mns, fgs) in list(hist.items()):
        sea = tg[(tg.team_id == team) & (tg.game_date == dts[0])]["season"]
        if not len(sea):
            continue
        # ARR: first game for this team, prior same-season other-team play
        season = sea.iloc[0]
        sc = sched[(season, team)]
        prior = [(dd, tt) for (dd, tt) in pdates[p]
                 if dd < dts[0] and tt != team and dd >= sc[0][1]]
        if prior:
            old_team = prior[-1][1]
            tro = trail(p, old_team, prior[-1][0] + pd.Timedelta(days=1))
            if tro and tro[0] >= STAR_MIN:
                arr_events.append(dict(star=p, team=team, season=season,
                                       arrive=dts[0], star_trail=tro[0]))
        # DEP: last game for this team, later same-season other-team play
        sea2 = tg[(tg.team_id == team) & (tg.game_date == dts[-1])]["season"]
        if not len(sea2):
            continue
        season2 = sea2.iloc[0]
        sc2 = sched[(season2, team)]
        season_end = sc2[-1][1]
        later = [dd for (dd, tt) in pdates[p]
                 if dd > dts[-1] and dd <= season_end and tt != team]
        if not later:
            continue
        d0, m0, _ = hist[(p, team)]
        i = bisect_left(d0, dts[-1]) + 1               # incl. the last game
        if i < TRAIL_MINGAMES:
            continue
        tmin = float(np.mean(m0[max(0, i - TRAIL_N):i]))
        if tmin < STAR_MIN:
            continue
        dep_events.append(dict(star=p, team=team, season=season2,
                               last=dts[-1], exec_date=min(later),
                               star_trail=tmin))
    print(f"events: DEP={len(dep_events)} ARR={len(arr_events)}")
    if max_events:
        dep_events, arr_events = dep_events[:max_events], arr_events[:max_events]

    # team stars per (team, season) for the other-star-out exclusion
    def fresh_other_star_out(team, gid, gdate, exclude):
        for (q, t2), (d2, m2, _) in hist.items():
            if t2 != team or q in exclude:
                continue
            i = bisect_left(d2, gdate)
            if i < TRAIL_MINGAMES:
                continue
            if np.mean(m2[max(0, i - TRAIL_N):i]) < STAR_MIN:
                continue
            if q in played_set[(gid, team)]:
                continue
            if i > 0 and 0 < (gdate - d2[i - 1]).days <= FRESH_DAYS:
                return True
        return False

    # ---------------- row construction ----------------
    ev_rows, pre_rows = [], []
    eid = 0
    for kind, events in (("DEP", dep_events), ("ARR", arr_events)):
        for ev in events:
            team, season, star = ev["team"], ev["season"], ev["star"]
            ev_date = ev["arrive"] if kind == "ARR" else ev["last"]
            sc = sched[(season, team)]
            # frozen pool at event date
            pool = []
            for (q, t2), _ in list(hist.items()):
                if t2 != team or q == star:
                    continue
                tr = trail(q, team, ev_date)
                if tr and tr[0] >= ROT_MIN:
                    pool.append((q, tr[0], tr[1]))
            if len(pool) < 3:
                continue
            pool_ids = {q for q, _, _ in pool}
            if kind == "ARR":       # drop other-side-of-trade departures
                dropped = set()
                for (q, _, _) in pool:
                    dq = hist[(q, team)][0]
                    lastb = dq[-1]
                    if -1 <= (lastb - ev_date).days <= 5:
                        dropped.add(q)
                pool = [t for t in pool if t[0] not in dropped]
                pool_ids -= dropped
                if len(pool) < 3:
                    continue
            batt = np.array([b for _, _, b in pool])
            cuts = np.quantile(batt, [1 / 3, 2 / 3])
            terc = {q: int(np.searchsorted(cuts, b, side="right"))
                    for (q, _, b), b in zip(pool, batt)}
            s_pos = positions.get(int(star))
            # star-out lift for the base arm on DEP rows (shipped behavior)
            lift_full = starout.compute_lift(weights, pool_ids, star, default=1.0)
            lift_prod = 1.0 + starout.RESID_ATT_SCALE * (lift_full - 1.0)
            # window games
            games = []
            if kind == "DEP":
                post = [(gid, gd) for gid, gd in sc if gd > ev["last"]]
                for k, (gid, gd) in enumerate(post[:KMAX], start=1):
                    if gd >= ev["exec_date"]:
                        games.append((k, gid, gd))
            else:
                k = 0
                for gid, gd in sc:
                    if gd < ev_date or star not in played_set[(gid, team)]:
                        continue
                    k += 1
                    if k > KMAX:
                        break
                    games.append((k, gid, gd))
            # pre-window: last 10 team games before the event (DEP: star played)
            prewin = [(gid, gd) for gid, gd in sc if gd < ev_date]
            if kind == "DEP":
                prewin = [(gid, gd) for gid, gd in prewin
                          if star in played_set[(gid, team)]]
            prewin = prewin[-10:]
            for is_pre, glist in ((False, games), (True, prewin)):
                for item in glist:
                    k, gid, gd = (0, *item) if is_pre else item
                    if not is_pre and fresh_other_star_out(team, gid, gd, {star}):
                        continue
                    for q in pool_ids:
                        r = rowmap.get((gid, q))
                        if r is None or r.mins < EVAL_MIN:
                            continue
                        rates = player_rates_from_stats(con, int(q), before=gd)
                        if not rates or rates["n_games"] < 8 or rates["proj_min"] < 15:
                            continue
                        mu = max((rates["rate_rim"] + rates["rate_mid"] +
                                  rates["rate_thr"]) * rates["proj_min"], 0.2)
                        row = dict(eid=eid, kind=kind, season=season, k=k,
                                   player=int(q), star=int(star), team=team,
                                   gdate=gd, ev_date=ev_date, pre=is_pre,
                                   terc=terc[q],
                                   same=(None if not (positions.get(int(q)) and s_pos)
                                         else bool({c for c in positions.get(int(q)) if c in "GFC"} &
                                                   {c for c in s_pos if c in "GFC"})),
                                   y_pts=float(r.pts), y_reb=float(r.reb),
                                   y_ast=float(r.ast), y_shots=float(r.shots),
                                   y_min=float(r.mins), mu=mu,
                                   proj_min=rates["proj_min"],
                                   mu_reb=rates["reb_per_min"] * rates["proj_min"],
                                   mu_ast=rates["ast_per_min"] * float(np.clip(rates["proj_min"], 10, 44)),
                                   fresh=(kind == "DEP" and not is_pre and
                                          0 < (gd - ev["last"]).days <= FRESH_DAYS),
                                   lift_prod=lift_prod,
                                   p_pos=positions.get(int(q)), s_pos=s_pos)
                        (pre_rows if is_pre else ev_rows).append((row, rates))
            eid += 1
            if eid % 25 == 0:
                print(f"  ...{eid} events, {len(ev_rows)} rows", flush=True)
    print(f"eval rows: {len(ev_rows)}  pre rows: {len(pre_rows)}  events: {eid}")

    evdf = pd.DataFrame([r for r, _ in ev_rows])
    prdf = pd.DataFrame([r for r, _ in pre_rows])

    # ---------------- walk-forward calibration ----------------
    def bucket(k, buckets):
        for lo, hi in buckets:
            if lo <= k <= hi:
                return f"{lo}-{hi}"
        return None

    cut = {}
    for kind in ("DEP", "ARR"):
        ed = evdf[evdf.kind == kind].groupby("eid").ev_date.first()
        cut[kind] = np.quantile(pd.to_datetime(ed).astype(np.int64),
                                TRAIN_Q) if len(ed) else 0
    evdf["train"] = [pd.Timestamp(r.ev_date).value <= cut[r.kind]
                     for r in evdf.itertuples()]
    prdf["train"] = [pd.Timestamp(r.ev_date).value <= cut[r.kind]
                     for r in prdf.itertuples()]

    def net_ratio(sub, pre, col_y="y_shots", col_mu="mu"):
        if not len(sub) or not len(pre):
            return np.nan
        return (sub[col_y].sum() / sub[col_mu].sum()) / \
               (pre[col_y].sum() / pre[col_mu].sum())

    print("\n=== TRAIN residuals on the LIVE baseline (netted vs pre-window) ===")
    tr_a = evdf[(evdf.kind == "ARR") & evdf.train]
    pr_a = prdf[(prdf.kind == "ARR") & prdf.train]
    tr_d = evdf[(evdf.kind == "DEP") & evdf.train]
    pr_d = prdf[(prdf.kind == "DEP") & prdf.train]

    num = den = 0.0
    print("ARR attempts (mid/top terciles; shape = regime-B roll residual):")
    for (lo, hi, shape) in starout.ARR_SHAPE:
        m = tr_a[(tr_a.k >= lo) & (tr_a.k <= hi) & (tr_a.terc > 0)]
        nr = net_ratio(m, pr_a[pr_a.terc > 0])
        print(f"  k{lo}-{hi}: n={len(m):4d} net {nr:+.4f} (shape {shape:+.2f})")
        if np.isfinite(nr) and shape != 0:
            num += len(m) * shape * (nr - 1.0)
            den += len(m) * shape * shape
    s_att_arr = round(float(np.clip(num / den if den else 0.0, 0.0, 1.5)), 2)
    m0 = tr_a[tr_a.terc == 0]
    print(f"  bottom tercile (shield check): net {net_ratio(m0, pr_a[pr_a.terc == 0]):+.4f} (n={len(m0)})")

    dmin_ev = (tr_a.y_min - tr_a.proj_min).mean() if len(tr_a) else np.nan
    dmin_pre = (pr_a.y_min - pr_a.proj_min).mean() if len(pr_a) else np.nan
    full_tilt = np.array([starout.ARR_TILT_SAME if s is True else
                          starout.ARR_TILT_DIFF if s is False else
                          starout.ARR_TILT_FLAT for s in tr_a.same])
    dnet = dmin_ev - dmin_pre
    s_min_arr = round(float(np.clip(dnet / full_tilt.mean() if len(full_tilt)
                                    else 0.0, 0.0, 1.5)), 2)
    print(f"ARR minutes: net dmin {dnet:+.3f} (ev {dmin_ev:+.3f} pre {dmin_pre:+.3f}), "
          f"mean full tilt {full_tilt.mean():+.2f} -> s_min {s_min_arr}")
    for lab, msk in (("same-pos", tr_a.same == True), ("diff-pos", tr_a.same == False)):  # noqa: E712
        m = tr_a[msk]
        if len(m):
            print(f"  {lab}: ev dmin {(m.y_min - m.proj_min).mean():+.3f} n={len(m)}")
    # reb/ast flow check: do per-game reb/ast fall with the lost minutes?
    for col_y, col_mu, lab in (("y_reb", "mu_reb", "reb"), ("y_ast", "mu_ast", "ast")):
        print(f"  ARR {lab} per-game net ratio: {net_ratio(tr_a, pr_a, col_y, col_mu):+.4f}")

    print("DEP attempts (post-execution, flat shape -0.08):")
    for lo, hi in DEP_BUCKETS:
        m = tr_d[(tr_d.k >= lo) & (tr_d.k <= hi)]
        print(f"  k{lo}-{hi}: n={len(m):4d} net {net_ratio(m, pr_d):+.4f}")
    nr_d = net_ratio(tr_d, pr_d)
    s_dep = round(float(np.clip((nr_d - 1.0) / starout.DEP_ATT_SHAPE, 0.0, 1.5)), 2)
    print(f"  pooled net {nr_d:+.4f} -> s_dep {s_dep}")
    print(f"\nFROZEN SCALES: ARR_ATT_SCALE={s_att_arr} ARR_MIN_SCALE={s_min_arr} "
          f"DEP_ATT_SCALE={s_dep}")

    # ---------------- arms + sims ----------------
    def arr_factor(k, terc):
        if terc == 0:
            return 1.0
        shape = next((v for lo, hi, v in starout.ARR_SHAPE if lo <= k <= hi), 0.0)
        return float(np.clip(1.0 + s_att_arr * shape, starout.ARR_FACTOR_LO, 1.0))

    def arr_tilt(same):
        full = (starout.ARR_TILT_SAME if same is True else
                starout.ARR_TILT_DIFF if same is False else starout.ARR_TILT_FLAT)
        return s_min_arr * full

    dep_factor = float(np.clip(1.0 + s_dep * starout.DEP_ATT_SHAPE,
                               starout.DEP_FACTOR_LO, 1.0))

    def scale_attempts(rates, f):
        out = dict(rates)
        for kk in starout.ATTEMPT_KEYS:
            if out.get(kk) is not None:
                out[kk] = float(out[kk]) * f
        return out

    res = {m: {a: [] for a in ("base",) + ARMS} for m in MARKETS}
    ll = []
    for i, (row, rates) in enumerate(ev_rows):
        if row["kind"] == "DEP":
            if row["fresh"]:      # shipped: lift while fresh (D83 has no supp)
                tiltp = starout.production_tilt(row["p_pos"], row["s_pos"])
                base_r = starout.adjust_rates(rates, row["lift_prod"], tiltp)
            else:
                base_r = rates
            supp_r = rates
            rd_r = scale_attempts(rates, dep_factor)
        else:
            base_r = supp_r = rates
            f = arr_factor(row["k"], row["terc"])
            rd_r = scale_attempts(rates, f)
            pm_old = float(rd_r.get("proj_min", 30.0))
            rd_r["proj_min"] = float(max(pm_old + arr_tilt(row["same"]),
                                         starout.ARR_MIN_FLOOR))
        arms = {"base": base_r, "supp": supp_r, "rd": rd_r}
        y = {"points": row["y_pts"], "rebounds": row["y_reb"],
             "assists": row["y_ast"]}
        sims = {}
        for a, rr in arms.items():
            key = id(rr)
            if key not in sims:
                sims[key] = simulate_player(rr, SIMS, seed=i)
            for m in MARKETS:
                res[m][a].append(crps(sims[key][m], y[m]))

        def _ll(rr):
            mu = max((rr["rate_rim"] + rr["rate_mid"] + rr["rate_thr"]) *
                     rr["proj_min"], 0.2)
            return row["y_shots"] * np.log(mu) - mu
        ll.append(tuple(_ll(arms[a]) for a in ("base",) + ARMS))
        if (i + 1) % 1000 == 0:
            print(f"  ...sims {i + 1}/{len(ev_rows)}", flush=True)
    con.close()
    ll = np.array(ll)
    players = evdf.player.values

    def table(mask, label):
        mask = mask.values if hasattr(mask, "values") else mask
        if mask.sum() < 20:
            print(f"\n--- {label}: n={mask.sum()} too thin ---")
            return
        print(f"\n--- {label} (n={mask.sum()}, "
              f"{len(np.unique(players[mask]))} players) ---")
        print(f"{'market':10} {'arm':5} {'CRPS':>8} {'delta':>9} {'CI95':>20}  verdict")
        for m in MARKETS:
            base = np.array(res[m]["base"])[mask]
            print(f"{m:10} {'base':5} {base.mean():8.4f}")
            for a in ARMS:
                arm = np.array(res[m][a])[mask]
                d = base - arm
                lo, hi = cluster_boot(d, players[mask])
                v = "BETTER" if lo > 0 else ("WORSE" if hi < 0 else "~ (CI spans 0)")
                print(f"{'':10} {a:5} {arm.mean():8.4f} {d.mean():+9.4f} "
                      f"[{lo:+.4f},{hi:+.4f}]  {v}")
        print("attempts (shots Poisson LL, + = better):")
        for j, a in enumerate(ARMS):
            d = (ll[:, j + 1] - ll[:, 0])[mask]
            lo, hi = cluster_boot(d, players[mask], seed=1)
            v = "BETTER" if lo > 0 else ("WORSE" if hi < 0 else "~")
            print(f"  {a:5} dLL {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  {v}")

    ho = ~evdf.train.values.astype(bool)
    table(ho & (evdf.kind == "ARR").values, "HELD-OUT ARR (arrival compression)")
    table(ho & (evdf.kind == "DEP").values, "HELD-OUT DEP (post-execution)")
    table(ho & evdf.fresh.values.astype(bool), "HELD-OUT DEP-FRESH (suppression isolated)")
    table(ho, "HELD-OUT POOLED")
    table(np.ones(len(evdf), bool), "POOLED (incl. train — for the record)")
    print("\nVALIDATE_ROSTERDELTA_DONE", flush=True)


if __name__ == "__main__":
    main(max_events=int(sys.argv[1]) if len(sys.argv) > 1 else None)
