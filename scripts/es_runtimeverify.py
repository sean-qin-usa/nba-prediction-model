#!/usr/bin/env python3
"""HUNT ROUND 2 — RUNTIME verification of shipped-component aliveness.

Instruments production.fit_production at 6 representative dates and, for 3
sample matchups per date, DECOMPOSES the margin from the Predictor's OWN
closure (no rebuild-mismatch risk) plus an independent same-run control
rebuild of every component (sched layer, ff row counts) as a cross-check.

Checks (per task):
  (1) cold-start prior nonzero + fading correctly early season post-csfix
  (2) schedule-layer coefficients sane at each date and ACTUALLY added
  (3) ff.ready flips when expected (~200 factor rows)
  (4) composition PIT darko lookup varies across dates for the same player
  (5) D20 team-home devs: in the production path or not, at each date;
      quantify games where they are dead
  (6) w_comp: no-op once ff.ready (runtime-proved, not code-read)
Plus: quantify runtime-dead schedule terms on each CALLER path (capstone
walk-forward vs live predict_today) — flags the Predictor supports but the
caller never passes.

READ-ONLY DuckDB. Never edits nbapred/. Output: printed report + JSON.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from nbapred.db import connect
from nbapred.model.composition import CompositionModel
from nbapred.model.four_factors import FourFactors, factor_game_rows
from nbapred.model.production import (DEAD_GP, DEAD_WPCT, fit_production,
                                      fit_schedule_layer, last_season_prior)
from nbapred.model.team_ratings import TeamRatings, game_rows

# make sure the FF luck variant is OFF — baseline production
os.environ.pop("FF_LUCK", None)

PROBES = [
    ("2025-26", dt.date(2025, 10, 28)),   # Oct week 1
    ("2025-26", dt.date(2025, 11, 15)),   # Nov
    ("2025-26", dt.date(2026, 1, 15)),    # Jan
    ("2025-26", dt.date(2026, 3, 15)),    # Mar
    ("2024-25", dt.date(2024, 10, 29)),   # Oct week 1
    ("2024-25", dt.date(2025, 1, 15)),    # Jan
]
TOL = 1e-6
REPORT = {"probes": [], "checks": {}, "counts": {}}
FAILS = []


def closure_of(fn):
    return dict(zip(fn.__code__.co_freevars,
                    [c.cell_contents for c in fn.__closure__]))


def check(name, ok, detail=""):
    tag = "OK " if ok else "FAIL"
    print(f"  [{tag}] {name}: {detail}")
    if not ok:
        FAILS.append(f"{name}: {detail}")
    return ok


def games_on(con, season, d):
    """3 sample matchups (home_id, away_id, gid) on first game date >= d."""
    rows = con.execute("""
        SELECT game_id, team_id, team_abbrev, matchup, game_date FROM nba_games
        WHERE season=? AND game_id LIKE '002%' AND game_date >= ?
        ORDER BY game_date, game_id""", [season, d]).fetchall()
    by = {}
    order = []
    d0 = rows[0][4] if rows else None
    for gid, tid, ab, m, gd in rows:
        if gd != d0:
            break
        if gid not in by:
            order.append(gid)
        by.setdefault(gid, []).append((tid, ab, m))
    out = []
    for gid in order[:3]:
        recs = by[gid]
        if len(recs) != 2:
            continue
        m = recs[0][2]
        host = m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        h = next((x for x in recs if x[1] == host), None)
        a = next((x for x in recs if x[1] != host), None)
        if h and a:
            out.append((int(h[0]), int(a[0]), gid))
    return out, d0


def ff_ready_flip_date(con, season, start):
    """First date where factor_game_rows >= 200 (monotone in date)."""
    lo, hi = start, start + dt.timedelta(days=60)
    if len(factor_game_rows(con, season, before=hi)) < 200:
        return None
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if len(factor_game_rows(con, season, before=mid)) >= 200:
            hi = mid
        else:
            lo = mid + dt.timedelta(days=1)
    return lo


def season_game_dates(con, season):
    """game-level (gid, date, home_id, away_id, home_gp_before, away_gp_before,
    b2b_home, b2b_away, dead_home, dead_away)."""
    meta = con.execute("""
        SELECT game_id, team_id, team_abbrev, matchup, wl, game_date FROM nba_games
        WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL
        ORDER BY game_date, game_id""", [season]).fetchall()
    by, order = {}, []
    tdates, hist = {}, {}
    for gid, tid, ab, m, wl, gd in meta:
        if gid not in by:
            order.append(gid)
        by.setdefault(gid, []).append((tid, ab, m, wl, gd))
        tdates.setdefault(tid, set()).add(gd)
        hist.setdefault(tid, []).append((gd, wl == "W"))
    for t in hist:
        hist[t].sort()
    gp_run = {}
    out = []
    for gid in order:
        recs = by[gid]
        if len(recs) != 2:
            continue
        m = recs[0][2]
        host = m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        h = next((x for x in recs if x[1] == host), None)
        a = next((x for x in recs if x[1] != host), None)
        if not h or not a:
            continue
        gd = h[4]
        def b2b(tid):
            return (gd - dt.timedelta(days=1)) in tdates.get(tid, set())
        def dead(tid):
            past = [w for (dd, w) in hist.get(tid, []) if dd < gd]
            return len(past) >= DEAD_GP and (sum(past) / len(past)) < DEAD_WPCT
        out.append(dict(gid=gid, date=gd, h=h[0], a=a[0],
                        gph=gp_run.get(h[0], 0), gpa=gp_run.get(a[0], 0),
                        b2bh=b2b(h[0]), b2ba=b2b(a[0]),
                        deadh=dead(h[0]), deada=dead(a[0])))
        gp_run[h[0]] = gp_run.get(h[0], 0) + 1
        gp_run[a[0]] = gp_run.get(a[0], 0) + 1
    return out


def main():
    con = connect(read_only=True)

    # ---------- per-probe decomposition ----------
    darko_probe_players = None
    darko_by_date = {}
    for season, before in PROBES:
        print(f"\n===== PROBE {season} @ {before} =====")
        model = fit_production(con, season, before=before, w_comp=0.7)
        cl = closure_of(type(model).margin)
        clr = closure_of(type(model).ratings_margin)
        comp, ff, tr = cl["comp"], cl["ff"], cl["tr"]
        he, b_hb2b, b_ab2b = cl["he"], cl["b_hb2b"], cl["b_ab2b"]
        b_hdead, b_adead = cl["b_hdead"], cl["b_adead"]
        w_comp = cl["w_comp"]
        prior, gp_map, id2ab = clr["prior"], clr["games_played"], clr["id2ab"]

        # independent same-run control rebuild (cross-check, not the oracle)
        sched_ctrl = fit_schedule_layer(con, before)
        nrows_ff = len(factor_game_rows(con, season, before=before))
        check("control sched == closure sched",
              max(abs(sched_ctrl[i] - (he, b_hb2b, b_ab2b, b_hdead, b_adead)[i])
                  for i in range(5)) < TOL,
              f"{tuple(round(x,3) for x in sched_ctrl)}")

        # (3) ff.ready
        check("ff.ready consistent with row count",
              ff.ready == (nrows_ff >= 200),
              f"ready={ff.ready} rows={nrows_ff}")

        # (2) sched coefficient sanity — split APPLIED (he/b2b: callers pass
        # these) from FIT-ONLY (dead: no caller passes dead flags, D47 ns).
        check("sched APPLIED coefficients sane",
              0.5 <= he <= 4.5 and -4.5 <= b_hb2b <= 0.5 and -0.5 <= b_ab2b <= 5.0,
              f"he={he:.2f} hb2b={b_hb2b:.2f} ab2b={b_ab2b:.2f}")
        # dead coeffs: estimated on only ~70-150 of ~2460 trailing rows (dead
        # teams exist only post-game-60), wpct control only partially
        # de-confounds, shrink w=0.80 -> large/unstable (-4..-9 observed).
        # Latent hazard IF anyone flips dead flags on; zero runtime impact now.
        check("sched FIT-ONLY dead coefficients bounded (runtime-dead)",
              abs(b_hdead) <= 12 and abs(b_adead) <= 12,
              f"hdead={b_hdead:.2f} adead={b_adead:.2f} (fit-only)")

        mus, d0 = games_on(con, season, before)
        probe_rec = dict(season=season, before=str(before), ff_ready=ff.ready,
                         n_ff_rows=nrows_ff,
                         sched=dict(he=he, b_hb2b=b_hb2b, b_ab2b=b_ab2b,
                                    b_hdead=b_hdead, b_adead=b_adead),
                         matchups=[])
        for h, a, gid in mus:
            cm = comp.margin(h, a, None, None, before, home_edge=0.0)
            rm_full = model.ratings_margin(h, a)
            rm = rm_full - tr.home
            fm = ff.margin_neutral(h, a) if ff.ready else None
            m_model = model.margin(h, a, None, None, before)
            if ff.ready:
                m_expect = 0.5 * fm + 0.5 * cm + he
            else:
                m_expect = w_comp * cm + (1 - w_comp) * rm + he
            check(f"decomposition matches ({gid})",
                  abs(m_model - m_expect) < TOL,
                  f"model={m_model:+.4f} recomposed={m_expect:+.4f} "
                  f"cm={cm:+.2f} fm={None if fm is None else round(fm,2)} "
                  f"rm={rm:+.2f} he={he:+.2f}")
            # (2) sched ACTUALLY added — runtime deltas
            dh = model.margin(h, a, None, None, before, b2b_home=True) - m_model
            da = model.margin(h, a, None, None, before, b2b_away=True) - m_model
            dd = model.margin(h, a, None, None, before, dead_home=True,
                              dead_away=True) - m_model
            check(f"b2b/dead deltas == coefficients ({gid})",
                  abs(dh - b_hb2b) < TOL and abs(da - b_ab2b) < TOL
                  and abs(dd - (b_hdead + b_adead)) < TOL,
                  f"d_hb2b={dh:+.3f} d_ab2b={da:+.3f} d_dead={dd:+.3f}")
            # (1) prior term: alive in ratings_margin?
            wh = max(0.0, 1 - gp_map.get(h, 0) / 20.0)
            wa = max(0.0, 1 - gp_map.get(a, 0) / 20.0)
            ph = prior.get(id2ab.get(h, ""), 0.0)
            pa = prior.get(id2ab.get(a, ""), 0.0)
            base = tr.pred_margin(h, a)
            check(f"prior applied in ratings_margin ({gid})",
                  abs(rm_full - (base + wh * ph - wa * pa)) < TOL,
                  f"gp=({gp_map.get(h,0)},{gp_map.get(a,0)}) w=({wh:.2f},{wa:.2f}) "
                  f"prior=({ph:+.2f},{pa:+.2f}) term={wh*ph - wa*pa:+.2f}")
            # (5) D20 home_dev: measure its contribution to the FULL margin
            hd = tr.home_dev.get(h, 0.0)
            d20_in_path = not ff.ready   # rm keeps home_dev; ff path drops it
            # runtime proof: recompose ff-path without home_dev and compare
            if ff.ready:
                check(f"D20 absent from ff-path margin ({gid})",
                      abs(m_model - (0.5 * fm + 0.5 * cm + he)) < TOL,
                      f"home_dev({id2ab.get(h)})={hd:+.3f} unused")
            else:
                no_dev = base - hd  # pred_margin includes home_dev exactly once
                rm_nodev = (no_dev + wh * ph - wa * pa) - tr.home
                m_nodev = w_comp * cm + (1 - w_comp) * rm_nodev + he
                check(f"D20 present in fallback margin ({gid})",
                      abs((m_model - m_nodev) - (1 - w_comp) * hd) < TOL,
                      f"home_dev({id2ab.get(h)})={hd:+.3f} "
                      f"contribution={(1-w_comp)*hd:+.4f}")
            probe_rec["matchups"].append(dict(
                gid=gid, h=id2ab.get(h), a=id2ab.get(a), margin=m_model,
                cm=cm, fm=fm, rm=rm, prior_term=wh * ph - wa * pa,
                home_dev=hd, d20_in_path=d20_in_path))

        # (1) prior sanity at the fit level
        check("prior dict populated", len(prior) >= 28, f"n={len(prior)}")
        check("games_played respects cutoff",
              all(v <= (before - dt.date(int(season[:4]), 10, 1)).days
                  for v in gp_map.values()) and
              (max(gp_map.values(), default=0) <= 82),
              f"max gp={max(gp_map.values(), default=0)}")
        # D20 magnitudes at this date
        devs = sorted(tr.home_dev.items(), key=lambda kv: -abs(kv[1]))[:3]
        print(f"  top |home_dev|: "
              f"{[(id2ab.get(t), round(v,2)) for t, v in devs]}")
        probe_rec["top_home_dev"] = [(id2ab.get(t), v) for t, v in devs]

        # (4) PIT darko across dates — collect for cross-date comparison
        if darko_probe_players is None:
            top = sorted(comp.players.items(),
                         key=lambda kv: -kv[1]["trail_min"])[:3]
            darko_probe_players = [pid for pid, _ in top]
        dk = CompositionModel._darko_asof(con, before)
        darko_by_date[(season, before)] = {p: dk.get(p) for p in darko_probe_players}

        # (6) w_comp no-op test (runtime): refit with w_comp=0.2
        model2 = fit_production(con, season, before=before, w_comp=0.2)
        deltas = [abs(model.margin(h, a, None, None, before)
                      - model2.margin(h, a, None, None, before))
                  for h, a, _ in mus]
        if ff.ready:
            check("w_comp is a NO-OP (ff ready)", max(deltas) < TOL,
                  f"max delta w_comp 0.7->0.2 = {max(deltas):.2e}")
        else:
            check("w_comp ACTIVE (ff not ready)", max(deltas) > 0.05,
                  f"max delta w_comp 0.7->0.2 = {max(deltas):.3f}")
        probe_rec["w_comp_max_delta"] = max(deltas)
        REPORT["probes"].append(probe_rec)

    # (4) darko PIT: same player, different dates -> different values
    print("\n===== PIT DARKO CROSS-DATE =====")
    for p in darko_probe_players:
        vals = {k: v for k, v in
                ((k, darko_by_date[k].get(p)) for k in darko_by_date)
                if v is not None}
        uniq = len({round(v, 6) for v in vals.values()})
        check(f"darko varies across dates (player {p})", uniq >= 3,
              f"{uniq} distinct values over {len(vals)} dates: "
              f"{[round(v,3) for v in vals.values()]}")
    REPORT["checks"]["darko_by_date"] = {
        f"{s}@{d}": v for (s, d), v in darko_by_date.items()}

    # ---------- aliveness quantification ----------
    print("\n===== ALIVENESS QUANTIFICATION =====")
    for season, start in (("2025-26", dt.date(2025, 10, 21)),
                          ("2024-25", dt.date(2024, 10, 22)),
                          ("2023-24", dt.date(2023, 10, 24))):
        flip = ff_ready_flip_date(con, season, start)
        games = season_game_dates(con, season)
        n = len(games)
        # per-day refit (predict_today semantics): fallback iff date < flip
        n_fb_daily = sum(1 for g in games if g["date"] < flip)
        # weekly refit (capstone prod_by_season semantics)
        refit, last = {}, None
        for g in games:
            if last is None or (g["date"] - last).days >= 7:
                last = g["date"]
            refit[g["gid"]] = last
        n_fb_weekly = sum(1 for g in games if refit[g["gid"]] < flip)
        # prior design window: games where a team is under 20 gp
        n_prior_design = sum(1 for g in games if g["gph"] < 20 or g["gpa"] < 20)
        n_prior_dead = sum(1 for g in games
                           if (g["gph"] < 20 or g["gpa"] < 20)
                           and g["date"] >= flip)
        # schedule-context games (terms the Predictor supports)
        n_b2b = sum(1 for g in games if g["b2bh"] or g["b2ba"])
        n_dead = sum(1 for g in games if g["deadh"] or g["deada"])
        REPORT["counts"][season] = dict(
            n=n, ff_flip=str(flip), n_fallback_daily=n_fb_daily,
            n_fallback_weekly=n_fb_weekly, n_prior_design=n_prior_design,
            n_prior_designed_but_dead=n_prior_dead, n_b2b=n_b2b, n_dead=n_dead)
        print(f"{season}: n={n} ff_ready_flip={flip} "
              f"fallback games daily={n_fb_daily} weekly={n_fb_weekly} | "
              f"prior design-window={n_prior_design} "
              f"(dead post-ff={n_prior_dead}) | b2b games={n_b2b} "
              f"dead-team games={n_dead}")

    con.close()
    REPORT["fails"] = FAILS
    out = Path(os.environ.get("ES_OUT_DIR", "/tmp"))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "es_runtimeverify.json", "w") as f:
        json.dump(REPORT, f, indent=1, default=str)
    print(f"\n{'ALL RUNTIME CHECKS PASSED' if not FAILS else 'FAILURES:'}")
    for x in FAILS:
        print(" -", x)


if __name__ == "__main__":
    main()
