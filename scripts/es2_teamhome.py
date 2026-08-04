"""GATE B — D20 restoration: team-specific home advantage in the SCHEDULE layer.

Context (journal wf_c1c677ac, scripts/es_runtimeverify.py): the gate-passed D20
component (per-team home-advantage deviations, TeamRatings team_home_ridge=200,
+0.0020 CI +0.0004..+0.0036) is EFFECTIVELY DEAD in shipped production — it
lives only inside the ratings fallback (103-108 games/season) and is shrunk to
<=0.19 pts there. This experiment restores it where it belongs: the schedule
layer, which applies to EVERY game.

ONE pre-registered config (k=200, the originally gated value). NO sweeps.

Variant ("teamhome"):
    sched_home_edge_team = he_global + home_dev_team
  * he_global, b2b betas: production fit_schedule_layer, UNCHANGED.
  * home_dev_team: walk-forward at each weekly refit from the SAME trailing
    730-day window of finals that fit_schedule_layer uses (game_date in
    [before-730d, before)). Replicates the D20 construction at the margin
    level: ridge regression
        home_margin_i ~ quality_+1/-1 columns (ridge 30 — the codebase's
                        margin-scale team-quality penalty, from
                        last_season_prior)
                      + unpenalized intercept (global home edge)
                      + home-team indicator columns (ridge k=200 — D20's
                        originally gated shrinkage)
    home_dev_team = the ridge-200 coefficient (deviation vs global, quality-
    controlled, not demeaned — exactly D20's identification). ~41-82 home
    games/team in window -> shrinkage factor n/(n+200) ~ 0.17-0.29.
  * Applied in BOTH branches (it is the schedule layer). In the ratings
    fallback the within-season D20 dev is STRIPPED from the ratings margin
    (rm - tr.home - tr.home_dev[home]) so the venue effect is MOVED, not
    duplicated ("restore it where it belongs").

Control = same-run exact replication of shipped production (copied from
scripts/es_continuity.py season_run, which verified vs
data/capstone_pergame_csfix.csv to ~1e-14). Replication re-checked here.

Gate: paired bootstrap 2000x 95% CI, variant vs control. Report: pooled,
per-season, early-season (either team gp<20), gp[0,5) bucket (min gp<5), and
DEN/UTA (altitude) home games. Positive delta = variant better.

Read-only DB. NEW file (scripts/es2_*.py). No edits to nbapred/ or existing
scripts. Outputs to scratchpad only.
"""
import bisect
import datetime as dt
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from nbapred.db import connect
from nbapred.model.composition import CompositionModel
from nbapred.model.four_factors import FourFactors
from nbapred.model.production import (SCALE, fit_schedule_layer,
                                      last_season_prior, sigmoid)
from nbapred.model.team_ratings import TeamRatings, game_rows

OUT_DIR = Path(os.environ.get(
    "ES_OUT",
    "/tmp/claude-1004/-hdd-steveqin-sean-dev/4912ce58-69cd-488d-a5f4-0d3ed2eed30c/scratchpad"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = ("2023-24", "2024-25", "2025-26")
W_COMP = 0.7
QUALITY_RIDGE = 30.0     # last_season_prior's margin-scale team penalty
DEV_RIDGE = 200.0        # D20's gated team_home_ridge — THE config
WINDOW_DAYS = 730        # fit_schedule_layer's trailing window
ALTITUDE = ("DEN", "UTA")


def fetch_margin_rows(con):
    """All regular-season finals (all seasons): (date, home_id, away_id,
    home_margin), date-sorted — the fit_schedule_layer data, prefetched once."""
    rows = con.execute("""
        WITH t AS (SELECT game_id, game_date, team_id, is_home, pts
                   FROM nba_games WHERE game_id LIKE '002%' AND pts IS NOT NULL)
        SELECT h.game_date, h.team_id ht, a.team_id a_t, h.pts - a.pts AS margin
        FROM t h JOIN t a USING (game_id) WHERE h.is_home AND NOT a.is_home
        ORDER BY h.game_date
    """).fetchall()
    out = []
    for d, ht, at, m in rows:
        d = d.date() if hasattr(d, "date") else d
        out.append((d, int(ht), int(at), float(m)))
    return out


def fit_home_dev(window):
    """D20 construction on the margin scale: {team_id: ridge-200 home dev}."""
    if len(window) < 200:
        return {}
    teams = sorted({t for _, h, a, _ in window for t in (h, a)})
    idx = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    n = len(window)
    X = np.zeros((n, 2 * T + 1))
    y = np.zeros(n)
    for i, (_, ht, at, m) in enumerate(window):
        X[i, idx[ht]] += 1.0          # quality (+home)
        X[i, idx[at]] -= 1.0          # quality (-away)
        X[i, T] = 1.0                 # global home edge (unpenalized)
        X[i, T + 1 + idx[ht]] = 1.0   # per-team home dev (ridge 200)
        y[i] = m
    P = np.zeros(2 * T + 1)
    P[:T] = QUALITY_RIDGE
    P[T + 1:] = DEV_RIDGE
    beta = np.linalg.solve(X.T @ X + np.diag(P), X.T @ y)
    return {t: float(beta[T + 1 + idx[t]]) for t in teams}


def season_run(season: str, margin_rows, margin_dates):
    """prod_by_season.py loop (default oracle-outs): control + variant margins
    side by side from SHARED components. Control path copied verbatim from
    scripts/es_continuity.py (verified vs capstone_pergame_csfix ~1e-14)."""
    t0 = time.time()
    con = connect(read_only=True)

    raw_prior = last_season_prior(con, season, regress=1.0)
    prior_ctrl = {t: 0.75 * r for t, r in raw_prior.items()}
    ab = dict(con.execute(
        "SELECT DISTINCT team_id, team_abbrev FROM nba_games WHERE season=?",
        [season]).fetchall())
    id2ab = {t: a for t, a in ab.items()}
    ab2id = {a: t for t, a in ab.items()}

    # ---- game meta / market / outs infra (copied from prod_by_season.py) --
    pm = con.execute("""SELECT game_id, team_id, player_id, seconds/60.0 AS mins
        FROM player_game_stats WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    played = {(g, t): set(grp.player_id) for (g, t), grp in pm.groupby(["game_id", "team_id"])}
    meta = con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl, game_date
        FROM nba_games WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL
        ORDER BY game_date""", [season]).fetchdf()
    mkt = {(str(r[0])[:10], r[1], r[2]): r[3] for r in con.execute(
        "SELECT game_date, home, away, p_home_spread FROM odds_market WHERE season_end=?",
        [int(season[:4]) + 1]).fetchall()}
    by, order = {}, []
    for x in meta.itertuples():
        if x.game_id not in by:
            order.append(x.game_id)
        by.setdefault(x.game_id, []).append(x)
    tdates = {}
    for x in meta.itertuples():
        d = x.game_date.date() if hasattr(x.game_date, "date") else x.game_date
        tdates.setdefault(x.team_id, set()).add(d)

    def b2b(tid, d):
        return (d - dt.timedelta(days=1)) in tdates.get(tid, set())

    # ---- walk-forward loop -------------------------------------------------
    rows = []
    gp_live = {}
    comp = tr = None
    sched = None
    ff_ctrl = None
    home_dev = {}
    dev_hist = []            # (refit_date, dev map) for diagnostics
    games_played = {}
    last = None
    nrefit = 0
    n_fallback = 0
    for gid in order:
        recs = by[gid]
        if len(recs) != 2:
            continue
        m = recs[0].matchup
        host = m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        h = next((x for x in recs if x.team_abbrev == host), None)
        a = next((x for x in recs if x.team_abbrev != host), None)
        if not h or not a:
            continue
        gd = h.game_date.date() if hasattr(h.game_date, "date") else h.game_date
        if last is None or (gd - last).days >= 7:
            nrefit += 1
            comp = CompositionModel(con, before=gd)
            sched = fit_schedule_layer(con, before=gd)
            tr = TeamRatings(ridge=25.0).fit(game_rows(con, before=gd, season=season))
            ff_ctrl = FourFactors().fit(con, season, before=gd)
            games_played = dict(con.execute("""
                SELECT team_id, count(*) FROM nba_games WHERE season=? AND
                game_id LIKE '002%' AND wl IS NOT NULL AND game_date < ?
                GROUP BY 1""", [season, gd]).fetchall())
            lo = bisect.bisect_left(margin_dates, gd - dt.timedelta(days=WINDOW_DAYS))
            hi = bisect.bisect_left(margin_dates, gd)
            home_dev = fit_home_dev(margin_rows[lo:hi])
            dev_hist.append((str(gd), dict(home_dev)))
            last = gd
        pmv = mkt.get((str(gd)[:10], h.team_abbrev, a.team_abbrev))
        gph = gp_live.get(h.team_id, 0)
        gpa = gp_live.get(a.team_id, 0)
        gp_live[h.team_id] = gph + 1
        gp_live[a.team_id] = gpa + 1
        if pmv is None:
            continue
        outs = {}
        for t in (h.team_id, a.team_id):
            pl = played.get((gid, t), set())
            outs[t] = {p for p, d0 in comp.players.items()
                       if d0["team_id"] == t and (gd - d0["last_played"]).days <= 12
                       and p not in pl}
        b2bh, b2ba = b2b(h.team_id, gd), b2b(a.team_id, gd)
        he, b_hb2b, b_ab2b = sched[0], sched[1], sched[2]
        sch = he + (b_hb2b if b2bh else 0.0) + (b_ab2b if b2ba else 0.0)
        cm = comp.margin(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id],
                         gd, home_edge=0.0)
        dev_h = home_dev.get(h.team_id, 0.0)

        # ---- control (exact production) + variant (sched dev restored) ----
        if ff_ctrl is not None and ff_ctrl.ready:
            base = 0.5 * ff_ctrl.margin_neutral(h.team_id, a.team_id) + 0.5 * cm + sch
            m_ctrl = base
            m_var = base + dev_h
            fb = 0
        else:
            mm = tr.pred_margin(h.team_id, a.team_id)
            gh = games_played.get(h.team_id, 0)
            ga = games_played.get(a.team_id, 0)
            wh = max(0.0, 1 - gh / 20.0)
            wa = max(0.0, 1 - ga / 20.0)
            rm = (mm + wh * prior_ctrl.get(id2ab.get(h.team_id, ""), 0.0)
                  - wa * prior_ctrl.get(id2ab.get(a.team_id, ""), 0.0))
            m_ctrl = W_COMP * cm + (1 - W_COMP) * (rm - tr.home) + sch
            # variant: venue effect MOVED to sched layer — strip within-season
            # D20 from the ratings margin, add trailing-730d sched dev
            hd_ts = tr.home_dev.get(h.team_id, 0.0)
            m_var = (W_COMP * cm + (1 - W_COMP) * (rm - tr.home - hd_ts)
                     + sch + dev_h)
            fb = 1
            n_fallback += 1

        rows.append(dict(
            season=season, game_id=gid, game_date=str(gd)[:10],
            home=h.team_abbrev, away=a.team_abbrev, y=int(h.wl == "W"),
            p_mkt=float(pmv), gp_home=gph, gp_away=gpa, fallback=fb,
            dev_home=round(dev_h, 4),
            control=float(sigmoid(m_ctrl / SCALE)),
            teamhome=float(sigmoid(m_var / SCALE))))
    con.close()

    # dev diagnostics: per-season mean dev for DEN/UTA + spread
    devsum = {}
    for x in ALTITUDE:
        tid = ab2id.get(x)
        vals = [dv.get(tid, 0.0) for _, dv in dev_hist if tid]
        devsum[x] = dict(mean=round(float(np.mean(vals)), 3),
                         min=round(float(np.min(vals)), 3),
                         max=round(float(np.max(vals)), 3)) if vals else None
    lastdate, lastdev = dev_hist[-1]
    top = sorted(lastdev.items(), key=lambda kv: -kv[1])[:5]
    bot = sorted(lastdev.items(), key=lambda kv: kv[1])[:5]
    devsum["last_refit"] = dict(
        date=lastdate,
        top5={id2ab.get(t, t): round(v, 3) for t, v in top},
        bot5={id2ab.get(t, t): round(v, 3) for t, v in bot},
        mean_abs=round(float(np.mean([abs(v) for v in lastdev.values()])), 3))
    print(f"[{season}] n={len(rows)} refits={nrefit} fallback_games={n_fallback} "
          f"({time.time()-t0:.0f}s)", flush=True)
    print(f"[{season}] devs: {json.dumps(devsum)}", flush=True)
    return rows, devsum, n_fallback


def paired_ci(d, B=2000, seed=42):
    d = np.asarray(d, float)
    if len(d) == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (B, len(d)))
    means = d[idx].mean(axis=1)
    return (float(d.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def ll_vec(y, p):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    con = connect(read_only=True)
    margin_rows = fetch_margin_rows(con)
    con.close()
    margin_dates = [r[0] for r in margin_rows]

    all_rows, devs, nfb = [], {}, {}
    for s in SEASONS:
        r, d, f = season_run(s, margin_rows, margin_dates)
        all_rows += r
        devs[s] = d
        nfb[s] = f

    import csv
    with open(OUT_DIR / "es2_teamhome_pergame.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(all_rows)

    y = np.array([r["y"] for r in all_rows])
    seas = np.array([r["season"] for r in all_rows])
    gpmin = np.array([min(r["gp_home"], r["gp_away"]) for r in all_rows])
    early = gpmin < 20
    gp05 = gpmin < 5
    alt_home = np.array([r["home"] in ALTITUDE for r in all_rows])
    alt_any = np.array([r["home"] in ALTITUDE or r["away"] in ALTITUDE
                        for r in all_rows])
    ll_c = ll_vec(y, [r["control"] for r in all_rows])
    ll_v = ll_vec(y, [r["teamhome"] for r in all_rows])
    ll_m = ll_vec(y, [r["p_mkt"] for r in all_rows])
    d = ll_c - ll_v          # positive = variant better

    # control vs shipped baseline CSV (replication check)
    base = {}
    with open(REPO / "data" / "capstone_pergame_csfix.csv") as f:
        for r in csv.DictReader(f):
            base[(r["season"], r["game_id"])] = float(r["p_us"])
    diffs = [abs(base[(r["season"], r["game_id"])] - r["control"])
             for r in all_rows if (r["season"], r["game_id"]) in base]
    repl = dict(n_matched=len(diffs), n_ours=len(all_rows),
                max_abs_diff=float(max(diffs)) if diffs else None)

    def sub(mask):
        return dict(n=int(mask.sum()), delta=paired_ci(d[mask]),
                    ll_control=round(float(ll_c[mask].mean()), 5),
                    ll_variant=round(float(ll_v[mask].mean()), 5))

    res = dict(
        config=dict(dev_ridge=DEV_RIDGE, quality_ridge=QUALITY_RIDGE,
                    window_days=WINDOW_DAYS, gate="paired bootstrap 2000x 95% CI"),
        replication=repl, n_fallback=nfb, devs=devs,
        control_ll=dict(
            pooled=round(float(ll_c.mean()), 5),
            market=round(float(ll_m.mean()), 5),
            per_season={s: round(float(ll_c[seas == s].mean()), 5)
                        for s in SEASONS}),
        variant_ll=dict(
            pooled=round(float(ll_v.mean()), 5),
            per_season={s: round(float(ll_v[seas == s].mean()), 5)
                        for s in SEASONS}),
        gate=dict(
            pooled=sub(np.ones(len(d), bool)),
            per_season={s: sub(seas == s) for s in SEASONS},
            early=sub(early),
            gp05=sub(gp05),
            den_uta_home=sub(alt_home),
            den_uta_any=sub(alt_any)),
        diag_buckets={
            "gp[5,10)": sub((gpmin >= 5) & (gpmin < 10)),
            "gp[10,20)": sub((gpmin >= 10) & (gpmin < 20)),
            "gp[20,)": sub(gpmin >= 20)})
    with open(OUT_DIR / "es2_teamhome_results.json", "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
