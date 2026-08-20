"""GATE B (rerun) — D20 restoration: team-specific home advantage in the
SCHEDULE layer.

Context (journal wf_c1c677ac / D59, scripts/es_runtimeverify.py): the
gate-passed D20 component (per-team home-advantage deviations, TeamRatings
team_home_ridge=200, +0.0020 CI +0.0004..+0.0036) is EFFECTIVELY DEAD in
shipped production — it lives only inside the ratings fallback (~105
games/season) and is shrunk to <=0.19 pts there. This experiment restores it
in the schedule layer, which applies to EVERY game:

    sched_home_edge_team = he_global + home_dev_team

ONE pre-registered config (k=200, the originally gated value). NO sweeps.

home_dev_team: walk-forward at each weekly refit from the SAME trailing
730-day window of finals that fit_schedule_layer uses (game_date in
[before-730d, before), strict PIT). Replicates the D20 construction at the
margin level: ridge regression
    home_margin_i ~ quality_+1/-1 team columns   (ridge 30, the codebase's
                                                  margin-scale quality penalty)
                  + unpenalized intercept         (global home edge)
                  + home-team indicator columns   (ridge k=200 — D20's gated
                                                  shrinkage)
home_dev_team = the ridge-200 coefficient (deviation vs global, quality-
controlled, not demeaned — D20's identification). ~40-80 home games/team in
window -> shrinkage n/(n+200) ~ 0.17-0.29.

Control = shipped production EXACTLY: import fit_production (which now
includes the D62 carry via continuity_map) and run the prod_by_season.py loop
verbatim (default oracle-outs path, weekly refit). Variant = control margin +
home_dev(home_team) — applied on top in BOTH branches. (Caveat, accepted at
pre-registration: in the ~105 fallback games/season the within-season D20 dev
(<=0.19 pts) remains in the ratings margin, so those games double-count
slightly; the main ff-ready branch — ~97% of games — is clean.)

Verification: control per-game probs vs data/capstone_pergame_carry.csv (the
in-flight carry capstone) if finished; else data/capstone_pergame_csfix.csv
with the diff noted (csfix predates the D62 carry, so early-season games WILL
differ there).

Gate: paired bootstrap 2000x 95% CI on per-game logloss deltas, variant vs
control. Report: pooled, per-season, DEN/UTA home games (altitude teams —
where the effect should concentrate). Sanity: home_dev for DEN/UTA/BOS at 3
sample refit dates.

Read-only DB. NEW file (scripts/es2b_teamhome.py). No edits to nbapred/ or
existing scripts.
"""
import bisect
import csv
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
from nbapred.model.production import SCALE, fit_production, sigmoid

OUT_DIR = REPO / "data"
SEASONS = ("2023-24", "2024-25", "2025-26")
QUALITY_RIDGE = 30.0     # margin-scale team-quality penalty (last_season_prior)
DEV_RIDGE = 200.0        # D20's gated team_home_ridge — THE config
WINDOW_DAYS = 730        # fit_schedule_layer's trailing window
ALTITUDE = ("DEN", "UTA")
SANITY_TEAMS = ("DEN", "UTA", "BOS")
SANITY_DATES = (dt.date(2024, 1, 15), dt.date(2025, 1, 15), dt.date(2026, 1, 15))


def fetch_margin_rows(con):
    """All regular-season finals: (date, home_id, away_id, home_margin),
    date-sorted — the fit_schedule_layer universe, prefetched once."""
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


def season_run(season, margin_rows, margin_dates, dev_hist):
    """prod_by_season.py loop VERBATIM (default oracle-outs path, weekly
    refit); control = fit_production margin, variant = control + home_dev."""
    t0 = time.time()
    con = connect(read_only=True)
    pm = con.execute("""SELECT game_id, team_id, player_id, seconds/60.0 AS mins
        FROM player_game_stats WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    played = {(g, t): set(grp.player_id)
              for (g, t), grp in pm.groupby(["game_id", "team_id"])}
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

    id2ab = dict(con.execute(
        "SELECT DISTINCT team_id, team_abbrev FROM nba_games WHERE season=?",
        [season]).fetchall())

    rows = []
    gp_live = {}
    model = comp = None
    home_dev = {}
    last = None
    nrefit = 0
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
            model = fit_production(con, season, before=gd, w_comp=0.7)
            comp = CompositionModel(con, before=gd)
            lo = bisect.bisect_left(margin_dates, gd - dt.timedelta(days=WINDOW_DAYS))
            hi = bisect.bisect_left(margin_dates, gd)
            home_dev = fit_home_dev(margin_rows[lo:hi])
            dev_hist.append((gd, {id2ab.get(t, t): v for t, v in home_dev.items()}))
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
        # control: EXACTLY model.p_home (same margin, dead flags not passed —
        # matches prod_by_season.py); variant: + home_dev on top
        mm = model.margin(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id],
                          gd, b2b_home=b2b(h.team_id, gd), b2b_away=b2b(a.team_id, gd))
        dev_h = home_dev.get(h.team_id, 0.0)
        rows.append(dict(
            season=season, game_id=gid, game_date=str(gd)[:10],
            home=h.team_abbrev, away=a.team_abbrev, y=int(h.wl == "W"),
            p_mkt=float(pmv), gp_home=gph, gp_away=gpa,
            dev_home=round(dev_h, 4),
            p_ctrl=float(sigmoid(mm / SCALE)),
            p_var=float(sigmoid((mm + dev_h) / SCALE))))
    con.close()
    print(f"[{season}] n={len(rows)} refits={nrefit} ({time.time()-t0:.0f}s)",
          flush=True)
    return rows


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

    all_rows = []
    dev_hist = []            # (refit_date, {abbrev: dev}) across all seasons
    for s in SEASONS:
        all_rows += season_run(s, margin_rows, margin_dates, dev_hist)

    with open(OUT_DIR / "es2b_teamhome_pergame.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(all_rows)

    y = np.array([r["y"] for r in all_rows])
    seas = np.array([r["season"] for r in all_rows])
    gpmin = np.array([min(r["gp_home"], r["gp_away"]) for r in all_rows])
    alt_home = np.array([r["home"] in ALTITUDE for r in all_rows])
    ll_c = ll_vec(y, [r["p_ctrl"] for r in all_rows])
    ll_v = ll_vec(y, [r["p_var"] for r in all_rows])
    ll_m = ll_vec(y, [r["p_mkt"] for r in all_rows])
    d = ll_c - ll_v          # positive = variant better

    # ---- control replication check vs shipped capstone CSV ----------------
    carry = OUT_DIR / "capstone_pergame_carry.csv"
    csfix = OUT_DIR / "capstone_pergame_csfix.csv"
    baseline_path = carry if carry.exists() else csfix
    base = {}
    with open(baseline_path) as f:
        for r in csv.DictReader(f):
            base[(r["season"], r["game_id"])] = float(r["p_us"])
    diffs = [abs(base[(r["season"], r["game_id"])] - r["p_ctrl"])
             for r in all_rows if (r["season"], r["game_id"]) in base]
    repl = dict(baseline=baseline_path.name, n_matched=len(diffs),
                n_ours=len(all_rows),
                max_abs_diff=float(max(diffs)) if diffs else None,
                note=("exact-replication expected (same fit_production incl. "
                      "D62 carry)" if baseline_path is carry else
                      "csfix PREDATES the D62 carry -> early-season games "
                      "expected to differ; re-check vs capstone_pergame_carry.csv "
                      "when the in-flight run finishes"))

    # ---- home_dev sanity: DEN/UTA/BOS at 3 sample refit dates -------------
    sanity = {}
    for sd in SANITY_DATES:
        cand = [(rd, dv) for rd, dv in dev_hist if rd <= sd]
        if not cand:
            continue
        rd, dv = max(cand, key=lambda x: x[0])
        sanity[str(sd)] = dict(refit_date=str(rd),
                               **{t: round(dv.get(t, 0.0), 3) for t in SANITY_TEAMS})
    lastdev = dev_hist[-1][1]
    devsummary = dict(
        last_refit=str(dev_hist[-1][0]),
        top5=dict(sorted(lastdev.items(), key=lambda kv: -kv[1])[:5]),
        bot5=dict(sorted(lastdev.items(), key=lambda kv: kv[1])[:5]),
        mean_abs=round(float(np.mean([abs(v) for v in lastdev.values()])), 3))
    devsummary["top5"] = {k: round(v, 3) for k, v in devsummary["top5"].items()}
    devsummary["bot5"] = {k: round(v, 3) for k, v in devsummary["bot5"].items()}

    def sub(mask):
        return dict(n=int(mask.sum()), delta=paired_ci(d[mask]),
                    ll_control=round(float(ll_c[mask].mean()), 5),
                    ll_variant=round(float(ll_v[mask].mean()), 5))

    res = dict(
        config=dict(dev_ridge=DEV_RIDGE, quality_ridge=QUALITY_RIDGE,
                    window_days=WINDOW_DAYS,
                    gate="paired bootstrap 2000x 95% CI, variant vs control"),
        replication=repl,
        control_ll=dict(
            pooled=round(float(ll_c.mean()), 5),
            market=round(float(ll_m.mean()), 5),
            per_season={s: round(float(ll_c[seas == s].mean()), 4)
                        for s in SEASONS}),
        variant_ll=dict(
            pooled=round(float(ll_v.mean()), 5),
            per_season={s: round(float(ll_v[seas == s].mean()), 4)
                        for s in SEASONS}),
        gate=dict(
            pooled=sub(np.ones(len(d), bool)),
            per_season={s: sub(seas == s) for s in SEASONS},
            den_uta_home=sub(alt_home)),
        diag_buckets={
            "gp[0,5)": sub(gpmin < 5),
            "gp[5,20)": sub((gpmin >= 5) & (gpmin < 20)),
            "gp[20,)": sub(gpmin >= 20)},
        home_dev_sanity=sanity,
        home_dev_summary=devsummary)
    with open(OUT_DIR / "es2b_teamhome_results.json", "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
