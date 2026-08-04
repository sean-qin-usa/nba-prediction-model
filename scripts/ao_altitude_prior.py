"""AO TEST 1 — ALTITUDE WITH A PHYSIO PRIOR (D70 retry; Sean's directive,
pre-registered, ONE config, no sweeps).

D70's Gate B (es2b_teamhome.py) fit per-team home-devs for ALL 30 teams with
ridge-200 shrinkage toward ZERO and failed (pooled -0.00002 exact null; the
DEN/UTA altitude subset showed no concentration — drowned by 28 teams of
nonstationary noise). Sean's critique: altitude is a PHYSICAL CONSTANT — the
correct prior is a positive floor that decays with era (rest science, travel,
load management) but never vanishes, and the construction should carry ONLY
the two altitude cities.

CONSTRUCTION (all constants fixed at pre-registration; only the data-side
estimate updates walk-forward):
  * home_dev estimated for DEN and UTA ONLY. At each weekly refit, from the
    SAME trailing 730-day window of regular-season finals fit_schedule_layer
    uses (strict PIT), fit a generalized ridge on the margin scale:
        home_margin ~ quality +1/-1 team columns   (ridge 30)
                    + unpenalized global home intercept
                    + DEN-home indicator + UTA-home indicator
                      (ridge 200 — D70's gated shrinkage strength — toward the
                       PHYSIO PRIOR instead of zero:
                       solve (X'X + P) b = X'y + P*prior)
  * physio prior (points, era-decaying; fixed here from pre-2023 public
    altitude-HCA estimates, decay rate fixed):
        prior_t(DEN) = 1.0 * 0.85 ** years_since_2015
        prior_t(UTA) = 0.6 * 0.85 ** years_since_2015
        years_since_2015 = (refit_date - 2015-01-01).days / 365.25
  * application: variant margin = control margin + home_dev(home team) for
    DEN/UTA HOME GAMES ONLY. All other games bitwise-unchanged.
  * window < 200 finals (never happens in 2023-26) -> dev = prior (the floor).

CONTROL = same-run unmodified fit_production (CURRENT production: D62 carry +
D73 tank + codex-round-6 fixes). Loop is the prod_by_season.py loop verbatim
(default oracle-outs / bought-availability tier, weekly refit). Replication
checked vs data/capstone_pergame_tank.csv (expected ~1e-14 rerun jitter, D63).

GATE (pre-registered): paired bootstrap 2000x 95% CI on per-game logloss
deltas (control - variant; positive = variant better): POOLED + DEN/UTA-HOME
subset (n~246). Per-season and per-venue splits reported as diagnostics.

Read-only DB. NEW file scripts/ao_altitude_prior.py only; nbapred/ untouched.
Outputs: data/ao_altitude_pergame.csv, data/ao_altitude_results.json.
Run:      python scripts/ao_altitude_prior.py            (full walk-forward)
Analyze:  python scripts/ao_altitude_prior.py --analyze  (bootstrap from CSV)
"""
import bisect
import csv
import datetime as dt
import json
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
QUALITY_RIDGE = 30.0          # margin-scale team-quality penalty
DEV_RIDGE = 200.0             # D70's gated shrinkage strength (unchanged)
WINDOW_DAYS = 730             # fit_schedule_layer's trailing window
DEN_ID, UTA_ID = 1610612743, 1610612762
PRIOR_2015 = {DEN_ID: 1.0, UTA_ID: 0.6}   # pts, pre-2023 public estimates
DECAY = 0.85                               # per-year era decay (fixed)
EPOCH = dt.date(2015, 1, 1)
PERGAME_CSV = OUT_DIR / "ao_altitude_pergame.csv"
RESULTS_JSON = OUT_DIR / "ao_altitude_results.json"
CAPSTONE = OUT_DIR / "capstone_pergame_tank.csv"


def physio_prior(refit_date):
    yrs = (refit_date - EPOCH).days / 365.25
    return {t: p * DECAY ** yrs for t, p in PRIOR_2015.items()}


def fetch_margin_rows(con):
    """All regular-season finals (date, home_id, away_id, home_margin),
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


def fit_alt_dev(window, prior):
    """Generalized ridge toward the physio prior; {team_id: dev} for DEN/UTA.

    Quality columns ridge-30 toward 0, global home intercept unpenalized,
    the TWO altitude home-dev columns ridge-200 toward prior_t (penalty
    p*(b - prior)^2 -> normal equations (X'X + P) b = X'y + P*prior)."""
    if len(window) < 200:
        return dict(prior)
    teams = sorted({t for _, h, a, _ in window for t in (h, a)})
    idx = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    n = len(window)
    dev_col = {DEN_ID: T + 1, UTA_ID: T + 2}
    X = np.zeros((n, T + 3))
    y = np.zeros(n)
    for i, (_, ht, at, m) in enumerate(window):
        X[i, idx[ht]] += 1.0            # quality (+home)
        X[i, idx[at]] -= 1.0            # quality (-away)
        X[i, T] = 1.0                   # global home edge (unpenalized)
        if ht in dev_col:
            X[i, dev_col[ht]] = 1.0     # altitude home dev
        y[i] = m
    P = np.zeros(T + 3)
    P[:T] = QUALITY_RIDGE
    P[T + 1:] = DEV_RIDGE
    m0 = np.zeros(T + 3)
    m0[T + 1] = prior[DEN_ID]
    m0[T + 2] = prior[UTA_ID]
    beta = np.linalg.solve(X.T @ X + np.diag(P), X.T @ y + P * m0)
    return {DEN_ID: float(beta[T + 1]), UTA_ID: float(beta[T + 2])}


def season_run(season, margin_rows, margin_dates, dev_hist):
    """prod_by_season.py loop VERBATIM (default oracle-outs, weekly refit);
    control = fit_production margin, variant = control + altitude dev on
    DEN/UTA home games."""
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

    rows = []
    gp_live = {}
    model = comp = None
    alt_dev = {}
    prior = {}
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
            prior = physio_prior(gd)
            lo = bisect.bisect_left(margin_dates, gd - dt.timedelta(days=WINDOW_DAYS))
            hi = bisect.bisect_left(margin_dates, gd)
            alt_dev = fit_alt_dev(margin_rows[lo:hi], prior)
            dev_hist.append((gd, dict(prior), dict(alt_dev)))
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
        mm = model.margin(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id],
                          gd, b2b_home=b2b(h.team_id, gd), b2b_away=b2b(a.team_id, gd))
        dev_h = alt_dev.get(h.team_id, 0.0) if h.team_id in (DEN_ID, UTA_ID) else 0.0
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


def analyze(all_rows, dev_hist):
    y = np.array([r["y"] for r in all_rows])
    seas = np.array([r["season"] for r in all_rows])
    home = np.array([r["home"] for r in all_rows])
    alt_home = (home == "DEN") | (home == "UTA")
    ll_c = ll_vec(y, [r["p_ctrl"] for r in all_rows])
    ll_v = ll_vec(y, [r["p_var"] for r in all_rows])
    ll_m = ll_vec(y, [r["p_mkt"] for r in all_rows])
    d = ll_c - ll_v                       # positive = variant better

    # ---- control replication vs shipped tank capstone ---------------------
    base = {}
    with open(CAPSTONE) as f:
        for r in csv.DictReader(f):
            base[(r["season"], r["game_id"])] = float(r["p_us"])
    diffs = [abs(base[(r["season"], r["game_id"])] - r["p_ctrl"])
             for r in all_rows if (r["season"], r["game_id"]) in base]
    repl = dict(baseline=CAPSTONE.name, n_matched=len(diffs),
                n_ours=len(all_rows),
                max_abs_diff=float(max(diffs)) if diffs else None,
                note="expected ~1e-14 rerun jitter (D63)")

    def sub(mask):
        return dict(n=int(mask.sum()), delta=paired_ci(d[mask]),
                    ll_control=round(float(ll_c[mask].mean()), 5),
                    ll_variant=round(float(ll_v[mask].mean()), 5),
                    ll_market=round(float(ll_m[mask].mean()), 5))

    sanity = {}
    if dev_hist:
        for gd, pr, dv in (dev_hist[0], dev_hist[len(dev_hist) // 2], dev_hist[-1]):
            sanity[str(gd)] = dict(
                prior_DEN=round(pr[DEN_ID], 3), dev_DEN=round(dv[DEN_ID], 3),
                prior_UTA=round(pr[UTA_ID], 3), dev_UTA=round(dv[UTA_ID], 3))
    res = dict(
        config=dict(dev_ridge=DEV_RIDGE, quality_ridge=QUALITY_RIDGE,
                    window_days=WINDOW_DAYS,
                    prior_2015={"DEN": PRIOR_2015[DEN_ID], "UTA": PRIOR_2015[UTA_ID]},
                    decay_per_year=DECAY, epoch=str(EPOCH),
                    gate="paired bootstrap 2000x 95% CI; POOLED + DEN/UTA-home"),
        replication=repl,
        gate=dict(pooled=sub(np.ones(len(d), bool)),
                  den_uta_home=sub(alt_home)),
        diagnostics=dict(
            per_season={s: sub(seas == s) for s in SEASONS},
            den_home=sub(home == "DEN"),
            uta_home=sub(home == "UTA"),
            den_uta_home_per_season={s: sub(alt_home & (seas == s))
                                     for s in SEASONS},
            mean_dev_applied=round(float(np.mean(
                [r["dev_home"] for r in all_rows if r["dev_home"] != 0.0])), 3)),
        dev_sanity=sanity)
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


def main():
    if "--analyze" in sys.argv:
        with open(PERGAME_CSV) as f:
            all_rows = []
            for r in csv.DictReader(f):
                r.update(y=int(r["y"]), p_mkt=float(r["p_mkt"]),
                         p_ctrl=float(r["p_ctrl"]), p_var=float(r["p_var"]),
                         dev_home=float(r["dev_home"]),
                         gp_home=int(r["gp_home"]), gp_away=int(r["gp_away"]))
                all_rows.append(r)
        analyze(all_rows, [])
        return
    con = connect(read_only=True)
    margin_rows = fetch_margin_rows(con)
    con.close()
    margin_dates = [r[0] for r in margin_rows]
    all_rows = []
    dev_hist = []            # (refit_date, prior, dev) across all seasons
    for s in SEASONS:
        all_rows += season_run(s, margin_rows, margin_dates, dev_hist)
    with open(PERGAME_CSV, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(all_rows)
    analyze(all_rows, dev_hist)


if __name__ == "__main__":
    main()
