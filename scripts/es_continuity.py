"""ES experiment: cross-season CONTINUITY-WEIGHTED rolling of the team side.

Sean's question: "should we roll everything season to season?"

Today's production: each season's FourFactors starts empty (not ready until 200
team-game rows ~ 3 weeks) and the ratings fallback uses last-season ratings x
flat 0.75 fading linearly over 20 games.

Variants (pre-registered, no wider grid):
 (a) SEED the per-factor ridges with LAST season's factor_game_rows as
     pseudo-observations, weight w_carry = w0 * 0.5**(gp_team/h): FF 'ready'
     from opening night. Grid: h in {10,20} x w0 in {0.3,0.6} = 4 configs.
 (b) CONTINUITY-weighted prior regression: regress_t = 0.5 + 0.45*continuity_t
     (continuity_t = minutes-weighted share of last season's team minutes
     returning on this season's opening roster; opening roster = players
     appearing in the team's first 5 games — sanctioned PIT approximation:
     rosters are known before opening night). Applied to the ratings-fallback
     prior AND as a multiplier on carried FF pseudo-row weights.

GATED configs = the 4 (h,w0) combined (a)+(b) variants. Diagnostic (ungated)
arms: B_prior_only ((b) on the prior alone, no seeding — the only arm where the
prior half of (b) is actually exercised, since seeded FF never hits the
fallback) and A-only flat-carry twins of the 4 grid points (isolate the
continuity multiplier).

DATA CONSTRAINT: player_game_stats starts 2023-24 -> no 2022-23 factor rows or
minutes. For season 2023-24 every variant reduces to control BY CONSTRUCTION
(deltas exactly 0); the experiment is live for 2024-25 and 2025-26 only.

Control = same-run exact replication of shipped production (prod_by_season.py
loop, default oracle-outs, weekly refit), verified against
data/capstone_pergame_csfix.csv. Gate: paired bootstrap 2000x, 95% CI, pooled +
per-season + first-20-team-games subset (either team gp<20; the target window).

Read-only DB. Outputs to scratchpad only.
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
from nbapred.eval.metrics import log_loss
from nbapred.model.composition import CompositionModel
from nbapred.model.four_factors import FACTORS, FourFactors, factor_game_rows
from nbapred.model.production import (SCALE, fit_schedule_layer,
                                      last_season_prior, sigmoid)
from nbapred.model.team_ratings import TeamRatings, game_rows

OUT_DIR = Path(os.environ.get(
    "ES_OUT",
    "/tmp/claude-1004/-hdd-steveqin-sean-dev/4912ce58-69cd-488d-a5f4-0d3ed2eed30c/scratchpad"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = ("2023-24", "2024-25", "2025-26")
W_COMP = 0.7
DEFAULT_CONT = 0.5555555555555556  # 0.5 + 0.45*c == 0.75 (flat fallback)

# name -> (seed_ff, h, w0, cont_rows, cont_prior)
CONFIGS = {
    # gated (pre-registered grid, (a)+(b) combined)
    "AB_h10_w03": (True, 10.0, 0.3, True, True),
    "AB_h10_w06": (True, 10.0, 0.6, True, True),
    "AB_h20_w03": (True, 20.0, 0.3, True, True),
    "AB_h20_w06": (True, 20.0, 0.6, True, True),
    # diagnostic arms (not gated; same grid points, components isolated)
    "B_prior_only": (False, None, None, False, True),
    "A_h10_w03": (True, 10.0, 0.3, False, False),
    "A_h10_w06": (True, 10.0, 0.6, False, False),
    "A_h20_w03": (True, 20.0, 0.3, False, False),
    "A_h20_w06": (True, 20.0, 0.6, False, False),
}
GATED = ["AB_h10_w03", "AB_h10_w06", "AB_h20_w03", "AB_h20_w06"]


class SeededFF:
    """FourFactors with prior-season pseudo-rows. Replicates
    nbapred.model.four_factors.FourFactors.fit math exactly (ridge=25, no
    luck adjust, factor x100, factor->ortg lstsq on the same rows) with row
    weights added; with no carried rows and unit weights it IS the production
    fit."""

    def __init__(self, ridge: float = 25.0):
        self.ridge = ridge
        self.fms = {}
        self.W = None

    def fit(self, prev_rows, cur_rows, prev_w):
        rows = prev_rows + cur_rows
        if len(rows) < 200:
            return self
        w = (np.concatenate([prev_w, np.ones(len(cur_rows))])
             if len(prev_rows) else np.ones(len(cur_rows)))
        self.fms = {f: TeamRatings(ridge=self.ridge, team_home_ridge=None).fit(
            [(x["tid"], x["oid"], x["home"], x[f] * 100) for x in rows],
            weights=w) for f in FACTORS}
        X = np.array([[self.fms[f].pred_ortg(x["tid"], x["oid"], x["home"])
                       for f in FACTORS] for x in rows])
        y = np.array([x["ortg"] for x in rows])
        A = np.c_[X, np.ones(len(X))]
        sw = np.sqrt(w)
        self.W = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        return self

    def eortg(self, tid, oid, is_home):
        xf = np.array([self.fms[f].pred_ortg(tid, oid, is_home) for f in FACTORS])
        return float(xf @ self.W[:4] + self.W[4])

    def margin_neutral(self, home_id, away_id):
        return (self.eortg(home_id, away_id, False)
                - self.eortg(away_id, home_id, False))

    @property
    def ready(self):
        return self.W is not None and bool(self.fms)


def prev_season(season: str) -> str:
    y0 = int(season[:4])
    return f"{y0-1}-{str(y0)[-2:]}"


def continuity_map(con, season: str):
    """{team_id: minutes-weighted share of last season's team minutes present
    on this season's opening roster (players in the team's first 5 games)}.
    None if prior-season player data is missing."""
    prev = prev_season(season)
    pm = con.execute("""
        SELECT s.team_id, s.player_id, sum(s.seconds)/60.0 mins
        FROM player_game_stats s
        JOIN nba_games g ON g.game_id = s.game_id AND g.team_id = s.team_id
        WHERE g.season = ? AND s.game_id LIKE '002%'
        GROUP BY 1, 2""", [prev]).fetchall()
    if not pm:
        return None
    roster = con.execute("""
        WITH tg AS (
          SELECT team_id, game_id,
                 row_number() OVER (PARTITION BY team_id ORDER BY game_date, game_id) rn
          FROM nba_games WHERE season = ? AND game_id LIKE '002%')
        SELECT tg.team_id, s.player_id
        FROM tg JOIN player_game_stats s
          ON s.game_id = tg.game_id AND s.team_id = tg.team_id
        WHERE tg.rn <= 5 GROUP BY 1, 2""", [season]).fetchall()
    if not roster:
        return None
    ros = {}
    for t, p in roster:
        ros.setdefault(int(t), set()).add(int(p))
    tot, ret = {}, {}
    for t, p, m in pm:
        t = int(t)
        tot[t] = tot.get(t, 0.0) + m
        if int(p) in ros.get(t, set()):
            ret[t] = ret.get(t, 0.0) + m
    return {t: ret.get(t, 0.0) / tot[t] for t in tot if tot[t] > 0}


def season_run(season: str):
    """prod_by_season.py loop (default oracle-outs) with control + all variant
    margins computed side by side from SHARED components."""
    t0 = time.time()
    con = connect(read_only=True)

    # ---- season-constant pieces -------------------------------------------
    cont = continuity_map(con, season)                    # None for 2023-24
    prev_rows = factor_game_rows(con, prev_season(season))  # [] for 2023-24
    live = bool(prev_rows) and cont is not None
    raw_prior = last_season_prior(con, season, regress=1.0)
    prior_ctrl = {t: 0.75 * r for t, r in raw_prior.items()}
    ab = dict(con.execute(
        "SELECT DISTINCT team_id, team_abbrev FROM nba_games WHERE season=?",
        [season]).fetchall())
    id2ab = {t: a for t, a in ab.items()}
    ab2id = {a: t for t, a in ab.items()}
    if live:
        cont_by_ab = {id2ab[t]: c for t, c in cont.items() if t in id2ab}
        prior_cont = {a: (0.5 + 0.45 * cont_by_ab.get(a, DEFAULT_CONT)) * r
                      for a, r in raw_prior.items()}
    else:
        prior_cont = dict(prior_ctrl)
    # current-season factor rows, fetched once, date-sliced per refit (identical
    # to the before= SQL filter)
    cur_rows_all = sorted(factor_game_rows(con, season), key=lambda x: x["date"])
    cur_dates = [x["date"] for x in cur_rows_all]
    prev_tids = np.array([x["tid"] for x in prev_rows], dtype=int) if prev_rows else None
    cont_prev = (np.array([cont.get(int(t), DEFAULT_CONT) for t in prev_tids])
                 if live else None)

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
    names = list(CONFIGS)
    rows = []
    gp_live = {}          # per-team games completed so far (for subset label)
    comp = tr = None
    sched = None
    ff_ctrl = None
    ff_seeded = {}
    games_played = {}
    last = None
    nrefit = 0
    first_refit_diag = None
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
            if live:
                k = bisect.bisect_left(cur_dates, gd)
                cur = cur_rows_all[:k]
                gp_arr = np.array([games_played.get(int(t), 0) for t in prev_tids],
                                  dtype=float)
                for name, (seed, hh, w0, c_rows, _cp) in CONFIGS.items():
                    if not seed:
                        continue
                    w = w0 * 0.5 ** (gp_arr / hh)
                    if c_rows:
                        w = w * cont_prev
                    ff_seeded[name] = SeededFF().fit(prev_rows, cur, w)
                if first_refit_diag is None:
                    # sanity: opening-night seeded margins vs prev-season FF
                    tids = sorted({int(t) for t in prev_tids})
                    ffp = FourFactors().fit(con, prev_season(season))
                    ref = np.array([ffp.margin_neutral(t, tids[0]) for t in tids])
                    diag = {}
                    for name, ffs in ff_seeded.items():
                        v = np.array([ffs.margin_neutral(t, tids[0]) for t in tids])
                        diag[name] = round(float(np.corrcoef(ref, v)[0, 1]), 4)
                    first_refit_diag = dict(
                        date=str(gd), corr_vs_prev_ff=diag,
                        n_cur_rows=k, ff_ctrl_ready=bool(ff_ctrl.ready))
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

        def ratings_margin(prior):
            mm = tr.pred_margin(h.team_id, a.team_id)
            gh = games_played.get(h.team_id, 0)
            ga = games_played.get(a.team_id, 0)
            wh = max(0.0, 1 - gh / 20.0)
            wa = max(0.0, 1 - ga / 20.0)
            return (mm + wh * prior.get(id2ab.get(h.team_id, ""), 0.0)
                    - wa * prior.get(id2ab.get(a.team_id, ""), 0.0))

        def margin_for(ffm, prior):
            if ffm is not None and ffm.ready:
                return 0.5 * ffm.margin_neutral(h.team_id, a.team_id) + 0.5 * cm + sch
            rm = ratings_margin(prior) - tr.home
            return W_COMP * cm + (1 - W_COMP) * rm + sch

        ps = {"control": float(sigmoid(margin_for(ff_ctrl, prior_ctrl) / SCALE))}
        for name, (seed, hh, w0, c_rows, c_prior) in CONFIGS.items():
            if not live:
                ps[name] = ps["control"]
                continue
            ffm = ff_seeded[name] if seed else ff_ctrl
            prior = prior_cont if c_prior else prior_ctrl
            ps[name] = float(sigmoid(margin_for(ffm, prior) / SCALE))
        rows.append(dict(
            season=season, game_id=gid, game_date=str(gd)[:10],
            home=h.team_abbrev, away=a.team_abbrev, y=int(h.wl == "W"),
            p_mkt=float(pmv), gp_home=gph, gp_away=gpa, **ps))
    con.close()
    print(f"[{season}] n={len(rows)} refits={nrefit} live={live} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if first_refit_diag:
        print(f"[{season}] first-refit diag: {first_refit_diag}", flush=True)
    if live:
        cshow = sorted(((id2ab.get(t, t), round(c, 3)) for t, c in cont.items()),
                       key=lambda x: x[1])
        print(f"[{season}] continuity min/med/max: {cshow[0]} "
              f"{cshow[len(cshow)//2]} {cshow[-1]}", flush=True)
    return rows, first_refit_diag, (cont if live else None)


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
    all_rows, diags, conts = [], {}, {}
    for s in SEASONS:
        r, d, c = season_run(s)
        all_rows += r
        diags[s] = d
        conts[s] = {str(k): v for k, v in (c or {}).items()}

    import csv
    with open(OUT_DIR / "es_continuity_pergame.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(all_rows)

    y = np.array([r["y"] for r in all_rows])
    seas = np.array([r["season"] for r in all_rows])
    early = np.array([(r["gp_home"] < 20 or r["gp_away"] < 20) for r in all_rows])
    ll_c = ll_vec(y, [r["control"] for r in all_rows])
    ll_m = ll_vec(y, [r["p_mkt"] for r in all_rows])

    # control vs shipped baseline CSV (replication check)
    base = {}
    with open(REPO / "data" / "capstone_pergame_csfix.csv") as f:
        for r in csv.DictReader(f):
            base[(r["season"], r["game_id"])] = float(r["p_us"])
    diffs = [abs(base[(r["season"], r["game_id"])] - r["control"])
             for r in all_rows if (r["season"], r["game_id"]) in base]
    repl = dict(n_matched=len(diffs), n_ours=len(all_rows),
                max_abs_diff=float(max(diffs)) if diffs else None)

    res = dict(replication=repl, diags=diags,
               continuity=conts,
               n=dict(total=len(all_rows), early=int(early.sum()),
                      per_season={s: int((seas == s).sum()) for s in SEASONS}),
               control_ll=dict(
                   pooled=round(float(ll_c.mean()), 5),
                   market=round(float(ll_m.mean()), 5),
                   early=round(float(ll_c[early].mean()), 5),
                   market_early=round(float(ll_m[early].mean()), 5),
                   per_season={s: round(float(ll_c[seas == s].mean()), 5)
                               for s in SEASONS},
                   early_per_season={s: round(float(ll_c[(seas == s) & early].mean()), 5)
                                     for s in SEASONS}),
               variants={})
    for name in CONFIGS:
        ll_v = ll_vec(y, [r[name] for r in all_rows])
        d = ll_c - ll_v          # positive = variant better
        ent = dict(
            pooled=paired_ci(d),
            early=paired_ci(d[early]),
            early_ll=round(float(ll_v[early].mean()), 5),
            pooled_ll=round(float(ll_v.mean()), 5),
            per_season={s: paired_ci(d[seas == s]) for s in SEASONS},
            early_per_season={s: paired_ci(d[(seas == s) & early]) for s in SEASONS},
            gated=name in GATED)
        res["variants"][name] = ent
    with open(OUT_DIR / "es_continuity_results.json", "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
