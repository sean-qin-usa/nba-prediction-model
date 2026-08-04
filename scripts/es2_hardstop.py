"""ES2 GATE A (pre-registered follow-up from es_continuity, journal 786628):
cross-season FourFactors carry with HARD STOP.

ONE config — no sweeps, no post-hoc variants:
  Seed the FourFactors fit with LAST season's factor_game_rows as
  pseudo-observations, per-row weight w = 0.3 * continuity_team, where
  continuity_team = minutes-weighted share of last season's team minutes
  returning on this season's opening roster (players in the team's first 5
  games — identical construction to es_continuity.continuity_map, sanctioned
  PIT approximation: rosters are known before opening night).
  Carried rows are used ONLY while the current-season FF is NOT ready
  (<200 current factor rows at the weekly refit). The moment ff_ctrl is
  ready, carried rows are dropped ENTIRELY (hard stop) — the variant is
  bitwise identical to control from that refit on.

Evidence basis (es_continuity): gp[0,5) carry value +0.0158 CI(+0.004,+0.027)
in all 4 grid configs; losses appeared only where carry persisted past
readiness. The hard stop keeps the proven window and removes the proven harm.

Control = same-run exact replication of shipped production (prod_by_season.py
loop, default oracle-outs, weekly refit), verified against
data/capstone_pergame_csfix.csv (expected max|diff| ~1e-14).

Gate (pre-registered): paired bootstrap 2000x, 95% CI, variant-vs-control.
Primary endpoint: pooled 3-season delta (CI excluding 0 in favor = PASS).
Also reported: per-season, early subset (either team gp<20), gp buckets by
min(gp_home, gp_away) — [0,5) is the evidence bucket — and the carry-active
subset (games scored with carried rows in the blend).

DATA CONSTRAINT: player_game_stats starts 2023-24, so 2023-24 has no prior
rows and is inert (delta exactly 0 by construction); live evidence is
2024-25 and 2025-26.

Read-only DB. Outputs to scratchpad only.
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
DEFAULT_CONT = 0.5555555555555556   # same fallback as es_continuity
CARRY_W0 = 0.3                      # pre-registered; NOT tuned here


class SeededFF:
    """FourFactors with prior-season pseudo-rows (copied verbatim from
    scripts/es_continuity.py). Replicates FourFactors.fit math exactly
    (ridge=25, no luck adjust, factor x100, factor->ortg lstsq on the same
    rows) with row weights added; with no carried rows and unit weights it IS
    the production fit."""

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
    None if prior-season player data is missing. (Verbatim from
    scripts/es_continuity.py.)"""
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
    """prod_by_season.py loop (default oracle-outs) with control + hardstop
    variant computed side by side from SHARED components (control pattern
    copied from scripts/es_continuity.py)."""
    t0 = time.time()
    con = connect(read_only=True)

    # ---- season-constant pieces -------------------------------------------
    cont = continuity_map(con, season)                      # None for 2023-24
    prev_rows = factor_game_rows(con, prev_season(season))  # [] for 2023-24
    live = bool(prev_rows) and cont is not None
    raw_prior = last_season_prior(con, season, regress=1.0)
    prior_ctrl = {t: 0.75 * r for t, r in raw_prior.items()}
    ab = dict(con.execute(
        "SELECT DISTINCT team_id, team_abbrev FROM nba_games WHERE season=?",
        [season]).fetchall())
    id2ab = {t: a for t, a in ab.items()}
    # current-season factor rows, fetched once, date-sliced per refit
    # (identical to the before= SQL filter)
    cur_rows_all = sorted(factor_game_rows(con, season), key=lambda x: x["date"])
    cur_dates = [x["date"] for x in cur_rows_all]
    if live:
        prev_tids = np.array([x["tid"] for x in prev_rows], dtype=int)
        # PRE-REGISTERED carry weight: w0 * continuity_team, constant while
        # active (the hard stop replaces any decay schedule).
        w_prev = CARRY_W0 * np.array(
            [cont.get(int(t), DEFAULT_CONT) for t in prev_tids])
    else:
        w_prev = None

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
    ff_hard = None            # seeded FF; None once the hard stop fires
    games_played = {}
    last = None
    nrefit = 0
    handover = None           # first refit date with ff_ctrl ready
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
            if live and not ff_ctrl.ready:
                # HARD-STOP CONDITION: carried rows exist only on this branch.
                k = bisect.bisect_left(cur_dates, gd)
                ff_hard = SeededFF().fit(prev_rows, cur_rows_all[:k], w_prev)
                if first_refit_diag is None:
                    tids = sorted({int(t) for t in prev_tids})
                    ffp = FourFactors().fit(con, prev_season(season))
                    ref = np.array([ffp.margin_neutral(t, tids[0]) for t in tids])
                    v = np.array([ff_hard.margin_neutral(t, tids[0]) for t in tids])
                    first_refit_diag = dict(
                        date=str(gd),
                        corr_vs_prev_ff=round(float(np.corrcoef(ref, v)[0, 1]), 4),
                        n_cur_rows=k, ff_ctrl_ready=bool(ff_ctrl.ready))
            else:
                if live and ff_hard is not None and handover is None:
                    handover = str(gd)
                ff_hard = None    # hard stop: carried rows dropped entirely
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

        p_ctrl = float(sigmoid(margin_for(ff_ctrl, prior_ctrl) / SCALE))
        if ff_hard is not None:
            p_var = float(sigmoid(margin_for(ff_hard, prior_ctrl) / SCALE))
            carried = 1
        else:
            p_var = p_ctrl        # identical by construction after hard stop
            carried = 0
        rows.append(dict(
            season=season, game_id=gid, game_date=str(gd)[:10],
            home=h.team_abbrev, away=a.team_abbrev, y=int(h.wl == "W"),
            p_mkt=float(pmv), gp_home=gph, gp_away=gpa,
            control=p_ctrl, hardstop=p_var, carried=carried))
    con.close()
    ncar = sum(r["carried"] for r in rows)
    print(f"[{season}] n={len(rows)} refits={nrefit} live={live} "
          f"carry_games={ncar} handover={handover} ({time.time()-t0:.0f}s)",
          flush=True)
    if first_refit_diag:
        print(f"[{season}] first-refit diag: {first_refit_diag}", flush=True)
    return rows, first_refit_diag, handover


def paired_ci(d, B=2000, seed=42):
    d = np.asarray(d, float)
    if len(d) == 0:
        return (0.0, 0.0, 0.0, 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (B, len(d)))
    means = d[idx].mean(axis=1)
    return (float(d.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)), int(len(d)))


def ll_vec(y, p):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    all_rows, diags, handovers = [], {}, {}
    for s in SEASONS:
        r, d, ho = season_run(s)
        all_rows += r
        diags[s] = d
        handovers[s] = ho

    with open(OUT_DIR / "es2_hardstop_pergame.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(all_rows)

    y = np.array([r["y"] for r in all_rows])
    seas = np.array([r["season"] for r in all_rows])
    gph = np.array([r["gp_home"] for r in all_rows])
    gpa = np.array([r["gp_away"] for r in all_rows])
    early = (gph < 20) | (gpa < 20)
    gmin = np.minimum(gph, gpa)
    carried = np.array([r["carried"] for r in all_rows], bool)
    ll_c = ll_vec(y, [r["control"] for r in all_rows])
    ll_v = ll_vec(y, [r["hardstop"] for r in all_rows])
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

    def slice_lls(mask):
        return dict(control=round(float(ll_c[mask].mean()), 5),
                    hardstop=round(float(ll_v[mask].mean()), 5),
                    market=round(float(ll_m[mask].mean()), 5), n=int(mask.sum()))

    res = dict(
        config=dict(w0=CARRY_W0, weight="w0 * continuity_team (constant)",
                    stop="carried rows dropped the refit ff_ctrl.ready flips"),
        replication=repl, diags=diags, handovers=handovers,
        n=dict(total=len(all_rows), early=int(early.sum()),
               carried=int(carried.sum()),
               carried_per_season={s: int((carried & (seas == s)).sum())
                                   for s in SEASONS},
               per_season={s: int((seas == s).sum()) for s in SEASONS}),
        lls=dict(pooled=slice_lls(np.ones(len(y), bool)),
                 early=slice_lls(early),
                 gp05=slice_lls(gmin < 5),
                 carried=slice_lls(carried),
                 per_season={s: slice_lls(seas == s) for s in SEASONS},
                 early_per_season={s: slice_lls((seas == s) & early)
                                   for s in SEASONS}),
        gate=dict(
            pooled=paired_ci(d),
            per_season={s: paired_ci(d[seas == s]) for s in SEASONS},
            early=paired_ci(d[early]),
            early_per_season={s: paired_ci(d[(seas == s) & early]) for s in SEASONS},
            gp_buckets={f"[{lo},{hi})": paired_ci(d[(gmin >= lo) & (gmin < hi)])
                        for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 200))},
            gp05_per_season={s: paired_ci(d[(seas == s) & (gmin < 5)])
                             for s in SEASONS},
            carried_subset=paired_ci(d[carried])))
    with open(OUT_DIR / "es2_hardstop_results.json", "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
