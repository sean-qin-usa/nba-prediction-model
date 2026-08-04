"""LATE-STATE LAYER v1 GATE (D84-D top candidate — pre-registered, ONE config).

Spec (verbatim from the regime-D deep-dive's top_candidate_spec, wf_f9bb526c):

  TERM (inside Predictor.margin, applied on top of the shipped stack):
      active iff either team gp_before >= 55 (identical activation to
      tanking.py; adds exactly 0.0 otherwise):
          m += c_f * (form5_h - form5_a) + c_o * (nout_h - nout_a)

  FEATURES (PIT):
      form5   = same-season trailing-5-game mean signed margin, shift(1),
                0 until 5 games
      nout_*  = the OUT-set counts already passed to p_home (headline tier:
                oracle outs — same tier as capstone_pergame_tank.csv)

  FIT (production idiom, mirrors fit_schedule_layer / tanking.py):
      daily walk-forward OLS  y = home_margin,
      X = [1, tsd, fdiff, outdiff, wdiff]
      over ALL completed ACTIVE games from 2022-23 onward (outdiff = 0 for
      2022-23 burn-in rows; in-run seasons use the counts computed by this
      same run), coefficients shrunk n/(n+600), sign guards c_f = max(0, .),
      c_o = min(0, .), both 0 until 20 active rows.  tsd and wdiff are
      FIT-TIME CONTROLS ONLY — never applied.  The shipped tank k is NOT
      refit (L4 evidence: joint refit hurts).

  CONTROL = shipped production EXACTLY (fit_production incl. D62 carry +
  D73 tank), prod_by_season.py loop verbatim (weekly refit, oracle-outs
  path).  Replication check vs data/capstone_pergame_tank.csv (headline).

  GATE (pre-registered): paired bootstrap 2000x 95% CI on per-game logloss
  deltas (control - variant; positive = variant better).  SHIP iff the
  POOLED CI excludes 0.  Report active window, per-season, Jan/Feb-Mar-Apr
  pockets, and d-vs-market on the active window (expected +0.0168 shipped
  -> +0.0112 with the layer; expected active delta +0.0056
  CI(+0.0015,+0.0099), pooled ~+0.0019).

Read-only DB; no sweeps; new file — nbapred/ untouched by the gate run.

REPRODUCTION NOTE (post-ship): this gate ran on PRE-D90 production (control
replicated capstone_pergame_tank.csv at 2e-14) and also predates the D84-A
October-bridge ship. To reproduce the control arm today, run with
LATE_STATE=0 OCT_BRIDGE=0; without them the "control" would already contain
the late-state layer (double-count) and the bridge's week-1 changes.
"""
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
from nbapred.model.tanking import get_tank_model

OUT_DIR = REPO / "data"
SEASONS = ("2023-24", "2024-25", "2025-26")
GP_ACTIVE = 55
FORM_N = 5
C_MIN_ACTIVE = 20            # coefs 0 until this many active fit rows
C_SHRINK = 600.0             # n/(n+600) shrink toward 0 (= SCHED_SHRINK)


# ---------------------------------------------------------------------------
# late-state layer: features + daily walk-forward coefficient fit
# ---------------------------------------------------------------------------

class LateStateLayer:
    """form5 map (all seasons incl 2022-23) + active-row history + daily
    (c_f, c_o) estimator.  2022-23 burn-in rows are built from the DB with
    outdiff=0 (per spec); 2023-24 onward rows are appended by the run loop
    with the same oracle-tier out counts passed to p_home."""

    def __init__(self, con, tank):
        self.tank = tank
        # -- form5 per (team_id, game_id): trailing-5 mean signed margin ----
        tg = con.execute("""
            SELECT g.season, g.game_id, g.game_date, g.team_id, g.pts,
                   o.pts AS opp_pts
            FROM nba_games g
            JOIN nba_games o ON o.game_id = g.game_id
                            AND o.team_id <> g.team_id
            WHERE g.game_id LIKE '002%' AND g.pts IS NOT NULL
              AND o.pts IS NOT NULL AND g.season >= '2022-23'
            ORDER BY g.team_id, g.game_date, g.game_id""").fetchall()
        self.form = {}                    # (team_id, game_id) -> form5
        hist = {}                         # (season, team_id) -> [sm, ...]
        for season, gid, d, tid, pts, opp in tg:
            k = (season, int(tid))
            h = hist.setdefault(k, [])
            self.form[(int(tid), gid)] = (
                float(np.mean(h[-FORM_N:])) if len(h) >= FORM_N else 0.0)
            h.append(float(pts - opp))
        # -- 2022-23 burn-in active rows (outdiff = 0 per spec) -------------
        g22 = con.execute("""
            SELECT season, game_id, game_date, team_id, team_abbrev,
                   matchup, pts, wl
            FROM nba_games WHERE game_id LIKE '002%' AND pts IS NOT NULL
              AND season = '2022-23' ORDER BY game_date, game_id""").fetchall()
        by = {}
        order = []
        for r in g22:
            if r[1] not in by:
                order.append(r[1])
            by.setdefault(r[1], []).append(r)
        self.rows = []                    # (date, margin, tsd, fdiff, outdiff, wdiff)
        wl_state = {}                     # team_id -> [wins, games]
        for gid in order:
            recs = by[gid]
            if len(recs) != 2:
                continue
            m = recs[0][5]
            host = (m.split("@")[-1].strip() if "@" in m
                    else m.split("vs.")[0].strip())
            h = next((x for x in recs if x[4] == host), None)
            a = next((x for x in recs if x[4] != host), None)
            if h is None or a is None:
                continue
            gd = h[2].date() if hasattr(h[2], "date") else h[2]
            ht, at = int(h[3]), int(a[3])
            wh = wl_state.setdefault(ht, [0, 0])
            wa = wl_state.setdefault(at, [0, 0])
            gp_h = self.tank.score(ht, gd)[1]
            gp_a = self.tank.score(at, gd)[1]
            if gp_h >= GP_ACTIVE or gp_a >= GP_ACTIVE:
                fdiff = (self.form.get((ht, gid), 0.0)
                         - self.form.get((at, gid), 0.0))
                wdiff = ((wh[0] / wh[1] if wh[1] else 0.5)
                         - (wa[0] / wa[1] if wa[1] else 0.5))
                self.rows.append((gd, float(h[6] - a[6]),
                                  float(self.tank.diff(ht, at, gd)),
                                  float(fdiff), 0.0, float(wdiff)))
            hw = h[6] > a[6]
            wh[0] += int(hw); wh[1] += 1
            wa[0] += int(not hw); wa[1] += 1
        self.n_burnin = len(self.rows)
        self._coef_date = None
        self._coef = (0.0, 0.0)

    def append(self, gd, margin, tsd, fdiff, outdiff, wdiff):
        self.rows.append((gd, float(margin), float(tsd), float(fdiff),
                          float(outdiff), float(wdiff)))

    def coefs(self, gd):
        """Daily walk-forward (c_f, c_o) on active rows strictly before gd.
        Rows are appended in date order, so all stored rows with date < gd
        are exactly the completed active games before gd."""
        if gd == self._coef_date:
            return self._coef
        rows = [r for r in self.rows if r[0] < gd]
        n = len(rows)
        if n < C_MIN_ACTIVE:
            self._coef_date, self._coef = gd, (0.0, 0.0)
            return self._coef
        A = np.array([(1.0, r[2], r[3], r[4], r[5]) for r in rows])
        y = np.array([r[1] for r in rows])
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
        sh = n / (n + C_SHRINK)
        self._coef_date = gd
        self._coef = (float(max(0.0, beta[2] * sh)),
                      float(min(0.0, beta[3] * sh)))
        return self._coef


# ---------------------------------------------------------------------------
# season loop: prod_by_season.py VERBATIM control + late-state variant
# ---------------------------------------------------------------------------

def season_run(season, layer, coef_hist):
    t0 = time.time()
    con = connect(read_only=True)
    tank = layer.tank
    pm = con.execute("""SELECT game_id, team_id, player_id, seconds/60.0 AS mins
        FROM player_game_stats WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    played = {(g, t): set(grp.player_id)
              for (g, t), grp in pm.groupby(["game_id", "team_id"])}
    meta = con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl,
        game_date, pts FROM nba_games WHERE season=? AND game_id LIKE '002%'
        AND wl IS NOT NULL ORDER BY game_date""", [season]).fetchdf()
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
    model = comp = None
    last = None
    nrefit = 0
    wl_state = {}                         # team_id -> [wins, games]
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
            last = gd
        # ---- late-state per-game state (PIT, before W/L update) ----------
        wh = wl_state.setdefault(h.team_id, [0, 0])
        wa = wl_state.setdefault(a.team_id, [0, 0])
        wdiff = ((wh[0] / wh[1] if wh[1] else 0.5)
                 - (wa[0] / wa[1] if wa[1] else 0.5))
        gp_h = tank.score(h.team_id, gd)[1]
        gp_a = tank.score(a.team_id, gd)[1]
        active = gp_h >= GP_ACTIVE or gp_a >= GP_ACTIVE
        tsd = tank.diff(h.team_id, a.team_id, gd)
        fdiff = (layer.form.get((int(h.team_id), gid), 0.0)
                 - layer.form.get((int(a.team_id), gid), 0.0))
        # oracle-tier OUT sets (verbatim capstone construction)
        outs = {}
        for t in (h.team_id, a.team_id):
            pl = played.get((gid, t), set())
            outs[t] = {p for p, d0 in comp.players.items()
                       if d0["team_id"] == t and (gd - d0["last_played"]).days <= 12
                       and p not in pl}
        outdiff = float(len(outs[h.team_id]) - len(outs[a.team_id]))
        c_f, c_o = layer.coefs(gd)        # daily fit, rows strictly < gd
        margin_home = float(h.pts - a.pts)
        hw = h.pts > a.pts
        # append THIS game to the fit history (used for dates > gd only) —
        # every completed active game enters, market-matched or not
        if active:
            layer.append(gd, margin_home, tsd, fdiff, outdiff, wdiff)
        wh[0] += int(hw); wh[1] += 1
        wa[0] += int(not hw); wa[1] += 1
        pmv = mkt.get((str(gd)[:10], h.team_abbrev, a.team_abbrev))
        if pmv is None:
            continue
        mm = model.margin(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id],
                          gd, b2b_home=b2b(h.team_id, gd), b2b_away=b2b(a.team_id, gd))
        add = (c_f * fdiff + c_o * outdiff) if active else 0.0
        if not coef_hist or coef_hist[-1][0] != str(gd):
            coef_hist.append((str(gd), round(c_f, 4), round(c_o, 4),
                              len(layer.rows)))
        rows.append(dict(
            season=season, game_id=gid, game_date=str(gd)[:10],
            home=h.team_abbrev, away=a.team_abbrev, y=int(h.wl == "W"),
            p_mkt=float(pmv), active=int(active), tsd=round(float(tsd), 6),
            gp_h=gp_h, gp_a=gp_a, fdiff=round(float(fdiff), 4),
            outdiff=outdiff, c_f=round(c_f, 4), c_o=round(c_o, 4),
            p_ctrl=float(sigmoid(mm / SCALE)),
            p_var=float(sigmoid((mm + add) / SCALE))))
    con.close()
    print(f"[{season}] n={len(rows)} refits={nrefit} fit_rows={len(layer.rows)} "
          f"({time.time()-t0:.0f}s)", flush=True)
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
    tank = get_tank_model(con)            # primes the process cache too
    layer = LateStateLayer(con, tank)
    con.close()
    print(f"burn-in active rows (2022-23, outdiff=0): {layer.n_burnin}",
          flush=True)

    coef_hist = []
    all_rows = []
    for s in SEASONS:
        all_rows += season_run(s, layer, coef_hist)

    with open(OUT_DIR / "ov_latestate_pergame.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(all_rows)

    y = np.array([r["y"] for r in all_rows])
    seas = np.array([r["season"] for r in all_rows])
    month = np.array([int(r["game_date"][5:7]) for r in all_rows])
    active = np.array([r["active"] for r in all_rows], bool)
    tsd = np.array([r["tsd"] for r in all_rows])
    ll_c = ll_vec(y, [r["p_ctrl"] for r in all_rows])
    ll_v = ll_vec(y, [r["p_var"] for r in all_rows])
    ll_m = ll_vec(y, [r["p_mkt"] for r in all_rows])
    d = ll_c - ll_v              # positive = variant better
    dm_c = ll_c - ll_m           # + = we lose to market (control)
    dm_v = ll_v - ll_m           # + = we lose to market (variant)

    # ---- control replication check vs shipped capstone (tank headline) ----
    base = {}
    with open(OUT_DIR / "capstone_pergame_tank.csv") as f:
        for r in csv.DictReader(f):
            base[(r["season"], r["game_id"])] = float(r["p_us"])
    diffs = [abs(base[(r["season"], r["game_id"])] - r["p_ctrl"])
             for r in all_rows if (r["season"], r["game_id"]) in base]
    repl = dict(baseline="capstone_pergame_tank.csv", n_matched=len(diffs),
                n_ours=len(all_rows),
                max_abs_diff=float(max(diffs)) if diffs else None)

    pocket = np.where(np.isin(month, [1, 2]), "JanFeb",
                      np.where(month == 3, "Mar",
                               np.where(month == 4, "Apr", "other")))

    def sub(mask):
        return dict(n=int(mask.sum()), delta=paired_ci(d[mask]),
                    ll_control=round(float(ll_c[mask].mean()), 5),
                    ll_variant=round(float(ll_v[mask].mean()), 5),
                    ll_market=round(float(ll_m[mask].mean()), 5),
                    d_vs_mkt_control=round(float(dm_c[mask].mean()), 5),
                    d_vs_mkt_variant=round(float(dm_v[mask].mean()), 5))

    apr_c = [(cf, co) for (dd, cf, co, _) in coef_hist if dd[5:7] == "04"]
    pooled = sub(np.ones(len(d), bool))
    ship = pooled["delta"][1] > 0.0       # pooled 95% CI excludes 0 (better)
    res = dict(
        config=dict(
            term="m += c_f*(form5_h-form5_a) + c_o*(nout_h-nout_a), "
                 "active iff either gp_before>=55 (tanking.py activation)",
            fit="daily walk-forward OLS margin~[1,tsd,fdiff,outdiff,wdiff] "
                "on all completed active games since 2022-23 (outdiff=0 "
                "22-23), n/(n+600) shrink, sign guards c_f>=0/c_o<=0, "
                "0 until 20 active rows; tsd/wdiff fit-time controls only; "
                "shipped tank k NOT refit",
            outs_tier="oracle (headline capstone tier)",
            gate="paired bootstrap 2000x 95% CI, ONE config; "
                 "SHIP iff pooled CI excludes 0"),
        replication=repl,
        window_check=dict(
            n_active=int(active.sum()),
            n_tsd_nonzero=int((tsd != 0).sum()),
            n_mismatch=int((active != (tsd != 0)).sum())),
        control_ll=dict(pooled=round(float(ll_c.mean()), 5),
                        market=round(float(ll_m.mean()), 5),
                        per_season={s: round(float(ll_c[seas == s].mean()), 4)
                                    for s in SEASONS}),
        variant_ll=dict(pooled=round(float(ll_v.mean()), 5),
                        per_season={s: round(float(ll_v[seas == s].mean()), 4)
                                    for s in SEASONS}),
        gate=dict(
            pooled=pooled,
            active=sub(active),
            per_season={s: sub(seas == s) for s in SEASONS},
            per_season_active={s: sub(active & (seas == s)) for s in SEASONS},
            pockets_active={p: sub(active & (pocket == p))
                            for p in ("JanFeb", "Mar", "Apr")}),
        diag=dict(
            burnin_rows=layer.n_burnin,
            total_fit_rows=len(layer.rows),
            coefs_april_mean=(
                [round(float(np.mean([c[0] for c in apr_c])), 4),
                 round(float(np.mean([c[1] for c in apr_c])), 4)]
                if apr_c else None),
            coef_history_sample=[coef_hist[i] for i in range(len(coef_hist))
                                 if i % 25 == 0 or i >= len(coef_hist) - 3]),
        verdict=("SHIP (pooled CI excludes 0)" if ship
                 else "NO-SHIP -> FREEZE F7 (pooled CI includes 0)"),
    )
    with open(OUT_DIR / "ov_latestate_results.json", "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
