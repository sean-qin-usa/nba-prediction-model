"""STAT crossfitW — is the FourFactors factors->ortg map (W) contaminated by
generated-regressor overfit?

DIAGNOSIS. `nbapred/model/four_factors.py::FourFactors.fit` builds, for each of
the 4 factors, an opponent-adjusted ridge on ALL trailing rows, then forms the
design matrix X from those ridges' FITTED values on the SAME rows and regresses
realized `ortg` on X. Row i's X_i therefore contains row i's own noise (the
ridge saw y_i's factor realization), so cov(X_i, eps_i) != 0 and the OLS map W
is biased AWAY from zero — it flatters in-sample fit and can misweight factors
relative to each other (a factor whose ridge overfits harder gets more credit).

TREATMENT (standard cross-fitting / DML-style orthogonalisation):
  * K=5 folds over the trailing rows (seeded permutation, deterministic).
  * For each fold k: refit all 4 per-factor ridges on the other 4/5 of rows and
    predict the held-out fold -> out-of-fold factor predictions for ALL rows.
  * Fit W by OLS on those OOF predictions vs realized ortg.
  * The per-factor ridges USED AT PREDICTION TIME are refit on all rows
    (standard cross-fitting: nuisance functions final-fit on full data, the
    second-stage coefficient learned on orthogonalised signals).
Everything else (row extraction, factor definitions, ridge value, the
efficiency->margin composition, schedule layer, composition model) is the
unmodified production code.

HARNESS. Full 3-season walk-forward capstone, loop copied from
scripts/prod_by_season.py default path (oracle OUT-sets, weekly refit, no
oracle minutes, no dead-team flags), with a SAME-RUN CONTROL that instantiates
the stock, unmodified FourFactors on the identical refit dates and shares every
other component with the variant. Gate = paired bootstrap (2000 resamples, 95%
CI) on the per-game log-loss delta variant - control. Shipped capstone CSV is
read only as an informational fidelity check, never as the gate baseline
(baselines have drifted vs the DB).

RULES OBSERVED: DuckDB opened read_only=True; nothing under nbapred/ or any
existing script is modified; PIT strict (every fit uses `before=game_date`).

Run:  python scripts/stat_crossfitW.py
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from nbapred.db import connect
from nbapred.eval.metrics import log_loss
from nbapred.model.composition import CompositionModel
from nbapred.model.four_factors import FACTORS, FourFactors, factor_game_rows
from nbapred.model.production import (SCALE, fit_schedule_layer,
                                      last_season_prior, sigmoid)
from nbapred.model.team_ratings import TeamRatings, game_rows

SEASONS = ("2023-24", "2024-25", "2025-26")
# ctrl    = stock FourFactors (same-run control; the gate baseline)
# xfit    = cross-fitted W (the variant under test)
# xfit_sc = MECHANISM DECOMPOSITION arm, not a ship candidate: the cross-fitted
#           factor MIX (W direction) affinely rescaled so its fitted eortg has
#           the control's mean and sd. Isolates "does cross-fitting misweight
#           the factors?" from "does cross-fitting compress the FF margin?".
ARMS = ("ctrl", "xfit", "xfit_sc")
RIDGE = 25.0
W_COMP = 0.7                      # ratings-fallback blend (production default)
NBOOT = 2000
BOOT_SEED = 7
BASELINE_CSV = ROOT / "data" / "capstone_pergame.csv"      # informational only
OUT_CSV = ROOT / "data" / "stat_crossfitW_pergame.csv"
OUT_JSON = ROOT / "data" / "stat_crossfitW_results.json"


# --------------------------------------------------------------------------
# variant: cross-fitted factors -> ortg map
# --------------------------------------------------------------------------
class CrossFitFF(FourFactors):
    """FourFactors with the second-stage map W learned on OUT-OF-FOLD factor
    predictions. Per-factor ridges (self.fms), the eortg/margin/margin_neutral
    algebra and `ready` are inherited unchanged — only how W is estimated
    differs, so any delta is attributable to the generated-regressor bias."""

    K = 5
    SEED = 20260730

    def __init__(self, ridge: float = 25.0):
        super().__init__(ridge=ridge)
        self.diag = {}
        self.W_scaled = None

    def _ridges(self, rows, weights):
        return {f: TeamRatings(ridge=self.ridge, team_home_ridge=None).fit(
            [(x["tid"], x["oid"], x["home"], x[f] * 100) for x in rows],
            weights=weights) for f in FACTORS}

    @staticmethod
    def _design(fms, rows):
        return np.array([[fms[f].pred_ortg(x["tid"], x["oid"], x["home"])
                          for f in FACTORS] for x in rows])

    @staticmethod
    def _wls(A, y, w):
        if w is None:
            return np.linalg.lstsq(A, y, rcond=None)[0]
        sw = np.sqrt(w)
        return np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]

    def fit(self, con, season: str, before=None, half_life_days: float | None = None,
            luck_adjust_3p: bool = False):
        # ---- identical row extraction / guards / optional transforms as stock
        rows = factor_game_rows(con, season, before)
        if len(rows) < 200:
            return self
        if luck_adjust_3p:                       # parity with stock (unused here)
            lg3p = sum(x["thrm"] for x in rows) / max(sum(x["thra"] for x in rows), 1)
            for x in rows:
                exp3 = lg3p * x["thra"]
                x["efg"] = x["efg"] + 0.5 * (exp3 - x["thrm"]) / x["fga"]
                x["ortg"] = x["ortg"] + 100 * 3 * (exp3 - x["thrm"]) / x["poss"]
        w = None
        if half_life_days and before:
            age = np.array([(before - x["date"]).days for x in rows], float)
            w = 0.5 ** (age / half_life_days)

        n = len(rows)
        y = np.array([x["ortg"] for x in rows])

        # ---- nuisance stage, FINAL fit on all rows (used at prediction time)
        self.fms = self._ridges(rows, w)
        X_full = self._design(self.fms, rows)

        # ---- nuisance stage, CROSS-FIT: OOF predictions for every row
        rng = np.random.default_rng(self.SEED)
        perm = rng.permutation(n)
        fold = np.empty(n, dtype=int)
        fold[perm] = np.arange(n) % self.K
        X_oof = np.empty_like(X_full)
        for k in range(self.K):
            te = np.flatnonzero(fold == k)
            tr = np.flatnonzero(fold != k)
            fms_k = self._ridges([rows[i] for i in tr],
                                 None if w is None else w[tr])
            X_oof[te] = self._design(fms_k, [rows[i] for i in te])

        # ---- second stage: W on the orthogonalised (OOF) design
        self.W = self._wls(np.c_[X_oof, np.ones(n)], y, w)
        W_naive = self._wls(np.c_[X_full, np.ones(n)], y, w)

        # ---- decomposition arm: cross-fitted DIRECTION, control SCALE.
        # affine map of the xfit fitted values onto the control's fitted-value
        # mean/sd (both evaluated on the full-data design actually used at
        # prediction time), so the only surviving difference is the factor mix.
        zc = X_full @ W_naive[:4] + W_naive[4]
        zx = X_full @ self.W[:4]
        c = float(zc.std() / zx.std()) if zx.std() > 0 else 1.0
        self.W_scaled = np.r_[c * self.W[:4], float(zc.mean() - c * zx.mean())]

        # diagnostics: shrinkage of the OOF design vs the in-sample design and
        # the resulting coefficient movement
        self.diag = {
            "n_rows": n,
            "W_xfit": [float(v) for v in self.W],
            "W_naive": [float(v) for v in W_naive],
            "sd_X_full": [float(s) for s in X_full.std(0)],
            "sd_X_oof": [float(s) for s in X_oof.std(0)],
            "r2_naive": float(1 - np.var(y - np.c_[X_full, np.ones(n)] @ W_naive)
                              / np.var(y)),
            "r2_xfit_oof": float(1 - np.var(y - np.c_[X_oof, np.ones(n)] @ self.W)
                                 / np.var(y)),
            "scale_c": c,
            "sd_eortg_ctrl": float(zc.std()),
            "sd_eortg_xfit": float((zx + self.W[4]).std()),
        }
        return self

    def as_scaled(self):
        """Sibling predictor sharing this fit's per-factor ridges but using the
        scale-matched cross-fitted map (decomposition arm)."""
        sib = FourFactors(ridge=self.ridge)
        sib.fms = self.fms
        sib.W = self.W_scaled if self.W is not None else None
        return sib


# --------------------------------------------------------------------------
# walk-forward capstone (copied from scripts/prod_by_season.py, default path)
# --------------------------------------------------------------------------
def season_run(season: str):
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
        return (d - _dt.timedelta(days=1)) in tdates.get(tid, set())

    # season-level pieces of production's ratings fallback (verbatim)
    prior = last_season_prior(con, season)
    id2ab = dict(con.execute(
        "SELECT DISTINCT team_id, team_abbrev FROM nba_games WHERE season=?",
        [season]).fetchall())

    y, rows_out, diags = [], [], []
    P = {a: [] for a in ARMS}
    comp = tr = sched5 = ff = None
    ffs = {}
    games_played = {}
    last = None
    t0 = time.time()
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
            # --- shared components (identical for both arms) ---
            comp = CompositionModel(con, before=gd)
            sched5 = fit_schedule_layer(con, gd)
            tr = TeamRatings(ridge=RIDGE).fit(game_rows(con, before=gd, season=season))
            games_played = dict(con.execute("""
                SELECT team_id, count(*) FROM nba_games WHERE season=?
                AND game_id LIKE '002%' AND wl IS NOT NULL AND game_date < ?
                GROUP BY 1""", [season, gd]).fetchall())
            # --- the only thing that differs: the FF component ---
            xf = CrossFitFF().fit(con, season, before=gd)
            ffs = {"ctrl": FourFactors().fit(con, season, before=gd),   # SAME-RUN CONTROL
                   "xfit": xf, "xfit_sc": xf.as_scaled()}
            if ffs["xfit"].diag:
                diags.append({"season": season, "date": str(gd), **ffs["xfit"].diag})
            last = gd
            print(f"  [{season}] refit {gd} ({time.time()-t0:5.1f}s) "
                  f"ready={ffs['ctrl'].ready}", flush=True)
        pmv = mkt.get((str(gd)[:10], h.team_abbrev, a.team_abbrev))
        if pmv is None:
            continue
        outs = {}
        for t in (h.team_id, a.team_id):
            pl = played.get((gid, t), set())
            outs[t] = {p for p, d0 in comp.players.items()
                       if d0["team_id"] == t and (gd - d0["last_played"]).days <= 12
                       and p not in pl}
        y.append(int(h.wl == "W"))
        he, b_hb2b, b_ab2b, _hd, _ad = sched5
        sched = (he + (b_hb2b if b2b(h.team_id, gd) else 0.0)
                 + (b_ab2b if b2b(a.team_id, gd) else 0.0))
        cm = comp.margin(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id],
                         gd, home_edge=0.0)
        for arm in ARMS:
            ff = ffs[arm]
            if ff.ready:
                fm = ff.margin_neutral(h.team_id, a.team_id)
                mg = 0.5 * fm + 0.5 * cm + sched
            else:
                # production fallback (verbatim): ratings + cold-start prior,
                # global home coeff stripped, w_comp=0.7 — arm-independent
                gh = games_played.get(h.team_id, 0)
                ga = games_played.get(a.team_id, 0)
                rm = (tr.pred_margin(h.team_id, a.team_id)
                      + max(0.0, 1 - gh / 20.0) * prior.get(id2ab.get(h.team_id, ""), 0.0)
                      - max(0.0, 1 - ga / 20.0) * prior.get(id2ab.get(a.team_id, ""), 0.0)
                      ) - tr.home
                mg = W_COMP * cm + (1 - W_COMP) * rm + sched
            P[arm].append(float(sigmoid(mg / SCALE)))
        rows_out.append([season, gid, str(gd)[:10], h.team_abbrev, a.team_abbrev,
                         y[-1]] + [P[arm][-1] for arm in ARMS]
                        + [float(pmv), len(outs[h.team_id]), len(outs[a.team_id])])
    con.close()
    yv = np.array(y)
    res = {"season": season, "n": len(yv),
           **{a: round(log_loss(yv, P[a]), 4) for a in ARMS},
           "mkt": round(log_loss(yv, [r[-3] for r in rows_out]), 4)}
    print(res, flush=True)
    return res, rows_out, diags


# --------------------------------------------------------------------------
def paired_boot(delta: np.ndarray, nboot: int = NBOOT, seed: int = BOOT_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), (nboot, len(delta)))
    means = delta[idx].mean(axis=1)
    return (float(delta.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def ll_vec(yv, p, eps=1e-15):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    yv = np.asarray(yv, float)
    return -(yv * np.log(p) + (1 - yv) * np.log(1 - p))


def main():
    all_rows, per_season, all_diags = [], [], []
    for s in SEASONS:
        res, rws, dg = season_run(s)
        per_season.append(res)
        all_rows.extend(rws)
        all_diags.extend(dg)

    with open(OUT_CSV, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["season", "game_id", "game_date", "home", "away", "y"]
                     + [f"p_{a}" for a in ARMS]
                     + ["p_mkt", "n_out_home", "n_out_away"])
        wtr.writerows(all_rows)

    seas = np.array([r[0] for r in all_rows])
    yv = np.array([r[5] for r in all_rows], float)
    P = {a: np.array([r[6 + i] for r in all_rows], float)
         for i, a in enumerate(ARMS)}
    ll = {a: ll_vec(yv, P[a]) for a in ARMS}

    gates = {}
    for a in ARMS:
        if a == "ctrl":
            continue
        d = ll[a] - ll["ctrl"]                          # <0 = variant better
        pl = paired_boot(d)
        gates[a] = {"pooled": pl,
                    "per_season": {s: paired_boot(d[seas == s]) for s in SEASONS},
                    "verdict": "PASS" if pl[2] < 0 else "FAIL" if pl[1] > 0 else "NS"}
    pooled = gates["xfit"]["pooled"]                     # the primary gate
    per_s = gates["xfit"]["per_season"]
    verdict = gates["xfit"]["verdict"]

    # informational fidelity of the same-run control vs the shipped capstone CSV
    fid = None
    try:
        import pandas as pd
        base = pd.read_csv(BASELINE_CSV, dtype={"game_id": str})
        ours = pd.read_csv(OUT_CSV, dtype={"game_id": str})
        j = base.merge(ours, on=["season", "game_id"], suffixes=("_b", ""))
        fid = {"joined": int(len(j)),
               "max_abs_dp": float(np.abs(j.p_ctrl - j.p_us).max()),
               "mean_abs_dp": float(np.abs(j.p_ctrl - j.p_us).mean()),
               "ll_ctrl": float(log_loss(j.y_b, j.p_ctrl)),
               "ll_shipped_csv": float(log_loss(j.y_b, j.p_us))}
    except Exception as e:                                    # pragma: no cover
        fid = {"error": repr(e)}

    # mean |dp| vs control — how much each map actually moved predictions
    move = {a: {"mean_abs_dp": float(np.abs(P[a] - P["ctrl"]).mean()),
                "max_abs_dp": float(np.abs(P[a] - P["ctrl"]).max())}
            for a in ARMS if a != "ctrl"}

    out = {"per_season": per_season, "gates": gates, "pooled": pooled,
           "per_season_boot": per_s, "verdict": verdict,
           "control_fidelity_vs_shipped_csv": fid,
           "arm_movement": move, "K": CrossFitFF.K, "nboot": NBOOT,
           "diagnostics": all_diags}
    json.dump(out, open(OUT_JSON, "w"), indent=1)

    print("\n=== GATE: arm - ctrl (same-run), per-game log loss ===", flush=True)
    for a, g in gates.items():
        pl, ps = g["pooled"], g["per_season"]
        print(f"{a:8s} pooled d={pl[0]:+.5f} CI({pl[1]:+.5f},{pl[2]:+.5f}) {g['verdict']}")
        for s in SEASONS:
            print(f"         {s}  d={ps[s][0]:+.5f}  CI({ps[s][1]:+.5f},{ps[s][2]:+.5f})")
    print("arm movement:", move)
    print("control fidelity vs shipped CSV:", fid)
    print("wrote", OUT_CSV, OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
