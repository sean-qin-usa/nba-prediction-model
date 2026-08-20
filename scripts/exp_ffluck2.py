"""EXPERIMENT ffluck2 — refined 3PT-luck adjustment for the FourFactors ridge.

The blunt version (FF_LUCK=1: replace ALL 3PM with league-avg% x 3PA in the
ridge targets) made 2023-24/2024-25 WORSE (+0.0024/+0.0035) because it also
destroys real offensive 3P skill. Refined variants tested here:

  ctrl      unmodified FourFactors — harness-fidelity control (must replicate
            baseline p_us from data/capstone_pergame_sched.csv)
  defonly   DEFENSE-ONLY luck removal: per factor, fit TWO ridge sets —
            OFFENSE credit from RAW rows, DEFENSE credit from rows whose 3PM
            is replaced by league-avg% x 3PA. A team's defensive rating no
            longer absorbs opponents' 3P conversion luck, while its own
            offensive 3P skill stays fully priced. factor->ortg map (W) is fit
            on the hybrid predictions vs RAW ortg (the row's luck belongs to
            the offense, which we deliberately keep raw).
  team_w05  both-sides shrink of 3PM toward TEAM-TRAILING 3P% (30-game
  team_w10  strictly-before-date window, PIT within the train set; EB
            stabilizer: 100 pseudo-3PA of league avg), shrink weight w=0.5 /
            w=1.0. Team-specific expectation preserves real shooting skill
            while removing game-level luck. Applied to efg ridge target AND
            the ortg target of the W map (mirrors the blunt construction).

Walk-forward capstone loop copied from scripts/prod_by_season.py (default
path: oracle OUT-sets, weekly refit, no oracle minutes). Production nbapred
modules imported READ-ONLY; nothing in nbapred/ is modified. DB opened
read_only=True only. Gate: paired bootstrap (2000 resamples) 95% CI on
per-game log-loss delta vs data/capstone_pergame_sched.csv.

Run:  python scripts/exp_ffluck2.py
"""
import sys, warnings, json, csv
import datetime as _dt
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from nbapred.db import connect
from nbapred.eval.metrics import log_loss
from nbapred.model.four_factors import factor_game_rows, FACTORS
from nbapred.model.team_ratings import TeamRatings, game_rows
from nbapred.model.production import (SCALE, sigmoid, fit_schedule_layer,
                                      last_season_prior)
from nbapred.model.composition import CompositionModel

VARIANTS = ["ctrl", "defonly", "team_w05", "team_w10"]
TRAIL_WINDOW = 30          # team-trailing 3P% window (games)
TRAIL_PSEUDO_ATT = 100.0   # EB stabilizer: pseudo-3PA of league average
RIDGE = 25.0
SEASONS = ("2023-24", "2024-25", "2025-26")
BASELINE_CSV = ROOT / "data" / "capstone_pergame_sched.csv"
OUT_CSV = ROOT / "data" / "capstone_pergame_ffluck2.csv"
OUT_JSON = ROOT / "data" / "ffluck2_results.json"


class HybridFF:
    """FourFactors clone whose per-factor ridge can differ between the
    offense-credit side (mu/home/off) and the defense-credit side (deff)."""

    def __init__(self, fms_off: dict, fms_def: dict, W: np.ndarray):
        self.fms_off, self.fms_def, self.W = fms_off, fms_def, W

    def pred(self, f: str, tid: int, oid: int, is_home: bool) -> float:
        o, d = self.fms_off[f], self.fms_def[f]
        return (o.mu + o.off.get(tid, 0.0) - d.deff.get(oid, 0.0)
                + (o.home if is_home else 0.0))

    def eortg(self, tid: int, oid: int, is_home: bool) -> float:
        xf = np.array([self.pred(f, tid, oid, is_home) for f in FACTORS])
        return float(xf @ self.W[:4] + self.W[4])

    def margin_neutral(self, home_id: int, away_id: int) -> float:
        return (self.eortg(home_id, away_id, False)
                - self.eortg(away_id, home_id, False))


def fit_ff_variants(con, season: str, before):
    """Fit all four FF variants from ONE factor_game_rows extraction.
    Returns dict variant -> HybridFF, or None when <200 rows (early season —
    production falls back to ratings+prior, identical across variants)."""
    rows = factor_game_rows(con, season, before)
    if len(rows) < 200:
        return None
    lg3p = sum(x["thrm"] for x in rows) / max(sum(x["thra"] for x in rows), 1)

    # team-trailing 3P%: strictly-before-date games (PIT inside the train
    # window — every row already predates `before`), last TRAIL_WINDOW games,
    # EB-shrunk toward the train-set league average.
    hist: dict[int, list] = {}
    for x in sorted(rows, key=lambda r: r["date"]):
        hist.setdefault(x["tid"], []).append((x["date"], x["thrm"], x["thra"]))

    def trail3p(tid, date):
        h = [(m, a) for (d, m, a) in hist.get(tid, []) if d < date][-TRAIL_WINDOW:]
        m = sum(t[0] for t in h)
        a = sum(t[1] for t in h)
        return (m + TRAIL_PSEUDO_ATT * lg3p) / (a + TRAIL_PSEUDO_ATT)

    def adj(x, new3pm):
        """(efg, ortg) with the row's 3PM replaced by new3pm."""
        d3 = new3pm - x["thrm"]
        return (x["efg"] + 0.5 * d3 / x["fga"],
                x["ortg"] + 100 * 3 * d3 / x["poss"])

    def ridge_fit(vals):
        return TeamRatings(ridge=RIDGE, team_home_ridge=None).fit(
            [(x["tid"], x["oid"], x["home"], v * 100) for x, v in zip(rows, vals)])

    # base (raw) ridges — the exact production fit
    base = {f: ridge_fit([x[f] for x in rows]) for f in FACTORS}
    # league-avg-adjusted efg ridge (only its DEFENSE side is consumed)
    efg_lg = ridge_fit([adj(x, lg3p * x["thra"])[0] for x in rows])
    # team-trailing-shrunk efg ridges + matching adjusted ortg targets
    team_fits, team_y = {}, {}
    for lab, w in (("team_w05", 0.5), ("team_w10", 1.0)):
        efgs, ortgs = [], []
        for x in rows:
            new3pm = (1 - w) * x["thrm"] + w * trail3p(x["tid"], x["date"]) * x["thra"]
            ae, ao = adj(x, new3pm)
            efgs.append(ae)
            ortgs.append(ao)
        team_fits[lab] = ridge_fit(efgs)
        team_y[lab] = np.array(ortgs)

    y_raw = np.array([x["ortg"] for x in rows])

    def build(fms_off, fms_def, y):
        ff = HybridFF(fms_off, fms_def, None)
        X = np.array([[ff.pred(f, x["tid"], x["oid"], x["home"])
                       for f in FACTORS] for x in rows])
        A = np.c_[X, np.ones(len(X))]
        ff.W = np.linalg.lstsq(A, y, rcond=None)[0]
        return ff

    out = {"ctrl": build(base, base, y_raw),
           "defonly": build(base, {**base, "efg": efg_lg}, y_raw)}
    for lab in ("team_w05", "team_w10"):
        fms = {**base, "efg": team_fits[lab]}
        out[lab] = build(fms, fms, team_y[lab])
    return out


def season_run(season: str):
    """Walk-forward capstone for one season — faithful copy of
    scripts/prod_by_season.py's default path, with the FF component swapped
    per variant. Shared components (composition, schedule layer, ratings
    fallback) computed once per refit and reused across variants."""
    con = connect(read_only=True)
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
        return (d - _dt.timedelta(days=1)) in tdates.get(tid, set())

    # season-level pieces of production's ratings fallback (verbatim copies)
    prior = last_season_prior(con, season)
    ab = dict(con.execute(
        "SELECT DISTINCT team_id, team_abbrev FROM nba_games WHERE season=?",
        [season]).fetchall())
    id2ab = {t: a for t, a in ab.items()}
    games_played = dict(con.execute("""
        SELECT team_id, count(*) FROM nba_games WHERE season=? AND game_id LIKE '002%'
        AND wl IS NOT NULL GROUP BY 1""", [season]).fetchall())

    y, rows_out = [], []
    P = {v: [] for v in VARIANTS}
    comp = tr = ffs = sched5 = None
    last = None
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
            comp = CompositionModel(con, before=gd)
            sched5 = fit_schedule_layer(con, gd)
            tr = TeamRatings(ridge=RIDGE).fit(game_rows(con, before=gd, season=season))
            ffs = fit_ff_variants(con, season, gd)
            last = gd
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
        he, b_hb2b, b_ab2b, _bhd, _bad = sched5
        sched = (he + (b_hb2b if b2b(h.team_id, gd) else 0.0)
                 + (b_ab2b if b2b(a.team_id, gd) else 0.0))
        cm = comp.margin(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id],
                         gd, home_edge=0.0)
        if ffs is not None:
            for v in VARIANTS:
                fm = ffs[v].margin_neutral(h.team_id, a.team_id)
                P[v].append(float(sigmoid((0.5 * fm + 0.5 * cm + sched) / SCALE)))
        else:
            # production fallback (verbatim): ratings + cold-start prior,
            # global home coeff stripped, w_comp=0.7 — identical across variants
            gh = games_played.get(h.team_id, 0)
            ga = games_played.get(a.team_id, 0)
            wh = max(0.0, 1 - gh / 20.0)
            wa = max(0.0, 1 - ga / 20.0)
            rm = (tr.pred_margin(h.team_id, a.team_id)
                  + wh * prior.get(id2ab.get(h.team_id, ""), 0.0)
                  - wa * prior.get(id2ab.get(a.team_id, ""), 0.0)) - tr.home
            p = float(sigmoid((0.7 * cm + 0.3 * rm + sched) / SCALE))
            for v in VARIANTS:
                P[v].append(p)
        rows_out.append([season, gid, str(gd)[:10], h.team_abbrev, a.team_abbrev,
                         y[-1]] + [P[v][-1] for v in VARIANTS]
                        + [float(pmv), len(outs[h.team_id]), len(outs[a.team_id])])
    con.close()
    yv = np.array(y)
    res = {"season": season, "n": len(yv),
           **{v: round(log_loss(yv, P[v]), 4) for v in VARIANTS}}
    print(res, flush=True)
    return res, rows_out


def paired_boot(delta: np.ndarray, nboot: int = 2000, seed: int = 7):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), (nboot, len(delta)))
    means = delta[idx].mean(axis=1)
    return (float(delta.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def main():
    all_rows = []
    per_season = []
    for s in SEASONS:
        res, rows = season_run(s)
        per_season.append(res)
        all_rows.extend(rows)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["season", "game_id", "game_date", "home", "away", "y"]
                   + [f"p_{v}" for v in VARIANTS]
                   + ["p_mkt", "n_out_home", "n_out_away"])
        w.writerows(all_rows)

    # ---- paired gate vs the shipped-production baseline CSV ----
    import pandas as pd
    base = pd.read_csv(BASELINE_CSV, dtype={"game_id": str})
    ours = pd.read_csv(OUT_CSV, dtype={"game_id": str})
    j = base.merge(ours, on=["season", "game_id"], suffixes=("_b", ""))
    print(f"joined {len(j)}/{len(base)} baseline games "
          f"(ours {len(ours)})", flush=True)
    eps = 1e-15

    def ll_vec(y, p):
        p = np.clip(np.asarray(p, float), eps, 1 - eps)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    yb = j["y_b"].values.astype(float)
    ll_base = ll_vec(yb, j["p_us"].values)
    fidelity = float(np.abs(j["p_ctrl"].values - j["p_us"].values).max())
    print(f"ctrl-vs-baseline fidelity: max|dp|={fidelity:.2e}", flush=True)

    gates = {}
    for v in VARIANTS:
        d = ll_vec(yb, j[f"p_{v}"].values) - ll_base   # <0 = variant better
        pooled = paired_boot(d)
        seasons = {}
        for s in SEASONS:
            msk = (j["season"] == s).values
            seasons[s] = paired_boot(d[msk])
        verdict = ("PASS" if pooled[2] < 0 else
                   "FAIL" if pooled[1] > 0 else "NS")
        gates[v] = {"pooled": pooled, "per_season": seasons, "verdict": verdict}
        print(f"{v:9s} pooled dLL {pooled[0]:+.5f} CI({pooled[1]:+.5f},"
              f"{pooled[2]:+.5f}) {verdict}  " + "  ".join(
                  f"{s}: {seasons[s][0]:+.5f}" for s in SEASONS), flush=True)

    json.dump({"per_season": per_season, "gates": gates,
               "ctrl_fidelity_max_dp": fidelity},
              open(OUT_JSON, "w"), indent=1)
    print("wrote", OUT_CSV, OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
