"""AM MEASUREMENT — how non-stationary is team strength, by season phase?

MEASURE FIRST, MODEL SECOND.  This script contains NO endpoint scoring (no
win-probability log loss vs the certified control).  It measures the physical
quantity the owner's hypothesis is about: does the optimal MEMORY LENGTH for
team-strength estimation shorten as the season progresses?

Three measurements, all PIT (strictly-prior games only), all on the scorable
5-season corpus (E3-E6, 2021-22..2025-26):

  M-A  OPTIMAL EXPONENTIAL HALF-LIFE BY PHASE.  For a grid of half-lives h
       (in games), x_h = decay-weighted mean signed margin of home minus away,
       using only that team's strictly-prior CURRENT-SEASON games.  For each
       games-played bucket, fit y (actual home margin) ~ 1 + c*x_h with
       LEAVE-ONE-SEASON-OUT cross-fitting and record held-out RMSE.  argmin_h
       = the optimal memory at that phase.  Falling h with gp supports the
       "late season is less stationary" theory; flat h refutes it.

  M-B  RECENT-vs-OLD WEIGHT SHARE BY PHASE.  y ~ 1 + a*x_recent5 + b*x_older,
       where x_recent5 uses the last 5 games and x_older the rest of the
       season to date.  Report the PER-GAME weight ratio
       (a/5) / (b/(gp-5)) by phase.  Ratio 1.0 = a uniform season-to-date
       average is optimal (stationary).  Ratio > 1 and RISING with gp = the
       owner's hypothesis.

  M-C  BLOCK AUTOCORRELATION DECAY BY PHASE.  Opponent-adjusted margins
       (season SRS removed from the opponent, home edge removed) averaged in
       non-overlapping 5-game blocks; corr(b_k, b_{k+m}) as a function of the
       lag m and of the phase of block k.  A stationary team gives a flat
       profile in m; drift makes it decay.  The DECAY RATE by phase is the
       direct physical measurement of non-stationarity.

Read-only DB.  Writes data/am_measure.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from nbapred.db import connect  # noqa: E402

SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
# theory-set phase edges on min(gp_home, gp_away): opener ramp / early / mid /
# late-pre-deadline / post-deadline+tank window (55 = the D71/D73 gate)
BUCKETS = [(10, 20), (20, 30), (30, 41), (41, 55), (55, 82)]
HALFLIVES = [2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 1e9]
BLOCK = 5


def load_games(con):
    rows = con.execute("""
        SELECT season, game_id, game_date, team_id, is_home, pts
        FROM nba_games WHERE game_id LIKE '002%' AND pts IS NOT NULL
          AND season IN ('2021-22','2022-23','2023-24','2024-25','2025-26')
        ORDER BY game_date, game_id""").fetchall()
    byg = {}
    for season, gid, d, tid, ish, pts in rows:
        d = d.date() if hasattr(d, "date") else d
        g = byg.setdefault(gid, dict(season=season, date=d, gid=gid))
        g["h" if ish else "a"] = (int(tid), float(pts))
    games = sorted((g for g in byg.values() if "h" in g and "a" in g),
                   key=lambda g: (g["date"], g["gid"]))
    return games


def build_panel(games):
    """Chronological pass: for every game, snapshot each team's strictly-prior
    current-season margin history, then append.  Fully PIT."""
    hist = {}                                   # (season, tid) -> [margins]
    recs = []
    for g in games:
        s = g["season"]
        ht, hp = g["h"]
        at, ap = g["a"]
        mh = hist.setdefault((s, ht), [])
        ma = hist.setdefault((s, at), [])
        recs.append(dict(season=s, gid=g["gid"], date=str(g["date"]),
                         home=ht, away=at, y=hp - ap,
                         mh=list(mh), ma=list(ma)))
        mh.append(hp - ap)
        ma.append(ap - hp)
    return recs


def wmean(ms, h):
    """Decay-weighted mean of a chronological margin list; weight 0.5**(age/h)
    with age 0 = most recent.  h >= 1e9 is the uniform (infinite-memory) case."""
    n = len(ms)
    if n == 0:
        return 0.0
    if h >= 1e8:
        return float(np.mean(ms))
    age = np.arange(n - 1, -1, -1, dtype=float)
    w = 0.5 ** (age / h)
    return float(np.dot(w, ms) / w.sum())


def loso_linfit(X, y, seasons):
    """Leave-one-season-out cross-fit of y ~ X (X includes intercept col).
    Returns (held-out RMSE, held-out R2 vs the held-out mean, pooled beta)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    seasons = np.asarray(seasons)
    pred = np.zeros_like(y)
    for s in np.unique(seasons):
        te = seasons == s
        tr = ~te
        if tr.sum() < X.shape[1] + 5 or te.sum() == 0:
            pred[te] = y[tr].mean() if tr.sum() else 0.0
            continue
        b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
        pred[te] = X[te] @ b
    resid = y - pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    sst = float(np.mean((y - y.mean()) ** 2))
    r2 = float(1.0 - np.mean(resid ** 2) / sst) if sst else float("nan")
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return rmse, r2, beta.tolist()


def main():
    con = connect(read_only=True)
    games = load_games(con)
    recs = build_panel(games)
    out = {"corpus": {"seasons": list(SEASONS), "n_games": len(recs),
                      "eras": "E3+E4+E5+E6", "buckets": BUCKETS,
                      "halflives": HALFLIVES}}

    # ---------------- M-A : optimal half-life by phase ---------------------
    ma = {}
    for lo, hi in BUCKETS:
        sel = [r for r in recs
               if lo <= min(len(r["mh"]), len(r["ma"])) < hi]
        if len(sel) < 100:
            continue
        y = np.array([r["y"] for r in sel], float)
        ss = np.array([r["season"] for r in sel])
        row = {"n": len(sel), "mean_gp": float(np.mean(
            [min(len(r["mh"]), len(r["ma"])) for r in sel])), "by_h": {}}
        for h in HALFLIVES:
            x = np.array([wmean(r["mh"], h) - wmean(r["ma"], h) for r in sel])
            X = np.column_stack([np.ones(len(x)), x])
            rmse, r2, beta = loso_linfit(X, y, ss)
            row["by_h"][f"h={h:g}"] = {"rmse": round(rmse, 5),
                                    "r2_heldout": round(r2, 6),
                                    "coef": round(beta[1], 5),
                                    "intercept": round(beta[0], 4)}
        best = min(row["by_h"].items(), key=lambda kv: kv[1]["rmse"])
        row["best_h"] = best[0]
        row["best_rmse"] = best[1]["rmse"]
        row["rmse_inf"] = row["by_h"]["h=1e+09"]["rmse"]
        row["gain_vs_inf"] = round(row["rmse_inf"] - best[1]["rmse"], 5)
        ma[f"gp[{lo},{hi})"] = row
    out["M_A_optimal_halflife_by_phase"] = ma

    # ---------------- M-B : recent-vs-old per-game weight ratio ------------
    mb = {}
    for lo, hi in BUCKETS:
        sel = [r for r in recs
               if lo <= min(len(r["mh"]), len(r["ma"])) < hi
               and len(r["mh"]) > 5 and len(r["ma"]) > 5]
        if len(sel) < 100:
            continue
        y = np.array([r["y"] for r in sel], float)
        ss = np.array([r["season"] for r in sel])
        xr = np.array([np.mean(r["mh"][-5:]) - np.mean(r["ma"][-5:])
                       for r in sel])
        xo = np.array([np.mean(r["mh"][:-5]) - np.mean(r["ma"][:-5])
                       for r in sel])
        X = np.column_stack([np.ones(len(y)), xr, xo])
        rmse, r2, beta = loso_linfit(X, y, ss)
        # bootstrap the per-game weight ratio (season-clustered)
        rng = np.random.default_rng(20260801)
        seas = np.unique(ss)
        groups = [np.flatnonzero(ss == s) for s in seas]
        ratios = []
        gp_bar = float(np.mean([min(len(r["mh"]), len(r["ma"])) for r in sel]))
        for _ in range(1000):
            pick = rng.integers(0, len(groups), size=len(groups))
            idx = np.concatenate([groups[i] for i in pick])
            b = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
            if b[2] > 1e-9:
                ratios.append((b[1] / 5.0) / (b[2] / max(gp_bar - 5.0, 1.0)))
        ratio = ((beta[1] / 5.0) / (beta[2] / max(gp_bar - 5.0, 1.0))
                 if beta[2] > 1e-9 else float("nan"))
        mb[f"gp[{lo},{hi})"] = {
            "n": len(sel), "mean_gp": round(gp_bar, 2),
            "a_recent5": round(beta[1], 5), "b_older": round(beta[2], 5),
            "per_game_weight_ratio_recent_over_old": round(ratio, 3),
            "ratio_ci": [round(float(np.percentile(ratios, 2.5)), 3),
                         round(float(np.percentile(ratios, 97.5)), 3)]
            if len(ratios) > 50 else None,
            "heldout_rmse": round(rmse, 5), "heldout_r2": round(r2, 6)}
    out["M_B_recent_vs_old_weight_by_phase"] = mb

    # ---------------- M-C : block autocorrelation decay by phase -----------
    # season SRS: margin_g = r_home - r_away + HE  (least squares, per season)
    mc = {}
    resid_by_team = {}
    for s in SEASONS:
        gs = [g for g in games if g["season"] == s]
        teams = sorted({g["h"][0] for g in gs} | {g["a"][0] for g in gs})
        ti = {t: i for i, t in enumerate(teams)}
        A = np.zeros((len(gs), len(teams) + 1))
        b = np.zeros(len(gs))
        for i, g in enumerate(gs):
            A[i, ti[g["h"][0]]] = 1.0
            A[i, ti[g["a"][0]]] = -1.0
            A[i, -1] = 1.0                      # home edge
            b[i] = g["h"][1] - g["a"][1]
        A = np.column_stack([A, np.zeros(len(gs))])   # pad for ridge-free lstsq
        A = A[:, :-1]
        beta = np.linalg.lstsq(A, b, rcond=None)[0]
        r = beta[:len(teams)]
        r = r - r.mean()
        he = beta[-1]
        for g in gs:
            m = g["h"][1] - g["a"][1]
            eh = m - (r[ti[g["h"][0]]] - r[ti[g["a"][0]]] + he)
            resid_by_team.setdefault((s, g["h"][0]), []).append(eh)
            resid_by_team.setdefault((s, g["a"][0]), []).append(-eh)
    # blocks
    blocks = {}
    for key, seq in resid_by_team.items():
        bs = [float(np.mean(seq[i:i + BLOCK]))
              for i in range(0, len(seq) - BLOCK + 1, BLOCK)]
        blocks[key] = bs
    phase_edges = [(0, 4), (4, 8), (8, 12), (12, 17)]   # block index -> phase
    for pe in phase_edges:
        prof = {}
        for m in (1, 2, 3, 4, 6, 8):
            xs, ys = [], []
            for key, bs in blocks.items():
                for k in range(pe[0], min(pe[1], len(bs))):
                    if k + m < len(bs):
                        xs.append(bs[k])
                        ys.append(bs[k + m])
            if len(xs) > 200:
                prof[f"lag{m}"] = {"n": len(xs),
                                   "corr": round(float(np.corrcoef(xs, ys)[0, 1]), 4)}
        lab = f"blocks[{pe[0]},{pe[1]}) = gp[{pe[0]*BLOCK},{pe[1]*BLOCK})"
        mc[lab] = prof
    out["M_C_block_autocorr_by_phase"] = mc
    out["M_C_note"] = ("residual = per-game margin minus the season-SRS "
                       "prediction, i.e. the team's within-season deviation "
                       "from its own season-long strength (+ noise). Blocks "
                       "are non-overlapping 5-game means. A STATIONARY team "
                       "gives corr ~ 0 at every lag with no trend; a DRIFTING "
                       "team gives positive corr at lag 1 decaying in m. The "
                       "phase comparison is the measurement.")

    (ROOT / "data" / "am_measure.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
