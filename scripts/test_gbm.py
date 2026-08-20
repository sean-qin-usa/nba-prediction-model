"""GBM challenger (handoff I.5 mandates a GBM baseline): HistGradientBoosting
over our validated margin features [ff_margin, comp_margin, ratings_margin,
rest_adv] — does nonlinearity/interaction add anything beyond the linear blend?
Walk-forward: train on first 60% of 2025-26 test games' features, predict rest.
Features are themselves walk-forward (no leakage); the GBM train/test split is
temporal.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect
from nbapred.eval.metrics import log_loss
from nbapred.model.team_ratings import TeamRatings
from nbapred.model.composition import CompositionModel
from scripts.test_ff_gbm import factor_rows

SCALE = 7.2
sig = lambda x: 1 / (1 + np.exp(-np.asarray(x)))
FACTORS = ["efg", "tovr", "orbr", "ftr"]


def build_features(season="2025-26"):
    """Walk-forward per-game features: ff/comp/ratings margins + rest adv."""
    con = connect(read_only=True)
    rows = factor_rows(con, season)
    pm = con.execute("""SELECT game_id, team_id, player_id FROM player_game_stats
        WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    played = {(g, t): set(grp.player_id) for (g, t), grp in pm.groupby(["game_id", "team_id"])}
    rest = {}
    for r in con.execute("""SELECT game_id, is_home, days_rest FROM schedule_features
                          WHERE season=?""", [season]).fetchall():
        rest.setdefault(r[0], {})[bool(r[1])] = r[2]
    games = {}
    for r in rows:
        games.setdefault(r["gid"], []).append(r)
    order = [g for g in dict.fromkeys(r["gid"] for r in rows)]

    out = []
    last = None; fms = {}; rm = None; W = None; comp = None; hist = []
    for gid in order:
        recs = games[gid]
        if len(recs) != 2:
            continue
        h = next(x for x in recs if x["home"]); a = next(x for x in recs if not x["home"])
        d = h["date"]; dd = d.date() if hasattr(d, "date") else d
        if last is None or (dd - last).days >= 7:
            if len(hist) > 200:
                fms = {f: TeamRatings(ridge=25.0, team_home_ridge=None).fit(
                    [(x["tid"], x["oid"], x["home"], x[f] * 100) for x in hist]) for f in FACTORS}
                rm = TeamRatings(ridge=25.0).fit(
                    [(x["tid"], x["oid"], x["home"], x["ortg"]) for x in hist])
                X = np.array([[fms[f].pred_ortg(x["tid"], x["oid"], x["home"])
                               for f in FACTORS] for x in hist])
                yy = np.array([x["ortg"] for x in hist])
                W = np.linalg.lstsq(np.c_[X, np.ones(len(X))], yy, rcond=None)[0]
                comp = CompositionModel(con, before=dd)
            last = dd
        if fms and rm is not None and comp is not None:
            def eortg(t, o, hm):
                xf = np.array([fms[f].pred_ortg(t, o, hm) for f in FACTORS])
                return float(xf @ W[:4] + W[4])
            ffm = eortg(h["tid"], a["tid"], True) - eortg(a["tid"], h["tid"], False)
            rtm = (rm.pred_ortg(h["tid"], a["tid"], True)
                   - rm.pred_ortg(a["tid"], h["tid"], False))
            outs = {}
            for t in (h["tid"], a["tid"]):
                pl = played.get((gid, t), set())
                outs[t] = {p for p, d0 in comp.players.items()
                           if d0["team_id"] == t and (dd - d0["last_played"]).days <= 12
                           and p not in pl}
            cpm = comp.margin(h["tid"], a["tid"], outs[h["tid"]], outs[a["tid"]], dd)
            rg = rest.get(gid, {})
            radv = 0.0
            if True in rg and False in rg and rg[True] is not None and rg[False] is not None:
                radv = float(np.clip(rg[True] - rg[False], -3, 3))
            out.append(dict(date=dd, y=h["win"], ff=ffm, cp=cpm, rt=rtm, rest=radv))
        hist.append(h); hist.append(a)
    con.close()
    return out


def main():
    feats = build_features()
    feats.sort(key=lambda r: r["date"])
    cut = int(len(feats) * 0.6)
    tr, te = feats[:cut], feats[cut:]
    Xtr = np.array([[r["ff"], r["cp"], r["rt"], r["rest"]] for r in tr])
    ytr = np.array([r["y"] for r in tr])
    Xte = np.array([[r["ff"], r["cp"], r["rt"], r["rest"]] for r in te])
    yte = np.array([r["y"] for r in te])
    print(f"train {len(tr)}  test {len(te)}")

    # linear blend baseline (fit logistic on train)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1.0).fit(Xtr, ytr)
    p_lr = lr.predict_proba(Xte)[:, 1]
    # fixed 50/50 ff+comp for reference
    p_fix = sig((0.5 * Xte[:, 0] + 0.5 * Xte[:, 1]) / SCALE)
    # GBM
    from sklearn.ensemble import HistGradientBoostingClassifier
    gbm = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                         max_iter=300, l2_regularization=1.0,
                                         early_stopping=True).fit(Xtr, ytr)
    p_gbm = gbm.predict_proba(Xte)[:, 1]
    print(f"fixed 50/50 ff+comp : {log_loss(yte, p_fix):.4f}")
    print(f"logistic blend      : {log_loss(yte, p_lr):.4f}")
    print(f"GBM (nonlinear)     : {log_loss(yte, p_gbm):.4f}")
    d = (-(yte * np.log(np.clip(p_lr, 1e-9, 1)) + (1 - yte) * np.log(np.clip(1 - p_lr, 1e-9, 1)))) \
        - (-(yte * np.log(np.clip(p_gbm, 1e-9, 1)) + (1 - yte) * np.log(np.clip(1 - p_gbm, 1e-9, 1))))
    rng = np.random.default_rng(0)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"GBM vs logistic delta {d.mean():+.5f} CI ({lo:+.5f},{hi:+.5f}) "
          f"-> {'KEEP GBM' if lo > 0 else 'linear suffices'}")


if __name__ == "__main__":
    main()
