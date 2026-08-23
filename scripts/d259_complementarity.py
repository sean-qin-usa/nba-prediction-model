#!/usr/bin/env python3
"""D259 — DOES LINEUP TENDENCY MIX EXPLAIN WHAT THE ADDITIVE MODEL GETS WRONG?

The synergy question, finally asked of the data. The composition model is
additive in players (talent x minutes, summed) and the ratings model is additive
in teams (off_i - def_j). Neither can express "these two fit together". D257
showed tendencies are real and reliable; D258 built a PIT estimator with fitted
per-axis constants. This asks whether any of it matters.

TWO BARS, in order, because they have different answers and the register keeps
finding an effect that clears the first and not the second:

  BAR 1  Does complementarity predict the ADDITIVE MODEL'S OWN ERROR?
         That is the synergy hypothesis on its own terms. If a team's offence
         systematically beats or misses its additive prediction depending on how
         its available players' tendencies fit together, the additive form is
         mis-specified in a way tendencies explain.

  BAR 2  Does it predict the MARKET residual? Nothing in this register has
         cleared that bar, and an effect can be real at Bar 1 and worth nothing
         at Bar 2 (D255: pair effects persisted and were worth 0.03 points).

EVERYTHING IS POINT-IN-TIME. Tendencies come from D258's shrinkage on prior
games only. Player weights are TRAILING minutes (the 10-game average production
already uses), never minutes actually played tonight -- using realised minutes
would let the outcome pick its own weights.

THE FEATURES are mix, not level. Level is already in the model via talent; only
DISPERSION and FIT are new information:

    sd_fg3   weighted spread of three-point rate across the lineup  (spacing mix)
    sd_ast   spread of playmaking                     (one creator or several)
    sd_rim   spread of rim pressure
    hhi      minutes concentration                    (star-dependence)
    fit_sr   weighted corr(fg3_i, rim_i) across the lineup -- NEGATIVE means
             the spacers and the rim attackers are different players, which is
             the classic complementary shape; POSITIVE means the roster doubles
             up on one archetype
    ast_gap  playmaking supply minus shot-creation demand

DISCIPLINE. D253b showed 24 plausible features cost 80% of out-of-sample R^2, so
the headline is walk-forward OOS R^2 against a permutation null over the WHOLE
procedure, and an ablation against the level-only control. A feature set that
cannot beat its own shuffled labels is not a finding.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import importlib.util                                             # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from scipy import stats                                           # noqa: E402
from sklearn.linear_model import Ridge                            # noqa: E402

_s = importlib.util.spec_from_file_location(
    "d258", ROOT / "scripts" / "d258_tendency_estimator.py")
D258 = importlib.util.module_from_spec(_s); _s.loader.exec_module(D258)
_s2 = importlib.util.spec_from_file_location(
    "d255", ROOT / "scripts" / "d255_matchup_residual.py")
D255 = importlib.util.module_from_spec(_s2); _s2.loader.exec_module(D255)

K = {"fg3_rate": 8, "rim_rate": 16, "ast_rate": 32}   # D258 fitted
MIX = ["sd_fg3", "sd_ast", "sd_rim", "hhi", "fit_sr", "ast_gap"]
LEVEL = ["w_fg3", "w_ast", "w_rim"]


def wstat(w, x):
    w = np.asarray(w, float); x = np.asarray(x, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if ok.sum() < 3:
        return np.nan, np.nan
    w, x = w[ok], x[ok]
    m = np.average(x, weights=w)
    v = np.average((x - m) ** 2, weights=w)
    return m, math_sqrt(v)


def math_sqrt(v):
    return float(np.sqrt(max(v, 0.0)))


def build_features():
    d = D258.load()
    seasons = sorted(d.season.unique())
    nxt = {seasons[i]: seasons[i + 1] for i in range(len(seasons) - 1)}
    prev = {v: k for k, v in nxt.items()}

    # ---- PIT tendency per player-game via D258 shrinkage ----------
    for ax, (nc, dc) in D258.AXES.items():
        if ax not in K:
            continue
        num = d[nc].to_numpy(float); den = d[dc].to_numpy(float)
        key = [d.player_id, d.season]
        cn = d.groupby(key)[nc].cumsum().to_numpy(float) - num
        cd = d.groupby(key)[dc].cumsum().to_numpy(float) - den
        ps = d.groupby(["player_id", "season"])[[nc, dc]].sum().reset_index()
        ps["rate"] = ps[nc] / ps[dc].replace(0, np.nan)
        ps["tgt"] = ps.season.map(nxt)
        pri = ps.dropna(subset=["tgt"]).set_index(["tgt", "player_id"]).rate
        lg = d.groupby("season").apply(
            lambda x: x[nc].sum() / max(x[dc].sum(), 1e-9), include_groups=False)
        lgp = d.season.map({s: lg.get(prev.get(s), lg.mean()) for s in seasons})
        base = np.array([pri.get((s, p), np.nan)
                         for s, p in zip(d.season, d.player_id)])
        base = np.where(np.isfinite(base), base, lgp.to_numpy(float))
        d[f"t_{ax}"] = (cn + K[ax] * base) / (cd + K[ax])

    # ---- PIT trailing minutes (what production uses) ---------------
    d["mins"] = d.seconds / 60.0
    d["trail_min"] = (d.groupby(["player_id", "season"])["mins"]
                      .transform(lambda s: s.shift(1).rolling(10, min_periods=3)
                                 .mean()))
    d = d.dropna(subset=["trail_min", "t_fg3_rate", "t_ast_rate", "t_rim_rate"])
    d = d[d.trail_min >= 8.0]

    # ---- team-game mix features ------------------------------------
    rows = []
    for (gid, season), g in d.groupby(["gid", "season"], sort=False):
        pass
    # regroup by team: need team_id, which load() does not carry -> re-pull
    import duckdb
    con = duckdb.connect(str(ROOT / "data" / "nba.duckdb"), read_only=True)
    tm = con.execute("""SELECT CAST(game_id AS VARCHAR) gid, player_id, team_id
                        FROM player_game_stats
                        WHERE CAST(game_id AS VARCHAR) LIKE '002%'""").df()
    con.close()
    tm["gid"] = tm.gid.str.zfill(10)
    d = d.merge(tm, on=["gid", "player_id"], how="inner")

    for (gid, tid), g in d.groupby(["gid", "team_id"], sort=False):
        w = g.trail_min.to_numpy(float)
        if w.sum() <= 0 or len(g) < 5:
            continue
        f3, ar, rr = (g.t_fg3_rate.to_numpy(float), g.t_ast_rate.to_numpy(float),
                      g.t_rim_rate.to_numpy(float))
        m3, s3 = wstat(w, f3); ma, sa = wstat(w, ar); mr, sr = wstat(w, rr)
        p = w / w.sum()
        if not np.isfinite(s3):
            continue
        # weighted correlation between spacing and rim pressure
        c = np.average((f3 - m3) * (rr - mr), weights=w)
        fit = c / (s3 * sr) if (s3 > 1e-9 and sr > 1e-9) else 0.0
        rows.append(dict(gid=gid, team_id=tid, season=g.season.iloc[0],
                         w_fg3=m3, w_ast=ma, w_rim=mr,
                         sd_fg3=s3, sd_ast=sa, sd_rim=sr,
                         hhi=float((p ** 2).sum()), fit_sr=float(fit),
                         ast_gap=float(ma - m3)))
    return pd.DataFrame(rows)


def clus(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    k = len(v); se = v.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return v.mean(), v.mean() - tc * se, v.mean() + tc * se, k


def walk(df, feats, target, seasons):
    pred = np.full(len(df), np.nan)
    for i, s in enumerate(seasons):
        if i < 3:
            continue
        tr = (df.season < s).to_numpy(); te = (df.season == s).to_numpy()
        if tr.sum() < 2000 or te.sum() < 200:
            continue
        X, Xt = df.loc[tr, feats].to_numpy(float), df.loc[te, feats].to_numpy(float)
        mu, sd = X.mean(0), X.std(0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        m = Ridge(alpha=50.0).fit((X - mu) / sd, target[tr])
        pred[te] = m.predict((Xt - mu) / sd)
    ok = np.isfinite(pred)
    r = target[ok] - pred[ok]; b = target[ok] - target[ok].mean()
    return float(1 - (r ** 2).sum() / (b ** 2).sum()), ok


def main():
    feat = build_features()
    print(f"{len(feat):,} team-games with a PIT mix profile")

    # ---- outcome: additive-model ortg residual ---------------------
    tg = D255.build_team_games()
    parts = []
    for s, g in tg.groupby("season"):
        g = g.copy(); g["resid"] = D255.fit_additive(g); parts.append(g)
    tg = pd.concat(parts)
    df = feat.merge(tg[["gid", "team_id", "resid", "ab", "opp_ab"]],
                    on=["gid", "team_id"], how="inner")
    df = df.dropna(subset=MIX + LEVEL + ["resid"]).reset_index(drop=True)
    df = df.sort_values("season").reset_index(drop=True)
    seasons = sorted(df.season.unique())
    print(f"{len(df):,} team-games joined to an additive residual, "
          f"{len(seasons)} seasons\n")

    print("=" * 76)
    print("BAR 1  does tendency MIX explain the additive model's own error?")
    print("=" * 76)
    y = df.resid.to_numpy(float)
    r2_mix, _ = walk(df, MIX, y, seasons)
    r2_lvl, _ = walk(df, LEVEL, y, seasons)
    r2_all, _ = walk(df, LEVEL + MIX, y, seasons)
    print(f"  level only  (w_fg3,w_ast,w_rim)   OOS R^2 {r2_lvl:+.5f}")
    print(f"  MIX only    (dispersion + fit)    OOS R^2 {r2_mix:+.5f}")
    print(f"  level + mix                       OOS R^2 {r2_all:+.5f}")

    rng = np.random.default_rng(259)
    scode = pd.factorize(df.season)[0]
    idx = [np.flatnonzero(scode == i) for i in range(scode.max() + 1)]
    null = []
    for _ in range(60):
        yp = y.copy()
        for ix in idx:
            yp[ix] = rng.permutation(y[ix])
        null.append(walk(df, MIX, yp, seasons)[0])
    null = np.array(null)
    p = float((null >= r2_mix).mean())
    print(f"\n  permutation null (60 draws, residual shuffled within season):")
    print(f"    median {np.median(null):+.5f}, 95th {np.percentile(null,95):+.5f}")
    print(f"    observed {r2_mix:+.5f}  p = {p:.3f}  "
          f"{'MIX MATTERS' if p < 0.05 else 'NULL — additive form is not violated in a way mix explains'}")

    print("\n" + "=" * 76)
    print("BAR 2  does it predict the MARKET residual?")
    print("=" * 76)
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
    f["game_id"] = (f.game_id.astype(str).str.replace(r"\.0$", "", regex=True)
                    .str.zfill(10))
    f = f.dropna(subset=["margin_actual", "close_margin", "open_margin"])
    # team-perspective market residual
    h = f.assign(ab=f.home, mres=f.margin_actual - f.close_margin,
                 mres_o=f.margin_actual - f.open_margin)
    a = f.assign(ab=f.away, mres=-(f.margin_actual - f.close_margin),
                 mres_o=-(f.margin_actual - f.open_margin))
    L = pd.concat([h[["game_id", "ab", "mres", "mres_o"]],
                   a[["game_id", "ab", "mres", "mres_o"]]])
    m = df.merge(L, left_on=["gid", "ab"], right_on=["game_id", "ab"],
                 how="inner").dropna(subset=["mres"])
    m = m.sort_values("season").reset_index(drop=True)
    ms = sorted(m.season.unique())
    print(f"  {len(m):,} team-games with a market residual")
    for tgt, lab in (("mres", "vs CLOSE"), ("mres_o", "vs OPEN")):
        r2, _ = walk(m, LEVEL + MIX, m[tgt].to_numpy(float), ms)
        print(f"    {lab}: OOS R^2 {r2:+.6f}  "
              f"{'' if r2 > 0 else '(worse than predicting the mean)'}")

    json.dump({"bar1_mix": r2_mix, "bar1_level": r2_lvl, "bar1_all": r2_all,
               "bar1_p": p}, open(ROOT / "data" / "d259_complementarity.json", "w"),
              default=float)
    print("\nwrote data/d259_complementarity.json")


if __name__ == "__main__":
    main()
