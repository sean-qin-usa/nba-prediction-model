"""EXPERIMENT homesens — player home/road sensitivity (Sean's idea).

Question: does a roster-aggregated, EB-shrunk player home-lift predict home
margin BEYOND the schedule layer's global home edge + B2B terms?

Standalone experiment file. READS ONLY (DuckDB read_only=True). Never edits
nbapred/ or prod scripts. No odds/market inputs to any feature.

Parts:
  1. DESCRIPTIVE (full-sample, labelled as such — not used in PIT features):
     per-player home-minus-road pts/36, TS%, FT% differentials; DerSimonian-
     Laird tau^2; distribution of raw diffs, shrink factors, shrunk values.
  2. PIT FEATURE: for each capstone game, team home-sensitivity = minutes-
     weighted sum of trailing-window EB-shrunk player home-lifts for the
     trailing roster (last 10 team games within 35 days; splits + tau^2
     recomputed per date from games strictly BEFORE the date).
     Theory: home margin lift = (HS_home + HS_away)/2 — a home-sensitive
     roster gains at home AND loses on the road, so the SUM enters.
  3. GATES:
     A. pooled + per-season OLS: margin ~ 1 + b2b_h + b2b_a + dead_h + dead_a
        + wpct_diff + HS_sum; bootstrap (2000) CI on the HS coefficient.
     B. production-controlled OLS: margin ~ 1 + m_prod(=7.2*logit(p_us))
        + b2b_h + b2b_a + HS_sum. The honest "beyond the shipped model" test.
     C. paired log-loss gate vs data/capstone_pergame_sched.csv: walk-forward
        ridge beta on trailing games only, p_adj = sigmoid((m_prod +
        beta_t*HS)/7.2); paired bootstrap (2000) 95% CI on per-game log-loss
        delta. Also a theory-fixed beta=0.5 variant (no fitting at all).
"""
import sys, warnings, json, datetime as dt
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from nbapred.db import connect

SCALE = 7.2
SCRATCH = Path("/tmp/claude-1004/-hdd-steveqin-sean-dev/4912ce58-69cd-488d-a5f4-0d3ed2eed30c/scratchpad")
SCRATCH.mkdir(parents=True, exist_ok=True)

MIN_SEC = 240          # >=4 min for rate rows (avoid garbage-minute blowups)
DESC_MIN_G = 10        # descriptive: games per side
DESC_MIN_FTA = 25      # descriptive: FTA per side
PIT_MIN_G = 5          # PIT: games per side (shrinkage handles the noise)
PIT_MIN_FTA = 15
ROSTER_G, ROSTER_DAYS = 10, 35   # trailing roster window
RIDGE_PSEUDO = 300.0   # gate C: pseudo-games of beta=0 prior


# ---------------------------------------------------------------- data load
def load(con):
    pg = con.execute("""
        SELECT s.game_id, s.player_id, s.team_id, g.game_date, g.is_home,
               s.seconds, s.pts, s.fga, s.fta, s.ftm
        FROM player_game_stats s
        JOIN nba_games g ON s.game_id = g.game_id AND s.team_id = g.team_id
        WHERE g.game_id LIKE '002%' AND s.seconds > 0
        ORDER BY g.game_date""").fetchdf()
    pg["game_date"] = pd.to_datetime(pg.game_date).dt.date
    pg["mins"] = pg.seconds / 60.0
    pg["pts36"] = np.where(pg.seconds >= MIN_SEC, pg.pts * 36.0 / pg.mins, np.nan)
    pg["tsa"] = pg.fga + 0.44 * pg.fta
    pg["ts"] = np.where((pg.seconds >= MIN_SEC) & (pg.tsa >= 3),
                        pg.pts / (2 * pg.tsa), np.nan)
    games = con.execute("""
        SELECT season, game_id, game_date, team_id, team_abbrev, is_home, wl, pts
        FROM nba_games WHERE game_id LIKE '002%' AND pts IS NOT NULL
        ORDER BY game_date""").fetchdf()
    games["game_date"] = pd.to_datetime(games.game_date).dt.date
    return pg, games


# ------------------------------------------------- weighted split machinery
def _side_agg(df, val, wcol):
    """per (player, side): n games, sum w, weighted mean, var of the mean."""
    d = df.dropna(subset=[val]).copy()
    w = d[wcol].values
    x = d[val].values
    d["_w"], d["_wx"], d["_wx2"], d["_w2"] = w, w * x, w * x * x, w * w
    g = d.groupby(["player_id", "is_home"])[["_w", "_wx", "_wx2", "_w2"]].sum()
    g["n"] = d.groupby(["player_id", "is_home"]).size()
    mean = g._wx / g._w
    varw = np.maximum(g._wx2 / g._w - mean ** 2, 0.0)
    neff = g._w ** 2 / g._w2
    vmean = varw / np.maximum(neff - 1.0, 1.0)
    return pd.DataFrame({"n": g.n, "mean": mean, "vmean": vmean})


def rate_diffs(df, val, wcol, min_g):
    """home-minus-road diff + sampling variance per eligible player."""
    a = _side_agg(df, val, wcol).unstack("is_home")
    try:
        nh, nr = a[("n", True)], a[("n", False)]
        mh, mr = a[("mean", True)], a[("mean", False)]
        vh, vr = a[("vmean", True)], a[("vmean", False)]
    except KeyError:
        return pd.DataFrame(columns=["d", "v"])
    ok = (nh >= min_g) & (nr >= min_g)
    out = pd.DataFrame({"d": (mh - mr)[ok], "v": (vh + vr)[ok]}).dropna()
    return out[out.v > 0]


def ft_diffs(df, min_fta):
    d = df[df.fta > 0]
    g = d.groupby(["player_id", "is_home"])[["ftm", "fta"]].sum().unstack("is_home")
    try:
        fmh, fah = g[("ftm", True)], g[("fta", True)]
        fmr, far = g[("ftm", False)], g[("fta", False)]
    except KeyError:
        return pd.DataFrame(columns=["d", "v"])
    ok = (fah >= min_fta) & (far >= min_fta)
    p = (fmh + fmr) / (fah + far)          # pooled per-player FT% for variance
    v = p * (1 - p) * (1 / fah + 1 / far)
    out = pd.DataFrame({"d": (fmh / fah - fmr / far)[ok], "v": v[ok]}).dropna()
    return out[out.v > 0]


def dl_shrink(dv):
    """DerSimonian-Laird tau^2; returns (tau2, prec-weighted mean, shrunk
    DEVIATIONS from the global mean — the global home effect belongs to the
    schedule layer, not this feature)."""
    if len(dv) < 20:
        return 0.0, 0.0, pd.Series(0.0, index=dv.index)
    d, v = dv.d.values, dv.v.values
    u = 1.0 / v
    db = (u * d).sum() / u.sum()
    Q = (u * (d - db) ** 2).sum()
    k = len(d)
    c = u.sum() - (u ** 2).sum() / u.sum()
    tau2 = max(0.0, (Q - (k - 1)) / c)
    shrunk = tau2 / (tau2 + v) * (d - db)
    return tau2, db, pd.Series(shrunk, index=dv.index)


# --------------------------------------------------------- part 1: descriptive
def descriptive(pg):
    rep = {}
    for name, dv in (("pts36", rate_diffs(pg, "pts36", "mins", DESC_MIN_G)),
                     ("ts", rate_diffs(pg, "ts", "tsa", DESC_MIN_G)),
                     ("ft", ft_diffs(pg, DESC_MIN_FTA))):
        tau2, db, sh = dl_shrink(dv)
        f = tau2 / (tau2 + dv.v)     # shrink factor kept (1 = no shrink)
        q = lambda s, ps=(5, 25, 50, 75, 95): {f"p{p}": round(float(np.percentile(s, p)), 4) for p in ps}
        rep[name] = {
            "n_players": int(len(dv)), "tau": round(float(np.sqrt(tau2)), 4),
            "global_home_lift": round(float(db), 4),
            "raw_diff_sd": round(float(dv.d.std()), 4),
            "mean_sampling_sd": round(float(np.sqrt(dv.v).mean()), 4),
            "raw_diff_pctiles": q(dv.d),
            "shrink_factor_pctiles": q(f),
            "shrunk_dev_pctiles": q(sh),
        }
    return rep


# ------------------------------------------------------ part 2: PIT features
# STEELMAN variant: DL tau^2 sits at the boundary (0) once data accumulates,
# zeroing the feature. A fixed A-PRIORI tau floor (tau=0.5 pts/36 — "players
# plausibly differ by +-0.5 pts36 in home sensitivity"; constant chosen from
# hypothesis scale, never from outcomes -> leakage-free) keeps the feature
# alive in all 3 seasons. Suffix "F" columns.
TAU_FLOOR_PTS2 = 0.25


def pit_features(pg, games, cap):
    # per-game minutes lookup + per-team ordered schedule
    minmap = {}
    for gid, grp in pg.groupby("game_id"):
        minmap[gid] = dict(zip(grp.player_id.astype(int), grp.mins))
    team_sched = {}
    tg = games.drop_duplicates(["game_id", "team_id"])
    for r in tg.itertuples():
        team_sched.setdefault(r.team_id, []).append((r.game_date, r.game_id))
    for t in team_sched:
        team_sched[t].sort()

    ab2id = {}
    for r in games.itertuples():
        ab2id[(r.season, r.team_abbrev)] = r.team_id

    def roster_weights(tid, d):
        past = [(dd, gid) for dd, gid in team_sched.get(tid, [])
                if dd < d and (d - dd).days <= ROSTER_DAYS][-ROSTER_G:]
        if not past:
            return {}
        tot = {}
        for _, gid in past:
            for pid, m in minmap.get(gid, {}).items():
                tot[pid] = tot.get(pid, 0.0) + m
        s = sum(tot.values())
        if s <= 0:
            return {}
        return {pid: 240.0 * m / s for pid, m in tot.items()}   # sum to 240

    cap = cap.sort_values("game_date").reset_index(drop=True)
    pg_dates = pg.game_date.values
    keys = ["hs_pts_h", "hs_pts_a", "hs_ts_h", "hs_ts_a", "hs_ft_h", "hs_ft_a",
            "hs_ptsF_h", "hs_ptsF_a"]
    feats = {k: np.zeros(len(cap)) for k in keys}
    taus = []
    for d, idx in cap.groupby("game_date").groups.items():
        hist = pg[pg_dates < d]
        lifts = {}
        for name, dv in (("pts", rate_diffs(hist, "pts36", "mins", PIT_MIN_G)),
                         ("ts", rate_diffs(hist, "ts", "tsa", PIT_MIN_G)),
                         ("ft", ft_diffs(hist, PIT_MIN_FTA))):
            tau2, _, sh = dl_shrink(dv)
            lifts[name] = dict(zip((int(i) for i in sh.index), sh.values))
            if name == "pts":
                taus.append((str(d), float(np.sqrt(tau2)), len(dv)))
                # steelman: fixed a-priori tau floor, same centered deviations
                if len(dv) >= 20:
                    t2f = max(tau2, TAU_FLOOR_PTS2)
                    u = 1.0 / dv.v.values
                    db = (u * dv.d.values).sum() / u.sum()
                    shF = t2f / (t2f + dv.v.values) * (dv.d.values - db)
                    lifts["ptsF"] = dict(zip((int(i) for i in dv.index), shF))
                else:
                    lifts["ptsF"] = {}
        for i in idx:
            row = cap.loc[i]
            for side, ab in (("h", row.home), ("a", row.away)):
                tid = ab2id.get((row.season, ab))
                w = roster_weights(tid, d) if tid is not None else {}
                for met in ("pts", "ptsF"):
                    feats[f"hs_{met}_{side}"][i] = sum(
                        (m / 36.0) * lifts[met].get(p, 0.0) for p, m in w.items())
                for met in ("ts", "ft"):
                    feats[f"hs_{met}_{side}"][i] = sum(
                        (m / 240.0) * lifts[met].get(p, 0.0) for p, m in w.items())
    for k, v in feats.items():
        cap[k] = v
    # theory: extra home margin ~ (HS_home + HS_away)/2 -> SUM feature
    for met in ("pts", "ptsF", "ts", "ft"):
        cap[f"x_{met}"] = cap[f"hs_{met}_h"] + cap[f"hs_{met}_a"]
    return cap, taus


# ------------------------------------------- schedule covariates for the gate
def sched_covariates(games, cap):
    meta = games[games.wl.notna()]
    tdates, hist = {}, {}
    for r in meta.itertuples():
        tdates.setdefault((r.season, r.team_id), set()).add(r.game_date)
        hist.setdefault((r.season, r.team_id), []).append((r.game_date, r.wl == "W"))
    for k in hist:
        hist[k].sort()
    from nbapred.model.production import DEAD_WPCT, DEAD_GP

    def b2b(season, tid, d):
        return (d - dt.timedelta(days=1)) in tdates.get((season, tid), set())

    def wpct(season, tid, d):
        past = [w for (dd, w) in hist.get((season, tid), []) if dd < d]
        return (sum(past) / len(past)) if past else 0.5, len(past)

    def dead(season, tid, d):
        w, n = wpct(season, tid, d)
        return n >= DEAD_GP and w < DEAD_WPCT

    ab2id = {(r.season, r.team_abbrev): r.team_id for r in games.itertuples()}
    pts = {(r.game_id, r.team_abbrev): r.pts for r in games.itertuples()}
    rows = []
    for r in cap.itertuples():
        ht, at = ab2id[(r.season, r.home)], ab2id[(r.season, r.away)]
        d = r.game_date
        wh, _ = wpct(r.season, ht, d)
        wa, _ = wpct(r.season, at, d)
        rows.append((pts[(r.game_id, r.home)] - pts[(r.game_id, r.away)],
                     float(b2b(r.season, ht, d)), float(b2b(r.season, at, d)),
                     float(dead(r.season, ht, d)), float(dead(r.season, at, d)),
                     wh - wa))
    cap[["margin", "hb2b", "ab2b", "hdead", "adead", "wdiff"]] = \
        pd.DataFrame(rows, index=cap.index)
    return cap


# ----------------------------------------------------------------- gates
def ols_boot(X, y, j, B=2000, seed=42):
    """coef j of lstsq + bootstrap CI."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(seed)
    n = len(y)
    bs = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        bs[b] = np.linalg.lstsq(X[i], y[i], rcond=None)[0][j]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(beta[j]), float(lo), float(hi)


def gate_ols(cap):
    out = {}
    m_prod = SCALE * np.log(cap.p_us / (1 - cap.p_us)).values
    y = cap.margin.values.astype(float)
    for met in ("pts", "ptsF", "ts", "ft"):
        x = cap[f"x_{met}"].values
        XA = np.c_[np.ones(len(cap)), cap.hb2b, cap.ab2b, cap.hdead, cap.adead,
                   cap.wdiff, x]
        XB = np.c_[np.ones(len(cap)), m_prod, cap.hb2b, cap.ab2b, x]
        resA = ols_boot(XA, y, XA.shape[1] - 1)
        resB = ols_boot(XB, y, XB.shape[1] - 1)
        per = {}
        for s, grp in cap.groupby("season"):
            i = grp.index.values
            per[s] = {
                "A": [round(v, 3) for v in ols_boot(XA[i], y[i], XA.shape[1] - 1)],
                "B": [round(v, 3) for v in ols_boot(XB[i], y[i], XB.shape[1] - 1)],
                "sd_x": round(float(np.std(x[i])), 3)}
        out[met] = {"sd_x": round(float(np.std(x)), 4),
                    "A_coef_ci": [round(v, 3) for v in resA],
                    "B_coef_ci": [round(v, 3) for v in resB],
                    "per_season": per}
    return out


def gate_paired(cap, met="pts"):
    """Gate C: walk-forward ridge beta on trailing games; paired log-loss."""
    cap = cap.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    m_prod = SCALE * np.log(cap.p_us / (1 - cap.p_us)).values
    x = cap[f"x_{met}"].values
    r = cap.margin.values - m_prod          # production residual
    n = len(cap)
    Sxy = np.cumsum(x * r)
    Sxx = np.cumsum(x * x)
    Sx2m = np.cumsum(x * x) / np.arange(1, n + 1)     # trailing mean x^2
    beta = np.zeros(n)
    beta[1:] = Sxy[:-1] / (Sxx[:-1] + RIDGE_PSEUDO * np.maximum(Sx2m[:-1], 1e-9))
    res = {}
    for tag, bvec in (("wf_ridge", beta), ("theory_0.5", np.full(n, 0.5))):
        p_adj = 1 / (1 + np.exp(-(m_prod + bvec * x) / SCALE))
        p_adj = np.clip(p_adj, 1e-6, 1 - 1e-6)
        p0 = np.clip(cap.p_us.values, 1e-6, 1 - 1e-6)
        yv = cap.y.values
        ll0 = -(yv * np.log(p0) + (1 - yv) * np.log(1 - p0))
        ll1 = -(yv * np.log(p_adj) + (1 - yv) * np.log(1 - p_adj))
        delta = ll0 - ll1                    # >0 = improvement
        rng = np.random.default_rng(42)
        bs = np.array([delta[rng.integers(0, n, n)].mean() for _ in range(2000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        per = {s: {"n": int(len(g)),
                   "ll_base": round(float(np.mean(ll0[g.index])), 4),
                   "ll_adj": round(float(np.mean(ll1[g.index])), 4),
                   "delta": round(float(np.mean(delta[g.index])), 5)}
               for s, g in cap.groupby("season")}
        res[tag] = {"pooled_delta": round(float(delta.mean()), 6),
                    "ci": [round(float(lo), 6), round(float(hi), 6)],
                    "final_beta": round(float(bvec[-1]), 4),
                    "per_season": per}
    return res


def main():
    con = connect(read_only=True)
    pg, games = load(con)
    con.close()
    cap = pd.read_csv(Path(__file__).resolve().parent.parent /
                      "data/capstone_pergame_sched.csv", dtype={"game_id": str})
    cap["game_date"] = pd.to_datetime(cap.game_date).dt.date

    print("== PART 1: descriptive EB (full-sample, 3 seasons) ==")
    rep = descriptive(pg)
    print(json.dumps(rep, indent=1))

    print("\n== PART 2: PIT features ==")
    cap, taus = pit_features(pg, games, cap)
    cap = sched_covariates(games, cap)
    fpath = SCRATCH / "homesens_features.csv"
    cap.to_csv(fpath, index=False)
    print("features ->", fpath)
    print("tau_pts36 over time (first/mid/last):", taus[0], taus[len(taus)//2], taus[-1])
    print(cap[["x_pts", "x_ptsF", "x_ts", "x_ft"]].describe().round(4).to_string())
    print("sd of x_ptsF by season:",
          cap.groupby("season").x_ptsF.std().round(3).to_dict())

    print("\n== PART 3A/B: OLS gates (coef, 95% CI) ==")
    og = gate_ols(cap)
    print(json.dumps(og, indent=1))

    print("\n== PART 3C: paired log-loss gate vs baseline ==")
    pgate = {met: gate_paired(cap, met) for met in ("pts", "ptsF")}
    print(json.dumps(pgate, indent=1))

    json.dump({"descriptive": rep, "ols": og, "paired": pgate},
              open(SCRATCH / "homesens_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
