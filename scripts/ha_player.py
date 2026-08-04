"""HA-(4) PLAYER-LEVEL HOME SENSITIVITY (Sean's long-standing idea, never built).

Per player-season, an EB-shrunk home-vs-road split in a PER-MINUTE performance
measure:
    pts36   = 36 * pts / minutes
    gmsc36  = 36 * GameScore / minutes   (Hollinger composite)
    ts_pct  = pts / (2*(fga + 0.44*fta))     (efficiency, no volume)

For each player-season we fit, by weighted least squares with minutes as
weights,
    rate_i = mu_p + delta_p * 1{home} + e_i
so delta_p is the home-minus-road per-36 split in points, with an honest SE
(the WLS variance under var(rate_i) ∝ 1/minutes_i, which is what a counting
stat divided by minutes actually has).

QUESTIONS ANSWERED
  (4a) how big is the raw spread of delta_p, and how much of it is sampling
       noise (method-of-moments EB, exactly as in deliverable 1)?
  (4b) split-half reliability WITHIN a season, and lag-1 ACROSS seasons.
  (4c) is the distribution of TRUE player home-sensitivity distinguishable
       from zero spread?
  (4d) THE PRIZE: aggregate prior-season player deltas by roster minutes into a
       team-level, point-in-time roster home edge. Does it predict the team's
       realised home deviation d_t BETTER than the team's own prior-season d_t?

(4d) is the only PIT-disciplined part: the roster aggregate for season y uses
only player splits estimated from seasons STRICTLY BEFORE y. (4a)-(4c) are
descriptive.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import numpy as np
import pandas as pd

from ha_core import CONTROLS, boot_ci, eb_shrink, fit_season, load_panel

SEED = 20260801
REPO = Path(__file__).resolve().parent.parent
DB = REPO / "data" / "nba.duckdb"
OUT = Path("/tmp/claude-1004/-hdd-steveqin-sean-dev/"
           "4912ce58-69cd-488d-a5f4-0d3ed2eed30c/scratchpad")
SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
NORMAL = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
MIN_SEC_GAME = 300          # >=5 min to count a game
MIN_SIDE_MIN = 250          # >=250 minutes on EACH side for a usable split


def connect_ro(retries=12, wait=5.0):
    last = None
    for _ in range(retries):
        try:
            return duckdb.connect(str(DB), read_only=True)
        except Exception as e:
            last = e
            time.sleep(wait)
    raise last


def load_players() -> pd.DataFrame:
    con = connect_ro()
    q = """
    WITH g AS (
        SELECT game_id, season, game_date, team_abbrev, matchup, is_home
        FROM nba_games WHERE game_id LIKE '002%'
    ), host AS (
        SELECT game_id,
               max(CASE WHEN position('@' in matchup) > 0
                        THEN trim(split_part(matchup,'@',2))
                        WHEN position('vs.' in matchup) > 0
                        THEN trim(split_part(matchup,'vs.',1)) END) AS host,
               sum(CASE WHEN is_home THEN 1 ELSE 0 END) AS n_home
        FROM g GROUP BY game_id
    )
    SELECT p.game_id, p.player_id, g.season, g.game_date, g.team_abbrev AS team,
           h.host, (g.team_abbrev = h.host) AS is_home,
           p.seconds, p.pts, p.fga, p.fgm, p.fta, p.ftm, p.fg3m,
           p.oreb, p.dreb, p.ast, p.stl, p.blk, p.tov, p.pf
    FROM player_game_stats p
    JOIN g ON g.game_id = p.game_id
    JOIN nba_players np ON np.player_id = p.player_id
    JOIN host h ON h.game_id = p.game_id
    WHERE p.seconds >= ? AND h.n_home = 1
      AND g.team_abbrev = (SELECT team_abbrev FROM g g2
                           WHERE g2.game_id = p.game_id
                             AND g2.team_id = p.team_id LIMIT 1)
    """
    try:
        df = con.execute(q, [MIN_SEC_GAME]).fetchdf()
    except Exception:
        # simpler, explicit join on team_id
        df = con.execute("""
        WITH g AS (
            SELECT game_id, season, game_date, team_id, team_abbrev, matchup, is_home
            FROM nba_games WHERE game_id LIKE '002%'
        ), host AS (
            SELECT game_id,
                   max(CASE WHEN position('@' in matchup) > 0
                            THEN trim(split_part(matchup,'@',2))
                            WHEN position('vs.' in matchup) > 0
                            THEN trim(split_part(matchup,'vs.',1)) END) AS host,
                   sum(CASE WHEN is_home THEN 1 ELSE 0 END) AS n_home
            FROM g GROUP BY game_id
        )
        SELECT p.game_id, p.player_id, g.season, g.game_date,
               g.team_abbrev AS team, h.host,
               (g.team_abbrev = h.host) AS is_home,
               p.seconds, p.pts, p.fga, p.fgm, p.fta, p.ftm, p.fg3m,
               p.oreb, p.dreb, p.ast, p.stl, p.blk, p.tov, p.pf
        FROM player_game_stats p
        JOIN g ON g.game_id = p.game_id AND g.team_id = p.team_id
        JOIN host h ON h.game_id = p.game_id
        WHERE p.seconds >= ? AND h.n_home = 1
        """, [MIN_SEC_GAME]).fetchdf()
    con.close()
    df["min"] = df["seconds"] / 60.0
    df["gmsc"] = (df.pts + 0.4 * df.fgm - 0.7 * df.fga
                  - 0.4 * (df.fta - df.ftm) + 0.7 * df.oreb + 0.3 * df.dreb
                  + df.stl + 0.7 * df.ast + 0.7 * df.blk - 0.4 * df.pf - df.tov)
    df["pts36"] = 36.0 * df.pts / df["min"]
    df["gmsc36"] = 36.0 * df.gmsc / df["min"]
    tsa = 2.0 * (df.fga + 0.44 * df.fta)
    df["ts"] = np.where(tsa > 0, df.pts / tsa, np.nan)
    df["is_home"] = df["is_home"].astype(bool)
    return df


def splits(df: pd.DataFrame, metric: str, key=("season", "player_id")):
    """Minutes-weighted home-minus-road split + WLS SE, per group."""
    rows = []
    for k, g in df.groupby(list(key), sort=False):
        h = g[g.is_home]; a = g[~g.is_home]
        Mh, Ma = h["min"].sum(), a["min"].sum()
        if Mh < MIN_SIDE_MIN or Ma < MIN_SIDE_MIN:
            continue
        y = g[metric].to_numpy(float)
        w = g["min"].to_numpy(float)
        ok = np.isfinite(y)
        y, w = y[ok], w[ok]
        hh = g.is_home.to_numpy()[ok].astype(float)
        X = np.column_stack([np.ones(len(y)), hh])
        W = w
        XtW = X.T * W
        try:
            beta = np.linalg.solve(XtW @ X, XtW @ y)
        except np.linalg.LinAlgError:
            continue
        r = y - X @ beta
        # scale c such that var(y_i) = c / w_i
        c = float((W * r * r).sum() / max(len(y) - 2, 1))
        cov = c * np.linalg.inv(XtW @ X)
        rows.append(dict(zip(key, k if isinstance(k, tuple) else (k,))) |
                    dict(delta=float(beta[1]), se=float(np.sqrt(max(cov[1, 1], 1e-12))),
                         min_home=float(Mh), min_road=float(Ma),
                         n_games=len(g), base=float(beta[0])))
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(SEED)
    df = load_players()
    print(f"player-game rows (>= {MIN_SEC_GAME}s): {len(df)}")
    print(df.groupby("season").agg(rows=("pts", "size"),
                                   players=("player_id", "nunique")).to_string())
    res = {}

    # ---------- league-level per-36 home effect ---------------------------
    print("\n=== (4-0) LEAGUE-WIDE per-minute home effect (context) ===")
    for m in ["pts36", "gmsc36", "ts"]:
        h = df[df.is_home]; a = df[~df.is_home]
        wm = lambda s, w: np.nansum(s * w) / np.nansum(w * np.isfinite(s))
        vh = wm(h[m].to_numpy(float), h["min"].to_numpy(float))
        va = wm(a[m].to_numpy(float), a["min"].to_numpy(float))
        print(f"  {m:8s} home {vh:.4f}  road {va:.4f}  diff {vh-va:+.4f}")

    # ---------- (4a) spread and signal share ------------------------------
    print(f"\n=== (4a) PLAYER-SEASON HOME SPLITS (>= {MIN_SIDE_MIN} min each side) ===")
    S = {}
    for m in ["pts36", "gmsc36"]:
        s = splits(df, m)
        S[m] = s
        sn = s[s.season.isin(NORMAL)]
        tau2, shr, share = eb_shrink(sn.delta.to_numpy(), sn.se.to_numpy())
        print(f"  {m}: n={len(sn)} player-seasons (normal 5)")
        print(f"     mean delta      = {sn.delta.mean():+.4f}")
        print(f"     sd(delta_hat)   = {sn.delta.std(ddof=1):.4f}")
        print(f"     rms sampling SE = {np.sqrt((sn.se**2).mean()):.4f}")
        print(f"     tau (true SD)   = {np.sqrt(tau2):.4f}")
        print(f"     SIGNAL SHARE    = {share*100:.2f}%")
        res[f"{m}_eb"] = dict(n=len(sn), mean=float(sn.delta.mean()),
                              sd=float(sn.delta.std(ddof=1)),
                              rms_se=float(np.sqrt((sn.se**2).mean())),
                              tau=float(np.sqrt(tau2)), share=float(share))

    # ---------- (4b) split-half + lag-1 -----------------------------------
    print("\n=== (4b) RELIABILITY OF PLAYER HOME SPLITS ===")
    for m in ["pts36", "gmsc36"]:
        # split-half: randomly halve each player's games
        cors = []
        for _ in range(30):
            coin = rng.random(len(df)) < 0.5
            s1 = splits(df[coin], m)
            s2 = splits(df[~coin], m)
            j = s1.merge(s2, on=["season", "player_id"], suffixes=("_1", "_2"))
            j = j[j.season.isin(NORMAL)]
            if len(j) > 50:
                cors.append(np.corrcoef(j.delta_1, j.delta_2)[0, 1])
        rh = float(np.mean(cors))
        sb = 2 * rh / (1 + rh)
        # lag-1 across seasons
        s = S[m].copy()
        s["yi"] = s.season.map({y: i for i, y in enumerate(SEASONS)})
        nxt = s.copy(); nxt["yi"] = nxt["yi"] - 1
        pair = s.merge(nxt, on=["player_id", "yi"], suffixes=("_0", "_1"))
        pair = pair[pair.season_0.isin(NORMAL) | pair.season_1.isin(NORMAL)]
        r1 = float(np.corrcoef(pair.delta_0, pair.delta_1)[0, 1]) if len(pair) > 30 else np.nan
        # lag-2
        nxt2 = s.copy(); nxt2["yi"] = nxt2["yi"] - 2
        pair2 = s.merge(nxt2, on=["player_id", "yi"], suffixes=("_0", "_2"))
        r2 = float(np.corrcoef(pair2.delta_0, pair2.delta_2)[0, 1]) if len(pair2) > 30 else np.nan
        # bootstrap CI on lag-1 (cluster by player)
        pl = pair.player_id.unique()
        bs = []
        for _ in range(2000):
            pick = rng.choice(pl, len(pl), replace=True)
            sel = pair.set_index("player_id").loc[pick]
            bs.append(np.corrcoef(sel.delta_0, sel.delta_1)[0, 1])
        lo, hi = boot_ci(np.array(bs))
        print(f"  {m}: split-half r={rh:+.4f} -> Spearman-Brown {sb:+.4f}")
        print(f"       lag-1 r={r1:+.4f} CI({lo:+.4f},{hi:+.4f}) n={len(pair)} "
              f"{'SIG' if lo>0 or hi<0 else 'NS'} | lag-2 r={r2:+.4f} n={len(pair2)}")
        res[f"{m}_rel"] = dict(split_half=rh, sb=sb, lag1=r1, lag1_lo=lo,
                               lag1_hi=hi, n_lag1=len(pair), lag2=r2,
                               n_lag2=len(pair2))

    # ---------- (4c) is true spread > 0? ----------------------------------
    print("\n=== (4c) IS TRUE PLAYER HOME-SENSITIVITY SPREAD > 0? ===")
    for m in ["pts36", "gmsc36"]:
        sn = S[m][S[m].season.isin(NORMAL)]
        obs = float(sn.delta.std(ddof=1))
        # parametric null: delta_p ~ N(mean, se_p^2), no true spread
        nulls = []
        for _ in range(4000):
            sim = sn.delta.mean() + rng.normal(0, sn.se.to_numpy())
            nulls.append(sim.std(ddof=1))
        nulls = np.array(nulls)
        p = float((nulls >= obs).mean())
        print(f"  {m}: observed sd={obs:.4f}  null sd={nulls.mean():.4f} "
              f"[p5,p95]=({np.percentile(nulls,5):.4f},{np.percentile(nulls,95):.4f})"
              f"  p={p:.4f}  {'SPREAD REAL' if p<0.05 else 'CANNOT REJECT ZERO SPREAD'}")
        res[f"{m}_spread_test"] = dict(obs=obs, null=float(nulls.mean()), p=p)

    # ---------- (4d) THE PRIZE: roster aggregation, PIT --------------------
    print("\n=== (4d) ROSTER-AGGREGATED HOME EDGE vs TEAM IDENTITY (PIT) ===")
    d = load_panel()
    dev = {}
    for s in SEASONS:
        f = fit_season(d[d.season == s], CONTROLS)
        dev[s] = pd.Series(f["d"], index=f["teams"])
    DEV = pd.DataFrame(dev)

    metric = "gmsc36"
    sp = S[metric]
    # EB-shrink each player-season split toward 0 using the pooled tau
    tau2, _, _ = eb_shrink(sp.delta.to_numpy(), sp.se.to_numpy())
    sp = sp.assign(w=lambda x: tau2 / (tau2 + x.se ** 2))
    sp["shrunk"] = sp["w"] * (sp["delta"] - sp["delta"].mean())

    # minutes per player-season-team (for roster weights)
    mins = (df.groupby(["season", "team", "player_id"])["min"].sum()
              .reset_index().rename(columns={"min": "tmin"}))

    rows = []
    for i, y in enumerate(SEASONS):
        if i == 0:
            continue
        prior_seasons = SEASONS[:i]
        # player home sensitivity from STRICTLY PRIOR seasons, minutes-weighted
        pri = sp[sp.season.isin(prior_seasons)]
        if pri.empty:
            continue
        pri_agg = (pri.assign(wt=pri.min_home + pri.min_road)
                      .groupby("player_id")
                      .apply(lambda g: np.average(g.shrunk, weights=g.wt))
                      .rename("psens").reset_index())
        cur = mins[mins.season == y].merge(pri_agg, on="player_id", how="left")
        cur["psens"] = cur["psens"].fillna(0.0)
        # team roster edge = sum_p psens_p * minutes_share (scaled to a 5-man floor)
        tot = cur.groupby("team")["tmin"].transform("sum")
        cur["share"] = cur["tmin"] / tot
        agg = cur.groupby("team").apply(
            lambda g: float((g.psens * g.share).sum())).rename("roster_edge")
        cover = cur.assign(cov=lambda x: (x.psens != 0) * x.share).groupby(
            "team")["cov"].sum()
        t = pd.DataFrame({"roster_edge": agg, "coverage": cover,
                          "realised_d": DEV[y], "prior_d": DEV[SEASONS[i - 1]]})
        t["season"] = y
        rows.append(t.reset_index().rename(columns={"index": "team"}))
    P = pd.concat(rows)
    P = P.dropna()
    print(f"  panel: {len(P)} team-seasons, mean minute-coverage of prior "
          f"player splits = {P.coverage.mean():.3f}")

    print("\n  HEAD-TO-HEAD: which predicts a team's realised home deviation?")
    for label, sub in (("all seasons w/ a prior", P),
                       ("normal seasons only", P[P.season.isin(NORMAL)])):
        if len(sub) < 30:
            continue
        r_roster = np.corrcoef(sub.roster_edge, sub.realised_d)[0, 1]
        r_prior = np.corrcoef(sub.prior_d, sub.realised_d)[0, 1]

        def bci(x, y, B=4000):
            bs = []
            idx = np.arange(len(x))
            for _ in range(B):
                p = rng.integers(0, len(x), len(x))
                bs.append(np.corrcoef(x.iloc[p], y.iloc[p])[0, 1])
            return boot_ci(np.array(bs))
        lo1, hi1 = bci(sub.roster_edge, sub.realised_d)
        lo2, hi2 = bci(sub.prior_d, sub.realised_d)
        print(f"  {label} (n={len(sub)}):")
        print(f"    ROSTER-AGGREGATED player sensitivity  r={r_roster:+.4f} "
              f"CI({lo1:+.4f},{hi1:+.4f}) {'SIG' if lo1>0 or hi1<0 else 'NS'}")
        print(f"    TEAM IDENTITY (prior-season d_t)      r={r_prior:+.4f} "
              f"CI({lo2:+.4f},{hi2:+.4f}) {'SIG' if lo2>0 or hi2<0 else 'NS'}")
        # difference
        bs = []
        for _ in range(4000):
            p = rng.integers(0, len(sub), len(sub))
            s2 = sub.iloc[p]
            bs.append(np.corrcoef(s2.roster_edge, s2.realised_d)[0, 1] -
                      np.corrcoef(s2.prior_d, s2.realised_d)[0, 1])
        lo, hi = boot_ci(np.array(bs))
        print(f"    DIFFERENCE (roster - identity)        {r_roster-r_prior:+.4f} "
              f"CI({lo:+.4f},{hi:+.4f}) {'SIG' if lo>0 or hi<0 else 'NS'}")
        res[f"h2h_{label}"] = dict(n=len(sub), r_roster=float(r_roster),
                                   r_prior=float(r_prior),
                                   diff=float(r_roster - r_prior),
                                   diff_lo=lo, diff_hi=hi)

    # is the roster aggregate itself more stationary (autocorrelated)?
    print("\n  STATIONARITY of the two candidate signals themselves:")
    for col in ["roster_edge", "prior_d"]:
        piv = P.pivot(index="team", columns="season", values=col)
        cs = []
        cols = [c for c in SEASONS if c in piv.columns]
        for a, b in zip(cols, cols[1:]):
            v = piv[[a, b]].dropna()
            if len(v) > 20:
                cs.append(np.corrcoef(v[a], v[b])[0, 1])
        print(f"    {col:12s} lag-1 autocorrelation = {np.mean(cs):+.4f} "
              f"(per-pair {[round(c,3) for c in cs]})")
        res[f"auto_{col}"] = float(np.mean(cs))

    P.to_csv(OUT / "ha_roster_panel.csv", index=False)
    for m, s in S.items():
        s.to_csv(OUT / f"ha_player_splits_{m}.csv", index=False)
    (OUT / "ha_player.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {OUT/'ha_player.json'}")


if __name__ == "__main__":
    main()
