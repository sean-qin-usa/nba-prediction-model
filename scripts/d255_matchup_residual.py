#!/usr/bin/env python3
"""D255 — IS THERE ANY TEAM-PAIR STRUCTURE IN THE RATINGS RESIDUAL, AND DOES IT
PERSIST?

The opponent-adjusted ratings are ADDITIVE and rank-1:

    ortg_ij = mu + off_i - def_j + home*is_home + home_dev_i*is_home

Each team is two numbers, so team i's offence is assumed to perform the same
amount better or worse against EVERY defence. Sean's question is whether a
persistent stylistic propensity — how a team lines up, how flexible a coach is
with matchups — breaks that assumption.

BEFORE PARAMETERISING AN INTERACTION, ESTABLISH THAT ONE EXISTS. A free
`off_i x def_j` interaction is 870 ordered pairs; on ~2.4 meetings per pair per
season it will fit noise perfectly and predict nothing. So this entry does not
fit an interaction at all. It asks the prior question:

    does the pair-level residual REPLICATE?

If team A genuinely matches up badly with team B's scheme, that is a property of
the pair and it must show up again — in their next meeting, and next season if
the rosters and coaches persist. If it does not replicate, there is nothing to
model and every interaction term downstream would be fitting noise.

THREE ESTIMANDS, all on the residual `r = ortg_observed - ortg_additive_fit`:

  R1  WITHIN-SEASON SPLIT-HALF. Split each ordered pair's meetings into first
      and later. Correlate the pair means. This is the reliability of a
      same-season pair effect.
  R2  ACROSS-SEASON PERSISTENCE. Correlate a pair's mean residual in season s
      with the same pair in season s+1. Stylistic propensity that survives an
      offseason should show here; a one-off does not.
  R3  VARIANCE DECOMPOSITION. How much of the residual variance could a perfect
      pair effect explain at most? This bounds the whole enterprise before any
      model is fitted.

THE MECHANICAL CAVEAT, STATED UP FRONT. `off_i` and `def_j` are estimated from
the same games that produce the residual, so each game pulls its own fit
slightly toward itself and induces a small negative bias in any residual
correlation. With ~82 games per team the leverage is ~1/82, so the bias is small
and NEGATIVE — it works against finding persistence, which is the safe
direction for a null result and the dangerous one for a positive.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from scipy import stats                                           # noqa: E402

RIDGE = 25.0


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def build_team_games():
    import duckdb
    con = duckdb.connect(str(ROOT / "data" / "nba.duckdb"), read_only=True)
    df = con.execute("""
        SELECT CAST(game_id AS VARCHAR) gid, team_id,
               sum(pts) pts, sum(fga) fga, sum(fta) fta,
               sum(oreb) oreb, sum(tov) tov, sum(fg3a) fg3a,
               sum(ast) ast, sum(dreb) dreb
        FROM player_game_stats
        WHERE CAST(game_id AS VARCHAR) LIKE '002%'
        GROUP BY game_id, team_id""").df()
    con.close()
    df["gid"] = df.gid.str.zfill(10)
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz",
                    usecols=["game_id", "season", "game_date", "home", "away"])
    f["game_id"] = zf(f["game_id"])
    f["game_date"] = pd.to_datetime(f["game_date"])
    df = df.merge(f, left_on="gid", right_on="game_id", how="inner")
    # opponent columns
    opp = df[["gid", "team_id", "pts", "fga", "fta", "oreb", "tov"]].copy()
    opp.columns = ["gid", "opp_id", "opp_pts", "opp_fga", "opp_fta",
                   "opp_oreb", "opp_tov"]
    m = df.merge(opp, on="gid")
    m = m[m.team_id != m.opp_id].copy()
    m["poss"] = m.fga + 0.44 * m.fta - m.oreb + m.tov
    m = m[m.poss >= 50]
    m["ortg"] = 100.0 * m.pts / m.poss
    # home identification, learned from box order (away listed first)
    votes = {}
    for gid, sub in m.groupby("gid", sort=False):
        tids = list(dict.fromkeys(sub.team_id))
        if len(tids) != 2:
            continue
        r0 = sub.iloc[0]
        for t, ab in zip(tids, [r0.away, r0.home]):
            votes.setdefault(t, {}).setdefault(ab, 0)
            votes[t][ab] += 1
    tid2ab = {t: max(d, key=d.get) for t, d in votes.items()}
    m["ab"] = m.team_id.map(tid2ab)
    m["opp_ab"] = m.opp_id.map(tid2ab)
    m = m.dropna(subset=["ab", "opp_ab"])
    m["is_home"] = (m.ab == m.home).astype(float)
    return m


def fit_additive(g):
    """Ridge fit of ortg ~ mu + off_i - def_j + home. Returns residuals."""
    teams = sorted(set(g.ab) | set(g.opp_ab))
    idx = {t: i for i, t in enumerate(teams)}
    T, n = len(teams), len(g)
    X = np.zeros((n, 2 + 2 * T))
    X[:, 0] = 1.0
    X[:, 1] = g.is_home.to_numpy(float)
    for r, (o, d) in enumerate(zip(g.ab, g.opp_ab)):
        X[r, 2 + idx[o]] = 1.0
        X[r, 2 + T + idx[d]] = -1.0
    y = g.ortg.to_numpy(float)
    P = np.zeros(2 + 2 * T); P[2:] = RIDGE
    beta = np.linalg.solve(X.T @ X + np.diag(P), X.T @ y)
    return y - X @ beta


def main():
    m = build_team_games()
    print(f"{len(m):,} team-games, {m.season.nunique()} seasons")
    m = m.sort_values(["season", "game_date"]).reset_index(drop=True)
    res = []
    for s, g in m.groupby("season"):
        g = g.copy()
        g["resid"] = fit_additive(g)
        g["meet"] = g.groupby(["ab", "opp_ab"]).cumcount()
        res.append(g)
    m = pd.concat(res)
    print(f"residual sd {m.resid.std():.3f} ortg points; "
          f"mean meetings per ordered pair per season "
          f"{m.groupby(['season','ab','opp_ab']).size().mean():.2f}")
    out = {}

    # ---------- R3 first: the ceiling -------------------------------
    print("\n" + "=" * 74)
    print("R3  HOW MUCH RESIDUAL VARIANCE COULD A PERFECT PAIR EFFECT EXPLAIN?")
    print("=" * 74)
    tot = m.resid.var()
    pair_mean = m.groupby(["season", "ab", "opp_ab"]).resid.transform("mean")
    print(f"  total residual variance          {tot:8.3f}")
    print(f"  variance of the pair means       {pair_mean.var():8.3f}  "
          f"({100*pair_mean.var()/tot:.1f}% of total)")
    print("  NOTE: with ~2.4 games per pair, most of that share is the sampling")
    print("  noise of a 2-3 game average, not a real pair effect. R1/R2 separate")
    print("  the two by asking whether it REPLICATES.")
    out["R3_pair_share"] = float(pair_mean.var() / tot)

    # ---------- R1 within-season split-half -------------------------
    print("\n" + "=" * 74)
    print("R1  WITHIN-SEASON: first meeting vs later meetings")
    print("=" * 74)
    rows = []
    for s, g in m.groupby("season"):
        a = g[g.meet == 0].groupby(["ab", "opp_ab"]).resid.mean()
        b = g[g.meet > 0].groupby(["ab", "opp_ab"]).resid.mean()
        j = pd.concat([a.rename("first"), b.rename("later")], axis=1).dropna()
        if len(j) > 50:
            rows.append(dict(season=s, n=len(j),
                             r=float(np.corrcoef(j["first"], j.later)[0, 1])))
    d1 = pd.DataFrame(rows)
    v = d1.r.to_numpy()
    se = v.std(ddof=1) / np.sqrt(len(v)); tc = stats.t.ppf(.975, len(v) - 1)
    print(d1.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    print(f"\n  mean r {v.mean():+.4f}  CI [{v.mean()-tc*se:+.4f}, "
          f"{v.mean()+tc*se:+.4f}]  positive in {int((v>0).sum())}/{len(v)}")
    out["R1"] = dict(mean_r=float(v.mean()),
                     ci=[float(v.mean()-tc*se), float(v.mean()+tc*se)],
                     k=len(v))

    # ---------- R2 across-season persistence ------------------------
    print("\n" + "=" * 74)
    print("R2  ACROSS-SEASON: pair residual in s vs the same pair in s+1")
    print("=" * 74)
    pm = m.groupby(["season", "ab", "opp_ab"]).resid.mean().reset_index()
    seasons = sorted(m.season.unique())
    nxt = {seasons[i]: seasons[i + 1] for i in range(len(seasons) - 1)}
    pm["nseason"] = pm.season.map(nxt)
    j = pm.merge(pm, left_on=["nseason", "ab", "opp_ab"],
                 right_on=["season", "ab", "opp_ab"], suffixes=("", "_n"))
    rows = []
    for s, g in j.groupby("season"):
        if len(g) > 50:
            rows.append(dict(season=s, n=len(g),
                             r=float(np.corrcoef(g.resid, g.resid_n)[0, 1])))
    d2 = pd.DataFrame(rows)
    v2 = d2.r.to_numpy()
    se2 = v2.std(ddof=1) / np.sqrt(len(v2)); tc2 = stats.t.ppf(.975, len(v2) - 1)
    print(d2.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    print(f"\n  mean r {v2.mean():+.4f}  CI [{v2.mean()-tc2*se2:+.4f}, "
          f"{v2.mean()+tc2*se2:+.4f}]  positive in {int((v2>0).sum())}/{len(v2)}")
    sig = (v2.mean() - tc2 * se2 > 0)
    print(f"  >>> {'PAIR EFFECTS PERSIST ACROSS SEASONS' if sig else 'NO ACROSS-SEASON PERSISTENCE — nothing for an interaction term to learn'}")
    out["R2"] = dict(mean_r=float(v2.mean()),
                     ci=[float(v2.mean()-tc2*se2), float(v2.mean()+tc2*se2)],
                     k=len(v2))

    json.dump(out, open(ROOT / "data" / "d255_matchup.json", "w"), default=float)
    print("\nwrote data/d255_matchup.json")


if __name__ == "__main__":
    main()
