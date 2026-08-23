#!/usr/bin/env python3
"""D257 — DO TENDENCIES EXIST, AND HOW FAST DO THEY SWITCH?

The composition model reduces every player to ONE scalar (`talent` = DARKO_net)
weighted by minutes. A pass-first guard and a rim-runner with equal talent are
interchangeable in it. Sean wants tendencies represented, and raised the right
objection before any modelling: tendencies presumably move slowly (a new coach
changing a team's shot profile "takes a while to get down") -- but is that
actually true?

It matters because it decides whether a tendency is ESTIMABLE POINT-IN-TIME. A
slow-moving tendency can be measured from prior games and used tonight. A
tendency that resets on opening night cannot, and any PIT estimate of it would
be systematically stale exactly when it matters.

Tendencies are computable from `player_game_stats`, which carries shot-location
splits the model never touches: `rima/rimm`, `mida/midm`, `thra/thrm`, plus
`ast`, `tov`, `fta`. Five axes, all rates so team quality divides out:

    fg3_rate = fg3a / fga        spacing
    rim_rate = rima / fga        rim pressure
    ast_rate = ast  / fgm        playmaking / ball movement
    ftr      = fta  / fga        foul drawing
    tov_rate = tov  / poss       ball security

THREE QUESTIONS, in the order that decides whether the next step is worth taking:

  A  RELIABILITY. Split each team-season into odd and even games and correlate.
     If a tendency does not even agree with itself within a season, it is noise
     and nothing downstream can use it.

  B  PERSISTENCE. Correlate a team's tendency in season s with season s+1.
     High persistence across an offseason -- through roster churn and coaching
     changes -- is what "slow-moving" actually means.

  C  THE SWITCHING QUESTION, which is Sean's. For each team-season, does the
     PRIOR season predict the rest of this season better than THIS season's
     first 10 games do?

       prior wins  -> tendencies carry over; a new regime takes a while to land
       first-10 wins -> the new profile asserts itself immediately
       both strong -> tendency is stable and knowable from day one

     Reported per axis, because spacing and playmaking need not behave alike:
     a coach can order more threes on night one, while ball movement may take
     months of habit to change.

  D  BIG MOVERS. Restricted to team-seasons whose tendency moved most from the
     prior year -- the actual regime changes -- how many games until the new
     level is established? That is the JJ Redick question asked of the data
     rather than of intuition.
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

AXES = ["fg3_rate", "rim_rate", "ast_rate", "ftr", "tov_rate"]


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def clus(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    k = len(v)
    if k < 2:
        return np.nan, np.nan, np.nan, k
    se = v.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return v.mean(), v.mean() - tc * se, v.mean() + tc * se, k


def build():
    import duckdb
    con = duckdb.connect(str(ROOT / "data" / "nba.duckdb"), read_only=True)
    df = con.execute("""
        SELECT CAST(game_id AS VARCHAR) gid, team_id,
               sum(fga) fga, sum(fgm) fgm, sum(fg3a) fg3a, sum(fta) fta,
               sum(oreb) oreb, sum(tov) tov, sum(ast) ast,
               sum(COALESCE(rima,0)) rima, sum(COALESCE(mida,0)) mida
        FROM player_game_stats
        WHERE CAST(game_id AS VARCHAR) LIKE '002%'
        GROUP BY game_id, team_id""").df()
    con.close()
    df["gid"] = df.gid.str.zfill(10)
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz",
                    usecols=["game_id", "season", "game_date", "home", "away"])
    f["game_id"] = zf(f["game_id"]); f["game_date"] = pd.to_datetime(f.game_date)
    df = df.merge(f, left_on="gid", right_on="game_id", how="inner")
    # team_id -> abbreviation, from box order (away listed first)
    votes = {}
    for gid, sub in df.groupby("gid", sort=False):
        t = list(dict.fromkeys(sub.team_id))
        if len(t) != 2:
            continue
        r = sub.iloc[0]
        for tid, ab in zip(t, [r.away, r.home]):
            votes.setdefault(tid, {}).setdefault(ab, 0)
            votes[tid][ab] += 1
    df["ab"] = df.team_id.map({t: max(d, key=d.get) for t, d in votes.items()})
    df = df.dropna(subset=["ab"])
    poss = df.fga + 0.44 * df.fta - df.oreb + df.tov
    df["fg3_rate"] = df.fg3a / df.fga.replace(0, np.nan)
    df["rim_rate"] = df.rima / df.fga.replace(0, np.nan)
    df["ast_rate"] = df.ast / df.fgm.replace(0, np.nan)
    df["ftr"] = df.fta / df.fga.replace(0, np.nan)
    df["tov_rate"] = df.tov / poss.replace(0, np.nan)
    df = df.sort_values(["season", "ab", "game_date"])
    df["gn"] = df.groupby(["season", "ab"]).cumcount()
    return df


def main():
    d = build()
    have_rim = d.rim_rate.notna().mean()
    seasons = sorted(d.season.unique())
    print(f"{len(d):,} team-games, {len(seasons)} seasons "
          f"({seasons[0]}..{seasons[-1]})")
    print(f"rim_rate available on {100*have_rim:.1f}% of team-games "
          f"(shot-location split is not in every era)")
    out = {}

    # ---------------- A reliability -------------------------------
    print("\n" + "=" * 76)
    print("A  RELIABILITY — odd vs even games within a season")
    print("=" * 76)
    print(f"  {'axis':10} {'mean r':>9} {'CI':>22} {'seasons':>8}")
    for ax in AXES:
        rs = []
        for s, g in d.groupby("season"):
            a = g[g.gn % 2 == 0].groupby("ab")[ax].mean()
            b = g[g.gn % 2 == 1].groupby("ab")[ax].mean()
            j = pd.concat([a.rename("x"), b.rename("y")], axis=1).dropna()
            if len(j) >= 20:
                rs.append(float(np.corrcoef(j.x, j.y)[0, 1]))
        if len(rs) >= 3:
            m, lo, hi, k = clus(rs)
            print(f"  {ax:10} {m:+9.3f} [{lo:+.3f},{hi:+.3f}] {k:8}")
            out[f"A_{ax}"] = dict(r=float(m), ci=[float(lo), float(hi)], k=k)

    # ---------------- B across-season persistence -----------------
    print("\n" + "=" * 76)
    print("B  PERSISTENCE — season s vs season s+1 (through the offseason)")
    print("=" * 76)
    ts = d.groupby(["season", "ab"])[AXES].mean().reset_index()
    nxt = {seasons[i]: seasons[i + 1] for i in range(len(seasons) - 1)}
    ts["ns"] = ts.season.map(nxt)
    j = ts.merge(ts, left_on=["ns", "ab"], right_on=["season", "ab"],
                 suffixes=("", "_n"))
    print(f"  {'axis':10} {'mean r':>9} {'CI':>22}")
    for ax in AXES:
        rs = []
        for s, g in j.groupby("season"):
            gg = g[[ax, f"{ax}_n"]].dropna()
            if len(gg) >= 20:
                rs.append(float(np.corrcoef(gg[ax], gg[f"{ax}_n"])[0, 1]))
        if len(rs) >= 3:
            m, lo, hi, k = clus(rs)
            print(f"  {ax:10} {m:+9.3f} [{lo:+.3f},{hi:+.3f}]")
            out[f"B_{ax}"] = dict(r=float(m), ci=[float(lo), float(hi)], k=k)

    # ---------------- C the switching question --------------------
    print("\n" + "=" * 76)
    print("C  WHAT PREDICTS THE REST OF THE SEASON: last year, or the first 10?")
    print("=" * 76)
    print(f"  {'axis':10} {'prior season':>14} {'first 10 gms':>14} "
          f"{'both (R^2)':>12}   verdict")
    for ax in AXES:
        rp, rf, rb = [], [], []
        for s, g in d.groupby("season"):
            if s not in nxt.values() and s == seasons[0]:
                continue
            prev_s = {v: k for k, v in nxt.items()}.get(s)
            if prev_s is None:
                continue
            prior = ts[ts.season == prev_s].set_index("ab")[ax]
            first = g[g.gn < 10].groupby("ab")[ax].mean()
            rest = g[g.gn >= 10].groupby("ab")[ax].mean()
            jj = pd.concat([prior.rename("p"), first.rename("f"),
                            rest.rename("r")], axis=1).dropna()
            if len(jj) < 20:
                continue
            rp.append(float(np.corrcoef(jj.p, jj.r)[0, 1]))
            rf.append(float(np.corrcoef(jj.f, jj.r)[0, 1]))
            X = np.column_stack([np.ones(len(jj)), jj.p, jj.f])
            b = np.linalg.lstsq(X, jj.r.to_numpy(), rcond=None)[0]
            pred = X @ b
            rb.append(1 - ((jj.r - pred) ** 2).sum()
                      / ((jj.r - jj.r.mean()) ** 2).sum())
        if len(rp) >= 3:
            mp, lp, hp, _ = clus(rp)
            mf, lf, hf, _ = clus(rf)
            mb, *_ = clus(rb)
            v = ("carries over" if mp > mf + 0.05 else
                 "resets fast" if mf > mp + 0.05 else "both")
            print(f"  {ax:10} {mp:+9.3f}     {mf:+9.3f}     {mb:9.3f}   {v}")
            out[f"C_{ax}"] = dict(prior=float(mp), first10=float(mf),
                                  both_r2=float(mb), verdict=v)

    # ---------------- D big movers --------------------------------
    print("\n" + "=" * 76)
    print("D  BIG MOVERS — when a team's profile really changes, how fast?")
    print("=" * 76)
    print("  Team-seasons in the top decile of |this season - prior season|.")
    print("  Fraction of the FULL-SEASON move already present in games 1-10,")
    print("  11-20, 21-40 -- 1.00 means the new level landed immediately.\n")
    print(f"  {'axis':10} {'n':>5} {'g1-10':>8} {'g11-20':>8} {'g21-40':>8} "
          f"{'g41+':>8}")
    for ax in AXES:
        rows = []
        for s in seasons:
            prev_s = {v: k for k, v in nxt.items()}.get(s)
            if prev_s is None:
                continue
            prior = ts[ts.season == prev_s].set_index("ab")[ax]
            g = d[d.season == s]
            full = g.groupby("ab")[ax].mean()
            for ab in full.index:
                if ab not in prior.index or not np.isfinite(prior[ab]):
                    continue
                move = full[ab] - prior[ab]
                if not np.isfinite(move):
                    continue
                sub = g[g.ab == ab]
                w = {}
                for lab, m in (("g1", sub.gn < 10), ("g2", (sub.gn >= 10) & (sub.gn < 20)),
                               ("g3", (sub.gn >= 20) & (sub.gn < 40)), ("g4", sub.gn >= 40)):
                    val = sub[m][ax].mean()
                    w[lab] = val - prior[ab] if np.isfinite(val) else np.nan
                rows.append(dict(ab=ab, season=s, move=move, **w))
        r = pd.DataFrame(rows).dropna()
        if len(r) < 30:
            continue
        thr = r.move.abs().quantile(0.90)
        big = r[r.move.abs() >= thr]
        fr = [np.nanmedian(big[c] / big.move) for c in ("g1", "g2", "g3", "g4")]
        print(f"  {ax:10} {len(big):5} {fr[0]:8.2f} {fr[1]:8.2f} "
              f"{fr[2]:8.2f} {fr[3]:8.2f}")
        out[f"D_{ax}"] = dict(n=len(big), frac=[float(x) for x in fr])

    json.dump(out, open(ROOT / "data" / "d257_tendency.json", "w"), default=float)
    print("\nwrote data/d257_tendency.json")


if __name__ == "__main__":
    main()
