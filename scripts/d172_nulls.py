#!/usr/bin/env python3
"""D172 — the nulls that the first two passes owed, plus one broken check fixed.

D164 measured this project's manufacturing capacity at +16.92 ROI points from
a 600-cell search, 100% artifact.  Any statistic of the form "splitting the
data explains more variance" inherits that problem: SPLITTING ALWAYS EXPLAINS
MORE.  Two such statistics were reported without a null and are corrected here:

  (a) "coach spell explains 1.5-2.0% more behaviour variance than team-season"
  (b) "owner spell explains 18.7% more youth-share variance than team"

The null for both: keep the spell STRUCTURE (same number of spells, same
lengths) and randomise WHERE the boundaries fall.  If the real split does no
better than a random split of identical shape, the extra variance is the split,
not the coach/owner.

FIXED: d172_ownership.py's "drop the transition season" robustness check
silently no-opped (it filtered on a column that is 0 for every row it kept, and
fell back to the unfiltered frame).  Redone properly here by comparing the last
FULL season of the old regime with the first FULL season of the new one.

READ-ONLY.  Writes data/d172_nulls.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import duckdb                                                     # noqa: E402
import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from scipy import stats as st                                     # noqa: E402

from nbapred.teams import FRANCHISE                               # noqa: E402

DATA = ROOT / "data"
RES: dict = {}
T0 = time.time()
NPERM = 2000


def ro(attempts=10, wait=60.0):
    for i in range(attempts):
        try:
            return duckdb.connect(str(DATA / "nba.duckdb"), read_only=True)
        except Exception as e:                                    # noqa: BLE001
            if ("lock" not in str(e).lower() and "held" not in str(e).lower()) \
               or i == attempts - 1:
                raise
            time.sleep(wait)


def cci(v, cl):
    d = pd.DataFrame({"v": np.asarray(v, float), "c": np.asarray(cl)}).dropna()
    if len(d) < 2:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, K=0, p=np.nan, n=len(d))
    m = d.groupby("c").v.mean()
    K = len(m)
    mu, sd = float(m.mean()), float(m.std(ddof=1))
    se = sd / np.sqrt(K)
    t = st.t.ppf(0.975, K - 1)
    ts = mu / se if se > 0 else np.nan
    p = 2 * (1 - st.t.cdf(abs(ts), K - 1)) if se > 0 else np.nan
    return dict(mean=mu, lo=mu - t * se, hi=mu + t * se, K=int(K),
                p=float(p) if p == p else np.nan, n=int(len(d)))


print("=" * 100)
print("D172 — PERMUTATION NULLS FOR THE TWO 'SPLITTING EXPLAINS MORE' CLAIMS")
print("=" * 100)

# ============================================== (a) COACH SPELLS vs BEHAVIOUR
print("\n" + "-" * 100)
print("(a) COACH SPELL vs a RANDOM split of the same shape — behaviours")
print("-" * 100)
B = pd.read_csv(DATA / "d172_behaviour.csv.gz")
BEHS = ["n_used", "hhi", "top8_share", "poss", "t5_turnover"]
rng = np.random.default_rng(20260804)

print(f"{'behaviour':<14}{'real extra':>12}{'null mean':>11}{'null p95':>10}"
      f"{'perm p':>9}   verdict")
coach_null = {}
for b in BEHS:
    d = B[["season", "team_abbrev", "coach", "gi", b]].dropna().copy()
    d["z"] = d[b] - d.groupby("season")[b].transform("mean")
    tot = d.z.var(ddof=1)
    bt = d.groupby(["season", "team_abbrev"]).z.mean().var(ddof=1)
    real = (d.groupby(["season", "team_abbrev", "coach"]).z.mean().var(ddof=1)
            - bt) / tot
    # null: for each team-season, keep the number of spells and their lengths,
    # but put the boundaries at random game indices.
    key = ["season", "team_abbrev"]
    grp = {k: v for k, v in d.groupby(key)}
    shape = {k: sorted(v.groupby("coach").size().tolist())
             for k, v in grp.items()}
    nulls = []
    for _ in range(200):
        lbl = []
        for k, v in grp.items():
            n, sizes = len(v), shape[k]
            if len(sizes) == 1:
                lbl.append(pd.Series(0, index=v.index))
                continue
            cuts = np.sort(rng.choice(np.arange(5, max(6, n - 5)),
                                      size=len(sizes) - 1, replace=False))
            lbl.append(pd.Series(np.searchsorted(cuts, np.arange(n)),
                                 index=v.index))
        d["_p"] = pd.concat(lbl)
        nulls.append((d.groupby(["season", "team_abbrev", "_p"]).z.mean()
                      .var(ddof=1) - bt) / tot)
    nulls = np.array(nulls)
    p = float((nulls >= real).mean())
    coach_null[b] = dict(real_extra=float(real), null_mean=float(nulls.mean()),
                         null_p95=float(np.quantile(nulls, .95)), perm_p=p)
    print(f"{b:<14}{real:>11.2%}{nulls.mean():>11.2%}"
          f"{np.quantile(nulls,.95):>10.2%}{p:>9.3f}   "
          f"{'REAL' if p < 0.05 else 'INDISTINGUISHABLE FROM A RANDOM SPLIT'}")
RES["coach_spell_variance_null"] = coach_null

# ================================================ (b) OWNER SPELLS vs YOUTH
print("\n" + "-" * 100)
print("(b) OWNER SPELL vs a RANDOM split of the same shape — youth minutes")
print("-" * 100)
con = ro()
youth = con.execute("""
    WITH s AS (
      SELECT g.season, g.team_abbrev, s.player_id, s.seconds
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, season, team_id, team_abbrev
                FROM nba_games WHERE game_id LIKE '002%') g
          ON g.game_id = s.game_id AND g.team_id = s.team_id
       WHERE s.seconds > 0),
    d AS (SELECT player_id, MIN(draft_year) dy FROM draft_history GROUP BY 1)
    SELECT s.season, s.team_abbrev, SUM(s.seconds) AS sec,
           SUM(CASE WHEN d.dy IS NOT NULL
                     AND CAST(SUBSTR(s.season,1,4) AS INT) - d.dy <= 2
                    THEN s.seconds ELSE 0 END) AS sec_youth
      FROM s LEFT JOIN d USING (player_id) GROUP BY 1,2""").fetchdf()
con.close()
youth["youth_share"] = youth.sec_youth / youth.sec
youth["team"] = youth.team_abbrev.map(lambda a: FRANCHISE.get(a, a))
OW = pd.read_csv(DATA / "d172_owner_spells.csv")
Y = youth.merge(OW, on=["season", "team"], how="inner").sort_values(
    ["team", "yr"])
Y["z"] = Y.youth_share - Y.groupby("season").youth_share.transform("mean")
tot = Y.z.var(ddof=1)
v_team = Y.groupby("team").z.mean().var(ddof=1)
real = (Y.groupby(["team", "owner"]).z.mean().var(ddof=1) - v_team) / tot

grp = {k: v for k, v in Y.groupby("team")}
shape = {k: v.owner.nunique() for k, v in grp.items()}
nulls = []
for _ in range(NPERM):
    lbl = []
    for k, v in grp.items():
        n, ns = len(v), shape[k]
        if ns == 1:
            lbl.append(pd.Series(0, index=v.index))
            continue
        cuts = np.sort(rng.choice(np.arange(1, n), size=ns - 1, replace=False))
        lbl.append(pd.Series(np.searchsorted(cuts, np.arange(n)), index=v.index))
    Y["_p"] = pd.concat(lbl)
    nulls.append((Y.groupby(["team", "_p"]).z.mean().var(ddof=1) - v_team) / tot)
nulls = np.array(nulls)
p = float((nulls >= real).mean())
print(f"  real OWNER marginal variance share : {real:>7.2%}")
print(f"  null (same spell counts, random boundaries), {NPERM} draws:")
print(f"     mean {nulls.mean():.2%}   p95 {np.quantile(nulls,.95):.2%}   "
      f"max {nulls.max():.2%}")
print(f"  PERMUTATION p = {p:.4f}   -> "
      f"{'REAL' if p < 0.05 else 'INDISTINGUISHABLE FROM A RANDOM SPLIT'}")
RES["owner_spell_variance_null"] = dict(
    real_extra=float(real), null_mean=float(nulls.mean()),
    null_p95=float(np.quantile(nulls, .95)), perm_p=p, n_perm=NPERM)

# ================================== (c) the ownership robustness check, FIXED
print("\n" + "-" * 100)
print("(c) OWNERSHIP CHANGE, TRANSITION SEASON PROPERLY EXCLUDED")
print("-" * 100)
A = pd.read_csv(DATA / "ats19_frame.csv.gz")
long = []
for side, sgn in (("home", 1), ("away", -1)):
    d = A[["season", side, "m_us", "open_margin", "margin_actual"]].rename(
        columns={side: "team"})
    d["r_model"] = sgn * (d.margin_actual - d.m_us)
    d["r_open"] = sgn * (d.margin_actual - d.open_margin)
    long.append(d[["season", "team", "r_model", "r_open"]])
L = pd.concat(long, ignore_index=True)
TS = L.groupby(["season", "team"]).agg(
    r_model=("r_model", "mean"), r_open=("r_open", "mean")).reset_index()
TS = TS.merge(OW[["season", "team", "owner", "yr", "tenure"]],
              on=["season", "team"], how="inner")

# last FULL season of the old regime = tenure -1 relative to change year;
# first FULL season of the new regime = tenure 1.  The tenure-0 season, in
# which the sale closed, is dropped from BOTH sides.
chg = OW[OW.tenure == 0][["team", "yr"]]
rows = []
for r in chg.itertuples():
    pre = TS[(TS.team == r.team) & (TS.yr == r.yr - 1)]
    post = TS[(TS.team == r.team) & (TS.yr == r.yr + 1)]
    if len(pre) and len(post):
        rows.append(dict(team=r.team, yr=r.yr, season=post.season.iloc[0],
                         d_model=float(post.r_model.iloc[0] - pre.r_model.iloc[0]),
                         d_open=float(post.r_open.iloc[0] - pre.r_open.iloc[0])))
D = pd.DataFrame(rows)
print(f"  ownership changes with BOTH a clean pre and a clean post season: "
      f"{len(D)}")
for lbl, c in (("vs OUR MODEL", "d_model"), ("vs the MARKET", "d_open")):
    if len(D) >= 3:
        r = cci(D[c], D.season)
        RES.setdefault("owner_change_clean", {})[lbl] = r
        print(f"    {lbl:<16}{r['mean']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] "
              f"p={r['p']:.4f} n={r['n']} K={r['K']}")

RES["elapsed_s"] = round(time.time() - T0, 1)
(DATA / "d172_nulls.json").write_text(json.dumps(RES, indent=1, default=str))
print(f"\nWROTE data/d172_nulls.json ({RES['elapsed_s']}s)")
