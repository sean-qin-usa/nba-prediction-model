#!/usr/bin/env python3
"""D172 — stress the ONE result that survived, and BH the whole family.

d172_nulls.py (c) found the only non-null in the study: in the first FULL
season under a new control owner, a team's margin residual against the MARKET
falls by -1.310 pts [-2.069,-0.552] p=0.0025 (n=27, K=14).

Before that is allowed anywhere near a gate it has to survive the obvious
alternative: teams are SOLD after a good run, so year+1 vs year-1 could be
pure regression to the mean on a selected sample.  Tested here by
  (1) reporting the LEVELS, not just the delta;
  (2) a placebo of random (team, year) pairs at the same 2-season spacing,
      matched on the pre-season residual;
  (3) leave-one-event-out and leave-one-season-out;
  (4) Benjamini-Hochberg across the whole D172 family.

READ-ONLY.  Writes data/d172_stress.json.
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

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from scipy import stats as st                                     # noqa: E402

from nbapred.teams import FRANCHISE                               # noqa: E402

DATA = ROOT / "data"
RES: dict = {}
T0 = time.time()


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
    n=("r_model", "size"), r_model=("r_model", "mean"),
    r_open=("r_open", "mean")).reset_index()
TS["yr"] = TS.season.str[:4].astype(int)
OW = pd.read_csv(DATA / "d172_owner_spells.csv")

print("=" * 100)
print("D172 — STRESSING THE ONE SURVIVOR: the first full season under a new owner")
print("=" * 100)

chg = OW[OW.tenure == 0][["team", "yr"]]
rows = []
for r in chg.itertuples():
    pre = TS[(TS.team == r.team) & (TS.yr == r.yr - 1)]
    post = TS[(TS.team == r.team) & (TS.yr == r.yr + 1)]
    if len(pre) and len(post):
        rows.append(dict(team=r.team, yr=int(r.yr), season=post.season.iloc[0],
                         pre_model=float(pre.r_model.iloc[0]),
                         post_model=float(post.r_model.iloc[0]),
                         pre_open=float(pre.r_open.iloc[0]),
                         post_open=float(post.r_open.iloc[0])))
D = pd.DataFrame(rows)
D["d_open"] = D.post_open - D.pre_open
D["d_model"] = D.post_model - D.pre_model

print("\n1. LEVELS, not just the delta")
print(f"   events with clean pre and post: {len(D)}")
for lbl, a, b in (("vs the MARKET", "pre_open", "post_open"),
                  ("vs OUR MODEL", "pre_model", "post_model")):
    ra, rb = cci(D[a], D.season), cci(D[b], D.season)
    print(f"   {lbl:<16} PRE  {ra['mean']:+.3f} [{ra['lo']:+.3f},{ra['hi']:+.3f}]"
          f" p={ra['p']:.4f}")
    print(f"   {'':<16} POST {rb['mean']:+.3f} [{rb['lo']:+.3f},{rb['hi']:+.3f}]"
          f" p={rb['p']:.4f}")
    RES.setdefault("levels", {})[lbl] = dict(pre=ra, post=rb)
print("\n   READ: if PRE is strongly POSITIVE and POST is ~0, the 'effect' is\n"
      "   regression to the mean on a sample selected for having just run hot.")

print("\n2. MATCHED PLACEBO — random (team, year) at the same 2-season spacing,")
print("   matched on the pre-season residual (calliper 1.0 pt)")
rng = np.random.default_rng(20260804)
pool = []
evset = set(zip(D.team, D.yr))
for t, g in TS.groupby("team"):
    g = g.set_index("yr")
    for y in g.index:
        if (t, y + 1) in evset:
            continue
        if (y - 1) in g.index and (y + 1) in g.index:
            pool.append(dict(team=t, yr=y, season=g.loc[y + 1, "season"],
                             pre_open=g.loc[y - 1, "r_open"],
                             post_open=g.loc[y + 1, "r_open"],
                             pre_model=g.loc[y - 1, "r_model"],
                             post_model=g.loc[y + 1, "r_model"]))
POOL = pd.DataFrame(pool)
POOL["d_open"] = POOL.post_open - POOL.pre_open
POOL["d_model"] = POOL.post_model - POOL.pre_model
print(f"   placebo pool: {len(POOL)} (team, year) cells")
CAL = 1.0
mm = []
for r in D.itertuples():
    c = POOL[POOL.pre_open.sub(r.pre_open).abs() <= CAL]
    if len(c):
        mm.append(dict(season=r.season, ev=r.d_open, ct=c.d_open.mean(),
                       ev_m=r.d_model, ct_m=c.d_model.mean(), n_ctrl=len(c)))
MM = pd.DataFrame(mm)
for lbl, e, c in (("vs the MARKET", "ev", "ct"), ("vs OUR MODEL", "ev_m", "ct_m")):
    r = cci(MM[e] - MM[c], MM.season)
    RES.setdefault("placebo_did", {})[lbl] = r
    print(f"   {lbl:<16} event {MM[e].mean():+.3f}  control "
          f"{MM[c].mean():+.3f}  DiD {r['mean']:+.3f} "
          f"[{r['lo']:+.3f},{r['hi']:+.3f}] p={r['p']:.4f} n={r['n']}")

print("\n3. LEAVE-ONE-OUT")
base = cci(D.d_open, D.season)
loo_ev = []
for i in range(len(D)):
    loo_ev.append(cci(D.drop(D.index[i]).d_open, D.drop(D.index[i]).season)["mean"])
loo_s = {}
for s in sorted(D.season.unique()):
    d = D[D.season != s]
    if d.season.nunique() >= 2:
        loo_s[s] = cci(d.d_open, d.season)["mean"]
print(f"   full sample          {base['mean']:+.3f}")
print(f"   leave-one-EVENT-out  min {min(loo_ev):+.3f}  max {max(loo_ev):+.3f}")
print(f"   leave-one-SEASON-out min {min(loo_s.values()):+.3f}  "
      f"max {max(loo_s.values()):+.3f}")
worst = max(loo_s, key=lambda k: loo_s[k])
print(f"   most influential season: {worst} (dropping it -> {loo_s[worst]:+.3f})")
RES["loo"] = dict(full=base["mean"], ev_min=min(loo_ev), ev_max=max(loo_ev),
                  season_min=min(loo_s.values()), season_max=max(loo_s.values()),
                  most_influential_season=worst)
print("\n   the events themselves:")
print(D.sort_values("d_open")[["team", "yr", "pre_open", "post_open", "d_open"]]
      .to_string(index=False))

# ================================================================ 4. BH
print("\n" + "-" * 100)
print("4. BENJAMINI-HOCHBERG ACROSS THE WHOLE D172 FAMILY")
print("-" * 100)
fam = []


def add(name, p):
    if p == p:
        fam.append((name, float(p)))


meas = json.loads((DATA / "d172_measure.json").read_text())
dec = json.loads((DATA / "d172_decompose.json").read_text())
nul = json.loads((DATA / "d172_nulls.json").read_text())
own = json.loads((DATA / "d172_ownership.json").read_text())

for k, v in meas.get("behaviour_at_change", {}).items():
    add(f"coach: behaviour jump {k}", v["p"])
for k, v in dec.get("post_vs_zero", {}).items():
    add(f"coach: {k} vs 0", v["p"])
for k, v in dec.get("matched_did", {}).items():
    add(f"coach: matched DiD {k}", v["did"]["p"])
for k, v in dec.get("persistence_decomposed", {}).items():
    add(f"coach: lag1 {k}", v["p"])
for k, v in dec.get("coach_travels", {}).items():
    add(f"coach: travels {k}", v["p"])
for k, v in nul.get("coach_spell_variance_null", {}).items():
    add(f"coach: spell-variance perm {k}", v["perm_p"])
add("owner: spell-variance perm youth",
    nul["owner_spell_variance_null"]["perm_p"])
for k, v in nul.get("owner_change_clean", {}).items():
    add(f"owner: clean change {k}", v["p"])
for k, v in own.get("ownership_change_event", {}).items():
    add(f"owner: yoy change {k}", v["new_owner"]["p"])
add("owner: youth share at change",
    own["youth_at_owner_change"]["new_owner"]["p"])
for k, v in RES.get("placebo_did", {}).items():
    add(f"owner: placebo DiD {k}", v["p"])

F = pd.DataFrame(fam, columns=["test", "p"]).sort_values("p").reset_index(drop=True)
m = len(F)
F["rank"] = F.index + 1
F["bh_crit"] = F["rank"] / m * 0.05
F["pass_bh"] = F.p <= F.bh_crit
# step-up: everything up to the largest passing rank
last = F[F.pass_bh].index.max() if F.pass_bh.any() else -1
F["reject"] = F.index <= last if last >= 0 else False
print(f"family size m = {m}, FDR 0.05\n")
print(F[["test", "p", "bh_crit", "reject"]].to_string(index=False))
RES["bh"] = F.astype(str).to_dict("records")
RES["bh_rejected"] = F[F.reject].test.tolist()
print(f"\nSURVIVES BH: {F[F.reject].test.tolist()}")

RES["elapsed_s"] = round(time.time() - T0, 1)
(DATA / "d172_stress.json").write_text(json.dumps(RES, indent=1, default=str))
D.to_csv(DATA / "d172_owner_events.csv", index=False)
print(f"\nWROTE data/d172_stress.json ({RES['elapsed_s']}s)")
