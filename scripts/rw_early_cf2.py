"""REGIME A counterfactuals round 2 (EXPLORATORY):
 - margin-MSE endpoint (better powered than ll for margin-level fixes)
 - rookie-penalty grid: m += gamma * rookie_ps_share_d * fade10
 - ps_pd beta grid (reconciling the S5 regression vs CF-D ll result)
 - carry-window power numbers for the pre-registered gate spec
Reuses rw_early_cf machinery outputs; read-only.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from nbapred.db import connect
from nbapred.model.four_factors import FourFactors, factor_game_rows
from nbapred.model.production import CARRY_W0, SCALE, _prev_season

SEASONS = ("2023-24", "2024-25", "2025-26")
OPENERS = {"2023-24": "2023-10-24", "2024-25": "2024-10-22",
           "2025-26": "2025-10-21"}


def ll_vec(y, p):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_ci(d, B=2000, seed=42):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (B, len(d)))
    means = d[idx].mean(axis=1)
    return dict(mean=round(float(d.mean()), 5),
                lo=round(float(np.percentile(means, 2.5)), 5),
                hi=round(float(np.percentile(means, 97.5)), 5), n=int(len(d)),
                sd=round(float(d.std()), 4))


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


df = pd.read_csv(REPO / "data" / "rw_early_decomp_pergame.csv")
sigs = pd.read_csv(REPO / "data" / "rw_early_signals.csv")
refits = json.load(open(REPO / "data" / "rw_early_decomp_refits.json"))
skey = sigs.set_index(["season", "team"])

con = connect(read_only=True)
ab2id = {}
for s in SEASONS:
    for t, a in con.execute(
            "SELECT DISTINCT team_id, team_abbrev FROM nba_games WHERE season=?",
            [s]).fetchall():
        ab2id[(s, a)] = int(t)

# rebuild fm_A and cm_ps identically to rw_early_counterfactuals.py ---------
fm_A = {}
game_refit = {}
for s in SEASONS:
    rl = [r for r in refits if r["season"] == s]
    rdates = sorted(r["date"] for r in rl)
    active = {r["date"]: r["carry_active"] for r in rl}
    prev_rows = factor_game_rows(con, _prev_season(s), before=None)
    ps_cont = {ab2id[(s, t)]: float(skey.loc[(s, t), "ps_cont_any"])
               for t in sigs[sigs.season == s].team}
    mean_cont = float(np.mean(list(ps_cont.values())))
    import datetime as dt
    for rd in rdates:
        if not active[rd]:
            continue
        w = [CARRY_W0 * ps_cont.get(x["tid"], mean_cont) for x in prev_rows]
        fm_A[(s, rd)] = FourFactors().fit(
            con, s, before=dt.date.fromisoformat(rd),
            carry_rows=prev_rows, carry_weights=w)
    for i, r in df[df.season == s].iterrows():
        game_refit[(s, r.game_id)] = max(d for d in rdates if d <= r.game_date)
df["refit_date"] = [game_refit[(s, g)] for s, g in zip(df.season, df.game_id)]
fmA_col = []
for r in df.itertuples():
    ff = fm_A.get((r.season, r.refit_date))
    if ff is None or not r.carry_active:
        fmA_col.append(r.fm)
    else:
        fmA_col.append(ff.margin_neutral(ab2id[(r.season, r.home)],
                                         ab2id[(r.season, r.away)]))
df["fm_A"] = fmA_col

cm_ps_map = {}
for s in SEASONS:
    opener = OPENERS[s]
    ps_ros = {}
    for t, p, m in con.execute("""
        SELECT s.team_id, s.player_id, sum(s.seconds)/60.0
        FROM player_game_stats s
        JOIN nba_games g ON g.game_id = s.game_id AND g.team_id = s.team_id
        WHERE g.season = ? AND s.game_id LIKE '001%' AND s.seconds > 0
        GROUP BY 1,2""", [s]).fetchall():
        ps_ros.setdefault(int(p), {})[int(t)] = float(m)
    assign = {p: max(d, key=d.get) for p, d in ps_ros.items()}
    prev_team = dict(con.execute("""
        SELECT s.player_id, arg_max(s.team_id, g.game_date)
        FROM player_game_stats s
        JOIN nba_games g ON g.game_id = s.game_id AND g.team_id = s.team_id
        WHERE g.season = ? AND s.game_id LIKE '002%' AND s.seconds > 0
        GROUP BY 1""", [_prev_season(s)]).fetchall())
    for p, t in prev_team.items():
        if int(p) not in assign:
            assign[int(p)] = int(t)
    trail = dict(con.execute("""
        WITH pg AS (
          SELECT s.player_id, s.seconds/60.0 m,
                 row_number() OVER (PARTITION BY s.player_id
                                    ORDER BY g.game_date DESC) rn
          FROM player_game_stats s
          JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g
            USING (game_id)
          WHERE s.game_id LIKE '002%' AND s.seconds >= 720 AND g.game_date < ?)
        SELECT player_id, avg(m) FROM pg WHERE rn <= 10 GROUP BY 1""",
        [opener]).fetchall())
    darko = dict(con.execute("""
        SELECT player_id, dpm FROM (
          SELECT player_id, dpm,
                 row_number() OVER (PARTITION BY player_id
                                    ORDER BY date DESC) rn
          FROM darko_history WHERE date < ?) WHERE rn = 1""",
        [opener]).fetchall())
    for p, t in assign.items():
        tm = trail.get(p)
        if tm is None:
            continue
        cm_ps_map[(s, t)] = cm_ps_map.get((s, t), 0.0) + darko.get(p, 0.0) * tm / 48.0
con.close()

df["cm_ps"] = [cm_ps_map.get((s, ab2id[(s, h)]), 0.0)
               - cm_ps_map.get((s, ab2id[(s, a)]), 0.0)
               for s, h, a in zip(df.season, df.home, df.away)]
for c in ("ps_pd", "rookie_ps_share"):
    df[f"{c}_d"] = [float(skey.loc[(s, h), c]) - float(skey.loc[(s, a), c])
                    for s, h, a in zip(df.season, df.home, df.away)]
df["gp_min"] = df[["gp_home", "gp_away"]].min(axis=1)
df["fade10"] = np.maximum(0.0, 1 - df.gp_min / 10.0)
df["week1"] = (df.cm == 0).astype(int)

y = df.y.values
am = df.am.values.astype(float)
m_ship = df.m_us.values
mAB = np.where(df.week1 == 1, 0.5 * df.fm_A + 0.5 * df.cm_ps + df.sched,
               np.where(df.carry_active == 1,
                        0.5 * df.fm_A + 0.5 * df.cm + df.sched, m_ship))

out = {}
subsets = dict(week1=df.week1.values == 1, carry=df.carry_active.values == 1,
               gp_min_lt10=(df.gp_min < 10).values,
               early_all=np.ones(len(df), bool))

# 1) margin-MSE endpoint for AB and C
mC = np.where(df.week1 == 1, df.fm + df.sched, m_ship)
for nm, mv in (("AB", mAB), ("C", mC)):
    e = {}
    for sn, mask in subsets.items():
        d = ((am - m_ship) ** 2 - (am - np.asarray(mv)) ** 2)[mask]
        e[sn] = paired_ci(d)   # positive = variant closer in points^2
    out[f"MSE_{nm}_vs_ship"] = e
# market MSE reference
mktm = SCALE * np.log(np.clip(df.p_mkt, 1e-12, 1 - 1e-12)
                      / np.clip(1 - df.p_mkt, 1e-12, 1))
out["MSE_levels"] = {sn: dict(
    ship=round(float(((am - m_ship) ** 2)[m].mean()), 1),
    AB=round(float(((am - mAB) ** 2)[m].mean()), 1),
    mkt=round(float(((am - mktm) ** 2)[m].mean()), 1), n=int(m.sum()))
    for sn, m in subsets.items()}

# 2) rookie gamma grid on shipped margins
ll_ship = ll_vec(y, sig(m_ship / SCALE))
for gam in (-4.0, -8.0, -12.0):
    mv = m_ship + gam * df.rookie_ps_share_d * df.fade10
    e = {sn: paired_ci((ll_ship - ll_vec(y, sig(mv / SCALE)))[m])
         for sn, m in subsets.items()}
    out[f"rookie_gamma={gam}"] = {k: dict(mean=v["mean"], lo=v["lo"], hi=v["hi"])
                                  for k, v in e.items()}

# 3) ps_pd beta grid on shipped margins
for b in (0.05, 0.10, 0.15):
    mv = m_ship + b * df.ps_pd_d * df.fade10
    e = {sn: paired_ci((ll_ship - ll_vec(y, sig(mv / SCALE)))[m])
         for sn, m in subsets.items()}
    out[f"pspd_beta={b}"] = {k: dict(mean=v["mean"], lo=v["lo"], hi=v["hi"])
                             for k, v in e.items()}
# ps_pd on top of AB
ll_AB = ll_vec(y, sig(mAB / SCALE))
for b in (0.05, 0.10):
    mv = mAB + b * df.ps_pd_d * df.fade10
    e = {sn: paired_ci((ll_AB - ll_vec(y, sig(mv / SCALE)))[m])
         for sn, m in subsets.items()}
    out[f"AB+pspd_beta={b}_vs_AB"] = {k: dict(mean=v["mean"], lo=v["lo"],
                                              hi=v["hi"])
                                      for k, v in e.items()}
# rookie on top of AB
for gam in (-4.0, -8.0):
    mv = mAB + gam * df.rookie_ps_share_d * df.fade10
    e = {sn: paired_ci((ll_AB - ll_vec(y, sig(mv / SCALE)))[m])
         for sn, m in subsets.items()}
    out[f"AB+rookie_gamma={gam}_vs_AB"] = {k: dict(mean=v["mean"], lo=v["lo"],
                                                   hi=v["hi"])
                                           for k, v in e.items()}

# 4) power numbers for the gate spec (ll endpoint, AB vs ship)
d_carry = (ll_ship - ll_AB)[df.carry_active.values == 1]
out["gate_power"] = dict(
    carry_n=int(len(d_carry)), diff_sd=round(float(d_carry.std()), 4),
    mde80_at_n=round(2.802 * float(d_carry.std()) / np.sqrt(len(d_carry)), 5),
    note="paired ll diffs, carry window; MSE endpoint power in MSE section")

# 5) per-season AB ll deltas (carry window)
for s in SEASONS:
    m = (df.season == s).values & (df.carry_active.values == 1)
    out[f"AB_carry_{s}"] = paired_ci((ll_ship - ll_AB)[m])

json.dump(out, open(REPO / "data" / "rw_early_cf2_results.json", "w"), indent=1)
print(json.dumps(out, indent=1))
