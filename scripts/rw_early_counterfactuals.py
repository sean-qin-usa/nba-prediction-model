"""REGIME A counterfactual constructions (EXPLORATORY — not gates; the top
candidate gets a separate pre-registered spec).

Built on the bitwise-verified dump (rw_early_decomp_pergame.csv):

  CF-C  drop the DEAD comp leg in week 1: p = sig((fm + sched)/S) on cm==0
        games (shipped blends a literal 0-margin at 50% weight there)
  CF-A  preseason-roster continuity in the carry weights (replaces the
        uniform DEFAULT 0.5556 at refit 1 and the partial first-5 at later
        carry refits): w = 0.3 * ps_cont_any(team)
  CF-B  preseason-roster composition for week 1: cm_ps = sum over the
        preseason-defined roster of DARKO(as-of) x prior-season trailing
        minutes / 48 (vets only; rookies contribute 0 like shipped comp)
  CF-D  preseason point-diff margin term: m += 0.10 * ps_pd_diff *
        max(0, 1 - gp_min/10)  (FIXED conservative coef, no fitting)
  Combos: A+C, A+B, A+B+D (the "October reconstruction")

Deltas = paired ll(shipped) - ll(variant), positive = variant better;
bootstrap CI 2000x. Read-only DB; results to data/rw_early_cf_results.json.
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
    if len(d) == 0:
        return dict(mean=0.0, lo=0.0, hi=0.0, n=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (B, len(d)))
    means = d[idx].mean(axis=1)
    return dict(mean=round(float(d.mean()), 5),
                lo=round(float(np.percentile(means, 2.5)), 5),
                hi=round(float(np.percentile(means, 97.5)), 5), n=int(len(d)))


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

# ---------- CF-A: refit carry FF with preseason continuity ---------------
fm_A = {}     # (season, refit_date) -> FourFactors
game_refit = {}
for s in SEASONS:
    rl = [r for r in refits if r["season"] == s]
    rdates = sorted(r["date"] for r in rl)
    active = {r["date"]: r["carry_active"] for r in rl}
    prev_rows = factor_game_rows(con, _prev_season(s), before=None)
    ps_cont = {ab2id[(s, t)]: float(skey.loc[(s, t), "ps_cont_any"])
               for t in sigs[sigs.season == s].team}
    mean_cont = np.mean(list(ps_cont.values()))
    for rd in rdates:
        if not active[rd]:
            continue
        w = [CARRY_W0 * ps_cont.get(x["tid"], mean_cont) for x in prev_rows]
        import datetime as dt
        d0 = dt.date.fromisoformat(rd)
        ff = FourFactors().fit(con, s, before=d0,
                               carry_rows=prev_rows, carry_weights=w)
        fm_A[(s, rd)] = ff
    # assign each dumped game to its refit date
    for i, r in df[df.season == s].iterrows():
        rd = max([d for d in rdates if d <= r.game_date])
        game_refit[(s, r.game_id)] = rd

df["refit_date"] = [game_refit[(s, g)] for s, g in zip(df.season, df.game_id)]
fmA_col = []
for r in df.itertuples():
    ff = fm_A.get((r.season, r.refit_date))
    if ff is None or not r.carry_active:
        fmA_col.append(r.fm)
    else:
        hid, aid = ab2id[(r.season, r.home)], ab2id[(r.season, r.away)]
        fmA_col.append(ff.margin_neutral(hid, aid))
df["fm_A"] = fmA_col

# ---------- CF-B: preseason-roster composition ---------------------------
cm_ps_map = {}   # (season, team_id) -> strength
scale_check = {}
for s in SEASONS:
    opener = OPENERS[s]
    # preseason participation this season (any 001 minutes)
    ps_ros = {}
    for t, p, m in con.execute("""
        SELECT s.team_id, s.player_id, sum(s.seconds)/60.0
        FROM player_game_stats s
        JOIN nba_games g ON g.game_id = s.game_id AND g.team_id = s.team_id
        WHERE g.season = ? AND s.game_id LIKE '001%' AND s.seconds > 0
        GROUP BY 1,2""", [s]).fetchall():
        ps_ros.setdefault(int(p), {})[int(t)] = float(m)
    assign = {p: max(d, key=d.get) for p, d in ps_ros.items()}
    # prior-season primary team for no-preseason vets
    prev_team = dict(con.execute("""
        SELECT s.player_id, arg_max(s.team_id, g.game_date)
        FROM player_game_stats s
        JOIN nba_games g ON g.game_id = s.game_id AND g.team_id = s.team_id
        WHERE g.season = ? AND s.game_id LIKE '002%' AND s.seconds > 0
        GROUP BY 1""", [_prev_season(s)]).fetchall())
    for p, t in prev_team.items():
        if int(p) not in assign:
            assign[int(p)] = int(t)     # assumed still on last season's team
    # trailing minutes: last-10 002 games with >=12 min before opener
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
    strength = {}
    for p, t in assign.items():
        tm = trail.get(p)
        if tm is None:
            continue                     # rookies / no prior NBA minutes
        strength[t] = strength.get(t, 0.0) + darko.get(p, 0.0) * tm / 48.0
    for t in strength:
        cm_ps_map[(s, t)] = strength[t]
    scale_check[s] = dict(
        sd=round(float(np.std(list(strength.values()))), 2),
        min=round(min(strength.values()), 2),
        max=round(max(strength.values()), 2))
con.close()

df["cm_ps"] = [cm_ps_map.get((s, ab2id[(s, h)]), 0.0)
               - cm_ps_map.get((s, ab2id[(s, a)]), 0.0)
               for s, h, a in zip(df.season, df.home, df.away)]

# ---------- CF-D: preseason point-diff term ------------------------------
df["ps_pd_d"] = [float(skey.loc[(s, h), "ps_pd"]) - float(skey.loc[(s, a), "ps_pd"])
                 for s, h, a in zip(df.season, df.home, df.away)]
df["gp_min"] = df[["gp_home", "gp_away"]].min(axis=1)
df["w_fade10"] = np.maximum(0.0, 1 - df.gp_min / 10.0)
BETA_PSPD = 0.10

# ---------- assemble variants --------------------------------------------
df["week1"] = (df.cm == 0).astype(int)
y = df.y.values
ll_ship = ll_vec(y, df.p_us)
ll_mkt = ll_vec(y, df.p_mkt)

variants = {}
# C: drop dead comp leg on week1 games
mC = np.where(df.week1 == 1, df.fm + df.sched, df.m_us)
variants["C_drop_dead_leg"] = mC
# A: ps-continuity carry (all carry games)
mA = np.where(df.carry_active == 1,
              0.5 * df.fm_A + 0.5 * df.cm + df.sched, df.m_us)
variants["A_ps_continuity_carry"] = mA
# A+C
mAC = np.where(df.week1 == 1, df.fm_A + df.sched, mA)
variants["AC_pscarry_plus_dropdead"] = mAC
# B: ps-roster comp on week1 (keep shipped fm)
mB = np.where(df.week1 == 1, 0.5 * df.fm + 0.5 * df.cm_ps + df.sched, df.m_us)
variants["B_psroster_comp_week1"] = mB
# A+B: ps carry + ps comp on week1, ps carry weeks 2-3
mAB = np.where(df.week1 == 1, 0.5 * df.fm_A + 0.5 * df.cm_ps + df.sched, mA)
variants["AB_october_reconstruction"] = mAB
# D alone on shipped margin
mD = df.m_us + BETA_PSPD * df.ps_pd_d * df.w_fade10
variants["D_pspd_term"] = mD
# A+B+D
mABD = mAB + BETA_PSPD * df.ps_pd_d * df.w_fade10
variants["ABD_full"] = mABD

res = dict(scale_check=scale_check,
           cm_sd_when_alive=round(float(df[df.cm != 0].cm.std()), 2),
           cm_ps_sd=round(float(df.cm_ps.std()), 2))
subsets = dict(week1=df.week1 == 1, carry=df.carry_active == 1,
               early_all=np.ones(len(df), bool),
               gp_min_lt10=df.gp_min < 10)
for name, mv in variants.items():
    pv = sig(np.asarray(mv) / SCALE)
    llv = ll_vec(y, pv)
    e = {}
    for sn, mask in subsets.items():
        m = np.asarray(mask)
        e[sn] = paired_ci((ll_ship - llv)[m])
        e[sn]["ll_var"] = round(float(llv[m].mean()), 4)
        e[sn]["ll_ship"] = round(float(ll_ship[m].mean()), 4)
        e[sn]["ll_mkt"] = round(float(ll_mkt[m].mean()), 4)
    res[name] = e

json.dump(res, open(REPO / "data" / "rw_early_cf_results.json", "w"), indent=1)
print(json.dumps(res, indent=1))
