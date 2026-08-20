"""ABSENCE AUDIT — comp-leg DESIGN diagnostic and POWER calculation.

Runs BEFORE any endpoint scoring (GATE_POLICY_V2 §5.5 power floor).

KEY IDENTITY that makes this exact: changing `trail_min` changes the comp leg
ADDITIVELY and changes nothing else (comp has no fit; the ROSTER_DAYS
membership test reads `last_played`, not `trail_min`; the OUT sets are built
from `last_played` + the oracle; ff / sched / tank are untouched). Therefore
    margin_new = margin_old + 0.5 * (dstrength_home - dstrength_away)
and, since p_us = sigmoid(margin/7.2),
    p_new = sigmoid(logit(p_us) + dmargin/7.2)
exactly. This script reports only MARGIN-scale quantities (rms points) and the
implied MDE80 arithmetic; it does NOT score log loss.

Reads data/ab_comp_rows.csv.gz (from ab_comp_diag.py). Writes
data/ab_comp_design.json.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
SCALE = 7.2
W_COMP = 0.5

MISS_B = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, 10)]
GP_B = [(1, 2), (3, 5), (6, 9), (10, 14), (15, 19), (20, 10**6)]


def bucket_idx(v, buckets):
    for i, (lo, hi) in enumerate(buckets):
        if lo <= v <= hi:
            return i
    return len(buckets) - 1


def wf_table(df, key, buckets):
    """{season: [b per bucket]} using ONLY strictly-prior seasons (walk-forward).
    The earliest season has no prior data -> all-zero table (term inert)."""
    out = {}
    for s in SEASONS:
        prior = df[df.season < s]
        if len(prior) == 0:
            out[s] = [0.0] * len(buckets)
            continue
        bi = prior[key].map(lambda v: bucket_idx(v, buckets))
        m = prior.groupby(bi)["bias"].mean()
        out[s] = [float(m.get(i, 0.0)) for i in range(len(buckets))]
    return out


def apply_arm(df, key, buckets, tab):
    b = np.array([tab[s][bucket_idx(v, buckets)]
                  for s, v in zip(df.season, df[key])])
    return b


def team_margin(df, dmin_col):
    """dstrength per team-game from a per-player minutes CORRECTION (subtracted
    from trail_min): dstrength = -sum(talent * corr / 48)."""
    g = df.assign(ds=-df.talent * df[dmin_col] / 48.0)
    return g.groupby(["season", "game_id", "team_id"])["ds"].sum().reset_index()


def to_game(ts, home_map):
    """(season, game_id) -> dmargin = W_COMP*(ds_home - ds_away)."""
    ts = ts.copy()
    ts["is_home"] = [home_map.get((g, int(t))) for g, t in zip(ts.game_id, ts.team_id)]
    ts = ts[ts.is_home.notna()]
    piv = ts.pivot_table(index=["season", "game_id"], columns="is_home",
                         values="ds", aggfunc="sum").fillna(0.0)
    cols = list(piv.columns)
    h = piv[True] if True in cols else 0.0
    a = piv[False] if False in cols else 0.0
    piv["dmargin"] = W_COMP * (h - a)
    return piv.reset_index()[["season", "game_id", "dmargin"]]


def main():
    df = pd.read_csv("data/ab_comp_rows.csv.gz", dtype={"game_id": str})
    df["game_id"] = df["game_id"].str.zfill(10)
    from nbapred.db import connect
    con = connect(read_only=True)
    hm = {}
    for gid, tid, mu, ab in con.execute("""
        SELECT game_id, team_id, matchup, team_abbrev FROM nba_games
        WHERE game_id LIKE '002%' AND wl IS NOT NULL""").fetchall():
        host = mu.split("@")[-1].strip() if "@" in mu else mu.split("vs.")[0].strip()
        hm[(gid, int(tid))] = (ab == host)
    con.close()

    out = {}

    # ---------- ARM A: b(miss10s), walk-forward ------------------------------
    tabA = wf_table(df, "miss10s", MISS_B)
    df["corrA"] = apply_arm(df, "miss10s", MISS_B, tabA)
    # ---------- ARM B: b(gp), walk-forward (the literal D133 shape) ----------
    tabB = wf_table(df, "gp", GP_B)
    df["corrB"] = apply_arm(df, "gp", GP_B, tabB)
    # ---------- ARM C: minutes conservation (blind to TEAMMATES' absence) ----
    # rescale each available set so sum(trail_min) equals the walk-forward mean
    # realized sum for a set of that size; correction = trail_min*(1 - k_team)
    tm = df.groupby(["season", "game_id", "team_id"]).agg(
        st=("trail_min", "sum"), sr=("real_min", "sum"),
        n=("player_id", "size")).reset_index()
    ratios = {}
    for s in SEASONS:
        prior = tm[tm.season < s]
        if len(prior) == 0:
            ratios[s] = {}
            continue
        ratios[s] = prior.groupby("n").apply(
            lambda x: float(x.sr.sum() / x.st.sum())).to_dict()
    tm["k"] = [ratios[s].get(int(n), 1.0) for s, n in zip(tm.season, tm.n)]
    kmap = {(s, g, int(t)): k for s, g, t, k in
            zip(tm.season, tm.game_id, tm.team_id, tm.k)}
    df["corrC"] = [tm_ * (1.0 - kmap.get((s, g, int(t)), 1.0))
                   for s, g, t, tm_ in zip(df.season, df.game_id, df.team_id,
                                           df.trail_min)]
    # ---------- ORACLE: the ceiling (perfect minutes) ------------------------
    df["corrO"] = df["trail_min"] - df["real_min"]

    # ---------- margin footprints -------------------------------------------
    caps = pd.read_csv("data/capstone_pergame_d132.csv", dtype={"game_id": str})
    caps = caps[["season", "game_id", "y", "p_us", "p_mkt"]]
    res = {}
    for arm in ("A", "B", "C", "O"):
        gm = to_game(team_margin(df, f"corr{arm}"), hm)
        j = caps.merge(gm, on=["season", "game_id"], how="left")
        j["dmargin"] = j["dmargin"].fillna(0.0)
        dm = j["dmargin"].to_numpy()
        # implied per-game probability move (exact, no re-run needed)
        lg = np.log(j.p_us / (1 - j.p_us)) + dm / SCALE
        pn = 1.0 / (1.0 + np.exp(-lg))
        dp = pn - j.p_us.to_numpy()
        res[arm] = {
            "n_games": int(len(j)),
            "rms_dmargin_pts": round(float(np.sqrt((dm ** 2).mean())), 4),
            "mean_dmargin_pts": round(float(dm.mean()), 4),
            "max_abs_dmargin": round(float(np.abs(dm).max()), 4),
            "frac_games_moved": round(float((np.abs(dm) > 1e-9).mean()), 4),
            "rms_dp": round(float(np.sqrt((dp ** 2).mean())), 5),
            "max_abs_dp": round(float(np.abs(dp).max()), 5),
        }
        np.save(f"data/ab_dmargin_{arm}.npy", dm)
        print(arm, json.dumps(res[arm]))
    out["arms"] = res
    out["wf_table_A_miss10s"] = {k: [round(x, 4) for x in v] for k, v in tabA.items()}
    out["wf_table_B_gp"] = {k: [round(x, 4) for x in v] for k, v in tabB.items()}

    # ---------- POWER (§5.5): MDE80 = 2.802 * sd / sqrt(n) ------------------
    # sd of the PAIRED per-game log-loss delta is what matters. Bound it:
    # dloss = -[y*log(pn/p) + (1-y)*log((1-pn)/(1-p))]; compute its sd under
    # the actual y (this is a VARIANCE calculation, not the point estimate --
    # the point estimate is deliberately NOT printed here).
    y = caps.y.to_numpy(float)
    p = caps.p_us.to_numpy(float)
    pw = {}
    for arm in ("A", "B", "C", "O"):
        dm = np.load(f"data/ab_dmargin_{arm}.npy")
        lg = np.log(p / (1 - p)) + dm / SCALE
        pn = 1.0 / (1.0 + np.exp(-lg))
        d = -(y * np.log(pn / p) + (1 - y) * np.log((1 - pn) / (1 - p)))
        sd = float(d.std(ddof=1))
        pw[arm] = {"sd_paired_delta": round(sd, 5),
                   "MDE80_n6148": round(2.802 * sd / np.sqrt(len(d)), 5)}
        print("POWER", arm, pw[arm])
    out["power"] = pw
    json.dump(out, open("data/ab_comp_design.json", "w"), indent=1)
    print("wrote data/ab_comp_design.json")


if __name__ == "__main__":
    main()
