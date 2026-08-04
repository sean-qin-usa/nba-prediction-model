"""INCREMENTAL GATE — rookie draft-slot prior inside the October composition
bridge (D84-A candidate #4, part 2). ONE config, no sweeps.

CONTROL  = the D84-A bridge exactly as pre-registered in rw_early_v1_gate.py
           (cm_ps over the 001-preseason roster, DARKO(as-of) x trail/48,
           oracle-outs = played-set filter; rookies contribute 0).
VARIANT  = identical, plus each CURRENT-CLASS drafted rookie on the 001 roster
           (no prior-season 002 trail) contributes
               curve_dpm(pick; walk-forward fit, classes 2017..Y-1)
               x ps_mpg(player's avg 001 minutes per appearance) / 48
           subject to the SAME outs convention. Undrafted rookies and
           stash-class debuts stay 0 (no slot prior / out of spec).
MINUTES DECISION (ex-ante, from PIT-clean class-2022 data in
data/rookie_draft_curve.json): raw preseason mpg unless that check showed a
systematic scale break — see MIN_MAP below, set from that JSON's c0/c1 only.

ENDPOINT (pre-registered): paired bootstrap 2000x 95% CI of
ll(control)-ll(variant) on the ACTIVE window (cm==0), 3 seasons pooled.
PASS = CI > 0. Secondary: per-season, vs-shipped, margin-MSE, mechanism stats.
Read-only DB. Writes data/rookie_bridge_gate.json.
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
from nbapred.model.production import SCALE, _prev_season

SEASONS = ("2023-24", "2024-25", "2025-26")


def ll_vec(y, p):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_ci(d, B=2000, seed=42):
    d = np.asarray(d, float)
    if len(d) == 0:
        return dict(mean=0.0, lo=0.0, hi=0.0, n=0)
    rng = np.random.default_rng(seed)
    mm = d[rng.integers(0, len(d), (B, len(d)))].mean(axis=1)
    return dict(mean=round(float(d.mean()), 5),
                lo=round(float(np.percentile(mm, 2.5)), 5),
                hi=round(float(np.percentile(mm, 97.5)), 5), n=int(len(d)))


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def main():
    curve = json.load(open(REPO / "data" / "rookie_draft_curve.json"))
    wf = {int(k): v for k, v in curve["walk_forward"].items()}
    # MIN_MAP decision (per the pre-declared rule in the docstring): the
    # class-2022 check DID show a systematic break — reg mpg = -4.24 + 1.089*ps
    # (corr .747, mean 16.7 -> 13.9): raw ps mpg overstates fringe-rookie
    # minutes. Use the fitted map, clipped at 0 (a negative implied-minute
    # times a negative dpm must not become a positive contribution).
    mmap = curve["minutes_map_class2022"]
    MIN_MAP = lambda x: max(0.0, mmap["c0"] + mmap["c1"] * x)  # noqa: E731

    df = pd.read_csv(REPO / "data" / "rw_early_decomp_pergame.csv",
                     dtype={"game_id": str})
    df["game_id"] = df.game_id.str.zfill(10)
    con = connect(read_only=True)
    ab2id, openers = {}, {}
    for s in SEASONS:
        for t, a in con.execute(
                "SELECT DISTINCT team_id, team_abbrev FROM nba_games "
                "WHERE season=?", [s]).fetchall():
            ab2id[(s, a)] = int(t)
        openers[s] = str(con.execute(
            "SELECT min(game_date) FROM nba_games WHERE season=? AND "
            "game_id LIKE '002%'", [s]).fetchone()[0])[:10]

    P, PR = {}, {}         # control / rookie-extended: season -> {p: (t, contrib)}
    mech_ros = {}
    for s in SEASONS:
        opener = openers[s]
        yr = int(s[:4])
        ps_ros, ps_gp = {}, {}
        for t, p, m, gp in con.execute("""
            SELECT s.team_id, s.player_id, sum(s.seconds)/60.0, count(*)
            FROM player_game_stats s
            JOIN nba_games g ON g.game_id=s.game_id AND g.team_id=s.team_id
            WHERE g.season=? AND s.game_id LIKE '001%' AND s.seconds>0
            GROUP BY 1,2""", [s]).fetchall():
            ps_ros.setdefault(int(p), {})[int(t)] = float(m)
            ps_gp[(int(p), int(t))] = int(gp)
        assign = {p: max(d, key=d.get) for p, d in ps_ros.items()}
        for p, t in con.execute("""
            SELECT s.player_id, arg_max(s.team_id, g.game_date)
            FROM player_game_stats s
            JOIN nba_games g ON g.game_id=s.game_id AND g.team_id=s.team_id
            WHERE g.season=? AND s.game_id LIKE '002%' AND s.seconds>0
            GROUP BY 1""", [_prev_season(s)]).fetchall():
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
              WHERE s.game_id LIKE '002%' AND s.seconds>=720
              AND g.game_date < ?)
            SELECT player_id, avg(m) FROM pg WHERE rn<=10 GROUP BY 1""",
            [opener]).fetchall())
        darko = dict(con.execute("""
            SELECT player_id, dpm FROM (
              SELECT player_id, dpm,
                     row_number() OVER (PARTITION BY player_id
                                        ORDER BY date DESC) rn
              FROM darko_history WHERE date < ?) WHERE rn=1""",
            [opener]).fetchall())
        picks = dict(con.execute(
            "SELECT player_id, overall_pick FROM draft_history "
            "WHERE draft_year=? AND overall_pick IS NOT NULL", [yr]).fetchall())
        P[s] = {p: (t, darko.get(p, 0.0) * trail[p] / 48.0)
                for p, t in assign.items() if p in trail}
        a, b = wf[yr]["a"], wf[yr]["b"]
        PR[s] = dict(P[s])
        n_rook, contribs = 0, []
        for p, t in assign.items():
            if p in trail or p not in picks or p not in ps_ros:
                continue
            mpg = MIN_MAP(ps_ros[p][t] / max(ps_gp[(p, t)], 1))
            c = (a + b * np.log(picks[p])) * mpg / 48.0
            PR[s][p] = (t, c)
            n_rook += 1
            contribs.append(c)
        mech_ros[s] = dict(rookies_on_rosters=n_rook,
                           mean_contrib=round(float(np.mean(contribs)), 3),
                           min_contrib=round(float(np.min(contribs)), 3))
    played = {}
    for g, t, p in con.execute(
            "SELECT game_id, team_id, player_id FROM player_game_stats "
            "WHERE game_id LIKE '002%' AND seconds>0").fetchall():
        played.setdefault((g, int(t)), set()).add(int(p))
    con.close()

    active = (df.cm == 0)

    def cm_col(PP):
        out, nrk = [], []
        for r in df.itertuples():
            if r.cm != 0:
                out.append(0.0); nrk.append(0)
                continue
            hid, aid = ab2id[(r.season, r.home)], ab2id[(r.season, r.away)]
            sH = sum(c for p, (t, c) in PP[r.season].items()
                     if t == hid and p in played.get((r.game_id, hid), set()))
            sA = sum(c for p, (t, c) in PP[r.season].items()
                     if t == aid and p in played.get((r.game_id, aid), set()))
            nrk.append(sum(1 for p, (t, c) in PP[r.season].items()
                           if p not in P[r.season]
                           and ((t == hid and p in played.get((r.game_id, hid), set()))
                                or (t == aid and p in played.get((r.game_id, aid), set())))))
            out.append(sH - sA)
        return np.array(out), np.array(nrk)

    cm_ctrl, _ = cm_col(P)
    cm_rk, n_rk_played = cm_col(PR)
    m_ctrl = np.where(active, 0.5 * df.fm + 0.5 * cm_ctrl + df.sched, df.m_us)
    m_rk = np.where(active, 0.5 * df.fm + 0.5 * cm_rk + df.sched, df.m_us)
    ll_c = ll_vec(df.y, sig(m_ctrl / SCALE))
    ll_r = ll_vec(df.y, sig(m_rk / SCALE))
    ll_s = ll_vec(df.y, sig(df.m_us / SCALE))
    d = ll_c - ll_r                       # >0 = rookie prior BETTER
    am = df.am.values.astype(float)
    a_ = active.values
    res = dict(
        primary_active=paired_ci(d[a_]),
        per_season_active={s: paired_ci(d[((df.season == s).values) & a_])
                           for s in SEASONS},
        early_pooled=paired_ci(d),
        vs_shipped_active=paired_ci((ll_s - ll_r)[a_]),
        control_vs_shipped_active=paired_ci((ll_s - ll_c)[a_]),
        mse_active=paired_ci(((am - m_ctrl) ** 2 - (am - m_rk) ** 2)[a_]),
        lls_active=dict(shipped=round(float(ll_s[a_].mean()), 4),
                        control=round(float(ll_c[a_].mean()), 4),
                        rookie=round(float(ll_r[a_].mean()), 4),
                        market=round(float(ll_vec(df.y, df.p_mkt)[a_].mean()), 4)),
        mechanism=dict(
            rosters=mech_ros,
            active_n=int(a_.sum()),
            games_with_rookie_played=int(((n_rk_played > 0) & a_).sum()),
            mean_rookies_played=round(float(n_rk_played[a_].mean()), 2),
            mean_abs_margin_delta=round(float(np.abs(m_rk - m_ctrl)[a_].mean()), 3),
            max_abs_margin_delta=round(float(np.abs(m_rk - m_ctrl)[a_].max()), 3)),
        subset_rookie_games=paired_ci(d[a_ & (n_rk_played > 0)]),
        zero_outside_window=bool((m_rk[~a_] == df.m_us[~a_]).all()),
        curve_used={s: wf[int(s[:4])] for s in SEASONS})
    json.dump(res, open(REPO / "data" / "rookie_bridge_gate.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
