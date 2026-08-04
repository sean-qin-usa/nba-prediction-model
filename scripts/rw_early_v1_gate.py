"""PRE-REGISTERABLE GATE SPEC — "October composition bridge" (REGIME A top
candidate, ONE config, no sweeps).

DEFECT BEING FIXED (measured, rw_early_decomp/analysis): during each season's
first refit window (~week 1, 51-53 games/season), CompositionModel has an
EMPTY rotation (ROSTER_DAYS=12 excludes everyone in October; comp is 002-only)
so the shipped blend averages a LITERAL 0-margin at 50% weight AND the outs
channel is dead (outs are drawn from comp.players). Shipped week-1 margins are
half-scale (sd 3.3 vs market 6.5, logit slope 1.70 = underconfident) and cost
+0.0355/game vs market — 49%% of the entire early-season (gp<20) loss.

CONSTRUCTION (all inputs strictly pre-game-date; no fitting, no tuned params):
  ACTIVE only when the game is scored while the comp rotation is empty
  (equivalently cm == 0 via empty roster; all other games bitwise unchanged).
  roster(t) = {p appeared (seconds>0) in a 001 preseason game for t this
               season; team = argmax 001 minutes}
            UNION {p whose PRIOR-season primary 002 team (arg_max game_date)
               is t and who appears in no team's 001 this season}
  contrib(p) = DARKO_dpm(as-of < game) x trail_min(p)/48
               trail_min = avg of last 10 games with seconds>=720 in 002 data
               before the game (spans prior season; rookies -> no contrib)
  cm_ps(h,a) = sum contrib over roster(h) minus outs  -  same for away
               outs convention = SAME TIER as the rest of the backtest
               (oracle: roster member not in the game's played set; live:
               injury feed / official inactives)
  margin = 0.5*fm + 0.5*cm_ps + sched      (identical production blend; only
                                            the dead comp leg is replaced)

ENDPOINTS (pre-registered):
  primary  = paired bootstrap 2000x, 95%% CI, ll(shipped)-ll(variant) on the
             ACTIVE window, 3 seasons pooled (n~156). PASS = CI > 0.
  secondary= early window (either gp<20) pooled; per-season report.
DISCLOSURE: this config was DISCOVERED on the same 3 seasons (one adaptation:
outs convention added after diagnosing 2025-26 opening-night star absences).
A rerun of this script is a REPLICATION, not an independent confirm ->
route per GATE_POLICY_V2 as freeze-list live-October-2026 confirm; 2022-23
fallback-path analog becomes a quasi-holdout when 2021-22 lands.

Read-only DB; writes data/rw_early_v1_gate.json.
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
    rng = np.random.default_rng(seed)
    mm = d[rng.integers(0, len(d), (B, len(d)))].mean(axis=1)
    return dict(mean=round(float(d.mean()), 5),
                lo=round(float(np.percentile(mm, 2.5)), 5),
                hi=round(float(np.percentile(mm, 97.5)), 5), n=int(len(d)))


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def main():
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
    P = {}
    for s in SEASONS:
        opener = openers[s]
        ps_ros = {}
        for t, p, m in con.execute("""
            SELECT s.team_id, s.player_id, sum(s.seconds)/60.0
            FROM player_game_stats s
            JOIN nba_games g ON g.game_id=s.game_id AND g.team_id=s.team_id
            WHERE g.season=? AND s.game_id LIKE '001%' AND s.seconds>0
            GROUP BY 1,2""", [s]).fetchall():
            ps_ros.setdefault(int(p), {})[int(t)] = float(m)
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
        P[s] = {p: (t, darko.get(p, 0.0) * trail[p] / 48.0)
                for p, t in assign.items() if p in trail}
    played = {}
    for g, t, p in con.execute(
            "SELECT game_id, team_id, player_id FROM player_game_stats "
            "WHERE game_id LIKE '002%' AND seconds>0").fetchall():
        played.setdefault((g, int(t)), set()).add(int(p))
    con.close()

    active = (df.cm == 0)
    cmo = []
    for r in df.itertuples():
        if r.cm != 0:
            cmo.append(0.0)
            continue
        hid, aid = ab2id[(r.season, r.home)], ab2id[(r.season, r.away)]
        sH = sum(c for p, (t, c) in P[r.season].items()
                 if t == hid and p in played.get((r.game_id, hid), set()))
        sA = sum(c for p, (t, c) in P[r.season].items()
                 if t == aid and p in played.get((r.game_id, aid), set()))
        cmo.append(sH - sA)
    df["cm_ps"] = cmo
    m_var = np.where(active, 0.5 * df.fm + 0.5 * df.cm_ps + df.sched, df.m_us)
    d = ll_vec(df.y, sig(df.m_us / SCALE)) - ll_vec(df.y, sig(m_var / SCALE))
    llm = ll_vec(df.y, df.p_mkt)
    res = dict(
        active_window=paired_ci(d[active.values]),
        early_pooled=paired_ci(d),
        per_season={s: paired_ci(d[(df.season == s).values])
                    for s in SEASONS},
        per_season_active={s: paired_ci(d[((df.season == s) & active).values])
                           for s in SEASONS},
        lls_active=dict(
            shipped=round(float(ll_vec(df.y, sig(df.m_us / SCALE))[active].mean()), 4),
            variant=round(float(ll_vec(df.y, sig(m_var / SCALE))[active].mean()), 4),
            market=round(float(llm[active].mean()), 4)),
        n_active_per_season={s: int((active & (df.season == s)).sum())
                             for s in SEASONS},
        zero_outside_window=bool((m_var[~active.values]
                                  == df.m_us[~active.values]).all()))
    json.dump(res, open(REPO / "data" / "rw_early_v1_gate.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
