#!/usr/bin/env python3
"""GATE 1 DESIGN DIAGNOSTIC — starout trail_min absence-awareness.

DETECTOR LEVEL ONLY. No endpoint (points CRPS) number is produced here; this is
the D133/D145 protocol verbatim (run and disclose the design diagnostic first,
pre-register afterwards, never touch the endpoint before the pre-registration).

Extends scripts/ab_starout_diag.py (D145 §5) from ONE construction to the THREE
that GATE 1 pre-registers, and adds the false-firing measurement the veto needs:

  ctrl    trail_min = avg over the last TRAIL_GAMES BOX ROWS (DNP zeros in)
  A       trail_min = avg over the last TRAIL_GAMES PLAYED rows (m > 0);
          n_tr becomes a PLAYED count
  B       ctrl levels, plus a played-count FLOOR: a player with fewer than
          PLAYED_FLOOR played rows inside the trailing ROW window is excluded
          from BOTH star candidacy and the rotation pool

Reference ("oracle star"): a player is genuinely a star-and-still-a-rotation-
player on date d iff his mean minutes over his last TRAIL_GAMES games ACTUALLY
PLAYED (>= 12 min, the props/composition convention) is >= STAR_TRAILING_MIN,
he has >= MIN_TRAIL_GAMES such games, AND he played >= PLAYED_FLOOR of his
team's last 10 games. The last clause is what "no longer a rotation player"
means, so a fire on a non-oracle star is a FALSE FIRE.

Also sizes the eval universe (how many rows each arm actually moves) and the
per-arm freshness behaviour after the D146 fix.

READ-ONLY. Writes data/qg_starout_design.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from nbapred.db import connect
from nbapred.engine.starout import (FRESH_DAYS, MIN_TRAIL_GAMES, PLAYED_FLOOR,
                                    ROT_TRAILING_MIN, STAR_TRAILING_MIN,
                                    TRAIL_GAMES)

SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
BIG = 12.0          # minutes defining a "meaningfully played" game (props rule)
KBUCKETS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 9), (10, 20), (21, 200)]


def _d(x):
    return x.date() if hasattr(x, "date") else x


def main():
    con = connect(read_only=True)
    pm = con.execute("""
        SELECT s.game_id, s.team_id, s.player_id, s.seconds/60.0 AS m,
               g.game_date, g.season
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games
              WHERE wl IS NOT NULL) g USING (game_id)
        WHERE s.game_id LIKE '002%'""").fetchdf()
    pm["game_date"] = [_d(x) for x in pm["game_date"]]
    sched = defaultdict(list)
    for s, t, d in con.execute("""
        SELECT season, team_id, game_date FROM nba_games
        WHERE game_id LIKE '002%' AND wl IS NOT NULL""").fetchall():
        sched[(s, int(t))].append(_d(d))
    for k in sched:
        sched[k] = sorted(set(sched[k]))
    con.close()
    pm = pm[pm["season"].isin(SEASONS)]
    print(f"loaded {len(pm)} 002 rows, seasons {SEASONS}", flush=True)

    per = defaultdict(list)                 # (season, tid, pid) -> [(date, m)]
    for r in pm.itertuples():
        per[(r.season, int(r.team_id), int(r.player_id))].append(
            (r.game_date, float(r.m or 0.0)))
    for k in per:
        per[k].sort()

    recs = []
    for (s, t, p), ent in per.items():
        dates = [d for d, _ in ent]
        mins = {d: m for d, m in ent}
        big = [d for d, m in ent if m >= BIG]
        if len(big) < MIN_TRAIL_GAMES:
            continue
        sc = sched[(s, t)]
        scset = set(sc)
        played_dates = {d for d, m in ent if m > 0}
        k_miss = 0
        for gd in sc:
            playing = mins.get(gd, 0.0) > 0
            if playing:
                k_miss = 0
                continue
            k_miss += 1
            prior_rows = [d for d in dates if d < gd]
            prior_played = [d for d in prior_rows if mins[d] > 0]
            prior_big = [d for d in big if d < gd]
            # ---- ctrl: last TRAIL_GAMES ROWS, zeros included
            if len(prior_rows) < MIN_TRAIL_GAMES:
                continue
            tail = prior_rows[-TRAIL_GAMES:]
            tm_ctrl = float(np.mean([mins[d] for d in tail]))
            n_tr_ctrl = len(tail)
            n_pl_win = sum(1 for d in tail if mins[d] > 0)
            # ---- A: last TRAIL_GAMES PLAYED rows
            tm_A = (float(np.mean([mins[d] for d in prior_played[-TRAIL_GAMES:]]))
                    if prior_played else None)
            n_tr_A = min(len(prior_played), TRAIL_GAMES)
            # ---- freshness (D146: last PLAYED date)
            ld = prior_played[-1] if prior_played else None
            fresh = (ld is not None) and (0 < (gd - ld).days <= FRESH_DAYS)
            # ---- oracle reference: >=12-min trailing mean + rotation membership
            if len(prior_big) < MIN_TRAIL_GAMES:
                continue
            tm_or = float(np.mean([mins[d] for d in prior_big[-TRAIL_GAMES:]]))
            prior_sched = [d for d in sc if d < gd][-10:]
            first_big = min(prior_big)
            played10 = sum(1 for d in prior_sched
                           if d >= first_big and d in played_dates)
            oracle = int(tm_or >= STAR_TRAILING_MIN
                         and played10 >= PLAYED_FLOOR and fresh)
            fire_ctrl = int(tm_ctrl >= STAR_TRAILING_MIN
                            and n_tr_ctrl >= MIN_TRAIL_GAMES and fresh)
            fire_A = int(tm_A is not None and tm_A >= STAR_TRAILING_MIN
                         and n_tr_A >= MIN_TRAIL_GAMES and fresh)
            fire_B = int(tm_ctrl >= STAR_TRAILING_MIN
                         and n_tr_ctrl >= MIN_TRAIL_GAMES
                         and n_pl_win >= PLAYED_FLOOR and fresh)
            # rotation-pool membership of a PLAYING teammate is measured below
            recs.append((s, t, p, gd, k_miss, tm_ctrl, tm_A, tm_or,
                         n_pl_win, played10, int(fresh), oracle,
                         fire_ctrl, fire_A, fire_B))
    df = pd.DataFrame(recs, columns=[
        "season", "team_id", "player_id", "date", "k_miss", "tm_ctrl", "tm_A",
        "tm_oracle", "n_pl_win", "played10", "fresh", "oracle",
        "fire_ctrl", "fire_A", "fire_B"])
    df.to_csv("data/qg_starout_rows.csv.gz", index=False, compression="gzip")
    out = {"n_absent_player_games": int(len(df)),
           "seasons": list(SEASONS),
           "constants": dict(STAR=STAR_TRAILING_MIN, ROT=ROT_TRAILING_MIN,
                             TRAIL_GAMES=TRAIL_GAMES,
                             MIN_TRAIL=MIN_TRAIL_GAMES, FLOOR=PLAYED_FLOOR,
                             FRESH_DAYS=FRESH_DAYS, BIG=BIG)}
    print(f"\nabsent player-games with >=5 prior rows: {len(df)}")

    # ---------- (1) P(fire) by consecutive-missed bucket, on ORACLE stars ----
    tab = []
    for lo, hi in KBUCKETS:
        s = df[(df.k_miss >= lo) & (df.k_miss <= hi) & (df.oracle == 1)]
        if not len(s):
            tab.append([f"{lo}-{hi}", 0] + [None] * 6)
            continue
        tab.append([f"{lo}-{hi}", int(len(s)),
                    round(float(s.tm_oracle.mean()), 2),
                    round(float(s.tm_ctrl.mean()), 2),
                    round(float(s.tm_A.mean()), 2),
                    round(float(s.fire_ctrl.mean()), 4),
                    round(float(s.fire_A.mean()), 4),
                    round(float(s.fire_B.mean()), 4)])
    out["pfire_by_k_oracle"] = tab
    print("\n(k missed | n oracle stars out | oracle trail_min | ctrl trail_min |"
          " A trail_min | P(fire) ctrl | A | B):")
    for r in tab:
        print("   ", r)

    # ---------- (2) recall / false fires --------------------------------
    tot_or = int((df.oracle == 1).sum())
    rec = {}
    for a in ("ctrl", "A", "B"):
        f = df[f"fire_{a}"]
        rec[a] = {
            "fires": int(f.sum()),
            "true_fires": int(((f == 1) & (df.oracle == 1)).sum()),
            "false_fires": int(((f == 1) & (df.oracle == 0)).sum()),
            "missed": int(((f == 0) & (df.oracle == 1)).sum()),
            "recall": round(float(((f == 1) & (df.oracle == 1)).sum() / max(tot_or, 1)), 4),
            "precision": round(float(((f == 1) & (df.oracle == 1)).sum() / max(int(f.sum()), 1)), 4),
        }
    out["oracle_star_out_total"] = tot_or
    out["detector"] = rec
    print(f"\noracle star-outs total {tot_or}")
    print(json.dumps(rec, indent=1))

    # false fires decomposed: which oracle clause fails?
    fd = {}
    for a in ("ctrl", "A", "B"):
        s = df[(df[f"fire_{a}"] == 1) & (df.oracle == 0)]
        fd[a] = {"n": int(len(s)),
                 "fails_rotation_only": int(((s.tm_oracle >= STAR_TRAILING_MIN)
                                             & (s.played10 < PLAYED_FLOOR)).sum()),
                 "fails_star_level_only": int(((s.tm_oracle < STAR_TRAILING_MIN)
                                               & (s.played10 >= PLAYED_FLOOR)).sum()),
                 "fails_both": int(((s.tm_oracle < STAR_TRAILING_MIN)
                                    & (s.played10 < PLAYED_FLOOR)).sum()),
                 "mean_played10": round(float(s.played10.mean()), 2) if len(s) else None}
    out["false_fire_decomposition"] = fd
    print("\nfalse-fire decomposition:", json.dumps(fd, indent=1))

    # ---------- (3) how much does each arm MOVE? -------------------------
    mv = {}
    for a in ("A", "B"):
        d = df[f"fire_{a}"] != df["fire_ctrl"]
        mv[a] = {"team_games_changed": int(d.sum()),
                 "share_of_absent_rows": round(float(d.mean()), 4),
                 "newly_fires": int(((df[f"fire_{a}"] == 1) & (df.fire_ctrl == 0)).sum()),
                 "no_longer_fires": int(((df[f"fire_{a}"] == 0) & (df.fire_ctrl == 1)).sum())}
    out["arm_movement"] = mv
    print("\narm movement:", json.dumps(mv, indent=1))

    # ---------- (4) per-season stability of the ctrl decay ---------------
    ss = {}
    for s_ in SEASONS:
        sub = df[(df.season == s_) & (df.oracle == 1)]
        if not len(sub):
            continue
        a = sub[sub.k_miss <= 1].fire_ctrl.mean()
        b = sub[sub.k_miss >= 4].fire_ctrl.mean()
        ss[s_] = [int((sub.k_miss <= 1).sum()), round(float(a), 4),
                  int((sub.k_miss >= 4).sum()), round(float(b), 4),
                  round(float(b - a), 4)]
    out["ctrl_decay_by_season"] = ss
    print("\nctrl P(fire) k<=1 vs k>=4, by season [n1, p1, n4, p4, diff]:")
    for k, v in ss.items():
        print("   ", k, v)

    json.dump(out, open("data/qg_starout_design.json", "w"), indent=1, default=float)
    print("\nwrote data/qg_starout_design.json")
    print("QG_STAROUT_DESIGN_DONE", flush=True)


if __name__ == "__main__":
    main()
