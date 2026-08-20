"""ABSENCE AUDIT — starout.team_context (inventory site #6).

`team_context` computes each player's trailing-10 `trail_min`, `trail_att` and
`last_date` over **every** box-score row for the team, with NO `seconds` filter,
unlike composition (>=720) and props (>=720). Two consequences to measure:

  (i)  DNP rows (seconds = 0) are in the trailing window, so an ABSENT star's
       own `trail_min` FALLS. After enough missed games he drops below
       STAR_TRAILING_MIN = 28 and the D33/D39 star-out redistribution SILENTLY
       STOPS FIRING while he is still out.
  (ii) `last_date = max(game_date)` over the same unfiltered rows, so a DNP row
       counts as "played" and the documented `FRESH_DAYS <= 12` guard
       ("else his absence is already embedded in teammates' trailing rates")
       never expires for a player the box score keeps listing.

First: how complete are DNP (seconds = 0) rows in this corpus at all?
Then: the detector's firing rate against a CLEAN (>=12-min-conditioned) star
definition, as a function of the star's consecutive missed games k.

READ-ONLY. Writes data/ab_starout_diag.json.
"""
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from nbapred.db import connect
from nbapred.engine.starout import (FRESH_DAYS, MIN_TRAIL_GAMES,
                                    STAR_TRAILING_MIN, TRAIL_GAMES)

SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")


def _d(x):
    return x.date() if hasattr(x, "date") else x


def main():
    con = connect(read_only=True)
    pm = con.execute("""
        SELECT s.game_id, s.team_id, s.player_id, s.seconds, g.game_date, g.season
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

    out = {}
    # ---- (0) DNP-row coverage ---------------------------------------------
    # For each (season, team, player): team games between his first and last
    # >=12-min game vs the rows he actually has.
    rowset = defaultdict(dict)     # (season, tid, pid) -> {date: seconds}
    for r in pm.itertuples():
        rowset[(r.season, int(r.team_id), int(r.player_id))][r.game_date] = int(r.seconds or 0)
    span_games = miss_with_row = miss_no_row = 0
    for (s, t, p), dd in rowset.items():
        big = [d for d, sec in dd.items() if sec >= 720]
        if len(big) < 5:
            continue
        lo, hi = min(big), max(big)
        for d in sched[(s, t)]:
            if lo <= d <= hi and dd.get(d, 0) < 720:
                span_games += 1
                if d in dd:
                    miss_with_row += 1
                else:
                    miss_no_row += 1
    out["dnp_row_coverage"] = {
        "player_team_games_missed_inside_span": span_games,
        "with_a_box_row": miss_with_row,
        "with_NO_box_row": miss_no_row,
        "coverage": round(miss_with_row / max(span_games, 1), 4)}
    print("DNP-row coverage:", json.dumps(out["dnp_row_coverage"]))

    # ---- (1) detector comparison ------------------------------------------
    # For every (season, team, game_date), find players who are OUT (oracle:
    # no row with seconds>0) and ask whether each definition calls them a star.
    per = defaultdict(list)        # (season, tid, pid) -> [(date, sec)]
    for r in pm.itertuples():
        per[(r.season, int(r.team_id), int(r.player_id))].append(
            (r.game_date, int(r.seconds or 0)))
    for k in per:
        per[k].sort()

    recs = []
    for (s, t, p), ent in per.items():
        dates = [d for d, _ in ent]
        secs = {d: sec for d, sec in ent}
        big = [d for d, sec in ent if sec >= 720]
        if len(big) < MIN_TRAIL_GAMES:
            continue
        sc = sched[(s, t)]
        k_miss = 0
        for gd in sc:
            playing = secs.get(gd, 0) > 0
            if playing:
                k_miss = 0
                continue
            k_miss += 1
            prior = [d for d in dates if d < gd]
            if len(prior) < MIN_TRAIL_GAMES:
                continue
            # CURRENT construction: all rows, last TRAIL_GAMES, no seconds filter
            tail = prior[-TRAIL_GAMES:]
            tm_cur = float(np.mean([secs[d] / 60.0 for d in tail]))
            ld_cur = prior[-1]
            fresh_cur = 0 < (gd - ld_cur).days <= FRESH_DAYS
            # CLEAN construction: >=12-min games only (composition / props rule)
            pbig = [d for d in big if d < gd]
            if len(pbig) < MIN_TRAIL_GAMES:
                continue
            tailb = pbig[-TRAIL_GAMES:]
            tm_cln = float(np.mean([secs[d] / 60.0 for d in tailb]))
            ld_cln = pbig[-1]
            fresh_cln = 0 < (gd - ld_cln).days <= FRESH_DAYS
            recs.append((s, t, p, gd, k_miss, tm_cur, tm_cln,
                         int(fresh_cur), int(fresh_cln),
                         int(tm_cur >= STAR_TRAILING_MIN and fresh_cur),
                         int(tm_cln >= STAR_TRAILING_MIN and fresh_cln)))
    df = pd.DataFrame(recs, columns=["season", "team_id", "player_id", "date",
                                     "k_miss", "tm_cur", "tm_clean",
                                     "fresh_cur", "fresh_clean",
                                     "star_cur", "star_clean"])
    df.to_csv("data/ab_starout_rows.csv.gz", index=False, compression="gzip")
    out["n_absent_player_games"] = int(len(df))
    print("\nabsent player-games with >=5 prior rows:", len(df))

    # restrict to players who WERE genuine stars on the eve of the absence
    tab = []
    for lo, hi in [(1, 1), (2, 2), (3, 3), (4, 5), (6, 9), (10, 20), (21, 200)]:
        s = df[(df.k_miss >= lo) & (df.k_miss <= hi) & (df.star_clean == 1)]
        if len(s) == 0:
            tab.append((f"{lo}-{hi}", 0, None, None, None, None))
            continue
        tab.append((f"{lo}-{hi}", int(len(s)),
                    round(float(s.tm_clean.mean()), 2),
                    round(float(s.tm_cur.mean()), 2),
                    round(float(s.star_cur.mean()), 4),
                    round(float(s.fresh_cur.mean()), 4)))
    out["fire_rate_by_k"] = tab
    print("\n(k missed, n CLEAN stars out, clean trail_min, CURRENT trail_min, "
          "P(current detector fires), P(current fresh)):")
    for r in tab:
        print("   ", r)

    # the reverse error: current fires where clean says not a star
    fp = df[(df.star_cur == 1) & (df.star_clean == 0)]
    fn = df[(df.star_cur == 0) & (df.star_clean == 1)]
    out["disagreement"] = {
        "clean_star_out_total": int((df.star_clean == 1).sum()),
        "current_fires_total": int((df.star_cur == 1).sum()),
        "false_negatives": int(len(fn)),
        "false_positives": int(len(fp)),
        "recall": round(float((df.star_cur == 1).sum() / max((df.star_clean == 1).sum(), 1)), 4)}
    print("\ndisagreement:", json.dumps(out["disagreement"]))
    # freshness pathology: how often does the DNP row keep `fresh` alive?
    kept = df[(df.fresh_cur == 1) & (df.fresh_clean == 0)]
    out["freshness_kept_alive_by_dnp_rows"] = int(len(kept))
    print("rows where a DNP box row keeps FRESH_DAYS alive but the clean "
          f"(>=12-min) last game is already stale: {len(kept)}")

    json.dump(out, open("data/ab_starout_diag.json", "w"), indent=1)
    print("wrote data/ab_starout_diag.json")


if __name__ == "__main__":
    main()
