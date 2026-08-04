"""Draft-slot -> rookie-season DPM curve (D84-A candidate #4, part 1).

HYPOTHESIS (pre-declared, no sweeps): rookie impact declines log-linearly in
draft slot — dpm_rookie = a + b*log(overall_pick) — the canonical convex draft
value curve. Target = MEAN darko_history dpm over the rookie season (the
realized DARKO estimate of the player's impact while a rookie; chosen ex-ante
over end-of-season dpm which overweights within-year development).

Fit universe: draft classes 2017+ (task spec) with any darko rows in their
rookie-season window. PIT: the gate consumer fits WALK-FORWARD — for eval
season s only classes with rookie seasons strictly before s enter the fit.

Also measured here (both PIT-clean for the 2023-26 eval window):
  - per-class darko coverage (survivorship check on pre-2022 classes)
  - rookie preseason(001) mpg vs first-10 regular(002) mpg on class 2022
    (rookie season 2022-23 < all eval seasons) -> minutes mapping decision
  - undrafted-rookie realized mean dpm (reference; they stay 0 in the bridge)

Read-only DB. Writes data/rookie_draft_curve.json.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from nbapred.db import connect

CLASSES = list(range(2017, 2026))          # rookie seasons 2017-18 .. 2025-26


def season_window(draft_year):
    return f"{draft_year}-10-01", f"{draft_year + 1}-06-30"


def fit_loglin(picks, dpms):
    x = np.log(np.asarray(picks, float))
    y = np.asarray(dpms, float)
    A = np.vstack([np.ones_like(x), x]).T
    (a, b), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ np.array([a, b])
    ss = 1 - ((y - yhat) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    return float(a), float(b), float(ss)


def main():
    con = connect(read_only=True)
    rows = con.execute(
        "SELECT player_id, draft_year, overall_pick FROM draft_history "
        "WHERE draft_year >= 2017 AND overall_pick IS NOT NULL").fetchall()
    # realized rookie-season mean dpm per drafted player
    data = []          # (draft_year, pick, mean_dpm, n_rows)
    for pid, dy, pick in rows:
        if dy not in CLASSES:
            continue
        lo, hi = season_window(dy)
        r = con.execute(
            "SELECT avg(dpm), count(*) FROM darko_history "
            "WHERE player_id=? AND date BETWEEN ? AND ?", [pid, lo, hi]).fetchone()
        if r[0] is not None and r[1] >= 5:      # >=5 as-of dates = actually played
            data.append((dy, int(pick), float(r[0]), int(r[1])))

    per_class = {}
    for dy in CLASSES:
        cls = [d for d in data if d[0] == dy]
        n_picks = con.execute(
            "SELECT count(*) FROM draft_history WHERE draft_year=? AND "
            "overall_pick IS NOT NULL", [dy]).fetchone()[0]
        per_class[dy] = dict(covered=len(cls), picks=int(n_picks),
                             coverage=round(len(cls) / max(n_picks, 1), 2),
                             mean_dpm=round(float(np.mean([d[2] for d in cls])), 3)
                             if cls else None)

    # pooled fit (all classes) + walk-forward fits for each eval season
    out = dict(per_class=per_class)
    pooled = fit_loglin([d[1] for d in data], [d[2] for d in data])
    out["pooled_fit"] = dict(a=round(pooled[0], 4), b=round(pooled[1], 4),
                             r2=round(pooled[2], 3), n=len(data))
    out["walk_forward"] = {}
    for eval_year in (2023, 2024, 2025, 2026):
        train = [d for d in data if d[0] < eval_year]
        a, b, r2 = fit_loglin([d[1] for d in train], [d[2] for d in train])
        out["walk_forward"][eval_year] = dict(
            a=round(a, 4), b=round(b, 4), r2=round(r2, 3), n=len(train),
            pred=dict(p1=round(a, 2), p5=round(a + b * np.log(5), 2),
                      p14=round(a + b * np.log(14), 2),
                      p30=round(a + b * np.log(30), 2),
                      p60=round(a + b * np.log(60), 2)))

    # empirical bin means (shape check on the log-linear hypothesis)
    bins = [(1, 5), (6, 10), (11, 20), (21, 30), (31, 45), (46, 60)]
    out["bin_means"] = {
        f"{lo}-{hi}": dict(
            mean=round(float(np.mean([d[2] for d in data if lo <= d[1] <= hi])), 3),
            n=len([d for d in data if lo <= d[1] <= hi]))
        for lo, hi in bins}

    # undrafted rookies (reference): first darko season 2017+, no draft row
    und = con.execute("""
        WITH fd AS (SELECT player_id, min(date) fd FROM darko_history GROUP BY 1)
        SELECT count(*), avg(m) FROM (
          SELECT h.player_id, avg(h.dpm) m
          FROM darko_history h
          JOIN fd USING (player_id)
          WHERE year(fd.fd) >= 2017
            AND h.date <= fd.fd + INTERVAL 250 DAY
            AND h.player_id NOT IN (SELECT player_id FROM draft_history)
          GROUP BY 1)""").fetchone()
    out["undrafted_first_year"] = dict(n=int(und[0]),
                                       mean_dpm=round(float(und[1]), 3))

    # minutes map: class-2022 rookies, 001 mpg (2022-23) vs first-10 002 mpg
    mm = con.execute("""
        WITH rook AS (SELECT player_id FROM draft_history WHERE draft_year=2022),
        ps AS (
          SELECT s.player_id, avg(s.seconds/60.0) mpg
          FROM player_game_stats s
          JOIN nba_games g ON g.game_id=s.game_id AND g.team_id=s.team_id
          WHERE g.season='2022-23' AND s.game_id LIKE '001%' AND s.seconds>0
          GROUP BY 1),
        reg AS (
          SELECT player_id, avg(m) mpg10 FROM (
            SELECT s.player_id, s.seconds/60.0 m,
                   row_number() OVER (PARTITION BY s.player_id
                                      ORDER BY g.game_date) rn
            FROM player_game_stats s
            JOIN nba_games g ON g.game_id=s.game_id AND g.team_id=s.team_id
            WHERE g.season='2022-23' AND s.game_id LIKE '002%' AND s.seconds>0)
          WHERE rn<=10 GROUP BY 1)
        SELECT ps.mpg, reg.mpg10 FROM rook
        JOIN ps USING (player_id) JOIN reg USING (player_id)""").fetchall()
    con.close()
    x = np.array([m[0] for m in mm]); yv = np.array([m[1] for m in mm])
    A = np.vstack([np.ones_like(x), x]).T
    (c0, c1), *_ = np.linalg.lstsq(A, yv, rcond=None)
    out["minutes_map_class2022"] = dict(
        n=len(mm), corr=round(float(np.corrcoef(x, yv)[0, 1]), 3),
        c0=round(float(c0), 3), c1=round(float(c1), 3),
        mean_ps=round(float(x.mean()), 1), mean_reg=round(float(yv.mean()), 1))

    json.dump(out, open(REPO / "data" / "rookie_draft_curve.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
