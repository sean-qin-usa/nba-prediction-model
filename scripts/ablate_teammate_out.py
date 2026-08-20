"""Sean's oracle-availability test for PROPS: when a star teammate (>=28 trailing
min, fresh absence) does NOT play, a player's usage should jump — do props improve
if we model it?

Step 1 EMPIRICAL: measure the actual usage lift (shots/min, FTA/min, pts/min)
when a top teammate is out vs in. If real, Step 2: apply the empirical lift
(estimated on TRAIN period only, leakage-safe) in the prop sim and gate on CRPS,
overall AND on the star-out subset (where the edge would live).
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from nbapred.db import connect
from nbapred.engine.props import player_rates_from_stats, simulate_player

STAR_MIN = 28.0


def crps(s, y):
    s = np.sort(s); n = len(s)
    return float(np.mean(np.abs(s - y)) - 0.5 * (2 * np.arange(1, n + 1) - n - 1) @ s / n**2)


def main(sims=3000, max_eval=1100):
    con = connect(read_only=True)
    pg = con.execute("""SELECT s.game_id, s.player_id, s.team_id, g.game_date,
        s.seconds, s.pts, s.rima+s.mida+s.thra shots, s.fta
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING(game_id)
        WHERE s.game_id LIKE '002%' ORDER BY g.game_date""").fetchdf()
    con.close()
    pg["mins"] = pg.seconds / 60.0

    # trailing star status: player's avg minutes over prior 10 played games
    pg = pg.sort_values(["player_id", "game_date"])
    pg["avg10"] = pg.groupby("player_id")["mins"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=5).mean())
    pg["last_played"] = pg.groupby("player_id")["game_date"].shift(1)

    # for each team-game: list of stars expected (avg10>=STAR_MIN, played within 10d)
    played = pg[pg.mins >= 8].groupby(["game_id", "team_id"])["player_id"].apply(set)
    stars = pg[(pg.avg10 >= STAR_MIN)]
    # expected stars per (team, date): star row exists for a game they PLAYED; to
    # detect absence we need per team-game the stars whose last game was recent.
    star_hist = stars[["player_id", "team_id", "game_date"]].copy()

    # build per (game_id, team): star_out flag
    games = pg[["game_id", "team_id", "game_date"]].drop_duplicates()
    star_by_team = {}
    for r in star_hist.itertuples():
        star_by_team.setdefault(r.team_id, []).append((r.game_date, r.player_id))
    star_out = {}
    for r in games.itertuples():
        outs = set()
        hist = star_by_team.get(r.team_id, [])
        # stars who played for this team within 12 days before this game
        recent = {p for (d, p) in hist if 0 < (r.game_date - d).days <= 12}
        present = played.get((r.game_id, r.team_id), set())
        outs = {p for p in recent if p not in present}
        star_out[(r.game_id, r.team_id)] = outs

    # STEP 1: empirical usage lift for rotation players (>=15 trailing min, not the star)
    rot = pg[(pg.avg10 >= 15) & (pg.mins >= 12)].copy()
    rot["star_absent"] = [len(star_out.get((g, t), set()) - {p}) > 0
                          for g, t, p in zip(rot.game_id, rot.team_id, rot.player_id)]
    for col, label in (("shots", "shots/min"), ("fta", "FTA/min"), ("pts", "pts/min"),
                       ("mins", "minutes")):
        base = rot[col] / rot.mins if col != "mins" else rot[col]
        a = base[rot.star_absent]; b = base[~rot.star_absent]
        print(f"{label:10}: star OUT {a.mean():.4f} (n={len(a)})  IN {b.mean():.4f}  "
              f"lift {a.mean()/b.mean()-1:+.1%}")

    # STEP 2: sim gate — apply lifts (train-estimated, global) on star-out games
    cut = pg.game_date.quantile(0.6)
    tr = rot[rot.game_date <= cut]
    lift_shots = (tr[tr.star_absent].shots / tr[tr.star_absent].mins).mean() / \
                 (tr[~tr.star_absent].shots / tr[~tr.star_absent].mins).mean()
    lift_min = tr[tr.star_absent].mins.mean() / tr[~tr.star_absent].mins.mean()
    print(f"\nTRAIN lifts: shots/min x{lift_shots:.3f}  minutes x{lift_min:.3f}")

    test = rot[(rot.game_date > cut)].copy()
    test = test.iloc[::max(1, len(test) // (max_eval * 2))]
    con = connect(read_only=True)
    b_all, e_all, b_out, e_out, n = [], [], [], [], 0
    for r in test.itertuples():
        if n >= max_eval:
            break
        rates = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
        if not rates or rates["n_games"] < 8 or rates["proj_min"] < 15:
            continue
        rn = dict(rates); rn.pop("minutes_hist", None)
        y = r.pts
        sb = simulate_player(rn, sims, seed=n)["points"]
        cb = crps(sb, y)
        if r.star_absent:
            r2 = dict(rn)
            for k in ("rate_rim", "rate_mid", "rate_thr", "fta_per_min"):
                r2[k] = rn[k] * lift_shots
            r2["proj_min"] = min(rn["proj_min"] * lift_min, 44)
            se = simulate_player(r2, sims, seed=n)["points"]
            ce = crps(se, y)
            b_out.append(cb); e_out.append(ce)
        else:
            ce = cb
        b_all.append(cb); e_all.append(ce)
        n += 1
    con.close()

    print(f"\ntest n={n}  (star-out subset: {len(b_out)})")
    print(f"ALL     : base {np.mean(b_all):.4f}  oracle-adjusted {np.mean(e_all):.4f}")
    if len(b_out) > 30:
        d = np.array(b_out) - np.array(e_out)
        rng = np.random.default_rng(0)
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"STAR-OUT: base {np.mean(b_out):.4f}  adjusted {np.mean(e_out):.4f}  "
              f"delta {d.mean():+.4f}  CI ({lo:+.4f},{hi:+.4f}) -> {'KEEP' if lo > 0 else 'no'}")


if __name__ == "__main__":
    main()
