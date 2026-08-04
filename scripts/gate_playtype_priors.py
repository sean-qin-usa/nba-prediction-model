#!/usr/bin/env python3
"""Gate: Synergy playtype 'styles make fights' prior.

H (pre-stated): a player's playtype mix (share of possessions by action type)
interacted with the opponent's per-playtype defensive quality predicts
player-game scoring-efficiency deviations from the player's own trailing
baseline. Zone-level opponent defense (shipped) sees rim/mid/3 geography but
not scheme (PnR coverage, iso defense, transition control) — playtypes are the
scheme axis.

PIT: playtype mix and opponent playtype defense are built ONLY from 2024-25
season aggregates (prior season, frozen); evaluated on 2025-26 player-games.
Roster churn attenuates the signal — a pass here understates the live version.

Predictor  x_ig = sum_t m_i(t) * d_T(t)
  m_i(t) = player i's 2024-25 offensive POSS_PCT in playtype t (renormalized)
  d_T(t) = opponent T's 2024-25 defensive PPP allowed in playtype t, centered
           vs possession-weighted league mean (positive = leaky defense)
Outcome    y_ig = pts-per-36 in game g minus player's trailing-20-game pts/36.
Gate: bootstrap 95% CI on the OLS slope of y on x (cluster-robust by game via
game-level bootstrap); PASS if CI excludes 0 in the predicted (positive)
direction AND out-of-fold MSE improves.
"""
import json
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nbapred.db import connect  # noqa: E402

RNG = np.random.default_rng(7)


def synergy_frames(season: str, grouping: str) -> pd.DataFrame:
    rows = []
    for f in glob.glob("data/raw/nba_api/synergy/*.json"):
        d = json.load(open(f))
        p = d["params"]
        if p["season"] != season or p["type_grouping_nullable"] != grouping:
            continue
        rs = d["response"].get("resultSets") or [d["response"].get("resultSet")]
        rs = rs if isinstance(rs, list) else [rs]
        df = pd.DataFrame(rs[0]["rowSet"], columns=rs[0]["headers"])
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    # --- 2024-25 priors (frozen) ---------------------------------------------
    off = synergy_frames("2024-25", "offensive")
    dfn = synergy_frames("2024-25", "defensive")
    # playtype universe = cached 9
    types = sorted(off.PLAY_TYPE.unique())

    # player mix: POSS_PCT renormalized over cached types; require >=100 total poss
    off["POSS"] = off.POSS_PCT * 1.0  # POSS_PCT is share of player's possessions
    mix = off.pivot_table(index="PLAYER_ID", columns="PLAY_TYPE",
                          values="POSS_PCT", aggfunc="sum").fillna(0.0)
    mix = mix.div(mix.sum(axis=1).replace(0, np.nan), axis=0).dropna()

    # opponent playtype defense: aggregate player defensive rows -> team PPP
    # allowed per playtype, weighted by defensive possessions (GP * POSS_PCT is
    # per-game share; use raw possession proxy GP*POSS_PCT).
    dfn = dfn[dfn.PPP.notna()].copy()
    dfn["W"] = (dfn.GP * dfn.POSS_PCT).clip(lower=0)
    td = (dfn.groupby(["TEAM_ID", "PLAY_TYPE"])
             .apply(lambda g: np.average(g.PPP, weights=g.W) if g.W.sum() > 0 else np.nan,
                    include_groups=False)
             .unstack())
    league = {}
    for t in list(types):
        sub = dfn[(dfn.PLAY_TYPE == t) & (dfn.W > 0)]
        if len(sub) < 30:            # playtype untracked on defense: drop axis
            types.remove(t)
            continue
        league[t] = np.average(sub.PPP, weights=sub.W)
    dcen = td - pd.Series(league)          # + = allows more than league PPP
    dcen = dcen.fillna(0.0)

    # --- 2025-26 outcomes ----------------------------------------------------
    con = connect(read_only=True)
    pg = con.execute("""
        SELECT s.player_id, s.team_id, s.game_id, g.game_date, s.pts AS points,
               s.seconds/60.0 AS minutes,
               (SELECT o.team_id FROM nba_games o
                 WHERE o.game_id = s.game_id AND o.team_id <> s.team_id LIMIT 1) AS opp_id
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g USING (game_id)
        WHERE g.season = '2025-26' AND s.game_id LIKE '002%' AND s.seconds >= 720
        ORDER BY s.player_id, g.game_date
    """).fetchdf()
    con.close()

    pg["p36"] = pg.points / pg.minutes * 36.0
    pg["trail"] = (pg.groupby("player_id")["p36"]
                     .transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean()))
    pg = pg.dropna(subset=["trail", "opp_id"])
    pg["y"] = pg.p36 - pg.trail

    # predictor
    mixd = {int(i): r.values for i, r in mix[types].iterrows()}
    dced = {int(i): r.reindex(types).values for i, r in dcen.iterrows()}
    x = np.full(len(pg), np.nan)
    for k, (pid, oid) in enumerate(zip(pg.player_id.astype(int), pg.opp_id.astype(int))):
        m = mixd.get(pid)
        d = dced.get(oid)
        if m is not None and d is not None:
            x[k] = float(m @ d)
    pg["x"] = x
    pg = pg.dropna(subset=["x"])
    pg["x"] -= pg.x.mean()
    n = len(pg)
    print(f"n player-games: {n}; players w/ mix: {len(mixd)}; x sd: {pg.x.std():.4f}")

    # --- gate ----------------------------------------------------------------
    # CONFOUND CONTROL (G4 checklist): x embeds "opponent defense is bad
    # overall", which team-level production already prices. Demean x AND y
    # within opponent so only the STYLE-MATCH residual is tested: does player
    # i deviate more vs T than the average visitor to T, in proportion to how
    # i's mix lines up with T's playtype weaknesses?
    if "--style-only" in sys.argv:
        pg["x"] = pg.x - pg.groupby("opp_id").x.transform("mean")
        pg["y"] = pg.y - pg.groupby("opp_id").y.transform("mean")
        print("mode: opponent fixed effects (pure style-match)")
    X = pg.x.values
    Y = pg.y.values
    slope = float(np.polyfit(X, Y, 1)[0])
    games = pg.game_id.unique()
    gmap = pg.groupby("game_id").indices
    boots = []
    for _ in range(2000):
        gs = RNG.choice(games, size=len(games), replace=True)
        idx = np.concatenate([gmap[g] for g in gs])
        boots.append(np.polyfit(X[idx], Y[idx], 1)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # OOS MSE check: split by date halves, fit slope on H1, score H2
    cut = pg.game_date.quantile(0.5)
    a, b = pg[pg.game_date <= cut], pg[pg.game_date > cut]
    s1 = np.polyfit(a.x, a.y, 1)
    mse_base = float((b.y ** 2).mean())
    mse_x = float(((b.y - np.polyval(s1, b.x)) ** 2).mean())
    print(f"slope {slope:+.3f}  CI[{lo:+.3f},{hi:+.3f}]  (pts/36 per unit expected-PPP tilt)")
    print(f"OOS MSE base {mse_base:.4f} -> with-x {mse_x:.4f}  (delta {mse_base - mse_x:+.5f})")
    effect = slope * pg.x.std()
    print(f"1-sd effect: {effect:+.3f} pts/36")
    verdict = "PASS" if (lo > 0 and mse_x < mse_base) else "FAIL"
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
