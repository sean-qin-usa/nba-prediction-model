#!/usr/bin/env python3
"""Reproduce the pre-schedule-layer production model (PIT DARKO baseline, D44)
per game, for the paired bootstrap gate on the schedule layer. Exact copy of
production.fit_production as of the D44 capstone (hardcoded HOME_EDGE=3.0 via
composition default, home-in-FF margins, no schedule term)."""
import sys, warnings, csv
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nbapred.db import connect
from nbapred.model.production import (SCALE, sigmoid, last_season_prior)
from nbapred.model.team_ratings import TeamRatings, game_rows
from nbapred.model.composition import CompositionModel
from nbapred.model.four_factors import FourFactors


def fit_production_legacy(con, season, before=None, w_comp=0.7):
    comp = CompositionModel(con, before=before)
    ff = FourFactors().fit(con, season, before=before)
    tr = TeamRatings(ridge=25.0).fit(game_rows(con, before=before, season=season))
    prior = last_season_prior(con, season)
    ab = dict(con.execute(
        "SELECT DISTINCT team_id, team_abbrev FROM nba_games WHERE season=?", [season]).fetchall())
    id2ab = {t: a for t, a in ab.items()}
    games_played = dict(con.execute("""
        SELECT team_id, count(*) FROM nba_games WHERE season=? AND game_id LIKE '002%'
        AND wl IS NOT NULL GROUP BY 1""", [season]).fetchall())

    class Predictor:
        def ratings_margin(self, home_id, away_id):
            m = tr.pred_margin(home_id, away_id)
            gh = games_played.get(home_id, 0); ga = games_played.get(away_id, 0)
            wh = max(0.0, 1 - gh / 20.0); wa = max(0.0, 1 - ga / 20.0)
            ph = prior.get(id2ab.get(home_id, ""), 0.0)
            pa = prior.get(id2ab.get(away_id, ""), 0.0)
            return m + wh * ph - wa * pa

        def margin(self, home_id, away_id, out_home=None, out_away=None, game_date=None):
            cm = comp.margin(home_id, away_id, out_home, out_away, game_date)  # HE=3.0 default
            if ff.ready:
                fm = ff.margin(home_id, away_id)
                return 0.5 * fm + 0.5 * cm
            rm = self.ratings_margin(home_id, away_id)
            return w_comp * cm + (1 - w_comp) * rm

        def p_home(self, home_id, away_id, out_home=None, out_away=None, game_date=None):
            return float(sigmoid(self.margin(home_id, away_id, out_home,
                                             out_away, game_date) / SCALE))
    return Predictor()


def season_run(season, wtr):
    con = connect(read_only=True)
    pm = con.execute("""SELECT game_id, team_id, player_id FROM player_game_stats
        WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    played = {(g, t): set(grp.player_id) for (g, t), grp in pm.groupby(["game_id", "team_id"])}
    meta = con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl, game_date
        FROM nba_games WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL
        ORDER BY game_date""", [season]).fetchdf()
    mkt = {(str(r[0])[:10], r[1], r[2]): r[3] for r in con.execute(
        "SELECT game_date, home, away, p_home_spread FROM odds_market WHERE season_end=?",
        [int(season[:4]) + 1]).fetchall()}
    by = {}; order = []
    for x in meta.itertuples():
        if x.game_id not in by: order.append(x.game_id)
        by.setdefault(x.game_id, []).append(x)
    model = comp = None; last = None; n = 0
    for gid in order:
        recs = by[gid]
        if len(recs) != 2: continue
        m = recs[0].matchup
        host = m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        h = next((x for x in recs if x.team_abbrev == host), None)
        a = next((x for x in recs if x.team_abbrev != host), None)
        if not h or not a: continue
        gd = h.game_date.date() if hasattr(h.game_date, "date") else h.game_date
        if last is None or (gd - last).days >= 7:
            model = fit_production_legacy(con, season, before=gd, w_comp=0.7)
            comp = CompositionModel(con, before=gd)
            last = gd
        pmv = mkt.get((str(gd)[:10], h.team_abbrev, a.team_abbrev))
        if pmv is None: continue
        outs = {}
        for t in (h.team_id, a.team_id):
            pl = played.get((gid, t), set())
            outs[t] = {p for p, d0 in comp.players.items()
                       if d0["team_id"] == t and (gd - d0["last_played"]).days <= 12
                       and p not in pl}
        p = model.p_home(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id], gd)
        wtr.writerow([season, gid, str(gd)[:10], h.team_abbrev, a.team_abbrev,
                      int(h.wl == "W"), round(p, 6), round(pmv, 5)])
        n += 1
    con.close()
    print(f"{season}: {n}", flush=True)


if __name__ == "__main__":
    with open("data/capstone_pergame_legacy.csv", "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["season", "game_id", "game_date", "home", "away", "y", "p_us", "p_mkt"])
        for s in ("2023-24", "2024-25", "2025-26"):
            season_run(s, wtr)
    print("LEGACY_DONE")
