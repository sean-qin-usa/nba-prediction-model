#!/usr/bin/env python3
"""Walk-forward component dump for the rejection re-gate battery (Sean's
directive 2026-07-29: retry every rejection with full ability on the 3-season
corpus). One expensive pass records per-game MARGINS for every component;
blend variants are then re-gated offline for free by regate_blends.py.

Per game: y, p_mkt, m_ff (shipped 4F), m_ff60 (60d-recency 4F — the flipped
NS rejection), m_comp (PIT composition), m_ratings (opp-adj + cold-start).
"""
import sys, warnings, csv
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nbapred.db import connect
from nbapred.model.production import fit_production
from nbapred.model.composition import CompositionModel
from nbapred.model.four_factors import FourFactors

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
    prod = comp = ff60 = None; last = None; n = 0
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
            prod = fit_production(con, season, before=gd, w_comp=0.7)
            comp = CompositionModel(con, before=gd)
            ff60 = FourFactors().fit(con, season, before=gd, half_life_days=60.0)
            last = gd
        pmv = mkt.get((str(gd)[:10], h.team_abbrev, a.team_abbrev))
        if pmv is None: continue
        outs = {}
        for t in (h.team_id, a.team_id):
            pl = played.get((gid, t), set())
            outs[t] = {p for p, d0 in comp.players.items()
                       if d0["team_id"] == t and (gd - d0["last_played"]).days <= 12
                       and p not in pl}
        cm = comp.margin(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id], gd)
        # prod.margin's ff lives in a closure; refit here would double cost —
        # recover m_ff from the blend identity instead: blend = .5*fm + .5*cm
        blend = prod.margin(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id], gd)
        rm = prod.ratings_margin(h.team_id, a.team_id)
        fm = 2 * blend - cm if abs(blend - (0.7 * cm + 0.3 * rm)) > 1e-9 else None
        f6 = ff60.margin(h.team_id, a.team_id) if ff60.ready else None
        wtr.writerow([season, gid, str(gd)[:10], h.team_abbrev, a.team_abbrev,
                      int(h.wl == "W"), round(pmv, 5), round(cm, 4), round(rm, 4),
                      None if fm is None else round(fm, 4),
                      None if f6 is None else round(f6, 4)])
        n += 1
    con.close()
    print(f"{season}: {n} games dumped", flush=True)

if __name__ == "__main__":
    with open("data/component_pergame.csv", "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["season", "game_id", "game_date", "home", "away", "y",
                      "p_mkt", "m_comp", "m_ratings", "m_ff", "m_ff60"])
        for s in ("2023-24", "2024-25", "2025-26"):
            season_run(s, wtr)
    print("COMPONENTS_DONE")
