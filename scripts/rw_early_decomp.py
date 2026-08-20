"""REGIME A decomposition — early-season (either team gp<20) component dump.

Replicates the SHIPPED production loop (prod_by_season.py caller x
fit_production internals, D62 carry + D46 sched; D73 tank term is exactly 0
in this window, asserted against the shipped CSV) and records per-game
COMPONENT margins so the early loss vs market can be attributed:

  ff-ready:   m = 0.5*fm + 0.5*cm + sched            (fm carry-seeded while
                                                      cur rows < 200)
  pre-ready:  m = 0.7*cm + 0.3*(rm_core + prior_term) + sched

Dumped per game: cm (comp, home_edge=0), fm (ff margin_neutral or NaN),
rm_core (ratings - tr.home), prior_term (wh*ph - wa*pa), sched, m_us, p_us,
carry_active, n_cur_rows, plus actual scores. Per refit: the continuity map
actually used (task 2b: opening-night DEFAULT-continuity quantification).

Fidelity check: p_us must match data/capstone_pergame_tank.csv to <1e-9 on
every covered game (same weekly cadence, same oracle outs, same b2b flags).

Read-only DB; outputs to data/rw_early_decomp_pergame.csv + refits json.
"""
import csv
import datetime as dt
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from nbapred.db import connect
from nbapred.model.composition import CompositionModel
from nbapred.model.four_factors import FourFactors, factor_game_rows
from nbapred.model.production import (CARRY_CONT_DEFAULT, CARRY_W0, SCALE,
                                      _prev_season, continuity_map,
                                      fit_schedule_layer, last_season_prior,
                                      sigmoid)
from nbapred.model.team_ratings import TeamRatings, game_rows

SEASONS = ("2023-24", "2024-25", "2025-26")
W_COMP = 0.7
OUT = REPO / "data"


def season_run(season):
    t0 = time.time()
    con = connect(read_only=True)
    pm = con.execute("""SELECT game_id, team_id, player_id FROM player_game_stats
        WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    played = {(g, t): set(grp.player_id)
              for (g, t), grp in pm.groupby(["game_id", "team_id"])}
    meta = con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl,
        game_date, pts FROM nba_games
        WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL
        ORDER BY game_date""", [season]).fetchdf()
    mkt = {(str(r[0])[:10], r[1], r[2]): r[3] for r in con.execute(
        "SELECT game_date, home, away, p_home_spread FROM odds_market WHERE season_end=?",
        [int(season[:4]) + 1]).fetchall()}
    by, order = {}, []
    for x in meta.itertuples():
        if x.game_id not in by:
            order.append(x.game_id)
        by.setdefault(x.game_id, []).append(x)
    tdates = {}
    for x in meta.itertuples():
        d = x.game_date.date() if hasattr(x.game_date, "date") else x.game_date
        tdates.setdefault(x.team_id, set()).add(d)

    def b2b(tid, d):
        return (d - dt.timedelta(days=1)) in tdates.get(tid, set())

    prev_rows = factor_game_rows(con, _prev_season(season), before=None)
    ab = dict(con.execute(
        "SELECT DISTINCT team_id, team_abbrev FROM nba_games WHERE season=?",
        [season]).fetchall())
    id2ab = {t: a for t, a in ab.items()}
    nteam = len(id2ab)

    rows, refit_log = [], []
    gp_live = {}
    comp = ff = tr = None
    sched5 = None
    prior = None
    games_played = {}
    cont_used = {}
    n_cur = 0
    carry_active = False
    last = None
    for gid in order:
        recs = by[gid]
        if len(recs) != 2:
            continue
        m = recs[0].matchup
        host = m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        h = next((x for x in recs if x.team_abbrev == host), None)
        a = next((x for x in recs if x.team_abbrev != host), None)
        if not h or not a:
            continue
        gd = h.game_date.date() if hasattr(h.game_date, "date") else h.game_date
        # stop entirely once no early game can occur (all teams gp>=20)
        if (len(gp_live) == nteam
                and min(gp_live.values()) >= 20):
            break
        if last is None or (gd - last).days >= 7:
            comp = CompositionModel(con, before=gd)
            cont_used = continuity_map(con, season, before=gd) or {}
            carry_rows = carry_w = None
            if cont_used:
                if prev_rows:
                    carry_rows = prev_rows
                    carry_w = [CARRY_W0 * cont_used.get(x["tid"], CARRY_CONT_DEFAULT)
                               for x in prev_rows]
            n_cur = len(factor_game_rows(con, season, before=gd))
            ff = FourFactors().fit(con, season, before=gd,
                                   carry_rows=carry_rows, carry_weights=carry_w)
            carry_active = bool(n_cur < 200 and carry_rows)
            sched5 = fit_schedule_layer(con, before=gd)
            tr = TeamRatings(ridge=25.0).fit(
                game_rows(con, before=gd, season=season))
            prior = last_season_prior(con, season)
            games_played = dict(con.execute("""
                SELECT team_id, count(*) FROM nba_games WHERE season=? AND
                game_id LIKE '002%' AND wl IS NOT NULL AND game_date < ?
                GROUP BY 1""", [season, gd]).fetchall())
            # log the continuity actually used at this refit
            refit_log.append(dict(
                season=season, date=str(gd), n_cur_rows=n_cur,
                carry_active=carry_active, ff_ready=bool(ff.ready),
                n_teams_with_roster_cont=len([1 for t in cont_used
                                              if cont_used.get(t) is not None]),
                cont={id2ab.get(t, str(t)): round(v, 4)
                      for t, v in sorted(cont_used.items())}))
            last = gd
        gph = gp_live.get(h.team_id, 0)
        gpa = gp_live.get(a.team_id, 0)
        gp_live[h.team_id] = gph + 1
        gp_live[a.team_id] = gpa + 1
        if gph >= 20 and gpa >= 20:
            continue
        pmv = mkt.get((str(gd)[:10], h.team_abbrev, a.team_abbrev))
        if pmv is None:
            continue
        outs = {}
        for t in (h.team_id, a.team_id):
            pl = played.get((gid, t), set())
            outs[t] = {p for p, d0 in comp.players.items()
                       if d0["team_id"] == t
                       and (gd - d0["last_played"]).days <= 12 and p not in pl}
        b2bh, b2ba = b2b(h.team_id, gd), b2b(a.team_id, gd)
        he, b_hb2b, b_ab2b = sched5[0], sched5[1], sched5[2]
        sch = he + (b_hb2b if b2bh else 0.0) + (b_ab2b if b2ba else 0.0)
        cm = comp.margin(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id],
                         gd, home_edge=0.0)
        rm_core = tr.pred_margin(h.team_id, a.team_id) - tr.home
        gh = games_played.get(h.team_id, 0)
        ga = games_played.get(a.team_id, 0)
        wh = max(0.0, 1 - gh / 20.0)
        wa = max(0.0, 1 - ga / 20.0)
        prior_term = (wh * prior.get(id2ab.get(h.team_id, ""), 0.0)
                      - wa * prior.get(id2ab.get(a.team_id, ""), 0.0))
        if ff.ready:
            fm = ff.margin_neutral(h.team_id, a.team_id)
            m_us = 0.5 * fm + 0.5 * cm + sch
        else:
            fm = float("nan")
            m_us = W_COMP * cm + (1 - W_COMP) * (rm_core + prior_term) + sch
        rows.append(dict(
            season=season, game_id=gid, game_date=str(gd)[:10],
            home=h.team_abbrev, away=a.team_abbrev, y=int(h.wl == "W"),
            pts_h=int(h.pts), pts_a=int(a.pts), am=int(h.pts - a.pts),
            p_mkt=float(pmv), gp_home=gph, gp_away=gpa,
            ff_ready=int(ff.ready), carry_active=int(carry_active),
            n_cur_rows=n_cur, n_out_home=len(outs[h.team_id]),
            n_out_away=len(outs[a.team_id]), b2b_h=int(b2bh), b2b_a=int(b2ba),
            cm=round(cm, 6), fm=(round(fm, 6) if fm == fm else ""),
            rm_core=round(rm_core, 6), prior_term=round(prior_term, 6),
            sched=round(sch, 6),
            cont_h=round(cont_used.get(h.team_id, float("nan")), 4)
            if cont_used else "",
            cont_a=round(cont_used.get(a.team_id, float("nan")), 4)
            if cont_used else "",
            m_us=round(m_us, 6), p_us=float(sigmoid(m_us / SCALE))))
    con.close()
    print(f"[{season}] early games={len(rows)} refits={len(refit_log)} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return rows, refit_log


def main():
    all_rows, logs = [], []
    for s in SEASONS:
        r, lg = season_run(s)
        all_rows += r
        logs += lg
    with open(OUT / "rw_early_decomp_pergame.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    with open(OUT / "rw_early_decomp_refits.json", "w") as f:
        json.dump(logs, f, indent=1)

    # fidelity check vs shipped capstone
    base = {}
    with open(REPO / "data" / "capstone_pergame_tank.csv") as f:
        for r in csv.DictReader(f):
            base[(r["season"], r["game_id"])] = (float(r["p_us"]),
                                                 float(r["tsd"]))
    diffs, tsd_bad, missing = [], 0, 0
    for r in all_rows:
        k = (r["season"], r["game_id"])
        if k not in base:
            missing += 1
            continue
        pb, tsd = base[k]
        diffs.append(abs(pb - r["p_us"]))
        if tsd != 0.0:
            tsd_bad += 1
    print(json.dumps(dict(n=len(all_rows), matched=len(diffs),
                          missing=missing, tank_active_in_window=tsd_bad,
                          max_abs_diff=max(diffs) if diffs else None)))


if __name__ == "__main__":
    main()
