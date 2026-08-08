#!/usr/bin/env python3
"""Daily prediction entrypoint (the October live path).

For each scheduled NBA game today (nba_api scoreboard):
  * production win-prob (opponent-adjusted ratings + cold-start prior)
  * blend with de-vigged market if lines are logged (blend is for BETTING output
    only — model stays market-blind; handoff I.6)
  * prop distributions (points) for each team's top rotation players

Offseason: no games scheduled -> prints that and exits cleanly. Wire into cron
before the season (after the odds logger, so lines exist).
"""
import datetime as dt
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect
from nbapred.engine.props import (player_rates_from_stats, prop_prob,
                                  simulate_player)
from nbapred.engine import starout
from nbapred.engine.slate import (filter_regular_season, slate_context,
                                  todays_games)

MODEL_WEIGHT = 0.3   # blend: final = w*model + (1-w)*market (shift toward model
                     # inside flagged lineup windows once injury feed is live)


def main(season=None):
    from nbapred.config import current_season
    season = season or current_season()   # Oct 2026 -> '2026-27' (was a
    # hardcoded '2025-26', which would have mislabeled every virtual tank row
    # AND fit the whole production stack on last season at the October opener)
    # D178 FIX 1: REGULAR SEASON ONLY (game_id prefix 002).  The nba_api
    # scoreboard returns preseason games in early October and All-Star games in
    # February; the model is fit and priced on 002 only.
    games = filter_regular_season(todays_games(), where="predict_today")
    if not games:
        print(f"No NBA regular-season games scheduled today "
              f"({dt.date.today()}). "
              "(Offseason — entrypoint is armed for October.)")
        return
    con = connect(read_only=True)
    today = dt.date.today()
    # Shared slate assembly (nbapred/engine/slate.py): tank priming ->
    # fit_production (D73 tank + D84-A October bridge) -> injury-feed OUT
    # sets + departed filter + b2b — ONE code path with the bet engine.
    ctx_slate = slate_context(con, season, games, today)
    model, comp_view = ctx_slate["model"], ctx_slate["comp_view"]
    yday = ctx_slate["b2b"]
    names = dict(con.execute("SELECT player_id, full_name FROM nba_players").fetchall())
    # D82 lean-in: star-out redistribution for props (D33 softmax attempts lift
    # + D39 positional minutes tilt; efficiency untouched per D34). Load the
    # usage prior + position map once for the slate.
    usage_w = starout.load_usage_weights()   # None -> trailing-attempts fallback
    positions = starout.load_positions()
    for gid, hid, aid in games:
        outs = ctx_slate["outs"][gid]
        p = model.p_home(hid, aid, outs.get(hid), outs.get(aid), today,
                         b2b_home=hid in yday, b2b_away=aid in yday)
        tsd = model.tank_diff(hid, aid, today)
        print(f"\n=== game {gid}: P(home win) model={p:.3f} "
              f"(margin {model.margin(hid, aid, outs.get(hid), outs.get(aid), today, hid in yday, aid in yday):+.1f}"
              f", tank {model.tank_k * tsd:+.2f} [k={model.tank_k:+.2f} tsd={tsd:+.2f}])")
        # top-3 rotation players each side: points distribution snapshot
        for tid, side in ((hid, "HOME"), (aid, "AWAY")):
            t_outs = {int(p) for p in outs.get(tid, set())}
            # D82: star-out context for this team (None when no >=28-trailing-
            # min fresh star is in the OUT set -> everything below is a no-op).
            ctx = starout.team_context(con, tid, t_outs, today,
                                       weights=usage_w, positions=positions)
            if ctx:
                print(f"  [{side}] star-out: {names.get(ctx['star'], ctx['star'])} -> "
                      f"attempts x{ctx['lift']:.3f} (residual of softmax "
                      f"x{ctx['lift_softmax']:.3f}, {ctx['usage_source']}, "
                      f"pool {ctx['n_pool']}), minutes +pos-tilt (D33/D39)")
            # D85 roster-delta: trade-aware window for this team (regime-B).
            # DEP = star traded away (executed) -> seller incumbents attenuated
            # (the star-out lift is suppressed inside team_context); ARR =
            # mid-season star arrival -> incumbent usage compression + minutes
            # tilt in games the arriver plays (None when arriver is OUT).
            rd = starout.roster_delta_context(con, tid, today, out_ids=t_outs,
                                              positions=positions)
            if rd and rd.get("dep"):
                d = rd["dep"]
                print(f"  [{side}] roster-delta DEP: "
                      f"{names.get(d['star'], d['star'])} departed (game "
                      f"{d['k'] + 1}/30) -> incumbent attempts "
                      f"x{d['att_factor']:.3f}")
            if rd and rd.get("arr"):
                a = rd["arr"]
                print(f"  [{side}] roster-delta ARR: "
                      f"{names.get(a['arriver'], a['arriver'])} arrived "
                      f"(k={a['k']}) -> incumbent compression "
                      f"(tercile-shielded) + minutes tilt")
            # Rank rotation by seconds in the team's LAST 15 games (all-time
            # totals surfaced departed players / years-old rosters), and keep
            # only players whose CURRENT team (arg_max by date) is this team.
            # D178 FIX 1: both CTEs now carry `game_id LIKE '002%'`.  Without
            # it (a) `recent` could spend slots of the 15-game rotation window
            # on preseason/All-Star box scores, and (b) `current_team`'s
            # arg_max could resolve a player's CURRENT team to an All-Star or
            # exhibition side (team_ids that are not NBA franchises at all —
            # 315 such spine rows, 97 codes), which drops every player on that
            # roster from the printout. Same filter as `star_out_live` and the
            # `last_team` departed filter in slate.py.
            top = con.execute("""
                WITH recent AS (
                  SELECT DISTINCT g.game_id, g.game_date
                  FROM player_game_stats s
                  JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g
                    USING (game_id)
                  WHERE s.team_id=? AND s.game_id LIKE '002%'
                  ORDER BY g.game_date DESC LIMIT 15
                ), current_team AS (
                  SELECT s.player_id, arg_max(s.team_id, g.game_date) ct
                  FROM player_game_stats s
                  JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g
                    USING (game_id)
                  WHERE s.game_id LIKE '002%'
                  GROUP BY 1
                )
                SELECT s.player_id, sum(s.seconds) sec
                FROM player_game_stats s
                JOIN recent r USING (game_id)
                JOIN current_team c ON c.player_id = s.player_id
                WHERE s.team_id=? AND c.ct=?
                GROUP BY 1 ORDER BY 2 DESC LIMIT 12""",
                [tid, tid, tid]).fetchall()
            # remaining players only: never sim (or lift) tonight's OUT players
            top = [(int(p), s) for p, s in top if int(p) not in t_outs][:3]
            for pid, _ in top:
                rates = player_rates_from_stats(con, int(pid))
                if not rates:
                    continue
                # D58: opp-defense shift removed from live path — ledger says
                # NS x4 (unproven); gate it in before October or drop it.
                rates = starout.adjust_player_rates(rates, pid, ctx, positions)
                rates = starout.adjust_player_rates_rd(rates, pid, rd, positions)
                sim = simulate_player(rates, n=4000, seed=pid)["points"]
                med = float(np.median(sim))
                pp = prop_prob(sim, med + 0.5)
                tag = ((" *star-out adj*" if ctx else "") +
                       (" *roster-delta*" if rd else ""))
                print(f"  [{side}] {names.get(pid, pid):22} pts median {med:.0f} "
                      f"P(over {med + 0.5:.1f})={pp['p_over']:.2f}{tag}")
    con.close()


if __name__ == "__main__":
    main()
