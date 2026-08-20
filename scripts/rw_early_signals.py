"""REGIME A untapped-data signals — per (season, team) PIT-safe October facts.

All signals below are knowable BEFORE opening night (preseason 001 games end
~4-7 days before the opener; prior-season data is archival):

  ps_n, ps_pd        preseason games and mean point differential (001 pairs)
  ps_cont_any        prior-season 002 team-minutes share returning among ALL
                     preseason participants (any 001 minutes)
  ps_cont_last2      ... among players appearing in the team's LAST TWO
                     preseason games (dress-rehearsal roster)
  ps_cont_wt         prior-season minutes weighted by the player's preseason
                     minutes share (soft rotation weighting)
  ps_top5_ret        of the last preseason game's top-5 by minutes, how many
                     were in the PRIOR season's team top-8 by minutes
  rookie_ps_share    share of team preseason minutes to players with NO
                     darko_history entry before Aug 1 (never rated = never
                     played an NBA game -> rookies + camp unknowns)
  rookie_re_share10  REALIZED share of team minutes to those same players in
                     the first 10 regular-season games (diagnostic only,
                     not PIT for prediction)
  first5_cont        the carry's eventual first-5-games continuity
                     (continuity_map with full-season roster info — what the
                     shipped carry converges to after ~2 refits)
  open_cont          continuity using ONLY the opening-night (game-1) roster

Read-only DB; output data/rw_early_signals.csv.
"""
import csv
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nbapred.db import connect
from nbapred.model.production import _prev_season, continuity_map

SEASONS = ("2023-24", "2024-25", "2025-26")


def season_signals(con, season):
    prev = _prev_season(season)
    yr = int(season[:4])
    cutoff = f"{yr}-08-01"

    # prior-season 002 minutes per (team, player)
    pm = con.execute("""
        SELECT s.team_id, s.player_id, sum(s.seconds)/60.0 mins
        FROM player_game_stats s
        JOIN nba_games g ON g.game_id = s.game_id AND g.team_id = s.team_id
        WHERE g.season = ? AND s.game_id LIKE '002%' GROUP BY 1,2""",
        [prev]).fetchall()
    prev_min = {}
    for t, p, m in pm:
        prev_min.setdefault(int(t), {})[int(p)] = float(m)
    prev_tot = {t: sum(d.values()) for t, d in prev_min.items()}
    prev_top8 = {t: set(sorted(d, key=d.get, reverse=True)[:8])
                 for t, d in prev_min.items()}

    # rated-before-cutoff set (veterans)
    rated = {int(p) for (p,) in con.execute(
        "SELECT DISTINCT player_id FROM darko_history WHERE date < ?",
        [cutoff]).fetchall()}

    # preseason participation (001) this season
    ps = con.execute("""
        SELECT s.team_id, s.player_id, s.game_id, g.game_date,
               s.seconds/60.0 mins
        FROM player_game_stats s
        JOIN nba_games g ON g.game_id = s.game_id AND g.team_id = s.team_id
        WHERE g.season = ? AND s.game_id LIKE '001%' AND s.seconds > 0""",
        [season]).fetchall()
    ps_by_team = {}
    for t, p, gid, gd, m in ps:
        ps_by_team.setdefault(int(t), []).append((int(p), gid, gd, float(m)))

    # preseason team point differential from 001 nba_games pairs
    g001 = con.execute("""
        SELECT game_id, team_id, pts FROM nba_games
        WHERE season=? AND game_id LIKE '001%' AND pts IS NOT NULL""",
        [season]).fetchall()
    byg = {}
    for gid, t, pts in g001:
        byg.setdefault(gid, []).append((int(t), int(pts)))
    ps_pd, ps_n = {}, {}
    for gid, recs in byg.items():
        if len(recs) != 2:
            continue
        (t1, p1), (t2, p2) = recs
        for t, own, opp in ((t1, p1, p2), (t2, p2, p1)):
            ps_pd[t] = ps_pd.get(t, 0.0) + (own - opp)
            ps_n[t] = ps_n.get(t, 0) + 1
    ps_pd = {t: ps_pd[t] / ps_n[t] for t in ps_pd}

    # regular-season first-10 games minutes (realized rookie share diagnostic)
    rs = con.execute("""
        WITH tg AS (
          SELECT team_id, game_id,
                 row_number() OVER (PARTITION BY team_id
                                    ORDER BY game_date, game_id) rn
          FROM nba_games WHERE season = ? AND game_id LIKE '002%')
        SELECT tg.team_id, s.player_id, sum(s.seconds)/60.0 mins
        FROM tg JOIN player_game_stats s
          ON s.game_id = tg.game_id AND s.team_id = tg.team_id
        WHERE tg.rn <= 10 GROUP BY 1, 2""", [season]).fetchall()
    rs_by_team = {}
    for t, p, m in rs:
        rs_by_team.setdefault(int(t), {})[int(p)] = float(m)

    # opening-night (game 1) rosters
    open_ros = {}
    for t, p in con.execute("""
        WITH tg AS (
          SELECT team_id, game_id,
                 row_number() OVER (PARTITION BY team_id
                                    ORDER BY game_date, game_id) rn
          FROM nba_games WHERE season = ? AND game_id LIKE '002%')
        SELECT tg.team_id, s.player_id
        FROM tg JOIN player_game_stats s
          ON s.game_id = tg.game_id AND s.team_id = tg.team_id
        WHERE tg.rn = 1 AND s.seconds > 0 GROUP BY 1, 2""",
            [season]).fetchall():
        open_ros.setdefault(int(t), set()).add(int(p))

    first5 = continuity_map(con, season) or {}

    ab = dict(con.execute(
        "SELECT DISTINCT team_id, team_abbrev FROM nba_games WHERE season=? "
        "AND game_id LIKE '002%'", [season]).fetchall())

    out = []
    for t in sorted(ab):
        plays = ps_by_team.get(t, [])
        pmins = {}
        for p, gid, gd, m in plays:
            pmins[p] = pmins.get(p, 0.0) + m
        tot_ps = sum(pmins.values())
        # last two preseason games (by date, then id)
        gids = sorted({(gd, gid) for _, gid, gd, _ in plays})
        last2 = {gid for _, gid in gids[-2:]}
        last1 = gids[-1][1] if gids else None
        ros_any = set(pmins)
        ros_last2 = {p for p, gid, _, _ in plays if gid in last2}
        # top-5 by minutes in the final preseason game
        lastg = [(p, m) for p, gid, _, m in plays if gid == last1]
        top5_last = {p for p, _ in sorted(lastg, key=lambda x: -x[1])[:5]}

        pmv = prev_min.get(t, {})
        ptot = prev_tot.get(t, 0.0)

        def cont(roster):
            if not ptot:
                return None
            return sum(m for p, m in pmv.items() if p in roster) / ptot

        cont_wt = (sum(pmv.get(p, 0.0) * min(1.0, pmins[p] / 48.0)
                       for p in pmins) / ptot if ptot else None)
        rook_ps = (sum(m for p, m in pmins.items() if p not in rated) / tot_ps
                   if tot_ps else None)
        rsm = rs_by_team.get(t, {})
        rs_tot = sum(rsm.values())
        rook_rs = (sum(m for p, m in rsm.items() if p not in rated) / rs_tot
                   if rs_tot else None)
        out.append(dict(
            season=season, team_id=t, team=ab[t],
            ps_n=ps_n.get(t, 0),
            ps_pd=round(ps_pd.get(t, 0.0), 3),
            ps_cont_any=round(cont(ros_any), 4) if ptot else "",
            ps_cont_last2=round(cont(ros_last2), 4) if ptot else "",
            ps_cont_wt=round(cont_wt, 4) if ptot else "",
            ps_top5_ret=len(top5_last & prev_top8.get(t, set())),
            rookie_ps_share=round(rook_ps, 4) if rook_ps is not None else "",
            rookie_rs_share10=round(rook_rs, 4) if rook_rs is not None else "",
            first5_cont=round(first5.get(t, float("nan")), 4),
            open_cont=round(cont(open_ros.get(t, set())), 4) if ptot else "",
            n_ps_players=len(ros_any),
            n_ps_rookie_players=len([p for p in ros_any if p not in rated])))
    return out


def main():
    con = connect(read_only=True)
    rows = []
    for s in SEASONS:
        rows += season_signals(con, s)
        print(f"[{s}] {len(rows)} cumulative team rows", flush=True)
    con.close()
    with open(REPO / "data" / "rw_early_signals.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", REPO / "data" / "rw_early_signals.csv")


if __name__ == "__main__":
    main()
