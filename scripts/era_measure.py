#!/usr/bin/env python3
"""ERA MEASUREMENT — the data-measured backing for docs/ERAS.md.

Every era boundary claimed in ERAS.md must be backed by a number computed here,
not by a recollection of NBA news. READ-ONLY on data/nba.duckdb.

Signatures measured per season (regular season, game_id LIKE '002%'):
  * home margin, home win rate            (the D131/D70 channel)
  * travel km per team-game, |tz| crossings, b2b / 3in4 rate, mean days rest
  * repeat-opponent rate                  (the 2020-21 series-style schedule)
  * pace proxy (possessions/game), pts/game, 3PA share
  * absence: players used per team-game, CORE-player DNP rate
  * star-DNP rate by month                (Omicron Dec-2021; PPP 2023-24)
  * schedule structure: n games, span, NBA-Cup marker (game_id 006)

Output: data/era_signatures.json  (+ stdout table)
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nbapred.model import travel as TV  # noqa: E402

DB = ROOT / "data" / "nba.duckdb"
OUT = ROOT / "data" / "era_signatures.json"

BUBBLE_LO, BUBBLE_HI = "2020-07-30", "2020-10-11"


def connect(retry_s: int = 60):
    t0 = time.time()
    while True:
        try:
            return duckdb.connect(str(DB), read_only=True)
        except Exception:
            if time.time() - t0 > retry_s:
                raise
            time.sleep(2)


def main() -> None:
    con = connect()
    res: dict = {}

    # ---------------------------------------------------------------- games
    games = con.execute("""
        SELECT season, game_id, game_date,
               max(CASE WHEN is_home THEN team_abbrev END)  AS home,
               max(CASE WHEN NOT is_home THEN team_abbrev END) AS away,
               max(CASE WHEN is_home THEN pts END)          AS pts_h,
               max(CASE WHEN NOT is_home THEN pts END)      AS pts_a,
               max(CASE WHEN is_home THEN team_id END)      AS tid_h,
               max(CASE WHEN NOT is_home THEN team_id END)  AS tid_a,
               max(matchup)                                 AS mu
        FROM nba_games WHERE game_id LIKE '002%'
        GROUP BY 1,2,3 ORDER BY 3,2
    """).fetchall()

    by_season = defaultdict(list)
    for r in games:
        by_season[r[0]].append(r)

    # ------------------------------------------------------- travel state
    state = TV.build_state(con)

    # ------------------------------------------------- team-game sequences
    seq = defaultdict(list)   # (season, team_abbrev) -> [(date, opp, is_home)]
    rows = con.execute("""
        SELECT season, game_id, game_date, team_abbrev, is_home, team_id, matchup
        FROM nba_games WHERE game_id LIKE '002%' ORDER BY game_date, game_id
    """).fetchall()
    gteams = defaultdict(list)
    for season, gid, gd, ab, ish, tid, mu in rows:
        gteams[gid].append((season, gd, ab, ish, tid))
    for gid, recs in gteams.items():
        if len(recs) != 2:
            continue
        (s0, d0, a0, h0, t0), (s1, d1, a1, h1, t1) = recs
        seq[(s0, a0)].append((d0, a1, h0, t0))
        seq[(s1, a1)].append((d1, a0, h1, t1))
    for k in seq:
        seq[k].sort()

    # -------------------------------------------------- box-score aggregates
    box = con.execute("""
        SELECT g.season, p.game_id, p.team_id,
               sum(p.pts) pts, sum(p.fga) fga, sum(p.fta) fta,
               sum(p.oreb) oreb, sum(p.tov) tov, sum(p.fg3a) fg3a,
               count(*) n_used, sum(CASE WHEN p.seconds > 0 THEN 1 ELSE 0 END) n_played
        FROM player_game_stats p
        JOIN (SELECT DISTINCT game_id, season FROM nba_games WHERE game_id LIKE '002%') g
          USING (game_id)
        GROUP BY 1,2,3
    """).fetchall()
    box_by_season = defaultdict(list)
    for r in box:
        box_by_season[r[0]].append(r)

    # ------------------------------------------------------ CORE-player set
    # CORE = played >=20 games in the season at >=28.0 mpg (1680 s).  Assigned
    # to the team he played the most games for that season.  In-season, not PIT
    # — this is a DESCRIPTIVE era signature, never a model feature.
    core = con.execute("""
        WITH ps AS (
          SELECT g.season, p.player_id, p.team_id, count(*) gp, avg(p.seconds) sec
          FROM player_game_stats p
          JOIN (SELECT DISTINCT game_id, season FROM nba_games WHERE game_id LIKE '002%') g
            USING (game_id)
          WHERE p.seconds > 0
          GROUP BY 1,2,3
        ), agg AS (
          SELECT season, player_id, sum(gp) gp, sum(gp*sec)/sum(gp) sec,
                 arg_max(team_id, gp) team_id
          FROM ps GROUP BY 1,2
        )
        SELECT season, player_id, team_id, gp, sec FROM agg
        WHERE gp >= 20 AND sec >= 1680
    """).fetchall()
    core_by_team = defaultdict(list)     # (season, team_id) -> [player_id]
    for season, pid, tid, gp, sec in core:
        core_by_team[(season, tid)].append(pid)

    played = defaultdict(set)            # (game_id, team_id) -> {player_id played}
    for gid, tid, pid in con.execute("""
        SELECT p.game_id, p.team_id, p.player_id
        FROM player_game_stats p
        JOIN (SELECT DISTINCT game_id FROM nba_games WHERE game_id LIKE '002%') g
          USING (game_id)
        WHERE p.seconds > 0
    """).fetchall():
        played[(gid, tid)].add(pid)

    box_games = {r[1] for r in box}      # games with any box coverage

    # ---------------------------------------------------------- per season
    per_season = {}
    per_month = {}
    for season in sorted(by_season):
        gs = by_season[season]
        n = len(gs)
        marg = [r[5] - r[6] for r in gs if r[5] is not None and r[6] is not None]
        hw = [1.0 if (r[5] or 0) > (r[6] or 0) else 0.0 for r in gs]
        dates = sorted(r[2] for r in gs)

        # travel / density over team-games
        tk, tz, b2b, d34, rest, repeat = [], [], [], [], [], []
        for (sn, ab), lst in seq.items():
            if sn != season:
                continue
            prev_opp, prev_date = None, None
            for (d, opp, ish, tid) in lst:
                st = state.get((tid, d.date() if hasattr(d, "date") else d))
                if st:
                    tk.append(st["travel_km"])
                    tz.append(abs(st["tz_east"]))
                    b2b.append(1.0 if st["b2b"] else 0.0)
                    d34.append(st["is_3in4"])
                if prev_date is not None:
                    rest.append((d - prev_date).days)
                    repeat.append(1.0 if opp == prev_opp else 0.0)
                prev_opp, prev_date = opp, d

        bx = box_by_season.get(season, [])
        poss = [r[4] + 0.44 * r[5] - r[6] + r[7] for r in bx]
        pts = [r[3] for r in bx]
        fg3a = [r[8] for r in bx]
        fga = [r[4] for r in bx]
        nplay = [r[10] for r in bx]

        # CORE DNP rate
        exp_ct = miss_ct = 0
        for r in gs:
            gid = r[1]
            if gid not in box_games:
                continue
            for tid in (r[7], r[8]):
                cs = core_by_team.get((season, tid), [])
                pl = played.get((gid, tid), set())
                exp_ct += len(cs)
                miss_ct += sum(1 for p in cs if p not in pl)

        per_season[season] = dict(
            n_games=n,
            date_lo=str(dates[0])[:10], date_hi=str(dates[-1])[:10],
            box_games=sum(1 for r in gs if r[1] in box_games),
            home_margin=round(sum(marg) / len(marg), 4) if marg else None,
            home_margin_se=round((sum((m - sum(marg) / len(marg)) ** 2 for m in marg)
                                  / (len(marg) - 1) / len(marg)) ** 0.5, 4) if len(marg) > 2 else None,
            home_wr=round(sum(hw) / len(hw), 4),
            travel_km_per_tg=round(sum(tk) / len(tk), 1) if tk else None,
            tz_abs_per_tg=round(sum(tz) / len(tz), 4) if tz else None,
            b2b_rate=round(sum(b2b) / len(b2b), 4) if b2b else None,
            is3in4_rate=round(sum(d34) / len(d34), 4) if d34 else None,
            mean_rest_days=round(sum(rest) / len(rest), 3) if rest else None,
            rest_ge3_rate=round(sum(1 for x in rest if x >= 3) / len(rest), 4) if rest else None,
            repeat_opp_rate=round(sum(repeat) / len(repeat), 4) if repeat else None,
            poss_per_tg=round(sum(poss) / len(poss), 2) if poss else None,
            pts_per_tg=round(sum(pts) / len(pts), 2) if pts else None,
            fg3a_share=round(sum(fg3a) / sum(fga), 4) if fga else None,
            players_used_per_tg=round(sum(nplay) / len(nplay), 3) if nplay else None,
            core_n=len({p for (s, t), ps in core_by_team.items() if s == season for p in ps}),
            core_dnp_rate=round(miss_ct / exp_ct, 4) if exp_ct else None,
            core_expected=exp_ct,
        )

        # monthly absence + margin
        mm = defaultdict(lambda: dict(exp=0, miss=0, marg=[], n=0))
        for r in gs:
            gid, gd = r[1], r[2]
            ym = str(gd)[:7]
            slot = mm[ym]
            slot["n"] += 1
            if r[5] is not None and r[6] is not None:
                slot["marg"].append(r[5] - r[6])
            if gid in box_games:
                for tid in (r[7], r[8]):
                    cs = core_by_team.get((season, tid), [])
                    pl = played.get((gid, tid), set())
                    slot["exp"] += len(cs)
                    slot["miss"] += sum(1 for p in cs if p not in pl)
        per_month[season] = {
            ym: dict(n=v["n"],
                     core_dnp_rate=round(v["miss"] / v["exp"], 4) if v["exp"] else None,
                     home_margin=round(sum(v["marg"]) / len(v["marg"]), 3) if v["marg"] else None)
            for ym, v in sorted(mm.items())
        }

    res["per_season"] = per_season
    res["per_month"] = per_month

    # ------------------------------------------- weekly absence (Omicron)
    # ISO-week core-DNP rate, so the Dec-2021 boundary is set by the data
    # rather than by "December 2021" as a round number.
    weekly = {}
    for season in sorted(by_season):
        wk = defaultdict(lambda: dict(exp=0, miss=0, n=0))
        for r in by_season[season]:
            gid, gd = r[1], r[2]
            d = gd.date() if hasattr(gd, "date") else gd
            iso = d.isocalendar()
            key = f"{iso[0]}-W{iso[1]:02d}"
            wk[key]["n"] += 1
            if gid in box_games:
                for tid in (r[7], r[8]):
                    cs = core_by_team.get((season, tid), [])
                    pl = played.get((gid, tid), set())
                    wk[key]["exp"] += len(cs)
                    wk[key]["miss"] += sum(1 for p in cs if p not in pl)
        weekly[season] = {k: dict(n=v["n"],
                                  core_dnp=round(v["miss"] / v["exp"], 4) if v["exp"] else None)
                          for k, v in sorted(wk.items())}
    res["weekly_core_dnp"] = weekly

    # ------------------------------------------------------- bubble stratum
    bub = [r for r in by_season.get("2019-20", []) if BUBBLE_LO <= str(r[2])[:10] <= BUBBLE_HI]
    pre = [r for r in by_season.get("2019-20", []) if str(r[2])[:10] < "2020-03-12"]
    hiatus = [r for r in by_season.get("2019-20", [])
              if "2020-03-12" <= str(r[2])[:10] < BUBBLE_LO]

    def strat(rows_):
        if not rows_:
            return None
        marg = [r[5] - r[6] for r in rows_ if r[5] is not None]
        hw = [1.0 if (r[5] or 0) > (r[6] or 0) else 0.0 for r in rows_]
        tk = []
        for r in rows_:
            gd = r[2].date() if hasattr(r[2], "date") else r[2]
            for tid in (r[7], r[8]):
                st = state.get((tid, gd))
                if st:
                    tk.append(st["travel_km"])
        return dict(n=len(rows_),
                    home_margin=round(sum(marg) / len(marg), 4),
                    home_wr=round(sum(hw) / len(hw), 4),
                    travel_km_per_tg_ASCONSTRUCTED=round(sum(tk) / len(tk), 1) if tk else None,
                    box_games=sum(1 for r in rows_ if r[1] in box_games))

    res["covid_strata"] = dict(pre_shutdown_2019_20=strat(pre),
                               hiatus_games=strat(hiatus),
                               bubble=strat(bub))

    # ------------------------------------------------- NBA Cup / IST marker
    res["nba_cup_006"] = {s: c for s, c in con.execute(
        "SELECT season, count(DISTINCT game_id) FROM nba_games "
        "WHERE game_id LIKE '006%' GROUP BY 1 ORDER BY 1").fetchall()}

    # ------------------------------------------- scorability probe (D131)
    res["scorability"] = {}
    for s in sorted(by_season):
        tot = len(by_season[s])
        cov = sum(1 for r in by_season[s] if r[1] in box_games)
        res["scorability"][s] = dict(games=tot, box_covered=cov,
                                     coverage=round(cov / tot, 4))

    # ------------------------------------------- injury-feed / inactives cov
    res["feed_coverage"] = dict(
        injury_reports_pit=[dict(ym=r[0], n=r[1]) for r in con.execute(
            "SELECT strftime(game_date,'%Y-%m'), count(*) FROM injury_reports_pit "
            "GROUP BY 1 ORDER BY 1").fetchall()],
        game_inactives_by_season=[dict(season=r[0], games=r[1], rows=r[2]) for r in con.execute(
            "SELECT g.season, count(DISTINCT i.game_id), count(*) FROM game_inactives i "
            "JOIN (SELECT DISTINCT game_id, season FROM nba_games) g USING (game_id) "
            "GROUP BY 1 ORDER BY 1").fetchall()],
    )

    OUT.write_text(json.dumps(res, indent=1, default=str))

    hdr = ("season   n   HMarg  HWR    trvkm  b2b   3in4  rest  rep   poss   pts   "
           "used  coreDNP")
    print(hdr)
    for s, v in per_season.items():
        print(f"{s} {v['n_games']:4d} {v['home_margin']:+6.3f} {v['home_wr']:.4f} "
              f"{v['travel_km_per_tg'] or 0:6.1f} {v['b2b_rate'] or 0:.3f} "
              f"{v['is3in4_rate'] or 0:.3f} {v['mean_rest_days'] or 0:.2f} "
              f"{v['repeat_opp_rate'] or 0:.3f} {v['poss_per_tg'] or 0:6.2f} "
              f"{v['pts_per_tg'] or 0:6.2f} {v['players_used_per_tg'] or 0:5.2f} "
              f"{v['core_dnp_rate'] if v['core_dnp_rate'] is not None else float('nan'):.4f}")
    print()
    print("COVID strata:", json.dumps(res["covid_strata"], indent=1))
    print("NBA Cup (006) by season:", res["nba_cup_006"])
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
