"""Observation stream for the M1 team DLM (read-only DB access).

One observation per completed game, chronological:
    (date, kind, home_key, away_key, y1, y2)
      kind='eff'    y1, y2 = (ortg_home, ortg_away) per 100 possessions —
                    built from player_game_stats box sums (game_rows math,
                    poss = fga + 0.44*fta - oreb + tov, <50 poss dropped);
                    regular-season (002) games, available 2022-23+.
      kind='margin' y1 = home score margin (pts ~ ortg-margin scale) — from
                    odds_market finals for (a) seasons before possession data
                    exists and (b) playoff/play-in rows (real signal for the
                    NEXT season's carry). Never duplicates an 'eff' game.

Team key = abbreviation (stable across the filter window; odds_market and
nba_games agree on all 30 for 2021+).
"""
from __future__ import annotations

import datetime as dt


def season_boundaries(con, start: dt.date, end: dt.date) -> list[dt.date]:
    """First regular-season game date per season in [start, end] — the
    season-boundary shock dates."""
    rows = con.execute("""
        SELECT season, min(game_date) FROM nba_games
        WHERE game_id LIKE '002%' AND game_date BETWEEN ? AND ?
        GROUP BY 1 ORDER BY 2""", [start, end]).fetchall()
    return [r[1] for r in rows]


def eff_obs(con, start: dt.date, end: dt.date) -> dict:
    """{(date, home_ab, away_ab): (date,'eff',home,away,ortg_h,ortg_a)} for
    002 games with possession data."""
    df = con.execute("""
        SELECT s.game_id, s.team_id,
               sum(s.pts) pts, sum(s.fga) fga, sum(s.fta) fta,
               sum(s.oreb) oreb, sum(s.tov) tov,
               any_value(g.matchup) matchup, any_value(g.team_abbrev) abbr,
               any_value(g.game_date) gdate
        FROM player_game_stats s
        JOIN nba_games g ON g.game_id = s.game_id AND g.team_id = s.team_id
        WHERE s.game_id LIKE '002%' AND g.game_date BETWEEN ? AND ?
        GROUP BY s.game_id, s.team_id""", [start, end]).fetchdf()
    by = {}
    for r in df.itertuples():
        by.setdefault(r.game_id, []).append(r)
    out = {}
    for gid, recs in by.items():
        if len(recs) != 2:
            continue
        m = recs[0].matchup
        host = m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        h = next((x for x in recs if x.abbr == host), None)
        a = next((x for x in recs if x.abbr != host), None)
        if h is None or a is None:
            continue
        ph = h.fga + 0.44 * h.fta - h.oreb + h.tov
        pa = a.fga + 0.44 * a.fta - a.oreb + a.tov
        if ph < 50 or pa < 50:
            continue
        d = h.gdate.date() if hasattr(h.gdate, "date") else h.gdate
        out[(d, h.abbr, a.abbr)] = (d, "eff", h.abbr, a.abbr,
                                    100.0 * float(h.pts) / ph,
                                    100.0 * float(a.pts) / pa)
    return out


def margin_obs(con, start: dt.date, end: dt.date) -> dict:
    """{(date, home, away): (date,'margin',home,away,margin,None)} from
    odds_market finals (includes playoffs; scores only, no market fields —
    G2 stays intact: the DLM never reads a price)."""
    rows = con.execute("""
        SELECT game_date, home, away, score_home - score_away
        FROM odds_market
        WHERE game_date BETWEEN ? AND ?
          AND score_home IS NOT NULL AND score_away IS NOT NULL""",
        [start, end]).fetchall()
    out = {}
    for d, h, a, m in rows:
        out[(d, h, a)] = (d, "margin", h, a, float(m), None)
    return out


def build_team_obs(con, start: dt.date, end: dt.date) -> list[tuple]:
    """Merged chronological stream: 'eff' preferred, 'margin' fills the rest."""
    eff = eff_obs(con, start, end)
    mar = margin_obs(con, start, end)
    merged = dict(mar)
    merged.update(eff)                       # eff wins on key collision
    return sorted(merged.values(), key=lambda o: (o[0], o[2], o[3]))


def team_keys(obs) -> list[str]:
    return sorted({o[2] for o in obs} | {o[3] for o in obs})
