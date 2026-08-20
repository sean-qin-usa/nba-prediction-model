"""Four-factors margin model (production). Gate-passed +0.0178 (2025-26) and
replicated +0.0078 (2024-25); 8-factor extension tied -> parsimony keeps 4.

Per factor (eFG, TOV rate, OREB rate, FT rate): an opponent-adjusted ridge
(TeamRatings reused, factor scaled x100). Factors -> expected ortg via a linear
map fit on the SAME trailing games. margin = eortg(home) - eortg(away).
"""
from __future__ import annotations

import numpy as np

from .team_ratings import TeamRatings

FACTORS = ["efg", "tovr", "orbr", "ftr"]


def factor_game_rows(con, season: str, before=None):
    date_clause = "AND g.game_date < ?" if before else ""
    params = [season] + ([before] if before else [])
    df = con.execute(f"""SELECT s.game_id, s.team_id, g.game_date, g.matchup, g.team_abbrev,
        sum(s.fgm) fgm, sum(s.fga) fga, sum(s.thrm) thrm, sum(s.thra) thra, sum(s.tov) tov,
        sum(s.oreb) oreb, sum(s.dreb) dreb, sum(s.fta) fta, sum(s.pts) pts
        FROM player_game_stats s JOIN nba_games g ON g.game_id=s.game_id AND g.team_id=s.team_id
        WHERE g.season=? AND s.game_id LIKE '002%' {date_clause}
        GROUP BY 1,2,3,4,5
        ORDER BY 1,2""", params).fetchdf()   # D230: pin order — see below
    by = {}
    for r in df.itertuples():
        by.setdefault(r.game_id, []).append(r)
    rows = []
    for gid, recs in by.items():
        if len(recs) != 2:
            continue
        m = recs[0].matchup
        host = m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        h = next((x for x in recs if x.team_abbrev == host), None)
        a = next((x for x in recs if x.team_abbrev != host), None)
        if not h or not a:
            continue
        for t, o, hm in ((h, a, True), (a, h, False)):
            poss = t.fga + 0.44 * t.fta - t.oreb + t.tov
            if poss < 50:
                continue
            rows.append(dict(
                tid=int(t.team_id), oid=int(o.team_id), home=hm,
                date=(t.game_date.date() if hasattr(t.game_date, "date")
                      else t.game_date),
                efg=(t.fgm + 0.5 * t.thrm) / t.fga, tovr=t.tov / poss,
                orbr=t.oreb / (t.oreb + o.dreb), ftr=t.fta / t.fga,
                thrm=t.thrm, thra=t.thra, pts=t.pts, fga=t.fga, poss=poss,
                ortg=100 * t.pts / poss))
    return rows


class FourFactors:
    def __init__(self, ridge: float = 25.0):
        self.ridge = ridge
        self.fms = {}
        self.W = None

    def fit(self, con, season: str, before=None, half_life_days: float | None = None,
            luck_adjust_3p: bool = False,
            carry_rows: list | None = None, carry_weights: list | None = None):
        """half_life_days: optional recency weighting (rejection re-review —
        the 60d retest flipped direction but was NS on 2 seasons; re-gated on
        3). Weights apply to the per-factor ridges AND the factor->ortg map.

        carry_rows/carry_weights (D62, gate-passed +0.00097 pooled, +0.0039
        early): prior-season factor rows as pseudo-observations, weighted
        0.3 x roster-continuity — used ONLY while the current season has <200
        rows (hard stop: the moment current rows suffice, carry is ignored
        entirely; contamination structurally impossible)."""
        rows = factor_game_rows(con, season, before)
        carry_active = False
        if len(rows) < 200:
            if not carry_rows:
                return self
            carry_active = True
            n_cur = len(rows)
            rows = list(carry_rows) + rows
            w_carry = np.array(list(carry_weights) + [1.0] * n_cur, float)
        if luck_adjust_3p:
            # D-research #1: opponent 3P% is ~noise year-over-year; defenses
            # control attempts/location, not conversion. Replace realized 3PM
            # with league-avg% x 3PA in the ridge TARGETS (efg + ortg), so
            # neither offense hot streaks nor defense "3P luck allowed"
            # contaminate the fitted ratings. League avg from the train rows
            # themselves (PIT: all rows predate `before`).
            lg3p = sum(x["thrm"] for x in rows) / max(sum(x["thra"] for x in rows), 1)
            for x in rows:
                exp3 = lg3p * x["thra"]
                x["efg"] = x["efg"] + 0.5 * (exp3 - x["thrm"]) / x["fga"]
                x["ortg"] = x["ortg"] + 100 * 3 * (exp3 - x["thrm"]) / x["poss"]
        w = None
        if carry_active:
            w = w_carry
        elif half_life_days and before:
            age = np.array([(before - x["date"]).days for x in rows], float)
            w = 0.5 ** (age / half_life_days)
        self.fms = {f: TeamRatings(ridge=self.ridge, team_home_ridge=None).fit(
            [(x["tid"], x["oid"], x["home"], x[f] * 100) for x in rows],
            weights=w) for f in FACTORS}
        X = np.array([[self.fms[f].pred_ortg(x["tid"], x["oid"], x["home"])
                       for f in FACTORS] for x in rows])
        y = np.array([x["ortg"] for x in rows])
        A = np.c_[X, np.ones(len(X))]
        if w is not None:
            sw = np.sqrt(w)
            self.W = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        else:
            self.W = np.linalg.lstsq(A, y, rcond=None)[0]
        return self

    def eortg(self, tid: int, oid: int, is_home: bool) -> float:
        xf = np.array([self.fms[f].pred_ortg(tid, oid, is_home) for f in FACTORS])
        return float(xf @ self.W[:4] + self.W[4])

    def margin(self, home_id: int, away_id: int) -> float:
        return self.eortg(home_id, away_id, True) - self.eortg(away_id, home_id, False)

    def margin_neutral(self, home_id: int, away_id: int) -> float:
        """Team-quality margin with the home coefficient stripped from both
        sides — home advantage is applied explicitly by the schedule layer."""
        return self.eortg(home_id, away_id, False) - self.eortg(away_id, home_id, False)

    @property
    def ready(self) -> bool:
        return self.W is not None and bool(self.fms)
