"""Compose engine inputs from player skills (the skills->engine bridge).

Turns a team's roster of per-player shooting skills (EB estimates from
skill_priors, or fitted posteriors later) into the possession engine's
per-possession rates — and applies the OPPONENT's defense. This is what makes
the engine lineup-sensitive (drop an injured player, recompose) and
defense-aware: when team A shoots, team B's defensive quality lowers the make
probability. Individual shot-by-shot matchup weights (who guards the shooter)
are v1.5 (needs tracking data); this is the team-level version of II.3.4's
"shooter skill - defender skill".

Offense: zone FG% and zone shares are shot-volume-weighted blends of the
rostered players' rates. Defense: a team factor = how much the team suppresses
opponent make-rate vs league (from opponents' shooting in the team's games),
applied as a logit shift to the opponent's zone FG%.
"""
from __future__ import annotations

import numpy as np

from .possession import LEAGUE


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def team_offense(con, team_id: int, before=None, min_shots: int = 5) -> dict:
    """Shot-volume-weighted offensive rates for a team's rostered players.
    `before` (date) restricts to trailing games (leakage-safe)."""
    date_clause = "AND g.game_date < ?" if before else ""
    params = [team_id] + ([before] if before else [])
    rows = con.execute(f"""
        SELECT s.player_id,
               sum(s.rimm) rimm, sum(s.rima) rima, sum(s.midm) midm, sum(s.mida) mida,
               sum(s.fg3m) fg3m, sum(s.fg3a) fg3a, sum(s.ftm) ftm, sum(s.fta) fta,
               sum(s.tov) tov, sum(s.oreb) oreb, sum(s.dreb) dreb
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
        WHERE s.team_id = ? AND s.game_id LIKE '002%' {date_clause}
        GROUP BY s.player_id
    """, params).fetchdf()
    if rows.empty:
        return dict(LEAGUE)

    tot = rows[["rima", "mida", "fg3a"]].sum()
    fga = float(tot.sum())
    if fga < 50:
        return dict(LEAGUE)
    shares = {"rim": tot["rima"] / fga, "mid": tot["mida"] / fga, "thr": tot["fg3a"] / fga}
    # volume-weighted zone FG% across the roster (a player weights by their attempts)
    def zfg(mk, at):
        num, den = float(rows[mk].sum()), float(rows[at].sum())
        return num / den if den > 0 else LEAGUE["zone_fg"][{"rimm":"rim","midm":"mid","fg3m":"thr"}[mk]]
    zone_fg = {"rim": zfg("rimm", "rima"), "mid": zfg("midm", "mida"), "thr": zfg("fg3m", "fg3a")}
    poss = fga + 0.44 * float(rows["fta"].sum()) - float(rows["oreb"].sum()) + float(rows["tov"].sum())
    rates = dict(LEAGUE)
    rates.update(zone_share=shares, zone_fg=zone_fg,
                 tov_per_poss=float(rows["tov"].sum()) / max(poss, 1),
                 oreb_rate=float(rows["oreb"].sum()) / max(float(rows["oreb"].sum()) + float(rows["dreb"].sum()), 1),
                 foul_per_shot=0.44 * float(rows["fta"].sum()) / max(fga, 1) / 2 + 0.02)
    return rates


def team_defense_shift(con, team_id: int, before=None) -> float:
    """Logit shift a team's defense applies to opponent make-rate. Negative =
    good defense (suppresses shooting). From opponents' FG% in this team's games
    vs league. Team-level stand-in for per-defender skill."""
    date_clause = "AND g.game_date < ?" if before else ""
    params = [team_id, team_id] + ([before] if before else [])
    row = con.execute(f"""
        SELECT sum(o.fgm) fgm, sum(o.fga) fga
        FROM player_game_stats o
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
        WHERE o.game_id IN (SELECT DISTINCT game_id FROM player_game_stats WHERE team_id = ?)
          AND o.team_id <> ? AND o.game_id LIKE '002%' {date_clause}
    """, params).fetchone()
    if not row or not row[1]:
        return 0.0
    league_fg = 0.470
    # EB shrinkage toward league: small samples must not swing the shift wildly.
    # k ~ a few hundred opponent FGA of "prior" pull (regularizes early season).
    k = 400.0
    fgm, fga = float(row[0]), float(row[1])
    opp_fg = (fgm + k * league_fg) / (fga + k)
    return _logit(opp_fg) - _logit(league_fg)


def matchup_rates(con, off_team: int, def_team: int, before=None) -> dict:
    """Offensive team's rates with the defensive team's suppression applied."""
    off = team_offense(con, off_team, before)
    shift = team_defense_shift(con, def_team, before)
    adj = dict(off)
    adj["zone_fg"] = {z: float(_sigmoid(_logit(p) + shift)) for z, p in off["zone_fg"].items()}
    return adj
