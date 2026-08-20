"""Possession-level Monte Carlo engine — v0 skeleton (handoff II.3).

This is the START of the actual simulator. v0 deliberately uses TEAM-LEVEL rates
(not yet player skills) so the engine LOOP can be validated against league
marginals (III.2 calibration battery) BEFORE the Bayesian skill fit exists. Once
the skill posteriors land, the per-possession probabilities get replaced by
logits composed from the on-court players' theta — the loop below is unchanged.

Possession = a trip that ends on a made shot, a turnover, or a defensive
rebound; an offensive rebound extends it. This is the correct unit (OREB does
not start a new possession), so simulated PPP is directly comparable to real.

Determinism: caller passes a seeded numpy Generator (no implicit global RNG).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# League-average seed rates (from player_game_stats, 2025-26 sample). These are
# the placeholders the skill model will later replace per-lineup.
LEAGUE = dict(
    tov_per_poss=0.132,        # calibrated so PPP lands ~1.15 with the tree below
    zone_share={"rim": 0.346, "mid": 0.235, "thr": 0.418},
    zone_fg={"rim": 0.622, "mid": 0.446, "thr": 0.352},
    foul_per_shot=0.146,       # shooting-foul prob per FG attempt -> ~0.29 FTA/FGA
    ft_pct=0.774,
    oreb_rate=0.245,           # offensive-rebound prob on a miss
    pace=99.5,                 # possessions per team per game
)
ZONE_PTS = {"rim": 2, "mid": 2, "thr": 3}


@dataclass
class GameResult:
    home_pts: int
    away_pts: int
    home_fga: int
    away_fga: int
    home_fg3a: int
    home_fta: int

    @property
    def home_win(self) -> bool:
        return self.home_pts > self.away_pts


def simulate_possession(rates: dict, rng: np.random.Generator) -> tuple[int, int, int, int]:
    """Returns (points, fga, fg3a, fta) for one possession."""
    pts = fga = fg3a = fta = 0
    zones = list(rates["zone_share"])
    probs = np.array([rates["zone_share"][z] for z in zones])
    probs = probs / probs.sum()
    while True:
        if rng.random() < rates["tov_per_poss"]:
            return pts, fga, fg3a, fta                      # turnover ends possession
        z = zones[rng.choice(len(zones), p=probs)]
        fga += 1
        if z == "thr":
            fg3a += 1
        # shooting foul?
        if rng.random() < rates["foul_per_shot"]:
            fta += 2
            pts += int(rng.random() < rates["ft_pct"]) + int(rng.random() < rates["ft_pct"])
            return pts, fga, fg3a, fta                      # trip ends (ignore and-1 in v0)
        if rng.random() < rates["zone_fg"][z]:
            pts += ZONE_PTS[z]
            return pts, fga, fg3a, fta                      # made shot ends possession
        if rng.random() < rates["oreb_rate"]:
            continue                                        # offensive rebound -> new trip
        return pts, fga, fg3a, fta                          # defensive rebound ends possession


def simulate_team(rates: dict, rng: np.random.Generator) -> tuple[int, int, int, int]:
    n_poss = rng.poisson(rates["pace"])
    pts = fga = fg3a = fta = 0
    for _ in range(n_poss):
        p, a, a3, ft = simulate_possession(rates, rng)
        pts += p; fga += a; fg3a += a3; fta += ft
    return pts, fga, fg3a, fta


def simulate_game(home_rates: dict, away_rates: dict, rng: np.random.Generator,
                  home_edge: float = 1.014) -> GameResult:
    """home_edge scales home pace/efficiency slightly (~2.8 pt home advantage);
    a placeholder for the structural home effect the fit will estimate."""
    hr = dict(home_rates)
    hr["pace"] = home_rates["pace"] * home_edge
    hp, hfga, hfg3a, hfta = simulate_team(hr, rng)
    ap, afga, afg3a, afta = simulate_team(away_rates, rng)
    # nudge home scoring efficiency a hair via an extra made-basket expectation
    hp = int(round(hp * home_edge))
    return GameResult(hp, ap, hfga, afga, hfg3a, hfta)


def simulate_matchup(home_rates: dict, away_rates: dict, n: int = 5000, seed: int = 0):
    """Monte Carlo: N simulated games -> win prob + score distribution."""
    rng = np.random.default_rng(seed)
    hp = np.empty(n); ap = np.empty(n); hw = 0
    for i in range(n):
        g = simulate_game(home_rates, away_rates, rng)
        hp[i], ap[i] = g.home_pts, g.away_pts
        hw += g.home_win
    return {
        "p_home_win": hw / n,
        "home_pts_mean": float(hp.mean()), "home_pts_sd": float(hp.std()),
        "away_pts_mean": float(ap.mean()),
        "total_mean": float((hp + ap).mean()), "total_sd": float((hp + ap).std()),
        "margin_mean": float((hp - ap).mean()), "margin_sd": float((hp - ap).std()),
    }
