"""Vectorized possession engine — same model as possession.py, ~100x faster.

The Python loop in possession.py simulates one possession at a time; here every
possession across every simulated game advances together as numpy arrays.
Offensive rebounds (which extend a possession) are handled ROUND-BY-ROUND: all
active trips resolve at once, the subset that earns an OREB stays active for
another round, and since OREB~0.25 the active set empties in a few rounds.

Identical rates dict as possession.py (tov_per_poss, zone_share, zone_fg,
foul_per_shot, ft_pct, oreb_rate, pace). Deterministic given a seeded Generator.
"""
from __future__ import annotations

import numpy as np

ZONE_ORDER = ["rim", "mid", "thr"]
ZONE_PTS = np.array([2, 2, 3])


def _team_points(rates: dict, n_poss: np.ndarray, rng: np.random.Generator):
    """n_poss[g] = possessions for game g. Returns (points[g], fga[g], fg3a[g], fta[g])."""
    n_games = len(n_poss)
    total = int(n_poss.sum())
    game_of = np.repeat(np.arange(n_games), n_poss)     # game index per possession
    pts = np.zeros(total)
    fga = np.zeros(total); fg3a = np.zeros(total); fta = np.zeros(total)

    shares = np.array([rates["zone_share"][z] for z in ZONE_ORDER])
    shares = shares / shares.sum()
    zfg = np.array([rates["zone_fg"][z] for z in ZONE_ORDER])

    active = np.arange(total)                            # possessions still in a trip
    while active.size:
        m = active.size
        # turnover ends the trip (and possession) with no shot
        tov = rng.random(m) < rates["tov_per_poss"]
        shooters = active[~tov]
        s = shooters.size
        if s:
            zone = rng.choice(3, size=s, p=shares)
            fga[shooters] += 1
            fg3a[shooters] += (zone == 2)
            foul = rng.random(s) < rates["foul_per_shot"]
            # fouled trips: 2 FTs, possession ends
            fouled = shooters[foul]
            if fouled.size:
                fta[fouled] += 2
                made_ft = (rng.random(fouled.size) < rates["ft_pct"]).astype(float) \
                    + (rng.random(fouled.size) < rates["ft_pct"]).astype(float)
                pts[fouled] += made_ft
            # non-fouled: shot make/miss
            live = shooters[~foul]; zlive = zone[~foul]
            if live.size:
                made = rng.random(live.size) < zfg[zlive]
                pts[live[made]] += ZONE_PTS[zlive[made]]
                # misses -> OREB extends (stay active), else possession ends
                missers = live[~made]; zmiss = zlive[~made]
                if missers.size:
                    keep = rng.random(missers.size) < rates["oreb_rate"]
                    next_active = missers[keep]
                else:
                    next_active = missers[:0]
            else:
                next_active = live[:0]
        else:
            next_active = shooters[:0]
        active = next_active

    def by_game(arr):
        return np.bincount(game_of, weights=arr, minlength=n_games)
    return by_game(pts), by_game(fga), by_game(fg3a), by_game(fta)


def simulate_matchup_fast(home_rates: dict, away_rates: dict, n: int = 5000,
                          seed: int = 0, home_edge: float = 1.014) -> dict:
    rng = np.random.default_rng(seed)
    hp_pace = rng.poisson(home_rates["pace"] * home_edge, n)
    ap_pace = rng.poisson(away_rates["pace"], n)
    hp, _, _, _ = _team_points(home_rates, hp_pace, rng)
    ap, _, _, _ = _team_points(away_rates, ap_pace, rng)
    hp = np.round(hp * home_edge)
    total = hp + ap; margin = hp - ap
    return {
        "p_home_win": float((hp > ap).mean()),
        "home_pts_mean": float(hp.mean()), "home_pts_sd": float(hp.std()),
        "away_pts_mean": float(ap.mean()),
        "total_mean": float(total.mean()), "total_sd": float(total.std()),
        "margin_mean": float(margin.mean()), "margin_sd": float(margin.std()),
    }
