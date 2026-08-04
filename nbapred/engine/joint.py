"""Joint game simulator — correlated player stat lines (the parlay/SGP edge).

The single biggest NON-DATA edge: the market prices same-game parlays by bolting
ad-hoc correlation onto individual legs, but a simulator that plays out a shared
game OWNS the true joint distribution. Two structural correlations emerge for
free here:
  * shared PACE: high-possession sims lift EVERY player -> positive correlation
    between a player's points and the team total.
  * shared SHOT POOL: usage is allocated multinomially among teammates each sim
    -> two teammates' points NEGATIVELY correlate (they compete for shots).

Books often assume near-independence or a crude single correlation; getting the
sign and size right is the edge, even when each individual leg is fair.

v1 keeps it lightweight: one team's possessions are Poisson(pace); each is
allocated to a shooter by usage weight (shot-rate x minute-share); the shot is
scored by that player's zone rates (optionally opponent-adjusted). Rebounds/
assists similarly attributed. Output: per-player point samples (n_sims x
n_players) -> any joint prop probability.
"""
from __future__ import annotations

import numpy as np


def _team_player_arrays(players: list[dict]):
    """players: [{player_id, proj_min, rate_rim, rate_mid, rate_thr, fg_rim,
    fg_mid, fg_thr, fta_per_min, ft_pct, ...}]. Returns usage weights + rate
    arrays for vectorized allocation."""
    pm = np.array([p["proj_min"] for p in players])
    # per-minute shot rate * minutes = expected shots -> usage weight
    shot_rate = np.array([p["rate_rim"] + p["rate_mid"] + p["rate_thr"] for p in players])
    usage = shot_rate * pm
    usage = usage / usage.sum() if usage.sum() > 0 else np.ones(len(players)) / len(players)
    zshare = np.array([[p["rate_rim"], p["rate_mid"], p["rate_thr"]] for p in players])
    zshare = zshare / zshare.sum(axis=1, keepdims=True).clip(1e-9)
    fg = np.array([[p["fg_rim"], p["fg_mid"], p["fg_thr"]] for p in players])
    ftpm = np.array([p.get("fta_per_min", 0) * p["proj_min"] for p in players])  # exp FTA
    ftp = np.array([p.get("ft_pct", 0.77) for p in players])
    return usage, zshare, fg, ftpm, ftp


def simulate_team_joint(players: list[dict], pace: float, n: int, rng) -> np.ndarray:
    """Return points[n_sims, n_players] with shared-pace + shared-pool correlation."""
    P = len(players)
    usage, zshare, fg, ftpm, ftp = _team_player_arrays(players)
    ZP = np.array([2, 2, 3])
    poss = rng.poisson(pace, n)                       # shared pace per sim
    pts = np.zeros((n, P))
    for s in range(n):
        k = poss[s]
        # allocate this game's shot attempts among players by usage
        shots_per = rng.multinomial(k, usage)         # shared pool -> teammates compete
        for j in range(P):
            a = shots_per[j]
            if a == 0:
                continue
            zc = rng.multinomial(a, zshare[j])        # split into rim/mid/3
            made = rng.binomial(zc, fg[j])
            pts[s, j] += int(made @ ZP)
        # free throws (independent-ish, scaled by own volume)
        fta = rng.poisson(ftpm)
        pts[s] += rng.binomial(fta, ftp)
    return pts


def simulate_game_joint(home: list[dict], away: list[dict], home_pace: float,
                        away_pace: float, n: int = 4000, seed: int = 0):
    """Returns {player_id: points_samples} for all players + the two teams."""
    rng = np.random.default_rng(seed)
    hp = simulate_team_joint(home, home_pace, n, rng)
    ap = simulate_team_joint(away, away_pace, n, rng)
    out = {}
    for j, p in enumerate(home):
        out[p["player_id"]] = hp[:, j]
    for j, p in enumerate(away):
        out[p["player_id"]] = ap[:, j]
    return out, hp.sum(axis=1), ap.sum(axis=1)


def parlay_prob(samples: dict, legs: list[tuple]) -> dict:
    """legs = [(player_id, 'over'/'under', line)]. Returns joint P(all hit) vs
    the naive independent product — the gap is the correlation mispricing."""
    masks = []
    for pid, side, line in legs:
        s = samples[pid]
        masks.append(s > line if side == "over" else s < line)
    joint = np.mean(np.all(masks, axis=0))
    indep = np.prod([m.mean() for m in masks])
    return {"joint": float(joint), "independent": float(indep),
            "correlation_edge": float(joint - indep)}
