"""Zone-split defensive RAPM — de-confounded rim / mid / perimeter defense.

Raw on-court allowance (defense_zone.py) is contaminated: a player's rim-D
reflects the rim protector next to him. Here, for each zone, we ridge-regress
the shot MAKE indicator on the five defenders on the floor, so each player's
zone-defense coefficient is estimated controlling for his linemates — the same
disentangling RAPM does for net impact, applied per shot zone.

Per zone z: make_i ~ mu_z + Σ_{d in def5(i)} beta_{d,z}   (weighted logit-scale
via linear-prob ridge for tractability). We store -beta so POSITIVE = good D
(suppresses makes in that zone). Output: rim_def / mid_def / thr_def per player.

Reads cached rotation+PBP only (no DB). Attribution reuses the stint segments.
"""
from __future__ import annotations

import numpy as np
import orjson
from scipy import sparse

from .rapm import SEC_PER_POSS  # noqa: F401 (kept for parity)
from ..features.defense_zone import _game_segments, _zone
from ..features.possessions_v2 import _team_ids
from ..features.stints import _elapsed


def collect_shots(limit=None, allowed_gids=None, only_002: bool = False, before=None):
    """Return per-zone (defender_id_lists, made) records from the cache.
    allowed_gids: restrict to these game_ids (leakage-safe train split).
    only_002/before: same regular-season / point-in-time filters as
    `defense_zone.build_zone_defense` (D100). Defaults are OFF here so the
    already-published def-RAPM re-runs stay bit-reproducible; pass them
    explicitly for any new PIT work."""
    from ..features.cache_index import game_index
    rots = game_index("gamerotation")
    pbps = game_index("playbyplayv3")
    gids = sorted(set(rots) & set(pbps))
    if only_002:
        gids = [g for g in gids if g.startswith("002")]
    if before is not None:
        from ..features.defense_zone import games_before
        ok = games_before(before, only_002=only_002)
        gids = [g for g in gids if g in ok]
    if allowed_gids is not None:
        allowed_gids = set(allowed_gids)
        gids = [g for g in gids if g in allowed_gids]
    if limit:
        gids = gids[:limit]
    recs = {"rim": [], "mid": [], "thr": []}   # each: (def5 tuple, made 0/1)
    for gid in gids:
        try:
            rot = orjson.loads(open(rots[gid], "rb").read())["response"]
            pbp = orjson.loads(open(pbps[gid], "rb").read())["response"]
        except Exception:
            continue
        # D100: the cached playbyplayv3 `game` object has NO homeTeamId (probe-
        # verified: keys are only gameId/videoAvailable/actions), so the old
        # `pbp['game'].get('homeTeamId')` was ALWAYS None and
        # `defenders = a5 if teamId == home_id else h5` attributed EVERY shot to
        # the home five. Derive the ids from the rotation feed instead (same fix
        # D81 applied to possessions_v2 / defense_zone).
        home_id, _away_id = _team_ids(rot, pbp)
        if home_id is None:
            continue
        segs = _game_segments(rot, pbp)
        if not segs:
            continue
        seg_t0 = np.array([s[0] for s in segs])
        for a in pbp.get("game", {}).get("actions", []):
            if a.get("actionType") not in ("Made Shot", "Missed Shot"):
                continue
            t = _elapsed(a.get("period"), a.get("clock"))
            if t is None:
                continue
            k = int(np.searchsorted(seg_t0, t, side="right") - 1)
            if k < 0 or k >= len(segs):
                continue
            _, _, h5, a5 = segs[k]
            defenders = a5 if a.get("teamId") == home_id else h5
            z = _zone(a.get("shotDistance"), a.get("shotValue"), a.get("subType"))
            recs[z].append((tuple(defenders), int(a.get("shotResult") == "Made")))
    return recs


def fit_zone(records, ridge: float = 800.0):
    """Ridge-regress make on defender indicators for one zone.
    Returns {player_id: def_rating} (positive = suppresses makes = good D)."""
    players = sorted({p for defs, _ in records for p in defs})
    if len(players) < 10 or len(records) < 500:
        return {}, 0.0
    idx = {p: i for i, p in enumerate(players)}
    P = len(players)
    rows, cols, vals, y = [], [], [], []
    for r, (defs, made) in enumerate(records):
        rows.append(r); cols.append(0); vals.append(1.0)      # mu
        for d in defs:
            rows.append(r); cols.append(1 + idx[d]); vals.append(1.0)
        y.append(made)
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(records), 1 + P))
    y = np.asarray(y, float)
    A = (X.T @ X).toarray()
    reg = np.full(1 + P, ridge); reg[0] = 0.0
    A[np.diag_indices_from(A)] += reg
    b = X.T @ y
    beta = np.linalg.solve(A, b)
    eff = beta[1:]; eff -= eff.mean()
    # positive beta = raises make-rate = BAD D; store -beta*100 so +=good, ~pts scale
    return {players[i]: float(-eff[i] * 100) for i in range(P)}, float(beta[0])


def fit_all(limit=None, ridge=800.0, allowed_gids=None):
    recs = collect_shots(limit, allowed_gids=allowed_gids)
    out = {}
    league = {}
    for z in ("rim", "mid", "thr"):
        ratings, mu = fit_zone(recs[z], ridge)
        league[z] = mu
        for p, v in ratings.items():
            out.setdefault(p, {})[z] = v
    return out, league, {z: len(recs[z]) for z in recs}
