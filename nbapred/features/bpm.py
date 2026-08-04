"""In-house Box Plus/Minus 2.0 — exact daily PIT from our own box scores.

D85 build decision (docs/EXTERNAL_MODELS.md #2): BPM 2.0 is a deterministic
PUBLISHED formula (basketball-reference.com/about/bpm2.html, raw-cached at
data/raw/ext_bbref/about_bpm2.html) on box-score rates + a team adjustment.
Our DuckDB holds every box score, so computing BPM as-of any date gives daily
exact PIT with zero fetches — the third talent-ensemble leg, and the only
mechanically different one (no plus-minus stream => least error-correlated
with DARKO/EPM).

Implementation is the season-level method from the published page:
 1. Per-100-team-possession stats while on floor
    (poss on floor = team_poss * MP / (team_MP/5)).
 2. Points adjusted for team shooting context:
    adj_pts = pts - TSA * (team_pts_per_TSA - 1.00), TSA = FGA + 0.44*FTA.
    (Baseline 1.00 verified against the page's 2017 LeBron example:
    34.9 -> 30.4 with TSA=28.2 and CLE at 1.157 pts/TSA.)
 3. Position (1=PG..5=C) and offensive role (1=creator..5=receiver) estimated
    from % of team stats accrued on floor (published regressions), blended
    with a 50-minute prior, team-renormalized to minutes-weighted 3.0 with
    recursive clipping to [1, 5].
 4. Coefficients linear in position (FGA/FTA terms: linear in role), plus the
    non-linear position (-0.818 at pos 1) and role (+/-2.774) constants.
 5. Team adjustment: constant per team so that sum(raw_bpm * pct_min) equals
    5 x the team's lead-adjusted SOS-adjusted net rating per 100
    (rating_adj = srs100 * (1 + 0.175 * poss_per_game/200); avg lead ~
    half the per-game adjusted margin, 0.35/2 pts effect per lead point).
 6. Published low-minutes stabilizer for the as-of (rolling) view:
    est = -4.75 + 0.175 * ReMPG, ReMPG = MP/(G+4), weight (450-MP)/3 min 0.

DOCUMENTED DEVIATIONS from basketball-reference (all forced by data reality,
none fitted): (a) no "listed position" source in our DB -> the 50-minute
position prior is neutral 3.0 (B-R uses the listed position; the team
renormalization limits the impact, validated corr>0.95 in
scripts/validate_bpm.py); (b) season BPM here is the season-level method —
B-R aggregates game-level runs since 1985, "very close" fit per their page;
(c) possessions are the B-R estimator formula on our box totals, not B-R's
own possession counts.

PIT rule: every query filters game_date < before (exclusive) — a game on
date d never sees d.  bpm_asof(con, d) is a trailing-365-day rolling view
(spans the season boundary so October dates are covered by last season's
games), which is what "rolling BPM per player as-of any date" means here.
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict

# ---- published BPM 2.0 coefficient tables (position 1 -> position 5) ------
# variables on per-100-team-possession stats
COEF_POS = {          # varies linearly in POSITION
    "adj_pts": (0.860, 0.860),
    "fg3m":    (0.389, 0.389),
    "ast":     (0.580, 1.034),
    "tov":     (-0.964, -0.964),
    "oreb":    (0.613, 0.181),
    "dreb":    (0.116, 0.181),
    "stl":     (1.369, 1.008),
    "blk":     (1.327, 0.703),
    "pf":      (-0.367, -0.367),
}
COEF_ROLE = {         # varies linearly in OFFENSIVE ROLE
    "fga": (-0.560, -0.780),
    "fta": (-0.246, -0.343),
}
POS_CONSTANT = -0.818     # at position 1, linear to 0.0 at position 3+
ROLE_CONSTANT = -2.774    # at role 1, linear through 0 at 3 to +2.774 at 5

# position regression on % of team stats while on floor (current era)
POS_REG = dict(intercept=2.130, trb=8.668, stl=-2.486, pf=0.992,
               ast=-3.536, blk=1.667)
# offensive role regression
ROLE_REG = dict(intercept=6.00, ast=-6.642, thresh_pts=-8.544)
THRESH_OFFSET = 0.33      # threshold pts/TSA = team pts/TSA - 0.33
PTS_TSA_BASELINE = 1.00   # league baseline points per true shot attempt

POS_PRIOR_MIN = 50.0      # 50 minutes of prior ...
POS_PRIOR = 3.0           # ... at neutral position (no listed positions in DB)
ROLE_PRIOR = 4.0          # published: 50 minutes at role 4.0

LEAD_EFFECT = 0.35        # pts/100 worse per point of lead (half to each team)

# published low-minutes stabilizer (game-level section of the BPM 2.0 page)
REG_INTERCEPT, REG_SLOPE = -4.75, 0.175
REG_WEIGHT_CAP, REG_WEIGHT_DIV = 450.0, 3.0


def _clip(x, lo=1.0, hi=5.0):
    return lo if x < lo else hi if x > hi else x


def _team_shift(vals: dict, mp: dict, target: float = 3.0) -> dict:
    """Add a per-team constant (before clipping to [1,5]) so the
    minutes-weighted mean of the clipped values equals `target` (bisection —
    the page specifies this is done recursively because of the limits)."""
    tot_mp = sum(mp.values())
    if tot_mp <= 0:
        return {k: _clip(v) for k, v in vals.items()}
    lo, hi = -6.0, 6.0
    for _ in range(60):
        c = 0.5 * (lo + hi)
        m = sum(_clip(vals[k] + c) * mp[k] for k in vals) / tot_mp
        if m < target:
            lo = c
        else:
            hi = c
    c = 0.5 * (lo + hi)
    return {k: _clip(vals[k] + c) for k in vals}


def _interp(pair, x):
    """Linear between position/role 1 and 5."""
    c1, c5 = pair
    return c1 + (x - 1.0) / 4.0 * (c5 - c1)


def _team_games(con, season, before, since):
    """Team + opponent box totals per (game, team) from player_game_stats."""
    cond, params = "", [season]
    if before is not None:
        cond += " AND g.game_date < ?"
        params.append(before)
    if since is not None:
        cond += " AND g.game_date >= ?"
        params.append(since)
    return con.execute(f"""
        WITH tg AS (
          SELECT s.game_id, s.team_id,
                 sum(s.seconds)/60.0 mp, sum(s.fga) fga, sum(s.fgm) fgm,
                 sum(s.fg3m) fg3m, sum(s.fta) fta, sum(s.ftm) ftm,
                 sum(s.ast) ast, sum(s.tov) tov, sum(s.oreb) oreb,
                 sum(s.dreb) dreb, sum(s.stl) stl, sum(s.blk) blk,
                 sum(s.pf) pf, sum(s.pts) pts
          FROM player_game_stats s
          JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g
            USING (game_id)
          WHERE s.game_id LIKE '002%' AND g.season = ? {cond}
          GROUP BY 1, 2)
        SELECT a.*, b.team_id opp_id, b.fga o_fga, b.fgm o_fgm, b.fta o_fta,
               b.tov o_tov, b.oreb o_oreb, b.dreb o_dreb, b.pts o_pts
        FROM tg a JOIN tg b ON a.game_id = b.game_id
                          AND a.team_id <> b.team_id
    """, params).fetchdf()


def _possessions(r) -> float:
    """B-R possession estimator (average of the two team halves)."""
    def half(fga, fta, fgm, orb, opp_drb, tov):
        den = orb + opp_drb
        orb_pct = orb / den if den > 0 else 0.0
        return fga + 0.4 * fta - 1.07 * orb_pct * (fga - fgm) + tov
    return 0.5 * (half(r.fga, r.fta, r.fgm, r.oreb, r.o_dreb, r.tov)
                  + half(r.o_fga, r.o_fta, r.o_fgm, r.o_oreb, r.dreb, r.o_tov))


def _team_context(con, season, before, since):
    """Per-team season(-window) totals, possessions, pace and the
    lead-adjusted SOS-adjusted net rating per 100 (SRS on per-100 margins)."""
    df = _team_games(con, season, before, since)
    if df.empty:
        return {}
    teams: dict = defaultdict(lambda: dict(
        games=[], mp=0.0, poss=0.0, fga=0.0, fta=0.0, pts=0.0, trb=0.0,
        stl=0.0, pf=0.0, ast=0.0, blk=0.0, tsa=0.0, n=0))
    for r in df.itertuples():
        p = _possessions(r)
        t = teams[int(r.team_id)]
        m100 = (r.pts - r.o_pts) / p * 100.0 if p > 0 else 0.0
        t["games"].append((int(r.opp_id), m100))
        t["mp"] += r.mp
        t["poss"] += p
        t["fga"] += r.fga
        t["fta"] += r.fta
        t["pts"] += r.pts
        t["trb"] += r.oreb + r.dreb
        t["stl"] += r.stl
        t["pf"] += r.pf
        t["ast"] += r.ast
        t["blk"] += r.blk
        t["n"] += 1
    # SRS iteration on per-100 margins
    r0 = {tid: (sum(m for _, m in t["games"]) / len(t["games"]))
          for tid, t in teams.items()}
    rate = dict(r0)
    for _ in range(200):
        new = {}
        for tid, t in teams.items():
            sos = sum(rate.get(o, 0.0) for o, _ in t["games"]) / len(t["games"])
            new[tid] = r0[tid] + sos
        if max(abs(new[t] - rate[t]) for t in rate) < 1e-9:
            rate = new
            break
        rate = new
    out = {}
    for tid, t in teams.items():
        t["tsa"] = t["fga"] + 0.44 * t["fta"]
        t["pts_per_tsa"] = t["pts"] / t["tsa"] if t["tsa"] > 0 else PTS_TSA_BASELINE
        poss_pg = t["poss"] / t["n"] if t["n"] else 0.0
        srs100 = rate[tid]
        # avg lead ~ (per-game adjusted margin)/2; effect 0.35/2 per point
        avg_lead = srs100 * poss_pg / 100.0 / 2.0
        t["rating_adj"] = srs100 + (LEAD_EFFECT / 2.0) * avg_lead
        out[tid] = t
    return out


def compute_bpm(con, season: str, before: dt.date | None = None,
                since: dt.date | None = None) -> dict:
    """Season(-window) BPM per (player_id, team_id) stint.

    Returns {(pid, tid): {'bpm', 'mp', 'games', 'pos', 'role'}} for all
    stints with mp > 0 in `season` restricted to [since, before).
    """
    ctx = _team_context(con, season, before, since)
    if not ctx:
        return {}
    cond, params = "", [season]
    if before is not None:
        cond += " AND g.game_date < ?"
        params.append(before)
    if since is not None:
        cond += " AND g.game_date >= ?"
        params.append(since)
    pdf = con.execute(f"""
        SELECT s.player_id, s.team_id, count(*) games,
               sum(s.seconds)/60.0 mp, sum(s.fga) fga, sum(s.fgm) fgm,
               sum(s.fg3m) fg3m, sum(s.fta) fta, sum(s.ast) ast,
               sum(s.tov) tov, sum(s.oreb) oreb, sum(s.dreb) dreb,
               sum(s.stl) stl, sum(s.blk) blk, sum(s.pf) pf, sum(s.pts) pts
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g
          USING (game_id)
        WHERE s.game_id LIKE '002%' AND g.season = ? {cond} AND s.seconds > 0
        GROUP BY 1, 2
    """, params).fetchdf()
    out = {}
    for tid_np, grp in pdf.groupby("team_id"):
        tid = int(tid_np)
        t = ctx.get(tid)
        if t is None or t["mp"] <= 0:
            continue
        slot_mp = t["mp"] / 5.0            # one floor-slot's total minutes
        pos_raw, role_raw, mp_map = {}, {}, {}
        rows = {}
        for r in grp.itertuples():
            pid = int(r.player_id)
            if r.mp <= 0:
                continue
            share = slot_mp / r.mp         # converts share-of-team to on-floor pct

            def pct(x, tx):
                return (x / tx) * share if tx > 0 else 0.0

            p_pos = (POS_REG["intercept"]
                     + POS_REG["trb"] * pct(r.oreb + r.dreb, t["trb"])
                     + POS_REG["stl"] * pct(r.stl, t["stl"])
                     + POS_REG["pf"] * pct(r.pf, t["pf"])
                     + POS_REG["ast"] * pct(r.ast, t["ast"])
                     + POS_REG["blk"] * pct(r.blk, t["blk"]))
            tsa = r.fga + 0.44 * r.fta
            thresh_p = r.pts - (t["pts_per_tsa"] - THRESH_OFFSET) * tsa
            thresh_t = THRESH_OFFSET * t["tsa"]   # team sum of threshold pts
            p_role = (ROLE_REG["intercept"]
                      + ROLE_REG["ast"] * pct(r.ast, t["ast"])
                      + ROLE_REG["thresh_pts"] * pct(thresh_p, thresh_t))
            # 50-minute priors (position prior neutral: no listed positions)
            pos_raw[pid] = (p_pos * r.mp + POS_PRIOR * POS_PRIOR_MIN) \
                / (r.mp + POS_PRIOR_MIN)
            role_raw[pid] = (p_role * r.mp + ROLE_PRIOR * POS_PRIOR_MIN) \
                / (r.mp + POS_PRIOR_MIN)
            mp_map[pid] = float(r.mp)
            rows[pid] = r
        if not rows:
            continue
        pos = _team_shift(pos_raw, mp_map)
        role = _team_shift(role_raw, mp_map)
        raw = {}
        for pid, r in rows.items():
            poss_floor = t["poss"] * (r.mp / slot_mp)
            if poss_floor <= 0:
                continue

            def per100(x):
                return x / poss_floor * 100.0

            tsa100 = per100(r.fga + 0.44 * r.fta)
            adj_pts = per100(r.pts) - (t["pts_per_tsa"] - PTS_TSA_BASELINE) * tsa100
            v = dict(adj_pts=adj_pts, fg3m=per100(r.fg3m), ast=per100(r.ast),
                     tov=per100(r.tov), oreb=per100(r.oreb),
                     dreb=per100(r.dreb), stl=per100(r.stl),
                     blk=per100(r.blk), pf=per100(r.pf),
                     fga=per100(r.fga), fta=per100(r.fta))
            s = sum(_interp(COEF_POS[k], pos[pid]) * v[k] for k in COEF_POS)
            s += sum(_interp(COEF_ROLE[k], role[pid]) * v[k] for k in COEF_ROLE)
            if pos[pid] < 3.0:
                s += (3.0 - pos[pid]) * (POS_CONSTANT / 2.0)
            s += (3.0 - role[pid]) * (ROLE_CONSTANT / 2.0)
            raw[pid] = s
        # team adjustment: sum(raw * pct_min) + 5c = adjusted team rating
        wsum = sum(raw[pid] * (mp_map[pid] / slot_mp) for pid in raw)
        c = (t["rating_adj"] - wsum) / 5.0
        for pid in raw:
            r = rows[pid]
            out[(pid, tid)] = dict(bpm=raw[pid] + c, mp=mp_map[pid],
                                   games=int(r.games), pos=pos[pid],
                                   role=role[pid])
    return out


def _seasons_in_window(con, since: dt.date, before: dt.date) -> list[str]:
    return [s for (s,) in con.execute(
        "SELECT DISTINCT season FROM nba_games WHERE game_id LIKE '002%' "
        "AND game_date >= ? AND game_date < ? ORDER BY season",
        [since, before]).fetchall()]


def bpm_asof(con, asof: dt.date, window_days: int = 365,
             regressed: bool = True) -> dict:
    """Rolling PIT BPM: {player_id: {'bpm', 'mp', 'games'}} using only games
    in [asof - window_days, asof). Stints (player, team, season) are combined
    minutes-weighted; the published low-minutes stabilizer is applied when
    `regressed` (default — this is the exact-daily-PIT ensemble input)."""
    since = asof - dt.timedelta(days=window_days)
    agg: dict = defaultdict(lambda: dict(wsum=0.0, mp=0.0, games=0))
    for season in _seasons_in_window(con, since, asof):
        for (pid, _tid), v in compute_bpm(con, season, before=asof,
                                          since=since).items():
            a = agg[pid]
            a["wsum"] += v["bpm"] * v["mp"]
            a["mp"] += v["mp"]
            a["games"] += v["games"]
    out = {}
    for pid, a in agg.items():
        if a["mp"] <= 0:
            continue
        bpm = a["wsum"] / a["mp"]
        if regressed:
            rempg = a["mp"] / (a["games"] + 4.0)
            est = REG_INTERCEPT + REG_SLOPE * rempg
            w = max(0.0, (REG_WEIGHT_CAP - a["mp"]) / REG_WEIGHT_DIV)
            bpm = (bpm * a["mp"] + est * w) / (a["mp"] + w)
        out[pid] = dict(bpm=float(bpm), mp=float(a["mp"]), games=a["games"])
    return out
