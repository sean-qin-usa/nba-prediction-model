"""Production win-prob model: the validated stack, one entrypoint.

Stack (each component ablation-gated; see journal):
  * 50/50 four-factors + availability-composition margin, D62 carry making
    ff.ready from opening night; schedule layer; D73 tank term
  * margin -> P(win) via sigmoid(margin / 7.2)

The pre-ff-ready ratings/cold-start fallback branch (D16 prior, D20 devs,
w_comp blend) was DELETED: it moved 0 of 6,148 games across all 5 corpus
seasons and 0 refits could reach it live (D110/D112 ablation exactly 0.00000;
carry needs only prior-season player_game_stats, present from opening night).
fit_production now FAILS LOUD if FourFactors cannot fit (D54 precedent) —
that state means the DB is missing the entire prior season.

Explicitly NOT in the stack (tested, rejected): recency-weighting, rest,
composition-RAPM, injury adjustments (x3). See COMPLEXITY.md discipline.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

SCALE = 7.2
PRIOR_REGRESS = 0.75


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def last_season_prior(con, season: str, regress: float = PRIOR_REGRESS) -> dict:
    """End-of-last-season margin ratings (from odds_market scores, which cover
    all seasons), regressed toward the mean for roster turnover. Keyed by team
    ABBREVIATION; caller maps to ids."""
    end = int(season[:4]) + 1              # '2025-26' -> season_end 2026
    rows = con.execute("""SELECT home, away, score_home - score_away FROM odds_market
        WHERE season_end = ?""", [end - 1]).fetchall()   # prior season
    if len(rows) < 200:
        return {}
    teams = sorted({t for h, a, _ in rows for t in (h, a)})
    idx = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    X = np.zeros((len(rows), T + 1)); y = np.zeros(len(rows))
    for k, (h, a, m) in enumerate(rows):
        X[k, idx[h]] += 1; X[k, idx[a]] -= 1; X[k, T] = 1; y[k] = m
    P = np.full(T + 1, 30.0); P[T] = 0
    beta = np.linalg.solve(X.T @ X + np.diag(P), X.T @ y)
    r = beta[:T] - beta[:T].mean()
    return {t: regress * float(r[idx[t]]) for t in teams}


SCHED_PRIOR = (2.3, -1.5, +1.5, 0.0, 0.0)   # home edge, home-b2b, away-b2b, home-dead, away-dead
SCHED_SHRINK = 600.0              # games of prior mass
DEAD_WPCT, DEAD_GP = 0.35, 60     # "dead team" = lottery-bound after game 60
                                  # (structural threshold, fixed; only the
                                  # margin COEFFICIENT is estimated, walk-forward)

# D131 COVID CROWD-REGIME GUARD — MEASURED, NOT SHIPPED: **DEFAULT OFF**.
# Set COVID_GUARD=1 to enable. The production path is BITWISE UNCHANGED with
# the switch at its default, so no registered gate is disturbed.
# WHY IT EXISTS: the distortion below is real and large at the INPUT level.
# WHY IT IS OFF: end-to-end it is a null. Paired bootstrap (2000x, seed 7) on
# the certified 5-season corpus, guard-ON minus guard-OFF, + = guard better:
#   POOLED  +0.000079 CI(-0.000417,+0.000569) NS, p_wrongside 0.367 (n=6,148)
#   2021-22 +0.000501 CI(-0.001519,+0.002420) NS
#   2022-23 -0.000103 CI(-0.001636,+0.001479) NS
#   2023-24 / 2024-25 / 2025-26 bitwise unchanged (max|dp| 9.8e-15)
# The repo's standing rule (D124/D127/D130) is that a CI straddling 0 does not
# ship, and 2021-22+2022-23 are the D112 HOLDOUT — moving them on a NS result
# after looking at them is exactly what the gate policy forbids. Note also that
# the guard is INERT ON EVERY FUTURE LIVE SLATE: from 2023-10-24 onward no
# trailing-730d window touches the excluded range, so enabling it could only
# ever change historical backtests of 2021-22 / 2022-23.
# HONEST CAVEAT: the null is somewhat STARVED, not decisive — MDE80 is ~0.0007
# pooled while theory predicts an effect of order 0.0002-0.0006 pooled, so the
# CI cannot exclude the size the input distortion predicts. Do not re-test on
# this corpus (second-look trap); it is eligible only as a fresh-corpus check.
#
# THE DISTORTION IT TARGETS. The 2020 Orlando bubble (neutral site, no
# crowd, from 2020-07-30) and the 2020-21 season (no/heavily-restricted
# attendance throughout) are a different home-advantage regime: realized home
# margin 2020-21 is +0.944 vs +1.958 on 2021-22..2025-26, i.e. -1.014 pts.
# The corpus grew to 7 seasons, so the trailing-730d window below reaches into
# that regime at EVERY 2021-22 cutoff and most of 2022-23. Left in, it depresses
# the APPLIED home_edge by ~0.5 pts (raw OLS ~1.0 pt, halved by SCHED_SHRINK)
# and drags the b2b slopes (b_hb2b -1.95 contaminated vs -3.25 clean).
# Boundaries are calendar facts, not tuned parameters: 2020-07-30 is the first
# bubble seeding game; 2021-06-30 sits after the last 2020-21 game and before
# 2021-22 opens. 2019-20 PRE-shutdown is deliberately NOT excluded — it measured
# +2.174, a normal-crowd season, and keeping it is what stops the guard from
# starving the 2021-22 opener down to a pure-prior home edge.
# Only the MARGIN FIT FRAME is filtered; the b2b and standings lookups are left
# alone, so every retained game keeps its exact b2b flag and as-of wpct.
COVID_NO_CROWD = (dt.date(2020, 7, 30), dt.date(2021, 6, 30))


def fit_schedule_layer(con, before) -> tuple[float, float, float]:
    """Walk-forward (home_edge, beta_home_b2b, beta_away_b2b) from trailing
    2 seasons of finals. Rejection re-review find (2026-07-29): B2B is ALIVE
    at the schedule-margin level (home-b2b home margin -1.35 vs +2.04 rested;
    away-b2b +3.58) — the old props-level test measured the wrong aggregation.
    Also fixes the hardcoded HOME_EDGE=3.0 (actual ~1.9): the source of the
    all-bucket home-side calibration bias found in the residual autopsy."""
    import numpy as np
    import os as _os
    if before is None:
        before = dt.date.today()
    # D131: drop the COVID no-crowd regime from the margin fit frame.
    # DEFAULT OFF (measured null, see COVID_NO_CROWD above); COVID_GUARD=1
    # enables it, which is how the registered numbers were produced.
    _lo = before - dt.timedelta(days=730)
    _params = [before, _lo]
    _excl = ""
    if _os.environ.get("COVID_GUARD", "0") != "0":
        _excl = "AND NOT (game_date >= ? AND game_date <= ?)"
        _params += list(COVID_NO_CROWD)
    g = con.execute(f"""
        WITH t AS (SELECT game_id, game_date, team_id, is_home, pts
                   FROM nba_games WHERE game_id LIKE '002%' AND pts IS NOT NULL
                   AND game_date < ? AND game_date >= ? {_excl})
        SELECT h.game_date, h.team_id ht, a.team_id a_t, h.pts - a.pts AS margin
        FROM t h JOIN t a USING (game_id) WHERE h.is_home AND NOT a.is_home
    """, _params).fetchdf()
    n = len(g)
    if n == 0:
        return SCHED_PRIOR
    lastg = {}
    hb, ab = [], []
    for r in con.execute("""
        SELECT team_id, game_date FROM nba_games WHERE game_id LIKE '002%'
        AND pts IS NOT NULL AND game_date < ? AND game_date >= ?
        ORDER BY game_date""", [before, before - dt.timedelta(days=760)]).fetchall():
        lastg.setdefault(r[0], []).append(r[1])
    prev = {t: {d: i for i, d in enumerate(ds)} for t, ds in lastg.items()}
    def is_b2b(t, d):
        ds = lastg.get(t, [])
        i = prev[t].get(d) if t in prev else None
        return i is not None and i > 0 and (d - ds[i - 1]).days == 1
    # standings as-of each trailing game (for the dead-team incentive term)
    wl = con.execute("""
        SELECT season, team_id, game_date, wl FROM nba_games
        WHERE game_id LIKE '002%' AND wl IS NOT NULL AND game_date < ?
        AND game_date >= ? ORDER BY game_date""",
        [before, before - dt.timedelta(days=760)]).fetchall()
    gp, wins, stand = {}, {}, {}
    for season, t, d0, w0 in wl:
        k = (season, t)
        stand[(t, d0)] = (gp.get(k, 0), wins.get(k, 0) / max(gp.get(k, 1), 1))
        gp[k] = gp.get(k, 0) + 1
        wins[k] = wins.get(k, 0) + (w0 == "W")
    def dead(t, d0):
        s = stand.get((t, d0))
        return s is not None and s[0] >= DEAD_GP and s[1] < DEAD_WPCT
    hd, ad, qd = [], [], []
    for r in g.itertuples():
        d = r.game_date.date() if hasattr(r.game_date, "date") else r.game_date
        hb.append(is_b2b(r.ht, d)); ab.append(is_b2b(r.a_t, d))
        hd.append(dead(r.ht, d)); ad.append(dead(r.a_t, d))
        sh = stand.get((r.ht, d)); sa = stand.get((r.a_t, d))
        qd.append((sh[1] if sh else 0.5) - (sa[1] if sa else 0.5))
    # wpct-diff CONTROL de-confounds the dead flags from team quality (dead
    # teams are bad; without it beta_dead absorbs badness the components
    # already price). Control coefficient is fit-only, never applied.
    X = np.c_[np.ones(n), np.array(hb, float), np.array(ab, float),
              np.array(hd, float), np.array(ad, float), np.array(qd, float)]
    beta = np.linalg.lstsq(X, g.margin.values, rcond=None)[0]
    w = n / (n + SCHED_SHRINK)
    return tuple(w * beta[i] + (1 - w) * SCHED_PRIOR[i] for i in range(5))


def fit_schedule_layer_ext(con, before, arms=(), state=None):
    """D46 layer EXTENDED with pre-registered travel / circadian / density
    regressors (data/travel_prereg.md, sha256 d3d334b9...).

    *** MEASUREMENT INFRASTRUCTURE — **NOT IN THE PRODUCTION PATH**. ***
    Nothing in fit_production calls this; `fit_schedule_layer` above is
    BITWISE UNCHANGED (proven: the same-run control matched all 3,690 dev games
    of data/capstone_pergame.csv at max|dp| 1.688e-14, 0 games moved). There is
    no SCHED_TRAVEL switch because the terms did not earn one.

    GATE RESULT (dev 2023-24..2025-26, n=3,690, paired 2000x seed 20260801,
    + = better; scripts/tv_gate.py):
        ARM A travel     +0.00011 CI(-0.00028,+0.00050) ns   MDE80 0.00055
        ARM B circadian  -0.00009 CI(-0.00068,+0.00054) ns   sign-KILLED
        ARM C road-trip  -0.00038 CI(-0.00116,+0.00034) ns
        ARM D density    -0.00020 CI(-0.00088,+0.00043) ns
        PORTFOLIO ABCD   -0.00066 CI(-0.00179,+0.00047) ns   (negative point)
    NO-SHIP. The holdout endpoint was deliberately NOT scored (pre-registered:
    once, only if dev passes), so 2021-22/2022-23 remain clean.

    WHY IT FAILED IS NOT "TRAVEL IS FAKE". The margin-scale coefficients are
    sign-correct and stable — d3in4 sign-correct in 100% of the 75 walk-forward
    refits, travel 98.7%, road-trip length 96.0% — and travel is significant on
    the 8,279-game margin frame at -0.309 pts/1,000 km CI(-0.583,-0.035). The
    terms simply move margin by rms 0.195-0.609 pts, while the break-even for
    the log-loss endpoint at n=3,690 is rms 1.485 pts. ARM A would need ~88,500
    games (72 seasons) to pass. Kept here so the retest on a larger corpus is a
    one-line call rather than a rebuild.

    ARM D also demonstrably RE-LABELS the shipped b2b terms rather than adding a
    channel: the joint refit moves home-b2b by +0.335 and away-b2b by -0.336.

    Returns (base5, extra) where base5 is the same 5-tuple fit_schedule_layer
    returns and `extra` is {regressor_name: shrunk coefficient in points}.

    DISCIPLINE, identical to fit_schedule_layer / D46:
      * walk-forward, trailing 730 days strictly before `before`;
      * the wpct-diff CONTROL `qd` is in the design matrix and is FIT-ONLY,
        never applied;
      * shrinkage w = n/(n+SCHED_SHRINK) toward SCHED_PRIOR for the five
        shipped terms and toward **0.0** for every new term;
      * the SHIPPED b2b regressors stay in the design matrix, so a travel
        coefficient is by construction the INCREMENT over b2b, and because the
        whole layer is refit jointly nothing is double-counted at apply time.

    arms=() reproduces fit_schedule_layer EXACTLY (asserted by the gate
    harness, scripts/tv_gate.py; measured max|diff| 1.1e-13, pure lstsq
    round-off from a different array assembly order).

    `state` accepts a prebuilt travel.build_state(con) map so a walk-forward
    harness does not rebuild 16.5k team-games at every weekly refit.
    """
    import os as _os

    import numpy as np

    from .travel import arm_columns, build_state, term_value
    if before is None:
        before = dt.date.today()
    _lo = before - dt.timedelta(days=730)
    _params = [before, _lo]
    _excl = ""
    if _os.environ.get("COVID_GUARD", "0") != "0":
        _excl = "AND NOT (game_date >= ? AND game_date <= ?)"
        _params += list(COVID_NO_CROWD)
    g = con.execute(f"""
        WITH t AS (SELECT game_id, game_date, team_id, is_home, pts
                   FROM nba_games WHERE game_id LIKE '002%' AND pts IS NOT NULL
                   AND game_date < ? AND game_date >= ? {_excl})
        SELECT h.game_date, h.team_id ht, a.team_id a_t, h.pts - a.pts AS margin
        FROM t h JOIN t a USING (game_id) WHERE h.is_home AND NOT a.is_home
    """, _params).fetchdf()
    n = len(g)
    cols = arm_columns(arms)
    if n == 0:
        return SCHED_PRIOR, {c: 0.0 for c, _ in cols}
    lastg = {}
    for r in con.execute("""
        SELECT team_id, game_date FROM nba_games WHERE game_id LIKE '002%'
        AND pts IS NOT NULL AND game_date < ? AND game_date >= ?
        ORDER BY game_date""", [before, before - dt.timedelta(days=760)]).fetchall():
        lastg.setdefault(r[0], []).append(r[1])
    prev = {t: {d: i for i, d in enumerate(ds)} for t, ds in lastg.items()}

    def is_b2b(t, d):
        ds = lastg.get(t, [])
        i = prev[t].get(d) if t in prev else None
        return i is not None and i > 0 and (d - ds[i - 1]).days == 1
    wl = con.execute("""
        SELECT season, team_id, game_date, wl FROM nba_games
        WHERE game_id LIKE '002%' AND wl IS NOT NULL AND game_date < ?
        AND game_date >= ? ORDER BY game_date""",
        [before, before - dt.timedelta(days=760)]).fetchall()
    gp, wins, stand = {}, {}, {}
    for season, t, d0, w0 in wl:
        k = (season, t)
        stand[(t, d0)] = (gp.get(k, 0), wins.get(k, 0) / max(gp.get(k, 1), 1))
        gp[k] = gp.get(k, 0) + 1
        wins[k] = wins.get(k, 0) + (w0 == "W")

    def dead(t, d0):
        s = stand.get((t, d0))
        return s is not None and s[0] >= DEAD_GP and s[1] < DEAD_WPCT
    st = state if state is not None else (build_state(con) if cols else {})
    hb, ab, hd, ad, qd = [], [], [], [], []
    ex = {c: [] for c, _ in cols}
    keep = []
    for r in g.itertuples():
        d = r.game_date.date() if hasattr(r.game_date, "date") else r.game_date
        if cols:
            sh, sa = st.get((r.ht, d)), st.get((r.a_t, d))
            if sh is None or sa is None:
                continue
            for c, fn in cols:
                ex[c].append(term_value(c, fn, sh, sa))
        keep.append(r.margin)
        hb.append(is_b2b(r.ht, d)); ab.append(is_b2b(r.a_t, d))
        hd.append(dead(r.ht, d)); ad.append(dead(r.a_t, d))
        sh_, sa_ = stand.get((r.ht, d)), stand.get((r.a_t, d))
        qd.append((sh_[1] if sh_ else 0.5) - (sa_[1] if sa_ else 0.5))
    n = len(keep)
    if n == 0:
        return SCHED_PRIOR, {c: 0.0 for c, _ in cols}
    base = [np.ones(n), np.array(hb, float), np.array(ab, float),
            np.array(hd, float), np.array(ad, float)]
    X = np.c_[tuple(base + [np.array(ex[c], float) for c, _ in cols]
                    + [np.array(qd, float)])]
    beta = np.linalg.lstsq(X, np.array(keep, float), rcond=None)[0]
    w = n / (n + SCHED_SHRINK)
    base5 = tuple(w * beta[i] + (1 - w) * SCHED_PRIOR[i] for i in range(5))
    extra = {c: w * float(beta[5 + j]) for j, (c, _) in enumerate(cols)}
    return base5, extra


CARRY_W0 = 0.3          # D62 pre-registered carry weight (x continuity)
CARRY_CONT_DEFAULT = 0.5555555555555556


def _prev_season(season: str) -> str:
    y0 = int(season[:4])
    return f"{y0 - 1}-{str(y0)[-2:]}"


def ps_continuity(con, season: str, before=None,
                  prev_minutes: list | None = None) -> dict:
    """D84-A refit-1 proxy: {team_id: prior-season 002 team-minutes share
    returning among this season's preseason (001) participants} — the
    ps_cont_any signal (rw_early_signals.py), corr 0.93 / MAE 0.034 vs the
    realized first-5 continuity on 2023-26. Knowable on opening night
    (preseason ends days before the opener). Empty dict when no 001 data."""
    prev = _prev_season(season)
    if prev_minutes is None:
        prev_minutes = con.execute("""
            SELECT s.team_id, s.player_id, sum(s.seconds)/60.0 mins
            FROM player_game_stats s
            JOIN nba_games g ON g.game_id = s.game_id AND g.team_id = s.team_id
            WHERE g.season = ? AND s.game_id LIKE '002%' GROUP BY 1, 2""",
            [prev]).fetchall()
    date_clause = "AND g.game_date < ?" if before else ""
    params = [season] + ([before] if before else [])
    ros = {}
    for t, p in con.execute(f"""
        SELECT s.team_id, s.player_id FROM player_game_stats s
        JOIN nba_games g ON g.game_id = s.game_id AND g.team_id = s.team_id
        WHERE g.season = ? AND s.game_id LIKE '001%' AND s.seconds > 0
        {date_clause} GROUP BY 1, 2""", params).fetchall():
        ros.setdefault(int(t), set()).add(int(p))
    tot, ret = {}, {}
    for t, p, m in prev_minutes:
        t = int(t)
        tot[t] = tot.get(t, 0.0) + m
        if int(p) in ros.get(t, set()):
            ret[t] = ret.get(t, 0.0) + m
    return {t: ret.get(t, 0.0) / tot[t]
            for t in tot if tot[t] > 0 and t in ros}


def continuity_map(con, season: str, before=None) -> dict | None:
    """{team_id: minutes-weighted share of last season's minutes returning on
    this season's early roster}. PIT: roster from first-5 games strictly
    before `before`. Teams with no observed 002 games yet (refit-1 / opening
    night) use the preseason-001 continuity proxy (D84-A: the old uniform
    CARRY_CONT_DEFAULT fallback ran ALL 30 teams at 0.5556 on opening night;
    ps_cont_any is a corr-0.93 proxy for what the carry converges to);
    CARRY_CONT_DEFAULT remains the last-resort when 001 data is absent."""
    prev = _prev_season(season)
    pm = con.execute("""
        SELECT s.team_id, s.player_id, sum(s.seconds)/60.0 mins
        FROM player_game_stats s
        JOIN nba_games g ON g.game_id = s.game_id AND g.team_id = s.team_id
        WHERE g.season = ? AND s.game_id LIKE '002%' GROUP BY 1, 2""",
        [prev]).fetchall()
    if not pm:
        return None
    date_clause = "AND game_date < ?" if before else ""
    params = [season] + ([before] if before else [])
    roster = con.execute(f"""
        WITH tg AS (
          SELECT team_id, game_id,
                 row_number() OVER (PARTITION BY team_id ORDER BY game_date, game_id) rn
          FROM nba_games WHERE season = ? AND game_id LIKE '002%' {date_clause})
        SELECT tg.team_id, s.player_id FROM tg
        JOIN player_game_stats s ON s.game_id = tg.game_id AND s.team_id = tg.team_id
        WHERE tg.rn <= 5 GROUP BY 1, 2""", params).fetchall()
    ros = {}
    for t, p in roster:
        ros.setdefault(int(t), set()).add(int(p))
    tot, ret = {}, {}
    for t, p, m in pm:
        t = int(t)
        tot[t] = tot.get(t, 0.0) + m
        if int(p) in ros.get(t, set()):
            ret[t] = ret.get(t, 0.0) + m
    ps = {}
    import os as _os
    if (_os.environ.get("OCT_BRIDGE", "1") != "0"      # one switch for the
            # whole D84-A package (bridge + ps-cont carry): OCT_BRIDGE=0 is
            # the F6 one-shot same-run control = exact old shipped behavior
            and any(t not in ros for t in tot)):       # some team pre-002
        ps = ps_continuity(con, season, before=before, prev_minutes=pm)
    return {t: (ret.get(t, 0.0) / tot[t] if t in ros
                else ps.get(t, CARRY_CONT_DEFAULT))
            for t in tot if tot[t] > 0}


def fit_production(con, season: str, before=None, w_comp: float = 0.7):
    """Fit the production model:
      * availability-COMPOSITION strength (validated 0.5455 vs ratings 0.5815) —
        the primary component; reacts to injury-feed OUT lists in-season
      * four-factors margin (D62-carried from opening night)
    w_comp is VESTIGIAL (it weighted the deleted pre-ff-ready fallback);
    kept only so existing callers passing w_comp=0.7 keep working."""
    from .composition import CompositionModel
    from .four_factors import FourFactors
    comp = CompositionModel(con, before=before)
    import os as _os
    # D84-A OCTOBER BRIDGE: when the comp rotation is EMPTY for a game (the
    # week-1 dead state: ROSTER_DAYS excludes everyone, cm is a literal 0 AND
    # the outs channel is dead), the comp leg is replaced by the preseason-
    # bridge composition (001 rosters + prior-season trailing minutes + PIT
    # DARKO). Built only when some team is in the dead state at the cutoff;
    # every game with a live rotation is bitwise unchanged. OCT_BRIDGE=0
    # restores dead-zero behavior (one-shot same-run control, F6).
    bridge = None
    if _os.environ.get("OCT_BRIDGE", "1") != "0":
        from .october_bridge import OctoberBridge, missing_rotation_teams
        if missing_rotation_teams(con, comp, season, before):
            # FROZEN for 2026-27: trail_seasons=2 is D105's declared primary
            # for the F6 one-shot (measured: 5 players moved, max |dcm_ps|
            # 0.418 vs uncapped). OCT_BRIDGE_TRAIL="" reproduces the uncapped
            # legacy construction (D122 certification ran uncapped).
            _tr = _os.environ.get("OCT_BRIDGE_TRAIL", "2")
            bridge = OctoberBridge(con, season, before=before,
                                   trail_seasons=int(_tr) if _tr else None)
    from .october_bridge import rotation_empty as _rot_empty
    from .four_factors import factor_game_rows
    # D62 carry: prior-season rows x 0.3 x continuity, hard-stopped at readiness
    carry_rows = carry_w = None
    cont = continuity_map(con, season, before=before)
    if cont is not None:
        prev_rows = factor_game_rows(con, _prev_season(season), before=None)
        if prev_rows:
            carry_rows = prev_rows
            carry_w = [CARRY_W0 * cont.get(x["tid"], CARRY_CONT_DEFAULT)
                       for x in prev_rows]
    ff = FourFactors().fit(con, season, before=before,
                           luck_adjust_3p=_os.environ.get("FF_LUCK") == "1",
                           carry_rows=carry_rows, carry_weights=carry_w)
    if not ff.ready:
        # The deleted D16/ratings fallback could only fire here, and only on
        # a DB missing the ENTIRE prior season (carry supplies ff.ready from
        # opening night otherwise: 0 of 6,148 corpus games, 0 of 125 weekly
        # refits, live 2026-27 probe ready — D110/D112 + t1_coldstart re-test).
        # In that state the fallback itself was near-garbage (TeamRatings
        # unfit at <30 rows, last_season_prior empty): fail loud instead.
        raise RuntimeError(
            f"FourFactors not ready for {season} (before={before}): fewer "
            f"than 200 current-season rows AND no carry rows from "
            f"{_prev_season(season)}. Prior-season player_game_stats/"
            "nba_games are missing from this DB — restore or backfill them; "
            "the pre-ff-ready ratings fallback was deleted (see DECISIONS).")
    he, b_hb2b, b_ab2b, b_hdead, b_adead = fit_schedule_layer(con, before)
    # D73 tank term: margin += k * (tank_home - tank_away), active only when
    # the scored team's gp >= 55 (exactly 0 otherwise -> games with both
    # teams gp<55 are bitwise-unchanged). Composite + k estimator live in
    # tanking.py; the per-(team, date) map is prefit (heavy build, cached
    # per process) and applied INSIDE margin — callers pass nothing new,
    # which is what keeps backtest and predict_today on one code path
    # (live slates are primed via get_tank_model(con, virtual_games=...)).
    # TANK_TERM=0 zeroes k for same-run controls (mirrors LATE_STATE /
    # OCT_BRIDGE); the tank MAP is still built because the D90 layer reads its
    # gp state from it, so only the applied margin term goes to 0.
    # D112 STATUS: DEMOTED BUT STILL ON. The held-out effect with warm
    # coefficients is +0.00147 CI(-0.00036,+0.00329) NS on 21-23 (+0.00090 NS
    # on 22-23) against +0.00334 PASS on dev — the registered "PASSES
    # DECISIVELY" claim does NOT reproduce out-of-sample. It stays enabled
    # because the held-out point estimate is POSITIVE and removing the term is
    # worse in all five seasons including both holdout seasons (see D112);
    # TANK_TERM=0 applies the strict reading in one env var.
    from .tanking import get_tank_model
    tank = get_tank_model(con)
    tank_k = (0.0 if _os.environ.get("TANK_TERM", "1") == "0"
              else tank.fit_k(before))
    # D90 late-state layer: margin += c_f*(form5_h-form5_a) + c_o*(nout_h-
    # nout_a), active only when either team's gp >= 55 (exactly 0 otherwise
    # -> games with both teams gp<55 are bitwise-unchanged). Coefficients
    # are DAILY walk-forward inside the layer (evaluated at game_date, not
    # at this refit — matches the pre-registered gate); the tank k above is
    # NOT refit jointly (regime-D L4 evidence). Composite lives in
    # latestate.py; callers pass nothing new (the OUT sets already passed
    # to margin/p_home supply the outs counts; live parity via the primed
    # tank model, D68 discipline).
    #
    # *** D112 REVERTED — DEFAULT OFF (2026-08-01). ***
    # D90 shipped on a pooled gate measured entirely on 2023-26. With the D112
    # corpus floor derived from the data, the layer is warm on the genuine
    # holdout (2021-22 + 2022-23, never scorable during the campaign) and its
    # pre-registered out-of-sample effect is a PRECISE NULL:
    #     held-out 21-23  +0.00014  CI(-0.00085,+0.00108)  NS
    #     dev      24-26  +0.00267  CI(+0.00039,+0.00498)  PASS
    #     gate     23-24  -0.00085  CI(-0.00327,+0.00151)  NS (negative, and
    #                     the layer was active on 33.6% of games there, so the
    #                     old cold-burn-in excuse is gone)
    #     DiD dev-minus-held +0.00253 CI(+0.00005,+0.00492) SIG
    # i.e. it helped MEASURABLY more on the seasons it was developed on and
    # contributes nothing where no gate could select on it. Under the D112
    # pre-registered decision rule that is a DEMOTION, so the switch default
    # flips: LATE_STATE=1 re-enables the layer (the code is intact and every
    # test still covers it); LATE_STATE unset/0 is the shipped behaviour.
    late = None
    if _os.environ.get("LATE_STATE", "0") != "0":
        from .latestate import get_latestate_model
        late = get_latestate_model(con)

    class Predictor:
        def tank_diff(self, home_id: int, away_id: int, game_date) -> float:
            """gp>=55-gated tank_home - tank_away (0.0 when game_date=None)."""
            if game_date is None:
                return 0.0
            return tank.diff(home_id, away_id, game_date)

        def margin(self, home_id: int, away_id: int, out_home: set | None = None,
                   out_away: set | None = None, game_date=None,
                   b2b_home: bool = False, b2b_away: bool = False,
                   dead_home: bool = False, dead_away: bool = False) -> float:
            # schedule layer: explicit, walk-forward-estimated (home edge was
            # hardcoded 3.0 = biased; b2b was a false rejection — see
            # fit_schedule_layer docstring)
            sched = (he + (b_hb2b if b2b_home else 0.0) + (b_ab2b if b2b_away else 0.0)
                     + (b_hdead if dead_home else 0.0) + (b_adead if dead_away else 0.0))
            # D73 tank term (0.0 outside the gp>=55 window / without a date)
            tk = tank_k * self.tank_diff(home_id, away_id, game_date)
            # D90 late-state layer (0.0 outside the either-gp>=55 window /
            # without a date; daily walk-forward c_f/c_o inside the layer)
            lt = (late.term(home_id, away_id, out_home, out_away, game_date)
                  if late is not None else 0.0)
            cm = comp.margin(home_id, away_id, out_home, out_away, game_date,
                             home_edge=0.0)
            if bridge is not None and _rot_empty(comp, home_id, away_id,
                                                 game_date):
                # D84-A: dead comp leg -> preseason-bridge cm_ps (outs live)
                cm = bridge.margin(home_id, away_id, out_home, out_away)
            # validated stack: 50/50 four-factors + composition (fixed
            # weights beat fitted — GBM/logistic challengers rejected).
            # ff.ready is guaranteed by the fit-time guard in fit_production.
            fm = ff.margin_neutral(home_id, away_id)
            return 0.5 * fm + 0.5 * cm + sched + tk + lt

        def p_home(self, home_id: int, away_id: int, out_home: set | None = None,
                   out_away: set | None = None, game_date=None,
                   b2b_home: bool = False, b2b_away: bool = False,
                   dead_home: bool = False, dead_away: bool = False) -> float:
            return float(sigmoid(self.margin(home_id, away_id, out_home,
                                             out_away, game_date, b2b_home,
                                             b2b_away, dead_home, dead_away) / SCALE))

    Predictor.tank_k = tank_k               # D73: walk-forward tank coefficient
    Predictor.latestate = late              # D90: late-state layer (audit/tests)
    return Predictor()
