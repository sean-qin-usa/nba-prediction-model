"""Star-out redistribution for the LIVE props path (D82 production lean-in).

Wires the gate-passed star-out science into predict_today's props section —
until now it lived only in analysis scripts (gate_redistribution_crps.py,
audit_usage_pit.py):

  * D33 [e66896]: softmax ATTEMPTS redistribution — when a >=28-trailing-min
    star is out, remaining rotation players' shot attempts lift by
    S / (S - exp(u_star)), u from the conditional-logit usage fit
    (P(shooter | on-floor five), fit_v2_usage.py). Poisson LL +0.044
    CI +0.033..+0.055 vs no-lift; survives PIT decontamination per D57
    [4d7ecc]: +0.0437 CI +0.027..+0.061.
  * D39 [abc76d]: positional MINUTES tilt — when a star sits, same-position
    players gain +2.91 min, different-position +2.09 (t=8.2, n=24k).
    Position from the defended_fg (ptdefend) cache PLAYER_POSITION field;
    +2.4 flat fallback when either position is unknown.
  * D34/D35 [345a5a/7d093b]: efficiency deliberately UNCHANGED — attempts move
    ~5x more than points (usage-efficiency skill curve); the naive points-lift
    failed its CRPS gate. Conservative ship: volume-only adjustment, FG%/FT%
    untouched.

Usage prior: data/v2_usage.npz (exp(u) weights). If the file is not loadable,
falls back to trailing-attempts shares (exp(u_p) := trailing-10 mean shot
attempts per game), which yields the same S/(S-w_star) form; the context dict
reports which source was used in "usage_source".

RESIDUAL CALIBRATION (the validation finding, scratchpad diag 2026-07-31):
the D33/D39 magnitudes were measured against WEAKER baselines than the live
props engine. D33's +0.044/+0.064 Poisson-LL win used trailing avg_shots over
ALL games (short-minute stubs included); D39's +2.91/+2.09 is the move vs a
star-IN baseline. The live props baseline (>=12-min-conditioned EWMA rates and
minutes, the journal-063f19 universe fix) already absorbs ~2/3 of the
redistribution by game time (absences cluster + EWMA role tracking): on the
2025-26 star-out universe the TOTAL residual volume is only +7.3% (fresh
stratum +8.1%) vs the full-strength +19.8% x (+2.5 min), and full-strength
application makes points/reb/ast CRPS WORSE (points -0.63) — the D34/D35
double-count, now measured. Production therefore applies the gated SHAPE at
the RESIDUAL magnitude:
    lift = 1 + RESID_ATT_SCALE * (softmax_lift - 1)      (per-min rate channel)
    tilt = RESID_MIN_SCALE * D39_tilt                    (minutes channel)
with scales estimated walk-forward on the pre-q0.6 2025-26 split (moment/MLE:
MIN 0.387, ATT 0.161; stable 0.36-0.41 / 0.15-0.18 across q=0.5/0.6/0.7 and
conservative vs held-out rate residual +0.063). REB/AST volumes are held
neutral under the tilt (see adjust_rates: the star-out reb/ast residual is
right-tail-only; the typical row shows none). scripts/validate_starout.py
scores the shipped (residual) and full-strength variants on held-out rows.
October caveat: with last-season trailing windows the baseline embeds less of
an ongoing absence, so residual scales are conservative for truly fresh news.

Design choices (documented, gate-parity):
  * Single star per team: if several >=28-min players are out, adjust for the
    HIGHEST-usage one only — exactly what the D33 gate validated (it picked
    max-u among outs). No stacking of multiple lifts/tilts.
  * Freshness <=12 days: the star must have played for this team within 12
    days, else his absence is already embedded in teammates' trailing rates
    (D35 contamination) and re-lifting would double-count.
  * Raw softmax lift capped to [1.0, 1.6] (never shrinks; D82 spec cap, vs the
    gate's 1.5) BEFORE residual scaling.
  * Trailing minutes are team-scoped (games for THIS team) so a departed
    star's old-team games don't qualify him on his new team.

The caller (predict_today) owns OUT sets; this module never mutates the input
rates dict and is a pure pre-simulate_player transform — simulate_player's
internals and the margin path are untouched.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

STAR_TRAILING_MIN = 28.0     # D33/D39 star definition (trailing-10 mean minutes)
ROT_TRAILING_MIN = 15.0      # rotation pool floor (gate parity)
TRAIL_GAMES = 10             # trailing window (gate: rolling(10, min_periods=5))
MIN_TRAIL_GAMES = 5
FRESH_DAYS = 12              # star must have played for this team within 12d
LIFT_LO, LIFT_HI = 1.0, 1.6  # D82 spec cap (gate capped at 1.5)
TILT_SAME_POS = 2.91         # D39: same-position minutes gain (full effect)
TILT_DIFF_POS = 2.09         # D39: different-position minutes gain (full effect)
TILT_FLAT = 2.4              # fallback when either position is unknown
# residual scales: fraction of the full D33/D39 effect NOT already absorbed by
# the live props baseline (walk-forward moment estimates, pre-q0.6 2025-26;
# see module docstring "RESIDUAL CALIBRATION")
RESID_ATT_SCALE = 0.16
RESID_MIN_SCALE = 0.39
PROJ_MIN_CAP = 46.0          # sanity cap on tilted projection (sim clips at 48)
ATTEMPT_KEYS = ("rate_rim", "rate_mid", "rate_thr", "fta_per_min")

# --- GATE SWITCHES (D146 follow-up + D129 open question) ---------------------
# STAROUT_TRAIL — how team_context() builds trail_min / trail_att / n_tr.
#   "current" (DEFAULT)  every box row in the last TRAIL_GAMES ROWS, DNP zeros
#                        included. D145 measured the consequence: the star
#                        DE-STARS HIMSELF while out (P(fire) 0.842 -> 0.705 as
#                        he misses 1 -> 6-9 games, recall 0.789).
#   "played"             the last TRAIL_GAMES PLAYED rows (m > 0); n_tr becomes
#                        a played count, so MIN_TRAIL_GAMES applies to games
#                        actually played.
#   "floor"              "current" levels, PLUS an explicit played-count floor:
#                        a player with fewer than PLAYED_FLOOR played rows in
#                        the trailing ROW window is excluded from BOTH star
#                        candidacy and the rotation pool (neither detected nor
#                        silently down-weighted).
# STAROUT_USAGE — which weights feed the D33 softmax lift S/(S-w_star).
#   "softmax" (DEFAULT)  exp(u) from data/v2_usage.npz (the fitted artifact).
#   "null_u"             uniform weights => lift = N/(N-1), POOL ARITHMETIC
#                        ONLY. D129's registered hypothesis: what is
#                        load-bearing is the LIFT MAGNITUDE, not the fit.
#   "trailatt"           the module's own documented fallback (trailing-attempt
#                        shares); measured because production TAKES this path
#                        whenever the npz is unloadable.
# Both default to the shipped behaviour; a mode is only reachable by setting
# the env var, so production is bitwise unchanged until a gate flips a default.
PLAYED_FLOOR = MIN_TRAIL_GAMES   # ARM B floor (reuses the module constant; not tuned)
_TRAIL_MODES = ("current", "played", "floor")
_USAGE_MODES = ("softmax", "null_u", "trailatt")


def trail_mode() -> str:
    v = (os.environ.get("STAROUT_TRAIL") or "current").strip().lower()
    return v if v in _TRAIL_MODES else "current"


def usage_mode() -> str:
    v = (os.environ.get("STAROUT_USAGE") or "softmax").strip().lower()
    return v if v in _USAGE_MODES else "softmax"


def load_usage_weights(path=None):
    """exp(u) per player from the D33 conditional-logit usage fit.
    Returns {player_id: weight} or None if the npz is not loadable (caller /
    team_context then falls back to trailing-attempts shares)."""
    p = Path(path) if path else ROOT / "data" / "v2_usage.npz"
    try:
        z = np.load(p)
        return {int(i): float(np.exp(u))
                for i, u in zip(z["player_ids"], z["u"])}
    except Exception:  # noqa: BLE001 — missing/corrupt file -> documented fallback
        return None


def load_positions(raw_dir=None):
    """{player_id: PLAYER_POSITION} from the defended_fg (ptdefend) raw cache.
    Newest cached season wins on conflicts. Empty dict if cache absent
    (minutes tilt then uses the +2.4 flat fallback everywhere)."""
    import orjson
    d = Path(raw_dir) if raw_dir else ROOT / "data" / "raw" / "nba_api" / "ptdefend"
    docs = []
    for f in sorted(d.glob("*.json")) if d.is_dir() else []:
        try:
            j = orjson.loads(f.read_bytes())
            docs.append((j.get("params", {}).get("season") or "", j))
        except Exception:  # noqa: BLE001
            continue
    out = {}
    for _, j in sorted(docs, key=lambda t: t[0]):   # ascending -> newest last, wins
        try:
            rs = j["response"]["resultSets"][0]
            hdr = rs["headers"]
            ii = hdr.index("CLOSE_DEF_PERSON_ID")
            ip = hdr.index("PLAYER_POSITION")
        except Exception:  # noqa: BLE001
            continue
        for row in rs["rowSet"]:
            if row[ip]:
                out[int(row[ii])] = str(row[ip])
    return out


def minutes_tilt(pos_player, pos_star):
    """D39 positional minutes tilt for one remaining player given the out
    star's position. Positions are ptdefend strings ('G', 'F-C', ...); overlap
    of position letters counts as same-position (G-F vs F -> same)."""
    a = {c for c in str(pos_player or "").upper() if c in "GFC"}
    b = {c for c in str(pos_star or "").upper() if c in "GFC"}
    if not a or not b:
        return TILT_FLAT
    return TILT_SAME_POS if a & b else TILT_DIFF_POS


def compute_lift(weights, pool_ids, star_id, default=1.0):
    """D33 softmax attempts lift (FULL effect): S / (S - w_star) over the
    remaining-rotation pool + star, capped to [LIFT_LO, LIFT_HI]. `weights`
    maps player -> exp(u) (or trailing attempts in fallback mode); `default`
    fills unknown players (1.0 = exp(0), the gate's u.get(p, 0.0) convention)."""
    pool = set(pool_ids) | {star_id}
    S = sum(weights.get(p, default) for p in pool)
    w_star = weights.get(star_id, default)
    return float(np.clip(S / max(S - w_star, 1e-9), LIFT_LO, LIFT_HI))


def production_lift(weights, pool_ids, star_id, default=1.0):
    """Residual-calibrated attempts lift actually applied in production:
    1 + RESID_ATT_SCALE * (softmax_lift - 1). Keeps the gated softmax SHAPE,
    applies only the magnitude the live baseline has not already absorbed."""
    return 1.0 + RESID_ATT_SCALE * (compute_lift(weights, pool_ids, star_id,
                                                 default=default) - 1.0)


def production_tilt(pos_player, pos_star):
    """Residual-calibrated minutes tilt actually applied in production:
    RESID_MIN_SCALE * D39 tilt (positional ordering preserved)."""
    return RESID_MIN_SCALE * minutes_tilt(pos_player, pos_star)


def adjust_rates(rates, lift, tilt):
    """Return a NEW rates dict with attempts scaled by `lift` and proj_min
    shifted by `tilt`. Efficiency (fg_*, ft_pct) untouched (D34).

    REB/AST are held VOLUME-NEUTRAL under the tilt (per-min rates scaled by
    proj_old/proj_new): the validation shows the star-out reb/ast residual is
    tail-only (median per-row residual NEGATIVE, P(y>mu)=.48/.49, vs minutes
    median +0.88, P=.56) — a uniform exposure bump harms the typical row's
    CRPS (-0.4%/-0.3% CI-solid). The redistribution is a SHOOTING-role effect;
    minutes tilt flows to the shooting side only.

    minutes_hist is kept — simulate_player recenters empirical draws on the
    (tilted) proj_min, so the tilt flows through the D57-recentered path."""
    out = dict(rates)
    for k in ATTEMPT_KEYS:
        if out.get(k) is not None:
            out[k] = float(out[k]) * lift
    pm_old = float(out.get("proj_min", 30.0))
    pm_new = float(min(pm_old + tilt, PROJ_MIN_CAP))
    out["proj_min"] = pm_new
    if pm_new > 0:
        for k in ("reb_per_min", "ast_per_min"):   # volume-neutral vs base
            if out.get(k) is not None:
                out[k] = float(out[k]) * pm_old / pm_new
    return out


def team_context(con, team_id, out_ids, before, weights=None, positions=None):
    """Detect tonight's star-out for one team and precompute the attempts lift.

    out_ids: the OUT set predict_today already builds (injury-report Outs +
    departed players). Returns None when no >=28-trailing-min fresh star is
    out, else {star, lift, star_pos, usage_source, n_pool}.
    """
    out_ids = {int(p) for p in (out_ids or set())}
    if not out_ids:
        return None
    tmode = trail_mode()
    rows = con.execute(f"""
        WITH tg AS (
          SELECT s.player_id, s.seconds/60.0 AS m,
                 s.rima + s.mida + s.thra AS att, g.game_date,
                 row_number() OVER (PARTITION BY s.player_id
                                    ORDER BY g.game_date DESC) AS rn,
                 CASE WHEN s.seconds > 0 THEN
                   row_number() OVER (PARTITION BY s.player_id, s.seconds > 0
                                      ORDER BY g.game_date DESC) END AS rp
          FROM player_game_stats s
          JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
          WHERE s.team_id = ? AND s.game_id LIKE '002%' AND g.game_date < ?
        )
        SELECT player_id,
               avg(m)   FILTER (WHERE rn <= {TRAIL_GAMES}) AS trail_min,
               count(*) FILTER (WHERE rn <= {TRAIL_GAMES}) AS n_tr,
               max(game_date) FILTER (WHERE m > 0)         AS last_date,
               avg(att) FILTER (WHERE rn <= {TRAIL_GAMES}) AS trail_att,
               avg(m)   FILTER (WHERE rp <= {TRAIL_GAMES}) AS trail_min_pl,
               count(*) FILTER (WHERE rp <= {TRAIL_GAMES}) AS n_played,
               avg(att) FILTER (WHERE rp <= {TRAIL_GAMES}) AS trail_att_pl,
               count(*) FILTER (WHERE rn <= {TRAIL_GAMES} AND m > 0) AS n_pl_win
        FROM tg GROUP BY player_id
    """, [int(team_id), before]).fetchall()
    # SELECT order: 0 player_id | 1 trail_min | 2 n_tr | 3 last_date |
    #               4 trail_att | 5 trail_min_pl | 6 n_played | 7 trail_att_pl |
    #               8 n_pl_win
    # stats value tuple: (trail_min, n_tr, last_date, trail_att, n_played_in_window)
    if tmode == "played":
        stats = {int(r[0]): (r[5], int(r[6]), r[3], r[7], int(r[8])) for r in rows}
    else:
        stats = {int(r[0]): (r[1], int(r[2]), r[3], r[4], int(r[8])) for r in rows}
    if tmode == "floor":   # ARM B: mostly-absent players leave the model entirely
        stats = {p: v for p, v in stats.items() if v[4] >= PLAYED_FLOOR}
    stats = {p: v for p, v in stats.items() if v[0] is not None}

    def fresh(pid):
        """D146 (T0 correctness): last_date must be the last game the player
        actually PLAYED, which is what FRESH_DAYS is documented to mean ("star
        must have played for this team within 12d"). It was max(game_date) over
        ALL rows, and player_game_stats carries 38,311 DNP rows (18% of the 002
        table) with seconds=0 — so a benched player kept refreshing his own
        timestamp and the guard was inert (D145 measured P(fresh)=1.0000 in
        every absence bucket; 806 player-team pairs had last-row after
        last-played, by up to 1,710 days). None = never played for this team
        inside the window, which is not fresh."""
        ld = stats[pid][2]
        if ld is None:
            return False
        return 0 < (before - ld).days <= FRESH_DAYS

    star_cands = [p for p in out_ids if p in stats
                  and stats[p][1] >= MIN_TRAIL_GAMES
                  and stats[p][0] >= STAR_TRAILING_MIN and fresh(p)]
    # D85 trade-aware SUPPRESSION (regime-B, journal bd0208): once a departed
    # star's trade EXECUTES (he has since played for ANOTHER team), the
    # remaining-rotation lift collapses to placebo (post-trade k1-3 att 1.005
    # [0.918,1.093] vs benched 1.115; incoming players absorb 17->31% of team
    # FGA) — the redistribution science travels with absence-roster-intact,
    # not removal. Such stars are excluded from candidacy here; the seller-team
    # side is handled by roster_delta_context (attenuation) instead.
    star_cands = [p for p in star_cands
                  if not _departed_elsewhere(con, p, team_id, before)]
    if not star_cands:
        return None

    umode = usage_mode()
    if umode == "null_u":
        # D129 open question: pool arithmetic ONLY. Every player weighs 1, so
        # lift = N/(N-1) over the pool+star and data/v2_usage.npz is not read.
        w, default, src = {}, 1.0, "null_u"
    elif umode == "trailatt" or weights is None:
        # documented fallback: trailing-attempts shares (same S/(S-w) form)
        w = {p: max(float(stats[p][3] or 0.0), 1e-3) for p in stats}
        default = float(np.median(list(w.values()))) if w else 1.0
        src = "trailing-attempts"
    else:
        w, default, src = weights, 1.0, "v2_usage.npz"

    if umode == "null_u":
        # uniform weights cannot order the candidates; the pre-registered
        # tie-break is trailing ATTEMPTS (a box-score quantity, no fitted
        # artifact), which is also the module's own fallback ordering.
        sel = {p: max(float(stats[p][3] or 0.0), 1e-3) for p in stats}
        sel_default = 0.0
    else:
        sel, sel_default = w, default
    star = max(star_cands, key=lambda p: sel.get(p, sel_default))  # gate parity: max-usage
    pool = {p for p in stats if p not in out_ids
            and stats[p][1] >= MIN_TRAIL_GAMES
            and stats[p][0] >= ROT_TRAILING_MIN and fresh(p)}
    lift_full = compute_lift(w, pool, star, default=default)
    return {"star": star,
            "lift": 1.0 + RESID_ATT_SCALE * (lift_full - 1.0),  # production
            "lift_softmax": lift_full,                          # full D33 effect
            "star_pos": (positions or {}).get(star),
            "usage_source": src, "n_pool": len(pool),
            # diagnostics for the gate harnesses (not read by production)
            "star_trail_min": float(stats[star][0]), "pool": pool,
            "trail_mode": tmode, "usage_mode": umode}


def adjust_player_rates(rates, player_id, ctx, positions=None):
    """Apply the star-out adjustment to one remaining player's rates dict
    BEFORE simulate_player. No-op (same object back) when there is no star-out
    context, no rates, or the player IS the out star."""
    if not ctx or rates is None or int(player_id) == ctx["star"]:
        return rates
    tilt = production_tilt((positions or {}).get(int(player_id)), ctx.get("star_pos"))
    return adjust_rates(rates, ctx["lift"], tilt)


# ===================== D85 ROSTER-DELTA (trade-aware) =======================
# Regime-B science (topic nba-star-transitions, entries bd0208 / b89f4a /
# f3286e; scripts/rw_star_transitions.py, rw_star_joins.py, rw_c_split.py,
# rw_rtm_placebo.py — all placebo-calibrated):
#
#   DEPARTURE (seller team, trade EXECUTED = star has played for another
#   team): remaining-rotation lift collapses to placebo (post 1.005 vs benched
#   1.115) while newcomers absorb 17->31% of team FGA and incumbent trailing
#   rates OVER-project ~6-10% persistently through k~30 (roll 0.979-1.028 vs
#   roll-null 1.087; "seller-team equilibrium keeps shifting"). Term:
#   (a) suppression of the D33/D39 lift in team_context (above), (b) flat
#   incumbent attempts attenuation over the first RD_KMAX post-departure team
#   games.
#
#   ARRIVAL (buyer team, mid-season >=28-trailing-min arriver): incumbent
#   usage COMPRESSION, delayed and persistent — residual vs the LIVE trailing
#   baseline ~0 in k1-3 (arriver ramps, D60), -12% k4-7, -7% k8-12, -3..-5%
#   k13-30; NOT uniform (bottom-usage tercile shielded ~0, mid/top bear it);
#   minutes adj -2.2/game persistent, same-pos -3.6 vs diff -2.5 (D39 mirror).
#   Applies only to games the arriver PLAYS.
#
# MAGNITUDES are walk-forward residual-calibrated on the LIVE props baseline
# (scripts/validate_rosterdelta.py, train = events <= q0.6 by event date,
# netted against the same players' pre-event residuals) — the D83 lesson:
# science measured against a trailing-mean baseline double-counts if applied
# full-strength on top of the live EWMA baseline. Shapes are from the science;
# only the scales come from the residual fit.

RD_KMAX = 30                 # regime-B measurement window (team games)
ARR_STAR_MIN = 28.0          # arriver trailing-min at the OLD team
# (k_lo, k_hi, residual-vs-trailing shape) — b89f4a roll residuals
ARR_SHAPE = ((1, 3, 0.0), (4, 7, -0.12), (8, 12, -0.07), (13, RD_KMAX, -0.04))
ARR_TILT_SAME = -3.6         # incumbent minutes vs arriver position (full)
ARR_TILT_DIFF = -2.5
ARR_TILT_FLAT = -2.9         # unknown-position fallback (pool-weighted mean)
DEP_ATT_SHAPE = -0.08        # seller incumbent over-projection, flat k1-30
# residual scales (fraction of the shape magnitudes surviving on the LIVE
# baseline; FROZEN from the validate_rosterdelta.py train fit, events <=
# q0.6, netted vs pre-event windows — see D85). Measured train residuals:
# ARR net attempts 0.899/0.918/0.930 (k4-7/8-12/13-30, mid+top terciles;
# bottom 0.963 — shield holds), net dmin -1.95; DEP net 0.973 (the live
# EWMA absorbs most of the regime-B -8%: the D83 lesson applied).
ARR_ATT_SCALE = 1.08
ARR_MIN_SCALE = 0.64
DEP_ATT_SCALE = 0.34
ARR_MIN_FLOOR = 8.0          # tilted projection floor (sim truncates at 10)
ARR_FACTOR_LO = 0.70         # sanity clips on the applied attempt factors
DEP_FACTOR_LO = 0.85


def _departed_elsewhere(con, player_id, team_id, before):
    """True when the player's most recent 002 game strictly before `before`
    was for a DIFFERENT team — i.e. his trade/waiver has EXECUTED (PIT,
    box-score-observable). Mirrors predict_today's departed filter."""
    row = con.execute("""
        SELECT arg_max(s.team_id, g.game_date)
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
        WHERE s.player_id = ? AND s.game_id LIKE '002%' AND g.game_date < ?
    """, [int(player_id), before]).fetchone()
    return row is not None and row[0] is not None and int(row[0]) != int(team_id)


def arr_att_shape(k):
    """Arrival-compression residual shape at arriver-played game k (1-based)."""
    for lo, hi, v in ARR_SHAPE:
        if lo <= k <= hi:
            return v
    return 0.0


def arrival_att_factor(k, tercile):
    """Attempt factor for an incumbent at arriver-played game k. Bottom-usage
    tercile (tercile == 0) is SHIELDED (b89f4a: placebo-adjusted low 0.995 vs
    mid/top 0.910 — compression is not uniform-proportional)."""
    if tercile == 0:
        return 1.0
    return float(np.clip(1.0 + ARR_ATT_SCALE * arr_att_shape(k),
                         ARR_FACTOR_LO, 1.0))


def arrival_tilt(pos_player, pos_arriver):
    """Residual-calibrated incumbent minutes tilt (NEGATIVE) given the
    arriver's position; same-position incumbents lose most (D39 mirror)."""
    a = {c for c in str(pos_player or "").upper() if c in "GFC"}
    b = {c for c in str(pos_arriver or "").upper() if c in "GFC"}
    if not a or not b:
        full = ARR_TILT_FLAT
    else:
        full = ARR_TILT_SAME if a & b else ARR_TILT_DIFF
    return ARR_MIN_SCALE * full


def departure_att_factor():
    """Flat seller-team incumbent attempt factor over post-departure k1-30."""
    return float(np.clip(1.0 + DEP_ATT_SCALE * DEP_ATT_SHAPE,
                         DEP_FACTOR_LO, 1.0))


def _team_season_rows(con, team_id, before):
    """(season, schedule dates, per-player played rows) for the team's current
    season strictly before `before`. Played = >=8 min (regime-B convention)."""
    sea = con.execute("""
        SELECT season FROM nba_games
        WHERE team_id = ? AND game_id LIKE '002%' AND game_date < ?
        ORDER BY game_date DESC LIMIT 1""", [int(team_id), before]).fetchone()
    if not sea:
        return None, [], {}
    season = sea[0]
    sched = [r[0] for r in con.execute("""
        SELECT DISTINCT game_date FROM nba_games
        WHERE team_id = ? AND season = ? AND game_id LIKE '002%'
          AND game_date < ? ORDER BY game_date""",
        [int(team_id), season, before]).fetchall()]
    rows = con.execute("""
        SELECT s.player_id, g.game_date, s.seconds/60.0,
               s.rima + s.mida + s.thra
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games
              WHERE season = ?) g USING (game_id)
        WHERE s.team_id = ? AND s.game_id LIKE '002%' AND g.game_date < ?
        ORDER BY g.game_date""", [season, int(team_id), before]).fetchall()
    per = {}
    for p, d, m, att in rows:
        if m >= 8.0:
            per.setdefault(int(p), []).append((d, float(m), float(att)))
    return season, sched, per


def _trail(entries, before_date, lo=MIN_TRAIL_GAMES, n=TRAIL_GAMES):
    """(trailing minutes, trailing attempts) over the last <=n played games
    strictly before before_date; None if fewer than lo games."""
    prior = [(m, a) for (d, m, a) in entries if d < before_date]
    if len(prior) < lo:
        return None
    tail = prior[-n:]
    return (float(np.mean([m for m, _ in tail])),
            float(np.mean([a for _, a in tail])))


def roster_delta_context(con, team_id, before, out_ids=None, positions=None):
    """D85: detect an ACTIVE roster-delta window for this team tonight.

    Returns {"arr": ..., "dep": ...} (either side None) or None when no
    active window. All queries strictly before `before` (PIT).

      dep: a >=28-trailing star's last game for this team lies within the
           last RD_KMAX team games AND he has since played for another team
           (trade executed). Frozen incumbents get departure_att_factor().
      arr: a mid-season arriver (>=28 trailing at the old team as of the
           move) joined within the last RD_KMAX team games and is not OUT
           tonight. Frozen incumbents get arrival_att_factor(k, tercile) +
           arrival_tilt(); k = 1 + arriver's played games since arrival
           (compression is conditional on the arriver being on the floor).
    """
    out_ids = {int(p) for p in (out_ids or set())}
    season, sched, per = _team_season_rows(con, team_id, before)
    if not sched or not per:
        return None
    # this-season other-team rows for this team's players (arrival/departure)
    pids = list(per)
    ph = ",".join("?" * len(pids))
    other = con.execute(f"""
        SELECT s.player_id, s.team_id, g.game_date, s.seconds/60.0
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games
              WHERE season = ?) g USING (game_id)
        WHERE s.player_id IN ({ph}) AND s.team_id != ?
          AND s.game_id LIKE '002%' AND g.game_date < ?
        ORDER BY g.game_date""",
        [season] + pids + [int(team_id), before]).fetchall()
    oth = {}
    for p, t, d, m in other:
        if m >= 8.0:
            oth.setdefault(int(p), []).append((d, int(t), float(m)))

    def games_since(date, inclusive=False):
        return sum(1 for d in sched if (d >= date if inclusive else d > date))

    # ---- DEPARTURE: executed trade within the window
    dep = None
    for p, ent in per.items():
        last = ent[-1][0]
        post_other = [d for (d, t, m) in oth.get(p, []) if d > last]
        if not post_other:
            continue
        k_team = games_since(last)          # tonight would be game k_team + 1
        if k_team + 1 > RD_KMAX:
            continue
        # trailing INCLUDING the last game (event date = first game after it)
        if len(ent) < MIN_TRAIL_GAMES:
            continue
        tail = ent[-TRAIL_GAMES:]
        tmin = float(np.mean([m for (_, m, _) in tail]))
        if tmin < STAR_TRAILING_MIN:
            continue
        if dep is None or tmin > dep["star_trail"]:
            ev_date = min(d for d in sched if d > last)
            pool = set()
            for q, e2 in per.items():
                if q == p:
                    continue
                tr2 = _trail(e2, ev_date)
                if tr2 and tr2[0] >= ROT_TRAILING_MIN:
                    pool.add(q)
            dep = {"star": p, "star_trail": tmin, "k": k_team,
                   "pool": pool, "att_factor": departure_att_factor()}
    # ---- ARRIVAL: mid-season star joined within the window, not OUT tonight
    arr = None
    for p, ent in per.items():
        first = ent[0][0]
        prior_other = [(d, t, m) for (d, t, m) in oth.get(p, [])
                       if d < first]
        if not prior_other:
            continue
        k_team = games_since(first, inclusive=True)   # tonight = k_team + 1
        if k_team + 1 > RD_KMAX or p in out_ids:
            continue
        old_team = prior_other[-1][1]
        old_tail = [m for (d, t, m) in prior_other if t == old_team][-TRAIL_GAMES:]
        if len(old_tail) < MIN_TRAIL_GAMES or np.mean(old_tail) < ARR_STAR_MIN:
            continue
        if arr is None or float(np.mean(old_tail)) > arr["star_trail"]:
            pool, base_att = [], []
            for q, e2 in per.items():
                if q == p:
                    continue
                tr2 = _trail(e2, first)
                if tr2 and tr2[0] >= ROT_TRAILING_MIN:
                    pool.append(q)
                    base_att.append(tr2[1])
            terc = {}
            if pool:
                cuts = np.quantile(base_att, [1 / 3, 2 / 3])
                terc = {q: int(np.searchsorted(cuts, a, side="right"))
                        for q, a in zip(pool, base_att)}
            k_next = 1 + len(ent)     # arrival game itself was k=1
            arr = {"arriver": p, "star_trail": float(np.mean(old_tail)),
                   "k": min(k_next, RD_KMAX), "pool": set(pool),
                   "tercile": terc,
                   "arr_pos": (positions or {}).get(int(p))}
    if not dep and not arr:
        return None
    return {"dep": dep, "arr": arr}


def adjust_player_rates_rd(rates, player_id, rd, positions=None):
    """Apply the roster-delta adjustment to one incumbent's rates BEFORE
    simulate_player (after adjust_player_rates). No-op when no context or the
    player is outside the frozen incumbent pools. reb/ast per-min rates are
    left untouched under the arrival tilt (per-game reb/ast FLOW DOWN with
    the lost minutes — validated in validate_rosterdelta.py, unlike the
    star-out case where the residual was tail-only)."""
    if not rd or rates is None:
        return rates
    pid = int(player_id)
    out, changed = dict(rates), False
    dep, arr = rd.get("dep"), rd.get("arr")
    if dep and pid in dep["pool"] and pid != dep["star"]:
        for k in ATTEMPT_KEYS:
            if out.get(k) is not None:
                out[k] = float(out[k]) * dep["att_factor"]
        changed = True
    if arr and pid in arr["pool"] and pid != arr["arriver"]:
        f = arrival_att_factor(arr["k"], arr["tercile"].get(pid, 1))
        for k in ATTEMPT_KEYS:
            if out.get(k) is not None:
                out[k] = float(out[k]) * f
        tilt = arrival_tilt((positions or {}).get(pid), arr.get("arr_pos"))
        out["proj_min"] = float(max(float(out.get("proj_min", 30.0)) + tilt,
                                    ARR_MIN_FLOOR))
        changed = True
    return out if changed else rates
