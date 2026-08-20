"""Per-player prop simulator — the engine's real edge zone (H-B).

Given a player's trailing per-minute rates + a minutes projection, Monte-Carlo
their stat line (points, threes, rebounds, assists) and read off prop
probabilities P(over line). This is the piece a team-ratings model CANNOT
produce: a full per-player stat DISTRIBUTION, natively.

Generative model per simulated game:
  minutes  ~ Normal(proj_min, sd_min)                     (clipped >=0)
  zone attempts_z ~ Poisson(rate_z * minutes)             z in {rim,mid,thr}
  makes_z         ~ Binomial(attempts_z, fg%_z)           (EB-shrunk fg%)
  ft attempts     ~ Poisson(fta_per_min * minutes)
  ft makes        ~ Binomial(ft_att, ft%)
  points = 2*rim_m + 2*mid_m + 3*thr_m + ft_m
  rebounds ~ Poisson(reb_per_min * minutes)
  assists  ~ Poisson(ast_per_min * minutes)

Poisson(attempts) x Binomial(makes) gives the right over-dispersed count
variance for props. FG%_z should come EB-shrunk (skill_priors) so thin samples
don't produce absurd tails. Rates fed in TRAILING (leakage-safe; caller owns).
"""
from __future__ import annotations

import datetime as dt
import os

import numpy as np


def _if_none(v, default):
    """Fallback ONLY when the estimate is missing (None). A valid 0.0 must
    survive — `or` treated 0-for-N shooters as missing and silently swapped in
    the league average (FEATURE_LEDGER hall-of-shame #6, silent defaults)."""
    return default if v is None else v


# --- D133 EARLY-MINUTES RAMP -------------------------------------------------
# The minutes projection conditions its history on games actually PLAYED
# (seconds >= 720), so it is structurally BLIND TO ABSENCE: a player returning
# from missed games is projected at his pre-absence role and does not get that
# role back for ~10 games. This is the fix D79 queued ("explicit gated early
# minutes ramp"), D86 named ("the right gated fix") and D128 quantified.
#
# The defect is NOT calendar-driven. The identical decay in the player's own
# games-played appears at team-gp>=30 (February/March), where no October
# rotation-expansion story can apply; October only *looks* special because in
# October the whole universe has low gp (a composition effect). Independent
# register corroboration: D76/H2 measured the return-from-absence minutes cap
# at -4.9 min with a ~3-game ramp, at the win-probability endpoint.
#
# b(gp) = mean(proj_min - realized minutes) on the props eval universe
# (002, seconds>=720, n_games>=8, proj_min>=20), fit on 2019-20..2025-26,
# n=116,638. Gated walk-forward (each season scored with coefficients fit only
# on strictly-prior seasons); the table is stationary across all five fit
# cutoffs (gp0 leg 2.75/2.87/3.08/3.10/3.17).
# Zero at gp>=20: K=20 from D48's "cold-start ramp /20 games" precedent and the
# estimator's own memory (minutes_hist is exactly 20 long; with
# half_life_games=10 the current-season weight share is 0.75 at gp=20).
#
# GATE (D133, scripts/pr_ramp_gate.py, prereg data/props_ramp_prereg.md):
# dev 2023-24..2025-26, n=12,618 Oct-Nov rows / 481 players, 4,000 sims/row,
# paired cluster bootstrap by player. Points dCRPS +0.03909
# CI[+0.03161,+0.04697] SIG (3.5x realized MDE80), PIT 0.4820 -> 0.4979
# (October alone 0.4579 -> 0.5013), Dec-Jun +0.00423 SIG better, 3/3 seasons.
# A matched single-constant level knob captures only 42.6% (A - A0 = +0.02245
# SIG), and the spread-only ablation is SIG HARMFUL — the defect is LOCATION.
# Set PROPS_MIN_RAMP=0 to restore the pre-D133 behaviour (same-run controls).
MINUTES_RAMP = ((1, 3.041), (3, 2.016), (6, 0.992), (10, 0.340),
                (15, -0.014), (20, -0.112))       # (gp_exclusive_upper, bias)


def minutes_ramp(gp: int) -> float:
    """Minutes to SUBTRACT from proj_min for a player with `gp` games played
    this season before the cutoff. 0.0 at gp>=20 (and when the switch is off),
    so the term is exactly zero outside its pre-registered window."""
    if os.environ.get("PROPS_MIN_RAMP", "1") == "0":
        return 0.0
    for hi, b in MINUTES_RAMP:
        if gp < hi:
            return b
    return 0.0


# --- D145 ABSENCE RAMP -------------------------------------------------------
# D133 fixed the PROXY, not the AXIS. Its ramp is keyed on games-PLAYED and is
# IDENTICALLY ZERO at gp>=20 — 65% of the eval universe — yet the mechanism it
# named is ABSENCE, and absence is fully present there. Measured residual of the
# SHIPPED (D133-ramped) estimator vs realized minutes, by games missed among the
# player's team's last 10 (scripts/ab_props_design.py, n=115,274):
#
#   miss10                0       1       2      3-4     5-7     8-10
#   gp>=20 (ramp == 0)  -0.316  -0.117  +0.001  +0.035  +0.827  +2.968
#   ... & team-gp>=30   -0.317  -0.112  -0.021  +0.012  +0.831  +2.962
#
# The mid-season row is numerically identical, so this is NOT a calendar effect
# (D133's own control). Per-season contrast (miss>=3 minus miss==0, gp>=20):
# +0.791/+0.940/+0.926/+0.808/+1.191 — 5/5 seasons.
#
# b(miss10) = mean(proj_min_after_D133_ramp - realized) on the D128 eval
# universe (002, seconds>=720, n_games>=8, proj_min>=20), fit on
# 2019-20..2025-26, n=114,624. ZERO for miss10<=4: the walk-forward fit there is
# -0.03..+0.10 with an UNSTABLE SIGN across all five cutoffs (the estimator is
# unbiased), while bucket{5,7} = 0.653/0.670/0.752/0.770/0.785 and bucket{8,10}
# = 3.019/2.740/2.692/2.756/2.853 are stationary. Forcing zero outside the
# window makes the term provably an absence correction, not a level knob.
#
# GATE (D145, scripts/ab_props_gate.py, prereg data/absence_prereg.md sha256
# c6c7044c...): 5 seasons, n=8,924 window rows / 591 players, 4,000 sims/row.
# Points CRPS +0.06001, SEASON-CLUSTERED CI[+0.04743,+0.07505] SIG (the §9.1
# shipping CI), cluster-mean t at 4 dof CI[+0.03845,+0.08328] SIG, block
# bootstrap SIG, rolling-origin 4/4 SIG, LOSO 5/5 SIG, ERA-STABLE (I2=41%,
# Q p=0.166), PIT 0.4741 -> 0.5036. In the D133-INERT region (gp>=20) alone:
# +0.05755 CI[+0.04285,+0.07205] SIG. A matched constant knob on the same
# window captures 82.5% (A - A0 = +0.01051 SIG), and replacing the gp ramp
# entirely is WORSE (A - C = +0.00407 SIG) — the two axes are complementary.
# Set PROPS_ABSENCE_RAMP=0 to restore pre-D145 behaviour (same-run controls).
MISS_RAMP = ((5, 0.0), (8, 0.858), (11, 2.987))   # (miss_exclusive_upper, bias)


def absence_ramp(miss10: int) -> float:
    """Minutes to SUBTRACT from proj_min for a player who missed `miss10` of
    his team's last 10 games before the cutoff. Exactly 0.0 for miss10<=4 and
    when the switch is off, so the term is zero outside its window."""
    if os.environ.get("PROPS_ABSENCE_RAMP", "1") == "0":
        return 0.0
    for hi, b in MISS_RAMP:
        if miss10 < hi:
            return b
    return MISS_RAMP[-1][1]   # clamp UP, mirroring the gate's bucket_of()


# --- PER-CHANNEL RAMP (D133 open item 3 / D145 open item 17b) ---------------
# The D133 gp ramp and the D145 absence ramp both subtract minutes from the SAME
# `proj_min`, and every channel then consumes that one number. Both were fitted
# POINTS-optimally. D133 registered the consequence: October REBOUND PIT
# overshoots 0.4833 -> 0.5229, because rebounds scale ~linearly in minutes while
# points do not, so the points-optimal shift OVER-corrects rebounds.
#
# STRUCTURAL FACT (verified, not assumed): only REBOUNDS and ASSISTS are
# separable. `points` and `threes` are read off the SAME zone-attempt draws, so
# a channel-specific minutes exposure for threes necessarily moves points; a
# per-channel treatment for those two is not defined. Points/threes are
# therefore BITWISE UNCHANGED by every mode below.
#
# lam_c = the fraction of the total ramp subtraction that channel c should
# receive (1.0 = today's shared behaviour). Fitted by the first-moment
# condition E[rate_c * (m0 - lam_c*D)] = E[y_c] on the fit seasons only.
# Mode "A": shift the channel's minutes exposure (location).
# Mode "B": keep the shared exposure, scale the channel's rate to the SAME
#           first moment (dispersion-matched ablation, D133 ARM B's role).
# PROPS_CHANNEL_RAMP unset/"0" = OFF (pre-gate behaviour, bitwise).
CHANNEL_LAM = {"reb": 1.0, "ast": 1.0}


def channel_ramp_mode() -> str:
    v = (os.environ.get("PROPS_CHANNEL_RAMP") or "0").strip().upper()
    return v if v in ("A", "B") else "0"


def games_missed_last10(con, df, season_now, before) -> int:
    """D145 absence axis, PIT by construction.

    Number of games among the player's team's LAST 10 games strictly before the
    cutoff (same season, at or after his first >=12-min game of that season for
    that team) in which he did NOT record a >=12-min game. "His team" = the team
    of his most recent >=12-min 002 game strictly before the cutoff.

    `df` is the already-fetched >=12-min history (must carry team_id and
    game_date). One extra query — the team's schedule — which is why D133
    declined this axis; D145 pays it.
    """
    from ..config import current_season
    cut = before if before is not None else dt.date.today()
    team = int(df["team_id"].to_numpy()[-1])
    own = {(d.date() if hasattr(d, "date") else d)
           for d, s, t in zip(df["game_date"], df["_season"], df["team_id"])
           if s == season_now and int(t) == team}
    if not own:
        return 0
    first = min(own)
    rows = con.execute("""
        SELECT DISTINCT game_date FROM nba_games
        WHERE season = ? AND team_id = ? AND game_id LIKE '002%'
          AND game_date < ?
        ORDER BY game_date DESC LIMIT 10""",
        [season_now, team, cut]).fetchall()
    prior = [(r[0].date() if hasattr(r[0], "date") else r[0]) for r in rows]
    return int(sum(1 for d in prior if d >= first and d not in own))


def simulate_player(rates: dict, n: int = 10000, seed: int = 0) -> dict:
    """rates keys: proj_min, sd_min, rate_rim, rate_mid, rate_thr (attempts/min),
    fg_rim, fg_mid, fg_thr, fta_per_min, ft_pct, reb_per_min, ast_per_min.
    Returns arrays for points, threes, rebounds, assists."""
    rng = np.random.default_rng(seed)
    hist = rates.get("minutes_hist")
    if hist is not None and len(hist) >= 5:
        # sample from the player's EMPIRICAL recent minutes (captures the real
        # skew: blowout/foul-trouble/DNP low games), not a symmetric Normal.
        # D57 re-gate finding: this branch's CENTER was the flat hist mean,
        # not the recency-weighted proj_min the gates evaluate — recentering
        # the empirical draws on proj_min keeps the validated shape (skew,
        # blowout tail) and fixes the center (+0.09-0.12 min MAE, PASS 10/10
        # seeds, all 3 seasons; scripts/audit_xminutes_720.py).
        h = np.asarray(hist, float)
        base = rng.choice(h, n) - h.mean() + float(rates.get("proj_min", h.mean()))
        mins = np.clip(base + rng.normal(0, 2.0, n), 0, 48)
    else:
        # TRUNCATED (not clipped) at 10 min: graded props condition on the player
        # actually playing, so sub-10-min draws shouldn't exist in the scored
        # universe — truncation shifts the conditional mean up, matching the
        # played-more/played-less asymmetry seen in PIT localization.
        mins = rng.normal(rates["proj_min"], rates.get("sd_min", 4.0), n)
        bad = mins < 10
        for _ in range(4):
            if not bad.any():
                break
            mins[bad] = rng.normal(rates["proj_min"], rates.get("sd_min", 4.0), bad.sum())
            bad = mins < 10
        mins = np.clip(mins, 10, 48)

    def zone_pts(rate, fg, val):
        att = rng.poisson(np.maximum(rate, 0) * mins)
        made = rng.binomial(att, np.clip(fg, 0, 1))
        return made, made * val

    rim_m, rim_p = zone_pts(rates["rate_rim"], rates["fg_rim"], 2)
    mid_m, mid_p = zone_pts(rates["rate_mid"], rates["fg_mid"], 2)
    thr_m, thr_p = zone_pts(rates["rate_thr"], rates["fg_thr"], 3)
    ft_att = rng.poisson(np.maximum(rates.get("fta_per_min", 0), 0) * mins)
    ft_m = rng.binomial(ft_att, np.clip(rates.get("ft_pct", 0.77), 0, 1))

    points = rim_p + mid_p + thr_p + ft_m
    # --- per-channel ramp (OFF by default; points/threes above are already
    # drawn, so they are bitwise unaffected in every mode) -------------------
    cmode = channel_ramp_mode()
    delta = float(rates.get("ramp_delta", 0.0))       # minutes the ramps removed
    reb_mins, reb_mult = mins, 1.0
    ast_expo = float(np.clip(rates.get("proj_min", 30.0), 10, 44))
    ast_mult = 1.0
    if cmode != "0" and delta > 0.0:
        pm = float(rates.get("proj_min", 30.0))
        lr, la = CHANNEL_LAM["reb"], CHANNEL_LAM["ast"]
        if cmode == "A":     # location: give the channel back (1-lam)*delta
            reb_mins = np.clip(mins + (1.0 - lr) * delta, 0, 48)
            ast_expo = float(np.clip(pm + (1.0 - la) * delta, 10, 44))
        else:                # B: same first moment, shared exposure (dispersion)
            reb_mult = (pm + (1.0 - lr) * delta) / pm if pm > 0 else 1.0
            ast_mult = ((float(np.clip(pm + (1.0 - la) * delta, 10, 44)) / ast_expo)
                        if ast_expo > 0 else 1.0)
    rebounds = rng.poisson(np.maximum(rates.get("reb_per_min", 0), 0) * reb_mult * reb_mins)
    # assists: independent exposure (proj_min scalar), NOT the shared minutes
    # draw — real within-player corr(pts,ast)=+0.04; sharing minutes made the
    # sim +0.26 (6x too coupled -> overpriced pts+ast parlays). D30.
    assists = rng.poisson(np.maximum(rates.get("ast_per_min", 0), 0) * ast_mult * ast_expo, size=n)
    return {"points": points, "threes": thr_m, "rebounds": rebounds, "assists": assists}


def team_pace(con, team_id: int, before=None):
    """Team possessions per game (pace), for scaling a player's counting stats:
    facing a fast team -> more possessions -> more attempts. Trailing/leakage-safe."""
    date_clause = "AND g.game_date < ?" if before else ""
    params = [team_id] + ([before] if before else [])
    row = con.execute(f"""
        SELECT sum(s.fga + 0.44*s.fta - s.oreb + s.tov) poss, count(DISTINCT s.game_id) g
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
        WHERE s.team_id = ? AND s.game_id LIKE '002%' {date_clause}
    """, params).fetchone()
    if not row or not row[1]:
        return 99.5
    return float(row[0]) / row[1]


LEAGUE_PACE = 99.5


def apply_pace(rates: dict, opp_pace: float, own_pace: float) -> dict:
    """Scale volume rates by the projected game pace vs the pace the player's
    rates were measured at. Game pace ~ average of the two teams' paces."""
    game_pace = 0.5 * (own_pace + opp_pace)
    mult = game_pace / LEAGUE_PACE
    out = dict(rates)
    for k in ("rate_rim", "rate_mid", "rate_thr", "fta_per_min", "reb_per_min", "ast_per_min"):
        out[k] = rates[k] * mult
    return out


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def team_zone_defense(con, team_id: int, before=None, league=None):
    """What a team ALLOWS by zone (opponent FG% in the team's games), EB-shrunk.
    Returns logit shifts {rim,mid,thr}: shooter make-prob shifts by this when
    facing this defense. Negative = good defense (suppresses that zone)."""
    lg = league or {"rim": 0.613, "mid": 0.44, "thr": 0.359}
    date_clause = "AND g.game_date < ?" if before else ""
    params = [team_id, team_id] + ([before] if before else [])
    row = con.execute(f"""
        SELECT sum(o.rimm) rm, sum(o.rima) ra, sum(o.midm) mm, sum(o.mida) ma,
               sum(o.thrm) tm, sum(o.thra) ta
        FROM player_game_stats o
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
        WHERE o.game_id IN (SELECT DISTINCT game_id FROM player_game_stats
                            WHERE team_id = ? AND game_id LIKE '002%')
          AND o.game_id LIKE '002%' AND o.team_id <> ? {date_clause}
    """, params).fetchone()
    if not row or not row[1]:
        return {"rim": 0.0, "mid": 0.0, "thr": 0.0}
    K = {"rim": 200, "mid": 250, "thr": 300}
    shift = {}
    for z, (mk, at) in zip(("rim", "mid", "thr"),
                           ((row[0], row[1]), (row[2], row[3]), (row[4], row[5]))):
        if not at:
            shift[z] = 0.0; continue
        allow = (mk + K[z] * lg[z]) / (at + K[z])
        shift[z] = float(_logit(allow) - _logit(lg[z]))   # <0 => good D
    return shift


def apply_opp_defense(rates: dict, opp_shift: dict) -> dict:
    """Shift a shooter's zone FG% by the opponent's zone-defense (defender-aware)."""
    out = dict(rates)
    for z, key in (("rim", "fg_rim"), ("mid", "fg_mid"), ("thr", "fg_thr")):
        out[key] = float(1 / (1 + np.exp(-(_logit(rates[key]) + opp_shift.get(z, 0.0)))))
    return out


def player_rates_kalman(con, player_id: int, before=None):
    """REJECTED, NOT ON ANY LIVE PATH (kept only because audit_kalman_720.py /
    cg_kalman_clean.py import it to reproduce their numbers).

    Same rate profile as player_rates_from_stats, but each per-minute rate is a
    Kalman-filtered latent estimate instead of EWMA. Decisively beaten by EWMA
    once the universe was aligned and the 002 filter was on: -0.00137 (-0.90%)
    CI[-0.00166,-0.00107] SIG, n=63,393 player-games, sign-consistent 3/3
    seasons (D103). No named future gate — do not wire this into props.py's
    live path or predict_today without re-gating it first."""
    from ..model.form_filter import FormFilter
    date_clause = "AND g.game_date < ?" if before else ""
    params = [player_id] + ([before] if before else [])
    df = con.execute(f"""
        SELECT g.game_date, s.seconds, s.rima, s.mida, s.thra, s.fta, s.oreb, s.dreb,
               s.ast, s.rimm, s.midm, s.thrm, s.ftm
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
        WHERE s.player_id = ? AND s.game_id LIKE '002%' {date_clause}
        ORDER BY g.game_date
    """, params).fetchdf()
    df = df[df["seconds"] > 0]
    if len(df) < 3:
        return None
    mins = df["seconds"].to_numpy() / 60.0
    dates = np.array([d.toordinal() for d in df["game_date"]])
    # D128: the D79 forward step is REVERTED to the no-op it replaced. D79
    # advanced the filter from the last observed game to the prediction date
    # (P += Q*dt, phi**dt), which is the form_filter DESIGN — but it measured
    # WORSE than the no-op on the pre-registered endpoint: CRPS +0.00378
    # CI[+0.00112,+0.00643] SIG, MAE +0.00461 SIG, PIT unchanged (0.5023 vs
    # 0.5022). D118 precedent: a principled implementation that measures OOS
    # harm loses to the measurement. Set PROPS_KALMAN_FWD=1 to restore it.
    target = before if before is not None else dt.date.today()
    fwd_days = (max(0.0, float(target.toordinal() - dates[-1]))
                if os.environ.get("PROPS_KALMAN_FWD") == "1" else 0.0)

    def kfilt(counts):
        rate = counts / mins
        f = FormFilter(float(np.average(rate, weights=mins)), prior_var=0.05,
                       Q=1e-4, meas_base=6.0)
        last = None
        for i in range(len(df)):
            f.predict(0.0 if last is None else dates[i] - last)
            f.update(rate[i], mins[i]); last = dates[i]
        f.predict(fwd_days)  # forward step: last game -> prediction date
        return max(f.theta, 0.0)

    def fg(mk, at):
        num, den = df[mk].sum(), df[at].sum()
        return float(num / den) if den > 5 else None

    proj_min = float(np.average(mins, weights=np.arange(1, len(mins) + 1)))  # recency-lean minutes
    return {
        "proj_min": proj_min, "sd_min": max(float(mins.std()), 2.0),
        "rate_rim": kfilt(df["rima"].to_numpy()), "rate_mid": kfilt(df["mida"].to_numpy()),
        "rate_thr": kfilt(df["thra"].to_numpy()),
        "fg_rim": _if_none(fg("rimm", "rima"), 0.60),
        "fg_mid": _if_none(fg("midm", "mida"), 0.42),
        "fg_thr": _if_none(fg("thrm", "thra"), 0.35),
        "fta_per_min": kfilt(df["fta"].to_numpy()),
        "ft_pct": _if_none(fg("ftm", "fta"), 0.77),
        "reb_per_min": kfilt((df["oreb"] + df["dreb"]).to_numpy()),
        "ast_per_min": kfilt(df["ast"].to_numpy()), "n_games": len(df),
    }


def prop_prob(samples: np.ndarray, line: float) -> dict:
    """P(over), P(under) for a prop line (half-points avoid pushes)."""
    over = float(np.mean(samples > line))
    return {"line": line, "p_over": over, "p_under": 1 - over,
            "mean": float(samples.mean()), "sd": float(samples.std())}


def player_rates_from_stats(con, player_id: int, before=None, half_life_games: float = 10.0):
    """Build a player's trailing per-minute rate profile from player_game_stats
    (EWMA by recency). before = date cutoff (leakage-safe)."""
    date_clause = "AND g.game_date < ?" if before else ""
    params = [player_id] + ([before] if before else [])
    df = con.execute(f"""
        SELECT g.game_date, s.team_id, s.seconds, s.rima, s.rimm, s.mida, s.midm,
               s.thra, s.thrm, s.fta, s.ftm, s.oreb, s.dreb, s.ast
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
        WHERE s.player_id = ? AND s.game_id LIKE '002%' {date_clause}
        ORDER BY g.game_date
    """, params).fetchdf()
    # Condition on games MEANINGFULLY played (>=12 min): the eval/prop universe
    # is games the player actually played, and mixing in DNP-ish stubs biased
    # minutes/rates down (the persistent PIT~0.54 under-prediction).
    df = df[df["seconds"] >= 720]
    if len(df) < 3:
        return None
    # recency weights (newer games weigh more)
    age = np.arange(len(df))[::-1]
    w = 0.5 ** (age / half_life_games)
    mins = df["seconds"].to_numpy() / 60.0
    wm = w * mins  # weight rates by minutes*recency

    def per_min(col):
        return float(np.sum(w * df[col].to_numpy()) / np.sum(wm))

    def fg(mk, at):
        num, den = np.sum(w * df[mk].to_numpy()), np.sum(w * df[at].to_numpy())
        return float(num / den) if den > 5 else None

    # NOTE: EWMA(hl5) minutes won raw one-step MAE (4.231 vs 4.306) but FAILED
    # the conditioned prop-CRPS gate (5.473 vs 5.336) — it chases dips that
    # don't apply to games the player actually plays 20+. Keep recency-lean avg.
    proj_min = float(np.sum(w * mins) / np.sum(w))
    sd_min = float(np.sqrt(np.average((mins - proj_min) ** 2, weights=w)))
    # D133 early-minutes ramp: gp = this player's 002/>=12-min games THIS season
    # strictly before the cutoff (all rows here already are). Season is derived
    # from the game date, so no extra query and no season argument is needed.
    from ..config import current_season
    season_now = current_season(before)
    df = df.assign(_season=[current_season(d) for d in df["game_date"]])
    gp = int((df["_season"] == season_now).sum())
    proj_min_raw = proj_min
    proj_min = max(proj_min - minutes_ramp(gp), 0.0)
    # D145 absence ramp: subtract the residual bias the games-PLAYED ramp above
    # cannot see (it is identically zero at gp>=20). Exactly 0.0 for miss10<=4,
    # i.e. on 89.7% of the universe, so the pre-D145 path is bitwise preserved
    # there and PROPS_ABSENCE_RAMP=0 restores it everywhere.
    miss10 = games_missed_last10(con, df, season_now, before)
    proj_min = max(proj_min - absence_ramp(miss10), 0.0)
    return {
        "proj_min": proj_min, "sd_min": max(sd_min, 2.0),
        # total minutes the two ramps removed; consumed only by the per-channel
        # ramp (default OFF) and by the gate harnesses.
        "ramp_delta": float(proj_min_raw - proj_min),
        "minutes_hist": mins[-20:],   # recent actual minutes (empirical, skewed)
        "rate_rim": per_min("rima"), "rate_mid": per_min("mida"), "rate_thr": per_min("thra"),
        "fg_rim": _if_none(fg("rimm", "rima"), 0.60),
        "fg_mid": _if_none(fg("midm", "mida"), 0.42),
        "fg_thr": _if_none(fg("thrm", "thra"), 0.35),
        "fta_per_min": per_min("fta"),
        "ft_pct": _if_none(fg("ftm", "fta"), 0.77),
        "reb_per_min": float(np.sum(w * (df["oreb"] + df["dreb"]).to_numpy()) / np.sum(wm)),
        "ast_per_min": per_min("ast"),
        "n_games": len(df),
    }
