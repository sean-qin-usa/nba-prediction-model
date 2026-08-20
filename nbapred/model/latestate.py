"""Late-state layer v1 (D90; gate scripts/ov_latestate_gate.py, pooled
+0.00189 CI(+0.00053,+0.00329), active window +0.00560 CI(+0.00122,+0.00976)).

production margin += c_f * (form5_home - form5_away)
                   + c_o * (n_out_home - n_out_away)
ACTIVE ONLY when either team's games-played (before the game) >= 55 —
identical activation to the D73 tank term (tanking.py); adds exactly 0.0
otherwise, so games with both teams gp<55 are bitwise-unchanged.

FEATURES (PIT):
  form5   = same-season trailing-5-completed-game mean signed margin,
            strictly before the game date; 0.0 until 5 games.
  n_out_* = size of the OUT sets the caller already passes to margin/p_home
            (live: injury-report OUT sets; headline backtest: oracle outs).

COEFFICIENTS (production idiom, mirrors tanking.TankModel.fit_k /
fit_schedule_layer, but DAILY walk-forward — evaluated at each game_date,
not at the weekly refit; this matches the pre-registered gate exactly):
  rows: ALL completed games strictly before game_date with either team
        gp>=55 (CORPUS FLOOR onward; out_diff = 0 for the floor season's
        rows, per the pre-registered spec — the out-count reconstruction
        needs a season of roster history behind it)
  OLS:  home_margin ~ [1, tank_diff, form5_diff, out_diff, wpct_diff]
        tank_diff and wpct_diff are FIT-TIME CONTROLS ONLY (they de-confound
        the form/outs terms from the shipped tank term and from team
        quality); the shipped tank k is NOT refit (regime-D L4 evidence:
        joint refit is worse than riding the shipped k).
  c_f = max(0, n/(n+600) * beta_form);  c_o = min(0, n/(n+600) * beta_out);
  both 0 until 20 active rows.

FIT-FRAME OUT COUNTS (design difference vs the gate, documented like
D73-SHIPPED's k-estimator difference): the gate accumulated the counts its
own run passed to p_home (weekly-refit CompositionModel roster, <=6 days
stale); production reconstructs them self-contained from player_game_stats
as-of each game date — a player is OUT for team t at date d iff his most
recent >=12-minute appearance strictly before d was with t within 12 days
(the CompositionModel roster definition: seconds>=720 builds the roster)
and he did not appear (seconds>0) in the game. Verified vs the gate arm by
the registered capstone check (per-season |delta| <= 0.0005).

LIVE PATH (D68 discipline): the layer reads gp state from the SAME
TankModel predict_today primes with tonight's virtual rows, form5/coefs
from completed games strictly before the date, and out counts from the OUT
sets the live caller passes — so the live number is by construction what
the backtest later computes. get_latestate_model keys its cache on the
tank-model instance: re-priming the tank rebuilds the layer.

CORPUS FLOOR (D112): the burn-in season was the hardcoded literal
`BURN_IN_SEASON = "2022-23"` until 2026-08-01 (hall-of-shame #8 in the D110
audit — it made this layer identically ZERO on 2021-22 and cold on 2022-23,
so the term was UNTESTABLE on the only genuine holdout we have). It is now
taken from tanking.season_floor() via the TankModel this layer is built on,
so the two stay coherent (this layer reads its gp state from that same tank
map, and a season below the tank floor has no gp state at all).
TANK_SEASON_FLOOR overrides both.
"""
from __future__ import annotations

import datetime as dt
from bisect import bisect_left

import numpy as np

from .tanking import GP_ACTIVE, get_tank_model

FORM_N = 5
ROSTER_DAYS = 12              # = CompositionModel.ROSTER_DAYS
ROSTER_MIN_SECONDS = 720      # = CompositionModel roster filter (12+ min)
C_MIN_ACTIVE = 20             # coefs stay 0 until this many active rows
C_SHRINK = 600.0              # n/(n+600) shrink toward 0 (= SCHED_SHRINK)
BURN_IN_FALLBACK = "2022-23"  # pre-D112 literal (used only if tank.floor absent)


def _to_date(d):
    if hasattr(d, "date") and not isinstance(d, dt.date):
        return d.date()
    if isinstance(d, dt.datetime):
        return d.date()
    return d


class LateStateModel:
    """form5 lookup + self-contained active fit frame + daily (c_f, c_o)."""

    def __init__(self, con, tank):
        self.tank = tank
        # burn-in season = the tank model's corpus floor (D112); out_diff is
        # zeroed there and out counts are reconstructed for later seasons only
        self.burn_in = getattr(tank, "floor", None) or BURN_IN_FALLBACK
        rows = con.execute("""
            SELECT g.season, g.game_id, g.game_date, g.team_id, g.team_abbrev,
                   g.matchup, g.pts, o.pts AS opp_pts
            FROM nba_games g
            JOIN nba_games o ON o.game_id = g.game_id
                            AND o.team_id <> g.team_id
            WHERE g.game_id LIKE '002%' AND g.pts IS NOT NULL
              AND o.pts IS NOT NULL AND g.season >= ?
            ORDER BY g.game_date, g.game_id""", [self.burn_in]).fetchall()
        # -- per-team completed-game history for form5 ----------------------
        self.hist: dict[int, list[tuple[dt.date, str, float]]] = {}
        for season, gid, d, tid, ab, m, pts, opp in rows:
            self.hist.setdefault(int(tid), []).append(
                (_to_date(d), season, float(pts - opp)))
        self._hist_dates = {t: [r[0] for r in h] for t, h in self.hist.items()}
        # -- self-contained historical out counts per (team_id, game_id) ----
        outc = self._out_counts(con)
        # -- active fit frame (date-ordered) --------------------------------
        ids_by_gid: dict[str, dict] = {}
        for season, gid, d, tid, ab, m, pts, opp in rows:
            ids_by_gid.setdefault(gid, {})[ab] = int(tid)
        wl: dict[tuple[str, int], list[int]] = {}
        seen: set = set()
        fd, fy, fx = [], [], []
        for season, gid, d, tid, ab, m, pts, opp in rows:
            if gid in seen:
                continue
            seen.add(gid)
            ids = ids_by_gid[gid]
            host = (m.split("@")[-1].strip() if "@" in m
                    else m.split("vs.")[0].strip())
            if host not in ids or len(ids) != 2:
                continue
            ht = ids[host]
            at = next(v for k, v in ids.items() if k != host)
            margin = float(pts - opp) if ht == int(tid) else float(opp - pts)
            d = _to_date(d)
            wh = wl.setdefault((season, ht), [0, 0])
            wa = wl.setdefault((season, at), [0, 0])
            gp_h = self.tank.score(ht, d)[1]
            gp_a = self.tank.score(at, d)[1]
            if gp_h >= GP_ACTIVE or gp_a >= GP_ACTIVE:
                od = (0.0 if season == self.burn_in else
                      float(outc.get((ht, gid), 0) - outc.get((at, gid), 0)))
                fd.append(np.datetime64(d))
                fy.append(margin)
                fx.append((self.tank.diff(ht, at, d),
                           self.form5(ht, d) - self.form5(at, d), od,
                           (wh[0] / wh[1] if wh[1] else 0.5)
                           - (wa[0] / wa[1] if wa[1] else 0.5)))
            hw = margin > 0
            wh[0] += int(hw); wh[1] += 1
            wa[0] += int(not hw); wa[1] += 1
        self._act_dates = np.array(fd, dtype="datetime64[D]")
        self._act_y = np.array(fy, float)
        self._act_X = np.array(fx, float).reshape(len(fy), 4)
        self._coef_cache: dict[dt.date, tuple[float, float]] = {}

    # ------------------------------------------------------------------ --
    def _out_counts(self, con) -> dict[tuple[int, str], int]:
        """{(team_id, game_id): oracle out count} reconstructed as-of each
        game date (see module docstring). Burn-in season excluded (spec)."""
        app = con.execute("""
            SELECT s.player_id, s.team_id, g.game_date, s.game_id,
                   s.seconds >= ? AS big
            FROM player_game_stats s
            JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games
                  WHERE game_id LIKE '002%') g USING (game_id)
            WHERE s.seconds > 0 AND g.season > ?
            ORDER BY g.game_date""",
            [ROSTER_MIN_SECONDS, self.burn_in]).fetchall()
        tg = con.execute("""
            SELECT game_date, game_id, team_id FROM nba_games
            WHERE game_id LIKE '002%' AND pts IS NOT NULL AND season > ?
            ORDER BY game_date, game_id""", [self.burn_in]).fetchall()
        by_date_ros: dict[dt.date, list] = {}
        played: dict[str, set] = {}
        for pid, tid, d, gid, big in app:
            played.setdefault(gid, set()).add((int(tid), int(pid)))
            if big:
                by_date_ros.setdefault(_to_date(d), []).append(
                    (int(pid), int(tid)))
        by_date_tg: dict[dt.date, list] = {}
        for d, gid, tid in tg:
            by_date_tg.setdefault(_to_date(d), []).append((gid, int(tid)))
        roster: dict[int, dict[int, dt.date]] = {}   # team -> {player: last}
        where: dict[int, int] = {}                   # player -> roster team
        out: dict[tuple[int, str], int] = {}
        for d in sorted(set(by_date_ros) | set(by_date_tg)):
            for gid, tid in by_date_tg.get(d, ()):
                pl = played.get(gid, set())
                c = 0
                for p, last in roster.get(tid, {}).items():
                    if (d - last).days <= ROSTER_DAYS and (tid, p) not in pl:
                        c += 1
                out[(tid, gid)] = c
            for pid, tid in by_date_ros.get(d, ()):
                old = where.get(pid)
                if old is not None and old != tid:
                    roster.get(old, {}).pop(pid, None)
                roster.setdefault(tid, {})[pid] = d
                where[pid] = tid
        return out

    # ------------------------------------------------------------------ --
    def form5(self, team_id: int, game_date) -> float:
        """Same-season trailing-5 mean signed margin strictly before
        game_date; 0.0 until 5 same-season completed games."""
        d = _to_date(game_date)
        h = self.hist.get(int(team_id))
        if not h:
            return 0.0
        i = bisect_left(self._hist_dates[int(team_id)], d)
        if i == 0:
            return 0.0
        season = h[i - 1][1]
        lo = max(0, i - FORM_N)
        if any(h[j][1] != season for j in range(lo, i)):
            return 0.0            # fewer than 5 same-season games yet
        return float(np.mean([h[j][2] for j in range(lo, i)])) \
            if i - lo == FORM_N else 0.0

    def coefs(self, game_date) -> tuple[float, float]:
        """Daily walk-forward (c_f, c_o) on active rows strictly before
        game_date; n/(n+600) shrink; sign guards; 0 until 20 rows."""
        d = _to_date(game_date)
        c = self._coef_cache.get(d)
        if c is not None:
            return c
        m = self._act_dates < np.datetime64(d)
        n = int(m.sum())
        if n < C_MIN_ACTIVE:
            c = (0.0, 0.0)
        else:
            X = np.c_[np.ones(n), self._act_X[m]]
            beta = np.linalg.lstsq(X, self._act_y[m], rcond=None)[0]
            sh = n / (n + C_SHRINK)
            c = (float(max(0.0, beta[2] * sh)), float(min(0.0, beta[3] * sh)))
        self._coef_cache[d] = c
        return c

    def term(self, home_id: int, away_id: int, out_home, out_away,
             game_date) -> float:
        """The late-state margin addition; exactly 0.0 outside the gp>=55
        window or without a date."""
        if game_date is None:
            return 0.0
        d = _to_date(game_date)
        gp_h = self.tank.score(home_id, d)[1]
        gp_a = self.tank.score(away_id, d)[1]
        if gp_h < GP_ACTIVE and gp_a < GP_ACTIVE:
            return 0.0
        c_f, c_o = self.coefs(d)
        if c_f == 0.0 and c_o == 0.0:
            return 0.0
        fdiff = self.form5(home_id, d) - self.form5(away_id, d)
        odiff = float(len(out_home or ()) - len(out_away or ()))
        return c_f * fdiff + c_o * odiff


# --------------------------------------------------------------------------
# module cache (heavy build once per process / per DB state / per tank prime)
# --------------------------------------------------------------------------

_CACHE: dict = {}


def get_latestate_model(con) -> LateStateModel:
    """Cached LateStateModel for the current DB state AND current (possibly
    virtual-primed) TankModel instance — re-priming the tank rebuilds the
    layer so live gp lookups stay coherent (predict_today primes the tank
    with tonight's slate BEFORE fit_production, exactly as for D73)."""
    tank = get_tank_model(con)
    sig = con.execute("""SELECT count(*), max(game_date) FROM nba_games
        WHERE game_id LIKE '002%'""").fetchone()
    key = (int(sig[0]), str(sig[1]), id(tank))
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = LateStateModel(con, tank)
    return _CACHE[key]
