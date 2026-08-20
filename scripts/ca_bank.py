#!/usr/bin/env python3
"""CARRY-ALL harness — the column BANK and the walk-forward subset solver.

WHAT THIS IS.  The owner's proposal is: stop asking "does this term PASS a
gate?" and start asking "does carrying it COST anything?".  Under the D46
shrunk schedule layer a dead coefficient shrinks toward zero and contributes
little, while retaining option value for eras/subsets where it is live.

The measurement therefore needs, for every candidate term, its value as an
ADDITIVE MARGIN COLUMN inside `production.fit_schedule_layer`'s design matrix,
fitted walk-forward with the SAME `n/(n+600)` shrinkage and the SAME fit-only
wpct control.  That is exactly `fit_schedule_layer_ext` (D136), generalised
from 4 registered arms to the whole rejected pile.

WHY A BANK.  The expensive part of a naive implementation is re-querying and
re-deriving every column at every weekly refit for every subset.  Instead we
evaluate every candidate column ONCE per game (they are all pregame-knowable
schedule / standings / venue quantities), cache the matrix, and then a refit
for ANY subset is a single lstsq on a row-slice of the cached matrix.  A
40-subset ladder then costs the same as one arm.

PIT DISCIPLINE.  Every column is a function of information strictly before the
game's own tip:
  * travel/venue/density/rest come from the SCHEDULE (known months ahead) and
    from the team's own PREVIOUS game;
  * standings columns (dead, urgency, lock, quit) use emit-before-update, i.e.
    the standings state as of the morning of the game;
  * form uses the team's trailing 5 COMPLETED games.
The same construction is used at fit time and at apply time (no second
construction — hall-of-shame #15).

READ-ONLY on the DB.  Nothing under nbapred/ is touched.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import nbapred.threads as _threads  # noqa: E402

_threads.pin(1)                     # BEFORE numpy: 5.5x on small solves

import numpy as np  # noqa: E402

from nbapred.model.production import (  # noqa: E402
    DEAD_GP, DEAD_WPCT, SCALE, SCHED_PRIOR, SCHED_SHRINK)
from nbapred.model.travel import build_state, venues  # noqa: E402

BANK_NPZ = REPO / "data" / "carryall_bank.npz"

# ---------------------------------------------------------------------------
# THE CLASS-(i) TERM LIST.  Each entry: (column name, source channel, register
# row it implements).  Order is FROZEN here and is the order the pre-registered
# ladder walks (see data/carryall_prereg.md).
# ---------------------------------------------------------------------------
TERMS = [
    # -- D47: already a FITTED column in fit_schedule_layer, never APPLIED.
    #    Carrying it means applying the coefficient the layer already computes.
    ("dead_h",       "dead",     "D47 dead-team FE (home)"),
    ("dead_a",       "dead",     "D47 dead-team FE (away)"),
    # -- D136 ARM A / B / C / D, verbatim constructions from travel.ARM_TERMS
    ("dtrav_kkm",    "travel",   "D136 ARM A travel fatigue, pts/1000km"),
    ("dtz_east",     "circad",   "D136 ARM B signed timezone crossings"),
    ("hret_h",       "roadtrip", "D136 ARM C homestand return"),
    ("rlen_extra_a", "roadtrip", "D136 ARM C road-trip length"),
    ("d3in4",        "density",  "D136 ARM D 3-in-4"),
    ("d5in7",        "density",  "D136 ARM D 5-in-7"),
    # -- D17/D48 rest advantage (win-prob), the channel beyond b2b
    ("drest",        "rest",     "D17/D48 rest-day advantage beyond b2b"),
    # -- D96 altitude with physio prior
    ("alt_home_km",  "altitude", "D96 home-venue altitude (physio prior)"),
    ("delev_km",     "altitude", "D96 acute altitude change (elev gain diff)"),
    # -- D130 ARM A / B / C late-season incentive family
    ("urg_d",        "urgency",  "D130 ARM A late-season urgency differential"),
    ("lock_d",       "clinch",   "D130 ARM B clinched/locked-seed letdown"),
    ("quit_d",       "quiturg",  "D130 ARM C quit x urgent interaction (proxy)"),
    # -- D71 F1 late-gated form
    ("form_d_late",  "form",     "D71 F1 late-gated form (gp>=55)"),
]
TERM_NAMES = [t[0] for t in TERMS]
N_TERMS = len(TERMS)

# D70/D20 team-specific home advantage is carried as a SEPARATE BLOCK of
# centred home-team dummies (one per franchise in the fit window, min-norm
# lstsq => sum-to-zero identification, exactly like team_ratings' home_dev).
TEAMHOME_BLOCK = "teamhome"

GP_ACTIVE = 55          # D130 / D71 late gate, taken verbatim
FORM_K = 5              # trailing games in the form window (D71 F1)
REST_CAP = 4.0


def _rng_noise(game_id: str, j: int) -> float:
    """Deterministic seeded N(0,1) noise column j for a game.

    Seeded on (game_id, j, salt) so the value is identical at fit time and at
    apply time and identical across runs, but is by construction independent of
    every real column and of the outcome.  This is the null benchmark: the cost
    of PARAMETER COUNT itself.
    """
    h = hashlib.sha256(f"carryall|{game_id}|{j}|20260802".encode()).digest()
    u = int.from_bytes(h[:8], "big") / 2 ** 64
    v = int.from_bytes(h[8:16], "big") / 2 ** 64
    u = min(max(u, 1e-12), 1 - 1e-12)
    return float(np.sqrt(-2.0 * np.log(u)) * np.cos(2.0 * np.pi * v))


def build_bank(con, n_noise: int = 48, verbose: bool = True) -> dict:
    """One row per regular-season game with a final score, chronological.

    Returns a dict of numpy arrays:
        gid, season, date (days since epoch), home_id, away_id, margin, y
        hb2b, ab2b, qd  (the incumbent/control columns)
        X   (n, N_TERMS)     the class-(i) candidate columns, in TERMS order
        Z   (n, n_noise)     seeded pure-noise columns
    """
    t0 = time.time()
    # DATA-QUALITY (found 2026-08-02, registered in data/carryall_notes.md):
    # 10 games -- 5 in 2024-25 and 5 in 2025-26, including the two 2025-26 NBA
    # Cup knockout games 0022501229/0022501230 -- carry `is_home = false` on
    # BOTH rows.  `fit_schedule_layer`'s `WHERE h.is_home AND NOT a.is_home`
    # silently drops them from the FIT frame, while `prod_by_season` /
    # `tv_gate` resolve the host from `matchup` and SCORE them.  We reproduce
    # both behaviours exactly: identity comes from `matchup` (so every scored
    # game is present), and `fit_ok` marks the rows the shipped fit frame
    # actually contains.
    raw = con.execute("""
        SELECT season, game_id, game_date, team_id, team_abbrev, matchup,
               is_home, pts, wl
        FROM nba_games WHERE game_id LIKE '002%' AND pts IS NOT NULL
        ORDER BY game_date, game_id
    """).fetchall()
    by_gid: dict[str, list] = {}
    for r in raw:
        by_gid.setdefault(r[1], []).append(r)
    g, fit_ok_l = [], []
    for gid in sorted(by_gid, key=lambda k: (by_gid[k][0][2], k)):
        recs = by_gid[gid]
        if len(recs) != 2:
            continue
        mu = recs[0][5] or ""
        host = mu.split("@")[-1].strip() if "@" in mu else mu.split("vs.")[0].strip()
        h = next((x for x in recs if x[4] == host), None)
        a = next((x for x in recs if x[4] != host), None)
        if h is None or a is None:
            continue
        g.append((h[0], gid, h[2], int(h[3]), h[4], int(a[3]), a[4],
                  float(h[7] - a[7]), h[8]))
        fit_ok_l.append(bool(h[6]) and not bool(a[6]))
    if verbose:
        print(f"[bank] {len(g)} games from DB in {time.time()-t0:.1f}s "
              f"(fit-frame eligible {sum(fit_ok_l)})", flush=True)

    # ---- per-team game-date chains (b2b, rest) -----------------------------
    lastg: dict[int, list] = {}
    for r in con.execute("""
        SELECT team_id, game_date FROM nba_games
        WHERE game_id LIKE '002%' AND pts IS NOT NULL ORDER BY game_date
    """).fetchall():
        d = r[1].date() if hasattr(r[1], "date") else r[1]
        lastg.setdefault(int(r[0]), []).append(d)
    prevpos = {t: {d: i for i, d in enumerate(ds)} for t, ds in lastg.items()}

    def prev_date(t, d):
        i = prevpos.get(t, {}).get(d)
        if i is None or i == 0:
            return None
        return lastg[t][i - 1]

    def is_b2b(t, d):
        p = prev_date(t, d)
        return p is not None and (d - p).days == 1

    def rest_days(t, d):
        p = prev_date(t, d)
        if p is None:
            return REST_CAP           # season opener: fully rested
        return min(float((d - p).days) - 1.0, REST_CAP)

    # ---- standings state, emit-before-update (D55 discipline) --------------
    wl = con.execute("""
        SELECT season, team_id, game_date, wl FROM nba_games
        WHERE game_id LIKE '002%' AND wl IS NOT NULL ORDER BY game_date
    """).fetchall()
    gp, wins, stand = {}, {}, {}
    for season, t, d0, w0 in wl:
        d0 = d0.date() if hasattr(d0, "date") else d0
        k = (season, int(t))
        stand[(int(t), d0)] = (gp.get(k, 0), wins.get(k, 0) / max(gp.get(k, 1), 1))
        gp[k] = gp.get(k, 0) + 1
        wins[k] = wins.get(k, 0) + (w0 == "W")

    def dead(t, d0):
        s = stand.get((int(t), d0))
        return 1.0 if (s is not None and s[0] >= DEAD_GP and s[1] < DEAD_WPCT) else 0.0

    # ---- urgency / lock, D130 construction verbatim ------------------------
    urg_map = _urgency_map(con, verbose=verbose)

    # ---- travel/venue state ------------------------------------------------
    t1 = time.time()
    st = build_state(con)
    A = venues()
    if verbose:
        print(f"[bank] travel state {len(st)} team-games in {time.time()-t1:.1f}s",
              flush=True)

    # ---- trailing-5 form, per team-season ----------------------------------
    form_hist: dict[int, list] = {}          # team -> [(date, margin)]
    for season, gid, d, ht, hab, a_t, aab, margin, hwl in g:
        d = d.date() if hasattr(d, "date") else d
        form_hist.setdefault(int(ht), []).append((d, float(margin)))
        form_hist.setdefault(int(a_t), []).append((d, -float(margin)))
    for t in form_hist:
        form_hist[t].sort()
    form_pos = {t: {d: i for i, (d, _) in enumerate(v)}
                for t, v in form_hist.items()}

    def form5(t, d):
        v = form_hist.get(int(t))
        i = form_pos.get(int(t), {}).get(d)
        if v is None or i is None or i < FORM_K:
            return 0.0
        return float(np.mean([m for _, m in v[i - FORM_K:i]]))

    # ---- assemble ----------------------------------------------------------
    n = len(g)
    col = {c: np.zeros(n) for c in TERM_NAMES}
    gid_a = np.empty(n, dtype=object)
    season_a = np.empty(n, dtype=object)
    date_a = np.empty(n, dtype="datetime64[D]")
    home_a = np.zeros(n, dtype=np.int64)
    away_a = np.zeros(n, dtype=np.int64)
    marg_a = np.zeros(n)
    y_a = np.zeros(n)
    hb_a = np.zeros(n)
    ab_a = np.zeros(n)
    qd_a = np.zeros(n)
    trav_ok = np.ones(n, dtype=bool)

    for i, (season, gid, d, ht, hab, a_t, aab, margin, hwl) in enumerate(g):
        d = d.date() if hasattr(d, "date") else d
        ht, a_t = int(ht), int(a_t)
        gid_a[i] = gid
        season_a[i] = season
        date_a[i] = np.datetime64(d, "D")
        home_a[i], away_a[i] = ht, a_t
        marg_a[i] = float(margin)
        y_a[i] = 1.0 if hwl == "W" else 0.0
        hb_a[i] = 1.0 if is_b2b(ht, d) else 0.0
        ab_a[i] = 1.0 if is_b2b(a_t, d) else 0.0
        sh_, sa_ = stand.get((ht, d)), stand.get((a_t, d))
        qd_a[i] = (sh_[1] if sh_ else 0.5) - (sa_[1] if sa_ else 0.5)

        dh, da = dead(ht, d), dead(a_t, d)
        col["dead_h"][i] = dh
        col["dead_a"][i] = da

        sh, sa = st.get((ht, d)), st.get((a_t, d))
        if sh is None or sa is None:
            trav_ok[i] = False
        else:
            ok = bool(sh["travel_valid"] and sa["travel_valid"])
            trav_ok[i] = ok
            col["dtrav_kkm"][i] = (sh["travel_km"] - sa["travel_km"]) / 1000.0
            col["dtz_east"][i] = sh["tz_east"] - sa["tz_east"]
            col["hret_h"][i] = sh["home_return"]
            col["rlen_extra_a"][i] = max(sa["road_len"] - 1.0, 0.0)
            col["d3in4"][i] = sh["is_3in4"] - sa["is_3in4"]
            col["d5in7"][i] = sh["is_5in7"] - sa["is_5in7"]
            col["delev_km"][i] = (sh["elev_gain_m"] - sa["elev_gain_m"]) / 1000.0
            v = sh.get("venue")
            col["alt_home_km"][i] = (A[v]["elev_m"] / 1000.0) if v in A else 0.0

        col["drest"][i] = rest_days(ht, d) - rest_days(a_t, d)

        uh, _, lh, gph = urg_map.get((ht, d), (0.0, 1, 0, 0))
        ua, _, la, gpa = urg_map.get((a_t, d), (0.0, 1, 0, 0))
        uh = uh if gph >= GP_ACTIVE else 0.0
        ua = ua if gpa >= GP_ACTIVE else 0.0
        lh = lh if gph >= GP_ACTIVE else 0
        la = la if gpa >= GP_ACTIVE else 0
        col["urg_d"][i] = uh - ua
        col["lock_d"][i] = float(lh - la)
        # D130 ARM C: quit x urgent.  PROXY (disclosed): the shipped tank score
        # is replaced by the dead-team flag, which is the same channel
        # (lottery-bound team) on data available for every historical season.
        col["quit_d"][i] = dh * ua - da * uh

        if gph >= GP_ACTIVE and gpa >= GP_ACTIVE:
            col["form_d_late"][i] = form5(ht, d) - form5(a_t, d)

    X = np.column_stack([col[c] for c in TERM_NAMES])
    Z = np.zeros((n, n_noise))
    for i in range(n):
        for j in range(n_noise):
            Z[i, j] = _rng_noise(str(gid_a[i]), j)

    if verbose:
        print(f"[bank] assembled {n} x {N_TERMS} (+{n_noise} noise) in "
              f"{time.time()-t0:.1f}s; travel_valid {trav_ok.mean():.4f}",
              flush=True)
    return dict(gid=gid_a, season=season_a, date=date_a, home=home_a,
                away=away_a, margin=marg_a, y=y_a, hb2b=hb_a, ab2b=ab_a,
                qd=qd_a, X=X, Z=Z, trav_ok=trav_ok, names=np.array(TERM_NAMES),
                fit_ok=np.array(fit_ok_l, dtype=bool))


def _urgency_map(con, verbose=True) -> dict:
    """(team_id, date) -> (urg, alive, lock, gp), D130 construction VERBATIM.

    `scripts/pg_urgency2.UrgencyModel` carries a `season >= FLOOR` corpus-floor
    literal (hall-of-shame #8; D138 measured what it costs: k_u 0.95/0.69 cold
    on the holdout vs 1.84/2.59/2.87 warm).  We monkey-patch the floor to the
    earliest season in the DB so the estimator is WARM on every season we
    score, and import the class otherwise unchanged.
    """
    import importlib
    sys.argv = [sys.argv[0]]                 # the module reads argv at import
    m = importlib.import_module("pg_urgency2")
    floor = con.execute(
        "SELECT min(season) FROM nba_games WHERE game_id LIKE '002%'").fetchone()[0]
    m.FLOOR = floor
    t0 = time.time()
    um = m.UrgencyModel(con)
    if verbose:
        print(f"[bank] urgency map {len(um.map)} team-dates (floor {floor}) "
              f"in {time.time()-t0:.1f}s", flush=True)
    return um.map


# ---------------------------------------------------------------------------
# THE WALK-FORWARD SUBSET SOLVER
# ---------------------------------------------------------------------------
class Layer:
    """Refits `fit_schedule_layer` and any carried subset of it at a date.

    Reproduces the shipped estimator EXACTLY when `cols=()` and
    `teamhome=False` (asserted in `ca_verify`): same trailing 730-day frame,
    same design matrix order [1, hb2b, ab2b, dead_h, dead_a, <extras>, qd],
    same `np.linalg.lstsq`, same `w = n/(n+SCHED_SHRINK)` shrinkage toward
    SCHED_PRIOR for the five shipped slots and toward 0.0 for every extra.
    The wpct control `qd` stays LAST and is FIT-ONLY, never applied.
    """

    # STANDINGS ARE WINDOW-TRUNCATED IN THE SHIPPED LAYER, and this is not
    # cosmetic.  `fit_schedule_layer` rebuilds `gp`/`wins` from the games it
    # pulled over a **760-day** window, keyed by (season, team).  Whenever a
    # season in the fit frame began BEFORE that 760-day cutoff, its games are
    # counted only from the cutoff, so `gp >= DEAD_GP` and the `wpct` control
    # are computed on a PARTIAL season record.  Measured effect on the fitted
    # dead coefficients: up to 3.4 points; leak into the APPLIED terms
    # (home edge / b2b): up to 0.12 points.  It costs nothing today because
    # `dead_h`/`dead_a` are fit but never applied -- but it is precisely the
    # term the owner is proposing to start carrying, so it is reproduced here
    # exactly rather than silently corrected.  (hall-of-shame #8, live.)
    STAND_WINDOW = 760

    # ---- PART B adaptation configs (data/carryall_prereg.md §6) -----------
    CP_LOOKBACK = 45      # C3 recent block, calendar days   (fixed ex ante)
    CP_Z = 2.5            # C3 break threshold, sigmas       (fixed ex ante)
    LONGPRIOR_DAYS = 1826  # C2 shrinkage target, 5 seasons  (fixed ex ante)

    def __init__(self, bank, window_days: int = 730, shrink: float = SCHED_SHRINK,
                 half_life_days: float | None = None, prior=SCHED_PRIOR,
                 trend: bool = False, prior_mode: str = "literal",
                 changepoint: bool = False, teamhome_ridge: float | None = None):
        self.b = bank
        self.window = window_days
        self.shrink = shrink
        self.half_life = half_life_days
        self.prior = tuple(prior)
        self.trend = bool(trend)              # C1 local-linear
        self.prior_mode = prior_mode          # C2 "literal" | "data5"
        self.changepoint = bool(changepoint)  # C3 variance-adaptive window
        # RECONCILIATION WITH D153: it carried the same D70 channel under an
        # explicit ridge (`team_home_ridge=200`, the shipped team_ratings
        # pattern) and measured -0.000613 ns; carried under the LAYER'S OWN
        # global n/(n+600) alone it is -0.01253 SIG.  Same channel, same
        # corpus, same layer -- the penalty is the only difference, so this
        # option makes the comparison internal instead of cross-harness.
        self.teamhome_ridge = teamhome_ridge
        self._break = None                    # C3 state, advances with time
        self.breaks = []
        self.dates = bank["date"].astype("datetime64[D]")
        self.di = self.dates.astype("int64")
        self._build_standings()
        self._cache = {}

    def _build_standings(self):
        """(team, season) -> (sorted date ints, cumulative wins) for the
        emit-before-update truncated standings."""
        rec: dict[tuple, list] = {}
        for i in range(len(self.di)):
            s = str(self.b["season"][i])
            d = int(self.di[i])
            hw = bool(self.b["y"][i])
            rec.setdefault((int(self.b["home"][i]), s), []).append((d, hw))
            rec.setdefault((int(self.b["away"][i]), s), []).append((d, not hw))
        self.st = {}
        for k, v in rec.items():
            v.sort()
            ds = np.array([x[0] for x in v], dtype=np.int64)
            cw = np.concatenate([[0], np.cumsum([1 if x[1] else 0 for x in v])])
            self.st[k] = (ds, cw)

    def _gp_w(self, team, season, d, lo):
        """(gp, wins) for `team` in `season` over games in [lo, d)."""
        e = self.st.get((int(team), season))
        if e is None:
            return 0, 0
        ds, cw = e
        j1 = int(np.searchsorted(ds, d, "left"))
        j0 = int(np.searchsorted(ds, lo, "left"))
        return j1 - j0, int(cw[j1] - cw[j0])

    def _dead_qd(self, idx, lo):
        """Window-truncated dead flags + wpct control for bank rows `idx`."""
        dh = np.zeros(len(idx)); da = np.zeros(len(idx)); qd = np.zeros(len(idx))
        for k, i in enumerate(idx):
            s = str(self.b["season"][i]); d = int(self.di[i])
            gh, wh = self._gp_w(self.b["home"][i], s, d, lo)
            ga, wa = self._gp_w(self.b["away"][i], s, d, lo)
            # NB `wins/max(gp,1)` is 0.0 -- NOT 0.5 -- at gp==0.  The shipped
            # `(sh[1] if sh else 0.5)` fallback fires only when the team-date
            # is absent from the window entirely, which never happens for a
            # row inside the fit frame.  A team's FIRST game of a season is
            # therefore entered into the wpct control at 0.000, not at 0.500.
            ph = wh / max(gh, 1); pa = wa / max(ga, 1)
            dh[k] = 1.0 if (gh >= DEAD_GP and ph < DEAD_WPCT) else 0.0
            da[k] = 1.0 if (ga >= DEAD_GP and pa < DEAD_WPCT) else 0.0
            qd[k] = ph - pa
        return dh, da, qd

    def _lo(self, before: dt.date):
        b = int(np.datetime64(before, "D").astype("int64"))
        lo = b - int(self.window)
        if self.changepoint:
            self._detect(before, b, lo)
            if self._break is not None:
                lo = max(lo, self._break)
        return lo

    def _detect(self, before, b, lo_full):
        """C3: declare a break when the most recent CP_LOOKBACK days depart
        from the incumbent window mean by more than CP_Z of the recent block's
        own SE.  Threshold fixed ex ante from the null, never from the
        endpoint (data/carryall_prereg.md §6)."""
        okm = self.b["fit_ok"]
        rec = np.where((self.di < b) & (self.di >= b - self.CP_LOOKBACK) & okm)[0]
        if len(rec) < 60:
            return
        win = np.where((self.di < b) & (self.di >= lo_full) & okm)[0]
        if self._break is not None:
            win = win[self.di[win] >= self._break]
        if len(win) < 200:
            return
        mr = float(self.b["margin"][rec].mean())
        mw = float(self.b["margin"][win].mean())
        se = float(self.b["margin"][rec].std(ddof=1) / np.sqrt(len(rec)))
        if se > 0 and abs(mr - mw) > self.CP_Z * se:
            nb = b - self.CP_LOOKBACK
            if self._break is None or nb > self._break:
                self._break = nb
                self.breaks.append((str(before), round(mr, 3), round(mw, 3),
                                    round((mr - mw) / se, 2)))

    def _prior(self, before: dt.date):
        """C2: shrink the home edge toward the trailing 5-season DATA mean
        instead of the frozen SCHED_PRIOR[0] literal."""
        if self.prior_mode != "data5":
            return self.prior
        b = int(np.datetime64(before, "D").astype("int64"))
        m = np.where((self.di < b) & (self.di >= b - self.LONGPRIOR_DAYS)
                     & self.b["fit_ok"])[0]
        if len(m) < 500:
            return self.prior
        return (float(self.b["margin"][m].mean()),) + tuple(self.prior[1:])

    def _rows(self, before: dt.date, lo: int):
        b = int(np.datetime64(before, "D").astype("int64"))
        return np.where((self.di < b) & (self.di >= lo) & self.b["fit_ok"])[0]

    def fit(self, before: dt.date, cols=(), teamhome=False, noise=0,
            extra_weight=None):
        """Return (base5, extras dict, thdev dict, n, w).

        `cols` are indices into bank['X']; `noise` is a count of leading
        columns of bank['Z'].
        """
        lo_fit = self._lo(before)              # advances C3 state exactly once
        idx = self._rows(before, lo_fit)
        n = len(idx)
        if n == 0:
            return self.prior, {}, {}, 0, 0.0, []
        key = (before, lo_fit)
        if key not in self._cache:
            lo = int(np.datetime64(before, "D").astype("int64")) - self.STAND_WINDOW
            self._cache[key] = self._dead_qd(idx, lo)
        dh, da, qdv = self._cache[key]
        # Columns 0 and 1 (dead_h / dead_a) are ALREADY in the shipped design
        # matrix -- production FITS them and simply never APPLIES them (D47).
        # "Carrying" those two therefore means applying the coefficient the
        # layer already computes, NOT adding a duplicate column; re-adding
        # them would make the design exactly collinear and split one
        # coefficient across two slots.
        cols = [int(c) for c in cols if int(c) not in (0, 1)]
        parts = [np.ones(n), self.b["hb2b"][idx], self.b["ab2b"][idx], dh, da]
        names = []
        for c in cols:
            parts.append(self.b["X"][idx][:, c])
            names.append(int(c))
        th_teams = []
        if teamhome:
            hs = self.b["home"][idx]
            th_teams = sorted(set(hs.tolist()))
            k = len(th_teams)
            for t in th_teams:
                parts.append((hs == t).astype(float) - 1.0 / k)
        nz = int(noise)
        for j in range(nz):
            parts.append(self.b["Z"][idx][:, j])
        # C1: centred time regressor, in SEASONS before `before`.  The layer's
        # intercept then estimates the era parameter AT `before` rather than
        # at the window's mean age (~1.0 season ago), which is the boxcar's
        # lag bias.  Shrunk toward 0 with the same w; it is APPLIED only
        # through the intercept (its own value at t=0 is exactly 0).
        n_trend = 0
        if self.trend:
            age = (int(np.datetime64(before, "D").astype("int64"))
                   - self.di[idx]) / 365.25
            parts.append(-age.astype(float))
            n_trend = 1
        parts.append(qdv)                              # fit-only control, LAST
        X = np.column_stack(parts)
        yv = self.b["margin"][idx]

        w_row = None
        if self.half_life is not None:
            age = (np.datetime64(before, "D") - self.dates[idx]) / np.timedelta64(1, "D")
            w_row = 0.5 ** (age.astype(float) / float(self.half_life))
        if extra_weight is not None:
            w_row = extra_weight(idx) if w_row is None else w_row * extra_weight(idx)
        if w_row is not None:
            sw = np.sqrt(w_row)
            beta = np.linalg.lstsq(X * sw[:, None], yv * sw, rcond=None)[0]
            n_eff = float(w_row.sum() ** 2 / max((w_row ** 2).sum(), 1e-12))
        else:
            if self.teamhome_ridge is not None and teamhome:
                P = np.zeros(X.shape[1])
                th0 = 5 + len(names)
                P[th0:th0 + len(th_teams)] = float(self.teamhome_ridge)
                beta = np.linalg.solve(X.T @ X + np.diag(P), X.T @ yv)
            else:
                beta = np.linalg.lstsq(X, yv, rcond=None)[0]
            n_eff = float(n)

        w = n_eff / (n_eff + self.shrink)
        pri = self._prior(before)
        base5 = tuple(w * beta[i] + (1 - w) * pri[i] for i in range(5))
        self.last = dict(n=n, n_eff=n_eff, w=w, prior0=pri[0],
                         slope=(float(beta[-2]) if self.trend else 0.0),
                         lo=int(lo_fit))
        off = 5
        extras = {}
        for j, c in enumerate(names):
            extras[c] = w * float(beta[off + j])
        off += len(names)
        thdev = {}
        for j, t in enumerate(th_teams):
            thdev[int(t)] = w * float(beta[off + j])
        off += len(th_teams)
        noise_b = [w * float(beta[off + j]) for j in range(nz)]
        return base5, extras, thdev, n, w, noise_b

    def sched_value(self, i, base5, extras, thdev, noise_b, apply_dead=()):
        """Schedule-layer margin contribution for bank row i.

        `apply_dead` is the subset of {0, 1} being carried: 0 applies the
        already-fitted home dead-team coefficient, 1 the away one.
        """
        s = base5[0]
        if self.b["hb2b"][i]:
            s += base5[1]
        if self.b["ab2b"][i]:
            s += base5[2]
        if 0 in apply_dead:
            s += base5[3] * self.b["X"][i, 0]
        if 1 in apply_dead:
            s += base5[4] * self.b["X"][i, 1]
        for c, v in extras.items():
            s += v * self.b["X"][i, c]
        if thdev:
            s += thdev.get(int(self.b["home"][i]), 0.0)
        for j, v in enumerate(noise_b):
            s += v * self.b["Z"][i, j]
        return s


def load_bank(con=None, n_noise=48, rebuild=False):
    if BANK_NPZ.exists() and not rebuild:
        z = np.load(BANK_NPZ, allow_pickle=True)
        return {k: z[k] for k in z.files}
    if con is None:
        from nbapred.db import connect
        con = connect(read_only=True)
    b = build_bank(con, n_noise=n_noise)
    np.savez_compressed(BANK_NPZ, **b)
    return b


if __name__ == "__main__":
    from nbapred.db import connect
    con = connect(read_only=True)
    b = load_bank(con, rebuild=("--rebuild" in sys.argv))
    print("bank rows", len(b["gid"]), "cols", b["X"].shape)
    import collections
    cnt = collections.Counter(b["season"].tolist())
    for s in sorted(cnt):
        print(f"  {s} {cnt[s]}")
