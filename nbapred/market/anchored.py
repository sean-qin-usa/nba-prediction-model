"""MARKET-ANCHORED CLV MODEL — predict the CLOSE from the OPEN (D147).

A PARALLEL model, explicitly NOT a replacement for the market-blind production
stack (nbapred/model/production.py et al.), which is untouched by this file.

METHOD, and the one place it differs from its ancestor.  The reference
architecture is /hdd/steveqin/sean_dev/football_exercise/submission/
sean_qin_model.py: it inverts each PAST match's devigged CLOSING price into a
strength observation and accumulates those observations in a ridge with
geometric forgetting, on the theory that a devigged close is a lower-noise
reading of team strength than the game's own box score.  (Its own measured
finding: goals/shots become monotonically HARMFUL as strength observations once
prices are known.)  We reproduce that engine in NBA margin space and test the
same claim here — `PriceRidge` is run three ways, over past CLOSES, past OPENS
and past RESULTS, and the three are horse-raced.

THE REFRAME.  The football model's target is the OUTCOME.  Ours is the CLOSING
PRICE — equivalently the line movement open->close.  We do not need to beat the
market's final answer, only its FIRST answer, and D121/D126 already show we do
(CLV +0.0097 universe / +0.019..+0.050 on the frozen rules).  An outcome-
targeted arm is fitted alongside for comparison, not as the primary.

STRICT PIT DISCIPLINE.  Every feature must be knowable BEFORE the bet is
placed, and the bet is placed at the OPEN.
  * The current game's CLOSE is the LABEL and is never a feature.
  * Past games' closes ARE legitimate: a close from a strictly earlier DATE is
    known by the time today's opener is posted.  `PriceRidge` therefore
    absorbs a day's games only AFTER every game on that day has been predicted
    (`flush_day`), so same-day closes can never leak sideways.
  * `assert_pit()` and tests/test_market_anchored.py enforce this by shuffling
    the future of the label and requiring bit-identical features.

FEATURE TIERS, because "knowable at bet time" is not one thing:
  * TIER A (LIVE) — the opening price of the game being predicted, the
    price-anchored strength ridge over past closes, our certified market-blind
    p_us, and schedule/tank state.  All posted or computable before the opener.
  * TIER B (DIAGNOSTIC, NOT LIVE) — the availability/OUT sets and the star-out
    flag.  The official inactive list lands ~30 minutes before tip, i.e. AFTER
    the open.  These are carried to MEASURE how much of the movement is
    news-arrival rather than mispricing; a Tier-B arm is an upper bound and is
    labelled as one everywhere it is reported.  (The frozen D75/D78/D82 rule
    STAR_FAV_SHARPER consumes star-out at the open by registered convention;
    that convention is inherited, not extended.)

Note on p_us: at D132 certified defaults INACTIVE_OUTS/REPORT_OUTS are UNSET,
so the production model does not consume the inactive list and p_us is
availability-blind — which is why it sits in Tier A.
"""
from __future__ import annotations

import numpy as np

SPREAD_SCALE = 6.96          # nbapred/ingest/kaggle_odds.py, program-wide
OVERROUND = 1.043            # D120 measured opening-ML overround (1.0431/1.0433)
MIN_DEC = 1.01

# ---- PriceRidge defaults.  Chosen to mirror the reference model's structure
# (a ridge that IS the cold-start prior + geometric forgetting), rescaled to
# NBA margin units.  h0 = the program's pooled home edge in points.
LAM_S = 3.0                  # ridge on team strengths (doubles as cold start)
LAM_H = 25.0                 # ridge on the home edge
H0 = 2.6                     # home-edge prior, points
DECAY = 0.995                # forgetting per absorbed game (82-game season;
                             # ~ half-life 138 team-games == the D143 h=21
                             # per-team half-life at 30 teams / 2 sides)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def am2dec(a):
    """American odds -> decimal.  NaN-safe.  (D120 caught a sign slip here.)"""
    a = np.asarray(a, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(a > 0, 1.0 + a / 100.0, 1.0 + 100.0 / np.abs(a))
    return np.where(np.isnan(a) | (a == 0), np.nan, d)


class PriceRidge:
    """Team strengths accumulated from PAST games' prices, ridge + forgetting.

        minimise  sum_k w_k (y_k - x_k . beta)^2 + sum_i lam_i (beta_i - m_i)^2

    x_k is +1 on the home team, -1 on the away team, +1 on the home-edge column;
    y_k is the observation being inverted from that past game.  With no data
    beta == prior, so the ridge term IS the cold-start rule (reference model's
    `Ridge`).  `decay()` forgets data geometrically while re-injecting the
    prior mass so it never decays away.

    The observation y is whatever channel the caller feeds:
      * `close_margin`  -> the devigged CLOSING price as a strength reading
                           (the football model's channel);
      * `open_margin`   -> the OPENING price (control: is the close really the
                           lower-noise reading?);
      * realised margin -> the box score (the channel the reference model
                           measured to be HARMFUL once prices are known).

    PIT: `observe()` only stages a game; `flush_day()` commits it.  Callers
    must flush a date only after every game on that date has been predicted.
    """

    def __init__(self, teams, lam=LAM_S, lam_h=LAM_H, h0=H0, decay=DECAY):
        self.teams = list(teams)
        self.idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams) + 1
        self.n = n
        self.hcol = n - 1
        self.lam = np.full(n, float(lam))
        self.lam[self.hcol] = float(lam_h)
        self.prior = np.zeros(n)
        self.prior[self.hcol] = float(h0)
        self.decay_f = float(decay)
        self.A = np.diag(self.lam).astype(float)
        self.b = self.lam * self.prior
        self._dg = np.arange(n)
        self._sol = self.prior.copy()
        self._dirty = False
        self._pending: list[tuple[int, int, float]] = []

    def beta(self):
        if self._dirty:
            self._sol = np.linalg.solve(self.A, self.b)
            self._dirty = False
        return self._sol

    def predict(self, home, away):
        """Implied HOME margin for a future game, from past prices only."""
        be = self.beta()
        hi = self.idx.get(home)
        ai = self.idx.get(away)
        if hi is None or ai is None:
            return float(be[self.hcol])
        return float(be[hi] - be[ai] + be[self.hcol])

    def predict_many(self, home_ix, away_ix):
        be = self.beta()
        return be[home_ix] - be[away_ix] + be[self.hcol]

    def observe(self, home, away, y):
        """STAGE one past game.  Not visible until flush_day()."""
        hi = self.idx.get(home)
        ai = self.idx.get(away)
        if hi is None or ai is None or not np.isfinite(y):
            return
        self._pending.append((hi, ai, float(y)))

    def flush_day(self):
        """COMMIT every staged game.  Call once per date, after predicting."""
        if not self._pending:
            return
        d = self.decay_f
        for hi, ai, y in self._pending:
            if d < 1.0:
                self.A *= d
                self.b *= d
                self.A[self._dg, self._dg] += (1.0 - d) * self.lam
                self.b += (1.0 - d) * self.lam * self.prior
            self.A[hi, hi] += 1.0
            self.A[ai, ai] += 1.0
            self.A[self.hcol, self.hcol] += 1.0
            self.A[hi, ai] -= 1.0
            self.A[ai, hi] -= 1.0
            self.A[hi, self.hcol] += 1.0
            self.A[self.hcol, hi] += 1.0
            self.A[ai, self.hcol] -= 1.0
            self.A[self.hcol, ai] -= 1.0
            self.b[hi] += y
            self.b[ai] -= y
            self.b[self.hcol] += y
        self._pending.clear()
        self._dirty = True


def run_price_ridge(home, away, day, obs, **kw):
    """Vectorised walk: returns the PIT ridge prediction for every row.

    `day` must be sorted ascending.  For every row the ridge has absorbed
    exactly the games on STRICTLY EARLIER dates — never today's, never the
    row's own label.  That is the PIT guarantee the shuffle test checks.
    """
    home = np.asarray(home)
    away = np.asarray(away)
    day = np.asarray(day)
    obs = np.asarray(obs, float)
    assert np.all(np.diff(day) >= 0), "rows must be date-sorted"
    teams = sorted(set(home.tolist()) | set(away.tolist()))
    r = PriceRidge(teams, **kw)
    hix = np.array([r.idx[t] for t in home])
    aix = np.array([r.idx[t] for t in away])
    out = np.empty(len(home))
    i, L = 0, len(home)
    while i < L:
        j = i
        while j < L and day[j] == day[i]:
            j += 1
        out[i:j] = r.predict_many(hix[i:j], aix[i:j])
        for k in range(i, j):
            r.observe(home[k], away[k], obs[k])
        r.flush_day()
        i = j
    return out


# --------------------------------------------------------------- PIT guard --
def assert_pit(build_fn, df, label_cols, day_col="day", seed=20260802,
               cuts=(0.35, 0.6, 0.85)):
    """Shuffle the FUTURE of the label; require the PAST-AND-PRESENT identical.

    The PIT contract for a bet placed at the OPEN of game i is:

        feature_i may read labels of games on dates STRICTLY BEFORE date_i,
        and nothing else.

    So the test is two-sided, and both halves matter:

      (1) FUTURE-BLINDNESS.  Permute every label column over all rows with
          date >= T.  Every feature on rows with date <= T must be
          bit-identical.  Note the `<=`: the rows ON the cut date are the
          interesting ones — they are inside the permuted block, so if a
          feature read a SAME-DAY label (a sideways leak within the slate)
          it moves here and nowhere else.
      (2) NON-VACUITY.  Permuting the WHOLE label history must MOVE the
          features, otherwise (1) passes for the trivial reason that the
          feature never reads the label at all.  A guard that cannot fail is
          not a guard (the D144 fresh-guard lesson).

    build_fn(df) -> (X, names).  Returns (violations, moved_under_full_shuffle).
    """
    X0, names = build_fn(df)
    rng = np.random.default_rng(seed)
    day = np.asarray(df[day_col].values)
    bad: list[str] = []
    for frac in cuts:
        T = np.quantile(day, frac)
        fut = day >= T
        keep = day <= T                       # includes the cut date itself
        if fut.sum() < 10 or keep.sum() < 10:
            continue
        d2 = df.copy()
        perm = rng.permutation(int(fut.sum()))
        for c in label_cols:
            v = d2[c].values.copy()
            v[fut] = v[fut][perm]
            d2[c] = v
        X1, _ = build_fn(d2)
        for k, nm in enumerate(names):
            a = np.nan_to_num(X0[keep, k], nan=-9e9)
            b = np.nan_to_num(X1[keep, k], nan=-9e9)
            if not np.array_equal(a, b) and nm not in bad:
                bad.append(nm)
    d3 = df.copy()
    for c in label_cols:
        d3[c] = rng.permutation(d3[c].values)
    X2, _ = build_fn(d3)
    moved = [names[k] for k in range(len(names))
             if not np.allclose(np.nan_to_num(X0[:, k], nan=0.0),
                                np.nan_to_num(X2[:, k], nan=0.0))]
    return bad, moved


# -------------------------------------------------------------- estimation --
def ridge_fit(X, y, lam=1.0, w=None):
    """Ridge with an unpenalised intercept.  X already carries a 1-column."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    p = X.shape[1]
    P = np.eye(p) * lam
    P[0, 0] = 0.0
    if w is None:
        A = X.T @ X + P
        b = X.T @ y
    else:
        w = np.asarray(w, float)
        A = (X * w[:, None]).T @ X + P
        b = (X * w[:, None]).T @ y
    return np.linalg.solve(A, b)


def standardise(Xtr, Xte):
    """z-score by TRAIN moments only (the intercept column is left alone)."""
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    mu[0], sd[0] = 0.0, 1.0
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd, mu, sd
