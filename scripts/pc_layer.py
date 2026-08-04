#!/usr/bin/env python3
"""PCA harness — a WALK-FORWARD rotation installer plus a per-term EB shrinkage
subclass of D154's `ca_bank.Layer`.

`ca_bank.Layer` is REUSED UNMODIFIED for arms A and B: principal components are
computed at each refit date from the fit rows only and written into RESERVED
extra columns of the bank's design matrix, so `Layer.fit(..., cols=[...])` and
`Layer.sched_value` need no changes at all.  Arm C needs a per-column shrinkage
weight, which the parent cannot express, so `EBLayer` subclasses it; with
`eb=False` the subclass is asserted BIT-IDENTICAL to the parent (see
`pc_leaktest.py`), the same discipline `ca_verify.py` applies to the shipped
layer.

Pre-registration: data/pca_prereg.md
sha256 0c3720ba99b668ff769b147d23aa45fa507eecd7b038a5e35bfdff44e1b63938

LEAKAGE (prereg §5).  `Rotation.fit` takes the row index set and ASSERTS that
every row it uses is dated strictly before the refit date.  Signs are pinned
deterministically.  The rotation is then applied to future rows; it is never
refit on a scored row.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import nbapred.threads as _t  # noqa: E402

_t.pin(1)
import datetime as dt  # noqa: E402

import numpy as np  # noqa: E402

from ca_bank import Layer  # noqa: E402

PILE13 = [i for i in range(15) if i not in (0, 1)]   # dead_h/dead_a excluded
PC_SLOT0 = 15                                        # reserved bank X columns
MAX_PC = 48


class Rotation:
    """A walk-forward correlation-matrix PCA of a design block.

    `fit(bank, rows, before, teamhome)` -> self, with `mu`, `sd`, `V`
    (columns = components, variance-ordered, signs pinned), `ev`, `r_rule`.
    """

    def __init__(self, var_target: float = 0.90, cols=None, teamhome=False):
        self.var_target = float(var_target)
        self.cols = list(PILE13 if cols is None else cols)
        self.teamhome = bool(teamhome)
        self.mu = self.sd = self.V = self.ev = None
        self.teams = []

    # -- block construction, identical at fit time and at apply time --------
    def block(self, bank, rows):
        M = bank["X"][rows][:, self.cols]
        if not self.teamhome:
            return M
        hs = bank["home"][rows]
        k = len(self.teams)
        D = np.column_stack([(hs == t).astype(float) - 1.0 / k
                             for t in self.teams])
        return np.column_stack([M, D])

    def fit(self, bank, rows, before: dt.date):
        # ---- R1: the rotation may only ever see strictly-prior rows -------
        bb = np.datetime64(before, "D")
        assert len(rows) > 0, "empty rotation frame"
        assert bank["date"][rows].max() < bb, (
            f"LEAK: rotation frame contains a row dated >= {before}")
        if self.teamhome:
            self.teams = sorted(set(bank["home"][rows].tolist()))
        M = self.block(bank, rows)
        self.mu = M.mean(0)
        sd = M.std(0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)
        Z = (M - self.mu) / self.sd
        C = (Z.T @ Z) / len(Z)
        ev, V = np.linalg.eigh(C)
        o = np.argsort(-ev)
        ev, V = np.maximum(ev[o], 0.0), V[:, o]
        # ---- R3: deterministic sign convention ---------------------------
        for j in range(V.shape[1]):
            i = int(np.argmax(np.abs(V[:, j])))
            if V[i, j] < 0:
                V[:, j] = -V[:, j]
        self.ev, self.V = ev, V
        cum = np.cumsum(ev) / ev.sum()
        self.r_rule = int(np.searchsorted(cum, self.var_target - 1e-12) + 1)
        self.p = M.shape[1]
        return self

    def project(self, bank, rows, r):
        """Component scores for `rows` under the frozen rotation."""
        M = self.block(bank, rows)
        return ((M - self.mu) / self.sd) @ self.V[:, :r]

    def key(self):
        """Hashable fingerprint, for the permutation proof."""
        return (np.round(self.mu, 12).tobytes(), np.round(self.sd, 12).tobytes(),
                np.round(self.ev, 12).tobytes(), np.round(self.V, 12).tobytes(),
                self.r_rule)


def install(bank, rot, rows_all, r):
    """Write `r` component scores for EVERY bank row into the reserved slots.

    The rotation was fit on strictly-prior rows; projecting future rows through
    it is the walk-forward APPLY step and is not leakage (prereg §5 R4).
    """
    assert r <= MAX_PC, "raise MAX_PC"
    P = rot.project(bank, rows_all, r)
    bank["X"][:, PC_SLOT0:PC_SLOT0 + r] = 0.0
    bank["X"][rows_all[:, None], np.arange(PC_SLOT0, PC_SLOT0 + r)[None, :]] = P
    return list(range(PC_SLOT0, PC_SLOT0 + r))


def widen(bank):
    """Return a copy of the bank whose X has MAX_PC reserved trailing slots."""
    b = dict(bank)
    n = len(bank["gid"])
    b["X"] = np.hstack([np.array(bank["X"], dtype=float),
                        np.zeros((n, MAX_PC))])
    return b


# ---------------------------------------------------------------------------
# ARM C — per-term empirical-Bayes shrinkage.
# ---------------------------------------------------------------------------
class EBLayer(Layer):
    """`Layer` with an optional per-coefficient shrinkage weight.

    `eb=False` reproduces the parent EXACTLY (asserted in pc_leaktest.py).
    `eb=True` replaces the global scalar `w = n/(n+600)` by
        w_j = t_j^2 / (1 + t_j^2),   t_j = beta_j / se_j
    on the CARRIED coefficients only: the extras, the team-home deviations, the
    noise columns, and slots 3/4 (the dead coefficients, whose prior is 0.0 so
    the substitution is clean).  Slots 0/1/2 -- the SHIPPED home edge and the
    two b2b terms -- keep the global `w`, so the arm is exactly "carry the pile
    under evidence-scaled shrinkage" and not a change to production's own
    terms (prereg §3 ARM C).
    """

    def __init__(self, *a, eb: bool = False, ridge: float | None = None,
                 ridge_dead: bool = False, **kw):
        super().__init__(*a, **kw)
        self.eb = bool(eb)
        self.ridge = ridge          # L2 penalty on the CARRIED block only
        self.ridge_dead = bool(ridge_dead)
        self.eb_diag = []
        self.edf = []

    def fit(self, before, cols=(), teamhome=False, noise=0, extra_weight=None):
        base5, extras, thdev, n, w, noise_b = super().fit(
            before, cols=cols, teamhome=teamhome, noise=noise,
            extra_weight=extra_weight)
        if (not self.eb and self.ridge is None) or n == 0:
            return base5, extras, thdev, n, w, noise_b

        # Rebuild the SAME design the parent just used.  Verified against the
        # parent's own coefficients below, so a divergence is caught, not
        # silently carried (ca_verify discipline).
        lo_fit = self.last["lo"]
        idx = self._rows(before, lo_fit)
        dh, da, qdv = self._cache[(before, lo_fit)]
        cols_x = [int(c) for c in cols if int(c) not in (0, 1)]
        parts = [np.ones(n), self.b["hb2b"][idx], self.b["ab2b"][idx], dh, da]
        for c in cols_x:
            parts.append(self.b["X"][idx][:, c])
        th_teams = []
        if teamhome:
            hs = self.b["home"][idx]
            th_teams = sorted(set(hs.tolist()))
            k = len(th_teams)
            for t in th_teams:
                parts.append((hs == t).astype(float) - 1.0 / k)
        for j in range(int(noise)):
            parts.append(self.b["Z"][idx][:, j])
        parts.append(qdv)
        X = np.column_stack(parts)
        yv = self.b["margin"][idx]
        beta = np.linalg.lstsq(X, yv, rcond=None)[0]
        # identity check against the parent's own numbers
        chk = max(abs(w * beta[i] - (base5[i] - (1 - w) * self._prior(before)[i]))
                  for i in range(5))
        assert chk < 1e-8, f"EBLayer design diverged from Layer.fit ({chk:.2e})"

        # ---- RIDGE FAMILY: L2 on the carried block only, lambda=0 == parent
        if self.ridge is not None:
            nb_ = 5 + len(cols_x) + len(th_teams) + int(noise)
            XtX = X.T @ X
            # SCALE-FREE penalty: P_j = lambda * (X'X)_jj / n, i.e. lambda on
            # the STANDARDISED column.  For the z-scored extras installed by
            # `install(..., V=I)` this is exactly lambda; for the raw 0/1 dead
            # columns it is the same penalty measured in their own units.
            sc = np.diag(XtX) / max(n, 1)
            P = np.zeros(X.shape[1])
            P[5:nb_] = float(self.ridge) * sc[5:nb_]
            if self.ridge_dead:
                P[3:5] = float(self.ridge) * sc[3:5]
            beta = np.linalg.solve(XtX + np.diag(P), X.T @ yv)
            H = np.linalg.solve(XtX + np.diag(P), XtX)
            lo_ = 3 if self.ridge_dead else 5
            self.edf.append(float(np.trace(H[lo_:nb_, lo_:nb_])))
            pri = self._prior(before)
            nb5 = [w * beta[i] + (1 - w) * pri[i] for i in range(5)]
            off = 5
            ex2 = {c: float(w * beta[off + j]) for j, c in enumerate(cols_x)}
            off += len(cols_x)
            th2 = {int(tt): float(w * beta[off + j])
                   for j, tt in enumerate(th_teams)}
            off += len(th_teams)
            nz2 = [float(w * beta[off + j]) for j in range(int(noise))]
            return tuple(nb5), ex2, th2, n, w, nz2

        p = X.shape[1]
        resid = yv - X @ beta
        s2 = float(resid @ resid) / max(n - p, 1)
        XtX = X.T @ X
        try:
            cov = np.linalg.pinv(XtX) * s2
        except np.linalg.LinAlgError:
            return base5, extras, thdev, n, w, noise_b
        se = np.sqrt(np.maximum(np.diag(cov), 1e-30))
        t = beta / se
        web = (t ** 2) / (1.0 + t ** 2)

        pri = self._prior(before)
        nb5 = list(base5)
        for i in (3, 4):                       # dead slots, prior 0.0
            nb5[i] = web[i] * beta[i] + (1 - web[i]) * pri[i]
        off = 5
        ex2 = {}
        for j, c in enumerate(cols_x):
            ex2[c] = float(web[off + j] * beta[off + j])
        off += len(cols_x)
        th2 = {int(tt): float(web[off + j] * beta[off + j])
               for j, tt in enumerate(th_teams)}
        off += len(th_teams)
        nz2 = [float(web[off + j] * beta[off + j]) for j in range(int(noise))]
        self.eb_diag.append(dict(
            date=str(before), n=int(n), w_global=round(float(w), 5),
            t_extras={int(c): round(float(t[5 + j]), 3)
                      for j, c in enumerate(cols_x)},
            w_eb={int(c): round(float(web[5 + j]), 4)
                  for j, c in enumerate(cols_x)},
            w_eb_dead=[round(float(web[3]), 4), round(float(web[4]), 4)],
            t_dead=[round(float(t[3]), 3), round(float(t[4]), 3)],
            mean_w_eb=round(float(np.mean(web[3:off])), 4) if off > 3 else 0.0))
        return tuple(nb5), ex2, th2, n, w, nz2
