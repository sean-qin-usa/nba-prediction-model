"""audit_kalman_720.py — AUDIT: was the Kalman-vs-trailing props ablation confounded?

Finding that motivated this script
----------------------------------
nbapred/engine/props.py has two rate builders that are documented as "the same
rate profile", but they train on DIFFERENT universes:

    player_rates_kalman      -> df = df[df["seconds"] > 0]     (any playing time)
    player_rates_from_stats  -> df = df[df["seconds"] >= 720]  (>=12 min only)

scripts/ablate_kalman_props.py compared those two functions head to head, so its
"Kalman is a wash" result is a JOINT test of (estimator) x (training universe)
x (minutes model: the EWMA path ships `minutes_hist` into simulate_player, the
Kalman path does not and falls back to the truncated Normal).  Any one of the
three could own the result.  D12 in docs/DECISIONS.md ("wash in props") rests on
that confounded comparison.

What this script does
---------------------
Isolates the ESTIMATOR by aligning the universe and scoring the rate directly
(no simulate_player, so the minutes-model confound is gone too):

  arm ewma720   incumbent EWMA (half_life_games=10) on seconds>=720   [as shipped]
  arm kal720    Kalman/FormFilter, ALIGNED to seconds>=720            [the fix]
  arm kal0      Kalman/FormFilter on seconds>0                        [as shipped]
  arm kal720fwd kal720 + the forward predict step to the target date  [diagnostic]
  arm career720 minutes-weighted career-to-date mean = the Kalman's own prior
                centre; bounds how much the filter adds over its anchor, and
                shows the metric has signal at all                    [diagnostic]
  arm kal720p   kal720 with Poisson-implied measurement noise R=lambda/m instead
                of the shipped constant meas_base=6.0                 [exploratory]

The third confound, for the record: the EWMA path returns `minutes_hist` and the
Kalman path does not, so simulate_player draws minutes from the EMPIRICAL
distribution for one arm and a truncated Normal for the other.  Scoring rates
directly removes that too -- but it means the original CRPS ablation stays
uninterpretable even after the universe is aligned.

Target = one-step-ahead per-minute scoring rates (rim/mid/thr attempts per
minute) on player-games with seconds>=720, regular season (game_id LIKE '002%').
Score = minutes-weighted mean absolute error,

    WMAE_z = sum_i m_i * |rate_hat_z,i - rate_z,i| / sum_i m_i

which equals mean |predicted attempts - actual attempts| per minute of exposure.
Reported per zone and summed over the three zones ("all3").

PIT discipline: every arm sees only games with game_date STRICTLY < the target
game_date, exactly as props.py's `before=` clause does.  History is NOT filtered
by game type (preseason/playoffs are absorbed) because props.py does not filter
it either; only the SCORED universe is regular season.

Inference: paired bootstrap, 2000 resamples, 95% percentile CI, CLUSTERED BY
PLAYER (samples are player-games).  Positive delta = Kalman better.

Implementation note: the rate paths are replicated in-script with exact O(1)
incremental recursions instead of re-querying per player-game (the original
ablation issued 2 queries per row and could only afford max_eval=1200).  The
recursions are algebraically identical to the shipped code and are checked
against the real props.py functions on a random sample (--verify) before any
numbers are reported.

Read-only: nbapred.db.connect(read_only=True); imports props.py/form_filter.py
but never mutates them.  Writes nothing.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect

ZONES = ("rim", "mid", "thr")
COL = {"rim": "rima", "mid": "mida", "thr": "thra"}

# --- constants copied verbatim from the shipped code paths ---------------------
HALF_LIFE = 10.0          # player_rates_from_stats(half_life_games=10.0)
K_PRIOR_VAR = 0.05        # player_rates_kalman -> FormFilter(prior_var=0.05,
K_Q = 1e-4                #                                   Q=1e-4,
K_MEAS_BASE = 6.0         #                                   meas_base=6.0)
K_PHI = 0.985             # FormFilter default phi
MIN_SEC = 720             # player_rates_from_stats universe
MIN_HIST = 3              # both paths: `if len(df) < 3: return None`
SEASONS = ("2023-24", "2024-25", "2025-26")

# --- exploratory arm only: Poisson-scaled noise, derived, NOT grid-searched ----
# Counts ~ Poisson(lambda*m) => Var(rate) = lambda/m, so R must scale with the
# rate.  Prior sd = 0.5*lambda at t=0; latent rate drifts ~0.3*lambda over a
# 200-day season => Q = (0.3*lambda)^2 / 200.  No competitor was consulted to
# pick these; the steady-state gain they imply is reported in the output.
LAM_FLOOR = 0.01          # attempts/min floor so a 0-attempt history cannot give R=0
P_PRIOR_FRAC = 0.25       # prior_var = 0.25 * lambda^2  -> prior sd = 0.5*lambda
Q_FRAC = 4.5e-4           # Q/day     = (0.3*lambda)^2 / 200


# ------------------------------------------------------------------ rate paths
class EwmaState:
    """player_rates_from_stats, incrementally.

    per_min(col) = sum(w*col) / sum(w*mins) with w_i = 0.5**((n-1-i)/half_life).
    Absorbing a game is exactly  N <- r*N + c  with r = 0.5**(1/half_life),
    because every existing game ages by exactly one index.
    """

    R = 0.5 ** (1.0 / HALF_LIFE)

    def __init__(self):
        self.n = 0
        self.num = {z: 0.0 for z in ZONES}   # sum(w * counts)
        self.wm = 0.0                        # sum(w * mins)
        self.w = 0.0                         # sum(w)

    def absorb(self, mins: float, counts: dict):
        r = self.R
        for z in ZONES:
            self.num[z] = r * self.num[z] + counts[z]
        self.wm = r * self.wm + mins
        self.w = r * self.w + 1.0
        self.n += 1

    def rates(self) -> dict:
        return {z: self.num[z] / self.wm for z in ZONES}

    def proj_min(self) -> float:
        return self.wm / self.w          # sum(w*mins)/sum(w), as in props.py


class KalmanState:
    """player_rates_kalman's kfilt(), incrementally.

    FormFilter is affine in the observations, so with theta_-1 = m (the
    minutes-weighted mean of the history rates, which is what props.py passes as
    prior_mean) the filtered state can be written

        theta_T = m_T * (1 - S_T) + D_T

    where, with A_i = phi**dt_i and K_i the Kalman gain,
        S_i = (1-K_i)*A_i*S_{i-1} + K_i          (S_-1 = 0)
        D_i = (1-K_i)*A_i*D_{i-1} + K_i*y_i      (D_-1 = 0)

    S and D do not depend on m, so they carry forward across targets even though
    m_T is recomputed from the whole history at every target.  The identity holds
    for ANY K path that is itself independent of m.  m_T = sum(counts)/sum(mins)
    because np.average(rate, weights=mins) with rate = counts/mins collapses to
    exactly that.

    S_T is also the readable "responsiveness" of the filter: theta = m + sum_i
    w_i (y_i - m) with S = sum_i w_i, so S near 0 means the estimate IS the
    career mean and S near 1 means it is fully data-driven.

    mode='ship'    R = meas_base / mins, meas_base = 6.0            [as shipped]
    mode='poisson' R = lambda_hat / mins                            [exploratory]
        Counts are Poisson(lambda*m), so the per-game rate has variance
        lambda/m -- the measurement noise SCALES WITH THE RATE.  A fixed
        meas_base=6.0 is only correct at one rate level.
    """

    def __init__(self, mode: str = "ship"):
        self.mode = mode
        self.n = 0
        self.last = None                     # last game date ordinal
        self.P = {z: (K_PRIOR_VAR if mode == "ship" else None) for z in ZONES}
        self.S = {z: 0.0 for z in ZONES}
        self.D = {z: 0.0 for z in ZONES}
        self.tot = {z: 0.0 for z in ZONES}   # sum counts
        self.tot_min = 0.0                   # sum minutes

    def absorb(self, ordinal: int, mins: float, counts: dict):
        dt = 0.0 if self.last is None else float(ordinal - self.last)
        dt = max(dt, 0.0)
        A = K_PHI ** dt
        for z in ZONES:
            y = counts[z] / mins
            if self.mode == "ship":
                self.P[z] += K_Q * dt
                R = K_MEAS_BASE / max(mins, 1.0)
            else:
                # lambda_hat from PRIOR games only (PIT-clean within the history)
                lam = (self.tot[z] / self.tot_min) if self.tot_min > 0 else y
                lam = max(lam, LAM_FLOOR)
                if self.P[z] is None:
                    self.P[z] = P_PRIOR_FRAC * lam * lam
                self.P[z] += Q_FRAC * lam * lam * dt
                R = lam / max(mins, 1.0)
            K = self.P[z] / (self.P[z] + R)
            self.P[z] *= (1 - K)
            g = (1 - K) * A
            self.S[z] = g * self.S[z] + K
            self.D[z] = g * self.D[z] + K * y
            self.tot[z] += counts[z]
        self.tot_min += mins
        self.last = ordinal
        self.n += 1

    def career(self) -> dict:
        """The filter's own prior centre: minutes-weighted career-to-date mean."""
        return {z: self.tot[z] / self.tot_min for z in ZONES}

    def theta(self) -> dict:
        """props.py's kfilt return: trailing f.predict(dt) is a NO-OP because
        `last` already equals dates[-1] there, so dt is always 0.  Replicated."""
        m = self.career()
        return {z: max(m[z] * (1 - self.S[z]) + self.D[z], 0.0) for z in ZONES}

    def theta_fwd(self, ordinal: int) -> dict:
        """Diagnostic arm: the forward step props.py MEANT to take -- decay the
        state toward the player's mean over the real gap to the target game."""
        dt = max(float(ordinal - self.last), 0.0)
        A = K_PHI ** dt
        m = self.career()
        out = {}
        for z in ZONES:
            th = m[z] * (1 - self.S[z]) + self.D[z]
            out[z] = max(m[z] + A * (th - m[z]), 0.0)
        return out


# ------------------------------------------------------------------- main walk
def build(con):
    """One pass over every player-game; emit one scored row per eligible target."""
    df = con.execute("""
        SELECT s.player_id, g.game_date, g.season, s.game_id, s.seconds,
               s.rima, s.mida, s.thra
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g USING (game_id)
        ORDER BY s.player_id, g.game_date, s.game_id
    """).fetchdf()
    df = df[df["seconds"].notna()]
    df["ordinal"] = df["game_date"].map(lambda d: d.toordinal())

    rows = []   # dicts, one per scored target
    for pid, grp in df.groupby("player_id", sort=False):
        e720, k720, k0 = EwmaState(), KalmanState(), KalmanState()
        kp = KalmanState(mode="poisson")
        secs = grp["seconds"].to_numpy()
        mins_all = secs / 60.0
        ords = grp["ordinal"].to_numpy()
        cnts = {z: grp[COL[z]].to_numpy().astype(float) for z in ZONES}
        seasons = grp["season"].to_numpy()
        gids = grp["game_id"].to_numpy()
        nrow = len(grp)

        i = 0
        while i < nrow:
            j = i
            while j < nrow and ords[j] == ords[i]:   # same-date block: score all
                j += 1                               # BEFORE absorbing any of it
            for t in range(i, j):
                if secs[t] < MIN_SEC or e720.n < MIN_HIST:
                    continue
                if not (str(gids[t]).startswith("002") and seasons[t] in SEASONS):
                    continue
                m_t = mins_all[t]
                pred = {"ewma720": e720.rates(), "kal720": k720.theta(),
                        "kal0": k0.theta(), "kal720fwd": k720.theta_fwd(int(ords[t])),
                        "career720": k720.career(), "kal720p": kp.theta()}
                rec = {"player_id": pid, "season": seasons[t], "minutes": m_t,
                       "n_games": e720.n, "proj_min": e720.proj_min(),
                       "n_hist_0": k0.n,
                       "S_ship": float(np.mean([k720.S[z] for z in ZONES])),
                       "S_pois": float(np.mean([kp.S[z] for z in ZONES]))}
                for z in ZONES:
                    y = cnts[z][t] / m_t
                    rec[f"y_{z}"] = y
                    for arm in ARMS:
                        rec[f"{arm}_{z}"] = abs(pred[arm][z] - y)
                rows.append(rec)
            for t in range(i, j):                    # now absorb the date's games
                c = {z: cnts[z][t] for z in ZONES}
                if secs[t] >= MIN_SEC:
                    e720.absorb(mins_all[t], c)
                    k720.absorb(int(ords[t]), mins_all[t], c)
                    kp.absorb(int(ords[t]), mins_all[t], c)
                if secs[t] > 0:
                    k0.absorb(int(ords[t]), mins_all[t], c)
            i = j
    return rows


# ------------------------------------------------------------------- inference
ARMS = ("ewma720", "kal720", "kal0", "kal720fwd", "career720", "kal720p")


def cluster_boot(pid, w, num_a, num_b, n_boot=2000, seed=0):
    """Paired bootstrap of WMAE(a) - WMAE(b), clustering by player.

    The statistic is a ratio of sums, so resampling whole player clusters only
    needs each player's summed numerator/weight -- exact, not an approximation.
    """
    uid, inv = np.unique(pid, return_inverse=True)
    P = len(uid)
    W = np.bincount(inv, weights=w, minlength=P)
    A = np.bincount(inv, weights=num_a, minlength=P)
    B = np.bincount(inv, weights=num_b, minlength=P)
    point = (A.sum() - B.sum()) / W.sum()
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.integers(0, P, P)
        out[b] = (A[s].sum() - B[s].sum()) / W[s].sum()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return point, lo, hi, P


def report(rows, label, gate=None):
    if gate is not None:
        rows_l = [r for r in rows if gate(r)]
    else:
        rows_l = rows
    if not rows_l:
        print(f"\n[{label}] no rows"); return None
    pid = np.array([r["player_id"] for r in rows_l])
    w = np.array([r["minutes"] for r in rows_l])
    err = {}
    for arm in ARMS:
        for z in ZONES:
            err[(arm, z)] = np.array([r[f"{arm}_{z}"] for r in rows_l]) * w
    ybar = {z: np.average([r[f"y_{z}"] for r in rows_l], weights=w) for z in ZONES}

    print(f"\n{'='*78}\n[{label}]  player-games={len(rows_l)}  players={len(set(pid))}  "
          f"minutes={w.sum():,.0f}")
    print(f"  mean realized rate/min: " +
          "  ".join(f"{z}={ybar[z]:.4f}" for z in ZONES) +
          f"   (total {sum(ybar.values()):.4f})")
    print(f"  filter responsiveness S = total weight on OBSERVED games "
          f"(1-S goes to the career mean):")
    print(f"      kal720 (shipped consts)={np.mean([r['S_ship'] for r in rows_l]):.3f}   "
          f"kal720p (poisson-scaled)={np.mean([r['S_pois'] for r in rows_l]):.3f}   "
          f"ewma720=1.000 by construction (no career-mean anchor)")
    hdr = f"  {'arm':<11}" + "".join(f"{z:>10}" for z in ZONES) + f"{'all3':>10}"
    print(hdr)
    tot = {}
    for arm in ARMS:
        vals = [err[(arm, z)].sum() / w.sum() for z in ZONES]
        tot[arm] = sum(vals)
        print(f"  {arm:<11}" + "".join(f"{v:>10.5f}" for v in vals) + f"{tot[arm]:>10.5f}")

    comps = [("PRIMARY  aligned estimator  ewma720 - kal720", "ewma720", "kal720", True),
             ("         universe effect    kal0    - kal720", "kal0", "kal720", False),
             ("         ORIG confounded    ewma720 - kal0   ", "ewma720", "kal0", False),
             ("DIAG     fwd-step           kal720  - kal720fwd", "kal720", "kal720fwd", False),
             ("DIAG     filter vs anchor   career720 - kal720", "career720", "kal720", False),
             ("DIAG     metric has signal  career720 - ewma720", "career720", "ewma720", False),
             ("EXPLOR   poisson-scaled K   ewma720 - kal720p", "ewma720", "kal720p", True)]
    res = {}
    print(f"  {'-'*74}")
    print("  paired bootstrap 2000x, clustered by player; + = SECOND arm better")
    for name, a, b, detail in comps:
        num_a = sum(err[(a, z)] for z in ZONES)
        num_b = sum(err[(b, z)] for z in ZONES)
        pt, lo, hi, P = cluster_boot(pid, w, num_a, num_b)
        sig = "SIG" if (lo > 0 or hi < 0) else "ns "
        rel = 100.0 * pt / tot[a] if tot[a] else 0.0
        print(f"  {name}: {pt:+.5f} ({rel:+.2f}%)  95% CI [{lo:+.5f},{hi:+.5f}] {sig}")
        res[f"{a}-{b}"] = (pt, lo, hi, rel, sig)
        if detail:
            for z in ZONES:                               # per-zone detail
                ptz, loz, hiz, _ = cluster_boot(pid, w, err[(a, z)], err[(b, z)])
                print(f"        {z}: {ptz:+.5f}  CI [{loz:+.5f},{hiz:+.5f}]"
                      f" {'SIG' if (loz>0 or hiz<0) else 'ns'}")
    return res


# ---------------------------------------------------------------- verification
def verify(con, rows, n_check=40, seed=7):
    """Fidelity gate: the in-script recursions must reproduce the SHIPPED
    functions on random targets, or nothing below is trustworthy."""
    from nbapred.engine.props import player_rates_from_stats, player_rates_kalman

    rng = np.random.default_rng(seed)
    dates = con.execute("""
        SELECT s.player_id, g.game_date FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g USING (game_id)
        WHERE g.season='2025-26' AND s.game_id LIKE '002%' AND s.seconds>=720
    """).fetchdf()
    # re-walk to grab the predictions for the sampled (player,date) pairs
    idx = rng.choice(len(dates), size=min(n_check, len(dates)), replace=False)
    worst_e = worst_k = worst_0 = 0.0
    checked = 0
    # Replay each sampled target's history through the in-script recursions and
    # compare against the shipped functions called with before=<target date>.
    allpg = con.execute("""
        SELECT s.player_id, g.game_date, s.seconds, s.rima, s.mida, s.thra
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
        ORDER BY s.player_id, g.game_date, s.game_id
    """).fetchdf()
    allpg = allpg[allpg["seconds"].notna()]
    by_p = {p: g for p, g in allpg.groupby("player_id", sort=False)}

    for k in idx:
        pid = int(dates.player_id.iloc[k]); gd = dates.game_date.iloc[k]
        g = by_p[pid]
        hist = g[g["game_date"] < gd]
        e, k7, k0 = EwmaState(), KalmanState(), KalmanState()
        for row in hist.itertuples():
            c = {"rim": float(row.rima), "mid": float(row.mida), "thr": float(row.thra)}
            m = row.seconds / 60.0
            if row.seconds >= MIN_SEC:
                e.absorb(m, c); k7.absorb(row.game_date.toordinal(), m, c)
            if row.seconds > 0:
                k0.absorb(row.game_date.toordinal(), m, c)
        if e.n < MIN_HIST:
            continue
        ship_e = player_rates_from_stats(con, pid, before=gd)
        ship_k = player_rates_kalman(con, pid, before=gd)
        mine_e, mine_k7, mine_k0 = e.rates(), k7.theta(), k0.theta()
        for z, key in (("rim", "rate_rim"), ("mid", "rate_mid"), ("thr", "rate_thr")):
            worst_e = max(worst_e, abs(ship_e[key] - mine_e[z]))
            worst_0 = max(worst_0, abs(ship_k[key] - mine_k0[z]))
        # kal720: shipped function with the universe swapped -- recompute the
        # shipped kfilt directly on the >=720 history to check the recursion.
        from nbapred.model.form_filter import FormFilter
        h = hist[hist["seconds"] >= MIN_SEC]
        mm = h["seconds"].to_numpy() / 60.0
        dd = np.array([d.toordinal() for d in h["game_date"]])
        for z in ZONES:
            cc = h[COL[z]].to_numpy() / mm
            f = FormFilter(float(np.average(cc, weights=mm)), prior_var=K_PRIOR_VAR,
                           Q=K_Q, meas_base=K_MEAS_BASE)
            last = None
            for i in range(len(h)):
                f.predict(0.0 if last is None else dd[i] - last)
                f.update(cc[i], mm[i]); last = dd[i]
            f.predict(0.0 if last is None else max(0, dd[-1] - last))
            worst_k = max(worst_k, abs(max(f.theta, 0.0) - mine_k7[z]))
        checked += 1
    print(f"\n[verify] {checked} random 2025-26 targets re-checked against the "
          f"SHIPPED functions")
    print(f"[verify] max |in-script - shipped| : ewma720={worst_e:.3e}  "
          f"kal0={worst_0:.3e}  kal720={worst_k:.3e}")
    ok = max(worst_e, worst_k, worst_0) < 1e-9
    print(f"[verify] {'PASS' if ok else 'FAIL'} (tol 1e-9)")
    return ok


def main():
    con = connect(read_only=True)
    print("building rate paths (one pass, PIT: history strictly < game_date) ...")
    rows = build(con)
    ok = verify(con, rows)
    if not ok:
        print("\n!! replication does not match shipped code -- ABORTING")
        con.close(); return
    con.close()

    print(f"\nscored targets (regular season, seconds>=720, >=3 prior >=720s games): "
          f"{len(rows)}")
    report([r for r in rows if r["season"] == "2025-26"], "2025-26  (PRIMARY)")
    for s in ("2023-24", "2024-25"):
        report([r for r in rows if r["season"] == s], f"{s}  (secondary)")
    report(rows, "POOLED 2023-26")
    report([r for r in rows if r["season"] == "2025-26"
            and r["n_games"] >= 8 and r["proj_min"] >= 20],
           "2025-26 under the ORIGINAL ablation gate (n_games>=8, proj_min>=20)")


if __name__ == "__main__":
    main()
