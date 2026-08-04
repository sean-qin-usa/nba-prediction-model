"""W49 TASK 2 — THE PRE-REGISTERED GATE: uncertainty-gated variance inflation.

DECLARED BEFORE THE RUN (one config, no sweeps, no threshold search):

MOTIVATION.  D74 closed the GLOBAL calibration vein (uniform underconfidence
fix does not generalise walk-forward).  A market-referenced cap is FORBIDDEN
(G2 market-blind).  The only remaining shape is a LOCAL variance inflation
driven by OUR OWN uncertainty, firing on a minority of games and exactly
zero elsewhere.

SIGNAL (all market-blind, all PIT at tip).  The three axes the w49_sep.py
overconfidence screen put above zero, in the direction of overconfidence
(hindsight-optimal scale multiplier k > 1):
    x1 = |fm - cm|                                  leg disagreement — our two
         independent legs (FourFactors, Composition) telling different stories
         is epistemic uncertainty the fixed SCALE 7.2 does not price.
         (screen: k=1.093 at the p75 gate)
    x2 = |out_talent_home| + |out_talent_away|       availability talent churn
         (screen: k=1.165, the largest ceiling of any PIT signal)
    x3 = 1{min(gp_home, gp_away) < 5}                week-1 dead-zone (D84-A)
         (screen: k=1.309, the largest multiplier)

GATE SHAPE (fires only in the upper quartile; identically zero elsewhere, so
this cannot smuggle in a global recalibration):
    z_k = max(0, (x_k - q75_d(x_k)) / (q75_d(x_k) - q25_d(x_k)))   k in {1,2}
    z_3 = x_3
    scale_i = SCALE * (1 + sum_k lambda_dk * z_ik),   lambda >= 0
    p_i     = sigmoid(m_tot_i / scale_i)
q75_d / q25_d are computed on COMPLETED GAMES STRICTLY BEFORE d only.

FIT.  lambda_d by walk-forward MLE on every completed game strictly before d,
sign-constrained lambda >= 0 (uncertainty may only INFLATE variance — the same
sign convention as ds_rt4_blend's w in [0,1]), shrunk toward 0 by n/(n+600)
(the repo's standard SCHED_SHRINK / FORM_SHRINK / K_SHRINK / BLEND_SHRINK),
lambda = 0 until 200 fittable games (BURN_IN, as ds_rt4_blend).  Weekly Monday
refits, matching the production refit cadence.

PRIMARY = the shrunk arm.  The unshrunk MLE arm is a diagnostic only.

CONTROL = the same-run shipped p_full (unchanged), so the pairing is exact.
GATE CRITERION = pooled paired bootstrap 2000x seed 20260801; ship only if the
pooled 95% CI excludes 0.
PRE-DECLARED SUBSETS reported regardless of the verdict: fired games only,
per-season, top-decile conf_us, early (gp<20), late (gp>=55).

HONESTY: the three axes were chosen by a screen run on THESE FOUR SEASONS, so
this is a third look at spent data.  The screen already bounded the hindsight
CEILING at +0.00043/game with a CI spanning zero, i.e. it predicts NS before
the walk-forward is run; this gate is the confirmation, not the discovery.

Read-only inputs.  Output: data/w49_gate.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from ds_corpus import paired_bootstrap  # noqa: E402

FRAME = ROOT / "data" / "w49_frame.csv"
OUT = ROOT / "data" / "w49_gate.json"
SCALE = 7.2
SHRINK = 600.0
BURN_IN = 200
SEED = 20260801
N_BOOT = 2000
EPS = 1e-12


def nll(lam, z, m, y):
    s = SCALE * (1.0 + z @ np.asarray(lam))
    p = 1.0 / (1.0 + np.exp(-m / s))
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def zmat(x1, x2, x3, q):
    """Gate transform with quantiles q = {(q25,q75) per continuous axis}."""
    z = np.zeros((len(x1), 3))
    for j, x in enumerate((x1, x2)):
        lo, hi = q[j]
        iqr = hi - lo
        if iqr > 0:
            z[:, j] = np.maximum(0.0, (x - hi) / iqr)
    z[:, 2] = x3
    return z


def main() -> None:
    d = pd.read_csv(FRAME, dtype={"game_id": str})
    d["game_date"] = pd.to_datetime(d.game_date)
    d = d.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    d["x1"] = (d.fm - d.cm).abs()
    d["x2"] = d.out_talent_home.abs() + d.out_talent_away.abs()
    d["x3"] = (d.gp_min < 5).astype(float)

    m = d.m_tot.values
    y = d.y.values.astype(float)
    x1, x2, x3 = d.x1.values, d.x2.values, d.x3.values
    dates = d.game_date.values

    # weekly Monday refit dates covering the corpus
    days = pd.to_datetime(sorted(d.game_date.unique()))
    refits = sorted({dt for dt in days if dt.weekday() == 0} | {days[0]})

    lam_shrunk = np.zeros((len(d), 3))
    lam_raw = np.zeros((len(d), 3))
    z_all = np.zeros((len(d), 3))
    trace = []
    for i, rd in enumerate(refits):
        nxt = refits[i + 1] if i + 1 < len(refits) else \
            pd.Timestamp(days[-1]) + pd.Timedelta(days=1)
        blk = (dates >= np.datetime64(rd)) & (dates < np.datetime64(nxt))
        if not blk.any():
            continue
        past = dates < np.datetime64(rd)
        n = int(past.sum())
        if n < BURN_IN:
            continue
        q = [(np.percentile(x1[past], 25), np.percentile(x1[past], 75)),
             (np.percentile(x2[past], 25), np.percentile(x2[past], 75))]
        zp = zmat(x1[past], x2[past], x3[past], q)
        r = minimize(nll, x0=np.zeros(3), args=(zp, m[past], y[past]),
                     method="L-BFGS-B", bounds=[(0.0, 5.0)] * 3)
        lam = np.maximum(r.x, 0.0)
        sh = n / (n + SHRINK)
        z_all[blk] = zmat(x1[blk], x2[blk], x3[blk], q)
        lam_raw[blk] = lam
        lam_shrunk[blk] = lam * sh
        trace.append({"date": str(pd.Timestamp(rd).date()), "n_fit": n,
                      "lam_raw": lam.tolist(),
                      "lam_shrunk": (lam * sh).tolist(), "shrink": float(sh)})

    def probs(L):
        s = SCALE * (1.0 + (z_all * L).sum(axis=1))
        return 1.0 / (1.0 + np.exp(-m / s))

    p_ctl = d.p_full.values
    p_var = probs(lam_shrunk)
    p_unshrunk = probs(lam_raw)

    def ll(p):
        p = np.clip(p, EPS, 1 - EPS)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    l_ctl, l_var, l_uns = ll(p_ctl), ll(p_var), ll(p_unshrunk)
    delta = l_ctl - l_var                       # >0 = variant better
    d["delta"] = delta
    d["fired"] = (z_all.sum(axis=1) > 0)
    d["l_ctl"], d["l_var"] = l_ctl, l_var

    # bitwise control check: where nothing fires the variant must equal control
    off = ~d.fired.values
    maxdiff = float(np.abs(p_var[off] - p_ctl[off]).max()) if off.any() else 0.0
    print(f"ZERO-OUTSIDE-WINDOW CHECK: max|p_var - p_ctl| on non-firing games "
          f"= {maxdiff:.3e}   (n_off={int(off.sum())})")

    res = {"n": len(d), "n_fired": int(d.fired.sum()),
           "fired_share": float(d.fired.mean()),
           "zero_outside_window_maxdiff": maxdiff,
           "lambda_trace": trace}

    scored = d[lam_shrunk.sum(axis=1) >= 0].copy()   # all rows are scored
    print(f"\nfired on {int(d.fired.sum())}/{len(d)} games "
          f"({100*d.fired.mean():.1f}%)")
    if trace:
        lt = np.array([t["lam_shrunk"] for t in trace])
        print(f"walk-forward lambda (shrunk)  mean {lt.mean(axis=0).round(4)}  "
              f"last {np.array(trace[-1]['lam_shrunk']).round(4)}  "
              f"max {lt.max(axis=0).round(4)}")
        ltr = np.array([t["lam_raw"] for t in trace])
        print(f"walk-forward lambda (raw MLE) mean {ltr.mean(axis=0).round(4)}"
              f"  max {ltr.max(axis=0).round(4)}")
        res["lambda_mean_shrunk"] = lt.mean(axis=0).tolist()
        res["lambda_mean_raw"] = ltr.mean(axis=0).tolist()

    print(f"\n{'='*92}")
    print("GATE — variance inflation gated on our own uncertainty "
          "(PRIMARY = shrunk arm)")
    print(f"{'='*92}")
    print(f"{'slice':34s} {'n':>6} {'delta/game':>12} "
          f"{'95% CI':>26} {'verdict':>8}")

    def row(label, mask):
        b = paired_bootstrap(delta[mask], n_boot=N_BOOT, seed=SEED)
        print(f"{label:34s} {b['n']:>6} {b['mean']:>+12.5f} "
              f"[{b['lo']:+.5f},{b['hi']:+.5f}]".rjust(0) +
              f" {b['verdict']:>8}")
        return {"slice": label, **b}

    rows = [row("POOLED (primary)", np.ones(len(d), bool))]
    rows.append(row("  fired games only", d.fired.values))
    for ssn in sorted(d.season.unique()):
        rows.append(row(f"  season {ssn}", (d.season == ssn).values))
    thr = d.conf_us.quantile(0.90)
    rows.append(row("  top-decile conf_us", (d.conf_us >= thr).values))
    rows.append(row("  early (gp_min<20)", (d.early == 1).values))
    rows.append(row("  late (gp_min>=55)", (d.late == 1).values))
    res["gate"] = rows

    b_uns = paired_bootstrap(l_ctl - l_uns, n_boot=N_BOOT, seed=SEED)
    print(f"\nDIAGNOSTIC unshrunk-MLE arm: {b_uns['mean']:+.5f} "
          f"[{b_uns['lo']:+.5f},{b_uns['hi']:+.5f}] {b_uns['verdict']}")
    res["unshrunk_arm"] = b_uns

    # what the gate did to the catastrophic tail it was built for
    srt = d.sort_values("l_ctl" if False else "delta")
    tail = d.nlargest(49, "exc") if "exc" in d.columns else None
    if tail is not None:
        print(f"\nEFFECT ON THE WORST 49 (the games this was built for): "
              f"delta {tail.delta.mean():+.5f}/game, total "
              f"{tail.delta.sum():+.3f} nats of the 30.909 they cost; "
              f"fired on {int(tail.fired.sum())}/49")
        res["worst49_effect"] = {"delta_per_game": float(tail.delta.mean()),
                                 "total_nats": float(tail.delta.sum()),
                                 "n_fired": int(tail.fired.sum())}
    print(f"\nheadline ll  control {l_ctl.mean():.5f}  variant "
          f"{l_var.mean():.5f}  market {d.l_mkt.mean():.5f}")
    res["ll"] = {"control": float(l_ctl.mean()), "variant": float(l_var.mean()),
                 "market": float(d.l_mkt.mean())}
    _ = srt
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
