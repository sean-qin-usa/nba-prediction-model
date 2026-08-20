#!/usr/bin/env python3
"""D232 — GATE: absence-response correction. Prereg sha256 fe77ff1e...

    challenger   m' = m_total + beta * out_diff
    control arm  m' = m_total + beta * out_diff + gamma * m_total

beta = 0 is the shipped model, so the null is the incumbent. Walk-forward:
beta fitted on seasons 1..k, scored on k+1 alone. Each arm converts margin to
probability with its OWN scale fitted on the training block, so neither is
handicapped by the other's calibration (D193's rule).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from scipy import stats                                           # noqa: E402
from scipy.optimize import minimize_scalar                        # noqa: E402


def nll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_scale(m, y):
    return float(minimize_scalar(
        lambda s: nll(1 / (1 + np.exp(-m / s)), y).mean(),
        bounds=(2, 25), method="bounded").x)


def zfill(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def load():
    cap = pd.read_csv(ROOT / "data" / "capstone_2019_26.csv")
    chan = pd.read_csv(ROOT / "data" / "channel_pergame.csv")
    pit = pd.read_csv(ROOT / "data" / "pit_frame.csv.gz")
    for d in (cap, chan, pit):
        d["game_id"] = zfill(d["game_id"])
    f = (chan.merge(cap[["game_id", "n_out_home", "n_out_away",
                         "eo_home", "eo_away"]], on="game_id",
                    validate="one_to_one")
             .merge(pit[["game_id", "margin_actual"]], on="game_id",
                    validate="one_to_one"))
    assert len(f) > 0.9 * len(chan)
    f = f.dropna(subset=["margin_actual", "m_total"]).copy()
    # PRIMARY is the PRE-REGISTERED quantity: EXPECTED absences, sum of P(out).
    # `head_diff` is the headcount of players carrying any doubt -- a different
    # feature (corr 0.84, mean 1.70 vs 0.96) that an earlier run used by mistake
    # because prod_by_season's n_out column is len(dict), not sum(dict). Both
    # are reported; the prereg governs which one is the arm.
    f["out_diff"] = f["eo_home"] - f["eo_away"]
    f["head_diff"] = f["n_out_home"] - f["n_out_away"]
    f["y"] = (f["margin_actual"] > 0).astype(float)
    return f.sort_values(["season", "game_date", "game_id"]).reset_index(drop=True)


FEATURE = "out_diff"


def run(f, seasons, control: bool, permute_seed=None):
    """Walk-forward. Returns per-season deltas vs the incumbent."""
    rng = np.random.default_rng(permute_seed) if permute_seed is not None else None
    g = f.copy()
    if rng is not None:                       # within-season permutation null
        for s in seasons:
            m = (g["season"] == s).to_numpy()
            g.loc[m, FEATURE] = rng.permutation(g.loc[m, FEATURE].to_numpy())
    rows = []
    for i, s in enumerate(seasons):
        if i == 0:
            continue
        tr = g[g["season"].isin(seasons[:i])]
        te = g[g["season"] == s]
        cols = [FEATURE] + (["m_total"] if control else [])
        Xtr = np.column_stack([tr[c].to_numpy(float) for c in cols])
        Xte = np.column_stack([te[c].to_numpy(float) for c in cols])
        # target is the incumbent's own residual, so beta=0 IS the incumbent
        rtr = (tr["margin_actual"] - tr["m_total"]).to_numpy(float)
        beta = np.linalg.lstsq(Xtr, rtr, rcond=None)[0]
        m_tr = tr["m_total"].to_numpy(float) + Xtr @ beta
        m_te = te["m_total"].to_numpy(float) + Xte @ beta
        ytr, yte = tr["y"].to_numpy(float), te["y"].to_numpy(float)
        s_inc = fit_scale(tr["m_total"].to_numpy(float), ytr)
        s_chal = fit_scale(m_tr, ytr)
        p_inc = 1 / (1 + np.exp(-te["m_total"].to_numpy(float) / s_inc))
        p_chal = 1 / (1 + np.exp(-m_te / s_chal))
        rows.append(dict(season=s, n=len(te), beta=float(beta[0]),
                         gamma=float(beta[1]) if control else None,
                         ll_inc=float(nll(p_inc, yte).mean()),
                         ll_chal=float(nll(p_chal, yte).mean()),
                         p_inc=float(p_inc.mean()), p_chal=float(p_chal.mean()),
                         base=float(yte.mean())))
        rows[-1]["delta"] = rows[-1]["ll_chal"] - rows[-1]["ll_inc"]
    return rows


def clustered(d):
    d = np.asarray(d, float)
    k = len(d)
    se = d.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return d.mean(), d.mean() - tc * se, d.mean() + tc * se, d.mean() / se, k


def main():
    f = load()
    seasons = sorted(f["season"].unique())
    print(f"frame {len(f)} games, {len(seasons)} seasons")

    # ---- MDE80 from a permutation null, BEFORE the endpoint --------------
    nulls = []
    for sd in range(10):
        nulls += [r["delta"] for r in run(f, seasons, False, permute_seed=sd)]
    k = len(seasons) - 1
    mde80 = ((stats.t.ppf(0.975, k - 1) + stats.t.ppf(0.80, k - 1))
             * np.std(nulls, ddof=1) / np.sqrt(k))
    print(f"MDE80 (stated first, permutation null): {mde80:.5f} nats")

    global FEATURE
    for FEATURE in ("out_diff", "head_diff"):
      print(f"\n{'='*66}\nFEATURE = {FEATURE} "
            f"({'PRE-REGISTERED: expected absences' if FEATURE=='out_diff' else 'headcount under doubt (secondary)'})")
      for control in (False, True):
          tag = "WITH STRENGTH CONTROL" if control else "MAIN ARM"
          rows = run(f, seasons, control)
          r = pd.DataFrame(rows)
          print(f"\n=== {tag} ===")
          show = ["season", "n", "beta"] + (["gamma"] if control else []) + \
                 ["ll_inc", "ll_chal", "delta"]
          print(r[show].to_string(index=False,
                                  float_format=lambda v: f"{v:9.5f}"))
          mean, lo, hi, t, kk = clustered(r["delta"])
          print(f"  season-clustered mean delta {mean:+.6f}")
          print(f"  95% CI ({kk-1} dof)          [{lo:+.6f}, {hi:+.6f}]")
          print(f"  t {t:+.2f}   better in {int((r['delta']<0).sum())}/{kk}")
          print(f"  mean beta {r['beta'].mean():+.4f}")
          # calibration veto
          di = (r["p_inc"] - r["base"]).abs().mean()
          dc = (r["p_chal"] - r["base"]).abs().mean()
          print(f"  calibration drift: incumbent {di:.4f} challenger {dc:.4f} "
                f"-> {'PASS' if dc <= di + 1e-6 else 'VETO'}")
          verdict = "SHIP" if (hi < 0 and dc <= di + 1e-6) else "NO SHIP"
          print(f"  VERDICT: {verdict}")
          json.dump({"rows": rows, "mean": mean, "ci": [lo, hi], "t": t,
                     "mde80": mde80, "verdict": verdict},
                    open(ROOT / "data" /
                         f"d232_gate_{FEATURE}_{'control' if control else 'main'}.json", "w"),
                    default=float)


if __name__ == "__main__":
    main()
