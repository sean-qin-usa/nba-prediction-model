"""OF-1c — EFFICACY PER POINT OF APPLIED MARGIN: separates "the term does not
transfer" from "the term's walk-forward estimator was COLD on the held-out
season".

D73 (tanking.py) and D90 (latestate.py) fit their coefficients on active games
`season >= '2022-23'` — a corpus-floor LITERAL (hall-of-shame #8). Consequence,
measured in scripts/of_transfer_ablation.py:
    2021-22   term identically 0.0        -> UNTESTABLE
    2022-23   term runs at a fraction of production strength (burn-in)
    2023-26   term at full strength
So a raw per-game effect comparison confounds "no signal on held-out" with
"almost no term applied on held-out".

FIX: back the applied term out of the arms exactly
    t = SCALE * (logit(p_base) - logit(p_ablated))
and report, on ACTIVE games only,
    efficacy = sum(per-game log-loss gain) / sum(|t|)   [nats per margin point]
which is invariant to how big the coefficient happened to be. Bootstrapped as a
ratio (numerator and denominator resampled together).

Usage: python scripts/of_term_efficacy.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from of_transfer_ablation import SEED, TERMS  # noqa: E402
from of_transfer_table import DEV, GATE, HELD, HELD1  # noqa: E402

SCALE = 7.2


def ll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def ratio_boot(num, den, n_boot=2000, seed=SEED):
    """Bootstrap the ratio sum(num)/sum(den) (paired resample of games)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if len(num) == 0 or den.sum() == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    r = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return {"n_active": int(len(num)), "efficacy": float(num.sum() / den.sum()),
            "lo": float(np.percentile(r, 2.5)), "hi": float(np.percentile(r, 97.5)),
            "mean_abs_term_pts": float(np.abs(den).mean()),
            "effect_per_active_game": float(num.mean())}


def main():
    rows = list(csv.DictReader(open(ROOT / "data" / "of_transfer_pergame.csv")))
    for r in rows:
        for k in r:
            if k.startswith(("p_", "act_", "gp_")) or k == "y":
                r[k] = float(r[k])

    groups = [("DEV 24-26", DEV), ("HELD-OUT 22-23", HELD1),
              ("HELD-OUT 21-23", HELD), ("gate 23-24", GATE)]
    out = {}
    print("efficacy = nats of log-loss gained per POINT of margin the term "
          "actually applied,\nmeasured on ACTIVE games only "
          "(invariant to a cold walk-forward coefficient)\n")
    for arm, name in TERMS.items():
        print(f"{name}")
        out[arm] = {"term": name, "groups": {}}
        for gname, ss in groups:
            rs = [r for r in rows if r["season"] in ss]
            y = np.array([r["y"] for r in rs])
            pb = np.array([r["p_base"] for r in rs])
            pa = np.array([r[f"p_{arm}"] for r in rs])
            t = SCALE * (logit(pb) - logit(pa))
            gain = ll(y, pa) - ll(y, pb)          # positive = term helps
            act = np.abs(t) > 1e-12
            res = ratio_boot(gain[act], np.abs(t[act]))
            out[arm]["groups"][gname] = res
            if res is None:
                print(f"  {gname:16s} term never applied — UNTESTABLE")
                continue
            print(f"  {gname:16s} n_act={res['n_active']:5d} "
                  f"mean|term|={res['mean_abs_term_pts']:.3f}pts  "
                  f"gain/active-game={res['effect_per_active_game']:+.5f}  "
                  f"EFFICACY={res['efficacy']:+.5f} "
                  f"({res['lo']:+.5f},{res['hi']:+.5f})")
        # transfer ratio on the efficacy scale
        d = out[arm]["groups"]["DEV 24-26"]
        h = out[arm]["groups"]["HELD-OUT 22-23"]
        if d and h and d["efficacy"] != 0:
            out[arm]["efficacy_transfer_ratio_2223_over_dev"] = h["efficacy"] / d["efficacy"]
            print(f"  -> held-out efficacy is "
                  f"{100*h['efficacy']/d['efficacy']:.0f}% of dev efficacy")
        print()

    json.dump(out, open(ROOT / "data" / "of_term_efficacy.json", "w"), indent=1)
    print("wrote data/of_term_efficacy.json")


if __name__ == "__main__":
    main()
