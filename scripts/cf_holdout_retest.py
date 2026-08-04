"""CF-1 — THE DECIDING RE-TEST (D112 JOB 2): does D73 / D90 survive on the
genuine holdout once the corpus-floor literal is gone and the coefficients
are WARM?

BACKGROUND. The D110 audit could not decide D73 or D90 because tanking.py and
latestate.py carried a hardcoded `season >= '2022-23'` corpus floor. On
2021-22 both terms were identically 0.0 (UNTESTABLE); on 2022-23 they ran
COLD (tank k -0.23..-1.32 vs -1.6..-2.3 later; late-state mean |term| 0.151
pts vs 1.377). D110's raw holdout nulls therefore partly measured a cold
estimator, not an absent signal. D112 derives the floor from the data
(tanking.season_floor -> 2021-22 on the current corpus) and this script
re-runs the SAME per-term ablation with warm coefficients.

DESIGN. `season_run` / `Pieces` / `paired_bootstrap` / SEASONS / GROUPS /
TERMS are imported VERBATIM from scripts/of_transfer_ablation.py — the
construction, the refit cadence, the OUT-set convention, the market join, the
bootstrap seed and the base-arm anchoring to the literal fit_production are
byte-identical to the D110 run. The ONLY thing that differs between the two
arms is the value of TANK_SEASON_FLOOR:

    ARM_OLD  TANK_SEASON_FLOOR=2022-23  = the pre-D112 shipped state
                                          (must reproduce D110's numbers)
    ARM_NEW  TANK_SEASON_FLOOR=2021-22  = the derived floor, warm coefficients

Run as two separate processes (CF_ARM=old|new) so each gets its own module
cache; the DB is opened read-only, so they may run concurrently.

--------------------------------------------------------------------------
PRE-REGISTERED DECISION RULE (written 2026-08-01, BEFORE any arm was run)
--------------------------------------------------------------------------
EFFECT   = per-game log-loss delta, ll(term OFF) - ll(term ON).
           POSITIVE = the term HELPS. Paired bootstrap 2000x, seed 20260801.
PRIMARY ENDPOINT for each of D73 and D90:
           the POOLED HELDOUT_21_23 effect (2021-22 + 2022-23), computed on
           ARM_NEW (warm coefficients).
VERDICT:
  CONFIRMED  iff the primary effect is POSITIVE and its 95% CI EXCLUDES 0.
  DEMOTED    otherwise (i.e. the effect is ~0, or negative, or the CI covers
             0). DEMOTED means: the term is reverted from production — its
             kill switch default flips to OFF — and the register is marked
             accordingly.
SECONDARY, REPORTED BUT NOT ALLOWED TO OVERTURN THE PRIMARY: HELDOUT_2223_only
  (the fully-warm holdout season; 2021-22 is now itself the burn-in season and
  is still partly cold), per-season effects, active-only restrictions, and the
  per-point efficacy ratio. A secondary result may NOT rescue a term that
  fails the primary, and may NOT convict a term that passes it.
NO OTHER ENDPOINT, SUBGROUP OR WINDOW WILL BE ADDED AFTER SEEING THE NUMBERS.
--------------------------------------------------------------------------

Read-only DB. Usage:
    CF_ARM=new python scripts/cf_holdout_retest.py
    CF_ARM=old python scripts/cf_holdout_retest.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

ARM = os.environ.get("CF_ARM", "new")
FLOORS = {"old": "2022-23", "new": "2021-22"}
os.environ["TANK_SEASON_FLOOR"] = FLOORS[ARM]

import numpy as np  # noqa: E402

from nbapred.db import connect  # noqa: E402
from nbapred.eval.metrics import log_loss  # noqa: E402
from of_transfer_ablation import (GROUPS, SEASONS, TERMS,  # noqa: E402
                                  paired_bootstrap, season_run)

LN2 = 0.6931471805599453


def ll_pg(r, key):
    p = min(max(r[key], 1e-12), 1 - 1e-12)
    return -(r["y"] * np.log(p) + (1 - r["y"]) * np.log(1 - p))


def main():
    anchor = os.environ.get("OF_ANCHOR", "1") != "0"
    con = connect(read_only=True)
    from nbapred.model.tanking import season_floor
    floor = season_floor(con)
    print(f"ARM={ARM}  TANK_SEASON_FLOOR={floor}", flush=True)

    allrows, anchors = [], {}
    for s in SEASONS:
        t0 = time.time()
        rr, ad = season_run(con, s, anchor)
        allrows += rr
        anchors[s] = ad
        print(f"{s}: n={len(rr)} anchor_max|dp|={ad:.2e} "
              f"({time.time()-t0:.0f}s)", flush=True)
    con.close()

    tag = f"cf_holdout_{ARM}"
    hdr = list(allrows[0].keys())
    with open(ROOT / "data" / f"{tag}_pergame.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerows(allrows)

    res = {"arm": ARM, "floor": floor, "anchor_max_abs_dp": anchors,
           "seed": 20260801, "seasons": {}, "activation": {}, "terms": {}}

    for s in SEASONS:
        rs = [r for r in allrows if r["season"] == s]
        y = np.array([r["y"] for r in rs])
        ll_us = float(log_loss(y, np.array([r["p_base"] for r in rs])))
        ll_mk = float(log_loss(y, np.array([r["p_mkt"] for r in rs])))
        res["seasons"][s] = {
            "n": len(rs), "ll_base": round(ll_us, 5), "ll_mkt": round(ll_mk, 5),
            "gap": round(ll_us - ll_mk, 5),
            "mkt_skill": round(LN2 - ll_mk, 5),
            "norm_gap_pct": round(100 * (ll_us - ll_mk) / (LN2 - ll_mk), 2)}
        res["activation"][s] = {
            "tank_nonzero": float(np.mean([r["act_tank"] != 0 for r in rs])),
            "late_nonzero": float(np.mean([r["act_late"] != 0 for r in rs])),
            "bridge_fired": int(sum(r["act_bridge"] for r in rs)),
            "prior_active_fallback": int(sum(r["act_prior_active"] for r in rs))}
        print(f"{s}: ll {ll_us:.5f} mkt {ll_mk:.5f} gap {ll_us-ll_mk:+.5f} "
              f"norm {res['seasons'][s]['norm_gap_pct']:.2f}%", flush=True)

    for arm_key, name in TERMS.items():
        entry = {"term": name, "per_season": {}, "groups": {}}
        for s in SEASONS:
            rs = [r for r in allrows if r["season"] == s]
            d = np.array([ll_pg(r, f"p_{arm_key}") - ll_pg(r, "p_base")
                          for r in rs])
            entry["per_season"][s] = paired_bootstrap(d)
            entry["per_season"][s]["n_moved"] = int(
                sum(abs(r[f"p_{arm_key}"] - r["p_base"]) > 1e-12 for r in rs))
        for g, ss in GROUPS.items():
            rs = [r for r in allrows if r["season"] in ss]
            d = np.array([ll_pg(r, f"p_{arm_key}") - ll_pg(r, "p_base")
                          for r in rs])
            bs = paired_bootstrap(d)
            bs["n_moved"] = int(sum(abs(r[f"p_{arm_key}"] - r["p_base"]) > 1e-12
                                    for r in rs))
            mv = np.array([i for i, r in enumerate(rs)
                           if abs(r[f"p_{arm_key}"] - r["p_base"]) > 1e-12])
            bs["active_only"] = paired_bootstrap(d[mv]) if len(mv) else None
            entry["groups"][g] = bs
        res["terms"][arm_key] = entry
        print(f"\n{name}", flush=True)
        for g in GROUPS:
            b = entry["groups"][g]
            print(f"  {g:20s} {b['mean']:+.5f} CI({b['lo']:+.5f},{b['hi']:+.5f})"
                  f" n={b['n']} moved={b['n_moved']} {b['verdict']}", flush=True)

    json.dump(res, open(ROOT / "data" / f"{tag}_results.json", "w"), indent=1)
    print(f"\nwrote data/{tag}_results.json", flush=True)


if __name__ == "__main__":
    main()
