#!/usr/bin/env python3
"""RETROSPECTIVE RE-SCORING of the decisions where the single dev/holdout split
was load-bearing, under the D139 multi-split + era-stratified + CLUSTERED
protocol (nbapred/eval/splits.py).

No model is re-run.  Every arm here is an ALREADY-REGISTERED per-game / per-row
artifact on disk, so the deltas are bit-for-bit the quantities the original
entries gated on — only the INFERENCE changes.  That is the point: if a verdict
flips, it flips because the statistics were wrong, not because the numbers moved.

Artifacts consumed (all read-only):
  data/of_transfer_pergame.csv     D110 ablations at the OLD corpus floor
  data/cf_holdout_new_pergame.csv  D112 ablations at the WARM (derived) floor
  data/f2_settle_pergame.csv       D124 F2 event-recency
  data/pg_urgency2_pergame.csv     D130 urgency arms (DEV ONLY, 3 seasons)
  data/pr_ramp_dev_rows.npz +
  data/pr_ramp_holdout_rows.npz    D133 props early-minutes ramp (SHIPPED)
  data/capstone_pergame.csv        D132 certified baseline (vs market)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nbapred.eval.splits import Panel, format_report, full_report  # noqa: E402

DATA = ROOT / "data"
B, SEED = 2000, 20260801


def load_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def panel_from_pergame(rows, ctrl_col, treat_col, label, meta=None):
    """delta = ll(ctrl) - ll(treat); POSITIVE = treat better.

    For an ABLATION artifact the CONTROL is the term-OFF arm and the TREATMENT
    is the full model, so a positive delta means THE TERM HELPS — identical to
    the D110 convention `effect = ll(term OFF) - ll(term ON)`.
    """
    return Panel.from_logloss(
        season=[r["season"] for r in rows],
        y=[float(r["y"]) for r in rows],
        p_ctrl=[float(r[ctrl_col]) for r in rows],
        p_treat=[float(r[treat_col]) for r in rows],
        date=[r["game_date"] for r in rows],
        label=label, meta=meta or {})


def run(panel, out, key, note=""):
    rep = full_report(panel, B, SEED)
    rep["note"] = note
    out[key] = rep
    print("\n" + "=" * 92)
    print(format_report(rep))
    if note:
        print(f"NOTE: {note}")
    return rep


def main() -> None:
    out: dict = {}

    of = load_csv(DATA / "of_transfer_pergame.csv")
    cf = load_csv(DATA / "cf_holdout_new_pergame.csv")

    # ---------------------------------------------------------------- D46
    run(panel_from_pergame(of, "p_no_sched", "p_base",
                           "D46 SCHEDULE LAYER (old floor, D110 arms)"),
        out, "D46_sched_oldfloor",
        "The only term D111 called an individual out-of-sample PASS.")
    run(panel_from_pergame(cf, "p_no_sched", "p_base",
                           "D46 SCHEDULE LAYER (warm floor, D112 arms)"),
        out, "D46_sched_warmfloor")

    # ---------------------------------------------------------------- D73
    run(panel_from_pergame(of, "p_no_tank", "p_base",
                           "D73 TANK TERM (old floor = COLD coefficient)"),
        out, "D73_tank_oldfloor",
        "D110's cold-coefficient confound: the term is identically 0 in 2021-22.")
    run(panel_from_pergame(cf, "p_no_tank", "p_base",
                           "D73 TANK TERM (warm floor, D112 arms) — SHIPPED, default ON"),
        out, "D73_tank_warmfloor",
        "This is the arm D112 used to WITHDRAW D73's exoneration.")

    # ---------------------------------------------------------------- D62
    run(panel_from_pergame(cf, "p_no_carry", "p_base",
                           "D62 CROSS-SEASON CARRY (warm floor) — SHIPPED"),
        out, "D62_carry_warmfloor")

    # ------------------------------------------------- D90 negative control
    run(panel_from_pergame(cf, "p_no_late", "p_base",
                           "D90 LATE-STATE (warm floor) — ALREADY REVERTED (D112)"),
        out, "D90_latestate_warmfloor",
        "Negative control: a term the register has already convicted of overfit. "
        "If the harness cannot see it, the harness is not sensitive enough.")

    # ---------------------------------------------------------------- D91
    run(panel_from_pergame(cf, "p_no_bridge", "p_base",
                           "D91 OCTOBER BRIDGE (warm floor)"),
        out, "D91_bridge_warmfloor")

    # ---------------------------------------------------------------- D124
    f2 = load_csv(DATA / "f2_settle_pergame.csv")
    run(panel_from_pergame(f2, "p_ctl", "p_exp",
                           "D124 F2 EVENT-RECENCY — RETIRED on a pooled NS"),
        out, "D124_F2",
        "Was the retirement era-driven? Positive delta = F2 helps.")

    # ---------------------------------------------------------------- D130
    ug = load_csv(DATA / "pg_urgency2_pergame.csv")
    run(panel_from_pergame(ug, "p_ctl", "p_a",
                           "D130 ARM A URGENCY — DEV ONLY (2023-26), all games"),
        out, "D130_armA_full",
        "STRUCTURAL LIMIT: this artifact has NO holdout seasons, so rolling-origin "
        "has 2 folds and the era decomposition sees only E5 and E6. The single "
        "split was load-bearing here in the strongest possible sense — there was "
        "no second era to check.")

    # ---------------------------------------------- D138 NS-portfolio joint
    # Registered the same day as this entry; its holdout pass
    # (+0.001979 CI(+0.000283,+0.003652), 2.29 sigma) was computed with an
    # i.i.d. game-level bootstrap (scratch_nsport_joint.py:307), so it is
    # in scope for the clustering check.
    ns = load_csv(DATA / "nsport_joint_pergame.csv")
    run(panel_from_pergame(ns, "p_ctl", "p_joint",
                           "D138 NS-PORTFOLIO JOINT BUNDLE — all 5 seasons"),
        out, "D138_joint_all5",
        "D138 ships PROVISIONAL on the holdout arm; re-scored here under "
        "clustered inference.")
    ho_rows = [r for r in ns if r["season"] in ("2021-22", "2022-23")]
    run(panel_from_pergame(ho_rows, "p_ctl", "p_joint",
                           "D138 NS-PORTFOLIO JOINT — HOLDOUT arm only (the authorizing number)"),
        out, "D138_joint_holdout",
        "K=2 seasons: the cluster machinery is at its structural limit here and "
        "the t interval has 1 dof. Read the design effect, not the interval.")

    # ------------------------------------------------- D132 certified gap
    cap = load_csv(DATA / "capstone_pergame.csv")
    run(panel_from_pergame(cap, "p_mkt", "p_us",
                           "D132 CERTIFIED HEADLINE — us vs market"),
        out, "D132_vs_market",
        "Negative = the market beats us (expected). Re-scored to get an honest "
        "clustered interval on the headline gap.")

    # ---------------------------------------------------------------- D133
    dv = np.load(DATA / "pr_ramp_dev_rows.npz", allow_pickle=True)
    ho = np.load(DATA / "pr_ramp_holdout_rows.npz", allow_pickle=True)

    def cat(k):
        return np.concatenate([dv[k], ho[k]])

    seas, mon = cat("season"), cat("month")
    pid = cat("player_id")
    for arm, name in (("A", "ARM A (SHIPPED)"), ("A0", "ARM A0 (level-only control)"),
                      ("C", "ARM C (two-axis, not shipped)")):
        ctrl, tre = cat("crps_ctrl"), cat(f"crps_{arm}")
        # PRE-REGISTERED PRIMARY WINDOW: Oct-Nov points CRPS
        m = np.isin(mon, [10, 11])
        p = Panel.from_losses(seas[m], ctrl[m], tre[m], date=None,
                              cluster=pid[m],
                              label=f"D133 {name} — Oct-Nov points CRPS (primary window)")
        run(p, out, f"D133_{arm}_octnov",
            "Cluster = player (the D128/D133 convention). Positive = ramp better.")
    # all-season secondary
    ctrl, tre = cat("crps_ctrl"), cat("crps_A")
    run(Panel.from_losses(seas, ctrl, tre, cluster=pid,
                          label="D133 ARM A — ALL-SEASON points CRPS"),
        out, "D133_A_allseason")

    (DATA / "cv_rescore.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n-> {DATA / 'cv_rescore.json'}")


if __name__ == "__main__":
    main()
