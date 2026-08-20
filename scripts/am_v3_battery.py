"""AM V3 BATTERY — GATE_POLICY_V2.md sections 8-11 on the three pre-registered arms.

Uses nbapred.eval.splits (no hand-rolled bootstrap).  Reports for each arm:
pooled i.i.d. (secondary), season-clustered CI (THE SHIPPING CI, section 9.1),
ICC + ANOVA design effect, cluster-mean t at K-1=4 dof (the conservative bound,
section 9.3), rolling-origin, LOSO (diagnostic only), legacy dev/holdout,
7-day block bootstrap, era decomposition, and adjudicate().

Also: the pre-registered concentration windows, the B-C contrast (the direct
test of the owner's phase-varying hypothesis), the A-vs-C same-channel check,
the COVID-frame sensitivity, and realized MDE80 everywhere.

Input data/am_pergame.csv.  Output data/am_v3_battery.json.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from nbapred.eval import splits as S  # noqa: E402

ROWS = list(csv.DictReader(open(ROOT / "data" / "am_pergame.csv")))
for r in ROWS:
    for k in ("y", "gp_home", "gp_away", "neutral"):
        r[k] = int(r[k])
    for k in ("p_mkt", "p_ctl", "p_A", "p_B", "p_C",
              "term_A", "term_B", "term_C"):
        r[k] = float(r[k])

SEASON = np.array([r["season"] for r in ROWS])
DATE = np.array([r["game_date"] for r in ROWS])
Y = np.array([r["y"] for r in ROWS])
GPH = np.array([r["gp_home"] for r in ROWS])
GPA = np.array([r["gp_away"] for r in ROWS])


def panel(pc, pt, mask=None, label=""):
    m = np.ones(len(ROWS), bool) if mask is None else np.asarray(mask, bool)
    return S.Panel.from_logloss(SEASON[m], Y[m], np.asarray(pc)[m],
                                np.asarray(pt)[m], date=DATE[m], label=label)


def slim(rep):
    """Trim full_report to the fields the entry must quote."""
    cl = rep["clustering"]
    o = {"n": rep["n"], "pooled_iid": rep["pooled"],
         "pooled_mde80": rep["pooled_mde80"],
         "clustering": cl,
         "per_season": rep["per_season"],
         "rolling_origin": {"folds": [{"test": f["test"],
                                       "est": f["fold"]["est"],
                                       "lo": f["fold"]["lo"],
                                       "hi": f["fold"]["hi"],
                                       "sig": f["fold"]["sig"],
                                       "mde80": f["mde80"]}
                                      for f in rep["rolling_origin"]["folds"]],
                            "sign_consistency":
                                rep["rolling_origin"]["sign_consistency"],
                            "drift_per_season":
                                rep["rolling_origin"]["drift_per_season"]},
         "loso": {"sign_consistency": rep["loso"]["sign_consistency"],
                  "sd_of_folds": rep["loso"]["sd_of_folds"],
                  "min_fold": rep["loso"]["min_fold"],
                  "max_fold": rep["loso"]["max_fold"],
                  "jackknife_range": rep["loso"]["jackknife_range"],
                  "independent_folds": rep["loso"]["independent_folds"],
                  "per_season_test_on": {f["left_out"]: f["test_on"]["est"]
                                         for f in rep["loso"]["folds"]}},
         "legacy": rep["legacy"],
         "era": rep["era"],
         "block_bootstrap": rep.get("block_bootstrap"),
         "temporal_design_effect": rep.get("temporal_design_effect"),
         "verdict": rep["verdict"]}
    return o


def main():
    out = {"prereg_sha256": "a3aae24b3193c2cf04c5e438b4c3c0b7ea51211ac5100c47"
                            "4f3d3dec64ec3bc9",
           "control": "data/capstone_pergame_d132.csv (D132 certified)",
           "n_rows": len(ROWS)}
    pc = np.array([r["p_ctl"] for r in ROWS])
    arms = {a: np.array([r[f"p_{a}"] for r in ROWS]) for a in "ABC"}

    windows = {
        "A": ("either_gp_ge55", (GPH >= 55) | (GPA >= 55)),
        "B": ("both_gp_ge20", (GPH >= 20) & (GPA >= 20)),
        "C": ("both_gp_ge20", (GPH >= 20) & (GPA >= 20)),
    }
    for a in "ABC":
        rep = S.full_report(panel(pc, arms[a], label=f"ARM {a}"))
        out[f"arm_{a}"] = slim(rep)
        wname, wmask = windows[a]
        wrep = S.full_report(panel(pc, arms[a], wmask,
                                   label=f"ARM {a} @ {wname}"))
        out[f"arm_{a}_window_{wname}"] = slim(wrep)
        # second pre-registered window for B/C
        if a in "BC":
            m2 = (GPH >= 55) & (GPA >= 55)
            out[f"arm_{a}_window_both_gp_ge55"] = slim(
                S.full_report(panel(pc, arms[a], m2, label=f"ARM {a} @ late")))
        # RT2-comparable universe (drop the 10 neutral-site games)
        nn = np.array([r["neutral"] == 0 for r in ROWS])
        out[f"arm_{a}_rt2_universe"] = {
            "n": int(nn.sum()),
            "pooled": S.paired_bootstrap(panel(pc, arms[a], nn).d)}
        # COVID-frame sensitivity: drop 2021-22 (E3, whose 730d fit frame
        # reaches into E2/E1)
        no21 = SEASON != "2021-22"
        out[f"arm_{a}_drop_2021_22"] = {
            "n": int(no21.sum()),
            "pooled": S.paired_bootstrap(panel(pc, arms[a], no21).d),
            "cluster": S.cluster_mean_t_interval(panel(pc, arms[a], no21).d,
                                                 SEASON[no21])}

    # ---- THE CONTRAST: does phase-varying beat constant? ------------------
    pB, pC = arms["B"], arms["C"]
    con_panel = panel(pC, pB, label="ARM B minus ARM C (phase-varying - constant)")
    out["contrast_B_minus_C"] = slim(S.full_report(con_panel))
    m2 = (GPH >= 55) & (GPA >= 55)
    out["contrast_B_minus_C_window_both_gp_ge55"] = slim(
        S.full_report(panel(pC, pB, m2, label="B-C @ both gp>=55")))
    m41 = (GPH >= 41) & (GPA >= 41)
    out["contrast_B_minus_C_window_both_gp_ge41"] = slim(
        S.full_report(panel(pC, pB, m41, label="B-C @ both gp>=41")))

    # ---- A vs C same-channel check on the gp>=55 window -------------------
    mA = (GPH >= 55) | (GPA >= 55)
    dA = panel(pc, arms["A"], mA).d
    dC = panel(pc, arms["C"], mA).d
    out["A_vs_C_same_channel"] = {
        "window": "either_gp_ge55", "n": int(mA.sum()),
        "A_mean": float(dA.mean()), "C_mean": float(dC.mean()),
        "corr_of_per_game_deltas": float(np.corrcoef(dA, dC)[0, 1]),
        "corr_of_terms": float(np.corrcoef(
            np.array([r["term_A"] for r in ROWS])[mA],
            np.array([r["term_C"] for r in ROWS])[mA])[0, 1])}

    (ROOT / "data" / "am_v3_battery.json").write_text(json.dumps(
        out, indent=1, default=float))

    # ---- console table ----------------------------------------------------
    def line(tag, o):
        p, cl = o["pooled_iid"], o["clustering"]
        t = cl["season_mean_t"]
        print(f"{tag:44s} n={o['n']:5d} est={p['est']:+.5f} "
              f"iid({p['lo']:+.5f},{p['hi']:+.5f}){'SIG' if p['sig'] else ' ns'} "
              f"| seasclus({cl['season_cluster_boot']['lo']:+.5f},"
              f"{cl['season_cluster_boot']['hi']:+.5f})"
              f"{'SIG' if cl['season_cluster_boot']['sig'] else ' ns'} "
              f"| t4dof({t['lo']:+.5f},{t['hi']:+.5f})"
              f"{'SIG' if t['sig'] else ' ns'} "
              f"| MDE80={o['pooled_mde80']:.5f}")
    for k, v in out.items():
        if isinstance(v, dict) and "pooled_iid" in v:
            line(k, v)
    print()
    for k, v in out.items():
        if isinstance(v, dict) and "verdict" in v:
            print(f"{k:44s} {v['verdict']['tier']}")
            for f in v["verdict"]["flags"]:
                print(f"      - {f}")


if __name__ == "__main__":
    main()
