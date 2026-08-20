#!/usr/bin/env python3
"""CLUSTERING AUDIT — which registered gates used i.i.d. resampling, how much
too narrow their CIs were, and whether that explains D110/D111.

(1) enumerates every bootstrap in scripts/ + nbapred/ and classifies it;
(2) measures the intra-season ICC and design effect of the per-game delta for
    every load-bearing shipped arm;
(3) applies the measured design effect to the registered z's of the shipped
    terms and reports which would still have cleared a 95% bar.

Read-only.  Output: data/cv_cluster_audit.json
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"

# Hand-verified classification of every bootstrap that authorized a REGISTERED
# gate (the regex sweep below covers the rest mechanically).
GATE_BOOTSTRAPS = {
    "D46 schedule layer": ("scripts/es_*/early campaign + of_transfer_ablation.py:90",
                           "iid-game", "SHIPPED"),
    "D62 cross-season carry": ("scripts/es2_hardstop.py:319", "iid-game", "SHIPPED"),
    "D73 April tank program": ("scripts/apr_program.py:643", "iid-game", "SHIPPED"),
    "D90 late-state layer": ("scripts/ov_latestate_gate.py", "iid-game", "REVERTED D112"),
    "D110/D112 transfer ablations": ("scripts/of_transfer_ablation.py:90", "iid-game", "audit"),
    "D124 F2 event-recency": ("scripts/pg_eventrecency.py", "iid-game", "RETIRED"),
    "D130 urgency arms": ("scripts/pg_urgency2.py:330", "iid-game", "no-ship"),
    "D136 travel arms": ("scripts/tv_gate.py:149", "iid-game", "no-ship"),
    "GATE_POLICY_V2 power table": ("scripts/ba_gatepower.py:44", "iid-game", "policy"),
    "D64/D71 NS-portfolio joint": ("scripts/ba_portfolio.py", "iid-game", "no-ship"),
    "D128 props reversal review": ("scripts/d79_reversal_review.py", "cluster-PLAYER", "audit"),
    "D133 props early-minutes ramp": ("scripts/pr_ramp_gate.py:122", "cluster-PLAYER", "SHIPPED"),
}

# Registered z's of the shipped terms, backed out of published CIs (D110 sec 3).
REGISTERED_Z = {"D46 sched": 3.46, "D73 tank gate": 2.81, "D90 late-state": 2.68,
                "D55 cold-start": 2.26, "D62 carry": 2.20}

# key in data/cv_rescore.json used to source the measured design effect
DEFF_SOURCE = {"D46 sched": "D46_sched_warmfloor", "D73 tank gate": "D73_tank_oldfloor",
               "D90 late-state": "D90_latestate_warmfloor",
               "D62 carry": "D62_carry_warmfloor"}


def sweep() -> dict:
    iid, clus = [], []
    for p in sorted(list(ROOT.glob("scripts/*.py")) + list(ROOT.glob("nbapred/**/*.py"))):
        t = p.read_text(errors="ignore")
        if "integers(" not in t:
            continue
        has_iid = bool(re.search(r"integers\(0,\s*len\((\w+)\)", t)) or \
            bool(re.search(r"integers\(0,\s*n,\s*size=", t))
        has_cl = bool(re.search(r"integers\(0,\s*len\(uniq", t)) or \
            bool(re.search(r"integers\(0,\s*(K|Gn|len\(groups\))", t))
        rel = p.relative_to(ROOT).as_posix()
        if has_cl:
            clus.append(rel)
        if has_iid:
            iid.append(rel)
    return dict(iid_game_level=iid, clustered=clus,
                n_iid=len(iid), n_clustered=len(clus))


def main() -> None:
    rs = json.loads((DATA / "cv_rescore.json").read_text())
    out = {"sweep": sweep(), "gate_bootstraps": GATE_BOOTSTRAPS}

    rows = []
    for k, rep in rs.items():
        c = rep.get("clustering")
        if not c:
            continue
        rows.append(dict(
            arm=rep["label"], key=k, n=rep["n"],
            est=rep["pooled"]["est"],
            se_iid=c["iid"]["se"], sig_iid=c["iid"]["sig"],
            se_season_boot=c["season_cluster_boot"]["se"],
            sig_season_boot=c["season_cluster_boot"]["sig"],
            deff_boot=c["design_effect_season"],
            icc=c["icc_season"]["icc"], deff_anova=c["icc_season"]["deff"],
            t_lo=c["season_mean_t"]["lo"], t_hi=c["season_mean_t"]["hi"],
            sig_t=c["season_mean_t"]["sig"],
            flips=c["flips"], flips_on_t=c["flips_on_t"]))
    out["design_effects"] = rows

    print(f"{'arm':58s} {'est':>9s} {'SE_iid':>8s} {'SE_seas':>8s} {'DEFFb':>6s} "
          f"{'ICC':>9s} {'DEFFa':>6s}  iid  seas  t")
    for r in rows:
        print(f"{r['arm'][:58]:58s} {r['est']:+9.5f} {r['se_iid']:8.5f} "
              f"{r['se_season_boot']:8.5f} {r['deff_boot']:6.2f} {r['icc']:+9.5f} "
              f"{r['deff_anova']:6.2f}  "
              f"{'SIG' if r['sig_iid'] else 'ns ':>3s}  "
              f"{'SIG' if r['sig_season_boot'] else 'ns ':>3s}  "
              f"{'SIG' if r['sig_t'] else 'ns '}")

    # --------- does clustering explain D110/D111? apply DEFF to registered z
    print("\nREGISTERED z ADJUSTED BY THE MEASURED DESIGN EFFECT")
    print(f"{'term':16s} {'z_reg':>6s} {'DEFFb':>6s} {'DEFFa':>6s} {'z/sqrt? no: z/DEFF_se':>0s}")
    adj = {}
    for term, z in REGISTERED_Z.items():
        src = DEFF_SOURCE.get(term)
        if src is None:
            adj[term] = dict(z_registered=z, note="not measurable (structurally dead)")
            print(f"{term:16s} {z:6.2f}   n/a    n/a   structurally dead (D110 1b)")
            continue
        c = rs[src]["clustering"]
        db, da = c["design_effect_season"], c["icc_season"]["se_inflation"]
        zb, za = z / max(db, 1e-9), z / max(da, 1e-9)
        adj[term] = dict(z_registered=z, deff_boot=db, se_infl_anova=da,
                         z_clustered_boot=zb, z_clustered_anova=za,
                         still_passes_95_boot=bool(zb > 1.96),
                         still_passes_95_anova=bool(za > 1.96))
        print(f"{term:16s} {z:6.2f} {db:6.2f} {da:6.2f}  -> z_boot {zb:.2f} "
              f"{'PASS' if zb > 1.96 else 'FAIL'} | z_anova {za:.2f} "
              f"{'PASS' if za > 1.96 else 'FAIL'}")
    out["registered_z_adjusted"] = adj

    # --------- selection-vs-clustering decomposition of the D111 shortfall
    # D110: apparent term-sum +0.0134; holdout says +0.0074 of +0.0145 transfers.
    out["explanation_test"] = dict(
        claim="Anti-conservative i.i.d. gates explain D110/D111's evaporation.",
        verdict="PARTIAL — necessary but far from sufficient.",
        detail=("Measured season design effects on the sides arms are 0.85-1.39 "
                "(bootstrap) / 0.89-2.29 (ANOVA), i.e. the i.i.d. CIs were "
                "0-40% too narrow, not 2-3x. Correcting the registered z's by "
                "the measured factor removes the pass for D90 (2.68->1.93) and "
                "puts D73 on the line (2.81->2.21, and its season-mean t "
                "interval straddles zero) while leaving D46 clearly through "
                "(3.46->3.09). That is EXACTLY D111's per-term verdict, "
                "reproduced from the inference alone — so clustering would "
                "have caught the two convicted terms AT GATE TIME. But the "
                "selection channel is bigger: E[max z] at 96-511 comparisons "
                "is 2.34-2.91 all by itself, and no per-gate SE correction "
                "touches that."))

    (DATA / "cv_cluster_audit.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\nsweep: {out['sweep']['n_iid']} files with i.i.d. resampling, "
          f"{out['sweep']['n_clustered']} with clustered resampling")
    print(f"-> {DATA / 'cv_cluster_audit.json'}")


if __name__ == "__main__":
    main()
