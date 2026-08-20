"""AM — calibration battery, BH q=0.10 family recount, and CONSTRUCTION IDENTITY.

(1) CONSTRUCTION IDENTITY (V2 section 6.6, the D141 lesson: verify by
    construction, not by name).  Three proofs for ARM A:
      a. `fit_form_k` and every FORM_* constant resolve to
         scripts/ba_windowed.py (the D71 gate script) by module file path —
         imported, never re-typed.
      b. the walk-forward k trajectory is compared date-by-date against D102
         RT2's stored trajectory (`data/ds_rt2_results.json`), which was
         itself verbatim-D71.  Bit-identical k on shared dates proves the
         coefficient path is the same estimator.
      c. the term is EXACTLY zero outside its gp>=55 gate.
    For ARMS B/C: the h function actually evaluated is printed for every
    phase, so the switch cannot be named after a hypothesis it does not
    implement.

(2) CALIBRATION BATTERY (diagnostic; the veto only bites on a ship).
    ECE10, ECE20, Brier, logit-link slope, Hosmer-Lemeshow, per-season sign.

(3) BH q=0.10 over the append-only family register data/bh_family.csv,
    K = 106 at D141 + 3 arms = 109, using the section-9-mandated CLUSTERED p.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import ba_windowed  # noqa: E402
from am_gate import h_phase, h_const  # noqa: E402

ROWS = list(csv.DictReader(open(ROOT / "data" / "am_pergame.csv")))
Y = np.array([int(r["y"]) for r in ROWS])
SEASON = np.array([r["season"] for r in ROWS])
P = {k: np.array([float(r[f"p_{k}"]) for r in ROWS]) for k in ("ctl", "A", "B", "C")}
BATT = json.load(open(ROOT / "data" / "am_v3_battery.json"))


def ece(y, p, bins):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -1e-9, 1 + 1e-9
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            tot += m.sum() / len(y) * abs(y[m].mean() - p[m].mean())
    return float(tot)


def link_slope(y, p):
    x = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    # one Newton step logistic fit of y ~ a + b*x, started at (0,1)
    a, b = 0.0, 1.0
    for _ in range(60):
        z = a + b * x
        mu = 1 / (1 + np.exp(-z))
        W = mu * (1 - mu)
        X = np.column_stack([np.ones_like(x), x])
        g = X.T @ (y - mu)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H + 1e-9 * np.eye(2), g)
        a, b = a + step[0], b + step[1]
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(b), float(a)


def hl_p(y, p, g=10):
    from math import erfc, sqrt
    edges = np.quantile(p, np.linspace(0, 1, g + 1))
    edges[0], edges[-1] = -1e-9, 1 + 1e-9
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, g - 1)
    stat = 0.0
    for b in range(g):
        m = idx == b
        if m.sum() < 2:
            continue
        o, e = y[m].sum(), p[m].sum()
        v = (p[m] * (1 - p[m])).sum()
        if v > 0:
            stat += (o - e) ** 2 / v
    dof = g - 2
    # survival of chi2 via Wilson-Hilferty
    z = ((stat / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / np.sqrt(2 / (9 * dof))
    return float(stat), float(0.5 * erfc(z / sqrt(2)))


def main():
    out = {}

    # ---------------- (1) construction identity ---------------------------
    ident = {"fit_form_k_module": ba_windowed.fit_form_k.__module__,
             "fit_form_k_file": ba_windowed.__file__,
             "FORM_GP": ba_windowed.FORM_GP, "FORM_N": ba_windowed.FORM_N,
             "FORM_SHRINK": ba_windowed.FORM_SHRINK,
             "FORM_WINDOW_D": ba_windowed.FORM_WINDOW_D,
             "h_phase_evaluated": {f"gp={g}": h_phase(g)
                                   for g in (20, 30, 40, 41, 50, 54, 55, 70, 81)},
             "h_const_evaluated": {f"gp={g}": h_const(g)
                                   for g in (20, 30, 40, 41, 50, 54, 55, 70, 81)}}
    gate = json.load(open(ROOT / "data" / "am_gate_results.json"))
    mine = {x["date"]: x["k_form"] for x in gate["k_trajectory"]}
    rt2 = json.load(open(ROOT / "data" / "ds_rt2_results.json"))
    shared = [(x["date"], x["k"], mine.get(x["date"]))
              for x in rt2["k_trajectory_tail"] if x["date"] in mine]
    ident["k_trajectory_vs_RT2"] = [
        {"date": d, "rt2_k": a, "am_k": b, "abs_diff": abs(a - b)}
        for d, a, b in shared]
    ident["k_trajectory_max_abs_diff_vs_RT2"] = (
        max((abs(a - b) for _, a, b in shared), default=None))
    ident["zero_outside_window_max_abs_dp"] = {
        a: gate[f"arm_{a}"]["zero_outside_window_max_abs_dp"] for a in "ABC"}
    ident["n_touched"] = {a: gate[f"arm_{a}"]["n_touched"] for a in "ABC"}
    out["construction_identity"] = ident

    # ---------------- (2) calibration battery -----------------------------
    calib = {}
    for a in "ABC":
        row = {}
        for tag, p in (("ctl", P["ctl"]), ("arm", P[a])):
            sl, ic = link_slope(Y, p)
            hs, hp = hl_p(Y, p)
            row[tag] = {"ece10": ece(Y, p, 10), "ece20": ece(Y, p, 20),
                        "brier": float(np.mean((Y - p) ** 2)),
                        "link_slope": sl, "link_intercept": ic,
                        "hl_stat": hs, "hl_p": hp,
                        "logloss": float(np.mean(
                            -(Y * np.log(p) + (1 - Y) * np.log(1 - p))))}
        row["delta"] = {k: row["arm"][k] - row["ctl"][k]
                        for k in ("ece10", "ece20", "brier", "logloss")}
        row["delta"]["link_slope_toward_1"] = (
            abs(row["ctl"]["link_slope"] - 1) - abs(row["arm"]["link_slope"] - 1))
        per = {}
        for s in sorted(set(SEASON)):
            m = SEASON == s
            per[s] = float(np.mean(-(Y[m] * np.log(P["ctl"][m])
                                     + (1 - Y[m]) * np.log(1 - P["ctl"][m])))
                           - np.mean(-(Y[m] * np.log(P[a][m])
                                       + (1 - Y[m]) * np.log(1 - P[a][m]))))
        row["per_season_delta"] = per
        row["seasons_better"] = int(sum(v > 0 for v in per.values()))
        calib[a] = row
    out["calibration"] = calib

    # ---------------- (3) BH q=0.10 ---------------------------------------
    fam = list(csv.DictReader(open(ROOT / "data" / "bh_family.csv")))
    K0 = len(fam)
    new = []
    for a, name in (("A", "AM ARM A — F1 late-gated form re-gate on the D132 "
                          "certified stack (post-D118, D90 gone)"),
                    ("B", "AM ARM B — phase-varying memory correction "
                          "h=34/13/8 (owner's hypothesis)"),
                    ("C", "AM ARM C — constant memory correction h=21 "
                          "(measurement's answer)")):
        b = BATT[f"arm_{a}"]
        s = b["clustering"]["season_cluster_boot"]
        new.append({"name": name, "source": "AM (this run)",
                    "est": s["est"], "lo": s["lo"], "hi": s["hi"],
                    "se": s["se"],
                    "z": s["est"] / s["se"] if s["se"] else float("nan"),
                    "p_onesided": s["p_wrongside"],
                    "basis": "season_clustered_bootstrap"})
    def _p(v):
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
        return x if np.isfinite(x) else None
    fam_p = [(r["name"], _p(r["p_onesided"])) for r in fam]
    n_missing = sum(1 for _, x in fam_p if x is None)
    allp = ([(n, x) for n, x in fam_p if x is not None]
            + [(r["name"], float(r["p_onesided"])) for r in new])
    allp.sort(key=lambda x: x[1])
    K = len(allp)
    q = 0.10
    bh = []
    for i, (n, p) in enumerate(allp, 1):
        bh.append({"rank": i, "name": n, "p": p, "thr": q * i / K,
                   "pass": p <= q * i / K})
    kmax = max((b["rank"] for b in bh if b["pass"]), default=0)
    out["bh"] = {"K_register_rows": K0, "K_with_usable_p": K, "q": q,
                 "register_rows_without_p": n_missing,
                 "largest_passing_rank": kmax,
                 "arms": [{"name": r["name"], "p": float(r["p_onesided"]),
                           "rank": next(b["rank"] for b in bh
                                        if b["name"] == r["name"]),
                           "thr": next(b["thr"] for b in bh
                                       if b["name"] == r["name"]),
                           "pass": next(b["pass"] for b in bh
                                        if b["name"] == r["name"])}
                          for r in new],
                 "top10": bh[:10]}

    # append-only registration of the three arms (V2 section 4: at
    # PRE-REGISTRATION time, winners and losers alike)
    with open(ROOT / "data" / "bh_family.csv", "a", newline="") as f:
        w = csv.writer(f)
        for r in new:
            w.writerow([r["name"], r["source"], r["est"], r["lo"], r["hi"],
                        r["se"], r["z"], r["p_onesided"], r["basis"]])

    (ROOT / "data" / "am_calib_bh.json").write_text(json.dumps(out, indent=1,
                                                               default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
