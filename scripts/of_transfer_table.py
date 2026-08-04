"""OF-1b — the TRANSFER TABLE: reads data/of_transfer_pergame.csv and reports,
per shipped term, dev-season effect vs held-out-season effect, the
difference-in-differences with a CI, the term's ACTIVATION RATE in each group
(so an untestable term is not mislabelled as a non-transferring one), and a
verdict.

DEV      = 2024-25 + 2025-26 (the seasons the feature set was developed on)
HELD     = 2022-23 (+ 2021-22) — never scorable during the campaign
GATE2324 = 2023-24 — in every pooled gate, reported for reference, NOT held out

DiD = dev_effect - held_effect, bootstrapped independently in each group
(different games, so the two resamples are independent) — a term whose CI on
the DiD excludes 0 helped MORE on the seasons it was developed on.

Usage: python scripts/of_transfer_table.py
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

DEV = ["2024-25", "2025-26"]
HELD = ["2022-23", "2021-22"]
HELD1 = ["2022-23"]
GATE = ["2023-24"]
ALLGATE = ["2023-24", "2024-25", "2025-26"]   # every season any gate could see
LN2 = 0.6931471805599453


def ll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot(d, n_boot=2000, seed=SEED):
    d = np.asarray(d, float)
    if len(d) == 0:
        return 0.0, 0.0, 0.0, None
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5)), means


def main():
    rows = list(csv.DictReader(open(ROOT / "data" / "of_transfer_pergame.csv")))
    for r in rows:
        for k, v in r.items():
            if k.startswith(("p_", "act_", "y", "gp_")):
                r[k] = float(v)

    def sub(seasons):
        return [r for r in rows if r["season"] in seasons]

    print("## season headline (normalized: gap / market skill above coin-flip)\n")
    print("| season | n | ours | market | gap | norm gap % | group |")
    print("|---|---|---|---|---|---|---|")
    for s in ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]:
        rs = sub([s])
        if not rs:
            continue
        y = np.array([r["y"] for r in rs])
        a = float(ll(y, np.array([r["p_base"] for r in rs])).mean())
        m = float(ll(y, np.array([r["p_mkt"] for r in rs])).mean())
        g = "HELD-OUT" if s in HELD else ("DEV" if s in DEV else "gate season")
        print(f"| {s} | {len(rs)} | {a:.5f} | {m:.5f} | {a-m:+.5f} | "
              f"{100*(a-m)/(LN2-m):.1f}% | {g} |")

    out = {}
    print("\n## per-term transfer table "
          "(effect = ll(term OFF) - ll(term ON); POSITIVE = term HELPS)\n")
    print("| term | DEV 24-26 | HELD-OUT 22-23 | HELD-OUT 21-23 | "
          "gate 23-24 | DiD (dev-held) | verdict |")
    print("|---|---|---|---|---|---|---|")
    for arm, name in TERMS.items():
        cell, draws, act = {}, {}, {}
        for key, ss in (("dev", DEV), ("held1", HELD1), ("held", HELD),
                        ("gate", GATE)):
            rs = sub(ss)
            y = np.array([r["y"] for r in rs])
            d = ll(y, np.array([r[f"p_{arm}"] for r in rs])) - \
                ll(y, np.array([r["p_base"] for r in rs]))
            m, lo, hi, dr = boot(d)
            moved = float(np.mean([abs(r[f"p_{arm}"] - r["p_base"]) > 1e-12
                                   for r in rs]))
            cell[key] = (m, lo, hi, moved)
            draws[key] = dr
            act[key] = moved
        did = draws["dev"] - draws["held1"]
        did_m = cell["dev"][0] - cell["held1"][0]
        did_lo, did_hi = float(np.percentile(did, 2.5)), float(np.percentile(did, 97.5))

        def f(k):
            m, lo, hi, mv = cell[k]
            return f"{m:+.5f} ({lo:+.5f},{hi:+.5f}) act{mv*100:.0f}%"

        if act["held1"] < 0.01:
            verdict = "UNTESTABLE (inactive held-out)"
        elif cell["held1"][1] > 0:
            verdict = "TRANSFERS"
        elif cell["held1"][0] <= 0 and cell["dev"][1] > 0:
            verdict = "DEV-ONLY (overfit signature)"
        elif cell["held1"][2] < 0:
            verdict = "REVERSES on held-out"
        else:
            verdict = "NS both / underpowered"
        print(f"| {name} | {f('dev')} | {f('held1')} | {f('held')} | "
              f"{f('gate')} | {did_m:+.5f} ({did_lo:+.5f},{did_hi:+.5f}) | "
              f"{verdict} |")
        out[arm] = {"term": name,
                    **{k: {"mean": cell[k][0], "lo": cell[k][1],
                           "hi": cell[k][2], "activation": cell[k][3]}
                       for k in cell},
                    "did_dev_minus_held2223": {"mean": did_m, "lo": did_lo,
                                               "hi": did_hi},
                    "verdict": verdict}

    print("\n## activation diagnostics (why some cells are untestable)\n")
    print("| season | tank!=0 | late!=0 | bridge games | ff-not-ready games "
          "(D16 prior alive) | carry flips ff.ready |")
    print("|---|---|---|---|---|---|")
    for s in ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]:
        rs = sub([s])
        if not rs:
            continue
        print(f"| {s} | {np.mean([r['act_tank']!=0 for r in rs])*100:.1f}% "
              f"| {np.mean([r['act_late']!=0 for r in rs])*100:.1f}% "
              f"| {int(sum(r['act_bridge'] for r in rs))} "
              f"| {int(sum(r['act_prior_active'] for r in rs))} "
              f"| {int(sum(r['act_carry_ff_ready_delta'] for r in rs))} |")

    # ---- sum of shipped terms, per group ---------------------------------
    print("\n## additive check: sum of the individual term effects, per group\n")
    for key, ss in (("DEV 24-26", DEV), ("HELD-OUT 22-23", HELD1),
                    ("HELD-OUT 21-23", HELD), ("gate 23-24", GATE)):
        tot = sum(out[a][{"DEV 24-26": "dev", "HELD-OUT 22-23": "held1",
                          "HELD-OUT 21-23": "held", "gate 23-24": "gate"}[key]]
                  ["mean"] for a in TERMS)
        print(f"  {key:16s} sum of per-term effects = {tot:+.5f}")

    json.dump(out, open(ROOT / "data" / "of_transfer_table.json", "w"), indent=1)
    print("\nwrote data/of_transfer_table.json")


if __name__ == "__main__":
    main()
