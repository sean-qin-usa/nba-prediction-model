"""OF-3 — SELECTION-BURDEN CENSUS: how many chances did the campaign have to
find a +0.004 that isn't there?

Three independent, mechanical counts (nothing hand-curated except the gate
table at the bottom, which is transcribed from docs/DECISIONS.md):

  N_configs  distinct model configurations SCORED PER GAME that still exist on
             disk = every non-market probability column in every data/*pergame*
             CSV. A hard LOWER bound: experiments that overwrote their CSV, or
             only kept a summary, are invisible here.
  N_compare  distinct bootstrap comparisons REPORTED = every {mean, lo, hi}
             triple in data/**.json. This is the number that matters for the
             multiple-comparison burden, because the campaign repeatedly
             shipped or promoted on a SUBSET/window result (D64 heavy-fav
             intersection, D71 late-gated form, D75 R4-low-t), not only on the
             pooled endpoint.
  N_dlines   decision-register entries.

Then the arithmetic the register never did:
  * SE of each shipped term's pooled gate, backed out of its published 95% CI.
  * E[z | z > 1.96] = 2.34 — the expected apparent size of a term that passes
    a 95% bar BY CHANCE, in units of its own SE.
  * E[max z] over N independent comparisons — what the BEST of N null terms
    looks like.
  * expected count of false passes = N * 0.025 (one-sided).

Read-only. Usage: python scripts/of_selection_census.py
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CI_KEYS = [{"lo", "hi"}, {"ci_lo", "ci_hi"}, {"lo95", "hi95"}]
SKIP_COLS = {"p_mkt", "p_market", "p_home_spread"}

# published POOLED gate results, transcribed from docs/DECISIONS.md
GATES = [
    ("D46 schedule layer", 0.00539, 0.00240, 0.00850, "SHIPPED"),
    ("D54/D55 cold-start revive", 0.00099, 0.00013, 0.00185, "SHIPPED"),
    ("D62 carry (authorizing gate)", 0.00097, 0.000085, 0.001816, "SHIPPED"),
    ("D63 carry (ship-confirm)", 0.00083, -0.00018, 0.00181, "confirm NS"),
    ("D73 tank (authorizing gate)", 0.00401, 0.00112, 0.00672, "SHIPPED"),
    ("D73 tank (shipped port re-measure)", 0.00418, 0.00239, 0.00589, "SHIPPED"),
    ("D90 late-state layer", 0.00189, 0.00053, 0.00329, "SHIPPED"),
    ("D64 NS-portfolio", 0.00266, -0.00068, 0.00592, "no-ship"),
    ("D71 windowed portfolio", 0.00155, -0.00268, 0.00562, "no-ship"),
    ("D102 RT4 fitted blend", 0.00077, -0.00036, 0.00187, "no-ship"),
    ("D86 talent ensemble", -0.00006, -0.00144, 0.00131, "no-ship"),
    ("D97 perfect-talent ORACLE (bound)", 0.00400, 0.00339, 0.00463, "oracle"),
]


def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def norm_sf(x):
    return 0.5 * math.erfc(x / math.sqrt(2))


def walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)


def is_triple(v):
    return (isinstance(v, list) and len(v) == 3
            and all(isinstance(x, (int, float)) for x in v))


def main():
    # ---- N_configs -------------------------------------------------------
    cfg, per_file = 0, {}
    for f in sorted(glob.glob(str(ROOT / "data" / "*.csv"))):
        try:
            with open(f) as fh:
                hdr = next(csv.reader(fh))
        except Exception:
            continue
        cols = [c for c in hdr
                if re.match(r"^(p_|prob)", c) and c not in SKIP_COLS]
        if cols:
            per_file[os.path.basename(f)] = cols
            cfg += len(cols)

    # ---- N_compare -------------------------------------------------------
    cmp_n, per_json = 0, {}
    for f in (sorted(glob.glob(str(ROOT / "data" / "*.json")))
              + sorted(glob.glob(str(ROOT / "data" / "logs" / "*.json")))):
        try:
            o = json.load(open(f))
        except Exception:
            continue
        n = 0
        for d in walk(o):
            ks = {k.lower() for k in d}
            if any(ck <= ks for ck in CI_KEYS):
                n += 1
            for k, v in d.items():
                if is_triple(v) and any(t in k.lower() for t in
                                        ("delta", "ci", "gate", "boot", "diff")):
                    n += 1
        if n:
            per_json[os.path.basename(f)] = n
            cmp_n += n

    # ---- N_dlines --------------------------------------------------------
    txt = (ROOT / "docs" / "DECISIONS.md").read_text()
    dlines = len([l for l in txt.split("\n") if re.match(r"^- D\d+", l)])
    scripts = len(glob.glob(str(ROOT / "scripts" / "*.py")))

    out = {"N_configs_scored_pergame": cfg,
           "N_configs_files": len(per_file),
           "N_bootstrap_comparisons_reported": cmp_n,
           "N_comparison_files": len(per_json),
           "N_decision_lines": dlines,
           "N_scripts": scripts,
           "note_uncounted": "D84-C's 49-feature confirmation battery kept no "
                             "per-game CSV; es_fadeshape 6 variants, D61 link "
                             "family and every overwritten capstone_pergame_*"
                             " are single surviving columns, so N_configs is a "
                             "hard lower bound."}

    print(f"N_configs (per-game prob columns on disk) : {cfg} "
          f"across {len(per_file)} files")
    print(f"N_compare (reported bootstrap CIs)        : {cmp_n} "
          f"across {len(per_json)} files")
    print(f"N_dlines / N_scripts                      : {dlines} / {scripts}")
    print()

    # ---- what chance produces at that depth -------------------------------
    trunc = norm_pdf(1.96) / norm_sf(1.96)
    out["E_z_given_pass"] = round(trunc, 3)
    print(f"E[z | z>1.96] = {trunc:.3f}  -> a NULL term that passes a 95% bar "
          f"looks like {trunc:.2f} x its own SE")
    out["by_N"] = {}
    for N in (25, 50, 94, 150, 300, cmp_n):
        a = math.sqrt(2 * math.log(N))
        emax = a - (math.log(math.log(N)) + math.log(4 * math.pi)) / (2 * a)
        out["by_N"][N] = {"expected_false_passes": round(N * 0.025, 1),
                          "E_max_z": round(emax, 2)}
        print(f"  N={N:4d}: E[#false passes] = {N*0.025:5.1f}   E[max z] = {emax:.2f}")
    print()

    # ---- shipped terms vs the chance bar ----------------------------------
    print(f"{'gate':38s} {'mean':>9s} {'SE':>8s} {'z':>6s} "
          f"{'chance-pass size':>17s}  verdict")
    out["gates"] = {}
    for n, m, lo, hi, v in GATES:
        se = (hi - lo) / (2 * 1.96)
        z = m / se
        out["gates"][n] = {"mean": m, "se": round(se, 6), "z": round(z, 2),
                           "chance_pass_size": round(trunc * se, 5),
                           "registered": v}
        print(f"{n:38s} {m:+9.5f} {se:8.5f} {z:6.2f} {trunc*se:+17.5f}  {v}")

    json.dump(out, open(ROOT / "data" / "of_selection_census.json", "w"), indent=1)
    print("\nwrote data/of_selection_census.json")


if __name__ == "__main__":
    main()
