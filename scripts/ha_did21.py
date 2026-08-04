"""HA-(2e) CI for the within-2020-21 crowd DOSE-RESPONSE difference-in-differences.

Limited attendance returned to a minority of NBA arenas from ~March 2021. If
the crowd mechanism in section 2 is right, 2020-21's home edge should RISE from
the first half of the season to the second, relative to the ordinary
within-season profile of a normal season (which section 5 shows is flat-to-
slightly-declining).

  DiD = (2020-21 late - 2020-21 early) - (normal late - normal early)

Same pooled specification as ha_decomp.py (season-team FE + schedule controls),
with the crowd stratum interacted with season half. Bootstrap over games.

DESCRIPTIVE, FULL-SAMPLE. Corroborating evidence only -- the primary crowd
estimate is the level contrast in section 2b.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ha_core import boot_ci, load_panel
from ha_decomp import fit

OUT = Path("/tmp/claude-1004/-hdd-steveqin-sean-dev/"
           "4912ce58-69cd-488d-a5f4-0d3ed2eed30c/scratchpad")


def prep(d):
    d = d.copy()
    med = d.groupby("season")["days_into"].transform("median")
    d["cell"] = d["stratum"] + "_" + np.where(d["days_into"] > med, "late", "early")
    return d


def did(sub):
    h, _, *_ = fit(sub, "cell")
    return ((h["nocrowd_late"] - h["nocrowd_early"]) -
            (h["normal_late"] - h["normal_early"]))


def main():
    rng = np.random.default_rng(20260801)
    d = prep(load_panel())
    pt = did(d)
    h, _, *_ = fit(d, "cell")
    print("pure HFA by (stratum x season half):")
    for k in sorted(h):
        print(f"  {k:16s} {h[k]:+.4f}")
    bs = []
    for _ in range(1200):
        s = d.iloc[rng.integers(0, len(d), len(d))]
        try:
            bs.append(did(s))
        except Exception:
            continue
    lo, hi = boot_ci(np.array(bs))
    print(f"\nDiD = {pt:+.4f}  95% CI ({lo:+.4f},{hi:+.4f})  "
          f"{'SIG' if lo > 0 or hi < 0 else 'NS'}   n_boot={len(bs)}")
    (OUT / "ha_did21.json").write_text(json.dumps(
        dict(did=float(pt), lo=lo, hi=hi, hfa=h), indent=2, default=float))


if __name__ == "__main__":
    main()
