#!/usr/bin/env python3
"""Certification manifest — what produced a number, and how exactly it repeats.

WHY. D230b established that this pipeline is not bit-reproducible: running
`prod_by_season.py` twice with identical code and settings differs by ~1e-14 on
p_us, because several DuckDB aggregations feeding float reductions never pinned
their row order. Two fixes cut that floor about 5x but not to zero. The floor is
eleven orders of magnitude below any gate delta, so no conclusion is at risk —
but it means `max|dp| = 0` is NOT an achievable control, and a control that
cannot reach zero cannot distinguish "inert" from "small".

So the honest instrument is not a zero, it is a MANIFEST plus a stated floor:
record what the code and inputs were, and state the numeric tolerance below
which two runs are indistinguishable. A later run either matches the manifest
(same inputs, same code) or it does not, and any residual difference is compared
against the floor rather than against zero.

    python scripts/cert_manifest.py            # write + print
    python scripts/cert_manifest.py --check    # compare to the stored manifest
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "cert_manifest.json"

#: measured in D230b: max|d p_us| between two runs of identical code/settings
#: AFTER the sorted-summation and ORDER BY fixes. Two runs closer than this are
#: indistinguishable; a difference above it is a real change.
NUMERIC_FLOOR_P = 3.3e-15
NUMERIC_FLOOR_MARGIN = 2.2e-13

CODE = ["nbapred", "scripts/prod_by_season.py", "scripts/bet_engine.py",
        "scripts/d230_channel_offset.py", "scripts/d232_absence_gate.py"]
DATA = ["data/offset_coefs.json", "data/p_out.csv.gz",
        "data/pit_frame.csv.gz", "data/channel_pergame.csv",
        "data/d230_prereg.sha256", "data/d232_prereg.sha256"]
#: env vars that CHANGE THE NUMBERS; recorded because an unset switch is a
#: silent default and D229 is a whole entry about silent defaults.
ENV = ["SOFT_AVAIL", "OFFSET_LAYER", "LATE_STATE", "TANK_TERM",
       "OPEN_TIME_OUTS", "REPORT_OUTS", "INACTIVE_OUTS", "ORACLE_PLAYED_OUTS",
       "ORACLE_MINUTES", "TANK_SEASON_FLOOR", "PROD_SEASONS",
       "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _code_hashes() -> dict:
    out = {}
    for c in CODE:
        p = ROOT / c
        if p.is_dir():
            for f in sorted(p.rglob("*.py")):
                if "__pycache__" in f.parts:
                    continue
                out[str(f.relative_to(ROOT))] = _sha(f)
        elif p.exists():
            out[c] = _sha(p)
    return out


def build() -> dict:
    code = _code_hashes()
    roll = hashlib.sha256(
        "".join(f"{k}:{v}" for k, v in sorted(code.items())).encode()).hexdigest()
    data = {d: (_sha(ROOT / d) if (ROOT / d).exists() else None) for d in DATA}
    try:
        import duckdb
        import numpy
        import pandas
        libs = {"numpy": numpy.__version__, "pandas": pandas.__version__,
                "duckdb": duckdb.__version__}
    except Exception:                                        # noqa: BLE001
        libs = {}
    try:
        git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        head = git.stdout.strip() or None
    except Exception:                                        # noqa: BLE001
        head = None
    coef = {}
    cp = ROOT / "data" / "offset_coefs.json"
    if cp.exists():
        c = json.load(open(cp))
        coef = dict(zip(c.get("features", []), c.get("coefs", [])))
    return {
        "code_rollup_sha256": roll,
        "code": code,
        "data": data,
        "env": {k: os.environ.get(k) for k in ENV},
        "libs": libs,
        "python": platform.python_version(),
        "git_head": head,
        "shipped_coefficients": {"offset": coef},
        "numeric_floor": {
            "p_us": NUMERIC_FLOOR_P,
            "margin": NUMERIC_FLOOR_MARGIN,
            "note": ("D230b: pipeline is NOT bit-reproducible; two runs of "
                     "identical code differ up to this much. Compare controls "
                     "against this floor, never against zero."),
        },
    }


def main() -> int:
    m = build()
    if "--check" in sys.argv:
        if not OUT.exists():
            print("no stored manifest — run without --check first")
            return 1
        old = json.load(open(OUT))
        diffs = []
        if old["code_rollup_sha256"] != m["code_rollup_sha256"]:
            for k, v in m["code"].items():
                if old["code"].get(k) != v:
                    diffs.append(f"CODE  {k}")
            for k in old["code"]:
                if k not in m["code"]:
                    diffs.append(f"CODE  {k} (removed)")
        for k, v in m["data"].items():
            if old["data"].get(k) != v:
                diffs.append(f"DATA  {k}")
        for k, v in m["env"].items():
            if old["env"].get(k) != v:
                diffs.append(f"ENV   {k}: {old['env'].get(k)!r} -> {v!r}")
        if diffs:
            print(f"MANIFEST DIFFERS ({len(diffs)} item(s)):")
            for d in diffs[:40]:
                print("  " + d)
            return 1
        print("manifest matches — same code, same inputs, same switches")
        return 0
    json.dump(m, open(OUT, "w"), indent=1, sort_keys=True)
    print(f"code rollup   {m['code_rollup_sha256'][:16]}...  "
          f"({len(m['code'])} files)")
    print(f"git HEAD      {m['git_head']}")
    print(f"python {m['python']}  " +
          "  ".join(f"{k} {v}" for k, v in m["libs"].items()))
    print("switches that change numbers:")
    for k, v in m["env"].items():
        print(f"    {k:22} {v if v is not None else '(unset -> code default)'}")
    print(f"offset coefficients: {m['shipped_coefficients']['offset']}")
    print(f"numeric floor: p_us {NUMERIC_FLOOR_P:.1e}  "
          f"margin {NUMERIC_FLOOR_MARGIN:.1e}  (NOT zero — D230b)")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
