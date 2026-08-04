#!/usr/bin/env python3
"""D94 — SECOND LOOK at the D86 talent-ensemble gate with the DAILY
re-identified EPM leg (epm_history_daily, built by scripts/epm_reid.py).

THIS IS A SECOND LOOK, NOT A FRESH PRE-REGISTRATION. The D86 gate
(scripts/ens_talent_gate.py) was spent 2026-07-31 and returned NULL/no-op
with the weekly-capture EPM leg (median asof gap 2/5/13 days by season).
Sean explicitly authorized the endpoint-grid re-identification and ONE
rerun of the same config with the denser leg. Differences vs D86, in full:
  1. EPM source: epm_history_daily (539 daily asof dates, re-id accuracy
     100% on 2,690 held-out top-5 row-dates, 98.6% on 6,449 held-out
     era-B Wayback pairs) instead of 101 weekly Wayback capture dates.
  2. Epistemic class of the EPM values: the daily grid is served by the
     CURRENT model version (PIT-in-data, NOT as-published — the endpoint
     re-runs history; D86 decision kept backtests on as-published capture
     values). This rerun therefore uses the D43-DARKO epistemic class for
     the EPM leg. Flagged, deliberate, and part of why the verdict is
     labeled second-look.
Everything else (weights, z-mapping, swap wiring, seed 20260730, bootstrap
2000x, seasons, windows) is IDENTICAL by construction: base module reused,
only epm_asof/epm_captures are overridden.

Outputs: data/ens_talent_gate_daily_pergame.csv,
         data/ens_talent_gate_daily_results.json
"""
import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import ens_talent_gate as base  # noqa: E402
from nbapred.db import connect  # noqa: E402

OUT_CSV = REPO / "data/ens_talent_gate_daily_pergame.csv"
OUT_JSON = REPO / "data/ens_talent_gate_daily_results.json"


def load_daily():
    con = connect(read_only=True)
    rows = con.execute("""
        SELECT asof_date, player_id, tot_epm
        FROM epm_history_daily ORDER BY asof_date""").fetchall()
    con.close()
    by_date = {}
    for d, p, t in rows:
        by_date.setdefault(d, {})[int(p)] = float(t)
    dates = sorted(by_date)
    print(f"daily EPM: {len(dates)} asof dates "
          f"({dates[0]} .. {dates[-1]}), "
          f"{sum(len(v) for v in by_date.values())} rows")
    return dates, by_date


def daily_epm_asof(caps, d):
    dates, by_date = caps
    lo, hi = 0, len(dates)
    while lo < hi:
        mid = (lo + hi) // 2
        if dates[mid] < d:
            lo = mid + 1
        else:
            hi = mid
    if lo == 0:
        return None, None
    best = dates[lo - 1]
    return dict(by_date[best]), (d - best).days


def main():
    caps = load_daily()
    base.epm_asof = daily_epm_asof            # module-global override
    base.PERGAME_CSV = OUT_CSV
    base.RESULTS_JSON = OUT_JSON
    if "--analyze" in sys.argv:
        import csv
        with open(OUT_CSV) as f:
            all_rows = []
            for r in csv.DictReader(f):
                r.update(y=int(r["y"]), p_mkt=float(r["p_mkt"]),
                         p_ctrl=float(r["p_ctrl"]), p_var=float(r["p_var"]),
                         cm_pit=float(r["cm_pit"]), cm_ens=float(r["cm_ens"]),
                         gp_home=int(r["gp_home"]), gp_away=int(r["gp_away"]))
                all_rows.append(r)
        diag = {}
        dj = OUT_JSON.with_suffix(".diag.json")
        if dj.exists():
            diag = json.loads(dj.read_text())
        base.analyze(all_rows, diag)
        return
    diag = {}
    all_rows = []
    for s in base.SEASONS:
        all_rows += base.season_run(s, caps, diag)
    import csv
    with open(OUT_CSV, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(all_rows)
    OUT_JSON.with_suffix(".diag.json").write_text(
        json.dumps(diag, default=str))
    base.analyze(all_rows, diag)
    print("\nSECOND-LOOK LABEL: this rerun re-spends no registration; the "
          "D86 NULL stands unless this shows a qualitatively different, "
          "CI-solid result (in which case it seeds a NEW pre-registration "
          "for a future one-shot, not a ship).")


if __name__ == "__main__":
    main()
