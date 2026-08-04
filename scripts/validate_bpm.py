#!/usr/bin/env python3
"""D85 step 2 validation — in-house BPM (nbapred/features/bpm.py) vs
basketball-reference season 'advanced' BPM values, 3 backtest seasons.
Pre-registered target: corr > 0.95 (docs/EXTERNAL_MODELS.md).

Comparison is at the (player, team-stint) level — B-R multi-team combined
rows (2TM/3TM/TOT) are excluded; our stints come from compute_bpm(season).
Names matched on accent-stripped lowercase (suffixes dropped); team codes
mapped B-R -> NBA (PHO->PHX, BRK->BKN, CHO->CHA). Pages raw-cached by
ext_bbref_bpm.fetch_current (one polite fetch per season, <=20 req/min)."""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ext_bbref_bpm import fetch_current  # noqa: E402
from nbapred.db import connect  # noqa: E402
from nbapred.features.bpm import compute_bpm  # noqa: E402

SEASONS = {"2023-24": 2024, "2024-25": 2025, "2025-26": 2026}
BR2NBA = {"PHO": "PHX", "BRK": "BKN", "CHO": "CHA"}
SUFFIX = re.compile(r"\s+(jr|sr|ii|iii|iv|v)\.?$")


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[.'’-]", "", s.lower()).strip()
    return SUFFIX.sub("", s)


def main() -> None:
    con = connect(read_only=True)
    names = dict(con.execute(
        "SELECT player_id, full_name FROM nba_players").fetchall())
    abbrev = dict(con.execute(
        "SELECT DISTINCT team_id, team_abbrev FROM nba_games "
        "WHERE season >= '2023-24'").fetchall())
    print(f"{'season':8} {'n':>4} {'corr':>6} {'corr500':>8} {'mae':>6} "
          f"{'mae500':>7}  unmatched")
    pooled, pooled500 = [], []
    for season, yyyy in SEASONS.items():
        br: dict = {}
        for r in fetch_current(yyyy):
            team = BR2NBA.get(r["team"], r["team"])
            if team and not team[0].isdigit() and team != "TOT":
                # page carries BOTH regular-season and playoff tables ->
                # duplicate keys; keep all rows and join on closest minutes
                br.setdefault((norm(r["player"]), team), []).append(
                    (r["bpm"], r["mp"]))
        ours = compute_bpm(con, season)
        rows, miss = [], []
        for (pid, tid), v in ours.items():
            key = (norm(names.get(pid, "")), abbrev.get(tid, ""))
            if key in br:
                bpm_br, mp_br = min(br[key],
                                    key=lambda x: abs(x[1] - v["mp"]))
                rows.append((v["bpm"], bpm_br, v["mp"], mp_br))
            else:
                miss.append((names.get(pid), abbrev.get(tid), round(v["mp"])))
        a = np.array(rows)
        big = a[a[:, 2] >= 500]
        corr = np.corrcoef(a[:, 0], a[:, 1])[0, 1]
        corr5 = np.corrcoef(big[:, 0], big[:, 1])[0, 1]
        mae = np.mean(np.abs(a[:, 0] - a[:, 1]))
        mae5 = np.mean(np.abs(big[:, 0] - big[:, 1]))
        pooled.append(a)
        pooled500.append(big)
        big_miss = sorted((m for m in miss if m[2] >= 200), key=lambda m: -m[2])
        print(f"{season:8} {len(a):4d} {corr:6.4f} {corr5:8.4f} {mae:6.3f} "
              f"{mae5:7.3f}  n={len(miss)} big={big_miss[:4]}")
    a = np.vstack(pooled)
    b = np.vstack(pooled500)
    print(f"{'POOLED':8} {len(a):4d} {np.corrcoef(a[:,0],a[:,1])[0,1]:6.4f} "
          f"{np.corrcoef(b[:,0],b[:,1])[0,1]:8.4f} "
          f"{np.mean(np.abs(a[:,0]-a[:,1])):6.3f} "
          f"{np.mean(np.abs(b[:,0]-b[:,1])):7.3f}")
    print(f"minutes join sanity: mean |mp_us - mp_br| = "
          f"{np.mean(np.abs(a[:,2]-a[:,3])):.1f} min (n={len(a)})")


if __name__ == "__main__":
    main()
