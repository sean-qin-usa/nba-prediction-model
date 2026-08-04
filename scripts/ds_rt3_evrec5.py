"""RT3 — F2 EVENT-RECENCY at 5-SEASON power, on the UN-STARVED corpus.

The last F2 verdict (journal dbd47e) was a "4-season power re-gate" that
reported the evidence had WEAKENED, its headline reason being: "NEW season
2022-23 NEGATIVE (-0.00095), 'three consistent positives' broken".

AUDIT FINDING (scripts/ds_starvation_diag.py): that 2022-23 arm was the most
DATA-STARVED season in program history. With nba_games floored at 2022-23:
  * fit_schedule_layer's 730d window held ZERO games at the first refit ->
    shrink weight 0.0 -> the schedule layer was the hardcoded SCHED_PRIOR
    constants for the opening weeks (median weight 0.507 vs 0.80 elsewhere)
  * continuity_map() returned None -> the D62 carry was ENTIRELY ABSENT
The season that "broke the consistency" was scored by a crippled control.

This rerun is the SAME script, SAME construction, SAME seed — only the corpus
changed (2021-22 + 2020-21 + 2019-20 schedules ingested) and 2021-22 is added
as a fifth eval season (odds_market season_end 2022 covers 1321 games;
player_game_stats 002_21 is complete at 1230/1230).

CAVEAT registered up front: 2021-22's own carry leans on 2020-21
player_game_stats, which is 780/1080 complete — its carry is degraded, not
absent. Reported separately as well as pooled.

Gate: unchanged from pg_eventrecency.py (paired bootstrap 2000x seed 7 on the
isolation delta p_exp vs same-run control), now pooled over 5 seasons.
Read-only DB.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import pg_eventrecency as pg  # noqa: E402

# corpus extension — the ONLY change vs the frozen re-gate
pg.SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
pg.PRIOR_SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")   # the 4-season run
pg.OUT_CSV = REPO / "data" / "ds_rt3_evrec5_pergame.csv"
pg.OUT_JSON = REPO / "data" / "ds_rt3_evrec5_summary.json"
# power comparison anchored on the STARVED-corpus 4-season re-gate, not the
# older 3-season run, so "power_vs_prior" isolates the corpus effect
pg.PRIOR_JSON = REPO / "data" / "pg_eventrecency_summary.json"

if __name__ == "__main__":
    pg.main()
