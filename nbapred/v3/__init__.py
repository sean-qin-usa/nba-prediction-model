"""v3 — the state-space restructure (docs/V3_SPEC.md is the contract).

Shadow package: NOTHING here touches the production stack (nbapred/model,
nbapred/engine). v3 reads the DB read-only and writes ONLY its own tables
(player_states, state_shocks, v3_predictions) through the guarded writer in
schema.py. Production modules flip to v3 internals one at a time, and only
on a passed gate (scripts/gate_v3.py, G1 protocol).

Milestones built here:
  M0 — StateBank skeleton + table schemas + shadow-run logging.
  M1 — team-level DLM pilot (team_dlm.py): daily-evolving team offense/defense
       states, obs = opponent-adjusted game efficiency margins, per-day process
       noise fit by marginal likelihood, season-boundary event shock.
"""
from .hyper import HyperParams, TeamHyper
from .state_bank import DIMS, StateBank
from .team_dlm import TeamDLM
