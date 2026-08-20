"""D142 GameRotation role-state construction: the properties the finding rests on.

Nothing in `nbapred/` consumes this construction — ARM R was a NO-SHIP. These
tests exist so the ARTIFACT and the CONSTRUCTION stay honest for the follow-up
arm the register queues, and so the two claims the entry makes cannot silently
rot:

  1. POINT-IN-TIME. `row_state` reads the player's history strictly before the
     scored day. GameRotation is post-game data; a leak here would void the
     whole finding.
  2. ZERO OUTSIDE THE WINDOW. The bucket is NA (and therefore the correction is
     exactly 0.0) unless the immediately-prior played game is rotation-covered
     AND a full 5-game covered window exists.
"""
import importlib.util as _ilu
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = _ilu.spec_from_file_location("ad_role_gate", ROOT / "scripts" / "ad_role_gate.py")
G = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(G)

ART = ROOT / "data" / "ad_role_flags.npz"


def _byp(n=30, starter_pattern=None, ords=None):
    """One synthetic player: `byp` entry of (ords, mins, seasons, starter)."""
    ords = np.arange(100, 100 + n) if ords is None else np.asarray(ords)
    mins = np.full(n, 30.0)
    seas = np.array(["2024-25"] * n, dtype=object)
    star = (np.zeros(n) if starter_pattern is None
            else np.asarray(starter_pattern, float))
    return {1: (ords, mins, seas, star)}


# ------------------------------------------------------------------ 1. PIT

def test_row_state_ignores_the_scored_game_and_everything_after():
    """Mutating the scored day's own row and every later row must not change
    the state computed for that day."""
    n, i = 30, 20
    star = np.array([0.0] * 15 + [1.0] * 15)
    byp = _byp(n, star)
    day = int(byp[1][0][i])
    before = G.row_state(byp, 1, day, "2024-25")

    ords, mins, seas, st = byp[1]
    st2, mins2 = st.copy(), mins.copy()
    st2[i:] = 1.0 - st2[i:]          # flip the present and the whole future
    mins2[i:] = 5.0
    after = G.row_state({1: (ords, mins2, seas, st2)}, 1, day, "2024-25")
    assert before == after


def test_row_state_uses_strictly_prior_games_only():
    """A player's 21st game must see exactly 20 prior games, and the gp count
    must not include the scored game."""
    byp = _byp(30, np.ones(30))
    day = int(byp[1][0][20])
    proj, n_hist, gp, rb, mb = G.row_state(byp, 1, day, "2024-25")
    assert n_hist == 20
    assert gp == 20


def test_artifact_has_no_duplicate_player_day_and_is_sorted():
    if not ART.exists():
        pytest.skip("data/ad_role_flags.npz not built")
    z = np.load(ART)
    key = np.stack([z["player_id"], z["ord"]], 1)
    assert len(np.unique(key, axis=0)) == len(key), "duplicate (player_id, ord)"
    assert set(np.unique(z["starter"]).tolist()) <= {0, 1}


# -------------------------------------------------- 2. zero outside window

def test_bucket_is_NA_when_the_immediately_prior_game_is_uncovered():
    """gap>0 -> NA. Cover games 0..14 then leave 15..19 uncovered; the state at
    game 20 must be NA even though a 5-game covered window exists earlier."""
    star = np.array([1.0] * 15 + [np.nan] * 15)
    byp = _byp(30, star)
    day = int(byp[1][0][20])
    assert G.row_state(byp, 1, day, "2024-25")[3] == "NA"


def test_bucket_is_NA_without_a_full_five_game_covered_window():
    star = np.full(30, np.nan)
    star[17:20] = 1.0                      # only 3 covered, all adjacent
    byp = _byp(30, star)
    day = int(byp[1][0][20])
    assert G.row_state(byp, 1, day, "2024-25")[3] == "NA"


def test_promoted_and_demoted_and_stable_partition():
    day_of = lambda byp, i: int(byp[1][0][i])
    # last game a start, majority of the last 5 NOT starts -> PROMOTED
    star = np.array([0.0] * 19 + [1.0] + [0.0] * 10)
    byp = _byp(30, star)
    assert G.row_state(byp, 1, day_of(byp, 20), "2024-25")[3] == "PROMOTED"
    # last game a bench game, majority of the last 5 starts -> DEMOTED
    star = np.array([1.0] * 19 + [0.0] + [1.0] * 10)
    byp = _byp(30, star)
    assert G.row_state(byp, 1, day_of(byp, 20), "2024-25")[3] == "DEMOTED"
    # unchanged role -> STABLE (and the shipped correction would be 0.0)
    byp = _byp(30, np.ones(30))
    assert G.row_state(byp, 1, day_of(byp, 20), "2024-25")[3] == "STABLE"


def test_five_game_window_is_odd_so_the_majority_never_ties():
    """The prereg's threshold-free claim: with a 5-game window sr5 can never be
    exactly 0.5, so `sr5 < 0.5` / `sr5 > 0.5` partition the space with no
    tuned cut point."""
    for k in range(6):
        assert abs(k / 5.0 - 0.5) > 1e-12
