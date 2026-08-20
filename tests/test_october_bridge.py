"""D84-A October bridge: construction parity, activation gating, and the
ps-continuity carry proxy.

The heavy test is D73-style live-parity: build a truncated in-memory DB
(everything strictly before the 2025-26 opener, as live would see it on
opening night), build OctoberBridge from it, and require its cm_ps margins
(with the gate's oracle outs convention) to match the pre-registered gate
construction table (data/rw_week1_psroster.csv cm_ps_o, verified 5e-15
against scripts/rw_early_v1_gate.py) on the 2025-26 week-1 active games.
The number predict_today computes on opening night 2026 must equal what
the gate/backtest would later compute for the same game.
"""
import datetime as dt
import sys
import warnings
from functools import lru_cache
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEASON = "2025-26"
OPENER = dt.date(2025, 10, 21)          # 2025-26 002 opener (rw_early_cf2)


# The gate table (data/rw_week1_psroster.csv) was built when `nba_games` held
# only 2022-23..2025-26. D101 ingested 2021-22/2020-21/2019-20 schedules, and
# OctoberBridge's trailing-minutes leg is season-AGNOSTIC by design ("last 10
# 002 games with >=12 min strictly before the cutoff, spans the prior season")
# and reaches player_game_stats THROUGH a join on nba_games — so 002 rows that
# had always existed for 2021-22/2020-21 became visible and the construction
# widened. Measured impact (scripts/ds_bridge_impact.py): 5 of 53 week-1 games
# move, mean |dcm_ps| 0.025 pts, max 0.604; 16 players change contribution and
# 6 join the roster (players whose only >=12-min 002 history predates 2022-23).
# The parity assertion below therefore pins the REGISTERED corpus so the F6
# pre-registration stays verifiable, and test_corpus_widening_bounded tracks
# the delta so it can never drift silently.
REGISTERED_CORPUS_FLOOR = "2022-23"


@lru_cache(maxsize=2)
def _truncated_mem_db(season_floor=None):
    """In-memory DuckDB holding only what live would see before the opener
    (includes this season's 001 preseason games — they end days earlier).
    season_floor pins `nba_games` to a corpus vintage; None = the real one."""
    import duckdb
    from nbapred.config import DB_PATH
    mem = duckdb.connect()
    mem.execute(f"ATTACH '{DB_PATH}' AS src (READ_ONLY)")
    flt = f" AND season >= '{season_floor}'" if season_floor else ""
    mem.execute(f"CREATE TABLE nba_games AS SELECT * FROM src.nba_games "
                f"WHERE game_date < ?{flt}", [OPENER])
    mem.execute("CREATE TABLE player_game_stats AS SELECT * FROM "
                "src.player_game_stats WHERE game_id IN "
                "(SELECT game_id FROM nba_games)")
    mem.execute("CREATE TABLE darko_history AS SELECT * FROM "
                "src.darko_history WHERE date < ?", [OPENER])
    return mem


@lru_cache(maxsize=1)
def _week1_frame():
    import pandas as pd
    df = pd.read_csv(ROOT / "data/rw_early_decomp_pergame.csv",
                     dtype={"game_id": str})
    ps = pd.read_csv(ROOT / "data/rw_week1_psroster.csv",
                     dtype={"game_id": str})
    sub = df[(df.season == SEASON) & (df.cm == 0)].merge(
        ps[["season", "game_id", "cm_ps_o"]], on=["season", "game_id"])
    assert len(sub) >= 50, "week-1 active window missing from gate tables"
    return sub


def _oracle_outs_and_ids(game_ids):
    """Oracle outs convention of the gate: roster member not in the game's
    played set (played sets come from the FULL DB — hindsight, used only to
    reproduce the gate's scoring; live uses the injury feed)."""
    from nbapred.db import connect
    con = connect(read_only=True)
    ab2id = {a: int(t) for t, a in con.execute(
        "SELECT DISTINCT team_id, team_abbrev FROM nba_games WHERE season=?",
        [SEASON]).fetchall()}
    ph = ",".join("?" * len(game_ids))
    played = {}
    for g, t, p in con.execute(
            f"SELECT game_id, team_id, player_id FROM player_game_stats "
            f"WHERE game_id IN ({ph}) AND seconds>0", list(game_ids)).fetchall():
        played.setdefault((g, int(t)), set()).add(int(p))
    con.close()
    return ab2id, played


def test_bridge_matches_gate_construction():
    """Live-visible OctoberBridge == pre-registered gate cm_ps (<1e-9)."""
    from nbapred.model.october_bridge import OctoberBridge
    sub = _week1_frame()
    gids = [g.zfill(10) for g in sub.game_id]
    ab2id, played = _oracle_outs_and_ids(tuple(gids))
    mem = _truncated_mem_db(REGISTERED_CORPUS_FLOOR)
    br = OctoberBridge(mem, SEASON, before=OPENER)
    roster = {}
    for p, (t, _c) in br.contrib.items():
        roster.setdefault(t, set()).add(p)
    checked = 0
    for r in sub.itertuples():
        gid = r.game_id.zfill(10)
        hid, aid = ab2id[r.home], ab2id[r.away]
        out_h = roster.get(hid, set()) - played.get((gid, hid), set())
        out_a = roster.get(aid, set()) - played.get((gid, aid), set())
        got = br.margin(hid, aid, out_h, out_a)
        assert abs(got - float(r.cm_ps_o)) < 1e-9, (gid, got, r.cm_ps_o)
        checked += 1
    assert checked == len(sub)


def test_corpus_widening_bounded():
    """D101 corpus fix vs the registered vintage: the bridge is a SEASON-
    AGNOSTIC trailing construction, so ingesting older schedules widens it.
    Keep the perturbation visible and bounded — if a future ingest moves this
    materially, the F6 one-shot owner must re-register before the opener."""
    from nbapred.model.october_bridge import OctoberBridge
    sub = _week1_frame()
    gids = tuple(g.zfill(10) for g in sub.game_id)
    ab2id, played = _oracle_outs_and_ids(gids)
    vals = {}
    for label, floor in (("reg", REGISTERED_CORPUS_FLOOR), ("now", None)):
        br = OctoberBridge(_truncated_mem_db(floor), SEASON, before=OPENER)
        roster = {}
        for p, (t, _c) in br.contrib.items():
            roster.setdefault(t, set()).add(p)
        vals[label] = {
            g.zfill(10): br.margin(
                ab2id[r.home], ab2id[r.away],
                roster.get(ab2id[r.home], set()) - played.get((g.zfill(10), ab2id[r.home]), set()),
                roster.get(ab2id[r.away], set()) - played.get((g.zfill(10), ab2id[r.away]), set()))
            for g, r in zip(sub.game_id, sub.itertuples())}
    d = [abs(vals["now"][g] - vals["reg"][g]) for g in vals["reg"]]
    n_moved = sum(x > 1e-9 for x in d)
    # TRIPPED BY D152 (2026-08-02), DELIBERATELY LEFT TRIPPED. The historical
    # backfill landed 2018-19, 2017-18, 2016-17, 2012-13, 2011-12 and the
    # 2019-20 bubble, which widens this SEASON-AGNOSTIC construction further:
    # measured 9/53 moved, max 1.2086, mean 0.1165 (bounds below are 8 / 1.0 /
    # 0.10). Per this test's own contract that is an F6 OWNER re-registration,
    # NOT something a later agent should silently widen.
    # THE SHIPPED PATH IS UNAFFECTED: production.fit_production builds the
    # bridge with trail_seasons=2 (OCT_BRIDGE_TRAIL default "2", frozen by
    # D105/D122), and at that cap the same measurement is 0/53 moved, max
    # 0.0000 — see test_shipped_bridge_immune_to_corpus_widening below.
    # So: live is bitwise unchanged; only this uncapped diagnostic moved.
    # RESOLVED D155 (F6 re-registration, on the owner's instruction). This test
    # was tripped by D152's backfill and deliberately left tripped, because its
    # own contract says a material move is an F6 OWNER decision, not something
    # an agent silently widens. That decision has now been taken: the F6
    # one-shot ships trail_seasons=2 — already D105's declared primary, already
    # the OCT_BRIDGE_TRAIL default since D122, and already what the D132
    # certification measured. The UNCAPPED construction is therefore retired:
    # nothing in production or in the certified table uses it.
    # So this is no longer a tripwire on a shipped path. It is now the POSITIVE
    # test of why the cap exists — the uncapped construction demonstrably
    # drifts when older seasons land, and the capped one demonstrably does not
    # (test_shipped_bridge_immune_to_corpus_widening, exactly 0.0).
    assert n_moved > 0 and max(d) > 0.5, (
        f"uncapped bridge no longer drifts under corpus widening "
        f"({n_moved}/{len(d)} moved, max {max(d):.4f}) — if this stops being "
        "true the cap may no longer be load-bearing; re-derive before relying "
        "on it. Historical: pre-D152 5/53 max 0.604; post-D152 9/53 max 1.2086.")
    # keep the magnitude visible so a 10x jump is still noticed by a human
    assert max(d) < 5.0, (
        f"uncapped drift max {max(d):.4f} pts is an order of magnitude beyond "
        "anything measured (1.2086 at D152) — investigate the corpus, not this "
        "bound.")


def test_shipped_bridge_immune_to_corpus_widening():
    """The construction that actually SHIPS must not move when older seasons
    are ingested. `fit_production` builds OctoberBridge with trail_seasons=2
    (D105 declared primary, frozen for the F6 one-shot; OCT_BRIDGE_TRAIL="2"),
    and a 2-season trailing window cannot see a season older than the corpus
    the gate table was built on. D152 backfilled 2011-12..2018-19 and the
    2019-20 bubble; at the shipped cap this is EXACTLY zero drift, which is
    what makes that backfill safe to land against a frozen pre-registration.
    """
    from nbapred.model.october_bridge import OctoberBridge
    sub = _week1_frame()
    gids = tuple(g.zfill(10) for g in sub.game_id)
    ab2id, played = _oracle_outs_and_ids(gids)
    vals = {}
    for label, floor in (("reg", REGISTERED_CORPUS_FLOOR), ("now", None)):
        br = OctoberBridge(_truncated_mem_db(floor), SEASON, before=OPENER,
                           trail_seasons=2)
        roster = {}
        for p, (t, _c) in br.contrib.items():
            roster.setdefault(t, set()).add(p)
        vals[label] = {
            g.zfill(10): br.margin(
                ab2id[r.home], ab2id[r.away],
                roster.get(ab2id[r.home], set()) - played.get((g.zfill(10), ab2id[r.home]), set()),
                roster.get(ab2id[r.away], set()) - played.get((g.zfill(10), ab2id[r.away]), set()))
            for g, r in zip(sub.game_id, sub.itertuples())}
    d = [abs(vals["now"][g] - vals["reg"][g]) for g in vals["reg"]]
    assert sum(x > 1e-9 for x in d) == 0, (
        f"SHIPPED bridge moved on {sum(x > 1e-9 for x in d)} of {len(d)} week-1 "
        f"games (max {max(d):.4f}) — the trail_seasons=2 cap is supposed to make "
        "the F6 pre-registration immune to corpus widening")


def test_trail_season_cap_reproduces_registered_construction():
    """D100: the gate table was built when `nba_games` held exactly
    2022-23..2025-26, i.e. THREE prior seasons of trailing minutes. The new
    `trail_seasons` cap makes that vintage reachable from the CURRENT corpus
    without pinning the corpus, so the registered construction stays
    reproducible as the schedule table keeps growing.

    Also pins the measured cap sweep (scripts/cg_bridge_reconfirm.py): a
    1-season cap is far too tight (30 of 53 games move), 3 seasons is exact,
    and D105's declared 2-season primary is a real 5-game / 0.42-pt departure
    that the F6 owner is choosing deliberately — not a drift.
    """
    from nbapred.model.october_bridge import OctoberBridge
    sub = _week1_frame()
    gids = tuple(g.zfill(10) for g in sub.game_id)
    ab2id, played = _oracle_outs_and_ids(gids)
    gate = {r.game_id.zfill(10): float(r.cm_ps_o) for r in sub.itertuples()}
    mem = _truncated_mem_db()                    # FULL current corpus
    moved = {}
    for cap in (1, 2, 3):
        br = OctoberBridge(mem, SEASON, before=OPENER, trail_seasons=cap)
        roster = {}
        for p, (t, _c) in br.contrib.items():
            roster.setdefault(t, set()).add(p)
        d = []
        for r in sub.itertuples():
            g = r.game_id.zfill(10)
            h, a = ab2id[r.home], ab2id[r.away]
            v = br.margin(h, a, roster.get(h, set()) - played.get((g, h), set()),
                          roster.get(a, set()) - played.get((g, a), set()))
            d.append(abs(v - gate[g]))
        moved[cap] = (sum(x > 1e-9 for x in d), max(d))
    assert moved[3][0] == 0 and moved[3][1] < 1e-9, moved[3]   # exact vintage
    assert moved[1][0] > 20, moved[1]                          # 1 season too tight
    assert moved[2][0] <= 8 and moved[2][1] < 1.0, moved[2]     # D105 primary


def test_rotation_empty_and_build_guard():
    """Opening night: every team's comp rotation is empty -> bridge builds and
    activates. Mid-season: no team is missing -> bridge never built, margins
    bitwise-unchanged by construction."""
    from nbapred.db import connect
    from nbapred.model.composition import CompositionModel
    from nbapred.model.october_bridge import (missing_rotation_teams,
                                              rotation_empty)
    mem = _truncated_mem_db()
    comp0 = CompositionModel(mem, before=OPENER)
    sub = _week1_frame()
    ab2id, _ = _oracle_outs_and_ids(
        tuple(g.zfill(10) for g in sub.game_id[:1]))
    missing = missing_rotation_teams(mem, comp0, SEASON, OPENER)
    assert len(missing) == 30, f"expected all 30 teams dead, got {len(missing)}"
    r = sub.iloc[0]
    assert rotation_empty(comp0, ab2id[r.home], ab2id[r.away], OPENER)
    # mid-season view: rotations alive, guard closed
    mid = dt.date(2026, 1, 15)
    con = connect(read_only=True)
    comp1 = CompositionModel(con, before=mid)
    assert missing_rotation_teams(con, comp1, SEASON, mid) == set()
    con.close()
    assert not rotation_empty(comp1, ab2id[r.home], ab2id[r.away], mid)


def test_ps_continuity_proxy_on_opening_night():
    """Refit-1 carry weights: continuity_map at the opener returns the
    preseason-001 continuity proxy (ps_cont_any, corr-0.93) instead of the
    uniform CARRY_CONT_DEFAULT for all 30 teams (the D84-A live gotcha)."""
    import pandas as pd
    from nbapred.model.production import CARRY_CONT_DEFAULT, continuity_map
    sig = pd.read_csv(ROOT / "data/rw_early_signals.csv")
    exp = {int(r.team_id): float(r.ps_cont_any)
           for r in sig[sig.season == SEASON].itertuples()}
    mem = _truncated_mem_db()
    got = continuity_map(mem, SEASON, before=OPENER)
    assert got is not None and len(got) == 30
    n_default = sum(1 for v in got.values()
                    if abs(v - CARRY_CONT_DEFAULT) < 1e-12)
    assert n_default == 0, f"{n_default} teams still on the uniform default"
    for t, v in exp.items():
        assert abs(got[t] - v) < 5e-5, (t, got[t], v)   # csv rounded to 4dp
    # OCT_BRIDGE=0 = F6 same-run control: exact old shipped behavior (uniform
    # default on opening night)
    import os
    os.environ["OCT_BRIDGE"] = "0"
    try:
        off = continuity_map(mem, SEASON, before=OPENER)
    finally:
        del os.environ["OCT_BRIDGE"]
    assert all(abs(v - CARRY_CONT_DEFAULT) < 1e-12 for v in off.values())


def test_continuity_map_unchanged_mid_season():
    """Once every team has observed 002 games, the proxy branch is inert and
    continuity_map reproduces the original first-5-roster construction."""
    from nbapred.db import connect
    from nbapred.model.production import _prev_season, continuity_map
    before = dt.date(2025, 1, 15)
    season = "2024-25"
    con = connect(read_only=True)
    got = continuity_map(con, season, before=before)
    # inline original construction (pre-D84-A code path)
    pm = con.execute("""
        SELECT s.team_id, s.player_id, sum(s.seconds)/60.0 mins
        FROM player_game_stats s
        JOIN nba_games g ON g.game_id = s.game_id AND g.team_id = s.team_id
        WHERE g.season = ? AND s.game_id LIKE '002%' GROUP BY 1, 2""",
        [_prev_season(season)]).fetchall()
    roster = con.execute("""
        WITH tg AS (
          SELECT team_id, game_id,
                 row_number() OVER (PARTITION BY team_id ORDER BY game_date, game_id) rn
          FROM nba_games WHERE season = ? AND game_id LIKE '002%'
          AND game_date < ?)
        SELECT tg.team_id, s.player_id FROM tg
        JOIN player_game_stats s ON s.game_id = tg.game_id AND s.team_id = tg.team_id
        WHERE tg.rn <= 5 GROUP BY 1, 2""", [season, before]).fetchall()
    con.close()
    ros = {}
    for t, p in roster:
        ros.setdefault(int(t), set()).add(int(p))
    assert len(ros) == 30            # every team past its first 5 games
    tot, ret = {}, {}
    for t, p, m in pm:
        t = int(t)
        tot[t] = tot.get(t, 0.0) + m
        if int(p) in ros.get(t, set()):
            ret[t] = ret.get(t, 0.0) + m
    exp = {t: ret.get(t, 0.0) / tot[t] for t in tot if tot[t] > 0}
    assert set(got) == set(exp)
    for t in exp:      # 1e-15 float-order jitter (DuckDB GROUP BY row order —
        # the D63 rerun-jitter class), not a behavior change
        assert abs(got[t] - exp[t]) < 1e-12, (t, got[t], exp[t])
