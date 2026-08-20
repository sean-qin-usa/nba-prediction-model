#!/usr/bin/env python3
"""D100 JOB-3: independent re-confirmation of the D91 October package after the
D101 corpus extension.

Question: `test_bridge_matches_gate_construction` began failing (10.4257 vs the
stored 9.9248). Is the bridge construction still CORRECT and the fixture merely
stale, or did the corpus extension break something?

Findings this script MEASURES (not asserts):

 1. WHICH LEG MOVED. The roster UNION leg is season-PINNED (`WHERE g.season =
    prev_season(season)`), so the corpus extension adds NO prior-season union
    members — contrary to the natural guess. What moved is the TRAILING-MINUTES
    leg, which reaches player_game_stats through an UNQUALIFIED join on
    nba_games and therefore had 2021-22/2020-21 rows silently dropped before
    D101 ingested those schedules. `contrib` membership is gated on `p in
    trail`, so widening that leg ADDS players to the roster.

 2. WHETHER THE ADDED PLAYERS ARE LEGITIMATE. For every player whose
    contribution changed, report the newest 002 game feeding his trailing
    average. A 2025-26 preseason camp body whose last >=12-min NBA game was in
    January 2022 is being projected as a rotation player.

 3. WHETHER A SEASON CAP RESTORES PARITY. D105 declares the trailing leg CAPPED
    AT 2 PRIOR SEASONS as the registered primary, but no cap exists in
    october_bridge.py. Measure parity vs the gate table under caps of 1..5
    prior seasons and uncapped.

 4. D91's OTHER PARITY CLAIMS on the current corpus: Predictor ON-OFF ==
    0.5*cm_ps (registered 3e-13) and mid-season ON-vs-OFF == 0 (registered
    4e-14).

Read-only. Writes data/cg_bridge_reconfirm.json.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import duckdb                                             # noqa: E402

from nbapred.config import DB_PATH                        # noqa: E402
from nbapred.db import connect                            # noqa: E402
from nbapred.model.october_bridge import OctoberBridge    # noqa: E402

SEASON = "2025-26"
OPENER = dt.date(2025, 10, 21)


def mem_db(season_floor=None):
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


def floor_for(n_prior: int) -> str:
    """Season label n_prior seasons before SEASON's start year."""
    y = int(SEASON[:4]) - n_prior
    return f"{y}-{(y + 1) % 100:02d}"


def margins(br, sub, ab2id, played):
    roster = {}
    for p, (t, _c) in br.contrib.items():
        roster.setdefault(t, set()).add(p)
    out = {}
    for r in sub.itertuples():
        gid = r.game_id.zfill(10)
        hid, aid = ab2id[r.home], ab2id[r.away]
        out[gid] = br.margin(hid, aid,
                             roster.get(hid, set()) - played.get((gid, hid), set()),
                             roster.get(aid, set()) - played.get((gid, aid), set()))
    return out


def main():
    import test_october_bridge as t
    sub = t._week1_frame()
    gids = tuple(g.zfill(10) for g in sub.game_id)
    ab2id, played = t._oracle_outs_and_ids(gids)
    gate = {r.game_id.zfill(10): float(r.cm_ps_o) for r in sub.itertuples()}
    res = {"n_week1_games": len(gate)}
    print(f"week-1 gate window: {len(gate)} games\n")

    # ---- 1/3: parity under each corpus vintage / prospective cap ----------
    print("=== parity vs the pre-registered gate table (cm_ps_o) ===")
    print(f"{'arm':<28}{'floor':<10}{'players':>8}{'moved':>7}{'max|d|':>10}{'mean|d|':>10}")
    arms = {}
    contribs = {}
    for label, floor in ([("uncapped (current corpus)", None)]
                         + [(f"cap {n} prior season(s)", floor_for(n)) for n in (1, 2, 3, 4, 5)]):
        br = OctoberBridge(mem_db(floor), SEASON, before=OPENER)
        v = margins(br, sub, ab2id, played)
        d = [abs(v[g] - gate[g]) for g in gate]
        moved = sum(x > 1e-9 for x in d)
        arms[label] = {"season_floor": floor, "players": len(br.contrib),
                       "n_moved": moved, "max_abs": max(d),
                       "mean_abs": sum(d) / len(d),
                       "parity_1e_9": moved == 0}
        contribs[label] = dict(br.contrib)
        print(f"{label:<28}{str(floor):<10}{len(br.contrib):>8}{moved:>7}"
              f"{max(d):>10.2e}{sum(d)/len(d):>10.2e}")
    res["parity_by_corpus_arm"] = arms

    # ---- 2: are the added players legitimate? -----------------------------
    reg = contribs["cap 3 prior season(s)"]     # == the registered vintage floor 2022-23
    now = contribs["uncapped (current corpus)"]
    added = sorted(set(now) - set(reg))
    changed = sorted(p for p in set(now) & set(reg)
                     if abs(now[p][1] - reg[p][1]) > 1e-9)
    con = connect(read_only=True)
    names = dict(con.execute("SELECT player_id, full_name FROM nba_players").fetchall())
    detail = []
    for p in added + changed:
        row = con.execute("""
            SELECT max(g.game_date) FROM player_game_stats s
            JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING(game_id)
            WHERE s.player_id=? AND s.game_id LIKE '002%' AND s.seconds>=720
              AND g.game_date < ?""", [p, OPENER]).fetchone()
        detail.append({"player_id": int(p), "name": names.get(p, "?"),
                       "status": "ADDED" if p in added else "changed",
                       "newest_qualifying_002": str(row[0]) if row and row[0] else None,
                       "contrib_registered": round(reg.get(p, (None, 0.0))[1], 4),
                       "contrib_now": round(now[p][1], 4)})
    con.close()
    detail.sort(key=lambda r: -abs(r["contrib_now"] - r["contrib_registered"]))
    res["players_added"] = len(added)
    res["players_changed"] = len(changed)
    res["player_detail"] = detail
    stale = [d for d in detail if d["status"] == "ADDED"
             and d["newest_qualifying_002"] and d["newest_qualifying_002"] < "2023-01-01"]
    res["added_with_pre_2023_last_game"] = len(stale)
    print(f"\n=== roster deltas vs the registered vintage ({len(added)} added, "
          f"{len(changed)} changed) ===")
    for d in detail[:12]:
        print(f"  {d['status']:<8}{d['player_id']:<9}{d['name']:<24}"
              f"last >=12min 002 {str(d['newest_qualifying_002']):<12}"
              f"contrib {d['contrib_registered']:+.4f} -> {d['contrib_now']:+.4f}")
    print(f"\n  ADDED players whose last qualifying game predates 2023: "
          f"{len(stale)}/{len(added)}")

    # ---- 4: D91's other parity claims on the current corpus ---------------
    print("\n=== D91 (b)/(c): Predictor ON-OFF parity on the CURRENT corpus ===")
    import os
    from nbapred.model.production import fit_production
    r = sub.iloc[0]
    hid, aid = ab2id[r.home], ab2id[r.away]
    mem = mem_db(None)
    probes = {}
    # fit_production is PIT through `before`, so the real (read-only) DB is the
    # faithful source here — mem_db only carries the three bridge tables.
    src = connect(read_only=True)
    for tag, when in (("opening_night", OPENER), ("mid_season", dt.date(2026, 1, 15))):
        vals = {}
        for sw in ("1", "0"):
            os.environ["OCT_BRIDGE"] = sw
            try:
                pr = fit_production(src, SEASON, before=when)
                vals[sw] = pr.margin(hid, aid, set(), set())
            finally:
                os.environ.pop("OCT_BRIDGE", None)
        diff = vals["1"] - vals["0"]
        probes[tag] = {"on": vals["1"], "off": vals["0"], "on_minus_off": diff}
        print(f"  {tag:<15} ON {vals['1']:+.6f}  OFF {vals['0']:+.6f}  "
              f"ON-OFF {diff:+.3e}")
    # ISOLATED bridge probe. OCT_BRIDGE toggles the WHOLE D84-A package (bridge
    # cm_ps AND the ps-continuity carry weights), so the raw ON-OFF above mixes
    # two legs. D91's "(ON-OFF) == 0.5*cm_ps to 3e-13" is a claim about the
    # BRIDGE leg alone -> hold OCT_BRIDGE=1 in both arms and neutralise only
    # OctoberBridge.margin.
    import nbapred.model.october_bridge as obmod
    orig = obmod.OctoberBridge.margin
    os.environ["OCT_BRIDGE"] = "1"
    try:
        pr_on = fit_production(src, SEASON, before=OPENER)
        m_on = pr_on.margin(hid, aid, set(), set())
        obmod.OctoberBridge.margin = lambda self, h, a, oh=None, oa=None: 0.0
        pr_off = fit_production(src, SEASON, before=OPENER)
        m_off = pr_off.margin(hid, aid, set(), set())
    finally:
        obmod.OctoberBridge.margin = orig
        os.environ.pop("OCT_BRIDGE", None)
    src.close()
    # ON-OFF should equal 0.5 * cm_ps on opening night (comp leg is 50% of blend)
    br = OctoberBridge(mem, SEASON, before=OPENER)
    roster = {}
    for p, (tm, _c) in br.contrib.items():
        roster.setdefault(tm, set()).add(p)
    cm_ps = br.margin(hid, aid, set(), set())
    err_pkg = abs(probes["opening_night"]["on_minus_off"] - 0.5 * cm_ps)
    err_iso = abs((m_on - m_off) - 0.5 * cm_ps)
    probes["opening_night"].update({
        "cm_ps": cm_ps, "abs_err_vs_half_cm_ps_PACKAGE_toggle": err_pkg,
        "bridge_isolated_on": m_on, "bridge_isolated_off": m_off,
        "abs_err_vs_half_cm_ps_BRIDGE_isolated": err_iso})
    print(f"  opening night: cm_ps {cm_ps:+.6f}")
    print(f"    package toggle (bridge + ps-cont carry): "
          f"|(ON-OFF) - 0.5*cm_ps| = {err_pkg:.3e}  <- carry leg included")
    print(f"    BRIDGE ISOLATED: ON {m_on:+.6f} OFF {m_off:+.6f}  "
          f"|(ON-OFF) - 0.5*cm_ps| = {err_iso:.3e}  (D91 registered 3e-13)")
    print(f"  mid-season ON-OFF |{probes['mid_season']['on_minus_off']:.3e}| "
          f"(D91 registered 4e-14) -> zero-outside-window "
          f"{'HOLDS' if abs(probes['mid_season']['on_minus_off']) < 1e-9 else 'BROKEN'}")
    res["predictor_probes"] = probes

    p = ROOT / "data/cg_bridge_reconfirm.json"
    p.write_text(json.dumps(res, indent=1, default=str))
    print(f"\nwrote {p}\nBRIDGE_RECONFIRM_DONE")


if __name__ == "__main__":
    main()
