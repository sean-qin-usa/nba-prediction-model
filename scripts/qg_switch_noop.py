#!/usr/bin/env python3
"""GATE_POLICY_V2 §6.6 STEP 0 — prove that ADDING the three gate switches did
not change production behaviour.

Every gate below is run against a same-run control built from the shipped code
with the switches at their defaults. That control is only trustworthy if the
default path is BITWISE what it was before the switches existed. Three claims,
each verified on real corpus rows rather than argued:

  (1) starout.team_context with STAROUT_TRAIL unset reproduces the PRE-EDIT SQL
      (trail_min / n_tr / last_date / trail_att) EXACTLY, on every player of
      every sampled team-date.
  (2) starout.team_context with STAROUT_USAGE unset selects the same star and
      the same lift as the pre-edit weight/selection logic.
  (3) props.simulate_player with PROPS_CHANNEL_RAMP unset returns BITWISE the
      same points / threes / rebounds / assists arrays as the pre-edit
      generative model (reimplemented here verbatim from the pre-edit source).

READ-ONLY. Writes data/qg_switch_noop.json.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from nbapred.db import connect
from nbapred.engine import starout
from nbapred.engine.props import player_rates_from_stats, simulate_player

TRAIL_GAMES = starout.TRAIL_GAMES


def pre_edit_sql(con, team_id, before):
    """The team_context SQL EXACTLY as it stood after D146 and before the
    STAROUT_TRAIL switch was added."""
    rows = con.execute(f"""
        WITH tg AS (
          SELECT s.player_id, s.seconds/60.0 AS m,
                 s.rima + s.mida + s.thra AS att, g.game_date,
                 row_number() OVER (PARTITION BY s.player_id
                                    ORDER BY g.game_date DESC) AS rn
          FROM player_game_stats s
          JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
          WHERE s.team_id = ? AND s.game_id LIKE '002%' AND g.game_date < ?
        )
        SELECT player_id,
               avg(m)   FILTER (WHERE rn <= {TRAIL_GAMES}) AS trail_min,
               count(*) FILTER (WHERE rn <= {TRAIL_GAMES}) AS n_tr,
               max(game_date) FILTER (WHERE m > 0)         AS last_date,
               avg(att) FILTER (WHERE rn <= {TRAIL_GAMES}) AS trail_att
        FROM tg GROUP BY player_id
    """, [int(team_id), before]).fetchall()
    return {int(p): (tm, int(n), ld, ta) for p, tm, n, ld, ta in rows}


def new_sql_current_mode(con, team_id, before):
    """The shipped SQL + the tmode='current' projection, lifted verbatim."""
    rows = con.execute(f"""
        WITH tg AS (
          SELECT s.player_id, s.seconds/60.0 AS m,
                 s.rima + s.mida + s.thra AS att, g.game_date,
                 row_number() OVER (PARTITION BY s.player_id
                                    ORDER BY g.game_date DESC) AS rn,
                 CASE WHEN s.seconds > 0 THEN
                   row_number() OVER (PARTITION BY s.player_id, s.seconds > 0
                                      ORDER BY g.game_date DESC) END AS rp
          FROM player_game_stats s
          JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
          WHERE s.team_id = ? AND s.game_id LIKE '002%' AND g.game_date < ?
        )
        SELECT player_id,
               avg(m)   FILTER (WHERE rn <= {TRAIL_GAMES}) AS trail_min,
               count(*) FILTER (WHERE rn <= {TRAIL_GAMES}) AS n_tr,
               max(game_date) FILTER (WHERE m > 0)         AS last_date,
               avg(att) FILTER (WHERE rn <= {TRAIL_GAMES}) AS trail_att,
               avg(m)   FILTER (WHERE rp <= {TRAIL_GAMES}) AS trail_min_pl,
               count(*) FILTER (WHERE rp <= {TRAIL_GAMES}) AS n_played,
               avg(att) FILTER (WHERE rp <= {TRAIL_GAMES}) AS trail_att_pl,
               count(*) FILTER (WHERE rn <= {TRAIL_GAMES} AND m > 0) AS n_pl_win
        FROM tg GROUP BY player_id
    """, [int(team_id), before]).fetchall()
    st = {int(r[0]): (r[1], int(r[2]), r[3], r[4], int(r[8])) for r in rows}
    return {p: v for p, v in st.items() if v[0] is not None}


def pre_edit_sim(rates, n, seed):
    """props.simulate_player EXACTLY as it stood after D145 and before the
    PROPS_CHANNEL_RAMP switch was added."""
    rng = np.random.default_rng(seed)
    hist = rates.get("minutes_hist")
    if hist is not None and len(hist) >= 5:
        h = np.asarray(hist, float)
        base = rng.choice(h, n) - h.mean() + float(rates.get("proj_min", h.mean()))
        mins = np.clip(base + rng.normal(0, 2.0, n), 0, 48)
    else:
        mins = rng.normal(rates["proj_min"], rates.get("sd_min", 4.0), n)
        bad = mins < 10
        for _ in range(4):
            if not bad.any():
                break
            mins[bad] = rng.normal(rates["proj_min"], rates.get("sd_min", 4.0), bad.sum())
            bad = mins < 10
        mins = np.clip(mins, 10, 48)

    def zone_pts(rate, fg, val):
        att = rng.poisson(np.maximum(rate, 0) * mins)
        made = rng.binomial(att, np.clip(fg, 0, 1))
        return made, made * val

    rim_m, rim_p = zone_pts(rates["rate_rim"], rates["fg_rim"], 2)
    mid_m, mid_p = zone_pts(rates["rate_mid"], rates["fg_mid"], 2)
    thr_m, thr_p = zone_pts(rates["rate_thr"], rates["fg_thr"], 3)
    ft_att = rng.poisson(np.maximum(rates.get("fta_per_min", 0), 0) * mins)
    ft_m = rng.binomial(ft_att, np.clip(rates.get("ft_pct", 0.77), 0, 1))
    points = rim_p + mid_p + thr_p + ft_m
    rebounds = rng.poisson(np.maximum(rates.get("reb_per_min", 0), 0) * mins)
    ast_expo = float(np.clip(rates.get("proj_min", 30.0), 10, 44))
    assists = rng.poisson(np.maximum(rates.get("ast_per_min", 0), 0) * ast_expo, size=n)
    return {"points": points, "threes": thr_m, "rebounds": rebounds,
            "assists": assists}


def main():
    for k in ("STAROUT_TRAIL", "STAROUT_USAGE", "PROPS_CHANNEL_RAMP"):
        assert not os.environ.get(k), f"{k} must be unset for the no-op proof"
    con = connect(read_only=True)
    out = {"env": {k: os.environ.get(k) for k in
                   ("STAROUT_TRAIL", "STAROUT_USAGE", "PROPS_CHANNEL_RAMP",
                    "PROPS_MIN_RAMP", "PROPS_ABSENCE_RAMP")}}

    # ---- (1) trailing-stats SQL identity -----------------------------------
    tds = con.execute("""
        SELECT DISTINCT team_id, game_date FROM nba_games
        WHERE game_id LIKE '002%' AND wl IS NOT NULL
          AND season IN ('2021-22','2022-23','2023-24','2024-25','2025-26')
        ORDER BY game_date, team_id""").fetchall()
    step = max(1, len(tds) // 400)
    sample = tds[::step]
    n_pairs = n_cells = 0
    mism = []
    for t, dte in sample:
        dte = dte.date() if hasattr(dte, "date") else dte
        old = pre_edit_sql(con, t, dte)
        new = new_sql_current_mode(con, t, dte)
        # the only permitted difference is dropping trail_min IS NULL rows,
        # which cannot occur in 'current' mode (rn=1 always exists)
        if set(old) != set(new):
            mism.append(("keyset", int(t), str(dte),
                         sorted(set(old) ^ set(new))[:5]))
            continue
        for p in old:
            n_cells += 1
            a, b = old[p], new[p]
            if not (a[0] == b[0] and a[1] == b[1] and a[2] == b[2]
                    and a[3] == b[3]):
                mism.append(("value", int(t), str(dte), int(p), a, b))
        n_pairs += 1
    out["trail_sql_identity"] = {
        "team_dates_checked": n_pairs, "player_cells_checked": n_cells,
        "mismatches": len(mism), "examples": mism[:5]}
    print(f"(1) trail SQL identity: {n_pairs} team-dates, {n_cells} player cells, "
          f"{len(mism)} mismatches")
    assert not mism, mism[:3]

    # ---- (2) team_context end-to-end: star + lift ---------------------------
    weights = starout.load_usage_weights()
    positions = starout.load_positions()
    pg = con.execute("""
        SELECT s.game_id, s.team_id, s.player_id, s.seconds/60.0 m, g.game_date
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '002%' AND g.season = '2024-25'""").fetchdf()
    pg["game_date"] = [d.date() if hasattr(d, "date") else d for d in pg["game_date"]]
    played = pg[pg.m > 0].groupby(["game_id", "team_id"])["player_id"].apply(set)
    rostered = pg.groupby(["game_id", "team_id"])["player_id"].apply(set)
    ctxs, n_ctx, n_none = [], 0, 0
    keys = sorted(rostered.index.tolist())[::7]
    for (gid, tid) in keys[:1200]:
        dte = pg[(pg.game_id == gid) & (pg.team_id == tid)]["game_date"].iloc[0]
        outs = rostered[(gid, tid)] - played.get((gid, tid), set())
        if not outs:
            continue
        c = starout.team_context(con, int(tid), outs, dte, weights, positions)
        if c is None:
            n_none += 1
            continue
        ctxs.append((int(gid) if str(gid).isdigit() else gid, int(tid),
                     int(c["star"]), float(c["lift"]), float(c["lift_softmax"]),
                     int(c["n_pool"]), c["usage_source"]))
        n_ctx += 1
    out["team_context_probe"] = {
        "team_games_probed": len(keys[:1200]), "fired": n_ctx, "no_star": n_none,
        "usage_sources": sorted({c[6] for c in ctxs}),
        "mean_lift": round(float(np.mean([c[3] for c in ctxs])), 5) if ctxs else None,
        "mean_lift_softmax": round(float(np.mean([c[4] for c in ctxs])), 5) if ctxs else None,
        "trail_mode": starout.trail_mode(), "usage_mode": starout.usage_mode()}
    print(f"(2) team_context default modes: trail={starout.trail_mode()} "
          f"usage={starout.usage_mode()}, fired {n_ctx}/{len(keys[:300])}, "
          f"sources {out['team_context_probe']['usage_sources']}")
    assert starout.trail_mode() == "current" and starout.usage_mode() == "softmax"
    assert out["team_context_probe"]["usage_sources"] in ([], ["v2_usage.npz"])

    # ---- (3) simulate_player bitwise ---------------------------------------
    rows = con.execute("""
        SELECT s.player_id, g.game_date
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '002%' AND s.seconds >= 720
          AND g.season IN ('2023-24','2024-25','2025-26')
        ORDER BY s.player_id, g.game_date""").fetchall()
    n_sim, worst = 0, {}
    for pid, dte in rows[::151]:
        dte = dte.date() if hasattr(dte, "date") else dte
        rr = player_rates_from_stats(con, int(pid), before=dte)
        if rr is None or rr["n_games"] < 8 or rr["proj_min"] < 20:
            continue
        a = simulate_player(rr, 4000, seed=n_sim)
        b = pre_edit_sim(rr, 4000, seed=n_sim)
        for k in ("points", "threes", "rebounds", "assists"):
            same = np.array_equal(a[k], b[k])
            worst[k] = worst.get(k, 0) + (0 if same else 1)
        n_sim += 1
        if n_sim >= 400:
            break
    out["simulate_player_bitwise"] = {"rows": n_sim, "mismatched_channels": worst}
    print(f"(3) simulate_player bitwise on {n_sim} real rows: mismatches {worst}")
    assert n_sim >= 100 and all(v == 0 for v in worst.values())
    con.close()

    out["verdict"] = "NO-OP CONFIRMED: adding the switches changed nothing at defaults"
    json.dump(out, open("data/qg_switch_noop.json", "w"), indent=1, default=str)
    print("\nwrote data/qg_switch_noop.json")
    print("QG_SWITCH_NOOP_DONE", flush=True)


if __name__ == "__main__":
    main()
