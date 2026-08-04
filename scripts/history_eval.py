#!/usr/bin/env python3
"""HISTORICAL OUT-OF-SAMPLE EVALUATION (D153) — score the certified stack, and
ablate every shipped sides term, on EVERY scorable season.

WHY THIS IS THE STRONGEST TEST WE HAVE.  The whole production stack was
designed, tuned and gated on 2021-22..2025-26.  D152's backfill made 2010-11,
2012-13, 2015-16..2019-20 (plus the 2011-12 lockout and 2020-21 strata)
scorable for the first time.  No gate in the campaign ever saw them, so the
per-season numbers below are strict out-of-sample in the only sense that
matters: the hypotheses themselves were never selected against this data
(GATE_POLICY_V2 §8.3).

WHAT IT PRODUCES, in ONE walk-forward pass per season:
  * `p_prod` — the LITERAL production predictor, `fit_production(...).p_home`,
    refit weekly, i.e. exactly what `scripts/prod_by_season.py` computes under
    the D132 environment.  This is deliverable (1).
  * `p_base` — the arm-assembly replica of the same predictor, anchored against
    `p_prod` at every game (`anchor_max_abs_dp` is reported per season and must
    be ~1e-14 for the ablation to be about the terms rather than about the
    replica).
  * one arm per SHIPPED sides term, each the same walk-forward pass with that
    term removed.  Effect of a term = ll(ablated) - ll(base), POSITIVE = the
    term HELPS.

DIFFERENCE FROM `scripts/of_transfer_ablation.py`, and why this file exists:
that harness's base arm still adds the D90 late-state term, which has been OFF
in production since D118.  Its anchor is therefore 0.086 on a historical season,
not 1e-14.  Here the base arm is late-state-free (matching the shipped default)
and D90 becomes an `add_late` DIAGNOSTIC arm instead.  Two blend arms are also
added so the D19/D21 composition/four-factors design can be read per era.

ARMS
  base        full production (D132 defaults)
  no_sched    D46 schedule layer OFF -> pre-D46 control (comp's hardcoded
              HOME_EDGE=3.0 + the FF home coefficient, no b2b terms)
  no_carry    D62 cross-season carry OFF (FF refit with carry_rows=None)
  no_tank     D73 April tank term OFF (k*tank_diff -> 0)
  no_bridge   D91 October package OFF (bridge + ps-continuity carry proxy)
  no_prior    D16 cold-start prior OFF
  no_ff       D21 four-factors leg REPLACED by the plain ridge-ratings margin
  no_comp     D19 composition leg REPLACED by the plain ridge-ratings margin
  ratings_only  both legs off (the pre-D19 reference stack)
  add_late    D90 late-state layer ON (diagnostic; reverted at D112)

READ-ONLY on the DB.  Nothing in nbapred/ is modified and no default is flipped.

  python scripts/history_eval.py                 # all scorable seasons
  python scripts/history_eval.py 2016-17 2017-18 # a subset
  HIST_TAG=floor2122 TANK_SEASON_FLOOR=2021-22 python scripts/history_eval.py ...
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nbapred import threads  # noqa: E402
threads.pin(1)

import numpy as np  # noqa: E402

from nbapred.db import connect  # noqa: E402
from nbapred.eval.metrics import log_loss  # noqa: E402
from nbapred.model.composition import HOME_EDGE  # noqa: E402
from nbapred.model.production import SCALE, fit_production, sigmoid  # noqa: E402

from of_transfer_ablation import Pieces, W_COMP  # noqa: E402

ARMS = ["base", "no_sched", "no_carry", "no_tank", "no_bridge", "no_prior",
        "no_ff", "no_comp", "ratings_only", "add_late"]
TERMS = {"no_sched": "D46 schedule layer", "no_carry": "D62 carry",
         "no_tank": "D73 tank", "no_bridge": "D91 October bridge",
         "no_prior": "D16 cold-start prior", "no_ff": "D21 four-factors leg",
         "no_comp": "D19 composition leg",
         "ratings_only": "D19+D21 both legs (pre-D19 reference)",
         "add_late": "D90 late-state (ADD; reverted at D112)"}
TAG = os.environ.get("HIST_TAG", "")


class HPieces(Pieces):
    """Pieces with a production-faithful base (no D90) and the blend arms."""

    def __init__(self, con, season, before):
        super().__init__(con, season, before)
        # of_transfer_ablation builds OctoberBridge with the UNCAPPED trailing
        # window; fit_production builds it with OCT_BRIDGE_TRAIL (default "2",
        # the D105/D122 freeze).  On the post-D152 corpus that difference is
        # worth up to 0.048 in p on October games — D152 §10 predicted exactly
        # this ("the trailing-minutes leg is season-AGNOSTIC and the backfill
        # widened it").  Rebuild the bridge the way production does so the
        # ablation is about the TERMS, not about the replica.
        if self.bridge is not None:
            from nbapred.model.october_bridge import OctoberBridge
            _tr = os.environ.get("OCT_BRIDGE_TRAIL", "2")
            self.bridge = OctoberBridge(con, season, before=before,
                                        trail_seasons=int(_tr) if _tr else None)

    def arms(self, h, a, out_h, out_a, gd, b2b_h, b2b_a):
        from nbapred.model.october_bridge import rotation_empty
        cm0 = self.comp.margin(h, a, out_h, out_a, gd, home_edge=0.0)
        rot_empty = (self.bridge is not None
                     and rotation_empty(self.comp, h, a, gd))
        cm_b = self.bridge.margin(h, a, out_h, out_a) if rot_empty else cm0
        sched = (self.he + (self.b_hb2b if b2b_h else 0.0)
                 + (self.b_ab2b if b2b_a else 0.0))
        tk = self.tank_k * self.tank.diff(h, a, gd)
        lt = self.late.term(h, a, out_h, out_a, gd)
        rm_p = self.ratings_margin(h, a, True)
        rm_n = self.ratings_margin(h, a, False)

        def blend(ff, cm, use_home_ff, rm):
            if ff.ready:
                fm = ff.margin(h, a) if use_home_ff else ff.margin_neutral(h, a)
                return 0.5 * fm + 0.5 * cm
            base_rm = rm if use_home_ff else rm - self.tr.home
            return W_COMP * cm + (1 - W_COMP) * base_rm

        core = blend(self.ff, cm_b, False, rm_p)
        rm_neutral = rm_p - self.tr.home          # ratings leg, home stripped
        m = {}
        # --- shipped stack (LATE_STATE off, the D118/D132 default) ----------
        m["base"] = core + sched + tk
        m["no_tank"] = core + sched
        m["add_late"] = core + sched + tk + lt
        m["no_prior"] = blend(self.ff, cm_b, False, rm_n) + sched + tk
        m["no_carry"] = blend(self.ff_nocarry, cm_b, False, rm_p) + sched + tk
        m["no_bridge"] = blend(self.ff_nobridge, cm0, False, rm_p) + sched + tk
        m["no_sched"] = blend(self.ff, cm_b + HOME_EDGE, True, rm_p) + tk
        # --- D19 / D21 blend legs -------------------------------------------
        if self.ff.ready:
            m["no_ff"] = 0.5 * rm_neutral + 0.5 * cm_b + sched + tk
            m["no_comp"] = (0.5 * self.ff.margin_neutral(h, a)
                            + 0.5 * rm_neutral) + sched + tk
        else:                       # blend already degenerate; arms are no-ops
            m["no_ff"] = core + sched + tk
            m["no_comp"] = core + sched + tk
        m["ratings_only"] = rm_neutral + sched + tk
        act = {"tank": float(self.tank.diff(h, a, gd)), "late": float(lt),
               "bridge": int(rot_empty), "ff_ready": int(self.ff.ready),
               "carry_ff_ready_delta": int(self.ff.ready != self.ff_nocarry.ready),
               "tank_k": float(self.tank_k), "home_edge": float(self.he)}
        return m, act


def season_run(con, season):
    pm = con.execute("""SELECT game_id, team_id, player_id FROM player_game_stats
        WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    played = {(g, t): set(grp.player_id)
              for (g, t), grp in pm.groupby(["game_id", "team_id"])}
    meta = con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl, game_date
        FROM nba_games WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL
        ORDER BY game_date""", [season]).fetchdf()
    mkt = {(str(r[0])[:10], r[1], r[2]): r[3] for r in con.execute(
        "SELECT game_date, home, away, p_home_spread FROM odds_market "
        "WHERE season_end=?", [int(season[:4]) + 1]).fetchall()}
    by, order = {}, []
    for x in meta.itertuples():
        if x.game_id not in by:
            order.append(x.game_id)
        by.setdefault(x.game_id, []).append(x)
    tdates = {}
    for x in meta.itertuples():
        d = x.game_date.date() if hasattr(x.game_date, "date") else x.game_date
        tdates.setdefault(x.team_id, set()).add(d)

    def b2b(tid, d):
        return (d - _dt.timedelta(days=1)) in tdates.get(tid, set())

    gp_hist = {}
    for x in meta.itertuples():
        d = x.game_date.date() if hasattr(x.game_date, "date") else x.game_date
        gp_hist.setdefault(x.team_id, []).append(d)

    rows, P, prod, last, anchor = [], None, None, None, 0.0
    n_sched, n_nomkt = 0, 0
    for gid in order:
        recs = by[gid]
        if len(recs) != 2:
            continue
        n_sched += 1
        mm = recs[0].matchup
        host = mm.split("@")[-1].strip() if "@" in mm else mm.split("vs.")[0].strip()
        h = next((x for x in recs if x.team_abbrev == host), None)
        a = next((x for x in recs if x.team_abbrev != host), None)
        if not h or not a:
            continue
        gd = h.game_date.date() if hasattr(h.game_date, "date") else h.game_date
        if last is None or (gd - last).days >= 7:
            P = HPieces(con, season, gd)
            prod = fit_production(con, season, before=gd, w_comp=W_COMP)
            last = gd
        pmv = mkt.get((str(gd)[:10], h.team_abbrev, a.team_abbrev))
        if pmv is None:
            n_nomkt += 1
            continue
        outs = {}
        for t in (h.team_id, a.team_id):
            pl = played.get((gid, t), set())
            outs[t] = {p for p, d0 in P.comp.players.items()
                       if d0["team_id"] == t
                       and (gd - d0["last_played"]).days <= 12 and p not in pl}
        bh, ba = b2b(h.team_id, gd), b2b(a.team_id, gd)
        m, act = P.arms(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id],
                        gd, bh, ba)
        pp = prod.p_home(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id],
                         gd, b2b_home=bh, b2b_away=ba)
        anchor = max(anchor, abs(pp - float(sigmoid(m["base"] / SCALE))))
        rows.append({"season": season, "game_id": gid, "game_date": str(gd)[:10],
                     "home": h.team_abbrev, "away": a.team_abbrev,
                     "y": int(h.wl == "W"), "p_prod": float(pp),
                     "p_mkt": float(pmv),
                     "gp_home": sum(1 for d in gp_hist.get(h.team_id, []) if d < gd),
                     "gp_away": sum(1 for d in gp_hist.get(a.team_id, []) if d < gd),
                     "n_out_home": len(outs[h.team_id]),
                     "n_out_away": len(outs[a.team_id]),
                     **{f"p_{k}": float(sigmoid(v / SCALE)) for k, v in m.items()},
                     **{f"act_{k}": v for k, v in act.items()}})
    return rows, anchor, n_sched, n_nomkt


LN2 = 0.6931471805599453


def main():
    con = connect(read_only=True)
    from history_scorable import sets
    pool, strat, detail = sets(con)
    want = sys.argv[1:] or (pool + strat)
    from nbapred.model.tanking import season_floor
    tank_floor = season_floor(con)
    print(f"tank season_floor = {tank_floor}", flush=True)
    allrows, meta = [], {}
    for s in want:
        t0 = time.time()
        rr, ad, nsched, nomkt = season_run(con, s)
        allrows += rr
        meta[s] = {"n": len(rr), "n_scheduled": nsched, "n_no_market": nomkt,
                   "anchor_max_abs_dp": ad, "secs": round(time.time() - t0, 1),
                   "poolable": s in pool, "stratum": detail[s]["separate_stratum"]}
        y = np.array([r["y"] for r in rr])
        llu = log_loss(y, np.array([r["p_prod"] for r in rr]))
        llm = log_loss(y, np.array([r["p_mkt"] for r in rr]))
        meta[s].update(ll_us=round(float(llu), 5), ll_mkt=round(float(llm), 5),
                       raw_gap=round(float(llu - llm), 5),
                       norm_gap_pct=round(100 * float(llu - llm) / float(LN2 - llm), 2))
        print(f"{s}: n={len(rr):5d} ll_us={llu:.5f} ll_mkt={llm:.5f} "
              f"raw={llu-llm:+.5f} norm={meta[s]['norm_gap_pct']:+.2f}%  "
              f"anchor={ad:.2e} nomkt={nomkt} ({meta[s]['secs']}s)", flush=True)
    con.close()
    suf = f"_{TAG}" if TAG else ""
    hdr = list(allrows[0].keys())
    with open(ROOT / "data" / f"history_pergame{suf}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerows(allrows)
    json.dump({"tank_season_floor": tank_floor, "tag": TAG, "arms": ARMS,
               "terms": TERMS, "seasons": meta,
               "env": {k: os.environ.get(k) for k in
                       ("LATE_STATE", "TANK_TERM", "ORACLE_MINUTES",
                        "INACTIVE_OUTS", "REPORT_OUTS", "TANK_SEASON_FLOOR",
                        "OCT_BRIDGE", "OCT_BRIDGE_TRAIL", "COVID_GUARD",
                        "FF_LUCK")}},
              open(ROOT / "data" / f"history_by_season{suf}.json", "w"), indent=1)
    print(f"\nwrote data/history_pergame{suf}.csv "
          f"({len(allrows)} games) and data/history_by_season{suf}.json")


if __name__ == "__main__":
    main()
