"""OF-1 — PER-SHIP TRANSFER TEST (the overfitting audit, D106 corollary).

QUESTION. Our two best NORMALIZED seasons (2024-25 4.1%, 2025-26 7.8% of
market skill above coin-flip still missing) are EXACTLY the seasons this
feature set was developed on; 2022-23 (12.1%) and 2023-24 (13.1%) sit at the
older level. D101 made 2021-22 and 2022-23 scorable for the first time, so we
finally have seasons NO GATE IN THE CAMPAIGN EVER SAW.

DESIGN. Same-run, per-game paired ablation of EVERY shipped campaign term,
measured separately on
    DEV  = {2024-25, 2025-26}   (the two development seasons)
    HELD = {2021-22, 2022-23}   (quasi-holdout: never scorable during the
                                 campaign, so no gate could select on them)
    GATE = {2023-24}            (in every pooled gate, but not a "best" season
                                 — reported separately, it is NOT held out)

ARMS (one walk-forward pass, all arms share every heavy object; the BASE arm
is the LITERAL production predictor, refit at the same cadence, so the
replica is anchored bitwise at every refit):
    base       full production
    no_sched   D46 schedule layer OFF -> pre-D46 control: home advantage
               reverts to comp's hardcoded HOME_EDGE=3.0 + the FF home
               coefficient (ff.margin, not margin_neutral), no b2b terms
    no_carry   D62 cross-season carry OFF (FF refit with carry_rows=None)
    no_tank    D73 April tank term OFF (k*tank_diff -> 0)
    no_late    D90 late-state layer OFF (mirrors LATE_STATE=0)
    no_bridge  D91 October package OFF (bridge + ps-continuity carry proxy;
               mirrors OCT_BRIDGE=0)
    no_prior   D16 cold-start prior OFF (last_season_prior -> {})

Effect of a term = ll(ablated) - ll(base) per game; POSITIVE = the term HELPS.
Paired bootstrap 2000x, seed 20260801, within each season group.

ACTIVATION RATES are reported per term per season: a term that is structurally
inactive on the holdout (tanking.py / latestate.py carry a hardcoded
`season >= '2022-23'` corpus-floor literal; 001 preseason boxscores only exist
from 2022-23) is UNTESTABLE there, not "non-transferring".

Read-only DB. nbapred/ untouched. Usage: python scripts/of_transfer_ablation.py
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
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from nbapred.db import connect  # noqa: E402
from nbapred.eval.metrics import log_loss  # noqa: E402
from nbapred.model.composition import HOME_EDGE, CompositionModel  # noqa: E402
from nbapred.model.four_factors import FourFactors, factor_game_rows  # noqa: E402
from nbapred.model.october_bridge import (OctoberBridge,  # noqa: E402
                                          missing_rotation_teams,
                                          rotation_empty)
from nbapred.model.production import (CARRY_CONT_DEFAULT, CARRY_W0,  # noqa: E402
                                      SCALE, _prev_season, continuity_map,
                                      fit_production, fit_schedule_layer,
                                      last_season_prior, sigmoid)
from nbapred.model.team_ratings import TeamRatings, game_rows  # noqa: E402

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
GROUPS = {"DEV_24_26": ["2024-25", "2025-26"],
          "HELDOUT_21_23": ["2021-22", "2022-23"],
          "HELDOUT_2223_only": ["2022-23"],
          "GATE_2324": ["2023-24"]}
ARMS = ["base", "no_sched", "no_carry", "no_tank", "no_late", "no_bridge",
        "no_prior"]
TERMS = {"no_sched": "D46 schedule layer", "no_carry": "D62 carry",
         "no_tank": "D73 tank", "no_late": "D90 late-state",
         "no_bridge": "D91 October bridge", "no_prior": "D16 cold-start prior"}
W_COMP = 0.7
SEED = 20260801


def paired_bootstrap(delta, n_boot=2000, seed=SEED):
    d = np.asarray(delta, float)
    if len(d) == 0:
        return {"n": 0, "mean": 0.0, "lo": 0.0, "hi": 0.0, "verdict": "EMPTY"}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"n": int(len(d)), "mean": float(d.mean()), "lo": float(lo),
            "hi": float(hi), "verdict": "PASS" if lo > 0 or hi < 0 else "NS"}


# ---------------------------------------------------------------------------
# per-refit component set (verbatim decomposition of production.fit_production)
# ---------------------------------------------------------------------------

class Pieces:
    """Everything fit_production builds, exposed so each arm can be assembled
    from the SAME objects. Constructed once per refit date."""

    def __init__(self, con, season, before):
        self.comp = CompositionModel(con, before=before)
        # --- D91 October package (bridge + ps-continuity carry proxy) -------
        self.bridge = None
        if missing_rotation_teams(con, self.comp, season, before):
            self.bridge = OctoberBridge(con, season, before=before)
        # --- D62 carry, with and without the D91 ps-cont proxy --------------
        os.environ["OCT_BRIDGE"] = "1"
        cont_on = continuity_map(con, season, before=before)
        os.environ["OCT_BRIDGE"] = "0"
        cont_off = continuity_map(con, season, before=before)
        os.environ["OCT_BRIDGE"] = "1"
        prev_rows = (factor_game_rows(con, _prev_season(season), before=None)
                     if cont_on is not None else None)

        def _ff(cont, carry):
            rows = w = None
            if carry and cont is not None and prev_rows:
                rows = prev_rows
                w = [CARRY_W0 * cont.get(x["tid"], CARRY_CONT_DEFAULT)
                     for x in prev_rows]
            return FourFactors().fit(con, season, before=before,
                                     carry_rows=rows, carry_weights=w)

        self.ff = _ff(cont_on, True)                       # production
        self.ff_nocarry = _ff(cont_on, False)              # D62 off
        same_cont = (cont_on == cont_off)
        self.ff_nobridge = self.ff if same_cont else _ff(cont_off, True)
        self.cont_same = same_cont
        # --- D46 schedule layer --------------------------------------------
        (self.he, self.b_hb2b, self.b_ab2b,
         self.b_hdead, self.b_adead) = fit_schedule_layer(con, before)
        # --- ratings + D16 cold-start prior ---------------------------------
        self.tr = TeamRatings(ridge=25.0).fit(
            game_rows(con, before=before, season=season))
        self.prior = last_season_prior(con, season)
        ab = dict(con.execute("SELECT DISTINCT team_id, team_abbrev FROM "
                              "nba_games WHERE season=?", [season]).fetchall())
        self.id2ab = {t: a for t, a in ab.items()}
        gp_clause = "AND game_date < ?" if before else ""
        gp_params = [season] + ([before] if before else [])
        self.games_played = dict(con.execute(
            f"""SELECT team_id, count(*) FROM nba_games WHERE season=?
                AND game_id LIKE '002%' AND wl IS NOT NULL {gp_clause}
                GROUP BY 1""", gp_params).fetchall())
        # --- D73 tank / D90 late-state (process-cached heavy builds) --------
        from nbapred.model.tanking import get_tank_model
        self.tank = get_tank_model(con)
        self.tank_k = self.tank.fit_k(before)
        from nbapred.model.latestate import get_latestate_model
        self.late = get_latestate_model(con)

    # -- ratings leg ------------------------------------------------------
    def ratings_margin(self, h, a, use_prior=True):
        m = self.tr.pred_margin(h, a)
        if not use_prior:
            return m
        wh = max(0.0, 1 - self.games_played.get(h, 0) / 20.0)
        wa = max(0.0, 1 - self.games_played.get(a, 0) / 20.0)
        return (m + wh * self.prior.get(self.id2ab.get(h, ""), 0.0)
                - wa * self.prior.get(self.id2ab.get(a, ""), 0.0))

    # -- all arms for one game -------------------------------------------
    def arms(self, h, a, out_h, out_a, gd, b2b_h, b2b_a):
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
        m = {}
        m["base"] = core + sched + tk + lt
        m["no_tank"] = core + sched + lt
        m["no_late"] = core + sched + tk
        m["no_prior"] = blend(self.ff, cm_b, False, rm_n) + sched + tk + lt
        m["no_carry"] = blend(self.ff_nocarry, cm_b, False, rm_p) + sched + tk + lt
        m["no_bridge"] = blend(self.ff_nobridge, cm0, False, rm_p) + sched + tk + lt
        # D46 OFF = the pre-D46 control: home advantage from comp's hardcoded
        # HOME_EDGE and the FF home coefficient; no explicit schedule layer.
        m["no_sched"] = blend(self.ff, cm_b + HOME_EDGE, True, rm_p) + tk + lt
        act = {"tank": float(self.tank.diff(h, a, gd)), "late": float(lt),
               "bridge": int(rot_empty),
               "carry_ff_ready_delta": int(self.ff.ready != self.ff_nocarry.ready),
               "prior_active": int(not self.ff.ready)}
        return m, act


# ---------------------------------------------------------------------------
# walk-forward (VERBATIM prod_by_season / ds_rt1_capstone conventions)
# ---------------------------------------------------------------------------

def season_run(con, season, anchor):
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

    rows, P, last, anchor_diff = [], None, None, 0.0
    for gid in order:
        recs = by[gid]
        if len(recs) != 2:
            continue
        mm = recs[0].matchup
        host = mm.split("@")[-1].strip() if "@" in mm else mm.split("vs.")[0].strip()
        h = next((x for x in recs if x.team_abbrev == host), None)
        a = next((x for x in recs if x.team_abbrev != host), None)
        if not h or not a:
            continue
        gd = h.game_date.date() if hasattr(h.game_date, "date") else h.game_date
        if last is None or (gd - last).days >= 7:
            P = Pieces(con, season, gd)
            prod = fit_production(con, season, before=gd, w_comp=W_COMP) if anchor else None
            last = gd
        pmv = mkt.get((str(gd)[:10], h.team_abbrev, a.team_abbrev))
        if pmv is None:
            continue
        outs = {}
        for t in (h.team_id, a.team_id):
            pl = played.get((gid, t), set())
            outs[t] = {p for p, d0 in P.comp.players.items()
                       if d0["team_id"] == t and (gd - d0["last_played"]).days <= 12
                       and p not in pl}
        bh, ba = b2b(h.team_id, gd), b2b(a.team_id, gd)
        m, act = P.arms(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id],
                        gd, bh, ba)
        if anchor:
            pp = prod.p_home(h.team_id, a.team_id, outs[h.team_id],
                             outs[a.team_id], gd, b2b_home=bh, b2b_away=ba)
            anchor_diff = max(anchor_diff,
                              abs(pp - float(sigmoid(m["base"] / SCALE))))
        r = {"season": season, "game_id": gid, "game_date": str(gd)[:10],
             "home": h.team_abbrev, "away": a.team_abbrev,
             "y": int(h.wl == "W"), "p_mkt": float(pmv),
             "gp_home": sum(1 for d in gp_hist.get(h.team_id, []) if d < gd),
             "gp_away": sum(1 for d in gp_hist.get(a.team_id, []) if d < gd),
             **{f"p_{k}": float(sigmoid(v / SCALE)) for k, v in m.items()},
             **{f"act_{k}": v for k, v in act.items()}}
        rows.append(r)
    return rows, anchor_diff


def ll_pg(r, key):
    p = min(max(r[key], 1e-12), 1 - 1e-12)
    return -(r["y"] * np.log(p) + (1 - r["y"]) * np.log(1 - p))


def main():
    anchor = os.environ.get("OF_ANCHOR", "1") != "0"
    con = connect(read_only=True)
    allrows, anchors = [], {}
    for s in SEASONS:
        t0 = time.time()
        rr, ad = season_run(con, s, anchor)
        allrows += rr
        anchors[s] = ad
        print(f"{s}: n={len(rr)} anchor_max|dp|={ad:.2e} ({time.time()-t0:.0f}s)",
              flush=True)
    con.close()

    hdr = list(allrows[0].keys())
    with open(ROOT / "data" / "of_transfer_pergame.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerows(allrows)

    res = {"design": "same-run per-term ablation; base arm anchored to the "
                     "literal fit_production predictor at every refit",
           "anchor_max_abs_dp": anchors, "seed": SEED,
           "seasons": {}, "groups": {}, "activation": {}}

    # --- per-season headline + normalized skill-gap ------------------------
    for s in SEASONS:
        rs = [r for r in allrows if r["season"] == s]
        y = np.array([r["y"] for r in rs])
        ll_us = log_loss(y, np.array([r["p_base"] for r in rs]))
        ll_mk = log_loss(y, np.array([r["p_mkt"] for r in rs]))
        res["seasons"][s] = {
            "n": len(rs), "ll_base": round(float(ll_us), 5),
            "ll_mkt": round(float(ll_mk), 5),
            "gap": round(float(ll_us - ll_mk), 5),
            "mkt_skill": round(float(0.6931471805599453 - ll_mk), 5),
            "norm_gap_pct": round(100 * float(ll_us - ll_mk)
                                  / float(0.6931471805599453 - ll_mk), 2)}
        res["activation"][s] = {
            "tank_nonzero": float(np.mean([r["act_tank"] != 0 for r in rs])),
            "late_nonzero": float(np.mean([r["act_late"] != 0 for r in rs])),
            "bridge_fired": int(sum(r["act_bridge"] for r in rs)),
            "prior_active_fallback": int(sum(r["act_prior_active"] for r in rs)),
            "carry_flips_ff_ready": int(sum(r["act_carry_ff_ready_delta"] for r in rs)),
            "tank_k_span": None}

    # --- per-term effects, per season and per group ------------------------
    for arm, name in TERMS.items():
        entry = {"term": name, "per_season": {}, "groups": {}}
        for s in SEASONS:
            rs = [r for r in allrows if r["season"] == s]
            d = np.array([ll_pg(r, f"p_{arm}") - ll_pg(r, "p_base") for r in rs])
            entry["per_season"][s] = paired_bootstrap(d)
            entry["per_season"][s]["n_moved"] = int(
                sum(abs(r[f"p_{arm}"] - r["p_base"]) > 1e-12 for r in rs))
        for g, ss in GROUPS.items():
            rs = [r for r in allrows if r["season"] in ss]
            d = np.array([ll_pg(r, f"p_{arm}") - ll_pg(r, "p_base") for r in rs])
            bs = paired_bootstrap(d)
            bs["n_moved"] = int(sum(abs(r[f"p_{arm}"] - r["p_base"]) > 1e-12
                                    for r in rs))
            # restricted to games the term actually moves (power-preserving)
            mv = np.array([i for i, r in enumerate(rs)
                           if abs(r[f"p_{arm}"] - r["p_base"]) > 1e-12])
            bs["active_only"] = (paired_bootstrap(d[mv]) if len(mv) else None)
            entry["groups"][g] = bs
        res.setdefault("terms", {})[arm] = entry
        print(f"\n{name}", flush=True)
        for g in GROUPS:
            b = entry["groups"][g]
            print(f"  {g:20s} {b['mean']:+.5f} CI({b['lo']:+.5f},{b['hi']:+.5f})"
                  f" n={b['n']} moved={b['n_moved']} {b['verdict']}", flush=True)

    json.dump(res, open(ROOT / "data" / "of_transfer_results.json", "w"), indent=1)
    print("\nwrote data/of_transfer_results.json", flush=True)


if __name__ == "__main__":
    main()
