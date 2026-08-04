"""SCRATCH — NS-PORTFOLIO FROZEN BUNDLE, joint gate on the 2021-22 + 2022-23
HOLDOUT (plus all five seasons for the LOSO stability diagnostic).

READ-ONLY on data/nba.duckdb (read_only=True, 60s retry).  Nothing in nbapred/
is imported for anything other than reading the shipped model; no production
file is edited.

Membership is FROZEN by data/nsport_prereg.md (sha256 recorded there) BEFORE
this script is run against any holdout number.  Do not edit the member list.

Members (see prereg for the objective rule and the exclusions):
  M1 DEFONLY   3P-luck defense-only hybrid FourFactors (exp_ffluck2 "defonly",
               construction lifted verbatim from scripts/ba_portfolio.py)
  M2 BLEND6040 0.6*composition + 0.4*FF instead of the shipped 0.5/0.5
  M3 URGENCY   D130 ARM A: margin += k_u*(u_h - u_a), k_u walk-forward,
               UrgencyModel imported VERBATIM from scripts/pg_urgency2.py
  M4 TRAVEL    D136 ARM A: jointly-refit schedule layer with dtrav_kkm,
               applied as a delta vs the shipped layer (tv_gate.py convention)

CONTROL = same-run replication of the shipped D132 production path
(prod_by_season.py construction: dead flags NOT passed, oracle OUT sets,
weekly refits), cross-checked per game against data/capstone_pergame.csv
(D134 control-hash rule).

Arms written per game: p_ctl, p_joint, and the four single-member marginals.
"""
import sys
import os
import csv
import json
import time
import datetime as _dt
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from nbapred.model.production import (SCALE, sigmoid, fit_production,
                                      CARRY_W0, CARRY_CONT_DEFAULT,
                                      continuity_map, _prev_season,
                                      fit_schedule_layer_ext)
from nbapred.model.four_factors import FACTORS, factor_game_rows
from nbapred.model.team_ratings import TeamRatings
from nbapred.model.composition import CompositionModel
from nbapred.model.travel import build_state, ARM_TERMS, term_value
from nbapred.model.october_bridge import rotation_empty as _rot_empty

# UrgencyModel verbatim from the D130 gate script (no re-implementation)
sys.path.insert(0, str(ROOT / "scripts"))
from pg_urgency2 import UrgencyModel, GP_ACTIVE            # noqa: E402

SEASONS = tuple(os.environ.get("NSPORT_SEASONS",
                "2021-22,2022-23,2023-24,2024-25,2025-26").split(","))
HOLDOUT = ("2021-22", "2022-23")
DEV = ("2023-24", "2024-25", "2025-26")

W_FF_JOINT = 0.4          # M2: 0.6*comp + 0.4*ff  (ba_portfolio W_FF_JOINT)
RIDGE = 25.0
ROSTER_DAYS = 12
TRAVEL_ARM = "A"

OUT_CSV = ROOT / "data" / "nsport_joint_pergame.csv"
OUT_JSON = ROOT / "data" / "nsport_joint_results.json"
CAPSTONE = ROOT / "data" / "capstone_pergame.csv"

ARMS = ["ctl", "joint", "defonly", "blend6040", "urgency", "travel"]


def connect_retry(attempts=120, wait_s=60):
    from nbapred.db import connect as _c
    last = None
    for _ in range(attempts):
        try:
            return _c(read_only=True)
        except Exception as e:                                  # pragma: no cover
            last = e
            print(f"DB connect failed ({e}); retry in {wait_s}s", flush=True)
            time.sleep(wait_s)
    raise last


def freevars(bound):
    fn = bound.__func__
    return dict(zip(fn.__code__.co_freevars,
                    (c.cell_contents for c in fn.__closure__)))


# ---------------- M1: defense-only 3P-luck hybrid FourFactors ----------------
class HybridFF:
    """Offense credit + mu/home from the production RAW ridges; efg DEFENSE
    credit from the luck-adjusted ridge.  Verbatim ba_portfolio.HybridFF."""

    def __init__(self, fms_off, fms_def, lg3p):
        self.fms_off, self.fms_def, self.lg3p = fms_off, fms_def, lg3p
        self.W = None

    def pred(self, f, tid, oid, is_home):
        o, d = self.fms_off[f], self.fms_def[f]
        return (o.mu + o.off.get(tid, 0.0) - d.deff.get(oid, 0.0)
                + (o.home if is_home else 0.0))

    def eortg(self, tid, oid, is_home):
        xf = np.array([self.pred(f, tid, oid, is_home) for f in FACTORS])
        return float(xf @ self.W[:4] + self.W[4])

    def margin_neutral(self, home_id, away_id):
        return (self.eortg(home_id, away_id, False)
                - self.eortg(away_id, home_id, False))


_PREV_ROWS_CACHE = {}


def fit_hybrid_ff(con, season, before, ff_prod, check_w=False):
    """Rebuild production's EXACT FF row set (current + D62 carry, same 200-row
    hard stop as four_factors.fit) and fit the defonly hybrid on it."""
    cur = factor_game_rows(con, season, before)
    rows, w = cur, None
    if len(cur) < 200:
        cont = continuity_map(con, season, before=before)
        prev = _PREV_ROWS_CACHE.get(season)
        if prev is None:
            prev = factor_game_rows(con, _prev_season(season), before=None)
            _PREV_ROWS_CACHE[season] = prev
        if cont is None or not prev:
            return None, None
        cw = [CARRY_W0 * cont.get(x["tid"], CARRY_CONT_DEFAULT) for x in prev]
        rows = list(prev) + cur
        w = np.array(cw + [1.0] * len(cur), float)
    lg3p = sum(x["thrm"] for x in rows) / max(sum(x["thra"] for x in rows), 1)
    efg_lg = TeamRatings(ridge=RIDGE, team_home_ridge=None).fit(
        [(x["tid"], x["oid"], x["home"],
          (x["efg"] + 0.5 * (lg3p * x["thra"] - x["thrm"]) / x["fga"]) * 100)
         for x in rows], weights=w)
    hy = HybridFF(ff_prod.fms, {**ff_prod.fms, "efg": efg_lg}, lg3p)
    X = np.array([[hy.pred(f, x["tid"], x["oid"], x["home"]) for f in FACTORS]
                  for x in rows])
    y = np.array([x["ortg"] for x in rows])
    A = np.c_[X, np.ones(len(X))]
    if w is not None:
        sw = np.sqrt(w)
        hy.W = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
    else:
        hy.W = np.linalg.lstsq(A, y, rcond=None)[0]
    wdev = None
    if check_w:                       # row-set reconstruction fidelity check
        Xc = np.array([[ff_prod.fms[f].pred_ortg(x["tid"], x["oid"], x["home"])
                        for f in FACTORS] for x in rows])
        Ac = np.c_[Xc, np.ones(len(Xc))]
        if w is not None:
            Wc = np.linalg.lstsq(Ac * sw[:, None], y * sw, rcond=None)[0]
        else:
            Wc = np.linalg.lstsq(Ac, y, rcond=None)[0]
        wdev = float(np.abs(Wc - ff_prod.W).max())
    return hy, wdev


# ------------------------------- season loop ---------------------------------
def season_run(season, urg, tstate, diag):
    t0 = time.time()
    con = connect_retry()
    pm = con.execute("""SELECT game_id, team_id, player_id, seconds/60.0 AS mins
        FROM player_game_stats WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    played = {(g, t): set(grp.player_id)
              for (g, t), grp in pm.groupby(["game_id", "team_id"])}
    meta = con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl, game_date
        FROM nba_games WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL
        ORDER BY game_date""", [season]).fetchdf()
    mkt = {(str(r[0])[:10], r[1], r[2]): r[3] for r in con.execute(
        "SELECT game_date, home, away, p_home_spread FROM odds_market WHERE season_end=?",
        [int(season[:4]) + 1]).fetchall()}
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

    rows_out = []
    model = comp = parts = hyb = None
    b5A = exA = None
    k_u, n_act = 0.0, 0
    last, wdev0 = None, None
    ctl_fid = 0.0
    n_refit = 0
    for gid in order:
        recs = by[gid]
        if len(recs) != 2:
            continue
        m = recs[0].matchup
        host = m.split("@")[-1].strip() if "@" in m else m.split("vs.")[0].strip()
        h = next((x for x in recs if x.team_abbrev == host), None)
        a = next((x for x in recs if x.team_abbrev != host), None)
        if not h or not a:
            continue
        gd = h.game_date.date() if hasattr(h.game_date, "date") else h.game_date
        if last is None or (gd - last).days >= 7:
            model = fit_production(con, season, before=gd, w_comp=0.7)
            comp = CompositionModel(con, before=gd)
            parts = freevars(model.margin)
            hyb, wdev = fit_hybrid_ff(con, season, gd, parts["ff"],
                                      check_w=(last is None))
            if wdev is not None:
                wdev0 = wdev
            b5A, exA = fit_schedule_layer_ext(con, gd, arms=(TRAVEL_ARM,),
                                              state=tstate)
            k_u, n_act = urg.fit_k(gd)
            last = gd
            n_refit += 1
        pmv = mkt.get((str(gd)[:10], h.team_abbrev, a.team_abbrev))
        if pmv is None:
            continue
        outs = {}
        for t in (h.team_id, a.team_id):
            pl = played.get((gid, t), set())
            outs[t] = {p for p, d0 in comp.players.items()
                       if d0["team_id"] == t
                       and (gd - d0["last_played"]).days <= ROSTER_DAYS
                       and p not in pl}
        yv = int(h.wl == "W")
        bh, ba = b2b(h.team_id, gd), b2b(a.team_id, gd)

        # ---- CONTROL: shipped production exactly as the capstone calls it ----
        p_ctl = model.p_home(h.team_id, a.team_id, outs[h.team_id],
                             outs[a.team_id], gd, b2b_home=bh, b2b_away=ba)

        # ---- shared pieces from the SAME fitted objects ----
        he, b_hb2b, b_ab2b = parts["he"], parts["b_hb2b"], parts["b_ab2b"]
        sched_ship = he + (b_hb2b if bh else 0.0) + (b_ab2b if ba else 0.0)
        cm = comp.margin(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id],
                         gd, home_edge=0.0)
        if parts["bridge"] is not None and _rot_empty(comp, h.team_id,
                                                      a.team_id, gd):
            cm = parts["bridge"].margin(h.team_id, a.team_id,
                                        outs[h.team_id], outs[a.team_id])
        ff = parts["ff"]
        fm_ctl = ff.margin_neutral(h.team_id, a.team_id)
        fm_def = hyb.margin_neutral(h.team_id, a.team_id) if hyb else fm_ctl
        tk = parts["tank_k"] * model.tank_diff(h.team_id, a.team_id, gd)

        # ---- M3 urgency (D130 ARM A) ----
        u_h, u_a = urg.u(h.team_id, gd), urg.u(a.team_id, gd)
        d_urg = k_u * (u_h - u_a)

        # ---- M4 travel (D136 ARM A), applied as a schedule-layer DELTA ----
        d_trav = 0.0
        sh_st, sa_st = tstate.get((h.team_id, gd)), tstate.get((a.team_id, gd))
        if sh_st is not None and sa_st is not None and b5A is not None:
            s_arm = b5A[0] + (b5A[1] if bh else 0.0) + (b5A[2] if ba else 0.0)
            for c, fn in ARM_TERMS[TRAVEL_ARM]:
                s_arm += exA[c] * term_value(c, fn, sh_st, sa_st)
            d_trav = s_arm - sched_ship

        # ---- manual control reconstruction (fidelity check) ----
        m_ctl_man = 0.5 * fm_ctl + 0.5 * cm + sched_ship + tk
        ctl_fid = max(ctl_fid, abs(float(sigmoid(m_ctl_man / SCALE)) - p_ctl))

        base_no_ff = sched_ship + tk
        marg = {
            "joint":     W_FF_JOINT * fm_def + (1 - W_FF_JOINT) * cm
                         + base_no_ff + d_urg + d_trav,
            "defonly":   0.5 * fm_def + 0.5 * cm + base_no_ff,
            "blend6040": W_FF_JOINT * fm_ctl + (1 - W_FF_JOINT) * cm + base_no_ff,
            "urgency":   0.5 * fm_ctl + 0.5 * cm + base_no_ff + d_urg,
            "travel":    0.5 * fm_ctl + 0.5 * cm + base_no_ff + d_trav,
        }
        ps = {k: float(sigmoid(v / SCALE)) for k, v in marg.items()}
        rows_out.append([season, gid, str(gd)[:10], h.team_abbrev, a.team_abbrev,
                         yv, p_ctl, ps["joint"], ps["defonly"], ps["blend6040"],
                         ps["urgency"], ps["travel"], float(pmv),
                         round(d_urg, 6), round(d_trav, 6), round(k_u, 5),
                         urg.gp(h.team_id, gd), urg.gp(a.team_id, gd),
                         urg.alive(h.team_id, gd), urg.alive(a.team_id, gd)])
    con.close()
    diag[season] = dict(
        n=len(rows_out), refits=n_refit, ctl_fidelity_max_dp=ctl_fid,
        w_reconstruct_dev=wdev0, k_u_last=k_u, k_u_frame_n=n_act,
        urgency_active_games=sum(1 for r in rows_out if r[13] != 0.0),
        travel_active_games=sum(1 for r in rows_out if r[14] != 0.0),
        secs=round(time.time() - t0, 1))
    print(f"[{season}] {diag[season]}", flush=True)
    return rows_out


# ------------------------------- analysis ------------------------------------
def pg_ll(y, p, eps=1e-15):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot(delta, n_boot=2000, seed=20260801):
    delta = np.asarray(delta, float)
    if len(delta) == 0:
        return dict(n=0, mean=None, lo=None, hi=None, se=None, verdict="NS")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    means = delta[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return dict(n=int(len(delta)), mean=float(delta.mean()), lo=float(lo),
                hi=float(hi), se=float(means.std(ddof=1)),
                p_wrongside=float((means <= 0).mean()),
                verdict="PASS" if lo > 0 else ("FAIL" if hi < 0 else "NS"))


def analyze():
    import pandas as pd
    df = pd.read_csv(OUT_CSV, dtype={"game_id": str})
    ll = {a: pg_ll(df.y, df[f"p_{a}"]) for a in ARMS}
    ll["mkt"] = pg_ll(df.y, df.p_mkt)
    res = {}

    # ---- D134 CONTROL-HASH FIELD ----
    cap = pd.read_csv(CAPSTONE, dtype={"game_id": str})
    j = cap.merge(df, on=["season", "game_id"], suffixes=("_c", ""))
    res["control_hash"] = dict(
        certified_csv="data/capstone_pergame.csv (D132)",
        joined=int(len(j)), certified_rows=int(len(cap)), run_rows=int(len(df)),
        max_abs_dp=float((j.p_us - j.p_ctl).abs().max()),
        frac_moved_gt_1e12=float(((j.p_us - j.p_ctl).abs() > 1e-12).mean()),
        per_season={s: float((g.p_us - g.p_ctl).abs().max())
                    for s, g in j.groupby("season")})

    res["logloss"] = {}
    for s, g in df.groupby("season"):
        res["logloss"][s] = {a: round(float(pg_ll(g.y, g[f"p_{a}"]).mean()), 5)
                             for a in ("ctl", "joint")}
        res["logloss"][s]["mkt"] = round(float(pg_ll(g.y, g.p_mkt).mean()), 5)
        res["logloss"][s]["n"] = int(len(g))

    d = {a: (ll["ctl"] - ll[a]) for a in ARMS if a != "ctl"}
    hold = df.season.isin(HOLDOUT).values
    dev = df.season.isin(DEV).values

    # ---- THE pre-registered PRIMARY endpoint ----
    res["PRIMARY_holdout_joint"] = boot(d["joint"][hold])
    res["secondary"] = dict(
        dev_joint=boot(d["joint"][dev]),
        all5_joint=boot(d["joint"]),
        per_season={s: boot(d["joint"][(df.season == s).values]) for s in SEASONS},
    )
    # per-member marginals on the holdout (pre-registered report-only)
    res["members_holdout"] = {a: boot(d[a][hold]) for a in ARMS if a != "ctl"}
    res["members_dev"] = {a: boot(d[a][dev]) for a in ARMS if a != "ctl"}
    res["members_per_season"] = {
        a: {s: float(d[a][(df.season == s).values].mean()) for s in SEASONS}
        for a in ARMS if a != "ctl"}
    _mh = [res["members_holdout"][a]["mean"] for a in ARMS if a != "ctl"]
    res["additivity_holdout"] = dict(
        sum_of_members=(float(sum(_mh)) if all(v is not None for v in _mh)
                        else None),
        joint=res["PRIMARY_holdout_joint"]["mean"])

    # ---- LOSO stability (coordinator item 2) ----
    loso = {s: boot(d["joint"][(df.season == s).values]) for s in SEASONS}
    vals = np.array([loso[s]["mean"] for s in SEASONS], float)
    PRE, POST = HOLDOUT, DEV
    pre = np.array([loso[s]["mean"] for s in PRE], float)
    post = np.array([loso[s]["mean"] for s in POST], float)
    grand = float(vals.mean())
    ss_b = float(len(pre) * (pre.mean() - grand) ** 2
                 + len(post) * (post.mean() - grand) ** 2)
    ss_w = float(((pre - pre.mean()) ** 2).sum() + ((post - post.mean()) ** 2).sum())
    ms_b, ms_w = ss_b / 1.0, ss_w / 3.0
    res["LOSO"] = dict(
        note="member configs are FROZEN, so 'select on 4 / score 1' reduces "
             "exactly to the per-season held-out estimate. Folds REUSE data "
             "(walk-forward fits are cumulative) => NOT 5 independent "
             "confirmations; this is a stability diagnostic only. Only "
             "2021-22 and 2022-23 are clean of hypothesis selection.",
        folds={s: loso[s] for s in SEASONS},
        mean=grand, sd=float(vals.std(ddof=1)),
        min=float(vals.min()), max=float(vals.max()),
        n_positive=int((vals > 0).sum()),
        era_anova=dict(pre_ppp_mean=float(pre.mean()), post_ppp_mean=float(post.mean()),
                       SS_between=ss_b, SS_within=ss_w, MS_between=ms_b,
                       MS_within=ms_w,
                       F=float(ms_b / ms_w) if ms_w > 0 else float("inf"),
                       between_share_of_SS=float(ss_b / (ss_b + ss_w))
                       if (ss_b + ss_w) > 0 else 0.0))
    # per-member era anova too
    res["LOSO_members"] = {}
    for a in ARMS:
        if a == "ctl":
            continue
        v = np.array([float(d[a][(df.season == s).values].mean()) for s in SEASONS])
        gm = float(v.mean())
        pb = v[:2]
        po = v[2:]
        sb = float(2 * (pb.mean() - gm) ** 2 + 3 * (po.mean() - gm) ** 2)
        sw = float(((pb - pb.mean()) ** 2).sum() + ((po - po.mean()) ** 2).sum())
        res["LOSO_members"][a] = dict(
            per_season=dict(zip(SEASONS, [float(x) for x in v])),
            mean=gm, sd=float(v.std(ddof=1)), n_positive=int((v > 0).sum()),
            pre_ppp_mean=float(pb.mean()), post_ppp_mean=float(po.mean()),
            between_share_of_SS=float(sb / (sb + sw)) if (sb + sw) > 0 else 0.0,
            F=float((sb / 1.0) / (sw / 3.0)) if sw > 0 else float("inf"))
    json.dump(res, open(OUT_JSON, "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


def main():
    if "--analyze-only" not in sys.argv:
        con = connect_retry()
        print("building travel state + urgency model ...", flush=True)
        tstate = build_state(con)
        urg = UrgencyModel(con)
        con.close()
        print(f"travel state {len(tstate)} team-games; urgency map "
              f"{len(urg.map)} team-dates", flush=True)
        diag = {}
        allrows = []
        for s in SEASONS:
            allrows += season_run(s, urg, tstate, diag)
        header = ["season", "game_id", "game_date", "home", "away", "y",
                  "p_ctl", "p_joint", "p_defonly", "p_blend6040", "p_urgency",
                  "p_travel", "p_mkt", "d_urg", "d_trav", "k_u",
                  "gp_h", "gp_a", "alive_h", "alive_a"]
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(allrows)
        json.dump(diag, open(ROOT / "data" / "nsport_joint_diag.json", "w"), indent=1)
    analyze()


if __name__ == "__main__":
    main()
