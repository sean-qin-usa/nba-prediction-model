"""BA-PORTFOLIO — pre-registered ONE-SHOT joint candidate (Task A, D44-D63 era).

Candidate = CURRENT SHIPPED production margin (D63: carry-shipped fit_production,
including the D62 continuity carry) PLUS, simultaneously:

  (1) DEFONLY 3PT-luck  — defense-only luck removal in the FourFactors ridge
      (construction verbatim from scripts/exp_ffluck2.py "defonly" arm):
      offense credit + mu/home from RAW ridges, efg DEFENSE credit from a ridge
      whose target replaces 3PM with league-avg% x 3PA; factor->ortg map W refit
      on hybrid predictions vs RAW ortg. Individual prior estimate +0.00041 NS.
  (2) EVENT RECENCY     — post-trade/star-return 15-game window FF blend
      (construction verbatim from scripts/exp_eventrecency.py: events = trade
      arrival >=25 min/g or star return >=30 min/g after >=15 days; per-factor
      off/def override w = k/(k+12)). Individual prior estimate +0.00138 NS
      (isolation gate; +0.00237 vs baseline CSV).
  (3) COMP-HEAVY 60/40  — blend 0.6*composition + 0.4*FF instead of 50/50
      (hairline-pass PROVISIONAL in the D46-era re-gates [34f412]).
  (4) DEAD-TEAM FE      — production's fit_schedule_layer already fits
      b_hdead/b_adead walk-forward (wpct-control de-confounded); simply PASS
      the dead flags at predict time (the capstone loop's dead() helper).
      Individual prior estimate +0.00038 NS (capstone_pergame_dead.csv).

SELECTION-BIAS HONESTY: these 4 were chosen BECAUSE they measured positive.
Gate bar therefore: paired bootstrap (2000x, 95% CI) vs the SAME-RUN control
must exclude zero AND the pooled point estimate must exceed +0.0015 (half the
naive sum ~+0.003). We report the naive sum of the individual prior estimates
vs the realized joint delta — the shortfall measures overlap/selection
inflation. Report-only decomposition arms (p_6040/p_dead/p_defonly/p_evrec,
each = control + ONE piece) are computed in the same pass for the overlap
autopsy; they play NO role in the ship decision.

CONTROL: same-run replication of shipped production via fit_production imports
(p_ctl = model.p_home with b2b flags, dead flags NOT passed — exactly the
scripts/prod_by_season.py capstone path). Cross-checked against
data/capstone_pergame_carry2.csv / _carry.csv: measured max |dp| ~1e-14 in
ALL THREE seasons (the 2022-23 backfill was evidently already complete when
those CSVs were written; the D63 "incomplete 548/1230" caveat is stale).

RULES HONORED: DuckDB read_only=True; nothing in nbapred/ or existing scripts
edited; PIT strict (events used only via completed games strictly before the
predicted date; all fits use before-cutoffs); market p is NEVER a model input —
p_mkt joins games (as in every capstone) and DEFINES the heavy-fav analysis
subset only.

Run:  python scripts/ba_portfolio.py            (full 3-season walk-forward)
      python scripts/ba_portfolio.py --analyze-only   (re-run gate analysis)
"""
import sys, os, json, csv, warnings
import datetime as _dt
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from nbapred.db import connect
from nbapred.eval.metrics import log_loss
from nbapred.model.production import (SCALE, sigmoid, fit_production,
                                      DEAD_WPCT, DEAD_GP, CARRY_W0,
                                      CARRY_CONT_DEFAULT, continuity_map,
                                      _prev_season)
from nbapred.model.four_factors import FACTORS, factor_game_rows
from nbapred.model.team_ratings import TeamRatings
from nbapred.market.windows import COACH_CHANGES

# ---- constants (all inherited from the source experiments; nothing tuned) ---
W_FF_JOINT = 0.4          # piece (3): 0.6*comp + 0.4*ff
K0 = 12.0                 # piece (2) ramp w = k/(k+K0)   (theory-set, exp_eventrecency)
WINDOW_GAMES = 15
TRADE_MIN = 25.0
STAR_MIN = 30.0
RETURN_DAYS = 15
MIN_PRIOR_G = 5
RIDGE = 25.0
GATE_MIN_POINT = 0.0015   # pre-registered: half the naive sum of individual estimates
HEAVY = 0.35              # D61 intersection: |p_mkt-.5|>.35 and |p_ctl-.5|<=.35

SEASONS = ("2023-24", "2024-25", "2025-26")
OUT_CSV = ROOT / "data" / "ba_portfolio_pergame.csv"
OUT_JSON = ROOT / "data" / "ba_portfolio_results.json"
BASE_CSVS = [ROOT / "data" / "capstone_pergame_carry2.csv",
             ROOT / "data" / "capstone_pergame_carry.csv"]

ARMS = ["ctl", "joint", "6040", "dead", "defonly", "evrec"]

# prior individual estimates (see docstring) for the naive-sum report
PRIOR_INDIV = {"defonly": 0.00041, "evrec_isolation": 0.00138,
               "6040": None, "dead": 0.00038}   # 6040: hairline-pass, magnitude unrecorded


# ---- closure extraction: the SAME fitted components production used ---------

def _freevars(bound):
    fn = bound.__func__
    return dict(zip(fn.__code__.co_freevars,
                    (c.cell_contents for c in fn.__closure__)))


def extract_parts(model):
    fv = _freevars(model.margin)
    fv2 = _freevars(model.ratings_margin)
    need = {"comp", "ff", "he", "b_hb2b", "b_ab2b", "b_hdead", "b_adead",
            "tr", "w_comp"}
    assert need <= set(fv), f"closure drift: {sorted(need - set(fv))}"
    return dict(comp=fv["comp"], ff=fv["ff"], he=fv["he"],
                b_hb2b=fv["b_hb2b"], b_ab2b=fv["b_ab2b"],
                b_hdead=fv["b_hdead"], b_adead=fv["b_adead"], tr=fv["tr"],
                w_comp=fv["w_comp"], games_played=fv2["games_played"],
                prior=fv2["prior"], id2ab=fv2["id2ab"])


# ---- piece (1): defonly hybrid FF on top of the production (carry) fit ------

class HybridFF:
    """Offense credit + mu/home from the production RAW ridges; efg DEFENSE
    credit from the luck-adjusted ridge. Supports event overrides (piece 2)."""

    def __init__(self, fms_off, fms_def, lg3p):
        self.fms_off, self.fms_def, self.lg3p = fms_off, fms_def, lg3p
        self.W = None

    def pred(self, f, tid, oid, is_home, ov=None):
        o, d = self.fms_off[f], self.fms_def[f]
        off = de = None
        if ov:
            off = ov.get(tid, {}).get(f, (None, None))[0]
            de = ov.get(oid, {}).get(f, (None, None))[1]
        if off is None:
            off = o.off.get(tid, 0.0)
        if de is None:
            de = d.deff.get(oid, 0.0)
        return o.mu + off - de + (o.home if is_home else 0.0)

    def eortg(self, tid, oid, is_home, ov=None):
        xf = np.array([self.pred(f, tid, oid, is_home, ov) for f in FACTORS])
        return float(xf @ self.W[:4] + self.W[4])

    def margin_neutral(self, home_id, away_id, ov=None):
        return (self.eortg(home_id, away_id, False, ov)
                - self.eortg(away_id, home_id, False, ov))


_PREV_ROWS_CACHE = {}


def fit_joint_ff(con, season, before, ff_prod, check_w=False):
    """Rebuild the production FF row set (current + D62 carry, identical
    construction) and fit the defonly hybrid on it. Reuses ff_prod.fms as the
    RAW ridges — guaranteed identical to the control's offense/def credit for
    the 3 untouched factors. Returns (HybridFF | None, w_check_dev)."""
    if not ff_prod.ready:
        return None, None
    cur = factor_game_rows(con, season, before)
    rows, w = cur, None
    if len(cur) < 200:                      # carry-active regime (D62 hard stop)
        cont = continuity_map(con, season, before=before)
        prev = _PREV_ROWS_CACHE.get(season)
        if prev is None:
            prev = factor_game_rows(con, _prev_season(season), before=None)
            _PREV_ROWS_CACHE[season] = prev
        if cont is None or not prev:
            return None, None               # production would not be ready either
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
    y = np.array([x["ortg"] for x in rows])          # RAW ortg target (defonly)
    A = np.c_[X, np.ones(len(X))]
    if w is not None:
        sw = np.sqrt(w)
        hy.W = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
    else:
        hy.W = np.linalg.lstsq(A, y, rcond=None)[0]
    wdev = None
    if check_w:
        # row-set reconstruction fidelity: refit the CONTROL W on our rows and
        # compare to production's ff.W (order-invariant; float noise only)
        Xc = np.array([[ff_prod.fms[f].pred_ortg(x["tid"], x["oid"], x["home"])
                        for f in FACTORS] for x in rows])
        Ac = np.c_[Xc, np.ones(len(Xc))]
        if w is not None:
            Wc = np.linalg.lstsq(Ac * sw[:, None], y * sw, rcond=None)[0]
        else:
            Wc = np.linalg.lstsq(Ac, y, rcond=None)[0]
        wdev = float(np.abs(Wc - ff_prod.W).max())
    return hy, wdev


# ---- piece (2): event detection + overrides (verbatim exp_eventrecency) -----

def player_logs(con, season):
    return con.execute("""
        SELECT s.player_id, s.team_id, g.game_date, s.seconds/60.0 AS mins
        FROM player_game_stats s
        JOIN nba_games g ON g.game_id = s.game_id AND g.team_id = s.team_id
        WHERE g.season = ? AND s.game_id LIKE '002%' AND s.seconds > 0
        ORDER BY g.game_date""", [season]).fetchall()


def detect_events(logs):
    byp = {}
    for pid, tid, d, m in logs:
        d = d.date() if hasattr(d, "date") else d
        byp.setdefault(int(pid), []).append((d, int(tid), float(m)))
    events, details = {}, []
    for pid, gs in byp.items():
        gs.sort()
        seen_teams = set()
        for i, (d, tid, m) in enumerate(gs):
            prior = gs[max(0, i - 10):i]
            avg = np.mean([x[2] for x in prior]) if prior else 0.0
            npr = len(prior)
            if tid not in seen_teams and seen_teams and npr >= MIN_PRIOR_G \
                    and avg >= TRADE_MIN:
                events.setdefault((tid, d), set()).add("trade")
                details.append(dict(kind="trade", player_id=pid, team=tid,
                                    date=str(d)))
            if i > 0 and (d - gs[i - 1][0]).days >= RETURN_DAYS \
                    and npr >= MIN_PRIOR_G and avg >= STAR_MIN:
                events.setdefault((tid, d), set()).add("return")
                details.append(dict(kind="return", player_id=pid, team=tid,
                                    date=str(d)))
            seen_teams.add(tid)
    assert not COACH_CHANGES, "registry populated; add coach events"
    ev_by_team = {}
    for (tid, d) in events:
        ev_by_team.setdefault(tid, []).append(d)
    for tid in ev_by_team:
        ev_by_team[tid].sort()
    return ev_by_team, details


class EventState:
    """Per-season event windows + full-season factor rows; PIT enforced by
    explicit date < gd filters at every use (exp_eventrecency construction)."""

    def __init__(self, con, season):
        self.events, self.details = detect_events(player_logs(con, season))
        rows = factor_game_rows(con, season)
        self.off_rows, self.def_rows = {}, {}
        for r in rows:
            self.off_rows.setdefault(r["tid"], []).append(r)
            self.def_rows.setdefault(r["oid"], []).append(r)
        for d in (self.off_rows, self.def_rows):
            for t in d:
                d[t].sort(key=lambda r: r["date"])

    def overrides(self, tid, gd, fms_off, fms_def, lg3p=None):
        """{tid: {factor: (off_blend, def_blend)}} or {}. lg3p not None =>
        defense-credit estimates strip 3P conversion luck (defonly-consistent);
        offense estimates always use RAW values (offense keeps its 3P skill)."""
        evs = [e for e in self.events.get(tid, []) if e < gd]
        if not evs:
            return {}, 0.0, 0
        e = evs[-1]
        post_o = [r for r in self.off_rows.get(tid, []) if e <= r["date"] < gd]
        post_d = [r for r in self.def_rows.get(tid, []) if e <= r["date"] < gd]
        k = len(post_o)
        if not (1 <= k <= WINDOW_GAMES - 1):
            return {}, 0.0, k
        w = k / (k + K0)

        def dval(r, f):                      # defense-credit observed value
            if f == "efg" and lg3p is not None:
                return r["efg"] + 0.5 * (lg3p * r["thra"] - r["thrm"]) / r["fga"]
            return r[f]

        ov = {}
        for f in FACTORS:
            o, d = fms_off[f], fms_def[f]
            off_est = float(np.mean([
                100.0 * r[f] - o.mu + d.deff.get(r["oid"], 0.0)
                - (o.home if r["home"] else 0.0) for r in post_o]))
            if post_d:
                def_est = float(np.mean([
                    o.mu + o.off.get(r["tid"], 0.0)
                    + (o.home if r["home"] else 0.0) - 100.0 * dval(r, f)
                    for r in post_d]))
                de = (1 - w) * d.deff.get(tid, 0.0) + w * def_est
            else:
                de = None
            ov[f] = ((1 - w) * o.off.get(tid, 0.0) + w * off_est, de)
        return {tid: ov}, w, k


class CtlFFView:
    """Event-override margin on the CONTROL FourFactors fit (for the report-
    only p_evrec arm) — exp_eventrecency's EventFourFactors, as a wrapper."""

    def __init__(self, ff):
        self.ff = ff

    def margin_neutral(self, home_id, away_id, ov=None):
        return (self._eortg(home_id, away_id, False, ov)
                - self._eortg(away_id, home_id, False, ov))

    def _eortg(self, tid, oid, is_home, ov):
        vals = []
        for f in FACTORS:
            m = self.ff.fms[f]
            off = de = None
            if ov:
                off = ov.get(tid, {}).get(f, (None, None))[0]
                de = ov.get(oid, {}).get(f, (None, None))[1]
            if off is None:
                off = m.off.get(tid, 0.0)
            if de is None:
                de = m.deff.get(oid, 0.0)
            vals.append(m.mu + off - de + (m.home if is_home else 0.0))
        xf = np.array(vals)
        return float(xf @ self.ff.W[:4] + self.ff.W[4])


# ---- walk-forward capstone loop (copied from scripts/prod_by_season.py) -----

def season_run(season):
    con = connect(read_only=True)
    ev = EventState(con, season)
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
    if os.environ.get("BA_SMOKE"):           # dev harness check only
        order = order[:int(os.environ["BA_SMOKE"])]
    tdates = {}
    for x in meta.itertuples():
        d = x.game_date.date() if hasattr(x.game_date, "date") else x.game_date
        tdates.setdefault(x.team_id, set()).add(d)

    def b2b(tid, d):
        return (d - _dt.timedelta(days=1)) in tdates.get(tid, set())

    # dead-team flags (PIT: standings from games already played this season) —
    # the prod_by_season dead() helper, now actually PASSED to the joint arm
    hist = {}
    for x in meta.itertuples():
        d = x.game_date.date() if hasattr(x.game_date, "date") else x.game_date
        hist.setdefault(x.team_id, []).append((d, x.wl == "W"))
    for t in hist:
        hist[t].sort()

    def past(tid, d):
        return [w for (dd, w) in hist.get(tid, []) if dd < d]

    def dead(tid, d):
        p = past(tid, d)
        return len(p) >= DEAD_GP and (sum(p) / len(p)) < DEAD_WPCT

    rows_out = []
    model = parts = jff = None
    wdev0 = None
    last = None
    ctl_fid = 0.0
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
            parts = extract_parts(model)
            jff, wdev = fit_joint_ff(con, season, gd, parts["ff"],
                                     check_w=(last is None))
            if wdev is not None:
                wdev0 = wdev
            last = gd
        pmv = mkt.get((str(gd)[:10], h.team_abbrev, a.team_abbrev))
        if pmv is None:
            continue
        comp = parts["comp"]
        outs = {}
        for t in (h.team_id, a.team_id):
            pl = played.get((gid, t), set())
            outs[t] = {p for p, d0 in comp.players.items()
                       if d0["team_id"] == t and (gd - d0["last_played"]).days <= 12
                       and p not in pl}
        yv = int(h.wl == "W")
        b2h, b2a = b2b(h.team_id, gd), b2b(a.team_id, gd)
        dh, da = dead(h.team_id, gd), dead(a.team_id, gd)
        # CONTROL: shipped production exactly as the capstone calls it
        p_ctl = model.p_home(h.team_id, a.team_id, outs[h.team_id],
                             outs[a.team_id], gd, b2b_home=b2h, b2b_away=b2a)
        # shared pieces from the SAME fitted objects
        sched_nd = (parts["he"] + (parts["b_hb2b"] if b2h else 0.0)
                    + (parts["b_ab2b"] if b2a else 0.0))
        sched_d = (sched_nd + (parts["b_hdead"] if dh else 0.0)
                   + (parts["b_adead"] if da else 0.0))
        cm = comp.margin(h.team_id, a.team_id, outs[h.team_id],
                         outs[a.team_id], gd, home_edge=0.0)
        ff = parts["ff"]
        wj_h = wj_a = 0.0
        if ff.ready and jff is not None:
            fm_ctl = ff.margin_neutral(h.team_id, a.team_id)
            # events on the control FF (report-only arm)
            ovh_c, _, _ = ev.overrides(h.team_id, gd, ff.fms, ff.fms)
            ova_c, _, _ = ev.overrides(a.team_id, gd, ff.fms, ff.fms)
            ov_c = {**ovh_c, **ova_c}
            cview = CtlFFView(ff)
            fm_ctl_ev = cview.margin_neutral(h.team_id, a.team_id, ov_c) if ov_c else fm_ctl
            # defonly hybrid, no events (report-only arm)
            fm_def = jff.margin_neutral(h.team_id, a.team_id)
            # JOINT: defonly hybrid + events on it
            ovh_j, wj_h, _ = ev.overrides(h.team_id, gd, jff.fms_off,
                                          jff.fms_def, jff.lg3p)
            ova_j, wj_a, _ = ev.overrides(a.team_id, gd, jff.fms_off,
                                          jff.fms_def, jff.lg3p)
            ov_j = {**ovh_j, **ova_j}
            fm_joint = jff.margin_neutral(h.team_id, a.team_id, ov_j) if ov_j else fm_def
            m_ctl_man = 0.5 * fm_ctl + 0.5 * cm + sched_nd
            marg = {"joint": W_FF_JOINT * fm_joint + (1 - W_FF_JOINT) * cm + sched_d,
                    "6040": W_FF_JOINT * fm_ctl + (1 - W_FF_JOINT) * cm + sched_nd,
                    "dead": 0.5 * fm_ctl + 0.5 * cm + sched_d,
                    "defonly": 0.5 * fm_def + 0.5 * cm + sched_nd,
                    "evrec": 0.5 * fm_ctl_ev + 0.5 * cm + sched_nd}
        else:
            # production fallback (ratings + cold-start prior, w_comp=0.7);
            # only the dead flags can differ from control here
            tr = parts["tr"]
            gh = parts["games_played"].get(h.team_id, 0)
            ga = parts["games_played"].get(a.team_id, 0)
            wh = max(0.0, 1 - gh / 20.0)
            wa = max(0.0, 1 - ga / 20.0)
            rm = (tr.pred_margin(h.team_id, a.team_id)
                  + wh * parts["prior"].get(parts["id2ab"].get(h.team_id, ""), 0.0)
                  - wa * parts["prior"].get(parts["id2ab"].get(a.team_id, ""), 0.0)
                  ) - tr.home
            base = parts["w_comp"] * cm + (1 - parts["w_comp"]) * rm
            m_ctl_man = base + sched_nd
            marg = {"joint": base + sched_d, "6040": base + sched_nd,
                    "dead": base + sched_d, "defonly": base + sched_nd,
                    "evrec": base + sched_nd}
        ctl_fid = max(ctl_fid, abs(float(sigmoid(m_ctl_man / SCALE)) - p_ctl))
        ps = {k: float(sigmoid(v / SCALE)) for k, v in marg.items()}
        gp_h, gp_a = len(past(h.team_id, gd)), len(past(a.team_id, gd))
        rows_out.append([season, gid, str(gd)[:10], h.team_abbrev, a.team_abbrev,
                         yv, p_ctl, ps["joint"], ps["6040"], ps["dead"],
                         ps["defonly"], ps["evrec"], float(pmv),
                         int(dh), int(da), gp_h, gp_a,
                         round(wj_h, 4), round(wj_a, 4),
                         len(outs[h.team_id]), len(outs[a.team_id])])
    con.close()
    n_ev = sum(len(v) for v in ev.events.values())
    print(f"[{season}] games={len(rows_out)} events={n_ev} "
          f"ctl_fidelity_max_dp={ctl_fid:.2e} W_reconstruct_dev={wdev0} "
          f"dead_flag_games={sum(1 for r in rows_out if r[13] or r[14])} "
          f"ev_active={sum(1 for r in rows_out if r[17] > 0 or r[18] > 0)}",
          flush=True)
    return rows_out, {"ctl_fidelity_max_dp": ctl_fid, "w_reconstruct_dev": wdev0,
                      "n_events": n_ev}


# ---- gate analysis ----------------------------------------------------------

def pg_ll(y, p, eps=1e-15):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot(delta, n_boot=2000, seed=7):
    delta = np.asarray(delta, float)
    if len(delta) == 0:
        return dict(n=0, mean=None, lo=None, hi=None, verdict="NS")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    means = delta[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    verdict = "PASS" if lo > 0 else ("FAIL" if hi < 0 else "NS")
    return dict(n=int(len(delta)), mean=float(delta.mean()), lo=float(lo),
                hi=float(hi), verdict=verdict)


def analyze(run_diag):
    import pandas as pd
    df = pd.read_csv(OUT_CSV, dtype={"game_id": str})
    ll = {a: pg_ll(df.y, df[f"p_{a}"]) for a in ARMS}
    ll["mkt"] = pg_ll(df.y, df.p_mkt)
    d_joint = ll["ctl"] - ll["joint"]          # positive = joint better

    res = {"config": dict(w_ff_joint=W_FF_JOINT, K0=K0, window=WINDOW_GAMES,
                          gate="CI excludes 0 AND mean > +0.0015 (pre-registered)",
                          heavy_def=f"|p_mkt-.5|>{HEAVY} & |p_ctl-.5|<={HEAVY}"),
           "run_diagnostics": run_diag}

    # cross-check control vs the shipped carry capstone CSVs
    xchk = {}
    for path in BASE_CSVS:
        if not path.exists():
            continue
        b = pd.read_csv(path, dtype={"game_id": str})
        j = b.merge(df, on=["season", "game_id"], suffixes=("_b", ""))
        per = {s: float((g.p_ctl - g.p_us).abs().max())
               for s, g in j.groupby("season")}
        xchk[path.name] = {"joined": int(len(j)), "max_dp_per_season": per}
    res["control_crosscheck"] = xchk
    res["note_crosscheck"] = ("measured: control matches carry/carry2 CSVs at "
                              "~1e-14 in ALL seasons (2022-23 backfill was "
                              "already complete when they were written); "
                              "same-run control is the gate baseline.")

    # headline log-losses
    res["logloss"] = {}
    for s, g in df.groupby("season"):
        res["logloss"][s] = {a: round(float(pg_ll(g.y, g[f"p_{a}"]).mean()), 4)
                             for a in ("ctl", "joint")}
        res["logloss"][s]["mkt"] = round(float(pg_ll(g.y, g.p_mkt).mean()), 4)
        res["logloss"][s]["n"] = int(len(g))

    # ---- THE pre-registered gate: joint vs same-run control ----
    seasons = {s: boot(d_joint[(df.season == s).values]) for s in SEASONS}
    pooled = boot(d_joint)
    early = ((df.gp_home < 20) | (df.gp_away < 20)).values
    heavy = ((np.abs(df.p_mkt - 0.5) > HEAVY)
             & (np.abs(df.p_ctl - 0.5) <= HEAVY)).values
    gate_pass = bool(pooled["lo"] > 0 and pooled["mean"] > GATE_MIN_POINT)
    res["gate"] = {
        "pooled": pooled,
        "point_bar": GATE_MIN_POINT,
        "SHIP": gate_pass,
        "per_season": seasons,
        "early_either_gp_lt20": boot(d_joint[early]),
        "mkt_heavy_not_us": boot(d_joint[heavy]),
        "early_n": int(early.sum()), "heavy_n": int(heavy.sum()),
        "per_season_subsets": {
            s: {"early": boot(d_joint[early & (df.season == s).values]),
                "heavy": boot(d_joint[heavy & (df.season == s).values])}
            for s in SEASONS},
    }

    # ---- overlap autopsy: report-only single-piece arms ----
    res["decomposition_report_only"] = {}
    realized = {}
    for a in ("6040", "dead", "defonly", "evrec"):
        d = ll["ctl"] - ll[a]
        realized[a] = float(d.mean())
        res["decomposition_report_only"][a] = boot(d)
    res["overlap"] = {
        "prior_individual_estimates": PRIOR_INDIV,
        "prior_naive_sum_approx": "~+0.003 (0.00041+0.00138+0.00038 + 60/40 "
                                  "hairline ~0.0008; magnitude of 60/40 regate "
                                  "not recorded)",
        "realized_individual_same_run": realized,
        "realized_individual_sum": float(sum(realized.values())),
        "realized_joint": pooled["mean"],
        "shortfall_joint_minus_sum": float(pooled["mean"] - sum(realized.values())),
    }
    json.dump(res, open(OUT_JSON, "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


def main():
    run_diag = {}
    if "--analyze-only" not in sys.argv:
        header = ["season", "game_id", "game_date", "home", "away", "y",
                  "p_ctl", "p_joint", "p_6040", "p_dead", "p_defonly",
                  "p_evrec", "p_mkt", "dead_home", "dead_away", "gp_home",
                  "gp_away", "w_ev_home", "w_ev_away", "n_out_home",
                  "n_out_away"]
        all_rows = []
        for s in SEASONS:
            rows, diag = season_run(s)
            all_rows += rows
            run_diag[s] = diag
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(all_rows)
    elif OUT_JSON.exists():
        run_diag = json.load(open(OUT_JSON)).get("run_diagnostics", {})
    analyze(run_diag)


if __name__ == "__main__":
    main()
