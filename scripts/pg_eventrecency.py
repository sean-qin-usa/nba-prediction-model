"""PRE-REGISTERED RE-GATE of F2 (event-recency window blend) at 4-season power.

F2 (freeze list, DECISIONS.md; D52/D64/D71): post-regime-event FF recency
blend — three consistent positives, last isolation gate +0.00138
CI(-0.00085,+0.00353) NS pooled over 3 seasons (2024-25 individually
PASSED). 2021-22 player_game_stats landed -> 2022-23 is now a fittable
eval season; this is the pre-registered power re-gate.

CONSTRUCTION EXACTLY scripts/exp_eventrecency.py — same event detection
(trade/arrival >=25 min/g, star return >=30 min/g after >=15 days,
within-season trailing averages, MIN_PRIOR_G=5; coach registry empty ->
skipped), same activation window (15 team-games post-event), same blend
ramp w = k/(k+12), same per-factor override math. Changes are corpus +
control only:

  * EVAL SEASONS: + 2022-23 (player_game_stats fully backfilled; market
    se2023 matches 1230/1230). 2021-22 itself is NOT evaluable and does
    NOT enter detection (within-season logs only, per the original spec) —
    nba_games has no 2021-22 rows.
  * CONTROL: SAME-RUN current production (fit_production = D62 carry +
    D73 tank), per the re-gate protocol. The original script paired
    against a hand-copied sched-era production; here p_ctl IS
    fit_production's margin (bitwise, incl. tank + carry) and p_exp
    swaps ONLY the FF term: m_exp = m_ctl + 0.5*(fm_event - fm_ctl) —
    algebraically identical isolation to the original (FF enters the
    ready-branch margin linearly at 0.5). The fitted FourFactors instance
    is read out of the Predictor closure (read-only; nbapred/ untouched).
    Control replication checked vs capstone_pergame_tank.csv (2023-24..
    2025-26; 2022-23 has no shipped capstone baseline).

PRIMARY GATE (pre-registered): paired bootstrap 2000x (seed 7) 95% CI on
per-game log-loss delta p_exp vs p_ctl (isolation, the F2 freeze metric),
pooled over the 4 eval seasons. Also reported: per-season, affected-games
subset, pooled over the prior 3 seasons (power comparison vs
data/exp_eventrecency_summary.json).

SHIP RULE: pooled isolation CI excludes 0 -> SHIP recommendation with
port spec; else stays FROZEN for 2026-27; report power gained (CI width
shrinkage). Read-only DB. New file scripts/pg_eventrecency.py.
"""
import sys, json, csv, time, warnings, datetime as dt
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np


def connect_retry(read_only=True, attempts=120, wait_s=30):
    """Read-only connect with retry (another session's batch loaders take the
    write lock transiently). Harness robustness only — no modeling change."""
    from nbapred.db import connect as _c
    last = None
    for _ in range(attempts):
        try:
            return _c(read_only=read_only)
        except Exception as e:
            last = e
            time.sleep(wait_s)
    raise last

from nbapred.db import connect
from nbapred.model.composition import CompositionModel
from nbapred.model.production import SCALE, fit_production, sigmoid
from nbapred.model.four_factors import FACTORS, factor_game_rows
from nbapred.market.windows import COACH_CHANGES

# ---- experiment constants (VERBATIM exp_eventrecency.py) --------------------
K0 = 12.0            # blend ramp w = k/(k+K0)
WINDOW_GAMES = 15    # team-games after an event that are "affected"
TRADE_MIN = 25.0     # avg min/g qualifying a trade arrival
STAR_MIN = 30.0      # avg min/g qualifying a star return
RETURN_DAYS = 15     # absence length qualifying a return
MIN_PRIOR_G = 5      # min games in the trailing average (stability)

SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")
PRIOR_SEASONS = ("2023-24", "2024-25", "2025-26")
OUT_CSV = REPO / "data" / "pg_eventrecency_pergame.csv"
OUT_JSON = REPO / "data" / "pg_eventrecency_summary.json"
PRIOR_JSON = REPO / "data" / "exp_eventrecency_summary.json"
CAPSTONE = REPO / "data" / "capstone_pergame_tank.csv"


# ---- event detection (VERBATIM exp_eventrecency.py) -------------------------

def player_logs(con, season):
    """Played player-games for the season: (player_id, team_id, date, mins)."""
    return con.execute("""
        SELECT s.player_id, s.team_id, g.game_date, s.seconds/60.0 AS mins
        FROM player_game_stats s
        JOIN nba_games g ON g.game_id = s.game_id AND g.team_id = s.team_id
        WHERE g.season = ? AND s.game_id LIKE '002%' AND s.seconds > 0
        ORDER BY g.game_date""", [season]).fetchall()


def detect_events(logs):
    """{team_id: sorted [event_date]}, plus detail list for reporting.
    Trailing averages are within-season only (original spec, unchanged)."""
    byp = {}
    for pid, tid, d, m in logs:
        d = d.date() if hasattr(d, "date") else d
        byp.setdefault(int(pid), []).append((d, int(tid), float(m)))
    events = {}          # (team, date) -> set(kinds)
    details = []
    for pid, gs in byp.items():
        gs.sort()
        seen_teams = set()
        for i, (d, tid, m) in enumerate(gs):
            prior = gs[max(0, i - 10):i]           # last up-to-10 games before
            avg = np.mean([x[2] for x in prior]) if prior else 0.0
            npr = len(prior)
            # (a) trade/arrival: first game with tid, had games w/ other teams
            if tid not in seen_teams and seen_teams and npr >= MIN_PRIOR_G \
                    and avg >= TRADE_MIN:
                events.setdefault((tid, d), set()).add("trade")
                details.append(dict(kind="trade", player_id=pid, team=tid,
                                    date=str(d), avg_min=round(float(avg), 1)))
            # (b) star return: same-season gap >= RETURN_DAYS
            if i > 0 and (d - gs[i - 1][0]).days >= RETURN_DAYS \
                    and npr >= MIN_PRIOR_G and avg >= STAR_MIN:
                events.setdefault((tid, d), set()).add("return")
                details.append(dict(kind="return", player_id=pid, team=tid,
                                    date=str(d), avg_min=round(float(avg), 1),
                                    days_out=(d - gs[i - 1][0]).days))
            seen_teams.add(tid)
    # (c) coach change — registry empty => skip (spec)
    assert not COACH_CHANGES, "registry populated; add coach events"
    ev_by_team = {}
    for (tid, d) in events:
        ev_by_team.setdefault(tid, []).append(d)
    for tid in ev_by_team:
        ev_by_team[tid].sort()
    return ev_by_team, details


# ---- event-blended FF override math (VERBATIM math, free functions over the
#      production-fitted FourFactors instance) --------------------------------

def _pred_f(ff, f, tid, oid, is_home, ov):
    m = ff.fms[f]
    off = ov.get(tid, {}).get(f, (None, None))[0]
    de = ov.get(oid, {}).get(f, (None, None))[1]
    if off is None:
        off = m.off.get(tid, 0.0)
    if de is None:
        de = m.deff.get(oid, 0.0)
    return m.mu + off - de + (m.home if is_home else 0.0)


def _eortg_ev(ff, tid, oid, is_home, ov):
    xf = np.array([_pred_f(ff, f, tid, oid, is_home, ov) for f in FACTORS])
    return float(xf @ ff.W[:4] + ff.W[4])


def margin_neutral_ev(ff, home_id, away_id, ov):
    return (_eortg_ev(ff, home_id, away_id, False, ov)
            - _eortg_ev(ff, away_id, home_id, False, ov))


class EventState:
    """Per-season event windows + full-season factor rows (PIT enforced by
    explicit date < gd filters at every use). VERBATIM exp_eventrecency.py."""

    def __init__(self, con, season):
        self.events, self.details = detect_events(player_logs(con, season))
        rows = factor_game_rows(con, season)     # full season; date-filtered on use
        self.off_rows, self.def_rows = {}, {}
        for r in rows:
            self.off_rows.setdefault(r["tid"], []).append(r)
            self.def_rows.setdefault(r["oid"], []).append(r)
        for d in (self.off_rows, self.def_rows):
            for t in d:
                d[t].sort(key=lambda r: r["date"])

    def overrides(self, tid, gd, ff):
        """{factor: (off_blend, def_blend)} for team tid predicting on gd, or
        {}. Returns (ov, w, k)."""
        evs = [e for e in self.events.get(tid, []) if e < gd]
        if not evs or not ff.ready:
            return {}, 0.0, 0
        e = evs[-1]                                   # most recent event
        post_o = [r for r in self.off_rows.get(tid, []) if e <= r["date"] < gd]
        post_d = [r for r in self.def_rows.get(tid, []) if e <= r["date"] < gd]
        k = len(post_o)
        # affected window: predicted game is game k+1 since event, need <= 15,
        # and >=1 completed post-event game to estimate from
        if not (1 <= k <= WINDOW_GAMES - 1):
            return {}, 0.0, k
        w = k / (k + K0)
        ov = {}
        for f in FACTORS:
            m = ff.fms[f]
            off_est = float(np.mean([
                100.0 * r[f] - m.mu + m.deff.get(r["oid"], 0.0)
                - (m.home if r["home"] else 0.0) for r in post_o]))
            if post_d:
                def_est = float(np.mean([
                    m.mu + m.off.get(r["tid"], 0.0)
                    + (m.home if r["home"] else 0.0) - 100.0 * r[f]
                    for r in post_d]))
                de = (1 - w) * m.deff.get(tid, 0.0) + w * def_est
            else:
                de = None
            ov[f] = ((1 - w) * m.off.get(tid, 0.0) + w * off_est, de)
        return {tid: ov}, w, k


def _closure_var(bound_method, name):
    """Read a captured variable out of a bound method's closure (read-only:
    lets the harness reach fit_production's fitted FourFactors without
    touching nbapred/)."""
    fn = bound_method.__func__
    return fn.__closure__[fn.__code__.co_freevars.index(name)].cell_contents


# ---- capstone loop: prod_by_season.py VERBATIM (oracle OUT-sets, weekly
#      refit); control = CURRENT fit_production margin; variant swaps FF ------

def season_run(season):
    import time
    t0 = time.time()
    con = connect_retry(read_only=True)
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
    by = {}; order = []
    for x in meta.itertuples():
        if x.game_id not in by:
            order.append(x.game_id)
        by.setdefault(x.game_id, []).append(x)
    tdates = {}
    for x in meta.itertuples():
        d = x.game_date.date() if hasattr(x.game_date, "date") else x.game_date
        tdates.setdefault(x.team_id, set()).add(d)

    def b2b(tid, d):
        return (d - dt.timedelta(days=1)) in tdates.get(tid, set())

    rows = []
    model = comp = ff = None
    last = None
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
            ff = _closure_var(model.margin, "ff")
            last = gd
        pmv = mkt.get((str(gd)[:10], h.team_abbrev, a.team_abbrev))
        if pmv is None:
            continue
        outs = {}
        for t in (h.team_id, a.team_id):
            pl = played.get((gid, t), set())
            outs[t] = {p for p, d0 in comp.players.items()
                       if d0["team_id"] == t and (gd - d0["last_played"]).days <= 12
                       and p not in pl}
        m_ctl = model.margin(h.team_id, a.team_id, outs[h.team_id],
                             outs[a.team_id], gd,
                             b2b_home=b2b(h.team_id, gd),
                             b2b_away=b2b(a.team_id, gd))
        if ff.ready:
            ovh, w_h, k_h = ev.overrides(h.team_id, gd, ff)
            ova, w_a, k_a = ev.overrides(a.team_id, gd, ff)
            ov = {**ovh, **ova}
            if ov:
                fm_ctl = ff.margin_neutral(h.team_id, a.team_id)
                fm_exp = margin_neutral_ev(ff, h.team_id, a.team_id, ov)
                m_exp = m_ctl + 0.5 * (fm_exp - fm_ctl)
            else:
                m_exp = m_ctl
        else:
            w_h = w_a = 0.0; k_h = k_a = 0
            m_exp = m_ctl
        rows.append(dict(season=season, game_id=gid, game_date=str(gd)[:10],
                         home=h.team_abbrev, away=a.team_abbrev,
                         y=int(h.wl == "W"),
                         p_ctl=float(sigmoid(m_ctl / SCALE)),
                         p_exp=float(sigmoid(m_exp / SCALE)),
                         p_mkt=float(pmv), w_home=round(w_h, 4),
                         w_away=round(w_a, 4), k_home=k_h, k_away=k_a))
    con.close()
    n_ev = sum(len(v) for v in ev.events.values())
    print(f"[{season}] games={len(rows)} events={n_ev} "
          f"(trade={sum(1 for d in ev.details if d['kind']=='trade')}, "
          f"return={sum(1 for d in ev.details if d['kind']=='return')}) "
          f"affected={sum(1 for r in rows if r['w_home'] > 0 or r['w_away'] > 0)} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return rows, ev.details


# ---- gates (VERBATIM boot_gate: 2000 resamples, seed 7) ---------------------

def pg_ll(y, p, eps=1e-15):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_gate(delta, n_boot=2000, seed=7):
    """delta per game (positive = experiment improves). Mean + 95% CI."""
    delta = np.asarray(delta, float)
    if len(delta) == 0:
        return dict(n=0, mean=None, lo=None, hi=None, verdict="NS")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    means = delta[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    verdict = ("PASS" if lo > 0 else "FAIL" if hi < 0 else "NS")
    return dict(n=int(len(delta)), mean=float(delta.mean()),
                lo=float(lo), hi=float(hi), verdict=verdict)


def main():
    import pandas as pd
    all_details = {}
    if "--analyze-only" in sys.argv and OUT_CSV.exists():
        df = pd.read_csv(OUT_CSV, dtype={"game_id": str})
    else:
        all_rows = []
        for s in SEASONS:
            rows, details = season_run(s)
            all_rows += rows
            all_details[s] = details
        with open(OUT_CSV, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            wtr.writeheader(); wtr.writerows(all_rows)
        df = pd.DataFrame(all_rows)
    df["game_id"] = df.game_id.astype(str).str.zfill(10)

    # control replication vs the shipped capstone (2023-24..2025-26 only)
    cap = pd.read_csv(CAPSTONE, dtype={"game_id": str})[
        ["season", "game_id", "p_us"]]
    cap["game_id"] = cap.game_id.str.zfill(10)
    mg = df.merge(cap, on=["season", "game_id"], how="inner")
    repl = dict(baseline="capstone_pergame_tank.csv", n_matched=int(len(mg)),
                n_ours=int(len(df)),
                note="2022-23 has no shipped capstone -> same-run control only",
                max_abs_diff=float((mg.p_ctl - mg.p_us).abs().max())
                if len(mg) else None)

    df["ll_ctl"] = pg_ll(df.y, df.p_ctl)
    df["ll_exp"] = pg_ll(df.y, df.p_exp)
    df["ll_mkt"] = pg_ll(df.y, df.p_mkt)
    df["aff"] = (df.w_home > 0) | (df.w_away > 0)
    d_iso = df.ll_ctl - df.ll_exp

    res = dict(config=dict(K0=K0, WINDOW_GAMES=WINDOW_GAMES, TRADE_MIN=TRADE_MIN,
                           STAR_MIN=STAR_MIN, RETURN_DAYS=RETURN_DAYS,
                           MIN_PRIOR_G=MIN_PRIOR_G,
                           coach_changes="skipped-empty",
                           control="same-run fit_production (carry+tank); "
                                   "variant = m_ctl + 0.5*(fm_event - fm_ctl)",
                           gate="paired bootstrap 2000x seed 7, isolation "
                                "p_exp vs p_ctl (F2 freeze metric)"),
               replication=repl,
               seasons={},
               events={s: len(v) for s, v in all_details.items()}
               or "see run log")
    for s, g in df.groupby("season"):
        res["seasons"][s] = dict(
            n=int(len(g)),
            ll_ctl=round(float(g.ll_ctl.mean()), 4),
            ll_exp=round(float(g.ll_exp.mean()), 4),
            ll_mkt=round(float(g.ll_mkt.mean()), 4),
            n_aff=int(g.aff.sum()),
            gate_vs_ctl=boot_gate(g.ll_ctl - g.ll_exp),
            aff_gate_vs_ctl=boot_gate((g.ll_ctl - g.ll_exp)[g.aff]))
    in2326 = df.season.isin(PRIOR_SEASONS)
    res["pooled"] = dict(
        n=int(len(df)),
        gate_vs_ctl=boot_gate(d_iso),
        aff_gate_vs_ctl=boot_gate(d_iso[df.aff]),
        gate_vs_ctl_2326=boot_gate(d_iso[in2326]),
        aff_gate_vs_ctl_2326=boot_gate(d_iso[in2326 & df.aff]),
        n_aff=int(df.aff.sum()))
    # ---- power comparison vs the frozen run -----------------------------
    if PRIOR_JSON.exists():
        pr = json.load(open(PRIOR_JSON))
        pp = pr["pooled"]["gate_vs_ctl"]
        pa = pr["pooled"]["aff_gate_vs_ctl"]
        np_ = res["pooled"]["gate_vs_ctl"]
        na = res["pooled"]["aff_gate_vs_ctl"]
        res["power_vs_prior"] = dict(
            prior_pooled_iso=dict(n=pp["n"], mean=pp["mean"], lo=pp["lo"],
                                  hi=pp["hi"], ci_width=pp["hi"] - pp["lo"]),
            new_pooled_iso=dict(n=np_["n"], mean=np_["mean"], lo=np_["lo"],
                                hi=np_["hi"], ci_width=np_["hi"] - np_["lo"]),
            prior_aff_iso=dict(n=pa["n"], mean=pa["mean"], lo=pa["lo"],
                               hi=pa["hi"], ci_width=pa["hi"] - pa["lo"]),
            new_aff_iso=dict(n=na["n"], mean=na["mean"], lo=na["lo"],
                             hi=na["hi"], ci_width=na["hi"] - na["lo"]))
    json.dump(res, open(OUT_JSON, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
