"""BA-WINDOWED — fresh pre-registered ONE-SHOT gate (second-look-aware).

CANDIDATE = "WINDOWED PORTFOLIO": current shipped production margin (D63
carry-shipped fit_production) PLUS three terms that are each EXACTLY ZERO
outside their window, so most games are bitwise-identical to control and the
gate variance shrinks to the windows' footprint:

  (1) DEAD-TEAM FE      — pass dead_home/dead_away at predict time; the
      coefficients are already fit walk-forward inside fit_schedule_layer
      (wpct-control de-confounded). Construction identical to
      scripts/ba_portfolio.py's "dead" piece. Window: team gp>=60 & wpct<.35.
  (2) EVENT-RECENCY     — post-trade/star-return 15-game FF window blend on
      the CONTROL FourFactors fit (scripts/exp_eventrecency.py construction,
      via ba_portfolio's EventState + CtlFFView "evrec" piece). Window: within
      15 games after a detected regime event.
  (3) LATE-GATED FORM   — NEW term from D65 ("late-season collapse cluster"):
      only when EITHER side has team games-played >= 55, add
      k * (form5_home - form5_away), form5 = trailing-5-game mean signed
      margin (this season). k is estimated WALK-FORWARD in the
      fit_schedule_layer style: trailing 730d LATE-SEASON games only (either
      side gp>=55 at the time), y = home margin, X = [1, form5_diff,
      wpct_diff]; the wpct-diff control de-confounds form from season quality
      (control coef fit-only, never applied — the dead-FE pattern);
      k = n/(n+600) * beta_form, shrunk toward the 0 prior. NO hindsight
      coefficient anywhere — D65's "closes 36%" was a hindsight bound; this
      walk-forward version is what gates.

SECOND-LOOK HONESTY: this candidate was informed by D64 (dead/evrec additive,
reach the D61 hole) and D65 (late-form hindsight bound). The bar therefore
stays strict despite the informed selection:

GATE (pre-registered, one shot): paired bootstrap 2000x 95% CI of
(ll_ctl - ll_wind) vs the SAME-RUN control (bitwise fit_production
replication), full 3-season walk-forward capstone.
SHIP iff pooled CI excludes 0 AND pooled point >= +0.0015.
Report (no role in the decision): per-season; early (either gp<20); late
(either gp>=55); D61 intersection (|p_mkt-.5|>.35 & |p_ctl-.5|<=.35, p_mkt
from data/capstone_pergame_carry.csv — subset-definition only, never a
feature); each piece's same-run isolation delta.

RULES HONORED: DuckDB read_only=True; new file scripts/ba_windowed.py only;
nothing in nbapred/ or existing scripts edited; PIT strict (every fit and
every feature uses only games strictly before the predicted date); market p
never a model input. Control cross-checked vs data/capstone_pergame_carry2.csv
(expect ~1e-14).

Run:  python scripts/ba_windowed.py                (full 3-season walk-forward)
      python scripts/ba_windowed.py --analyze-only (re-run gate analysis)
"""
import sys, os, json, csv, warnings
import datetime as _dt
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from nbapred.db import connect
from nbapred.model.production import (SCALE, sigmoid, fit_production,
                                      DEAD_WPCT, DEAD_GP)

# reuse ba_portfolio machinery verbatim (read-only import; no side effects)
from ba_portfolio import (extract_parts, EventState, CtlFFView, pg_ll, boot)

# ---- constants (inherited / pre-registered; nothing tuned after seeing data)
FORM_GP = 55              # late gate: either side team games-played >= 55
FORM_N = 5                # form5 = trailing-5-game mean signed margin
FORM_SHRINK = 600.0       # games of prior mass toward k=0 (fit_schedule_layer)
FORM_WINDOW_D = 730       # trailing window for the walk-forward k fit
GATE_MIN_POINT = 0.0015   # pre-registered ship bar on the pooled point
HEAVY = 0.35              # D61 intersection definition

SEASONS = ("2023-24", "2024-25", "2025-26")
OUT_CSV = ROOT / "data" / "ba_windowed_pergame.csv"
OUT_JSON = ROOT / "data" / "ba_windowed_results.json"
BASE_CSVS = [ROOT / "data" / "capstone_pergame_carry2.csv",
             ROOT / "data" / "capstone_pergame_carry.csv"]
CARRY_CSV = ROOT / "data" / "capstone_pergame_carry.csv"   # D61 subset p_mkt

ARMS = ["ctl", "wind", "dead", "evrec", "form"]


# ---- piece (3): walk-forward late-gated form coefficient --------------------

def fit_form_k(con, before):
    """k for the late-gated form term, fit_schedule_layer-style.

    Training rows: completed games in [before-730d, before) where EITHER team
    had season gp >= FORM_GP at tipoff. y = home margin;
    X = [1, form5_h - form5_a, wpct_h - wpct_a]. The wpct-diff control
    de-confounds form from season quality and is fit-only, never applied.
    Season context (gp/form5/wpct) is accumulated from full seasons strictly
    before `before` (all PIT), so the 730d truncation never undercounts gp.
    Returns (k, n, beta_raw)."""
    rows = con.execute("""
        SELECT season, game_id, game_date, team_id, is_home, pts
        FROM nba_games WHERE game_id LIKE '002%' AND pts IS NOT NULL
        AND game_date < ? ORDER BY game_date, game_id""", [before]).fetchall()
    byg = {}
    for season, gid, d, tid, ish, pts in rows:
        d = d.date() if hasattr(d, "date") else d
        g = byg.setdefault(gid, dict(season=season, date=d))
        g["h" if ish else "a"] = (int(tid), float(pts))
    glist = sorted((g for g in byg.values() if "h" in g and "a" in g),
                   key=lambda g: (g["date"], g["season"]))
    lo = before - _dt.timedelta(days=FORM_WINDOW_D)
    marg = {}                                   # (season, tid) -> [margins]
    X, y = [], []
    for g in glist:
        s = g["season"]
        ht, hp = g["h"]
        at, ap = g["a"]
        mh = marg.get((s, ht), [])
        ma = marg.get((s, at), [])
        if (g["date"] >= lo
                and (len(mh) >= FORM_GP or len(ma) >= FORM_GP)
                and len(mh) >= FORM_N and len(ma) >= FORM_N):
            f5d = float(np.mean(mh[-FORM_N:]) - np.mean(ma[-FORM_N:]))
            wd = (sum(m > 0 for m in mh) / len(mh)
                  - sum(m > 0 for m in ma) / len(ma))
            X.append([1.0, f5d, wd])
            y.append(hp - ap)
        marg.setdefault((s, ht), []).append(hp - ap)
        marg.setdefault((s, at), []).append(ap - hp)
    n = len(X)
    if n == 0:
        return 0.0, 0, 0.0
    beta = np.linalg.lstsq(np.array(X), np.array(y, float), rcond=None)[0]
    w = n / (n + FORM_SHRINK)
    return float(w * beta[1]), n, float(beta[1])


# ---- walk-forward capstone loop (ba_portfolio/prod_by_season construction) --

def season_run(season):
    con = connect(read_only=True)
    ev = EventState(con, season)
    pm = con.execute("""SELECT game_id, team_id, player_id, seconds/60.0 AS mins
        FROM player_game_stats WHERE game_id LIKE '002%' AND seconds>0""").fetchdf()
    played = {(g, t): set(grp.player_id)
              for (g, t), grp in pm.groupby(["game_id", "team_id"])}
    meta = con.execute("""SELECT game_id, team_id, team_abbrev, matchup, wl, pts,
        game_date FROM nba_games WHERE season=? AND game_id LIKE '002%'
        AND wl IS NOT NULL ORDER BY game_date""", [season]).fetchdf()
    mkt = {(str(r[0])[:10], r[1], r[2]): r[3] for r in con.execute(
        "SELECT game_date, home, away, p_home_spread FROM odds_market WHERE season_end=?",
        [int(season[:4]) + 1]).fetchall()}
    by, order = {}, []
    for x in meta.itertuples():
        if x.game_id not in by:
            order.append(x.game_id)
        by.setdefault(x.game_id, []).append(x)
    if os.environ.get("BW_SMOKE"):           # dev harness check only
        order = order[:int(os.environ["BW_SMOKE"])]
    tdates = {}
    for x in meta.itertuples():
        d = x.game_date.date() if hasattr(x.game_date, "date") else x.game_date
        tdates.setdefault(x.team_id, set()).add(d)

    def b2b(tid, d):
        return (d - _dt.timedelta(days=1)) in tdates.get(tid, set())

    # per-team season history with margins (PIT: only rows with date < gd used)
    hist = {}                                # tid -> [(date, win, margin)]
    for gid, recs in by.items():
        if len(recs) != 2:
            continue
        r0, r1 = recs
        d = r0.game_date.date() if hasattr(r0.game_date, "date") else r0.game_date
        hist.setdefault(r0.team_id, []).append((d, r0.wl == "W",
                                                float(r0.pts - r1.pts)))
        hist.setdefault(r1.team_id, []).append((d, r1.wl == "W",
                                                float(r1.pts - r0.pts)))
    for t in hist:
        hist[t].sort(key=lambda x: x[0])

    def past(tid, d):
        return [(w, m) for (dd, w, m) in hist.get(tid, []) if dd < d]

    def dead(tid, d):
        p = past(tid, d)
        return len(p) >= DEAD_GP and (sum(w for w, _ in p) / len(p)) < DEAD_WPCT

    def form5(tid, d):
        p = past(tid, d)
        if len(p) < FORM_N:
            return None
        return float(np.mean([m for _, m in p[-FORM_N:]]))

    rows_out = []
    model = parts = None
    k_form, n_form, beta_raw = 0.0, 0, 0.0
    k_traj = []
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
            k_form, n_form, beta_raw = fit_form_k(con, gd)
            k_traj.append(dict(date=str(gd), k=round(k_form, 5), n=n_form,
                               beta_raw=round(beta_raw, 5)))
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
        gp_h, gp_a = len(past(h.team_id, gd)), len(past(a.team_id, gd))
        # CONTROL: shipped production exactly as the capstone calls it
        p_ctl = model.p_home(h.team_id, a.team_id, outs[h.team_id],
                             outs[a.team_id], gd, b2b_home=b2h, b2b_away=b2a)
        # windowed pieces (each EXACTLY 0.0 outside its window)
        sched_nd = (parts["he"] + (parts["b_hb2b"] if b2h else 0.0)
                    + (parts["b_ab2b"] if b2a else 0.0))
        sched_d = (sched_nd + (parts["b_hdead"] if dh else 0.0)
                   + (parts["b_adead"] if da else 0.0))
        ft = 0.0
        if gp_h >= FORM_GP or gp_a >= FORM_GP:
            f5h, f5a = form5(h.team_id, gd), form5(a.team_id, gd)
            if f5h is not None and f5a is not None:
                ft = k_form * (f5h - f5a)
        cm = comp.margin(h.team_id, a.team_id, outs[h.team_id],
                         outs[a.team_id], gd, home_edge=0.0)
        ff = parts["ff"]
        wj_h = wj_a = 0.0
        if ff.ready:
            fm_ctl = ff.margin_neutral(h.team_id, a.team_id)
            ovh_c, wj_h, _ = ev.overrides(h.team_id, gd, ff.fms, ff.fms)
            ova_c, wj_a, _ = ev.overrides(a.team_id, gd, ff.fms, ff.fms)
            ov_c = {**ovh_c, **ova_c}
            cview = CtlFFView(ff)
            fm_ctl_ev = (cview.margin_neutral(h.team_id, a.team_id, ov_c)
                         if ov_c else fm_ctl)
            m_ctl_man = 0.5 * fm_ctl + 0.5 * cm + sched_nd
            marg = {"wind": 0.5 * fm_ctl_ev + 0.5 * cm + sched_d + ft,
                    "dead": 0.5 * fm_ctl + 0.5 * cm + sched_d,
                    "evrec": 0.5 * fm_ctl_ev + 0.5 * cm + sched_nd,
                    "form": m_ctl_man + ft}
        else:
            # production fallback (ratings + cold-start prior, w_comp=0.7);
            # events ride on FF so evrec is inert here; dead/form windows
            # cannot be open this early (gp>=60 / gp>=55) but keep the exact
            # construction anyway
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
            marg = {"wind": base + sched_d + ft, "dead": base + sched_d,
                    "evrec": base + sched_nd, "form": base + sched_nd + ft}
        ctl_fid = max(ctl_fid, abs(float(sigmoid(m_ctl_man / SCALE)) - p_ctl))
        ps = {k: float(sigmoid(v / SCALE)) for k, v in marg.items()}
        rows_out.append([season, gid, str(gd)[:10], h.team_abbrev, a.team_abbrev,
                         yv, p_ctl, ps["wind"], ps["dead"], ps["evrec"],
                         ps["form"], float(pmv), int(dh), int(da), gp_h, gp_a,
                         round(wj_h, 4), round(wj_a, 4), round(ft, 5),
                         round(k_form, 5)])
    con.close()
    n_ev = sum(len(v) for v in ev.events.values())
    n_dead = sum(1 for r in rows_out if r[12] or r[13])
    n_evact = sum(1 for r in rows_out if r[16] > 0 or r[17] > 0)
    n_form = sum(1 for r in rows_out if r[18] != 0.0)
    print(f"[{season}] games={len(rows_out)} events={n_ev} "
          f"ctl_fidelity_max_dp={ctl_fid:.2e} dead_games={n_dead} "
          f"ev_active={n_evact} form_active={n_form} "
          f"k_final={k_traj[-1] if k_traj else None}", flush=True)
    return rows_out, {"ctl_fidelity_max_dp": ctl_fid, "n_events": n_ev,
                      "dead_games": n_dead, "ev_active": n_evact,
                      "form_active": n_form, "k_trajectory": k_traj}


# ---- gate analysis ----------------------------------------------------------

def analyze(run_diag):
    import pandas as pd
    df = pd.read_csv(OUT_CSV, dtype={"game_id": str})
    ll = {a: pg_ll(df.y, df[f"p_{a}"]) for a in ARMS}
    ll["mkt"] = pg_ll(df.y, df.p_mkt)
    d_wind = ll["ctl"] - ll["wind"]            # positive = candidate better

    res = {"config": dict(
        pieces="dead-FE + event-recency(ctl-FF) + late-gated form",
        form=dict(gp_gate=FORM_GP, n=FORM_N, shrink=FORM_SHRINK,
                  window_d=FORM_WINDOW_D, coef="walk-forward, wpct-controlled"),
        gate="pooled CI excludes 0 AND pooled point >= +0.0015 (pre-registered)",
        heavy_def=f"|p_mkt_carry-.5|>{HEAVY} & |p_ctl-.5|<={HEAVY}"),
        "run_diagnostics": run_diag}

    # control cross-check vs shipped carry capstone CSVs
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

    # windowing property: how much of the sample is bitwise control
    ident = (df.p_wind.values == df.p_ctl.values)
    res["window_footprint"] = {
        "n_total": int(len(df)),
        "n_bitwise_identical_to_ctl": int(ident.sum()),
        "n_touched": int((~ident).sum()),
        "touched_frac": round(float((~ident).mean()), 4),
        "n_dead": int(((df.dead_home == 1) | (df.dead_away == 1)).sum()),
        "n_ev_active": int(((df.w_ev_home > 0) | (df.w_ev_away > 0)).sum()),
        "n_form_active": int((df.form_term != 0.0).sum()),
    }

    # headline log-losses
    res["logloss"] = {}
    for s, g in df.groupby("season"):
        res["logloss"][s] = {a: round(float(pg_ll(g.y, g[f"p_{a}"]).mean()), 4)
                             for a in ("ctl", "wind")}
        res["logloss"][s]["mkt"] = round(float(pg_ll(g.y, g.p_mkt).mean()), 4)
        res["logloss"][s]["n"] = int(len(g))

    # D61 intersection subset: p_mkt from the carry capstone CSV (definition
    # only, never a feature); p_us = same-run control
    heavy = np.zeros(len(df), bool)
    if CARRY_CSV.exists():
        b = pd.read_csv(CARRY_CSV, dtype={"game_id": str})
        pmk = df.merge(b[["season", "game_id", "p_mkt"]], on=["season", "game_id"],
                       how="left", suffixes=("", "_carry"))["p_mkt_carry"]
        pmk = pmk.fillna(df.p_mkt)
        heavy = ((np.abs(pmk - 0.5) > HEAVY)
                 & (np.abs(df.p_ctl - 0.5) <= HEAVY)).values

    early = ((df.gp_home < 20) | (df.gp_away < 20)).values
    late = ((df.gp_home >= FORM_GP) | (df.gp_away >= FORM_GP)).values

    pooled = boot(d_wind)
    gate_pass = bool(pooled["lo"] > 0 and pooled["mean"] >= GATE_MIN_POINT)
    res["gate"] = {
        "pooled": pooled,
        "point_bar": GATE_MIN_POINT,
        "SHIP": gate_pass,
        "per_season": {s: boot(d_wind[(df.season == s).values]) for s in SEASONS},
        "early_either_gp_lt20": boot(d_wind[early]),
        "late_either_gp_ge55": boot(d_wind[late]),
        "d61_intersection": boot(d_wind[heavy]),
        "early_n": int(early.sum()), "late_n": int(late.sum()),
        "heavy_n": int(heavy.sum()),
        "per_season_subsets": {
            s: {"early": boot(d_wind[early & (df.season == s).values]),
                "late": boot(d_wind[late & (df.season == s).values]),
                "d61": boot(d_wind[heavy & (df.season == s).values])}
            for s in SEASONS},
    }

    # same-run isolation deltas (report-only)
    res["isolation_report_only"] = {}
    for a in ("dead", "evrec", "form"):
        d = ll["ctl"] - ll[a]
        res["isolation_report_only"][a] = {
            "pooled": boot(d),
            "d61_intersection": boot(d[heavy]),
            "late": boot(d[late]),
        }
    iso_sum = float(sum((ll["ctl"] - ll[a]).mean()
                        for a in ("dead", "evrec", "form")))
    res["additivity"] = {"isolation_sum": iso_sum,
                         "realized_joint": pooled["mean"],
                         "joint_minus_sum": float(pooled["mean"] - iso_sum)}
    json.dump(res, open(OUT_JSON, "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


def main():
    run_diag = {}
    if "--analyze-only" not in sys.argv:
        header = ["season", "game_id", "game_date", "home", "away", "y",
                  "p_ctl", "p_wind", "p_dead", "p_evrec", "p_form", "p_mkt",
                  "dead_home", "dead_away", "gp_home", "gp_away",
                  "w_ev_home", "w_ev_away", "form_term", "k_form"]
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
