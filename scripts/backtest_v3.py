"""M1 backtest — team-DLM pilot vs production, same-run control (V3_SPEC M1).

One continuous team-DLM filter runs 2021-10 -> 2026-04 (margin obs for the
2021-22 warm-up + playoff rows, efficiency-pair obs 2022-23 on), hyperparams
refit MONTHLY walk-forward by trailing-2-season marginal likelihood, season
boundaries handled by the event shock (kappa continuity x variance inflation
— NOT a refit). In the same run the production stack is replicated exactly as
shipped (weekly refits, oracle outs, csfix construction — the es_continuity
control math, verified ~1e-14 vs data/capstone_pergame_csfix.csv).

Arms (all sigmoid(margin/7.2), link isolation per spec):
  ctrl        0.5*ff + 0.5*comp + sched   (production; = csfix p_us)
  swap_ff     0.5*DLM + 0.5*comp + sched  when ff ready, else ctrl fallback
              [PRIMARY GATE: 'swap fm -> dlm margin in the blend']
  dlm_always  0.5*DLM + 0.5*comp + sched  every game (DLM is always ready —
              the cross-season-continuity dividend shows early season)
  swap_comp   0.5*ff + 0.5*DLM + sched    when ff ready (V3_SPEC's variant)
  dlm_sched   DLM + sched                 (standalone diagnostic)
  ff_sched    ff  + sched                 when ready, else ctrl fallback
              (standalone FF twin for the component comparison)

Gate protocol: paired_bootstrap_delta vs ctrl per season + pooled + October
slice; ship rule G1 (pooled CI excludes 0). Outputs:
  data/v3_m1_pergame.csv, data/v3_m1_results.json
Shadow logging: swap_ff predictions -> v3_predictions (version m1.0-swapff),
season-boundary shocks -> state_shocks. Reads are read_only; writes go through
nbapred.v3.schema.v3_writer (single-writer respected).
"""
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

# tiny 62x62 kalman algebra: BLAS threading THRASHES (490% CPU, ~50x slower);
# pin to one thread before numpy loads
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from nbapred.db import connect
from nbapred.eval.ablate import paired_bootstrap_delta
from nbapred.eval.metrics import log_loss
from nbapred.model.composition import CompositionModel
from nbapred.model.four_factors import FourFactors
from nbapred.model.production import (SCALE, fit_schedule_layer,
                                      last_season_prior, sigmoid)
from nbapred.model.team_ratings import TeamRatings, game_rows
from nbapred.v3.filter_run import run_filter
from nbapred.v3.link import p_home

SEASONS = ("2023-24", "2024-25", "2025-26")
W_COMP = 0.7
CSFIX = REPO / "data" / "capstone_pergame_csfix.csv"
OUT_CSV = REPO / "data" / "v3_m1_pergame.csv"
OUT_JSON = REPO / "data" / "v3_m1_results.json"


def run_dlm(con, csfix: pd.DataFrame):
    """Continuous filter over the whole span; one-step-ahead neutral margins
    for every csfix game, captured BEFORE that date's results are fed."""
    by_date = {}
    for r in csfix.itertuples():
        d = dt.date.fromisoformat(r.game_date)
        by_date.setdefault(d, []).append((r.game_id, r.home, r.away))
    margins = {}

    def on_slate(d, dlm):
        for gid, h, a in by_date.get(d, []):
            margins[gid] = (dlm.margin_neutral(h, a),
                            dlm.margin_neutral_var(h, a))

    start, end = min(by_date), max(by_date)
    t0 = time.time()
    dlm, hyper, hyper_log = run_filter(con, start, end, hyperfit=True,
                                       on_slate=on_slate, verbose=True)
    print(f"DLM run: {time.time()-t0:.0f}s, {len(hyper_log)} hyperfits, "
          f"{len(margins)}/{len(csfix)} game margins", flush=True)
    missing = set(csfix.game_id) - set(margins)
    if missing:
        raise RuntimeError(f"{len(missing)} csfix games missing DLM margins: "
                           f"{sorted(missing)[:5]}")
    return margins, hyper, hyper_log


def season_run(con, season: str, dlm_margins: dict):
    """Production replication (control) + arm margins, one pass."""
    t0 = time.time()
    prior = last_season_prior(con, season)                # 0.75-regressed
    ab = dict(con.execute(
        "SELECT DISTINCT team_id, team_abbrev FROM nba_games WHERE season=?",
        [season]).fetchall())
    id2ab = {t: a for t, a in ab.items()}

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
        return (d - dt.timedelta(days=1)) in tdates.get(tid, set())

    rows = []
    comp = tr = ff = None
    sched = None
    games_played = {}
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
            comp = CompositionModel(con, before=gd)
            sched = fit_schedule_layer(con, before=gd)
            tr = TeamRatings(ridge=25.0).fit(game_rows(con, before=gd, season=season))
            ff = FourFactors().fit(con, season, before=gd)
            games_played = dict(con.execute("""
                SELECT team_id, count(*) FROM nba_games WHERE season=? AND
                game_id LIKE '002%' AND wl IS NOT NULL AND game_date < ?
                GROUP BY 1""", [season, gd]).fetchall())
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
        sch = (sched[0] + (sched[1] if b2b(h.team_id, gd) else 0.0)
               + (sched[2] if b2b(a.team_id, gd) else 0.0))
        cm = comp.margin(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id],
                         gd, home_edge=0.0)
        md, vd = dlm_margins[gid]

        def fallback():
            mm = tr.pred_margin(h.team_id, a.team_id)
            wh = max(0.0, 1 - games_played.get(h.team_id, 0) / 20.0)
            wa = max(0.0, 1 - games_played.get(a.team_id, 0) / 20.0)
            rm = (mm + wh * prior.get(id2ab.get(h.team_id, ""), 0.0)
                  - wa * prior.get(id2ab.get(a.team_id, ""), 0.0) - tr.home)
            return W_COMP * cm + (1 - W_COMP) * rm

        ready = ff.ready
        fm = ff.margin_neutral(h.team_id, a.team_id) if ready else np.nan
        m_ctrl = (0.5 * fm + 0.5 * cm if ready else fallback()) + sch
        m_swapf = (0.5 * md + 0.5 * cm if ready else fallback()) + sch
        m_alws = 0.5 * md + 0.5 * cm + sch
        m_swapc = (0.5 * fm + 0.5 * md if ready else fallback()) + sch
        m_dlm = md + sch
        m_ffs = (fm if ready else fallback()) + sch
        rows.append(dict(
            season=season, game_id=gid, game_date=str(gd)[:10],
            home=h.team_abbrev, away=a.team_abbrev, y=int(h.wl == "W"),
            p_mkt=float(pmv),
            p_ctrl=float(sigmoid(m_ctrl / SCALE)),
            p_swap_ff=float(sigmoid(m_swapf / SCALE)),
            p_dlm_always=float(sigmoid(m_alws / SCALE)),
            p_swap_comp=float(sigmoid(m_swapc / SCALE)),
            p_dlm_sched=float(sigmoid(m_dlm / SCALE)),
            p_ff_sched=float(sigmoid(m_ffs / SCALE)),
            m_ff=float(fm) if ready else np.nan, m_cm=float(cm),
            m_dlm=float(md), v_dlm=float(vd), ff_ready=int(ready)))
    print(f"[{season}] n={len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    return rows


def boot(df, col, base="p_ctrl"):
    r = paired_bootstrap_delta(df.y.values, df[base].values, df[col].values)
    r["logloss"] = log_loss(df.y, df[col])
    return r


def main():
    con = connect(read_only=True)
    csfix = pd.read_csv(CSFIX, dtype={"game_id": str})
    dlm_margins, hyper, hyper_log = run_dlm(con, csfix)
    rows = []
    for season in SEASONS:
        rows += season_run(con, season, dlm_margins)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    # ---- control replication vs csfix --------------------------------------
    j = csfix.merge(df[["game_id", "p_ctrl"]], on="game_id", validate="1:1")
    rep = float(np.max(np.abs(j.p_us - j.p_ctrl)))
    print(f"\ncsfix replication: n={len(j)}/{len(csfix)} max|p_ctrl-p_us|={rep:.2e}")

    res = {"n": len(df), "csfix_replication_maxdiff": rep,
           "hyper_final": {k: getattr(hyper, k) for k in hyper.FIT_KEYS},
           "hyper_log": [(str(d), {k: getattr(h, k) for k in h.FIT_KEYS},
                          round(ll, 4)) for d, h, ll in hyper_log],
           "logloss": {}, "gates": {}}
    for col in ("p_ctrl", "p_mkt", "p_swap_ff", "p_dlm_always", "p_swap_comp",
                "p_dlm_sched", "p_ff_sched"):
        res["logloss"][col] = {s: log_loss(g.y, g[col])
                               for s, g in df.groupby("season")}
        res["logloss"][col]["pooled"] = log_loss(df.y, df[col])
    print("\nlog loss by season (control | market | arms):")
    for col, v in res["logloss"].items():
        print(f"  {col:12s} " + " ".join(f"{v[s]:.4f}" for s in SEASONS)
              + f"  pooled {v['pooled']:.4f}")

    df["month"] = pd.to_datetime(df.game_date).dt.month
    oct_df = df[df.month == 10]
    for arm in ("p_swap_ff", "p_dlm_always", "p_swap_comp", "p_dlm_sched"):
        g = {"pooled": boot(df, arm)}
        for s, gg in df.groupby("season"):
            g[s] = boot(gg, arm)
        g["october"] = boot(oct_df, arm)
        g["october"]["n"] = int(len(oct_df))
        res["gates"][arm] = g
        lo, hi = g["pooled"]["ci95"]
        print(f"\nGATE {arm} vs ctrl: pooled {g['pooled']['delta_logloss']:+.5f} "
              f"CI({lo:+.5f},{hi:+.5f}) keep={g['pooled']['keep']}")
        for s in SEASONS + ("october",):
            lo, hi = g[s]["ci95"]
            print(f"    {s:9s} {g[s]['delta_logloss']:+.5f} CI({lo:+.5f},{hi:+.5f})")
    # component head-to-head on the ff-ready subset
    sub = df[df.ff_ready == 1]
    r = paired_bootstrap_delta(sub.y.values, sub.p_ff_sched.values,
                               sub.p_dlm_sched.values)
    res["gates"]["dlm_vs_ff_standalone_ready"] = {**r, "n": int(len(sub))}
    lo, hi = r["ci95"]
    print(f"\nDLM+sched vs FF+sched (ff-ready n={len(sub)}): "
          f"{r['delta_logloss']:+.5f} CI({lo:+.5f},{hi:+.5f}) keep={r['keep']}")

    OUT_JSON.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {OUT_CSV}\nwrote {OUT_JSON}")

    # ---- shadow logging (M0 harness): swap_ff -> v3_predictions ------------
    # DuckDB lock semantics (verified): a writer needs NO other connection on
    # the file, same process included — so gather lookups, CLOSE the reader,
    # then take the short write phase through v3_writer.
    do_write = "--no-write" not in sys.argv
    ab2id = dict(con.execute(
        "SELECT DISTINCT team_abbrev, team_id FROM nba_games "
        "WHERE game_id LIKE '002%' AND season IN ('2023-24','2024-25','2025-26')"
    ).fetchall())
    bounds = con.execute("""
        SELECT season, min(game_date) FROM nba_games
        WHERE game_id LIKE '002%' AND season IN ('2023-24','2024-25','2025-26')
        GROUP BY 1""").fetchall()
    con.close()
    if do_write:
        from nbapred.v3.schema import v3_writer
        from nbapred.v3.shocks import Shock, log_shocks
        preds = [(r.game_id,
                  dt.datetime.combine(dt.date.fromisoformat(r.game_date), dt.time()),
                  "side", float(r.m_dlm), float(np.sqrt(r.v_dlm + 121.0)),
                  float(r.p_swap_ff), "m1.0-swapff")
                 for r in df.itertuples()]
        with v3_writer() as w:
            w.executemany("INSERT OR REPLACE INTO v3_predictions VALUES "
                          "(?, ?, ?, ?, ?, ?, ?)", preds)
            for season, b in bounds:
                log_shocks(w, [Shock(int(t), "season_boundary", b,
                                     "season_calendar")
                               for t in sorted(set(ab2id.values()))])
        print(f"shadow-logged {len(preds)} v3_predictions rows (m1.0-swapff) "
              f"+ season-boundary shocks")


if __name__ == "__main__":
    main()
