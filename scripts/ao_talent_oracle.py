"""AO TEST 2 — PERFECT-TALENT ORACLE (Sean's directive, pre-registered, ONE
config). Bounds ALL paid talent products: EPM paid tier, LEBRON, CTG-derived
composites — anything non-unicorn. No talent metric can beat perfect talent
knowledge, so the improvement measured here is the CEILING for any
talent-metric purchase (PAID_ORACLES entry).

ORACLE CAVEAT (label): DELIBERATE LOOK-AHEAD. The composition model's talent
input (PIT DARKO dpm as-of the refit date) is replaced with HINDSIGHT "true
talent" = the player's realized dpm averaged over the SURROUNDING 60 days —
darko_history rows in [refit_date - 30d, refit_date + 30d], past AND future.
The future half of the window sees form/role/health information no live
product can have. Everything else (rosters, trailing minutes, outs, blend,
schedule layer, tank term) is bitwise the shipped construction.

CONSTRUCTION:
  * at each weekly refit: oracle_talent[pid] = avg(dpm) over darko_history in
    [refit-30d, refit+30d]; players with no window rows (edge cases: darko
    publishes on days played) fall back to their PIT talent — counted.
  * per game: cm_pit = comp.margin(..., home_edge=0.0) (the shipped leg);
    cm_or  = same sum with talent swapped (roster/trailing-min/outs
    identical; manual PIT recomputation asserted == comp.margin to 1e-9);
    variant margin = control margin - 0.5*cm_pit + 0.5*cm_or
    — the exact ORACLE_MINUTES swap precedent (prod_by_season.py, PAID_ORACLES
    #2). The 0.5 weight assumes ff.ready (true from opening night since the
    D62 carry, D67-R3); readiness re-verified at each season's first refit.

CONTROL = same-run unmodified fit_production (CURRENT production: D62 carry +
D73 tank + codex-round-6 fixes), prod_by_season.py loop verbatim (default
oracle-outs / bought-availability tier, weekly refit). Replication checked vs
data/capstone_pergame_tank.csv (~1e-14 rerun jitter expected, D63).

GATE / REPORT (pre-registered): paired bootstrap 2000x 95% CI on per-game
logloss deltas (control - variant): PER-SEASON + POOLED + MID-DISTRIBUTION
subset |p_mkt - 0.5| <= 0.35 (D77's precision region — does perfect talent
close the toss-up deficit? Flagged for the v3 program: this is the cheapest
test of whether v3-class talent precision can close mid-distribution at all).
Also reported: variant-vs-market deltas, gp buckets (diagnostics).

Read-only DB. NEW file scripts/ao_talent_oracle.py only; nbapred/ untouched.
Outputs: data/ao_talent_oracle_pergame.csv, data/ao_talent_oracle_results.json.
Run:      python scripts/ao_talent_oracle.py            (full walk-forward)
Analyze:  python scripts/ao_talent_oracle.py --analyze  (bootstrap from CSV)
"""
import csv
import datetime as dt
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from nbapred.db import connect
from nbapred.model.composition import ROSTER_DAYS, CompositionModel
from nbapred.model.production import (SCALE, _prev_season, continuity_map,
                                      fit_production, sigmoid)

OUT_DIR = REPO / "data"
SEASONS = ("2023-24", "2024-25", "2025-26")
HALF_WINDOW = 30                      # days each side -> 60-day surround
MID_BAND = 0.35                       # |p_mkt - 0.5| <= 0.35 (D77 region)
PERGAME_CSV = OUT_DIR / "ao_talent_oracle_pergame.csv"
RESULTS_JSON = OUT_DIR / "ao_talent_oracle_results.json"
CAPSTONE = OUT_DIR / "capstone_pergame_tank.csv"


def oracle_talent_map(con, refit_date):
    """{player_id: mean dpm over [refit-30d, refit+30d]} — hindsight talent."""
    return dict(con.execute(
        "SELECT player_id, avg(dpm) FROM darko_history "
        "WHERE date >= ? AND date <= ? GROUP BY 1",
        [refit_date - dt.timedelta(days=HALF_WINDOW),
         refit_date + dt.timedelta(days=HALF_WINDOW)]).fetchall())


def check_ff_ready(con, season, before):
    """Replicate fit_production's FourFactors readiness condition: rows exist
    (current-season factor rows or the D62 carry rows)."""
    from nbapred.model.four_factors import factor_game_rows
    cur = factor_game_rows(con, season, before=before)
    if cur:
        return True
    if continuity_map(con, season, before=before) is None:
        return False
    return bool(factor_game_rows(con, _prev_season(season), before=None))


def season_run(season, diag):
    """prod_by_season.py loop VERBATIM (default oracle-outs, weekly refit);
    control = fit_production margin; variant swaps the composition leg's
    talent for the 60-day-surround oracle."""
    t0 = time.time()
    con = connect(read_only=True)
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
    gp_live = {}
    model = comp = None
    otal = {}                 # pid -> oracle talent (window mean, PIT fallback)
    last = None
    nrefit = 0
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
            if last is None:                       # first refit of the season
                ready = check_ff_ready(con, season, gd)
                diag.setdefault("ff_ready_week1", {})[season] = bool(ready)
                if not ready:
                    raise RuntimeError(
                        f"ff not ready at {gd} — 0.5 blend swap invalid")
            nrefit += 1
            model = fit_production(con, season, before=gd, w_comp=0.7)
            comp = CompositionModel(con, before=gd)
            omap = oracle_talent_map(con, gd)
            nfall = 0
            otal = {}
            for pid, p in comp.players.items():
                if pid in omap:
                    otal[pid] = float(omap[pid])
                else:
                    otal[pid] = p["talent"]
                    nfall += 1
            diag.setdefault("fallbacks", []).append(
                (str(gd), nfall, len(comp.players)))
            last = gd
        pmv = mkt.get((str(gd)[:10], h.team_abbrev, a.team_abbrev))
        gph = gp_live.get(h.team_id, 0)
        gpa = gp_live.get(a.team_id, 0)
        gp_live[h.team_id] = gph + 1
        gp_live[a.team_id] = gpa + 1
        if pmv is None:
            continue
        outs = {}
        for t in (h.team_id, a.team_id):
            pl = played.get((gid, t), set())
            outs[t] = {p for p, d0 in comp.players.items()
                       if d0["team_id"] == t and (gd - d0["last_played"]).days <= 12
                       and p not in pl}
        mm = model.margin(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id],
                          gd, b2b_home=b2b(h.team_id, gd), b2b_away=b2b(a.team_id, gd))
        cm_pit = comp.margin(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id],
                             gd, home_edge=0.0)

        def cm_with(talent):        # replicate comp.strength diff, talent swapped
            s = 0.0
            for pid, p in comp.players.items():
                tid = p["team_id"]
                if tid == h.team_id and pid not in outs[h.team_id] \
                        and (gd - p["last_played"]).days <= ROSTER_DAYS:
                    s += talent[pid] * p["trail_min"] / 48.0
                elif tid == a.team_id and pid not in outs[a.team_id] \
                        and (gd - p["last_played"]).days <= ROSTER_DAYS:
                    s -= talent[pid] * p["trail_min"] / 48.0
            return s

        cm_check = cm_with({pid: p["talent"] for pid, p in comp.players.items()})
        if abs(cm_check - cm_pit) > 1e-9:
            raise RuntimeError(f"comp replication broke: {cm_check} vs {cm_pit}")
        cm_or = cm_with(otal)
        rows.append(dict(
            season=season, game_id=gid, game_date=str(gd)[:10],
            home=h.team_abbrev, away=a.team_abbrev, y=int(h.wl == "W"),
            p_mkt=float(pmv), gp_home=gph, gp_away=gpa,
            cm_pit=round(cm_pit, 4), cm_or=round(cm_or, 4),
            p_ctrl=float(sigmoid(mm / SCALE)),
            p_var=float(sigmoid((mm - 0.5 * cm_pit + 0.5 * cm_or) / SCALE))))
    con.close()
    print(f"[{season}] n={len(rows)} refits={nrefit} ({time.time()-t0:.0f}s)",
          flush=True)
    return rows


def paired_ci(d, B=2000, seed=42):
    d = np.asarray(d, float)
    if len(d) == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (B, len(d)))
    means = d[idx].mean(axis=1)
    return (float(d.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def ll_vec(y, p):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def analyze(all_rows, diag):
    y = np.array([r["y"] for r in all_rows])
    seas = np.array([r["season"] for r in all_rows])
    pmk = np.array([r["p_mkt"] for r in all_rows])
    gpmin = np.array([min(r["gp_home"], r["gp_away"]) for r in all_rows])
    mid = np.abs(pmk - 0.5) <= MID_BAND
    ll_c = ll_vec(y, [r["p_ctrl"] for r in all_rows])
    ll_v = ll_vec(y, [r["p_var"] for r in all_rows])
    ll_m = ll_vec(y, pmk)
    d = ll_c - ll_v                        # positive = oracle better
    dvm = ll_m - ll_v                      # positive = oracle beats market

    base = {}
    with open(CAPSTONE) as f:
        for r in csv.DictReader(f):
            base[(r["season"], r["game_id"])] = float(r["p_us"])
    diffs = [abs(base[(r["season"], r["game_id"])] - r["p_ctrl"])
             for r in all_rows if (r["season"], r["game_id"]) in base]
    repl = dict(baseline=CAPSTONE.name, n_matched=len(diffs),
                n_ours=len(all_rows),
                max_abs_diff=float(max(diffs)) if diffs else None,
                note="expected ~1e-14 rerun jitter (D63)")

    def sub(mask):
        return dict(n=int(mask.sum()), delta=paired_ci(d[mask]),
                    delta_vs_mkt=paired_ci(dvm[mask]),
                    ll_control=round(float(ll_c[mask].mean()), 5),
                    ll_variant=round(float(ll_v[mask].mean()), 5),
                    ll_market=round(float(ll_m[mask].mean()), 5))

    tal_diff = [abs(r["cm_or"] - r["cm_pit"]) for r in all_rows]
    res = dict(
        config=dict(window_days=2 * HALF_WINDOW, mid_band=MID_BAND,
                    swap="m - 0.5*cm_pit + 0.5*cm_oracle (ORACLE_MINUTES precedent)",
                    oracle_caveat="DELIBERATE LOOK-AHEAD: talent = darko dpm "
                                  "averaged over [refit-30d, refit+30d]; "
                                  "ceiling for ANY talent-metric purchase",
                    gate="paired bootstrap 2000x 95% CI; per-season + pooled "
                         "+ mid-distribution |p_mkt-0.5|<=0.35"),
        replication=repl,
        gate=dict(pooled=sub(np.ones(len(d), bool)),
                  per_season={s: sub(seas == s) for s in SEASONS},
                  mid_distribution=sub(mid)),
        diagnostics=dict(
            mid_per_season={s: sub(mid & (seas == s)) for s in SEASONS},
            confident=sub(~mid),
            gp_buckets={"gp[0,20)": sub(gpmin < 20), "gp[20,)": sub(gpmin >= 20)},
            mean_abs_cm_shift=round(float(np.mean(tal_diff)), 4),
            ff_ready_week1=diag.get("ff_ready_week1"),
            fallback_share_last_refit=(
                None if not diag.get("fallbacks") else
                round(diag["fallbacks"][-1][1] / diag["fallbacks"][-1][2], 4))))
    with open(RESULTS_JSON, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


def main():
    if "--analyze" in sys.argv:
        with open(PERGAME_CSV) as f:
            all_rows = []
            for r in csv.DictReader(f):
                r.update(y=int(r["y"]), p_mkt=float(r["p_mkt"]),
                         p_ctrl=float(r["p_ctrl"]), p_var=float(r["p_var"]),
                         cm_pit=float(r["cm_pit"]), cm_or=float(r["cm_or"]),
                         gp_home=int(r["gp_home"]), gp_away=int(r["gp_away"]))
                all_rows.append(r)
        analyze(all_rows, {})
        return
    diag = {}
    all_rows = []
    for s in SEASONS:
        all_rows += season_run(s, diag)
    with open(PERGAME_CSV, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(all_rows)
    analyze(all_rows, diag)


if __name__ == "__main__":
    main()
