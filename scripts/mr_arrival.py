"""BUILD A (pre-registered, Sean's messy-roster-switches directive) —
ARRIVAL ADJUSTMENT: D60 travel coefficients into the composition margin.

Basis (docs/DECISIONS.md D60, journal 7bdf31 + tradedecomp): what travels for
n=76 midseason movers — volume ~1:1 but EFFICIENCY HALF (TS travel 0.48),
minutes -2.6/g initially; the arriver's own transition cost, not chemistry.
Target region: Feb-Mar (+0.011/gm, ~13.9 nats — docs/ADVERSE_HYPOTHESES.md).

POPULATION — mid-season switchers, evaluated PIT at each game date gd:
a player is a switcher iff, in his game log (regular-season games actually
played, seconds>0, strictly before gd), a game for a DIFFERENT team than his
current team (= team of his most recent game) appears within his last 15
games, AND that different-team game is in the CURRENT season (offseason
movers excluded — "mid-season"). Let n = run-of-current-team-games + 1 =
tonight's game number with the new team. Adjustment window: n in 1..14
(fully recovered by game 15, per the ramp below).

ADJUSTMENT (ONE config, as pre-registered; applied to the player's
contribution wherever the composition model credits it, provided the player
is in the base strength: attributed team == scored team, within the 12-day
roster window, not in the OUT set):
  * minutes: minutes_expectation = trail_min - 2.6   (flat during window)
  * talent:  talent x m(n),  m(n) = 0.85 for n <= 10,
             0.85 + 0.15*(n-10)/5 for 10 < n < 15, 1.0 at n >= 15
  contribution = talent * m(n) * max(trail_min - 2.6, 0) / 48
(The pre-registered simple version of "shrink the efficiency portion per
D60 TS-travel 0.48": DARKO dpm is not separable into volume/efficiency, so
switcher talent x 0.85 with a linear ramp back to 1.0 by game 15.)

VARIANT: control margin + 0.5 * (delta_comp_home - delta_comp_away), where
0.5 is the ff-ready blend weight on the composition margin (~97% of games;
in the tiny pre-ff-ready fallback the true weight is 0.7 — mid-season
switchers are ~nonexistent there; count reported as a diagnostic).

CONTROL = shipped production EXACTLY (fit_production incl. D62 carry + D73
tank), prod_by_season.py loop verbatim (weekly refit, oracle-outs path).
Replication check vs data/capstone_pergame_tank.csv (current headline).

GATE: paired bootstrap 2000x 95% CI on per-game logloss deltas (control -
variant; positive = variant better). Report pooled, per-season, Feb-Mar
(the mandated 13.9-nat region), affected games (delta != 0), affected
Feb-Mar. Read-only DB. New file scripts/mr_arrival.py; nbapred/ untouched.
"""
import bisect
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
from nbapred.model.composition import CompositionModel, ROSTER_DAYS
from nbapred.model.production import SCALE, fit_production, sigmoid

OUT_DIR = REPO / "data"
SEASONS = ("2023-24", "2024-25", "2025-26")
MIN_SHIFT = 2.6          # D60: minutes -2.6/g initially
TALENT_MULT = 0.85       # D60 simple version of TS-travel 0.48
RAMP_START = 10          # games 1..10 at 0.85
RAMP_END = 15            # linear back to 1.0 by game 15
COMP_WEIGHT = 0.5        # ff-ready blend weight on composition
SANITY_DATES = (dt.date(2024, 2, 15), dt.date(2025, 2, 15), dt.date(2026, 2, 15))


def build_player_logs(con):
    """{pid: (dates list, team_ids list, seasons list)} — regular-season games
    actually played (seconds>0), date-sorted. Basis for PIT switcher state."""
    rows = con.execute("""
        SELECT s.player_id, s.team_id, g.game_date, g.season
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games
              WHERE game_id LIKE '002%') g USING (game_id)
        WHERE s.game_id LIKE '002%' AND s.seconds > 0
        ORDER BY s.player_id, g.game_date""").fetchall()
    logs = {}
    for pid, tid, d, season in rows:
        d = d.date() if hasattr(d, "date") else d
        logs.setdefault(int(pid), ([], [], []))
        L = logs[int(pid)]
        L[0].append(d); L[1].append(int(tid)); L[2].append(season)
    return logs


def switch_state(logs, pid, gd, season):
    """None, or (current_team, n, talent_mult) for a mid-season switcher whose
    tonight is new-team game n in 1..14 (PIT: log strictly before gd)."""
    L = logs.get(pid)
    if L is None:
        return None
    dates, teams, seasons = L
    i = bisect.bisect_left(dates, gd)      # games strictly before gd
    if i == 0:
        return None
    ct = teams[i - 1]
    j = i - 1
    while j >= 0 and teams[j] == ct:
        j -= 1
    run = (i - 1) - j                      # consecutive games with current team
    n = run + 1                            # tonight = game n with new team
    if j < 0 or n >= RAMP_END:             # no prior other-team game / recovered
        return None
    if seasons[j] != season:               # offseason move -> not mid-season
        return None
    if n <= RAMP_START:
        mult = TALENT_MULT
    else:
        mult = TALENT_MULT + (1.0 - TALENT_MULT) * (n - RAMP_START) / (RAMP_END - RAMP_START)
    return (ct, n, mult)


def comp_delta(comp, logs, team_id, out, gd, season, detail=None):
    """Sum over base-strength-included switchers of (adjusted - base)
    contribution; mirrors CompositionModel.strength inclusion exactly."""
    dlt, nsw = 0.0, 0
    for pid, p in comp.players.items():
        if p["team_id"] != team_id or pid in out:
            continue
        if (gd - p["last_played"]).days > ROSTER_DAYS:
            continue
        st = switch_state(logs, pid, gd, season)
        if st is None:
            continue
        _, n, mult = st
        base = p["talent"] * p["trail_min"] / 48.0
        adj = p["talent"] * mult * max(p["trail_min"] - MIN_SHIFT, 0.0) / 48.0
        dlt += adj - base
        nsw += 1
        if detail is not None:
            detail.append((pid, n, round(mult, 3), round(p["trail_min"], 1),
                           round(p["talent"], 2), round(adj - base, 3)))
    return dlt, nsw


def season_run(season, logs, sanity):
    """prod_by_season.py loop VERBATIM (default oracle-outs path, weekly
    refit); control = fit_production margin, variant = + 0.5*arrival delta."""
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

    id2ab = dict(con.execute(
        "SELECT DISTINCT team_id, team_abbrev FROM nba_games WHERE season=?",
        [season]).fetchall())

    rows = []
    model = comp = None
    last = None
    nrefit = 0
    sanity_done = set()
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
            nrefit += 1
            model = fit_production(con, season, before=gd, w_comp=0.7)
            comp = CompositionModel(con, before=gd)
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
        mm = model.margin(h.team_id, a.team_id, outs[h.team_id], outs[a.team_id],
                          gd, b2b_home=b2b(h.team_id, gd), b2b_away=b2b(a.team_id, gd))
        det_h = [] if (gd in SANITY_DATES and gd not in sanity_done) else None
        det_a = [] if det_h is not None else None
        dh, nh = comp_delta(comp, logs, h.team_id, outs[h.team_id], gd, season, det_h)
        da, na = comp_delta(comp, logs, a.team_id, outs[a.team_id], gd, season, det_a)
        if det_h is not None and (det_h or det_a):
            sanity.setdefault(str(gd), []).append(
                dict(game=f"{a.team_abbrev}@{h.team_abbrev}",
                     home=det_h, away=det_a))
        dcm = dh - da
        rows.append(dict(
            season=season, game_id=gid, game_date=str(gd)[:10],
            home=h.team_abbrev, away=a.team_abbrev, y=int(h.wl == "W"),
            p_mkt=float(pmv), n_sw_home=nh, n_sw_away=na,
            dcm=round(dcm, 5),
            p_ctrl=float(sigmoid(mm / SCALE)),
            p_var=float(sigmoid((mm + COMP_WEIGHT * dcm) / SCALE))))
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


def main():
    con = connect(read_only=True)
    logs = build_player_logs(con)
    names = dict(con.execute(
        "SELECT player_id, full_name FROM nba_players").fetchall())
    con.close()

    sanity = {}
    all_rows = []
    for s in SEASONS:
        all_rows += season_run(s, logs, sanity)

    with open(OUT_DIR / "mr_arrival_pergame.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(all_rows)

    y = np.array([r["y"] for r in all_rows])
    seas = np.array([r["season"] for r in all_rows])
    month = np.array([int(r["game_date"][5:7]) for r in all_rows])
    dcm = np.array([r["dcm"] for r in all_rows])
    ll_c = ll_vec(y, [r["p_ctrl"] for r in all_rows])
    ll_v = ll_vec(y, [r["p_var"] for r in all_rows])
    ll_m = ll_vec(y, [r["p_mkt"] for r in all_rows])
    d = ll_c - ll_v          # positive = variant better

    # ---- control replication check vs shipped capstone (tank headline) ----
    base = {}
    with open(OUT_DIR / "capstone_pergame_tank.csv") as f:
        for r in csv.DictReader(f):
            base[(r["season"], r["game_id"])] = float(r["p_us"])
    diffs = [abs(base[(r["season"], r["game_id"])] - r["p_ctrl"])
             for r in all_rows if (r["season"], r["game_id"]) in base]
    repl = dict(baseline="capstone_pergame_tank.csv", n_matched=len(diffs),
                n_ours=len(all_rows),
                max_abs_diff=float(max(diffs)) if diffs else None)

    # ---- readable sanity: switchers at 3 mid-Feb dates --------------------
    sanity_named = {}
    for sd, games in sanity.items():
        out = []
        for g in games:
            for side in ("home", "away"):
                for pid, n, mult, tmin, tal, dd in (g[side] or []):
                    out.append(f"{g['game']} {side}: {names.get(pid, pid)} "
                               f"game#{n} mult={mult} trail_min={tmin} "
                               f"talent={tal} d_contrib={dd}")
        if out:
            sanity_named[sd] = out[:12]

    affected = dcm != 0.0
    febmar = (month == 2) | (month == 3)
    octnov = (month == 10) | (month == 11)

    def sub(mask):
        return dict(n=int(mask.sum()), delta=paired_ci(d[mask]),
                    ll_control=round(float(ll_c[mask].mean()), 5),
                    ll_variant=round(float(ll_v[mask].mean()), 5),
                    ll_market=round(float(ll_m[mask].mean()), 5))

    res = dict(
        config=dict(min_shift=MIN_SHIFT, talent_mult=TALENT_MULT,
                    ramp=f"0.85 games 1-{RAMP_START}, linear to 1.0 by game {RAMP_END}",
                    comp_weight=COMP_WEIGHT, population="mid-season switchers "
                    "(different-team game within last 15 played, same season)",
                    gate="paired bootstrap 2000x 95% CI, variant vs control"),
        replication=repl,
        control_ll=dict(pooled=round(float(ll_c.mean()), 5),
                        market=round(float(ll_m.mean()), 5),
                        per_season={s: round(float(ll_c[seas == s].mean()), 4)
                                    for s in SEASONS}),
        variant_ll=dict(pooled=round(float(ll_v.mean()), 5),
                        per_season={s: round(float(ll_v[seas == s].mean()), 4)
                                    for s in SEASONS}),
        gate=dict(
            pooled=sub(np.ones(len(d), bool)),
            per_season={s: sub(seas == s) for s in SEASONS},
            feb_mar=sub(febmar),
            affected=sub(affected),
            affected_feb_mar=sub(affected & febmar)),
        diag=dict(
            n_affected=int(affected.sum()),
            n_affected_per_season={s: int((affected & (seas == s)).sum())
                                   for s in SEASONS},
            n_affected_oct_nov=int((affected & octnov).sum()),
            mean_abs_dcm_affected=round(float(np.abs(dcm[affected]).mean()), 4)
                if affected.any() else 0.0,
            max_abs_dcm=round(float(np.abs(dcm).max()), 4),
            mean_switchers_per_affected_game=round(float(np.mean(
                [r["n_sw_home"] + r["n_sw_away"]
                 for r in all_rows if r["dcm"] != 0.0])), 2)
                if affected.any() else 0.0),
        sanity_switchers=sanity_named)
    with open(OUT_DIR / "mr_arrival_results.json", "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
