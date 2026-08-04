#!/usr/bin/env python3
"""DO ANY KILLED FEATURES COME ALIVE IN THE OLD ERA? (D153 deliverable 4.)

Re-measures, on the newly scorable historical seasons and per ERA, the four
families that were closed as dead veins:

  * TEAM-SPECIFIC HOME ADVANTAGE (D20 shipped -> D70 killed on
    nonstationarity -> D137 quantified: tau 1.80 pts, signal share 26.1%,
    lag-1 +0.02 ns).  Method is D137's verbatim: one regression per season,
    team-strength FE + per-team home effect + schedule controls, minimum-norm
    lstsq so sum_t s_t = 0 and d_t is opponent- and own-quality-controlled;
    method-of-moments EB for tau and the signal share; lag-1 / lag-2
    correlation of d_t ACROSS consecutive seasons.
  * ALTITUDE (D96, two constructions two nulls): DEN and UTA home effects
    a_t per season, and the pooled DEN+UTA home effect per era.
  * TRAVEL / CIRCADIAN / DENSITY (D136 margin frame; D140 fixed the
    neutral-site bug).  Per-era coefficients with t-stats, on the same gated
    forms the pre-registration used, and with `travel_valid` COVERAGE reported
    first — D152 added 7 historical franchises to arenas.csv and a missing one
    silently scores travel 0.
  * LEAGUE HOME EDGE itself, per season, as the context for all of the above.

Panel is rebuilt from `nba_games` + `nbapred.model.travel.build_state`, NOT
from `nbapred.features.schedule.ARENAS` — that dict holds only the 30 CURRENT
franchises, so NJN / NOH / SEA / VAN / CHH / WSB / NOK would silently produce
NaN distances on exactly the seasons this task is about.

DIAGNOSTIC.  READ-ONLY.  A revival here would need its own pre-registered
gate; the honest framing for anything that shows up is "this term is
era-specific", not "we were wrong".

  python scripts/history_killed.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nbapred import threads  # noqa: E402
threads.pin(1)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ha_core import CONTROLS, eb_shrink, fit_season, ols  # noqa: E402
from nbapred.db import connect  # noqa: E402
from nbapred.eval import splits as S  # noqa: E402
from nbapred.model import travel as TV  # noqa: E402

SEED = 20260801


def build_panel(con, seasons):
    st = TV.build_state(con, since=dt.date(2010, 1, 1))
    q = """SELECT season, game_id, game_date, team_id, team_abbrev, matchup, pts, is_home
           FROM nba_games WHERE game_id LIKE '002%' AND pts IS NOT NULL
           ORDER BY game_date, game_id"""
    rows = con.execute(q).fetchall()
    by = {}
    for season, gid, gd, tid, ab, mu, pts, ih in rows:
        if season not in seasons:
            continue
        by.setdefault(gid, []).append(dict(season=season, gid=gid, gd=gd,
                                           tid=tid, ab=ab, mu=mu, pts=pts))
    recs = []
    for gid, rr in by.items():
        if len(rr) != 2:
            continue
        mu = rr[0]["mu"]
        host = mu.split("@")[-1].strip() if "@" in mu else mu.split("vs.")[0].strip()
        h = next((x for x in rr if x["ab"] == host), None)
        a = next((x for x in rr if x["ab"] != host), None)
        if not h or not a:
            continue
        sh, sa = st.get((h["tid"], h["gd"])), st.get((a["tid"], a["gd"]))
        if sh is None or sa is None:
            continue
        recs.append(dict(
            season=h["season"], game_id=gid, game_date=str(h["gd"])[:10],
            home=h["ab"], away=a["ab"], margin=float(h["pts"] - a["pts"]),
            neutral=int(bool(sh["neutral"] or sa["neutral"])),
            travel_valid=int(bool(sh["travel_valid"] and sa["travel_valid"])),
            h_b2b=float(sh["b2b"]), a_b2b=float(sa["b2b"]),
            h_3in4=float(sh["is_3in4"]), a_3in4=float(sa["is_3in4"]),
            h_travel=float(sh["travel_km"]), a_travel=float(sa["travel_km"]),
            h_tz=float(sh["tz_east"]), a_tz=float(sa["tz_east"]),
            h_elev=float(sh["elev_gain_m"]), a_elev=float(sa["elev_gain_m"]),
            h_rest=np.nan, a_rest=np.nan))
    df = pd.DataFrame(recs)
    # rest days, from the schedule itself
    dates = {}
    for season, gid, gd, tid, ab, mu, pts, ih in rows:
        dates.setdefault((season, ab), []).append(gd)
    for k in dates:
        dates[k] = sorted(set(dates[k]))
    def rest(season, ab, d):
        ds = dates.get((season, ab), [])
        prev = [x for x in ds if x < d]
        return (d - prev[-1]).days if prev else np.nan
    df["gd"] = pd.to_datetime(df["game_date"]).dt.date
    df["h_rest"] = [rest(s, t, d) for s, t, d in zip(df.season, df.home, df.gd)]
    df["a_rest"] = [rest(s, t, d) for s, t, d in zip(df.season, df.away, df.gd)]
    hr = df["h_rest"].fillna(3.0).clip(upper=6.0)
    ar = df["a_rest"].fillna(3.0).clip(upper=6.0)
    df["rest_diff"] = (hr - ar).clip(-4, 4)
    df["h_travel_k"] = df["h_travel"] / 1000.0
    df["a_travel_k"] = df["a_travel"] / 1000.0
    df["h_tz_abs"] = df["h_tz"].abs()
    df["a_tz_abs"] = df["a_tz"].abs()
    df["era"] = [S.era_of(s) for s in df.season]
    return df


def main():
    con = connect(read_only=True)
    from history_scorable import sets
    pool, strat, _ = sets(con)
    seasons = sorted(set(pool + strat))
    print("seasons:", seasons, flush=True)
    df = build_panel(con, set(seasons))
    con.close()
    out = {"seasons": seasons}

    # ---------------- travel_valid coverage FIRST ------------------------
    cov = {}
    for s in seasons:
        d = df[df.season == s]
        cov[s] = dict(n=int(len(d)), neutral=int(d.neutral.sum()),
                      travel_valid=int(d.travel_valid.sum()),
                      valid_frac=round(float(d.travel_valid.mean()), 4),
                      mean_travel_km_home=round(float(d.h_travel.mean()), 1),
                      mean_travel_km_away=round(float(d.a_travel.mean()), 1))
    out["travel_coverage"] = cov
    print("\n=== travel_valid COVERAGE (check before trusting any travel number) ===")
    for s in seasons:
        c = cov[s]
        print(f"  {s}  n={c['n']:5d}  valid={c['valid_frac']:.4f}  "
              f"neutral={c['neutral']:3d}  mean km home/away "
              f"{c['mean_travel_km_home']:6.1f}/{c['mean_travel_km_away']:6.1f}")

    # ---------------- per-season team home advantage ---------------------
    use = df[(df.neutral == 0) & (df.travel_valid == 1)].reset_index(drop=True)
    per = {}
    dmaps = {}
    for s in seasons:
        d = use[use.season == s]
        if len(d) < 400:
            continue
        f = fit_season(d, CONTROLS)
        tau2, shr, share = eb_shrink(f["d"], f["se_d"])
        dmaps[s] = dict(zip(f["teams"], f["d"]))
        per[s] = dict(era=S.era_of(s), n=f["n"], hfa=round(f["hfa"], 4),
                      se_hfa=round(f["se_hfa"], 4),
                      sd_d=round(float(np.std(f["d"], ddof=1)), 4),
                      rms_se_d=round(float(np.sqrt(np.mean(f["se_d"] ** 2))), 4),
                      tau=round(float(np.sqrt(tau2)), 4),
                      signal_share=round(float(share), 4),
                      den=round(float(dmaps[s].get("DEN", np.nan)), 3),
                      uta=round(float(dmaps[s].get("UTA", np.nan)), 3))
    out["team_home_per_season"] = per
    print("\n=== TEAM-SPECIFIC HOME ADVANTAGE, per season (D137 method) ===")
    print(f"  {'season':9s} {'era':4s} {'n':>5s} {'leagueHFA':>10s} {'sd(d)':>7s} "
          f"{'rms se':>7s} {'tau':>6s} {'signal':>7s}  {'DEN d':>7s} {'UTA d':>7s}")
    for s in per:
        p = per[s]
        print(f"  {s:9s} {p['era']:4s} {p['n']:5d} {p['hfa']:+10.3f} {p['sd_d']:7.3f} "
              f"{p['rms_se_d']:7.3f} {p['tau']:6.3f} {p['signal_share']:7.1%}  "
              f"{p['den']:+7.2f} {p['uta']:+7.2f}")

    # lag-1 / lag-2 across CONSECUTIVE seasons only
    ordered = [s for s in seasons if s in dmaps]
    def lag(k):
        xs, ys = [], []
        for i in range(len(ordered) - k):
            s0, s1 = ordered[i], ordered[i + k]
            if int(s1[:4]) - int(s0[:4]) != k:
                continue          # not consecutive in calendar terms
            common = set(dmaps[s0]) & set(dmaps[s1])
            for t in common:
                xs.append(dmaps[s0][t]); ys.append(dmaps[s1][t])
        if len(xs) < 20:
            return None
        return dict(n_pairs=len(xs), r=round(float(np.corrcoef(xs, ys)[0, 1]), 4))
    out["team_home_lag"] = {"lag1": lag(1), "lag2": lag(2)}
    # blocks: historical vs modern
    for name, block in (("historical", [s for s in ordered if s < "2019-20"]),
                        ("modern", [s for s in ordered if s >= "2021-22"])):
        xs, ys = [], []
        for i in range(len(block) - 1):
            s0, s1 = block[i], block[i + 1]
            if int(s1[:4]) - int(s0[:4]) != 1:
                continue
            for t in set(dmaps[s0]) & set(dmaps[s1]):
                xs.append(dmaps[s0][t]); ys.append(dmaps[s1][t])
        out["team_home_lag"][f"lag1_{name}"] = (
            dict(n_pairs=len(xs), r=round(float(np.corrcoef(xs, ys)[0, 1]), 4),
                 seasons=block) if len(xs) >= 20 else None)
    print("\n=== TEAM-HOME PERSISTENCE (d_t across consecutive seasons) ===")
    for k, v in out["team_home_lag"].items():
        print(f"  {k:20s} {v}")

    # ---------------- altitude, per era ----------------------------------
    alt = {}
    for e in sorted(set(use.era), key=lambda x: S.ERA_ORDER.index(x)
                    if x in S.ERA_ORDER else 99):
        d = use[use.era == e]
        if len(d) < 400:
            continue
        f = fit_season(d, CONTROLS)
        m = dict(zip(f["teams"], f["a"]))
        se = dict(zip(f["teams"], f["se_a"]))
        hi = float(np.mean([m[t] for t in f["teams"]]))
        rec = {}
        for t in ("DEN", "UTA"):
            if t in m:
                rec[t] = dict(a=round(m[t], 3), se=round(se[t], 3),
                              dev=round(m[t] - hi, 3),
                              t=round((m[t] - hi) / se[t], 2))
        # the visitor's elevation gain as a direct regressor, same frame
        X = np.c_[np.ones(len(d)), d.h_b2b, d.a_b2b, d.rest_diff,
                  (d.a_elev - d.h_elev) / 1000.0]
        b, cv, _, dof = ols(X, d.margin.to_numpy(float))
        rec["aelev_km_coef"] = round(float(b[4]), 4)
        rec["aelev_km_t"] = round(float(b[4] / np.sqrt(cv[4, 4])), 2)
        rec["league_hfa"] = round(hi, 3)
        rec["n"] = int(len(d))
        alt[e] = rec
    out["altitude_by_era"] = alt
    print("\n=== ALTITUDE by era (DEN/UTA home effect, and the visitor's climb) ===")
    for e, r in alt.items():
        den = r.get("DEN", {}); uta = r.get("UTA", {})
        print(f"  {e:4s} n={r['n']:5d} leagueHFA {r['league_hfa']:+6.3f} | "
              f"DEN a={den.get('a', float('nan')):+6.2f} dev={den.get('dev', float('nan')):+6.2f} "
              f"t={den.get('t', float('nan')):+5.2f} | "
              f"UTA a={uta.get('a', float('nan')):+6.2f} dev={uta.get('dev', float('nan')):+6.2f} "
              f"t={uta.get('t', float('nan')):+5.2f} | "
              f"visitor climb {r['aelev_km_coef']:+.3f} pts/km t={r['aelev_km_t']:+.2f}")

    # ---------------- travel / circadian / density, per era --------------
    tv = {}
    frames = {e: use[use.era == e] for e in alt}
    frames["HISTORICAL_NEW"] = use[use.season.isin(
        [s for s in seasons if s < "2019-20" and s != "2011-12"])]
    frames["CERTIFIED_5"] = use[use.season >= "2021-22"]
    frames["ALL"] = use
    for k, d in frames.items():
        if len(d) < 400:
            continue
        X = np.c_[np.ones(len(d)), d.h_b2b, d.a_b2b,
                  (d.h_travel_k - d.a_travel_k),
                  (d.h_tz - d.a_tz),
                  (d.h_3in4 - d.a_3in4)]
        b, cv, _, dof = ols(X, d.margin.to_numpy(float))
        names = ["const", "h_b2b", "a_b2b", "dtrav_kkm", "dtz_east", "d3in4"]
        tv[k] = {n: dict(coef=round(float(b[i]), 4),
                         t=round(float(b[i] / np.sqrt(cv[i, i])), 2))
                 for i, n in enumerate(names)}
        tv[k]["n"] = int(len(d))
    out["travel_by_era"] = tv
    print("\n=== TRAVEL / CIRCADIAN / DENSITY on the margin frame, per era ===")
    print("  (pre-registered signs: dtrav_kkm NEG, dtz_east POS, d3in4 NEG)")
    for k, r in tv.items():
        print(f"  {k:16s} n={r['n']:5d}  dtrav {r['dtrav_kkm']['coef']:+.4f} "
              f"t={r['dtrav_kkm']['t']:+5.2f} | dtz {r['dtz_east']['coef']:+.4f} "
              f"t={r['dtz_east']['t']:+5.2f} | d3in4 {r['d3in4']['coef']:+.4f} "
              f"t={r['d3in4']['t']:+5.2f}")

    json.dump(out, open(ROOT / "data" / "history_killed.json", "w"),
              indent=1, default=str)
    print("\nwrote data/history_killed.json")


if __name__ == "__main__":
    main()
