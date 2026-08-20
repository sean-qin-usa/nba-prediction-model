#!/usr/bin/env python
"""
STUDY 2 -- WHAT TRAVELS IN A TRADE (decomposition).

Descriptive/historical study (full-hindsight season means are used deliberately
and labeled as such -- this is NOT a PIT-predictive artifact).

For rotation players (>=20 trailing min/g over their last 15 games at team A)
who change teams midseason (>=10 games at A that season, then appear for B):

  1. Per stat family, compare last-15-at-A vs first-5/10/15-at-B windows,
     expressed as deviation from the player's own season-long (sum-based) mean.
  2. Travel coefficient table: arrival deviation, delta vs A-form, and a
     persistence slope (devB ~ devA across events). Player-clustered bootstrap CIs.
  3. Destination-context splits: new-team talent (minutes-weighted DARKO dpm of
     B teammates at arrival) and B team pace (box-estimated possessions/48).
  4. DARKO tracking speed post-trade: convergence curve of dpm toward its
     settled (t0+75d) value, and an "edge window": how long early realized
     production (first-5-at-B deviation) still predicts FUTURE darko drift.

DB access is read_only=True. Outputs printed to stdout + CSVs in --outdir.
"""

import argparse
import numpy as np
import pandas as pd
import duckdb

RNG = np.random.default_rng(7)
DB = "/hdd/steveqin/sean_dev/nba_model/data/nba.duckdb"

# ----------------------------------------------------------------------------
# Sum-based stat definitions over a window of games.
# Each entry: name -> (numerator_fn, denominator_fn) computed on summed cols.
# ----------------------------------------------------------------------------
def _min(d):  # minutes
    return d["seconds"] / 60.0

STATS = {
    "pts36":    (lambda d: 36.0 * d["pts"],               _min),
    "ts_pct":   (lambda d: d["pts"],                      lambda d: 2.0 * (d["fga"] + 0.44 * d["fta"])),
    "fga36":    (lambda d: 36.0 * d["fga"],               _min),   # usage proxy
    "ast36":    (lambda d: 36.0 * d["ast"],               _min),
    "tov36":    (lambda d: 36.0 * d["tov"],               _min),
    "reb36":    (lambda d: 36.0 * (d["oreb"] + d["dreb"]), _min),
    "stlblk36": (lambda d: 36.0 * (d["stl"] + d["blk"]),  _min),
    "ft_rate":  (lambda d: d["fta"],                      lambda d: d["fga"]),
    "fg3_pct":  (lambda d: d["fg3m"],                     lambda d: d["fg3a"]),
}
SUM_COLS = ["seconds", "pts", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm",
            "ast", "tov", "oreb", "dreb", "stl", "blk", "pf"]


def window_stats(df):
    """Sum-based stat values for a window (df of per-game rows)."""
    s = {c: float(df[c].sum()) for c in SUM_COLS}
    out = {}
    for name, (num, den) in STATS.items():
        d = den(s)
        out[name] = num(s) / d if d > 0 else np.nan
    out["gmsc36"] = gmsc36(s)
    return out


def gmsc36(s):
    """Game-score composite per 36 (for DARKO comparison)."""
    if s["seconds"] <= 0:
        return np.nan
    g = (s["pts"] + 0.4 * s["fgm"] - 0.7 * s["fga"] - 0.4 * (s["fta"] - s["ftm"])
         + 0.7 * s["oreb"] + 0.3 * s["dreb"] + s["stl"] + 0.7 * s["ast"]
         + 0.7 * s["blk"] - 0.4 * s["pf"] - s["tov"])
    return 36.0 * g / (s["seconds"] / 60.0)


def cluster_boot(values, clusters, n_boot=2000, stat=np.nanmean):
    """Bootstrap CI clustered by `clusters` (player_id). Returns (est, lo, hi)."""
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    ok = ~np.isnan(values)
    values, clusters = values[ok], clusters[ok]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    uniq = np.unique(clusters)
    idx_by = {c: np.where(clusters == c)[0] for c in uniq}
    est = stat(values)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[c] for c in pick])
        reps[b] = stat(values[idx])
    return est, np.nanpercentile(reps, 2.5), np.nanpercentile(reps, 97.5)


def cluster_boot_slope(x, y, clusters, n_boot=2000):
    """OLS slope of y~x (with intercept), player-clustered bootstrap CI."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    clusters = np.asarray(clusters)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y, clusters = x[ok], y[ok], clusters[ok]
    if len(x) < 10:
        return np.nan, np.nan, np.nan
    def slope(xx, yy):
        vx = np.var(xx)
        return np.cov(xx, yy)[0, 1] / vx if vx > 0 else np.nan
    uniq = np.unique(clusters)
    idx_by = {c: np.where(clusters == c)[0] for c in uniq}
    est = slope(x, y)
    reps = np.empty(n_boot)
    for b in range(n_boot):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[c] for c in pick])
        reps[b] = slope(x[idx], y[idx])
    return est, np.nanpercentile(reps, 2.5), np.nanpercentile(reps, 97.5)


# ----------------------------------------------------------------------------
def load_data(con):
    pgs = con.execute("""
        select g.season, g.game_date, p.*
        from player_game_stats p
        join (select distinct season, game_id, game_date from nba_games) g
          using (game_id)
        where substr(p.game_id,1,3) = '002'      -- regular season only
          and p.seconds is not null and p.seconds > 0
        order by p.player_id, g.game_date
    """).df()
    darko = con.execute("""
        select player_id, date, dpm from darko_history
        where date >= '2022-08-01'
    """).df()
    darko["date"] = pd.to_datetime(darko["date"])
    # box-based team pace: possessions/48 per team-game, then season mean
    pace = con.execute("""
        with tg as (
          select p.game_id, p.team_id, g.season,
                 sum(p.fga) + 0.44*sum(p.fta) - sum(p.oreb) + sum(p.tov) as poss,
                 sum(p.seconds)/5.0/60.0 as team_min
          from player_game_stats p
          join (select distinct season, game_id from nba_games) g using (game_id)
          where substr(p.game_id,1,3) = '002'
          group by 1,2,3
        )
        select season, team_id, avg(48.0 * poss / team_min) as pace48
        from tg group by 1,2
    """).df()
    return pgs, darko, pace


def detect_events(pgs):
    """Midseason team-change events with windows attached."""
    events = []
    pgs = pgs.sort_values(["player_id", "season", "game_date"])
    for (pid, season), d in pgs.groupby(["player_id", "season"], sort=False):
        d = d.reset_index(drop=True)
        teams = d["team_id"].values
        # change points
        for i in range(1, len(d)):
            if teams[i] != teams[i - 1]:
                team_a, team_b = teams[i - 1], teams[i]
                games_at_a = int((teams[:i] == team_a).sum())
                if games_at_a < 10:
                    continue
                a_rows = d.iloc[:i][d.iloc[:i]["team_id"] == team_a].tail(15)
                # rotation filter: trailing min/g over last-15-at-A >= 20
                if a_rows["seconds"].mean() / 60.0 < 20.0:
                    continue
                b_rows = d.iloc[i:][d.iloc[i:]["team_id"] == team_b]
                if len(b_rows) < 5:
                    continue
                events.append(dict(
                    player_id=pid, season=season, team_a=team_a, team_b=team_b,
                    trade_date=d.iloc[i]["game_date"],
                    n_games_a=games_at_a, n_games_b=len(b_rows),
                    a_rows=a_rows, b_rows=b_rows, season_rows=d,
                ))
    return events


def build_event_table(events):
    rows = []
    for ev in events:
        season_stats = window_stats(ev["season_rows"])   # HINDSIGHT full-season mean
        a15 = window_stats(ev["a_rows"])
        r = dict(player_id=ev["player_id"], season=ev["season"],
                 team_a=ev["team_a"], team_b=ev["team_b"],
                 trade_date=ev["trade_date"], n_games_a=ev["n_games_a"],
                 n_games_b=ev["n_games_b"],
                 min_pg_a=ev["a_rows"]["seconds"].mean() / 60.0,
                 min_pg_b10=ev["b_rows"].head(10)["seconds"].mean() / 60.0)
        for name in list(STATS) + ["gmsc36"]:
            r[f"season_{name}"] = season_stats[name]
            r[f"rawA15_{name}"] = a15[name]
            # NOTE: season mean includes both windows, so window deviations are
            # zero-sum-coupled (devA vs devB anti-correlate mechanically).
            # Use raw levels / raw deltas for persistence; deviations for dips.
            r[f"devA15_{name}"] = a15[name] - season_stats[name]
        for n in (5, 10, 15):
            bw = ev["b_rows"].head(n)
            if len(bw) >= n:
                ws = window_stats(bw)
                for name in list(STATS) + ["gmsc36"]:
                    r[f"rawB{n}_{name}"] = ws[name]
                    r[f"devB{n}_{name}"] = ws[name] - season_stats[name]
        # ramp buckets 6-10, 11-15
        for lab, sl in (("B6_10", slice(5, 10)), ("B11_15", slice(10, 15))):
            bw = ev["b_rows"].iloc[sl]
            if len(bw) == 5:
                ws = window_stats(bw)
                for name in list(STATS) + ["gmsc36"]:
                    r[f"dev{lab}_{name}"] = ws[name] - season_stats[name]
        rows.append(r)
    return pd.DataFrame(rows)


def darko_at(darko_p, date, tol_days=10):
    """dpm at nearest date <= date (within tol); darko_p sorted by date."""
    sub = darko_p[darko_p["date"] <= date]
    if len(sub) == 0:
        return np.nan
    row = sub.iloc[-1]
    if (date - row["date"]).days > tol_days:
        return np.nan
    return float(row["dpm"])


def darko_analysis(ev_df, events, darko):
    """Convergence of dpm to settled post-trade value + edge-window test."""
    darko = darko.sort_values(["player_id", "date"])
    dgrp = {pid: g for pid, g in darko.groupby("player_id")}
    ks = [0, 7, 14, 21, 30, 45, 60, 75]
    rows = []
    for ev in events:
        pid = ev["player_id"]
        if pid not in dgrp:
            continue
        dp = dgrp[pid]
        t0 = pd.Timestamp(ev["trade_date"])
        pre = darko_at(dp, t0 - pd.Timedelta(days=7))
        traj = {k: darko_at(dp, t0 + pd.Timedelta(days=k)) for k in ks}
        if np.isnan(pre) or np.isnan(traj[75]):
            continue
        rows.append(dict(player_id=pid, season=ev["season"],
                         trade_date=ev["trade_date"], pre=pre,
                         **{f"d{k}": traj[k] for k in ks}))
    keep = ["player_id", "season", "trade_date", "devB5_gmsc36", "devB10_gmsc36",
            "rawA15_gmsc36", "rawB5_gmsc36", "rawB10_gmsc36"]
    dk = pd.DataFrame(rows).merge(ev_df[keep],
                                  on=["player_id", "season", "trade_date"], how="left")
    dk["settle_change"] = dk["d75"] - dk["pre"]
    # baseline-free early realized change (no hindsight season mean involved)
    dk["early_change"] = dk["rawB5_gmsc36"] - dk["rawA15_gmsc36"]
    return dk, ks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data/scratch")
    ap.add_argument("--nboot", type=int, default=2000)
    args = ap.parse_args()

    con = duckdb.connect(DB, read_only=True)
    pgs, darko, pace = load_data(con)
    events = detect_events(pgs)
    ev_df = build_event_table(events)
    print(f"\n=== EVENTS: {len(ev_df)} midseason team-change events, "
          f"{ev_df['player_id'].nunique()} unique players ===")
    print(ev_df.groupby("season").size().to_string())
    print("NOTE: season-long means use FULL-SEASON HINDSIGHT (descriptive study).")

    # ---------------- travel coefficient table -----------------------------
    trav = []
    cl = ev_df["player_id"].values
    for name in STATS:
        rec = dict(stat=name, season_mean=float(ev_df[f"season_{name}"].mean()))
        for col, lab in [(f"devA15_{name}", "A15"), (f"devB5_{name}", "B5"),
                         (f"devB10_{name}", "B10"), (f"devB15_{name}", "B15")]:
            e, lo, hi = cluster_boot(ev_df[col], cl, args.nboot)
            rec[f"{lab}"], rec[f"{lab}_lo"], rec[f"{lab}_hi"] = e, lo, hi
        delta = ev_df[f"rawB10_{name}"] - ev_df[f"rawA15_{name}"]  # baseline-free
        e, lo, hi = cluster_boot(delta, cl, args.nboot)
        rec["dB10_minus_A15"], rec["delta_lo"], rec["delta_hi"] = e, lo, hi
        # TRAVEL COEFFICIENT: how well the pre-trade LEVEL predicts the
        # post-trade LEVEL (slope + corr of rawB10 on rawA15, across events).
        s, slo, shi = cluster_boot_slope(ev_df[f"rawA15_{name}"],
                                         ev_df[f"rawB10_{name}"], cl, args.nboot)
        rec["travel_slope"], rec["slope_lo"], rec["slope_hi"] = s, slo, shi
        xy = ev_df[[f"rawA15_{name}", f"rawB10_{name}"]].dropna()
        rec["travel_corr"] = float(np.corrcoef(xy.iloc[:, 0], xy.iloc[:, 1])[0, 1])
        trav.append(rec)
    trav = pd.DataFrame(trav)
    pd.set_option("display.width", 250)
    print("\n=== TRAVEL TABLE (player-clustered 95% CI) ===")
    print("devB* = deviation from own full-season mean (hindsight; zero-sum caveat);")
    print("dB10_minus_A15 = raw first-10-at-B minus raw last-15-at-A (baseline-free);")
    print("travel_slope/corr = rawB10 regressed on rawA15 across events (level persistence).")
    cols = ["stat", "season_mean", "A15", "B5", "B5_lo", "B5_hi", "B10", "B10_lo",
            "B10_hi", "B15", "dB10_minus_A15", "delta_lo", "delta_hi",
            "travel_slope", "slope_lo", "slope_hi", "travel_corr"]
    print(trav[cols].round(3).to_string(index=False))

    # ramp shape (5-game buckets at B)
    print("\n=== RAMP AT B (5-game buckets, dev from season mean) ===")
    ramp = []
    for name in STATS:
        r = {"stat": name}
        for col, lab in [(f"devB5_{name}", "g1_5"), (f"devB6_10_{name}", "g6_10"),
                         (f"devB11_15_{name}", "g11_15")]:
            e, lo, hi = cluster_boot(ev_df.get(col, pd.Series(np.nan, index=ev_df.index)),
                                     cl, args.nboot)
            r[lab], r[f"{lab}_lo"], r[f"{lab}_hi"] = e, lo, hi
        ramp.append(r)
    ramp = pd.DataFrame(ramp)
    print(ramp.round(3).to_string(index=False))

    # minutes at A vs B (context)
    e, lo, hi = cluster_boot(ev_df["min_pg_b10"] - ev_df["min_pg_a"], cl, args.nboot)
    print(f"\nMinutes/g change (first-10-at-B minus last-15-at-A): "
          f"{e:+.2f} [{lo:+.2f}, {hi:+.2f}]")

    # ---------------- destination context -----------------------------------
    # B-team talent: minutes-weighted mean darko dpm of B teammates in the
    # player's first-10-at-B games, darko as of trade date.
    darko_s = darko.sort_values(["player_id", "date"])
    dgrp = {pid: g for pid, g in darko_s.groupby("player_id")}
    talents = []
    for ev in events:
        t0 = pd.Timestamp(ev["trade_date"])
        bgids = ev["b_rows"]["game_id"].head(10).tolist()
        mates = pgs[(pgs["game_id"].isin(bgids)) & (pgs["team_id"] == ev["team_b"])
                    & (pgs["player_id"] != ev["player_id"])]
        agg = mates.groupby("player_id")["seconds"].sum()
        w, dv = [], []
        for mpid, sec in agg.items():
            if mpid in dgrp:
                v = darko_at(dgrp[mpid], t0)
                if not np.isnan(v):
                    w.append(sec); dv.append(v)
        talents.append(np.average(dv, weights=w) if w else np.nan)
    ev_df["b_talent"] = talents
    pace_map = {(r.season, r.team_id): r.pace48 for r in pace.itertuples()}
    ev_df["b_pace"] = [pace_map.get((s, t), np.nan)
                       for s, t in zip(ev_df["season"], ev_df["team_b"])]
    ev_df["a_pace"] = [pace_map.get((s, t), np.nan)
                       for s, t in zip(ev_df["season"], ev_df["team_a"])]

    print("\n=== DESTINATION CONTEXT SPLITS (devB10, player-clustered CI) ===")
    for ctx in ["b_talent", "b_pace"]:
        med = ev_df[ctx].median()
        lo_m = ev_df[ctx] <= med
        print(f"\n-- split by {ctx} (median {med:.2f}) --")
        for name in ["pts36", "fga36", "ts_pct", "ast36", "gmsc36"]:
            col = f"devB10_{name}"
            e1, l1, h1 = cluster_boot(ev_df.loc[lo_m, col], cl[lo_m.values], args.nboot)
            e2, l2, h2 = cluster_boot(ev_df.loc[~lo_m, col], cl[(~lo_m).values], args.nboot)
            print(f"  {name:8s} low: {e1:+.3f} [{l1:+.3f},{h1:+.3f}]   "
                  f"high: {e2:+.3f} [{l2:+.3f},{h2:+.3f}]")
        # slope of devB10_fga36 & pts36 on context (standardized context)
        z = (ev_df[ctx] - ev_df[ctx].mean()) / ev_df[ctx].std()
        for name in ["fga36", "pts36", "ts_pct"]:
            s, slo, shi = cluster_boot_slope(z, ev_df[f"devB10_{name}"], cl, args.nboot)
            print(f"  slope devB10_{name} per 1sd {ctx}: {s:+.3f} [{slo:+.3f},{shi:+.3f}]")

    # ---------------- DARKO tracking speed ----------------------------------
    dk, ks = darko_analysis(ev_df, events, darko)
    print(f"\n=== DARKO POST-TRADE TRACKING (n={len(dk)} events with darko coverage) ===")
    moved = dk[np.abs(dk["settle_change"]) >= 0.3].copy()
    print(f"Events with |settled dpm change| >= 0.3: {len(moved)} "
          f"(mean |change| {moved['settle_change'].abs().mean():.2f})")
    print("\nConvergence fraction toward settled (t0+75d) dpm value:")
    fr_rows = []
    for k in ks:
        frac = (moved[f"d{k}"] - moved["pre"]) / moved["settle_change"]
        med = float(np.nanmedian(frac))
        fr_rows.append((k, med))
        print(f"  day {k:>2}: median fraction {med:+.2f}")
    # days to 50% / 80%
    for target in (0.5, 0.8):
        day = next((k for k, f in fr_rows if f >= target), None)
        print(f"  -> first grid day reaching {int(target*100)}%: {day}")

    def corr_edge(pred_col, label):
        print(f"\nEdge window: corr({label}, FUTURE darko drift d75-dk):")
        clc = dk["player_id"].values
        for k in [0, 7, 14, 21, 30, 45]:
            drift = dk["d75"] - dk[f"d{k}"]
            x, y = dk[pred_col].values, drift.values
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() < 15:
                continue
            xx, yy, cc = x[ok], y[ok], clc[ok]
            uniq = np.unique(cc)
            idx_by = {c: np.where(cc == c)[0] for c in uniq}
            est = np.corrcoef(xx, yy)[0, 1]
            reps = []
            for b in range(args.nboot):
                pick = RNG.choice(uniq, size=len(uniq), replace=True)
                idx = np.concatenate([idx_by[c] for c in pick])
                if np.std(xx[idx]) > 0 and np.std(yy[idx]) > 0:
                    reps.append(np.corrcoef(xx[idx], yy[idx])[0, 1])
            lo, hi = np.percentile(reps, [2.5, 97.5])
            print(f"  from day {k:>2}: corr {est:+.3f} [{lo:+.3f},{hi:+.3f}]  (n={ok.sum()})")

    corr_edge("early_change", "gmsc36 first-5-at-B minus last-15-at-A (baseline-free)")
    corr_edge("devB5_gmsc36", "gmsc36 first-5-at-B dev from season mean (hindsight)")

    # save
    import os
    os.makedirs(args.outdir, exist_ok=True)
    ev_df.to_csv(f"{args.outdir}/ev_tradedecomp_events.csv", index=False)
    trav.to_csv(f"{args.outdir}/ev_tradedecomp_travel.csv", index=False)
    dk.to_csv(f"{args.outdir}/ev_tradedecomp_darko.csv", index=False)
    print(f"\nCSVs written to {args.outdir}")


if __name__ == "__main__":
    main()
