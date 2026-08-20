"""H2 test (docs/ADVERSE_HYPOTHESES.md): minutes-restriction returns.
Players returning from >=10-day absences (appearance gaps in
player_game_stats with >=3 team games missed, regular season) get capped
minutes in the first game back; the market knows tonight's cap, we assume
trailing minutes. Quantify the cap, then check whether games featuring a
returning >=25-trailing-min player are over-represented in our loss games
(d = L_us - L_mkt > 0.1).

Read-only DuckDB. Output: printed report only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import duckdb

DB = "/hdd/steveqin/sean_dev/nba_model/data/nba.duckdb"
CSV = "/hdd/steveqin/sean_dev/nba_model/data/capstone_pergame_tank.csv"
RNG = np.random.default_rng(23)


def ll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_dmean_ci(a, b, B=4000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ia = RNG.integers(0, len(a), (B, len(a)))
    ib = RNG.integers(0, len(b), (B, len(b)))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return tuple(np.percentile(d, [2.5, 97.5]))


def main():
    con = duckdb.connect(DB, read_only=True)
    app = con.execute("""
        SELECT s.player_id, s.game_id, s.team_id, s.seconds / 60.0 AS min,
               g.season, g.game_date
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, season, game_date FROM nba_games) g
          USING (game_id)
        WHERE s.game_id LIKE '002%' AND s.seconds > 0
        ORDER BY s.player_id, g.game_date""").df()
    sched = con.execute("""
        SELECT DISTINCT team_id, season, game_date FROM nba_games
        WHERE game_id LIKE '002%'""").df()
    con.close()
    app["game_date"] = pd.to_datetime(app.game_date)
    sched["game_date"] = pd.to_datetime(sched.game_date)
    team_dates = {k: np.sort(v.game_date.values)
                  for k, v in sched.groupby(["team_id", "season"])}

    events = []
    for (pid, season), gdf in app.groupby(["player_id", "season"], sort=False):
        if len(gdf) < 4:
            continue
        dates = gdf.game_date.values
        mins = gdf["min"].values
        tids = gdf.team_id.values
        gids = gdf.game_id.values
        for i in range(1, len(gdf)):
            gap = (dates[i] - dates[i - 1]) / np.timedelta64(1, "D")
            if gap < 10:
                continue
            td = team_dates.get((tids[i], season))
            if td is None:
                continue
            missed = int(((td > dates[i - 1]) & (td < dates[i])).sum())
            if missed < 3:
                continue
            j0 = max(0, i - 5)
            trail = mins[j0:i]
            if len(trail) < 3:
                continue
            events.append([pid, season, gids[i], str(dates[i])[:10],
                           float(gap), missed, float(trail.mean()),
                           float(mins[i])])
    ev = pd.DataFrame(events, columns=[
        "player_id", "season", "game_id", "game_date", "gap_days",
        "team_missed", "trail_min", "back_min"])
    ev["delta"] = ev.back_min - ev.trail_min
    print(f"return events (gap>=10d, >=3 team games missed): {len(ev)}")
    print(f"seasons: {ev.season.value_counts().sort_index().to_dict()}")

    def describe(sub, name):
        if not len(sub):
            return
        q = np.percentile(sub.delta, [10, 25, 50, 75, 90])
        print(f"\n{name}: n={len(sub)}  trailing {sub.trail_min.mean():.1f} -> "
              f"back {sub.back_min.mean():.1f}  delta mean {sub.delta.mean():+.2f}"
              f"  P10/25/50/75/90 {q.round(1)}")
        print(f"  share back < trail-5min: {(sub.delta < -5).mean():.2f}   "
              f"back < 0.75*trail: {(sub.back_min < 0.75 * sub.trail_min).mean():.2f}")

    describe(ev, "ALL returns")
    describe(ev[ev.trail_min >= 25], "trailing >= 25 min (rotation/stars)")
    describe(ev[ev.trail_min >= 30], "trailing >= 30 min")
    describe(ev[(ev.trail_min >= 25) & (ev.gap_days >= 21)],
             ">=25 min & gap >= 21d")
    # ramp: minutes in appearances 1..3 after long absence for >=25 group
    print("\n(cap persistence) mean minutes by appearance since return, "
          "trailing>=25 group: computed from next rows in app")
    stars = ev[ev.trail_min >= 25]
    key = app.set_index(["player_id", "season"])
    ramps = {1: [], 2: [], 3: []}
    for r in stars.itertuples():
        gdf = key.loc[(r.player_id, r.season)]
        if isinstance(gdf, pd.Series):
            continue
        gdf = gdf.reset_index()
        pos = gdf.index[gdf.game_id == r.game_id]
        if not len(pos):
            continue
        i = int(pos[0])
        for k in (1, 2, 3):
            if i + k - 1 < len(gdf):
                ramps[k].append(float(gdf["min"].iloc[i + k - 1]) - r.trail_min)
    for k in (1, 2, 3):
        v = np.array(ramps[k])
        print(f"  appearance +{k}: n={len(v)} mean delta vs trailing "
              f"{v.mean():+.2f} (median {np.median(v):+.1f})")

    # ---- over-representation in loss games ----
    df = pd.read_csv(CSV, dtype={"game_id": str})
    df["game_id"] = df.game_id.str.zfill(10)
    df["d"] = ll(df.p_us.values, df.y.values) - ll(df.p_mkt.values, df.y.values)
    ret_g = set(ev[ev.trail_min >= 25].game_id)
    df["ret25"] = df.game_id.isin(ret_g)
    df["loss"] = df.d > 0.1
    n11 = int((df.ret25 & df.loss).sum())
    n10 = int((df.ret25 & ~df.loss).sum())
    n01 = int((~df.ret25 & df.loss).sum())
    n00 = int((~df.ret25 & ~df.loss).sum())
    print("\n" + "=" * 72)
    print(f"LOSS-GAME over-representation (loss = d > 0.1, n_loss={n11 + n01})")
    print(f"  return>=25 games: {int(df.ret25.sum())} of {len(df)} "
          f"({df.ret25.mean():.3f})")
    print(f"  P(ret25 | loss) = {n11 / (n11 + n01):.3f}   "
          f"P(ret25 | not loss) = {n10 / (n10 + n00):.3f}")
    orat = (n11 * n00) / max(1, n10 * n01)
    try:
        from scipy.stats import fisher_exact
        pv = fisher_exact([[n11, n10], [n01, n00]])[1]
    except Exception:
        pv = np.nan
    print(f"  odds ratio {orat:.2f} (Fisher p={pv:.3f})")
    lo, hi = boot_dmean_ci(df.d[df.ret25], df.d[~df.ret25])
    print(f"  mean d: ret25 {df.d[df.ret25].mean():+.4f} vs "
          f"other {df.d[~df.ret25].mean():+.4f}  "
          f"diff CI95({lo:+.4f},{hi:+.4f})")
    # same but restricted to divergence-prone regions
    opp = (df.p_us - 0.5) * (df.p_mkt - 0.5) < 0
    print(f"  within opposite-side games: ret25 share {df.ret25[opp].mean():.3f}"
          f" vs elsewhere {df.ret25[~opp].mean():.3f}; "
          f"d(ret25&opp) {df.d[opp & df.ret25].mean():+.4f} "
          f"(n={int((opp & df.ret25).sum())})")


if __name__ == "__main__":
    main()
