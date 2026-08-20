"""Task 2: bench-split strength (top-5 vs 6-10) from lineup_stints, walk-forward PIT.

Construction (all trailing, within season, strictly prior to the game):
- top-5 of a team as of a game = 5 players with most cumulative seconds in that
  team's PRIOR games this season (from player_game_stats, full coverage).
- every lineup stint is classified by k = |on-floor lineup ∩ top-5 at that time|:
  S (k>=4, starter-heavy), B (k<=2, bench-heavy), M (k==3, mixed).
- per team-date: expanding sums of (seconds, pts for, pts against) per class over
  all covered prior games; net rating per 48 by class, expressed RELATIVE to the
  team's own overall net48 and EB-shrunk with w = sec/(sec + 36000) (600 min).
Features (home minus away): bench_rel, starter_rel, sb_gap (=S-B), bench_share.
Coverage: lineup_stints has 1191/764/1225 of 1230 regular-season games per season;
uncovered games simply don't contribute to the trailing sums.

Output: pkl of per-game bench features in scratchpad.
"""
import os
import sys
import numpy as np
import pandas as pd
import duckdb

DB = "/hdd/steveqin/sean_dev/nba_model/data/nba.duckdb"
OUTDIR = os.environ.get(
    "RW_OUT",
    "data/scratch",
)
SHRINK_SEC = 36000.0  # 600 minutes


def main():
    con = duckdb.connect(DB, read_only=True)
    meta = con.execute("""
        select season, game_id, game_date, team_abbrev as team, team_id, is_home
        from nba_games where game_id like '002%'
    """).fetchdf()
    meta["game_date"] = pd.to_datetime(meta.game_date)
    st = con.execute("""
        select game_id, home_team_id, away_team_id, seconds, home_lineup, away_lineup,
               home_pts, away_pts
        from lineup_stints where game_id like '002%'
    """).fetchdf()
    ps = con.execute("""
        select s.game_id, s.player_id, s.team_id, s.seconds
        from player_game_stats s where s.game_id like '002%'
    """).fetchdf()

    # long stint table: one row per stint per side
    home = st.rename(columns={"home_team_id": "team_id", "home_lineup": "lineup",
                              "home_pts": "pts_for", "away_pts": "pts_against"})[
        ["game_id", "team_id", "seconds", "lineup", "pts_for", "pts_against"]]
    away = st.rename(columns={"away_team_id": "team_id", "away_lineup": "lineup",
                              "away_pts": "pts_for", "home_pts": "pts_against"})[
        ["game_id", "team_id", "seconds", "lineup", "pts_for", "pts_against"]]
    long = pd.concat([home, away], ignore_index=True)
    long = long.merge(meta[["season", "game_id", "game_date", "team", "team_id"]],
                      on=["game_id", "team_id"], how="inner")

    # player cumulative seconds per (season, team) walk-forward
    ps = ps.merge(meta[["season", "game_id", "game_date", "team", "team_id"]],
                  on=["game_id", "team_id"], how="inner")
    ps = ps.sort_values(["season", "team", "game_date", "game_id"])

    # per (season, team, game): dict player->sec
    game_secs = ps.groupby(["season", "team", "game_date", "game_id"]).apply(
        lambda d: dict(zip(d.player_id, d.seconds)), include_groups=False).rename("psec")
    game_secs = game_secs.reset_index().sort_values(["season", "team", "game_date", "game_id"])

    top5_map = {}  # (team, game_id) -> frozenset of top-5 player ids entering the game
    for (season, team), sub in game_secs.groupby(["season", "team"]):
        cum = {}
        for _, r in sub.iterrows():
            if len(cum) >= 5 and sub.index.get_loc(r.name) is not None:
                pass
            if len(cum) >= 5:
                top5 = frozenset(sorted(cum, key=cum.get, reverse=True)[:5])
            else:
                top5 = frozenset()
            top5_map[(team, r.game_id)] = top5
            for p, s in r.psec.items():
                cum[p] = cum.get(p, 0) + s

    def classify(row):
        t5 = top5_map.get((row.team, row.game_id))
        if not t5:
            return "U"
        ids = {int(x) for x in row.lineup.split(",") if x}
        k = len(ids & t5)
        return "S" if k >= 4 else ("B" if k <= 2 else "M")

    long["cls"] = [classify(r) for r in long.itertuples()]
    print(long.cls.value_counts(), file=sys.stderr)
    long = long[long.cls != "U"]

    # per team-game per class sums
    pg = long.groupby(["season", "team", "game_date", "game_id", "cls"]).agg(
        sec=("seconds", "sum"), pf=("pts_for", "sum"), pa=("pts_against", "sum")
    ).reset_index()
    wide = pg.pivot_table(index=["season", "team", "game_date", "game_id"],
                          columns="cls", values=["sec", "pf", "pa"],
                          aggfunc="sum", fill_value=0.0).reset_index()
    wide.columns = ["".join(map(str, c)).strip() for c in wide.columns]
    for c in ["secS", "secB", "secM", "pfS", "pfB", "pfM", "paS", "paB", "paM"]:
        if c not in wide:
            wide[c] = 0.0
    wide = wide.sort_values(["season", "team", "game_date", "game_id"]).reset_index(drop=True)
    grp = wide.groupby(["season", "team"], sort=False)
    for c in ["secS", "secB", "secM", "pfS", "pfB", "pfM", "paS", "paB", "paM"]:
        wide["c_" + c] = grp[c].transform(lambda s: s.shift(1).expanding(1).sum())

    tot_sec = wide.c_secS + wide.c_secB + wide.c_secM
    tot_net = ((wide.c_pfS + wide.c_pfB + wide.c_pfM) -
               (wide.c_paS + wide.c_paB + wide.c_paM)) / tot_sec.clip(lower=1) * 2880
    out = wide[["season", "team", "game_date", "game_id"]].copy()
    for cls, name in (("S", "starter"), ("B", "bench")):
        sec = wide["c_sec" + cls]
        net = (wide["c_pf" + cls] - wide["c_pa" + cls]) / sec.clip(lower=1) * 2880
        rel = net - tot_net
        w = sec / (sec + SHRINK_SEC)
        out[name + "_rel"] = w * rel
        out[name + "_net_abs"] = (sec / (sec + SHRINK_SEC)) * net
        out[name + "_sec"] = sec
    out["sb_gap"] = out.starter_rel - out.bench_rel
    out["bench_share"] = wide.c_secB / tot_sec.clip(lower=1)
    out.loc[tot_sec < 3600 * 5, ["starter_rel", "bench_rel", "sb_gap", "bench_share",
                                 "starter_net_abs", "bench_net_abs"]] = np.nan

    # map onto ALL games (incl. uncovered): per team, forward-fill the trailing
    # value onto each schedule row (feature = last computed trailing state)
    sched = meta[["season", "game_id", "game_date", "team", "is_home"]].sort_values(
        ["season", "team", "game_date", "game_id"])
    # out rows exist only for covered games; the trailing state entering game g is
    # based on games strictly before g regardless of whether g is covered
    state_cols = ["starter_rel", "bench_rel", "sb_gap", "bench_share",
                  "starter_net_abs", "bench_net_abs", "bench_sec"]
    merged = sched.merge(out.drop(columns=["game_date"]), on=["season", "team", "game_id"],
                         how="left")
    merged = merged.sort_values(["season", "team", "game_date", "game_id"])
    # For uncovered games we need state as of that date: ffill within team-season.
    merged[state_cols] = merged.groupby(["season", "team"], sort=False)[state_cols].ffill()

    H = merged[merged.is_home][["game_id"] + state_cols].rename(
        columns={c: "bs_" + c + "_H" for c in state_cols})
    A = merged[~merged.is_home][["game_id"] + state_cols].rename(
        columns={c: "bs_" + c + "_A" for c in state_cols})
    game = H.merge(A, on="game_id", how="inner")
    path = os.path.join(OUTDIR, "regimec_bench.pkl")
    game.to_pickle(path)
    print(f"wrote {path}: {game.shape}", file=sys.stderr)
    print(game.describe().T[["count", "mean", "std"]], file=sys.stderr)


if __name__ == "__main__":
    main()
