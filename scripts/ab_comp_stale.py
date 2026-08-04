"""ABSENCE AUDIT — decompose comp's EXCLUDED-BUT-PLAYED leg.

A player who plays tonight can be missing from comp.strength() for two reasons:

  (S1) REFIT STALENESS. comp is rebuilt WEEKLY (prod_by_season / predict_today
       refit cadence). `last_played` and the team assignment are frozen at the
       refit cutoff. A player whose return (or trade) happened AFTER the refit
       is invisible until the next one, even though the return is a PIT-
       observable box score. Rebuilding comp at the GAME DATE fixes this with
       strictly-prior information only — no oracle, no new feed.

  (S2) GENUINE ABSENCE. Even at the game date his last >=12-min game is more
       than ROSTER_DAYS=12 days old, so the recency cut drops him. Recovering
       him requires either a longer window or the availability feed.

Same split for the WRONG_TEAM leg (post-trade assignment).

READ-ONLY. Writes data/ab_comp_stale.json.
"""
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from nbapred.db import connect
from nbapred.model.composition import ROSTER_DAYS, CompositionModel

SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")
SCALE, W_COMP = 7.2, 0.5


def _d(x):
    return x.date() if hasattr(x, "date") else x


class FastComp:
    """Vectorised CompositionModel: per-player (team, trail_min, last_played)
    as of an arbitrary cutoff, from the same universe (002, seconds>=720,
    last 10 games, arg_max team by date). Verified against CompositionModel."""

    def __init__(self, con):
        df = con.execute("""
            SELECT s.player_id, s.team_id, s.seconds/60.0 m, g.game_date
            FROM player_game_stats s
            JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
            WHERE s.game_id LIKE '002%' AND s.seconds >= 720
            ORDER BY s.player_id, g.game_date""").fetchdf()
        df["game_date"] = [_d(x) for x in df["game_date"]]
        self.by = {}
        for pid, g in df.groupby("player_id"):
            self.by[int(pid)] = (np.array([d.toordinal() for d in g.game_date]),
                                 g.team_id.to_numpy(np.int64),
                                 g.m.to_numpy(float))

    def asof(self, pid, cutoff):
        """(team_id, trail_min, last_played_ordinal) strictly before cutoff."""
        e = self.by.get(int(pid))
        if e is None:
            return None
        d, t, m = e
        i = np.searchsorted(d, cutoff.toordinal(), side="left")
        if i == 0:
            return None
        lo = max(0, i - 10)
        # arg_max(team_id, game_date) over the WHOLE prior history (SQL groups
        # after the rn<=10 filter, so the argmax is over the last-10 window)
        return int(t[i - 1]), float(m[lo:i].mean()), int(d[i - 1])


def main():
    con = connect(read_only=True)
    fc = FastComp(con)

    # --- fidelity check: FastComp == CompositionModel on 3 cutoffs -----------
    checks = []
    for cd in (__import__("datetime").date(2022, 1, 12),
               __import__("datetime").date(2024, 3, 5),
               __import__("datetime").date(2026, 2, 18)):
        cm = CompositionModel(con, before=cd)
        bad = 0
        for pid, p in cm.players.items():
            r = fc.asof(pid, cd)
            if (r is None or r[0] != p["team_id"]
                    or abs(r[1] - p["trail_min"]) > 1e-9
                    or r[2] != p["last_played"].toordinal()):
                bad += 1
        checks.append((str(cd), len(cm.players), bad))
        print("FIDELITY", checks[-1], flush=True)
    assert all(b == 0 for _, _, b in checks), "FastComp != CompositionModel"

    pm = con.execute("""
        SELECT s.game_id, s.team_id, s.player_id, s.seconds, g.game_date, g.season
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '002%'""").fetchdf()
    pm["game_date"] = [_d(x) for x in pm["game_date"]]
    played, mins = defaultdict(set), {}
    for r in pm.itertuples():
        if r.seconds and r.seconds > 0:
            played[(r.game_id, int(r.team_id))].add(int(r.player_id))
            mins[(r.game_id, int(r.player_id))] = float(r.seconds) / 60.0

    rows = []
    for season in SEASONS:
        meta = con.execute("""
            SELECT game_id, team_id, team_abbrev, matchup, wl, game_date
            FROM nba_games WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL
            ORDER BY game_date""", [season]).fetchdf()
        by, order = {}, []
        for x in meta.itertuples():
            if x.game_id not in by:
                order.append(x.game_id)
            by.setdefault(x.game_id, []).append(x)
        comp = None
        last = None
        for gid in order:
            recs = by[gid]
            if len(recs) != 2:
                continue
            gd = _d(recs[0].game_date)
            if last is None or (gd - last).days >= 7:
                comp = CompositionModel(con, before=gd)
                last = gd
            for x in recs:
                t = int(x.team_id)
                host = (x.matchup.split("@")[-1].strip() if "@" in x.matchup
                        else x.matchup.split("vs.")[0].strip())
                sgn = 1.0 if x.team_abbrev == host else -1.0
                for pid in played.get((gid, t), set()):
                    p = comp.players.get(pid)
                    m = mins.get((gid, pid), 0.0)
                    fr = fc.asof(pid, gd)
                    # what the PRODUCTION (weekly) comp does for this player
                    if p is None:
                        prod_in, prod_team, prod_tm, prod_tal = False, None, 0.0, 0.0
                    else:
                        dsl = (gd - p["last_played"]).days
                        prod_team = p["team_id"]
                        prod_tm, prod_tal = p["trail_min"], p["talent"]
                        prod_in = (prod_team == t and dsl <= ROSTER_DAYS)
                    # what a GAME-DATE-FRESH comp would do (same talent snapshot)
                    if fr is None:
                        fresh_in = False
                        fresh_team, fresh_tm, fdsl = None, 0.0, 999
                    else:
                        fresh_team, fresh_tm, lpo = fr
                        fdsl = gd.toordinal() - lpo
                        fresh_in = (fresh_team == t and fdsl <= ROSTER_DAYS)
                    if prod_in and fresh_in:
                        continue          # nothing to explain
                    rows.append((season, gid, t, sgn, pid, m, prod_tal,
                                 int(prod_in), int(fresh_in),
                                 prod_tm, fresh_tm, fdsl,
                                 int(p is not None and p["team_id"] != t)))
        print(season, "cum", len(rows), flush=True)
    con.close()

    df = pd.DataFrame(rows, columns=[
        "season", "game_id", "team_id", "sgn", "player_id", "real_min",
        "talent", "prod_in", "fresh_in", "prod_tm", "fresh_tm", "fdsl",
        "wrong_team"])
    df.to_csv("data/ab_comp_stale_rows.csv.gz", index=False, compression="gzip")
    out = {"n_rows": int(len(df))}
    print("\nplaying players NOT fully handled by the weekly comp:", len(df))
    for (pi, fi), g in df.groupby(["prod_in", "fresh_in"]):
        lab = {(0, 0): "S2 genuinely absent >12d at game date (both drop him)",
               (0, 1): "S1 REFIT STALENESS ONLY (game-date comp includes him)",
               (1, 0): "prod includes, fresh drops (reverse staleness)"}[(pi, fi)]
        print(f"  {lab:55s} n={len(g):6d} mean_min={g.real_min.mean():5.2f} "
              f"mean_talent={g.talent.mean():+.3f}")
        out[f"cell_{pi}{fi}"] = [int(len(g)), round(float(g.real_min.mean()), 2),
                                 round(float(g.talent.mean()), 3)]

    # margin footprint of each leg (talent x realized minutes / 48, x 0.5)
    caps = pd.read_csv("data/capstone_pergame_d132.csv", dtype={"game_id": str})
    def foot(sub, tag):
        if len(sub) == 0:
            return
        s = sub.assign(v=sub.sgn * sub.talent * sub.real_min / 48.0)
        gm = s.groupby(["season", "game_id"])["v"].sum().rename("dmargin").reset_index()
        j = caps.merge(gm, on=["season", "game_id"], how="left")
        dm = W_COMP * j.dmargin.fillna(0.0).to_numpy()
        p = caps.p_us.to_numpy(float)
        d = dm / SCALE
        r = {"rms_dmargin_pts": round(float(np.sqrt((dm ** 2).mean())), 4),
             "frac_games_moved": round(float((np.abs(dm) > 1e-9).mean()), 4),
             "max_abs": round(float(np.abs(dm).max()), 4),
             "best_case_dlogloss": round(float(0.5 * np.mean(d ** 2 * p * (1 - p))), 6)}
        out[tag] = r
        print(f"  {tag}: {r}")
        np.save(f"data/ab_dmargin_{tag}.npy", dm)

    print("\nMARGIN FOOTPRINT (points; comp weight 0.5 applied):")
    foot(df[(df.prod_in == 0) & (df.fresh_in == 1)], "S1_refit_stale")
    foot(df[(df.prod_in == 0) & (df.fresh_in == 0)], "S2_genuine_absence")
    foot(df[df.prod_in == 0], "S_all_missing")

    # the wrong-team leg costs TWICE (credited to the wrong side)
    wt = df[(df.wrong_team == 1)]
    out["wrong_team_n"] = int(len(wt))
    print(f"\nwrong-team rows: {len(wt)} (mean talent {wt.talent.mean():+.3f}, "
          f"mean min {wt.real_min.mean():.2f})")

    json.dump(out, open("data/ab_comp_stale.json", "w"), indent=1)
    print("wrote data/ab_comp_stale.json")


if __name__ == "__main__":
    main()
