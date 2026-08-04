"""ABSENCE AUDIT — the OTHER comp absence leg: players who PLAY tonight but are
dropped from comp.strength() by the ROSTER_DAYS=12 recency cut.

composition.strength() skips any player whose `last_played` (as of the WEEKLY
refit) is more than 12 days old. A player returning from a >12-day absence is
therefore worth EXACTLY ZERO to his team's comp strength until the next refit
re-observes him. This is the membership analogue of D133: the roster is learned
only from games the player PLAYED.

Also computes the §5.5 power bound for every arm at the logit level, using only
p (no outcomes): for a small additive logit shift d that removes a true bias,
    E[dLogLoss] ~= 0.5 * E[d^2 * p(1-p)]
This is a POWER quantity, not an endpoint score.

READ-ONLY. Writes data/ab_comp_excl.json.
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


def main():
    con = connect(read_only=True)
    pm = con.execute("""
        SELECT s.game_id, s.team_id, s.player_id, s.seconds, g.game_date, g.season
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '002%'""").fetchdf()
    pm["game_date"] = [_d(x) for x in pm["game_date"]]
    played = defaultdict(set)
    mins = {}
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
                is_home = (x.team_abbrev == host)
                for pid in played.get((gid, t), set()):
                    p = comp.players.get(pid)
                    m = mins.get((gid, pid), 0.0)
                    if p is None:
                        rows.append((season, gid, t, is_home, pid, "not_in_comp",
                                     0.0, 0.0, m, 999))
                        continue
                    dsl = (gd - p["last_played"]).days
                    if p["team_id"] != t:
                        rows.append((season, gid, t, is_home, pid, "wrong_team",
                                     p["trail_min"], p["talent"], m, dsl))
                    elif dsl > ROSTER_DAYS:
                        rows.append((season, gid, t, is_home, pid, "stale",
                                     p["trail_min"], p["talent"], m, dsl))
        print(season, "cum", len(rows), flush=True)
    con.close()

    df = pd.DataFrame(rows, columns=["season", "game_id", "team_id", "is_home",
                                     "player_id", "why", "trail_min", "talent",
                                     "real_min", "dsl"])
    df["missed_pts"] = df["talent"] * df["real_min"] / 48.0
    out = {"n_excluded_playing_rows": int(len(df))}
    print("\nexcluded-but-played rows by reason:")
    for w, g in df.groupby("why"):
        print(f"  {w:12s} n={len(g):7d} mean_min={g.real_min.mean():6.2f} "
              f"mean_talent={g.talent.mean():+6.3f} "
              f"sum_missed_pts/team-game={g.missed_pts.sum()/11645:+.4f}")
        out[f"why_{w}"] = [int(len(g)), round(float(g.real_min.mean()), 2),
                           round(float(g.talent.mean()), 3),
                           round(float(g.missed_pts.mean()), 4)]
    # the STALE leg only (the D133-class one: he played, he is on this team,
    # comp simply has not seen him since his absence began)
    st = df[df.why == "stale"]
    out["stale_by_talent"] = []
    for lo, hi, lab in [(-99, -1, "<-1"), (-1, 1, "-1..1"), (1, 3, "1..3"),
                        (3, 99, ">3")]:
        s = st[(st.talent > lo) & (st.talent <= hi)]
        out["stale_by_talent"].append(
            (lab, int(len(s)), round(float(s.real_min.mean()) if len(s) else 0, 2),
             round(float(s.missed_pts.sum()), 2)))
    print("stale by talent bucket:", out["stale_by_talent"])

    # per-game margin footprint of the STALE leg (what comp is missing)
    st = st.copy()
    st["sgn"] = np.where(st.is_home, 1.0, -1.0)
    gm = st.groupby(["season", "game_id"]).apply(
        lambda x: W_COMP * float((x.sgn * x.missed_pts).sum())).rename("dmargin").reset_index()
    caps = pd.read_csv("data/capstone_pergame_d132.csv", dtype={"game_id": str})
    j = caps.merge(gm, on=["season", "game_id"], how="left")
    j["dmargin"] = j["dmargin"].fillna(0.0)
    dm = j.dmargin.to_numpy()
    out["stale_margin"] = {
        "rms_dmargin_pts": round(float(np.sqrt((dm ** 2).mean())), 4),
        "mean": round(float(dm.mean()), 4),
        "max_abs": round(float(np.abs(dm).max()), 4),
        "frac_games_moved": round(float((np.abs(dm) > 1e-9).mean()), 4)}
    print("STALE leg margin footprint:", out["stale_margin"])
    np.save("data/ab_dmargin_S.npy", dm)

    # ---------- §5.5 POWER BOUND, outcomes never touched --------------------
    p = caps.p_us.to_numpy(float)
    pq = p * (1 - p)
    bnd = {}
    for arm in ("A", "B", "C", "O", "S"):
        d = np.load(f"data/ab_dmargin_{arm}.npy") / SCALE
        bnd[arm] = {
            "rms_logit_shift": round(float(np.sqrt((d ** 2).mean())), 5),
            "best_case_dlogloss": round(float(0.5 * np.mean(d ** 2 * pq)), 6)}
        print("BOUND", arm, bnd[arm])
    out["power_bound"] = bnd
    json.dump(out, open("data/ab_comp_excl.json", "w"), indent=1)
    print("wrote data/ab_comp_excl.json")


if __name__ == "__main__":
    main()
