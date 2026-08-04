"""W49 PASS 1 — build the enriched per-game forensic frame.

Base = data/ds_rt1_pergame.csv (4 seasons, 4920 games, p_full = shipped
production win prob, p_mkt = de-vig close).

Adds, for every game, everything that is PIT-available at tip:
  * component decomposition  m_tot = 0.5*fm + 0.5*cm + rest   (data/ds_rt4_components.csv,
    recovers p_full to 9e-15) -> which leg drives our confidence
  * outs both sides from `game_inactives` x trailing-25-minute star definition
    (trailing window strictly before game_date, PIT)
  * DARKO talent lost to outs (PIT snapshot <= game_date - 1)
  * schedule_features: rest / b2b / 3in4 / 4in5 / games_last_7 / travel / tz
  * season phase (month, gp), tank score diff (already in base)
  * team identities
and, for PROFILING ONLY (never a gate input), realised outcome facts:
  * final margin, blowout flag, OT flag (from total played seconds)

Read-only DB. Output: data/w49_frame.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ds_corpus import arm_connection  # noqa: E402

BASE = ROOT / "data" / "ds_rt1_pergame.csv"
COMP = ROOT / "data" / "ds_rt4_components.csv"
OUT = ROOT / "data" / "w49_frame.csv"

TRAIL_DAYS = 45          # trailing window for the "is a star" minutes test
STAR_MIN = 25.0          # >=25 trailing min/g == star (task definition)


def main() -> None:
    base = pd.read_csv(BASE, dtype={"game_id": str})
    base["game_id"] = base.game_id.str.zfill(10)
    base["game_date"] = pd.to_datetime(base.game_date)

    comp = pd.read_csv(COMP, dtype={"game_id": str})[
        ["game_id", "m_tot", "fm", "cm", "rest"]]
    d = base.merge(comp, on="game_id", how="left")
    assert d.m_tot.notna().all()

    con = arm_connection(None)

    # ---- team-level game rows (abbrev + pts + team_id) --------------------
    g = con.execute("""
        SELECT game_id, team_id, team_abbrev, is_home, pts
        FROM nba_games WHERE game_id IN (SELECT game_id FROM nba_games)
    """).df()
    g = g[g.game_id.isin(set(d.game_id))].drop_duplicates(["game_id", "team_id"])
    hm = g[g.is_home].drop_duplicates("game_id").set_index("game_id")
    aw = g[~g.is_home].drop_duplicates("game_id").set_index("game_id")
    d["home_team_id"] = d.game_id.map(hm.team_id)
    d["away_team_id"] = d.game_id.map(aw.team_id)
    d["pts_home"] = d.game_id.map(hm.pts)
    d["pts_away"] = d.game_id.map(aw.pts)
    d["margin"] = d.pts_home - d.pts_away

    # ---- OT detection: total player-seconds / 10 = game length -----------
    secs = con.execute("""
        SELECT game_id, SUM(seconds) AS s FROM player_game_stats GROUP BY 1
    """).df().set_index("game_id").s
    d["game_min"] = d.game_id.map(secs) / 600.0     # minutes of game clock
    d["is_ot"] = (d.game_min > 49.5).astype(int)

    # ---- schedule features (derived: schedule_features covers 25-26 only) --
    sched = con.execute("""
        SELECT DISTINCT game_id, team_id, game_date, season
        FROM nba_games WHERE game_id LIKE '002%'
    """).df()
    sched["game_date"] = pd.to_datetime(sched.game_date)
    sched = sched.sort_values(["team_id", "game_date"])
    sched["prev"] = sched.groupby("team_id").game_date.shift(1)
    sched["days_rest"] = (sched.game_date - sched.prev).dt.days
    # games in the trailing 7 days (strictly before)
    cnt7 = []
    for tid, grp in sched.groupby("team_id"):
        dts = grp.game_date.values
        c = [(np.sum((dts < dts[i]) & (dts >= dts[i] - np.timedelta64(7, "D"))))
             for i in range(len(dts))]
        cnt7.append(pd.Series(c, index=grp.index))
    sched["games_last_7"] = pd.concat(cnt7).reindex(sched.index)
    sched["is_b2b"] = (sched.days_rest == 1).astype(float)
    sched = sched[sched.game_id.isin(set(d.game_id))]

    for side, tcol in (("home", "home_team_id"), ("away", "away_team_id")):
        key = sched.set_index(["game_id", "team_id"])
        idx = pd.MultiIndex.from_arrays([d.game_id, d[tcol]])
        for c in ["days_rest", "is_b2b", "games_last_7"]:
            d[f"{c}_{side}"] = key[c].reindex(idx).values
    for side in ("home", "away"):
        d[f"is_b2b_{side}"] = d[f"is_b2b_{side}"].fillna(0.0).astype(int)
        d[f"days_rest_{side}"] = d[f"days_rest_{side}"].fillna(7).clip(upper=7)
    d["rest_adv"] = d.days_rest_home - d.days_rest_away
    d["b2b_either"] = ((d.is_b2b_home + d.is_b2b_away) > 0).astype(int)

    # ---- outs: inactives x trailing minutes (PIT) -------------------------
    # trailing minutes per (player, game_date) computed strictly before the date
    inact = con.execute("""
        SELECT i.game_id, i.player_id, i.team_id
        FROM game_inactives i
    """).df()
    inact = inact[inact.game_id.isin(set(d.game_id))]

    # PIT trailing minutes for every (player, date) pair we need
    dates = d[["game_id", "game_date"]].copy()
    con.execute("CREATE TEMP TABLE need AS SELECT * FROM dates")
    trail = con.execute(f"""
        WITH pgs AS (
            SELECT p.player_id, p.game_id, p.seconds/60.0 AS mins,
                   ng.game_date
            FROM player_game_stats p
            JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) ng
              ON ng.game_id = p.game_id
            WHERE ng.game_id LIKE '002%'
        )
        SELECT n.game_id, pgs.player_id,
               AVG(pgs.mins) AS trail_min, COUNT(*) AS trail_n
        FROM need n
        JOIN pgs ON pgs.game_date < n.game_date
                AND pgs.game_date >= n.game_date - INTERVAL {TRAIL_DAYS} DAY
        GROUP BY 1, 2
    """).df()
    tkey = trail.set_index(["game_id", "player_id"])

    ino = inact.merge(trail, on=["game_id", "player_id"], how="left")
    ino["trail_min"] = ino.trail_min.fillna(0.0)
    ino["is_star"] = (ino.trail_min >= STAR_MIN).astype(int)

    # DARKO talent of each out, PIT (latest snapshot strictly before tip)
    con.execute("CREATE TEMP TABLE outs AS SELECT * FROM ino")
    dk = con.execute("""
        WITH need AS (
            SELECT o.game_id, o.player_id, n.game_date
            FROM outs o JOIN need n ON n.game_id = o.game_id
        )
        SELECT nd.game_id, nd.player_id,
               (SELECT dpm FROM darko_history k
                 WHERE k.player_id = nd.player_id
                   AND k.date < nd.game_date
                 ORDER BY k.date DESC LIMIT 1) AS dpm
        FROM need nd
    """).df()
    ino = ino.merge(dk, on=["game_id", "player_id"], how="left")
    ino["dpm"] = ino.dpm.fillna(0.0)
    # talent lost = dpm x trailing share of a 48-min game (matches comp leg)
    ino["talent_lost"] = ino.dpm * (ino.trail_min / 48.0)

    ino["side"] = np.where(ino.team_id == ino.game_id.map(
        d.set_index("game_id").home_team_id), "home", "away")
    for side in ("home", "away"):
        s = ino[ino.side == side]
        agg = s.groupby("game_id").agg(
            n_out=("player_id", "size"),
            n_star_out=("is_star", "sum"),
            out_min=("trail_min", "sum"),
            out_talent=("talent_lost", "sum"),
            max_out_min=("trail_min", "max"),
        )
        for c in agg.columns:
            d[f"{c}_{side}"] = d.game_id.map(agg[c]).fillna(0.0)

    d["n_out"] = d.n_out_home + d.n_out_away
    d["n_star_out"] = d.n_star_out_home + d.n_star_out_away
    d["out_min_d"] = d.out_min_home - d.out_min_away
    d["out_talent_d"] = d.out_talent_home - d.out_talent_away

    # ---- derived model / market quantities --------------------------------
    eps = 1e-12
    cl = lambda p: np.clip(p, eps, 1 - eps)  # noqa: E731
    d["l_us"] = -(d.y * np.log(cl(d.p_full)) + (1 - d.y) * np.log(cl(1 - d.p_full)))
    d["l_mkt"] = -(d.y * np.log(cl(d.p_mkt)) + (1 - d.y) * np.log(cl(1 - d.p_mkt)))
    d["exc"] = d.l_us - d.l_mkt

    d["conf_us"] = (d.p_full - 0.5).abs()
    d["conf_mkt"] = (d.p_mkt - 0.5).abs()
    d["conf_gap"] = d.conf_us - d.conf_mkt
    d["div"] = d.p_full - d.p_mkt
    d["same_side"] = ((d.p_full > 0.5) == (d.p_mkt > 0.5)).astype(int)
    d["our_fav_home"] = (d.p_full > 0.5).astype(int)
    d["mkt_fav_home"] = (d.p_mkt > 0.5).astype(int)
    d["our_fav_won"] = ((d.p_full > 0.5) == (d.y == 1)).astype(int)
    d["mkt_fav_won"] = ((d.p_mkt > 0.5) == (d.y == 1)).astype(int)

    # confidence attribution: which leg carries the margin, and does it agree
    d["m_ff_leg"] = 0.5 * d.fm
    d["m_cm_leg"] = 0.5 * d.cm
    tot = d.m_ff_leg.abs() + d.m_cm_leg.abs() + d.rest.abs()
    d["share_cm"] = d.m_cm_leg.abs() / tot.replace(0, np.nan)
    d["share_ff"] = d.m_ff_leg.abs() / tot.replace(0, np.nan)
    d["legs_agree"] = (np.sign(d.fm) == np.sign(d.cm)).astype(int)
    d["leg_disagree_mag"] = (d.fm - d.cm).abs()
    # leg-level DISPERSION = our own (market-blind) uncertainty signal
    d["leg_spread"] = (d.m_ff_leg - d.m_cm_leg).abs()

    d["month"] = d.game_date.dt.month
    d["gp_min"] = d[["gp_home", "gp_away"]].min(axis=1)
    d["gp_max"] = d[["gp_home", "gp_away"]].max(axis=1)
    d["early"] = (d.gp_min < 20).astype(int)
    d["late"] = (d.gp_min >= 55).astype(int)
    d["abs_margin"] = d.margin.abs()
    d["blowout"] = (d.abs_margin >= 20).astype(int)
    d["tsd_abs"] = d.tsd.abs()

    d.to_csv(OUT, index=False)
    print(f"wrote {OUT}  n={len(d)}  cols={len(d.columns)}")
    print("null check:", d[["m_tot", "days_rest_home", "margin", "game_min"]]
          .isna().sum().to_dict())
    print("outs coverage by season (mean n_out):")
    print(d.groupby("season").n_out.mean().to_string())


if __name__ == "__main__":
    main()
