#!/usr/bin/env python3
"""D175: PRICE the archived-CBS availability feed on a pilot season.

Builds a CBS-derived `rout` (the same {(game_date, team_abbrev): {player_id}}
shape `report_out_map` returns) and feeds it to k19_t2.season_run UNMODIFIED,
so the CBS arm is `cbs_out UNION inactives` exactly as T2 is
`report_out UNION inactives`.  The baseline arm is `t2i` = inactives only, run
in the SAME process on the SAME DB state.

  CBS arm - T2i arm  ==  the CBS analogue of D171's T2-vs-T2i, which measured
  -0.741pp (n=9,516, season-clustered t=-5.02, K=8; honest range -0.25..-0.74).

POINT-IN-TIME DISCIPLINE. For game date D we use the LATEST archived snapshot
whose own target date is <= D, where a snapshot taken before 17:00 ET targets
that day's slate and one taken later targets the next day. No snapshot taken
after a game can inform that game. Carry-forward age is reported, not hidden.

READ-ONLY on data/nba.duckdb. Changes no default, re-runs no gate, and writes
no table.
"""
from __future__ import annotations
import json, os, sys, time
import datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("TANK_SEASON_FLOOR", "2020-21")   # k19_t2's documented run env

import nbapred.threads              # noqa: E402
nbapred.threads.pin(1)
import numpy as np, pandas as pd    # noqa: E402
from nbapred.db import connect      # noqa: E402
import k19_t2                       # noqa: E402

RAW = ROOT / "data" / "raw" / "unofficial" / "wayback"
CBSFIX = {"PHO": "PHX", "CHO": "CHA", "GS": "GSW", "NO": "NOP", "NY": "NYK",
          "SA": "SAS", "UTAH": "UTA", "BRK": "BKN", "WSH": "WAS"}


def norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def build_cbs_rout(con, tag: str, season: str, statuses, use_until=True):
    """use_until=True honours CBS's own 'out until at least <date>' expiry, so a
    carried-forward row stops asserting OUT once its stated return date passes.
    That is a PRECISION fix, and D175 measures the arm both ways."""
    d = pd.read_csv(RAW / f"{tag}_rows.csv")
    d["until_date"] = pd.to_datetime(d.get("until_date"), errors="coerce")
    d["snap_et_date"] = pd.to_datetime(d["snap_et_date"])
    d["target"] = d["snap_et_date"] + pd.to_timedelta(
        (d["snap_et_hour"] >= 17).astype(int), unit="D")
    d["pkey"] = d["player_slug"].map(norm)
    pl = con.execute("SELECT player_id, first_name, last_name FROM nba_players").fetchdf()
    pl["pkey"] = (pl.first_name.fillna("") + pl.last_name.fillna("")).map(norm)
    d = d.merge(pl.drop_duplicates("pkey")[["pkey", "player_id"]], on="pkey", how="left")
    d = d[d.player_id.notna()].copy()
    d["team_ab"] = d["team_abbr"].map(lambda a: CBSFIX.get(a, a))
    d = d[d.status.isin(statuses)]
    if use_until:
        # drop rows whose own stated return date is already in the past
        exp = d.until_date.notna() & (d.until_date <= d.target)
        d = d[~exp]

    # one out-set per (snapshot, team)
    snaps = sorted(d.target.unique())
    per = {}
    for (tgt, team), sub in d.groupby(["target", "team_ab"]):
        per[(pd.Timestamp(tgt), team)] = set(sub.player_id.astype(int))

    gd = con.execute("""SELECT DISTINCT game_date FROM nba_games
        WHERE season=? AND game_id LIKE '002%' AND wl IS NOT NULL
        ORDER BY 1""", [season]).fetchdf()["game_date"]
    gd = pd.to_datetime(gd)
    teams = [r[0] for r in con.execute(
        "SELECT DISTINCT team_abbrev FROM nba_games WHERE season=?", [season]).fetchall()]

    rout, rcov, ages = {}, set(), []
    sn = np.array(snaps, dtype="datetime64[ns]")
    for D in gd:
        i = np.searchsorted(sn, np.datetime64(D), side="right") - 1
        if i < 0:
            continue
        latest = pd.Timestamp(sn[i])
        ages.append((D - latest).days)
        rcov.add(str(D)[:10])
        for t in teams:
            s = per.get((latest, t))
            if s:
                rout[(str(D)[:10], k19_t2.fx(t))] = set(s)
    return rout, rcov, np.array(ages)


def main():
    season = sys.argv[1] if len(sys.argv) > 1 else "2015-16"
    tag = sys.argv[2] if len(sys.argv) > 2 else "cbs_2015_16"
    con = connect(read_only=True, retry_s=60)

    inact = {}
    for g, p in con.execute("SELECT game_id, player_id FROM game_inactives").fetchall():
        inact.setdefault(g, set()).add(int(p))

    res = {}
    print("=" * 92)
    print(f"D175 PRICING — season {season}, source {tag}, "
          f"TANK_SEASON_FLOOR={os.environ['TANK_SEASON_FLOOR']}")
    print("=" * 92)

    t0 = time.time()
    base = k19_t2.season_run(con, season, "t2i", {}, set(), inact)
    print(f"  BASELINE  t2i (inactives only)      "
          f"ll_us={base['ll_us']:.5f} ll_mkt={base['ll_mkt']:.5f} "
          f"norm={base['norm_gap_pct']:+.2f}%  outs/tm={base['mean_outs_per_team']:.3f} "
          f"n={base['n']}  ({time.time()-t0:.1f}s)")
    res["t2i"] = {k: v for k, v in base.items() if k != "rows"}

    for name, sts, uu in (("cbs_out_naive", ("OUT", "OUT_SEASON"), False),
                          ("cbs_out_expiry", ("OUT", "OUT_SEASON"), True),
                          ("cbs_out_expiry_D", ("OUT", "OUT_SEASON", "DOUBTFUL"), True)):
        rout, rcov, ages = build_cbs_rout(con, tag, season, sts, use_until=uu)
        t0 = time.time()
        r = k19_t2.season_run(con, season, "t2", rout, rcov, inact)
        dn = r["norm_gap_pct"] - base["norm_gap_pct"]
        print(f"  {name:<18} statuses={'+'.join(sts):<28} "
              f"ll_us={r['ll_us']:.5f} norm={r['norm_gap_pct']:+.2f}% "
              f"DELTA={dn:+.3f}pp  outs/tm={r['mean_outs_per_team']:.3f} "
              f"cov_days={len(rcov)}  carryfwd_age med={np.median(ages):.1f}d "
              f"p90={np.percentile(ages,90):.1f}d  ({time.time()-t0:.1f}s)")
        res[name] = {k: v for k, v in r.items() if k != "rows"}
        res[name]["delta_norm_pp"] = round(float(dn), 4)
        res[name]["carryfwd_median_d"] = float(np.median(ages))
        res[name]["carryfwd_p90_d"] = float(np.percentile(ages, 90))
        # paired per-game log-loss delta, for a t on the SAME games
        bp = {r0[1]: r0 for r0 in base["rows"]}
        dl = [(np.log(1 - x[6]) if x[5] == 0 else np.log(x[6]))
              - (np.log(1 - bp[x[1]][6]) if x[5] == 0 else np.log(bp[x[1]][6]))
              for x in r["rows"] if x[1] in bp]
        dl = np.array(dl)
        res[name]["paired_games"] = int(len(dl))
        res[name]["paired_mean_dll"] = float(-dl.mean())
        nz = int((np.abs(dl) > 1e-12).sum())
        res[name]["paired_games_changed"] = nz
        print(f"{'':>22}paired n={len(dl)} games whose p CHANGED={nz} "
              f"({100*nz/max(len(dl),1):.2f}%)  mean dLL={-dl.mean():+.6f}")

    json.dump(res, open(ROOT / "data" / f"d175_price_{tag}.json", "w"), indent=1)
    print(f"\nwrote data/d175_price_{tag}.json")


if __name__ == "__main__":
    main()
