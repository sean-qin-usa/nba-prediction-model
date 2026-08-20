#!/usr/bin/env python3
"""AVAILABILITY-DEPTH DESIGN DIAGNOSTIC 2 — fixes the two defects ad_design.py
exposed, still WITHOUT touching the endpoint.

Defect 1: `sr_last` in ad_design.py was "the most recent ROTATION-COVERED game",
which can be many games (or a whole season) stale, because rotation coverage is
27-100% depending on season. A stale role flag is not a role signal.
Defect 2: the +/-0.35 bucket edges were chosen by looking at the data.

This script therefore measures:
  (a) how the role-transition residual decays in the STALENESS GAP;
  (b) a threshold-free partition — sr_last is BINARY and sr5 uses a 5-game
      window, which is ODD so a majority can never tie:
        PROMOTED = sr_last==1 and sr5<0.5 ; DEMOTED = sr_last==0 and sr5>0.5 ;
        STABLE   = everything else;
  (c) walk-forward stationarity of b(bucket) across fit cutoffs (D133's
      stationarity check);
  (d) the ADVERSARIAL minutes-only control: the same partition rebuilt from
      last-game MINUTES rank instead of the rotation starter flag, so the gate
      can prove the rotation SOURCE is load-bearing and not just recency;
  (e) the national-TV proj_min shift, walk-forward.

Writes data/ad_design2.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from nbapred.db import connect
from nbapred.engine.props import minutes_ramp

HL = 10.0
SEASONS_ALL = ("2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
               "2024-25", "2025-26")


def cboot(d, g, B=2000, seed=20260801):
    uniq, inv = np.unique(g, return_inverse=True)
    sums = np.bincount(inv, weights=d, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), (B, len(uniq)))
    m = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), float(m.std(ddof=1))


def build_rows(con):
    df = con.execute("""
        SELECT s.player_id, s.team_id, s.game_id, g.season, g.game_date,
               s.seconds/60.0 AS mins
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g
          USING (game_id)
        WHERE s.game_id LIKE '002%' AND s.seconds >= 720
        ORDER BY s.player_id, g.game_date
    """).fetchdf()
    df["game_id"] = df["game_id"].astype(str)
    df["ord"] = df["game_date"].astype("datetime64[ns]").values.astype(
        "datetime64[D]").astype(int)
    rot = pd.read_csv(ROOT / "data" / "ad_rotation_pg.csv.gz",
                      dtype={"game_id": str})[
        ["game_id", "team_id", "player_id", "is_starter", "n_stints"]]
    tv = pd.read_csv(ROOT / "data" / "ad_natl_tv.csv", dtype={"game_id": str})
    d = df.merge(rot, on=["game_id", "team_id", "player_id"], how="left")
    d = d.merge(tv[["game_id", "is_natl_tv"]], on="game_id", how="left")

    recs = []
    for pid, sub in d.groupby("player_id", sort=False):
        sub = sub.sort_values("ord")
        mins = sub["mins"].to_numpy(float)
        seas = sub["season"].to_numpy(object)
        star = sub["is_starter"].to_numpy(float)
        tvf = sub["is_natl_tv"].to_numpy(float)
        ordv = sub["ord"].to_numpy()
        for i in range(len(sub)):
            if i < 8:
                continue
            w = 0.5 ** (np.arange(i)[::-1] / HL)
            proj_raw = float(np.sum(w * mins[:i]) / np.sum(w))
            gp = int((seas[:i] == seas[i]).sum())
            proj = max(proj_raw - minutes_ramp(gp), 0.0)
            if proj < 20:
                continue
            hs = star[:i]
            cov_idx = np.where(~np.isnan(hs))[0]
            if len(cov_idx) >= 1:
                last_idx = cov_idx[-1]
                gap = i - 1 - last_idx          # 0 = the immediately-prior game
                sr_last = float(hs[last_idx])
                w5 = cov_idx[-5:]
                sr5 = float(np.mean(hs[w5]))
                n5 = len(w5)
            else:
                gap, sr_last, sr5, n5 = 99, np.nan, np.nan, 0
            # adversarial minutes-only analogue: was the last PLAYED game a
            # "start-like" minutes game relative to the player's own median?
            med = float(np.median(mins[max(0, i - 20):i]))
            ml_last = 1.0 if mins[i - 1] >= med else 0.0
            m5w = mins[max(0, i - 5):i]
            ml5 = float(np.mean(m5w >= med))
            recs.append(dict(
                player_id=int(pid), season=seas[i], ord=int(ordv[i]), gp=gp,
                proj=proj, y=mins[i], resid=proj - mins[i],
                gap=int(gap), sr_last=sr_last, sr5=sr5, n5=n5,
                ml_last=ml_last, ml5=ml5,
                is_natl_tv=tvf[i]))
    return pd.DataFrame(recs)


def bucket(sr_last, sr5, n5, gap, max_gap):
    """PROMOTED / DEMOTED / STABLE, threshold-free (sr_last binary, 5-game
    window is odd so the majority never ties). Guarded: needs a full 5-game
    covered window AND the immediately-prior game(s) covered within max_gap."""
    if not np.isfinite(sr_last) or n5 < 5 or gap > max_gap:
        return "NA"
    if sr_last == 1.0 and sr5 < 0.5:
        return "PROMOTED"
    if sr_last == 0.0 and sr5 > 0.5:
        return "DEMOTED"
    return "STABLE"


def main():
    con = connect(read_only=True)
    r = build_rows(con)
    con.close()
    print(f"rows {len(r)}  players {r.player_id.nunique()}")
    out = {"n": int(len(r)), "players": int(r.player_id.nunique())}

    # ------------------------------------------------- (a) staleness decay
    print("\n=== (a) role-transition residual by STALENESS GAP ===")
    tab = []
    for gmax in (0, 1, 2, 5, 10, 99):
        b = np.array([bucket(a, c, n, g, gmax)
                      for a, c, n, g in zip(r.sr_last, r.sr5, r.n5, r.gap)])
        row = {"max_gap": gmax}
        for lab in ("PROMOTED", "DEMOTED", "STABLE"):
            m = b == lab
            if m.sum() < 50:
                continue
            pt, lo, hi, se = cboot(r.resid[m].to_numpy(), r.player_id[m].to_numpy())
            row[lab] = dict(n=int(m.sum()), resid=pt, lo=lo, hi=hi)
        row["NA_share"] = float((b == "NA").mean())
        tab.append(row)
        pr = row.get("PROMOTED", {}); de = row.get("DEMOTED", {})
        print(f"  gap<={gmax:2d}  PROMOTED n={pr.get('n',0):6d} "
              f"{pr.get('resid',float('nan')):+.4f}  DEMOTED n={de.get('n',0):6d} "
              f"{de.get('resid',float('nan')):+.4f}  "
              f"STABLE {row.get('STABLE',{}).get('resid',float('nan')):+.4f}  "
              f"NA {row['NA_share']:.3f}")
    out["staleness"] = tab

    # ------------------------------------ (b) frozen guard = gap<=0 (adjacent)
    GAP = 0
    r["bk"] = [bucket(a, c, n, g, GAP)
               for a, c, n, g in zip(r.sr_last, r.sr5, r.n5, r.gap)]
    print(f"\n=== (b) FROZEN CONFIG gap<=0 partition ===")
    print(r.bk.value_counts().to_string())
    out["partition_gap0"] = r.bk.value_counts().to_dict()
    out["active_share"] = float(r.bk.isin(["PROMOTED", "DEMOTED"]).mean())

    # per-season activity (rotation coverage drives this)
    ps = r.groupby("season").bk.value_counts().unstack(fill_value=0)
    print(ps.to_string())
    out["partition_by_season"] = ps.to_dict("index")

    # ---------------------------- (c) walk-forward stationarity of b(bucket)
    print("\n=== (c) walk-forward stationarity of b(bucket) ===")
    st = {}
    for cut in ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26"):
        prior = [s for s in SEASONS_ALL if s < cut]
        sub = r[r.season.isin(prior)]
        vals = {lab: (float(sub.resid[sub.bk == lab].mean()) if (sub.bk == lab).sum() >= 30
                      else None)
                for lab in ("PROMOTED", "DEMOTED", "STABLE")}
        vals["n"] = int(len(sub))
        st[cut] = vals
        print(f"  fit<{cut}: n={vals['n']:6d}  PROM {vals['PROMOTED']}  "
              f"DEM {vals['DEMOTED']}  STAB {vals['STABLE']}")
    out["stationarity"] = st

    # --------------------------- (d) ADVERSARIAL minutes-only control buckets
    def mbucket(ml_last, ml5, gp_ok=True):
        if not np.isfinite(ml_last):
            return "NA"
        if ml_last == 1.0 and ml5 < 0.5:
            return "PROMOTED"
        if ml_last == 0.0 and ml5 > 0.5:
            return "DEMOTED"
        return "STABLE"
    r["mk"] = [mbucket(a, c) for a, c in zip(r.ml_last, r.ml5)]
    print("\n=== (d) adversarial MINUTES-ONLY partition (no rotation input) ===")
    for lab in ("PROMOTED", "DEMOTED", "STABLE"):
        m = (r.mk == lab).to_numpy()
        if m.sum() < 50:
            continue
        pt, lo, hi, se = cboot(r.resid[m].to_numpy(), r.player_id[m].to_numpy())
        print(f"  {lab:9s} n={m.sum():6d} resid {pt:+.4f} CI[{lo:+.4f},{hi:+.4f}]")
        out.setdefault("minutes_only", {})[lab] = dict(n=int(m.sum()), resid=pt,
                                                       lo=lo, hi=hi)
    # cross-tab: does the rotation flag say anything the minutes flag does not?
    ct = pd.crosstab(r.bk, r.mk)
    print("\n  crosstab rotation-bucket x minutes-bucket:")
    print(ct.to_string())
    out["crosstab"] = ct.to_dict("index")
    # residual WITHIN the minutes-STABLE cell, split by the rotation flag --
    # this is the incremental content of the rotation source
    print("\n  residual inside minutes-bucket STABLE, split by rotation bucket:")
    for lab in ("PROMOTED", "DEMOTED", "STABLE"):
        m = ((r.bk == lab) & (r.mk == "STABLE")).to_numpy()
        if m.sum() < 50:
            continue
        pt, lo, hi, se = cboot(r.resid[m].to_numpy(), r.player_id[m].to_numpy())
        print(f"    rot={lab:9s} n={m.sum():6d} resid {pt:+.4f} CI[{lo:+.4f},{hi:+.4f}]")
        out.setdefault("incremental", {})[lab] = dict(n=int(m.sum()), resid=pt,
                                                      lo=lo, hi=hi)

    # ------------------------------------------ (e) national-TV proj shift
    print("\n=== (e) national-TV minutes residual, walk-forward table ===")
    tv = r[r.is_natl_tv.notna()]
    for cut in ("2023-24", "2024-25", "2025-26"):
        prior = [s for s in SEASONS_ALL if s < cut]
        sub = tv[tv.season.isin(prior)]
        if len(sub) < 100:
            print(f"  fit<{cut}: n={len(sub)} TOO THIN")
            continue
        b1 = float(sub.resid[sub.is_natl_tv == 1].mean())
        b0 = float(sub.resid[sub.is_natl_tv == 0].mean())
        print(f"  fit<{cut}: n={len(sub):6d}  b_natl {b1:+.4f}  b_local {b0:+.4f}"
              f"  diff {b1-b0:+.4f}")
        out.setdefault("natl_tv_wf", {})[cut] = dict(n=int(len(sub)), b_natl=b1,
                                                     b_local=b0, diff=b1 - b0)

    r.to_csv(ROOT / "data" / "ad_design2_rows.csv.gz", index=False,
             compression="gzip")
    (ROOT / "data" / "ad_design2.json").write_text(json.dumps(out, indent=2,
                                                              default=float))
    print("\nAD_DESIGN2_DONE")


if __name__ == "__main__":
    main()
