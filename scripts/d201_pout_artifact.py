#!/usr/bin/env python3
"""D201 — emit the POINT-IN-TIME P(out) artifact the composition leg consumes.

    data/p_out.csv.gz    game_date, player_id, p_out

Built from the D200 participation model, fitted WALK-FORWARD: the probabilities
used on season k+1 come from a model fitted on seasons 1..k only, so the artifact
is point-in-time by construction and nothing in it saw its own season.

Features are strictly as-of-open (T0): last published status BEFORE game day,
days since that report, consecutive prior Out reports, consecutive prior
Questionable/Doubtful reports, team rest. No game-state or competitiveness
feature appears anywhere (the blowout-leak decision, D200).

Read-only w.r.t. the DB. Writes one artifact.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

from d200_participation import (STATUSES, logistic_fit,           # noqa: E402
                                predict)
from nbapred.db import connect                                    # noqa: E402
from nbapred import teams as T                                    # noqa: E402

OUT = ROOT / "data" / "p_out.csv.gz"


def main():
    con = connect(read_only=True)
    r = pd.DataFrame(con.execute("""
        SELECT report_date, game_date, team, player, status
        FROM injury_reports_pit""").fetchall(),
        columns=["rd", "gd", "team", "player", "status"])
    xw = pd.DataFrame(con.execute(
        "SELECT player_id, lower(first_name||' '||last_name) fn "
        "FROM nba_players").fetchall(), columns=["pid", "fn"])
    gs = pd.DataFrame(con.execute(
        "SELECT DISTINCT game_date, team_abbrev FROM nba_games "
        "WHERE game_id LIKE '002%'").fetchall(), columns=["gd", "team"])
    con.close()

    amap, unres = T.resolve_map(sorted(r["team"].unique()))
    if unres:
        print(f"  [teams] unresolvable REPORTED: {unres}")
    r["team"] = r["team"].map(amap)
    r = r[r["team"].notna()].copy()
    r["rd"] = pd.to_datetime(r["rd"])
    r["gd"] = pd.to_datetime(r["gd"])
    gs["gd"] = pd.to_datetime(gs["gd"])

    # player_id join, same convention as prod_by_season's report_out_map
    def norm(nm):
        s = str(nm)
        if "," in s:
            a, b = [x.strip() for x in s.split(",")[:2]]
            return f"{b} {a}".lower()
        return s.lower()
    r["fn"] = r["player"].map(norm)
    r = r.merge(xw, on="fn", how="left")
    print(f"report rows {len(r)}, player_id resolved "
          f"{100*r['pid'].notna().mean():.1f}%")
    r = r[r["pid"].notna()].copy()
    r["pid"] = r["pid"].astype(int)

    gs = gs.sort_values("gd")
    gs["prev"] = gs.groupby("team")["gd"].shift(1)
    rest = {(x.team, x.gd): min((x.gd - x.prev).days, 7)
            if pd.notna(x.prev) else 7 for x in gs.itertuples()}

    lab = r[r["rd"] == r["gd"]][["gd", "team", "pid", "status"]].copy()
    lab["y_out"] = (lab["status"] == "Out").astype(float)

    # as-of-open features
    feats = []
    for (team, pid), g in r.sort_values("rd").groupby(["team", "pid"]):
        seen = []
        for x in g.itertuples():
            prior = [s for (d, s) in seen if d < x.gd]
            if prior:
                d_since = (x.gd - max(d for (d, s) in seen if d < x.gd)).days
                so = sq = 0
                for s in reversed(prior):
                    if s == "Out":
                        so += 1
                    else:
                        break
                for s in reversed(prior):
                    if s in ("Questionable", "Doubtful"):
                        sq += 1
                    else:
                        break
                feats.append((x.gd, team, pid, prior[-1], d_since, so, sq))
            seen.append((x.rd, x.status))
    F = pd.DataFrame(feats, columns=["gd", "team", "pid", "last_status",
                                     "days_since", "streak_out", "streak_q"])
    F = F.drop_duplicates(["gd", "team", "pid"], keep="last")
    F["rest"] = [rest.get((t, g), 7) for t, g in zip(F["team"], F["gd"])]
    for s in STATUSES:
        F[f"ls_{s}"] = (F["last_status"] == s).astype(float)
    FE = [f"ls_{s}" for s in STATUSES] + ["days_since", "streak_out",
                                          "streak_q", "rest"]

    F["season"] = np.where(F["gd"].dt.month >= 10,
                           F["gd"].dt.year.astype(str) + "-" +
                           (F["gd"].dt.year + 1).astype(str).str[2:],
                           (F["gd"].dt.year - 1).astype(str) + "-" +
                           F["gd"].dt.year.astype(str).str[2:])
    d = F.merge(lab[["gd", "team", "pid", "y_out"]], on=["gd", "team", "pid"],
                how="left")
    seas = sorted(d["season"].unique())
    print(f"feature rows {len(d)}, labelled {int(d['y_out'].notna().sum())}, "
          f"seasons {seas}")

    # WALK-FORWARD: season k+1 predicted by a model fitted on 1..k.
    # The first season has no prior fit; fall back to the hard rule (p = ls_Out),
    # which is exactly the incumbent, so it is never worse than status quo there.
    parts = []
    for i, s in enumerate(seas):
        te = d[d["season"] == s]
        tr = d[(d["season"].isin(seas[:i])) & d["y_out"].notna()]
        if i == 0 or len(tr) < 500:
            p = te["ls_Out"].to_numpy(float)
            src = "hard-rule fallback"
        else:
            w = logistic_fit(tr[FE].to_numpy(float),
                             tr["y_out"].to_numpy(float), lam=10.0)
            p = predict(w, te[FE].to_numpy(float))
            src = f"fit on {len(tr)} rows from {seas[:i]}"
        parts.append(pd.DataFrame({"game_date": te["gd"].dt.strftime("%Y-%m-%d"),
                                   "player_id": te["pid"].to_numpy(int),
                                   "p_out": p}))
        print(f"  {s}: {len(te):6d} rows  mean p_out {p.mean():.4f}   {src}")

    a = pd.concat(parts, ignore_index=True)
    a = a.groupby(["game_date", "player_id"], as_index=False)["p_out"].max()
    a.to_csv(OUT, index=False, compression="gzip")
    print(f"\nwrote {OUT}  ({len(a)} rows)")
    print(f"  p_out distribution: "
          f"mean {a['p_out'].mean():.4f}  "
          f"share>0.9 {100*(a['p_out']>0.9).mean():.1f}%  "
          f"share in (0.1,0.9) {100*a['p_out'].between(0.1,0.9).mean():.1f}%  "
          f"share<0.1 {100*(a['p_out']<0.1).mean():.1f}%")


if __name__ == "__main__":
    main()
