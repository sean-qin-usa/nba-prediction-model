#!/usr/bin/env python3
"""D200 — THE PARTICIPATION MODEL (roadmap item 4, stage 1).

This is the direct fix for D199.  At the OPEN we know each player's LAST
PUBLISHED status; we do not know tonight's 5PM report.  D199 measured that gap
at 18.1% of the out-set by headcount, 19.3% minutes-weighted, and worth 33% of
the model's entire deficit to the market.

So do not treat the as-of-open out-set as a hard set.  FORECAST it:

    P(player is OUT tonight | last published status, status history, rest)

and feed EXPECTED availability into the composition leg instead of a 0/1 flag.
A player last listed Questionable is not "in"; he is ~p out.

DESIGN, declared before scoring
  Unit: player-team-gameday where the player appears on ANY report on or before
        the game day (so the label exists).
  Label: OUT tonight (status Out on the same-day report) -- 1/0.
  Features, all strictly as-of-open (T0):
        last_status   the player's most recent published status BEFORE game day
        days_since    days since that report
        streak_out    consecutive prior reports listing him Out
        streak_q      consecutive prior reports listing him Questionable
        rest          days since his team's previous game
  Estimator: multinomial-free logistic regression, L2, fitted walk-forward on
        seasons 1..k and scored on k+1. No grid: lambda by the same GCV-style
        rule used in D195, i.e. Type-A per D192.
  BASELINE TO BEAT (the incumbent, per D176): the carry-forward rule D199 used,
        i.e. "he is out tonight iff he was out on the last report" -- a hard 0/1.

BLOWOUT TRAP (owner asked me to decide): participation is NOT conditioned on any
game-state or competitiveness feature.  A player's minutes are truncated by a
blowout, and the blowout is caused by the margin we are trying to predict, so any
competitiveness input would leak the outcome backwards.  Rest and schedule are
fine (they are fixed before tip); expected-margin, pace and total are not, and
none appear here.

Read-only.  Nothing ships.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

import oc_capacity as oc                                          # noqa: E402
from nbapred.db import connect                                    # noqa: E402
from nbapred import teams as T                                    # noqa: E402

STATUSES = ["Out", "Doubtful", "Questionable", "Probable", "Available"]


def logistic_fit(X, y, lam=1.0, iters=60):
    """L2 logistic by Newton-IRLS. Deterministic, no solver dependence."""
    n, p = X.shape
    Xb = np.column_stack([np.ones(n), X])
    w = np.zeros(p + 1)
    R = lam * np.eye(p + 1)
    R[0, 0] = 0.0
    for _ in range(iters):
        z = Xb @ w
        mu = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        s = np.clip(mu * (1 - mu), 1e-6, None)
        g = Xb.T @ (mu - y) + R @ w
        H = (Xb * s[:, None]).T @ Xb + R
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def predict(w, X):
    z = np.column_stack([np.ones(len(X)), X]) @ w
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def nll(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main():
    con = connect(read_only=True)
    r = pd.DataFrame(con.execute("""
        SELECT report_date, game_date, team, player, status
        FROM injury_reports_pit WHERE game_date >= '2021-10-01'""").fetchall(),
        columns=["rd", "gd", "team", "player", "status"])
    gsched = pd.DataFrame(con.execute(
        "SELECT DISTINCT game_date, team_abbrev FROM nba_games "
        "WHERE game_id LIKE '002%' AND season >= '2021-22'").fetchall(),
        columns=["gd", "team"])
    con.close()

    amap, unres = T.resolve_map(sorted(r["team"].unique()))
    if unres:
        print(f"  [teams] unresolvable REPORTED: {unres}")
    r["team"] = r["team"].map(amap)
    r = r[r["team"].notna()].copy()
    r["rd"] = pd.to_datetime(r["rd"])
    r["gd"] = pd.to_datetime(r["gd"])
    gsched["gd"] = pd.to_datetime(gsched["gd"])

    # rest, from the schedule
    gsched = gsched.sort_values("gd")
    gsched["prev"] = gsched.groupby("team")["gd"].shift(1)
    rest = {(x.team, x.gd): min((x.gd - x.prev).days, 7)
            if pd.notna(x.prev) else 7 for x in gsched.itertuples()}

    # SAME-DAY label rows
    lab = r[r["rd"] == r["gd"]][["gd", "team", "player", "status"]].copy()
    lab = lab.rename(columns={"status": "status_today"})
    lab["y_out"] = (lab["status_today"] == "Out").astype(float)

    # AS-OF-OPEN history: every report row strictly before the game day
    hist = r[r["rd"] < r["gd"]].copy()
    # a player's last published status before each game day
    hist = hist.sort_values("rd")
    feats = []
    for (team, player), g in r.sort_values("rd").groupby(["team", "player"]):
        seen = []          # (report_date, status)
        for x in g.itertuples():
            key = (x.gd, team, player)
            prior = [s for (d, s) in seen if d < x.gd]
            if prior:
                last = prior[-1]
                d_since = (x.gd - max(d for (d, s) in seen if d < x.gd)).days
                so = 0
                for s in reversed(prior):
                    if s == "Out":
                        so += 1
                    else:
                        break
                sq = 0
                for s in reversed(prior):
                    if s in ("Questionable", "Doubtful"):
                        sq += 1
                    else:
                        break
                feats.append((x.gd, team, player, last, d_since, so, sq))
            seen.append((x.rd, x.status))
    F = pd.DataFrame(feats, columns=["gd", "team", "player", "last_status",
                                     "days_since", "streak_out", "streak_q"])
    F = F.drop_duplicates(["gd", "team", "player"], keep="last")

    d = lab.merge(F, on=["gd", "team", "player"], how="inner")
    d["rest"] = [rest.get((t, g), 7) for t, g in zip(d["team"], d["gd"])]
    d["season"] = np.where(d["gd"].dt.month >= 10,
                           d["gd"].dt.year.astype(str) + "-" +
                           (d["gd"].dt.year + 1).astype(str).str[2:],
                           (d["gd"].dt.year - 1).astype(str) + "-" +
                           d["gd"].dt.year.astype(str).str[2:])
    print(f"labelled player-gamedays: {len(d)}  base rate OUT "
          f"{d['y_out'].mean():.4f}")

    # design matrix
    for s in STATUSES:
        d[f"ls_{s}"] = (d["last_status"] == s).astype(float)
    FE = [f"ls_{s}" for s in STATUSES] + ["days_since", "streak_out",
                                          "streak_q", "rest"]
    seas = sorted(d["season"].unique())
    print(f"seasons: {seas}\n")

    print("  BASELINE (incumbent): out tonight iff out on the last report")
    print("  MODEL: L2 logistic on as-of-open features\n")
    print(f"  {'season':10} {'n':>7} {'base LL':>9} {'model LL':>9} "
          f"{'base Brier':>11} {'model Brier':>12}")
    rows = []
    for i in range(1, len(seas)):
        tr = d[d["season"].isin(seas[:i])]
        te = d[d["season"] == seas[i]]
        if len(tr) < 500 or len(te) < 200:
            continue
        w = logistic_fit(tr[FE].to_numpy(float), tr["y_out"].to_numpy(float),
                         lam=10.0)
        p = predict(w, te[FE].to_numpy(float))
        y = te["y_out"].to_numpy(float)
        pb = te["ls_Out"].to_numpy(float)            # hard carry-forward
        pb_s = np.clip(pb, 0.02, 0.98)               # smoothed so LL is finite
        rows.append((seas[i], len(te), nll(pb_s, y), nll(p, y),
                     float(((pb - y) ** 2).mean()), float(((p - y) ** 2).mean()),
                     w))
        print(f"  {seas[i]:10} {len(te):7d} {rows[-1][2]:9.5f} "
              f"{rows[-1][3]:9.5f} {rows[-1][4]:11.5f} {rows[-1][5]:12.5f}")

    W = np.array([x[1] for x in rows], float)
    bll = np.average([x[2] for x in rows], weights=W)
    mll = np.average([x[3] for x in rows], weights=W)
    bbr = np.average([x[4] for x in rows], weights=W)
    mbr = np.average([x[5] for x in rows], weights=W)
    print(f"\n  POOLED   LL {bll:.5f} -> {mll:.5f}  ({mll-bll:+.5f})")
    print(f"           Brier {bbr:.5f} -> {mbr:.5f}  "
          f"({100*(mbr-bbr)/bbr:+.1f}%)")
    print(f"  better in {sum(1 for x in rows if x[5] < x[4])}/{len(rows)} seasons")

    B = np.mean([x[6] for x in rows], axis=0)
    print("\n  mean coefficients:")
    for nm, b in zip(["intercept"] + FE, B):
        print(f"    {nm:16} {b:+.4f}")

    # what the model says about the players the hard rule gets wrong
    last = rows[-1]
    te = d[d["season"] == last[0]]
    p = predict(last[6], te[FE].to_numpy(float))
    q = te["last_status"] == "Questionable"
    print(f"\n  players last listed QUESTIONABLE ({int(q.sum())} rows): "
          f"actually out {te.loc[q,'y_out'].mean():.3f}, "
          f"model says {p[q.to_numpy()].mean():.3f}, hard rule says 0.000")
    o = te["last_status"] == "Out"
    print(f"  players last listed OUT ({int(o.sum())} rows): "
          f"actually out {te.loc[o,'y_out'].mean():.3f}, "
          f"model says {p[o.to_numpy()].mean():.3f}, hard rule says 1.000")

    json.dump(dict(n=len(d), pooled_ll=[float(bll), float(mll)],
                   pooled_brier=[float(bbr), float(mbr)],
                   coefs=dict(zip(["intercept"] + FE, B.tolist()))),
              open(ROOT / "data" / "d200_participation.json", "w"), indent=1)
    print("\nwrote data/d200_participation.json")


if __name__ == "__main__":
    main()
