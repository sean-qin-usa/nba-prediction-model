#!/usr/bin/env python3
"""D245c — VALIDATION PASS before L3/L4. Six clarifications, no redesign.

  1. ARM-SPECIFIC CAP COUNTERS + AN UNCAPPED N1. D245b's `over_48_iter=125` was
     a single counter shared by every arm, so it could not tell whether N1's
     -0.0781 was normalisation or normalisation-PLUS-cap. If capped and uncapped
     N1 coincide, the "43% is mass normalisation" statement is justified.
  2. `pout_missing` -> `pout_default_zero`. Absence from the artifact is a
     STRUCTURAL DEFAULT under production semantics, not a data failure, and
     naming it "missing" overstates it.
  3. OVERTIME-NORMALISED S*. Raw S* uses realised minutes, which exceed 240 in
     overtime while every allocator predicts exactly 240 — injecting game-length
     variation into the target. A normalised target is scored alongside, and the
     tier benefit must survive it.
  4. COVERAGE STRATIFICATION, WHICH D245b PROMISED AND DID NOT IMPLEMENT.
     PRIMARY-minus-N1 split by OTHER == 0 vs OTHER > 0, plus a descriptive slope
     of the L1 delta on OTHER share. This is what decides whether the gain is a
     genuine rank-weight correction or roster regularisation.
  5. IDENTITY TEST PERSISTED. The 400/400 production match existed only in
     register prose; it is computed here and written to the artifact.
  6. Placebo is reported as a SANITY CHECK, not as decisive validity evidence.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from scipy import stats                                           # noqa: E402

from nbapred.db import connect                                    # noqa: E402
from nbapred.model.composition import CompositionModel            # noqa: E402

FROM, ROSTER_DAYS, TRAIL_N, MIN_SEC = "2019-20", 12, 10, 720
FB: Counter = Counter()


def cap48(m, arm):
    m = np.asarray(m, float).copy()
    it = 0
    for _ in range(50):
        over = m > 48.0
        if not over.any():
            break
        it += 1
        FB[f"{arm}_over_48_players"] += int(over.sum())
        ex = float((m[over] - 48.0).sum())
        m[over] = 48.0
        free = m < 48.0
        fs = float(m[free].sum())
        if fs <= 0:
            break
        m[free] += ex * m[free] / fs
    if it:
        FB[f"{arm}_over_48_games"] += 1
        FB[f"{arm}_waterfill_iterations"] += it
    assert m.max() <= 48.0 + 1e-6
    return m


def identity_test(con, dates=(dt.date(2022, 12, 1), dt.date(2023, 1, 15),
                              dt.date(2024, 3, 1), dt.date(2025, 2, 10))):
    box = con.execute("""
        SELECT s.player_id, s.seconds/60.0 mins, g.game_date
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games
              WHERE game_id LIKE '002%') g USING (game_id)
        WHERE s.seconds >= 720 ORDER BY s.player_id, g.game_date""").fetchdf()
    box["game_date"] = pd.to_datetime(box["game_date"])
    out = []
    for d in dates:
        cm = CompositionModel(con, before=d)
        h = box[box["game_date"] < pd.Timestamp(d)]
        n = mism = 0
        worst = 0.0
        for pid in cm.players:
            g = h[h["player_id"] == pid].tail(TRAIL_N)
            if g.empty:
                continue
            diff = abs(float(g["mins"].mean()) - cm.players[pid]["trail_min"])
            worst = max(worst, diff)
            n += 1
            if diff > 1e-9:
                mism += 1
        out.append(dict(date=str(d), comparisons=n, mismatches=mism,
                        max_abs_diff=worst))
        print(f"    {d}: {n} comparisons, {mism} mismatches, "
              f"max|d| {worst:.2e}")
    return out


def build(con):
    box = con.execute("""
        SELECT s.game_id, s.player_id, s.team_id, s.seconds/60.0 mins,
               g.season, g.game_date
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, team_id, season, game_date FROM nba_games
              WHERE game_id LIKE '002%') g USING (game_id, team_id)
        ORDER BY g.game_date""").fetchdf()
    dk_df = con.execute("SELECT player_id, date, dpm FROM darko_history "
                        "ORDER BY player_id, date").fetchdf()
    box["game_date"] = pd.to_datetime(box["game_date"])
    dk_df["date"] = pd.to_datetime(dk_df["date"])
    play = box[box["mins"] * 60 >= MIN_SEC].copy().sort_values(
        ["player_id", "game_date"])
    play["tr_min"] = (play.groupby("player_id")["mins"]
                      .transform(lambda s: s.rolling(TRAIL_N, min_periods=1).mean()))
    hist = play[["player_id", "team_id", "game_date", "tr_min"]]
    po = pd.read_csv(ROOT / "data" / "p_out.csv.gz")
    po["game_date"] = pd.to_datetime(po["game_date"])
    pm = {(r.game_date, int(r.player_id)): float(r.p_out) for r in po.itertuples()}
    dk = {int(p): (g["date"].to_numpy(), g["dpm"].to_numpy())
          for p, g in dk_df.groupby("player_id")}

    def tal(pid, gd):
        e = dk.get(int(pid))
        if e is None:
            FB["talent_default_zero"] += 1
            return 0.0
        i = np.searchsorted(e[0], np.datetime64(gd)) - 1
        if i < 0:
            FB["talent_default_zero"] += 1
            return 0.0
        return float(e[1][i])

    rows = []
    for (tid, gd), g in box.groupby(["team_id", "game_date"], sort=False):
        if g["season"].iloc[0] < FROM:
            continue
        h = hist[(hist["team_id"] == tid) & (hist["game_date"] < gd)]
        if h.empty:
            continue
        c = h.sort_values("game_date").groupby("player_id").tail(1)
        c = c[(gd - c["game_date"]).dt.days <= ROSTER_DAYS].dropna(subset=["tr_min"])
        if len(c) < 3:
            FB["too_few_candidates"] += 1
            continue
        act = dict(zip(g["player_id"], g["mins"]))
        tot = float(g["mins"].sum())
        if tot <= 0:
            continue
        pids = [int(p) for p in c["player_id"]]
        cs = set(pids)
        pv = []
        for p in pids:
            if (gd, p) not in pm:
                FB["pout_default_zero"] += 1
            pv.append(pm.get((gd, p), 0.0))
        s_raw = float(sum(tal(p, gd) * m / 48.0 for p, m in act.items() if m > 0))
        s_nrm = float(sum(tal(p, gd) * (240.0 * m / tot) / 48.0
                          for p, m in act.items() if m > 0))
        rows.append(dict(season=g["season"].iloc[0], game_date=gd,
                         pids=pids, tr=[float(x) for x in c["tr_min"]],
                         pout=pv, tal=[tal(p, gd) for p in pids],
                         actual=[float(act.get(p, 0.0)) for p in pids],
                         tot_actual=tot,
                         other=float(sum(m for p, m in act.items() if p not in cs)),
                         s_raw=s_raw, s_nrm=s_nrm))
    return pd.DataFrame(rows)


def alloc(u, alpha, mode, arm):
    u = np.asarray(u, float); n = len(u); s = u.sum()
    if mode == "raw":
        return u
    if s <= 0:
        FB[f"{arm}_zero_total"] += 1
        return u
    if mode == "n1":
        return cap48(240.0 * u / s, arm)
    if mode == "n1_nocap":
        return 240.0 * u / s
    if n < 6:
        FB[f"{arm}_fewer_than_6"] += 1
        return cap48(240.0 * u / s, arm)
    o = np.argsort(-u); top, rest = o[:5], o[5:]
    ut, ur = u[top].sum(), u[rest].sum()
    if ut <= 0 or ur <= 0:
        FB[f"{arm}_zero_tier"] += 1
        return cap48(240.0 * u / s, arm)
    a = alpha if mode == "alpha" else 160.0 / 240.0
    m = np.zeros(n)
    m[top] = 240.0 * a * u[top] / ut
    m[rest] = 240.0 * (1 - a) * u[rest] / ur
    return cap48(m, arm)


def clus(v):
    v = np.asarray(v, float); k = len(v)
    se = v.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return v.mean(), v.mean() - tc * se, v.mean() + tc * se, k


def main():
    con = connect(read_only=True)
    print("=== (5) IDENTITY TEST vs CompositionModel, now PERSISTED ===")
    ident = identity_test(con)
    f = build(con)
    con.close()
    print(f"\nframe {len(f):,} team-games")

    f["u"] = [(1 - np.asarray(p)) * np.asarray(t)
              for p, t in zip(f["pout"], f["tr"])]
    f["oshare"] = f["other"] / f["tot_actual"]

    def s5(r):
        u = np.asarray(r["u"]); a = np.asarray(r["actual"])
        return (np.nan if len(u) < 5 or r["tot_actual"] <= 0
                else a[np.argsort(-u)[:5]].sum() / r["tot_actual"])
    f["s5"] = f.apply(s5, axis=1)
    seasons = sorted(f.season.unique())
    alphas = {s: float(f[f.season.isin(seasons[:i])]["s5"].mean())
              for i, s in enumerate(seasons) if i}

    ARMS = ["ctrl", "n1", "n1_nocap", "prim", "fix", "plac"]
    MODE = {"ctrl": "raw", "n1": "n1", "n1_nocap": "n1_nocap",
            "prim": "alpha", "fix": "fixed", "plac": "alpha"}
    rng = np.random.default_rng(2452)
    rows, strat = [], []
    for i, s in enumerate(seasons):
        if i == 0:
            continue
        te = f[f.season == s]; a_f = alphas[s]
        acc = {f"{m}_{k}": [] for m in ("tv", "l2r", "l2n") for k in ARMS}
        for r in te.itertuples():
            u = np.asarray(r.u); act = np.asarray(r.actual)
            tal = np.asarray(r.tal); tot = r.tot_actual
            if tot <= 0 or len(u) < 3:
                continue
            q_act = np.append(act / tot, r.other / tot)
            P = {}
            for k in ARMS:
                uu = u.copy()
                if k == "plac":
                    rng.shuffle(uu)
                P[k] = alloc(uu, a_f, MODE[k], k)
            for k, m in P.items():
                sm = m.sum()
                acc[f"tv_{k}"].append(
                    0.5 * np.abs(np.append(m / sm if sm > 0 else m, 0.0) - q_act).sum())
                pred = float((tal * m / 48.0).sum())
                acc[f"l2r_{k}"].append(abs(pred - r.s_raw))
                acc[f"l2n_{k}"].append(abs(pred - r.s_nrm))
            strat.append(dict(season=s, oshare=r.oshare,
                              d_tv=acc["tv_prim"][-1] - acc["tv_n1"][-1],
                              d_l2=acc["l2r_prim"][-1] - acc["l2r_n1"][-1]))
        rows.append(dict(season=s, alpha=a_f,
                         **{k: float(np.mean(v)) for k, v in acc.items()}))
    d = pd.DataFrame(rows); st = pd.DataFrame(strat)

    print("\n=== (1) IS N1's GAIN NORMALISATION, OR NORMALISATION+CAP? ===")
    cap_keys = {k: v for k, v in FB.items() if "over_48" in k}
    print(f"  arm-specific cap counters: {cap_keys or 'NONE — cap never bound'}")
    m, lo, hi, k = clus(d["l2r_n1"] - d["l2r_n1_nocap"])
    print(f"  N1(capped) - N1(uncapped) at L2: {m:+.6f}  CI [{lo:+.6f}, {hi:+.6f}]")
    verdict = ("the cap contributes essentially nothing; the 43% claim stands"
               if abs(m) < 1e-4
               else "the cap DOES contribute; the 43% claim must be restated")
    print(f"  -> {verdict}")

    print("\n=== (3) TIER BENEFIT UNDER RAW vs OVERTIME-NORMALISED S* ===")
    for tag, lab in (("l2r", "raw S*"), ("l2n", "OT-normalised S*")):
        a1, lo1, hi1, _ = clus(d[f"{tag}_prim"] - d[f"{tag}_ctrl"])
        a2, lo2, hi2, kk = clus(d[f"{tag}_prim"] - d[f"{tag}_n1"])
        print(f"  {lab:20} prim-ctrl {a1:+.4f} [{lo1:+.4f},{hi1:+.4f}]   "
              f"prim-n1 {a2:+.4f} [{lo2:+.4f},{hi2:+.4f}]  "
              f"{int(((d[f'{tag}_prim']-d[f'{tag}_n1'])<0).sum())}/{kk}")

    print("\n=== (4) COVERAGE STRATIFICATION — the promised diagnostic ===")
    z = st[st.oshare <= 1e-9]; nz = st[st.oshare > 1e-9]
    print(f"  OTHER == 0 : n={len(z):,}  tier benefit dTV {z.d_tv.mean():+.5f}  "
          f"dL2 {z.d_l2.mean():+.4f}")
    print(f"  OTHER >  0 : n={len(nz):,}  tier benefit dTV {nz.d_tv.mean():+.5f}  "
          f"dL2 {nz.d_l2.mean():+.4f}")
    for col, lab in (("d_tv", "L1"), ("d_l2", "L2")):
        b = np.polyfit(st.oshare, st[col], 1)
        print(f"  slope of {lab} tier benefit on OTHER share: {b[0]:+.5f} "
              f"(intercept {b[1]:+.5f})")
    print("  interpretation: benefit at OTHER==0 is a genuine rank-weight")
    print("  correction; benefit only where OTHER>0 would be roster regularisation.")

    json.dump({"identity_test": ident, "alphas": alphas, "rows": rows,
               "fallbacks": dict(FB),
               "strat": {"other_zero_n": int(len(z)),
                         "other_zero_dtv": float(z.d_tv.mean()),
                         "other_pos_n": int(len(nz)),
                         "other_pos_dtv": float(nz.d_tv.mean())}},
              open(ROOT / "data" / "d245c_validation.json", "w"), default=float)
    print("\nwrote data/d245c_validation.json")


if __name__ == "__main__":
    main()
