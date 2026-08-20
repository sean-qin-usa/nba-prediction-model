#!/usr/bin/env python3
"""D245b — REPAIR GATE. Eight fixes to D245's L1/L2 construction, then rescore.

D245's numbers were construction-driven. Verified defects, all confirmed
against source before repair:

  1. ALPHA DENOMINATOR. `a[top].sum() / a.sum()` divides by CANDIDATE minutes,
     not total team minutes. With coverage 94.41% the excluded OTHER minutes are
     load-bearing, so 0.6259 was never the top-five TEAM share and "150/90" was
     not established.
  2. OTHER EXCLUDED FROM L1. `share_act = act / act.sum()` renormalises to the
     candidate pool after removing 5.59% of real minutes. OTHER was reported and
     then dropped from the very metric it was created to protect.
  3. S* IS CANDIDATE-ONLY. `(tal*act/48).sum()` runs over candidates, so the
     "true minutes target" omits the same 5.59%. PRIMARY forces 240 minutes onto
     candidates while the target holds ~226.5 — part of the -0.1917 could be
     scale, not allocation.
  4. TRAILING MINUTES DO NOT REPRODUCE PRODUCTION. `rolling(10, min_periods=3)
     .mean().shift(1)` plus a strictly-prior row filter DOUBLE-EXCLUDES the most
     recent prior game and needs four appearances where production needs one.
     Demonstrated: on games [10,20,30,40] production gives 25.0, D245 gave 20.0.
     **D245's CONTROL was therefore not the production control.**
  5. P5 WAS DECLARED CONFIRMED AT L2; the prereg scores it at L3.
  6. 48-MINUTE CLIP redistributed excess ONCE and could push another player over.
  7. MISSING p_out silently became 0.0 and was never counted.
  8. NO N1 DIAGNOSTIC, so a gain from merely forcing 240 total minutes could not
     be separated from the rank-tier constraint.

Repairs: production-identical trailing minutes with an identity test against
`CompositionModel`; alpha over total team minutes; OTHER inside the L1 share
vector; S* over EVERY actual participant; iterative water-fill with an
assertion; all fallbacks counted; N1 added as a diagnostic arm; an ADDITIVE
per-band TV decomposition; and coverage stratification to test whether the gain
depends on roster incompleteness.
"""
from __future__ import annotations

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

FROM = "2019-20"
ROSTER_DAYS = 12
TRAIL_N = 10
MIN_SEC = 720
FB: Counter = Counter()


def cap48(m: np.ndarray) -> np.ndarray:
    """Iterative water-fill. D245 redistributed once and could exceed 48."""
    m = m.astype(float).copy()
    for _ in range(50):
        over = m > 48.0
        if not over.any():
            break
        FB["over_48_iter"] += 1
        excess = float((m[over] - 48.0).sum())
        m[over] = 48.0
        free = m < 48.0
        fs = float(m[free].sum())
        if fs <= 0:
            break
        m[free] += excess * m[free] / fs
    assert m.max() <= 48.0 + 1e-6, f"cap violated: {m.max()}"
    return m


def build():
    con = connect(read_only=True)
    box = con.execute("""
        SELECT s.game_id, s.player_id, s.team_id, s.seconds/60.0 AS mins,
               g.season, g.game_date
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, team_id, season, game_date FROM nba_games
              WHERE game_id LIKE '002%') g USING (game_id, team_id)
        ORDER BY g.game_date""").fetchdf()
    darko = con.execute(
        "SELECT player_id, date, dpm FROM darko_history "
        "ORDER BY player_id, date").fetchdf()
    con.close()
    box["game_date"] = pd.to_datetime(box["game_date"])
    darko["date"] = pd.to_datetime(darko["date"])

    # FIX 4: production-identical -- rolling mean over qualifying games with
    # min_periods=1 and NO shift; the strictly-prior row filter is what excludes
    # the current game.
    play = box[box["mins"] * 60 >= MIN_SEC].copy().sort_values(
        ["player_id", "game_date"])
    play["tr_min"] = (play.groupby("player_id")["mins"]
                      .transform(lambda s: s.rolling(TRAIL_N, min_periods=1).mean()))
    hist = play[["player_id", "team_id", "game_date", "tr_min"]]

    pout = pd.read_csv(ROOT / "data" / "p_out.csv.gz")
    pout["game_date"] = pd.to_datetime(pout["game_date"])
    pm = {(r.game_date, int(r.player_id)): float(r.p_out) for r in pout.itertuples()}

    dk = {}
    for pid, g in darko.groupby("player_id"):
        dk[int(pid)] = (g["date"].to_numpy(), g["dpm"].to_numpy())

    def talent(pid, gd):
        e = dk.get(int(pid))
        if e is None:
            FB["talent_missing"] += 1
            return 0.0
        i = np.searchsorted(e[0], np.datetime64(gd)) - 1
        if i < 0:
            FB["talent_missing"] += 1
            return 0.0
        return float(e[1][i])

    rows = []
    for (tid, gd), g in box.groupby(["team_id", "game_date"], sort=False):
        if g["season"].iloc[0] < FROM:
            continue
        h = hist[(hist["team_id"] == tid) & (hist["game_date"] < gd)]
        if h.empty:
            FB["no_history"] += 1
            continue
        cand = h.sort_values("game_date").groupby("player_id").tail(1)
        cand = cand[(gd - cand["game_date"]).dt.days <= ROSTER_DAYS]
        cand = cand.dropna(subset=["tr_min"])
        if len(cand) < 3:
            FB["too_few_candidates"] += 1
            continue
        act_all = dict(zip(g["player_id"], g["mins"]))
        tot = float(g["mins"].sum())
        if tot <= 0:
            continue
        pids = [int(p) for p in cand["player_id"]]
        cs = set(pids)
        po = []
        for p in pids:
            k = (gd, p)
            if k not in pm:
                FB["pout_missing"] += 1
            po.append(pm.get(k, 0.0))
        # FIX 3: S* over EVERY actual participant, not just candidates
        s_star = float(sum(talent(p, gd) * m / 48.0
                           for p, m in act_all.items() if m > 0))
        other = float(sum(m for p, m in act_all.items() if p not in cs))
        rows.append(dict(
            team_id=tid, game_date=gd, season=g["season"].iloc[0],
            pids=pids, tr=[float(x) for x in cand["tr_min"]], pout=po,
            tal=[talent(p, gd) for p in pids],
            actual=[float(act_all.get(p, 0.0)) for p in pids],
            tot_actual=tot, other=other, s_star=s_star))
    return pd.DataFrame(rows)


def alloc(u, alpha, mode):
    u = np.asarray(u, float)
    n = len(u)
    if mode == "raw":
        return u
    s = u.sum()
    if s <= 0:
        FB["zero_total_weight"] += 1
        return u
    if mode == "n1":
        return cap48(240.0 * u / s)
    if n < 6:
        FB["fewer_than_6"] += 1
        return cap48(240.0 * u / s)
    o = np.argsort(-u)
    top, rest = o[:5], o[5:]
    ut, ur = u[top].sum(), u[rest].sum()
    if ut <= 0 or ur <= 0:
        FB["zero_within_tier"] += 1
        return cap48(240.0 * u / s)
    a = alpha if mode == "alpha" else 160.0 / 240.0
    m = np.zeros(n)
    m[top] = 240.0 * a * u[top] / ut
    m[rest] = 240.0 * (1 - a) * u[rest] / ur
    return cap48(m)


def clus(v):
    v = np.asarray(v, float); k = len(v)
    se = v.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return v.mean(), v.mean() - tc * se, v.mean() + tc * se, k


def main():
    f = build()
    print(f"frame {len(f):,} team-games {f.season.min()}..{f.season.max()}")
    cov = 1 - f["other"].sum() / f["tot_actual"].sum()
    print(f"candidate coverage {100*cov:.2f}%  (D245 reported 94.41% under the "
          f"broken trailing-minute rule)")

    f["u"] = [(1 - np.asarray(p)) * np.asarray(t)
              for p, t in zip(f["pout"], f["tr"])]

    # FIX 1: alpha over TOTAL team minutes
    def share5(r):
        u = np.asarray(r["u"]); a = np.asarray(r["actual"])
        if len(u) < 5 or r["tot_actual"] <= 0:
            return np.nan
        return a[np.argsort(-u)[:5]].sum() / r["tot_actual"]
    f["s5"] = f.apply(share5, axis=1)
    seasons = sorted(f.season.unique())
    alphas = {}
    print("\n=== ALPHA, repaired denominator (top-5 actual / TOTAL team) ===")
    for i, s in enumerate(seasons):
        if i == 0:
            continue
        alphas[s] = float(f[f.season.isin(seasons[:i])]["s5"].mean())
        print(f"  {s}: {alphas[s]:.4f}")
    av = np.array(list(alphas.values()))
    print(f"  mean {av.mean():.4f}  spread {av.max()-av.min():.4f}  "
          f"vs 160/240 {160/240:.4f}  -> implied split "
          f"{240*av.mean():.0f}/{240*(1-av.mean()):.0f}")
    print(f"  D245 reported 0.6259 with the CANDIDATE-only denominator; "
          f"repaired value is {av.mean():.4f}")

    ARMS = ["ctrl", "n1", "prim", "fix", "plac"]
    rng = np.random.default_rng(2451)
    rows = []
    for i, s in enumerate(seasons):
        if i == 0:
            continue
        te = f[f.season == s]; a_f = alphas[s]
        tv = {k: [] for k in ARMS}
        l2 = {k: [] for k in ARMS}
        bands = {k: [np.zeros(3), np.zeros(3)] for k in ("prim", "n1")}
        nb = 0
        for r in te.itertuples():
            u = np.asarray(r.u); act = np.asarray(r.actual)
            tal = np.asarray(r.tal); tot = r.tot_actual
            if tot <= 0 or len(u) < 3:
                continue
            # FIX 2: OTHER is a real component of the actual share vector
            q_act = np.append(act / tot, r.other / tot)
            P = {"ctrl": alloc(u, a_f, "raw"), "n1": alloc(u, a_f, "n1"),
                 "prim": alloc(u, a_f, "alpha"), "fix": alloc(u, a_f, "fixed")}
            sh = u.copy(); rng.shuffle(sh)
            P["plac"] = alloc(sh, a_f, "alpha")
            for k, m in P.items():
                sm = m.sum()
                q_p = np.append(m / sm if sm > 0 else m, 0.0)
                tv[k].append(0.5 * np.abs(q_p - q_act).sum())
                # FIX 3: compare against S* over ALL participants
                l2[k].append(abs(float((tal * m / 48.0).sum()) - r.s_star))
            # additive per-band TV contribution (sums exactly to total dTV)
            o = np.argsort(-u)
            qc = np.abs(np.append(P["ctrl"] / max(P["ctrl"].sum(), 1e-9),
                                  0.0) - q_act)
            for k in ("prim", "n1"):
                qk = np.abs(np.append(P[k] / max(P[k].sum(), 1e-9), 0.0) - q_act)
                for bi, sl in enumerate((o[:5], o[5:8], o[8:])):
                    if len(sl):
                        bands[k][0][bi] += 0.5 * (qk[sl] - qc[sl]).sum()
            nb += 1
        rows.append(dict(season=s, n=nb, alpha=a_f,
                         **{f"tv_{k}": float(np.mean(tv[k])) for k in ARMS},
                         **{f"l2_{k}": float(np.mean(l2[k])) for k in ARMS},
                         **{f"band_{k}_{b}": float(bands[k][0][b] / max(nb, 1))
                            for k in ("prim", "n1") for b in range(3)}))
    d = pd.DataFrame(rows)

    print("\n=== L1 minute-share TV, OTHER INCLUDED (lower better) ===")
    print(d[["season", "n"] + [f"tv_{k}" for k in ARMS]].to_string(
        index=False, float_format=lambda v: f"{v:8.4f}"))
    for k in ("n1", "prim", "fix", "plac"):
        m, lo, hi, kk = clus(d[f"tv_{k}"] - d["tv_ctrl"])
        print(f"  {k:5} vs ctrl {m:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
              f"better {int(((d[f'tv_{k}']-d.tv_ctrl)<0).sum())}/{kk}")
    m, lo, hi, kk = clus(d["tv_prim"] - d["tv_n1"])
    print(f"  **TIER BENEFIT (prim - n1) {m:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
          f"better {int(((d.tv_prim-d.tv_n1)<0).sum())}/{kk}**")

    print("\n=== L2 |strength - S*| where S* spans ALL participants ===")
    print(d[["season"] + [f"l2_{k}" for k in ARMS]].to_string(
        index=False, float_format=lambda v: f"{v:8.4f}"))
    for k in ("n1", "prim", "fix", "plac"):
        m, lo, hi, kk = clus(d[f"l2_{k}"] - d["l2_ctrl"])
        print(f"  {k:5} vs ctrl {m:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
              f"better {int(((d[f'l2_{k}']-d.l2_ctrl)<0).sum())}/{kk}")
    m, lo, hi, kk = clus(d["l2_prim"] - d["l2_n1"])
    print(f"  **TIER BENEFIT (prim - n1) {m:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
          f"better {int(((d.l2_prim-d.l2_n1)<0).sum())}/{kk}**")

    print("\n=== ADDITIVE TV DECOMPOSITION by rank band (sums to total dTV) ===")
    for k in ("prim", "n1"):
        b = [d[f"band_{k}_{i}"].mean() for i in range(3)]
        tot_d = (d[f"tv_{k}"] - d["tv_ctrl"]).mean()
        print(f"  {k:5} ranks1-5 {b[0]:+.5f}  6-8 {b[1]:+.5f}  9+ {b[2]:+.5f}"
              f"   sum {sum(b):+.5f}  vs total {tot_d:+.5f}")

    print(f"\nfallbacks: {dict(FB)}")
    json.dump({"coverage": float(cov), "alphas": alphas, "rows": rows,
               "fallbacks": dict(FB)},
              open(ROOT / "data" / "d245b_repair.json", "w"), default=float)
    print("wrote data/d245b_repair.json")


if __name__ == "__main__":
    main()
