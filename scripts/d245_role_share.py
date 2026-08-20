#!/usr/bin/env python3
"""D245 — ROLE-SHARE MINUTE ALLOCATION, layers 1-2. Prereg sha256 1d145732...

D242's N3 used a HAND-CHOSEN 160/80 tier split, selected from an eight-cell
sweep on these seasons, confounding tier share, a rank-5/6 discontinuity,
within-tier normalisation, extreme-weight shrinkage and roster-size
sensitivity. This replaces it with ONE parameter estimated from realised minute
shares -- never from margin or log loss -- and verifies the causal path from
player minutes upward instead of reading a margin RMSE and inferring a
mechanism.

Layers here:
  L1  minute allocation  -- total-variation distance on minute SHARES, MAE,
                            error by rank band, >48-minute violations
  L2  lineup strength    -- against S*_g = sum_i v_i * M_actual_i / 48, the
                            TRUE minutes target D242's "oracle" never was

CANDIDATE COVERAGE IS REPORTED FIRST AND IS LOAD-BEARING. If the 12-day
candidate window misses a material share of actual minutes, any gain from
rank-tiering may be roster REGULARISATION rather than role allocation, and the
non-candidate minutes are put in an explicit OTHER bucket rather than dropped.
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
ROSTER_DAYS = 12          # matches CompositionModel.ROSTER_DAYS
TRAIL_N = 10
MIN_SEC = 720             # the leg's own >=12-minute filter
FALLBACKS: Counter = Counter()


def build_frame() -> pd.DataFrame:
    con = connect(read_only=True)
    box = con.execute("""
        SELECT s.game_id, s.player_id, s.team_id, s.seconds/60.0 AS mins,
               g.season, g.game_date
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, team_id, season, game_date FROM nba_games
              WHERE game_id LIKE '002%') g USING (game_id, team_id)
        ORDER BY g.game_date""").fetchdf()
    darko = con.execute("""
        SELECT player_id, date, dpm FROM darko_history ORDER BY player_id, date
    """).fetchdf()
    con.close()
    box["game_date"] = pd.to_datetime(box["game_date"])
    darko["date"] = pd.to_datetime(darko["date"])

    # trailing minutes over the last TRAIL_N games with >= MIN_SEC, strictly prior
    play = box[box["mins"] * 60 >= MIN_SEC].copy().sort_values(
        ["player_id", "game_date"])
    play["tr_min"] = (play.groupby("player_id")["mins"]
                      .transform(lambda s: s.rolling(TRAIL_N, min_periods=3)
                                 .mean().shift(1)))
    last = play[["player_id", "team_id", "game_date", "tr_min"]]

    pout = pd.read_csv(ROOT / "data" / "p_out.csv.gz")
    pout["game_date"] = pd.to_datetime(pout["game_date"])

    rows = []
    for (tid, gd), g in box.groupby(["team_id", "game_date"], sort=False):
        if g["season"].iloc[0] < FROM:
            continue
        hist = last[(last["team_id"] == tid) & (last["game_date"] < gd)]
        if hist.empty:
            continue
        cand = hist.sort_values("game_date").groupby("player_id").tail(1)
        cand = cand[(gd - cand["game_date"]).dt.days <= ROSTER_DAYS]
        cand = cand.dropna(subset=["tr_min"])
        if len(cand) < 3:
            FALLBACKS["too_few_candidates"] += 1
            continue
        actual = dict(zip(g["player_id"], g["mins"]))
        tot_actual = float(g["mins"].sum())
        if tot_actual <= 0:
            continue
        cset = set(cand["player_id"])
        cand_actual = float(sum(v for k, v in actual.items() if k in cset))
        rows.append(dict(
            team_id=tid, game_date=gd, season=g["season"].iloc[0],
            game_id=g["game_id"].iloc[0],
            pids=list(cand["player_id"]),
            tr=list(cand["tr_min"]),
            actual=[float(actual.get(p, 0.0)) for p in cand["player_id"]],
            tot_actual=tot_actual, cand_actual=cand_actual,
            other_minutes=tot_actual - cand_actual))
    f = pd.DataFrame(rows)
    # opener-time p_out per (date, player)
    pm = {(r.game_date, int(r.player_id)): float(r.p_out)
          for r in pout.itertuples()}
    f["pout"] = [[pm.get((d, int(p)), 0.0) for p in pl]
                 for d, pl in zip(f["game_date"], f["pids"])]
    # PIT talent: last darko strictly before the game
    dk = {}
    for pid, grp in darko.groupby("player_id"):
        dk[int(pid)] = (grp["date"].to_numpy(), grp["dpm"].to_numpy())

    def talent(pid, gd):
        e = dk.get(int(pid))
        if e is None:
            return 0.0
        i = np.searchsorted(e[0], np.datetime64(gd)) - 1
        return float(e[1][i]) if i >= 0 else 0.0
    f["tal"] = [[talent(p, d) for p in pl]
                for d, pl in zip(f["game_date"], f["pids"])]
    return f


def allocate(u, alpha, mode):
    u = np.asarray(u, float)
    n = len(u)
    if mode == "raw":
        return u
    if n < 6:
        FALLBACKS["fewer_than_6_candidates"] += 1
        s = u.sum()
        return u * (240.0 / s) if s > 0 else u
    order = np.argsort(-u)
    top, rest = order[:5], order[5:]
    ut, ur = u[top].sum(), u[rest].sum()
    m = np.zeros(n)
    if ut <= 0 or ur <= 0:
        FALLBACKS["zero_within_tier_weight"] += 1
        s = u.sum()
        return u * (240.0 / s) if s > 0 else u
    a = alpha if mode == "alpha" else (160.0 / 240.0)
    m[top] = 240.0 * a * u[top] / ut
    m[rest] = 240.0 * (1 - a) * u[rest] / ur
    over = m > 48.0
    if over.any():
        FALLBACKS["over_48_clipped"] += int(over.sum())
        excess = (m[over] - 48.0).sum()
        m[over] = 48.0
        free = ~over
        if m[free].sum() > 0:
            m[free] += excess * m[free] / m[free].sum()
    return m


def clus(v):
    v = np.asarray(v, float); k = len(v)
    se = v.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return v.mean(), v.mean() - tc * se, v.mean() + tc * se, v.mean() / se, k


def main():
    f = build_frame()
    print(f"frame {len(f):,} team-games, seasons "
          f"{f.season.min()}..{f.season.max()}")

    # ---------- CANDIDATE COVERAGE (load-bearing) ---------------------
    cov = f["cand_actual"].sum() / f["tot_actual"].sum()
    print("\n" + "=" * 68)
    print("CANDIDATE COVERAGE — is the 12-day window missing real minutes?")
    print("=" * 68)
    print(f"  actual minutes belonging to candidates : {100*cov:.2f}%")
    print(f"  OTHER bucket (non-candidate minutes)   : "
          f"{100*(1-cov):.2f}%  = {f['other_minutes'].mean():.2f} min/team-game")
    print(f"  team-games with ANY non-candidate minutes: "
          f"{100*(f['other_minutes'] > 0.5).mean():.1f}%")
    print(f"  P6 {'CONFIRMED' if cov < 0.999 else 'REFUTED'}: coverage "
          f"{'<' if cov < 0.999 else '>='} 1.00")

    # ---------- alpha, from MINUTE SHARES only -------------------------
    f["u"] = [(1 - np.asarray(p)) * np.asarray(t)
              for p, t in zip(f["pout"], f["tr"])]

    def top5_share(row):
        u = np.asarray(row["u"]); a = np.asarray(row["actual"])
        if len(u) < 5 or a.sum() <= 0:
            return np.nan
        top = np.argsort(-u)[:5]
        return a[top].sum() / a.sum()
    f["share5"] = f.apply(top5_share, axis=1)
    seasons = sorted(f.season.unique())
    print("\n" + "=" * 68)
    print("ALPHA — estimated from realised minute shares, never from margin")
    print("=" * 68)
    alphas = {}
    for i, s in enumerate(seasons):
        if i == 0:
            continue
        tr = f[f.season.isin(seasons[:i])]
        alphas[s] = float(tr["share5"].mean())
        print(f"  fold {s}: alpha = {alphas[s]:.4f}   (n_train "
              f"{tr['share5'].notna().sum():,})")
    av = np.array(list(alphas.values()))
    print(f"  mean {av.mean():.4f}  spread {av.max()-av.min():.4f}  "
          f"vs 160/240 = {160/240:.4f}")
    print(f"  P1 {'CONFIRMED' if abs(av.mean()-2/3) < 0.05 else 'REFUTED'} "
          f"(|alpha - 2/3| = {abs(av.mean()-2/3):.4f})")
    print(f"  P2 {'CONFIRMED' if av.max()-av.min() < 0.05 else 'REFUTED'} "
          f"(spread {av.max()-av.min():.4f})")

    # ---------- L1 / L2 ------------------------------------------------
    rng = np.random.default_rng(245)
    rows = []
    for i, s in enumerate(seasons):
        if i == 0:
            continue
        te = f[f.season == s]
        a_f = alphas[s]
        acc = {k: [] for k in ("tv_ctrl", "tv_prim", "tv_fix", "tv_plac",
                               "l2_ctrl", "l2_prim", "l2_fix", "l2_orac")}
        band = {k: [[], [], []] for k in ("ctrl", "prim")}
        for r in te.itertuples():
            u = np.asarray(r.u); act = np.asarray(r.actual)
            tal = np.asarray(r.tal)
            ta = act.sum()
            if ta <= 0 or len(u) < 3:
                continue
            share_act = act / ta
            preds = {"ctrl": allocate(u, a_f, "raw"),
                     "prim": allocate(u, a_f, "alpha"),
                     "fix": allocate(u, a_f, "fixed")}
            pl = u.copy(); rng.shuffle(pl)
            preds["plac"] = allocate(pl, a_f, "alpha")
            for k, m in preds.items():
                sm = m.sum()
                acc[f"tv_{k}"].append(
                    0.5 * np.abs((m / sm if sm > 0 else m) - share_act).sum())
            s_star = float((tal * act / 48.0).sum())
            for k in ("ctrl", "prim", "fix"):
                m = preds[k]
                acc[f"l2_{k}"].append(
                    abs(float((tal * m / 48.0).sum()) - s_star))
            acc["l2_orac"].append(0.0)
            order = np.argsort(-u)
            for bi, sl in enumerate((order[:5], order[5:8], order[8:])):
                if len(sl):
                    for k in ("ctrl", "prim"):
                        band[k][bi].append(
                            float(np.abs(preds[k][sl] - act[sl]).mean()))
        rows.append(dict(season=s, n=len(te), alpha=a_f,
                         **{k: float(np.mean(v)) for k, v in acc.items()},
                         **{f"mae_{k}_{b}": float(np.mean(band[k][b]))
                            for k in ("ctrl", "prim") for b in range(3)}))
    d = pd.DataFrame(rows)
    print("\n" + "=" * 68)
    print("L1 — MINUTE-SHARE TOTAL-VARIATION DISTANCE (lower better)")
    print("=" * 68)
    print(d[["season", "n", "alpha", "tv_ctrl", "tv_prim", "tv_fix",
             "tv_plac"]].to_string(index=False,
                                   float_format=lambda v: f"{v:8.4f}"))
    for k, lab in (("tv_prim", "PRIMARY (alpha)"), ("tv_fix", "FIXED 160/80"),
                   ("tv_plac", "PLACEBO (ranks shuffled)")):
        m, lo, hi, t, kk = clus(d[k] - d["tv_ctrl"])
        print(f"  {lab:26} vs ctrl {m:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
              f"better {int(((d[k]-d['tv_ctrl'])<0).sum())}/{kk}")
    print(f"  P3 {'CONFIRMED' if (d.tv_prim-d.tv_ctrl).mean() < 0 else 'REFUTED'}")

    print("\n" + "=" * 68)
    print("L2 — |predicted strength - S*| where S* uses ACTUAL minutes")
    print("=" * 68)
    print(d[["season", "l2_ctrl", "l2_prim", "l2_fix"]].to_string(
        index=False, float_format=lambda v: f"{v:8.4f}"))
    for k, lab in (("l2_prim", "PRIMARY"), ("l2_fix", "FIXED 160/80")):
        m, lo, hi, t, kk = clus(d[k] - d["l2_ctrl"])
        print(f"  {lab:14} vs ctrl {m:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
              f"better {int(((d[k]-d['l2_ctrl'])<0).sum())}/{kk}")
    print(f"  P4 {'CONFIRMED' if (d.l2_prim-d.l2_ctrl).mean() < 0 else 'REFUTED'}")
    print(f"\n  P5 selection-bias estimate (FIXED minus PRIMARY at L2): "
          f"{(d.l2_fix - d.l2_prim).mean():+.4f}")

    print("\n  MAE by rank band (1-5 / 6-8 / 9+):")
    for k in ("ctrl", "prim"):
        print(f"    {k:5} " + "  ".join(f"{d[f'mae_{k}_{b}'].mean():6.3f}"
                                        for b in range(3)))
    print(f"\n  fallbacks: {dict(FALLBACKS)}")
    json.dump({"coverage": float(cov), "alphas": alphas,
               "rows": rows, "fallbacks": dict(FALLBACKS)},
              open(ROOT / "data" / "d245_role_share.json", "w"), default=float)
    print("\nwrote data/d245_role_share.json")


if __name__ == "__main__":
    main()
