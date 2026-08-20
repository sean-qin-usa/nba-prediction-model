#!/usr/bin/env python3
"""AUDIT: re-run the D45 x_minutes minutes-projector gate on the ALIGNED >=720s universe.

WHY THIS EXISTS
---------------
scripts/gate_xminutes.py conditions its eval universe (and therefore its
trailing baseline) on `s.seconds >= 600`, while EVERY props gate / training
path uses `>= 720` (nbapred/engine/props.py:215, nbapred/market/windows.py:67,
nbapred/model/composition.py:33, validate_props / ablate_* / audit_props_pit).
That is the known 600-vs-720 bug family. This script re-runs the same
comparison with the universe -- eval rows AND the trailing history the baseline
is built from -- aligned to >= 720 seconds.

WHAT IS COMPARED (minutes projector, one-step-ahead, MAE on actual minutes)
  trail_lin20  "shipped projector" per the audit brief: recency-weighted mean of
               the last 20 PLAYED games, weights = arange(1..n)  (the
               np.average(mins, weights=np.arange(1,len+1)) form in props.py)
  ewma_hl10    the LITERAL nbapred/engine/props.py:player_rates_from_stats
               formula: w = 0.5**(age/half_life_games=10) over ALL prior >=720s
               games, proj_min = sum(w*mins)/sum(w).  Included because the brief's
               "arange(1..n) over last 20" description and the actual shipped
               function disagree -- both are reported so the verdict does not
               depend on which one you call "shipped".
  trail_flat20 flat mean of last 20 played games (the ORIGINAL gate's strawman)
  xmin         DARKO x_minutes, as-of strictly before game_date (PIT)
  blend        0.5 * (trail + xmin), for each trail variant

WHICH BASELINE IS ACTUALLY "SHIPPED"?  All three are reported because they
disagree, and the answer matters for the verdict:
  - the brief says arange(1..n) over last 20        -> trail_lin20
  - props.py:player_rates_from_stats computes       -> ewma_hl10 (proj_min)
  - simulate_player() takes the `minutes_hist` branch whenever >=5 games exist
    (props.py:33-38: rng.choice(mins[-20:]) + N(0,2) jitter), so the LIVE
    central minutes estimate is E[draw] = mean(last 20 played) -> trail_flat20.
    proj_min only survives live as the assist-exposure scalar and the (dead)
    truncated-Normal fallback.  This matches journal 20260730_nba-model_bf3d11
    ("live minutes path is the LEDGER-REJECTED empirical-hist").

PIT DISCIPLINE
  - every input is strictly < game_date (trailing history uses date-based
    searchsorted, not row index, so same-date games cannot leak)
  - darko_history joined with allow_exact_matches=False (h.date < game_date)

STATS
  - metric MAE; paired per-observation |err| differences
  - delta = MAE(baseline) - MAE(challenger); POSITIVE = challenger better
  - 2000x bootstrap, 95% percentile CI, CLUSTERED BY PLAYER (samples are
    player-games).  iid CI printed alongside for the inflation factor.
  - reported per-season AND pooled.

READ-ONLY. Writes nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nbapred.db import connect  # noqa: E402

NBOOT = 2000
SEED = 11
SEASONS = ["2023-24", "2024-25", "2025-26"]
HL = 10.0          # props.py half_life_games
WINDOW = 20        # "last 20 played games"


# --------------------------------------------------------------------------- data
def fetch_games(con, sec_min: int) -> pd.DataFrame:
    return con.execute(f"""
        SELECT s.player_id, s.game_id, g.game_date, g.season, s.seconds/60.0 AS mins
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '002%' AND s.seconds >= {int(sec_min)}
        ORDER BY s.player_id, g.game_date, s.game_id
    """).fetchdf()


def fetch_darko(con) -> pd.DataFrame:
    dk = con.execute("""
        SELECT player_id, date, x_minutes
        FROM darko_history
        WHERE x_minutes IS NOT NULL
    """).fetchdf()
    dk["date"] = pd.to_datetime(dk["date"])
    return dk.sort_values("date").reset_index(drop=True)


def add_trailing(df: pd.DataFrame, within_season: bool) -> pd.DataFrame:
    """Trailing projectors computed ONLY from rows already in `df` (i.e. from the
    same >=Ns universe the eval runs on).  within_season=True restarts history at
    each season boundary (matches the original gate's season-scoped CTE);
    False uses the player's full cross-season history (matches production, which
    puts no season filter on player_rates_from_stats)."""
    keys = ["player_id", "season"] if within_season else ["player_id"]
    df = df.sort_values(keys + ["game_date", "game_id"]).reset_index(drop=True)
    n = len(df)
    flat = np.full(n, np.nan)
    lin = np.full(n, np.nan)
    ewma = np.full(n, np.nan)
    nprior = np.zeros(n, dtype=int)

    mins_all = df["mins"].to_numpy(float)
    dates_all = df["game_date"].to_numpy("datetime64[D]")
    lin_w = np.arange(1, WINDOW + 1, dtype=float)

    for idx in df.groupby(keys, sort=False).indices.values():
        idx = np.sort(idx)
        mins = mins_all[idx]
        dates = dates_all[idx]
        for j in range(len(idx)):
            k = int(np.searchsorted(dates, dates[j], side="left"))  # strict <
            row = idx[j]
            nprior[row] = k
            if k == 0:
                continue
            prior = mins[:k]
            w20 = prior[-WINDOW:]
            m = len(w20)
            flat[row] = w20.mean()
            lin[row] = float(np.dot(lin_w[WINDOW - m:], w20) / lin_w[WINDOW - m:].sum())
            w = 0.5 ** (np.arange(k)[::-1] / HL)
            ewma[row] = float(np.dot(w, prior) / w.sum())

    df["trail_flat20"] = flat
    df["trail_lin20"] = lin
    df["ewma_hl10"] = ewma
    df["n_prior"] = nprior
    return df


def attach_xmin(df: pd.DataFrame, dk: pd.DataFrame) -> pd.DataFrame:
    left = df.sort_values("game_date").reset_index(drop=True)
    left["game_date"] = pd.to_datetime(left["game_date"])
    out = pd.merge_asof(left, dk, left_on="game_date", right_on="date",
                        by="player_id", direction="backward",
                        allow_exact_matches=False)   # strictly BEFORE game_date
    out["xmin_stale_days"] = (out["game_date"] - out["date"]).dt.days
    return out.rename(columns={"x_minutes": "xmin"})


# --------------------------------------------------------------------------- stats
def cluster_boot(d: np.ndarray, clusters: np.ndarray, nboot=NBOOT, seed=SEED):
    """Percentile CI for mean(d), resampling CLUSTERS (players) with replacement."""
    rng = np.random.default_rng(seed)
    _, inv = np.unique(clusters, return_inverse=True)
    G = inv.max() + 1
    sums = np.bincount(inv, weights=d, minlength=G)
    cnts = np.bincount(inv, minlength=G).astype(float)
    pick = rng.integers(0, G, size=(nboot, G))
    bs = sums[pick].sum(1) / cnts[pick].sum(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def iid_boot(d: np.ndarray, nboot=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(d)
    bs = d[rng.integers(0, n, size=(nboot, n))].mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def gate(label: str, base: np.ndarray, chal: np.ndarray, y: np.ndarray,
         players: np.ndarray, indent="   "):
    e_b, e_c = np.abs(y - base), np.abs(y - chal)
    d = e_b - e_c                       # + = challenger better
    lo, hi = cluster_boot(d, players)
    ilo, ihi = iid_boot(d)
    infl = (hi - lo) / (ihi - ilo) if (ihi - ilo) > 0 else float("nan")
    verdict = "PASS" if lo > 0 else "FAIL"
    print(f"{indent}{label:<34s} dMAE {d.mean():+.4f} "
          f"clusterCI[{lo:+.4f},{hi:+.4f}] {verdict}   "
          f"(iidCI[{ilo:+.4f},{ihi:+.4f}] x{infl:.2f})")
    return dict(label=label, delta=float(d.mean()), lo=lo, hi=hi, verdict=verdict)


def seed_stability(label: str, base: np.ndarray, chal: np.ndarray, y: np.ndarray,
                   players: np.ndarray, seeds=range(1, 11)):
    """A hairline PASS must survive resampling noise. Re-run the cluster bootstrap
    under 10 seeds and report the worst-case lower bound."""
    d = np.abs(y - base) - np.abs(y - chal)
    los = [cluster_boot(d, players, seed=s)[0] for s in seeds]
    npass = sum(1 for lo in los if lo > 0)
    print(f"   [seed-stability] {label:<30s} delta {d.mean():+.4f} "
          f"lo in [{min(los):+.4f},{max(los):+.4f}]  PASS in {npass}/{len(los)} seeds")


def best_blend_w(trail, xmin, y):
    ws = np.arange(0.0, 1.001, 0.05)
    maes = [np.abs(y - (w * xmin + (1 - w) * trail)).mean() for w in ws]
    k = int(np.argmin(maes))
    return float(ws[k]), float(maes[k])


# --------------------------------------------------------------------------- run
def describe(tag: str, df: pd.DataFrame):
    y = df["mins"].to_numpy(float)
    print(f"\n{tag}")
    print(f"   n={len(df):,}  players={df.player_id.nunique():,}  "
          f"mean actual minutes {y.mean():.2f}  "
          f"median darko staleness {df.xmin_stale_days.median():.0f}d")
    for c in ("trail_lin20", "ewma_hl10", "trail_flat20", "xmin"):
        e = np.abs(y - df[c].to_numpy(float))
        bias = (y - df[c].to_numpy(float)).mean()
        print(f"   MAE {c:<13s} {e.mean():.4f}   bias(actual-proj) {bias:+.3f}")
    for tr in ("trail_lin20", "ewma_hl10"):
        b = 0.5 * (df[tr].to_numpy(float) + df["xmin"].to_numpy(float))
        print(f"   MAE blend50({tr[:11]:<11s}) {np.abs(y - b).mean():.4f}")


def run_block(df: pd.DataFrame, tag: str):
    describe(tag, df)
    y = df["mins"].to_numpy(float)
    pl = df["player_id"].to_numpy()
    tl = df["trail_lin20"].to_numpy(float)
    tw = df["ewma_hl10"].to_numpy(float)
    tf = df["trail_flat20"].to_numpy(float)
    xm = df["xmin"].to_numpy(float)
    res = []
    print("   -- baseline = trail_lin20 (brief's shipped projector) --")
    res.append(gate("xmin", tl, xm, y, pl))
    res.append(gate("blend50(lin20,xmin)", tl, 0.5 * (tl + xm), y, pl))
    print("   -- baseline = ewma_hl10 (literal props.py formula) --")
    res.append(gate("xmin", tw, xm, y, pl))
    res.append(gate("blend50(ewma,xmin)", tw, 0.5 * (tw + xm), y, pl))
    print("   -- reference: original gate's flat strawman baseline --")
    res.append(gate("xmin vs flat20", tf, xm, y, pl))
    res.append(gate("blend50(flat20,xmin) vs flat20", tf, 0.5 * (tf + xm), y, pl))
    res.append(gate("trail_lin20 vs flat20", tf, tl, y, pl))
    w, mae = best_blend_w(tl, xm, y)
    print(f"   [diag] MAE-optimal blend weight on xmin (in-sample, NOT a gate): "
          f"w*={w:.2f} MAE {mae:.4f} vs lin20 {np.abs(y - tl).mean():.4f}")
    return res


def main() -> None:
    con = connect(read_only=True)
    dk = fetch_darko(con)
    raw720 = fetch_games(con, 720)
    raw600 = fetch_games(con, 600)
    con.close()

    print("=" * 96)
    print("AUDIT audit_xminutes_720 -- D45 x_minutes gate on the ALIGNED >=720s universe")
    print(f"NBOOT={NBOOT} seed={SEED}  cluster=player  n_prior>=5  regular season only")
    print("=" * 96)

    # ---- 0) replicate the ORIGINAL (buggy) gate: >=600 universe, flat-20 baseline
    o = attach_xmin(add_trailing(raw600[raw600.season == "2025-26"].copy(), True), dk)
    o = o.dropna(subset=["trail_flat20", "xmin"])
    o = o[o.n_prior >= 5]
    yo = o["mins"].to_numpy(float)
    print("\n[0] REPLICATION of scripts/gate_xminutes.py (>=600s universe, flat trail-20)")
    print(f"   n={len(o):,}  MAE flat20 {np.abs(yo - o.trail_flat20).mean():.3f} | "
          f"xmin {np.abs(yo - o.xmin).mean():.3f} | "
          f"blend {np.abs(yo - 0.5*(o.trail_flat20 + o.xmin)).mean():.3f}")
    gate("xmin (orig universe)", o.trail_flat20.to_numpy(float), o.xmin.to_numpy(float),
         yo, o.player_id.to_numpy())
    gate("blend (orig universe)", o.trail_flat20.to_numpy(float),
         0.5 * (o.trail_flat20.to_numpy(float) + o.xmin.to_numpy(float)),
         yo, o.player_id.to_numpy())

    # ---- 0b) the registered D45 number: >=600 universe, RECENCY-WEIGHTED baseline
    print("\n[0b] REPLICATION of the registered D45 verdict "
          "(>=600s universe, recency-weighted baseline; DECISIONS.md: -0.049 CI -0.077..-0.023)")
    o2 = o.dropna(subset=["trail_lin20", "ewma_hl10"])
    y2 = o2["mins"].to_numpy(float)
    tl2 = o2.trail_lin20.to_numpy(float)
    tw2 = o2.ewma_hl10.to_numpy(float)
    xm2 = o2.xmin.to_numpy(float)
    print(f"   n={len(o2):,}  MAE lin20 {np.abs(y2-tl2).mean():.4f} | "
          f"ewma_hl10 {np.abs(y2-tw2).mean():.4f} | xmin {np.abs(y2-xm2).mean():.4f}")
    gate("blend vs lin20 (600s)", tl2, 0.5 * (tl2 + xm2), y2, o2.player_id.to_numpy())
    gate("blend vs ewma_hl10 (600s)", tw2, 0.5 * (tw2 + xm2), y2, o2.player_id.to_numpy())

    # ---- 1) ALIGNED >=720 universe, within-season history (original gate's scoping)
    ws = attach_xmin(add_trailing(raw720.copy(), True), dk)
    ws = ws.dropna(subset=["trail_lin20", "ewma_hl10", "xmin"])
    ws = ws[ws.n_prior >= 5]

    # ---- 2) ALIGNED >=720 universe, full cross-season history (production scoping)
    xs = attach_xmin(add_trailing(raw720.copy(), False), dk)
    xs = xs.dropna(subset=["trail_lin20", "ewma_hl10", "xmin"])
    xs = xs[xs.n_prior >= 5]

    print("\n" + "=" * 96)
    print("[1] ALIGNED >=720s UNIVERSE -- within-season trailing history "
          "(same scoping as the original gate)")
    print("=" * 96)
    headline = None
    for s in SEASONS:
        r = run_block(ws[ws.season == s], f"--- season {s} ---")
        if s == "2025-26":
            headline = r
    run_block(ws, "--- POOLED 2023-24..2025-26 ---")

    print("\n" + "=" * 96)
    print("[2] ALIGNED >=720s UNIVERSE -- full cross-season trailing history "
          "(production scoping, robustness)")
    print("=" * 96)
    for s in SEASONS:
        run_block(xs[xs.season == s], f"--- season {s} ---")
    run_block(xs, "--- POOLED 2023-24..2025-26 ---")

    # ---- 2b) MATCHED universe: cross-season projectors evaluated on the SAME rows
    #          as block [1], so history-scoping is isolated from universe expansion.
    key = ["player_id", "game_id"]
    xs_m = xs.merge(ws[key], on=key, how="inner")
    print("\n" + "=" * 96)
    print("[2b] MATCHED universe (block-[1] rows only), cross-season trailing history "
          "-- isolates history scoping from universe expansion")
    print("=" * 96)
    for s in SEASONS:
        run_block(xs_m[xs_m.season == s], f"--- season {s} ---")
    run_block(xs_m, "--- POOLED 2023-24..2025-26 ---")

    # ---- 2c) seed stability for every configuration that produced a PASS
    print("\n" + "=" * 96)
    print("[2c] SEED STABILITY of the borderline results (10 cluster-bootstrap seeds)")
    print("=" * 96)
    for tag, d in (("[1] within-season 25-26", ws[ws.season == "2025-26"]),
                   ("[2] cross-season 25-26", xs[xs.season == "2025-26"]),
                   ("[2] cross-season pooled", xs),
                   ("[2b] matched 25-26", xs_m[xs_m.season == "2025-26"])):
        yy = d["mins"].to_numpy(float); pp = d["player_id"].to_numpy()
        tl = d["trail_lin20"].to_numpy(float); tw = d["ewma_hl10"].to_numpy(float)
        tf = d["trail_flat20"].to_numpy(float); xm = d["xmin"].to_numpy(float)
        print(f"  {tag}  (n={len(d):,})")
        seed_stability("blend vs lin20", tl, 0.5 * (tl + xm), yy, pp)
        seed_stability("blend vs ewma_hl10", tw, 0.5 * (tw + xm), yy, pp)
        seed_stability("blend vs flat20 (LIVE center)", tf, 0.5 * (tf + xm), yy, pp)
        seed_stability("lin20 vs flat20 (free fix)", tf, tl, yy, pp)

    # ---- 3) mechanism: does the >=600 -> >=720 shift move x_minutes' bias?
    print("\n" + "=" * 96)
    print("[3] MECHANISM -- conditioning bias of x_minutes vs trailing")
    print("=" * 96)
    a = attach_xmin(add_trailing(raw600[raw600.season == "2025-26"].copy(), True), dk)
    a = a.dropna(subset=["trail_lin20", "xmin"]); a = a[a.n_prior >= 5]
    b = ws[ws.season == "2025-26"]
    for nm, d in (("600s universe", a), ("720s universe", b)):
        yy = d["mins"].to_numpy(float)
        print(f"   {nm}: n={len(d):,} mean_min {yy.mean():.2f} | "
              f"bias xmin {(yy - d.xmin).mean():+.3f} | "
              f"bias lin20 {(yy - d.trail_lin20).mean():+.3f} | "
              f"sd(xmin) {d.xmin.std():.2f} sd(lin20) {d.trail_lin20.std():.2f}")

    print("\nHEADLINE (2025-26, aligned >=720s, within-season, baseline trail_lin20):")
    for r in (headline or [])[:2]:
        print(f"   {r['label']:<26s} {r['delta']:+.4f} "
              f"CI[{r['lo']:+.4f},{r['hi']:+.4f}] {r['verdict']}")
    print("AUDIT_XMINUTES_720_DONE")


if __name__ == "__main__":
    main()
