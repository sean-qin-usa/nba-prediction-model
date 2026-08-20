#!/usr/bin/env python3
"""D79 HARMFUL-FIX REVERSAL REVIEW — props-ENDPOINT measurement.

D99/D100 (cg_kalman_clean.py) measured two of D79's own fixes as HARMFUL at the
one-step RATE level (WMAE of per-minute attempt rates), SIG 3/3 seasons:
  (a) the 002 filter on props.py rate history: ewma720_all - ewma720_002
      = -0.00056 (-0.36%) pooled — the pre-D79 unfiltered EWMA was BETTER;
  (b) the Kalman forward step ("fix to design"): kal720_002 - kal720f_002
      = -0.00023 (-0.15%) — the pre-D79 predict(0) no-op was BETTER.
Both were queued for reversal review. D99's own guidance: gate any reversal on
CRPS/PIT at the props ENDPOINT, because D79 attributed the pre-fix PIT centring
to bias cancellation against the (still-unfixed) minutes over-projection.

This script scores the two reversal candidates at the endpoint the register
uses for props verdicts (paired points-CRPS via simulate_player, cluster-
bootstrap by player; conditioning = validate_props: 002 targets, seconds>=720,
n_games>=8, proj_min>=20), 3 seasons, stride-sampled.

ARMS (same seed per row -> paired MC draws):
  ewma_002  player_rates_from_stats VERBATIM (production today, 002-filtered)
  ewma_all  same EWMA, 002 filter on the rate HISTORY removed (pre-D79) —
            the (a) REVERT candidate; eval targets stay 002-only
  kal_f     player_rates_kalman VERBATIM (forward step active; production-dead,
            only ablate_kalman_props.py calls it)
  kal_nof   same, forward step suppressed (pre-D79 no-op) — the (b) REVERT
            candidate

Metrics: points CRPS (primary), points PIT mean, |sim_mean - y| MAE.
Strata: pooled / per season / Oct+Nov vs rest / rows where the 002 filter
actually changes the rate profile ("affected") vs not.

Read-only DB. Writes data/d79_reversal_review.json. Touches nothing else.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from nbapred.db import connect
from nbapred.engine.props import (_if_none, player_rates_from_stats,
                                  player_rates_kalman, simulate_player)

SEASONS = ("2023-24", "2024-25", "2025-26")
ARMS = ("ewma_002", "ewma_all", "kal_f", "kal_nof")
SIMS = 4000
PER_SEASON = 2000
NBOOT = 2000
OUT = ROOT / "data" / "d79_reversal_review.json"


# ---------------------------------------------------------------- revert arms
def rates_ewma_all(con, player_id: int, before=None, half_life_games: float = 10.0):
    """player_rates_from_stats VERBATIM except the `game_id LIKE '002%'` filter
    on the rate HISTORY is removed (the pre-D79 universe: preseason/playoff/
    play-in rows re-admitted). Every other line identical to props.py."""
    date_clause = "AND g.game_date < ?" if before else ""
    params = [player_id] + ([before] if before else [])
    df = con.execute(f"""
        SELECT g.game_date, s.seconds, s.rima, s.rimm, s.mida, s.midm, s.thra, s.thrm,
               s.fta, s.ftm, s.oreb, s.dreb, s.ast
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
        WHERE s.player_id = ? {date_clause}
        ORDER BY g.game_date
    """, params).fetchdf()
    df = df[df["seconds"] >= 720]
    if len(df) < 3:
        return None
    age = np.arange(len(df))[::-1]
    w = 0.5 ** (age / half_life_games)
    mins = df["seconds"].to_numpy() / 60.0
    wm = w * mins

    def per_min(col):
        return float(np.sum(w * df[col].to_numpy()) / np.sum(wm))

    def fg(mk, at):
        num, den = np.sum(w * df[mk].to_numpy()), np.sum(w * df[at].to_numpy())
        return float(num / den) if den > 5 else None

    proj_min = float(np.sum(w * mins) / np.sum(w))
    sd_min = float(np.sqrt(np.average((mins - proj_min) ** 2, weights=w)))
    return {
        "proj_min": proj_min, "sd_min": max(sd_min, 2.0),
        "minutes_hist": mins[-20:],
        "rate_rim": per_min("rima"), "rate_mid": per_min("mida"), "rate_thr": per_min("thra"),
        "fg_rim": _if_none(fg("rimm", "rima"), 0.60),
        "fg_mid": _if_none(fg("midm", "mida"), 0.42),
        "fg_thr": _if_none(fg("thrm", "thra"), 0.35),
        "fta_per_min": per_min("fta"),
        "ft_pct": _if_none(fg("ftm", "fta"), 0.77),
        "reb_per_min": float(np.sum(w * (df["oreb"] + df["dreb"]).to_numpy()) / np.sum(wm)),
        "ast_per_min": per_min("ast"),
        "n_games": len(df),
    }


def rates_kalman_nofwd(con, player_id: int, before=None):
    """player_rates_kalman VERBATIM except fwd_days forced to 0 (the pre-D79
    no-op forward step). 002 filter kept (isolates the forward step)."""
    from nbapred.model.form_filter import FormFilter
    date_clause = "AND g.game_date < ?" if before else ""
    params = [player_id] + ([before] if before else [])
    df = con.execute(f"""
        SELECT g.game_date, s.seconds, s.rima, s.mida, s.thra, s.fta, s.oreb, s.dreb,
               s.ast, s.rimm, s.midm, s.thrm, s.ftm
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
        WHERE s.player_id = ? AND s.game_id LIKE '002%' {date_clause}
        ORDER BY g.game_date
    """, params).fetchdf()
    df = df[df["seconds"] > 0]
    if len(df) < 3:
        return None
    mins = df["seconds"].to_numpy() / 60.0
    dates = np.array([d.toordinal() for d in df["game_date"]])
    fwd_days = 0.0                       # <- the ONLY change vs props.py

    def kfilt(counts):
        rate = counts / mins
        f = FormFilter(float(np.average(rate, weights=mins)), prior_var=0.05,
                       Q=1e-4, meas_base=6.0)
        last = None
        for i in range(len(df)):
            f.predict(0.0 if last is None else dates[i] - last)
            f.update(rate[i], mins[i]); last = dates[i]
        f.predict(fwd_days)
        return max(f.theta, 0.0)

    def fg(mk, at):
        num, den = df[mk].sum(), df[at].sum()
        return float(num / den) if den > 5 else None

    proj_min = float(np.average(mins, weights=np.arange(1, len(mins) + 1)))
    return {
        "proj_min": proj_min, "sd_min": max(float(mins.std()), 2.0),
        "rate_rim": kfilt(df["rima"].to_numpy()), "rate_mid": kfilt(df["mida"].to_numpy()),
        "rate_thr": kfilt(df["thra"].to_numpy()),
        "fg_rim": _if_none(fg("rimm", "rima"), 0.60),
        "fg_mid": _if_none(fg("midm", "mida"), 0.42),
        "fg_thr": _if_none(fg("thrm", "thra"), 0.35),
        "fta_per_min": kfilt(df["fta"].to_numpy()),
        "ft_pct": _if_none(fg("ftm", "fta"), 0.77),
        "reb_per_min": kfilt((df["oreb"] + df["dreb"]).to_numpy()),
        "ast_per_min": kfilt(df["ast"].to_numpy()), "n_games": len(df),
    }


# ------------------------------------------------------------------- scoring
def crps(samples, y):
    s = np.sort(samples); n = len(s)
    return float(np.mean(np.abs(s - y)) - 0.5 * (2 * np.arange(1, n + 1) - n - 1) @ s / n ** 2)


def pit(samples, y, rng):
    lo = float(np.mean(samples < y))
    ties = float(np.mean(samples == y))
    return lo + rng.uniform() * ties


def cluster_boot(deltas, players, iters=NBOOT, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(players)
    per = {p: deltas[players == p] for p in uniq}
    means = []
    for _ in range(iters):
        pick = rng.choice(uniq, len(uniq), replace=True)
        means.append(np.concatenate([per[p] for p in pick]).mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(deltas)), float(lo), float(hi)


def main():
    t0 = time.time()
    con = connect(read_only=True)
    rows = []
    for season in SEASONS:
        pg = con.execute("""
            SELECT s.player_id, g.game_date, s.pts, s.seconds
            FROM player_game_stats s
            JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g
              USING (game_id)
            WHERE s.game_id LIKE '002%' AND s.seconds >= 720 AND g.season = ?
            ORDER BY g.game_date, s.player_id
        """, [season]).fetchdf()
        stride = max(1, len(pg) // (PER_SEASON * 2))   # over-sample; gate trims
        cand = pg.iloc[::stride]
        n_season = 0
        for r in cand.itertuples():
            if n_season >= PER_SEASON:
                break
            i = len(rows)
            base = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
            if base is None or base["n_games"] < 8 or base["proj_min"] < 20:
                continue
            alla = rates_ewma_all(con, int(r.player_id), before=r.game_date)
            kf = player_rates_kalman(con, int(r.player_id), before=r.game_date)
            knf = rates_kalman_nofwd(con, int(r.player_id), before=r.game_date)
            if alla is None or kf is None or knf is None:
                continue
            affected = any(abs(float(base[k]) - float(alla[k])) > 1e-12
                           for k in ("rate_rim", "rate_mid", "rate_thr",
                                     "fta_per_min", "reb_per_min", "ast_per_min",
                                     "proj_min"))
            rec = {"player_id": int(r.player_id), "season": season,
                   "month": r.game_date.month, "y": float(r.pts),
                   "affected": bool(affected)}
            prng = np.random.default_rng(10_000 + i)
            for arm, rr in (("ewma_002", base), ("ewma_all", alla),
                            ("kal_f", kf), ("kal_nof", knf)):
                sim = simulate_player(rr, SIMS, seed=i)["points"]
                rec[f"crps_{arm}"] = crps(sim, rec["y"])
                rec[f"pit_{arm}"] = pit(sim, rec["y"], prng)
                rec[f"mae_{arm}"] = abs(float(sim.mean()) - rec["y"])
            rows.append(rec)
            n_season += 1
            if n_season % 250 == 0:
                print(f"  {season} {n_season} rows ({time.time()-t0:.0f}s)", flush=True)
        print(f"{season}: scored {n_season}", flush=True)
    con.close()

    players = np.array([r["player_id"] for r in rows])
    seas = np.array([r["season"] for r in rows])
    month = np.array([r["month"] for r in rows])
    aff = np.array([r["affected"] for r in rows])

    out = {"n": len(rows), "sims": SIMS, "seasons": list(SEASONS),
           "affected_share": float(aff.mean()), "strata": {}}
    print(f"\nTOTAL rows {len(rows)}  players {len(set(players))}  "
          f"affected-by-002-filter {aff.mean():.1%}  ({time.time()-t0:.0f}s)")

    def get(metric, arm):
        return np.array([r[f"{metric}_{arm}"] for r in rows])

    COMPS = [("REVERT-A 002filter  ewma_002 - ewma_all  (+ = revert better)",
              "ewma_002", "ewma_all"),
             ("REVERT-B kalman fwd kal_f    - kal_nof   (+ = revert better)",
              "kal_f", "kal_nof")]

    def report(mask, label):
        blk = {"n": int(mask.sum()), "players": int(len(set(players[mask])))}
        print(f"\n--- {label} (n={mask.sum()}, {blk['players']} players) ---")
        for m in ("crps", "mae"):
            blk[m] = {}
            for name, a, b in COMPS:
                d = (get(m, a) - get(m, b))[mask]
                pt_, lo, hi = cluster_boot(d, players[mask])
                sig = "SIG" if (lo > 0 or hi < 0) else "ns"
                base_mean = float(get(m, a)[mask].mean())
                rel = 100 * pt_ / base_mean if base_mean else 0.0
                blk[m][f"{a}-{b}"] = dict(delta=pt_, lo=lo, hi=hi,
                                          rel_pct=rel, sig=sig)
                print(f"  {m:5s} {name}: {pt_:+.5f} ({rel:+.2f}%) "
                      f"CI[{lo:+.5f},{hi:+.5f}] {sig}")
        blk["pit_mean"] = {arm: float(get("pit", arm)[mask].mean()) for arm in ARMS}
        blk["crps_mean"] = {arm: float(get("crps", arm)[mask].mean()) for arm in ARMS}
        print("  PIT mean: " + "  ".join(f"{a} {blk['pit_mean'][a]:.4f}" for a in ARMS))
        out["strata"][label] = blk

    report(np.ones(len(rows), bool), "POOLED")
    for s in SEASONS:
        report(seas == s, s)
    report(np.isin(month, (10, 11)), "OCT+NOV")
    report(~np.isin(month, (10, 11)), "DEC-JUN")
    report(aff, "AFFECTED (002 filter changes rates)")
    if (~aff).sum() >= 100:
        report(~aff, "UNAFFECTED")

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}  ({time.time()-t0:.0f}s)")
    print("D79_REVERSAL_REVIEW_DONE", flush=True)


if __name__ == "__main__":
    main()
