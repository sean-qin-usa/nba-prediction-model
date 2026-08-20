#!/usr/bin/env python3
"""Gate: assist exposure partial-coupling (THREAD-5b, external review 6 props item).

Shipped construction (props.py:83-84, D30): assists use a SCALAR exposure
clip(proj_min,10,44) — fully decoupled from the simulated minutes draw, because
sharing the minutes draw made within-player corr(pts,ast) +0.26 vs real +0.04
(6x over-coupled -> overpriced pts+ast parlays). Cost: the assists marginal has
NO minutes-mixing variance — audit_props_pit measured it ~19% under-dispersed
(z2=1.19).

Candidates:
  - PARTIAL coupling — expo_i = (1-c)*clip(proj_min,10,44) + c*mins_i with
    c=0.5 (thread-named arm). c=0 is the shipped control (bitwise identical:
    the local sim copies simulate_player's exact rng draw order, and assists
    are the LAST draw, so all arms share identical minutes/points draws at the
    same seed). c=0.25 and c=1.0 run as DIAGNOSTIC CURVE only (not gated).
  - IND — assists exposure from an INDEPENDENT minutes draw (same generative
    hist+jitter procedure, separate rng): full minutes-mixing overdispersion
    with ZERO pts-coupling. This is ig_props_variants.py variant E ("astmix"),
    queued in-repo 2026-07-30 and never run — included as a co-primary because
    the assists MARGINAL (what CRPS scores) cannot distinguish coupling from
    independent mixing, while the D30 concern is purely the coupling side.

Gate (pre-registered): 2025-26 assists CRPS, paired bootstrap 2000x, 95% CI of
(control - variant) must exclude 0 in favor of the variant, for each of the two
primary arms {c=0.5, IND}. Ship choice if both pass: prefer IND on first
principles (sim corr(pts,ast) ~ 0 vs real +0.04; c=0.5 coupling ~3x real).
Guard: any shipped arm's corr(points,assists) must stay well below the D30
failure level +0.26; report it for every arm.

Universe: same conditioning as audit_props_pit — 2025-26 regular season,
seconds>=720, >=8 prior games, proj_min>=20, strided across the season.
Read-only DB.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect
from nbapred.engine.props import player_rates_from_stats, simulate_player


def crps(samples, y):
    s = np.sort(samples); n = len(s)
    return float(np.mean(np.abs(s - y)) - 0.5 * (2 * np.arange(1, n + 1) - n - 1) @ s / n**2)


def draw_minutes(rng, rates: dict, n: int):
    """Exact copy of simulate_player's minutes draw (consumes the same rng
    sequence)."""
    hist = rates.get("minutes_hist")
    if hist is not None and len(hist) >= 5:
        h = np.asarray(hist, float)
        base = rng.choice(h, n) - h.mean() + float(rates.get("proj_min", h.mean()))
        return np.clip(base + rng.normal(0, 2.0, n), 0, 48)
    mins = rng.normal(rates["proj_min"], rates.get("sd_min", 4.0), n)
    bad = mins < 10
    for _ in range(4):
        if not bad.any():
            break
        mins[bad] = rng.normal(rates["proj_min"], rates.get("sd_min", 4.0), bad.sum())
        bad = mins < 10
    return np.clip(mins, 10, 48)


def simulate_player_astc(rates: dict, n: int, seed: int, ast_c) -> dict:
    """Copy of props.simulate_player with assist exposure knob ast_c.
    ast_c=0.0 reproduces the shipped scalar-exposure path bitwise (same rng
    stream; assists are the final draw). ast_c='ind' -> exposure from an
    independent minutes draw (separate rng; shared draws untouched)."""
    rng = np.random.default_rng(seed)
    mins = draw_minutes(rng, rates, n)

    def zone_pts(rate, fg, val):
        att = rng.poisson(np.maximum(rate, 0) * mins)
        made = rng.binomial(att, np.clip(fg, 0, 1))
        return made, made * val

    rim_m, rim_p = zone_pts(rates["rate_rim"], rates["fg_rim"], 2)
    mid_m, mid_p = zone_pts(rates["rate_mid"], rates["fg_mid"], 2)
    thr_m, thr_p = zone_pts(rates["rate_thr"], rates["fg_thr"], 3)
    ft_att = rng.poisson(np.maximum(rates.get("fta_per_min", 0), 0) * mins)
    ft_m = rng.binomial(ft_att, np.clip(rates.get("ft_pct", 0.77), 0, 1))
    points = rim_p + mid_p + thr_p + ft_m
    rebounds = rng.poisson(np.maximum(rates.get("reb_per_min", 0), 0) * mins)

    scalar_expo = float(np.clip(rates.get("proj_min", 30.0), 10, 44))
    if ast_c == "ind":
        rng_ind = np.random.default_rng(seed + 10_000_019)   # independent stream
        expo = draw_minutes(rng_ind, rates, n)
    else:
        expo = (1.0 - ast_c) * scalar_expo + ast_c * mins    # ast_c=0 -> scalar
    lam = np.maximum(rates.get("ast_per_min", 0), 0) * expo
    if np.isscalar(lam) or np.ndim(lam) == 0:
        assists = rng.poisson(lam, size=n)
    else:
        assists = rng.poisson(lam)
    return {"points": points, "rebounds": rebounds, "assists": assists}


# 0.0 control; 0.5 and "ind" primary gate arms; 0.25/1.0 diagnostic curve
ARMS = (0.0, 0.25, 0.5, 1.0, "ind")


def main(min_prior_games=8, min_proj_min=20, sims=2000, max_eval=1500):
    con = connect(read_only=True)
    pg = con.execute("""
        SELECT s.player_id, g.game_date, s.pts, s.ast, s.seconds
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '00225%' AND s.seconds >= 720
        ORDER BY g.game_date, s.player_id
    """).fetchdf()
    print(f"2025-26 candidate player-games: {len(pg)}")
    stride = max(1, len(pg) // (max_eval * 2))
    rows = pg.iloc[::stride]

    crps_a = {c: [] for c in ARMS}
    pit_a = {c: [] for c in ARMS}
    z2_a = {c: [] for c in ARMS}
    corr_a = {c: [] for c in ARMS}
    parity_max = 0.0
    n = 0
    for r in rows.itertuples():
        if n >= max_eval:
            break
        rates = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
        if rates is None or rates["n_games"] < min_prior_games or rates["proj_min"] < min_proj_min:
            continue
        ya = float(r.ast)
        # parity check on first rows: local c=0 must equal shipped simulate_player
        if n < 25:
            ship = simulate_player(rates, n=sims, seed=n)["assists"]
            loc = simulate_player_astc(rates, n=sims, seed=n, ast_c=0.0)["assists"]
            parity_max = max(parity_max, float(np.abs(ship - loc).max()))
        for c in ARMS:
            sim = simulate_player_astc(rates, n=sims, seed=n, ast_c=c)
            ast = sim["assists"]
            crps_a[c].append(crps(ast, ya))
            pit_a[c].append(float(np.mean(ast < ya) + 0.5 * np.mean(ast == ya)))
            z2_a[c].append(((ya - ast.mean()) / max(ast.std(), 1e-6)) ** 2)
            corr_a[c].append(float(np.corrcoef(sim["points"], ast)[0, 1]))
        n += 1
    con.close()

    print(f"evaluated player-games: {n}")
    print(f"parity (shipped vs local c=0, first 25 rows): max|diff| = {parity_max}")
    print(f"\n{'arm':>6s} {'astCRPS':>8s} {'PITmean':>8s} {'PITstd':>7s} "
          f"{'z2':>6s} {'corr(pts,ast)':>14s}")
    for c in ARMS:
        print(f"  c={str(c):<4} {np.mean(crps_a[c]):8.4f} {np.mean(pit_a[c]):8.3f} "
              f"{np.std(pit_a[c]):7.3f} {np.mean(z2_a[c]):6.3f} "
              f"{np.mean(corr_a[c]):+14.3f}")

    rng = np.random.default_rng(0)
    print("\npaired bootstrap 2000x (control - variant; >0 = variant better):")
    for c in ARMS[1:]:
        d = np.array(crps_a[0.0]) - np.array(crps_a[c])
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        tag = " <-- PRIMARY GATE" if c in (0.5, "ind") else " (diagnostic)"
        verdict = "PASS" if lo > 0 else ("HARM" if hi < 0 else "NS")
        print(f"  c={c}: dCRPS {d.mean():+.4f}  CI ({lo:+.4f},{hi:+.4f})  {verdict}{tag}")


if __name__ == "__main__":
    main()
