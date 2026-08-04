#!/usr/bin/env python3
"""IG probe (read-only): paired variant battery for simulate_player tuning knobs.

Variants (same seed per row, paired):
  A live       — as shipped (empirical hist + N(0,2) jitter)
  B nojit      — hist path with jitter sd 0 (isolate the +N(0,2))
  C trunc10    — hist stripped, truncated-normal at 10 (ledger-ACCEPTED path)
  D trunc12    — same but truncated at 12 (matches the >=720s conditioning) [D54]
  E astmix     — live points; assists exposure = INDEPENDENT minutes draw
                 (recenters D30: decoupled from points but keeps minutes-mixing
                 overdispersion that the fixed-scalar exposure kills)

Also: minutes diagnostics (realized error sd vs sim sd; sd_min floor binding),
PIT/CRPS/z2/80%-coverage for points; PIT/z2 for assists (live vs astmix).
"""
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from nbapred.db import connect
from nbapred.engine.props import player_rates_from_stats, simulate_player


def sim_variant(rates, n, seed, jitter=2.0, trunc=None, ast_expo_mode="fixed"):
    """Reimplementation of simulate_player with knobs. trunc=None -> hist path."""
    rng = np.random.default_rng(seed)
    hist = rates.get("minutes_hist")
    if trunc is None and hist is not None and len(hist) >= 5:
        h = np.asarray(hist, float)
        base = rng.choice(h, n) - h.mean() + float(rates.get("proj_min", h.mean()))
        if jitter > 0:
            mins = np.clip(base + rng.normal(0, jitter, n), 0, 48)
        else:
            mins = np.clip(base + rng.normal(0, 2.0, n) * 0.0, 0, 48)  # keep rng stream aligned
    else:
        lo = trunc if trunc is not None else 10
        mins = rng.normal(rates["proj_min"], rates.get("sd_min", 4.0), n)
        bad = mins < lo
        for _ in range(4):
            if not bad.any():
                break
            mins[bad] = rng.normal(rates["proj_min"], rates.get("sd_min", 4.0), bad.sum())
            bad = mins < lo
        mins = np.clip(mins, lo, 48)

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

    if ast_expo_mode == "fixed":
        ast_expo = float(np.clip(rates.get("proj_min", 30.0), 10, 44))
        assists = rng.poisson(np.maximum(rates.get("ast_per_min", 0), 0) * ast_expo, size=n)
    else:  # independent minutes draw for assist exposure
        rng2 = np.random.default_rng(seed + 777)
        if hist is not None and len(hist) >= 5:
            h = np.asarray(hist, float)
            expo = np.clip(rng2.choice(h, n) - h.mean()
                           + float(rates.get("proj_min", h.mean()))
                           + rng2.normal(0, 2.0, n), 0, 48)
        else:
            expo = np.clip(rng2.normal(rates["proj_min"], rates.get("sd_min", 4.0), n), 10, 48)
        assists = rng.poisson(np.maximum(rates.get("ast_per_min", 0), 0) * expo)
    return {"points": points, "assists": assists, "mins": mins}


def crps(s, y):
    s = np.sort(s); n = len(s)
    return float(np.mean(np.abs(s - y)) - 0.5 * (2 * np.arange(1, n + 1) - n - 1) @ s / n**2)


def summarize(tag, pit, crp, z2, cov, n):
    p = np.array(pit)
    print(f"  {tag:10s} PIT mean {p.mean():.3f} std {p.std():.3f}  "
          f"CRPS {np.mean(crp):.4f}  z2 {np.mean(z2):.3f}  cov80 {cov/n:.3f}")


def main(sims=2000, max_eval=700):
    rng_u = np.random.default_rng(11)
    con = connect(read_only=True)
    pg = con.execute("""
        SELECT s.player_id, g.game_date, s.pts, s.ast, s.seconds
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '00225%' AND s.seconds >= 720
        ORDER BY g.game_date, s.player_id""").fetchdf()
    stride = max(1, len(pg) // (max_eval * 2))
    rows = pg.iloc[::stride]

    keys = ("live", "nojit", "trunc10", "trunc12")
    res = {k: {"pit": [], "crps": [], "z2": [], "cov": 0} for k in keys}
    ast_res = {k: {"pit": [], "z2": []} for k in ("ast_fixed", "ast_mix")}
    min_err, sim_min_sd, floor_bind = [], [], 0
    verified = False
    n = 0
    for r in rows.itertuples():
        if n >= max_eval:
            break
        rates = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
        if rates is None or rates["n_games"] < 8 or rates["proj_min"] < 20:
            continue
        y, ya, amin = float(r.pts), float(r.ast), r.seconds / 60.0

        if not verified:  # reimpl fidelity check vs shipped code, same seed
            a = simulate_player(rates, n=500, seed=123)["points"]
            b = sim_variant(rates, 500, 123)["points"]
            assert np.array_equal(a, b), "reimplementation diverges from shipped sim"
            rt = dict(rates); rt.pop("minutes_hist")
            a = simulate_player(rt, n=500, seed=123)["points"]
            b = sim_variant(rt, 500, 123, trunc=10)["points"]
            assert np.array_equal(a, b), "trunc reimpl diverges"
            verified = True

        rt = dict(rates); rt.pop("minutes_hist", None)
        u = rng_u.random()
        sims_out = {
            "live": sim_variant(rates, sims, seed=n),
            "nojit": sim_variant(rates, sims, seed=n, jitter=0.0),
            "trunc10": sim_variant(rt, sims, seed=n, trunc=10),
            "trunc12": sim_variant(rt, sims, seed=n, trunc=12),
        }
        for k in keys:
            pts = sims_out[k]["points"]
            res[k]["pit"].append(np.mean(pts < y) + u * np.mean(pts == y))
            res[k]["crps"].append(crps(pts, y))
            res[k]["z2"].append(((y - pts.mean()) / max(pts.std(), 1e-6)) ** 2)
            lo, hi = np.percentile(pts, [10, 90])
            res[k]["cov"] += int(lo <= y <= hi)
        amix = sim_variant(rates, sims, seed=n, ast_expo_mode="mix")["assists"]
        for k, arr in (("ast_fixed", sims_out["live"]["assists"]), ("ast_mix", amix)):
            ast_res[k]["pit"].append(np.mean(arr < ya) + u * np.mean(arr == ya))
            ast_res[k]["z2"].append(((ya - arr.mean()) / max(arr.std(), 1e-6)) ** 2)

        min_err.append(amin - rates["proj_min"])
        sim_min_sd.append(float(sims_out["live"]["mins"].std()))
        floor_bind += int(rates["sd_min"] == 2.0)
        n += 1
    con.close()

    print(f"evaluated rows: {n} (2025-26, >=720s, >=8 prior games, proj_min>=20)")
    print("\nPOINTS (paired across variants):")
    for k in keys:
        summarize(k, res[k]["pit"], res[k]["crps"], res[k]["z2"], res[k]["cov"], n)
    # paired CRPS deltas with bootstrap CI
    rng = np.random.default_rng(0)
    for a, b in (("live", "nojit"), ("trunc10", "trunc12"), ("live", "trunc10")):
        d = np.array(res[a]["crps"]) - np.array(res[b]["crps"])
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  CRPS {a}-{b}: {d.mean():+.4f}  CI ({lo:+.4f},{hi:+.4f})")

    print("\nASSISTS:")
    for k in ast_res:
        p = np.array(ast_res[k]["pit"])
        print(f"  {k:10s} PIT mean {p.mean():.3f} std {p.std():.3f} "
              f"(uniform .289)  z2 {np.mean(ast_res[k]['z2']):.3f}")

    me = np.array(min_err)
    print(f"\nMINUTES: realized err (actual-proj): mean {me.mean():+.2f} sd {me.std():.2f}")
    print(f"  sim minutes sd (live path, incl jitter): mean {np.mean(sim_min_sd):.2f}")
    print(f"  sd_min floor 2.0 binds on {floor_bind}/{n} rows "
          f"(floor only matters on the dead trunc path)")


if __name__ == "__main__":
    main()
