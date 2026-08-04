#!/usr/bin/env python3
"""VALIDATION (read-only) for the D82 live star-out adjustment (starout.py).

Universe: 2025-26 star-out player-games — a >=28-trailing-min star (rolling-10,
min 5, shifted) missing from the realized rotation while seen within 12 days;
eval rows are remaining rotation players (trailing >=15 min) who played >=12
min, with a rates profile (n_games>=8, proj_min>=15). Gate-parity with
scripts/gate_redistribution_crps.py except: 2025-26 only, and minutes_hist is
KEPT (live-path parity — the shipped sim recenters empirical draws on proj_min
per D57).

Arms per row (same seed -> paired MC draws):
  base : trailing rates untouched
  prod : the SHIPPED adjustment — residual-calibrated softmax attempts lift
         (1 + 0.16*(L-1)) + residual positional minutes tilt (0.39 * D39),
         exactly what nbapred/engine/starout.py applies before simulate_player
  full : the D33/D39 magnitudes as-written (softmax lift capped [1,1.6], full
         +2.91/+2.09/+2.4 tilt) — reported honestly for the record

Metrics: prop CRPS points/rebounds/assists + attempts-side shots Poisson LL
(the D33 claim). Paired per-row deltas, PLAYER-CLUSTERED bootstrap (2000x,
95% CI), delta > 0 = adjustment better. The residual scales were estimated on
the pre-q0.6 rows of this universe (moment/MLE, scratchpad diag), so the
POST-CUT section is the honest held-out read for `prod`.

DECLARED CAVEATS (same class as every prior redistribution gate):
  * OUT-set is ORACLE (inferred from the realized box score), identical in all
    arms — contrasts are fair, absolute CRPS optimistic.
  * D35 contamination: trailing rates already partially embed ongoing absences.
    The live baseline (>=12-min EWMA) absorbs ~2/3 of the redistribution —
    THAT is why full-strength D33/D39 magnitudes over-shoot here (points CRPS
    worse, the D34 result) while the gated attempts science replicates on its
    own weaker avg_shots baseline (+0.060 on these same rows). Fresh-news
    October cases are under-represented; residual scales are conservative
    there (module docstring).
  * v2_usage.npz u is fit on pooled-season pbp (in-sample for 2025-26); D57's
    PIT refit showed the softmax effect survives decontamination (+0.0437).
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nbapred.db import connect
from nbapred.engine import starout
from nbapred.engine.props import player_rates_from_stats, simulate_player

SIMS = 2000
MARKETS = ("points", "rebounds", "assists")
ARMS = ("prod", "full")
TRAIN_Q = 0.6   # residual scales were fit on rows <= this date quantile


def crps(s, y):
    s = np.sort(s); n = len(s)
    return float(np.mean(np.abs(s - y)) - 0.5 * (2 * np.arange(1, n + 1) - n - 1) @ s / n ** 2)


def cluster_boot(deltas, players, iters=2000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(players)
    per = {p: deltas[players == p] for p in uniq}
    means = []
    for _ in range(iters):
        pick = rng.choice(uniq, len(uniq), replace=True)
        means.append(np.concatenate([per[p] for p in pick]).mean())
    return np.percentile(means, [2.5, 97.5])


def main(max_eval=None):
    con = connect(read_only=True)
    pg = con.execute("""
        SELECT s.game_id, s.player_id, s.team_id, g.game_date, s.seconds/60.0 mins,
               s.pts, s.oreb + s.dreb AS reb, s.ast,
               s.rima + s.mida + s.thra AS shots
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date, season FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '002%' AND g.season = '2025-26'
        ORDER BY g.game_date""").fetchdf()
    weights = starout.load_usage_weights()
    src = "v2_usage.npz" if weights else "trailing-attempts (npz missing)"
    positions = starout.load_positions()
    pg = pg.sort_values(["player_id", "game_date"])
    pg["avg_min"] = pg.groupby("player_id")["mins"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=5).mean())
    played = pg[pg.mins >= 8].groupby(["game_id", "team_id"])["player_id"].apply(set)
    stars = pg[pg.avg_min >= starout.STAR_TRAILING_MIN]
    sbt = {}
    for r in stars[["player_id", "team_id", "game_date"]].itertuples():
        sbt.setdefault(r.team_id, []).append((r.game_date, r.player_id))
    rot = pg[(pg.avg_min >= starout.ROT_TRAILING_MIN) & (pg.mins >= 12)].copy()
    # per-player chronology of >=12-min games (for the D35 contamination share:
    # fraction of the eval player's trailing-10 games the star ALSO missed)
    hist12 = {p: list(zip(g.game_date, g.game_id))
              for p, g in pg[pg.mins >= 12].groupby("player_id")}

    res = {m: {a: [] for a in ("base",) + ARMS} for m in MARKETS}
    llrows = []                     # (ll_base, ll_prod, ll_full) shots Poisson LL
    contam, dates = [], []
    who, lifts, tilts, tiltkind = [], [], [], {"same": 0, "diff": 0, "flat": 0}
    n = 0
    for r in rot.itertuples():
        if max_eval and n >= max_eval:
            break
        recent = {p for (d0, p) in sbt.get(r.team_id, []) if 0 < (r.game_date - d0).days <= starout.FRESH_DAYS}
        outs = (recent - played.get((r.game_id, r.team_id), set())) - {r.player_id}
        if not outs:
            continue
        w = weights if weights is not None else {}
        star = max(outs, key=lambda p: w.get(p, 1.0))
        rates = player_rates_from_stats(con, int(r.player_id), before=r.game_date)
        if not rates or rates["n_games"] < 8 or rates["proj_min"] < 15:
            continue
        pool = {int(p) for p in rot[(rot.team_id == r.team_id) &
                                    (rot.game_date == r.game_date)].player_id}
        lift_full = starout.compute_lift(w, pool, star, default=1.0)
        lift_prod = 1.0 + starout.RESID_ATT_SCALE * (lift_full - 1.0)
        p_pos, s_pos = positions.get(int(r.player_id)), positions.get(int(star))
        tilt_full = starout.minutes_tilt(p_pos, s_pos)
        tilt_prod = starout.production_tilt(p_pos, s_pos)
        tiltkind["flat" if not (p_pos and s_pos) else
                 "same" if tilt_full == starout.TILT_SAME_POS else "diff"] += 1
        lifts.append(lift_prod); tilts.append(tilt_prod)
        arms = {"base": rates,
                "prod": starout.adjust_rates(rates, lift_prod, tilt_prod),
                "full": starout.adjust_rates(rates, lift_full, tilt_full)}
        y = {"points": r.pts, "rebounds": r.reb, "assists": r.ast}
        for a, rr in arms.items():
            sim = simulate_player(rr, SIMS, seed=n)
            for m in MARKETS:
                res[m][a].append(crps(sim[m], y[m]))
        # attempts-side Poisson LL (the D33 claim), mu = sum(rates)*proj_min
        def _ll(rr):
            mu = max((rr["rate_rim"] + rr["rate_mid"] + rr["rate_thr"]) * rr["proj_min"], 0.2)
            return r.shots * np.log(mu) - mu
        llrows.append(tuple(_ll(arms[a]) for a in ("base",) + ARMS))
        # D35 contamination: share of this player's trailing-10 (>=12min) games
        # in which the star was ALSO absent (trailing rates embed redistribution)
        prior = [gid for (d, gid) in hist12.get(r.player_id, [])
                 if d < r.game_date][-10:]
        contam.append(np.mean([star not in played.get((gid, r.team_id), set())
                               for gid in prior]) if prior else 1.0)
        dates.append(r.game_date)
        who.append(int(r.player_id)); n += 1
        if n % 500 == 0:
            print(f"  ...{n} rows", flush=True)
    con.close()

    players = np.array(who)
    contam = np.array(contam)
    dates = np.array(dates)
    heldout = dates > np.quantile(dates, TRAIN_Q)
    ll = np.array(llrows)
    print(f"\nstar-out validation universe: n={n} player-games, "
          f"{len(np.unique(players))} players, season 2025-26")
    print(f"usage source: {src}; mean PROD lift {np.mean(lifts):.3f}, "
          f"mean PROD tilt +{np.mean(tilts):.2f} min "
          f"(same-pos {tiltkind['same']}, diff {tiltkind['diff']}, flat {tiltkind['flat']})")
    print(f"D35 contamination (share of trailing-10 games star also missed): "
          f"mean {contam.mean():.2f}; fresh rows (<=0.3) {np.mean(contam <= 0.3):.1%}")

    def table(mask, label):
        print(f"\n--- {label} (n={mask.sum()}, {len(np.unique(players[mask]))} players) ---")
        print(f"{'market':10} {'arm':6} {'CRPS':>8} {'delta':>9} {'CI95':>20}  verdict")
        for m in MARKETS:
            base = np.array(res[m]["base"])[mask]
            print(f"{m:10} {'base':6} {base.mean():8.4f}")
            for a in ARMS:
                arm = np.array(res[m][a])[mask]
                d = base - arm                      # + = adjustment better
                lo, hi = cluster_boot(d, players[mask])
                v = "BETTER" if lo > 0 else ("WORSE" if hi < 0 else "~match (CI spans 0)")
                print(f"{'':10} {a:6} {arm.mean():8.4f} {d.mean():+9.4f} "
                      f"[{lo:+.4f},{hi:+.4f}]  {v}")
        print("attempts (shots Poisson LL, + = better):")
        for i, a in enumerate(ARMS):
            d = (ll[:, i + 1] - ll[:, 0])[mask]
            lo, hi = cluster_boot(d, players[mask], seed=1)
            v = "BETTER" if lo > 0 else ("WORSE" if hi < 0 else "~match")
            print(f"  {a:6} dLL {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  {v}")

    table(np.ones(n, bool), "POOLED (prod scales were fit on pre-cut rows)")
    table(heldout, f"HELD-OUT (date > q{TRAIN_Q} — clean OOS for prod)")

    print("\nprod stratified by D35 contamination (fresh = October-live analog):")
    for lab, mask in (("FRESH  (contam<=0.3)", contam <= 0.3),
                      ("STALE  (contam> 0.3)", contam > 0.3)):
        if mask.sum() < 30:
            print(f"  {lab}: n={mask.sum()} too thin"); continue
        parts = []
        for m in MARKETS:
            d = (np.array(res[m]["base"]) - np.array(res[m]["prod"]))[mask]
            lo, hi = cluster_boot(d, players[mask], seed=2)
            parts.append(f"{m[:3]} {d.mean():+.3f}[{lo:+.3f},{hi:+.3f}]")
        dll = (ll[:, 1] - ll[:, 0])[mask]
        lo, hi = cluster_boot(dll, players[mask], seed=3)
        parts.append(f"attLL {dll.mean():+.3f}[{lo:+.3f},{hi:+.3f}]")
        print(f"  {lab} n={mask.sum():5d}  " + "  ".join(parts))
    print("\nVALIDATE_STAROUT_DONE", flush=True)


if __name__ == "__main__":
    main(max_eval=int(sys.argv[1]) if len(sys.argv) > 1 else None)
