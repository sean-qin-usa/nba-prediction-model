#!/usr/bin/env python3
"""AUDIT: re-run the D38 conditional (C&S-tilted) redistribution gate with a
point-in-time-clean C&S feature.

THE LEAK BEING FIXED
--------------------
scripts/gate_conditional_redis.py loads ONE full-season Catch&Shoot aggregate
(season "2024-25", FGA_FREQUENCY over all 82 games) and applies it to star-out
player-games from ALL seasons in the DB (game_id LIKE '002%' -> 2023-24,
2024-25, 2025-26). That is two distinct violations at once:
  * 2023-24 rows  -> FUTURE leak (a feature built from a season that had not
                     happened yet at game_date),
  * 2024-25 rows  -> SAME-SEASON leak (the aggregate contains the very game
                     being predicted, plus the rest of that season),
  * 2025-26 rows  -> clean: 2024-25 is a completed PRIOR season, so the
                     aggregate is strictly < game_date.
Verdict on the leaked run was D38 REJECTED (delta -0.0128, CI negative).

WHAT THIS SCRIPT DOES
---------------------
  A. REPLICATION of the leaked gate (must reproduce n=20,566 / -0.0128) plus a
     per-season decomposition, which isolates the clean stratum inside it.
  B. PIT-CLEAN GATE: eval rows restricted to 2025-26 ONLY, tilt built from the
     2024-25 full-season C&S aggregate = legitimate prior-season feature.
  C. LEAK CONTRAST: same 2025-26 rows, tilt built from the 2025-26 (same-season,
     in-sample) aggregate. Difference B vs C = the price of the leak, measured
     on identical rows.
  D. PLACEBO: tilt values permuted across rows within season. Separates "the
     tilt carries wrong-signed information" from "any multiplicative jitter of
     this size hurts a Poisson LL".

HONEST CONSTRUCTION CAVEAT (kept from the original, NOT fixed here)
-------------------------------------------------------------------
The OUT-set is ORACLE. Who is out is inferred from the realized box score of
the game being predicted (`played` = players with >=8 min in THAT game), and
the peer pool `team_now` is likewise the realized >=12-min rotation of THAT
game. Production would have to use a pre-game injury report / projected
lineup. Both arms (uniform and tilted) share this oracle identically, so it
does not bias the tilt-vs-uniform contrast, but every absolute number here is
optimistic and this gate does NOT establish deployable performance.
Second common-mode caveat: the usage prior u (data/v2_usage.npz) is fit on
play-by-play pooled across all seasons with a random 70/30 split, so it is
itself in-sample. It enters the uniform lift L and both arms identically, so
again it is common-mode for this contrast, but it means L is flattered.

Ground rules honored: read-only DuckDB, new file only, PIT inputs strictly
< game_date (except the oracle OUT-set, declared above), paired bootstrap
2000x / 95% CI, CLUSTERED BY PLAYER (rows are player-games).
"""
import sys
import glob
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import orjson

from nbapred.db import connect
from nbapred.ingest.nba_stats import _frames

ROOT = Path(__file__).resolve().parent.parent
B = 2000
SEASON_OF = {"00223": "2023-24", "00224": "2024-25", "00225": "2025-26"}


# ----------------------------------------------------------------- features
def load_cs(season):
    """Full-season Catch&Shoot FGA_FREQUENCY by player_id for `season`."""
    out = {}
    for f in glob.glob(str(ROOT / "data/raw/nba_api/ptshot/*.json")):
        d = orjson.loads(open(f, "rb").read())
        if (d["params"].get("general_range_nullable") == "Catch and Shoot"
                and d["params"].get("season") == season):
            df = list(_frames(d["response"]).values())[0]
            for r in df.itertuples():
                out[int(r.PLAYER_ID)] = float(np.nan_to_num(r.FGA_FREQUENCY or 0))
    return out


def build_skeleton():
    """Star-out rows, C&S-agnostic. All PIT except the declared oracle OUT-set.

    Returns list of dicts: season, player_id, avg_shots (PIT 10-game mean of
    prior shot counts), shots (realized), L (uniform softmax lift), peers
    (realized rotation player_ids in that team-game).
    """
    con = connect(read_only=True)
    pg = con.execute("""
        SELECT s.game_id, s.player_id, s.team_id, g.game_date,
               s.seconds/60.0 mins, s.rima+s.mida+s.thra shots
        FROM player_game_stats s
        JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g USING (game_id)
        WHERE s.game_id LIKE '002%'
        ORDER BY g.game_date
    """).fetchdf()
    con.close()

    uz = np.load(ROOT / "data/v2_usage.npz")
    u = dict(zip(uz["player_ids"].tolist(), uz["u"].tolist()))

    pg = pg.sort_values(["player_id", "game_date"])
    # strictly-prior rolling windows (shift(1)) -> PIT
    pg["avg_min"] = pg.groupby("player_id")["mins"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=5).mean())
    pg["avg_shots"] = pg.groupby("player_id")["shots"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=5).mean())

    played = pg[pg.mins >= 8].groupby(["game_id", "team_id"])["player_id"].apply(set)
    played = played.to_dict()                                   # ORACLE
    stars = pg[pg.avg_min >= 28.0]
    sbt = {}
    for r in stars[["player_id", "team_id", "game_date"]].itertuples():
        sbt.setdefault(r.team_id, []).append((r.game_date, r.player_id))
    # dates come back as python date objects -> use ordinals for vectorized math
    sbt = {t: (np.array([d.toordinal() for d, _ in v]), np.array([p for _, p in v]))
           for t, v in sbt.items()}

    rot = pg[(pg.avg_min >= 15) & (pg.mins >= 12) & pg.avg_shots.notna()].copy()
    # peers of a team-game = the realized >=12-min rotation (ORACLE, as original)
    peers = {k: list(map(int, v)) for k, v in
             rot.groupby(["team_id", "game_date"])["player_id"].apply(list).to_dict().items()}

    recent_cache = {}

    def recent_stars(team_id, date):
        key = (team_id, date)
        if key not in recent_cache:
            if team_id not in sbt:
                recent_cache[key] = set()
            else:
                dts, pids = sbt[team_id]
                dd = date.toordinal() - dts          # strictly-prior window
                recent_cache[key] = set(pids[(dd > 0) & (dd <= 12)].tolist())
        return recent_cache[key]

    rows = []
    for r in rot.itertuples():
        recent = recent_stars(r.team_id, r.game_date)
        if not recent:
            continue
        outs = (recent - played.get((r.game_id, r.team_id), set())) - {r.player_id}
        if not outs:
            continue
        star = max(outs, key=lambda p: u.get(p, 0.0))
        pl = peers.get((r.team_id, r.game_date), [])
        pool = set(pl) | {int(star)}
        S = sum(np.exp(u.get(p, 0.0)) for p in pool)
        L = float(min(S / max(S - np.exp(u.get(star, 0.0)), 1e-9), 1.5))
        rows.append(dict(season=SEASON_OF[r.game_id[:5]], player_id=int(r.player_id),
                         avg_shots=float(r.avg_shots), shots=float(r.shots),
                         L=L, peers=pl))
    return rows


def arm_arrays(rows, cs_map, keep=None, tilt_override=None):
    """Apply a C&S map to skeleton rows -> (base, y, L, tilted_lift, pid).

    keep: optional index set to force identical row sets across configs.
    tilt_override: optional array of tilts (placebo), aligned to the kept rows.
    """
    idx, base, y, L, tilt, pid = [], [], [], [], [], []
    for i, r in enumerate(rows):
        if keep is not None and i not in keep:
            continue
        cs_i = cs_map.get(r["player_id"])
        team_cs = [cs_map[p] for p in r["peers"] if p in cs_map]
        if cs_i is None or len(team_cs) < 3:
            continue
        idx.append(i)
        base.append(r["avg_shots"]); y.append(r["shots"]); L.append(r["L"])
        tilt.append(float(np.clip(cs_i / max(np.mean(team_cs), 1e-6), 0.3, 2.5)))
        pid.append(r["player_id"])
    tilt = np.array(tilt)
    if tilt_override is not None:
        tilt = tilt_override
    return (np.array(idx), np.array(base), np.array(y), np.array(L),
            1 + (np.array(L) - 1) * tilt, np.array(pid), tilt)


# ----------------------------------------------------------------- statistics
def poisson_ll(pred, y):
    pred = np.clip(pred, 0.2, None)
    return y * np.log(pred) - pred


def boot_ci(d, pids, seed=0):
    """Paired bootstrap, 2000x, 95% CI. iid AND clustered-by-player."""
    rng = np.random.default_rng(seed)
    n = len(d)
    bs_iid = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    labs, inv = np.unique(pids, return_inverse=True)
    K = len(labs)
    sums = np.bincount(inv, weights=d, minlength=K)
    cnts = np.bincount(inv, minlength=K).astype(float)
    pick = rng.integers(0, K, (B, K))
    bs_cl = sums[pick].sum(axis=1) / cnts[pick].sum(axis=1)
    return (np.percentile(bs_iid, [2.5, 97.5]), np.percentile(bs_cl, [2.5, 97.5]),
            bs_iid.std(), bs_cl.std(), K)


def report(label, base, y, L, tilted, pid, tilt, seed=0, indent="  "):
    lu = poisson_ll(base * L, y)
    lt = poisson_ll(base * tilted, y)
    d = lt - lu
    (ilo, ihi), (clo, chi), se_i, se_c, K = boot_ci(d, pid, seed)
    verdict = "KEEP tilt" if clo > 0 else ("REJECT tilt" if chi < 0 else "INCONCLUSIVE")
    print(f"{indent}{label}")
    print(f"{indent}  n={len(d)} player-games, {K} unique players "
          f"({len(d)/max(K,1):.1f} rows/player)")
    print(f"{indent}  mean uniform-lift {L.mean():.3f}  mean tilted-lift {tilted.mean():.3f}"
          f"  mean tilt {tilt.mean():.3f} (sd {tilt.std():.3f})")
    print(f"{indent}  Poisson LL  uniform {lu.mean():.5f}   tilted {lt.mean():.5f}")
    print(f"{indent}  delta (tilted-uniform) {d.mean():+.5f}")
    print(f"{indent}    iid  bootstrap CI ({ilo:+.5f},{ihi:+.5f})  SE {se_i:.5f}")
    print(f"{indent}    player-cluster CI ({clo:+.5f},{chi:+.5f})  SE {se_c:.5f} "
          f"[x{se_c/max(se_i,1e-12):.2f} inflation]  -> {verdict}")
    return dict(n=len(d), K=K, delta=float(d.mean()), cl_lo=float(clo), cl_hi=float(chi),
                iid_lo=float(ilo), iid_hi=float(ihi), se_c=float(se_c), verdict=verdict,
                ll_u=float(lu.mean()), ll_t=float(lt.mean()), d=d, pid=pid)


def main():
    print("=" * 78)
    print("AUDIT: D38 conditional C&S-tilted redistribution, PIT-clean re-gate")
    print("OUT-set construction is ORACLE (realized >=8-min played-set of the game "
          "being\npredicted); kept as-is from the original. Common-mode to both arms.")
    print("=" * 78)

    rows = build_skeleton()
    seasons = np.array([r["season"] for r in rows])
    print(f"skeleton star-out rows (all seasons, pre-C&S filter): {len(rows)}")
    for s in ("2023-24", "2024-25", "2025-26"):
        print(f"   {s}: {(seasons == s).sum()}")

    cs24 = load_cs("2024-25")
    cs25 = load_cs("2025-26")
    print(f"C&S aggregates loaded: 2024-25 n={len(cs24)} players, "
          f"2025-26 n={len(cs25)} players (both FULL-season, GP up to 82)")

    # ---------------------------------------------------------------- A
    print("\n" + "=" * 78)
    print("A) REPLICATION of the LEAKED gate (2024-25 C&S applied to ALL seasons)")
    print("=" * 78)
    i_all, b_all, y_all, L_all, t_all, p_all, tl_all = arm_arrays(rows, cs24)
    r_all = report("POOLED (as shipped in D38)", b_all, y_all, L_all, t_all, p_all, tl_all)
    s_all = seasons[i_all]
    print("\n  per-season decomposition of the SAME leaked run:")
    leak_kind = {"2023-24": "FUTURE leak", "2024-25": "SAME-SEASON leak",
                 "2025-26": "CLEAN prior-season"}
    per_season = {}
    for s in ("2023-24", "2024-25", "2025-26"):
        m = s_all == s
        if m.sum() < 50:
            continue
        per_season[s] = report(f"{s}  [{leak_kind[s]}]", b_all[m], y_all[m], L_all[m],
                               t_all[m], p_all[m], tl_all[m], seed=1, indent="    ")

    # ---------------------------------------------------------------- B
    print("\n" + "=" * 78)
    print("B) PIT-CLEAN GATE: eval rows = 2025-26 ONLY, tilt = 2024-25 prior season")
    print("=" * 78)
    keep25 = {i for i, r in enumerate(rows) if r["season"] == "2025-26"}
    i_b, b_b, y_b, L_b, t_b, p_b, tl_b = arm_arrays(rows, cs24, keep=keep25)
    r_b = report("PIT-CLEAN D38 re-gate", b_b, y_b, L_b, t_b, p_b, tl_b, seed=2)

    # ---------------------------------------------------------------- C
    print("\n" + "=" * 78)
    print("C) LEAK CONTRAST on IDENTICAL rows: prior-season tilt vs same-season tilt")
    print("=" * 78)
    i_c, b_c, y_c, L_c, t_c, p_c, tl_c = arm_arrays(rows, cs25, keep=keep25)
    common = np.intersect1d(i_b, i_c)
    mb = np.isin(i_b, common); mc = np.isin(i_c, common)
    print(f"  rows qualifying under BOTH C&S vintages: {len(common)}")
    rb = report("tilt from 2024-25 (PIT-clean)", b_b[mb], y_b[mb], L_b[mb], t_b[mb],
                p_b[mb], tl_b[mb], seed=3)
    rc = report("tilt from 2025-26 (SAME-SEASON leak)", b_c[mc], y_c[mc], L_c[mc],
                t_c[mc], p_c[mc], tl_c[mc], seed=3)
    dd = (poisson_ll(b_c[mc] * t_c[mc], y_c[mc]) - poisson_ll(b_b[mb] * t_b[mb], y_b[mb]))
    (_, _), (lo, hi), _, se, _ = boot_ci(dd, p_b[mb], seed=4)
    print(f"  leak advantage (same-season tilt - prior-season tilt) {dd.mean():+.5f} "
          f"player-cluster CI ({lo:+.5f},{hi:+.5f})")
    print(f"  tilt correlation across vintages (same rows): "
          f"{np.corrcoef(tl_b[mb], tl_c[mc])[0,1]:+.3f}")

    # ---------------------------------------------------------------- D
    print("\n" + "=" * 78)
    print("D) PLACEBO: tilt permuted across the 2025-26 rows (destroys the pairing,")
    print("   preserves the tilt marginal) -> how much of the effect is mere jitter?")
    print("=" * 78)
    rngp = np.random.default_rng(11)
    ds = []
    for k in range(20):
        perm = rngp.permutation(len(tl_b))
        tt = 1 + (L_b - 1) * tl_b[perm]
        ds.append((poisson_ll(b_b * tt, y_b) - poisson_ll(b_b * L_b, y_b)).mean())
    ds = np.array(ds)
    print(f"  placebo delta over 20 permutations: mean {ds.mean():+.5f} "
          f"sd {ds.std():.5f}  range ({ds.min():+.5f},{ds.max():+.5f})")
    print(f"  real PIT-clean delta {r_b['delta']:+.5f}  -> "
          f"{'real tilt is WORSE than random tilt (wrong-signed signal)' if r_b['delta'] < ds.mean() else 'real tilt beats random tilt (some signal), sign vs 0 is the gate'}")

    # ---------------------------------------------------------------- F
    print("\n" + "=" * 78)
    print("F) EXPLORATORY (not a gate; multiplicity uncontrolled): tilt STRENGTH a,")
    print("   tilt_a = 1 + a*(tilt-1). a=1 is D38 as specified, a=0 is uniform.")
    print("=" * 78)
    lu_b = poisson_ll(b_b * L_b, y_b)
    for a in (0.25, 0.50, 0.75, 1.00):
        ta = 1 + (L_b - 1) * (1 + a * (tl_b - 1))
        da = poisson_ll(b_b * ta, y_b) - lu_b
        (_, _), (lo_a, hi_a), _, _, _ = boot_ci(da, p_b, seed=5)
        print(f"  a={a:.2f}  delta {da.mean():+.5f}  player-cluster CI "
              f"({lo_a:+.5f},{hi_a:+.5f})  "
              f"{'KEEP' if lo_a > 0 else ('REJECT' if hi_a < 0 else 'INCONCLUSIVE')}")

    # ---------------------------------------------------------------- power
    print("\n" + "=" * 78)
    print("E) POWER / MDE on the PIT-clean stratum")
    print("=" * 78)
    print(f"  player-cluster SE {r_b['se_c']:.5f} -> one-sided 2.5% MDE "
          f"{1.96*r_b['se_c']:+.5f} in mean Poisson LL")
    print(f"  leaked pooled n={r_all['n']} vs PIT-clean n={r_b['n']} "
          f"({r_b['n']/r_all['n']:.0%} of rows)")

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  ORIGINAL (leaked, iid CI) : delta {r_all['delta']:+.5f} "
          f"CI ({r_all['iid_lo']:+.5f},{r_all['iid_hi']:+.5f}) -> REJECTED")
    print(f"  PIT-CLEAN (player-cluster): delta {r_b['delta']:+.5f} "
          f"CI ({r_b['cl_lo']:+.5f},{r_b['cl_hi']:+.5f}) -> {r_b['verdict']}")
    flipped = (r_all['iid_hi'] < 0) and (r_b['cl_lo'] > 0)
    print(f"  D38 verdict FLIPPED by PIT-clean construction: {flipped}")
    print("AUDIT_CONDREDIS_PIT_DONE", flush=True)


if __name__ == "__main__":
    main()
