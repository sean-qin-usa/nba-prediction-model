"""REGIME D — LATE-SEASON RESIDUAL decomposition (post-tank +~0.017/gm, ~21 nats).

Heavy decomposition of the residual vs market close in the late-season window,
on the PRODUCTION per-game file (capstone_pergame_tank.csv = carry D62 + tank
D73 + sched D46). Sections:

  S0  Window reproduction: tsd!=0 (tank-eligible) vs Mar/Apr vs Feb+ overlap.
  S1  Residual splits: tank-moved vs tank-inert; seeding-locked (H4 clinch);
      urgency (F5, measurement only — frozen); playoff-preview (top8 v top8);
      rotation-shortening (starter-minutes-share trend, PIT).
  S2  Tank term's own residual: post-correction slope on tsd, binned shape
      (saturation vs linear vs acceleration), shrink undershoot (shipped k vs
      hindsight k), counterfactual re-links p' = sigmoid((m_base + k g(tsd))/7.2)
      where m_base = 7.2*logit(p_us) - k_ship*tsd is EXACT by construction.
  S3  Schedule-density endgame: 4-in-6 / 5-in-7 in the final month, fatigue
      asymmetry (margin residual) late vs earlier.
  S4  Hindsight closure table for construction candidates.

PIT: standings/rotation/density use games strictly before game_date only.
Market data (odds_market home_exp_margin / close) used ONLY as benchmark and
for D76-style region definitions. DuckDB read_only=True. No repo writes.
"""
from __future__ import annotations

import json
import sys

import duckdb
import numpy as np
import pandas as pd

DB = "/hdd/steveqin/sean_dev/nba_model/data/nba.duckdb"
CSV = "/hdd/steveqin/sean_dev/nba_model/data/capstone_pergame_tank.csv"
TANKSTATS = "/hdd/steveqin/sean_dev/nba_model/data/apr_tank_stats.csv"
URG = "/hdd/steveqin/sean_dev/nba_model/data/mr_urgency_pergame.csv"
SCALE = 7.2
RNG = np.random.default_rng(46)

EAST = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DET", "IND", "MIA", "MIL",
        "NYK", "ORL", "PHI", "TOR", "WAS"}


def ll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def boot_ci(x, B=4000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(x), (B, len(x)))
    return tuple(np.percentile(x[idx].mean(axis=1), [2.5, 97.5]))


def region(df, mask, name, dpool):
    g = df[mask]
    if len(g) == 0:
        print(f"  {name:44s} n=0")
        return
    lo, hi = boot_ci(g.d.values)
    share = g.d.sum() / dpool * 100.0
    print(f"  {name:44s} n={len(g):4d}  d/gm {g.d.mean():+.4f} "
          f"CI({lo:+.4f},{hi:+.4f})  nats {g.d.sum():+6.1f} ({share:+5.1f}% of pool)"
          f"  by-season " + " ".join(
              f"{s[-5:]}:{g[g.season == s].d.mean():+.3f}({(g.season == s).sum()})"
              for s in sorted(g.season.unique())))


def main():
    con = duckdb.connect(DB, read_only=True)
    tg = con.execute("""SELECT season, game_id, game_date, team_abbrev, wl
        FROM nba_games WHERE game_id LIKE '002%'
          AND season IN ('2023-24','2024-25','2025-26')""").df()
    om = con.execute("""SELECT game_date, home, away, score_home, score_away,
        home_exp_margin FROM odds_market WHERE season_end >= 2024""").df()
    pg = con.execute("""SELECT s.game_id, s.team_id, s.seconds
        FROM player_game_stats s
        WHERE s.game_id LIKE '002%' AND s.seconds IS NOT NULL""").df()
    tid = con.execute("""SELECT DISTINCT season, game_id, game_date, team_id,
        team_abbrev FROM nba_games WHERE game_id LIKE '002%'
          AND season IN ('2022-23','2023-24','2024-25','2025-26')""").df()
    con.close()

    tg["game_date"] = pd.to_datetime(tg.game_date)
    om["game_date"] = om.game_date.astype(str)
    tid["game_date"] = pd.to_datetime(tid.game_date)

    df = pd.read_csv(CSV, dtype={"game_id": str})
    df["game_id"] = df.game_id.str.zfill(10)
    df = df.merge(om, on=["game_date", "home", "away"], how="left")
    assert df.home_exp_margin.notna().all(), "odds merge failed"
    df["game_date"] = pd.to_datetime(df.game_date)
    df["month"] = df.game_date.dt.month
    df["margin_home"] = df.score_home - df.score_away
    df["m_us"] = SCALE * logit(df.p_us)
    df["m_base"] = df.m_us - df.k * df.tsd          # exact pre-tank margin
    df["d"] = ll(df.p_us.values, df.y.values) - ll(df.p_mkt.values, df.y.values)
    df["dp_tank"] = (df.p_us - sigmoid(df.m_base / SCALE)).abs()

    # gp_before + tank_score per side from the program's own stats file
    ts = pd.read_csv(TANKSTATS, dtype={"game_id": str})
    ts["game_id"] = ts.game_id.str.zfill(10)
    side = ts[["game_id", "team_abbrev", "gp_before", "tank_score"]]
    df = df.merge(side.rename(columns={"team_abbrev": "home",
                                       "gp_before": "gp_h",
                                       "tank_score": "tank_h"}),
                  on=["game_id", "home"], how="left")
    df = df.merge(side.rename(columns={"team_abbrev": "away",
                                       "gp_before": "gp_a",
                                       "tank_score": "tank_a"}),
                  on=["game_id", "away"], how="left")
    assert df.gp_h.notna().all() and df.gp_a.notna().all()

    # ---- standings (PIT: strictly before game_date) --------------------- #
    hist = {}
    for (season, team), g in tg.groupby(["season", "team_abbrev"]):
        g = g.sort_values("game_date")
        hist[(season, team)] = (g.game_date.values,
                                np.cumsum((g.wl == "W").values))

    def record_before(season, team, D):
        dates, wins = hist[(season, team)]
        i = int(np.searchsorted(dates, D))
        w = int(wins[i - 1]) if i > 0 else 0
        return w, i - w, i

    conf_cache = {}

    def conf_table(season, is_east, D):
        ck = (season, is_east, D)
        if ck not in conf_cache:
            teams = [t for (s, t) in hist if s == season
                     and ((t in EAST) == is_east)]
            rows = []
            for t in teams:
                w, l, gp = record_before(season, t, D)
                rows.append((t, w, l, gp, 82 - gp))
            rows.sort(key=lambda r: (-r[1], r[2]))
            conf_cache[ck] = rows
        return conf_cache[ck]

    def flags_for(season, team, D):
        tab = conf_table(season, team in EAST, D)
        w, l, gp = record_before(season, team, D)
        t7 = tab[6]
        clin = (w - (t7[1] + t7[4]) > 0) and team != t7[0]   # H4 clinch
        rank = next(i for i, r in enumerate(tab) if r[0] == team)
        return clin, rank < 8, gp

    fh = df.apply(lambda r: flags_for(r.season, r.home,
                                      np.datetime64(r.game_date)), axis=1)
    fa = df.apply(lambda r: flags_for(r.season, r.away,
                                      np.datetime64(r.game_date)), axis=1)
    df["clin_h"] = [x[0] for x in fh]
    df["top8_h"] = [x[1] for x in fh]
    df["clin_a"] = [x[0] for x in fa]
    df["top8_a"] = [x[1] for x in fa]

    # ---- rotation tightness (PIT trailing shares) ----------------------- #
    pg = pg.merge(tid, on=["game_id", "team_id"], how="inner")
    pg = pg.sort_values("seconds", ascending=False)
    pg["rk"] = pg.groupby(["game_id", "team_id"]).cumcount()
    tot = pg.groupby(["season", "game_id", "game_date", "team_abbrev"]
                     ).seconds.sum().rename("sec_tot")
    top7 = pg[pg.rk < 7].groupby(["season", "game_id", "game_date",
                                  "team_abbrev"]).seconds.sum().rename("sec7")
    rot = pd.concat([tot, top7], axis=1).reset_index()
    rot["share7"] = rot.sec7 / rot.sec_tot
    rot = rot.sort_values("game_date")
    rot["trail10"] = (rot.groupby(["season", "team_abbrev"]).share7
                      .transform(lambda s: s.shift(1).rolling(10, min_periods=8)
                                 .mean()))
    rot["base20"] = (rot.groupby(["season", "team_abbrev"]).share7
                     .transform(lambda s: s.shift(11).rolling(20, min_periods=15)
                                .mean()))
    rot["rot_trend"] = rot.trail10 - rot.base20
    rkey = rot[["season", "game_date", "team_abbrev", "rot_trend"]]
    df = df.merge(rkey.rename(columns={"team_abbrev": "home",
                                       "rot_trend": "rt_h"}),
                  on=["season", "game_date", "home"], how="left")
    df = df.merge(rkey.rename(columns={"team_abbrev": "away",
                                       "rot_trend": "rt_a"}),
                  on=["season", "game_date", "away"], how="left")

    # ---- schedule density 4in6 / 5in7 (calendar, PIT-safe) -------------- #
    dates_by_team = {k: v[0] for k, v in hist.items()}

    def dens(season, team, D):
        dates = dates_by_team[(season, team)]
        i = int(np.searchsorted(dates, D))
        prior5 = int(np.sum(dates[:i] >= D - np.timedelta64(5, "D")))
        prior6 = int(np.sum(dates[:i] >= D - np.timedelta64(6, "D")))
        return int(prior5 >= 3), int(prior6 >= 4)   # 4in6, 5in7 incl today

    dh = df.apply(lambda r: dens(r.season, r.home,
                                 np.datetime64(r.game_date)), axis=1)
    da = df.apply(lambda r: dens(r.season, r.away,
                                 np.datetime64(r.game_date)), axis=1)
    df["d46_h"] = [x[0] for x in dh]
    df["d57_h"] = [x[1] for x in dh]
    df["d46_a"] = [x[0] for x in da]
    df["d57_a"] = [x[1] for x in da]

    # urgency file (F5) — measurement only
    u = pd.read_csv(URG, dtype={"game_id": str})
    u["game_id"] = u.game_id.str.zfill(10)
    df = df.merge(u[["game_id", "u_home", "u_away"]], on="game_id", how="left")

    # ================= S0 window reproduction ============================ #
    print("=" * 100)
    print("S0 WINDOW: production capstone_pergame_tank.csv, n=%d" % len(df))
    act = df.tsd != 0
    print(f"  pooled d/gm {df.d.mean():+.4f}  total {df.d.sum():+.1f} nats")
    for name, m in [("tsd!=0 (tank-eligible window)", act),
                    ("Mar+Apr", df.month.isin([3, 4])),
                    ("either gp>=55", (df.gp_h >= 55) | (df.gp_a >= 55)),
                    ("both gp>=55", (df.gp_h >= 55) & (df.gp_a >= 55)),
                    ("tsd!=0 & Mar/Apr", act & df.month.isin([3, 4])),
                    ("tsd!=0 & pre-Mar", act & ~df.month.isin([3, 4]))]:
        g = df[m]
        print(f"  {name:32s} n={len(g):4d}  d/gm {g.d.mean():+.4f}  "
              f"nats {g.d.sum():+6.1f}")
    dpool = df[act].d.sum()
    late = act  # REGIME D window = tank-eligible late window

    # ================= S1 residual splits ================================ #
    print("=" * 100)
    print("S1 SPLITS inside tsd!=0 window (pool %.1f nats). d = L_us - L_mkt "
          "(negative = we beat close)" % dpool)
    q = df.dp_tank[late].quantile([0.5, 0.75]).values
    print(f"  tank |dp| moved quantiles: median {q[0]:.4f}, p75 {q[1]:.4f}")
    region(df, late & (df.dp_tank >= 0.01), "tank MOVED p>=.01", dpool)
    region(df, late & (df.dp_tank < 0.01), "tank inert (<.01)", dpool)
    one_clin = late & (df.clin_h ^ df.clin_a)
    region(df, one_clin, "seeding-locked one side (H4 clinch)", dpool)
    region(df, late & df.clin_h & df.clin_a, "both clinched", dpool)
    region(df, late & ~df.clin_h & ~df.clin_a, "neither clinched", dpool)
    uact = late & ((df.u_home - df.u_away) != 0)
    region(df, uact, "urgency active (F5 def)", dpool)
    region(df, late & df.top8_h & df.top8_a, "playoff preview (both top8)",
           dpool)
    region(df, late & df.top8_h & df.top8_a & ~(df.clin_h & df.clin_a),
           "playoff preview, not both-clinched", dpool)
    rknown = df.rt_h.notna() & df.rt_a.notna()
    rthr = 0.015
    short_h = (df.rt_h > rthr) & df.top8_h
    short_a = (df.rt_a > rthr) & df.top8_a
    region(df, late & rknown & (short_h ^ short_a),
           f"rotation-shortening one side (>{rthr:+.3f}, top8)", dpool)
    region(df, late & rknown & (short_h | short_a),
           "rotation-shortening any side", dpool)
    # complement: what's left after tank+clinch+urgency+preview coverage
    covered = (df.dp_tank >= 0.01) | one_clin | uact | (df.top8_h & df.top8_a)
    region(df, late & ~covered, "UNCOVERED remainder", dpool)
    # margin residual direction for named effort regions
    print("  -- margin residuals (team perspective, resid = margin - m_us) --")
    for nm, tm, sgn in [("clinched side", df.clin_h, 1),
                        ("clinched side (away)", df.clin_a, -1)]:
        g = df[late & tm & ~(df.clin_h & df.clin_a)]
        r = sgn * (g.margin_home - g.m_us)
        rm = sgn * (g.margin_home - g.home_exp_margin)
        print(f"    {nm:26s} n={len(g):4d} resid_us {r.mean():+5.2f} "
              f"resid_mkt {rm.mean():+5.2f}")
    for nm, x in [("rot-short side (home)",
                   df[late & rknown & short_h & ~short_a]),
                  ("rot-short side (away)",
                   df[late & rknown & short_a & ~short_h])]:
        sgn = 1 if "home" in nm else -1
        r = sgn * (x.margin_home - x.m_us)
        rm = sgn * (x.margin_home - x.home_exp_margin)
        print(f"    {nm:26s} n={len(x):4d} resid_us {r.mean():+5.2f} "
              f"resid_mkt {rm.mean():+5.2f}")

    # ================= S2 tank term's own residual ======================= #
    print("=" * 100)
    print("S2 TANK TERM SHAPE (active games)")
    a = df[late].copy()
    resid_post = a.margin_home - a.m_us          # after shipped correction
    resid_pre = a.margin_home - a.m_base         # before tank term
    for nm, r in [("pre-tank resid ~ tsd", resid_pre),
                  ("post-tank resid ~ tsd", resid_post)]:
        X = np.column_stack([np.ones(len(a)), a.tsd.values])
        beta, *_ = np.linalg.lstsq(X, r.values, rcond=None)
        e = r.values - X @ beta
        se = np.sqrt((e @ e) / (len(a) - 2)
                     * np.linalg.inv(X.T @ X)[1, 1])
        print(f"  {nm:26s} slope {beta[1]:+.3f} +-{1.96 * se:.3f} "
              f"(shipped k range {a.k.min():.2f}..{a.k.max():.2f})")
    # binned shape: decile means of pre-tank residual vs tsd
    a["bin"] = pd.qcut(a.tsd, 10, labels=False, duplicates="drop")
    print("  tsd decile -> mean tsd, mean pre-tank resid, n:")
    for b, g in a.groupby("bin"):
        print(f"    d{int(b)}: tsd {g.tsd.mean():+6.3f}  resid_pre "
              f"{(g.margin_home - g.m_base).mean():+6.2f}  n={len(g)}")

    def refit_p(kfun):
        m = df.m_base + kfun(df)
        return sigmoid(m / SCALE)

    def mle_k(g, active_mask):
        """hindsight pooled 1-D MLE for coef on g(tsd), active games only."""
        gv = np.where(active_mask, g, 0.0)
        ks = np.linspace(-8, 2, 401)
        best, bk = 1e18, 0.0
        y = df.y.values
        for k in ks:
            p = sigmoid((df.m_base.values + k * gv) / SCALE)
            L = ll(p, y)[active_mask].sum()
            if L < best:
                best, bk = L, k
        return bk

    amask = late.values
    tsd = df.tsd.values
    c75 = np.percentile(np.abs(tsd[amask]), 75)
    shapes = {
        "linear (refit k, no shrink)": tsd,
        f"saturating |tsd| cap {c75:.2f}": np.sign(tsd) * np.minimum(
            np.abs(tsd), c75),
        "accelerating sign*|tsd|^1.5": np.sign(tsd) * np.abs(tsd) ** 1.5,
        "deadzone-then-linear (|tsd|>p25)": np.sign(tsd) * np.maximum(
            np.abs(tsd) - np.percentile(np.abs(tsd[amask]), 25), 0.0),
    }
    print("  counterfactual re-links on active games (hindsight pooled k; "
          "shipped active d/gm %+0.4f):" % df[late].d.mean())
    shape_rows = {}
    for nm, g in shapes.items():
        k = mle_k(g, amask)
        p = sigmoid((df.m_base.values + np.where(amask, k * g, 0.0)) / SCALE)
        dd = ll(p, df.y.values) - ll(df.p_mkt.values, df.y.values)
        delta = ll(p, df.y.values) - ll(df.p_us.values, df.y.values)
        lo, hi = boot_ci(delta[amask])
        print(f"    {nm:34s} k*={k:+.2f}  d/gm {dd[amask].mean():+.4f}  "
              f"vs shipped {delta[amask].mean():+.5f} CI({lo:+.5f},{hi:+.5f})"
              "  by-season " + " ".join(
                  f"{s[-5:]}:{delta[amask & (df.season == s).values].mean():+.4f}"
                  for s in sorted(df.season.unique())))
        shape_rows[nm] = dict(k=k, d=float(dd[amask].mean()),
                              delta=float(delta[amask].mean()))
    # per-season hindsight linear k (shrink undershoot check)
    for s in sorted(df.season.unique()):
        m = amask & (df.season == s).values
        k = mle_k(tsd, m)
        print(f"    per-season hindsight linear k {s}: {k:+.2f} "
              f"(shipped mean {df.k[m].mean():+.2f})")

    # ================= S3 schedule-density endgame ======================= #
    print("=" * 100)
    print("S3 DENSITY ENDGAME (4in6/5in7), margin resid = actual - m_us "
          "(team perspective)")
    for label, mwin in [("Oct-Feb", ~df.month.isin([3, 4])),
                        ("Mar-Apr", df.month.isin([3, 4]))]:
        for dn, fh_, fa_ in [("4in6", df.d46_h, df.d46_a),
                             ("5in7", df.d57_h, df.d57_a)]:
            mm = mwin & ((fh_ == 1) ^ (fa_ == 1))
            g = df[mm]
            if len(g) == 0:
                continue
            sgn = np.where(g.d46_h if dn == "4in6" else g.d57_h, 1.0, -1.0)
            r = sgn * (g.margin_home - g.m_us)
            rm = sgn * (g.margin_home - g.home_exp_margin)
            lo, hi = boot_ci(r)
            dlo, dhi = boot_ci(g.d.values)
            print(f"  {label:7s} {dn} one-side n={len(g):4d}  "
                  f"tired-side resid_us {np.mean(r):+5.2f} CI({lo:+.2f},{hi:+.2f}) "
                  f"resid_mkt {np.mean(rm):+5.2f}  d/gm {g.d.mean():+.4f} "
                  f"CI({dlo:+.4f},{dhi:+.4f})")

    # ================= S4 hindsight closure table ======================== #
    print("=" * 100)
    print("S4 HINDSIGHT CLOSURE (single pooled coef, active window) — bound, "
          "not a gate")
    cands = {}
    # rotation-shortening continuous term (playoff-bound gated)
    rt_h = np.where(df.top8_h, df.rt_h.fillna(0), 0.0)
    rt_a = np.where(df.top8_a, df.rt_a.fillna(0), 0.0)
    cands["rot-shorten c*(rt_h-rt_a), top8-gated"] = (rt_h - rt_a) * 100.0
    # clinch correction
    cands["clinch c*(clin_a-clin_h)"] = (
        df.clin_a.astype(float) - df.clin_h.astype(float)).values
    # density-April term
    ddiff = (df.d46_h.astype(float) - df.d46_a.astype(float)).values
    cands["apr-density c*(d46_a-d46_h)"] = -ddiff
    # urgency (measurement echo; F5 frozen)
    cands["urgency c*(u_h-u_a) [F5 FROZEN]"] = (
        df.u_home.fillna(0) - df.u_away.fillna(0)).values
    closure = {}
    for nm, g in cands.items():
        k = mle_k(g, amask)
        p = sigmoid((df.m_base.values + df.k.values * tsd
                     + np.where(amask, k * g, 0.0)) / SCALE)
        delta = ll(p, df.y.values) - ll(df.p_us.values, df.y.values)
        lo, hi = boot_ci(delta[amask])
        print(f"  {nm:44s} c*={k:+.2f}  vs shipped {delta[amask].mean():+.5f} "
              f"CI({lo:+.5f},{hi:+.5f})  nats {delta[amask].sum():+5.1f}"
              "  by-season " + " ".join(
                  f"{s[-5:]}:{delta[amask & (df.season == s).values].mean():+.4f}"
                  for s in sorted(df.season.unique())))
        closure[nm] = dict(c=float(k), delta=float(delta[amask].mean()),
                           nats=float(delta[amask].sum()))

    out = dict(shapes=shape_rows, closure=closure,
               pool_nats=float(dpool), pool_n=int(late.sum()))
    with open(sys.argv[1] if len(sys.argv) > 1 else
              "/tmp/rw_regimeD_results.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
