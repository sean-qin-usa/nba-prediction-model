#!/usr/bin/env python3
"""D252 — DO WE PREDICT BETTER WHEN STARS SIT, AND WHERE DOES THE EDGE LIVE?

THE FRAMING MATTERS MORE THAN THE TEST. On games where a star is out, everyone
predicts worse — the outcome is more variable and every forecaster's log loss
rises. Asking "is our log loss lower on star-out games" therefore answers
nothing. The estimand is the GAP AGAINST THE MARKET on the same games:

    d_ll = ll_us - ll_mkt        negative means WE BEAT THE MARKET

If d_ll is more negative when a star sits, the market misprices absence relative
to us. If it is flat, absence is priced and the chaos is shared.

STAR DEFINITION IS POINT-IN-TIME. A player's minutes-per-game is taken over that
team's PRIOR games in the same season (expanding mean, shifted by one), so the
current game never informs who counts as a star. Games before the team's 5th are
dropped rather than back-filled from the season average, which would leak.

REALISED ABSENCE, USED AS A PARTITION NOT A PREDICTOR. Whether a star actually
played is known before tip, and here it only SPLITS the evaluation — it never
enters a forecast. That is legitimate and it buys all 19 seasons, where the
injury report would buy 7 (D186).

MULTIPLICITY IS THE REAL RISK. D239 measured this pipeline's capacity to
manufacture findings: best-of-N random subsets buy +2.54 ROI points from nothing,
and D237/D238 found regime gates underpowered rather than real. So the slices
below are FIXED BEFORE ANY IS READ, and the headline is not the best slice but
the best slice COMPARED AGAINST A PERMUTATION NULL OVER THE WHOLE FAMILY: labels
are shuffled within season, the max |gap| across every slice is recomputed, and
the observed max is scored against that distribution.

Reported for both models, because the difference is the point:
  m_us_blind  the market-blind margin (never sees a price)
  m_us        the shipped offset margin (blind margin spent against the opener)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402
from scipy import stats                                           # noqa: E402

SCALE = 6.96          # register convention for turning a margin into p(home win)
TOPK = 3              # "star" = top-3 by prior minutes on that team
MIN_PRIOR = 5         # team games needed before a star ranking is trusted


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def nll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def clus(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    k = len(v)
    if k < 2:
        return np.nan, np.nan, np.nan, k
    se = v.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return v.mean(), v.mean() - tc * se, v.mean() + tc * se, k


def build_star_table(f):
    """Per team-game: how many top-K players sat, and their prior minutes."""
    import duckdb
    con = duckdb.connect(str(ROOT / "data" / "nba.duckdb"), read_only=True)
    pgs = con.execute("""SELECT CAST(game_id AS VARCHAR) gid, player_id,
                                team_id, seconds FROM player_game_stats""").df()
    con.close()
    pgs["gid"] = pgs.gid.str.zfill(10)
    meta = f[["game_id", "game_date", "season", "home", "away"]]
    pgs = pgs.merge(meta, left_on="gid", right_on="game_id", how="inner")

    # team_id -> abbreviation, learned from box order (away listed first)
    votes = {}
    for gid, sub in pgs.groupby("gid", sort=False):
        tids = list(dict.fromkeys(sub.team_id))
        if len(tids) != 2:
            continue
        r = sub.iloc[0]
        for t, ab in zip(tids, [r.away, r.home]):
            votes.setdefault(t, {}).setdefault(ab, 0)
            votes[t][ab] += 1
    tid2ab = {t: max(d, key=d.get) for t, d in votes.items()}
    pgs["ab"] = pgs.team_id.map(tid2ab)
    pgs["min"] = pgs.seconds / 60.0
    pgs = pgs.dropna(subset=["ab"]).sort_values(["season", "ab", "game_date"])

    # PIT minutes: expanding mean over the team's PRIOR games only
    pgs["prior_mpg"] = (pgs.groupby(["season", "ab", "player_id"])["min"]
                        .transform(lambda s: s.shift(1).expanding().mean()))
    pgs["tg"] = (pgs.groupby(["season", "ab"])["game_date"]
                 .transform(lambda s: s.rank(method="dense").astype(int)))

    rows = []
    for (season, ab), g in pgs.groupby(["season", "ab"], sort=False):
        # roster-to-date = everyone who has played for this team this season
        seen = {}
        for gid, sub in g.groupby("gid", sort=False):
            tg = int(sub.tg.iloc[0])
            if tg > MIN_PRIOR and seen:
                rank = sorted(seen.items(), key=lambda kv: -kv[1])[:TOPK]
                dressed = set(sub.player_id)
                out = [(p, m) for p, m in rank if p not in dressed]
                rows.append(dict(game_id=gid, ab=ab, season=season,
                                 n_star_out=len(out),
                                 star_min_out=float(sum(m for _, m in out)),
                                 top_min=float(rank[0][1]) if rank else np.nan))
            for p, m in zip(sub.player_id, sub.prior_mpg):
                if np.isfinite(m):
                    seen[p] = m
    return pd.DataFrame(rows)


def main():
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
    f["game_id"] = zf(f["game_id"])
    f["game_date"] = pd.to_datetime(f["game_date"])
    f = f.dropna(subset=["open_margin", "close_margin", "margin_actual",
                         "p_us", "m_us_blind", "m_us"]).copy()

    cache = ROOT / "data" / "d252_stars.csv.gz"
    if cache.exists():
        st = pd.read_csv(cache); st["game_id"] = zf(st["game_id"])
    else:
        st = build_star_table(f)
        st.to_csv(cache, index=False, compression="gzip")
    print(f"star table: {len(st):,} team-games")

    piv = st.pivot_table(index="game_id", columns="ab", values="n_star_out",
                         aggfunc="first")
    d = f.copy()
    hs = st.merge(d[["game_id", "home"]], on="game_id")
    hs = hs[hs.ab == hs.home].set_index("game_id")
    as_ = st.merge(d[["game_id", "away"]], on="game_id")
    as_ = as_[as_.ab == as_.away].set_index("game_id")
    d = d.set_index("game_id")
    for lab, tb in (("h", hs), ("a", as_)):
        for c in ("n_star_out", "star_min_out"):
            d[f"{c}_{lab}"] = tb[c]
    d = d.reset_index().dropna(subset=["n_star_out_h", "n_star_out_a"])
    d["stars_out"] = d.n_star_out_h + d.n_star_out_a
    d["star_min_out"] = d.star_min_out_h + d.star_min_out_a
    print(f"{len(d):,} games with both sides' star status "
          f"({d.season.nunique()} seasons)")

    y = (d.margin_actual > 0).astype(float).to_numpy()
    p_mkt = 1 / (1 + np.exp(-d.close_margin.to_numpy(float) / SCALE))
    p_opn = 1 / (1 + np.exp(-d.open_margin.to_numpy(float) / SCALE))
    p_bl = 1 / (1 + np.exp(-d.m_us_blind.to_numpy(float) / SCALE))
    p_us = d.p_us.to_numpy(float)
    d["ll_mkt"], d["ll_open"] = nll(p_mkt, y), nll(p_opn, y)
    d["ll_blind"], d["ll_us"] = nll(p_bl, y), nll(p_us, y)
    d["d_shipped_vs_close"] = d.ll_us - d.ll_mkt
    d["d_shipped_vs_open"] = d.ll_us - d.ll_open
    d["d_blind_vs_open"] = d.ll_blind - d.ll_open

    # ---------------- Q1: STARS OUT -----------------------------------
    print("\n" + "=" * 78)
    print("Q1  DO WE DO BETTER WHEN STARS SIT?  (negative = we beat the market)")
    print("=" * 78)
    print(f"  {'stars out':12} {'n':>6} {'ll_mkt':>8} "
          f"{'shipped-close':>14} {'shipped-open':>13} {'blind-open':>11}")
    q1 = []
    for k, g in d.groupby(d.stars_out.clip(upper=3)):
        per = g.groupby("season")[["d_shipped_vs_close", "d_shipped_vs_open",
                                   "d_blind_vs_open"]].mean()
        m1, l1, h1, _ = clus(per.d_shipped_vs_close)
        m2, *_ = clus(per.d_shipped_vs_open)
        m3, *_ = clus(per.d_blind_vs_open)
        lab = f"{int(k)}{'+' if k == 3 else ''}"
        print(f"  {lab:12} {len(g):6} {g.ll_mkt.mean():8.4f} "
              f"{m1:+14.5f} {m2:+13.5f} {m3:+11.5f}")
        q1.append(dict(stars_out=lab, n=len(g), ll_mkt=float(g.ll_mkt.mean()),
                       shipped_close=float(m1), ci=[float(l1), float(h1)],
                       shipped_open=float(m2), blind_open=float(m3)))
    print("\n  Market log loss RISES with absences — the games really are harder.")
    print("  The question is whether OUR GAP moves, and by how much.")
    a = d[d.stars_out == 0]; b = d[d.stars_out >= 1]
    per = (d.assign(grp=(d.stars_out >= 1))
             .groupby(["season", "grp"]).d_shipped_vs_close.mean().unstack())
    if per.shape[1] == 2:
        diff = per[True] - per[False]
        m, lo, hi, k = clus(diff)
        print(f"\n  gap(>=1 star out) - gap(none): {m:+.5f} "
              f"CI [{lo:+.5f}, {hi:+.5f}] k={k}  "
              f"{'SIG' if (hi < 0 or lo > 0) else 'ns'}")
        print(f"  n: {len(a):,} clean vs {len(b):,} with an absence")

    # ---------------- Q2: THE PRE-REGISTERED FAMILY --------------------
    print("\n" + "=" * 78)
    print("Q2  WHERE IS THE EDGE?  Pre-registered slices, family-wide null")
    print("=" * 78)
    d["absopen"] = d.open_margin.abs()
    d["absedge"] = (d.m_us_blind - d.open_margin).abs()
    d["restd"] = 0.0
    d["days_in"] = (d.game_date - d.groupby("season")
                    .game_date.transform("min")).dt.days
    tot = pd.to_numeric(d.open_total, errors="coerce")
    d["totc"] = np.where((tot < 150) | (tot > 290), np.nan, tot)

    fam = {}
    fam["|open| band"] = pd.cut(d.absopen, [-.01, 2, 5, 8, 12, 99],
                                labels=["0-2", "2-5", "5-8", "8-12", "12+"])
    fam["stars out"] = d.stars_out.clip(upper=2).map(
        {0: "none", 1: "one", 2: "two+"})
    fam["|our edge| quintile"] = pd.qcut(d.absedge, 5, labels=[f"Q{i}" for i in
                                                               range(1, 6)],
                                         duplicates="drop")
    fam["total band"] = pd.qcut(d.totc, 4, labels=["low", "midlo", "midhi",
                                                   "high"], duplicates="drop")
    fam["month of season"] = pd.cut(d.days_in, [-1, 30, 60, 120, 999],
                                    labels=["m1", "m2", "m3-4", "m5+"])
    rng = np.random.default_rng(252)
    obs_max, results = 0.0, []
    for name, lab in fam.items():
        dd = d.assign(_g=lab).dropna(subset=["_g"])
        print(f"\n  --- {name} ---")
        for g, sub in dd.groupby("_g", observed=True):
            per = sub.groupby("season").d_shipped_vs_close.mean()
            m, lo, hi, k = clus(per)
            flag = "SIG" if (hi < 0 or lo > 0) else ""
            print(f"    {str(g):10} n={len(sub):6}  gap {m:+.5f} "
                  f"CI [{lo:+.5f}, {hi:+.5f}] k={k}  {flag}")
            obs_max = max(obs_max, abs(m))
            results.append(dict(family=name, level=str(g), n=len(sub),
                                gap=float(m), ci=[float(lo), float(hi)]))

    # family-wide permutation: shuffle slice labels within season
    print("\n  --- FAMILY-WIDE NULL: shuffle every slice label within season ---")
    vals = d.d_shipped_vs_close.to_numpy(float)
    seas = d.season.to_numpy()
    null = np.empty(2000)
    for i in range(2000):
        mx = 0.0
        for name, lab in fam.items():
            l = pd.Series(lab).to_numpy(object)
            perm = np.empty(len(l), object)
            for s in np.unique(seas):
                msk = seas == s
                perm[msk] = rng.permutation(l[msk])
            t = pd.DataFrame({"g": perm, "v": vals, "s": seas}).dropna()
            gm = t.groupby(["g", "s"], observed=True).v.mean().groupby("g").mean()
            if len(gm):
                mx = max(mx, float(gm.abs().max()))
        null[i] = mx
    p = float((null >= obs_max).mean())
    print(f"    observed max |gap| across the family: {obs_max:.5f}")
    print(f"    null max: median {np.median(null):.5f}, "
          f"95th {np.percentile(null,95):.5f}")
    print(f"    p = {p:.4f}  "
          f"{'A REAL POCKET' if p < 0.05 else 'NO SLICE BEATS THE FAMILY NULL'}")

    json.dump({"q1": q1, "q2": results, "family_p": p,
               "obs_max": float(obs_max)},
              open(ROOT / "data" / "d252_edge.json", "w"), default=float)
    print("\nwrote data/d252_edge.json")


if __name__ == "__main__":
    main()
