#!/usr/bin/env python3
"""D253 — CAN ANY OBSERVABLE CONDITION PREDICT WHERE WE BEAT THE MARKET?

Sean's question: season length, every production and rejected feature, and any
other differentiator, swept together with a regularised regression rather than
one slice at a time.

WHY A REGULARISED FIT AND NOT MORE SLICES. D252 tested five slice families and
none beat a family-wide null (p 0.737). Slicing one variable at a time cannot
find a condition that only exists as a COMBINATION, and testing more slices
one-by-one just inflates the multiplicity. An elastic net over everything at
once is the right tool — and it is also the single most dangerous thing in this
repo, because D239 measured the capacity directly: best-of-N random subsets buy
+2.54 ROI points from nothing.

SO THE HEADLINE IS NOT A COEFFICIENT. It is OUT-OF-SAMPLE R^2, walk-forward by
season, scored against a PERMUTATION NULL OVER THE ENTIRE PROCEDURE — outcome
shuffled within season, the whole fit rerun end to end, including feature
standardisation, the alpha/l1 search and the fold structure. A coefficient that
looks large in-sample means nothing here; only OOS R^2 exceeding its own null
does.

THE OUTCOME IS A DIFFERENCE, NOT OUR ERROR. Per game:

    y_close = ll(offset margin) - ll(closing line)     negative = we win
    y_open  = ll(offset margin) - ll(opening line)

Predicting our own error would just rediscover that some games are less
predictable. Predicting the DIFFERENCE asks the actual question: are there
conditions under which our information is worth more than the market's?

FEATURES. Everything joinable and known before tip: season length and the
shortened-season flag, rest and back-to-backs, line level and total, our own
disagreement, days into season, star absences (D252's PIT top-3 measure),
report and inactive counts, expected outs, the four-factor and composition
component margins, and national TV. Features that could not be joined for all
19 seasons are listed in the output rather than silently dropped.

LEAKAGE. The closing line is an INPUT to y_close by construction, so it must
never be a feature; nor may anything derived from it (close_total, the
open-to-close move). Asserted, not assumed.
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
from sklearn.linear_model import ElasticNet                       # noqa: E402

SCALE = 6.96
ALPHAS = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4]
L1S = [0.1, 0.5, 0.9, 1.0]
BANNED = ("close_margin", "close_spread", "close_total", "close_prob",
          "margin_actual", "y", "m_us", "p_us", "m_us696", "edge696")


def zf(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def nll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def clus(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    k = len(v)
    se = v.std(ddof=1) / np.sqrt(k)
    tc = stats.t.ppf(0.975, k - 1)
    return v.mean(), v.mean() - tc * se, v.mean() + tc * se, k


def assemble():
    f = pd.read_csv(ROOT / "data" / "ats19_frame_offset.csv.gz")
    f["game_id"] = zf(f["game_id"]); f["game_date"] = pd.to_datetime(f.game_date)
    f = f.dropna(subset=["open_margin", "close_margin", "margin_actual",
                         "m_us", "m_us_blind"]).copy()
    missing = []

    pit = pd.read_csv(ROOT / "data" / "pit_frame.csv.gz")
    pit["game_id"] = zf(pit["game_id"])
    keep = ["game_id", "rest_home", "rest_away", "absence_tr_home",
            "absence_tr_away", "outs_report_home", "outs_report_away",
            "outs_inactive_home", "outs_inactive_away", "m_ff", "m_comp"]
    f = f.merge(pit[[c for c in keep if c in pit.columns]], on="game_id",
                how="left")

    cap = pd.read_csv(ROOT / "data" / "capstone_2019_26.csv")
    cap["game_id"] = zf(cap["game_id"])
    f = f.merge(cap[["game_id", "eo_home", "eo_away"]], on="game_id", how="left")

    st = pd.read_csv(ROOT / "data" / "d252_stars.csv.gz")
    st["game_id"] = zf(st["game_id"])
    for side in ("home", "away"):
        t = st.merge(f[["game_id", side]], on="game_id")
        t = t[t.ab == t[side]].set_index("game_id")
        f[f"stars_out_{side}"] = f.game_id.map(t.n_star_out)
        f[f"starmin_out_{side}"] = f.game_id.map(t.star_min_out)

    try:
        tv = pd.read_csv(ROOT / "data" / "ad_natl_tv.csv")
        tv["game_id"] = zf(tv["game_id"])
        f = f.merge(tv[["game_id", "is_natl_tv"]].drop_duplicates("game_id"),
                    on="game_id", how="left")
    except Exception as e:
        missing.append(f"national TV ({e})")

    # ---- derived / season-level ------------------------------------
    n_by = f.groupby("season").game_id.transform("size")
    f["season_len"] = n_by
    f["is_short_season"] = (n_by < 1200).astype(float)
    f["season_idx"] = f.season.rank(method="dense")
    f["days_in"] = (f.game_date - f.groupby("season")
                    .game_date.transform("min")).dt.days
    f["rest_h"] = f.rest_home.clip(upper=7).fillna(2)
    f["rest_a"] = f.rest_away.clip(upper=7).fillna(2)
    f["rest_diff"] = f.rest_h - f.rest_a
    f["b2b_home"] = (f.rest_h <= 0).astype(float)
    f["b2b_away"] = (f.rest_a <= 0).astype(float)
    f["open_abs"] = f.open_margin.abs()
    f["our_edge"] = f.m_us_blind - f.open_margin
    f["our_edge_abs"] = f.our_edge.abs()
    tot = pd.to_numeric(f.open_total, errors="coerce")
    f["total"] = np.where((tot < 150) | (tot > 290), np.nan, tot)
    f["total"] = f.total.fillna(f.total.median())
    f["stars_out"] = f.stars_out_home.fillna(0) + f.stars_out_away.fillna(0)
    f["stars_diff"] = f.stars_out_home.fillna(0) - f.stars_out_away.fillna(0)
    f["starmin_out"] = (f.starmin_out_home.fillna(0)
                        + f.starmin_out_away.fillna(0))
    f["eo_diff"] = (f.eo_home - f.eo_away).fillna(0.0)
    f["eo_tot"] = (f.eo_home.fillna(0) + f.eo_away.fillna(0))
    f["is_natl_tv"] = f.get("is_natl_tv", pd.Series(0, index=f.index)).fillna(0)
    # Tier-B features are NOT filled. Zero-filling an absence count outside the
    # injury-report era would teach the model an ERA, not an absence: a 0 would
    # mean "nobody out" in 2021 and "we have no idea" in 2010.
    for c in ("outs_report_home", "outs_report_away", "outs_inactive_home",
              "outs_inactive_away", "absence_tr_home", "absence_tr_away"):
        if c in f.columns:
            f[c] = pd.to_numeric(f[c], errors="coerce")
        else:
            missing.append(c)
    return f, missing


# Coverage measured, not assumed (see D253 output):
#   Tier A  19 seasons, 2007-08..2025-26
#   Tier B   7 seasons, 2019-20..2025-26  (injury report starts 2018-12-17)
#   Tier C   3 seasons, 2023-24..2025-26  (m_ff / m_comp) — too few folds for a
#            walk-forward fit, so it is reported as unavailable rather than run
TIER_A = ["season_len", "is_short_season", "season_idx", "days_in",
          "open_margin", "open_abs", "total", "rest_h", "rest_a", "rest_diff",
          "b2b_home", "b2b_away", "our_edge", "our_edge_abs",
          "stars_out", "stars_diff", "starmin_out", "is_natl_tv"]
TIER_B = ["outs_report_home", "outs_report_away", "outs_inactive_home",
          "outs_inactive_away", "absence_tr_home", "absence_tr_away",
          "eo_diff", "eo_tot"]
TIER_C = ["m_ff", "m_comp"]


def main():
    f, missing = assemble()
    y = (f.margin_actual > 0).astype(float).to_numpy()
    def ll(m):
        return nll(1 / (1 + np.exp(-np.asarray(m, float) / SCALE)), y)
    f["y_close"] = ll(f.m_us) - ll(f.close_margin)
    f["y_open"] = ll(f.m_us) - ll(f.open_margin)

    MODELS = [("A: 19 seasons, market+schedule+stars", TIER_A),
              ("B: 7 seasons, + injury report & expected outs",
               TIER_A + TIER_B)]
    print(f"Tier C {TIER_C} covers only 3 seasons — no walk-forward fold "
          f"structure exists, so it is NOT fitted.")
    if missing:
        print("  NOT joined (listed, not silently dropped):", missing)
    all_out = {}
    for mname, FEATS in MODELS:
        FEATS = [c for c in FEATS if c in f.columns]
        for b in BANNED:
            assert b not in FEATS, f"LEAKAGE: {b} is a feature"
        d = f.dropna(subset=FEATS + ["y_close", "y_open"]).copy()
        seasons = sorted(d.season.unique())
        print("\n" + "#" * 76)
        print(f"# MODEL {mname}")
        print(f"# {len(d):,} games, {len(FEATS)} features, "
              f"{len(seasons)} seasons ({seasons[0]}..{seasons[-1]})")
        print("#" * 76)
        all_out[mname] = fit_report(d, FEATS, seasons)
    json.dump(all_out, open(ROOT / "data" / "d253_differentiators.json", "w"),
              default=float)
    print("\nwrote data/d253_differentiators.json")


def fit_report(d, FEATS, seasons):
    MIN_TRAIN = 3
    assert len(seasons) > MIN_TRAIN + 1, (
        f"only {len(seasons)} seasons — fewer than the {MIN_TRAIN+2} needed "
        f"for a walk-forward fit; refusing to report a number")

    def run(target, yvec=None, seed=0):
        """(closure over d, FEATS, seasons)"""
        """Walk-forward elastic net. Returns OOS R^2 and per-fold coefs."""
        preds = np.full(len(d), np.nan)
        coefs = []
        yy = d[target].to_numpy(float) if yvec is None else yvec
        for i, s in enumerate(seasons):
            if i < 3:
                continue
            tr = d.season.isin(seasons[:i]).to_numpy()
            te = (d.season == s).to_numpy()
            Xtr, Xte = d.loc[tr, FEATS].to_numpy(float), d.loc[te, FEATS].to_numpy(float)
            mu, sd = Xtr.mean(0), Xtr.std(0)
            sd = np.where(sd < 1e-9, 1.0, sd)
            Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
            ytr = yy[tr]
            # inner split: last training season is the validation block
            inner = d.loc[tr, "season"].to_numpy() == seasons[i - 1]
            best, bpair = np.inf, (ALPHAS[0], L1S[0])
            for a in ALPHAS:
                for l1 in L1S:
                    m = ElasticNet(alpha=a, l1_ratio=l1, max_iter=5000)
                    m.fit(Ztr[~inner], ytr[~inner])
                    e = ((m.predict(Ztr[inner]) - ytr[inner]) ** 2).mean()
                    if e < best:
                        best, bpair = e, (a, l1)
            m = ElasticNet(alpha=bpair[0], l1_ratio=bpair[1], max_iter=5000)
            m.fit(Ztr, ytr)
            preds[te] = m.predict(Zte)
            coefs.append(dict(season=s, alpha=bpair[0], l1=bpair[1],
                              **dict(zip(FEATS, m.coef_))))
        ok = np.isfinite(preds)
        resid = yy[ok] - preds[ok]
        base = yy[ok] - yy[ok].mean()
        r2 = 1 - (resid ** 2).sum() / (base ** 2).sum()
        return float(r2), pd.DataFrame(coefs), ok, preds

    out = {}
    for target in ("y_close", "y_open"):
        print("\n" + "=" * 76)
        print(f"TARGET {target}   (negative outcome = we beat that price)")
        print("=" * 76)
        r2, cf, ok, preds = run(target)
        print(f"  walk-forward OOS R^2 = {r2:+.5f}  on {int(ok.sum()):,} games, "
              f"{len(cf)} folds")
        nz = (cf[FEATS].abs() > 1e-10).mean().sort_values(ascending=False)
        print("\n  features selected in the most folds:")
        for c, v in nz.head(8).items():
            mn = cf[c].mean()
            print(f"    {c:22} in {100*v:5.1f}% of folds   mean coef {mn:+.6f}")
        # permutation null over the WHOLE procedure
        rng = np.random.default_rng(253)
        scode = pd.factorize(d.season)[0]
        idx = [np.flatnonzero(scode == i) for i in range(scode.max() + 1)]
        base = d[target].to_numpy(float)
        null = []
        for it in range(40):
            perm = base.copy()
            for ix in idx:
                perm[ix] = rng.permutation(base[ix])
            null.append(run(target, yvec=perm)[0])
        null = np.array(null)
        p = float((null >= r2).mean())
        print(f"\n  permutation null over the full procedure (40 draws, "
              f"outcome shuffled within season):")
        print(f"    null OOS R^2: median {np.median(null):+.5f}, "
              f"95th {np.percentile(null,95):+.5f}, max {null.max():+.5f}")
        print(f"    p = {p:.3f}   "
              f"{'PREDICTABLE' if p < 0.05 else 'NOT PREDICTABLE — no combination of these conditions forecasts our edge'}")
        out[target] = dict(r2=r2, p=p, null_median=float(np.median(null)),
                           null_p95=float(np.percentile(null, 95)),
                           top=list(nz.head(8).index))
    return out


if __name__ == "__main__":
    main()
