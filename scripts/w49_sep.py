"""W49 TASK 1b — the SEPARABILITY VERDICT.

Two things the first pass could not settle:

A. BALANCED matched controls.  w49_profile.py built control sets as the UNION
   of caliper matches, which does not balance the matching variables (conf_mkt
   SMD -1.40 in C3).  Here every catastrophe i gets weight 1 spread evenly over
   its caliper matches, so the control distribution is re-weighted TO the
   catastrophe distribution on (conf_us, div).  Weighted SMDs + a balance check.

B. THE ONLY QUESTION A GATE CAN BE BUILT ON: does any MARKET-BLIND PIT signal
   predict that OUR OWN stated probability is too confident?
   For each candidate signal S the test is a market-blind reliability split:
   inside our own confidence bins, is (realised hit rate - stated prob) worse
   when S fires than when it does not?  A variance-inflation gate can only pay
   if the answer is yes.  Reported with a paired bootstrap on the log-loss a
   *hindsight-optimal* scale inflation would have saved on S=1 — the CEILING
   any such gate could reach.  If the ceiling is ~0, Task 2 is dead on arrival.

C. Walk-forward market-blind classifier for top-1% membership (OOS AUC) vs a
   conf_us-only baseline: are catastrophes identifiable ex ante at all?

Read-only inputs.  Output: data/w49_sep.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import optimize  # noqa: E402

FRAME = ROOT / "data" / "w49_frame.csv"
OUT = ROOT / "data" / "w49_sep.json"
SEED = 20260801
SCALE = 7.2
N_BOOT = 2000

MATCH_FEATURES = [
    "conf_us", "conf_mkt", "conf_gap", "div", "same_side",
    "leg_spread", "legs_agree", "share_cm", "share_ff", "fm", "cm", "m_tot",
    "n_star_out", "n_star_out_home", "n_star_out_away", "out_talent_home",
    "out_talent_away", "out_min_home", "out_min_away", "out_min_d",
    "tsd_abs", "tsd", "gp_min", "early", "late", "month",
    "days_rest_home", "days_rest_away", "b2b_either", "rest",
    "abs_margin", "blowout", "is_ot", "margin", "our_fav_won",
]

# ---- MARKET-BLIND candidate signals (every one computable at tip) ---------
def signals(d: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "leg_spread>=p75": d.leg_spread >= d.leg_spread.quantile(.75),
        "leg_spread>=p90": d.leg_spread >= d.leg_spread.quantile(.90),
        "legs_disagree_sign": d.legs_agree == 0,
        "ff_dominant(share_ff>=p75)": d.share_ff >= d.share_ff.quantile(.75),
        "comp_dominant(share_cm>=p75)": d.share_cm >= d.share_cm.quantile(.75),
        "n_star_out>=3": d.n_star_out >= 3,
        "n_star_out>=4": d.n_star_out >= 4,
        "star_out_asym>=2": (d.n_star_out_home - d.n_star_out_away).abs() >= 2,
        "out_talent_load>=p75": (d.out_talent_home.abs() + d.out_talent_away.abs())
                                >= (d.out_talent_home.abs()
                                    + d.out_talent_away.abs()).quantile(.75),
        "early(gp<20)": d.early == 1,
        "week1(gp<5)": d.gp_min < 5,
        "late(gp>=55)": d.late == 1,
        "april": d.month == 4,
        "tsd_abs>=0.5": d.tsd_abs >= 0.5,
        "b2b_either": d.b2b_either == 1,
        "rest_leg_big(|rest|>=p90)": d.rest.abs() >= d.rest.abs().quantile(.90),
        "conf_us>=p90": d.conf_us >= d.conf_us.quantile(.90),
    }


def wsmd(a, b, wb):
    """SMD of catastrophe mean vs WEIGHTED control mean."""
    a = np.asarray(a, float)
    b, wb = np.asarray(b, float), np.asarray(wb, float)
    ok = ~np.isnan(b)
    b, wb = b[ok], wb[ok]
    a = a[~np.isnan(a)]
    if len(a) < 2 or len(b) < 2 or wb.sum() == 0:
        return np.nan, np.nan, np.nan
    mb = np.average(b, weights=wb)
    vb = np.average((b - mb) ** 2, weights=wb)
    s = np.sqrt((a.var(ddof=1) + vb) / 2)
    return float(a.mean()), float(mb), (float((a.mean() - mb) / s) if s > 0
                                        else np.nan)


def balanced_match(cat: pd.DataFrame, pool: pd.DataFrame, keys: dict):
    """Weight-1-per-catastrophe caliper matching against RIGHT games."""
    right = pool[pool.our_fav_won == 1]
    w = np.zeros(len(right))
    unmatched = 0
    for _, r in cat.iterrows():
        m = np.ones(len(right), dtype=bool)
        for c, cal in keys.items():
            m &= (right[c] - r[c]).abs().values <= cal
        k = m.sum()
        if k == 0:
            unmatched += 1
            continue
        w[m] += 1.0 / k
    return right, w, unmatched


def matched_table(cat, pool, keys, label, res):
    right, w, unm = balanced_match(cat, pool, keys)
    keep = w > 0
    print(f"\n{'='*104}")
    print(f"{label}   n_cat={len(cat)}  unmatched={unm}  "
          f"distinct controls used={int(keep.sum())}  "
          f"caliper={keys}")
    print(f"{'='*104}")
    print(f"{'feature':24s} {'catastrophe':>12s} {'wctl':>12s} {'wSMD':>8s}  "
          f"note")
    rows = []
    for f in MATCH_FEATURES:
        ma, mb, s = wsmd(cat[f].values, right[f].values, w)
        note = "MATCHED-ON" if f in keys else ""
        rows.append({"cmp": label, "feature": f, "cat": ma, "ctl": mb,
                     "wsmd": s, "matched_on": f in keys})
        print(f"{f:24s} {ma:12.4f} {mb:12.4f} {s:8.3f}  {note}")
    res.setdefault("matched", []).extend(rows)
    return rows


# ------------------------------------------------------------------- B
def reliability_split(d: pd.DataFrame, res: dict) -> None:
    """MARKET-BLIND: does our stated prob get worse when signal S fires?"""
    print(f"\n{'='*126}")
    print("B. MARKET-BLIND OVERCONFIDENCE SCREEN — inside our own confidence, "
          "is our stated probability too high when S fires?")
    print("   ceiling = log-loss/game a HINDSIGHT-OPTIMAL scale inflation on "
          "S=1 alone would have saved (upper bound on any gate).")
    print(f"{'='*126}")
    p_side = np.where(d.p_full > 0.5, d.p_full, 1 - d.p_full)
    hit = d.our_fav_won.values.astype(float)
    m = d.m_tot.values
    y = d.y.values.astype(float)
    rng = np.random.default_rng(SEED)

    def ll(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    base_ll = ll(d.p_full.values)
    print(f"{'signal':30s} {'n':>5} {'stated':>8} {'realis':>8} "
          f"{'shortfall pp':>13} {'excess/gm':>10} {'opt k':>7} "
          f"{'CEILING/gm':>11} {'ceiling 95% CI':>22}")
    rows = []
    for name, s in signals(d).items():
        s = s.values.astype(bool)
        if s.sum() < 60:
            continue
        # hindsight-optimal scale multiplier on the S=1 subset only
        def obj(k, mask=s):
            p = 1.0 / (1.0 + np.exp(-m[mask] / (SCALE * k)))
            p = np.clip(p, 1e-12, 1 - 1e-12)
            yy = y[mask]
            return float(-(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean())
        r = optimize.minimize_scalar(obj, bounds=(0.5, 4.0), method="bounded")
        k = float(r.x)
        p_new = 1.0 / (1.0 + np.exp(-m / (SCALE * k)))
        gain = np.zeros(len(d))
        gain[s] = base_ll[s] - ll(p_new)[s]          # >0 = inflation helps
        idx = rng.integers(0, len(gain), (N_BOOT, len(gain)))
        lo, hi = np.percentile(gain[idx].mean(axis=1), [2.5, 97.5])
        short = 100 * (hit[s].mean() - p_side[s].mean())
        rows.append({"signal": name, "n": int(s.sum()),
                     "stated": float(p_side[s].mean()),
                     "realised": float(hit[s].mean()),
                     "shortfall_pp": short,
                     "exc_per_game": float(d.exc.values[s].mean()),
                     "opt_k": k, "ceiling": float(gain.mean()),
                     "lo": float(lo), "hi": float(hi)})
        print(f"{name:30s} {s.sum():>5} {p_side[s].mean():>8.4f} "
              f"{hit[s].mean():>8.4f} {short:>+13.2f} "
              f"{d.exc.values[s].mean():>+10.4f} {k:>7.3f} "
              f"{gain.mean():>+11.5f}  [{lo:+.5f},{hi:+.5f}]")
    res["overconfidence_screen"] = rows

    # global reference: hindsight-optimal GLOBAL scale (the D74 vein)
    def obj_g(k):
        p = 1.0 / (1.0 + np.exp(-m / (SCALE * k)))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    rg = optimize.minimize_scalar(obj_g, bounds=(0.5, 4.0), method="bounded")
    p_g = 1.0 / (1.0 + np.exp(-m / (SCALE * rg.x)))
    print(f"\n  GLOBAL hindsight-optimal scale multiplier k = {rg.x:.4f} "
          f"(SCALE {SCALE} -> {SCALE*rg.x:.3f}); hindsight gain "
          f"{(base_ll - ll(p_g)).mean():+.5f}/game — the D74 vein, for "
          f"reference.")
    res["global_opt_k"] = float(rg.x)
    res["global_ceiling"] = float((base_ll - ll(p_g)).mean())


# ------------------------------------------------------------------- C
def wf_classifier(d: pd.DataFrame, res: dict) -> None:
    """Walk-forward MARKET-BLIND logistic for top-1% membership; OOS AUC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    print(f"\n{'='*104}")
    print("C. EX-ANTE IDENTIFIABILITY — walk-forward MARKET-BLIND logistic "
          "for top-1% membership")
    print(f"{'='*104}")
    d = d.sort_values("game_date").reset_index(drop=True)
    thr = d.exc.quantile(0.99)
    d["cat"] = (d.exc >= thr).astype(int)
    feats = ["conf_us", "leg_spread", "legs_agree", "share_cm", "share_ff",
             "n_star_out", "n_star_out_home", "n_star_out_away",
             "out_talent_home", "out_talent_away", "out_min_d", "tsd_abs",
             "gp_min", "early", "late", "b2b_either", "rest",
             "days_rest_home", "days_rest_away", "month"]
    X = d[feats].astype(float).fillna(0.0).values
    Xb = d[["conf_us"]].astype(float).values
    ytr = d.cat.values
    seasons = sorted(d.season.unique())
    out = []
    for i, ssn in enumerate(seasons):
        if i == 0:
            continue                      # first season = burn-in
        tr = d.season.isin(seasons[:i]).values
        te = (d.season == ssn).values
        if ytr[tr].sum() < 5 or ytr[te].sum() < 3:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        full = LogisticRegression(max_iter=2000, C=0.5).fit(
            (X[tr] - mu) / sd, ytr[tr])
        base = LogisticRegression(max_iter=2000, C=0.5).fit(
            Xb[tr], ytr[tr])
        a_full = roc_auc_score(ytr[te], full.predict_proba((X[te] - mu) / sd)[:, 1])
        a_base = roc_auc_score(ytr[te], base.predict_proba(Xb[te])[:, 1])
        print(f"  OOS {ssn}: n={te.sum():4d} catastrophes={int(ytr[te].sum()):3d}"
              f"  AUC full(20 PIT feats)={a_full:.3f}   "
              f"AUC conf_us-only={a_base:.3f}   delta={a_full-a_base:+.3f}")
        out.append({"season": ssn, "n": int(te.sum()),
                    "n_cat": int(ytr[te].sum()), "auc_full": float(a_full),
                    "auc_base": float(a_base)})
    if out:
        mf = np.mean([o["auc_full"] for o in out])
        mb = np.mean([o["auc_base"] for o in out])
        print(f"  MEAN OOS AUC  full={mf:.3f}  conf_us-only={mb:.3f}  "
              f"delta={mf-mb:+.3f}")
        res["wf_classifier"] = {"per_season": out, "mean_full": float(mf),
                                "mean_base": float(mb)}


def main() -> None:
    d = pd.read_csv(FRAME, dtype={"game_id": str})
    d["game_date"] = pd.to_datetime(d.game_date)
    d = d.sort_values("exc", ascending=False).reset_index(drop=True)
    res: dict = {}

    cat49, cat250 = d.head(49), d.head(250)
    matched_table(cat49, d, {"conf_us": 0.03, "div": 0.03},
                  "A1  WORST 49 vs BALANCED bet-matched RIGHT games", res)
    matched_table(cat250, d, {"conf_us": 0.03, "div": 0.03},
                  "A2  WORST 250 vs BALANCED bet-matched RIGHT games", res)
    matched_table(cat250, d, {"conf_us": 0.03},
                  "A3  WORST 250 vs BALANCED conf-matched RIGHT games "
                  "(market-blind matching)", res)

    reliability_split(d, res)
    wf_classifier(d, res)

    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
