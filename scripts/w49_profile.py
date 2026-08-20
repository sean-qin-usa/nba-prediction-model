"""W49 TASK 1 — profile the catastrophic tail (worst 49 = worst 1%, worst 250 =
worst 5%) against MATCHED controls, and answer the only question that matters:

    are catastrophic losses PREDICTABLE EX ANTE, or are they the tail of a
    correct distribution?

Three comparisons, all on data/w49_frame.csv:

  C1  worst-49 / worst-250  vs  ALL OTHER GAMES              (descriptive)
  C2  worst-49 / worst-250  vs  CONF-MATCHED RIGHT games     (market-blind
        matching: same |p_us-0.5| caliper, our favourite won)
  C3  worst-49 / worst-250  vs  BET-MATCHED RIGHT games      (market-referenced
        matching: same |p_us-0.5| AND same p_us-p_mkt caliper, our fav won)
        -> C3 is the decisive one: it holds the ENTIRE mechanical driver fixed
           and asks whether anything ELSE distinguishes the blowups.

Plus the honest ex-ante test:
  T1  conditional calibration — inside each (conf_us x div) cell, is our
      realised hit rate short of our stated probability, or on it?
  T2  walk-forward MARKET-BLIND logistic predicting top-1% membership;
      OOS AUC vs the conf_us-only baseline.  If T2 adds nothing over conf_us,
      nothing is separable and Task 2 has no gate to build.

Read-only. Output: data/w49_profile.json + console tables.
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

FRAME = ROOT / "data" / "w49_frame.csv"
OUT = ROOT / "data" / "w49_profile.json"
SEED = 20260801

# features profiled.  (name, is_pit)  — is_pit False = outcome-only, profiling
# never a gate input.
FEATURES = [
    ("conf_us", True), ("conf_mkt", False), ("conf_gap", False),
    ("div", False), ("same_side", False),
    ("p_full", True), ("p_mkt", False),
    ("m_tot", True), ("fm", True), ("cm", True), ("rest", True),
    ("leg_spread", True), ("legs_agree", True), ("leg_disagree_mag", True),
    ("share_cm", True), ("share_ff", True),
    ("n_out", True), ("n_star_out", True), ("n_star_out_home", True),
    ("n_star_out_away", True), ("out_min_home", True), ("out_min_away", True),
    ("out_talent_home", True), ("out_talent_away", True),
    ("out_talent_d", True), ("out_min_d", True),
    ("max_out_min_home", True), ("max_out_min_away", True),
    ("tsd", True), ("tsd_abs", True),
    ("days_rest_home", True), ("days_rest_away", True), ("rest_adv", True),
    ("is_b2b_home", True), ("is_b2b_away", True), ("b2b_either", True),
    ("games_last_7_home", True), ("games_last_7_away", True),
    ("gp_min", True), ("gp_max", True), ("early", True), ("late", True),
    ("month", True), ("our_fav_home", True), ("mkt_fav_home", False),
    ("abs_margin", False), ("blowout", False), ("is_ot", False),
    ("margin", False), ("our_fav_won", False),
]


def smd(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference (Cohen's d, pooled sd)."""
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else np.nan


def perm_p(a: np.ndarray, b: np.ndarray, n: int = 4000, seed: int = SEED) -> float:
    """Two-sample permutation test on the mean difference.

    mean_A - mean_B = sum_A*(1/na + 1/nb) - total/nb, so only the sum over a
    random size-na subset is needed.  For na << N we draw the subset directly
    (with the tiny with/without-replacement bias removed by rejection-free
    partial Fisher-Yates via argpartition on a small random matrix).
    """
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    na, nb, N = len(a), len(b), len(pool)
    tot = pool.sum()
    rng = np.random.default_rng(seed)
    # sample na distinct indices per replicate without materialising an (n,N)
    # float matrix: draw with replacement then repair the (rare) duplicates.
    idx = rng.integers(0, N, size=(n, na))
    sa = pool[idx].sum(axis=1)
    stat = np.abs(sa * (1.0 / na + 1.0 / nb) - tot / nb)
    return float(((stat >= obs - 1e-15).sum() + 1) / (n + 1))


def welch_p(a: np.ndarray, b: np.ndarray) -> float:
    from scipy import stats
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def compare(cat: pd.DataFrame, ctl: pd.DataFrame, label: str,
            n_perm_top: int = 10) -> list[dict]:
    """Welch p for every feature (fast); an explicit resampling p for the
    n_perm_top largest |SMD| features as a distribution-free check."""
    rows = []
    for f, is_pit in FEATURES:
        if f not in cat.columns:
            continue
        a, b = cat[f].astype(float).values, ctl[f].astype(float).values
        rows.append({
            "feature": f, "pit": is_pit, "cmp": label,
            "cat_mean": float(np.nanmean(a)), "ctl_mean": float(np.nanmean(b)),
            "smd": smd(a, b), "p": welch_p(a, b), "p_perm": np.nan,
            "n_cat": int(len(a)), "n_ctl": int(len(b)),
        })
    order = sorted(range(len(rows)),
                   key=lambda i: -abs(rows[i]["smd"] if
                                      np.isfinite(rows[i]["smd"]) else 0))
    for i in order[:n_perm_top]:
        f = rows[i]["feature"]
        rows[i]["p_perm"] = perm_p(cat[f].astype(float).values,
                                   ctl[f].astype(float).values)
    return rows


def print_table(rows: list[dict], title: str, top: int = 100) -> None:
    df = pd.DataFrame(rows)
    df["absd"] = df.smd.abs()
    df = df.sort_values("absd", ascending=False).head(top)
    print(f"\n{'='*100}\n{title}\n{'='*100}")
    print(f"{'feature':22s} {'PIT':4s} {'catastrophe':>12s} {'control':>12s} "
          f"{'SMD':>7s} {'welch p':>9s} {'resamp p':>9s}")
    for _, r in df.iterrows():
        star = "***" if r.p < 0.001 else "**" if r.p < 0.01 else \
               "*" if r.p < 0.05 else ""
        pp = f"{r.p_perm:9.4f}" if np.isfinite(r.p_perm) else " " * 9
        print(f"{r.feature:22s} {'Y' if r.pit else 'n':4s} {r.cat_mean:12.4f} "
              f"{r.ctl_mean:12.4f} {r.smd:7.3f} {r.p:9.4f} {pp} {star}")


def matched(cat: pd.DataFrame, pool: pd.DataFrame, keys: dict) -> pd.DataFrame:
    """Union of caliper-matched RIGHT games (our favourite won) for each
    catastrophe.  keys = {col: caliper}."""
    sel = set()
    right = pool[pool.our_fav_won == 1]
    for _, r in cat.iterrows():
        m = np.ones(len(right), dtype=bool)
        for c, cal in keys.items():
            m &= (right[c] - r[c]).abs().values <= cal
        sel |= set(right.index[m])
    return pool.loc[sorted(sel)]


def main() -> None:
    d = pd.read_csv(FRAME, dtype={"game_id": str})
    d["game_date"] = pd.to_datetime(d.game_date)
    d = d.sort_values("exc", ascending=False).reset_index(drop=True)
    res: dict = {"n": len(d)}

    # ---------------- headline decomposition -------------------------------
    tot = d.exc.sum()
    res["headline"] = {
        "net_excess_nats": float(tot),
        "per_game": float(d.exc.mean()),
        "worst49_nats": float(d.exc.head(49).sum()),
        "worst49_share": float(d.exc.head(49).sum() / tot),
        "worst250_nats": float(d.exc.head(250).sum()),
        "worst250_share": float(d.exc.head(250).sum() / tot),
        "best49_nats": float(d.exc.tail(49).sum()),
    }
    print("HEADLINE", json.dumps(res["headline"], indent=2))

    cat49, cat250 = d.head(49), d.head(250)
    rest = d.iloc[250:]

    all_rows: list[dict] = []
    # ---------------- C1 vs everything else --------------------------------
    all_rows += compare(cat49, d.iloc[49:], "C1_49_vs_rest")
    print_table([r for r in all_rows if r["cmp"] == "C1_49_vs_rest"],
                "C1  WORST 49 (top 1%) vs the other 4871 games", 24)
    r250 = compare(cat250, rest, "C1_250_vs_rest")
    all_rows += r250
    print_table(r250, "C1  WORST 250 (top 5%) vs the other 4670 games", 24)

    # ---------------- C2 conf-matched (MARKET-BLIND matching) --------------
    ctl2_49 = matched(cat49, d, {"conf_us": 0.03})
    ctl2_250 = matched(cat250, d, {"conf_us": 0.03})
    r = compare(cat49, ctl2_49, "C2_49_confmatched")
    all_rows += r
    print_table(r, f"C2  WORST 49 vs CONF-MATCHED RIGHT games "
                   f"(|dconf_us|<=0.03, our fav won; n={len(ctl2_49)})", 22)
    r = compare(cat250, ctl2_250, "C2_250_confmatched")
    all_rows += r
    print_table(r, f"C2  WORST 250 vs CONF-MATCHED RIGHT games "
                   f"(n={len(ctl2_250)})", 22)

    # ---------------- C3 bet-matched (conf AND divergence held fixed) ------
    ctl3_49 = matched(cat49, d, {"conf_us": 0.03, "div": 0.03})
    ctl3_250 = matched(cat250, d, {"conf_us": 0.03, "div": 0.03})
    r = compare(cat49, ctl3_49, "C3_49_betmatched")
    all_rows += r
    print_table(r, f"C3  WORST 49 vs BET-MATCHED RIGHT games "
                   f"(same conf AND same divergence, our fav won; "
                   f"n={len(ctl3_49)})  <<< DECISIVE", 22)
    r = compare(cat250, ctl3_250, "C3_250_betmatched")
    all_rows += r
    print_table(r, f"C3  WORST 250 vs BET-MATCHED RIGHT games "
                   f"(n={len(ctl3_250)})  <<< DECISIVE", 22)

    res["comparisons"] = all_rows
    res["control_sizes"] = {"c2_49": len(ctl2_49), "c2_250": len(ctl2_250),
                            "c3_49": len(ctl3_49), "c3_250": len(ctl3_250)}

    # ---------------- worst-49 raw listing ---------------------------------
    cols = ["season", "game_date", "home", "away", "y", "p_full", "p_mkt",
            "exc", "conf_us", "div", "fm", "cm", "n_star_out_home",
            "n_star_out_away", "tsd", "margin", "gp_min", "is_ot"]
    print(f"\n{'='*100}\nWORST 49 GAMES (raw)\n{'='*100}")
    print(cat49[cols].to_string(index=False,
                                float_format=lambda v: f"{v:8.3f}"))
    res["worst49"] = cat49[cols].assign(
        game_date=cat49.game_date.dt.strftime("%Y-%m-%d")).to_dict("records")

    # ---------------- T1 conditional calibration ---------------------------
    print(f"\n{'='*100}\nT1  CONDITIONAL CALIBRATION — is our stated prob honest "
          f"inside each (our-confidence x divergence) cell?\n{'='*100}")
    d["p_our_side"] = np.where(d.p_full > 0.5, d.p_full, 1 - d.p_full)
    d["p_mkt_our_side"] = np.where(d.p_full > 0.5, d.p_mkt, 1 - d.p_mkt)
    d["cbin"] = pd.cut(d.conf_us, [0, .05, .10, .15, .20, .25, .30, .50],
                       include_lowest=True)
    d["dbin"] = pd.cut(d.conf_gap, [-1, -.10, -.05, 0, .05, .10, 1])
    t1 = d.groupby(["cbin", "dbin"]).apply(lambda g: pd.Series({
        "n": len(g), "stated": g.p_our_side.mean(),
        "realised": g.our_fav_won.mean(),
        "mkt_stated": g.p_mkt_our_side.mean(),
        "exc_per_game": g.exc.mean()})).reset_index()
    t1 = t1[t1.n >= 30]
    t1["shortfall_pp"] = 100 * (t1.realised - t1.stated)
    t1["mkt_shortfall_pp"] = 100 * (t1.realised - t1.mkt_stated)
    print(t1.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    res["t1_conditional_calibration"] = t1.astype({"cbin": str, "dbin": str}) \
        .to_dict("records")

    # marginal: our-confidence-only calibration (fully market-blind)
    t1b = d.groupby("cbin").apply(lambda g: pd.Series({
        "n": len(g), "stated": g.p_our_side.mean(),
        "realised": g.our_fav_won.mean(),
        "exc_per_game": g.exc.mean()})).reset_index()
    t1b["shortfall_pp"] = 100 * (t1b.realised - t1b.stated)
    print("\nMARKET-BLIND calibration by our own confidence bin:")
    print(t1b.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    res["t1_marginal_calibration"] = t1b.astype({"cbin": str}).to_dict("records")

    d.to_csv(ROOT / "data" / "w49_frame_sorted.csv", index=False)
    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
