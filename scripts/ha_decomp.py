"""HA-(2) MECHANISM DECOMPOSITION of the league-wide home edge.

Additive decomposition, in POINTS with bootstrap CIs:

    raw home edge = travel/rest asymmetry
                  + crowd            (from D131's 2020-21 no-crowd experiment)
                  + residual         (familiarity / venue / sleep / unexplained)

Method. One pooled regression over all non-neutral regular-season games with
SEASON-SPECIFIC team strength fixed effects (so opponent and own quality are
controlled within season), a crowd-regime-specific home intercept, and the
schedule state of BOTH sides:

    margin = sum_{t,y} s_{t,y} * (1{home=t} - 1{away=t})
           + sum_r HFA_r * 1{stratum=r}
           + b_hb2b*h_b2b + b_ab2b*a_b2b + b_h34*h_3in4 + b_a34*a_3in4
           + b_rest*rest_diff + b_ht*h_travel_k + b_at*a_travel_k
           + b_htz*|h_tz| + b_atz*|a_tz|

HFA_r is then the home edge at ZERO schedule asymmetry ("pure" HFA), and the
travel/rest contribution is sum_j b_j * E[X_j] evaluated at the realised
schedule.  crowd = HFA_pure(normal) - HFA_pure(nocrowd).
ALTITUDE is handled separately: it is REDISTRIBUTIVE (every team hosts and
visits a balanced slate, so E[altitude gain] ~ 0) and therefore cannot explain
any of the LEAGUE-WIDE edge; only its DEN/UTA-specific size is reported.

DESCRIPTIVE, FULL-SAMPLE.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ha_core import boot_ci, load_panel

SEED = 20260801
OUT = Path("/tmp/claude-1004/-hdd-steveqin-sean-dev/"
           "4912ce58-69cd-488d-a5f4-0d3ed2eed30c/scratchpad")
SCHED = ["h_b2b", "a_b2b", "h_3in4", "a_3in4", "rest_diff",
         "h_travel_k", "a_travel_k", "h_tz_abs", "a_tz_abs"]


def build(df, strata_col="stratum"):
    teams = sorted(set(df["home"]) | set(df["away"]))
    seasons = sorted(df["season"].unique())
    ts = [(t, y) for y in seasons for t in teams]
    tsi = {k: i for i, k in enumerate(ts)}
    n = len(df)
    Z = np.zeros((n, len(ts)))
    rows = np.arange(n)
    hz = np.array([tsi[(h, y)] for h, y in zip(df["home"], df["season"])])
    az = np.array([tsi[(a, y)] for a, y in zip(df["away"], df["season"])])
    Z[rows, hz] += 1.0
    Z[rows, az] -= 1.0
    strata = sorted(df[strata_col].unique())
    S = np.zeros((n, len(strata)))
    si = {s: i for i, s in enumerate(strata)}
    S[rows, df[strata_col].map(si).to_numpy()] = 1.0
    C = df[SCHED].to_numpy(float)
    X = np.hstack([Z, S, C])
    return X, len(ts), strata


def fit(df, strata_col="stratum"):
    X, nts, strata = build(df, strata_col)
    y = df["margin"].to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    hfa = dict(zip(strata, coef[nts:nts + len(strata)]))
    b = dict(zip(SCHED, coef[nts + len(strata):]))
    return hfa, b, coef, X, y


def contributions(df, b):
    """points of the raw home edge attributable to each schedule term."""
    return {k: float(b[k] * df[k].mean()) for k in SCHED}


def main():
    rng = np.random.default_rng(SEED)
    d = load_panel()
    print(f"panel: {len(d)} non-neutral regular-season games")
    print(d.groupby("stratum").agg(n=("margin", "size"),
                                   raw=("margin", "mean")).round(4).to_string())

    # ---------- schedule asymmetry, raw ------------------------------------
    print("\n=== (2a) THE SCHEDULE ASYMMETRY THE VISITOR ACTUALLY FACES ===")
    asym = pd.DataFrame({
        "home": [d.h_b2b.mean(), d.h_3in4.mean(), d.h_rest.mean(),
                 d.h_travel.mean(), d.h_tz_abs.mean(), d.h_road_streak.mean()],
        "away": [d.a_b2b.mean(), d.a_3in4.mean(), d.a_rest.mean(),
                 d.a_travel.mean(), d.a_tz_abs.mean(), d.a_road_streak.mean()],
    }, index=["b2b rate", "3-in-4 rate", "days rest", "travel km",
              "|tz shift|", "prior consecutive road g"])
    asym["away-home"] = asym["away"] - asym["home"]
    print(asym.round(4).to_string())

    # per stratum (does 2020-21's compressed/series schedule differ?)
    print("\n  by stratum:")
    g = d.groupby("stratum").agg(
        h_b2b=("h_b2b", "mean"), a_b2b=("a_b2b", "mean"),
        h_trav=("h_travel", "mean"), a_trav=("a_travel", "mean"),
        h_tz=("h_tz_abs", "mean"), a_tz=("a_tz_abs", "mean"),
        rest_diff=("rest_diff", "mean"))
    print(g.round(3).to_string())

    # ---------- pooled fit -------------------------------------------------
    hfa, b, coef, X, y = fit(d)
    contrib = contributions(d, b)
    print("\n=== (2b) POOLED FIT (season-team FE, all 7 seasons) ===")
    print("  schedule coefficients (points):")
    for k in SCHED:
        print(f"    {k:12s} {b[k]:+8.4f}   mean(X)={d[k].mean():+8.4f}"
              f"   contribution={contrib[k]:+8.4f}")
    tot_sched = sum(contrib.values())
    print(f"    TOTAL schedule contribution to the raw home edge: {tot_sched:+.4f} pts")
    print("  pure HFA by crowd stratum (at zero schedule asymmetry):")
    for k, v in hfa.items():
        print(f"    {k:9s} {v:+.4f}")

    raw_norm = d[d.stratum == "normal"].margin.mean()
    raw_nc = d[d.stratum == "nocrowd"].margin.mean()
    crowd = hfa["normal"] - hfa["nocrowd"]
    print(f"\n  raw home edge normal={raw_norm:+.4f}  nocrowd={raw_nc:+.4f}"
          f"  raw diff={raw_norm-raw_nc:+.4f}")
    print(f"  CONTROLLED crowd effect (pure HFA normal - nocrowd) = {crowd:+.4f}")

    # ---------- bootstrap --------------------------------------------------
    B = 1500
    keys = ["crowd", "sched_total", "hfa_normal", "hfa_nocrowd", "hfa_pre",
            "residual"] + SCHED
    boots = {k: [] for k in keys}
    idx_all = np.arange(len(d))
    for _ in range(B):
        ix = rng.integers(0, len(d), len(d))
        sub = d.iloc[ix]
        try:
            h2, b2, *_ = fit(sub)
        except Exception:
            continue
        c2 = contributions(sub, b2)
        boots["crowd"].append(h2["normal"] - h2["nocrowd"])
        boots["sched_total"].append(sum(c2.values()))
        boots["hfa_normal"].append(h2["normal"])
        boots["hfa_nocrowd"].append(h2["nocrowd"])
        boots["hfa_pre"].append(h2.get("pre", np.nan))
        boots["residual"].append(h2["nocrowd"])
        for k in SCHED:
            boots[k].append(c2[k])
    ci = {k: boot_ci(np.array(v)) for k, v in boots.items() if len(v)}

    print("\n=== (2c) DECOMPOSITION TABLE (points of the ~1.96 pt home edge) ===")
    rows = [
        ("RAW home edge, normal seasons", raw_norm, None),
        ("  travel/rest asymmetry", tot_sched, ci["sched_total"]),
        ("  crowd (2020-21 natural experiment)", crowd, ci["crowd"]),
        ("  residual (no crowd, no schedule asym)", hfa["nocrowd"], ci["hfa_nocrowd"]),
        ("  [check] sum of the three", tot_sched + crowd + hfa["nocrowd"], None),
    ]
    for nm, v, c in rows:
        cs = f"  CI({c[0]:+.3f},{c[1]:+.3f})" if c else ""
        print(f"  {nm:42s} {v:+7.3f}{cs}")

    # ---------- altitude ---------------------------------------------------
    print("\n=== (2d) ALTITUDE ===")
    print(f"  E[alt_gain] over all home games = {d.alt_gain.mean():.2f} m "
          f"-> altitude is REDISTRIBUTIVE, it cannot produce league-wide edge")
    # DEN/UTA home-specific extra, controlled
    d2 = d.copy()
    d2["denuta"] = d2["home"].isin(["DEN", "UTA"]).astype(float)
    d2["den"] = (d2["home"] == "DEN").astype(float)
    d2["uta"] = (d2["home"] == "UTA").astype(float)
    for label, cols in (("DEN+UTA pooled", ["denuta"]), ("DEN / UTA separate", ["den", "uta"])):
        X0, nts, strata = build(d2)
        E = d2[cols].to_numpy(float)
        Xe = np.hstack([X0, E])
        cf, *_ = np.linalg.lstsq(Xe, d2["margin"].to_numpy(float), rcond=None)
        est = cf[-len(cols):]
        bs = []
        for _ in range(600):
            ix = rng.integers(0, len(d2), len(d2))
            s = d2.iloc[ix]
            X0b, _, _ = build(s)
            Xb = np.hstack([X0b, s[cols].to_numpy(float)])
            cb, *_ = np.linalg.lstsq(Xb, s["margin"].to_numpy(float), rcond=None)
            bs.append(cb[-len(cols):])
        bs = np.array(bs)
        for i, c in enumerate(cols):
            lo, hi = boot_ci(bs[:, i])
            print(f"  {label:20s} {c:8s} {est[i]:+7.3f} CI({lo:+.3f},{hi:+.3f})"
                  f"  {'SIG' if lo>0 or hi<0 else 'NS'}")
    # continuous altitude gain
    d2["altk"] = d2["alt_gain"] / 1000.0
    X0, nts, strata = build(d2)
    Xe = np.hstack([X0, d2[["altk"]].to_numpy(float)])
    cf, *_ = np.linalg.lstsq(Xe, d2["margin"].to_numpy(float), rcond=None)
    bs = []
    for _ in range(600):
        ix = rng.integers(0, len(d2), len(d2))
        s = d2.iloc[ix]
        X0b, _, _ = build(s)
        Xb = np.hstack([X0b, s[["altk"]].to_numpy(float)])
        cb, *_ = np.linalg.lstsq(Xb, s["margin"].to_numpy(float), rcond=None)
        bs.append(cb[-1])
    lo, hi = boot_ci(np.array(bs))
    print(f"  continuous: points per +1000 m of altitude GAIN for the host: "
          f"{cf[-1]:+.3f} CI({lo:+.3f},{hi:+.3f}) {'SIG' if lo>0 or hi<0 else 'NS'}")

    # ---------- 2020-21 dose-response (limited fans returned late) ---------
    print("\n=== (2e) WITHIN-2020-21 DOSE RESPONSE (fans partially returned "
          "from ~March 2021) ===")
    d3 = d.copy()
    d3["half"] = np.where(pd.to_datetime(d3["game_date"]).dt.dayofyear.values * 0 +
                          (d3["days_into"] > d3.groupby("season")["days_into"]
                           .transform("median")), "late", "early")
    d3["cell"] = d3["stratum"] + "_" + d3["half"]
    h4, b4, *_ = fit(d3, "cell")
    for k in sorted(h4):
        print(f"    {k:16s} pure HFA {h4[k]:+.4f}")
    did = ((h4["nocrowd_late"] - h4["nocrowd_early"]) -
           (h4["normal_late"] - h4["normal_early"]))
    print(f"  DiD (2020-21 late-minus-early) - (normal late-minus-early) = {did:+.4f}")
    print("   positive => home edge recovered as fans came back, net of the "
          "ordinary within-season profile")

    res = dict(raw_normal=float(raw_norm), raw_nocrowd=float(raw_nc),
               hfa_pure=hfa, sched_coef=b, sched_contrib=contrib,
               sched_total=float(tot_sched), crowd=float(crowd), ci=ci,
               did_2021=float(did), asym=asym.to_dict())
    (OUT / "ha_decomp.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {OUT/'ha_decomp.json'}")


if __name__ == "__main__":
    main()
