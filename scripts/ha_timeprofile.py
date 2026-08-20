"""HA-(5) WITHIN-SEASON TIME PROFILE of the league home edge.

Sean's hypothesis: home advantage "typically diminishes over the season" and
"should be extra strong on opening night and then weaken".

The confound this test lives or dies on: B2B rate, travel load and the rest
distribution all move with the calendar. Without controlling them you measure
the SCHEDULE, not the crowd. So every phase estimate below comes from

    margin = sum_{t,y} s_{t,y}*(1{home=t}-1{away=t})     # season team FE
           + sum_p HFA_p * 1{phase=p}                    # phase-specific HFA
           + schedule controls (b2b, 3in4, rest diff, travel, tz, both sides)

and is reported alongside the UNCONTROLLED raw mean so the schedule share of
any profile is visible.

Mechanism reading: a crowd/novelty effect should be strongest EARLY and fade;
a travel/fatigue effect should ACCUMULATE and be strongest LATE. Cross-checked
against the 2020-21 no-crowd stratum.

DESCRIPTIVE, FULL-SAMPLE. Nothing here ships.
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
OUT = Path("data/scratch")
SCHED = ["h_b2b", "a_b2b", "h_3in4", "a_3in4", "rest_diff",
         "h_travel_k", "a_travel_k", "h_tz_abs", "a_tz_abs"]
NORMAL = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def fit_phase(sub: pd.DataFrame, phase: pd.Series, controls=True, half_fe=False):
    teams = sorted(set(sub.home) | set(sub.away))
    if half_fe:   # allow team strength to drift at midseason
        key = sub.season + np.where(sub.mean_gp < 41, "_A", "_B")
    else:
        key = sub.season
    seas = sorted(set(key))
    ts = [(t, y) for y in seas for t in teams]
    tsi = {k: i for i, k in enumerate(ts)}
    n = len(sub)
    r = np.arange(n)
    Z = np.zeros((n, len(ts)))
    Z[r, [tsi[(h, y)] for h, y in zip(sub.home, key)]] += 1
    Z[r, [tsi[(a, y)] for a, y in zip(sub.away, key)]] -= 1
    ph = list(pd.unique(phase))
    pi = {p: i for i, p in enumerate(ph)}
    P = np.zeros((n, len(ph)))
    P[r, phase.map(pi).to_numpy()] = 1
    blocks = [Z, P]
    if controls:
        blocks.append(sub[SCHED].to_numpy(float))
    X = np.hstack(blocks)
    y = sub.margin.to_numpy(float)
    cf, *_ = np.linalg.lstsq(X, y, rcond=None)
    return pd.Series(cf[len(ts):len(ts) + len(ph)], index=ph), X, y, len(ts), len(ph)


def phase_with_ci(sub, phase, B=1200, rng=None, controls=True, half_fe=False):
    est, *_ = fit_phase(sub, phase, controls, half_fe)
    boots = {p: [] for p in est.index}
    sub = sub.reset_index(drop=True)
    phase = phase.reset_index(drop=True)
    for _ in range(B):
        ix = rng.integers(0, len(sub), len(sub))
        try:
            e, *_ = fit_phase(sub.iloc[ix], phase.iloc[ix], controls, half_fe)
        except Exception:
            continue
        for p in est.index:
            if p in e.index:
                boots[p].append(e[p])
    ci = {p: boot_ci(np.array(v)) for p, v in boots.items() if len(v) > 50}
    return est, ci


def report(name, sub, phase, order, rng, B=1000, half_fe=False):
    est, ci = phase_with_ci(sub, phase, B, rng, True, half_fe)
    est_nc, _ = fit_phase(sub, phase, controls=False, half_fe=half_fe)[0], None
    raw = sub.groupby(phase.values)["margin"].agg(["size", "mean"])
    print(f"\n--- {name} ---")
    print(f"{'phase':>14s} {'n':>6s} {'raw':>8s} {'ctrl':>8s} "
          f"{'95% CI':>20s}  {'b2b_asym':>9s} {'trav_asym':>10s}")
    rows = []
    for p in order:
        if p not in est.index:
            continue
        lo, hi = ci.get(p, (np.nan, np.nan))
        m = phase.values == p
        b2b_as = sub.loc[m, "a_b2b"].mean() - sub.loc[m, "h_b2b"].mean()
        tr_as = (sub.loc[m, "a_travel"].mean() - sub.loc[m, "h_travel"].mean()) / 1000
        print(f"{str(p):>14s} {int(raw.loc[p,'size']):>6d} {raw.loc[p,'mean']:>+8.3f} "
              f"{est[p]:>+8.3f} ({lo:+.3f},{hi:+.3f})  {b2b_as:>+9.3f} {tr_as:>+10.3f}")
        rows.append(dict(phase=str(p), n=int(raw.loc[p, "size"]),
                         raw=float(raw.loc[p, "mean"]), ctrl=float(est[p]),
                         lo=float(lo), hi=float(hi)))
    return rows


def main():
    rng = np.random.default_rng(SEED)
    d = load_panel()
    dn = d[d.season.isin(NORMAL)].reset_index(drop=True)
    res = {}

    # ---------- team-games-played buckets ---------------------------------
    def gp_bucket(g):
        for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 30), (30, 41),
                       (41, 52), (52, 62), (62, 72), (72, 100)]:
            if lo <= g < hi:
                return f"gp{lo}-{hi}"
        return "gp72+"
    order_gp = ["gp0-5", "gp5-10", "gp10-20", "gp20-30", "gp30-41",
                "gp41-52", "gp52-62", "gp62-72", "gp72-100"]
    ph = dn["mean_gp"].apply(gp_bucket)
    res["gp_normal"] = report("HFA by TEAM-GAMES-PLAYED bucket, 5 normal seasons",
                              dn, ph, order_gp, rng, B=800)

    # robustness: team strength allowed to drift at midseason
    res["gp_halffe"] = report("  [robustness] same, with HALF-SEASON team FE",
                              dn, ph, order_gp, rng, B=400, half_fe=True)

    # ---------- calendar month --------------------------------------------
    mn = {10: "Oct", 11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr"}
    phm = dn["month"].map(mn).fillna("other")
    res["month_normal"] = report("HFA by CALENDAR MONTH, 5 normal seasons", dn, phm,
                                 ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"],
                                 rng, B=800)

    # ---------- the opening-window claim, tested directly -----------------
    print("\n=== (5b) THE OPENING-NIGHT / EARLY-WINDOW CLAIM ===")
    tests = [
        ("opening night (days_into<=1)", dn.days_into <= 1),
        ("opening week (days_into<=7)", dn.days_into <= 7),
        ("first 5 team-games (mean_gp<5)", dn.mean_gp < 5),
        ("first 10 team-games (mean_gp<10)", dn.mean_gp < 10),
        ("first 20 team-games (mean_gp<20)", dn.mean_gp < 20),
        ("October only", dn.month == 10),
    ]
    rows = []
    for name, mask in tests:
        ph2 = pd.Series(np.where(mask, "EARLY", "REST"), index=dn.index)
        est, ci = phase_with_ci(dn, ph2, 1500, rng)
        diff_boot = []
        sub = dn.reset_index(drop=True); p2 = ph2.reset_index(drop=True)
        for _ in range(1500):
            ix = rng.integers(0, len(sub), len(sub))
            try:
                e, *_ = fit_phase(sub.iloc[ix], p2.iloc[ix])
            except Exception:
                continue
            if "EARLY" in e.index and "REST" in e.index:
                diff_boot.append(e["EARLY"] - e["REST"])
        lo, hi = boot_ci(np.array(diff_boot))
        dlt = est["EARLY"] - est["REST"]
        n = int(mask.sum())
        print(f"  {name:32s} n={n:5d}  EARLY {est['EARLY']:+.3f}  REST "
              f"{est['REST']:+.3f}  DIFF {dlt:+.3f} CI({lo:+.3f},{hi:+.3f}) "
              f"{'SIG' if lo>0 or hi<0 else 'NS'}")
        rows.append(dict(test=name, n=n, early=float(est["EARLY"]),
                         rest=float(est["REST"]), diff=float(dlt),
                         lo=float(lo), hi=float(hi)))
    res["opening"] = rows

    # ---------- monotone trend test ---------------------------------------
    print("\n=== (5c) IS THERE A MONOTONE TREND? (linear in team-games-played) ===")
    sub = dn.copy()
    sub["gpc"] = (sub["mean_gp"] - 41) / 41.0
    teams = sorted(set(sub.home) | set(sub.away))
    seas = sorted(sub.season.unique())
    ts = [(t, y) for y in seas for t in teams]
    tsi = {k: i for i, k in enumerate(ts)}
    n = len(sub); r = np.arange(n)
    Z = np.zeros((n, len(ts)))
    Z[r, [tsi[(h, y)] for h, y in zip(sub.home, sub.season)]] += 1
    Z[r, [tsi[(a, y)] for a, y in zip(sub.away, sub.season)]] -= 1
    ones = np.ones((n, 1))
    trend = sub[["gpc"]].to_numpy(float)
    quad = trend ** 2
    for label, extra in (("linear", trend), ("linear+quadratic",
                                             np.hstack([trend, quad]))):
        X = np.hstack([Z, ones, extra, sub[SCHED].to_numpy(float)])
        cf, *_ = np.linalg.lstsq(X, sub.margin.to_numpy(float), rcond=None)
        k = extra.shape[1]
        est = cf[len(ts) + 1:len(ts) + 1 + k]
        bs = []
        for _ in range(800):
            ix = rng.integers(0, n, n)
            cb, *_ = np.linalg.lstsq(X[ix], sub.margin.to_numpy(float)[ix], rcond=None)
            bs.append(cb[len(ts) + 1:len(ts) + 1 + k])
        bs = np.array(bs)
        for i in range(k):
            lo, hi = boot_ci(bs[:, i])
            print(f"  {label:18s} term{i+1} = {est[i]:+.4f} pts per half-season "
                  f"CI({lo:+.4f},{hi:+.4f}) {'SIG' if lo>0 or hi<0 else 'NS'}")
        res[f"trend_{label}"] = dict(est=[float(v) for v in est])
        print(f"    intercept (HFA at mid-season) = {cf[len(ts)]:+.4f}")

    # ---------- 2020-21 no-crowd cross-check -------------------------------
    print("\n=== (5d) NO-CROWD CROSS-CHECK: does 2020-21 show the same shape? ===")
    d21 = d[d.season == "2020-21"].reset_index(drop=True)
    def gp3(g):
        return "early(<10)" if g < 10 else ("mid" if g < 50 else "late(>=50)")
    for label, sub2 in (("2020-21 NO CROWD", d21),
                        ("normal 5 seasons", dn),
                        ("2019-20 pre-shutdown", d[d.season == "2019-20"].reset_index(drop=True))):
        ph3 = sub2["mean_gp"].apply(gp3)
        est, ci = phase_with_ci(sub2, ph3, 800, rng)
        s = "  ".join(f"{p}: {est[p]:+.3f} ({ci[p][0]:+.2f},{ci[p][1]:+.2f})"
                      for p in ["early(<10)", "mid", "late(>=50)"] if p in est.index)
        print(f"  {label:22s} {s}")
        res[f"nocrowd_{label}"] = {p: float(est[p]) for p in est.index}

    (OUT / "ha_timeprofile.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {OUT/'ha_timeprofile.json'}")


if __name__ == "__main__":
    main()
