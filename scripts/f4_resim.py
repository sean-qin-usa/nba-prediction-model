#!/usr/bin/env python3
"""F4 RE-SIM — the 2026-27 paper-trade registry with the D112 ship applied.

WHAT CHANGED vs the registry as it stood (D75/D78/D82, scripts/bet_engine.py):
  (b) UPPER CONFIDENCE-EXCESS CAP.  Skip a bet when conf_us - conf_mkt > 0.08.
      On a same-side bet conf_us - conf_mkt IS the edge, so this is an upper
      edge cap.  Pre-registered at 0.08; TENTATIVE BY SWEEP (D112 swept 8 caps
      x 6 rules x 2 frames with no selection protection) — the DIRECTION is
      what the rule-free Kelly slope and reliability curve support.
  (a) EDGE SHRINKAGE IN SIZING.  Kelly is computed from
      p_mkt_side + max(0, a + b*edge) instead of p_us_side, (a,b) from the
      Kelly-slope regression, refit on completed seasons only (walk-forward
      here, annually in the live engine).  See scripts/f4_shrinkage.py.
  (c) THREE SIZING ARMS RUN IN PARALLEL — flat (the honest control),
      raw-Kelly (what D75 ran), shrunk-Kelly (the D112 ship).  Nothing is
      chosen here; October settles sizing empirically.

WHAT IS REPORTED (task (d))
  * All 4 registered rules + the D112 headline diagnostic R4_LOWT(t=.04),
    UNCAPPED vs CAPPED, on all 4 seasons.
  * IS/OOS in BOTH directions: DEV-IN (in-sample = the two seasons every rule
    was developed on, 2023-24 + 2024-25) and DEV-OUT (those two seasons in the
    holdout — the harder arrangement, TRADING_STRATEGY.md s5), plus the
    REGISTERED split (IS 22-23..24-25 / OOS 25-26) for continuity with D112.
  * n / hit% / ROI% / ROIfair% / PnL / staked / sharpe / maxDD per arm.
  * NOISE-COMPATIBILITY for every positive-ROI cell: Monte-Carlo null in which
    we have NO edge (hit ~ Bernoulli(p_mkt_side)) at the same prices and the
    same stakes -> P(ROI_null >= ROI_obs), plus the family-wise expected-max
    arithmetic over the cells actually printed.
  * Paired bootstrap dPnL/bet for cap-vs-uncapped on the registered rules.

FRAMES: data/ds_rt1_pergame.csv (p_full, 4 seasons — primary) and
data/capstone_pergame_tank.csv (p_us, 3 seasons — the exact D75/D78 frame).
Pricing/vig/Kelly conventions imported from bet_sim3 verbatim.  DB read-only.

Run: python scripts/f4_resim.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import duckdb                                                    # noqa: E402
import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402

from ba_intersection import star_out_map, team_schedule          # noqa: E402
from bet_sim3 import (BANKROLL, KELLY_CAP, KELLY_FRAC, MIN_DEC,  # noqa: E402
                      OVERROUND, TANK_GP, max_drawdown)
from f4_shrinkage import (CONF_EXCESS_CAP, fit_kelly_slope,      # noqa: E402
                          kelly_fraction, shrink_edge)

DB = os.path.join(ROOT, "data", "nba.duckdb")
RT1 = os.path.join(ROOT, "data", "ds_rt1_pergame.csv")
TANK = os.path.join(ROOT, "data", "capstone_pergame_tank.csv")
OUT = os.path.join(ROOT, "data", "f4_resim.json")

# registry constants — copied from scripts/bet_engine.py, NOT re-chosen here
EDGE_R4 = 0.02
CONF_TIER = 0.20
DIV_LO, DIV_HI = 0.03, 0.10

DEV_SEASONS = {"2023-24", "2024-25"}         # where every rule was developed
NONDEV_SEASONS = {"2022-23", "2025-26"}
# The two IS/OOS DIRECTIONS asked for are the SAME partition scored once:
#   DEV-IN  direction: IS = DEV,    OOS = NONDEV   (the arrangement every
#                                                   registered rule was born in)
#   DEV-OUT direction: IS = NONDEV, OOS = DEV      (the harder arrangement —
#                                                   dev seasons in the holdout)
# Only the labels swap, so the table prints each half once and both readings
# are stated explicitly.  REGISTERED (IS 22-23..24-25 / OOS 25-26) is carried
# alongside for continuity with the D75/D78/D112 tables.
WINDOWS = [
    ("POOL",       {"2022-23", "2023-24", "2024-25", "2025-26"}),
    ("REG:IS",     {"2022-23", "2023-24", "2024-25"}),
    ("REG:OOS",    {"2025-26"}),
    ("DEV",        DEV_SEASONS),
    ("NONDEV",     NONDEV_SEASONS),
]
N_BOOT = 4000
N_NULL = 20000
SEED = 20260801


# ---------------------------------------------------------------- frame -----
def build(csv: str, pcol: str, want_star: bool) -> pd.DataFrame:
    df = pd.read_csv(csv, dtype={"game_id": str})
    df["game_id"] = df.game_id.str.zfill(10)
    df["game_date"] = pd.to_datetime(df.game_date)
    df = df.rename(columns={pcol: "p_us"})
    con = duckdb.connect(DB, read_only=True)
    try:
        ts = team_schedule(con)
        so = star_out_map(con) if want_star else None
    finally:
        con.close()
    keys = ["season", "game_id"]
    for side in ("home", "away"):
        m = ts.rename(columns={"team": side})
        df = df.merge(m[keys + [side, "gp_before"]], on=keys + [side],
                      how="left")
        df = df.rename(columns={"gp_before": f"{side[0]}_gp"})
    assert df.h_gp.notna().all() and df.a_gp.notna().all()

    df["pick_home"] = df.p_us > 0.5
    df["same_side"] = (df.p_us - 0.5) * (df.p_mkt - 0.5) > 0
    df["p_us_side"] = np.where(df.pick_home, df.p_us, 1 - df.p_us)
    df["p_mkt_side"] = np.where(df.pick_home, df.p_mkt, 1 - df.p_mkt)
    df["edge"] = df.p_us_side - df.p_mkt_side
    df["hit"] = np.where(df.pick_home, df.y == 1, df.y == 0).astype(int)
    df["conf_us"] = (df.p_us - 0.5).abs()
    df["conf_mkt"] = (df.p_mkt - 0.5).abs()
    df["conf_gap"] = df.conf_us - df.conf_mkt          # == edge on same-side
    q = df.p_mkt_side * OVERROUND
    df["dec"] = np.maximum(1.0 / q, MIN_DEC)
    df["dec_fair"] = 1.0 / df.p_mkt_side
    df["floored"] = (1.0 / q) < MIN_DEC
    df["late"] = (df.h_gp >= TANK_GP) | (df.a_gp >= TANK_GP)

    # STAR_FAV_SHARPER needs the MARKET favourite's star-out flag
    if want_star and so is not None:
        fav = np.where(df.p_mkt >= 0.5, df.home, df.away)
        df["fav_team"] = fav
        m = so.rename(columns={"team": "fav_team"})[
            ["game_id", "fav_team", "star_out"]]
        df = df.merge(m, on=["game_id", "fav_team"], how="left")
        df["fav_star_out"] = df.star_out.fillna(False).astype(bool)
        df = df.drop(columns=["star_out"])
    else:
        df["fav_star_out"] = False
    return df.sort_values("game_date").reset_index(drop=True)


def registry_masks(df: pd.DataFrame) -> dict:
    """The 4 SHIPPED F4 rules (exact operators from bet_engine.rules_fired)
    plus the D112 headline diagnostic.  Thresholds are NOT re-chosen here."""
    ss = df.same_side
    tails = df.conf_us > CONF_TIER
    band = (df.edge >= DIV_LO) & (df.edge <= DIV_HI)
    return {
        "R4_LOWT":          ss & (df.edge > EDGE_R4) & df.late,
        "T20_D03_10_W":     ss & tails & band & df.late,
        "T20_D03_10":       ss & tails & band,
        "STAR_FAV_SHARPER": ss & (df.edge > 0) & df.fav_star_out,
        # labelled diagnostic — D112's headline used t=.04; the SHIPPED rule
        # stays at its registered t=.02.  Printed, never selected.
        "[diag] R4_LOWT(t=.04)": ss & (df.edge > 0.04) & df.late,
    }


# ---------------------------------------------------------------- sizing ----
def walkforward_coeffs(df: pd.DataFrame) -> dict:
    """(a,b) per season fitted on COMPLETED PRIOR seasons only.  The first
    season has no prior -> None (cold start: shrunk-Kelly stakes nothing)."""
    seasons = sorted(df.season.unique())
    out: dict[str, dict | None] = {}
    for i, s in enumerate(seasons):
        prior = df[df.season.isin(seasons[:i]) & df.same_side]
        if i == 0 or len(prior) < 50:
            out[s] = None
            continue
        out[s] = fit_kelly_slope(prior.edge, prior.hit - prior.p_mkt_side)
    return out


def stakes(sub: pd.DataFrame, arm: str, price: str, wf: dict,
           frozen: dict) -> np.ndarray:
    dec = (sub.dec if price == "vig" else sub.dec_fair).values
    if arm == "flat":
        return np.ones(len(sub))
    if arm == "raw_kelly":
        p = sub.p_us_side.values
    else:                                   # shrunk_kelly_wf / shrunk_kelly_reg
        coeffs = wf if arm.endswith("_wf") else None
        p = np.empty(len(sub))
        for i, (s, e, pm) in enumerate(zip(sub.season.values, sub.edge.values,
                                           sub.p_mkt_side.values)):
            c = coeffs.get(s) if coeffs is not None else frozen
            p[i] = np.nan if c is None else pm + shrink_edge(e, c["a"], c["b"])
    f = np.array([0.0 if not np.isfinite(pi) else kelly_fraction(pi, di)
                  for pi, di in zip(p, dec)])
    return np.minimum(KELLY_FRAC * f * BANKROLL, KELLY_CAP)


def score(sub: pd.DataFrame, arm: str, wf: dict, frozen: dict,
          price: str = "vig") -> dict:
    st = stakes(sub, arm, price, wf, frozen)
    keep = st > 0
    sub, st = sub[keep], st[keep]
    if len(sub) == 0:
        return dict(n=0, staked=0.0, pnl=0.0, roi=np.nan, roi_fair=np.nan,
                    hit=np.nan, sharpe=np.nan, maxdd=0.0, pnl_arr=None,
                    n_cand=int(len(keep)))
    dec = (sub.dec if price == "vig" else sub.dec_fair).values
    won = sub.hit.values.astype(bool)
    pnl = np.where(won, st * (dec - 1), -st)
    pnlf = np.where(won, st * (sub.dec_fair.values - 1), -st)
    sd = float(pnl.std(ddof=1)) if len(pnl) > 1 else np.nan
    return dict(n=len(sub), staked=float(st.sum()), pnl=float(pnl.sum()),
                roi=float(pnl.sum() / st.sum()),
                roi_fair=float(pnlf.sum() / st.sum()),
                hit=float(sub.hit.mean()),
                sharpe=float(pnl.mean() / sd) if sd and sd > 0 else np.nan,
                maxdd=max_drawdown(np.cumsum(pnl)),
                stake_arr=st, p_arr=sub.p_mkt_side.values, dec_arr=dec,
                n_cand=int(len(keep)))


# ------------------------------------------------------------ noise math ----
def noise_p(r: dict, seed: int = SEED) -> float:
    """P(ROI >= observed | we have NO edge): resample outcomes from the
    de-vigged market probability at the same prices and the same stakes."""
    if not r["n"] or not np.isfinite(r["roi"]):
        return float("nan")
    rng = np.random.default_rng(seed)
    st, p, dec = r["stake_arr"], r["p_arr"], r["dec_arr"]
    win = rng.random((N_NULL, len(p))) < p
    pnl = np.where(win, st * (dec - 1), -st)
    roi = pnl.sum(axis=1) / st.sum()
    return float((roi >= r["roi"]).mean())


def boot_delta(pnl_unc: np.ndarray, keep: np.ndarray) -> dict:
    """Paired bootstrap of the per-bet PnL delta (capped minus uncapped) on the
    UNCAPPED bet set; the capped arm scores 0 where the cap skipped."""
    d = np.where(keep, 0.0, -pnl_unc)
    if len(d) == 0:
        return {"n": 0, "mean": 0.0, "lo": 0.0, "hi": 0.0, "verdict": "EMPTY"}
    rng = np.random.default_rng(SEED)
    m = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return {"n": int(len(d)), "mean": float(d.mean()), "lo": float(lo),
            "hi": float(hi), "verdict": "SIG" if lo > 0 or hi < 0 else "NS"}


# ---------------------------------------------------------------- report ----
ARMS = ("flat", "raw_kelly", "shrunk_kelly_wf", "shrunk_kelly_reg")
HDR = (f"    {'arm':<17}{'window':>8}{'n':>5}{'hit%':>7}{'ROI%':>8}"
       f"{'ROIfair%':>10}{'PnL':>9}{'staked':>9}{'sharpe':>8}{'maxDD':>8}"
       f"{'null p':>9}")
DIRECTIONS = ("DEV-IN  direction: IS = DEV, OOS = NONDEV   |   "
              "DEV-OUT direction: IS = NONDEV, OOS = DEV   "
              "(same partition, labels swap)")


def shrunk_diagnostic(df: pd.DataFrame, frozen: dict, res: dict,
                      label: str) -> None:
    """Why shrunk-Kelly stakes what it stakes.  The load-bearing number is
    V_max = (p_mkt_side + shrunk_edge)/p_mkt_side — the LARGEST overround at
    which quarter-Kelly on the CALIBRATED edge still puts money down.  The
    sims price at V=1.045; V_max < V means 'stake nothing at this price'."""
    a, b = frozen["a"], frozen["b"]
    be = (-a / b) if b > 0 else float("nan")
    print(f"\n  {'='*112}\n  SHRUNK-KELLY DIAGNOSTIC — what the calibrated "
          f"edge actually permits")
    print(f"    break-even claimed edge (shrunk_edge > 0): {be:+.4f}   "
          f"vs the registered upper cap {CONF_EXCESS_CAP:+.4f}")
    rows = []
    for rule, mask in registry_masks(df).items():
        base = df[mask]
        if len(base) < 10:
            continue
        cap = base[base.conf_gap <= CONF_EXCESS_CAP]
        if len(cap) == 0:
            continue
        se = np.array([shrink_edge(e, a, b) for e in cap.edge.values])
        vmax = (cap.p_mkt_side.values + se) / cap.p_mkt_side.values
        vraw = cap.p_us_side.values / cap.p_mkt_side.values
        rows.append({"rule": rule, "n_capped": int(len(cap)),
                     "frac_shrunk_pos": float((se > 0).mean()),
                     "vmax_mean": float(vmax.mean()),
                     "vmax_max": float(vmax.max()),
                     "vraw_mean": float(vraw.mean()),
                     "frac_vmax_over_vig": float((vmax > OVERROUND).mean())})
        print(f"    {rule:<24} n_capped={len(cap):>4}  "
              f"shrunk_edge>0 on {100*(se > 0).mean():>5.1f}%  "
              f"V_max mean {vmax.mean():.4f} max {vmax.max():.4f}  "
              f"(raw-Kelly V_max mean {vraw.mean():.4f})  "
              f"-> stakes at V={OVERROUND} on "
              f"{100*(vmax > OVERROUND).mean():.1f}% of bets")
    # SENSITIVITY: the slope itself is noisy (se ~0.10-0.20).  How large would
    # b have to be before the calibrated edge clears the vig on the CAPPED
    # registry?  If the answer is "far outside the CI", the sizing conclusion
    # does not depend on the exact slope, only on b << 1.
    allmask = np.zeros(len(df), bool)
    for rule, mask in registry_masks(df).items():
        if not rule.startswith("[diag]"):
            allmask |= mask.values
    cap = df[allmask & (df.conf_gap <= CONF_EXCESS_CAP)]
    sens = []
    for bb in np.arange(0.1, 1.01, 0.1):
        se = np.maximum(0.0, a + bb * cap.edge.values)
        vmax = (cap.p_mkt_side.values + se) / cap.p_mkt_side.values
        sens.append((float(bb), float((vmax > OVERROUND).mean())))
    print("    SENSITIVITY (a fixed, b swept) — % of capped registry bets a "
          "quarter-Kelly bettor would stake at V=1.045:")
    print("      " + "  ".join(f"b={b:.1f}:{100*f:>5.1f}%" for b, f in sens))
    res.setdefault("shrunk_sens", {})[label] = sens
    print(f"    READ: the calibrated edge clears a {OVERROUND} overround "
          f"essentially nowhere, so shrunk-Kelly stakes ZERO at the sims' "
          f"price.  It stakes LIVE only when the best shopped decimal beats "
          f"1/(p_mkt_side + shrunk_edge) — i.e. only when line shopping "
          f"delivers a price at or better than consensus fair.  That is the "
          f"quantified version of D75's 'line shopping covers the gap'.")
    res.setdefault("shrunk_diag", {})[label] = rows


def run_frame(df: pd.DataFrame, label: str, res: dict) -> None:
    wf = walkforward_coeffs(df)
    ss = df[df.same_side]
    frozen = fit_kelly_slope(ss.edge, ss.hit - ss.p_mkt_side)
    print(f"\n{'#'*118}\nFRAME: {label}   n={len(df)}  same-side={len(ss)}")
    print(f"{'#'*118}")
    print(f"  frozen (whole-frame, IN-SAMPLE for the coefficients) kelly slope:"
          f" realised = {frozen['a']:+.4f} {frozen['b']:+.4f} x claimed  "
          f"(se {frozen['se_b']:.4f}, t={frozen['t']:+.2f}, n={frozen['n']})")
    for s, c in wf.items():
        if c is None:
            print(f"  walk-forward {s}: COLD START (no completed prior season)"
                  f" -> shrunk-Kelly stakes 0")
        else:
            print(f"  walk-forward {s}: a={c['a']:+.4f} b={c['b']:+.4f} "
                  f"(se {c['se_b']:.4f}, n={c['n']}) -> break-even claimed "
                  f"edge {(-c['a']/c['b']) if c['b'] > 0 else float('nan'):+.4f}")
    res.setdefault("frames", {})[label] = {
        "frozen": frozen,
        "walkforward": {k: v for k, v in wf.items()}}
    shrunk_diagnostic(df, frozen, res, label)

    rows, positives = [], []
    for rule, mask in registry_masks(df).items():
        base = df[mask].sort_values("game_date")
        if len(base) < 10:
            print(f"\n  {rule}: n={len(base)} — too few to score")
            continue
        keep_all = (base.conf_gap <= CONF_EXCESS_CAP).values
        print(f"\n  {'='*112}\n  RULE {rule}   uncapped n={len(base)}   "
              f"capped n={int(keep_all.sum())}   "
              f"(cap removes {len(base)-int(keep_all.sum())} = "
              f"{100*(1-keep_all.mean()):.1f}%)")
        for variant, km in (("UNCAPPED", np.ones(len(base), bool)),
                            ("CAP=0.08", keep_all)):
            print(f"  --- {variant} ---   {DIRECTIONS}\n{HDR}")
            for wname, wseasons in WINDOWS:
                wsel = base.season.isin(wseasons).values
                sub = base[wsel & km]
                for arm in ARMS:
                    r = score(sub, arm, wf, frozen)
                    pnull = noise_p(r) if (r["n"] and
                                           np.isfinite(r["roi"]) and
                                           r["roi"] > 0) else float("nan")
                    print(f"    {arm:<17}{wname:>8}{r['n']:>5}"
                          f"{100*r['hit'] if r['n'] else 0:>7.1f}"
                          f"{100*r['roi'] if r['n'] else 0:>8.2f}"
                          f"{100*r['roi_fair'] if r['n'] else 0:>10.2f}"
                          f"{r['pnl']:>9.2f}{r['staked']:>9.1f}"
                          f"{r['sharpe'] if r['n'] > 1 else 0:>8.3f}"
                          f"{r['maxdd']:>8.2f}"
                          f"{pnull if np.isfinite(pnull) else float('nan'):>9.3f}")
                    rec = {"frame": label, "rule": rule,
                           "variant": variant, "window": wname,
                           "arm": arm, "n": r["n"],
                           "hit": r["hit"], "roi": r["roi"],
                           "roi_fair": r["roi_fair"], "pnl": r["pnl"],
                           "staked": r["staked"], "sharpe": r["sharpe"],
                           "maxdd": r["maxdd"], "null_p": pnull}
                    rows.append(rec)
                    if r["n"] and np.isfinite(r["roi"]) and r["roi"] > 0:
                        positives.append(rec)
            print()
        # paired cap-vs-uncapped bootstrap on flat stakes (D112's statistic)
        pnl_unc = np.where(base.hit.values.astype(bool),
                           base.dec.values - 1, -1.0)
        for wname, wseasons in WINDOWS:
            wsel = base.season.isin(wseasons).values
            bd = boot_delta(pnl_unc[wsel], keep_all[wsel])
            print(f"    dPnL/bet (flat, cap - uncapped) {wname:>14}: "
                  f"{bd['mean']:+.4f} [{bd['lo']:+.4f},{bd['hi']:+.4f}] "
                  f"{bd['verdict']}  (n_uncapped={bd['n']})")
            rows.append({"frame": label, "rule": rule, "variant": "dPnL",
                         "window": wname, "arm": "flat", **bd})
    res.setdefault("rows", []).extend(rows)

    # ---- family-wise arithmetic on this frame
    k = len([r for r in rows if r.get("variant") in ("UNCAPPED", "CAP=0.08")])
    print(f"\n  FAMILY-WISE: {k} rule x variant x window x arm cells printed "
          f"for this frame.  Under a global null of zero edge the expected "
          f"number with null p < 0.05 is {0.05*k:.1f}; "
          f"{len([r for r in positives if np.isfinite(r['null_p']) and r['null_p'] < 0.05])}"
          f" observed.  Cells are heavily overlapping (same bets re-scored), "
          f"so this is an upper bound on the surprise, not a test.")


def main() -> None:
    res: dict = {"cap": CONF_EXCESS_CAP, "kelly_frac": KELLY_FRAC,
                 "overround": OVERROUND, "kelly_cap": KELLY_CAP,
                 "bankroll_ref": BANKROLL}
    print("F4 RE-SIM — registry + D112 ship (upper conf-excess cap "
          f"{CONF_EXCESS_CAP} + edge shrinkage in sizing)")
    print(f"pricing: proportional overround {OVERROUND}, decimal floored at "
          f"{MIN_DEC}; quarter-Kelly on a {BANKROLL:.0f}u reference bankroll, "
          f"per-bet cap {KELLY_CAP:.0f}u")
    run_frame(build(RT1, "p_full", want_star=True),
              "PRIMARY rt1 p_full 4-season", res)
    run_frame(build(TANK, "p_us", want_star=True),
              "REPLICATION capstone_tank 3-season", res)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
