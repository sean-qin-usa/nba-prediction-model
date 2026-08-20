#!/usr/bin/env python3
"""D178 — RE-DERIVE THE MONTHLY CLV BANDS FROM SCRATCH ON THE D171/D173 DATA.

WHY THIS EXISTS.  `scripts/bet_engine.py` shipped RED = -0.0131 / GOOD =
+0.0200 from D120/D121.  D159 showed those were built wrong in TWO independent
ways, and the second one is the fatal one:

  (1) LEVEL — the band was CENTRED on the ALL-SAME-SIDE UNIVERSE (+0.00350)
      but WIDTHED by the per-bet sd of the RULE UNION, while
      `--monthly-report` scores the UNION against it.  Centre and width came
      from two different populations.
  (2) FRAME — it was built in the SP (spread-derived probability) convention,
      2022-23..2025-26, while the engine prices and settles MONEYLINES:
      `bet_engine.settle` computes `clv = close_implied - implied_p` where
      `implied_p` is the de-vigged CONSENSUS MONEYLINE probability on our side.

Together those made GOOD a +1.28-sigma event with a ~98-month expected wait —
MIS-SPECIFIED, not merely mis-levelled.  D159 published replacements
(-0.0006 / +0.0246 at 66 bets/month) but derived them on the PRE-BACKFILL
data.  THIS SCRIPT DERIVES THEM AGAIN, FROM SCRATCH, ON THE CURRENT CERTIFIED
DATA — nothing is pasted.

WHICH SPACE.  The engine scores CLV in PROBABILITY space on the REAL MONEYLINE
(`ML`), so that is the frame used here.  D173 reports two CLV numbers and they
are NOT interchangeable: the probability-space ML CLV is +0.01228 (essentially
unchanged from D159's +0.01197), while the ATS SPREAD-POINT CLV doubled to
+0.320 POINTS.  Points are not probabilities; the engine never sees a spread
point.  **Using the ATS number here would be a unit error of ~26x.**

THE CONSTRUCTION (fixed, and now printed by `--monthly-report` itself so the
bands can never again drift from what produced them):
    population  UNION of the four registered F4 rules, unique games, @OPEN
    frame       ML (real moneyline), de-vigged, probability space
    centre      MEAN CLV OF THE UNION           (not of the universe — fix 1)
    width       +- 2 * (per-bet CLV sd of the UNION) / sqrt(median union
                bets per month)
    months      calendar months with >= 3 union bets

ARMS.  HONEST (`data/capstone_pergame.csv`, the D171 certified artifact, the
availability construction October actually ships) is the one INSTALLED.
LEAKY_REG (`data/ds_rt1_pergame.csv`) is carried ONLY as a fidelity anchor —
it must reproduce D155/D159's registered union digits or nothing here is
believed.

WHAT THIS BAND IS FOR (D176, folded in deliberately).  D176 measured CLV and
ROI apart on three pre-registered selectors: the AVAILABILITY-DIVERGENCE
selector bought MORE CLV (6/6 cells, +0.143 pts) and LESS ROI (1/6, -1.16pp),
while the CLV-TARGETED selector bought essentially no extra CLV (+0.004) and
the MOST ROI.  **CLV IS NOT A SUFFICIENT STATISTIC FOR BET SELECTION**; a band
tuned purely on it can green-light a selector that is losing money.  These
bands are a fast-resolving MONITOR on execution and timing, NOT the objective.
Relatedly: nothing here is a net-of-null statistic — centre, sd and se are
plain sample moments — because D176 also found all three new arms beat their
own permutation nulls (p <= 0.048, BH-surviving) AND STILL LOST TO THE
INCUMBENT.  Beating a null is necessary, not sufficient.

Read-only on data/nba.duckdb (read_only=True, retry 60s) — inherited from
bo_openbacktest/bo_lineshop, which this script imports rather than
re-implements.  Touches NO certified artifact.

Run:  python scripts/d178_clvbands.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from math import sqrt

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import nbapred.threads                                            # noqa: E402
nbapred.threads.pin(1)

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

import bo_openbacktest as bo                                      # noqa: E402
import hc_honestclv as hc                                         # noqa: E402

OUT = os.path.join(ROOT, "data", "d178_clvbands.json")

# What bet_engine.py carried BEFORE this entry (D120/D121).
SHIPPED_RED, SHIPPED_GOOD = -0.0131, +0.0200
# D159's replacements, derived on PRE-BACKFILL data.  Carried ONLY to report
# the drift; never installed from here.
D159_RED, D159_GOOD = -0.0006, +0.0246
# The registered union digits the LEAKY_REG anchor must reproduce (D155/D159).
ANCHOR_ML_OPEN_UNION = (1378, 0.01590)

MIN_BETS_PER_MONTH = 3          # a month with < 3 union bets is not a month


def derive(m: pd.DataFrame, src: str) -> dict:
    """Union-centred CLV band for one frame in one price space."""
    masks, p_open, _dec, ok_o, _edge = hc.masks_of(m, src, when="open")
    p_close, _dc, ok_c = bo.price_cols(m, "close", src)
    ok = ok_o & ok_c
    clv = p_close - p_open                      # >0 = we beat the close
    union = np.asarray(masks["UNION"]) & ok
    same = ((m.p_us.values - 0.5) *
            (np.where(m.pick_home, p_open, 1 - p_open) - 0.5)) > 0
    universe = same & ok                        # the "ALL same-side" universe

    sub = m[union].copy()
    sub["clv"] = clv[union]
    sub["ym"] = sub.game_date.dt.to_period("M").astype(str)
    g = sub.groupby("ym").clv.agg(["size", "mean"])
    g = g[g["size"] >= MIN_BETS_PER_MONTH]

    centre = float(sub.clv.mean())              # THE UNION centre (fix 1)
    sd_bet = float(sub.clv.std(ddof=1))
    n_med = float(g["size"].median())
    se = sd_bet / sqrt(n_med)
    per_rule = {r: {"n": int((np.asarray(v) & ok).sum()),
                    "clv": float(clv[np.asarray(v) & ok].mean())}
                for r, v in masks.items()}
    return {
        "src": src,
        "n_union_bets": int(len(sub)),
        "n_universe": int(universe.sum()),
        "centre_union": centre,
        "centre_universe": float(clv[universe].mean()),
        "per_bet_sd": sd_bet,
        "n_months": int(len(g)),
        "median_bets_per_month": n_med,
        "se_at_median_n": se,
        "red": centre - 2 * se,
        "good": centre + 2 * se,
        "mean_of_month_means": float(g["mean"].mean()),
        "sd_of_month_means": float(g["mean"].std(ddof=1)),
        "frac_months_positive": float((g["mean"] > 0).mean()),
        "empirical_p05": float(g["mean"].quantile(0.05)),
        "empirical_p50": float(g["mean"].median()),
        "empirical_p95": float(g["mean"].quantile(0.95)),
        "seasons": sorted(sub.season.unique().tolist()),
        "per_rule": per_rule,
        "months": g.reset_index().to_dict("records"),
    }


def reachability(b: dict, red: float, good: float) -> dict:
    """Under monthly mean ~ N(centre, se): how often is a month GOOD/RED, and
    how long until 2 consecutive GOOD months?  Plus the EMPIRICAL counts."""
    from statistics import NormalDist
    nd = NormalDist(b["centre_union"], b["se_at_median_n"])
    p_good = 1 - nd.cdf(good)
    p_red = nd.cdf(red)
    v = np.array([r["mean"] for r in b["months"]])
    above = v > good
    pairs = int((above[:-1] & above[1:]).sum()) if len(above) > 1 else 0
    return {"good": good, "red": red,
            "sigma_of_good": (good - b["centre_union"]) / b["se_at_median_n"],
            "sigma_of_red": (red - b["centre_union"]) / b["se_at_median_n"],
            "p_month_good": float(p_good), "p_month_red": float(p_red),
            "expected_months_to_2consec":
                float(1 / p_good ** 2) if p_good > 0 else float("inf"),
            "empirical_months_above_good": int(above.sum()),
            "empirical_months_below_red": int((v < red).sum()),
            "empirical_consecutive_pairs": pairs,
            "n_months": len(v)}


def main() -> None:
    res = {"design": __doc__.split("\n")[0],
           "md5": {os.path.basename(p): hc.md5(p) for p in
                   (hc.HONEST, hc.LEAKY_REG,
                    os.path.join(ROOT, "data", "derived", "odds_open.csv"))}}
    print("=" * 100)
    print("D178 — MONTHLY CLV BANDS, RE-DERIVED FROM SCRATCH ON THE "
          "D171/D173 DATA")
    print("=" * 100)
    for k, v in res["md5"].items():
        print(f"    {v}  {k}")

    frames = hc.build_frames(res)
    for lab in ("HONEST", "LEAKY_REG"):
        m = frames[lab]
        print(f"    frame {lab:<11s} n={len(m):5d}  seasons="
              f"{sorted(m.season.unique())}")

    # ---------------------------------------------- fidelity anchor --------
    print(f"\n{'-'*100}\n[1] FIDELITY ANCHOR — LEAKY_REG must reproduce the "
          f"REGISTERED ML/open UNION digits (D155/D159)\n{'-'*100}")
    a = derive(frames["LEAKY_REG"], "ML")
    n_reg, clv_reg = ANCHOR_ML_OPEN_UNION
    u = a["per_rule"]["UNION"]
    ok = u["n"] == n_reg and abs(u["clv"] - clv_reg) < 5e-5
    print(f"    UNION @open ML   n={u['n']} (registered {n_reg})   "
          f"CLV={u['clv']:+.5f} (registered {clv_reg:+.5f})   "
          f"{'EXACT — the harness is the registered one' if ok else '*** MISMATCH ***'}")
    res["anchor"] = {"n": u["n"], "clv": u["clv"], "registered_n": n_reg,
                     "registered_clv": clv_reg, "exact": bool(ok)}

    # ---------------------------------------------- the derivation ---------
    print(f"\n{'-'*100}\n[2] THE DERIVATION.  Engine frame = ML (real "
          f"moneyline), PROBABILITY space, union @open, unique games.\n"
          f"    NOTE the other CLV number in D173 is the ATS SPREAD-POINT CLV "
          f"(+0.320 POINTS) — a different unit entirely; the engine\n"
          f"    never sees a spread point, so it is NOT used and would be a "
          f"~26x unit error here.\n{'-'*100}")
    out = {}
    for lab in ("HONEST", "LEAKY_REG"):
        for src in ("ML", "SP"):
            out[f"{src}|{lab}"] = derive(frames[lab], src)
    print(f"    {'frame|arm':<16}{'nbets':>7}{'mo':>4}{'med/mo':>8}"
          f"{'centre':>10}{'sd_bet':>9}{'se':>9}{'RED':>11}{'GOOD':>10}"
          f"  (universe centre)")
    for k, b in out.items():
        print(f"    {k:<16}{b['n_union_bets']:>7}{b['n_months']:>4}"
              f"{b['median_bets_per_month']:>8.0f}{b['centre_union']:>+10.5f}"
              f"{b['per_bet_sd']:>9.5f}{b['se_at_median_n']:>9.5f}"
              f"{b['red']:>+11.5f}{b['good']:>+10.5f}"
              f"      {b['centre_universe']:+.5f}")
    res["bands"] = out

    ship = out["ML|HONEST"]
    print(f"\n    >>> THE INSTALLED BAND (ML | HONEST — the frame the engine "
          f"scores in, on the availability construction it ships):")
    print(f"        centre (union mean CLV)   {ship['centre_union']:+.6f}")
    print(f"        per-bet CLV sd            {ship['per_bet_sd']:.6f}")
    print(f"        union bets / months       {ship['n_union_bets']} over "
          f"{ship['n_months']} months, MEDIAN "
          f"{ship['median_bets_per_month']:.0f}/month")
    print(f"        monthly se at that n      {ship['se_at_median_n']:.6f}")
    print(f"        RED  = centre - 2*se      {ship['red']:+.6f}")
    print(f"        GOOD = centre + 2*se      {ship['good']:+.6f}")
    print(f"        seasons                   {ship['seasons']}")

    # ---------------------------------------------- what changed -----------
    print(f"\n{'-'*100}\n[3] WHAT CHANGES, AND WHY THE OLD BAND WAS "
          f"MIS-SPECIFIED (not merely mis-levelled)\n{'-'*100}")
    rows = [("SHIPPED  D120/D121 (SP frame, universe-centred/union-widthed)",
             SHIPPED_RED, SHIPPED_GOOD),
            ("D159     (ML, union-centred; PRE-BACKFILL data)",
             D159_RED, D159_GOOD),
            ("D178     (ML, union-centred; D171/D173 data) *** INSTALLED ***",
             ship["red"], ship["good"])]
    print(f"    {'band':<62}{'RED':>10}{'GOOD':>10}{'sigma(GOOD)':>13}"
          f"{'E[months to 2 consec GOOD]':>28}")
    for lab, r, g in rows:
        rc = reachability(ship, r, g)
        em = rc["expected_months_to_2consec"]
        print(f"    {lab:<62}{r:>+10.4f}{g:>+10.4f}"
              f"{rc['sigma_of_good']:>+13.2f}"
              f"{(f'{em:.1f}' if np.isfinite(em) else 'inf'):>28}")
    res["comparison"] = [
        {"label": lab, "red": r, "good": g, **reachability(ship, r, g)}
        for lab, r, g in rows]

    print(f"\n    EMPIRICAL CHECK on the {ship['n_months']} months in the "
          f"ML|HONEST frame (what the bands would have flagged):")
    for lab, r, g in rows:
        rc = reachability(ship, r, g)
        print(f"      {lab:<62} GOOD months "
              f"{rc['empirical_months_above_good']:>2}/{rc['n_months']}, "
              f"consecutive GOOD pairs "
              f"{rc['empirical_consecutive_pairs']:>2}, RED months "
              f"{rc['empirical_months_below_red']:>2}")

    print(f"\n    MONTHS (ML|HONEST):")
    for r in ship["months"]:
        print(f"      {r['ym']}  n={r['size']:>3}  meanCLV={r['mean']:+.5f}")

    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
