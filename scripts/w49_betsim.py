"""W49 TASK 3 — the trading implication of the confidence asymmetry.

FACT BEING ACTED ON (measured on data/ds_rt1_pergame.csv, 4 seasons):
  when |p_us-0.5| > |p_mkt-0.5| we lose +0.0150 nats/game (n=1650); when we are
  LESS confident we lose only +0.0064 (n=3270).  2.3x.

STRUCTURAL IDENTITY (worth stating before any sim): on a SAME-SIDE bet,
    edge = p_us_side - p_mkt_side  ==  |p_us-0.5| - |p_mkt-0.5|  ==  conf_gap.
So "skip games where our confidence exceeds the market's by more than X" is
EXACTLY an upper cap on the betting edge.  The D78 registry rules already
carry a 0.10 cap (T20.D03-10); the D75 R4 low-threshold rule — the primary
2026-27 paper-trade rule — carries NONE.  That is the actionable gap.

USING THE MARKET HERE IS ALLOWED: this is BET SELECTION, not a model input
(G2 market-blind applies to the model only).

WHAT IS RUN
  A. Bet-sizing reliability curve: by conf_gap (=edge) bucket, our CLAIMED
     excess win prob vs the REALISED excess win rate, on same-side games.
     This is the bet-sizing answer independent of any rule.
  B. Confidence-excess cap sweep X in {0.04..0.20, inf} applied to the
     pre-registered rule family (R1/R4 low-t + the D78 T20 tail rules),
     flat and quarter-Kelly, IS(23-24,24-25) / OOS(25-26) / pooled, with
     paired bootstrap CIs on the per-bet PnL delta (capped minus uncapped,
     evaluated on the uncapped bet set so the pairing is exact).
  C. Same sweep replicated on data/capstone_pergame_tank.csv so the numbers
     line up with the registered D75/D78 tables.

Machinery (pricing, sizing, IS/OOS discipline) imported verbatim from
scripts/bet_sim3.py — nothing existing is edited.  Read-only DB.

Run: python scripts/w49_betsim.py
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

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ba_intersection import team_schedule  # noqa: E402
from bet_sim3 import (KELLY_CAP, KELLY_FRAC, BANKROLL, MIN_DEC,  # noqa: E402
                      OVERROUND, TANK_GP, score)

DB = os.path.join(ROOT, "data", "nba.duckdb")
RT1 = os.path.join(ROOT, "data", "ds_rt1_pergame.csv")
TANK = os.path.join(ROOT, "data", "capstone_pergame_tank.csv")
OUT = os.path.join(ROOT, "data", "w49_betsim.json")

IS_SEASONS = {"2022-23", "2023-24", "2024-25"}
OOS_SEASON = "2025-26"
CAPS = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, np.inf]
N_BOOT = 4000
SEED = 20260801


def build(csv: str, pcol: str) -> pd.DataFrame:
    df = pd.read_csv(csv, dtype={"game_id": str})
    df["game_id"] = df.game_id.str.zfill(10)
    df["game_date"] = pd.to_datetime(df.game_date)
    df = df.rename(columns={pcol: "p_us"})
    con = duckdb.connect(DB, read_only=True)
    try:
        ts = team_schedule(con)
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
    df["conf_gap"] = df.conf_us - df.conf_mkt
    q = df.p_mkt_side * OVERROUND
    df["dec"] = np.maximum(1.0 / q, MIN_DEC)
    df["floored"] = (1.0 / q) < MIN_DEC
    df["dec_fair"] = 1.0 / df.p_mkt_side
    df["late"] = (df.h_gp >= TANK_GP) | (df.a_gp >= TANK_GP)
    return df.sort_values("game_date").reset_index(drop=True)


# ------------------------------------------------------------------ A
def reliability(df: pd.DataFrame, res: dict) -> None:
    print(f"\n{'='*104}")
    print("A. BET-SIZING RELIABILITY — do we actually win MORE when we claim "
          "more edge?  (same-side games only)")
    print(f"{'='*104}")
    s = df[df.same_side].copy()
    bins = [0, .02, .04, .06, .08, .10, .15, .20, 1.0]
    s["b"] = pd.cut(s.edge, bins, include_lowest=True)
    rows = []
    rng = np.random.default_rng(SEED)
    print(f"{'edge band':>16} {'n':>5} {'claimed p':>10} {'mkt p':>8} "
          f"{'realised':>9} {'claimed dp':>11} {'REALISED dp':>12} "
          f"{'realised dp 95% CI':>22} {'flat ROI%':>10} {'fairROI%':>9}")
    for b, g in s.groupby("b"):
        if len(g) < 20:
            continue
        realdp = g.hit.mean() - g.p_mkt_side.mean()
        h = g.hit.values.astype(float)
        pm = g.p_mkt_side.values
        idx = rng.integers(0, len(g), (N_BOOT, len(g)))
        bs = h[idx].mean(axis=1) - pm[idx].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        pnl = np.where(g.hit.astype(bool), g.dec - 1, -1.0)
        pnlf = np.where(g.hit.astype(bool), g.dec_fair - 1, -1.0)
        print(f"{str(b):>16} {len(g):>5} {g.p_us_side.mean():>10.4f} "
              f"{g.p_mkt_side.mean():>8.4f} {g.hit.mean():>9.4f} "
              f"{g.edge.mean():>+11.4f} {realdp:>+12.4f} "
              f"[{lo:+.4f},{hi:+.4f}]".rjust(0) +
              f" {100*pnl.mean():>10.2f} {100*pnlf.mean():>9.2f}")
        rows.append({"band": str(b), "n": len(g),
                     "claimed_p": float(g.p_us_side.mean()),
                     "mkt_p": float(g.p_mkt_side.mean()),
                     "realised": float(g.hit.mean()),
                     "claimed_dp": float(g.edge.mean()),
                     "realised_dp": float(realdp),
                     "lo": float(lo), "hi": float(hi),
                     "roi_vig": float(pnl.mean()),
                     "roi_fair": float(pnlf.mean())})
    res["reliability"] = rows
    # regression of realised excess on claimed excess (the Kelly-slope test)
    b1, b0 = np.polyfit(s.edge, s.hit - s.p_mkt_side, 1)
    n = len(s)
    r = (s.hit - s.p_mkt_side) - (b0 + b1 * s.edge)
    se = np.sqrt((r ** 2).sum() / (n - 2) / ((s.edge - s.edge.mean()) ** 2).sum())
    print(f"\n  KELLY-SLOPE TEST  realised_excess = {b0:+.4f} + {b1:+.4f} x "
          f"claimed_excess   (se {se:.4f}, t={b1/se:+.2f}, n={n})")
    print(f"  slope 1.0 == our stated edge is fully real; slope 0 == our "
          f"stated edge carries NO information about realised excess.")
    res["kelly_slope"] = {"b0": float(b0), "b1": float(b1), "se": float(se),
                          "t": float(b1 / se), "n": int(n)}


# ------------------------------------------------------------------ B
def rule_family(df: pd.DataFrame) -> dict:
    r = {}
    for t in (0.02, 0.04):
        base = df.same_side & (df.edge > t)
        r[f"R1(t={t:.2f})"] = base
        r[f"R4_LOWT(t={t:.2f})"] = base & df.late          # D75 primary
    for c in (0.20,):
        b = df.same_side & (df.conf_us > c) & (df.edge >= 0.03)
        r[f"T{int(c*100)}.D03+"] = b
        r[f"T{int(c*100)}.D03+W"] = b & df.late
    return r


def boot_delta(pnl_unc: np.ndarray, keep: np.ndarray) -> dict:
    """Paired bootstrap on the per-bet PnL of the UNCAPPED set: capped arm
    scores 0 on skipped bets (capital not deployed).  Positive = cap better."""
    d = np.where(keep, 0.0, -pnl_unc)     # cap removes that bet's PnL
    rng = np.random.default_rng(SEED)
    if len(d) == 0:
        return {"n": 0, "mean": 0.0, "lo": 0.0, "hi": 0.0, "verdict": "EMPTY"}
    idx = rng.integers(0, len(d), (N_BOOT, len(d)))
    m = d[idx].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return {"n": int(len(d)), "mean": float(d.mean()), "lo": float(lo),
            "hi": float(hi), "verdict": "SIG" if lo > 0 or hi < 0 else "NS"}


def cap_sweep(df: pd.DataFrame, label: str, res: dict) -> None:
    print(f"\n{'='*118}")
    print(f"B. CONFIDENCE-EXCESS CAP SWEEP — {label}")
    print("   skip a bet when conf_us - conf_mkt > X  (== edge > X on "
          "same-side bets).  X=inf is the uncapped rule as registered.")
    print(f"{'='*118}")
    rules = rule_family(df)
    out = []
    for name, mask in rules.items():
        sub_all = df[mask].sort_values("game_date")
        if len(sub_all) < 15:
            continue
        print(f"\n  {name}   uncapped n={len(sub_all)}")
        print(f"    {'cap X':>7} {'window':>7} {'n':>5} {'hit%':>7} "
              f"{'ROI%':>8} {'ROIfair%':>9} {'PnL(u)':>9} {'sharpe':>8} "
              f"{'qk ROI%':>9} {'qk PnL':>9}  {'dPnL/bet vs uncapped (95% CI)'}")
        for X in CAPS:
            keep_all = (sub_all.conf_gap <= X).values
            for win, sel in (("IS", sub_all.season.isin(IS_SEASONS)),
                             ("OOS", sub_all.season == OOS_SEASON),
                             ("POOL", pd.Series(True, index=sub_all.index))):
                sub = sub_all[sel.values & keep_all]
                unc = sub_all[sel.values]
                f = score(sub, "flat")
                k = score(sub, "qkelly")
                if np.isinf(X):
                    bd = {"mean": 0.0, "lo": 0.0, "hi": 0.0, "verdict": "base"}
                else:
                    pnl_unc = np.where(unc.hit.astype(bool),
                                       unc.dec - 1, -1.0)
                    bd = boot_delta(pnl_unc, keep_all[sel.values])
                xs = "inf" if np.isinf(X) else f"{X:.2f}"
                print(f"    {xs:>7} {win:>7} {f['n']:>5} "
                      f"{(f['hit']*100 if f['n'] else 0):>7.1f} "
                      f"{(f['roi']*100 if f['n'] else 0):>8.2f} "
                      f"{(f['roi_fair']*100 if f['n'] else 0):>9.2f} "
                      f"{f['pnl']:>9.2f} "
                      f"{(f['sharpe'] if f['n'] > 1 else 0):>8.3f} "
                      f"{(k['roi']*100 if k['n'] else 0):>9.2f} "
                      f"{k['pnl']:>9.2f}  "
                      f"{bd['mean']:+.4f} [{bd['lo']:+.4f},{bd['hi']:+.4f}] "
                      f"{bd['verdict']}")
                out.append({"frame": label, "rule": name, "cap": xs,
                            "window": win, "n": f["n"],
                            "hit": f["hit"], "roi": f["roi"],
                            "roi_fair": f["roi_fair"], "pnl": f["pnl"],
                            "sharpe": f["sharpe"], "qk_roi": k["roi"],
                            "qk_pnl": k["pnl"], "dpnl": bd["mean"],
                            "lo": bd["lo"], "hi": bd["hi"],
                            "verdict": bd["verdict"]})
    res.setdefault("cap_sweep", []).extend(out)


def main() -> None:
    res: dict = {}
    print("#" * 118)
    print("PRIMARY FRAME — ds_rt1_pergame.csv, p_full (current production), "
          "4 seasons 2022-23..2025-26")
    print("#" * 118)
    df = build(RT1, "p_full")
    res["n_primary"] = len(df)
    reliability(df, res)
    cap_sweep(df, "rt1 p_full 4-season", res)

    print("\n" + "#" * 118)
    print("REPLICATION FRAME — capstone_pergame_tank.csv, p_us "
          "(the exact D75/D78 sim frame), 3 seasons")
    print("#" * 118)
    df2 = build(TANK, "p_us")
    res["n_replication"] = len(df2)
    r2: dict = {}
    reliability(df2, r2)
    res["replication_reliability"] = r2.get("reliability")
    res["replication_kelly_slope"] = r2.get("kelly_slope")
    cap_sweep(df2, "capstone_tank 3-season", res)

    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
