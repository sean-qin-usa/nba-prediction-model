#!/usr/bin/env python3
"""BET-SIM — betting simulator with strict IS/OOS discipline (Deliverable 1).

DATA: data/capstone_pergame_carry2.csv (p_us = our model home win prob,
p_mkt = DE-VIG CLOSE derived from the spread) + data/nba.duckdb (read_only)
for games-played (W6) and star-out flags (game_inactives join).

PRICING ASSUMPTIONS (stated up front, no line shopping, always get the close):
  * p_mkt is the de-vig CLOSE — the sharpest publicly available number.
    Beating it AFTER re-adding vig requires real edge, not noise.
  * Moneyline prices are reconstructed from p_mkt with a 4.5% total
    overround allocated proportionally (multiplicatively):
        q_side = p_mkt_side * 1.045,  decimal_side = 1 / q_side
    At a pick'em this gives 1.914 decimal = -109.4 American, i.e. the
    standard -110/-110 two-way market; at short favorite prices it produces
    realistic ML shading. Decimal odds are floored at 1.01 (books never pay
    less); the count of floored bets is reported.
  * Betting the "shared side" (both models on the same side of 0.5) is by
    construction betting the MARKET FAVORITE at short odds: hit rate alone
    is meaningless (breakeven at the vigged favorite price is
    p_mkt_side*1.045, e.g. 68.9% at a 66% close, not 52.38%).
  * ROI is also reported at the FAIR (no-vig) close, decimal = 1/p_mkt_side,
    to decompose "no real edge vs the close" from "edge eaten by vig".

PRE-REGISTERED RULE FAMILY (ALL 12 evaluated in-sample; no post-hoc adds):
  R1(t): same-side game AND (p_us_side - p_mkt_side) > t, t in {.02,.04,.06}
  R2(t): R1(t) AND W6 early-season window (either team gp < 20 —
         nbapred.market.windows.EARLY_SEASON_GP, D66 pre-registration)
  R3(t): R1(t) AND star-out game (either team has an inactive with trailing
         avg minutes >= 28 over last 10 games with 12+ min, strictly before
         the game — scripts/ba_intersection.star_out_map, game_inactives join)
  R4(t): R1(t) AND late season (game month in {3,4} — ba_intersection
         convention)
  Opposite-side games are NEVER bet (known net-negative). Taxonomy anchor
  (full 3 seasons): same-side-we-sharper n=981 hit 67.6%.

SIZING (both reported for every rule):
  * FLAT: 1u per bet.
  * QUARTER-KELLY on the edge p_us_side vs the VIGGED offered odds:
    f* = (p_us_side*d - 1)/(d - 1); stake = 0.25*f* of a FIXED 100u
    reference bankroll (non-compounding), capped at 10u; bets with f* <= 0
    (model edge does not clear the vig) are SKIPPED — so Kelly n_bets < flat
    n_bets by construction.

IS/OOS DISCIPLINE (pre-registered selection procedure, mechanical):
  * IN-SAMPLE = 2023-24 + 2024-25. All 12 rules evaluated IS.
  * SELECTION: top 2 rules by IS Sharpe-like (mean/sd of per-bet FLAT unit
    PnL) among rules with IS n_bets >= 40 (below that the Sharpe is noise).
  * OOS = 2025-26, scored ONLY for the selected rules, untouched otherwise.
  * HONESTY CAVEAT: the rule FAMILY was motivated by the win-vs-market
    taxonomy, whose aggregate counts (n=981/67.6%) were computed on all 3
    seasons. OOS is clean w.r.t. threshold/family SELECTION (done on IS
    only), but not w.r.t. the discovery that the shared-side-sharper class
    exists. Treat OOS numbers as validation of the selection, not discovery.

RULES HONORED: DuckDB read_only=True; new file scripts/bet_sim.py only;
nothing in nbapred/ or existing scripts edited (helpers are IMPORTED from
scripts/ba_intersection.py, not copied). Deterministic (no RNG).

Run:  python scripts/bet_sim.py
"""
from __future__ import annotations

import os
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from ba_intersection import team_schedule, star_out_map          # noqa: E402
from nbapred.market.windows import EARLY_SEASON_GP               # noqa: E402

DB = os.path.join(ROOT, "data", "nba.duckdb")
CSV = os.path.join(ROOT, "data", "capstone_pergame_carry2.csv")

IS_SEASONS = {"2023-24", "2024-25"}
OOS_SEASON = "2025-26"
OVERROUND = 1.045
MIN_DEC = 1.01
THRESHOLDS = (0.02, 0.04, 0.06)
KELLY_FRAC = 0.25
BANKROLL = 100.0
KELLY_CAP = 10.0
MIN_BETS_IS = 40
N_SELECT = 2


def build_frame() -> pd.DataFrame:
    df = pd.read_csv(CSV, dtype={"game_id": str})
    df["game_date"] = pd.to_datetime(df.game_date)
    con = duckdb.connect(DB, read_only=True)
    try:
        ts = team_schedule(con)
        so = star_out_map(con)
    finally:
        con.close()

    keys = ["season", "game_id"]
    for side in ("home", "away"):
        m = ts.rename(columns={"team": side})
        df = df.merge(m[keys + [side, "gp_before"]], on=keys + [side],
                      how="left")
        df = df.rename(columns={"gp_before": f"{side[0]}_gp"})
        s = so.rename(columns={"team": side})
        df = df.merge(s[["game_id", side, "star_out"]], on=["game_id", side],
                      how="left")
        df = df.rename(columns={"star_out": f"{side[0]}_star_out"})
        df[f"{side[0]}_star_out"] = (df[f"{side[0]}_star_out"]
                                     .astype("boolean").fillna(False)
                                     .astype(bool))
    assert df.h_gp.notna().all() and df.a_gp.notna().all(), "schedule join gap"

    # side pick + edge on the picked side
    df["pick_home"] = df.p_us > 0.5
    df["same_side"] = (df.p_us - 0.5) * (df.p_mkt - 0.5) > 0
    df["p_us_side"] = np.where(df.pick_home, df.p_us, 1 - df.p_us)
    df["p_mkt_side"] = np.where(df.pick_home, df.p_mkt, 1 - df.p_mkt)
    df["edge"] = df.p_us_side - df.p_mkt_side
    df["hit"] = np.where(df.pick_home, df.y == 1, df.y == 0)

    # offered odds (vigged) + fair close odds
    q = df.p_mkt_side * OVERROUND
    df["dec"] = np.maximum(1.0 / q, MIN_DEC)
    df["floored"] = (1.0 / q) < MIN_DEC
    df["dec_fair"] = 1.0 / df.p_mkt_side

    # window flags
    df["w6"] = np.minimum(df.h_gp, df.a_gp) < EARLY_SEASON_GP
    df["star_any"] = df.h_star_out.astype(bool) | df.a_star_out.astype(bool)
    df["late"] = df.game_date.dt.month.isin([3, 4])
    return df


def rule_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    rules = {}
    for t in THRESHOLDS:
        base = df.same_side & (df.edge > t)
        rules[f"R1(t={t:.2f})"] = base
        rules[f"R2(t={t:.2f})"] = base & df.w6
        rules[f"R3(t={t:.2f})"] = base & df.star_any
        rules[f"R4(t={t:.2f})"] = base & df.late
    return rules


def max_drawdown(cum: np.ndarray) -> float:
    if len(cum) == 0:
        return 0.0
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))
    return float(np.max(peak - np.concatenate([[0.0], cum])))


def score(sub: pd.DataFrame, sizing: str) -> dict:
    """sub must be date-sorted. sizing in {'flat','qkelly'}."""
    if sizing == "qkelly":
        fstar = (sub.p_us_side * sub.dec - 1) / (sub.dec - 1)
        stake = np.minimum(KELLY_FRAC * fstar * BANKROLL, KELLY_CAP)
        keep = stake > 0
        sub, stake = sub[keep], stake[keep].values
    else:
        stake = np.ones(len(sub))
    if len(sub) == 0:
        return dict(n=0, staked=0.0, pnl=0.0, roi=np.nan, hit=np.nan,
                    mean=np.nan, sd=np.nan, sharpe=np.nan, maxdd=np.nan,
                    roi_fair=np.nan, curve=np.array([]), floored=0)
    pnl = np.where(sub.hit, stake * (sub.dec - 1), -stake)
    pnl_fair = np.where(sub.hit, stake * (sub.dec_fair - 1), -stake)
    cum = np.cumsum(pnl)
    sd = float(pnl.std(ddof=1)) if len(pnl) > 1 else np.nan
    return dict(
        n=len(sub), staked=float(stake.sum()), pnl=float(pnl.sum()),
        roi=float(pnl.sum() / stake.sum()), hit=float(sub.hit.mean()),
        mean=float(pnl.mean()), sd=sd,
        sharpe=float(pnl.mean() / sd) if sd and sd > 0 else np.nan,
        maxdd=max_drawdown(cum),
        roi_fair=float(pnl_fair.sum() / stake.sum()),
        curve=cum, floored=int(sub.floored.sum()))


def fmt_row(name, sz, r):
    return (f"{name:<12}{sz:<8}{r['n']:>6}{r['hit']*100 if r['n'] else 0:>7.1f}"
            f"{r['roi']*100:>8.2f}{r['roi_fair']*100:>10.2f}"
            f"{r['pnl']:>9.2f}{r['staked']:>9.1f}"
            f"{r['sharpe']:>8.3f}{r['maxdd']:>8.2f}")


HDR = (f"{'rule':<12}{'sizing':<8}{'n':>6}{'hit%':>7}{'ROI%':>8}"
       f"{'ROI%fair':>10}{'PnL(u)':>9}{'staked':>9}{'sharpe':>8}{'maxDD':>8}")


def print_curve(label, cum, per_line=15):
    vals = [f"{v:+.1f}" for v in cum]
    print(f"  {label} cumulative PnL (u) after each bet [{len(vals)} bets]:")
    for i in range(0, len(vals), per_line):
        print("    " + " ".join(vals[i:i + per_line]))


def main():
    df = build_frame().sort_values(["game_date", "game_id"]).reset_index(drop=True)
    rules = rule_masks(df)
    is_m = df.season.isin(IS_SEASONS)
    oos_m = df.season == OOS_SEASON

    # taxonomy anchor cross-check (full 3 seasons — matches the known counts)
    sharper = df.same_side & (df.edge > 0)
    print("=" * 96)
    print("BET-SIM — pre-registered rule family, IS=2023-24+2024-25, "
          "OOS=2025-26 (untouched until selection)")
    print("=" * 96)
    print(f"taxonomy anchor: same-side-we-sharper n={int(sharper.sum())} "
          f"hit={df.hit[sharper].mean()*100:.1f}%  (expect 981 / 67.6%)  |  "
          f"opposite-side n={int((~df.same_side).sum())} NEVER BET")
    print(f"vig model: overround {OVERROUND} proportional -> pick'em decimal "
          f"{1/(0.5*OVERROUND):.3f} (~-110); odds floored at {MIN_DEC} on "
          f"{int(df.floored.sum())} of {len(df)} games (heaviest favorites)")
    print(f"avg vig cost on the shared side: "
          f"{(1/df.p_mkt_side - df.dec).mul(df.p_mkt_side).mean()*100:.2f}% "
          f"of stake per bet (fair-minus-offered payout x fair win prob)")

    # ---- IN-SAMPLE: all 12 rules x 2 sizings --------------------------------
    print("\n" + "-" * 96)
    print(f"IN-SAMPLE (n games={int(is_m.sum())})")
    print(HDR)
    is_res = {}
    for name, mask in rules.items():
        sub = df[mask & is_m]
        for sz in ("flat", "qkelly"):
            r = score(sub, sz)
            is_res[(name, sz)] = r
            print(fmt_row(name, sz, r))

    # ---- SELECTION (mechanical, pre-registered) -----------------------------
    elig = [(n, r["sharpe"]) for (n, sz), r in is_res.items()
            if sz == "flat" and r["n"] >= MIN_BETS_IS
            and np.isfinite(r["sharpe"])]
    elig.sort(key=lambda x: -x[1])
    selected = [n for n, _ in elig[:N_SELECT]]
    print("\n" + "-" * 96)
    print(f"SELECTION (top {N_SELECT} by IS flat Sharpe, n>=%d): %s"
          % (MIN_BETS_IS, ", ".join(selected) if selected else "NONE eligible"))
    for n in selected:
        r = is_res[(n, "flat")]
        print(f"  {n}: IS sharpe={r['sharpe']:+.3f} ROI={r['roi']*100:+.2f}% "
              f"n={r['n']} hit={r['hit']*100:.1f}%")
        print_curve(f"{n} IS flat", is_res[(n, 'flat')]["curve"])

    # ---- OOS: selected rules ONLY -------------------------------------------
    print("\n" + "-" * 96)
    print(f"OUT-OF-SAMPLE {OOS_SEASON} (n games={int(oos_m.sum())}) — "
          "selected rules only")
    print(HDR)
    verdict_profitable = []
    for name in selected:
        sub = df[rules[name] & oos_m]
        for sz in ("flat", "qkelly"):
            r = score(sub, sz)
            print(fmt_row(name, sz, r))
            if r["n"] > 0 and r["roi"] > 0:
                verdict_profitable.append((name, sz, r))
        print_curve(f"{name} OOS flat", score(sub, "flat")["curve"])

    # ---- VERDICT ------------------------------------------------------------
    print("\n" + "=" * 96)
    print("VERDICT")
    if verdict_profitable:
        for name, sz, r in verdict_profitable:
            print(f"  {name} [{sz}] is OOS-profitable at the -110-equivalent "
                  f"vig: ROI {r['roi']*100:+.2f}% on n={r['n']} bets "
                  f"(PnL {r['pnl']:+.2f}u). Small n — treat as fragile until "
                  "live CLV confirms (D66 protocol).")
    else:
        print("  NO selected rule is OOS-profitable at -110-equivalent "
              "pricing. The model's shared-side edge over the de-vig close "
              "is smaller than the ~4.4% vig at favorite prices.")
    print("  Honesty: p_mkt is the de-vig CLOSE — ROI%fair above shows the "
          "edge vs the fair close; the gap ROI%fair-ROI% is pure vig. "
          "No line shopping, no CLV capture, close always assumed available.")
    print("  Selection was mechanical on IS flat Sharpe; OOS untouched by "
          "selection. Family-level caveat: the taxonomy motivating the "
          "family was computed on all 3 seasons (see docstring).")
    print("  Coverage caveat: game_inactives covers 2025-26 for only "
          "20/1230 games, so R3 (star-out) is only evaluable IS; had R3 "
          "been selected, OOS scoring would have been impossible. IS "
          "(2023-24 full, 2024-25 1195/1230) is valid.")


if __name__ == "__main__":
    main()
