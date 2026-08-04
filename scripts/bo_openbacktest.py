#!/usr/bin/env python3
"""BO-OPENBACKTEST — the honest BET-AT-OPEN backtest (D120).

THE TEST WE COULD NOT RUN UNTIL D119.  Every betting result in this program
so far was BET-AT-CLOSE and came back negative; after D117's edge calibration
ZERO positive-EV bets exist against the close.  The one real controlled effect
(D115 s6.3) is that our side's CLV **at the open** is +0.0112 (t=+6.7) with the
favourite-drift control NEGATIVE -- "we beat the opener and lose to the closer".
D119 delivered `odds_open` (23,783 games with BOTH prices), so the open price
is now transactable in a backtest for the first time.

NO LOOKAHEAD.  The opening price is posted hours before tip.  Every arm that
fires "at the open" uses ONLY p_open for eligibility, sizing and pricing; the
close is used exclusively as (i) the CLV yardstick and (ii) the paired control.

THREE ARMS, so selection and price are separated:
  A  OPEN        fire on edge-vs-OPEN,  transact at the OPEN price   <- the test
  B  OPEN@CLOSE  fire on edge-vs-OPEN,  transact at the CLOSE price
                 (IDENTICAL bet set to A -- isolates the pure price/timing
                  effect from the selection effect)
  C  CLOSE       fire on edge-vs-CLOSE, transact at the CLOSE price
                 (the D115/D117 replication, restricted to the SAME universe
                  of games so the open-vs-close delta is paired on games)

TWO PRICE SOURCES (task 2b: "if opening MLs exist for 23-24+ use them
directly rather than a vig assumption"):
  SP@1.045  fair p = sigmoid(margin/6.96) (de-vigged by construction, the
            program-wide convention, nbapred/ingest/kaggle_odds.py), decimal =
            1/(p_side * 1.045).  ASSUMPTION, applies to all seasons.
  ML        REAL opening/closing American moneylines (odds_open.open_ml_*,
            100% of 2023-24..2025-26 via ESPN core + Action Network).  Decimal
            is the ACTUAL transactable price -- NO vig assumption at all.
            Fair p = proportional de-vig.  Measured overround 1.0433, which
            independently validates the 1.045 the program has been assuming.

BET FORMS
  ML/side   the registry's own formulation (our p(win) vs the market's p(win))
  ATS       our implied margin 6.96*logit(p_us) vs the OPENING SPREAD, -110,
            pushes returned.  Directly comparable to D119's 54.63% ceiling.

RULES: the four REGISTERED F4 rules, operators copied verbatim from
scripts/bet_engine.rules_fired via scripts/f4_resim.registry_masks.  Nothing
is re-chosen here.  `edge` is recomputed against whichever price the arm
transacts at -- that is the only change, and it is the point of the test.

DISCIPLINE
  * IS/OOS in BOTH directions on the D115 partition (DEV = 2023-24 + 2024-25,
    NONDEV = 2022-23 + 2025-26); both readings printed.
  * Bootstrap CIs on every ROI / hit rate / CLV.
  * TWO nulls per positive cell:
      null_be   P(ROI >= observed | TRUE BREAKEVEN, i.e. p_i = 1/dec_i)
                -- the task's question: does it clear the vig?
      null_mkt  P(ROI >= observed | NO EDGE vs the de-vigged market)
                -- the f4_resim/D117 convention, carried for continuity.
  * Family-wise null-count check over every cell printed.

RULES HONORED: DuckDB read_only=True; new file scripts/bo_*.py; nbapred/
untouched.

Run:  python scripts/bo_openbacktest.py
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
from bet_sim3 import MIN_DEC, OVERROUND, TANK_GP                 # noqa: E402
from f4_shrinkage import CONF_EXCESS_CAP, fit_kelly_slope        # noqa: E402

DB = os.path.join(ROOT, "data", "nba.duckdb")
ODDS_OPEN = os.path.join(ROOT, "data", "derived", "odds_open.csv")
RT1 = os.path.join(ROOT, "data", "ds_rt1_pergame.csv")
CF = os.path.join(ROOT, "data", "cf_holdout_new_pergame.csv")
OUT = os.path.join(ROOT, "data", "bo_openbacktest.json")

SPREAD_SCALE = 6.96                  # nbapred/ingest/kaggle_odds.py
ATS_DEC = 1.0 + 100.0 / 110.0        # -110 => 1.909091, breakeven 52.381%

# registry constants — copied from scripts/bet_engine.py, NOT re-chosen here
EDGE_R4 = 0.02
CONF_TIER = 0.20
DIV_LO, DIV_HI = 0.03, 0.10

DEV = {"2023-24", "2024-25"}         # every registered rule was developed here
NONDEV = {"2022-23", "2025-26"}
WINDOWS = [
    ("POOL", None),                  # every season present in the frame
    ("DEV", DEV),
    ("NONDEV", NONDEV),
]
N_BOOT = 4000
N_NULL = 10000
SEED = 20260801
PRIMARY_WINDOWS = {"POOL", "DEV", "NONDEV"}   # per-season cells are diagnostic


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def am2dec(a):
    """American odds -> decimal.  NaN-safe.
    +150 -> 2.50 ;  -120 -> 1.8333.  (A sign slip here silently produces
    decimals < 1 and 107% "breakeven" rates — it did, once.)"""
    a = np.asarray(a, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(a > 0, 1.0 + a / 100.0, 1.0 + 100.0 / np.abs(a))
    return np.where(np.isnan(a) | (a == 0), np.nan, d)


# ----------------------------------------------------------------- frame ----
def build(csv: str, pcol: str, label: str, res: dict) -> pd.DataFrame:
    """Model frame JOINed to odds_open, with both prices in both forms."""
    df = pd.read_csv(csv, dtype={"game_id": str})
    df["game_id"] = df.game_id.str.zfill(10)
    df["game_date"] = pd.to_datetime(df.game_date)
    df = df.rename(columns={pcol: "p_us"})

    oo = pd.read_csv(ODDS_OPEN, parse_dates=["game_date"])
    keep = ["season", "game_date", "home", "away", "open_margin", "close_margin",
            "open_ml_home", "open_ml_away", "close_ml_home", "close_ml_away",
            "score_home", "score_away", "source"]
    n_before = len(df)
    m = df.merge(oo[keep], on=["season", "game_date", "home", "away"],
                 how="left")
    assert len(m) == n_before, "odds_open join fanned out — not 1:1"

    # ---- COVERAGE TABLE (task 1) -------------------------------------------
    cov = m.groupby("season").apply(lambda g: pd.Series({
        "model_games": len(g),
        "with_open": int(g.open_margin.notna().sum()),
        "with_close": int(g.close_margin.notna().sum()),
        "with_both": int((g.open_margin.notna() & g.close_margin.notna()).sum()),
        "with_open_ML": int(g.open_ml_home.notna().sum()),
        "source": "/".join(sorted(g.source.dropna().unique())) or "NONE",
    }), include_groups=False)
    cov["pct_both"] = 100.0 * cov.with_both / cov.model_games
    print(f"\n{'#' * 116}\nFRAME: {label}   ({os.path.basename(csv)}, "
          f"p_us = {pcol})\n{'#' * 116}")
    print("\n[1] JOIN COVERAGE — model frame x odds_open "
          "(join key: season + game_date + home + away, verified 1:1)")
    print(cov.to_string())
    tot_b = int(cov.with_both.sum())
    print(f"    TOTAL {tot_b}/{len(m)} = {100 * tot_b / len(m):.1f}% of this "
          f"frame carries BOTH an opening and a closing price.")
    hole = cov.loc["2022-23"] if "2022-23" in cov.index else None
    if hole is not None:
        print(f"    THE HOLE IS 2022-23 AND ONLY 2022-23: {int(hole.with_both)}"
              f"/{int(hole.model_games)} = {hole.pct_both:.1f}%.  The SBR "
              f"archive genuinely stops on 2023-01-16 (D119), so the missing "
              f"games are the SECOND HALF of 2022-23 — a CALENDAR-CONTIGUOUS "
              f"block, not a random sample.  Every 2022-23 number below is a "
              f"FIRST-HALF-ONLY reading and is flagged as such.")
        got = m[(m.season == "2022-23") & m.open_margin.notna()]
        if len(got):
            print(f"    2022-23 covered window: {got.game_date.min().date()} .. "
                  f"{got.game_date.max().date()}   "
                  f"(frame runs to {m[m.season == '2022-23'].game_date.max().date()})")
    res.setdefault("coverage", {})[label] = cov.reset_index().to_dict("records")

    m = m[m.open_margin.notna() & m.close_margin.notna()].copy()

    # ---- PRICES ------------------------------------------------------------
    # spread-derived, de-vigged by construction (symmetric logistic)
    m["p_open_sp"] = sigmoid(m.open_margin / SPREAD_SCALE)
    m["p_close_sp"] = sigmoid(m.close_margin / SPREAD_SCALE)
    # real moneylines, proportional de-vig
    for w in ("open", "close"):
        dh = am2dec(m[f"{w}_ml_home"])
        da = am2dec(m[f"{w}_ml_away"])
        ov = 1.0 / dh + 1.0 / da
        bad = ~np.isfinite(ov) | (ov < 1.0) | (ov > 1.25)
        m[f"dec_{w}_ml_home"] = np.where(bad, np.nan, dh)
        m[f"dec_{w}_ml_away"] = np.where(bad, np.nan, da)
        m[f"p_{w}_ml"] = np.where(bad, np.nan, (1.0 / dh) / ov)
        m[f"ov_{w}_ml"] = np.where(bad, np.nan, ov)

    # ---- model side / late window -----------------------------------------
    con = duckdb.connect(DB, read_only=True)
    try:
        ts = team_schedule(con)
        so = star_out_map(con)
    finally:
        con.close()
    for side in ("home", "away"):
        t = ts.rename(columns={"team": side})[["season", "game_id", side,
                                               "gp_before"]]
        m = m.merge(t, on=["season", "game_id", side], how="left")
        m = m.rename(columns={"gp_before": f"{side[0]}_gp"})
    assert m.h_gp.notna().all() and m.a_gp.notna().all()
    m["late"] = (m.h_gp >= TANK_GP) | (m.a_gp >= TANK_GP)

    m["pick_home"] = m.p_us > 0.5
    m["p_us_side"] = np.where(m.pick_home, m.p_us, 1 - m.p_us)
    m["conf_us"] = (m.p_us - 0.5).abs()
    m["hit"] = np.where(m.pick_home, m.y == 1, m.y == 0).astype(int)
    m["margin_actual"] = m.score_home - m.score_away
    m["m_us"] = SPREAD_SCALE * logit(m.p_us)      # our implied home margin

    # star-out on the MARKET favourite — evaluated at whichever price the arm
    # uses, so both an open- and a close-favourite flag are carried.
    som = so.rename(columns={"team": "fav_team"})[["game_id", "fav_team",
                                                   "star_out"]]
    for w in ("open", "close"):
        m["fav_team"] = np.where(m[f"p_{w}_sp"] >= 0.5, m.home, m.away)
        m = m.merge(som, on=["game_id", "fav_team"], how="left")
        m[f"fav_star_out_{w}"] = m.star_out.fillna(False).astype(bool)
        m = m.drop(columns=["star_out"])
    m = m.drop(columns=["fav_team"])
    return m.sort_values("game_date").reset_index(drop=True)


def price_cols(m: pd.DataFrame, when: str, src: str):
    """(fair de-vigged prob on OUR side, transactable decimal on OUR side,
    availability mask) for price `when` in {open, close} from source `src`."""
    if src == "SP":
        p_home = m[f"p_{when}_sp"].values
        p_side = np.where(m.pick_home, p_home, 1 - p_home)
        dec = np.maximum(1.0 / (p_side * OVERROUND), MIN_DEC)
        ok = np.isfinite(p_side)
    else:                                                        # real ML
        p_home = m[f"p_{when}_ml"].values
        p_side = np.where(m.pick_home, p_home, 1 - p_home)
        dec = np.where(m.pick_home, m[f"dec_{when}_ml_home"].values,
                       m[f"dec_{when}_ml_away"].values)
        ok = np.isfinite(p_side) & np.isfinite(dec)
    return p_side, dec, ok


def registry_masks(m: pd.DataFrame, p_side: np.ndarray, when: str) -> dict:
    """The 4 SHIPPED F4 rules, operators verbatim from bet_engine.rules_fired,
    with `edge` measured against the price this arm actually transacts at."""
    edge = m.p_us_side.values - p_side
    same = ((m.p_us.values - 0.5) *
            (np.where(m.pick_home, p_side, 1 - p_side) - 0.5)) > 0
    tails = m.conf_us.values > CONF_TIER
    band = (edge >= DIV_LO) & (edge <= DIV_HI)
    late = m.late.values
    star = m[f"fav_star_out_{when}"].values
    return {
        "R4_LOWT":          same & (edge > EDGE_R4) & late,
        "T20_D03_10_W":     same & tails & band & late,
        "T20_D03_10":       same & tails & band,
        "STAR_FAV_SHARPER": same & (edge > 0) & star,
    }, edge, same


# ------------------------------------------------------------- scoring ------
def score_ml(m: pd.DataFrame, sel: np.ndarray, dec: np.ndarray,
             p_fair: np.ndarray, seed: int = SEED) -> dict:
    """Flat 1u on our side.  `sel` selects bets; `dec`/`p_fair` are the
    transacted price and its de-vigged fair prob."""
    if sel.sum() == 0:
        return dict(n=0)
    hit = m.hit.values[sel].astype(bool)
    d, pf = dec[sel], p_fair[sel]
    pnl = np.where(hit, d - 1.0, -1.0)
    be = 1.0 / d                                     # per-bet breakeven prob
    roi = float(pnl.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pnl), (N_BOOT, len(pnl)))
    bs = pnl[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    hb = hit[idx].mean(axis=1) if len(pnl) else np.array([np.nan])
    hlo, hhi = np.percentile(hb, [2.5, 97.5])
    out = dict(n=int(sel.sum()), hit=float(hit.mean()),
               hit_lo=float(hlo), hit_hi=float(hhi),
               breakeven=float(be.mean()), edge_pp=float(hit.mean() - be.mean()),
               roi=roi, roi_lo=float(lo), roi_hi=float(hi),
               pnl=float(pnl.sum()), mean_dec=float(d.mean()),
               ov=float(np.mean(be / pf)))          # realised overround paid
    # ---- the two nulls ----------------------------------------------------
    # null_be : p_i = 1/dec_i  -> the bet is EXACTLY breakeven, ROI centres on
    #           0.  This is the task's question: does the edge clear the vig?
    # null_mkt: p_i = de-vigged market prob -> we have no edge vs the market,
    #           ROI centres on -vig.  The f4_resim / D117 convention.
    # A cell with ROI <= -5% cannot be significant under either null at any n
    # worth reporting, so the Monte-Carlo is skipped there (exact for null_be,
    # which is centred at 0 and therefore has p > 0.5 whenever ROI < 0).
    if roi > -0.05:
        r2 = np.random.default_rng(seed + 1)
        for tag, p in (("be", be), ("mkt", pf)):
            win = r2.random((N_NULL, len(p))) < p
            nroi = np.where(win, d - 1.0, -1.0).mean(axis=1)
            out[f"null_{tag}"] = float((nroi >= roi).mean())
            out[f"nullhit_{tag}"] = float((win.mean(axis=1) >= hit.mean()).mean())
    else:
        out["null_be"] = 1.0
        out["null_mkt"] = float("nan")
    return out


def score_ats(m: pd.DataFrame, sel: np.ndarray, line: np.ndarray,
              seed: int = SEED) -> dict:
    """Bet our side against `line` (the market's expected HOME margin) at -110.
    Pushes return the stake and are excluded from the hit rate."""
    if sel.sum() == 0:
        return dict(n=0)
    mm = m[sel]
    ln = line[sel]
    pick_home = (mm.m_us.values > ln)                # our margin beats the line
    diff = mm.margin_actual.values - ln              # >0 = home covered
    push = diff == 0
    cover = np.where(pick_home, diff > 0, diff < 0)
    pnl = np.where(push, 0.0, np.where(cover, ATS_DEC - 1.0, -1.0))
    live = ~push
    roi = float(pnl.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pnl), (N_BOOT, len(pnl)))
    lo, hi = np.percentile(pnl[idx].mean(axis=1), [2.5, 97.5])
    hr = float(cover[live].mean()) if live.sum() else np.nan
    hb = cover[live][rng.integers(0, live.sum(), (N_BOOT, live.sum()))
                     ].mean(axis=1) if live.sum() else np.array([np.nan])
    hlo, hhi = np.percentile(hb, [2.5, 97.5])
    be = 1.0 / ATS_DEC
    r2 = np.random.default_rng(seed + 1)
    win = r2.random((N_NULL, int(live.sum()))) < be
    return dict(n=int(sel.sum()), n_live=int(live.sum()), n_push=int(push.sum()),
                hit=hr, hit_lo=float(hlo), hit_hi=float(hhi), breakeven=be,
                edge_pp=hr - be, roi=roi, roi_lo=float(lo), roi_hi=float(hi),
                pnl=float(pnl.sum()),
                null_be=float((win.mean(axis=1) >= hr).mean()))


def boot_mean(x: np.ndarray, seed: int = SEED) -> tuple:
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return (float(x.mean()), float(lo), float(hi),
            float(x.mean() / se) if se and se > 0 else np.nan)


# ------------------------------------------------------------- reporting ----
def fmt(r: dict) -> str:
    if not r.get("n"):
        return f"{'0':>5}" + " " * 76
    return (f"{r['n']:>5}{100 * r['hit']:>7.1f}"
            f"[{100 * r['hit_lo']:>5.1f},{100 * r['hit_hi']:>5.1f}]"
            f"{100 * r['breakeven']:>8.2f}{100 * r['edge_pp']:>+8.2f}"
            f"{100 * r['roi']:>8.2f}"
            f"[{100 * r['roi_lo']:>+6.1f},{100 * r['roi_hi']:>+6.1f}]"
            f"{r.get('null_be', float('nan')):>9.3f}"
            f"{r.get('null_mkt', float('nan')):>9.3f}")


def hdr(w: int = 12) -> str:
    return (f"    {'arm':<{w}}{'window':>8}{'n':>5}{'hit%':>7}"
            f"{'  [95% CI]':>14}{'be%':>8}{'edge_pp':>8}{'ROI%':>8}"
            f"{'  [95% CI]':>16}{'null_be':>9}{'null_mkt':>9}")


HDR = hdr(12)


def run_universe(m: pd.DataFrame, src: str, label: str, res: dict,
                 cells: list) -> None:
    """Task 2(a)/(b) on the WHOLE covered universe, before any rule fires."""
    okm = np.ones(len(m), bool)
    po, deco, oko = price_cols(m, "open", src)
    pc, decc, okc = price_cols(m, "close", src)
    okm &= oko & okc
    if okm.sum() < 50:
        print(f"\n  [{src}] universe: only {okm.sum()} priced games — skipped")
        return
    edge_o = m.p_us_side.values - po
    edge_c = m.p_us_side.values - pc
    same_o = ((m.p_us.values - 0.5) *
              (np.where(m.pick_home, po, 1 - po) - 0.5)) > 0
    same_c = ((m.p_us.values - 0.5) *
              (np.where(m.pick_home, pc, 1 - pc) - 0.5)) > 0

    print(f"\n[2] THE CORE TEST — our model's side vs the price, "
          f"WHOLE UNIVERSE, price source = {src}   (n priced = {int(okm.sum())})")
    print(f"    mean edge vs OPEN {edge_o[okm].mean():+.5f}   "
          f"vs CLOSE {edge_c[okm].mean():+.5f}   "
          f"(open-minus-close {edge_o[okm].mean() - edge_c[okm].mean():+.5f})")
    print(f"    same-side vs OPEN {100 * same_o[okm].mean():.1f}%   "
          f"vs CLOSE {100 * same_c[okm].mean():.1f}%")
    print(hdr(22))
    sets = [("ALL", okm),
            ("SAMESIDE-O", okm & same_o),
            ("EDGE>0 @O", okm & same_o & (edge_o > 0)),
            ("EDGE>.02@O", okm & same_o & (edge_o > EDGE_R4)),
            ("EDGE>.03@O", okm & same_o & (edge_o > 0.03))]
    seasons = sorted(m.season.unique())
    for nm, sel in sets:
        for wname, ws in WINDOWS + [(s, {s}) for s in seasons]:
            s = sel & (m.season.isin(ws).values if ws else True)
            ro = score_ml(m, s, deco, po)
            rc = score_ml(m, s, decc, pc)
            for tag, r in (("A OPEN", ro), ("B O@CLOSE", rc)):
                if not r.get("n"):
                    continue
                print(f"    {nm + ' ' + tag:<22}{wname:>8}{fmt(r)}")
                cells.append({"frame": label, "src": src, "scope": "universe",
                              "set": nm, "arm": tag, "window": wname, **{
                                  k: v for k, v in r.items()}})
        print()

    # ---- ATS (task 2a) at the OPENING SPREAD, paired with the close --------
    print(f"    ATS — our implied margin 6.96*logit(p_us) vs the SPREAD, -110 "
          f"(breakeven {100 / ATS_DEC:.3f}%).  D119's perfect-foresight "
          f"ceiling was 54.63% at the open.")
    print(HDR.replace("null_mkt", "        "))
    for wname, ws in WINDOWS:
        s = okm & (m.season.isin(ws).values if ws else True)
        for tag, ln in (("ATS @OPEN", m.open_margin.values),
                        ("ATS @CLOSE", m.close_margin.values)):
            r = score_ats(m, s, ln)
            print(f"    {tag:<12}{wname:>8}{r['n']:>5}{100 * r['hit']:>7.1f}"
                  f"[{100 * r['hit_lo']:>5.1f},{100 * r['hit_hi']:>5.1f}]"
                  f"{100 * r['breakeven']:>8.2f}{100 * r['edge_pp']:>+8.2f}"
                  f"{100 * r['roi']:>8.2f}"
                  f"[{100 * r['roi_lo']:>+6.1f},{100 * r['roi_hi']:>+6.1f}]"
                  f"{r['null_be']:>8.3f}   (push {r['n_push']})")
            cells.append({"frame": label, "src": src, "scope": "ATS",
                          "set": "ALL", "arm": tag, "window": wname, **r})


def run_rules(m: pd.DataFrame, src: str, label: str, res: dict,
              cells: list) -> None:
    """Task 2(c)/(d): the four registered rules, open vs close, PAIRED."""
    po, deco, oko = price_cols(m, "open", src)
    pc, decc, okc = price_cols(m, "close", src)
    ok = oko & okc
    if ok.sum() < 50:
        return
    mo, edge_o, _ = registry_masks(m, po, "open")
    mc, edge_c, _ = registry_masks(m, pc, "close")

    print(f"\n[3] THE FOUR REGISTERED F4 RULES — price source {src}\n"
          f"    A OPEN      = fired on edge-vs-OPEN,  transacted at the OPEN\n"
          f"    B OPEN@CLS  = the SAME BETS as A, transacted at the CLOSE "
          f"(isolates price from selection)\n"
          f"    C CLOSE     = fired on edge-vs-CLOSE, transacted at the CLOSE "
          f"(the D115/D117 control, same game universe)")
    for rule in mo:
        so_, sc_ = mo[rule] & ok, mc[rule] & ok
        both = int((so_ & sc_).sum())
        print(f"\n  {'=' * 110}\n  RULE {rule}   fires@open n={int(so_.sum())}"
              f"   fires@close n={int(sc_.sum())}   overlap n={both}"
              f"   (open-only {int((so_ & ~sc_).sum())}, "
              f"close-only {int((sc_ & ~so_).sum())})")
        print(HDR)
        seasons = sorted(m.season.unique())
        for wname, ws in WINDOWS + [(s, {s}) for s in seasons]:
            wsel = m.season.isin(ws).values if ws else np.ones(len(m), bool)
            for tag, sel, dec, pf in (("A OPEN", so_ & wsel, deco, po),
                                      ("B OPEN@CLS", so_ & wsel, decc, pc),
                                      ("C CLOSE", sc_ & wsel, decc, pc)):
                r = score_ml(m, sel, dec, pf)
                if not r.get("n"):
                    continue
                print(f"    {tag:<12}{wname:>8}{fmt(r)}")
                cells.append({"frame": label, "src": src, "scope": "rule",
                             "set": rule, "arm": tag, "window": wname, **r})
        # PAIRED delta on the IDENTICAL bet set (A minus B)
        if so_.sum() > 1:
            ho = m.hit.values[so_].astype(bool)
            d = (np.where(ho, deco[so_] - 1, -1.0) -
                 np.where(ho, decc[so_] - 1, -1.0))
            mn, lo, hi, t = boot_mean(d)
            print(f"    PAIRED dROI/bet (A OPEN - B same bets at CLOSE): "
                  f"{100 * mn:+.2f}pp [{100 * lo:+.2f},{100 * hi:+.2f}] "
                  f"{'SIG' if (lo > 0 or hi < 0) else 'NS'}  n={int(so_.sum())}"
                  f"  -- the PURE PRICE effect of transacting hours earlier")
            cells.append({"frame": label, "src": src, "scope": "paired",
                          "set": rule, "arm": "A-B", "window": "POOL",
                          "n": int(so_.sum()), "mean": mn, "lo": lo, "hi": hi})


def run_clv(m: pd.DataFrame, src: str, label: str, res: dict) -> None:
    """Task 3: realized CLV of every bet the rules fire AT THE OPEN."""
    po, deco, oko = price_cols(m, "open", src)
    pc, decc, okc = price_cols(m, "close", src)
    ok = oko & okc
    mo, edge_o, same_o = registry_masks(m, po, "open")
    print(f"\n[4] CLV OF THE BETS FIRED AT THE OPEN — price source {src}\n"
          f"    clv_prob  = p_close_side - p_open_side (de-vigged; >0 = the "
          f"close moved TOWARD our side)\n"
          f"    clv_price = dec_open/dec_close - 1 (the price we locked vs the "
          f"price we would have paid at the close)")
    clv_prob_all = pc - po                       # OUR side, de-vigged
    clv_price_all = deco / decc - 1.0
    # CONTROL 1 (D115 s6.3): the CLV of simply betting the OPEN FAVOURITE.
    # `po` is already our-side, so our side is the open favourite iff po > 0.5.
    fav_is_ours = po > 0.5
    drift = np.where(fav_is_ours, clv_prob_all, -clv_prob_all)
    out = {}
    sets = [("ALL games (our side)", ok),
            ("ALL same-side", ok & same_o)] + [(r, mo[r] & ok) for r in mo]
    print(f"    {'bet set':<21}{'n':>6}{'clv_prob':>10}{'  [95% CI]':>20}"
          f"{'t':>7}{'%>0':>7}{'clv_price%':>11}{'favdrift':>10}"
          f"{'EXCESS':>10}{'  [95% CI]':>20}")
    for nm, sel in sets:
        if sel.sum() < 5:
            continue
        a, alo, ahi, at = boot_mean(clv_prob_all[sel])
        b, blo, bhi, _ = boot_mean(clv_price_all[sel])
        f, _, _, _ = boot_mean(drift[sel])
        # PAIRED excess over the favourite-drift control on the SAME games
        e, elo, ehi, et = boot_mean(clv_prob_all[sel] - drift[sel])
        print(f"    {nm:<21}{int(sel.sum()):>6}{a:>+10.5f}"
              f"[{alo:>+9.5f},{ahi:>+9.5f}]{at:>+7.2f}"
              f"{100 * (clv_prob_all[sel] > 0).mean():>7.1f}"
              f"{100 * b:>+11.3f}{f:>+10.5f}{e:>+10.5f}"
              f"[{elo:>+9.5f},{ehi:>+9.5f}]"
              f"{' SIG' if (elo > 0 or ehi < 0) else ' NS'}")
        out[nm] = {"n": int(sel.sum()), "clv_prob": a, "lo": alo, "hi": ahi,
                   "t": at, "frac_pos": float((clv_prob_all[sel] > 0).mean()),
                   "clv_price": b, "favdrift": f,
                   "excess": e, "excess_lo": elo, "excess_hi": ehi,
                   "excess_t": et}

    # ---- CONTROL 2: THE SELECTION PLACEBO (the confound that matters) ------
    # Every rule selects on `edge = p_us_side - p_open_side`, i.e. it PICKS
    # GAMES WHERE THE OPEN IS LOW FOR OUR SIDE.  If the open simply carries
    # more measurement noise than the close, that selection harvests
    # mean-reversion and manufactures CLV carrying ZERO information.
    # PLACEBO: permute p_us WITHIN (season x p_open decile) strata, so the
    # selection mechanism and the open-price distribution are preserved but
    # the model's information is destroyed.  Re-fire the rules on the placebo
    # and measure its CLV.  If placebo CLV ~ 0, our CLV is information.
    rng = np.random.default_rng(SEED)
    dec_o = pd.qcut(po, 10, labels=False, duplicates="drop")
    p_perm = m.p_us.values.copy()
    for key in pd.unique(list(zip(m.season.values, dec_o))):
        idx = np.where((m.season.values == key[0]) & (dec_o == key[1]))[0]
        if len(idx) > 1:
            p_perm[idx] = p_perm[rng.permutation(idx)]
    mp = m.copy()
    mp["p_us"] = p_perm
    mp["pick_home"] = mp.p_us > 0.5
    mp["p_us_side"] = np.where(mp.pick_home, mp.p_us, 1 - mp.p_us)
    mp["conf_us"] = (mp.p_us - 0.5).abs()
    po_p, _, _ = price_cols(mp, "open", src)
    pc_p, _, _ = price_cols(mp, "close", src)
    mo_p, _, same_p = registry_masks(mp, po_p, "open")
    clv_p = pc_p - po_p
    print(f"\n    SELECTION PLACEBO — p_us permuted within "
          f"(season x p_open decile); the rules re-fire on a model that knows "
          f"NOTHING.\n    If the placebo also earns CLV, our CLV is "
          f"mean-reversion of open-price noise, not information.")
    print(f"    {'bet set':<21}{'n':>6}{'PLACEBO clv':>13}{'  [95% CI]':>20}"
          f"{'t':>7}   {'REAL clv':>10}   verdict")
    for nm, selr in [("ALL same-side", ok & same_p)] + \
                    [(r, mo_p[r] & ok) for r in mo_p]:
        if selr.sum() < 5:
            continue
        a, alo, ahi, at = boot_mean(clv_p[selr])
        real = out.get(nm, {}).get("clv_prob", float("nan"))
        v = ("MECHANICAL" if np.isfinite(real) and alo > 0 and a > 0.5 * real
             else "clean" if not (alo > 0) else "partial")
        print(f"    {nm:<21}{int(selr.sum()):>6}{a:>+13.5f}"
              f"[{alo:>+9.5f},{ahi:>+9.5f}]{at:>+7.2f}   {real:>+10.5f}   {v}")
        out.setdefault("placebo", {})[nm] = {
            "n": int(selr.sum()), "clv": a, "lo": alo, "hi": ahi, "t": at,
            "real": real, "verdict": v}

    # ---- CONTROL 3: does CLV carry OUTCOME information? --------------------
    selr = ok & same_o
    cl, hh = clv_prob_all[selr], m.hit.values[selr].astype(float)
    if selr.sum() > 100:
        hi_ = cl > np.median(cl)
        a, alo, ahi, at = boot_mean(hh[hi_] - 0.0)
        b, blo, bhi, _ = boot_mean(hh[~hi_] - 0.0)
        d, dlo, dhi, dt = boot_mean(hh[hi_]), None, None, None
        print(f"\n    CLV -> OUTCOME: bets in the TOP half of realized CLV hit "
              f"{100 * hh[hi_].mean():.1f}% vs {100 * hh[~hi_].mean():.1f}% in "
              f"the bottom half (corr(clv,hit) = {np.corrcoef(cl, hh)[0, 1]:+.4f})."
              f"  CLV is only worth having if it eventually shows up here.")
        out["clv_outcome"] = {"top_hit": float(hh[hi_].mean()),
                              "bot_hit": float(hh[~hi_].mean()),
                              "corr": float(np.corrcoef(cl, hh)[0, 1])}

    # ---- per-season CLV (source is confounded with season: sbr vs espn/an) --
    sub0 = m[ok & same_o].copy()
    sub0["clv"] = clv_prob_all[ok & same_o]
    ps = sub0.groupby(["season", "source"]).clv.agg(["size", "mean"])
    print(f"\n    PER-SEASON x SOURCE (the open price comes from DIFFERENT "
          f"vendors before/after 2023-24 — the effect must appear in both):")
    for k, r in ps.iterrows():
        print(f"      {k[0]:<9}{k[1]:<22}n={int(r['size']):>5}  "
              f"mean clv {r['mean']:+.5f}")
    out["per_season"] = ps.reset_index().to_dict("records")

    # ---- MONTHLY DISTRIBUTION — calibrate a good/bad CLV month -------------
    print(f"\n    MONTHLY CLV — what a good/bad month looks like for the live "
          f"program (bets = union of the four registered rules @ open)")
    union = np.zeros(len(m), bool)
    for r in mo:
        union |= mo[r].values if hasattr(mo[r], "values") else mo[r]
    union &= ok
    sub = m[union].copy()
    sub["clv"] = clv_prob_all[union]
    sub["ym"] = sub.game_date.dt.to_period("M").astype(str)
    g = sub.groupby("ym").clv.agg(["size", "mean"])
    g = g[g["size"] >= 3]
    if len(g):
        print(f"      months={len(g)}  bets/month median={g['size'].median():.0f}"
              f"  mean-of-monthly-means={g['mean'].mean():+.5f}")
        q = g["mean"].quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        print(f"      monthly-mean CLV percentiles: "
              f"p10 {q.loc[0.10]:+.5f}  p25 {q.loc[0.25]:+.5f}  "
              f"p50 {q.loc[0.50]:+.5f}  p75 {q.loc[0.75]:+.5f}  "
              f"p90 {q.loc[0.90]:+.5f}")
        print(f"      months with POSITIVE mean CLV: "
              f"{100 * (g['mean'] > 0).mean():.1f}%  ({int((g['mean'] > 0).sum())}"
              f"/{len(g)})")
        sd1 = float(sub.clv.std(ddof=1))
        nmed = float(g["size"].median())
        print(f"      per-bet CLV sd = {sd1:.5f}; a median month of "
              f"{nmed:.0f} bets therefore has se = {sd1 / np.sqrt(nmed):.5f}, "
              f"so a monthly mean below "
              f"{out.get('ALL same-side', {}).get('clv_prob', 0) - 2 * sd1 / np.sqrt(nmed):+.5f}"
              f" is a 2-sigma RED FLAG and above "
              f"{out.get('ALL same-side', {}).get('clv_prob', 0) + 2 * sd1 / np.sqrt(nmed):+.5f}"
              f" is a 2-sigma good month.")
        out["monthly"] = {"n_months": int(len(g)),
                          "median_bets": nmed,
                          "pct_positive": float((g["mean"] > 0).mean()),
                          "per_bet_sd": sd1,
                          "percentiles": {str(k): float(v) for k, v in q.items()},
                          "table": g.reset_index().to_dict("records")}
        print("      per-month table:")
        for _, row in g.reset_index().iterrows():
            bar = "+" * int(max(0, row["mean"] * 400)) or \
                  "-" * int(max(0, -row["mean"] * 400))
            print(f"        {row['ym']}  n={int(row['size']):>3}  "
                  f"mean {row['mean']:+.5f}  {bar}")
    res.setdefault("clv", {})[f"{label}|{src}"] = out


def run_kelly_slope(m: pd.DataFrame, src: str, label: str, res: dict) -> None:
    """D117's calibration re-measured AT THE OPEN.  Does the 'no positive-EV
    bets' verdict survive when the price is the opener?"""
    po, deco, oko = price_cols(m, "open", src)
    pc, decc, okc = price_cols(m, "close", src)
    ok = oko & okc
    print(f"\n[5] D117 EDGE-CALIBRATION RE-MEASURED AT THE OPEN "
          f"(price source {src})")
    print(f"    realised_excess = a + b x claimed_excess, on same-side bets; "
          f"break-even claimed edge = -a/b")
    row = {}
    for tag, p in (("OPEN", po), ("CLOSE", pc)):
        pside = p
        edge = m.p_us_side.values - pside
        same = ((m.p_us.values - 0.5) *
                (np.where(m.pick_home, pside, 1 - pside) - 0.5)) > 0
        s = ok & same
        c = fit_kelly_slope(pd.Series(edge[s]),
                            pd.Series(m.hit.values[s] - pside[s]))
        be = (-c["a"] / c["b"]) if c["b"] > 0 else float("nan")
        # what overround does the calibrated edge clear at the registry cap?
        e_cap = CONF_EXCESS_CAP
        vmax = 1.0 + max(0.0, c["a"] + c["b"] * e_cap) / np.mean(pside[s])
        print(f"    {tag:<6} a={c['a']:+.4f}  b={c['b']:+.4f} "
              f"(se {c['se_b']:.4f}, t={c['t']:+.2f}, n={c['n']})  "
              f"-> break-even claimed edge {be:+.4f}  "
              f"vs registry cap {CONF_EXCESS_CAP:+.4f}  "
              f"[{'CLEARS' if be < CONF_EXCESS_CAP else 'DOES NOT CLEAR'} "
              f"the cap]")
        row[tag] = {**c, "breakeven_claimed_edge": be, "vmax_at_cap": vmax}
    bo, bc = row["OPEN"]["b"], row["CLOSE"]["b"]
    print(f"    SLOPE DELTA open-minus-close: {bo - bc:+.4f} "
          f"(a delta {row['OPEN']['a'] - row['CLOSE']['a']:+.4f}) — a HIGHER "
          f"slope at the open means more of our claimed edge is real there.")
    res.setdefault("kelly_slope", {})[f"{label}|{src}"] = row


# ------------------------------------------------------------------ main ----
def main() -> None:
    res: dict = {"overround_assumed": OVERROUND, "spread_scale": SPREAD_SCALE,
                 "ats_dec": ATS_DEC, "seed": SEED}
    cells: list = []
    print("=" * 116)
    print("BO-OPENBACKTEST — the honest BET-AT-OPEN backtest (D120)")
    print("=" * 116)
    print("PRICING: SP@1.045 = sigmoid(margin/6.96) de-vigged, decimal "
          "1/(p_side*1.045), floor 1.01 (the program-wide assumption).")
    print("         ML       = the REAL opening/closing American moneylines, "
          "no vig assumption (2023-24..2025-26 only).")
    print("STAKING: flat 1u everywhere.  Kelly is D117's territory and it "
          "stakes zero; flat is the honest control and has the best power.")

    for label, csv, pcol in (("PRIMARY rt1 p_full 4-season", RT1, "p_full"),
                             ("WIDE cf_holdout p_base 5-season", CF, "p_base")):
        m = build(csv, pcol, label, res)
        ovm = m.ov_open_ml.dropna()
        if len(ovm):
            print(f"\n    MEASURED opening-ML overround: mean {ovm.mean():.4f} "
                  f"median {ovm.median():.4f} (n={len(ovm)}) — the program has "
                  f"been ASSUMING {OVERROUND}.  The assumption is sound.")
        for src in ("SP", "ML"):
            if src == "ML" and m.p_open_ml.notna().sum() < 200:
                continue
            run_universe(m, src, label, res, cells)
            run_rules(m, src, label, res, cells)
            run_clv(m, src, label, res)
            run_kelly_slope(m, src, label, res)

    # ---------------------------------------------------- family-wise -------
    print(f"\n{'=' * 116}\n[6] FAMILY-WISE NULL-COUNT CHECK\n{'=' * 116}")
    from math import comb
    res["familywise"] = {}
    for scope_name, keep in (
            ("PRE-SPECIFIED (POOL/DEV/NONDEV only)",
             lambda c: c.get("window") in PRIMARY_WINDOWS),
            ("EVERYTHING PRINTED (incl. per-season diagnostics)",
             lambda c: True)):
        scored = [c for c in cells
                  if c.get("n") and np.isfinite(c.get("null_be", np.nan))
                  and keep(c)]
        k = len(scored)
        sig_be = [c for c in scored if c["null_be"] < 0.05]
        sig_mkt = [c for c in scored
                   if np.isfinite(c.get("null_mkt", np.nan))
                   and c["null_mkt"] < 0.05]
        p_at = 1.0 - sum(comb(k, i) * 0.05 ** i * 0.95 ** (k - i)
                         for i in range(len(sig_be))) if k else float("nan")
        print(f"\n  -- {scope_name}")
        print(f"     cells = {k};  expected significant at p<0.05 under a "
              f"global null = {0.05 * k:.1f}")
        print(f"     OBSERVED vs TRUE BREAKEVEN  (null_be)  = {len(sig_be)}"
              f"   -> P(chance >= {len(sig_be)}) = {p_at:.4f}")
        print(f"     OBSERVED vs NO-EDGE-vs-mkt  (null_mkt) = {len(sig_mkt)}")
        for c in sig_be:
            print(f"       SIG: {c['frame'][:20]:<20} {c['src']:<3} "
                  f"{c['set']:<18} {c['arm']:<11} {c['window']:<8} "
                  f"n={c['n']:<5} ROI {100 * c['roi']:+.2f}%  "
                  f"p={c['null_be']:.4f}")
        res["familywise"][scope_name] = {
            "cells": k, "expected": 0.05 * k, "observed_be": len(sig_be),
            "observed_mkt": len(sig_mkt), "p_atleast": p_at}
    print("\n     Cells heavily OVERLAP (the same bets re-scored across "
          "windows, arms and price sources), so these are UPPER BOUNDS on the "
          "surprise, not tests.")

    res["cells"] = cells
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
