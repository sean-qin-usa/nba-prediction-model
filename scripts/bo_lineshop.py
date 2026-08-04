#!/usr/bin/env python3
"""BO-LINESHOP — does LINE SHOPPING at the open close D121's 1.47pp deficit?

THE QUESTION.  D120/D121: on real opening moneylines our side hits 68.14% vs a
69.61% breakeven — SHORT BY 1.47pp (n=3,682).  Every prior betting test in this
program priced ONE number per game, which is not how anyone actually bets.  The
TeamRankings spread-movement pages carry a PER-BOOK panel, so for the first time
the opening price can be shopped in a backtest.

WHAT THE PANEL ACTUALLY IS (measured, not assumed): the page renders three
columns but book3 is EMPTY for every NBA game in the scrape.  The maximum shop
this data can simulate is BEST-OF-**2**.

EXECUTION POLICIES on the SAME frozen bet set (no rule is re-selected, no rule
is added; the four registered F4 rules fire on the REGISTERED consensus open
exactly as in D120/D121/D126, and ONLY the transacted price changes):
  B1 / B2   one book only — what "price one number per game" actually means
  ONEBOOK   the expected result of a one-account bettor (average of B1, B2) —
            THE BASELINE the shop has to beat
  MID       the midpoint handicap
  CONS_TR   TeamRankings' summary Open cell (a THIRD quote: it lies OUTSIDE
            [min(b1,b2), max(b1,b2)] on 27.3% of games)
  CONS_REG  odds_open.open_margin — the price D120/D121/D126 transacted
  BEST2 / WORST2   best / worst of the two real books FOR THE SIDE WE BET
  BEST3 / BEST4 / WORST4   best/worst over 3 and 4 quotes (books + vendor
            snapshots) — UPPER BOUNDS, because the vendor snapshots are not
            simultaneously-transactable book prices
A secondary RE-FIRE arm lets the rules see the best price when computing edge
(that arm CHANGES the bet set and is reported separately, because selecting on
the price is exactly where a fake result would come from).

TWO HARNESS CONTROLS, both required to reproduce EXACTLY before anything else
is believed: (1) D126's four rules + union on the 2022-23 late window;
(2) D121's ML headline n=3,682 / 68.14% vs 69.61% / -1.47pp on the full frame.

PRICING CONVENTION — identical to D121/D126:
  p_open  = sigmoid(open_margin / 6.96)          [D120 PRIMARY|SP|OPEN frame,
            the frame frozen in bet_engine.OPEN_SHRUNK]
  decimal = max(1 / (p_side * 1.045), 1.01)      [D75/D78 proportional
            overround, validated by D120 against real MLs at 1.0431/1.0433]
  close   = odds_open.close_margin, same map (odds_market close for the
            2022-23 D126 reproduction, per D126).
D120 established this SP map is ~2pp PESSIMISTIC on breakeven vs real MLs; every
LEVEL below carries that haircut, but every DELTA between policies is a
difference of two numbers computed the same way, so the haircut cancels.

RULES HONORED: DuckDB read_only=True with a 60s retry; new file; nothing in
nbapred/, scripts/bet_engine.py or the frozen registry is touched.

Run:  python scripts/bo_lineshop.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from math import comb

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import duckdb                                                    # noqa: E402
import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402
from scipy import stats                                          # noqa: E402

import bo_openbacktest as bo                                     # noqa: E402
from ba_intersection import star_out_map                         # noqa: E402
from bet_sim3 import MIN_DEC, OVERROUND                          # noqa: E402

DB = os.path.join(ROOT, "data", "nba.duckdb")
TR_JSONL = os.path.join(ROOT, "data", "raw", "teamrankings",
                        "spread_movement.jsonl")
RT1 = os.path.join(ROOT, "data", "ds_rt1_pergame.csv")
CF = os.path.join(ROOT, "data", "cf_holdout_new_pergame.csv")
REG2223 = os.path.join(ROOT, "data", "bo_open2223_oos.json")
OUT = os.path.join(ROOT, "data", "bo_lineshop.json")

SEED = bo.SEED
SC = bo.SPREAD_SCALE                 # 6.96
CUT2223 = pd.Timestamp("2023-01-16")
D121_DEFICIT_PP = 1.47               # the number this test has to close

# fields kept when a scored cell is serialised to JSON
KEEP = ("n", "hit", "hit_lo", "hit_hi", "hit_lo_exact", "hit_hi_exact",
        "breakeven", "edge_pp", "roi", "roi_lo", "roi_hi", "mean_dec",
        "null_be", "null_mkt", "binom_p", "d_be_pp_1bk", "d_roi_pp_1bk",
        "d_be_pp_reg", "d_roi_pp_reg")

# teamrankings abbreviations -> ours (verbatim from scripts/build_odds_open.py)
TR_TEAMS = {"BK": "BKN", "GS": "GSW", "NO": "NOP", "NY": "NYK",
            "PHO": "PHX", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}


def ro_connect(attempts: int = 10, wait_s: float = 60.0):
    for i in range(attempts):
        try:
            return duckdb.connect(DB, read_only=True)
        except Exception as e:                                    # noqa: BLE001
            msg = str(e).lower()
            if ("lock" not in msg and "held" not in msg) or i == attempts - 1:
                raise
            print(f"reader blocked by writer lock, retry {i+1}/{attempts} in "
                  f"{wait_s:.0f}s")
            time.sleep(wait_s)


def binom_ci(k: int, n: int):
    lo = stats.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


# ------------------------------------------------------- per-book panel ------
def build_panel(res: dict) -> pd.DataFrame:
    """game -> {book_i: opening spread}, resolved onto OUR team keys the same
    way scripts/build_odds_open.py does (`_pair_join` against odds_market, the
    TR_TEAMS map, and the favourite-perspective sign convention).

    TeamRankings quotes from the FAVOURITE's side and never says who is home,
    so home/away come from the unordered-pair join, and every book's number is
    re-expressed as an expected HOME MARGIN (positive = home favoured), the
    convention odds_open.open_margin uses."""
    recs = [json.loads(x) for x in open(TR_JSONL) if x.strip()]
    tr = pd.DataFrame(recs)
    tr = tr[~tr.get("no_data", pd.Series(False, index=tr.index)).fillna(False)]
    tr = tr[tr.fav_open.notna() & tr.fav_team.notna()].copy()
    tr["game_date"] = pd.to_datetime(tr.game_date)

    ob = np.array([[(r or [None, None, None] + [None] * 3)[i]
                    for i in range(3)] for r in tr.open_books], dtype=object)
    for i in range(3):
        tr[f"b{i+1}"] = pd.to_numeric(pd.Series(ob[:, i], index=tr.index),
                                      errors="coerce")
    tr["n_books"] = tr[["b1", "b2", "b3"]].notna().sum(axis=1)

    print(f"\n{'=' * 116}\n[1] THE PER-BOOK OPENING PANEL\n{'=' * 116}")
    print(f"  scraped pages {len(recs)};  usable (has a favourite + an open) "
          f"{len(tr)}")
    print(f"  per-book NON-NULL counts in the page's own OPEN row: "
          f"book1={int(tr.b1.notna().sum())}  book2={int(tr.b2.notna().sum())} "
          f" book3={int(tr.b3.notna().sum())}")
    print(f"  *** book3 is EMPTY for every NBA game -> the maximum shop this "
          f"source can simulate is BEST-OF-2, not best-of-3. ***")
    print(f"  games by number of opening quotes: "
          f"{tr.n_books.value_counts().sort_index().to_dict()}")

    # ---- _pair_join, verbatim in behaviour ---------------------------------
    con = ro_connect()
    try:
        mk = con.execute("""SELECT season_end, game_date, home, away
                            FROM odds_market""").df()
    finally:
        con.close()
    mk["game_date"] = pd.to_datetime(mk.game_date)
    mk["pair"] = [frozenset(p) for p in zip(mk.home, mk.away)]
    tr["fav_team"] = tr.fav_team.replace(TR_TEAMS)
    tr["dog_team"] = tr.dog_team.replace(TR_TEAMS)
    tr["pair"] = [frozenset(p) for p in zip(tr.fav_team, tr.dog_team)]
    tr = tr.drop(columns=["season_end"], errors="ignore")  # odds_market owns it
    m = tr.merge(mk, on=["game_date", "pair"], how="inner")
    print(f"  pair-join against odds_market: {len(tr)} -> {len(m)} matched "
          f"(drops preseason / All-Star / exhibition rows for free)")

    fav_home = (m.fav_team == m.home).values
    sgn = np.where(fav_home, -1.0, +1.0)          # margin = -fav_line if fav home
    out = pd.DataFrame({
        "game_date": m.game_date, "home": m.home, "away": m.away,
        "season_end": m.season_end,
        "mb1": sgn * m.b1.values, "mb2": sgn * m.b2.values,
        "m_cons_tr": sgn * m.fav_open.values,
        "n_books": m.n_books.values,
    })
    out = out.drop_duplicates(subset=["game_date", "home", "away"],
                              keep="first").reset_index(drop=True)

    two = out[out.n_books >= 2]
    disp = (two[["mb1", "mb2"]].max(axis=1) - two[["mb1", "mb2"]].min(axis=1))
    print(f"\n  DISPERSION AT THE OPEN — |book1 - book2|, n={len(two)} "
          f"two-book games (THE SIZE OF THE PRIZE):")
    print(f"    mean {disp.mean():.4f} pts   median {disp.median():.2f}   "
          f"sd {disp.std():.3f}   max {disp.max():.1f}")
    for t in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        f = (disp == 0).mean() if t == 0 else (disp >= t).mean()
        print(f"    {'== 0.0' if t == 0 else f'>= {t:.1f}'} pts: "
              f"{100 * f:5.1f}%")
    print("    histogram: " + "  ".join(
        f"{k}:{v}" for k, v in disp.value_counts().sort_index().items()
        if v >= 5))
    print(f"\n  per-season two-book games: "
          f"{two.groupby('season_end').size().to_dict()}")
    cons_is_min = (two.m_cons_tr == two[["mb1", "mb2"]].min(axis=1)).mean()
    cons_is_max = (two.m_cons_tr == two[["mb1", "mb2"]].max(axis=1)).mean()
    print(f"  TR's summary Open cell (= what odds_open carries as 'the' open) "
          f"is the MIN of the two books {100*cons_is_min:.1f}% of the time and "
          f"the MAX {100*cons_is_max:.1f}% — it is ONE BOOK'S NUMBER, not an "
          f"average, so best-minus-consensus is ~HALF the dispersion.")
    res["panel"] = {
        "scraped": len(recs), "usable": int(len(tr)),
        "book_nonnull": {f"book{i+1}": int(tr[f'b{i+1}'].notna().sum())
                         for i in range(3)},
        "n_books_hist": {int(k): int(v) for k, v in
                         tr.n_books.value_counts().sort_index().items()},
        "matched": int(len(out)), "two_book": int(len(two)),
        "dispersion_pts": {"mean": float(disp.mean()),
                           "median": float(disp.median()),
                           "sd": float(disp.std()),
                           "frac_zero": float((disp == 0).mean()),
                           "frac_ge_0.5": float((disp >= 0.5).mean()),
                           "frac_ge_1.0": float((disp >= 1.0).mean()),
                           "frac_ge_1.5": float((disp >= 1.5).mean()),
                           "frac_ge_2.0": float((disp >= 2.0).mean())},
        "cons_is_min": float(cons_is_min), "cons_is_max": float(cons_is_max),
        "per_season_two_book": {int(k): int(v) for k, v in
                                two.groupby('season_end').size().items()},
    }
    return out


# ------------------------------------------------------------- policies -----
def _best(ph, quotes):
    q = np.vstack(quotes)
    return np.where(ph, np.nanmin(q, axis=0), np.nanmax(q, axis=0))


def _worst(ph, quotes):
    q = np.vstack(quotes)
    return np.where(ph, np.nanmax(q, axis=0), np.nanmin(q, axis=0))


def policy_margins(m: pd.DataFrame) -> dict:
    """Expected-HOME-MARGIN under each execution policy.

    Best FOR OUR SIDE = the number that makes our side look LEAST likely to the
    market, because that is the longest price.  Betting home -> the SMALLEST
    open_margin; betting away -> the LARGEST.

    FOUR quotes per game are available and they are NOT redundant:
      b1, b2      the two books in the page's own per-book Open row
      cons_tr     TeamRankings' summary Open cell — measured to lie OUTSIDE
                  [min(b1,b2), max(b1,b2)] on 27.3% of games, so it is a THIRD
                  independent quote, not an average of the other two
      cons_reg    odds_open.open_margin, the price D120/D121/D126 transacted
                  (SBR / ESPN+ActionNetwork) — a FOURTH, different-vendor quote
    The clean PURE-SHOPPING baseline is a ONE-BOOK bettor (B1 or B2), because
    that is what "price one number per game" actually means."""
    ph = m.pick_home.values
    b1, b2 = m.mb1.values, m.mb2.values
    ct, cr = m.m_cons_tr.values, m.open_margin.values
    return {"B1": b1, "B2": b2, "CONS_TR": ct, "CONS_REG": cr,
            "MID": np.nanmean(np.vstack([b1, b2]), axis=0),
            "BEST2": _best(ph, [b1, b2]), "WORST2": _worst(ph, [b1, b2]),
            "BEST3": _best(ph, [b1, b2, ct]),
            "BEST4": _best(ph, [b1, b2, ct, cr]),
            "WORST4": _worst(ph, [b1, b2, ct, cr])}


POL_ORDER = ["B1", "B2", "ONEBOOK", "MID", "CONS_TR", "CONS_REG", "BEST2",
             "WORST2", "BEST3", "BEST4", "WORST4"]


def priced(m: pd.DataFrame, margin: np.ndarray):
    """(our-side de-vigged prob, transactable decimal) under the D121/D126
    convention, for an arbitrary opening margin vector."""
    p_home = bo.sigmoid(margin / SC)
    p_side = np.where(m.pick_home.values, p_home, 1 - p_home)
    dec = np.maximum(1.0 / (p_side * OVERROUND), MIN_DEC)
    return p_side, dec


def policy_prices(m: pd.DataFrame, pols: dict) -> dict:
    """{policy: (p_side, decimal)}.  ONEBOOK is the EXPECTED result of a
    one-book bettor who holds an account at exactly one of the two books,
    i.e. the elementwise average of the B1 and B2 prices — the correct
    baseline for 'what does a second book buy you'."""
    pr = {k: priced(m, v) for k, v in pols.items()}
    pr["ONEBOOK"] = (0.5 * (pr["B1"][0] + pr["B2"][0]),
                     0.5 * (pr["B1"][1] + pr["B2"][1]))
    return pr


# -------------------------------------------------------------- reporting ---
ROW = ("    {pol:<9}{n:>5}{hit:>8.2f}{be:>9.2f}{edge:>+9.2f}{roi:>9.2f}"
       "[{lo:>+6.1f},{hi:>+6.1f}]{d1:>+9.2f}{d2:>+9.2f}")
HEAD = ("    {:<9}{:>5}{:>8}{:>9}{:>9}{:>9}{:>16}{:>9}{:>9}".format(
    "policy", "n", "hit%", "be%", "edge_pp", "ROI%", "  [95% CI]",
    "dBE/1bk", "dBE/reg"))


def score_policies(m, sel, pr, label="", order=None):
    """Score the SAME bet set at every execution policy.  dBE columns are the
    breakeven CHANGE vs the ONEBOOK baseline and vs the REGISTERED consensus
    (negative = the shop LOWERED our breakeven = the deficit shrank)."""
    out = {}
    for pol in (order or POL_ORDER):
        if pol not in pr:
            continue
        p_side, dec = pr[pol]
        ok = np.isfinite(p_side) & np.isfinite(dec)
        r = bo.score_ml(m, sel & ok, dec, p_side)
        if r.get("n"):
            k = int(round(r["hit"] * r["n"]))
            r["hit_lo_exact"], r["hit_hi_exact"] = binom_ci(k, r["n"])
            r["binom_p"] = float(stats.binomtest(
                k, r["n"], r["breakeven"], alternative="greater").pvalue)
        out[pol] = r
    for base, tag in (("ONEBOOK", "1bk"), ("CONS_REG", "reg")):
        b = out.get(base, {})
        for r in out.values():
            if r.get("n") and b.get("n"):
                r[f"d_be_pp_{tag}"] = 100 * (r["breakeven"] - b["breakeven"])
                r[f"d_roi_pp_{tag}"] = 100 * (r["roi"] - b["roi"])
    if label:
        print(f"\n  {label}")
        print(HEAD)
        for pol, r in out.items():
            if not r.get("n"):
                continue
            print(ROW.format(pol=pol, n=r["n"], hit=100 * r["hit"],
                             be=100 * r["breakeven"], edge=100 * r["edge_pp"],
                             roi=100 * r["roi"], lo=100 * r["roi_lo"],
                             hi=100 * r["roi_hi"],
                             d1=r.get("d_be_pp_1bk", 0.0),
                             d2=r.get("d_be_pp_reg", 0.0)))
    return out


def paired_droi(m, sel, pa, pb):
    """Bootstrapped paired per-bet ROI difference on the IDENTICAL bet set."""
    da, db = pa[1], pb[1]
    h = m.hit.values[sel].astype(bool)
    d = (np.where(h, da[sel] - 1, -1.0) - np.where(h, db[sel] - 1, -1.0))
    return bo.boot_mean(d)


def score_ats_fixed(m: pd.DataFrame, sel: np.ndarray, line: np.ndarray,
                    seed: int = SEED) -> dict:
    """ATS at a fixed -110 on OUR MODEL'S SIDE (m.pick_home, the same side the
    registry bets), against `line` (the market's expected HOME margin).

    NOTE this deliberately differs from bo.score_ats, which re-picks the side
    from `m_us > line` and therefore lets the SIDE change when the number
    changes — fine for D120's question, wrong for an EXECUTION test.  With the
    side held fixed, a better handicap can only weakly increase the cover rate,
    so BEST >= ONEBOOK >= WORST is a MONOTONICITY CHECK on the sign
    convention: if it fails, the 'best' price is not best."""
    if sel.sum() == 0:
        return dict(n=0)
    mm = m[sel]
    ln = line[sel]
    ph = mm.pick_home.values
    diff = mm.margin_actual.values - ln                # >0 = home covered
    push = diff == 0
    cover = np.where(ph, diff > 0, diff < 0)
    pnl = np.where(push, 0.0, np.where(cover, bo.ATS_DEC - 1.0, -1.0))
    live = ~push
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pnl), (bo.N_BOOT, len(pnl)))
    lo, hi = np.percentile(pnl[idx].mean(axis=1), [2.5, 97.5])
    hr = float(cover[live].mean()) if live.sum() else np.nan
    be = 1.0 / bo.ATS_DEC
    return dict(n=int(sel.sum()), n_live=int(live.sum()),
                n_push=int(push.sum()), hit=hr, breakeven=be,
                edge_pp=hr - be, roi=float(pnl.mean()),
                roi_lo=float(lo), roi_hi=float(hi))


# ------------------------------------------------------------------ main ----
def main() -> None:
    res: dict = {
        "convention": {
            "p_open": "sigmoid(open_margin/6.96) [D120 PRIMARY|SP|OPEN, the "
                      "frame frozen in bet_engine.OPEN_SHRUNK]",
            "decimal": f"max(1/(p_side*{OVERROUND}), {MIN_DEC}) [D75/D78 "
                       f"proportional overround, D120-validated 1.0431]",
            "close": "odds_open.close_margin (odds_market close for the D126 "
                     "reproduction window)",
            "known_bias": "SP map ~2pp pessimistic on breakeven vs real MLs "
                          "(D120); LEVELS carry it, DELTAS cancel it",
            "seed": SEED},
        "d121_deficit_pp": D121_DEFICIT_PP}

    panel = build_panel(res)

    # ---- model frame, built by the D120/D121 machinery verbatim ------------
    for attempt in range(10):
        try:
            m = bo.build(RT1, "p_full", "PRIMARY rt1 p_full 4-season", res)
            break
        except Exception as e:                                    # noqa: BLE001
            if "lock" not in str(e).lower() or attempt == 9:
                raise
            print("build blocked by writer lock, retry in 60s")
            time.sleep(60)

    n0 = len(m)
    m = m.merge(panel.drop(columns=["season_end"]),
                on=["game_date", "home", "away"], how="left")
    assert len(m) == n0, "panel join fanned out — not 1:1"
    m["n_books"] = m.n_books.fillna(0).astype(int)
    print(f"\n[1b] PANEL x MODEL FRAME: {n0} frame games with both prices; "
          f"{int((m.n_books >= 1).sum())} have >=1 TR book, "
          f"{int((m.n_books >= 2).sum())} have 2 books "
          f"({100 * (m.n_books >= 2).mean():.1f}%).")
    print(f"     per season, 2-book games: "
          f"{m[m.n_books >= 2].groupby('season').size().to_dict()}")
    res["panel"]["frame_two_book"] = int((m.n_books >= 2).sum())
    res["panel"]["frame_n"] = int(n0)

    # vendor bridge: TR's own consensus vs the registered consensus
    tw = m[m.n_books >= 2]
    dv = (tw.m_cons_tr - tw.open_margin)
    print(f"\n[1c] VENDOR BRIDGE (this matters: part of any 'best-minus-"
          f"consensus' gap could be vendor disagreement, not shopping):")
    print(f"     TR consensus open MINUS registered odds_open open: "
          f"mean {dv.mean():+.4f}  mean|diff| {dv.abs().mean():.4f} pts  "
          f"corr {np.corrcoef(tw.m_cons_tr, tw.open_margin)[0,1]:.4f}  "
          f"identical {100*(dv == 0).mean():.1f}%")
    for s, g in m[m.n_books >= 2].groupby("season"):
        d = (g.m_cons_tr - g.open_margin)
        print(f"       {s}  n={len(g):<5} mean|diff| {d.abs().mean():.3f} "
              f"(source: {'/'.join(sorted(g.source.dropna().unique()))})")
    res["vendor_bridge"] = {"mean": float(dv.mean()),
                            "mad": float(dv.abs().mean()),
                            "corr": float(np.corrcoef(tw.m_cons_tr,
                                                      tw.open_margin)[0, 1]),
                            "frac_identical": float((dv == 0).mean())}

    # =====================================================================
    # [2] CONTROL — reproduce D126 on the 2022-23 late window
    # =====================================================================
    print(f"\n{'=' * 116}\n[2] HARNESS CONTROL — REPRODUCE D126 "
          f"(2022-23 after {CUT2223.date()}, recovered TR opens)\n{'=' * 116}")
    con = ro_connect()
    try:
        om = con.execute("""SELECT game_date, home, away,
                                   home_exp_margin AS om_close_margin
                            FROM odds_market WHERE season_end = 2023""").fetchdf()
        so = star_out_map(con)
    finally:
        con.close()
    om["game_date"] = pd.to_datetime(om.game_date)

    m22 = m[(m.season == "2022-23") & (m.game_date > CUT2223)].copy()
    m22 = m22.merge(om, on=["game_date", "home", "away"], how="left")
    m22 = m22[m22.om_close_margin.notna()].reset_index(drop=True)
    m22["close_margin"] = m22.om_close_margin
    m22["p_close_sp"] = bo.sigmoid(m22.close_margin / SC)
    som = so.rename(columns={"team": "fav_team"})[["game_id", "fav_team",
                                                   "star_out"]]
    m22["fav_team"] = np.where(m22.p_close_sp >= 0.5, m22.home, m22.away)
    m22 = m22.drop(columns=["fav_star_out_close"]).merge(
        som, on=["game_id", "fav_team"], how="left")
    m22["fav_star_out_close"] = m22.star_out.fillna(False).astype(bool)
    m22 = m22.drop(columns=["star_out", "fav_team"])

    reg = json.load(open(REG2223))
    print(f"     n scored {len(m22)} (D126 registered {reg['frame']['n_scored']})"
          f"   late {int(m22.late.sum())} (D126 {reg['frame']['n_late']})")
    po22, deco22, ok22o = bo.price_cols(m22, "open", "SP")
    pc22, decc22, ok22c = bo.price_cols(m22, "close", "SP")
    ok22 = ok22o & ok22c
    mo22, _, _ = bo.registry_masks(m22, po22, "open")
    repro = {}
    ok_all = True
    print(f"     {'rule':<18}{'n':>5}{'hit%':>8}{'be%':>8}{'ROI%':>8}"
          f"{'dROI_pp':>9}{'CLV':>10}   vs D126 registered")
    clv22 = pc22 - po22
    for rule, msk in mo22.items():
        sel = msk & ok22
        r = bo.score_ml(m22, sel, deco22, po22)
        h = m22.hit.values[sel].astype(bool)
        dd = (np.where(h, deco22[sel] - 1, -1.0) -
              np.where(h, decc22[sel] - 1, -1.0))
        dmn, _, _, _ = bo.boot_mean(dd)
        cmn, _, _, _ = bo.boot_mean(clv22[sel])
        rr = reg["rules"][rule]
        ra = rr.get("A OPEN", {})
        okrow = (r["n"] == rr["n_open"]
                 and abs(r["hit"] - ra.get("hit", -9)) < 1e-9
                 and abs(r["roi"] - ra.get("roi", -9)) < 1e-9
                 and abs(dmn - rr["dROI_om"]["mean"]) < 1e-9
                 and abs(cmn - rr["CLV_om"]["mean"]) < 1e-9)
        ok_all &= okrow
        print(f"     {rule:<18}{r['n']:>5}{100*r['hit']:>8.2f}"
              f"{100*r['breakeven']:>8.2f}{100*r['roi']:>8.2f}"
              f"{100*dmn:>+9.2f}{cmn:>+10.5f}   "
              f"[D126 n={rr['n_open']} hit={100*ra.get('hit',np.nan):.2f} "
              f"ROI={100*ra.get('roi',np.nan):.2f} "
              f"dROI={100*rr['dROI_om']['mean']:+.2f} "
              f"CLV={rr['CLV_om']['mean']:+.5f}]  "
              f"{'EXACT' if okrow else '*** MISMATCH ***'}")
        repro[rule] = {"n": r["n"], "hit": r["hit"], "roi": r["roi"],
                       "droi": dmn, "clv": cmn, "exact": bool(okrow)}
    # union of the 4 rules on unique games (the D120 live-calibration convention)
    u22 = np.zeros(len(m22), bool)
    for msk in mo22.values():
        u22 |= np.asarray(msk)
    u22 &= ok22
    ru = bo.score_ml(m22, u22, deco22, po22)
    rc = bo.score_ml(m22, u22, decc22, pc22)
    hu = m22.hit.values[u22].astype(bool)
    du = (np.where(hu, deco22[u22] - 1, -1.0) -
          np.where(hu, decc22[u22] - 1, -1.0))
    dumn, _, _, _ = bo.boot_mean(du)
    cumn, culo, cuhi, cut_ = bo.boot_mean(clv22[u22])
    print(f"     {'UNION (unique)':<18}{ru['n']:>5}{100*ru['hit']:>8.2f}"
          f"{100*ru['breakeven']:>8.2f}{100*ru['roi']:>8.2f}"
          f"{100*dumn:>+9.2f}{cumn:>+10.5f}   "
          f"[D126 text: n=109 hit=71.6 be=70.03 ROI=+1.64 @close=-1.83 "
          f"dROI=+3.47 CLV=+0.0258]")
    repro["UNION"] = {"n": ru["n"], "hit": ru["hit"], "be": ru["breakeven"],
                      "roi": ru["roi"], "roi_close": rc["roi"],
                      "droi": dumn, "clv": cumn}
    print(f"     HARNESS VERDICT: "
          f"{'ALL FOUR RULES REPRODUCE D126 EXACTLY' if ok_all else 'MISMATCH'}")
    res["d126_reproduction"] = {"rules": repro, "all_exact": bool(ok_all)}


    # policy table on the D126 window itself (small n, reported for continuity)
    m22p = m22[m22.n_books >= 2].copy()
    if len(m22p) > 20:
        po_, _, okp = bo.price_cols(m22p, "open", "SP")
        mo_, _, _ = bo.registry_masks(m22p, po_, "open")
        pr22 = policy_prices(m22p, policy_margins(m22p))
        u = np.zeros(len(m22p), bool)
        for msk in mo_.values():
            u |= np.asarray(msk)
        print(f"\n     [D126 WINDOW, 2-book games only, n={len(m22p)}] "
              f"union bets n={int((u & okp).sum())}")
        s22 = score_policies(m22p, u & okp, pr22,
                             label="D126-window UNION, execution policies "
                                   "(small n — reported for continuity only)")
        res["d126_window_policies"] = {k: {kk: vv for kk, vv in v.items()
                                           if kk in KEEP}
                                       for k, v in s22.items() if v.get("n")}

    # =====================================================================
    # [3] THE HEADLINE — execution policies, frozen rules, full frame
    # =====================================================================
    print(f"\n{'=' * 116}\n[3] EXECUTION POLICIES ON THE FROZEN RULES "
          f"(bet set fixed by the REGISTERED consensus open — PURE EXECUTION, "
          f"nothing re-selected)\n{'=' * 116}")
    print("  ONEBOOK = the expected result of a bettor with ONE account "
          "(average of the B1-only and B2-only prices) — the honest baseline "
          "for 'what does shopping buy'.\n"
          "  CONS_REG = the price D120/D121/D126 actually transacted "
          "(odds_open) — the registered control.\n"
          "  dBE columns are BREAKEVEN CHANGES: NEGATIVE = the shop lowered "
          f"our breakeven = the {D121_DEFICIT_PP}pp deficit shrank by that "
          f"much.  hit%% CANNOT move: the bet set is identical.")
    ms = m[m.n_books >= 2].reset_index(drop=True)
    pols = policy_margins(ms)
    pr = policy_prices(ms, pols)
    po, deco, oko = bo.price_cols(ms, "open", "SP")
    pc, decc, okc = bo.price_cols(ms, "close", "SP")
    ok = oko & okc
    mo, edge_o, same_o = bo.registry_masks(ms, po, "open")

    def gain_pts(pol_a, pol_b):
        """Points of handicap our side gained going from pol_b to pol_a."""
        return np.where(ms.pick_home.values, pols[pol_b] - pols[pol_a],
                        pols[pol_a] - pols[pol_b])

    g_1bk = 0.5 * (gain_pts("BEST2", "B1") + gain_pts("BEST2", "B2"))
    g_reg = gain_pts("BEST2", "CONS_REG")
    print(f"\n  SHOP GAIN IN POINTS (our side): BEST2 vs a ONE-BOOK bettor "
          f"mean {g_1bk.mean():+.4f} pts;  BEST2 vs the registered consensus "
          f"mean {g_reg.mean():+.4f} pts;  BEST4 vs one book "
          f"{(0.5*(gain_pts('BEST4','B1')+gain_pts('BEST4','B2'))).mean():+.4f}"
          f" pts")
    res["shop_gain_pts"] = {
        "BEST2_vs_ONEBOOK": float(g_1bk.mean()),
        "BEST2_vs_CONS_REG": float(g_reg.mean()),
        "BEST4_vs_ONEBOOK": float(
            (0.5 * (gain_pts("BEST4", "B1") + gain_pts("BEST4", "B2"))).mean())}

    rules_out = {}
    union = np.zeros(len(ms), bool)
    for r in mo:
        union |= np.asarray(mo[r])
    # ALL_UNIVERSE is D121's OWN core test: bet our side on every priced game.
    # That is where the 1.47pp deficit was measured (n=3,682 on real MLs), NOT
    # on the rule union — so it has to be the primary row here too.
    sets = [("ALL_UNIVERSE", ok)] + list(mo.items()) + [("UNION", union)]
    for rule, msk in sets:
        sel = np.asarray(msk) & ok
        if sel.sum() < 5:
            continue
        sc = score_policies(ms, sel, pr,
                            label=f"RULE {rule}   (fires@open n={int(sel.sum())}"
                                  f", hit {100*ms.hit.values[sel].mean():.2f}%)")
        pd_out = {}
        for a, b in (("BEST2", "ONEBOOK"), ("BEST2", "CONS_REG"),
                     ("WORST2", "ONEBOOK"), ("BEST3", "ONEBOOK"),
                     ("BEST4", "ONEBOOK"), ("WORST4", "ONEBOOK")):
            mn, lo, hi, t = paired_droi(ms, sel, pr[a], pr[b])
            sig = "SIG" if (lo > 0 or hi < 0) else "NS"
            print(f"      PAIRED dROI {a:<7}- {b:<8}: {100*mn:+.3f}pp "
                  f"[{100*lo:+.3f},{100*hi:+.3f}] {sig}")
            pd_out[f"{a}-{b}"] = {"mean": mn, "lo": lo, "hi": hi, "t": t}
        clv_out = {}
        for pol in POL_ORDER:
            ps = pr[pol][0]
            c = pc - ps
            a_, l_, h_, t_ = bo.boot_mean(c[sel])
            clv_out[pol] = {"clv": a_, "lo": l_, "hi": h_, "t": t_,
                            "frac_pos": float((c[sel] > 0).mean())}
        base = clv_out["ONEBOOK"]["clv"]
        print("      CLV vs 1-book baseline: " + "  ".join(
            f"{k}={v['clv']:+.5f}({v['clv']-base:+.5f})"
            for k, v in clv_out.items()
            if k in ("ONEBOOK", "CONS_REG", "BEST2", "WORST2", "BEST4")))
        rules_out[rule] = {"scores": {k: {kk: vv for kk, vv in v.items()
                                          if kk in KEEP}
                                      for k, v in sc.items() if v.get("n")},
                           "paired": pd_out, "clv": clv_out}
    res["headline"] = rules_out

    # ---- per-window split discipline (D115 partition) ----------------------
    print(f"\n  PER-WINDOW (D115 partition: DEV = 2023-24+2024-25, "
          f"NONDEV = 2022-23+2025-26) — UNION, ONEBOOK vs BEST2 vs WORST2:")
    print(f"      {'window':<9}{'n':>5}{'hit%':>8}{'be 1bk':>9}{'be BEST2':>10}"
          f"{'dBE_pp':>9}{'ROI 1bk':>9}{'ROI BEST2':>11}{'ROI WORST2':>12}")
    wins = {}
    for wname, ws in (("POOL", None), ("DEV", bo.DEV), ("NONDEV", bo.NONDEV)):
        wsel = union & ok & (ms.season.isin(ws).values if ws
                             else np.ones(len(ms), bool))
        if wsel.sum() < 10:
            continue
        sw = score_policies(ms, wsel, pr, order=["ONEBOOK", "BEST2", "WORST2",
                                                 "CONS_REG", "BEST4"])
        print(f"      {wname:<9}{sw['ONEBOOK']['n']:>5}"
              f"{100*sw['ONEBOOK']['hit']:>8.2f}"
              f"{100*sw['ONEBOOK']['breakeven']:>9.2f}"
              f"{100*sw['BEST2']['breakeven']:>10.2f}"
              f"{sw['BEST2']['d_be_pp_1bk']:>+9.2f}"
              f"{100*sw['ONEBOOK']['roi']:>9.2f}"
              f"{100*sw['BEST2']['roi']:>11.2f}"
              f"{100*sw['WORST2']['roi']:>12.2f}")
        wins[wname] = {k: {kk: vv for kk, vv in v.items() if kk in KEEP}
                       for k, v in sw.items() if v.get("n")}
    res["windows"] = wins

    # ---- did BEST close the 1.47pp deficit? --------------------------------
    print(f"\n  {'=' * 112}\n  DID BEST-OF-N EXECUTION CLOSE THE "
          f"{D121_DEFICIT_PP}pp DEFICIT?  The deficit is a BREAKEVEN excess, "
          f"so shopping closes it by LOWERING breakeven.\n  {'=' * 112}")
    print(f"    {'rule':<18}{'n':>5}{'hit%':>8}{'be 1bk':>9}{'be BEST2':>10}"
          f"{'dBE':>8}{'%of1.47':>9}{'be BEST4':>10}{'dBE':>8}{'%of1.47':>9}"
          f"{'be WORST2':>11}{'dBE':>8}")
    closed = {}
    for rule, d in rules_out.items():
        s = d["scores"]
        if "ONEBOOK" not in s:
            continue
        d2 = -s["BEST2"]["d_be_pp_1bk"]
        d4 = -s["BEST4"]["d_be_pp_1bk"]
        dw = -s["WORST2"]["d_be_pp_1bk"]
        closed[rule] = {"BEST2_pp": d2, "BEST4_pp": d4, "WORST2_pp": dw,
                        "BEST2_pct_of_deficit": 100 * d2 / D121_DEFICIT_PP,
                        "BEST4_pct_of_deficit": 100 * d4 / D121_DEFICIT_PP}
        print(f"    {rule:<18}{s['BEST2']['n']:>5}{100*s['BEST2']['hit']:>8.2f}"
              f"{100*s['ONEBOOK']['breakeven']:>9.2f}"
              f"{100*s['BEST2']['breakeven']:>10.2f}{d2:>+8.2f}"
              f"{100*d2/D121_DEFICIT_PP:>8.0f}%"
              f"{100*s['BEST4']['breakeven']:>10.2f}{d4:>+8.2f}"
              f"{100*d4/D121_DEFICIT_PP:>8.0f}%"
              f"{100*s['WORST2']['breakeven']:>11.2f}{dw:>+8.2f}")
    res["closes_deficit"] = closed

    # =====================================================================
    # [4] HONEST CONTROLS
    # =====================================================================
    print(f"\n{'=' * 116}\n[4] HONEST CONTROLS\n{'=' * 116}")
    ctl = {}
    ub = union & ok

    # (0) MONOTONICITY / SIGN CHECK -----------------------------------------
    print("\n  (0) SIGN CHECK.  With the SIDE held fixed, a better handicap can "
          "only weakly raise the cover rate, so ATS hit% must be monotone "
          "BEST4 >= BEST2 >= ONEBOOK >= WORST2.  If it is not, 'best' is not "
          "best and everything below is wrong.")
    print(f"      {'policy':<9}{'n_live':>8}{'ATS hit%':>10}{'edge_pp':>9}"
          f"{'ROI%':>9}{'  [95% CI]':>18}")
    ats = {}
    for pol in ("WORST4", "WORST2", "B1", "B2", "MID", "CONS_TR",
                "CONS_REG", "BEST2", "BEST3", "BEST4"):
        if pol not in pols:
            continue
        r = score_ats_fixed(ms, ub, pols[pol])
        ats[pol] = r
        print(f"      {pol:<9}{r['n_live']:>8}{100*r['hit']:>10.2f}"
              f"{100*r['edge_pp']:>+9.2f}{100*r['roi']:>9.2f}"
              f"[{100*r['roi_lo']:>+6.1f},{100*r['roi_hi']:>+6.1f}]")
    mono = (ats["BEST4"]["hit"] >= ats["BEST2"]["hit"] >= ats["WORST2"]["hit"])
    print(f"      MONOTONE: {'YES — sign convention verified' if mono else '*** NO — SIGN BUG ***'}")
    print(f"      -- same, whole 2-book universe (n={int(ok.sum())} games, our "
          f"side vs the number, no rule filter):")
    for pol in ("WORST2", "B1", "B2", "CONS_REG", "BEST2", "BEST4"):
        r = score_ats_fixed(ms, ok, pols[pol])
        ats[f"UNIV_{pol}"] = r
        print(f"      {pol:<9}{r['n_live']:>8}{100*r['hit']:>10.2f}"
              f"{100*r['edge_pp']:>+9.2f}{100*r['roi']:>9.2f}"
              f"[{100*r['roi_lo']:>+6.1f},{100*r['roi_hi']:>+6.1f}]")
    ctl["ats_fixed_side"] = ats
    ctl["ats_monotone"] = bool(mono)

    # (i) is taking the best price a SELECTION on stale/informed lines? -------
    print("\n  (i) STALENESS / SELECTION-ON-PRICE.  Pure execution cannot move "
          "hit% (identical bet set).  The real worry: is a book's number best "
          "BECAUSE it is stale, in a way that correlates with the outcome?")
    g = g_1bk[ub]
    h = ms.hit.values[ub].astype(float)
    r_, p_ = stats.pearsonr(g, h)
    print(f"      corr(shop gain in pts, hit) = {r_:+.4f}  p={p_:.3f}  "
          f"n={len(g)}")
    zero = ub & (g_1bk <= 1e-9)
    pos = ub & (g_1bk > 1e-9)
    h0, h1 = ms.hit.values[zero].mean(), ms.hit.values[pos].mean()
    z = (h1 - h0) / np.sqrt(h0 * (1 - h0) / zero.sum() + h1 * (1 - h1) / pos.sum())
    print(f"      hit%% when shopping gains NOTHING (books agree) "
          f"{100*h0:.2f}% (n={int(zero.sum())})  vs when it gains something "
          f"{100*h1:.2f}% (n={int(pos.sum())}):  diff {100*(h1-h0):+.2f}pp  "
          f"z={z:+.2f}  p={2*(1-stats.norm.cdf(abs(z))):.3f}")
    print(f"      {'gain bucket':<14}{'n':>6}{'hit%':>8}{'be BEST2%':>11}"
          f"{'edge_pp':>9}{'ROI@BEST2':>11}")
    buck = {}
    for lo_, hi_, nm in ((-99, -1e-9, "< 0 (rare)"), (-1e-9, 1e-9, "0.0"),
                         (1e-9, 0.30, "0-0.3"), (0.30, 0.55, "~0.5"),
                         (0.55, 1.05, "~1.0"), (1.05, 99, ">=1.5")):
        s2 = ub & (g_1bk > lo_) & (g_1bk <= hi_)
        if s2.sum() < 5:
            continue
        rr2 = bo.score_ml(ms, s2, pr["BEST2"][1], pr["BEST2"][0])
        print(f"      {nm:<14}{rr2['n']:>6}{100*rr2['hit']:>8.2f}"
              f"{100*rr2['breakeven']:>11.2f}{100*rr2['edge_pp']:>+9.2f}"
              f"{100*rr2['roi']:>11.2f}")
        buck[nm] = {k: rr2[k] for k in ("n", "hit", "breakeven", "edge_pp",
                                        "roi")}
    # does the BEST book's number move further toward the close (= it was the
    # stale one), and does the game then go against us?
    mv_best = np.where(ms.pick_home.values,
                       ms.close_margin.values - pols["BEST2"],
                       pols["BEST2"] - ms.close_margin.values)
    mv_1bk = np.where(ms.pick_home.values,
                      ms.close_margin.values - 0.5 * (pols["B1"] + pols["B2"]),
                      0.5 * (pols["B1"] + pols["B2"]) - ms.close_margin.values)
    act = np.where(ms.pick_home.values, ms.margin_actual.values,
                   -ms.margin_actual.values)
    ab = np.where(ms.pick_home.values, -pols["BEST2"], pols["BEST2"])
    a1 = np.where(ms.pick_home.values,
                  -0.5 * (pols["B1"] + pols["B2"]),
                  0.5 * (pols["B1"] + pols["B2"]))
    print(f"      open->close drift AGAINST our side (pts; + = the market "
          f"moved away from the number we took): BEST2 {mv_best[ub].mean():+.4f}"
          f"  ONEBOOK {mv_1bk[ub].mean():+.4f}  (the gap IS the shop gain, "
          f"mechanically — it is not evidence either way)")
    print(f"      REALISED ATS margin vs the number we took (actual - handicap,"
          f" + = we cover): BEST2 {(act-ab)[ub].mean():+.4f} pts   ONEBOOK "
          f"{(act-a1)[ub].mean():+.4f} pts   difference "
          f"{((act-ab)-(act-a1))[ub].mean():+.4f} = the shop gain, and it is "
          f"the ONLY thing the shop changes")
    ctl["staleness"] = {"corr_gain_hit": float(r_), "p": float(p_),
                        "hit_gain_zero": float(h0), "n_zero": int(zero.sum()),
                        "hit_gain_pos": float(h1), "n_pos": int(pos.sum()),
                        "z": float(z),
                        "p_twosided": float(2 * (1 - stats.norm.cdf(abs(z)))),
                        "buckets": buck}

    # (ii) outlier / limit-and-void realism haircut --------------------------
    print("\n  (ii) OUTLIER HAIRCUT.  Real books limit or void obviously wrong "
          "numbers, so a 'best' price far from the other quotes is not "
          "reliably transactable at size.")
    dev = np.abs(pols["BEST2"] - 0.5 * (pols["B1"] + pols["B2"])) * 2.0
    for t in (1.0, 1.5, 2.0, 3.0):
        print(f"      fraction of BEST2 prices >{t} pts from the OTHER book: "
              f"{100*float((dev[ub] > t).mean()):.2f}%")
    keep = ub & (dev <= 1.5)
    s_ex = score_policies(ms, keep, pr,
                          order=["ONEBOOK", "CONS_REG", "BEST2", "WORST2",
                                 "BEST4"],
                          label="UNION EXCLUDING games where the two books "
                                "disagree by >1.5 pts")
    capped = np.where(ms.pick_home.values,
                      np.maximum(pols["BEST2"],
                                 0.5 * (pols["B1"] + pols["B2"]) - 0.75),
                      np.minimum(pols["BEST2"],
                                 0.5 * (pols["B1"] + pols["B2"]) + 0.75))
    pr_cap = dict(pr)
    pr_cap["BESTCAP"] = priced(ms, capped)
    s_cap = score_policies(ms, ub, pr_cap,
                           order=["ONEBOOK", "CONS_REG", "BEST2", "BESTCAP"],
                           label="UNION with the realised shop gain CAPPED at "
                                 "0.75 pts vs the midpoint (= 1.5 pts of "
                                 "book-to-book disagreement)")
    ctl["outliers"] = {
        "frac_gt": {str(t): float((dev[ub] > t).mean())
                    for t in (1.0, 1.5, 2.0, 3.0)},
        "excl_1.5": {k: {kk: vv for kk, vv in v.items() if kk in KEEP}
                     for k, v in s_ex.items() if v.get("n")},
        "capped": {k: {kk: vv for kk, vv in v.items() if kk in KEEP}
                   for k, v in s_cap.items() if v.get("n")}}

    # (iii) how far does a BIGGER shop go? ----------------------------------
    print("\n  (iii) A BIGGER SHOP: a MEASURED quote-count ladder, then an "
          "EXPLICIT EXTRAPOLATION.  Three books is a small shop and we do not "
          "even have three — the panel is 2 books.")
    from itertools import combinations
    quotes = ["B1", "B2", "CONS_TR", "CONS_REG"]
    ph = ms.pick_home.values
    lad = {}
    print(f"      MEASURED LADDER (average over ALL subsets of size k of the 4 "
          f"available quotes; k=1 is the one-number bettor):")
    print(f"      {'k quotes':<10}{'subsets':>9}{'gain pts':>11}{'be%':>9}"
          f"{'dBE_pp':>9}{'ROI%':>9}")
    base_be = None
    for k_ in (1, 2, 3, 4):
        subs = list(combinations(quotes, k_))
        bes, rois, gains = [], [], []
        for s_ in subs:
            marg = _best(ph, [pols[q] for q in s_])
            p_s, d_s = priced(ms, marg)
            r_ = bo.score_ml(ms, ub & np.isfinite(p_s), d_s, p_s)
            bes.append(r_["breakeven"])
            rois.append(r_["roi"])
            gains.append(np.where(ph, -marg, marg)[ub].mean())
        be_ = float(np.mean(bes))
        gp_ = float(np.mean(gains))
        if k_ == 1:
            base_be, base_g = be_, gp_
        lad[k_] = {"n_subsets": len(subs), "gain_pts": gp_ - base_g,
                   "be": be_, "dbe_pp": 100 * (be_ - base_be),
                   "roi": float(np.mean(rois))}
        print(f"      {k_:<10}{len(subs):>9}{gp_ - base_g:>+11.3f}"
              f"{100*be_:>9.2f}{100*(be_-base_be):>+9.2f}"
              f"{100*np.mean(rois):>9.2f}")
    two = ms[ms.n_books >= 2]
    _d = two[["mb1", "mb2"]].max(axis=1) - two[["mb1", "mb2"]].min(axis=1)
    dsp, sd_diff = float(_d.mean()), float(_d.std())
    sigma = dsp / (2 / np.sqrt(np.pi))          # E|X-Y| = 2*sigma/sqrt(pi)
    print(f"      CAVEAT ON THE k=3,4 ROWS: only TWO of the four quotes are "
          f"BOOKS observed simultaneously in one panel row. CONS_TR and "
          f"CONS_REG are VENDOR SNAPSHOTS (TeamRankings' summary cell; "
          f"SBR/ESPN+ActionNetwork's captured open) taken at different times "
          f"by different scrapers — their mutual disagreement (mean|diff| "
          f"{res['vendor_bridge']['mad']:.2f} pts) is LARGER than the "
          f"book-to-book disagreement ({dsp:.2f} pts), so the k>=3 rows MIX "
          f"real shopping with vendor/timing noise and are an UPPER BOUND, "
          f"not a clean 3- or 4-book shop.")
    EN = {1: 0.0, 2: 0.5642, 3: 0.8463, 4: 1.0294, 5: 1.1630, 6: 1.2672,
          7: 1.3522, 8: 1.4236}
    obs2 = -rules_out["UNION"]["scores"]["BEST2"]["d_be_pp_1bk"]
    g2 = float(g_1bk[ub].mean())
    print(f"\n      EXTRAPOLATION (Gaussian idiosyncratic-deviation model, "
          f"calibrated on the ONLY clean measurement — the 2 real books):")
    print(f"        per-book idiosyncratic sd sigma = {sigma:.4f} pts "
          f"(from E|b1-b2| = {dsp:.4f} = 2*sigma/sqrt(pi)).  A common "
          f"component shared by all books cancels in a best-of-N, so this is "
          f"the right sd to extrapolate on.")
    ext = {"ladder": lad, "measured_2book_pts": g2, "measured_2book_pp": obs2,
           "by_N": {}}
    for n_, e_ in EN.items():
        if n_ < 2:
            continue
        pts = sigma * e_
        scale = pts / (sigma * EN[2])
        ext["by_N"][n_] = {"pts": pts, "dbe_pp": obs2 * scale}
        print(f"        N={n_} books -> gain {pts:.3f} pts "
              f"({scale:.2f}x best-of-2) -> breakeven cut ~{obs2*scale:+.2f}pp"
              f"   {'CLOSES' if obs2*scale >= D121_DEFICIT_PP else 'SHORT OF'} "
              f"{D121_DEFICIT_PP}pp")
    print(f"      WHY THIS IS AN UPPER BOUND, NOT A FORECAST: the observed "
          f"deviation distribution is NOT Gaussian — 36% of book pairs are "
          f"EXACT TIES and sd/mean|diff| = {sd_diff/dsp:.2f} vs the Gaussian "
          f"1.25, i.e. a spike at zero plus a fat tail. A spike at zero means "
          f"extra books often add NOTHING (they copy), which the Gaussian "
          f"E[max] does not model.  Half-point granularity truncates small "
          f"gains, real accounts get limited on the outlier book, and the "
          f"5-8-book number below is therefore a CEILING.  The only number "
          f"here that is a MEASUREMENT is the 2-book one: {obs2:+.2f}pp.")
    ctl["extrapolation"] = {"sigma_pts": sigma, "dispersion_pts": dsp,
                            "sd_over_mean": sd_diff / dsp, **ext}

    # (iv) family-wise arithmetic --------------------------------------------
    print("\n  (iv) FAMILY-WISE ARITHMETIC (D121's 9-observed-vs-14.4-expected "
          "standard).")
    cells = []
    for rule, d in rules_out.items():
        for pol, r in d["scores"].items():
            if r.get("n") and np.isfinite(r.get("null_be", np.nan)):
                cells.append((f"{rule}|{pol}", r["null_be"], r["roi"], r["n"]))
    for w, d in wins.items():
        for pol, r in d.items():
            if r.get("n") and np.isfinite(r.get("null_be", np.nan)):
                cells.append((f"WIN_{w}|{pol}", r["null_be"], r["roi"],
                              r["n"]))
    k = len(cells)
    sig = [c for c in cells if c[1] < 0.05]
    p_at = (1.0 - sum(comb(k, i) * 0.05 ** i * 0.95 ** (k - i)
                      for i in range(len(sig)))) if k else np.nan
    print(f"      cells scored = {k};  expected significant at p<0.05 under a "
          f"global null = {0.05*k:.1f};  OBSERVED = {len(sig)}  "
          f"-> P(chance >= {len(sig)}) = {p_at:.4f}")
    for c in sig:
        print(f"        SIG {c[0]:<28} ROI {100*c[2]:+.2f}%  n={c[3]}  "
              f"p={c[1]:.4f}")
    print("      (cells heavily OVERLAP — the same bets re-priced — so this is "
          "an UPPER BOUND on the surprise, exactly as in D120.)")
    ctl["familywise"] = {"cells": k, "expected": 0.05 * k,
                         "observed": len(sig), "p_atleast": float(p_at),
                         "sig": [c[0] for c in sig]}

    # (v) spreads are not moneylines -----------------------------------------
    print("\n  (v) SPREADS ARE NOT MONEYLINES.  D121's 1.47pp headline was on "
          "REAL opening MLs; TeamRankings publishes SPREADS ONLY, so every "
          "number here goes through p = sigmoid(margin/6.96) and a 1.045 "
          "proportional overround.  THE CONVERSION IS THE ASSUMPTION.")
    mlm = m[m.open_ml_home.notna() & (m.n_books >= 2)].reset_index(drop=True)
    if len(mlm) > 100:
        psp, dsp_, oksp = bo.price_cols(mlm, "open", "SP")
        pml, dml, okml = bo.price_cols(mlm, "open", "ML")
        both = oksp & okml
        d_be = (1.0 / dsp_[both]) - (1.0 / dml[both])
        print(f"      on the {int(both.sum())} games carrying BOTH a real "
              f"opening ML and a 2-book TR panel, the SP convention's implied "
              f"breakeven is {100*d_be.mean():+.3f}pp HIGHER than the real "
              f"ML's (D120 measured +1.98pp on the rule bets) — the SP frame "
              f"is PESSIMISTIC by about that much, i.e. in our favour.")
        ctl["sp_minus_ml_be_pp"] = float(100 * d_be.mean())
    for sc_ in (6.5, 6.96, 7.5):
        print(f"      sensitivity: at spread scale {sc_} a 1-pt line "
              f"difference is worth {100*OVERROUND*0.71*0.29/sc_:.2f}pp of "
              f"breakeven at p=0.71 (the union's mean our-side price); the "
              f"measured BEST2 gain rescales to "
              f"{obs2*6.96/sc_:+.2f}pp")
    print(f"      => the ~2pp SP-vs-ML pessimism is a LEVEL bias that cancels "
          f"in every dBE above (both policies go through the same map); the "
          f"spread-scale choice moves the dBE by about +-7%, so the 2-book "
          f"gain is {obs2*6.96/7.5:.2f}-{obs2*6.96/6.5:.2f}pp under any "
          f"defensible scale.  Neither uncertainty reaches "
          f"{D121_DEFICIT_PP}pp.")

    # (v-b) THE LITERAL QUESTION, ON THE REAL-MONEYLINE FRAME ----------------
    print(f"\n  (v-b) THE LITERAL QUESTION ON D121'S OWN FRAME.  D121's "
          f"{D121_DEFICIT_PP}pp is hit-minus-breakeven on REAL opening "
          f"moneylines.  Here is that same quantity on the union bets that "
          f"ALSO carry a 2-book TR panel, with the measured shop gain applied "
          f"to it.")
    print(f"      D121's deficit was measured on the WHOLE UNIVERSE (bet our "
          f"side on every game, n=3,682), not on the rules — so the universe "
          f"row is the one that answers the literal question.")
    mlfull = m[m.open_ml_home.notna() & (m.n_books >= 2)].reset_index(drop=True)
    res["ml_frame"] = {}
    if len(mlfull) > 100:
        po_f, _, okf = bo.price_cols(mlfull, "open", "SP")
        mo_f, _, _ = bo.registry_masks(mlfull, po_f, "open")
        uf = np.zeros(len(mlfull), bool)
        for r in mo_f:
            uf |= np.asarray(mo_f[r])
        pml_f, dml_f, okml_f = bo.price_cols(mlfull, "open", "ML")
        obs_u = -rules_out["ALL_UNIVERSE"]["scores"]["BEST2"]["d_be_pp_1bk"]
        # --- SECOND HARNESS CONTROL + the 2-book subsample's own bias -------
        pml_a, dml_a, okml_a = bo.price_cols(m, "open", "ML")
        r_all = bo.score_ml(m, okml_a, dml_a, pml_a)
        r_2b = bo.score_ml(m, okml_a & (m.n_books >= 2).values, dml_a, pml_a)
        r_lt = bo.score_ml(m, okml_a & (m.n_books < 2).values, dml_a, pml_a)
        print(f"      HARNESS CONTROL #2 — D121's headline reproduced on the "
              f"FULL ML frame: n={r_all['n']} hit {100*r_all['hit']:.2f}% vs "
              f"breakeven {100*r_all['breakeven']:.2f}% = "
              f"{100*r_all['edge_pp']:+.2f}pp, ROI {100*r_all['roi']:+.2f}%  "
              f"[D121 registered n=3,682, 68.14 vs 69.61, -1.47pp, -1.64%]  "
              f"{'EXACT' if r_all['n'] == 3682 and abs(100*r_all['edge_pp'] + D121_DEFICIT_PP) < 0.01 else '*** MISMATCH ***'}")
        print(f"      *** THE 2-BOOK SUBSAMPLE IS NOT A RANDOM HALF ***  "
              f"2-book games: n={r_2b['n']} edge {100*r_2b['edge_pp']:+.2f}pp; "
              f"games WITHOUT 2 books: n={r_lt['n']} edge "
              f"{100*r_lt['edge_pp']:+.2f}pp.  TR's 2-book coverage collapses "
              f"in 2025-26 (the season the model did best in), so the shop is "
              f"measured on the WORSE half of the corpus.  The shop GAIN is a "
              f"price property and transfers; the LEVEL does not.")
        res["ml_frame"]["_harness"] = {
            "full": {k: r_all[k] for k in KEEP if k in r_all},
            "two_book": {k: r_2b[k] for k in KEEP if k in r_2b},
            "lt_two_book": {k: r_lt[k] for k in KEEP if k in r_lt},
            "reproduces_d121": bool(r_all["n"] == 3682 and abs(
                100 * r_all["edge_pp"] + D121_DEFICIT_PP) < 0.01)}
        print(f"      APPLIED TO D121'S OWN REGISTERED DEFICIT: "
              f"-{D121_DEFICIT_PP:.2f}pp + {obs_u:.2f}pp = "
              f"{obs_u - D121_DEFICIT_PP:+.2f}pp — best-of-2 closes "
              f"{100*obs_u/D121_DEFICIT_PP:.0f}% of it and leaves it "
              f"{'CLEAR' if obs_u >= D121_DEFICIT_PP else 'SHORT'}.")
        res["ml_frame"]["_applied_to_d121"] = obs_u - D121_DEFICIT_PP
        for nm, selml, gainpp in (
                ("ALL_UNIVERSE (D121's own test)", okf & okml_f, obs_u),
                ("UNION of the 4 rules", uf & okf & okml_f, obs2)):
            rml = bo.score_ml(mlfull, selml, dml_f, pml_f)
            resid = 100 * rml["edge_pp"] + gainpp
            print(f"      {nm}: n={rml['n']}  hit {100*rml['hit']:.2f}%  "
                  f"breakeven {100*rml['breakeven']:.2f}%  = "
                  f"{100*rml['edge_pp']:+.2f}pp  ROI {100*rml['roi']:+.2f}% "
                  f"[{100*rml['roi_lo']:+.1f},{100*rml['roi_hi']:+.1f}]"
                  f"   (D121 registered {-D121_DEFICIT_PP:+.2f}pp on n=3,682 "
                  f"universe)")
            print(f"        + the MEASURED 2-book shop gain ({gainpp:+.2f}pp "
                  f"of breakeven)  ->  residual {resid:+.2f}pp  "
                  f"{'CLEARS' if resid >= 0 else 'STILL SHORT'}")
            res["ml_frame"][nm] = {**{k: rml[k] for k in KEEP if k in rml},
                                   "deficit_pp": 100 * rml["edge_pp"],
                                   "shop_gain_pp": gainpp,
                                   "residual_pp": resid}
        print(f"      NOTE the transfer assumption: the gain was measured on "
              f"SPREAD quotes and is applied to an ML breakeven. It is the "
              f"same probability shift either way, but a real ML shop is not "
              f"guaranteed to disperse as much as a spread shop does; and "
              f"this 2-book subsample is NOT D121's n=3,682, so its own "
              f"deficit differs from -1.47pp by sampling.")

    # (vi) the RE-FIRE arm ---------------------------------------------------
    print("\n  (vi) RE-FIRE ARM (SECONDARY).  A real shopper computes edge "
          "against the BEST price, which CHANGES the bet set — that is "
          "selection on the price, so it is reported apart from the headline.")
    ms2 = ms.copy()
    p_best, dec_best = pr["BEST2"]
    ph_best = bo.sigmoid(pols["BEST2"] / SC)
    con = ro_connect()
    try:
        so2 = star_out_map(con)
    finally:
        con.close()
    som2 = so2.rename(columns={"team": "fav_team"})[["game_id", "fav_team",
                                                     "star_out"]]
    ms2["fav_team"] = np.where(ph_best >= 0.5, ms2.home, ms2.away)
    ms2 = ms2.drop(columns=["fav_star_out_open"]).merge(
        som2, on=["game_id", "fav_team"], how="left")
    ms2["fav_star_out_open"] = ms2.star_out.fillna(False).astype(bool)
    ms2 = ms2.drop(columns=["star_out", "fav_team"])
    mo_b, _, _ = bo.registry_masks(ms2, p_best, "open")
    union_b = np.zeros(len(ms2), bool)
    for r in mo_b:
        union_b |= np.asarray(mo_b[r])
    union_b &= ok
    added, kept, dropped = union_b & ~ub, union_b & ub, ub & ~union_b
    print(f"      union bets: consensus-fired {int(ub.sum())} -> best-fired "
          f"{int(union_b.sum())}  (+{int(added.sum())} added, "
          f"-{int(dropped.sum())} dropped, {int(kept.sum())} kept)")
    rb = bo.score_ml(ms2, union_b, dec_best, p_best)
    print(f"      RE-FIRE @BEST2   n={rb['n']:<5} hit={100*rb['hit']:.2f} "
          f"be={100*rb['breakeven']:.2f} edge={100*rb['edge_pp']:+.2f}pp "
          f"ROI={100*rb['roi']:+.2f}% [{100*rb['roi_lo']:+.1f},"
          f"{100*rb['roi_hi']:+.1f}]")
    ref = {"union": {k: rb[k] for k in KEEP if k in rb},
           "n_added": int(added.sum()), "n_dropped": int(dropped.sum())}
    for nm, s3 in (("ADDED bets only", added), ("KEPT bets", kept)):
        if s3.sum() < 5:
            continue
        r3 = bo.score_ml(ms2, s3, dec_best, p_best)
        print(f"      {nm:<16} n={r3['n']:<5} hit={100*r3['hit']:.2f} "
              f"be={100*r3['breakeven']:.2f} edge={100*r3['edge_pp']:+.2f}pp "
              f"ROI={100*r3['roi']:+.2f}%")
        ref[nm] = {k: r3[k] for k in KEEP if k in r3}
    print("      READ: if the ADDED bets are worse than the KEPT bets, the "
          "extra volume a shop unlocks is NEGATIVE selection and the "
          "re-fire arm is not a free upgrade over pure execution.")
    ctl["refire"] = ref

    res["controls"] = ctl

    # =====================================================================
    # [5] CLV — the live programme's actual target
    # =====================================================================
    print(f"\n{'=' * 116}\n[5] CLV UNDER EACH EXECUTION POLICY (CLV is what "
          f"the October programme measures)\n{'=' * 116}")
    print(f"    {'set':<18}{'n':>6}{'ONEBOOK':>10}{'CONS_REG':>10}"
          f"{'BEST2':>10}{'WORST2':>10}{'BEST4':>10}"
          f"{'   BEST2-ONEBOOK [95% CI]':>32}")
    clv_tab = {}
    for rule, msk in sets:
        sel = np.asarray(msk) & ok
        if sel.sum() < 5:
            continue
        row = {}
        for pol in POL_ORDER:
            a_, l_, h_, t_ = bo.boot_mean((pc - pr[pol][0])[sel])
            row[pol] = {"clv": a_, "lo": l_, "hi": h_, "t": t_}
        d_, dl_, dh_, dt_ = bo.boot_mean(
            ((pc - pr["BEST2"][0]) - (pc - pr["ONEBOOK"][0]))[sel])
        row["BEST2_minus_ONEBOOK"] = {"mean": d_, "lo": dl_, "hi": dh_,
                                      "t": dt_}
        print(f"    {rule:<18}{int(sel.sum()):>6}{row['ONEBOOK']['clv']:>+10.5f}"
              f"{row['CONS_REG']['clv']:>+10.5f}{row['BEST2']['clv']:>+10.5f}"
              f"{row['WORST2']['clv']:>+10.5f}{row['BEST4']['clv']:>+10.5f}"
              f"   {d_:>+9.5f}[{dl_:>+8.5f},{dh_:>+8.5f}]"
              f"{' SIG' if (dl_ > 0 or dh_ < 0) else ' NS'}")
        clv_tab[rule] = row
    res["clv"] = clv_tab
    print("    NOTE: shopping raises measured CLV ARITHMETICALLY — CLV = "
          "p_close_side - p_open_side and a better open lowers p_open_side. "
          "The CLV gain therefore EQUALS the price gain: real money in a "
          "CLV-scored programme, but NOT new information, and NOT a second "
          "independent confirmation of the ROI result.")

    # monthly CLV bands under each policy
    sub = ms[ub].copy()
    sub["clv_1bk"] = (pc - pr["ONEBOOK"][0])[ub]
    sub["clv_best"] = (pc - pr["BEST2"][0])[ub]
    sub["clv_reg"] = (pc - pr["CONS_REG"][0])[ub]
    sub["ym"] = sub.game_date.dt.to_period("M").astype(str)
    gm = sub.groupby("ym")[["clv_1bk", "clv_best", "clv_reg"]].agg(
        ["size", "mean"])
    gm = gm[gm[("clv_1bk", "size")] >= 3]
    if len(gm):
        nmed = float(gm[("clv_1bk", "size")].median())
        print(f"\n    MONTHLY CLV BANDS (union @open, {len(gm)} months, median "
              f"{nmed:.0f} bets/month):")
        mb = {}
        for c in ("clv_reg", "clv_1bk", "clv_best"):
            mu = float(gm[(c, "mean")].mean())
            sd = float(sub[c].std(ddof=1))
            mb[c] = {"mean": mu, "sd": sd,
                     "red": mu - 2 * sd / np.sqrt(nmed),
                     "good": mu + 2 * sd / np.sqrt(nmed),
                     "pct_pos": float((gm[(c, "mean")] > 0).mean())}
            print(f"      {c:<10} mean-of-months {mu:+.5f}  per-bet sd "
                  f"{sd:.5f}  ->  red flag {mb[c]['red']:+.5f}, good "
                  f"{mb[c]['good']:+.5f};  {100*mb[c]['pct_pos']:.0f}% of "
                  f"months positive")
        print(f"      (D120/D121 registered ML-frame bands: red -0.0131, "
              f"good +0.0200.  These SP-frame bands are NOT interchangeable "
              f"with them.)")
        res["monthly_clv"] = {"n_months": int(len(gm)), "median_bets": nmed,
                              "bands": mb}

    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
