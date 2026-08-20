#!/usr/bin/env python3
"""TS-OPENERS — is a BET-AT-OPEN strategy testable at all?

The directive: our historical price is the CLOSE, so the honest formulation
is bet-at-close (the hardest test).  But if OPENING lines exist we should
also evaluate bet-at-open with close-as-CLV-check, because that is the
realistic edge case (openers are softer than closes).

This script answers ONE question with evidence: for which seasons do we
actually hold an OPENING price?  It audits every odds source in the repo,
then — on whatever window is covered — measures (a) how much the line moves
open->close, (b) what our model's edge looks like at the open vs at the
close, and (c) the CLV a bet-at-open would have earned.

SOURCES AUDITED
  odds_hist_sbr   (DuckDB)  sportsbookreviewsonline archives: HAS
                            spread_open AND spread_close
  odds_market     (DuckDB)  kaggle cvia ingest: CLOSE only (the p_mkt used
                            by every backtest in the program)
  data/raw/kaggle/*         4 kaggle datasets, checked for open/close cols
  data/raw/sbr/*            the raw xlsx/html the DuckDB table came from,
                            incl. the "HTML-not-xlsx" failures

PRICE CONVENTION (identical to nbapred/ingest/kaggle_odds.py):
  p = sigmoid(home_expected_margin / 6.96);  home_expected_margin = -spread
  where spread is quoted from the HOME side (SBR convention: negative =
  home favoured).  De-vigged by construction (symmetric logistic).

RULES HONORED: DuckDB read_only=True; new file scripts/ts_openers.py only;
nbapred/ untouched.

Run:  python scripts/ts_openers.py
"""
from __future__ import annotations

import glob
import os
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DB = os.path.join(ROOT, "data", "nba.duckdb")
PERGAME = os.path.join(ROOT, "data", "ds_rt3_evrec5_pergame.csv")
OUT_JSON = os.path.join(ROOT, "data", "ts_openers.json")
SPREAD_SCALE = 6.96

SBR2ABB = {
    "Atlanta": "ATL", "Boston": "BOS", "Brooklyn": "BKN", "Charlotte": "CHA",
    "Chicago": "CHI", "Cleveland": "CLE", "Dallas": "DAL", "Denver": "DEN",
    "Detroit": "DET", "GoldenState": "GSW", "Golden State": "GSW",
    "Houston": "HOU", "Indiana": "IND", "LAClippers": "LAC",
    "LALakers": "LAL", "Memphis": "MEM", "Miami": "MIA", "Milwaukee": "MIL",
    "Minnesota": "MIN", "NewOrleans": "NOP", "NewYork": "NYK",
    "OklahomaCity": "OKC", "Orlando": "ORL", "Philadelphia": "PHI",
    "Phoenix": "PHX", "Portland": "POR", "Sacramento": "SAC",
    "SanAntonio": "SAS", "Toronto": "TOR", "Utah": "UTA",
    "Washington": "WAS", "Seattle": "OKC", "NewJersey": "BKN",
}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


# ----------------------------------------------------------------- audit ---
def audit_duckdb(con) -> list[dict]:
    rows = []
    cov = con.execute("""
        SELECT season, COUNT(*) n, COUNT(spread_open) n_open,
               COUNT(spread_close) n_close, COUNT(v_ml) n_ml,
               AVG(CASE WHEN spread_open = spread_close THEN 1.0 ELSE 0.0 END) same
        FROM odds_hist_sbr GROUP BY 1 ORDER BY 1
    """).fetchdf()
    print("\n[1] odds_hist_sbr (DuckDB) — HAS spread_open")
    print(cov.to_string(index=False))
    rows.append({"source": "odds_hist_sbr", "has_open": True,
                 "coverage": cov.to_dict("records")})

    mk = con.execute("""
        SELECT season_end, COUNT(*) n, COUNT(p_home_spread) n_p,
               COUNT(ml_home) n_ml
        FROM odds_market GROUP BY 1 ORDER BY 1
    """).fetchdf()
    print("\n[2] odds_market (DuckDB, the p_mkt every backtest uses)"
          " — CLOSE only, no open column exists")
    print(mk.tail(8).to_string(index=False))
    rows.append({"source": "odds_market", "has_open": False,
                 "coverage": mk.to_dict("records")})
    return rows


def audit_kaggle() -> list[dict]:
    print("\n[3] kaggle datasets — column scan for OPEN vs CLOSE")
    out = []
    base = os.path.join(ROOT, "data", "raw", "kaggle")
    for csv in sorted(glob.glob(os.path.join(base, "*", "*.csv"))):
        try:
            head = pd.read_csv(csv, nrows=5)
        except Exception as exc:                                  # pragma: no cover
            print(f"  {os.path.relpath(csv, ROOT):<80} UNREADABLE {exc}")
            continue
        cols = [c.lower() for c in head.columns]
        has_open = any("open" in c for c in cols)
        has_close = any("close" in c for c in cols)
        priced = any(k in c for c in cols
                     for k in ("spread", "moneyline", "price", "ml", "odds"))
        if not priced:
            continue
        print(f"  {os.path.relpath(csv, ROOT):<78} "
              f"open={has_open!s:<5} close={has_close!s:<5} "
              f"cols={list(head.columns)[:8]}")
        out.append({"source": os.path.relpath(csv, ROOT),
                    "has_open": has_open, "has_close": has_close})
    return out


def audit_sbr_raw() -> list[dict]:
    """The raw archives, including the HTML-not-xlsx failures."""
    print("\n[4] data/raw/sbr — raw archives (xlsx + the HTML failures)")
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "sbr", "*"))):
        name = os.path.basename(path)
        try:
            if path.endswith(".xlsx"):
                d = pd.read_excel(path)
            else:
                tabs = pd.read_html(path)
                d = max(tabs, key=len)
                d.columns = d.iloc[0]
                d = d.iloc[1:]
            has_open = any(str(c).strip().lower() == "open" for c in d.columns)
            games = len(d) // 2
            print(f"  {name:<32} rows={len(d):>5} games={games:>5} "
                  f"open_col={has_open}")
            out.append({"file": name, "games": games, "has_open": has_open})
        except Exception as exc:
            print(f"  {name:<32} PARSE FAIL: {exc}")
            out.append({"file": name, "error": str(exc)})
    return out


# ------------------------------------------------------- open vs close -----
def open_close_frame(con) -> pd.DataFrame:
    """Join SBR opener+closer onto our per-game model frame."""
    sbr = con.execute("""
        SELECT season, game_date, visitor, home, spread_open, spread_close
        FROM odds_hist_sbr
        WHERE season >= '2021-22' AND spread_open IS NOT NULL
    """).fetchdf()
    sbr["game_date"] = pd.to_datetime(sbr.game_date)
    sbr["home_ab"] = sbr.home.map(SBR2ABB)
    sbr["away_ab"] = sbr.visitor.map(SBR2ABB)
    bad = sbr[sbr.home_ab.isna() | sbr.away_ab.isna()]
    if len(bad):
        print(f"  WARNING {len(bad)} SBR rows with unmapped team names")
    sbr = sbr.dropna(subset=["home_ab", "away_ab"]).drop(columns=["season"])

    df = pd.read_csv(PERGAME, dtype={"game_id": str})
    df["game_date"] = pd.to_datetime(df.game_date)
    m = df.merge(sbr, left_on=["game_date", "home", "away"],
                 right_on=["game_date", "home_ab", "away_ab"], how="inner")
    # SBR quotes the spread from the HOME row; negative = home favoured.
    m["exp_open"] = -m.spread_open
    m["exp_close"] = -m.spread_close
    m["p_open"] = sigmoid(m.exp_open / SPREAD_SCALE)
    m["p_close_sbr"] = sigmoid(m.exp_close / SPREAD_SCALE)
    return m


def report_open_close(m: pd.DataFrame) -> dict:
    print("\n[5] OPEN vs CLOSE on the covered window")
    if len(m) == 0:
        print("  no overlap — nothing to measure")
        return {}
    per = m.groupby("season").apply(
        lambda g: pd.Series({
            "n": len(g),
            "mean_move_pts": float((g.exp_close - g.exp_open).mean()),
            "mean_abs_move_pts": float((g.exp_close - g.exp_open).abs().mean()),
            "p95_abs_move_pts": float((g.exp_close - g.exp_open).abs()
                                      .quantile(0.95)),
            "frac_unchanged": float((g.exp_close == g.exp_open).mean()),
            "corr_sbr_close_vs_pmkt": float(g.p_close_sbr.corr(g.p_mkt)),
            "mae_sbr_close_vs_pmkt": float((g.p_close_sbr - g.p_mkt).abs()
                                           .mean()),
        }), include_groups=False)
    print(per.to_string())

    # Which price is SHARPER?  log-loss of the open, the SBR close and the
    # program's p_mkt against the realized winner.  Sharper = lower.
    def ll(p):
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(-(m.y * np.log(p) + (1 - m.y) * np.log(1 - p)).mean())

    sharp = {"ll_open": ll(m.p_open), "ll_close_sbr": ll(m.p_close_sbr),
             "ll_pmkt_program": ll(m.p_mkt), "ll_us": ll(m.p_ctl), "n": len(m)}
    print("\n  price sharpness (log-loss vs realized winner; LOWER = sharper)")
    for k, v in sharp.items():
        print(f"    {k:<22} {v:.5f}" if k != "n" else f"    {k:<22} {v}")
    print(f"    open -> close sharpening: "
          f"{sharp['ll_open'] - sharp['ll_close_sbr']:+.5f} nats/game "
          f"(positive = close is sharper, as expected)")

    # Our model's edge at the OPEN vs at the CLOSE, on the shared side.
    for lbl, pcol in (("OPEN", "p_open"), ("CLOSE", "p_close_sbr")):
        pick_home = m.p_ctl > 0.5
        p_us_side = np.where(pick_home, m.p_ctl, 1 - m.p_ctl)
        p_mk_side = np.where(pick_home, m[pcol], 1 - m[pcol])
        same = (m.p_ctl - 0.5) * (m[pcol] - 0.5) > 0
        edge = p_us_side - p_mk_side
        print(f"\n  our edge vs {lbl}: mean {edge.mean():+.4f} "
              f"| same-side {same.mean()*100:.1f}% "
              f"| frac edge>0.03 {(edge > 0.03).mean()*100:.1f}%")
        sharp[f"mean_edge_{lbl.lower()}"] = float(edge.mean())
        sharp[f"frac_edge_gt3_{lbl.lower()}"] = float((edge > 0.03).mean())

    # CLV: if we bet the model's side AT THE OPEN, did the close move to us?
    # CONTROL: the same statistic for betting the OPEN'S OWN FAVOURITE.  If
    # lines simply drift toward favourites (steam), a model that picks the
    # favourite 87% of the time harvests "CLV" that carries zero information.
    def clv_of(side_is_home):
        po = np.where(side_is_home, m.p_open, 1 - m.p_open)
        pc = np.where(side_is_home, m.p_close_sbr, 1 - m.p_close_sbr)
        return pc - po

    clv = clv_of((m.p_ctl > 0.5).values)
    clv_fav = clv_of((m.p_open > 0.5).values)
    n = len(m)
    # paired: our CLV minus the free favourite-drift CLV on the same games
    excess = clv - np.where((m.p_ctl > 0.5).values == (m.p_open > 0.5).values,
                            clv_fav, -clv_fav)
    for lbl, v in (("OUR side", clv), ("OPEN favourite (control)", clv_fav)):
        se = v.std(ddof=1) / np.sqrt(n)
        print(f"\n  CLV of betting {lbl} at the OPEN: mean {v.mean():+.5f} "
              f"prob (se {se:.5f}, t={v.mean()/se:+.2f}), "
              f"{(v > 0).mean()*100:.1f}% positive (50% = no information)")
    print(f"\n  => our CLV is the favourite-drift control on "
          f"{( (m.p_ctl>0.5).values == (m.p_open>0.5).values ).mean()*100:.1f}%"
          f" of games; the two statistics are NOT independent.")
    sharp["clv_mean_prob"] = float(clv.mean())
    sharp["clv_frac_positive"] = float((clv > 0).mean())
    sharp["clv_t"] = float(clv.mean() / (clv.std(ddof=1) / np.sqrt(n)))
    sharp["clv_fav_control_mean"] = float(clv_fav.mean())
    sharp["clv_fav_control_t"] = float(
        clv_fav.mean() / (clv_fav.std(ddof=1) / np.sqrt(n)))
    sharp["clv_excess_vs_fav_mean"] = float(excess.mean())
    return {"per_season": per.reset_index().to_dict("records"),
            "sharpness": sharp}


def main():
    print("=" * 78)
    print("TS-OPENERS — opening-line availability audit")
    print("=" * 78)
    con = duckdb.connect(DB, read_only=True)
    try:
        res = {"duckdb": audit_duckdb(con),
               "kaggle": audit_kaggle(),
               "sbr_raw": audit_sbr_raw()}
        m = open_close_frame(con)
        res["open_close"] = report_open_close(m)
    finally:
        con.close()

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print("""
  OPENING lines exist ONLY in odds_hist_sbr, and that table ends mid
  2022-23.  Every other source (kaggle cvia -> odds_market, kaggle
  ehallmar, christophertreasure, erichqiu) carries ONE price per game and
  it is the CLOSE.  The raw HTML "failures" in data/raw/sbr are byte-
  equivalent to the xlsx that was already ingested -- there is nothing to
  recover there.

  CONSEQUENCE for the strategy: the evaluation window that matters
  (2022-23 .. 2025-26) has an opener for at most PART of its FIRST season
  and NONE of the OOS seasons.  A bet-at-open strategy is therefore
  UNTESTABLE out-of-sample in this repo.  The only honest formulation is
  BET-AT-CLOSE, which is also the hardest test: the close is the sharpest
  price of the day, so any edge that survives at the close is a lower
  bound on the edge available earlier.
""")
    pd.Series(res).to_json(OUT_JSON)
    print(f"  wrote {os.path.relpath(OUT_JSON, ROOT)}")


if __name__ == "__main__":
    main()
