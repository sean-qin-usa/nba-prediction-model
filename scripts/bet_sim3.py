#!/usr/bin/env python3
"""BET-SIM 3 — Sean's tail-betting thesis, refined by D77/D76 evidence.
Machinery copied from scripts/bet_sim2.py (same pricing, sizing, IS/OOS
selection discipline); NEW pre-registered rule family.

THESIS (D77 market-blowup autopsy, entry ab1add): once our mechanical
flatness is removed (shrink baseline logit(p_flat)=0.077+0.814*logit(p_mkt),
OLS of our logits on market logits over all games), our REAL deficit vs the
market is LARGER on toss-ups (-0.0116/gm) than on confident games
(-0.0065/gm) — the market's edge lives in the MIDDLE of the distribution.
So bet only in OUR confidence tails, on modest shared-side divergence, with
the divergence CAPPED per D13/H10 (extreme divergence = the market knows
something structural; D76-H1 killed the late-news story — the market's
divergence edge concentrates in QUIET games, so unresolved-Questionable
games are, if anything, our better divergences).

DATA: data/capstone_pergame_tank.csv (p_us with production tank term,
p_mkt = de-vig close from spread) + data/nba.duckdb read_only
(team_schedule for games-played; injury_reports_pit for the 5PM game-day
Questionable counts).

PRICING ASSUMPTIONS (identical to bet_sim.py / bet_sim2.py; no line
shopping, always get the close):
  * p_mkt is the de-vig CLOSE. Moneyline prices reconstructed with a 4.5%
    total overround allocated proportionally: q_side = p_mkt_side * 1.045,
    decimal = 1/q_side, floored at 1.01. Pick'em => 1.914 decimal (~-110).
  * ROI also reported at the FAIR (no-vig) close (dec_fair = 1/p_mkt_side)
    to decompose "no edge vs close" from "edge eaten by vig".

PRE-REGISTERED RULE FAMILY (ALL 16 evaluated in-sample; NOTHING added
after; combinatorial closure of 4 axes fixed before any data was scored):
  T-tier   : |p_us - 0.5| > c, c in {0.20, 0.28}   (OUR confidence tails,
             per D77: our real deficit is smallest there)
  Div band : shared side AND (p_us_side - p_mkt_side) in [t_lo, t_hi],
             (t_lo, t_hi) in {(0.02, 0.08), (0.03, 0.10)} — the CAP t_hi
             implements D13 (extreme divergence = adverse selection,
             excluded). Opposite-side games NEVER bet (known net-negative).
  +N overlay (optional): only games where the two teams combined have >= 1
             unresolved Questionable on the 5PM GAME-DAY report
             (injury_reports_pit, report_date = game_date; H1-reversal:
             high-news divergences are our good ones, d=-0.009 / 51.5% win
             vs quiet d=+0.088 / 35.7%).
             COVERAGE CAVEAT: injury_reports_pit runs 2023-10-24 ..
             2025-12-21, so for 2025-26 the +N rules are scored on the
             COVERED window only (games through 2025-12-21), labeled.
             Games with no game-day report are never bet under +N.
  +W overlay (optional): late-season window, either team gp >= 55 (the R4
             region — D73 tank-term activation; D75: the only
             flat-OOS-vig-positive diagnostic band in program history).
  4 base rules x {none, +N, +W, +NW} = 16 rules.

SIZING (both reported for every rule):
  * FLAT: 1u per bet.
  * QUARTER-KELLY on p_us_side vs the VIGGED offered odds:
    f* = (p_us_side*d - 1)/(d - 1); stake = 0.25*f* of a FIXED 100u
    reference bankroll (non-compounding), capped at 10u; f* <= 0 skipped.

IS/OOS DISCIPLINE (mechanical, pre-registered):
  * IN-SAMPLE = 2023-24 + 2024-25; all 16 rules evaluated IS.
  * SELECTION: top 2 by IS flat Sharpe (mean/sd per-bet flat PnL) among
    rules with IS n_bets >= 40.
  * OOS = 2025-26 scored for the selected rules; ALL 16 also reported OOS
    as labeled diagnostics (NO selection protection — informational only).
  * HONESTY CAVEAT: the T-tier axis is motivated by the D77 decile
    evidence, which was computed on ALL 3 seasons (the flat-shrink fit and
    the toss-up-vs-confident split saw 2025-26). OOS here is clean w.r.t.
    threshold/overlay SELECTION but NOT w.r.t. the D77 discovery that the
    middle is worse than the tails. Same class of caveat as bet_sim2's
    taxonomy anchor; treat OOS as validation of selection, not discovery.

DIAGNOSTIC (D77 method, requested): mechanical-flatness-corrected per-game
edge by |p_us - 0.5| decile — real = L_flat - L_us where
p_flat = sigmoid(a + b*logit(p_mkt)), (b, a) = polyfit over all games
(expected a=+0.077, b=0.814). Positive real = we genuinely add signal vs a
pure shrunk-market baseline. This is the "where are we ACTUALLY closest"
curve that justifies (or kills) tail-tiering.

RULES HONORED: DuckDB read_only=True; new file scripts/bet_sim3.py only
(machinery copied from bet_sim2.py; helpers IMPORTED from
scripts/ba_intersection.py, nothing existing edited). Bootstrap CIs use a
fixed seed; betting sim itself is deterministic.

Run:  python scripts/bet_sim3.py
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

from ba_intersection import team_schedule                        # noqa: E402

DB = os.path.join(ROOT, "data", "nba.duckdb")
CSV = os.path.join(ROOT, "data", "capstone_pergame_tank.csv")

IS_SEASONS = {"2023-24", "2024-25"}
OOS_SEASON = "2025-26"
OVERROUND = 1.045
MIN_DEC = 1.01
KELLY_FRAC = 0.25
BANKROLL = 100.0
KELLY_CAP = 10.0
MIN_BETS_IS = 40
N_SELECT = 2
TANK_GP = 55                     # +W window (either team gp >= 55, D73/D75)
CONF_TIERS = (0.20, 0.28)        # T-tier c
DIV_BANDS = ((0.02, 0.08), (0.03, 0.10))   # [t_lo, t_hi] on the shared side
N_BOOT = 4000
SEED = 77

ABB2FULL = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers", "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets",
    "IND": "Indiana Pacers", "LAC": "LA Clippers",
    "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves", "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs", "TOR": "Toronto Raptors", "UTA": "Utah Jazz",
    "WAS": "Washington Wizards"}


def logloss(p, y):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def build_frame() -> pd.DataFrame:
    df = pd.read_csv(CSV, dtype={"game_id": str})
    df["game_id"] = df.game_id.str.zfill(10)
    df["game_date"] = pd.to_datetime(df.game_date)
    con = duckdb.connect(DB, read_only=True)
    try:
        ts = team_schedule(con)
        # 5PM game-day report: per (game_date, team) count of Questionable
        rep = con.execute("""
            SELECT game_date, team,
                   COUNT(*) FILTER (WHERE status = 'Questionable') AS n_quest
            FROM injury_reports_pit
            WHERE report_date = game_date
            GROUP BY game_date, team
        """).df()
        rep_days = con.execute("""
            SELECT DISTINCT report_date FROM injury_reports_pit
            WHERE report_date = game_date
        """).df()
    finally:
        con.close()

    keys = ["season", "game_id"]
    for side in ("home", "away"):
        m = ts.rename(columns={"team": side})
        df = df.merge(m[keys + [side, "gp_before"]], on=keys + [side],
                      how="left")
        df = df.rename(columns={"gp_before": f"{side[0]}_gp"})
    assert df.h_gp.notna().all() and df.a_gp.notna().all(), "schedule join gap"

    # game-day 5PM Questionable counts (0 when covered but team not listed)
    rep["game_date"] = pd.to_datetime(rep.game_date)
    qmap = {(d, t): int(n) for d, t, n in
            zip(rep.game_date, rep.team, rep.n_quest)}
    covered_days = set(pd.to_datetime(rep_days.report_date))
    df["covered"] = df.game_date.isin(covered_days)
    df["quest"] = [
        qmap.get((d, ABB2FULL[h]), 0) + qmap.get((d, ABB2FULL[a]), 0)
        for d, h, a in zip(df.game_date, df.home, df.away)]

    # side pick + edge on the picked side
    df["pick_home"] = df.p_us > 0.5
    df["same_side"] = (df.p_us - 0.5) * (df.p_mkt - 0.5) > 0
    df["p_us_side"] = np.where(df.pick_home, df.p_us, 1 - df.p_us)
    df["p_mkt_side"] = np.where(df.pick_home, df.p_mkt, 1 - df.p_mkt)
    df["edge"] = df.p_us_side - df.p_mkt_side
    df["hit"] = np.where(df.pick_home, df.y == 1, df.y == 0)
    df["conf_us"] = (df.p_us - 0.5).abs()

    # offered odds (vigged) + fair close odds
    q = df.p_mkt_side * OVERROUND
    df["dec"] = np.maximum(1.0 / q, MIN_DEC)
    df["floored"] = (1.0 / q) < MIN_DEC
    df["dec_fair"] = 1.0 / df.p_mkt_side

    # overlay flags
    df["news"] = df.covered & (df.quest >= 1)          # +N (unbettable if uncovered)
    df["late"] = (df.h_gp >= TANK_GP) | (df.a_gp >= TANK_GP)   # +W
    return df


def rule_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """16 pre-registered rules: 2 conf tiers x 2 div bands x 4 overlay combos."""
    rules = {}
    for c in CONF_TIERS:
        for lo, hi in DIV_BANDS:
            base = (df.same_side & (df.conf_us > c)
                    & (df.edge >= lo) & (df.edge <= hi))
            stem = f"T{int(c*100)}.D{int(lo*100):02d}-{int(hi*100):02d}"
            rules[stem] = base
            rules[stem + "+N"] = base & df.news
            rules[stem + "+W"] = base & df.late
            rules[stem + "+NW"] = base & df.news & df.late
    return rules


def is_news(name: str) -> bool:
    return name.endswith("+N") or name.endswith("+NW")


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
                    roi_fair=np.nan, curve=np.array([]), floored=0,
                    n_wins=0)
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
        curve=cum, floored=int(sub.floored.sum()),
        n_wins=int(sub.hit.sum()))


def fmt_row(name, sz, r, note=""):
    return (f"{name:<15}{sz:<8}{r['n']:>6}{r['hit']*100 if r['n'] else 0:>7.1f}"
            f"{r['roi']*100:>8.2f}{r['roi_fair']*100:>10.2f}"
            f"{r['pnl']:>9.2f}{r['staked']:>9.1f}"
            f"{r['sharpe']:>8.3f}{r['maxdd']:>8.2f}  {note}")


HDR = (f"{'rule':<15}{'sizing':<8}{'n':>6}{'hit%':>7}{'ROI%':>8}"
       f"{'ROI%fair':>10}{'PnL(u)':>9}{'staked':>9}{'sharpe':>8}{'maxDD':>8}")


def print_curve(label, cum, per_line=15):
    vals = [f"{v:+.1f}" for v in cum]
    print(f"  {label} cumulative PnL (u) after each bet [{len(vals)} bets]:")
    for i in range(0, len(vals), per_line):
        print("    " + " ".join(vals[i:i + per_line]))


def gap_to_profit(r) -> str:
    """Avg price improvement (decimal points on winning bets) that would zero
    out a losing rule — the line-shopping/CLV gap."""
    if r["n"] == 0 or not np.isfinite(r["roi"]):
        return "n/a"
    if r["roi"] > 0:
        return "already profitable"
    if r["n_wins"] == 0:
        return "no wins — no price fixes this"
    dd = -r["pnl"] / r["n_wins"]
    return (f"needs +{dd:.3f} avg decimal on wins "
            f"(~{dd*100:.1f} cents/$1) i.e. ROI gap {-r['roi']*100:.2f}%")


def decile_table(df: pd.DataFrame):
    """D77-method diagnostic: mechanical-flatness-corrected per-game edge by
    |p_us-0.5| decile (all 3 seasons). real = L_flat - L_us."""
    rng = np.random.default_rng(SEED)
    b, a = np.polyfit(logit(df.p_mkt), logit(df.p_us), 1)   # slope, intercept
    p_flat = sigmoid(a + b * logit(df.p_mkt))
    L_us = logloss(df.p_us, df.y)
    L_mkt = logloss(df.p_mkt, df.y)
    L_flat = logloss(p_flat, df.y)
    raw = L_mkt - L_us          # >0 = we beat market (mechanical incl.)
    mech = L_mkt - L_flat       # flatness alone
    real = L_flat - L_us        # >0 = genuine signal beyond shrunk market
    dec = pd.qcut(df.conf_us, 10, labels=False, duplicates="drop")

    print(f"[flat baseline] logit(p_flat) = {a:+.4f} + {b:.4f}*logit(p_mkt)"
          f"  (D77 published: +0.077 + 0.814*logit; slope<1 = mechanically"
          " flatter)")
    print(f"{'dec':>4} {'|p_us-.5| range':>16} {'n':>5} {'mean_q_us':>10}"
          f" {'mean_q_mkt':>11} {'raw/gm':>9} {'mech/gm':>9} {'REAL/gm':>9}"
          f"  {'real 95% CI':>20}")
    tail_real, mid_real = [], []
    for d in sorted(pd.unique(dec)):
        m = (dec == d).values
        r = real[m]
        idx = rng.integers(0, len(r), (N_BOOT, len(r)))
        lo_ci, hi_ci = np.percentile(r[idx].mean(axis=1), [2.5, 97.5])
        cu = df.conf_us[m]
        qus = np.maximum(df.p_us[m], 1 - df.p_us[m]).mean()
        qmk = np.maximum(df.p_mkt[m], 1 - df.p_mkt[m]).mean()
        print(f"{d:>4} {cu.min():>7.3f}-{cu.max():<8.3f} {m.sum():>5}"
              f" {qus:>10.3f} {qmk:>11.3f} {raw[m].mean():>+9.4f}"
              f" {mech[m].mean():>+9.4f} {r.mean():>+9.4f}"
              f"  [{lo_ci:+.4f},{hi_ci:+.4f}]")
        (tail_real if d >= 8 else mid_real).append(r)
    tr = np.concatenate(tail_real)
    mr = np.concatenate(mid_real)
    print(f"  top-2 deciles (tails) REAL {tr.mean():+.4f}/gm (n={len(tr)})"
          f"  vs deciles 0-7 REAL {mr.mean():+.4f}/gm (n={len(mr)})")
    for ssn, g in df.assign(real=real, dec=dec.values).groupby("season"):
        t = g[g.dec >= 8]
        print(f"    {ssn}: tails REAL {t.real.mean():+.4f}/gm (n={len(t)}),"
              f" rest {g[g.dec < 8].real.mean():+.4f}/gm")


def main():
    df = build_frame().sort_values(["game_date", "game_id"]).reset_index(drop=True)
    rules = rule_masks(df)
    is_m = df.season.isin(IS_SEASONS)
    oos_m = df.season == OOS_SEASON
    oos_cov = oos_m & df.covered

    print("=" * 100)
    print("BET-SIM 3 (tail-betting thesis, D77/D76-refined) — "
          "IS=2023-24+2024-25, OOS=2025-26")
    print("=" * 100)
    print(f"vig model: overround {OVERROUND} proportional -> pick'em decimal "
          f"{1/(0.5*OVERROUND):.3f} (~-110); odds floored at {MIN_DEC} on "
          f"{int(df.floored.sum())} of {len(df)} games")
    print(f"avg vig cost on the picked side: "
          f"{(1/df.p_mkt_side - df.dec).mul(df.p_mkt_side).mean()*100:.2f}% "
          f"of stake per bet")
    print(f"news coverage (5PM game-day report exists): "
          f"IS {int((is_m & df.covered).sum())}/{int(is_m.sum())} games; "
          f"OOS {int(oos_cov.sum())}/{int(oos_m.sum())} games "
          f"(injury_reports_pit ends 2025-12-21) — +N/+NW rules are scored "
          f"on covered games only, labeled")
    print(f"+N pool (>=1 unresolved Questionable, both teams combined): "
          f"{int(df.news.sum())}/{int(df.covered.sum())} covered games; "
          f"+W pool (either gp>={TANK_GP}): {int(df.late.sum())} games")

    # ---- DIAGNOSTIC: D77 decile curve --------------------------------------
    print("\n" + "-" * 100)
    print("DIAGNOSTIC — real (mechanical-flatness-corrected) per-game edge "
          "by |p_us-0.5| decile, all 3 seasons")
    print("(raw = L_mkt-L_us incl. flatness; mech = L_mkt-L_flat; "
          "REAL = L_flat-L_us, >0 = genuine signal)")
    decile_table(df)

    # ---- IN-SAMPLE: all 16 rules x 2 sizings --------------------------------
    print("\n" + "-" * 100)
    print(f"IN-SAMPLE (n games={int(is_m.sum())}) — all 16 pre-registered "
          "rules")
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
    print("\n" + "-" * 100)
    print(f"SELECTION (top {N_SELECT} by IS flat Sharpe, n>={MIN_BETS_IS}): "
          f"{', '.join(selected) if selected else 'NONE eligible'}")
    print(f"  eligible (n>={MIN_BETS_IS}): "
          + (", ".join(f"{n} ({s:+.3f})" for n, s in elig) or "none"))
    for n in selected:
        r = is_res[(n, "flat")]
        print(f"  {n}: IS sharpe={r['sharpe']:+.3f} ROI={r['roi']*100:+.2f}% "
              f"n={r['n']} hit={r['hit']*100:.1f}%")
        print_curve(f"{n} IS flat", r["curve"])

    # ---- OOS: selected rules -----------------------------------------------
    print("\n" + "-" * 100)
    print(f"OUT-OF-SAMPLE {OOS_SEASON} (n games={int(oos_m.sum())}) — "
          "selected rules only")
    print(HDR)
    verdict_profitable = []
    oos_gap = {}
    for name in selected:
        m = rules[name] & (oos_cov if is_news(name) else oos_m)
        sub = df[m]
        note = "[covered window <=2025-12-21]" if is_news(name) else ""
        for sz in ("flat", "qkelly"):
            r = score(sub, sz)
            print(fmt_row(name, sz, r, note))
            if sz == "flat":
                oos_gap[name] = r
            if r["n"] > 0 and r["roi"] > 0:
                verdict_profitable.append((name, sz, r))
        print_curve(f"{name} OOS flat", score(sub, "flat")["curve"])

    # ---- ALL-16 OOS DIAGNOSTICS (pre-committed, labeled) --------------------
    print("\n" + "-" * 100)
    print("ALL-16 OOS DIAGNOSTICS (pre-committed reporting; non-selected "
          "rows carry NO selection protection — informational only; +N/+NW "
          "on covered window <=2025-12-21 only)")
    print(HDR)
    for name in rules:
        if name in selected:
            print(f"{name:<15}(selected — see OOS table above)")
            continue
        m = rules[name] & (oos_cov if is_news(name) else oos_m)
        sub = df[m]
        note = "[covered window]" if is_news(name) else ""
        for sz in ("flat", "qkelly"):
            r = score(sub, sz)
            print(fmt_row(name, sz, r, note))
            if sz == "flat":
                oos_gap[name] = r

    # ---- VERDICT ------------------------------------------------------------
    print("\n" + "=" * 100)
    print("VERDICT")
    sel_prof = [v for v in verdict_profitable if v[0] in selected]
    if sel_prof:
        for name, sz, r in sel_prof:
            print(f"  {name} [{sz}] is OOS-profitable at the -110-equivalent "
                  f"vig: ROI {r['roi']*100:+.2f}% on n={r['n']} bets "
                  f"(PnL {r['pnl']:+.2f}u). Small n — treat as fragile until "
                  "live CLV confirms (D66 protocol).")
    else:
        print("  NO selected rule is OOS-profitable at -110-equivalent "
              "pricing.")
    print("  GAP TO PROFIT (flat, OOS; = required line-shopping/CLV price "
          "improvement):")
    for name, r in oos_gap.items():
        tag = "selected" if name in selected else "diagnostic"
        print(f"    {name:<15}[{tag}] n={r['n']:>3} ROI={r['roi']*100:+7.2f}% "
              f"ROI@fair={r['roi_fair']*100:+7.2f}% -> {gap_to_profit(r)}")
    print("  Honesty: p_mkt is the de-vig CLOSE; ROI%fair shows the edge vs "
          "the fair close, the ROI%fair-ROI% gap is pure vig. No line "
          "shopping, close always assumed available.")
    print("  Selection was mechanical on IS flat Sharpe; OOS untouched by "
          "selection. Family-level caveat: the T-tier axis rests on the D77 "
          "decile evidence computed on all 3 seasons (see docstring) — OOS "
          "validates the selection, not the tail-vs-middle discovery.")
    print("  +N coverage caveat: injury_reports_pit ends 2025-12-21; +N/+NW "
          "OOS rows cover only the early-2025-26 window and by construction "
          "exclude late-season games (so +NW is near-empty OOS).")


if __name__ == "__main__":
    main()
