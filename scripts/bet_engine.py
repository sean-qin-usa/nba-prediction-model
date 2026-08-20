#!/usr/bin/env python3
"""PAPER-TRADE BET ENGINE — the 2026-27 pre-registered F4 registry
(D75 / D78 / D82 rules + the D112 ship), FOUR sizing arms, THREE timestamped
snapshot views, full book-panel telemetry, CLV logging.  Offseason-safe,
cron-ready.  October 2026 posture (D121 + external review product pass): a CLV-FARMING
MEASUREMENT PRODUCT, no real capital at any view.

THE THREE VIEWS (restructure per the external review product pass, D121 context).
The single 22:40 UTC emission is replaced by three timestamped emissions of
the SAME rule-selected bets, each booked at that view's price:

  OPEN         --scan-open, cron every 30 min 14:00-02:00 UTC.  Books a game
               the first time a two-sided h2h line appears for it.  "First" is
               bounded by the capture cadence (odds logger JSONL + 30-min ESPN
               poll), so quote_ts stores the ACTUAL timestamp of the freshest
               quote used, not the scan time.  A game already carrying OPEN
               rows in bet_quotes_panel is never re-booked.
  POST_REPORT  --emit-post-report, cron 22:10 UTC (~5:10PM ET), shortly after
               the 5PM ET injury report lands; the latest injury_reports
               capture ts for today is recorded in `detail` (report_ts=...).
  PRETIP       --emit-pretip, cron 23:55 UTC: last quote before tipoff.

Why: D120/D121 measured a REAL pure-timing edge (+1.4..+3.7pp dROI, SIG in
16/16 paired cells) that lands at breakeven, and a placebo-controlled,
outcome-predictive CLV at the open (+0.0112, t=6.7).  Booking the same rules
at three timestamps turns that into a LIVE per-view statistic.  Decision rule
(external review): if OPEN does not beat POST_REPORT and PRETIP on CLV by 2026-11-30,
stop talking about opener edge and keep only the latest-price paper book.

BOOK-PANEL TELEMETRY.  At every emission, ALL available book quotes for each
two-sided game on the slate are logged to bet_quotes_panel (game_id,
snapshot_kind, ts, book, side, price), and every bet row carries the book
identity of its best price, the consensus (median) decimal, and hence the
best-vs-consensus gap.  Purpose: by 2026-12-31, attribute positive CLV to
timing vs one soft book family (>=75% of positive CLV from one family =>
line shopping is the product; flat across books => model timing is).

MIGRATION (bet_paper).  New columns, added in place by _ensure_schema via
ALTER TABLE ADD COLUMN so existing rows stay valid:
    snapshot_kind    VARCHAR  DEFAULT 'POST_REPORT'  (legacy 22:40 rows were
                     emitted after the 5PM ET report — POST_REPORT is their
                     honest label)
    quote_ts         TIMESTAMPTZ   -- ts of the freshest quote actually used
    best_price       DOUBLE        -- best decimal, our side (== price_decimal)
    consensus_price  DOUBLE        -- MEDIAN decimal, our side, across books
    open_shrunk_edge DOUBLE        -- max(0, a_o + b_o*edge), open-arm calib
    stake_open_shrunk DOUBLE       -- arm 4 stake
    pnl_open_shrunk  DOUBLE        -- arm 4 pnl
(`book` was already a column.)  The PRIMARY KEY must gain snapshot_kind
(same game/rule now books up to 3x); DuckDB cannot ALTER a PK, so when an
existing table's PK lacks snapshot_kind the migration rebuilds it in place
(CREATE new-shape table -> INSERT SELECT -> DROP -> RENAME) inside the same
short write window.  bet_quotes_panel is created alongside.

D112 SHIP (unchanged, pre-registered 2026-08-01):
  (b) UPPER CONFIDENCE-EXCESS CAP — skip when conf_us - conf_mkt > 0.08.
  (a) EDGE SHRINKAGE IN SIZING — Kelly from max(0, a + b*edge) added to
      p_mkt_side (scripts/f4_shrinkage.py, refit annually Oct 1).
  (c) PARALLEL SIZING ARMS, none selected before season end.

FOUR SIZING ARMS (was three):
  flat          1.0u                                  — the honest control
  raw_kelly     quarter-Kelly on p_us_side            — what D75 ran
  shrunk_kelly  quarter-Kelly on p_mkt_side + max(0, a+b*edge), CLOSE-fit
                (a=-0.01396, b=+0.18418; f4_shrinkage.py) — the D112 ship
  open_shrunk   DIAGNOSTIC ONLY — same construction with the FROZEN OPEN-arm
                calibration below.  It sizes PAPER stakes at every view,
                never real ones.  Rationale: f4_shrinkage's coefficients were
                fit at CLOSE prices; D120/D121 measured the Kelly slope at
                the OPEN at b=0.569 vs 0.209 — "no positive-EV bets" is a
                statement about the close specifically.

FROZEN open_shrunk CONSTANTS (see OPEN_SHRUNK below).  D121 registers only
b=0.569, so (a, b) were re-derived exactly the way f4_shrinkage.py fits the
close arm — OLS of realised_excess (hit - p_open_side) on claimed_excess
(p_us_side - p_open_side), same-side games only — on data/ds_rt1_pergame.csv
(p_full, 4 seasons) joined 1:1 to data/derived/odds_open.csv, with
p_open = sigmoid(open_margin / 6.96) (the program's spread->prob map, the
D120 PRIMARY|SP|OPEN frame):
    a = -0.0037733442091709493   b = +0.5684830302091815
    se_b = 0.10294290368592647   n = 3848   t = +5.52
    break-even claimed edge -a/b = +0.0066 (vs +0.0758 at the close)
The recompute reproduces data/bo_openbacktest.json kelly_slope
["PRIMARY rt1 p_full 4-season|SP"]["OPEN"] to the last decimal.  FROZEN:
never refit mid-season; an annual refit would be a new D-line.

D178 CHANGES (October live-path close-out):
  FIX 1  GAME-TYPE FILTER.  The live path books REGULAR SEASON ONLY.  See
         nbapred/engine/slate.py — `todays_games()` and `slate_context()` both
         filter `game_id` prefix 002, and emit()/scan_open() filter their own
         copy of the list.  Before this, an early-October preseason game or a
         February All-Star game on the nba_api scoreboard would have been
         priced and BOOKED.
  FIX 2  >= 2 DISTINCT BOOKS AT THE OPEN (MIN_BOOKS_OPEN).  See that constant.
  FIX 3  CLV BANDS RE-DERIVED (CLV_BAND).  See that constant.

REAL-STAKES TRIGGER (--monthly-report; print-only, the engine NEVER acts):
real stakes are CONSIDERED only after 2 consecutive completed calendar
months with OPEN-view mean CLV > CLV_MONTH_GOOD, no completed OPEN-view month
< CLV_MONTH_RED (the D178 2-sigma bands: -0.000236 / +0.024797, union-centred
on the REAL-MONEYLINE frame at 67 bets/month — see CLV_BAND), AND
stake_open_shrunk > 0 on >= 10% of priced OPEN-view candidates in that window;
earliest 2027-01-01; if triggered, start FLAT 0.25u.  This codifies the external review
product-pass gate on D121's "open might justify capital".  NOTE: a symmetric
2-sigma GOOD line makes 2 consecutive GOOD months a ~1,932-month wait BY
CONSTRUCTION; --monthly-report prints that arithmetic every run.  Re-specifying
the TRIGGER (as opposed to the BAND) is a separate product decision.

RULES (frozen BEFORE opening night; operators verbatim from the sims):
  R4_LOWT / T20_D03_10_W / T20_D03_10 / STAR_FAV_SHARPER — see rules_fired.
  Registry vetoes: opposite side NEVER bet; conf-excess cap 0.08 (D112).

PRICES: per view, freshest h2h quotes at that view's time — today's/
yesterday's raw odds JSONL (data/raw/odds/*.jsonl) with the odds_quotes
table as fallback.  Per book: latest snapshot, de-vig implied
p = (1/d_side) / (1/d_home + 1/d_away); consensus p_mkt = MEDIAN implied
across books; candidate price = BEST decimal on our side.

CLV: at --settle, close = last pre-tip snapshot (median de-vig implied +
best decimal); clv = close_implied - implied_p (>0 = we beat the close);
outcome/pnl per arm from nba_games.  Rows stay unsettled until data lands.

WRITES: bet_paper + bet_quotes_panel only.  Every read connection is
read_only=True; writes are SHORT and batched (executemany inside one
window); on writer-lock failure the engine waits 60s and retries
(_write_retry).  INSERT OR IGNORE keeps the FIRST emission of a candidate
per (game, side, rule, snapshot_kind) — reruns never overwrite.

Run:  python scripts/bet_engine.py --scan-open          (cron */30, 14-02 UTC)
      python scripts/bet_engine.py --emit-post-report   (cron 22:10 UTC)
      python scripts/bet_engine.py --emit-pretip        (cron 23:55 UTC)
      python scripts/bet_engine.py --settle             (cron 10:40 UTC)
      python scripts/bet_engine.py --report             (scorecard: view x arm)
      python scripts/bet_engine.py --monthly-report     (CLV bands + trigger)
      python scripts/bet_engine.py --dry-run 2026-02-11 (historical rehearsal
                                    in a TEMP DB — never touches bet_paper)
      python scripts/bet_engine.py --emit               (legacy alias for
                                    --emit-post-report)
"""
from __future__ import annotations

import argparse
import math
import datetime as dt
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nbapred.config import RAW_ODDS                              # noqa: E402

from f4_shrinkage import (CONF_EXCESS_CAP, SIZING_ARMS,          # noqa: E402
                          load_coeffs, stake_units as arm_stake)

# Set to a path to redirect EVERY connection (dry runs / tests).
DB_OVERRIDE: str | None = None


def _connect(read_only: bool = False):
    """Indirection for tests (monkeypatch) and dry runs (DB_OVERRIDE)."""
    if DB_OVERRIDE is not None:
        import duckdb
        return duckdb.connect(str(DB_OVERRIDE), read_only=read_only)
    from nbapred.db import connect
    return connect(read_only=read_only)


def _write_retry(attempts: int = 10, wait_s: float = 60.0):
    """Writer connection; on writer-lock contention wait 60s and retry."""
    for i in range(attempts):
        try:
            return _connect()
        except Exception as e:  # noqa: BLE001 — duckdb.IOException and kin
            msg = str(e).lower()
            if ("lock" not in msg and "held" not in msg) or i == attempts - 1:
                raise
            print(f"writer lock held, retry {i + 1}/{attempts} in {wait_s:.0f}s")
            time.sleep(wait_s)


EDGE_R4 = 0.02           # R4 low threshold (D75)
CONF_TIER = 0.20         # |p_us - 0.5| tail tier (D78)
DIV_LO, DIV_HI = 0.03, 0.10   # capped divergence band (D78 / D13)
TANK_GP = 55             # late window: either team gp >= 55 (D73/D75)
STAR_MIN = 28.0          # star trailing-minutes threshold (ba_intersection)
STAR_TRAIL = 10
STAR_MIN_GP = 3
STAKE = 1.0              # flat paper stake per (game, rule, view)
# CONF_EXCESS_CAP (0.08) is imported from f4_shrinkage so the engine, the
# re-sim and the annual refit can never drift apart.

SNAPSHOT_KINDS = ("OPEN", "POST_REPORT", "PRETIP")

# ---- D178 FIX 2: >=2 DISTINCT BOOKS AT THE OPEN -----------------------------
# D142 measured the whole shop asset at the open: best-of-2 lifts CLV ~49%, and
# taking the WORSE of two books erases essentially all of it (+0.0092 ->
# -0.0007).  D125 shipped bet_quotes_panel telemetry but NOTHING required two
# books, so a thin-book night booked a single-book price at the open and that
# observation entered the CLV mean indistinguishably from a shopped one.
#
# THE RULE.  An OPEN row is BOOKED (non-zero stake in every arm, counted in the
# CLV scoring set) only when >= MIN_BOOKS_OPEN distinct two-sided books are
# present at that moment.  With one book the observation is STILL WRITTEN —
# stakes zeroed, single_book = TRUE, clv_eligible = FALSE — so a thin-book
# night degrades VOLUME, visibly, rather than silently polluting the
# measurement.  `--report` and `--monthly-report` score CLV on clv_eligible
# rows only.
#
# WHERE THE LIVE OPEN PANEL COMES FROM — VERIFIED, NOT ASSUMED (D174 context):
#   * Action Network  — CANNOT serve.  D174 (2)(c): AN carries NO per-book
#     opening price; its book_id 30 "Open" is a single CONSENSUS opener and the
#     per-book numbers are one scrape-time snapshot (D174 (3): that snapshot is
#     the CLOSE).
#   * ESPN            — CANNOT serve alone.  `nbapred/ingest/espn_lines.py` is
#     ONE BOOK by construction (ESPN BET on the public scoreboard).  MEASURED
#     on D174's own panel rows (data/bkp_panel_rows.csv.gz), share of games
#     with >=2 distinct operators AT THE OPEN:
#         2023-24  94.7%   (the syndicated era, 8 operators modal)
#         2024-25   0.0%   (ESPN BET only)
#         2025-26   3.4%
#     i.e. an ESPN-only open panel would refuse ~96.6% of OPEN bookings.
#   * THE ODDS API (`nbapred/ingest/odds_logger.py` -> data/raw/odds/*.jsonl,
#     `load_odds.flatten` -> one row per (book, market, outcome)) is the ONLY
#     live source whose response shape carries a real multi-book US panel
#     (`ev["bookmakers"][]`, regions=us).  The engine already prefers it
#     (_jsonl_quotes) and falls back to odds_quotes.
#   * OPERATIONAL CONSEQUENCE, NAMED SO OCTOBER CANNOT BE SURPRISED: the odds
#     logger has NEVER RUN IN-SEASON in this repo — data/raw/odds/ holds one
#     offseason day whose `data` is `[]`, and `odds_quotes` is EMPTY (0 rows).
#     If the logger is not up on opening night, every OPEN row will be
#     single-book and the CLV scoring set will be EMPTY — loudly, by design,
#     which is the point of this rule.  See docs/OCTOBER_RUNBOOK.md.
MIN_BOOKS_OPEN = 2

# ---- FROZEN open-arm calibration (diagnostic arm 4) -------------------------
# Derivation: header "FROZEN open_shrunk CONSTANTS".  Reproduces
# data/bo_openbacktest.json kelly_slope["PRIMARY rt1 p_full 4-season|SP"]
# ["OPEN"] exactly (the D120/D121 registered artifact).  b=0.569 is the value
# D121 registers.  DO NOT REFIT — frozen 2026-08-01, before any 2026-27 data.
OPEN_SHRUNK = {
    "a": -0.0037733442091709493,
    "b": 0.5684830302091815,
    "se_b": 0.10294290368592647,
    "n": 3848,
    "t": 5.522313922129048,
    "frame": "ds_rt1_pergame.csv p_full x odds_open (SP open, sigmoid(m/6.96))",
    "source": "D120/D121 bo_openbacktest kelly_slope PRIMARY|SP|OPEN",
    "fit_date": "2026-08-01",
    "frozen": True,
}

ALL_ARMS = tuple(SIZING_ARMS) + ("open_shrunk",)

# ---- real-stakes trigger (print-only; --monthly-report) ---------------------
# D178 FIX 3 — BANDS RE-DERIVED FROM SCRATCH (scripts/d178_clvbands.py ->
# data/d178_clvbands.json).  They REPLACE D120/D121's RED -0.0131 / GOOD
# +0.0200, which D159 showed were built wrong in two independent ways:
#   (1) LEVEL  — centred on the ALL-SAME-SIDE UNIVERSE (+0.0035) but widthed by
#                the per-bet sd of the RULE UNION, while --monthly-report scores
#                the UNION.  Centre and width came from different populations.
#   (2) FRAME  — built in the SP (spread-derived-probability) convention while
#                this engine prices and settles MONEYLINES: `settle()` computes
#                clv = close_implied - implied_p, and implied_p is the de-vigged
#                CONSENSUS MONEYLINE probability on our side.
# So GOOD was a +1.23-sigma event on the wrong frame — MIS-SPECIFIED, not
# mis-levelled.  D159 published -0.0006/+0.0246 but derived them PRE-BACKFILL;
# these are re-derived on the D171/D173 certified data, NOT pasted.
#
# THE SPACE MATTERS: the engine scores CLV in PROBABILITY space on the REAL
# MONEYLINE.  D173 also reports an ATS SPREAD-POINT CLV that doubled to +0.320
# POINTS — a different unit; using it here would be a ~26x error.  ML is used.
#
# DERIVATION INPUTS (ML | HONEST = capstone_pergame.csv, the availability
# construction October ships; union of the 4 F4 rules, unique games, @OPEN;
# 2023-24..2025-26 — the three full-T2 seasons, the only ones with a real
# moneyline open).  --monthly-report PRINTS these alongside the verdict so a
# band can never again drift from what produced it.
CLV_BAND = {
    "d": "D178",
    "frame": "ML (real moneyline), probability space, union @OPEN, unique games",
    "arm": "HONEST (data/capstone_pergame.csv, md5 695d40a3545e889267cad403b7acdce8)",
    "seasons": ("2023-24", "2024-25", "2025-26"),
    "centre": +0.012280275370533451,   # MEAN CLV OF THE UNION (fixes defect 1)
    "per_bet_sd": 0.05122628489879061,
    "n_bets": 1386,
    "n_months": 21,
    "bets_per_month": 67.0,      # MEDIAN union bets per calendar month
    "se_monthly": 0.006258286762523038,  # per_bet_sd / sqrt(bets_per_month)
    "k_sigma": 2.0,
    "source": "scripts/d178_clvbands.py -> data/d178_clvbands.json",
    "anchor": "LEAKY_REG ML@open UNION n=1378 CLV=+0.01590 == D155/D159 EXACT",
    "supersedes": "D120/D121 RED -0.0131 / GOOD +0.0200",
    # D176's headline caveat, carried onto the instrument it constrains.
    "status": "MONITOR, NOT OBJECTIVE",
}
# WHAT THIS BAND IS AND IS NOT — D176 MEASURED THE TWO APART, SO THIS IS NOT A
# HEDGE.  D176 ran three pre-registered selectors and found that the
# AVAILABILITY-DIVERGENCE selector bought MORE CLV (6/6 cells, +0.143 pts) and
# LESS ROI (1/6, -1.16pp), while the explicitly CLV-TARGETED selector bought
# essentially NO extra CLV (+0.004) and the MOST ROI.  **CLV IS NOT A SUFFICIENT
# STATISTIC FOR BET SELECTION.**  A band tuned purely on CLV can therefore
# green-light a selector that is losing money.  CLV is kept because it RESOLVES
# FAST — it is the monthly MONITOR, the early-warning instrument on execution
# and timing.  It is NOT the objective, and A GREEN CLV MONTH IS NOT EVIDENCE OF
# PROFITABILITY.  ROI, against the INCUMBENT, remains the objective.
# AND ON NULLS (D176's second lesson): nothing in this band is a net-of-null
# statistic.  Centre, sd and se are plain sample moments of the union CLV.  That
# is deliberate — D176 found all three new arms beat their own permutation nulls
# at p <= 0.048 and survived BH, AND ALL THREE STILL LOST TO THE INCUMBENT.
# "Beats a scrambled copy of itself" is necessary, not sufficient; it is true of
# the incumbent too.  Any future re-derivation that leans on a null must report
# against the INCUMBENT as well.
CLV_MONTH_RED = CLV_BAND["centre"] - CLV_BAND["k_sigma"] * CLV_BAND["se_monthly"]
CLV_MONTH_GOOD = CLV_BAND["centre"] + CLV_BAND["k_sigma"] * CLV_BAND["se_monthly"]
# -> RED -0.000236 / GOOD +0.024797.
# TWO CONSEQUENCES THAT MUST NOT BE DISCOVERED LATER (both printed by
# --monthly-report):
#   * RED IS NOW MUCH TIGHTER (-0.0002 vs -0.0131): with the band correctly
#     centred on +0.0123, "2 sigma below" is essentially "any negative month".
#     On the 21 in-frame months that flags 3 — 2024-10, 2025-04, 2025-10 — and
#     TWO OF THE THREE ARE OCTOBERS.  Expect an October RED; it means the month
#     was below the historical run rate, not that the engine is broken.
#   * THE TRIGGER IS UNREACHABLE BY CONSTRUCTION, AND ALWAYS WAS.  Asking for
#     TRIGGER_MONTHS consecutive months above a +k-sigma line has expected wait
#     1/P(+k sigma)^TRIGGER_MONTHS = ~1,932 months at k=2.  D120/D121's own
#     numbers gave ~85 months only because their GOOD line sat at +1.23 sigma of
#     THIS frame by accident.  Re-specifying the TRIGGER is a product decision
#     with its own D-line; this entry fixes the BAND and makes the arithmetic
#     visible instead of installing a number that hides it.
TRIGGER_MONTHS = 2           # consecutive good months required (OPEN view)
TRIGGER_OPEN_SHRUNK_FRAC = 0.10
TRIGGER_EARLIEST = dt.date(2027, 1, 1)
TRIGGER_START_STAKE = 0.25   # flat units, if ever triggered — printed only

BET_PAPER_DDL = """
CREATE TABLE IF NOT EXISTS {name} (
    candidate_ts  TIMESTAMPTZ NOT NULL,   -- when the candidate was emitted (PIT)
    snapshot_kind VARCHAR NOT NULL DEFAULT 'POST_REPORT',  -- OPEN/POST_REPORT/PRETIP
    quote_ts      TIMESTAMPTZ,            -- ts of the freshest quote used
    game_date     DATE NOT NULL,
    game_id       VARCHAR NOT NULL,
    home          VARCHAR, away VARCHAR,  -- team abbrevs
    side          VARCHAR NOT NULL,       -- 'home' / 'away'
    rule          VARCHAR NOT NULL,       -- R4_LOWT / T20_D03_10_W / T20_D03_10 / STAR_FAV_SHARPER
    p_us          DOUBLE,                 -- model P(home win), market-blind
    p_us_side     DOUBLE, p_mkt_side DOUBLE, edge DOUBLE,
    price_decimal DOUBLE, book VARCHAR,   -- best decimal on our side + book
    best_price    DOUBLE,                 -- == price_decimal (explicit)
    consensus_price DOUBLE,               -- MEDIAN decimal on our side
    implied_p     DOUBLE,                 -- consensus de-vig implied p (our side)
    event_id      VARCHAR, commence_ts TIMESTAMPTZ,
    stake_units   DOUBLE DEFAULT 1.0,     -- FLAT arm, 1u, paper
    close_price   DOUBLE, close_implied DOUBLE,
    clv           DOUBLE,                 -- close_implied - implied_p (>0 = beat close)
    outcome       INTEGER,                -- 1 our side won / 0 lost
    pnl_units     DOUBLE, settled_ts TIMESTAMPTZ,
    detail        VARCHAR,
    conf_excess   DOUBLE,                 -- conf_us - conf_mkt (== edge, same-side)
    cap_in_force  DOUBLE,                 -- the CONF_EXCESS_CAP this row obeyed
    shrunk_edge   DOUBLE,                 -- max(0, a + b*edge), close-fit (D112)
    shrink_a      DOUBLE,                 -- close-fit coefficients used, logged PIT
    shrink_b      DOUBLE,
    stake_raw_kelly    DOUBLE,            -- arm 2 (D75's sizing)
    stake_shrunk_kelly DOUBLE,            -- arm 3 (the D112 ship)
    pnl_raw_kelly      DOUBLE,
    pnl_shrunk_kelly   DOUBLE,
    open_shrunk_edge   DOUBLE,            -- max(0, a_o + b_o*edge), OPEN-arm frozen
    stake_open_shrunk  DOUBLE,            -- arm 4 (diagnostic, paper-only)
    pnl_open_shrunk    DOUBLE,
    n_books       INTEGER,                -- D178: distinct two-sided books used
    single_book   BOOLEAN,                -- D178: n_books < MIN_BOOKS_OPEN
    clv_eligible  BOOLEAN DEFAULT TRUE,   -- D178: in the CLV scoring set?
    PRIMARY KEY (game_date, game_id, side, rule, snapshot_kind)
);
"""

PANEL_DDL = """
CREATE TABLE IF NOT EXISTS bet_quotes_panel (
    game_date     DATE NOT NULL,
    game_id       VARCHAR NOT NULL,
    snapshot_kind VARCHAR NOT NULL,       -- OPEN / POST_REPORT / PRETIP
    ts            TIMESTAMPTZ,            -- the book's quote snapshot ts
    emitted_ts    TIMESTAMPTZ,            -- when the panel row was logged
    book          VARCHAR NOT NULL,
    side          VARCHAR NOT NULL,       -- 'home' / 'away'
    price         DOUBLE,                 -- decimal
    PRIMARY KEY (game_date, game_id, snapshot_kind, book, side)
);
"""

SCHEMA = BET_PAPER_DDL.format(name="bet_paper") + PANEL_DDL

# Columns ADDed in place when an older bet_paper predates them (order:
# D112 wave, then the 3-view/telemetry wave).  (name, type, default-SQL|None).
MIGRATION_COLUMNS = [
    ("conf_excess",        "DOUBLE", None),
    ("cap_in_force",       "DOUBLE", None),
    ("shrunk_edge",        "DOUBLE", None),
    ("shrink_a",           "DOUBLE", None),
    ("shrink_b",           "DOUBLE", None),
    ("stake_raw_kelly",    "DOUBLE", None),
    ("stake_shrunk_kelly", "DOUBLE", None),
    ("pnl_raw_kelly",      "DOUBLE", None),
    ("pnl_shrunk_kelly",   "DOUBLE", None),
    ("snapshot_kind",      "VARCHAR", "'POST_REPORT'"),
    ("quote_ts",           "TIMESTAMPTZ", None),
    ("best_price",         "DOUBLE", None),
    ("consensus_price",    "DOUBLE", None),
    ("open_shrunk_edge",   "DOUBLE", None),
    ("stake_open_shrunk",  "DOUBLE", None),
    ("pnl_open_shrunk",    "DOUBLE", None),
    # D178 FIX 2 wave.  clv_eligible defaults TRUE so every PRE-D178 row keeps
    # exactly the CLV status it already had — the rule is not applied
    # retroactively to a measurement it did not govern.
    ("n_books",            "INTEGER", None),
    ("single_book",        "BOOLEAN", "FALSE"),
    ("clv_eligible",       "BOOLEAN", "TRUE"),
]

INSERT_COLS = [
    "candidate_ts", "snapshot_kind", "quote_ts", "game_date", "game_id",
    "home", "away", "side", "rule", "p_us", "p_us_side", "p_mkt_side", "edge",
    "price_decimal", "book", "best_price", "consensus_price", "implied_p",
    "event_id", "commence_ts", "stake_units", "detail",
    "conf_excess", "cap_in_force", "shrunk_edge", "shrink_a", "shrink_b",
    "stake_raw_kelly", "stake_shrunk_kelly",
    "open_shrunk_edge", "stake_open_shrunk",
    "n_books", "single_book", "clv_eligible",
]

PANEL_COLS = ["game_date", "game_id", "snapshot_kind", "ts", "emitted_ts",
              "book", "side", "price"]


def _ensure_schema(con) -> None:
    """Create bet_paper/bet_quotes_panel if absent; migrate an existing
    bet_paper in place: ALTER TABLE ADD COLUMN (with defaults) for every
    missing column, then — because DuckDB cannot alter a PK — rebuild the
    table once if its PK predates snapshot_kind."""
    con.execute(SCHEMA)
    have = {r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'bet_paper'").fetchall()}
    for col, typ, default in MIGRATION_COLUMNS:
        if col not in have:
            ddl = f"ALTER TABLE bet_paper ADD COLUMN {col} {typ}"
            if default is not None:
                ddl += f" DEFAULT {default}"
            con.execute(ddl)
            if default is not None:   # legacy rows must be valid, not NULL
                con.execute(f"UPDATE bet_paper SET {col} = {default} "
                            f"WHERE {col} IS NULL")
    # PK rebuild: same (game, side, rule) may now book once per snapshot view.
    pk = con.execute("""
        SELECT constraint_column_names FROM duckdb_constraints()
        WHERE table_name = 'bet_paper' AND constraint_type = 'PRIMARY KEY'
        """).fetchall()
    pk_cols = set(pk[0][0]) if pk else set()
    if pk and "snapshot_kind" not in pk_cols:
        con.execute("DROP TABLE IF EXISTS bet_paper__mig")
        con.execute(BET_PAPER_DDL.format(name="bet_paper__mig"))
        cols = [r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'bet_paper__mig'").fetchall()]
        keep = ", ".join(c for c in cols)
        con.execute(f"INSERT INTO bet_paper__mig ({keep}) "
                    f"SELECT {keep} FROM bet_paper")
        con.execute("DROP TABLE bet_paper")
        con.execute("ALTER TABLE bet_paper__mig RENAME TO bet_paper")
        print("bet_paper migrated: PK now includes snapshot_kind")


# ---- rule registry (pure; unit-tested) --------------------------------------

def rules_fired(p_us: float, p_mkt: float, gp_home: int, gp_away: int,
                fav_star_out: bool, cap: float = CONF_EXCESS_CAP) -> list[str]:
    """Evaluate the pre-registered F4 rules for one game. Returns rule names
    (the pick side is our side of 0.5 — under same-side, also the favorite).

    Two registry-level vetoes run before any rule:
      1. OPPOSITE SIDE — never bet (known net-negative, D78).
      2. UPPER CONFIDENCE-EXCESS CAP — skip when conf_us - conf_mkt > `cap`
         (D112).  Pass cap=float('inf') to reproduce the pre-D112 registry.
    """
    same_side = (p_us - 0.5) * (p_mkt - 0.5) > 0
    if not same_side:
        return []                      # opposite side NEVER bet (known negative)
    if (abs(p_us - 0.5) - abs(p_mkt - 0.5)) > cap:
        return []                      # D112 upper confidence-excess cap
    pick_home = p_us > 0.5
    p_us_side = p_us if pick_home else 1 - p_us
    p_mkt_side = p_mkt if pick_home else 1 - p_mkt
    edge = p_us_side - p_mkt_side
    late = gp_home >= TANK_GP or gp_away >= TANK_GP
    tails = abs(p_us - 0.5) > CONF_TIER
    band = DIV_LO <= edge <= DIV_HI
    fired = []
    if edge > EDGE_R4 and late:
        fired.append("R4_LOWT")
    if tails and band and late:
        fired.append("T20_D03_10_W")
    if tails and band:
        fired.append("T20_D03_10")
    if edge > 0 and fav_star_out:
        fired.append("STAR_FAV_SHARPER")
    return fired


# ---- odds ------------------------------------------------------------------

def _jsonl_quotes(days: list[str]) -> list[tuple]:
    """h2h rows from raw odds JSONL via load_odds.flatten (no DB, no lock)."""
    from load_odds import flatten
    rows = []
    for d in days:
        path = RAW_ODDS / f"{d}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") not in (None, "main"):
                continue               # props records are per-event, not h2h
            try:
                rows += [r for r in flatten(rec, path.name, None)
                         if r[9] == "h2h"]
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def _table_quotes(con, game_date: dt.date) -> list[tuple]:
    """odds_quotes fallback, same tuple shape as load_odds.flatten output."""
    return con.execute("""
        SELECT snapshot_ts, ingest_ts, source, event_id, commence_time,
               home_team, away_team, bookmaker, book_last_update, market,
               outcome_name, outcome_desc, price_decimal, point, raw_file
        FROM odds_quotes WHERE market = 'h2h'
          AND commence_time >= ? AND commence_time < ?""",
        [game_date - dt.timedelta(days=1),
         game_date + dt.timedelta(days=2)]).fetchall()



# ---------------------------------------------------------------- D226 OFFSET
SPREAD_SCALE_MKT = 6.96      # the program's spread<->prob map (see header)
MODEL_SCALE = 7.2            # the model's own margin->prob link


def _offset_p(model, hid, aid, outs, today, yday, mkt, rest_diff=0.0):
    """Market-offset probability for one game (D224 gate, D226 promotion).

    The market-blind model forms its margin FIRST and without seeing a price;
    the opening number is then applied as a correction to it. If no market
    snapshot is available the blind probability is returned unchanged, so a dead
    odds feed degrades to the incumbent rather than to nothing.
    """
    from nbapred.market import offset as _off
    from nbapred.model.production import sigmoid
    m_blind = model.margin(hid, aid, outs.get(hid), outs.get(aid), today,
                           hid in yday, aid in yday)
    p_mkt = None if mkt is None else mkt.get("p_home")
    if p_mkt is None or not (0.0 < p_mkt < 1.0):
        return sigmoid(m_blind / MODEL_SCALE), m_blind, None
    open_margin = SPREAD_SCALE_MKT * math.log(p_mkt / (1.0 - p_mkt))
    m_off = _off.apply(m_blind, open_margin, rest_diff)
    return sigmoid(m_off / MODEL_SCALE), m_blind, m_off

def market_snapshot(quotes: list[tuple], home_name: str, away_name: str,
                    before_ts=None) -> dict | None:
    """Consensus + best-price + full-panel view of one game's h2h market.
    quotes: load_odds.flatten tuples. before_ts: only snapshots <= it (close).
    Returns {p_home, best: {side: (decimal, book)}, consensus: {side: median
    decimal}, panel: [(ts, book, side, price)], ts: freshest quote ts used,
    event_id, commence, n_books} or None when no book is two-sided."""
    by_book: dict[str, dict] = {}
    for q in quotes:
        (snap, _ing, _src, ev, commence, h, a, book, _upd, mkt, name, _desc,
         price, _pt, _raw) = q
        if mkt != "h2h" or h != home_name or a != away_name:
            continue
        if before_ts is not None and snap is not None and snap > before_ts:
            continue
        b = by_book.setdefault(book, {"snap": snap, "prices": {}})
        if snap is not None and (b["snap"] is None or snap > b["snap"]):
            b["snap"], b["prices"] = snap, {}
        if snap == b["snap"]:
            side = ("home" if name == home_name
                    else "away" if name == away_name else None)
            if side:
                b["prices"][side] = float(price)
                b["event_id"], b["commence"] = ev, commence
    imps, best, panel = [], {}, []
    per_side: dict[str, list[float]] = {"home": [], "away": []}
    ev_id = commence = quote_ts = None
    for book, b in by_book.items():
        d = b["prices"]
        if "home" not in d or "away" not in d or min(d.values()) <= 1.0:
            continue
        ih, ia = 1.0 / d["home"], 1.0 / d["away"]
        imps.append(ih / (ih + ia))
        for side in ("home", "away"):
            if side not in best or d[side] > best[side][0]:
                best[side] = (d[side], book)
            per_side[side].append(d[side])
            panel.append((b["snap"], book, side, d[side]))
        if b["snap"] is not None and (quote_ts is None or b["snap"] > quote_ts):
            quote_ts = b["snap"]
        ev_id, commence = b.get("event_id", ev_id), b.get("commence", commence)
    if not imps:
        return None
    consensus = {s: float(statistics.median(v)) for s, v in per_side.items()}
    return dict(p_home=float(statistics.median(imps)), best=best,
                consensus=consensus, panel=panel, ts=quote_ts,
                event_id=ev_id, commence=commence, n_books=len(imps))


# ---- star-out detection (live-feed analog of ba_intersection) ---------------

def star_out_live(con, out_players: set, before: dt.date) -> bool:
    """True iff any OUT player has trailing avg minutes >= STAR_MIN over the
    last STAR_TRAIL games with 12+ min, >= STAR_MIN_GP qualifying, < before."""
    outs = [int(p) for p in out_players or ()]
    if not outs:
        return False
    ph = ",".join("?" * len(outs))
    rows = con.execute(f"""
        WITH pg AS (
          SELECT s.player_id, s.seconds/60.0 m,
                 row_number() OVER (PARTITION BY s.player_id
                                    ORDER BY g.game_date DESC) rn
          FROM player_game_stats s
          JOIN (SELECT DISTINCT game_id, game_date FROM nba_games) g
            USING (game_id)
          WHERE s.game_id LIKE '002%' AND s.seconds >= 720
            AND g.game_date < ? AND s.player_id IN ({ph}))
        SELECT player_id, avg(m), count(*) FROM pg WHERE rn <= ?
        GROUP BY 1""", [before, *outs, STAR_TRAIL]).fetchall()
    return any(avg >= STAR_MIN and n >= STAR_MIN_GP for _, avg, n in rows)


# ---- emission core ----------------------------------------------------------

def _stakes(p_us_side: float, p_mkt_side: float, price, coeffs: dict) -> dict:
    """All four arms on the BEST shopped decimal.  open_shrunk reuses the
    shrunk-Kelly construction with the FROZEN open-arm coefficients."""
    if price:
        st = {a: arm_stake(a, p_us_side, p_mkt_side, price, coeffs)
              for a in SIZING_ARMS}
        st["open_shrunk"] = arm_stake("shrunk_kelly", p_us_side, p_mkt_side,
                                      price, OPEN_SHRUNK)
    else:
        st = {a: (STAKE if a == "flat" else 0.0) for a in ALL_ARMS}
    return st


def _candidate_rows(kind: str, today: dt.date, gid: str, home_ab: str,
                    away_ab: str, p_us: float, mkt: dict, gp_home: int,
                    gp_away: int, fso: bool, coeffs: dict,
                    extra_detail: str = "") -> list[list]:
    """bet_paper rows (INSERT_COLS order) for one game under one view."""
    p_mkt = mkt["p_home"]
    fired = rules_fired(p_us, p_mkt, gp_home, gp_away, fso)
    if not fired:
        return []
    pick_home = p_us > 0.5
    side = "home" if pick_home else "away"
    p_us_side = p_us if pick_home else 1 - p_us
    p_mkt_side = p_mkt if pick_home else 1 - p_mkt
    price, book = mkt["best"].get(side, (None, None))
    consensus = mkt.get("consensus", {}).get(side)
    implied = p_mkt_side          # consensus de-vig for our side
    edge = p_us_side - p_mkt_side
    conf_excess = abs(p_us - 0.5) - abs(p_mkt - 0.5)
    st = _stakes(p_us_side, p_mkt_side, price, coeffs)
    sh = max(0.0, coeffs["a"] + coeffs["b"] * edge)
    sh_open = max(0.0, OPEN_SHRUNK["a"] + OPEN_SHRUNK["b"] * edge)
    # ---- D178 FIX 2: the >=2-book gate ------------------------------------
    # n_books is the count of DISTINCT books that were TWO-SIDED at this
    # snapshot (market_snapshot only counts a book once it has both sides and
    # both decimals > 1.0), i.e. exactly the set the best-price shop ran over.
    n_books = int(mkt.get("n_books") or 0)
    single_book = n_books < MIN_BOOKS_OPEN
    # The gate BINDS AT THE OPEN ONLY.  That is where D142 measured the asset
    # (best-of-2 +49% CLV; worse-of-2 kills it) and where D167 decided to bet.
    # POST_REPORT/PRETIP still RECORD n_books/single_book for telemetry but
    # keep their CLV eligibility — narrowing them too would silently redefine
    # the two comparison views the whole three-view design exists to measure.
    refused = (kind == "OPEN") and single_book
    clv_eligible = not refused
    if refused:
        st = {a: 0.0 for a in ALL_ARMS}   # not booked: no arm deploys capital
    now = dt.datetime.now(dt.timezone.utc)
    detail = (f"n_books={n_books};fav_star_out={int(fso)};"
              f"gp={gp_home}/{gp_away}"
              + (f";single_book=1;clv_excluded=1;min_books={MIN_BOOKS_OPEN}"
                 if refused else "")
              + extra_detail)
    rows = []
    for rule in fired:
        rows.append([now, kind, mkt.get("ts"), today, str(gid), home_ab,
                     away_ab, side, rule, p_us, p_us_side, p_mkt_side, edge,
                     price, book, price, consensus, implied,
                     mkt["event_id"], mkt["commence"], st["flat"], detail,
                     conf_excess, CONF_EXCESS_CAP, sh,
                     coeffs["a"], coeffs["b"],
                     st["raw_kelly"], st["shrunk_kelly"],
                     sh_open, st["open_shrunk"],
                     n_books, single_book, clv_eligible])
        gap = (price - consensus) if price and consensus else float("nan")
        tag = (f"  *** SINGLE BOOK (k={n_books} < {MIN_BOOKS_OPEN}) — LOGGED, "
               f"NOT BOOKED, EXCLUDED FROM CLV (D178) ***" if refused else "")
        print(f"  [{kind}] {rule:<16} {away_ab}@{home_ab} {side} "
              f"p_us={p_us_side:.3f} p_mkt={p_mkt_side:.3f} edge={edge:+.3f} "
              f"best={price if price else 'n/a'}@{book} cons={consensus} "
              f"gap={gap:+.3f} k={n_books} stakes f={st['flat']:.2f} "
              f"rK={st['raw_kelly']:.2f} sK={st['shrunk_kelly']:.2f} "
              f"oS={st['open_shrunk']:.2f}{tag}")
    return rows


def _panel_rows(kind: str, today: dt.date, gid: str, mkt: dict) -> list[list]:
    now = dt.datetime.now(dt.timezone.utc)
    return [[today, str(gid), kind, ts, now, book, side, price]
            for ts, book, side, price in mkt.get("panel", [])]


def _flush(rows: list[list], panel: list[list]) -> None:
    """One SHORT batched write window for bet rows + panel telemetry."""
    if not rows and not panel:
        return
    w = _write_retry()
    _ensure_schema(w)
    if rows:
        w.executemany(
            "INSERT OR IGNORE INTO bet_paper (" + ",".join(INSERT_COLS) +
            ") VALUES (" + ",".join("?" * len(INSERT_COLS)) + ")", rows)
    if panel:
        w.executemany(
            "INSERT OR IGNORE INTO bet_quotes_panel (" + ",".join(PANEL_COLS) +
            ") VALUES (" + ",".join("?" * len(PANEL_COLS)) + ")", panel)
    w.close()


def _print_coeffs(coeffs: dict) -> None:
    print(f"registry: cap={CONF_EXCESS_CAP} | close shrinkage "
          f"a={coeffs['a']:+.4f} b={coeffs['b']:+.4f} "
          f"(fit {coeffs.get('fit_date', '?')} on "
          f"{','.join(coeffs.get('seasons', []))}) | open_shrunk FROZEN "
          f"a={OPEN_SHRUNK['a']:+.4f} b={OPEN_SHRUNK['b']:+.4f} (D120/D121)")


def emit(kind: str = "POST_REPORT", today: dt.date | None = None) -> int:
    """POST_REPORT / PRETIP emission: book the whole slate at this view's
    prices, plus full book-panel telemetry.  (OPEN uses scan_open.)"""
    assert kind in ("POST_REPORT", "PRETIP")
    from nbapred.config import current_season
    from nbapred.engine.slate import (filter_regular_season, slate_context,
                                      todays_games)
    today = today or dt.date.today()
    season = current_season(today)
    # D178 FIX 1: REGULAR SEASON ONLY.  todays_games() already filters, but the
    # engine keeps its own copy of the list and must not diverge from the list
    # slate_context built `outs` for — an unfiltered `games` here would KeyError
    # on ctx["outs"][gid] the first October night a preseason game is on the
    # scoreboard.  Never book a preseason/All-Star/exhibition game.
    games = filter_regular_season(todays_games(), where=kind)
    if not games:
        print(f"No NBA regular-season games today ({today}) — offseason no-op.")
        return 0
    from nba_api.stats.static import teams as _t
    id2name = {t["id"]: t["full_name"] for t in _t.get_teams()}
    id2ab = {t["id"]: t["abbreviation"] for t in _t.get_teams()}
    quotes = _jsonl_quotes([str(today), str(today - dt.timedelta(days=1))])
    con = _connect(read_only=True)
    if not quotes:
        quotes = _table_quotes(con, today)
        if quotes:
            print("using odds_quotes fallback (no raw JSONL for today)")
    extra = ""
    if kind == "POST_REPORT":
        try:   # hook: the 5PM ET injury report's capture time, logged PIT
            rts = con.execute("SELECT max(coalesce(report_ts, ingest_ts)) "
                              "FROM injury_reports WHERE game_date = ?",
                              [today]).fetchone()[0]
            extra = f";report_ts={rts}"
        except Exception:  # noqa: BLE001 — table absent on scratch DBs
            extra = ";report_ts=None"
    ctx = slate_context(con, season, games, today)
    model, gp, yday = ctx["model"], ctx["gp"], ctx["b2b"]
    coeffs = load_coeffs()                       # D112 shrinkage, logged PIT
    _print_coeffs(coeffs)
    rows, panel = [], []
    for gid, hid, aid in games:
        outs = ctx["outs"][gid]
        mkt = market_snapshot(quotes, id2name.get(hid, ""), id2name.get(aid, ""))
        p_us, _mb, _mo = _offset_p(model, hid, aid, outs, today, yday, mkt)
        if mkt is None:
            print(f"  {gid} {id2ab.get(aid)}@{id2ab.get(hid)}: no h2h lines — skipped")
            continue
        panel += _panel_rows(kind, today, gid, mkt)
        p_mkt = mkt["p_home"]
        fav_tid = hid if p_mkt >= 0.5 else aid
        fso = star_out_live(con, outs.get(fav_tid, set()), today)
        rows += _candidate_rows(kind, today, str(gid), id2ab.get(hid),
                                id2ab.get(aid), p_us, mkt, gp.get(hid, 0),
                                gp.get(aid, 0), fso, coeffs, extra)
    con.close()
    _flush(rows, panel)
    print(f"[{kind}] emitted {len(rows)} candidate rows, "
          f"{len(panel)} panel rows for {today}")
    return len(rows)


def scan_open(today: dt.date | None = None) -> int:
    """OPEN emission: book each game the FIRST time a two-sided h2h line
    appears for it (cron every 30 min, 14:00-02:00 UTC).  A game with OPEN
    panel rows already logged is never re-booked; quote_ts carries the actual
    quote timestamp, not the scan time.  Exits before any model fit when no
    new two-sided game exists (the cheap path, most scans)."""
    from nbapred.config import current_season
    from nbapred.engine.slate import (filter_regular_season, slate_context,
                                      todays_games)
    today = today or dt.date.today()
    season = current_season(today)
    # D178 FIX 1 — see emit().  OPEN is the view that books first and is the
    # one that runs every 30 min from 14:00 UTC in early October, i.e. the view
    # most exposed to a preseason slate.
    games = filter_regular_season(todays_games(), where="OPEN")
    if not games:
        print(f"No NBA regular-season games today ({today}) — offseason no-op.")
        return 0
    from nba_api.stats.static import teams as _t
    id2name = {t["id"]: t["full_name"] for t in _t.get_teams()}
    id2ab = {t["id"]: t["abbreviation"] for t in _t.get_teams()}
    quotes = _jsonl_quotes([str(today), str(today - dt.timedelta(days=1))])
    con = _connect(read_only=True)
    if not quotes:
        quotes = _table_quotes(con, today)
    try:
        seen = {r[0] for r in con.execute(
            "SELECT DISTINCT game_id FROM bet_quotes_panel "
            "WHERE game_date = ? AND snapshot_kind = 'OPEN'",
            [today]).fetchall()}
    except Exception:  # noqa: BLE001 — table not created yet
        seen = set()
    fresh = []
    for gid, hid, aid in games:
        if str(gid) in seen:
            continue
        mkt = market_snapshot(quotes, id2name.get(hid, ""), id2name.get(aid, ""))
        if mkt is not None:
            fresh.append((gid, hid, aid, mkt))
    if not fresh:
        con.close()
        print(f"[OPEN] no newly two-sided games ({len(seen)} already booked)")
        return 0
    thin = [(g, m["n_books"]) for g, _, _, m in fresh
            if int(m.get("n_books") or 0) < MIN_BOOKS_OPEN]
    print(f"[OPEN] {len(fresh)} newly two-sided game(s) — booking at first "
          f"quote; >={MIN_BOOKS_OPEN}-book rule (D178): "
          f"{len(fresh) - len(thin)} shoppable, {len(thin)} single-book "
          f"(logged, NOT booked, excluded from CLV)")
    for g, k in thin:
        print(f"  [OPEN] thin book panel game {g}: k={k}")
    ctx = slate_context(con, season, [(g, h, a) for g, h, a, _ in fresh], today)
    model, gp, yday = ctx["model"], ctx["gp"], ctx["b2b"]
    coeffs = load_coeffs()
    _print_coeffs(coeffs)
    rows, panel = [], []
    for gid, hid, aid, mkt in fresh:
        outs = ctx["outs"][gid]
        p_us, _mb, _mo = _offset_p(model, hid, aid, outs, today, yday, mkt)
        panel += _panel_rows("OPEN", today, gid, mkt)
        fav_tid = hid if mkt["p_home"] >= 0.5 else aid
        fso = star_out_live(con, outs.get(fav_tid, set()), today)
        rows += _candidate_rows("OPEN", today, str(gid), id2ab.get(hid),
                                id2ab.get(aid), p_us, mkt, gp.get(hid, 0),
                                gp.get(aid, 0), fso, coeffs)
    con.close()
    _flush(rows, panel)
    idx_ce = INSERT_COLS.index("clv_eligible")
    n_excl = sum(1 for r in rows if not r[idx_ce])
    print(f"[OPEN] emitted {len(rows)} candidate rows "
          f"({len(rows) - n_excl} booked, {n_excl} single-book/excluded), "
          f"{len(panel)} panel rows for {today}")
    return len(rows)


# ---- settle -----------------------------------------------------------------

def settle(today: dt.date | None = None) -> int:
    """Read phase (read_only) -> compute -> one SHORT batched write."""
    today = today or dt.date.today()
    w = _write_retry()                 # short: schema/migration only
    _ensure_schema(w)
    w.close()
    con = _connect(read_only=True)
    open_rows = con.execute("""
        SELECT game_date, game_id, side, rule, snapshot_kind, price_decimal,
               implied_p, event_id, commence_ts, stake_units,
               coalesce(stake_raw_kelly, 0.0), coalesce(stake_shrunk_kelly, 0.0),
               coalesce(stake_open_shrunk, 0.0)
        FROM bet_paper WHERE settled_ts IS NULL AND game_date < ?""",
        [today]).fetchall()
    if not open_rows:
        con.close()
        print("nothing to settle")
        return 0
    days = sorted({str(r[0]) for r in open_rows}
                  | {str(r[0] + dt.timedelta(days=1)) for r in open_rows})
    quotes = _jsonl_quotes(days)
    updates = []
    for (gdate, gid, side, rule, kind, price, implied, ev, commence, stake,
         stake_rk, stake_sk, stake_os) in open_rows:
        res = con.execute("""
            SELECT max(CASE WHEN is_home THEN wl END),
                   max(CASE WHEN is_home THEN team_abbrev END)
            FROM nba_games WHERE game_id = ? AND wl IS NOT NULL""",
            [gid]).fetchone()
        if not res or res[0] is None:
            continue                       # result not ingested yet — retry later
        home_won = res[0] == "W"
        outcome = int(home_won if side == "home" else not home_won)

        def _pnl(s):
            """PnL for one arm; a 0-stake arm scores exactly 0 (capital not
            deployed), never a loss."""
            if price is None or not s:
                return 0.0
            return s * (price - 1) if outcome else -s

        pnl, pnl_rk = _pnl(stake), _pnl(stake_rk)
        pnl_sk, pnl_os = _pnl(stake_sk), _pnl(stake_os)
        # close: last pre-tip snapshot for this event across sources
        close_price = close_implied = clv = None
        ev_quotes = [q for q in quotes if q[3] == ev] if ev else []
        if not ev_quotes:
            ev_quotes = [q for q in con.execute("""
                SELECT snapshot_ts, ingest_ts, source, event_id, commence_time,
                       home_team, away_team, bookmaker, book_last_update,
                       market, outcome_name, outcome_desc, price_decimal,
                       point, raw_file
                FROM odds_quotes WHERE market='h2h' AND event_id = ?""",
                [ev]).fetchall()] if ev else []
        if ev_quotes:
            hname, aname = ev_quotes[0][5], ev_quotes[0][6]
            snap = market_snapshot(ev_quotes, hname, aname, before_ts=commence)
            if snap:
                close_implied = (snap["p_home"] if side == "home"
                                 else 1 - snap["p_home"])
                cp = snap["best"].get(side)
                close_price = cp[0] if cp else None
                clv = (close_implied - implied) if implied is not None else None
        updates.append([outcome, pnl, pnl_rk, pnl_sk, pnl_os, close_price,
                        close_implied, clv, dt.datetime.now(dt.timezone.utc),
                        gdate, gid, side, rule, kind])
    con.close()
    if updates:
        w = _write_retry()             # short: one batched UPDATE window
        w.executemany("""UPDATE bet_paper SET outcome=?, pnl_units=?,
            pnl_raw_kelly=?, pnl_shrunk_kelly=?, pnl_open_shrunk=?,
            close_price=?, close_implied=?, clv=?, settled_ts=?
            WHERE game_date=? AND game_id=? AND side=? AND rule=?
              AND snapshot_kind=?""", updates)
        w.close()
    print(f"settled {len(updates)}/{len(open_rows)} open rows")
    return len(updates)


# ---- report -----------------------------------------------------------------

ARM_COLS = {"flat": ("stake_units", "pnl_units"),
            "raw_kelly": ("stake_raw_kelly", "pnl_raw_kelly"),
            "shrunk_kelly": ("stake_shrunk_kelly", "pnl_shrunk_kelly"),
            "open_shrunk": ("stake_open_shrunk", "pnl_open_shrunk")}


def report() -> None:
    """Registry scorecard, one block per (SNAPSHOT VIEW x SIZING ARM).  The
    arms run in parallel by design (D112 (c) + the open_shrunk diagnostic);
    the views run in parallel by design (timing attribution).  Do NOT pick an
    arm or a view from this table until the season's n is large enough — that
    decision is itself a selection and must be pre-declared."""
    con = _connect(read_only=True)
    try:
        con.execute("SELECT 1 FROM bet_paper LIMIT 1")
    except Exception:
        print("no bet_paper table yet")
        con.close()
        return
    have = {r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'bet_paper'").fetchall()}
    kinds = [r[0] for r in con.execute(
        "SELECT DISTINCT coalesce(snapshot_kind, 'POST_REPORT') "
        "FROM bet_paper ORDER BY 1").fetchall()] if "snapshot_kind" in have \
        else ["POST_REPORT"]
    # D178 FIX 2: CLV is scored on the ELIGIBLE set only (single-book OPEN rows
    # are logged but excluded).  `ELIG` degrades to "everything" on a table that
    # predates the column, so an old bet_paper reports exactly as it used to.
    ELIG = "coalesce(clv_eligible, TRUE)" if "clv_eligible" in have else "TRUE"
    NB = "n_books" if "n_books" in have else "NULL"
    SB = ("coalesce(single_book, FALSE)" if "single_book" in have else "FALSE")
    for kind in kinds:
        print(f"\n==== VIEW {kind} ====")
        print(f"{'rule':<18}{'n':>5}{'settled':>9}{'hit%':>7}{'PnL(u)':>9}"
              f"{'ROI%':>8}{'staked':>9}{'CLV':>9}{'n_clv':>6}{'1book':>7}"
              f"{'meanK':>7}")
        for arm, (scol, pcol) in ARM_COLS.items():
            if scol not in have or pcol not in have:
                print(f"\n-- {arm}: columns absent (older table) --")
                continue
            rows = con.execute(f"""
                SELECT rule, count(*) n, sum(CASE WHEN settled_ts IS NOT NULL
                         THEN 1 ELSE 0 END) settled,
                       avg(CASE WHEN {scol} > 0 THEN outcome END) hit,
                       sum({pcol}) pnl,
                       sum(CASE WHEN settled_ts IS NOT NULL THEN {scol} END) staked,
                       avg(CASE WHEN {ELIG} THEN clv END) mean_clv,
                       count(CASE WHEN {ELIG} THEN clv END) n_clv,
                       sum(CASE WHEN {SB} THEN 1 ELSE 0 END) n_single,
                       avg({NB}) mean_books
                FROM bet_paper
                WHERE coalesce(snapshot_kind, 'POST_REPORT') = ?
                GROUP BY 1 ORDER BY 1""", [kind]).fetchall()
            print(f"\n-- ARM {arm} --")
            for r, n, s, hit, pnl, staked, clv, n_clv, n1, mk in rows:
                roi = 100 * pnl / staked if staked else float("nan")
                print(f"{r:<18}{n:>5}{s or 0:>9}{100*(hit or 0):>7.1f}"
                      f"{(pnl or 0):>9.2f}{roi:>8.2f}{(staked or 0):>9.1f}"
                      f"{(clv if clv is not None else float('nan')):>9.4f}"
                      f"{n_clv:>6}{(n1 or 0):>7}"
                      f"{(mk if mk is not None else float('nan')):>7.2f}")
    con.close()


# ---- monthly report + real-stakes trigger (print-only) ----------------------

def monthly_report(today: dt.date | None = None) -> None:
    """Per-view monthly mean CLV vs the D178 bands, plus the codified
    REAL-STAKES TRIGGER status.  PRINTS ONLY — the engine never deploys
    capital, never sizes real stakes, never flips any switch.  CLV months are
    computed on UNIQUE games (a game firing k rules is one CLV observation,
    the D120 union-of-rules convention, 67 bets/month in the D178 frame), on
    clv_eligible rows only (D178 >=2-book rule).  The BAND'S DERIVATION INPUTS
    ARE PRINTED WITH IT — centre, per-bet sd, n/month, se, k — so the numbers
    can never again drift from the construction that produced them."""
    today = today or dt.date.today()
    con = _connect(read_only=True)
    try:
        con.execute("SELECT 1 FROM bet_paper LIMIT 1")
    except Exception:
        print("no bet_paper table yet")
        con.close()
        return
    have = {r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'bet_paper'").fetchall()}
    # D178 FIX 2: single-book OPEN rows are logged but NOT in the CLV set.
    ELIG = "coalesce(clv_eligible, TRUE)" if "clv_eligible" in have else "TRUE"
    SB = "coalesce(single_book, FALSE)" if "single_book" in have else "FALSE"
    rows = con.execute(f"""
        WITH g AS (SELECT DISTINCT coalesce(snapshot_kind,'POST_REPORT') kind,
                          strftime(game_date, '%Y-%m') ym, game_date, game_id,
                          side, clv, price_decimal, stake_open_shrunk,
                          {ELIG} elig, {SB} sb
                   FROM bet_paper)
        SELECT kind, ym, avg(CASE WHEN elig THEN clv END) mean_clv,
               count(CASE WHEN elig THEN clv END) n_clv,
               count(*) n_bets,
               sum(CASE WHEN price_decimal IS NOT NULL AND elig
                        THEN 1 ELSE 0 END) n_priced,
               sum(CASE WHEN stake_open_shrunk > 0 THEN 1 ELSE 0 END) n_os,
               sum(CASE WHEN sb THEN 1 ELSE 0 END) n_single
        FROM g GROUP BY 1, 2 ORDER BY 1, 2""").fetchall()
    con.close()
    # ---- D178: the DERIVATION INPUTS, printed with the verdict -------------
    b = CLV_BAND
    print(f"MONTHLY CLV REPORT (as of {today})")
    print(f"  BANDS ({b['d']}): RED < {CLV_MONTH_RED:+.6f} | "
          f"GOOD > {CLV_MONTH_GOOD:+.6f}")
    print(f"  DERIVATION INPUTS (a band and its inputs are printed together so "
          f"they can never drift apart):")
    print(f"    frame        {b['frame']}")
    print(f"    arm          {b['arm']}")
    print(f"    seasons      {', '.join(b['seasons'])}   "
          f"n={b['n_bets']} union bets over {b['n_months']} months")
    print(f"    centre       {b['centre']:+.6f}   (MEAN CLV OF THE UNION — "
          f"NOT of the all-same-side universe; that swap was defect 1)")
    print(f"    per-bet sd   {b['per_bet_sd']:.6f}")
    print(f"    n/month      {b['bets_per_month']:.0f}  (median union bets per "
          f"calendar month)")
    print(f"    monthly se   {b['se_monthly']:.6f}  = "
          f"{b['per_bet_sd']:.6f}/sqrt({b['bets_per_month']:.0f})")
    print(f"    bands        centre +- {b['k_sigma']:.0f} * se")
    print(f"    source       {b['source']}")
    print(f"    anchor       {b['anchor']}")
    print(f"    supersedes   {b['supersedes']}")
    print(f"  CLV IS SCORED ON clv_eligible ROWS ONLY (D178 >= "
          f"{MIN_BOOKS_OPEN}-book rule at the OPEN); `1book` counts logged-but-"
          f"excluded rows.")
    print(f"  *** CLV IS THE MONITOR, NOT THE OBJECTIVE (D176). *** D176 "
          f"measured the two apart: the availability-divergence selector bought")
    print(f"      MORE CLV (6/6 cells, +0.143 pts) and LESS ROI (1/6, -1.16pp), "
          f"while the CLV-TARGETED selector bought essentially no extra CLV")
    print(f"      (+0.004) and the MOST ROI. So CLV IS NOT A SUFFICIENT "
          f"STATISTIC FOR BET SELECTION, and A GREEN CLV MONTH BELOW IS NOT")
    print(f"      EVIDENCE OF PROFITABILITY — it is a fast-resolving read on "
          f"execution and timing. ROI, AGAINST THE INCUMBENT, is the objective.")
    by_kind: dict[str, list] = {}
    for kind, ym, mean_clv, n_clv, n_bets, n_priced, n_os, n_sb in rows:
        by_kind.setdefault(kind, []).append(
            (ym, mean_clv, n_clv, n_bets, n_priced, n_os, n_sb))
    for kind in sorted(by_kind):
        print(f"\n-- VIEW {kind} --")
        print(f"{'month':<9}{'meanCLV':>9}{'n_clv':>7}{'bets':>6}"
              f"{'priced':>8}{'openShr>0':>10}{'1book':>7}  flag")
        for ym, mean_clv, n_clv, n_bets, n_priced, n_os, n_sb in by_kind[kind]:
            flag = ""
            if mean_clv is not None:
                flag = ("RED" if mean_clv < CLV_MONTH_RED
                        else "GOOD" if mean_clv > CLV_MONTH_GOOD else "ok")
            print(f"{ym:<9}"
                  f"{(mean_clv if mean_clv is not None else float('nan')):>9.4f}"
                  f"{n_clv:>7}{n_bets:>6}{n_priced:>8}{n_os:>10}{n_sb:>7}"
                  f"  {flag}")
    # ---- D178: the band's own reachability, so it is never re-mis-specified -
    from statistics import NormalDist
    nd = NormalDist(b["centre"], b["se_monthly"])
    p_good = 1 - nd.cdf(CLV_MONTH_GOOD)
    p_red = nd.cdf(CLV_MONTH_RED)
    wait = (1 / p_good ** TRIGGER_MONTHS) if p_good > 0 else float("inf")
    print(f"\n  BAND ARITHMETIC (monthly mean ~ N(centre, se) at "
          f"{b['bets_per_month']:.0f} bets/month):")
    print(f"    P(month > GOOD) = {p_good:.4f}   P(month < RED) = "
          f"{p_red:.4f}   E[months to {TRIGGER_MONTHS} consecutive GOOD] = "
          f"{wait:,.0f}")
    print(f"    A symmetric +-{b['k_sigma']:.0f}-sigma band makes the "
          f"{TRIGGER_MONTHS}-consecutive-GOOD trigger effectively "
          f"UNREACHABLE. That is arithmetic, not a defect of this month's\n"
          f"    data, and it was ALSO true of D120/D121 — whose GOOD line only "
          f"looked reachable because it sat at +1.23 sigma of THIS frame by "
          f"accident.\n    RE-SPECIFYING THE TRIGGER IS A SEPARATE PRODUCT "
          f"DECISION AND NEEDS ITS OWN D-LINE. The band below is correct; the "
          f"gate above it is not this entry's to move.")
    # ---- trigger: OPEN view, completed calendar months only ----------------
    cur_ym = today.strftime("%Y-%m")
    om = [(ym, mc, np_, no) for ym, mc, nc, nb, np_, no, _sb in
          by_kind.get("OPEN", []) if ym < cur_ym and mc is not None]
    print("\nREAL-STAKES TRIGGER (codified; PRINT-ONLY — never acts):")
    print(f"  need: {TRIGGER_MONTHS} consecutive completed months OPEN mean "
          f"CLV > {CLV_MONTH_GOOD}; no completed OPEN month < {CLV_MONTH_RED}; "
          f"stake_open_shrunk > 0 on >= "
          f"{TRIGGER_OPEN_SHRUNK_FRAC:.0%} of priced OPEN candidates in that "
          f"window; earliest {TRIGGER_EARLIEST}.")
    if len(om) < TRIGGER_MONTHS:
        print(f"  status: NOT MET — only {len(om)} completed OPEN month(s) "
              f"with CLV data (need {TRIGGER_MONTHS}).")
        return
    last = om[-TRIGGER_MONTHS:]
    good = all(mc > CLV_MONTH_GOOD for _, mc, _, _ in last)
    consecutive = True
    for (ym0, *_), (ym1, *_) in zip(last, last[1:]):
        y0, m0 = int(ym0[:4]), int(ym0[5:7])
        y1, m1 = int(ym1[:4]), int(ym1[5:7])
        if (y1, m1) != ((y0 + 1, 1) if m0 == 12 else (y0, m0 + 1)):
            consecutive = False
    red = [ym for ym, mc, _, _ in om if mc < CLV_MONTH_RED]
    priced = sum(np_ for _, _, np_, _ in last)
    os_pos = sum(no for _, _, _, no in last)
    frac = (os_pos / priced) if priced else 0.0
    date_ok = today >= TRIGGER_EARLIEST
    checks = [
        (f"last {TRIGGER_MONTHS} completed months all > {CLV_MONTH_GOOD}",
         good, ", ".join(f"{ym}={mc:+.4f}" for ym, mc, _, _ in last)),
        ("months consecutive", consecutive, ""),
        (f"no completed month < {CLV_MONTH_RED}", not red,
         f"red months: {red}" if red else ""),
        (f"open_shrunk staked on >= {TRIGGER_OPEN_SHRUNK_FRAC:.0%} of priced "
         f"OPEN candidates", frac >= TRIGGER_OPEN_SHRUNK_FRAC,
         f"{os_pos}/{priced} = {frac:.1%}"),
        (f"date >= {TRIGGER_EARLIEST}", date_ok, str(today)),
    ]
    all_ok = all(ok for _, ok, _ in checks)
    for label, ok, note in checks:
        print(f"  [{'x' if ok else ' '}] {label}" + (f"  ({note})" if note else ""))
    if all_ok:
        print(f"  status: TRIGGER MET — real stakes may be CONSIDERED, "
              f"starting FLAT {TRIGGER_START_STAKE}u (never Kelly). "
              f"THIS ENGINE STILL TAKES NO ACTION; a human decision + a new "
              f"D-line are required.")
    else:
        print("  status: NOT MET — paper only.")


# ---- dry run (historical rehearsal; TEMP DB, never bet_paper) ---------------

def dry_run(date_str: str, db_path: str | None = None,
            books: int = 2) -> None:
    """Rehearse all three views + panel telemetry + all four arms + settle
    + reports on ONE historical date, writing ONLY to a temp DuckDB.  Prices
    come from data/derived/odds_open.csv (open ML -> OPEN view; close ML ->
    POST_REPORT/PRETIP — no intraday path exists historically), p_us from
    data/ds_rt1_pergame.csv p_full (the registered sim frame; no model fit),
    outcomes from the real nba_games (read-only).  fav_star_out is False (no
    historical live injury feed), so STAR_FAV_SHARPER cannot fire here.
    This is a PLUMBING rehearsal, not a backtest.

    `books` (D178): how many books the synthetic panel carries.
      1 — the HONEST historical shape.  data/derived/odds_open.csv is a single
          CONSENSUS open/close moneyline per game, and D174 (5) established
          that NO historical per-book ML OPEN panel exists for 2024-25/2025-26
          (ESPN collapsed to one operator; Action Network has no per-book
          opener at all).  Every OPEN row is therefore refused by the D178
          >=2-book rule — which is exactly what should be rehearsed.
      2 — adds a SECOND, CLEARLY-LABELLED book `spreadimplied_<tag>` priced off
          the SAME row's spread via the program's own map
          p = sigmoid(margin/6.96), dec = max(1/(p*OVERROUND), MIN_DEC).  It is
          real data from the same source, not noise, but it is NOT a second
          operator — it exists so the BOOKED branch of the >=2-book rule and
          all four sizing arms are exercised at the OPEN.  SYNTHETIC; never a
          measurement."""
    global DB_OVERRIDE
    import csv as _csv
    import tempfile
    date = dt.date.fromisoformat(date_str)
    day = str(date)
    rt1 = {}
    with open(ROOT / "data" / "ds_rt1_pergame.csv") as f:
        for r in _csv.DictReader(f):
            if r["game_date"] == day:
                rt1[(r["home"], r["away"])] = r
    oo = {}
    with open(ROOT / "data" / "derived" / "odds_open.csv") as f:
        for r in _csv.DictReader(f):
            if r["game_date"][:10] == day:
                oo[(r["home"], r["away"])] = r
    if not rt1:
        print(f"dry-run: no ds_rt1_pergame rows for {day} — pick another date")
        return
    def am2dec(a):
        try:
            a = float(a)
        except (TypeError, ValueError):
            return None
        if a == 0:
            return None
        return 1.0 + a / 100.0 if a > 0 else 1.0 + 100.0 / abs(a)
    t_open = dt.datetime(date.year, date.month, date.day, 14, 0,
                         tzinfo=dt.timezone.utc)
    t_close = dt.datetime(date.year, date.month, date.day, 23, 50,
                          tzinfo=dt.timezone.utc)
    commence = dt.datetime(date.year, date.month, date.day, 23, 59,
                           tzinfo=dt.timezone.utc)
    # D178: the synthetic second book, priced off the SAME row's spread.
    from bet_sim3 import MIN_DEC, OVERROUND
    SPREAD_SCALE = 6.96                 # nbapred/ingest/kaggle_odds.py

    def sp_dec(margin):
        """(dec_home, dec_away) from the spread, the program's own map."""
        try:
            mg = float(margin)
        except (TypeError, ValueError):
            return None, None
        import math
        ph = 1.0 / (1.0 + math.exp(-mg / SPREAD_SCALE))
        return (max(1.0 / (ph * OVERROUND), MIN_DEC),
                max(1.0 / ((1 - ph) * OVERROUND), MIN_DEC))

    open_q, close_q = [], []
    games = []
    for key, r in rt1.items():
        o = oo.get(key)
        if o is None:
            continue
        h, a = key
        src = (o.get("source") or "odds_open").replace("/", "+")
        gid = r["game_id"].zfill(10)
        games.append((gid, h, a, r))
        for (bucket, ts, mlh, mla, tag, mg) in (
                (open_q, t_open, o.get("open_ml_home"), o.get("open_ml_away"),
                 "open", o.get("open_margin")),
                (close_q, t_close, o.get("close_ml_home"),
                 o.get("close_ml_away"), "close", o.get("close_margin"))):
            panel = [(f"{src}_{tag}", am2dec(mlh), am2dec(mla))]
            if books >= 2:
                sh, sa = sp_dec(mg)
                panel.append((f"spreadimplied_{tag}", sh, sa))
            for book, dh, da in panel[:books]:
                if dh is None or da is None:
                    continue
                bucket.append((ts, None, "odds_open", gid, commence, h, a,
                               book, None, "h2h", h, None, dh, None, "dry"))
                bucket.append((ts, None, "odds_open", gid, commence, h, a,
                               book, None, "h2h", a, None, da, None, "dry"))
    if db_path is None:
        db_path = str(Path(tempfile.mkdtemp(prefix="bet_engine_dry_"))
                      / "dry.duckdb")
    print(f"DRY RUN {day}: {len(games)} games with model+odds rows; "
          f"TEMP DB {db_path} — real bet_paper is NOT touched.")
    print(f"  panel: books={books} "
          + ("(consensus ML only — the honest historical shape; every OPEN row "
             "should be REFUSED by the D178 >=2-book rule)" if books < 2 else
             "(consensus ML + SYNTHETIC spread-implied second book, D178 — "
             "plumbing only, NOT a second operator)"))
    # outcomes for settle, copied read-only from the real DB
    from nbapred.db import connect as _real_connect
    rcon = _real_connect(read_only=True)
    gids = [g for g, _, _, _ in games]
    ph = ",".join("?" * len(gids))
    ngrows = rcon.execute(
        f"SELECT season, game_id, game_date, team_id, team_abbrev, matchup, "
        f"is_home, wl, pts, ingest_ts FROM nba_games "
        f"WHERE game_id IN ({ph})", gids).fetchall()
    rcon.close()
    prev = DB_OVERRIDE
    DB_OVERRIDE = db_path
    try:
        w = _write_retry()
        _ensure_schema(w)
        w.execute("""CREATE TABLE IF NOT EXISTS nba_games (season VARCHAR,
            game_id VARCHAR, game_date DATE, team_id BIGINT,
            team_abbrev VARCHAR, matchup VARCHAR, is_home BOOLEAN, wl VARCHAR,
            pts INTEGER, ingest_ts TIMESTAMPTZ)""")
        w.execute("""CREATE TABLE IF NOT EXISTS odds_quotes (
            snapshot_ts TIMESTAMPTZ, ingest_ts TIMESTAMPTZ, source VARCHAR,
            event_id VARCHAR, commence_time TIMESTAMPTZ, home_team VARCHAR,
            away_team VARCHAR, bookmaker VARCHAR, book_last_update TIMESTAMPTZ,
            market VARCHAR, outcome_name VARCHAR, outcome_desc VARCHAR,
            price_decimal DOUBLE, point DOUBLE, raw_file VARCHAR)""")
        if ngrows:
            w.executemany("INSERT INTO nba_games VALUES (" +
                          ",".join("?" * 10) + ")", ngrows)
        if close_q:   # settle's close reconstruction reads odds_quotes
            w.executemany("INSERT INTO odds_quotes VALUES (" +
                          ",".join("?" * 15) + ")", close_q)
        w.close()
        coeffs = load_coeffs()
        _print_coeffs(coeffs)
        summary = {}
        for kind, quotes in (("OPEN", open_q), ("POST_REPORT", close_q),
                             ("PRETIP", close_q)):
            rows, panel = [], []
            for gid, h, a, r in games:
                mkt = market_snapshot(quotes, h, a)
                if mkt is None:
                    continue
                panel += _panel_rows(kind, date, gid, mkt)
                rows += _candidate_rows(
                    kind, date, gid, h, a, float(r["p_full"]), mkt,
                    int(float(r["gp_home"])), int(float(r["gp_away"])),
                    False, coeffs, ";dry_run=1")
            _flush(rows, panel)
            arms = {arm: 0.0 for arm in ALL_ARMS}
            idx = {c: i for i, c in enumerate(INSERT_COLS)}
            n_excl = 0
            for row in rows:
                arms["flat"] += row[idx["stake_units"]]
                arms["raw_kelly"] += row[idx["stake_raw_kelly"]]
                arms["shrunk_kelly"] += row[idx["stake_shrunk_kelly"]]
                arms["open_shrunk"] += row[idx["stake_open_shrunk"]]
                n_excl += 0 if row[idx["clv_eligible"]] else 1
            summary[kind] = (len(rows), len(panel), arms, n_excl)
        n_settled = settle(today=date + dt.timedelta(days=2))
        print("\nDRY-RUN SUMMARY")
        for kind, (nb, np_, arms, nx) in summary.items():
            print(f"  {kind:<12} bet rows {nb:>3} ({nb - nx} booked, {nx} "
                  f"single-book/CLV-excluded)  panel rows {np_:>3}  "
                  f"staked: " + "  ".join(f"{a}={v:.2f}"
                                          for a, v in arms.items()))
        print(f"  settled rows: {n_settled}")
        report()
        monthly_report(today=date + dt.timedelta(days=40))
    finally:
        DB_OVERRIDE = prev
    print(f"\nDRY RUN COMPLETE — temp DB left at {db_path} for inspection.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-open", action="store_true",
                    help="book OPEN view for games whose first two-sided line "
                         "just appeared (cron */30, 14:00-02:00 UTC)")
    ap.add_argument("--emit-post-report", action="store_true",
                    help="POST_REPORT view (cron 22:10 UTC, after the 5PM ET "
                         "injury report)")
    ap.add_argument("--emit-pretip", action="store_true",
                    help="PRETIP view (cron 23:55 UTC)")
    ap.add_argument("--emit", action="store_true",
                    help="legacy alias for --emit-post-report")
    ap.add_argument("--settle", action="store_true", help="settle open rows")
    ap.add_argument("--report", action="store_true", help="registry scorecard")
    ap.add_argument("--monthly-report", action="store_true",
                    help="monthly CLV vs bands + real-stakes trigger status "
                         "(print-only)")
    ap.add_argument("--dry-run", metavar="DATE",
                    help="historical rehearsal (YYYY-MM-DD) in a TEMP DB")
    ap.add_argument("--dry-db", metavar="PATH", default=None,
                    help="temp DB path for --dry-run (default: mkdtemp)")
    ap.add_argument("--dry-books", type=int, default=2, choices=(1, 2),
                    help="books in the --dry-run panel: 1 = honest historical "
                         "single consensus ML (every OPEN row refused by the "
                         "D178 >=2-book rule); 2 = + a SYNTHETIC "
                         "spread-implied book so the booked branch and all "
                         "four sizing arms are exercised (default 2)")
    args = ap.parse_args()
    ran = False
    if args.dry_run:
        dry_run(args.dry_run, args.dry_db, books=args.dry_books)
        ran = True
    if args.settle:
        settle()
        ran = True
    if args.scan_open:
        scan_open()
        ran = True
    if args.emit_post_report or args.emit:
        emit("POST_REPORT")
        ran = True
    if args.emit_pretip:
        emit("PRETIP")
        ran = True
    if args.monthly_report:
        monthly_report()
        ran = True
    if args.report:
        report()
        ran = True
    if not ran:
        emit("POST_REPORT")


if __name__ == "__main__":
    main()
