# OCTOBER RUNBOOK — the 2026-27 live path

> **DATA-COVERAGE CAVEAT — READ BEFORE QUOTING ANY NUMBER IN THIS FILE.**
> The daily NBA injury report, which the model's availability leg depends on,
> begins **2018-12-17 — mid-way through 2018-19**. Coverage of regular-season
> game dates is **0% before that, 63.7% in 2018-19, and 95–100% from 2019-20
> onward**. Only **2019-20 → 2025-26 (7 seasons)** is fully covered, and that is
> the only frame in which the model runs as designed. Earlier seasons score a
> *crippled variant* whose availability leg is fed inputs it was never meant to
> have. Any figure here spanning seasons before 2019-20 — including every
> 14-season and 19-season figure — blends two different models and should be
> read as historical context, not as a description of the shipped system.
> (`D186`)

Registered D178. Read this before opening night and again the morning after.
It covers: the cron layout, what each job does, **what breaks first and how you
can tell**, and every kill switch with its default.

Everything here is a PAPER product. `bet_engine.py` never deploys capital, never
sizes a real stake, and never flips a switch. `--monthly-report` prints a trigger
status; a human plus a new D-line is the only path to real money.

---

## 0. THE 60-SECOND HEALTH CHECK

Run these four, in order, on any morning you are unsure:

```sh
cd /opt/nba_model
python3 scripts/predict_today.py | head -20        # model side alive?
python3 scripts/bet_engine.py --report             # views x arms scorecard
python3 scripts/bet_engine.py --monthly-report     # CLV vs the D178 bands
tail -50 data/logs/bet_engine.log
```

Offseason / no slate is a CLEAN exit with `No NBA regular-season games ...`.
That message is success, not failure.

---

## 1. CRON LAYOUT (UTC)

Source of truth: `ops/crontab.example`. The odds logger is a **systemd service**
(`ops/odds-logger.service`), not a cron job.

| UTC | Job | What it does |
|---|---|---|
| service | `run_odds_logger.py` | The Odds API -> `data/raw/odds/YYYY-MM-DD.jsonl`, append-only. **Never opens DuckDB** (single-writer rule). Main lines every 30 min, tightened to 5 min inside 90 min of tip; free-tier pacing via `ODDS_MONTHLY_BUDGET`. |
| `*/15 * * * *` | `poll_injury_report.py` | Probes the league CDN for new injury PDFs, archives to `data/raw/injury_reports/`. |
| `*/30 * * * *` | `poll_espn_lines.py` | ESPN scoreboard odds. **ONE BOOK** (ESPN BET) — see §3. |
| `09:30` | `pull_nba_daily.py` | stats.nba.com daily pull. |
| `09:50` | `load_odds.py` | odds JSONL -> `odds_quotes`. |
| `10:20` | `build_player_stats.py` | `player_game_stats` from cached box/PBP, **including 001 preseason rows** — the D84-A October bridge needs them BEFORE the opener. |
| `10:40` | `bet_engine.py --settle` | Fills outcome/pnl/CLV on yesterday's rows. |
| `*/30` in `0-1,14-23` | `bet_engine.py --scan-open` | **OPEN** view: books each game the first time a two-sided h2h line appears. Cheap-exits before any model fit when no new two-sided game exists. |
| `22:10` | `bet_engine.py --emit-post-report` | **POST_REPORT** view, just after the 5PM ET injury report. |
| `23:55` | `bet_engine.py --emit-pretip` | **PRETIP** view, last quote before tip. |
| `11:00` on the 1st | `bet_engine.py --monthly-report` | CLV vs the D178 bands + real-stakes trigger status. PRINT-ONLY. |
| `08:00` on 1 Oct | `f4_shrinkage.py --refit` | ANNUAL edge-shrinkage refit on COMPLETED seasons only. **Never re-run mid-season chasing results.** |
| `10:15` Mondays | `scrape_2k.py` | 2K ratings. |

**`scripts/predict_today.py` IS NOT IN CRON.** It is the human-readable daily
printout and is run by hand. That is deliberate — it is not on the write path —
but it also means a break in it is only ever found by running it. Run it on
opening night.

### Ordering constraints that actually matter
* `load_odds` (09:50) must land before anything reads `odds_quotes`, but the
  emitters prefer the raw JSONL (`_jsonl_quotes`) and only fall back to the
  table, so a late load degrades rather than breaks.
* `build_player_stats` (10:20) must have run **before opening night** or the
  October bridge has no 001 rosters and every team is in the dead state.
* One writer at a time. Every read connection in the engine is
  `read_only=True`; writes are short, batched, and retry on lock for 60s x 10
  (`_write_retry`).

---

## 2. WHAT EACH BET-ENGINE VIEW BOOKS

The SAME rule-selected bets, booked three times at three prices, so timing is
measurable:

* **OPEN** — the first two-sided line. `quote_ts` is the ACTUAL quote timestamp,
  not the scan time. A game already carrying OPEN panel rows is never re-booked.
  **Since D178 an OPEN row is only BOOKED when >= 2 distinct books are
  two-sided** (§3).
* **POST_REPORT** — after the 5PM ET report; `detail` carries `report_ts=...`.
* **PRETIP** — last pre-tip quote.

Four sizing arms run in parallel and none is selected: `flat` (the honest
control), `raw_kelly`, `shrunk_kelly` (the D112 ship, close-fit), `open_shrunk`
(DIAGNOSTIC, frozen D120/D121 open calibration, paper only).

**Do not pick an arm or a view from `--report` mid-season.** That choice is
itself a selection and must be pre-declared.

---

## 3. THE >= 2-BOOK RULE AT THE OPEN (D178) — AND THE THING MOST LIKELY TO BITE

D142 measured the shop: best-of-2 lifts CLV ~49%, and taking the WORSE of two
books erases essentially all of it (+0.0092 -> -0.0007). So:

> An OPEN row is BOOKED only when `n_books >= MIN_BOOKS_OPEN` (2). With one
> book the observation is **still written** — every stake zeroed,
> `single_book = TRUE`, `clv_eligible = FALSE`, `detail` carries
> `single_book=1` — and `--report` / `--monthly-report` score CLV on
> `clv_eligible` rows only.

A thin-book night therefore degrades VOLUME, visibly, instead of silently
polluting the measurement. The gate binds **at the OPEN only**; POST_REPORT and
PRETIP record `n_books` for telemetry but keep their CLV eligibility, because
narrowing them too would redefine the two comparison views the three-view design
exists to measure.

### WHERE THE OPEN PANEL MUST COME FROM
* **Action Network — CANNOT SERVE.** D174 (2)(c): AN carries **no per-book
  opening price**; `book_id 30` "Open" is a single CONSENSUS opener, and its
  per-book numbers are one scrape-time snapshot which D174 (3) verified is the
  **close**.
* **ESPN — CANNOT SERVE ALONE.** `nbapred/ingest/espn_lines.py` is one book by
  construction. Measured on D174's panel rows, share of games with >= 2 distinct
  operators **at the open**: 2023-24 **94.7%**, 2024-25 **0.0%**, 2025-26
  **3.4%**. An ESPN-only open panel refuses ~96.6% of bookings.
* **THE ODDS API — the only live source that can.** `odds_logger` writes
  `ev["bookmakers"][]` (regions=us) and `load_odds.flatten` emits one row per
  (book, market, outcome).

### >>> THIS IS THE THING MOST LIKELY TO BREAK FIRST <<<
**The odds logger has never run in-season in this repo.** `data/raw/odds/` holds
exactly one offseason day whose `data` is `[]`, and `odds_quotes` is **empty (0
rows)**. If the systemd unit is not up and authenticated on opening night, every
OPEN row will be single-book and **the CLV scoring set will be empty**.

**How you can tell**, in order of speed:
1. `ls -la data/raw/odds/ | tail` — a file for today, growing.
2. `python3 -c "import json;print([json.loads(l)['quota'] for l in open('data/raw/odds/$(date -u +%F).jsonl')][-1])"` — `requests_remaining` falling but > 0.
3. `bet_engine.py --scan-open` prints `>=2-book rule (D178): N shoppable, M single-book`. **M ≫ N is the alarm.**
4. `--report` columns `1book` and `meanK`. `meanK` ≈ 1.00 on the OPEN view means one feed.
5. `--monthly-report` OPEN row: `n_clv` far below `bets`.

**What to do:** `systemctl status odds-logger`; check `ODDS_API_KEY` in `.env`;
check the monthly credit budget has not been exhausted (the logger degrades main
polling to hourly below `ODDS_CREDIT_FLOOR` and disables props first — it never
silently dies). Until it is up, the paper book keeps running and keeps logging;
it just books nothing at the OPEN. That is the designed failure mode.

---

## 4. THE CLV BANDS (D178)

`--monthly-report` prints the band **and its derivation inputs** every run, so
the two can never drift apart again:

```
frame        ML (real moneyline), probability space, union @OPEN, unique games
arm          HONEST (data/capstone_pergame.csv, md5 695d40a…)
seasons      2023-24, 2024-25, 2025-26   n=1386 union bets over 21 months
centre       +0.012280   (MEAN CLV OF THE UNION, not of the same-side universe)
per-bet sd   0.051226
n/month      67          (median union bets per calendar month)
monthly se   0.006258  = 0.051226/sqrt(67)
RED  < -0.000236     GOOD > +0.024797        (centre +- 2 se)
```

Two things to expect and NOT panic about:

* **RED is now tight.** Correctly centred on +0.0123, "2 sigma below" is
  essentially "any negative month". On the 21 in-frame months that flags three —
  **2024-10, 2025-04, 2025-10 — and two of the three are Octobers.** An October
  RED means the month ran below the historical rate, not that the engine is
  broken. October is a small-n month (29 bets in both flagged cases).
* **The trigger is unreachable by construction.** Two consecutive months above a
  +2-sigma line is a ~1,932-month wait. That was ALSO true of D120/D121 — their
  GOOD line only looked reachable because it sat at +1.23 sigma of this frame by
  accident. Re-specifying the TRIGGER (as opposed to the BAND) is a separate
  product decision and needs its own D-line.

Re-derive with `python3 scripts/d178_clvbands.py` (writes
`data/d178_clvbands.json`; the run prints a fidelity anchor that must reproduce
D155/D159's registered `n=1378 / CLV=+0.01590` exactly).

### >>> CLV IS THE MONITOR, NOT THE OBJECTIVE (D176) <<<
D176 measured the two apart, so this is not a hedge. Across three
pre-registered selectors:

* the **availability-divergence** selector bought **MORE CLV** (6/6 cells,
  +0.143 pts) and **LESS ROI** (1/6 cells, −1.16pp);
* the explicitly **CLV-targeted** selector bought essentially **no extra CLV**
  (+0.004) and the **MOST ROI**.

**CLV is not a sufficient statistic for bet selection.** A band tuned purely on
CLV can green-light a selector that is losing money. CLV is kept because it
resolves fast — it is the early-warning instrument on execution and timing.
**A green CLV month is not evidence of profitability.** ROI, measured against
the INCUMBENT, is the objective. `--monthly-report` prints this every run.

### On nulls
Nothing in this band is a net-of-null statistic — centre, sd and se are plain
sample moments of the union CLV — and that is deliberate. D176 found all three
new arms beat their own permutation nulls at p ≤ 0.048 and survived BH, **and
all three still lost to the incumbent.** A permutation null only asks "does this
beat a scrambled copy of itself", which is true of the incumbent too. Beating a
null is NECESSARY, NOT SUFFICIENT: any future re-derivation that leans on one
must report against the INCUMBENT as well.

---

## 5. GAME-TYPE FILTER (D178)

The live path books **regular season only** (`game_id` prefix `002`). Filtering
happens at four places so no caller can smuggle one in:
`slate.todays_games()`, `slate.slate_context()`, `bet_engine.emit()` /
`bet_engine.scan_open()`, and `predict_today.main()`. Each prints what it
dropped — silence means nothing was dropped.

This matters **in the first three weeks of October**: the nba_api scoreboard
serves preseason (`001`) games before the opener, and the odds feeds carry
preseason lines. It matters again in **February** (All-Star, `003`).

Expected log line, and it is not an error:
```
  [OPEN] EXCLUDED non-regular-season game 0012600007 (preseason) — the model is
  fit and priced on 002 only (D178)
```

`b2b` also filters `002` now, so a team whose only prior-day game was preseason
or the NBA Cup final is not flagged for a back-to-back it never played (measured:
1 date x 2 teams across the whole spine, 2025-12-17).

---

## 6. WHAT BREAKS FIRST — RANKED

| # | Failure | First symptom | Check | Blast radius |
|---|---|---|---|---|
| 1 | **Odds logger down / out of credits** | OPEN books 0 rows, `1book` high | §3 | CLV set empty; POST_REPORT/PRETIP still book |
| 2 | **Injury-report feed stalls** | `report_ts=None` in POST_REPORT `detail`; out-sets shrink | `ls data/raw/injury_reports/ \| tail` | Model reverts to a healthy-roster prior; D159: the availability feed is ~2/3 of the CLV asset |
| 3 | **Preseason on the scoreboard** | `EXCLUDED non-regular-season game` lines | §5 | **Handled** — this is the fix working |
| 4 | **stats.nba.com stall** | `pull_nba_daily` timeout in log | `tail data/logs/nba_daily.log` | `--settle` finds no result and RETRIES next day; rows stay unsettled, nothing is lost |
| 5 | **Writer-lock contention** | `writer lock held, retry i/10 in 60s` | log | Self-healing for 10 min; longer means a stuck process |
| 6 | **October bridge has no 001 rows** | Week-1 comp rotation empty, wild p_us | `build_player_stats` ran? | Predictions unusable in week 1 |
| 7 | **Scoreboard unavailable** | `scoreboard unavailable: ...` | log | Whole slate skipped that run; the next 30-min scan retries |

`--settle` is safe to re-run: rows stay unsettled until the result lands, and
`INSERT OR IGNORE` keeps the FIRST emission of a candidate per
(game, side, rule, snapshot_kind), so no rerun ever overwrites.

---

## 7. KILL SWITCHES (env vars) — DEFAULTS AND WHAT THEY DO

Set in the cron environment or `.env`. **All defaults below are the SHIPPED
behaviour**; you should need none of them on a normal night.

| Switch | Default | Set to | Effect |
|---|---|---|---|
| `LATE_STATE` | **`0` (OFF)** | `1` | Re-enables the D90 late-state layer (form5 + nout deltas, gp>=55 gated). DEMOTED under D112's pre-registered rule: it helped measurably more on the seasons it was developed on (DiD +0.00253, SIG) and contributes nothing where no gate could select on it. Code and tests are intact. |
| `TANK_TERM` | **`1` (ON)** | `0` | Sets the D73 tank coefficient `k` to exactly 0. The strict reading of the D112 evidence in one env var; the term ships ON because the held-out point estimate is positive and removing it is worse in all five seasons. |
| `OCT_BRIDGE` | **`1` (ON)** | `0` | Disables the WHOLE D84-A package (October bridge + ps-continuity carry) — the exact pre-D84 dead-zero behaviour, i.e. the F6 same-run control. Only fires when some team's comp rotation is empty; every game with a live rotation is bitwise unchanged either way. |
| `OCT_BRIDGE_TRAIL` | **`"2"`** | `""` | Trailing seasons for the bridge's prior-minutes leg. `2` is D105's declared primary, FROZEN for 2026-27. Empty string reproduces the UNCAPPED legacy construction (which is what D122's certification ran). |
| `COVID_GUARD` | **`0` (OFF)** | `1` | Drops the COVID no-crowd regime from the margin fit frame. Measured NULL, so it ships off — but the registered numbers were produced with it ON, so set it to `1` to reproduce them. |
| `PROPS_MIN_RAMP` | **`1` (ON)** | `0` | Disables the D133 minutes ramp (subtract from `proj_min` by games-played). Identically 0 at gp>=20, so it is a no-op outside its pre-registered window. |
| `PROPS_ABSENCE_RAMP` | **`1` (ON)** | `0` | Disables the D145 absence ramp (keyed on games missed of the team's last 10). Exactly 0 for miss10<=4. |
| `PROPS_KALMAN_FWD` | **unset (OFF)** | `1` | Restores forward-advancing the props Kalman filter to the prediction date. It is the filter's DESIGN but measured OOS harm (CRPS +0.00378 SIG, MAE +0.00461 SIG), and D118's precedent is that the measurement wins. |
| `ORACLE_PLAYED_OUTS` | **`0` (OFF)** | `1` | **LEAKAGE — CEILING RUNS ONLY.** Builds OUT-sets from tonight's box score (the played set). Forbidden on any live or reported path by `docs/LEAKAGE.md:131`; it exists so an oracle ceiling can be priced. If this is ever set in cron, every number the system emits is invalid. |
| `TANK_SEASON_FLOOR` | **`PINNED_SEASON_FLOOR` = `2020-21`** | a season | Overrides the tank corpus floor. PINNED since D155 because a coverage-derived floor moved twice on backfills and silently invalidated the certified table (D131, D153). Moving it is a MODEL CHANGE needing a gate + re-cert, not a bookkeeping edit. |
| `ODDS_MONTHLY_BUDGET` | `0` (no pacing) | credits | Free-tier pacing for the odds logger. |
| `ODDS_CREDIT_FLOOR` | `500` | int | Below this the logger stops props first, then degrades main polling to hourly. |

---

## 8. REHEARSAL

```sh
# all three views + panel + >=2-book rule + all four arms, in a TEMP DB
python3 scripts/bet_engine.py --dry-run 2026-03-27 --dry-books 2
# the honest historical single-book shape: every OPEN row refused
python3 scripts/bet_engine.py --dry-run 2026-03-27 --dry-books 1
```

`--dry-run` NEVER touches `bet_paper`; it writes to a mkdtemp DuckDB and leaves
it for inspection. `--dry-books 2` adds a clearly-labelled SYNTHETIC
spread-implied second book (no historical per-book ML open panel exists — D174
(5)); it is plumbing, never a measurement.

---

## 9. THINGS THAT ARE NOT BROKEN

* `No NBA regular-season games today (...) — offseason no-op.` — correct.
* `EXCLUDED non-regular-season game ...` in October/February — the fix working.
* `shrunk_kelly` staking 0.00 on essentially everything — the D112/D117 result;
  the calibrated close edge does not clear the vig. It deploys capital only via
  a shopped price better than consensus fair.
* An OPEN-view `RED` month in October — see §4.
* `settled N/M open rows` with N < M — the rest have no result ingested yet.
