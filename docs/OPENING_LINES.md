# Opening lines — what we have, where it came from, what is closed

**Status: SOLVED at $0.** We have opening spreads for every season we model.
This file exists so nobody re-litigates the search, and so the *limits* of the
data are on the record next to the win.

Written 2026-08-01 (D-line D118). Supersedes the working assumption in
`docs/PAID_OPTIONS.md` §2 that "nothing after 2022-23" was obtainable free.

---

## Why this mattered

`odds_market` carries exactly **one** price per game — the close. Every
strategy in `docs/TRADING_STRATEGY.md` that claims edge has to be executed at a
price *better* than the close, and with only a closing price that claim is
structurally unbacktestable: you cannot measure what you would have paid.

So the blocking question was never "is our model good" — it was "can we
observe a pre-close price at all".

## What we have now

Free sources, stitched, because none covers the whole span. SBR's hard stop
(2023-01-16) is almost exactly where the ESPN/Action Network feeds begin.

| source | seasons | what it gives | granularity |
|---|---|---|---|
| `sportsbookreviewsonline.com/scoresoddsarchives` | 2007-08 .. 2022-23 | open + close spread, open + close total, closing ML | open/close only, one composite book |
| **ESPN core API + Action Network** (primary for the gap) | 2023-24 .. 2025-26 | open + close **spread, total AND moneyline** | open/close; 12-16 books in 2023-24 |
| `teamrankings.com/nba/matchup/<slug>/spread-movement` | 2023-24 .. 2025-26 | open, last, high, low spread + **full timestamped intraday path** | every line change, up to 3 books |
| Kaggle `chevronronson/nba-stats-dataset` | 2003-04 .. 2025-26 | opening + closing **moneyline** + intraday path | per-change; moneyline only, no spread |

The last two are redundant for open/close but are the **only** sources of an
intraday path — neither ESPN nor Action Network serves line-movement history
retroactively (`/odds/{id}/history/0/movement` returns `count:0` for completed
games). Those endpoints do work going forward, so intraday can be captured
live from now on.

Tables / files:

- `odds_open` (DuckDB) — one row per game, our team keys, both sign
  conventions stored. Built by `scripts/build_odds_open.py`.
- `data/derived/odds_open.csv` — same content; readable while a loader holds
  the DuckDB single-writer lock (no parquet engine is installed).
- `odds_hist_sbr` (DuckDB) — SBR staging table, `nbapred/ingest/sbr_hist.py`.
- `data/raw/sbr/*.xlsx`, `data/raw/sbr_html/*.html` — SBR raw archives.
- `data/raw/sbr_ext/nba_open_close_*.csv` — ESPN + Action Network merged,
  one row per game, 42 cols (per-source raw jsonl kept alongside).
- `data/raw/teamrankings/spread_movement.jsonl` — per-game intraday paths.

### Sign conventions (read this before using the table)

    open_spread / close_spread    betting line ON THE HOME TEAM  (negative = home favored)
    open_margin / close_margin    expected HOME margin           (positive = home favored)
                                  == -spread; matches odds_market.home_exp_margin
    line_move = close_margin - open_margin      (positive = line moved TOWARD home)

Both are stored deliberately. Mixing them up is the obvious way to get a
sign-flipped backtest that looks like alpha.

---

## The measurement: what the open is actually worth

`scripts/measure_line_movement.py`, n = 23,758 games with both prices.

### 1. How far does the line move?

    mean |close - open|   1.218 pts   95% CI (1.202, 1.235)   sd 1.330
    median                1.0 pts     p75 1.5   p90 2.5   p95 3.5   p99 6.0
    P(no move at all)     17.9%
    P(|move| > 1 pt)      34.0%       P(> 2) 14.1%   P(> 3) 6.3%   P(> 5) 1.5%
    signed move           -0.032 pts  (essentially no systematic home/away drift)
    totals |move|         2.141 pts mean, 1.5 median

**Movement has roughly doubled over the sample** — mean |move| 0.87 pts in
2008-09 → 1.86 pts in 2024-25. The premise gets *better* with time, not worse.

### 2. Is the move informative? Yes, significantly.

    MAE vs actual margin    open  9.848    close  9.744
    paired MAE(close) - MAE(open)  = -0.1036   95% CI (-0.1255, -0.0822)
    line moves TOWARD the eventual result:  54.63%   95% CI (53.92%, 55.32%)

The close genuinely beats the open — the movement is not noise. So an early
price *is* a better price when the market later moves your way. And the signal
is **stronger in the seasons we actually model**: move-toward% runs 56.0%
(2021-22), 55.4% (2023-24), **60.1% (2024-25)**, 57.1% (2025-26) against a
2008-13 baseline near 51-53%.

### 3. The ceiling — the number to actually plan against

Betting the side the market later moved toward, graded on the real result:

    at the OPEN price   10498-8719-298   54.63%   CI (53.92, 55.32)
    at the CLOSE price   9596-9632-287   49.91%   CI (49.19, 50.62)
    breakeven at -110                    52.38%

**The entire prize for perfect open-to-close timing is 4.72 pp of ATS win rate
(54.63% vs 49.91%), and it clears the vig by only 2.25 pp.** That is the
ceiling *with perfect ex-ante foresight of which way the line will move* — an
assumption nothing in the model currently supports. Any realistic capture is a
fraction of 2.25 pp.

This bounds the strategy directly. Read against D117 ("after calibrating our
own edge estimates we have no positive-EV bets against the close; CLV is the
only viable target"): CLV is indeed the only target, and this is how big that
target is. It is positive, it is real, and it is small.

### 4. Coverage against the model corpus

| season_end | corpus games | with opening line | % | source |
|---|---|---|---|---|
| 2022 (2021-22) | 1321 | 1317 | 99.7% | sbr |
| 2023 (2022-23) | 1320 | 659 | 49.9% | sbr (site stopped 2023-01-16) |
| 2024 (2023-24) | 1319 | 1319 | 100% | espn+actionnetwork |
| 2025 (2024-25) | 1321 | 1321 | 100% | espn+actionnetwork |
| 2026 (2025-26) | 1322 | 1322 | 100% | espn+actionnetwork |

**5,938 of 6,603 model-corpus games (89.9%)** have an opening line; the only
hole is the second half of 2022-23. Full history (2007-08 onward) is 23,758
games with both prices.

### 5. Cross-source validation

`close_margin` checked against the independently-sourced
`odds_market.home_exp_margin`:

| source | n | mean abs diff | exact |
|---|---|---|---|
| sbr | 19,806 | 0.0000 | 100.0% |
| espn+actionnetwork | 3,960 | 0.2363 | 70.6% (p95 = 1.00) |

SBR reproduces the Kaggle close exactly. ESPN/AN differ by a fraction of a
point — expected, since they are different books, and far too small to be a
sign error (a flip would show ~5+). Sanity: mean `open_margin` +2.734, mean
`close_margin` +2.702, actual mean home margin +2.613.

ESPN and Action Network also cross-validate against each other at corr
0.973 / 0.956 / 0.970 across the three seasons, median diff 0.000.

Opening moneyline vs opening spread agree on which side is favored in
**97.1%** of the 3,952 games that have both, and the disagreement is
concentrated exactly where it should be — 47% at |spread| ≤ 1, 7.5% at 1-2.5,
0.7% at 2.5-5, 0.5% above 5. That is the pick'em pattern, not a sign bug.

---

## Avenues tried, and their verdicts

Kept in full, including the dead ones, so they are not re-worked.

### 1. sportsbookreviewsonline.com — **WORKS, with a hard end date**
The standard free archive. Two rows per game (V then H); the `Open`/`Close`
columns *mix* spread and total — per game the larger cell is the total and the
smaller is the spread, carried on the favorite's row.

- The site **reorganized**: the old `wp-content` xlsx download links are gone
  and each season page now renders the table **inline as HTML**. Earlier
  sessions saved those pages believing they were error pages; they were not —
  they were the data.
- Current index: `/scoresoddsarchives/nba/nbaoddsarchives.htm`. It lists
  `nba-odds-2007-08` .. `nba-odds-2022-23` and **nothing after**.
- `/scoresoddsarchives/nba-odds-2023-24/` (and -2024-25, -2025-26) return
  **HTTP 200 but soft-404 to the homepage** — 67 KB, zero `<tr>`. Do not treat
  the 200 as success.
- 2022-23 is genuinely **truncated at 2023-01-16** (664 games) in *both* the
  xlsx and the HTML. SBR stopped publishing mid-season. Not a fetch bug.
- Full-season xlsx for most seasons are recoverable via the Wayback Machine
  (`fetch_season_xlsx`); 2008-09 and 2009-10 have no xlsx snapshot and fall
  back to the live HTML, which is complete for those two.

### 2. ESPN core API + Action Network — **WORKS, and is the primary gap source**
Both keyless, both public, no rate-limit problems.

- ESPN: `sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{id}/competitions/{id}/odds`.
  Every provider item carries `open{}` / `close{}` / `current{}` blocks for
  `pointSpread`, `spread` (juice) and `moneyLine`, plus top-level open/close
  `total`. 12-16 books per game in 2023-24; ESPN narrowed to ESPN BET in
  2024-25 and DraftKings in 2025-26.
- Action Network: `api.actionnetwork.com/web/v2/scoreboard/nba?date=YYYYMMDD`,
  one request per date. `markets` is keyed by book_id and **book_id 30 = "Open"**
  (book_id 15 = "Consensus"), confirmed against `/web/v1/books`.
- Fetchers: `scripts/fetch_espn_odds_open_close.py`,
  `scripts/fetch_actionnetwork_odds.py`, merged by
  `scripts/build_nba_open_close.py` into `data/raw/sbr_ext/nba_open_close_*.csv`.
- **Two traps, both of which produce plausible-looking garbage:**
  1. ESPN's `open.pointSpread.value` holds the **decimal price** (1.80, 1.87),
     *not* the handicap, in some vintages — all of 2024-25. The handicap is
     only in the display string. Parsing `value` gives corr 0.056 with Action
     Network and a fake 7.34-pt mean move.
  2. ESPN's scoreboard date is **UTC**; Action Network's is **ET**. Joining on
     the raw ESPN date matched 182 of ~1300 games.
- The raw feeds include preseason, All-Star and exhibition games (Melbourne
  United, Team Chuck, `EAST`/`WEST`). `build_odds_open.py` drops them for free
  by joining the unordered team pair against `odds_market`.
- Neither feed has open lines for 2022-23 or earlier (ESPN returns books but
  zero open fields; AN returns no markets), so there is **no season where
  these overlap SBR's Open column** — the cross-validation is ESPN vs Action
  Network only, never three-way.

### 3. teamrankings.com per-game movement pages — **WORKS, but superseded**
Kept because it is the only *intraday* source; superseded by ESPN+AN for
open/close because it carries spread only (no total, no moneyline).
Found by following the `source_url` column inside the Kaggle
`chevronronson/nba-stats-dataset` moneyline dump, which was itself scraped
from teamrankings. The moneyline page has two siblings.

- Correct slug is **`/spread-movement`**. `/point-spread-movement` is a decoy:
  it returns HTTP 200 with a valid-looking page and **no movement data**.
- Gives `Open / High / Low / Last` per side plus a full timestamped history
  table (per book, `--` where a book has no quote).
- Spread history exists for **2023-24 onward only**. Verified absent for
  2022-23 and 2021-22 (page 200s, no Open table) — which is exactly where SBR
  takes over, so the stitch is clean.
- Matchup URLs for all 29,736 games 2003-04..2025-26 come free with the
  Kaggle file; no URL discovery crawl needed.
- Scraper: `scripts/scrape_teamrankings.py` (resumable, ~1 req/s).

### 4. Kaggle cached datasets — **mostly NO, one YES, one gap-filler**
Every cached file under `data/raw/kaggle/` was audited column by column, and
every `dataset.zip` was re-extracted to confirm no file had been missed.

- `ehallmar/nba-historical-stats-and-betting-data` — **NO**. `nba_betting_spread.csv`
  is one row per (game, book) with **no timestamp**; verified all 131,690 rows
  are unique on that key. Cross-book dispersion (median 0.5 pts) is book
  disagreement, not line movement. Same for the ML and totals files.
- `christophertreasure/nba-odds-data` — **NO**. One row per (date, team),
  single value per market.
- `cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024` — **NO**, and it is
  the source behind `odds_market`. Its `id_spread` / `id_total` columns look
  like a second price but decode as **outcome labels** (verified 100%:
  `id_spread`==2 ⇔ push, ==1 ⇔ favorite covered).
- `erichqiu/nba-odds-and-scores` — **YES**, and we already had it.
  `vegas.txt` carries explicit `Open_Line_Spread`, `Open_Line_ML`,
  `Open_Line_OU` alongside per-book closes. 9,188 games, 2012-13..2018-19.
  Entirely *inside* SBR's span, so it is a **cross-validation set, not a
  gap-filler** — use it to check the SBR open, not to extend it.
- `chevronronson/nba-stats-dataset` — **YES for moneyline**. Opening ML for
  2023-24 / 2024-25 / 2025-26 at 99.9% / 99.8% / 97.0% coverage, plus the
  intraday path. **No spread column** — this is why the teamrankings scrape
  was still necessary.
- Rejected after inspection: `caseydurfee/mgm-grand-nba-betting-data`
  (states outright it is closing-line only), `zachht/wnba-odds-history`
  (real spread movement but median window is only **2.75 h before tip** — late
  intraday, not an open), `oliviersportsdata/...` (50-row teaser for a paid
  product), plus ~8 others screened on season range or single-price schema.

Kaggle access note: `nbapred/ingest/kaggle_web.py` needs a logged-in Chrome
cookie and currently fails. But the **public Kaggle API works anonymously** for
search, metadata, file listing, and *single-file* download —
`https://www.kaggle.com/api/v1/datasets/download/{owner}/{slug}?file_name=<path>`
pulled a 95 MB file out of a 10.9 GB dataset with no auth. Worth wiring in.

### 5. The Odds API historical endpoint — **CLOSED at $0**
- "This endpoint is only available on paid usage plans" on all three
  `/v4/historical/*` endpoints. No trial, no sample, no demo key.
- **Trap:** the pricing table appears to grant Historical Odds to the free
  Starter tier in any markdown-converted view. The raw HTML has it inside
  `<s>...</s>` — struck through. It is not included.
- Free quota is 500 credits/mo; historical costs 10× per region per market, so
  one `us` h2h+spreads+totals snapshot = 30 credits ≈ 16 snapshots even if it
  were free. Our key in `.env` is a free-tier key and cannot reach it.
- It also would not answer this question directly: it returns point-in-time
  snapshots you must walk backward, not a labelled opening line.

### 6. Wayback / other archives — **only useful for SBR's own xlsx**
Wayback is how we recover the full-season SBR spreadsheets. It does **not**
help for 2023-24+: SBR never published those pages, so there is nothing
archived to recover. Per-game line-history pages at covers/oddsshark/
vegasinsider are not captured densely enough to reconstruct a season.

---

## Known limits — do not paper over these

1. **One composite book, 2007-08..2022-23.** SBR's open/close is a single
   consensus line. No cross-book dispersion, so no "best available price"
   backtest before 2023-24.
2. **No intraday path before 2023-24.** SBR gives two points. Anything about
   *when* to bet within the day is unanswerable for those seasons.
3. **No opening moneyline from SBR.** Its `ML` column is the **closing**
   moneyline. `open_ml_*` is NULL for all 19,821 SBR rows — a source
   limitation, not a parsing gap. Opening ML exists for 2023-24 onward
   (3,960 games, ESPN/Action Network) and, redundantly, in the Kaggle
   chevronronson file.
4. **2022-23 has an opening line only through 2023-01-16** (664 of 1320
   games). The rest of that season has close only. Any season-level statistic
   on 2022-23 is a first-half statistic.
5. **`odds_open` is not a full-corpus join.** Check `source` and expect NULLs;
   a naive left join will silently look like data where there is absence.
   `build_odds_open.py` now prints a model-corpus coverage table and shouts
   about any season below 99%, so the absence is loud rather than silent.
6. **No three-way validation is possible.** Neither ESPN nor Action Network
   carries open lines for 2022-23 or earlier, so no season has both SBR's
   `Open` column and a feed `open` — the cross-checks are SBR-vs-Kaggle-close
   and ESPN-vs-AN, never all three.

## Fixed along the way

- **Year-inference bug (`nbapred/ingest/sbr_hist.py`).** SBR dates are `MMDD`
  with no year. The old rule (`month >= 9 -> season start year`) misdated the
  entire 2019-20 bubble: **109 games** played Aug–Oct 2020 were stored a full
  year early, and the Sep/Oct 2020 finals collided with the Oct 2019 tip-off in
  the same month bucket. Now the year advances only on the calendar wrap, so a
  season may legitimately run Oct 2019 → Oct 2020. Regression test:
  `tests/test_basics.py::test_sbr_year_inference_survives_the_bubble`.
- **Wayback negative cache.** 2008-09 / 2009-10 have no archived xlsx, and
  re-probing cost ~150 s of rate-limited retries on *every* rebuild. Misses are
  now recorded as `data/raw/sbr/.no-xlsx-<season>`.
- **Fabricated spreads from corrupt source rows.** The spread/total split is
  inferred from magnitude, so a source row where *both* cells are large
  produced a 216-point "spread" and a 1105-point "total" (real example:
  2019-12-09 IND/LAC). Those two rows alone moved the sample max |move| to
  221.5 and inflated the mean. `resolve()` now returns NULL when the resolved
  spread exceeds 40 pts or the total falls outside 120-300. ~20 rows across
  17k are nulled; a NULL is recoverable, a fabricated number is not.
- **Row-at-a-time DuckDB insert.** `load_all` issued 17,712 single-row
  `INSERT`s, each its own transaction — it sat on the single write lock for
  >12 minutes and blocked every reader in the repo. Now one bulk
  `INSERT ... SELECT`.
- **Short abbreviations.** teamrankings and ESPN use `BK/GS/NO/NY/PHO/SA/
  UTAH/WSH` where we use `BKN/GSW/NOP/NYK/PHX/SAS/UTA/WAS`. Unmapped, this
  silently dropped every game involving those teams — a 63% join rate that
  looked like a scrape problem rather than a mapping bug. Now 100%.

## Parse validation

SBR's parsed `close_spread` was checked against the independently-sourced
`odds_market.home_exp_margin` (Kaggle) across all 19,806 joinable games:
**100.0% agree exactly**, mean absolute difference 0.0000. The
spread/total disambiguation
and the favorite-row convention are correct.
