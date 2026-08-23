# Master Decision Register

Every material decision, win, loss, and rejection — chronological, each with its
evidence trail. Journal ids refer to `.priorstates/journal/entries/` (full TL;DR
bodies there); scripts/logs are in-repo. Companion docs: FEATURE_LEDGER.md
(per-feature verdicts + false-rejection review), COMPLEXITY.md (the gate),
LEAKAGE.md (PIT rules), LIMITATIONS.md (caveats).

## Ground rules that govern everything below
- G1 Ablation gate: a feature ships only if OOS improvement's bootstrap 95% CI
  excludes zero (COMPLEXITY.md). Pre-registered before any live line existed.
- G2 Market-blind simulator: odds are benchmark/CLV/blend only, never model
  input (handoff I.6).
- G3 Walk-forward only; PIT as-of joins; trailing windows (LEAKAGE.md).
- G4 Construction-artifact checklist applied to acceptances AND rejections
  (FEATURE_LEDGER.md bottom; grew from real bugs found).

## 2026-07-26 — data layer


# DECISION REGISTER — CURRENT, D170 ..

Entries D1..D169 are in **[DECISIONS_ARCHIVE.md](DECISIONS_ARCHIVE.md)**, split
out on 2026-08-08 because the combined file passed 1.08 MB and GitHub stops
rendering markdown above ~1 MB. The split is byte-exact; nothing was edited.

Frequently cited archive entries: D112 (late-state demotion), D133/D144
(promoted-replacement mechanism), D147 (line-movement model), D157 (PCA /
the penalty-not-the-basis finding).

- D170 THE AVAILABILITY-DATA GAP WAS NOT A SOURCE GAP — **THREE FEEDS
  BACKFILLED ACROSS THE WHOLE 19-SEASON FRAME, AND THE OWNER'S SUSPICION IS
  CONFIRMED WITH A NUMBER: 58.4% OF 2024-25's APPARENT SUPERIORITY WAS DATA
  COMPLETENESS, NOT THE SEASON.** The 19-season pooled normalized gap falls
  **+20.89% -> +9.50%** and D161's "every level is a LOWER BOUND" is now
  quantified rather than asserted. **No default changed, no gate re-run, the
  eval corpus is NOT widened, and D158 remains the certified baseline / D159
  the trading baseline.**
  (0) **WHY THIS EXISTS.** D158 measured that scoring a season availability-
  blind costs it enormously (2021-22 blind 29.52% vs 2024-25 6.30% with full
  feeds). D159 found the availability feed is ~2/3 of the CLV asset. D161 then
  ran ALL 19 seasons BLIND *because the feeds did not exist historically* and
  labelled every level a lower bound. The owner asked the right question: **is
  2024-25 "best" partly because it is the only era with complete data?** That
  was untestable while the feeds were missing. This entry makes it testable.
  (1) **THE HEADLINE FINDING, BEFORE ANY MODEL RUNS: ALL THREE "MISSING" FEEDS
  WERE INGEST GAPS, NOT SOURCE GAPS.** Probed live, not assumed:
  * `game_inactives` — BoxScoreSummaryV2's `InactivePlayers` result set is
    populated **from 2006-07 onward** and returns 0 rows for 2005-06 and
    earlier. 2006-07 IS the source floor; everything above it was ours.
    **0 -> 1230/1230 games on every season 2007-08..2021-22.**
  * `injury_reports_pit` — daily probe of `Injury-Report_YYYY-MM-DD_05PM.pdf`,
    1.15s spacing, control URL verified at both ends: **403 on every day from
    season open 2018-10-16 through 2018-12-16, first 200 on 2018-12-17.**
    That IS a source floor and it is stated as one: **DO NOT CLAIM COVERAGE
    BEFORE 2018-12-17.** From there to 2023-10-23 was ours. **45,369 ->
    125,695 rows, 2 seasons -> 8.**
  * `darko_history` — **darko.app serves the FULL daily series back to
    1996-11-01, for RETIRED players too** (Kobe 1996-11-01..2016-04-13 n=1777;
    Duncan, Iverson, Nash, Dirk, Yao, KG likewise). We had fetched 1,009 of the
    3,934 player_ids in `player_game_stats` — the modern roster universe. **THE
    RAMP WAS OURS.** 354,600 -> **1,103,818 rows**, 837 -> **2,909 players**.
  **METHOD NOTE — the CDN IP-blocks and a 403 is therefore AMBIGUOUS.** A ~250-
  request sweep produced 403s on dates that had returned 200 seconds earlier;
  the block cleared in ~4 minutes. An early sweep "showed" a 2019-01 start that
  was pure throttling. Every miss in this entry is re-probed on a later pass
  behind a known-good control; nothing is declared absent on one look.
  (2) **COVERAGE, BEFORE -> AFTER.**
  `game_inactives` (games covered / total): 2007-08..2021-22 all **0 -> 100%**
  (2011-12 980/990 = 99.0%, 2014-15 1229/1230 = 99.9%; 2006-07 also landed at
  1224/1230); 2022-23..2024-25 unchanged 100%; 2025-26 1227/1230. Rows
  **42,313 -> 145,029.** Mean inactives per team-game, and this is itself a
  measured era signal rather than noise: **0.31-0.48 across 2009-10..2016-17,
  then 3.90-4.51 from 2017-18 on** — the load-management era arriving in the
  official lists (raw density 2.20-2.64 in 2006-11, 1.61-2.09 in 2011-17,
  3.90-4.51 in 2017-26).
  `injury_reports_pit` report-days per season: 2018-19 **110** (partial by
  source, from 2018-12-17), 2019-20 **156**, 2020-21 **139**, 2021-22 **168**,
  2022-23 **169**, 2023-24 168, 2024-25 167, 2025-26 **159 (was ~44)**. All
  absences are genuine and control-verified: All-Star breaks and the COVID
  shutdown (2020-03-12..2020-07-29, exactly 140 days).
  `darko_history` **PIT MINUTE COVERAGE** — the number that actually matters,
  because `CompositionModel` scores an unrated player as league-average:
  2007-08 **3.63 -> 99.95%**, 2010-11 **11.24 -> 99.94%**, 2015-16 36.87 ->
  99.90%, 2020-21 81.21 -> 99.89%, 2021-22 88.75 -> 99.89%; every season
  1996-97..2025-26 now **98.68-99.95%**. 1,025 of 3,934 player_ids 404 on
  darko.app (deep-bench / 10-day); they are worth ~0.1% of minutes.
  (3) **THE DIRECT CORRECTNESS TEST, AS ASKED.** A player listed inactive must
  not appear in `player_game_stats` with minutes for that game. Non-tautological
  (V2 `InactivePlayers` vs BoxScoreTraditionalV3). **145,029 inactive rows, 0
  with minutes>0 — VIOLATION RATE 0.0000%, on every season individually.** The
  stronger form also holds: `viol_any` = 0, i.e. an inactive player does not
  appear in that game's box score at all.
  (4) **THE FORWARD HOLE NOBODY HAD NOTICED.** The league renamed the PDF on
  2026-01-01 (`_05PM` -> `_05_00PM`). `injury_pdf.py`'s filename regex did not
  match, `parse_pdf` raised AttributeError, `load_all()` swallowed it as a parse
  failure — and **97 report-days (2026-01-01..2026-04-12, the whole back half of
  2025-26) sat in `data/raw/injury_reports/` and had never reached the table.**
  D167 §11's "the report feed stops 2025-12-21" was a parser bug, not the NBA.
  (5) **A LANDMINE FOUND BEFORE LOADING, NOT AFTER.** The legacy PDFs use four
  different column layouts. In the 2018-12-17..2019-11-11 layout the columns are
  `... Player Name | Category | Reason | Current Status | Previous Status`, and
  the existing heuristic parser takes the LAST status-like token on the line —
  **i.e. Previous Status** — and dumps Reason text into `team` (measured: 45-56
  distinct "teams" in a 30-team league). Loading those files unfixed would have
  written **systematically wrong statuses into the table the T1/T2 tier is built
  from.** Fixed with a header-POSITION parser for legacy headers only, which
  also handles per-PAGE column drift and two-word team names wrapped onto their
  own line. **The modern path is untouched and verified byte-identical: 40
  modern report-days re-parsed against the rows already in the DB, 28/28 of the
  previously-loaded ones match EXACTLY** (the other 12 were newly downloaded).
  After the fix, 0 bad team strings on every probed legacy file; the
  name-join yield to `nba_players` is **93.0-96.5% in EVERY season**, i.e. the
  legacy rows join as well as the modern ones.
  (6) **A PRE-EXISTING SILENT DROP, REPORTED AND DELIBERATELY NOT FIXED.** The
  PDFs name the Clippers **"LA Clippers"**; `report_out_map()` maps through
  nba_api's `full_name` **"Los Angeles Clippers"**; the lookup returns None and
  the row is discarded. **All 2,514 Clippers OUT rows have never entered a T1 or
  T2 out-set — in the certified seasons too. One team has been scored
  report-blind throughout.** 30 more rows are lost to the heuristic parser's
  upper-case-surname assumption ("da Silva, Tristan" lands in `team`). NOT
  FIXED HERE: `report_out_map` IS the T2 tier definition and changing it changes
  the certified baseline's inputs. **Owner's call.** It makes us weaker, not
  stronger, so it is not leakage — but "T2" has never actually meant T2 for
  1/30th of the league.
  (7) **THE RE-SCORE. FOUR ARMS, EACH CONTRAST CHANGING EXACTLY ONE THING.**
  `darko_history` and `injury_reports_pit` are read INSIDE `nbapred/`
  (composition talent; `tanking.py::_comp_c_shutdown`) so they move the FIT;
  **`game_inactives` is read NOWHERE inside `nbapred/`** (verified by grep) and
  enters only through the caller's out-set construction, so the tier contrast
  cannot move the fit. Ordering follows from that.
  A = blind, nothing backfilled (**= D161**); B1 = blind + DARKO; B2 = blind +
  DARKO + reports; C = **T2** (5PM report OUT-set UNION official pregame
  inactives, roster window, EMPTY where no feed) + everything.
  **HARNESS VALIDATION: arm A reproduces D161's `k19_model.json` ll_us to 5
  decimals on 18 of 19 seasons (delta exactly 0.00000).** The single exception
  is 2025-26 (0.58929 -> 0.58946, +0.14pp normalized) and its cause is known and
  benign: the §4 reload gave 2025-26 its missing report-days, which feed the
  tank shutdown component. Arm C's 2024-25 cell lands at **+6.22%** against
  D158's independently-certified full-feed **6.30%** — a second, unforced
  corroboration.
  **PER-SEASON NORMALIZED GAP (lower is better; tier label is arm C's):**
```
  season   tier(C)          A(=D161)      B1      B2       C |   B1-A    C-B2     C-A
  2007-08  T2i               +26.87   +7.18   +7.18   +6.54 | -19.69   -0.64  -20.33
  2008-09  T2i               +17.85   -2.44   -2.44   -2.01 | -20.29   +0.43  -19.86
  2009-10  T2i               +25.69   +4.24   +4.24   +3.78 | -21.45   -0.46  -21.91
  2010-11  T2i               +23.61   +4.22   +4.22   +1.58 | -19.39   -2.64  -22.03
  2011-12  T2i-partial       +27.61   +9.74   +9.74   +8.55 | -17.87   -1.19  -19.06
  2012-13  T2i               +16.67   +7.33   +7.33   +7.06 |  -9.34   -0.27   -9.61
  2013-14  T2i               +23.90  +11.67  +11.67   +9.82 | -12.23   -1.85  -14.08
  2014-15  T2i-partial       +20.16  +11.18  +11.18   +9.80 |  -8.98   -1.38  -10.36
  2015-16  T2i               +13.05   +5.78   +5.78   +5.07 |  -7.27   -0.71   -7.98
  2016-17  T2i                +9.71   +9.15   +9.15   +7.91 |  -0.56   -1.24   -1.80
  2017-18  T2i               +27.48  +25.58  +25.58  +22.17 |  -1.90   -3.41   -5.31
  2018-19  T2                +22.60  +17.27  +17.27  +14.98 |  -5.33   -2.29   -7.62
  2019-20  T2                +12.12  +11.75  +11.75   +6.26 |  -0.37   -5.49   -5.86
  2020-21  T2                +36.76  +34.89  +35.14  +26.85 |  -1.87   -8.29   -9.91
  2021-22  T2                +29.52  +27.42  +27.10  +16.79 |  -2.10  -10.31  -12.73
  2022-23  T2                +22.34  +23.37  +23.69  +13.21 |  +1.03  -10.48   -9.13
  2023-24  T2                +20.16  +20.64  +20.20  +16.35 |  +0.48   -3.85   -3.81
  2024-25  T2                +11.72  +11.80  +11.59   +6.22 |  +0.08   -5.37   -5.50
  2025-26  T2                +15.01  +14.92  +15.12  +12.21 |  -0.09   -2.91   -2.80
  POOLED (22,804 games)      +20.89  +12.39  +12.37   +9.50
```
  **TIER DISCIPLINE (D158's rule), STATED PER SEASON AND NEVER POOLED SILENTLY:
  2007-08..2017-18 = T2i (inactives only; no report feed exists), 2011-12 and
  2014-15 = T2i-partial (one/ten games short), 2018-19..2025-26 = T2.** Mean OUT
  players per team-game at C: 0.31-0.48 on 2009-10..2016-17, 0.70-0.87 on
  2017-18..2019-20, 1.09-1.38 on 2020-21..2025-26.
  (8) **THE ANSWER THE OWNER ASKED FOR, DIRECTLY.** Pre-registered metric:
  `ADV(s) = mean(norm gap of the other 18) - norm gap(s)`.
  **2024-25: ADV +10.01pp and RANK 2/19 in D161's arm A. Under B2 — blind
  still, but with the DARKO ramp removed — ADV collapses to +1.96pp and it
  falls to RANK 10/19, i.e. MERELY AVERAGE. At the honest T2 tier it recovers
  to ADV +4.16pp, RANK 5/19.**
  **SHARE OF 2024-25's APPARENT SUPERIORITY THAT WAS DATA COMPLETENESS: 58.4%.
  RESIDUAL GENUINELY THE SEASON: 41.6% (+4.16pp of the original +10.01pp).**
  Decomposed: **the DARKO ramp alone accounts for 82.5% of it**; the report
  backfill -2.1%; and the availability tier **-22.0%, i.e. the tier GIVES
  2024-25 ADVANTAGE BACK.** That last sign is the honest complication and it is
  not buried: the modern era has far more absence to know about (1.32 OUT
  players per team-game against 0.31-0.48 in 2012-2016), so an honest
  availability feed helps a modern season more. **Two readings, both defensible,
  and the register should carry both: (i) 2024-25's edge is 58% data artefact
  and what remains is mostly "the feed is worth more when there is more
  absence", which is a property of the ERA and arguably still not the season's
  own merit; (ii) more conservatively, once every season is scored at the best
  tier it can reach, 2024-25 is the 5th-best of 19 rather than the 2nd, and 8 of
  the 11 seasons now ahead of or near it are seasons D161 ranked at the bottom.**
  Either way the strong form of the D161 reading — "2024-25 is exceptional" —
  **does not survive.**
  (9) **THE MECHANISM IS MEASURED, NOT INFERRED, AND THE DESIGN GOT A PLACEBO IT
  DID NOT HAVE TO BE GIVEN.** corr(DARKO minute coverage GAINED, change in gap
  B1-A) = **-0.886**; corr(mean OUTs per team-game, change in gap C-B2) =
  **-0.809**. 16/19 seasons improve on the DARKO arm, 18/19 on the tier arm.
  **THE PLACEBO: the three seasons that ALREADY had ~100% DARKO coverage
  (2023-24, 2024-25, 2025-26) move by +0.48 / +0.08 / -0.09pp on B1-A — i.e.
  not at all — while the seasons with the largest coverage gain fall by up to
  22pp.** A confound that moved everything equally would have been an artefact;
  this one moves exactly what it should.
  (10) **WHAT THIS DOES TO THE REGISTER's HISTORICAL READINGS.** D153's
  "about HALF the oldest-season deficit is our own starved talent feed, not the
  era" was directionally right and QUANTITATIVELY TOO MODEST: it is **~73% on
  2007-08 and ~82% on 2010-11**, and on 2008-09 the model at T2 posts
  **-2.01%, i.e. it BEATS the market on a season D161 scored at +17.85%.** The
  four D153 correlations in ERAS.md §2b.1 were computed against a coverage
  variable that is now CONSTANT at ~99.9% and can no longer be recomputed; they
  are marked as historical. **The `darko_history` floor of 2003-10-29, recorded
  in ERAS.md §7 as "THE BINDING FLOOR FOR THE MODEL", is gone — 1996-97..2003-04
  are now FITTABLE.** They remain UNSCORABLE (`odds_market` starts 2007-08), so
  **the scorable frame is unchanged at 19 seasons and this entry does not widen
  the eval corpus.**
  (11) **ERA STATEMENT (GATE_POLICY_V2 §10).** Eval universe: 2007-08..2025-26,
  19 seasons, 22,804 games, unchanged from D161. ERA-AVAILABILITY: **this is the
  row that changed.** `game_inactives` now covers E-3 through E6; the only
  genuinely absent channel on the old eras is the injury report before
  2018-12-17, which is a SOURCE floor. AVAILABILITY TIER: **no longer constant
  across the frame — T2i on 2007-08..2017-18, T2 on 2018-19..2025-26 — and it is
  labelled per season at every appearance.** NO PLAYED-SET ORACLE is constructed
  anywhere in `scripts/k19_t2.py`; `player_game_stats` is never used to build an
  out set. COVID FRAME: 2019-20 and 2020-21 scored in every arm; 2020-21 remains
  the worst season in the frame at every tier (+36.76 -> +26.85). CLUSTERING:
  season, K=19; per-season paired SE of the model-minus-market per-game log loss
  is 0.0047-0.0065, so single-season moves under ~1.3pp normalized are inside
  noise and the small C-B2 deltas on the oldest seasons are NOT individually
  significant — the 8-10pp moves on 2020-21/2021-22/2022-23 and the 17-21pp
  DARKO moves are far outside it.
  (12) **WHAT THIS ENTRY DOES NOT CLAIM.** It does not claim the model is good —
  the pooled gap is still **+9.50%** against the market and the market still
  wins every season but 2008-09. It does not re-certify anything: **D158 remains
  the certified baseline, D159 the trading baseline, and no gate was re-run, no
  default flipped, no eval corpus widened.** It does not re-run D161's betting
  rules, D162's ATS track, D166's equity path or D168's structural ladder on the
  new feeds — every ROI/CLV number in the register is still computed on the old
  data and is now KNOWN to be conservative by an unmeasured amount. It does not
  fix the §6 Clippers defect. It does not ingest the `01PM` report edition,
  which the probe shows exists back to 2018-12-18 and which D167 §11 wrongly
  recorded as absent — deliberately, because `report_out_map` has no edition
  predicate and loading a second edition would silently union two different
  information sets into T2. **AND IT DOES NOT DECIDE THE OBVIOUS FOLLOW-UP: the
  certified baseline was fitted and certified on a DB whose historical
  availability and talent feeds were a small fraction of what is now present.
  Whether D158 should be re-certified on the backfilled DB is the owner's call,
  and this entry stops at saying it plainly.**
  [code scripts/bf_inactives_hist_fetch.py, scripts/bf_inactives_hist_load.py,
   scripts/bf_darko_hist_fetch.py, scripts/bf_darko_hist_load.py,
   scripts/bf_injury_pdf_fetch.py, scripts/bf_injury_load.py,
   scripts/bf_avail_verify.py, scripts/k19_t2.py, scripts/k19_t2_compare.py
   (all new); nbapred/ingest/injury_pdf.py MODIFIED (filename regex + legacy
   columnar parser; modern path verified byte-identical);
   scripts/prod_by_season.py IMPORTED UNMODIFIED (report_out_map, so the T2
   definition cannot drift), scripts/k19_model.py UNMODIFIED and NOT RE-RUN;
   data/avail_backfill_notes.md (full working, checkpointed as the run
   proceeded); data/k19_blindA.json + _pergame.csv (arm A),
   data/k19_blindB1.*, data/k19_blindB2.*, data/k19_t2.* (22,804 games each);
   data/logs/bf_inact_hist_{0,1,2}.log, data/logs/bf_darko_hist_*.log,
   data/logs/bf_darko_r2_*.log, data/logs/bf_injury_pdf.log,
   data/logs/bf_pdf_probe5.log, data/logs/k19_{blindA,blindB1,blindB2,t2}.log;
   raw ground truth data/raw/nba_api/boxscoresummaryv2/ (4,914 -> 24,040),
   data/raw/darko_history/ (1,009 -> 3,934),
   data/raw/injury_reports/ (500 -> 1,259 PDFs);
   DB WRITES (batched INSERT..SELECT off registered DataFrames, all network and
   parsing OUTSIDE the lock, lock-retry yielding 60s to the cron pullers):
   game_inactives 42,313 -> 145,029 (writes 0.46s/0.32s/0.58s),
   darko_history 354,600 -> 1,103,818 (11.00s),
   injury_reports_pit 45,369 -> 125,695 (1.48s/2.17s);
   nbapred/model/ UNTOUCHED, nbapred/engine/ UNTOUCHED,
   data/capstone_pergame.csv NOT WRITTEN, no gate re-run, no default changed]

- D171 **THE CLIPPERS DEFECT IS FIXED AND THE BASELINE IS RE-CERTIFIED ON THE
  BACKFILLED DB: POOLED 14.95% -> 12.87%. SUPERSEDES D158 AS THE CERTIFIED
  BASELINE**, exactly as D158 superseded D132 and D132 superseded D122. The
  number got BETTER by -2.08pp, and almost all of that is one season
  (2021-22, which D158 had to run availability-BLIND and which now runs full
  T2). D159 remains the trading baseline. **Every ROI/CLV number in the
  register is still computed on the pre-D170 data and is still a lower bound —
  this entry does NOT re-run any betting study, and §9 ranks them.**
  (0a) **CONTROL-HASH FIELD (D134 rule).** New certified artifact
  `data/capstone_pergame.csv` (md5 **695d40a3545e889267cad403b7acdce8**) vs the
  superseded one, preserved byte-identical as `data/capstone_pergame_D158.csv`
  (md5 **3b7bbbb78ac73c63273c18a8aa30013c**, copied aside and hash-verified
  BEFORE the run): **6,148/6,148 games matched, 0 new-only, 0 old-only; p_us
  moved on 3,647 games (59.32%), max|dp| 0.154922, mean|dp| 0.011063; p_mkt
  max|dp| 0.0e+00** (the market column is untouched, as it must be). Mean OUT
  players per team-game **0.937 -> 1.261 (+0.323)**. Per season, p_us moved:
  2021-22 **99.6%** (max|dp| .1549), 2022-23 84.6%, 2023-24 37.6%, 2024-25
  34.2%, 2025-26 40.6%. **THE MOVEMENT IS LARGE AND THE REASON IS NOT SUBTLE:
  under D158, 2021-22 had NO availability feed in existence (outs/team-game
  0.000) and ran BLIND; D170 gave it inactives AND reports, so it now runs full
  T2 on every game.** Env: LATE_STATE, TANK_TERM, ORACLE_MINUTES, INACTIVE_OUTS,
  REPORT_OUTS, ORACLE_PLAYED_OUTS, OCT_BRIDGE, OCT_BRIDGE_TRAIL, COVID_GUARD,
  FF_LUCK all UNSET; **TANK_SEASON_FLOOR=2020-21 pinned explicitly (D155)**.
  The run PRINTED ITS OWN TIER (`AVAILABILITY TIER: T2-HONEST`) and every season
  resolved to `full T2-HONEST`; **ORACLE_PLAYED_OUTS was unset and the oracle
  redirect never fired**. COMMAND: `env -u ORACLE_PLAYED_OUTS -u ORACLE_MINUTES
  -u REPORT_OUTS -u INACTIVE_OUTS TANK_SEASON_FLOOR=2020-21
  PROD_JSON_OUT=data/prod_by_season_D171.json python scripts/prod_by_season.py`.
  (1) **THE DEFECT D170 §6 REPORTED AND LEFT: FIXED, AND IT WAS IN FIVE PLACES,
  NOT ONE.** The injury PDFs spell the Clippers **"LA Clippers"**; nba_api's
  `full_name` is **"Los Angeles Clippers"**. Five consumers each built
  `{full_name: abbreviation}` inline and wrote `if ab:` — so an unresolvable
  name was discarded **without a word**. The INGEST side had already
  special-cased the alias (`nbapred/ingest/injury_pdf.py`), so the table was
  always right and only the readers were wrong:
  1. `scripts/prod_by_season.py::report_out_map` — **the T1/T2 tier definition
     itself**, i.e. the certification input;
  2. `nbapred/model/tanking.py::_comp_c_shutdown` — **INSIDE the fit**: the
     Clippers' rest/management shutdown signal has been structurally blank in
     every fit this project has ever run;
  3. `nbapred/engine/slate.py` — **the LIVE path.** Unlike D158's defect, this
     one DID reach production: tonight's Clippers games were predicted with an
     empty injury-report out-set. Direction is unfavourable to us, so it is not
     leakage, but it was a live correctness bug;
  4. `scripts/bp_ladder.py` (a verbatim copy) — fixed, output NOT re-run;
  5. `scripts/apr_program.py`.
  **THE FIX IS A CANONICAL MODULE, NOT A SPECIAL CASE:** new `nbapred/teams.py`
  (`abbrev_for` / `team_id_for` / `known_report_names` / `resolve_map`),
  resolving exact full_name -> explicit alias -> unique-nickname suffix, and
  **REPORTING any unresolved string with its row count instead of dropping it.**
  That last property is the point: this is the THIRD instance of this bug class
  in the register (D119's "63% scrape failure" that was a mapping bug; D161's
  938 games lost to era abbreviations), and all three were silent.
  (2) **THE 30-TEAM AUDIT, BOTH DIRECTIONS, AS ASKED**
  (`scripts/d171_team_audit.py`, read-only). `injury_reports_pit` emits **31**
  distinct `team` strings. 29 matched exactly. Two did not:
```
  DROP  LA Clippers          rows=2514  OUT=2119  OUT@sameday=1919  2018-12-17..2026-04-12
  DROP  da Silva, Tristan    rows=  30  OUT=  16  OUT@sameday=  10  (parser artefact, NOT a team)
```
  **REVERSE DIRECTION: of the 30 nba_api `full_name`s, exactly ONE never appears
  in any PDF — `Los Angeles Clippers`. No other franchise has a mismatch in
  either direction.** After the fix: **30/31 strings resolve; 0 of 30 franchises
  unrepresented**; the remaining `da Silva, Tristan` (10 same-day OUT rows,
  0.015% of slots — D170 §6's upper-case-surname parser assumption) is now
  PRINTED by every run rather than swallowed, and is not fixed here because
  fixing it means re-parsing PDFs, i.e. a DB write.
  Also surfaced and dismissed with a reason: `nba_games` carries era abbrevs
  CHH/NJN/NOH/NOK/SEA/VAN absent from nba_api, but they cannot touch
  `report_out_map` (the report feed starts 2018-12-17, by which date every
  franchise uses its modern abbrev) and k19 already carries D161's crosswalk.
  (3) **CONFINEMENT, VERIFIED RATHER THAN ASSERTED** (`d171_fix_verify.py`).
  Diffing the old and new out-maps: **LAC is the ONLY team whose out-set
  changed.** +1,718 player-slots, 571 new (date,team) cells, **zero cells
  lost**; total OUT slots 64,580 -> 66,298 (+2.66%).
  (4) **THE NEW CERTIFIED TABLE** (norm = (ll_us − ll_mkt)/(ln2 − ll_mkt)):
```
  season    n     ll_us    ll_mkt      raw    norm   | D158 ll_us  D158 norm   delta
  2021-22  1228  0.61935  0.60429  +0.01506  16.95%  |   0.63053     29.52%  -12.57pp
  2022-23  1230  0.63346  0.62437  +0.00909  13.21%  |   0.63385     13.78%   -0.57pp
  2023-24  1230  0.59921  0.58086  +0.01835  16.34%  |   0.59906     16.21%   +0.13pp
  2024-25  1230  0.58872  0.58155  +0.00717   6.43%  |   0.58857      6.30%   +0.13pp
  2025-26  1230  0.58631  0.57114  +0.01517  12.43%  |   0.58553     11.79%   +0.64pp
  POOLED   6148  0.60541  0.59244  +0.01297  12.87%  |   0.60750     14.95%   -2.08pp
```
  D158's registered numbers were **RE-DERIVED from the preserved artifact and
  match the register to five decimals on all six rows** — the comparison
  baseline is verified, not quoted from memory.
  (5) **DECOMPOSITION: HOW MUCH IS D170's BACKFILL, HOW MUCH IS THE FIX.** An
  extra arm ran the identical command with the fix reverted
  (`data/capstone_pergame_D171_noclipfix.csv`):
```
  season     D158   backfill  +clipfix |  backfill  clipfix   total
  2021-22  29.52%    16.79%    16.95% |   -12.73    +0.16    -12.57
  2022-23  13.78%    13.21%    13.21% |    -0.57    +0.00     -0.57
  2023-24  16.21%    16.35%    16.34% |    +0.14    -0.01     +0.13
  2024-25   6.30%     6.22%     6.43% |    -0.08    +0.21     +0.13
  2025-26  11.79%    12.21%    12.43% |    +0.42    +0.22     +0.64
  POOLED   14.95%    12.75%    12.87% |    -2.20    +0.12     -2.08
```
  **THE CLIPPERS FIX COSTS +0.12pp — IT MAKES THE MODEL SLIGHTLY WORSE. That is
  the sign D170 §6 predicted, and it is the sign that confirms this was never
  leakage.** It ships because it is CORRECT, not because it helps. Fix-only
  per-game: 2,077/6,148 moved (33.78%), max|dp| 0.082886; **410 LAC games
  (148 moved) but ALSO 1,929 non-LAC games**, because `_comp_c_shutdown` feeds a
  GLOBAL tank-k fit — k(2026-04-09) at the old floor **-2.17831 -> -2.08251**.
  (6) **THE 19-SEASON RE-EVAL, EVERY SEASON LABELLED WITH THE TIER IT CAN
  ACTUALLY REACH (D158's rule, never silently mixed).** `k19_t2.py --tier t2`,
  plus a NEW `--tier t2i` arm (inactives only) added so the 5PM report can be
  PRICED rather than assumed:
```
  season   tier          n     ll_us    ll_mkt     norm | D170 C    d  |   T2i   report worth  outs/tm
  2007-08  T2i         1230  0.57737  0.56927   +6.54% |  +6.54  +0.00 |     -        -        0.42
  2008-09  T2i         1230  0.56313  0.56568   -2.01% |  -2.01  +0.00 |     -        -        0.46
  2009-10  T2i         1230  0.58322  0.57889   +3.78% |  +3.78  +0.00 |     -        -        0.39
  2010-11  T2i         1230  0.57824  0.57640   +1.58% |  +1.58  +0.00 |     -        -        0.48
  2011-12  T2i-partial  990  0.59635  0.58731   +8.55% |  +8.55  +0.00 |     -        -        0.39
  2012-13  T2i         1228  0.58949  0.58161   +7.06% |  +7.06  +0.00 |     -        -        0.32
  2013-14  T2i         1230  0.59708  0.58662   +9.82% |  +9.82  +0.00 |     -        -        0.31
  2014-15  T2i-partial 1230  0.58587  0.57422   +9.80% |  +9.80  +0.00 |     -        -        0.41
  2015-16  T2i         1230  0.57897  0.57288   +5.07% |  +5.07  +0.00 |     -        -        0.38
  2016-17  T2i         1230  0.61814  0.61169   +7.91% |  +7.91  +0.00 |     -        -        0.44
  2017-18  T2i         1230  0.61585  0.59383  +22.17% | +22.17  +0.00 |     -        -        0.70
  2018-19  T2          1230  0.60939  0.59464  +14.98% | +14.98  +0.00 | +15.45%  -0.47pp     0.83
  2019-20  T2          1058  0.61670  0.61173   +6.10% |  +6.26  -0.16 |  +6.99%  -0.89pp     0.87
  2020-21  T2          1080  0.63823  0.61794  +26.98% | +26.85  +0.13 | +28.13%  -1.15pp     1.13
  2021-22  T2          1228  0.61935  0.60429  +16.95% | +16.79  +0.16 | +18.35%  -1.40pp     1.38
  2022-23  T2          1230  0.63346  0.62437  +13.21% | +13.21  +0.00 | +14.76%  -1.55pp     1.19
  2023-24  T2          1230  0.59921  0.58086  +16.34% | +16.35  -0.01 | +16.97%  -0.63pp     1.09
  2024-25  T2          1230  0.58872  0.58155   +6.43% |  +6.22  +0.21 |  +7.04%  -0.61pp     1.32
  2025-26  T2          1230  0.58631  0.57114  +12.43% | +12.21  +0.22 | +12.36%  +0.07pp     1.33
  POOLED (tier labelled per season)  22804  0.59832  0.58833  +9.53% | +9.50  +0.03
```
  K=19 season-cluster-mean t on the raw pooled gap **[+0.00695, +0.01307] SIG**.
  **The 11 pre-2018-19 seasons are bit-identical to D170's C arm, as they must
  be** — no report feed exists there and TANK_SEASON_FLOOR=2020-21 keeps the
  Clippers-sensitive shutdown component out of them. Only report-covered seasons
  move, which is itself an independent confinement check on the fix.
  (7) **THE OWNER'S QUESTION, ANSWERED DIRECTLY: "now that we have all the DARKO
  information, is there any other gap from the rest of the seasons to 2024-25?"**
  **THE DECISIVE FACT IS WHICH TABLES THE MODEL OPENS.** Grepped over the shipped
  path (`fit_production` + `CompositionModel` + four_factors / travel /
  october_bridge / latestate / tanking): the certified model reads **exactly
  SEVEN tables** — `nba_games`, `player_game_stats`, `nba_players`,
  `darko_history`, `darko_dpm`, `injury_reports_pit`, `odds_market` — plus
  `game_inactives`, which is read nowhere inside `nbapred/` and enters only
  through the caller's out-set. Per-season census in
  `scripts/d171_gap_census.py` and docs/STAT_INVENTORY.md. Enumerated:
  **(a) `game_inactives` — CLOSED.** 99.0-100% on all 19 seasons.
  **(b) `darko_history` — CLOSED.** PIT minute coverage **100.0% on every one of
  the 19 seasons**. There is no DARKO ceiling and no residual DARKO gap.
  **(c) THE INJURY REPORT BEFORE 2018-12-17 — PERMANENT, AND THE ONLY REMAINING
  MODEL-VISIBLE ASYMMETRY.** Priced directly rather than extrapolated from
  D170's C−B2 contrast (which confounds the tier with the era's absence
  density): paired per-game T2 vs T2i on **n=9,516** modern games, mean paired
  dLL **-0.000707**, season-clustered **-0.000709 ± 0.000141, t = -5.02 (K=8),
  7/8 seasons improve** = **-0.741pp normalized**. Whether it would be worth
  that much to a 2012-16 season is **NOT estimable at K=8** (corr with absence
  density +0.36 / -0.11, both noise), so it is given as a RANGE and not a point:
  the report adds only **+0.08..+0.20 OUT/team-game** on top of inactives at
  modern density, and 2012-16 carries **0.39** total OUT/team-game against
  **1.14** modern (ratio 0.34), giving **UPPER bound -0.74pp (no density
  scaling), LOWER bound -0.25pp (worth scales with the information added)**.
  Either end is small beside the 17-21pp the DARKO backfill moved those seasons.
  **(d) THE ERA PROPERTY — REAL DIFFERENCE, NOT A DATA GAP.** Three independent
  reasons (`scripts/d171_era_density.py`, full table in docs/ERAS.md): (i)
  players listed inactive who nevertheless logged minutes = **0 on every one of
  the 19 seasons**, so the official list is complete and correct in BOTH eras;
  (ii) the RAW official density roughly **DOUBLES at exactly 2017-18** (3.6-4.2
  -> 7.8-9.0), the season the NBA instituted its injury-report policy — that is
  the league's list, not ours; (iii) rotation depth is **FLAT** (10.0-10.6 vs
  10.5-10.8 players used per team-game), so it is not a roster-size artefact.
  The one real change in our own filter points the same way: `kept%` (share of
  the inactive list surviving the 12-day roster window) rises **8.1-10.5% ->
  12-18%**, i.e. modern absences are concentrated among players IN the active
  window — the load-management signature exactly. **VERDICT: the feed is worth
  more now because there is more absence to know about. That is the ERA.**
  **(e) EVERY OTHER COVERAGE ASYMMETRY IS INVISIBLE TO THE CERTIFIED MODEL.**
  `possessions_v2` (PBP/zone) 0% before 2018-19 and only **62.1% on 2024-25**;
  `lineup_stints` identical; `game_officials` 0% before 2022-23; tracking /
  `defended_fg` sparse throughout; `schedule_features` **2025-26 ONLY** (D135
  confirmed — 0 rows on all 18 other seasons); `odds_hist_sbr` stops after
  2022-23. **None of these is read by the shipped model**, so none contributes
  to any per-season gap today. They bound FUTURE work, not this number.
  `odds_market` itself is complete on all 19 seasons (~1,310/season), so the
  BENCHMARK has no era gap; odds panel DEPTH does (D163: 2 books early vs up to
  9 operators modern), but that is an EXECUTION-study asymmetry.
  **(f) IS 2024-25 STILL SPECIAL? ADV(s) = mean(norm gap of the other 18) −
  norm gap(s), D170's pre-registered metric:**
```
  arm                                        2024-25 gap     ADV    rank   best season
  D161 arm A (blind, pre-backfill DARKO)          11.72%  +10.01    2/19   2016-17
  D170 B2 (blind, full DARKO + reports)           11.59%   +1.96   10/19   2008-09
  D170 C  (T2, pre-Clippers-fix)                   6.22%   +4.16    5/19   2008-09
  D171    (T2, Clippers-FIXED)                     6.43%   +3.97    6/19   2008-09
```
  **FINAL RESIDUAL: ADV +3.97pp, RANK 6/19. 60.3% of 2024-25's apparent
  superiority was data completeness; 39.7% survives** (D170 said 58.4%/41.6%;
  the Clippers fix moves it 1.9pp further toward "data"). **The best season in
  the whole 19 is 2008-09, where the model BEATS the market at -2.01%.** The
  strong form of D161's reading — "2024-25 is exceptional" — does not survive;
  the conservative reading stands: once every season is scored at the best tier
  it can reach, 2024-25 is the 6th-best of 19, and most of what remains is the
  era property in (d) rather than the season's own merit.
  (8) **CHARTS REGENERATED (9), each rendered and inspected, titles naming the
  new certification and tier**: `logloss_by_season_normalized.png`,
  `progress_by_ship.png`, `logloss_continuous_current.png` (make_charts_cert);
  `status_logloss_h2h.png`, `status_trading_h2h.png` (make_status_charts);
  `history_normalized_gap.png`, `history_logloss_by_season.png`
  (make_history_charts via the new `d171_history_charts.py`);
  `k19_model_and_rules.png` (k19_chart with `K19_STATS=`);
  `walkforward_equity.png` (re-rendered for its staleness label ONLY — the study
  behind it was NOT re-run). **FOUR REAL LABEL-COLLISION FIXES, found by looking
  at the renders, not by assuming**: (i) the cert suptitle overflowed the figure
  at BOTH ends once "CERTIFIED D171" was added — split to two lines; (ii) the
  status suptitle could not go to two lines at all (the barh panel's own
  multi-line title collides) — kept to one and raised to y=1.02; (iii)
  `chart_normgap` hard-coded `ylim(0, ...)`, which **silently clipped 2008-09's
  -2.01%, the single most important point on the chart** — now floors below the
  minimum and draws a "we BEAT the market below this line" rule; the same
  latent bug in `k19_chart`'s `xlim(0, ...)` is fixed the same way, and its
  negative bar label is parked right of zero because to the left it collided
  with the season tick; (iv) single-season era bands (E3/E4/E5) are narrower
  than their own labels — now staggered vertically instead of running together.
  (9) **WHAT THIS ENTRY DELIBERATELY DID NOT RE-RUN, AND THE RANKED LIST.** D170
  flagged that every ROI/CLV number in the register is computed on the OLD thin
  data and is therefore **conservative by an unmeasured amount**; that is still
  true and this entry does not change it. **NONE of D161/D162/D166/D168/D169 was
  re-run.** Ranked by (expected movement x load-bearingness), as follow-up the
  owner must decide on separately:
  **1. D162's ATS/19-season betting track** (`ats19_score.py`) — the model's
     probabilities moved on 59.3% of certified games and on every report-covered
     season; every downstream rule inherits it. Biggest expected move.
  **2. D161's frozen-rules ladder across 19 seasons** (`k19_rules.py`,
     `bp_ladder.py`) — its out-sets were blind AND Clippers-blind; the RULES
     panels of `k19_model_and_rules.png` are still its old vintage and are
     labelled STALE on the chart.
  **3. D166/D168's walk-forward equity + structural ladder** (`wf_equity.py`,
     `sl_components.py`, `oc_capacity.py`) — the +3.54% headline and the whole
     6-rung ablation are computed on pre-backfill probabilities.
  **4. D159's trading baseline / CLV decomposition** — D159 attributed ~2/3 of
     the CLV asset to the availability feed; that feed is now materially richer.
  **5. D169's era-local selection** (`el_eralocal.py`) — a null result, so the
     re-run is confirmatory rather than corrective, but its era arms are exactly
     where the backfill bites hardest.
  **6. D163's line-shopping / execution panels** — least affected (they price
     the MARKET, not us), re-run last.
  (10) **ERA STATEMENT (GATE_POLICY_V2 §10).** Eval universe unchanged: certified
  corpus 2021-22..2025-26, 6,148 games; 19-season frame 2007-08..2025-26, 22,804
  games. **THE CORPUS IS NOT WIDENED BY THIS ENTRY.** AVAILABILITY TIER:
  T2-HONEST on all five certified seasons (a change from D158, where 2021-22 was
  BLIND and 2022-23 was inactives-only) and era-varying on the 19 — **T2i
  2007-08..2017-18, T2 2018-19..2025-26, labelled at every appearance**. NO
  PLAYED-SET ORACLE anywhere: `ORACLE_PLAYED_OUTS` unset, the run printed its
  tier, and `player_game_stats` is never used to build an out set. COVID FRAME:
  2019-20 and 2020-21 scored and labelled as separate strata, never pooled.
  CLUSTERING: season, K=19 (K=8 for the report-price contrast); per-season paired
  SE 0.0047-0.0065, so single-season moves under ~1.3pp are inside noise — which
  covers the +0.13/+0.13/+0.64pp moves on 2023-24/2024-25/2025-26 and the
  per-season report-worth numbers, but NOT the -12.57pp on 2021-22 or the
  -0.741pp pooled report price.
  (11) **TESTS.** Full suite: **127 passed, 2 failed** — `test_tanking.py::
  test_live_virtual_rows_match_gate_table` and `::test_fit_k_walkforward_old_
  floor_is_unchanged`. **BOTH WERE ALREADY RED BEFORE THIS ENTRY'S CHANGES,
  PROVEN BY BISECT** (stash the D171 edits, re-run: identical two failures).
  They are **D170 collateral**: both pin a CONSTANT derived from a DB vintage
  that no longer exists, and D170's report backfill moved `_comp_c_shutdown`'s
  inputs under them. Exact drift, measured directly:
  registered ship value **-2.26990**; D170 backfilled DB without the Clippers
  fix **-2.17831**; with it **-2.08251**. **NOT re-pinned here** — re-pinning a
  certification-adjacent fixture is the owner's call, on the same principle by
  which D170 left the Clippers map alone, and it is on the follow-up list.
  (12) **WHAT THIS ENTRY DOES NOT CLAIM.** It does not claim the model is good:
  the certified pooled gap is still **+12.87%** against the market and the market
  still wins **18 of 19** seasons. It does not claim the Clippers fix improved
  anything — it costs +0.12pp. It does not widen the eval corpus, flip a default
  other than the (now correct) name resolution, or re-run any gate. It does not
  re-run a single betting/ROI study, so every such number in the register remains
  a lower bound of unmeasured size (§9). It does not fix the `da Silva, Tristan`
  parser artefact (10 same-day OUT rows) or ingest the `01PM` report edition. It
  does not re-pin the two failing tanking fixtures. And it does not touch
  `capstone_pergame_pre_cert.csv`, `_d122cert.csv` or `_d132_leaky.csv`.
  [code nbapred/teams.py (NEW, canonical name resolution);
   nbapred/model/tanking.py, nbapred/engine/slate.py, nbapred/ingest/injury_pdf.py,
   scripts/prod_by_season.py, scripts/bp_ladder.py, scripts/apr_program.py
   MODIFIED (name resolution only);
   scripts/k19_t2.py MODIFIED (adds the `t2i` tier arm; `blind`/`t2` untouched);
   scripts/make_charts_cert.py, make_status_charts.py, make_history_charts.py,
   k19_chart.py, wf_equity_chart.py MODIFIED (titles + label-collision fixes);
   scripts/d171_team_audit.py, d171_fix_verify.py, d171_cert_table.py,
   d171_gap_census.py, d171_era_density.py, d171_k19_analyze.py,
   d171_report_price.py, d171_k19_stats.py, d171_history_charts.py (all NEW);
   data/recert_notes.md (full working, checkpointed as the run proceeded);
   data/capstone_pergame.csv REWRITTEN (md5 695d40a3545e889267cad403b7acdce8),
   data/capstone_pergame_D158.csv (the superseded artifact, preserved
   byte-identical, md5 3b7bbbb78ac73c63273c18a8aa30013c),
   data/capstone_pergame_D171_noclipfix.csv (the decomposition arm),
   data/prod_by_season_D171.json, data/d171_*.json,
   data/k19_d171_t2.json + _pergame.csv (22,804 games),
   data/k19_d171_t2i.json + _pergame.csv (9,516 games),
   data/logs/d171_cert.log, d171_cert_noclipfix.log, d171_k19_t2.log,
   d171_k19_t2i.log;
   charts/ 9 files regenerated (§8);
   docs/LEAKAGE.md, docs/ERAS.md, docs/STAT_INVENTORY.md updated;
   DB data/nba.duckdb READ-ONLY THROUGHOUT — retry_s=60 on every connect, ZERO
   writes, no table altered; scripts/k19_model.py, k19_analyze.py, k19_rules.py,
   ats19_score.py, wf_equity.py, sl_components.py, el_eralocal.py NOT RUN]

- D172 THE FULL TEAM-NAME JOIN AUDIT (THIRD INSTANCE OF THE BUG CLASS), AND
  COACH / OWNERSHIP MEASURED RATHER THAN OPINED. **HEADLINE: THE JOIN AUDIT
  FINDS NO FOURTH CLIPPERS — THE ONLY REAL ROW LOSS ANYWHERE IN THE REPO IS
  D171's KNOWN `da Silva, Tristan` PARSER ARTEFACT (30 ROWS). COACH IS DEAD:
  THE IN-SEASON COACH-CHANGE EFFECT IS 100% MEAN REVERSION (MATCHED DiD
  -0.181 pts [-1.137,+0.774] p=0.694) AND THE EFFECT DOES NOT TRAVEL WITH THE
  MAN (lag-1 r=+0.0009 vs THE MARKET). OWNERSHIP PRODUCES ONE SURVIVOR
  (-0.948 pts/game [-1.554,-0.343] p=0.0049 IN A NEW OWNER'S FIRST FULL
  SEASON) WHICH IS **PRE-REGISTERED AND BLOCKED, NOT GATED**, BECAUSE ITS
  OWNERSHIP TABLE IS HAND-BUILT AND UNSOURCED.** NO PRODUCTION DEFAULT
  CHANGED. NO CERTIFIED ARTIFACT TOUCHED. NO GATE RUN.
  (1) **THE AUDIT, EVERY SURFACE, BOTH DIRECTIONS** (`d172_join_audit.py`,
  READ-ONLY; `data/d172_surfaces.csv`, `d172_unresolved.csv`). 22 surfaces
  enumerated: `nba_games.team_abbrev` (the spine), `odds_market`/`odds_open`/
  `odds_hist_sbr`, `injury_reports_pit`, `schedule_features`, `epm_history`(+
  `_daily`), `ratings_2k`, `data/arenas.csv`, `game_officials`, the
  TeamRankings jsonl, the ESPN and Action Network per-season csvs, and five
  Kaggle sources. **`nba_games` carries 131 distinct abbreviations over 30
  seasons: 30 modern, 6 era, and 95 that are NOT NBA FRANCHISES.**
  (2) **THE SIX ERA CODES ARE FULLY COVERED AND THE D161 CROSSWALK IS
  COMPLETE.** NJN 1,396 / SEA 1,021 / NOH 802 / CHH 495 / VAN 378 / NOK 178
  = **4,270 team-game rows** (2,135 games); `team_id` disagrees with
  `team_abbrev` on exactly those 6 pairs and nowhere else; after the map the
  set difference (nba_games − odds codes) is **empty on every season and every
  odds table**. Without it those 4,270 rows have no market price — D161's bug,
  re-measured from scratch and confirmed fixed. **WSB is the one relocation
  code we do NOT need** (nba_games back-stamps the 1996-97 Bullets as WAS);
  `data/arenas.csv` carries a WSB row that joins to nothing, documented not
  deleted.
  (3) **A NEW FINDING: 95 NON-NBA "TEAMS", 315 ROWS, QUARANTINED BY LUCK NOT
  BY DESIGN.** 167 All-Star rows (`003`: Team Chuck/Shaq/Kenny/Candace,
  EST/WST, USA/WLD) and 148 preseason rows (`001`: Real Madrid, Maccabi,
  Barcelona, Fenerbahce, Melbourne, Adelaide, Sydney, Perth, Alba, Bayern,
  Flamengo, Guangzhou …), 146 of them paired against a real NBA team. Every
  production consumer filters `game_id LIKE '002%'` — **except
  `nbapred/engine/slate.py:52`, the LIVE path**, which does `SELECT DISTINCT
  team_id FROM nba_games WHERE game_date = ?` with no game-type filter.
  **FLAGGED, NOT CHANGED** (it is the LIVE path and the owner's call).
  (4) **THE UNRESOLVED-NAME TABLE.** After classifying every unmatched value,
  exactly ONE is a real row of real data being dropped: **`da Silva, Tristan`,
  30 rows in `injury_reports_pit.team`** — D171's known parser artefact, still
  unfixed, and `teams.py` REPORTS it rather than dropping it silently, which
  is the D171 design working as intended. Everything else is (a) an All-Star /
  exhibition entity correctly absent from a franchise map (ESPN 82 rows,
  ActionNetwork 68), (b) a defunct pre-1996 franchise in an out-of-corpus
  Kaggle file (SYR/MNL/PHW/STL/ROC/FTW/CIN — 106,774 rows, never ingested), or
  (c) a vendor short code that IS mapped (`odds_hist_sbr`'s 35 SBR city
  strings all resolve via `build_odds_open.TEAMS`, zero unmapped; ESPN's
  `GS/NO/NY/SA/UTAH/WSH` 8,171 rows all resolve via `ABBR`). **`game_officials`
  joins by `game_id` at 100.00% and is not a name-join risk at all.**
  `ratings_2k.team_slug` and `epm_history_daily.team_alias` (98.9% NULL) are
  CARRIED BUT NEVER JOINED — a coverage note, not a defect.
  (5) **WHAT WAS ROUTED THROUGH `nbapred/teams.py`, AND WHAT DELIBERATELY WAS
  NOT.** ADDED: `teams.FRANCHISE` (the era crosswalk, canonical),
  `teams.modern()` (idempotent), `teams.BBREF_TO_US`. **The crosswalk had
  existed as THREE byte-identical inline copies** (`k19_model.py`, `k19_t2.py`,
  `ats19_score.py`) **and in no importable module.** Those three are **NOT
  EDITED** — they produce certified artifacts, and routing them is a provable
  no-op that still re-touches a scored script. Guarded instead by
  `tests/test_teams_canon.py` (8 tests, all green), which asserts each inline
  dict equals the canonical one, so divergence goes red without re-running a
  certification. NOT ROUTED, with reason: `build_odds_open.TEAMS/TR_TEAMS`,
  `bo_lineshop.TR_TEAMS`, `build_nba_open_close.ABBR`, `kaggle_odds.TEAM_MAP`,
  `cm_arb.TR_TEAMS` — these map VENDOR vocabularies, not NBA full names, so
  the nickname-suffix resolver does not apply; all are measured at 100% on
  live data. **`TR_TEAMS` is duplicated in 3 files and `ABBR` is a 6-key
  subset of the 8-key `TR_TEAMS` doing the same job — a divergence waiting to
  happen, FLAGGED, not merged** (touching `build_odds_open.py` rebuilds
  `odds_open`).
  (6) **COACH DATA, $0, AND A NEW DATA-QUALITY FINDING: nba_api's COACH FEED
  IS NOT POINT-IN-TIME AND WAS REJECTED.** `commonteamroster`'s Coaches result
  set returns Mike D'Antoni for HOU 2015-16 and Kenny Atkinson for BKN 2015-16
  — both 2016-17 hires — while the men who actually coached (McHale/
  Bickerstaff, Hollins/Brown) are absent. Quantified over all 30x30 (807 rows,
  **799 of 892 team-seasons, 93 missing outright**): **EXACT match on the
  head-coach set 78.6%; MISSES a coach who really coached 21.0%; NAMES a coach
  who did not 12.1%**; worst on 1996-2000 at **50.7%**. Kept as a cross-check
  artifact only. **SOURCE OF TRUTH: Basketball-Reference `NBA_<end>_coaches`,
  free, cached to `data/raw/ext_bbref/coaches/`** — one row per (coach, team,
  season) with that coach's regular-season G, in chronological order within a
  team, so the cumulative game count locates each change EXACTLY in the
  schedule. **COVERAGE: 30/30 seasons 1996-97..2025-26, 1,013 coach-team-
  seasons, 191 coaches, 121 in-season changes; BBRef G-sum vs nba_games game
  count mismatches on 0 of 892 team-seasons; a coach is assigned to 71,092 /
  71,092 = 100.00% of regular-season team-games; team vocabulary resolves 100%
  both directions** (BBRef's BRK/CHO/PHO/WSB crosswalked — a FOURTH team
  vocabulary, handled at the door).
  (7) **MEASURE BEFORE MODELLING — WHAT A COACH DEMONSTRABLY MOVES.** Jump AT
  the in-season change (same roster, same season, n=118, season-clustered):
```
    behaviour               pre     post    delta   95% CI               p
    players used / game  10.393   10.186   -0.211  [-0.327,-0.095]  0.0009
    top-5 churn / game    1.105    1.234   +0.164  [+0.084,+0.244]  0.0002
    top-8 minute share   0.9199   0.9249  +0.0041  [-0.0006,+0.0088] 0.086
    minutes Herfindahl   0.1203   0.1212  +0.0004  [-0.0008,+0.0016] 0.464
    pace (possessions)    96.08    95.87   -0.055  [-0.632,+0.521]  0.846
```
  **A new coach shortens the rotation and churns the starting five. He does
  not change pace or minute concentration.** Both survive BH. **A FIRST-PASS
  VARIANCE DECOMPOSITION IS RETRACTED IN THIS ENTRY:** "coach spell explains
  1.5-2.0% more than team-season" was reported without a null; against a
  permutation null that keeps spell count and lengths and randomises the
  boundaries, **4 of 5 behaviours are indistinguishable from a random split**
  (perm p 0.070-0.365). Splitting always explains more — D164's lesson applied
  to my own statistic.
  (8) **THE COACH-CHANGE EVENT STUDY — AND A SECOND SELF-RETRACTION.** Frame:
  `ats19_frame.csv.gz`, 19 seasons, 22,742 games, `m_us` COACH-BLIND; 45,484
  team-games joined; **62 changes in frame, 52 with >=10 games each side.**
  FIRST PASS SAID post-minus-pre **+1.78 pts vs the market, p=0.012,
  "unpriced". THAT IS WRONG AND IS RETRACTED HERE.** Two defects, both mine:
  the placebo was NOT matched (it cut non-event seasons whose pre-window
  residual was ~0, against events at -2.28), and **post-minus-pre is the wrong
  statistic** — a coach is fired AFTER a bad run, so pre is negative by
  selection and the delta is mechanically positive. CORRECTED:
```
    quantity                              mean   95% CI              p
    PRE-change  residual vs MARKET      -2.208  [-2.941,-1.474]  0.0000
    POST-change residual vs MARKET      -0.429  [-1.371,+0.513]  0.350
    POST first 20 games  vs MARKET      -0.482  [-1.601,+0.638]  0.377
    PRE-change  residual vs OUR MODEL   -2.005  [-2.684,-1.326]  0.0000
    POST-change residual vs OUR MODEL   -0.682  [-1.714,+0.351]  0.182
```
  MATCHED DiD (calliper 0.75 pts on the pre-window market residual, mean 89
  controls per event, **52/52 matched**): fired-coach teams went **-2.23 ->
  -0.22 (+2.013)**; teams that ran EQUALLY BADLY and KEPT the coach went
  **-2.15 -> -0.19 (+1.960)**. **DiD -0.181 [-1.137,+0.774] p=0.694 vs the
  market; -0.551 [-1.423,+0.320] p=0.200 vs our model. THE COACH-CHANGE EFFECT
  IS MEAN REVERSION.**
  (9) **PERSISTENCE — D137's TEST, DECOMPOSED, AND IT IS THE TEAM'S MEMORY:**
```
    contrast                          vs OUR MODEL      vs the MARKET
    SAME coach, same team (n=378)   +0.400 (p<1e-15)   +0.120 (p=0.020)
    DIFFERENT coach, same team(159) +0.071 (p=0.375)   -0.130 (p=0.103)
    SAME coach, DIFFERENT team (29) +0.180 (p=0.351)  +0.0009 (p=0.996)
```
  Signal share: vs our model tau **1.559 pts / 50.9%**; vs the MARKET tau
  **0.665 pts / 16.4%** (D137's home advantage: tau 1.80, 26%). **The +0.400
  is our own model's team-level miscalibration persisting through roster
  continuity — it collapses to +0.120 against the market and to +0.0009 when
  the coach changes team, which is the only coach-specific persistence test
  there is. A coach effect with no memory is a description, not a feature —
  and this one has none.**
  (10) **OWNERSHIP. THE TABLE IS HAND-BUILT AND SAYS SO.** `OWNER_SPELLS` in
  `d172_ownership.py`: control ownership only, 73 spells over 30 teams,
  892/892 team-seasons assigned, **built by the agent from its own public
  knowledge — not scraped, not machine-verified**; 4 events spot-checked
  against Wikipedia, 4/4 correct. **STRUCTURALLY, OWNER IS BARELY A DIFFERENT
  VARIABLE FROM TEAM:** 73 levels for 892 team-seasons nested in 30 teams = a
  team dummy with 43 extra splits; **mean spell 12.2 seasons**; 65.2% of
  team-seasons sit in their team's modal spell. D70/D137 already killed the
  team dummy. **THE DIRECT "ON TOP OF D73" TEST CANNOT BE RUN AND THIS ENTRY
  SAYS SO PLAINLY:** the tank composite exists on **4 seasons only**
  (2022-23..2025-26, 120 team-seasons) on which just 6 teams have more than one
  owner — on that window owner and team ARE the same variable. Tested instead
  against a tank-outcome proxy on all 30 seasons (share of minutes to players
  within 2 years of their draft, comp-A's own direction): owner's marginal
  variance share **18.68% against a permutation null whose MEAN is 24.20%
  (p95 33.68%) -> perm p = 0.8705.** **The real split explains LESS than a
  random split of the same shape. Dead.** Youth share at an ownership change
  -0.0415 [-0.0918,+0.0089] p=0.102, also null.
  (11) **THE ONE SURVIVOR, AND WHY IT IS NOT GATED.** First FULL season under
  a new control owner (the closing season excluded from both sides), n=27
  events with a clean pre and post, K=14 seasons:
```
    season BEFORE the sale,      vs MARKET  +0.362 [-0.331,+1.054] p=0.279
    first FULL season new owner, vs MARKET  -0.948 [-1.554,-0.343] p=0.0049
    first FULL season new owner, vs MODEL   -1.247 [-1.970,-0.523] p=0.0026
    MATCHED-PLACEBO DiD,         vs MARKET  -0.918 [-1.489,-0.347] p=0.0041
```
  **It is not regression to the mean — the pre-sale season is
  indistinguishable from zero.** Leave-one-event-out -1.126..-1.489;
  leave-one-season-out the same range. **BH over the whole D172 family (m=31,
  FDR 0.05) rejects 9**: coach spell-variance perm `poss`; coach lag-1
  same-coach-same-team vs model; coach PRE-change residual vs market and vs
  model; coach behaviour jump `t5_turnover` and `n_used`; owner clean change
  vs model and vs market; owner placebo DiD vs market. **Every coach rejection
  is a selection artifact, a roster confound, or a rotation-shape description
  nobody can price. Not one is a feature.** The ownership survivor is
  pre-registered in **`data/coach_prereg.md` (sha256
  `972a7ea7b0d17ea57ff897418c40f489bf6cec34d22110bce4c68d1ddc77b803`)** with
  **at most 2 arms** (NEW_OWNER_Y1; NEW_OWNER_Y1 residualised on the D73 tank
  composite) and the full GATE_POLICY_V2 §8-§11 design — rolling-origin, LOSO,
  era decomposition with COVID as separate strata, season-clustered K-1 dof t
  interval as the shipping statistic, calibration veto, BH against the
  extended family, **and the D164 permutation null run alongside as a
  mandatory arm**. **IT IS BLOCKED AND NOT RUN.** The blocker is not date
  error (which attenuates toward zero) but **recall bias in which sales were
  remembered at all**: if the agent's memory of franchise sales correlates
  with the sale having been followed by a memorable collapse, the estimate is
  manufactured and no within-sample stress test can detect it. Clearance
  requires re-deriving every row from a citable source WITHOUT reference to
  the current table, reporting the count of sales the agent failed to recall,
  and discarding the result outright if that exceeds 15%.
  (12) **ERA STATEMENT (GATE_POLICY_V2 §10).** Audit universe: the FULL corpus
  **1996-97..2025-26, 80,267 team-game rows, 30 seasons** — wider than any
  scored frame, deliberately, because a join hole hides where nobody scores.
  Coach panel: the same 30 seasons, 71,092 regular-season team-games. Coach and
  ownership EFFECTS: the 19-season frame **2007-08..2025-26, 22,742 games**,
  because that is where `m_us` and an opening market price both exist.
  **COVID: 2019-20 and 2020-21 appear only as separate season clusters and are
  never pooled into a headline.** CLUSTERING: season throughout, K=18-19 on the
  event studies and K=14 on the ownership contrast — **K=14 at n=27 is small
  and every ownership number should be read with that in front of it.**
  AVAILABILITY TIER: not applicable — no arm of D172 consumes an out-set.
  (13) **WHAT THIS ENTRY DOES NOT CLAIM.** It does not claim a coach does not
  matter to a basketball team; it claims that **against the market, net of the
  roster, a coach effect is not measurable, does not persist, and does not
  travel with the man** — a confounded shadow of roster quality, which is the
  answer the brief expected and it is now measured rather than opined. It does
  not re-run any gate, move any certified number, or change any production
  default. It does not fix `da Silva, Tristan`, the `slate.py` game-type
  filter, or the duplicated `TR_TEAMS`/`ABBR` maps — all three are FLAGGED for
  the owner. It does not gate the ownership result and explicitly forbids
  doing so until the table is sourced. And it supersedes NOTHING: the prior
  "coach fixed effects low-value (confounded with roster; the W5 window
  suffices)" register opinion is now **CONFIRMED BY MEASUREMENT** rather than
  asserted.
  (14) **TESTS.** Full suite **135 passed, 2 failed**. The 2 are
  `test_tanking.py::test_live_virtual_rows_match_gate_table` and
  `::test_fit_k_walkforward_old_floor_is_unchanged` — **THE SAME PAIR D171 §11
  RECORDED AS ALREADY RED BEFORE ITS OWN CHANGES**, D170 backfill collateral,
  still not re-pinned (the owner's call). Isolated to confirm attribution:
  `test_tanking.py` alone is 2 failed / 4 passed, everything else is
  **131 passed / 0 failed**. D172 cannot have caused them — its only code
  change is three NEW names in `nbapred/teams.py` (`FRANCHISE`, `modern`,
  `BBREF_TO_US`) that `model/tanking.py` does not import. The pass count rises
  127 -> 135 exactly because `tests/test_teams_canon.py` adds 8.
  [code nbapred/teams.py MODIFIED (ADDITIVE ONLY: FRANCHISE, modern(),
   BBREF_TO_US — no existing call site changes behaviour);
   tests/test_teams_canon.py (NEW, 8 tests, green);
   scripts/d172_join_audit.py, d172_coach_pull.py, d172_coach_bbref.py,
   d172_coach_measure.py, d172_coach_decompose.py, d172_ownership.py,
   d172_nulls.py, d172_stress.py, d172_chart.py (all NEW);
   data/coach_notes.md (full working, checkpointed as the run proceeded),
   data/coach_prereg.md + .sha256 (PRE-REGISTERED, BLOCKED, NOT RUN);
   data/d172_surfaces.csv, d172_unresolved.csv, d172_coach_bbref.csv,
   d172_coach_nbaapi.csv, d172_behaviour.csv.gz, d172_coach_seasons.csv,
   d172_events.csv, d172_event_rows.csv, d172_matched.csv,
   d172_owner_spells.csv, d172_owner_events.csv, d172_measure.json,
   d172_decompose.json, d172_ownership.json, d172_nulls.json,
   d172_stress.json;
   data/raw/ext_bbref/coaches/NBA_1997..2026_coaches.html (30 cached pages);
   data/logs/d172_coach_pull.log;
   charts/coach_effects.png (NEW, 150dpi, rendered and inspected — four
   label-collision fixes made by looking at the render: panel A's DiD value
   collided with its own error bar, panel B's annotations were clipped at the
   left axis limit, panel C's near-zero bars had no room for their labels on
   their own side, and the two-line footer was clipped off the canvas);
   scripts/k19_model.py, k19_t2.py, ats19_score.py NOT MODIFIED AND NOT RUN;
   NO GATE RUN; NO PRODUCTION DEFAULT CHANGED; NO CERTIFIED ARTIFACT TOUCHED;
   DB data/nba.duckdb READ-ONLY THROUGHOUT — read_only=True with retry_s=60 on
   every connect, ZERO writes, no table created or altered]
- D173 **THE BETTING/ROI STUDIES RE-RUN ON D170/D171's BACKFILLED DATA, AS D171
  §9 RANKED THEM — AND TWO OF THE REGISTER'S THREE NEGATIVE HEADLINES DO NOT
  SURVIVE. D162's ATS REJECTION IS GONE (-3.25% [-4.46,-2.08] SIG NEG ->
  -0.68% [-2.13,+0.70] ns; 13 significantly-negative cells -> ZERO). D168's
  STRUCTURAL-CONTAMINATION EXPLANATION IS GONE (the ablation ladder no longer
  decays: V4_STRIPPED -3.70% -> +3.91%, and the -6.31 SIG paired delta that
  carried the argument becomes +0.82 ns). D161's RULES ARE RESCUED ON 19
  SEASONS (-5.60% SIG NEG -> -0.88% ns) BUT REMAIN **REJECTED ON THE OWNER'S
  PRIMARY FRAME AT THE OPEN: UNION -4.20% [-8.66,-0.36] SIG NEG, K=8**. D166
  and D159 are UNCHANGED in kind (+3.54% -> +3.35% ns; CLV +0.01197 ->
  +0.01228 SIG).** NOT A NEW GATE: every config, rule, threshold, window, band
  and seed is carried exactly as frozen and NOTHING was re-selected. No
  production default changed, eval corpus not widened, `nbapred/` and
  `scripts/bet_engine.py` UNTOUCHED, DB `read_only=True` retry_s=60, ZERO
  writes. Full working checkpointed in `data/rerun_notes.md`.
  (0) **CONTROL-HASH FIELD (D134 rule).** `data/capstone_pergame.csv` md5
  **695d40a3545e889267cad403b7acdce8** BEFORE and AFTER, i.e. UNCHANGED — this
  entry never runs `prod_by_season.py`. `capstone_pergame_D158.csv` preserved
  at **3b7bbbb78ac73c63273c18a8aa30013c**. Copied aside first:
  `capstone_pergame_D173_BACKUP_rerun.csv`, `ats19_frame_D162.csv.gz`,
  `ats19_D162.json`, `k19_rules_D161.json`, `wf_equity_D166.json`,
  `sl_components_D168.{json,csv.gz}`, `hc_honestclv_D159.json`,
  `k19_model_stats_D161.json`.
  (0a) **THE TWO FRAMES, THE OWNER'S SCOPING DECISION, APPLIED EVERYWHERE.**
  **PRIMARY = the report era 2018-19..2025-26, 8 seasons, tier T2** (5PM injury
  report UNION official pregame inactives) — injury reports begin **2018-12-17**
  and are central to the strategy, so this is the frame that matches how the
  model will run live. K=8 -> **7 dof**. Carried caveat: 2018-19 has report
  coverage on **788/1230** games, so it is T2 but not uniformly so.
  **SECONDARY = 19 seasons at best-available tier**, T2i (official inactives)
  2007-08..2017-18 and T2 2018-19..2025-26, K=19 -> 18 dof, for continuity with
  D161/D162/D166. **TIERS ARE NEVER POOLED SILENTLY (D158's lesson); n and tier
  are stated per season in every table in `data/rerun_notes.md`.**
  (1) **HARNESSES REUSED, NOT REBUILT; DEFAULTS BYTE-IDENTICAL.** The only
  edits are env-gated path overrides, all off by default: `ats19_score.py`
  (ATS19_K19/ATS19_TAG/ATS19_REPORT_ERA), `k19_rules.py`
  (K19_PERGAME/K19_RULES_TAG/K19_REPORT_ERA), `oc_capacity.py`
  (OC_FRAME/OC_TAG), `as_adaptive.py` (AS_TAG), `wf_equity.py`
  (WF_TAG/AS_TAG/WF_PERBET), `sl_components.py` (SL_TIER, default `blind` =
  D168 exactly), `sl_score.py` (SL_TAG/SL_SCORE_TAG), `hc_honestclv.py`
  (HC_HONEST/HC_PBS/HC_TAG). `sl_components.py`'s `t2` branch is the out-set
  builder from `k19_t2.py::season_run` VERBATIM. **ONE STATISTICAL FIX:**
  `ats19_score.py::mde80_clustered`'s t-table was sparse and fell back to dof
  18's 2.101 for any dof it did not list, which UNDERSTATES MDE80 at the K=8
  frame (dof 7 -> 2.365); completed to match `lb_longshot.cluster_mean_t`.
  Reporting statistic only — no point estimate or CI moves.
  (2) **CONTROLS. TWO PASS EXACTLY; THE THIRD CANNOT AND THAT IS ITSELF THE
  FINDING.** (a) **ATS19 vs D162: PASSES.** Re-run on the old `k19_pergame.csv`
  reproduces `ats19_D162.json` with exactly three classes of difference — the
  3 `mde80_clustered` fields the t-table fix corrects, the 2 capstone md5
  fields, and 42 `frozen_rules_ats` fields (that block reads
  `capstone_pergame.csv`, already D171's). **Every ATS statistic — cover, ROI,
  CI, CLV, placebo, era, battery — is bit-identical.** (b) **wf_equity vs D166:
  PASSES** — prints `ANCHOR D162 POOL19 ROI -3.245% cover 50.654%`, all five
  window arms reproduce D165 `EXACT`, headline `+3.54% / +54.9u`. (c)
  **hc_honestclv vs D159: PASSES** — on `capstone_pergame_D158.csv` it returns
  UNION honest **+0.01197 [+0.01099,+0.01295]** and matched-control alpha at
  the open **+4.93% ns**, D159's registered numbers to five decimals. (d)
  **k19_rules vs D161 CANNOT be exact**: the rule bet SETS are built from live
  DB features (`star_out_map` reads the availability tables), so D170's
  backfill changes which games fire. Carried as a DECOMPOSITION arm (old
  probabilities + new DB), not a control. Its signature: **STAR_FAV_SHARPER
  goes from K=4 seasons to K=19** — the rule could not fire before 2022-23
  because no inactives existed.
  (3) **STUDY 1 — D162, ATS AT THE OPENING SPREAD. THE REJECTION IS LIFTED ON
  EVERY WINDOW.** T=0 (ALL GAMES, NO SELECTION, the pre-registered PRIMARY),
  breakeven 52.3810%:

        window        arm      n    cover%    ROI%      [K-1 t CI]    sig   K  MDE80
        REPORT8       OLD   9469   50.662   -3.25  [ -6.24, -0.41]  SIG-   8   3.95
        REPORT8       NEW   9469   51.473   -1.71  [ -4.90, +1.30]    ns   8   4.20
        POOL19        OLD  22742   50.654   -3.25  [ -4.46, -2.08]  SIG-  19   1.67
        POOL19        NEW  22742   52.021   -0.68  [ -2.13, +0.70]    ns  19   1.98
        OOS14         OLD  16634   50.438   -3.64  [ -4.76, -2.56]  SIG-  14   1.53
        OOS14         NEW  16634   52.097   -0.53  [ -2.08, +0.91]    ns  14   2.08
        OOS_DEEP15    NEW  17858   52.047   -0.63  [ -2.07, +0.73]    ns  15   1.94
        NOCOVID17     NEW  20611   52.225   -0.29  [ -1.66, +1.09]    ns  17   1.92

  **Family-wise, the cleanest statement of the flip: 13 significantly-NEGATIVE
  primary ROI cells become ZERO; 0 of 20 are significantly positive against 1.0
  expected; exactly 1 of 20 SECONDARY cells is significantly positive — the
  chance rate exactly.** Seasons covering >52.381%: 1/19 -> 8/19; >50.000%:
  13/19 -> 17/19. **THE VERDICT MOVES FROM REJECTED TO NOT PROVEN EITHER WAY.
  It is NOT replaced by a positive result:** cover 52.021% is still below
  breakeven and no window's ROI interval excludes zero from below.
  Spread-point CLV roughly DOUBLES and stays SIG+ on every window: POOL19
  **+0.16623 [+0.06699,+0.26363] -> +0.32001 [+0.20244,+0.43467]**, REPORT8
  +0.31958 -> **+0.51563 [+0.29855,+0.72388]**. It is still worth far less than
  the vig.
  (4) **STUDY 2 — D161, THE FROZEN-RULES LADDER. THE ANSWER IS FRAME-DEPENDENT
  AND BOTH HALVES MATTER.** UNION:

        arm / window          OLD                       NEW
        CLOSE|ML POOL19       -3.40 [-7.87,-0.53] SIG-  -0.92 [-3.24,+1.54] ns
        CLOSE|ML OOS_DEEP     -5.60 [-9.60,-0.74] SIG-  -0.88 [-3.84,+1.68] ns
        CLOSE|ML REPORT8      (not run)                 -2.89 [-7.16,+1.93] ns
        OPEN|SP  POOL19       -4.97 [-7.23,-1.23] SIG-  -1.67 [-4.08,+0.23] ns
        OPEN|SP  OOS_DEEP     -5.03 [-7.91,-0.19] SIG-  -1.32 [-4.33,+0.87] ns
        OPEN|SP  REPORT8      (not run)                 -4.20 [-8.66,-0.36] SIG-

  **On the 19-season SECONDARY frame the rejection is lifted at both prices,
  including D161's headline OOS_DEEP -5.60% -> -0.88% ns. On the owner's
  PRIMARY report-era frame the rejection SURVIVES AT THE OPEN** — UNION -4.20%
  SIG NEG at K=8, with `T20_D03_10` -4.16% SIG- and `STAR_FAV_SHARPER` -5.11%
  SIG-; at the close the same frame is -2.89% ns. **THE RULES ARE THEREFORE
  STILL REJECTED WHERE IT COUNTS: priced at the open, in the era the strategy
  will actually run.** The 19-season rescue is real but is driven by the 11
  pre-report seasons. Family-wise: **0 of 115 pre-specified ROI cells SIG
  POSITIVE against 5.75 expected, 16 SIG NEGATIVE** (D161: 0 / 92 / 23); CLV
  cells SIG positive 37 of 110 (D161: 19 of 88). The union's bet count nearly
  doubles (2,954 -> 5,706 at the close) because the out-sets now exist.
  (5) **STUDY 3a — D166, THE EQUITY PATH. CONCLUSION UNCHANGED, PRECISION MUCH
  BETTER.** k=5 measured+haircut (the firm default) **+3.54% [-3.51,+8.98] ->
  +3.35% [-1.12,+8.04]**, cum +54.9u -> +58.1u; k=1 +1.66% -> +1.83%; k=8 raw
  +4.92% -> +4.58%. Still NOT SIGNIFICANT. What improves is resolution:
  **MDE80 8.10pp -> 5.94pp**, seasons-to-resolve 36 -> 22, seasons positive
  7/14 -> 10/14. The window-choice rule picks **ALL-HISTORY** in both runs. The
  only SIG cell is the exchange arm, which D166 already labels ARITHMETIC ONLY
  (we hold zero exchange data) — not a result. **PRIMARY FRAME (report era, 8
  of the 14 scored seasons, K=8 -> 7 dof): pooled +2.01% on 1,132 bets,
  season-clustered mean +1.06% [-4.58,+6.70] ns, sd 6.7pp, cum +22.8u** —
  BELOW the 14-season track and still not significant.
  (6) **STUDY 3b — D168, THE STRUCTURAL LADDER. THIS ONE FLIPS, AND IT IS THE
  MOST IMPORTANT THING IN THIS ENTRY AFTER §3.**

        variant                      OLD ROI%  OLD paired-vs-V0      NEW ROI%  NEW paired-vs-V0
        V0_FULL                        +3.54   —                       +3.35   —
        V1_noTANK                      +1.68   -1.03 [-2.44,+0.39]     +3.49   +0.42 [-1.51,+2.35]
        V2_noTANK_noBRIDGE             +1.35   -1.02 [-3.42,+1.38]     +3.02   +0.39 [-2.57,+3.35]
        V3_..._noCARRY                 -0.16   -2.15 [-4.68,+0.38]     +4.08   +1.32 [-0.94,+3.58]
        V4_STRIPPED                    -3.70   -6.31 [-12.30,-0.32]SIG +3.91   +0.82 [-5.78,+7.42]
        V5_FF_ONLY                     -1.79   -3.85 [-10.64,+2.94]    -1.79   -4.57 [-10.30,+1.16]
        V6_NOHOME                      -2.32   -4.71 [-10.73,+1.30]    +3.42   +0.41 [-5.73,+6.55]

  **D168's headline was that structural contamination explains ALL of D166's
  +3.54%: the ladder decayed monotonically to -3.70% as the era-specific terms
  were stripped, and V4_STRIPPED vs V0 was -6.31 SIG. ON THE BACKFILLED DATA
  THE LADDER DOES NOT DECAY AT ALL.** V4_STRIPPED is **+3.91%**, ABOVE the full
  stack; every paired delta except V5 is positive-signed and NONE is
  significant, including the one that was. The result no longer depends on the
  terms whose gates were era-specific, so **the contamination mechanism D168
  proposed is NOT SUPPORTED on the better data. THIS IS NOT A POSITIVE FINDING
  AND MUST NOT BE READ AS ONE** — every 13-dof interval in the NEW column spans
  zero except V6_NOHOME (+3.42% [+1.18,+6.56]), 1 SIG cell of 7 pre-specified
  variants against 0.35 expected. What died is D168's EXPLANATION, not the
  absence of an edge. ANCHOR: `sl_score` V0 vs the ATS frame max |dp| =
  **1.910e-14** on 22,742 games (EXACT); the T2 component pass reproduces
  `k19_d171_t2.json`'s per-season `ll_us`; the linear-sum property of
  `production.Predictor.margin` that made this cheap still holds.
  (7) **STUDY 4 — D159, THE HONEST CLV BASELINE. CONCLUSION UNCHANGED.** UNION
  CLV honest at the open **+0.01197 [+0.01099,+0.01295] -> +0.01228
  [+0.01009,+0.01452]** (+2.6%), still SIG; the leak falls from **13.4% to
  11.1%** of the leaky level; matched-control alpha at the open **+4.93% ns ->
  +5.82% ns**, at the close +3.19% ns both times. D159's verdict stands in
  every particular. D159's ML corpus (2023-24..2025-26) sits ENTIRELY inside
  the report era, so PRIMARY and SECONDARY coincide here; **K=3 is the binding
  limitation, not the data**. Caveat now carried on the chart: the LEAKY arm is
  still the frozen D158 oracle-ceiling vintage, so honest-minus-leaky is no
  longer a same-vintage pair and **the leak share is an UPPER bound**.
  (8) **WHAT WAS DELIBERATELY NOT RE-RUN.** **D163's line-shopping / execution
  panels — SKIPPED**, ranked last by D171 §9 because they price the MARKET not
  us, and `bo_lineshop.py` sources its probabilities from the frozen
  `ds_rt1_pergame.csv` vintage, so re-running would not pick up the new model
  probabilities without a structural change to that harness. The LEFT panel of
  `status_trading_h2h.png` is therefore still D163's and is LABELLED
  not-re-run ON THE CHART; the RIGHT (CLV) panel IS re-sourced onto this
  entry's honest re-run. **D169's era-local selection — SKIPPED** (a null
  result; confirmatory not corrective). The two red `test_tanking.py` fixtures
  D171 left are NOT re-pinned — still D170 collateral, still the owner's call.
  (9) **CHARTS — THE DELIVERABLE, ALL RENDERED AND VISUALLY INSPECTED, house
  style per `make_status_charts.py`.** `charts/walkforward_equity.png`
  (rebuilt: cumulative P&L by execution tier, ROI-by-season bars with the
  pooled mean and +-1sd band, the window-choice panel, and a NEW panel (d) for
  the report-era-only PRIMARY frame); `charts/recent_equity_perbet.png` (NEW,
  the owner's direct request: BET-BY-BET cumulative equity across 2024-25 and
  2025-26 only, n=321, x = sequential bet in date order, with a 50-bet rolling
  mean over the unsmoothed raw path); `charts/ats19_open.png`;
  `charts/k19_model_and_rules.png` (its RULES panels were labelled STALE by
  D171 and are now re-run); `charts/status_trading_h2h.png`;
  `charts/honest_clv.png`. **THE D171 CHARTING BUG CLASS WAS AUDITED
  EVERYWHERE and three live instances were fixed:** `hc_chart.py`'s two
  hard-coded `set_ylim(0, ...)` and one `set_ylim(-0.017, ...)` are now
  data-driven; `k19_chart.py` now ASSERTS that no ROI or CLV POINT estimate
  falls outside its (deliberately clamped, and flagged) whisker window. **A
  REAL BUG WAS FOUND AND FIXED IN THE NEW WORK ITSELF:** the report-era mean
  line in `walkforward_equity.png` panel (b) was drawn as a FRACTION on a
  percent axis, putting a +1.06% line on top of the zero line — caught by
  looking at the rendered image, not the code. Stale hard-coded verdict prose
  ("SIGNIFICANTLY NEGATIVE", "ROI -3.25% [-4.46,-2.08] SIG NEG", "0 of 92
  cells") is now COMPUTED FROM THE DATA in both `ats19_chart.py` and
  `k19_chart.py`, so a chart cannot again assert a conclusion its own numbers
  contradict.
  (9a) **THE LOG-LOSS CHARTS ARE CURRENT — VERIFIED, NOT ASSUMED.** This entry
  never rewrites `capstone_pergame.csv` or `k19_d171_t2.json`, so the inputs to
  `logloss_by_season_normalized.png`, `logloss_continuous_current.png`,
  `history_normalized_gap.png` and `history_logloss_by_season.png` are
  unchanged. Re-ran `make_charts_cert.py` and `d171_history_charts.py` and
  diffed: **all four PNGs byte-identical.** 2008-09's **-2.01%** renders below
  the zero line with the "we BEAT the market below this line" rule visible —
  the D171 clipping bug is confirmed fixed on the current data.
  (10) **D-NUMBER COLLISION, HANDLED.** D172 was claimed IN FLIGHT by a
  concurrent coach-effects study (`scripts/d172_*.py`, `data/d172_*`,
  `charts/coach_effects.png`) while this work was running; it appended to the
  register first. This entry therefore took **D173** and its artifact tags were
  renamed `_D172` -> `_D173` to match. No file collision occurred (their prefix
  is lower-case `d172_`, this entry's suffix was upper-case `_D172`).
  (11) **ERA STATEMENT (GATE_POLICY_V2 §10).** Eval universe UNCHANGED and NOT
  widened: 19-season frame 2007-08..2025-26, 22,804 model games / 22,742 with
  an opening spread; certified corpus untouched at 6,148. AVAILABILITY TIER
  best-available and LABELLED per season — T2i 2007-08..2017-18, T2
  2018-19..2025-26 — never silently pooled. NO PLAYED-SET ORACLE anywhere:
  `ORACLE_PLAYED_OUTS` unset in every arm; `TANK_SEASON_FLOOR=2020-21` pinned;
  `player_game_stats` never used to build an out set. COVID FRAME: 2019-20 and
  2020-21 scored and labelled as separate strata, never pooled into a headline.
  CLUSTERING: season, K=19 (18 dof) on the secondary frame, K=8 (7 dof) on the
  primary, K=14 (13 dof) on the walk-forward scored track, K=3 (2 dof) on
  D159's ML frame.
  (12) **DISCIPLINE, RESTATED WHEREVER A POSITIVE NUMBER APPEARS.** **D165's
  family-wise burden: +2.13pp for 7 procedures. D164's manufacturing capacity:
  +16.92 ROI points, 100% search artifact.** Both exceed every positive number
  in this entry. MDE80: ATS19 T=0 POOL19 1.67 -> 1.98pp and REPORT8 3.95 ->
  4.20pp; wf_equity k=5+haircut 8.10 -> 5.94pp; **the 2-season window on
  `recent_equity_perbet.png` has MDE80 = 61 ROI points at K=2 -> 1 dof, larger
  than any effect anyone has claimed for this model, and the chart says so on
  its face.** That chart also states that 2024-25 alone carries **38.8%** of
  the 14-season result (it was **76.3%** pre-backfill, so the backfill roughly
  HALVED the single-season dependence but did not remove it) and that D171
  measured 60.3% of that season's apparent superiority as data completeness.
  (13) **WHAT THIS ENTRY DOES NOT CLAIM.** It does not claim the model is
  profitable: not one pre-specified ROI cell on either frame is significantly
  positive, and the ATS cover rate (52.021%) is still below the 52.381%
  breakeven. It does not claim the frozen rules work — on the PRIMARY frame at
  the open they are still significantly negative. It does not re-select,
  re-gate, widen the corpus, or change a default. It does not re-run D163 or
  D169. It does not refresh `bo_lineshop.json`, `history_analysis.json`
  (superseded for these charts by `d171_history_analysis.json`), or the D153
  ablation battery behind `history_feature_by_era.png`. It does not re-pin the
  two failing tanking fixtures. **It changes the sign of the evidence on D162
  and D168 and it does not pretend that is the same thing as an edge.**
  [code scripts/ats19_score.py, k19_rules.py, oc_capacity.py, as_adaptive.py,
   wf_equity.py, sl_components.py, sl_score.py, hc_honestclv.py MODIFIED
   (env-gated path overrides + the one MDE80 t-table completion; defaults
   byte-identical);
   scripts/wf_equity_chart.py, ats19_chart.py, k19_chart.py, hc_chart.py,
   make_status_charts.py MODIFIED (new panels, data-driven verdicts,
   data-driven axis limits, collision fixes);
   scripts/d173_perbet_chart.py NEW;
   data/rerun_notes.md NEW (full working, checkpointed as the run proceeded);
   data/ats19_frame.csv.gz REBUILT from the new capstone
   (md5 78bbba1fa9cf14e6dc9563aab73bdc0f; the D162 vintage preserved as
   ats19_frame_D162.csv.gz, md5 b9e59afdd54247083184256e2d10a112);
   data/ats19_D173.json, ats19_frame_D173.csv.gz, ats19_D173blind.json,
   ats19_frame_D173blind.csv.gz (the tier-matched blind arm),
   ats19_D162repro.json, ats19_D162repro8.json (the controls),
   k19_rules_D173.json, k19_rules_D161repro.json, k19_rules_D161repro8.json,
   as_adaptive_D173.json, wf_equity_D173.json, wf_perbet_D173.json,
   sl_components_D173.{json,csv.gz}, sl_score_D173.json, sl_frames_D173/,
   hc_honestclv_D173.json, hc_honestclv_D159repro.json;
   data/logs/d173_*.log (15 files);
   charts/walkforward_equity.png, recent_equity_perbet.png (NEW),
   ats19_open.png, k19_model_and_rules.png, status_trading_h2h.png,
   status_logloss_h2h.png, honest_clv.png REGENERATED;
   charts/logloss_by_season_normalized.png, logloss_continuous_current.png,
   history_normalized_gap.png, history_logloss_by_season.png RE-RUN AND
   VERIFIED BYTE-IDENTICAL (already current);
   data/capstone_pergame.csv NOT WRITTEN (md5 695d40a3545e889267cad403b7acdce8
   before and after); scripts/prod_by_season.py, k19_t2.py, k19_model.py,
   bo_lineshop.py, el_eralocal.py NOT RUN;
   DB data/nba.duckdb READ-ONLY THROUGHOUT — read_only=True with retry_s=60 on
   every connect, ZERO writes, no table created or altered]
- D174 THE EXTRAPOLATED EXECUTION BAND, REPLACED WITH MEASURED PANELS BUILT
  FROM FILES WE ALREADY OWNED — **THE POST-2024 JUMP IS REAL. OF 2024-25's
  +17.28% ROI, 14.70 POINTS ARE THE MODEL AT ONE BOOK AND THE SHOP IS WORTH
  +1.66, NOT THE +2.58 THAT WAS EXTRAPOLATED. THE 14-SEASON PATH RE-PRICES
  FROM +3.54%/+54.9u TO +3.13%/+48.6u — MEASUREMENT COSTS 0.43 ROI POINTS AND
  6.6 UNITS, DOES NOT FLIP A SIGN, AND BOTH INTERVALS STILL SPAN ZERO.** AND
  THE HEADLINE METHOD RESULT: **THE NAIVE SEASON-OVER-SEASON LADDER COLLAPSE
  (k=2 CLOSE 0.2569 -> 0.1702 -> 0.1431) IS AN ARTEFACT OF A SHRINKING PANEL,
  NOT A SHRINKING MARKET — ON THE FIXED BASKET OF THE THREE BOOKS PRESENT IN
  ALL THREE SEASONS THE GAIN IS 0.1435 / 0.1392 / 0.1418, FLAT TO 3.0%.**
  DIAGNOSTIC. No production default changed, no gate re-run, eval corpus NOT
  widened, `nbapred/` and `scripts/bet_engine.py` UNTOUCHED,
  `scripts/prod_by_season.py` NOT RUN, no walk-forward re-run (a concurrent
  agent owns that lane — this entry produces the panels and the ladders for it
  to apply). DB `data/nba.duckdb` **READ-ONLY THROUGHOUT** — `read_only=True`
  with `retry_s=60`, ZERO writes, and opened for ONE query (the `odds_market`
  canonical game list). `nbapred.threads.pin(1)` called before numpy in both
  new scripts.
  (1) **THE BRIEF.** D163/D166 put the multi-book execution tier on a record
  that is **7 of 14 seasons MEASURED and 7 EXTRAPOLATED**, with 2024-25 —
  which carries ~76% of the equity result — sitting in an extrapolated band.
  D163 §18 explicitly left Action Network "identified, not built". This entry
  builds it. **EVERYTHING BELOW IS FROM FILES ALREADY ON DISK; ZERO NETWORK
  REQUESTS WERE MADE BY THIS ENTRY'S OWN SCRIPTS.**
  (2) **WHAT THE LOCAL SOURCES ACTUALLY CONTAIN, COUNTED FROM THE RAW JSONL SO
  NOTHING IS BLAMED ON A CSV WRITER. THREE CORRECTIONS TO D163's REGISTRATION.**
  **(a) ESPN's multi-book panel is a 2023-24-ONLY artefact:** distinct
  providers **16 (2023-24) -> 2 (2024-25) -> 4 (2025-26)**. 2024-25 is ESPN BET
  plus its own `- Live Odds` in-game feed and **nothing else**. ESPN stopped
  syndicating rival books when ESPN BET launched as its house book.
  **(b) D163 registered AN as "5 real books (68 DK, 69 FD, 71 BetRivers, 75
  MGM, 76 Caesars)". Caesars is book_id 49, not 76, and only THREE books are
  dense** — DK/FanDuel/BetRivers at 1,336-1,365 games; BetMGM covers 100-319
  games (7-24%) and Caesars 0-99 (0-7%). **The usable AN panel is k=3.**
  **(c) ACTION NETWORK CARRIES NO PER-BOOK OPENING PRICE AT ALL.** book_id 30
  "Open" is a single CONSENSUS opener; the per-book numbers are one snapshot
  taken at scrape time.
  (3) **WHICH PHASE IS THE AN SNAPSHOT? VERIFIED, NOT ASSUMED, BY D163's OWN
  TIE-RATE METHOD — THE SAME OPERATOR THROUGH TWO INDEPENDENT SCRAPERS.**
      comparison                                        n     tie%   mean|d|
      AN[DraftKings] vs ESPN[DraftKings], ESPN CLOSE 23-24  1208  81.54  0.1474
      AN[DraftKings] vs ESPN[DraftKings], ESPN CLOSE 25-26   985  96.45  0.0396
      AN[DraftKings] vs ESPN[DraftKings], ESPN OPEN  23-24  1115  13.99  1.5897
      AN[DraftKings] vs ESPN[DraftKings], ESPN OPEN  25-26   992  25.71  1.4839
      AN[Caesars]    vs ESPN[Caesars],    ESPN CLOSE 23-24  1193  50.63  0.3734
  **THE AN SNAPSHOT IS THE CLOSE.** And 81-96% is neither one resold feed (a
  true duplicate ties at **100.00%** with mean|d| exactly 0.0000 — D163 §4's
  control) nor two independent operators (**36.52%** — D163's cross-operator
  rate). **VERDICT ON INDEPENDENCE: AN AND ESPN ARE INDEPENDENT SCRAPERS OF
  OVERLAPPING OPERATORS.** The residual 3-18% is capture-instant jitter on one
  book, not two books disagreeing. Caesars sits at 50.63% because ESPN's is the
  **NJ state skin** and AN's is **Caesars NV** — two skins of one operator,
  consistent with D163's 91-94% skin band once timing jitter is added.
  (4) **THE DEDUP RULE, DOCUMENTED.** One operator per game whatever the feed
  or skin (D163's rule carried: a skin is not a book; `- Live Odds` are IN-GAME
  markets; `accuscore`/`betegy` are MODELS). At the **OPEN** ESPN wins a shared
  operator — it is the only source with a true per-book open. At the **CLOSE**
  Action Network wins — a real book snapshot in one HTTP response. Games are
  keyed through `odds_market` by `build_odds_open._pair_join`'s unordered-pair
  join **with +/-1 day tolerance**. **THE TOLERANCE IS LOAD-BEARING: ESPN's
  DATES ARE UTC AND AN's ARE ET, AND WITHOUT IT 75% OF THE CROSS-FEED JOIN
  SILENTLY FAILS AND THE TWO FEEDS LOOK INDEPENDENT BECAUSE THEY NEVER MEET**
  (key overlap 310/1278 before, 1269/1278 after). Team strings routed through
  D171's `nbapred/teams.py`, which REPORTS rather than drops: **the only
  unresolved names in any source are 12 All-Star/exhibition sides** (Team
  Stars, Team Stripes, World, Eastern/Western Conf All-Stars, Team Chuck …) —
  no franchise was silently lost.
  (5) **PANEL COVERAGE, MEASURED. THE STRUCTURAL FINDING OF THE ENTRY.**
      season   OPEN                          CLOSE
      2023-24  MEASURED 9 ops, modal 8       MEASURED 11 ops, modal 10
      2024-25  **1 operator — NO PANEL**     MEASURED  5 ops, modal 4
      2025-26  2 ops on 44 games — NO PANEL  MEASURED  6 ops, modal 3
  **A MEASURED MULTI-BOOK PANEL FOR 2024-25/2025-26 EXISTS ONLY AT THE CLOSE.
  AT THE OPEN — THE PHASE D167 DECIDED TO BET — THERE IS EXACTLY ONE
  OPERATOR.** Best-worst dispersion (MEASURED): ESPN open 1.7651 (n=1205) /
  ESPN close 1.6126 (n=1217) / AN close 0.6042 (n=1310) in 2023-24; AN close
  **0.4487** (n=1287) in 2024-25; AN close **0.4577** (n=1254) in 2025-26.
  HARNESS ANCHOR: the ESPN 2023-24 OPEN ladder reproduces D163 §5 to **0.003
  pts** at every k (0.3278/0.4914/0.6924/0.8746 vs 0.3289/0.4952/0.6982/0.8769).
  (6) **THE CONFOUND THAT WOULD HAVE FAKED THE ANSWER, AND THE CONTROL THAT
  CATCHES IT.** Naively, at the CLOSE:
      k   2023-24 (10 ops)  2024-25 (4 ops)  2025-26 (3 ops)
      2       0.2569            0.1702           0.1431
      3       0.3854            0.2550           0.2140
      5       0.5569            0.3118           0.2350
      8       0.7265            0.3118           0.2350
  That reads as a market whose dispersion collapsed 2.4x in two seasons. **IT
  IS AN ARTEFACT.** k=5 and k=8 SATURATE because the panel holds 3-4 books.
  On the FIXED basket of the three books AN carries densely in all three
  seasons (DraftKings, FanDuel, BetRivers — same feed, same instant):
      k   2023-24 (n=1304)  2024-25 (n=1002)  2025-26 (n=1202)   spread
      2       0.1435            0.1392            0.1418     0.0043 (3.0%)
      3       0.2153            0.2088            0.2128     0.0065 (3.0%)
  **ON IDENTICAL BOOKS THE CROSS-BOOK DISPERSION LAW IS FLAT TO 3.0% ACROSS
  2023-24 -> 2025-26. THE MARKET DID NOT CHANGE; THE OBSERVABLE PANEL SHRANK.
  D166's ACT OF CARRYING 2023-24's LADDER FORWARD WAS RIGHT AS A *LAW* AND
  WRONG ONLY IN THE *BOOK COUNT* IT ALSO CARRIED.** NEVER COMPARE A LADDER
  ACROSS SEASONS WITHOUT FIXING THE OPERATOR SET.
  (7) **WAS THE ASSUMED LADDER OPTIMISTIC, PESSIMISTIC OR RIGHT? LAW vs LAW,
  WHICH IS THE COMPARISON THAT DOES NOT DEPEND ON A BACK-OUT.** What D166
  carried forward is D163 §10's pooled ESPN23 law, k=5 +outlier = **0.4261**.
      season   EXTRAPOLATED   MEASURED(close)   +§8 phase bridge   ratio
      2024-25     0.4261         0.2527             0.2806      **1.69x** (1.52x)
      2025-26     0.4261         0.2154             0.2392      **1.98x** (1.78x)
  **OPTIMISTIC, BY 1.7-2.0x ON THE OBSERVABLE PANEL.** Never a raw number
  without its haircut twin — the raw measured k=5 cells are 0.3118 (2024-25)
  and 0.2350 (2025-26) against haircut twins 0.2527 and 0.2154.
  (8) **OPEN vs CLOSE, MATCHED GAMES *AND* MATCHED OPERATORS (2023-24, the
  only season with a panel at both phases; 1,205 games, modal 8 ops).**
      k     OPEN     CLOSE   open/close
      2   0.3278   0.2905     **1.1285**
      3   0.4914   0.4354       1.1288
      5   0.6924   0.6235     **1.1105**
      8   0.8746   0.7973       1.0969
  **THE OPEN IS WORTH ~11% MORE THAN THE CLOSE ON IDENTICAL BOOKS.** This is
  the only defensible bridge from a measured CLOSE ladder to the OPEN tier the
  strategy actually bets, and it CANNOT be re-measured on the AN basket
  because AN has no per-book open.
  (9) **CONTEMPORANEOUS DISPERSION — D163's TIME-vs-DISAGREEMENT CAVEAT,
  SETTLED.** D163 §8/trap 4 could not test simultaneity at the open because
  ESPN's `open` block carries no timestamp. **The Action Network panel settles
  it BY CONSTRUCTION: every book in a game comes from ONE HTTP RESPONSE, so
  the AN cross-section is SIMULTANEOUS.** It is the only genuinely
  contemporaneous multi-operator cross-section in the project.
      basis                                                    gain_2   vs 0.3247
      TeamRankings 2-book OPEN, books a median **2.9h apart**  0.3247      100%
      AN 2023-24 CONTEMPORANEOUS                               0.1533    **47.2%**
      AN 2024-25 CONTEMPORANEOUS                               0.1503    **46.3%**
      AN 2025-26 CONTEMPORANEOUS                               0.1464    **45.1%**
      ESPN 2023-24 OPEN, untimestamped, matched ops            0.3278      101%
      ESPN 2023-24 CLOSE, pinned to tip-off, matched ops       0.2905       89%
  **D163's SUSPICION IS CONFIRMED ALMOST EXACTLY AND REPRODUCES INDEPENDENTLY
  IN THREE SEASONS: A CONTEMPORANEOUS 2-BOOK CROSS-SECTION IS WORTH 45-47% OF
  THE 2-BOOK "OPENING DISPERSION" D142 PRICED. ROUGHLY HALF OF THAT NUMBER WAS
  TIME, NOT DISAGREEMENT.** This does NOT retro-discount D163's ESPN panel,
  whose CLOSE arm is simultaneous and lands within 11% of its OPEN arm (§8) —
  **the TeamRankings 2-book panel was the non-simultaneous one, and it is the
  one D142's +0.331 at N=2 came from.**
  (10) **THE OFFSHORE LAW REPLICATED ON A SECOND INDEPENDENT SOURCE, AND ONE
  HOLE SEASON CLOSED.** `data/raw/kaggle/erichqiu__nba-odds-and-scores/` has
  been on disk since 2026-07-26 and carries **per-book columns**
  (`Pinnacle_/5dimes_/Heritage_/Bovada_/Betonline_` line AND odds, for spread,
  ML and OU) for **2012-13..2018-19**. Pooled 7 seasons, n=9,184, K=7:
  k=2 **0.1210 [0.1027,0.1393]**, k=3 **0.1815 [0.1541,0.2089]**, k=5 **0.2666
  [0.2217,0.3116]**, against D163's KAG (`ehallmar`, 9 offshore operators)
  0.1002 / 0.1527 / 0.2254. **TWO INDEPENDENT OFFSHORE SCRAPES, SAME BAND.**
  Per-season k=2: .1175 .1013 .1136 .1186 .1223 .1631 **.1107 (2018-19)**.
  **2018-19 MOVES FROM EXTRAPOLATED TO MEASURED ON DATA WE ALREADY OWNED.**
  (11) **THE REMAINING HOLE, 2019-20..2022-23 — AND THE VERDICT IS LEGAL, NOT
  TECHNICAL.** robots.txt fetched and read for EVERY host before any content
  request (`data/hole_2018_2023_sources.md`).
  * **Action Network back-history: BLOCKED.** `api.actionnetwork.com/robots.txt`
    is 25 bytes — `User-agent: *` / `Disallow: /`. **The highest-value single
    test could not legitimately be run and WAS NOT SENT.**
  * **ESPN core API back-history: HAS THE DATA, ToS FORBIDS IT.** Per-provider
    panels exist for all five seasons — **4 real operators/game in 2018-19
    rising to 10 by 2022-23, 98/98 probed games, 8/8 and 9/9 on two exhaustive
    slates**. robots.txt is absent (HTTP 403), but the Disney Terms of Use
    prohibit automated access "including … data mining or web scraping" and
    building "any collection of data, data set or database", and
    `www.espn.com/robots.txt` names **`AI-crawler user-agents` / `Disallow: /`**.
    **RECORDED AS ToS-BLOCKED AND STOPPED; NO EXTRACTION PERFORMED.**
  * **Wayback: TOO SPARSE.** VegasInsider 380 snapshot-days, SBR 191, Covers
    131, DonBest 64. Best case ~1,500 **phase-ambiguous** observations against
    ~5,800 games (~25% ceiling), at arbitrary intraday timestamps aligning to
    neither open nor close.
  * **oddsportal: NOT FETCHED, NOT EVEN VIA WAYBACK** — pulling the identical
    blocked pages from an archive mirror evades the publisher's own rule.
    D163's stop is upheld and EXTENDED to the archive route.
  * covers.com dated endpoint: HTTP 404, does not exist. Local Kaggle sets
    `chevronronson`, `cviaxmiwnptr`, `christophertreasure` all checked and all
    single-consensus, NO per-book columns.
  **2019-20..2022-23 MUST STAY EXTRAPOLATED. A DOCUMENTED "AND HERE IS WHY IT
  MUST" IS THE ANSWER.** The blocked audit did nonetheless establish, from a
  ~98-game evaluation sample that was NOT extracted or stored as a panel, that
  the cross-book tie rate runs **50.00% (2018-19) -> 58.67 -> 55.23 -> 44.62 ->
  34.22% (2022-23)** against D163's anchors of 68.22% offshore and 36.52% for
  2023-24 retail. **D163 §16 said the 2.9x offshore->retail transition could
  not be dated. IT IS DATABLE, IT IS GRADUAL, AND IT COMPLETES IN 2022-23.**
  Reported as an audit observation from a blocked source; **NOTHING IN §12
  USES IT.**
  (12) **THE RE-PRICED EQUITY PATH.** Bridge anchor: reconstructing D166 §4's
  k=5+haircut path from its own numbers gives pooled **+3.56% / +55.2u**
  against D166's published **+3.54% / +54.9u** — 0.02pp. Only seasons whose
  LABEL changes move; every other row is D166 verbatim.
      season     n  D166 g  MEAS g  D166cov  NEWcov  D166ROI  NEWROI  label
      2018-19  101  0.1192  0.2223   44.41   44.76   -15.22  -14.54  MEASURED(new)
      2024-25  243  0.3925  0.2527   61.43   60.95   +17.28  +16.36  MEASURED(new)
      2025-26  174  0.6600  0.2154   49.40   47.87    -5.69   -8.61  MEASURED(new)
  **POOLED, 14 SEASONS, 1,553 BETS: +3.56% (+55.2u) -> +3.13% (+48.6u).
  SEASON-CLUSTERED K=14: +2.76% [-3.51,+9.03] -> +2.54% [-3.75,+8.82].
  MEASUREMENT COSTS 0.43 ROI POINTS AND 6.6 UNITS, DOES NOT FLIP A SIGN, AND
  BOTH INTERVALS STILL SPAN ZERO — AND THAT LAST CLAUSE MAY NOT BE QUOTED
  WITHOUT THE FIRST.** Bet-weighted mean gain 0.2803 -> 0.2153.
  **CAVEAT THAT LIMITS THIS TABLE, STATED RATHER THAN BURIED:** inverting
  D166's per-season dcover to recover an "applied gain" is **~8x noisier than
  the underlying law** — over the six MEASURED offshore seasons the backed-out
  gains run 0.0378-0.3983 (**sd 0.1238**) while the measured law runs
  0.2000-0.2428 (**sd 0.0154**). They agree on AVERAGE (0.2060 vs 0.2267),
  which is what validates the bridge in aggregate, but **NO SINGLE-SEASON
  BACK-OUT IS A FINDING.** In particular **2018-19's move is mostly noise
  correction, NOT evidence that the offshore extrapolation under-shot.** The
  2024-25 (n=243) and 2025-26 (n=174) cells are the two largest, and §7's
  law-vs-law table does not depend on the back-out at all.
  (13) **THE OWNER'S QUESTION, ANSWERED WITH THE MEASURED LADDER: IS THE JUMP
  AFTER 2024 REAL, OR AN ARTEFACT OF THE ASSUMED SHOPPING GAIN?**
      2024-25, 243 bets      cover     ROI
      ONE BOOK, no shop      60.08%  +14.70%
      D166 EXTRAPOLATED      61.43%  +17.28%
      **MEASURED k=5+HC**    **60.95%**  **+16.36%**
  **THE JUMP IS REAL AND IT IS NOT A SHOPPING ARTEFACT. OF 2024-25's +17.28%,
  14.70 POINTS ARE THE MODEL AT ONE BOOK AND THE SHOP CONTRIBUTES +1.66, NOT
  +2.58. THE EXTRAPOLATION INFLATED THE SEASON BY 0.92 ROI POINTS — 5.3% OF
  ITS HEADLINE — AND THE OTHER 94.7% SURVIVES MEASUREMENT UNTOUCHED.** The
  season was never a shopping story: **at k=1 there is no shop at all and it
  still returns +14.70%.** Drop-2024-25 on the re-priced path: **+0.68% /
  +8.9u** (D166's own drop test: +1.00%/+13.0u at k=5+HC, **-0.76%/-10.0u** at
  one book). **SO THE RE-PRICED RECORD STILL RESTS ON 2024-25 IN THE SENSE
  THAT REMOVING IT LEAVES A NUMBER NO TEST HERE COULD DISTINGUISH FROM ZERO —
  BUT WHAT 2024-25 IS, IS A MODEL SEASON, NOT AN EXECUTION ASSUMPTION.**
  (14) **THE HONEST BAND ON 2024-25's SHOP GAIN, EVERY CELL LABELLED.**
      basis                                                  gain    label
      MEASURED, 3-4 observable books, CLOSE, +haircut       0.2527   MEASURED
      + §8's open/close bridge (x1.11)                      0.2806   PART-MEASURED
      D166's 9-book law carried forward                     0.3925   EXTRAPOLATED
      §6's law transferred at 2023-24's book COUNT (k=5)    0.5401   law MEASURED, count ASSUMED
  **THE GAP BETWEEN THE BOUNDS IS A BOOK-COUNT QUESTION, NOT A MARKET
  QUESTION** — §6 proves the law is unchanged and the only thing in dispute is
  how many books a 2024-25 bettor could reach. **NO FREE SOURCE ANSWERS THAT:
  ESPN STOPPED SYNDICATING RIVALS WHEN ESPN BET LAUNCHED, AND ACTION NETWORK's
  SCOREBOARD SHOWS A FIXED FEATURED SET. FEED POLICY AND MARKET TRUTH ARE NOT
  DISTINGUISHABLE FROM THESE FILES AND THIS ENTRY DOES NOT PRETEND
  OTHERWISE.** Two corrections also run in OPPOSITE directions and both are
  measured: the phase bridge pushes the measured number UP ~11% (§8), the
  saturation of k=5/k=8 on a 3-4 book panel pushes it UP by more, and §6 says
  nothing justifies scaling the LAW itself.
  (15) **ERA / CLUSTERING / AVAILABILITY (GATE_POLICY_V2 §10).** Eval universe:
  erichqiu = 2012-13..2018-19 (**K=7**, the only ladder here carrying a
  clustered interval); AN/ESPN modern = 2023-24..2025-26 (**K=3**, too few
  seasons for a ladder CI and no CI is quoted on them); the re-priced path is
  **K=14** and every ROI interval above is the 13-dof cluster-mean interval.
  **ERA-AVAILABILITY: unchanged for 2019-20..2022-23 (none) and IMPROVED for
  2018-19 (now MEASURED) and 2024-25/2025-26 (now MEASURED AT THE CLOSE ONLY).
  The record is now 9 of 14 seasons MEASURED and 5 EXTRAPOLATED, against
  D166's 7 and 7.** COVID: 2019-20 and 2020-21 remain EXTRAPOLATED and are
  never pooled into a headline. AVAILABILITY TIER: not applicable — no arm
  consumes an out-set.
  (16) **UNPRICED BIAS, RESTATED ONCE AS REQUIRED.** Best-of-k **always**
  transacts at the most offside book in the panel, and **nothing here, in D163
  or in D166 charges for being LIMITED, RESTRICTED OR VOIDED.** Every number
  in this entry is gross of that cost and it remains the largest unmodelled
  risk in the execution layer. D163 §11's finding that the biggest shop gains
  WIN more (corr +0.0707 / +0.0422) still means the outlier haircut is
  justified by LIMITS, not by information, and this entry does not let it
  borrow an adverse-selection story the data refuses.
  (17) **WHAT THIS ENTRY DOES NOT CLAIM.** It does not re-run the walk-forward
  (a concurrent agent owns that lane; §12 is an ANALYTIC re-pricing through
  D166's own published conversion, anchored to 0.02pp, not a re-run). It does
  not build a per-book OPENING panel for 2024-25/2025-26 — **none exists in
  any free source and that is the entry's central limitation.** It does not
  close 2019-20..2022-23 and says plainly that it cannot. It does not extract
  anything from ESPN's back-history or Action Network's back-history, both of
  which were STOPPED on robots/ToS. It does not re-certify, gate, ship or
  change any default. **AND IT FLAGS FOR THE OWNER, WITHOUT ACTING ON IT: the
  ESPN and Action Network files already on disk were collected from those same
  hosts under those same rules — a PRE-EXISTING COMPLIANCE EXPOSURE that D174
  neither created nor extends, and which the owner should decide about before
  any further scrape of either host.**
  [code scripts/bkp_panel.py, scripts/bkp_ladder.py (both NEW, neither under
   `nbapred/`);
   data/bookpanel_notes.md (full working, checkpointed as the run proceeded),
   data/hole_2018_2023_sources.md (the robots/ToS audit, every host's literal
   rule recorded);
   data/bkp_panel.json, data/bkp_panel_rows.csv.gz (the MEASURED panels —
   per-(game, book) OPEN and CLOSE on our team keys, 2023-24..2025-26 plus
   erichqiu 2012-13..2018-19, for the betting lane to apply),
   data/bkp_ladder.json;
   data/logs/bkp_panel.log, data/logs/bkp_ladder.log;
   docs/OPENING_LINES.md updated (multi-book section rewritten with the three
   corrections to D163's registration, the phase coverage table, the dedup
   rule, the fixed-basket confound and the contemporaneous result; new §8 on
   the 2018-23 hole);
   inputs data/raw/sbr_ext/{an,espn}_nba_odds_raw_{2023-24,2024-25,2025-26}.jsonl,
   data/raw/kaggle/erichqiu__nba-odds-and-scores/<7 seasons>/vegas*.txt,
   data/nba.duckdb (read_only=True retry_s=60, ONE query: the `odds_market`
   canonical game list);
   `nbapred/` UNTOUCHED, scripts/bet_engine.py UNTOUCHED,
   scripts/prod_by_season.py NOT RUN, no gate re-run, no default changed,
   eval corpus unchanged, no walk-forward re-run, no chart written;
   DB READ-ONLY THROUGHOUT — ZERO writes, no table created or altered]

- D175 THE HUNT FOR UNOFFICIAL HISTORICAL INJURY/AVAILABILITY DATA, AND A
  CONCLUSION THAT REVERSED TWICE UNDER ITS OWN MEASUREMENTS. **HEADLINE: A
  FREE, PERMITTED, SYSTEMATIC ARCHIVE EXISTS FOR 5 OF THE 11 PRE-REPORT
  SEASONS (2013-14..2017-18) AND FOR NONE OF THE OTHER 6. PILOTED ON THREE OF
  THEM IT RETURNS +0.100pp / +0.220pp / -2.030pp — MEAN -0.570pp, t(2)=-0.78,
  p=0.52, WRONG SIGN ON 2 OF 3, AND THE SIGN FLIPS WHEN 2016-17 IS DROPPED.
  NOT ACTIONABLE; RECOMMENDATION IS TO KEEP THE POST-2018 FOCUS. SEPARATELY
  AND MORE IMPORTANTLY, A PERFECT AVAILABILITY ORACLE IS WORTH -1.76pp
  (2012-13) / -2.61pp (2015-16) / -3.75pp (2023-24), SO OLD-ERA HEADROOM IS
  2.4-3.5x THE REPORT'S -0.741pp AND HAD NEVER BEEN MEASURED; AND THE OUT-SET
  IS **NON-MONOTONE IN COMPLETENESS** — A PARTIAL FEED IS WORSE THAN NONE.**
  NOTHING INGESTED, NO TABLE CREATED OR ALTERED, NO DEFAULT CHANGED, NO GATE
  RUN, NO CERTIFIED ARTIFACT TOUCHED. DB READ-ONLY THROUGHOUT.
  (1) **ToS / robots FIRST, AND TWO CANDIDATES DIE AT THE DOOR.**
  **prosportstransactions.com** — the brief's strongest a-priori candidate —
  returns **HTTP 403 to every path INCLUDING `/robots.txt`**, with a Cloudflare
  **managed-challenge** body (`_cf_chl_opt`, `cType:'managed'`, "Enable
  JavaScript and cookies to continue"). That is an active anti-automation
  access control; a headless browser / CF-solver / spoofed-browser-UA replay
  would be CIRCUMVENTING it, not polite crawling. **Dropped, not attempted.**
  **Basketball-Reference** serves robots.txt 200 and `User-agent: *` carries
  **`Disallow: */gamelog/`** — which is exactly the brief's BBRef candidate
  ("player game logs mark DNP/Inactive with reasons"). **Dropped.** (The
  prohibition is path-specific: `/leagues/NBA_<yr>_coaches.html`, which D172
  already caches, stays allowed. BBRef box-score pages are allowed but list
  inactives WITHOUT reasons, i.e. strictly dominated by `game_inactives`.)
  **web.archive.org** publishes **no robots.txt (HTTP 404 = unrestricted under
  RFC 9309)** and its CDX index is a documented public API — **permitted**, and
  used at 1 request at a time, >=1.1s apart, honest UA, everything cached to
  disk so a re-run costs the archive nothing.
  (2) **`boxscoresummaryv2.InactivePlayers` HAS NO REASON COLUMN** — headers
  are PLAYER_ID/FIRST_NAME/LAST_NAME/JERSEY_NUM/TEAM_ID/TEAM_CITY/TEAM_NAME/
  TEAM_ABBREVIATION. It is exactly `game_inactives`. The brief's suspicion
  confirmed; that avenue is closed.
  (3) **BUT A SIBLING ENDPOINT WE HAD ALREADY DOWNLOADED CARRIES REASONS BACK
  TO 1996-97.** `data/raw/nba_api/boxscoretraditionalv3/` — **36,596 cached
  files, 36,595 distinct game_ids** already on local disk — gives every player
  a `comment`, the NBA's own post-game DNP/DND/NWT memo:
  `DNP - Coach's Decision` 121,893 / `DND - Injury/Illness` 4,927 /
  `DND - Sprained Left Ankle` 410 / `DND - Rest` 351 / `DNP - Rest` 95 /
  `NWT - Rest` 77, plus a long body-part tail. **Coverage is SYSTEMATIC on all
  30 seasons at 93.3%-97.5% of completed 002 games** — nine seasons deeper than
  `game_inactives`, twenty-two deeper than the report PDF. **Validity is near
  perfect**: of commented rows, DNP 9 of 125,728 logged minutes (0.007%),
  **DND 0 of 20,061, NWT 0 of 6,392**.
  (4) **AND IT IS USELESS FOR THE THING WE NEEDED, FOR A STRUCTURAL REASON.**
  **The comment population and `game_inactives` are DISJOINT — exactly 0
  overlap across all 26,453 DND/NWT rows in all 30 seasons.** From 2006-07 the
  NBA moved did-not-dress players OUT of the box score's `players` array and
  into the separate `InactivePlayers` set. Per-game rates show the handover:
  avg inactives/game 0.00 (1996-97..2005-06) -> 3.2-5.3 -> 6.3-9.0, against
  avg DND+NWT comments/game 1.5-1.8 -> 0.5-1.2 -> 0.3-1.0. **So the box-score
  memo attaches a reason to 0.00% of `game_inactives` rows in every one of the
  20 seasons.** It also carries **NO advance notice** (post-game) and **NO
  probabilistic status**. A NEW DATA-QUALITY FACT falls out: **the NBA changed
  memo convention at 2017-18** — specific body-part memos run 643-1,434/season
  BEFORE it and 57-147/season after (generic `Injury/Illness` 11-85 -> 468-1,156).
  **The OLD era has BETTER box-score reason granularity than the modern one**,
  the inverse of the report gap.
  (5) **THE WAYBACK ARCHIVE IS SYSTEMATIC FROM 2013-14 AND SPORADIC BEFORE.**
  C48 = share of NBA game days with a distinct-content snapshot in the prior
  48h (`collapse=digest`, statuscode 200):
```
             07-08 08-09 09-10 10-11 11-12 12-13 13-14 14-15 15-16 16-17 17-18
  cbssports     0%    2%    0%    2%  6.8% 11.1% 80.9% 67.3% 85.6% 91.0% 30.3%
  usatoday       0     0     0     0     0     0  9.5% 77.4% 77.6% 90.0% 96.0%
  UNION         0%    2%    0%    2%  6.8% 11.1% 80.9% 83.4% 90.0% 93.5% 96.0%
```
  2012-13's honest union over EVERY structured table (cbs+espn+yahoo+donbest)
  is **40.7% at 48h**. espn/yahoo/fox/realgm/rotowire never clear the bar.
  **Rotoworld is a trap**: n=65 snapshots in 2012-13 looks best-in-class, but
  each archived page carries ~8 blurbs spanning ~20 minutes — under 1% of the
  news stream; excluded from every union above. **So the 11 old seasons split
  6/5: 2007-08..2012-13 have no systematic archive at all.**
  (6) **THE PILOT. IT IS GOOD DATA, AND IT IS THE ONLY SURFACE THAT CARRIES
  ALL THREE CRITERIA.** `d175_wayback_{fetch,parse,validate}.py`; 418 archived
  pages over three seasons, server-rendered `<table class="data">`, parsed with
  `Updated | Player | Pos | Injury | Expected Return` and a CBS player-id link.
  2015-16: 120/120 snapshots, 7,728 rows, 30 teams, 373 players, name-match to
  `nba_players` 94.97%. **Reason non-empty 100.00%. News-break `Updated` date
  non-empty 100.00%, and >=1 day before the game on 90.63% (median 2d)** —
  i.e. genuinely EARLIER than the 5PM report. Snapshot lead vs a 19:00 ET tip
  median 14.6h, >=5h on 95.65%. **The status ladder is monotone and is real
  information** (most-recent-snapshot per (player,game), n=3,149):
```
  status        n     % played    status        n     % played
  PROBABLE     42      88.10      DOUBTFUL     24      12.50
  GTD         641      44.31      OUT        1617       7.11
  QUESTIONABLE 307     35.83      OUT_SEASON  512       0.00
```
  Agreement on OUT+OUT_SEASON: **did NOT play 94.60% (2015-16) / 96.30%
  (2016-17)**, n=2,129 / 2,704. **REASON COVERAGE OF `game_inactives`** (the
  brief's fraction): **36.40% / 48.88%** of all inactive rows, rising with
  importance to **56.43% / 75.12%** at mpg>=25.
  (7) **TWO PARSER BUGS OF MINE, BOTH CAUGHT BY RUNNING A SECOND SEASON, BOTH
  RECORDED BECAUSE THE FIRST VERSION OF THIS ENTRY'S VERDICT RESTED ON THEM.**
  (a) **CBS changed wording between seasons** — 2015-16 writes `Out until at
  least Feb 15`, 2016-17 writes `Expected to be out until at least Feb 1`; my
  classifier matched only the first, so **3,209 of 6,001 2016-17 rows fell into
  OTHER and the season parsed with ZERO `OUT` rows** and moved 3 games. (b)
  **carried-forward rows never expired**: CBS states its own return date, and a
  row saying "out until at least Nov 25" read on Dec 5 is not a claim the
  player is out; `until_date` is now parsed (48.5%/52.3% of rows) and a row
  stops asserting OUT once its date passes. On 2015-16 the fix moved the arm
  from **+0.490pp to +0.220pp** — the source deserved its best shot and this
  entry gives it.
  (8) **THE PRICE, THREE SEASONS, SEASON-CLUSTERED.** CBS arm =
  `cbs_out UNION inactives` fed to `k19_t2.season_run` **UNMODIFIED**, so it is
  the exact analogue of `T2 = report_out UNION inactives`; baseline `t2i` =
  inactives only, same process, same DB state. **HARNESS CHECK: every baseline
  reproduces the certified per-season number exactly** (2014-15 +9.80%,
  2015-16 +5.07%, 2016-17 +7.91%). Point-in-time discipline: for game date D
  we use the latest snapshot whose own target date is <= D, where a snapshot
  before 17:00 ET targets that day's slate and one after targets the next day.
```
  season   snaps  C48     t2i     CBS arm   DELTA      outs/tm base->CBS  games chg
  2014-15   109  67.3%   +9.80%   +9.90%  +0.100pp      0.415 -> 0.528      237
  2015-16   120  85.6%   +5.07%   +5.29%  +0.220pp      0.375 -> 0.526      260
  2016-17   189  91.0%   +7.91%   +5.88%  -2.030pp      0.437 -> 0.546      210

  mean -0.570pp   sd 1.266   se 0.731   t(2) = -0.780   p = 0.5171
  95% CI [-3.714, +2.574] pp    median +0.100pp    WRONG SIGN 2 of 3
  leave-one-season-out means: -0.905 / -0.965 / +0.160  (SIGN FLIPS on 2016-17)
```
  **The interval spans the benchmark (-0.741pp), zero, and the opposite sign,
  and the entire pooled effect is one season.** A pre-registered prediction
  (written into `data/unofficial_injury_notes.md` BEFORE running the third
  season) said 2014-15, the sparsest archive, should come in HARMFUL and WORSE
  than 2015-16: **correct on sign, wrong on ordering** (+0.100 vs +0.220), so
  the "density law" is only half supported and is recorded as such.
  (9) **THE FINDING THAT OUTLASTS THE VERDICT: OUT-SET VALUE IS NON-MONOTONE
  IN COMPLETENESS, AND OLD-ERA HEADROOM IS 2.4-3.5x THE REPORT.**
  `d175_oracle_bound.py` sets the out-set to every player who logged 0 minutes
  in that exact game — maximal leakage, a BOUND not a result:
```
  season     t2i    outs/tm    PERFECT-ORACLE   outs/tm     DELTA
  2012-13   +7.06%   0.319         +5.30%        1.306    -1.760pp
  2015-16   +5.07%   0.375         +2.46%        1.339    -2.610pp
  2023-24  +16.97%   0.974        +13.22%        1.794    -3.750pp
```
  **So the old-era gap is NOT "there is nothing left to know" — the headroom is
  REAL and 2.4-3.5x the report's -0.741pp.** But a PARTIAL feed landing at
  0.526-0.546 between a 0.375-0.437 baseline and a ~1.3 optimum can be WORSE
  THAN NOTHING. **A half-filled out-set is not half a good.** That is the
  brief's "sporadic data is worse than a clean gap" risk, measured rather than
  asserted, and it generalises beyond this source.
  (10) **WHY THE OFFICIAL REPORT HELPS AND THIS DOES NOT — MARGINAL-SET
  PRECISION.** Because `game_inactives` is already complete and correct
  (D171: `viol`=0 on all 19 seasons), a pregame source can ONLY add names that
  are not on it, so the marginal set carries the entire effect:
```
                     OUT rows   already in inactives   marginal   of marginal, PLAYED
  CBS      2015-16      2,129          74.78%           25.22%         21.42%
  OFFICIAL 2022-23      8,049          94.47%            5.53%          3.37%
  OFFICIAL 2023-24      8,928          92.79%            7.21%          1.24%
```
  The report adds a SMALL (5.5-7.2%), ~98%-true increment on top of a DENSE
  modern base. The archive adds a LARGE (25.2%), ~79%-true increment on top of
  a THIN old base. **But precision is not the whole story either:**
  `d175_headroom.py` re-ran 2015-16 with every false positive REMOVED BY ORACLE
  (276 names) and recovered only **+0.490 -> +0.440pp** — 0.05pp of a 0.49pp
  loss. The surviving marginal names are real rotation players (median season
  22.4 mpg, 38.4% at mpg>=25), so the damage is the DENSITY effect of (9), not
  merely bad names.
  (11) **THE TANK COMPOSITE, COMPONENT (c) — COMPUTABLE, AND STILL BLOCKED.**
  (c) counts rest/management/maintenance Out listings in the trailing 14 days.
  Reconstructed per team-date:
```
  source / season           team-dates   nonzero    mean    max
  CBS       2015-16              2,460    10.65%   0.265     14
  OFFICIAL  2022-23              2,460    21.10%   0.517     14
  OFFICIAL  2024-25              2,460    34.47%   1.141     19
```
  **So (c) DOES become computable for an old season from the archive** —
  non-degenerate, ~50% of the official 2022-23 event density. (The box-score
  route is far weaker: 58 mpg>=25 rest events in 2015-16 vs CBS's 153. And in
  2003-04..2012-13 the box-score rest vocabulary is 0-4 rows/season, which is
  mostly TRUE — league-wide load management is a modern practice; the official
  feed itself runs 72 rest events in 2018-19 vs 486-629 in 2024-25/2025-26.)
  **BUT IT CANNOT ENTER THE MODEL AND THIS ENTRY DOES NOT MAKE IT:**
  `tanking.PINNED_SEASON_FLOOR = "2020-21"` (D155), so the composite is not
  built for 2015-16 at all, and `tanking.py` states in terms that moving the
  pinned floor "is a MODEL CHANGE and needs a gate + re-cert; it is not a
  bookkeeping edit". **Component (c) becomes computable; the tank term does not
  become active; the season's gap cannot improve by that route without a
  separate gate.** Not attempted — the brief forbids re-certifying.
  (12) **WHAT THE SOURCE IS GOOD AT THAT WE STILL CANNOT USE.** The Q/D/P
  ladder in (6) is genuine, calibrated information about players who MIGHT sit
  — the brief's criterion 3, and the one thing neither `game_inactives` nor the
  box-score memo supplies. **The model has no channel to consume it**: the
  out-set interface is BINARY, so a 36%-likely-to-play QUESTIONABLE enters as a
  hard OUT. Building a probabilistic-availability channel is a MODEL CHANGE
  requiring a gate. **Logged as a future direction, NOT attempted.**
  (13) **ERA STATEMENT (GATE_POLICY_V2 §10).** Source census: all 30 seasons
  1996-97..2025-26 for the box-score memo; the Wayback probe covers 2007-08..
  2017-18. Priced frame: three seasons **2014-15, 2015-16, 2016-17**, 1,230
  games each, chosen as archivable-and-old; the oracle bound adds 2012-13 and
  2023-24. **CLUSTERING: season, K=3 — and K=3 is why nothing here is
  actionable; every number in (8) should be read with that in front of it.**
  COVID seasons never enter. AVAILABILITY TIER: the baseline arm is **T2i** on
  every priced season (correct — no report feed exists there), and the CBS arm
  is a NEW UNCERTIFIED tier that this entry deliberately does not name or
  register. `TANK_SEASON_FLOOR=2020-21` on every run, matching k19_t2's
  documented environment.
  (14) **WHAT THIS ENTRY DOES NOT CLAIM.** It does NOT claim the archive is
  worthless — 2016-17 alone returns -2.030pp, 2.7x the benchmark, and the data
  is demonstrably accurate (96.3% agreement, 100% reason coverage, a real
  news-break date). It claims only that **at K=3, with 2 of 3 seasons the wrong
  sign and the pooled sign flipping on one season's removal, there is no
  evidence to act on.** It does not create a table, ingest a row, name a new
  tier, change a default, run a gate, touch a certified artifact, or move the
  pinned tank floor. It does not re-certify or widen the eval corpus. It does
  not pilot 2013-14 or 2017-18 — **2017-18 is the highest-value unpiloted
  season (it abuts the 2018-12-17 floor) and needs the USA Today parser, since
  CBS collapses to C48 30.3% there; those two seasons would take K from 3 to 5
  for ~400 more archived pages, all free and permitted. THAT IS THE OWNER'S
  CALL, NOT A CONCLUSION OF THIS ENTRY.** **RECOMMENDATION: KEEP THE POST-2018
  FOCUS.** For 2007-08..2012-13 the gap is PERMANENT (no systematic free
  archive exists); for 2013-14..2017-18 the data exists, is permitted, is good,
  and is not measurably worth ingesting on the evidence in hand.
  [code scripts/d175_wayback_fetch.py, d175_wayback_parse.py,
   d175_wayback_validate.py, d175_price_cbs.py, d175_headroom.py,
   d175_oracle_bound.py (all NEW);
   data/unofficial_injury_notes.md (full working, checkpointed as the run
   proceeded, including the RETRACTED CHECKPOINT-10 verdict and the
   pre-registered CHECKPOINT-13 prediction);
   data/raw/unofficial/v3_comments.csv, v3_comment_vocab.json,
   v3_coverage_by_season.csv;
   data/raw/unofficial/wayback/ — cdx indices + 418 archived pages
   (cbs_2014_15/ 109, cbs_2015_16/ 120, cbs_2016_17/ 189) + *_rows.csv +
   cbs_2015_16_joined.csv.gz, cbs_2016_17_joined.csv.gz;
   data/d175_price_cbs_{2014_15,2015_16,2016_17}.json, d175_pooled.json,
   d175_headroom_cbs_2015_16.json, d175_oracle_bound.json,
   d175_cbs_{2015_16,2016_17}_validate.json;
   data/logs/d175_fetch_{1415,1617}.log;
   docs/ERAS.md, docs/STAT_INVENTORY.md updated;
   NO nbapred/ FILE MODIFIED; scripts/k19_t2.py IMPORTED, NOT EDITED AND NOT
   RE-RUN in its own right; NO GATE RUN; NO PRODUCTION DEFAULT CHANGED; NO
   CERTIFIED ARTIFACT TOUCHED; DB data/nba.duckdb READ-ONLY THROUGHOUT —
   read_only=True with retry_s=60 on every connect, ZERO writes, no table
   created or altered]

- D176 **THE STRATEGY SPACE RE-OPENED ON THE BACKFILLED MODEL — THREE
  PRE-REGISTERED SELECTORS, TWELVE CELLS, AND NOTHING BEATS THE INCUMBENT.
  THE BETTER DATA DID NOT OPEN A NEW EDGE, AND THE NULL IS WHY THAT IS
  CREDIBLE.** (2026-08-05)
  **THE WARRANT, AND ITS LIMIT.** Every betting rule in this register was
  selected against a model running 3.6-40% DARKO coverage on the pre-2021
  seasons and no availability feed at all. D170/D171 fixed that and D173 re-ran
  everything: the ATS rejection went from **-3.25% SIG NEG to -0.68% ns**, 13
  significantly-negative primary cells became **zero**, and 19-season spread CLV
  **doubled to +0.32001**. By D45 (gate against the SHIPPED baseline) the rules
  were chosen against a different object, so the strategy space may be re-opened
  ONCE. It does not follow that a search is licensed: D164 measured that a blind
  600-cell search manufactures **+16.917 ROI points from pure noise** (null
  +17.464, net **-0.55**, p=0.685) and that capacity scales with cell count
  (6 -> +3.078, 30 -> +7.888, 120 -> +14.764, 600 -> +16.917); D165 measured the
  family-wise burden of 7 procedures at **+2.13 points**. **This entry therefore
  scores TWELVE cells, not six hundred.**
  (1) **PRE-REGISTRATION.** `data/newstrat_prereg.md`, sha256
  **`db293ad8548a2e7099586a66847d6d1a71d8e83e4ab171e0917584980e9f1fc7`**
  (`data/newstrat_prereg.sha256`), written and hashed BEFORE any arm was scored,
  including the MDE80 table and the five-condition decision rule. Substrate is
  `data/ats19_frame.csv.gz` md5 `78bbba1fa9cf14e6dc9563aab73bdc0f` (byte-identical
  to `ats19_frame_D173.csv.gz`, i.e. POST-fix, built from `k19_d171_t2_pergame.csv`
  md5 `657458e52980aad702cc3b63766fa2f0`), 22,742 games, 19 seasons, tier **T2**.
  Registered correction: that JSON's `design` string still reads
  "availability-BLIND", which is stale D162 text — its own `md5` block and its
  non-zero `n_out_*` columns are T2. **THREE arms and ONE control, not the four
  permitted**: the burden rises monotonically in F (+1.32 at F=3 vs +1.63 at F=4
  from D165's stored 200x7 null matrix) and a 4th arm not motivated by a
  REGISTERED measurement buys +0.31 points of free ROI and no knowledge.
  (2) **THE ARMS.** Bet universe, side and price are IDENTICAL across all four —
  ATS at the OPEN, bet home iff `m_us > open_margin`, payoff {+100/110, -1, 0},
  `oc_capacity.payoff_and_masks` verbatim. Only the SELECTION differs.
  `s = +1` on a home bet. Grid for each: top-q fraction, q in {0.50, 0.25, 0.10}.
```
  A0 CONTROL      |m_us - open_margin|            incumbent EDGE logic, benchmark
  A1 CLV-TARGET   s * pred_dm                     D147/D167: 86.5% of our CLV is
                                                  banked pre-report, so select on
                                                  PREDICTED movement, not on edge
  A2 AVAIL-DIV    s * (m_us_T2 - m_us_BLIND)      D156/D159: the feed is now
                                                  complete and is ~2/3 of the CLV
  A3 RETURN       s * (retmin_h - retmin_a)       D133/D145 absence-blindness,
                                                  generalised to the SIDES path
```
  `pred_dm` is rebuilt on all 19 seasons from the D147 primitives
  (`nbapred/market/anchored.py`) because D147's own artifact carries it on only 4
  seasons; TIER A features only, ridge refit every 500 games on all prior games,
  **D147 `assert_pit` two-sided guard: violations `[]`, moved-under-full-shuffle
  `['resid_close','resid_res']` — non-vacuous.** R^2 vs the naive no-movement
  baseline +0.0876 pooled, **+0.109/+0.124/+0.120/+0.140/+0.110 on
  2021-22..2025-26** (D147 registered +0.171 on a richer set and 4 seasons).
  `m_us_BLIND` is `ats19_frame_D173blind.csv.gz`, the tier-matched blind twin
  (divergence mean +0.0099, sd 0.7894, exactly zero on 30.8%). `retmin` sums prior
  mpg of rotation players (mpg>=10 over the trailing 20 team-games) who missed the
  team's most recent game and are NOT on tonight's inactive list — **component (c)
  reads `game_inactives`, the same T2 tier `p_us` consumes but NOT Tier-A-live at
  the open, so A3 is an UPPER BOUND and is labelled one.**
  (3) **DESIGN.** D164 ARM B verbatim (select on seasons [0,k) by pooled ROI,
  eligibility n >= 100k, tie-break larger-n then lower index, freeze, score k,
  roll; pool bet-weighted; interval = equal-weight K-1 dof cluster-mean t).
  PRIMARY = report era 2018-19..2025-26, min_history 3, **scored folds
  2021-22..2025-26, K=5** — the two COVID seasons are TRAINING-ONLY and are never
  scored there. SECONDARY = 19 seasons at best-available tier (**T2i**
  2007-08..2017-18, **T2** 2018-19..2025-26 — never pooled silently, D158),
  min_history 5, **scored folds 2012-13..2025-26, K=14**.
  **MDE80 STATED BEFORE SCORING** (`data/ns_mde80.json`, selector-independent,
  random subsets of the universe payoff): paired **4.51 / 7.61 / 13.58** points at
  q = .50/.25/.10 on PRIMARY, **2.85 / 4.91 / 8.56** on SECONDARY.
  (4) **THE PRE-REGISTERED ANSWER: NO ARM CLEARS, AND ALL THREE LOSE TO THE
  INCUMBENT ON THE PRIMARY FRAME.**
```
  PRIMARY  K=5   universe n=9,469  ROI -1.715%  cover 51.473%  CLV +0.5156
  arm          q-path                n     ROI%   cover     CLV   paired vs A0
  A0 CONTROL   .10 x5              742   +7.83   56.504  +1.872   (benchmark)
  A1 CLV       .25/.25/.10/.10/.10 1475   +6.76   55.951  +1.547   -0.43 [-5.52,+4.67] ns
  A2 AVAIL     .10 x5              680   +6.46   55.804  +2.187   -1.25 [-14.18,+11.69] ns
  A3 RETURN    .10 x5              627   +1.44   53.140  +0.649   -4.54 [-14.52,+5.44] ns
  SECONDARY K=14 universe n=22,742 ROI -0.677%  cover 52.021%  CLV +0.3200
  A0 CONTROL                      2023   +3.95   54.468  +1.108   (benchmark)
  A1 CLV                          6649   +2.90   53.924  +0.746   +0.68 [-3.96,+5.32] ns
  A2 AVAIL                        3351   +0.28   52.530  +1.066   -2.96 [-7.77,+1.85] ns
  A3 RETURN                       3172   -1.24   51.719  +0.361   -4.03 [-10.42,+2.36] ns
```
  Every arm's own K-1 t interval straddles zero on both frames. A1's paired sign
  **FLIPS between frames** (-0.43 / +0.68). Decision-rule conditions (1), (2) and
  (5) all fail. **Nothing here is a result.**
  (5) **THE NULL, AND THE TRAP IT EXPOSES — THE MOST TRANSFERABLE THING IN THIS
  ENTRY.** 400 draws, seed 20260805, two nulls, identical procedure per arm:
  NULL-S permutes the arm's OWN selector within date-slate; NULL-M permutes every
  model-derived quantity jointly within date-slate (D164/D173 convention).
```
  PRIMARY   real   NULL-S    net      p   paired real  NULL-Sp    net      p  FWERp
  A1       +6.76    +0.30  +6.46  0.000        -0.43    -6.72  +6.29  0.003  0.005
  A2       +6.46    -1.25  +7.70  0.003        -1.25    -8.41  +7.16  0.018  0.025
  A3       +1.44    -1.69  +3.13  0.048        -4.54    -8.57  +4.02  0.037  0.215
```
  **ALL THREE ARMS BEAT THEIR OWN NULL AT p <= 0.048 AND ALL THREE SURVIVE BH —
  AND ALL THREE LOSE TO THE BENCHMARK.** A net-of-null of **+6.29** sits on a
  paired estimate of **-0.43**. NULL-S asks "does this selector beat a permuted
  copy of ITSELF", and the answer is yes for all four selectors INCLUDING the
  incumbent; it does not ask whether the arm beats what we already ship.
  **Reporting net-of-null without the benchmark alongside it would have
  manufactured three wins out of three losses.** That is a general defect in the
  register's reporting convention, found here by accident, and it is the reason
  D45's shipped-baseline comparison has to be the PRIMARY statistic and the null
  a SECONDARY one — never the other way round.
  **FAMILY-WISE BURDEN MEASURED AT F=3** by D165's own prescription (all 3 arms
  through the SAME permuted realisation, per-draw max): E[max|null] +1.045% vs
  mean single-arm null -0.881% = **+1.926 points** (PRIMARY), +0.261% vs -0.742%
  = **+1.003 points** (SECONDARY); paired **+2.110 / +1.152**. D165's F=3
  interpolation was +1.32; measured here at +1.00-2.11 on a different object.
  **Every paired estimate above is inside that tie band except A3's, which is
  negative.**
  (6) **THE SELECTION LAYER, NOT THE SELECTOR, IS WHAT LOST — AND THE CONTROL
  CHEATED WITHOUT MEANING TO.** Walk-forward pooled ROI minus the SAME arm with q
  FROZEN at 0.10 (cut-point still learned PIT-safely on [0,k)):
```
  frame        A0          A1          A2          A3
  PRIMARY    +0.00 (1q)  -1.65 (2q)  +0.00 (1q)  +0.00 (1q)
  SECONDARY  +0.00 (1q)  -2.92 (3q)  -0.71 (3q)  -0.54 (2q)
```
  **A0's walk-forward is DEGENERATE — it selected q=0.10 on all 19 folds across
  both frames and paid NO adaptation cost, while A1 paid -1.65 and -2.92.** The
  pre-registered paired statistic is therefore biased AGAINST the arms by an
  amount comparable to the effect sought. This reproduces D165 §3's "within-season
  adaptation costs -8.76 points paired" on a new object at a coarser cadence, and
  it is a design lesson for any future paired arm-vs-control test in this
  register: **a control whose selection is degenerate is not a fair control.**
  (7) **FROZEN-q RE-READ (POST-HOC, LABELLED): A1 IS POSITIVE ON 6 OF 6 CELLS AND
  IS STILL NOT A RESULT.** Paired ROI vs A0, q frozen, all three pre-registered q
  on both frames:
```
  arm    PRIMARY .50/.25/.10    SECONDARY .50/.25/.10   +/6   mean
  A1    +0.92 / +2.14 / +0.88   +0.58 / +0.64 / +2.61   6/6  +1.30pp
  A2    -1.28 / -1.84 / -1.25   +0.16 / -0.29 / -2.49   1/6  -1.16pp
  A3    -2.36 / -8.49 / -4.54   -2.77 / -4.21 / -3.59   0/6  -4.33pp
```
  A1 is the only arm whose own level CI ever excludes zero (PRIMARY q=.25 +6.87%
  [+0.23,+12.86]; SECONDARY q=.10 +5.81% [+1.79,+8.57]) and the only arm
  **sign-consistent across all four scorable eras** on the secondary frame
  (K-B +1.92, K-C +0.18, K-D +0.50, K-E +7.95 -> **ERA-CONDITIONAL**), where the
  INCUMBENT A0 is **ERA-SPECIFIC** and flips (K-B **-8.51**, K-C +0.63, K-D
  +9.14, K-E +6.31). **BUT not one of the six paired intervals excludes zero; the
  mean +1.30pp is BELOW the measured family burden (+1.00..+2.11) and far below
  the paired MDE80 (2.85-13.58); the six cells are NESTED (q .10 in .25 in .50,
  PRIMARY in SECONDARY) so the 6/6 sign-test p=0.0156 is NOT a valid p-value and
  is recorded as a consistency observation only; and A1's paired delta FLIPS SIGN
  under deletion of a single fold on BOTH frames (LOSO [-1.70,+0.84] and
  [-0.41,+1.89]).** A1 is a hypothesis worth carrying to fresh seasons, not a
  finding. **IT DOES NOT SHIP AND NOTHING IS RE-CERTIFIED.**
  (8) **THE ONE CLEAN DISSOCIATION: CLV AND ROI ARE SEPARABLE BY SELECTOR.**
  Same six frozen-q cells, paired CLV vs A0 in spread points:
```
  arm    +/6 CLV   mean CLV    +/6 ROI   mean ROI
  A1       4/6     +0.004        6/6     +1.30pp
  A2       6/6     +0.143        1/6     -1.16pp
  A3       0/6     -0.524        0/6     -4.33pp
```
  **A2 selects bets the market moves TOWARD more than the incumbent does — 6/6,
  mean +0.143 points, peaking at +0.422 on the PRIMARY q=.10 cell — while
  selecting bets that WIN LESS (1/6, -1.16pp). And the explicitly CLV-TARGETED arm
  A1 buys essentially NO extra CLV (+0.004, 4/6) while buying the most ROI.** No
  interval excludes zero, so this is an observation and not a measurement. But it
  is a direct caution about this register's use of CLV as the yardstick: **CLV is
  not a sufficient statistic for bet selection, and a selector can be optimised
  into more CLV without more money.** D167 §12 already qualified D159 in the same
  direction; this is a second, independent route to the same warning.
  (9) **A3 IS DECISIVELY WRONG-SIGNED — THE ONLY UNAMBIGUOUS VERDICT HERE, AND IT
  IS NEGATIVE.** 0/6 on ROI (mean -4.33pp; **SIG NEG** at PRIMARY q=.25, -8.49
  [-16.76,-0.22], and at SECONDARY q=.50, -2.77 [-4.90,-0.64]) and **0/6 on CLV
  with ALL SIX paired CLV intervals excluding zero** (-0.266, -0.632, -1.109,
  -0.170, -0.358, -0.609). Selecting on returning rotation minutes picks bets with
  LESS CLV and LESS ROI. **The D133/D145 absence-blindness insight does NOT
  transfer to the SIDES path in the direction hypothesised**; on this evidence the
  market prices returning rotation minutes MORE efficiently than the games we
  would otherwise be betting. Recorded so it is not re-proposed.
  (10) **V3 BATTERY (GATE_POLICY_V2 §8-§11).** ROLLING-ORIGIN is the walk-forward;
  sign consistency across folds is **FALSE for every arm on both frames**
  (A0 3/5 and 8/14 positive, A1 4/5 and 9/14, A2 4/5 and 7/14, A3 2/5 and 7/14),
  so by §11 tie-break 3 **all four, INCLUDING THE INCUMBENT, are disqualified from
  T1 on this statistic.** LOSO is reported as a stability diagnostic with
  `independent_folds = 1`, never as k proofs (§8.2). ERA: A0 ERA-SPECIFIC, A1
  ERA-CONDITIONAL, A2 and A3 ERA-STABLE at/below zero; the PRIMARY frame is
  entirely K-E so its era decomposition is **UNDETERMINED** and is reported as
  such rather than as stability. BLOCK BOOTSTRAP (7-day blocks, 2,000 draws) on
  the q=.10 cell: A1 [+2.46,+14.67] PRIMARY and [+2.49,+8.85] SECONDARY —
  **the only intervals in the whole run that exclude zero for A1, and §9.1
  explicitly SUBORDINATES the block bootstrap to the season-clustered interval,
  which straddles zero. The shipping statistic says no.** ICC -0.003..+0.006,
  DEFF 0.61-1.89, consistent with §9.2's "sides deltas are dominated by
  irreducible outcome noise". BH across the family: 3/3 rejected on PRIMARY, 1/3
  on SECONDARY — immaterial, see (5).
  (11) **ERA STATEMENT (§10).** Eval universe by era code: PRIMARY = **K-E only**
  (scored 2021-22..2025-26), training K-C+K-D; SECONDARY = **K-B, K-C, K-D, K-E**
  (scored 2012-13..2025-26), training K-A+K-B. ERA-AVAILABILITY: every input
  exists on every scored season — `open_margin`/`close_margin` on all 19,
  `game_inactives` complete to 2006-07 after D170/D171, and **A2/A3 consume the
  inactive list only, NOT `injury_reports_pit`, precisely so that neither arm is
  structurally inert on the secondary frame** (the D110 §1a trap). The 5PM report
  enters only through the frame's own T2 `p_us`, which is why the report era is
  the PRIMARY frame. COVID-FRAME CHECK: **2019-20 and 2020-21 are TRAINING-ONLY
  on the primary frame and are never scored there**; on the secondary they are
  scored inside K-D, where A0 is strongest (+9.14%) — a reason to distrust A0's
  pooled level, not to like it. TIERS: T2 throughout the primary, **T2i
  2007-08..2017-18 + T2 2018-19..2025-26** on the secondary, named and never
  pooled silently. CLUSTERING: season, K=5 (primary) and K=14 (secondary);
  **K=5 is why nothing on the primary frame is actionable and every number there
  should be read with that in front of it.**
  (12) **EXECUTION.** Priced flat -110 at the OPEN. `data/bkp_panel_rows.csv.gz`
  had **NOT** grown when this was priced (493,371 bytes, mtime 2026-08-04 22:35,
  unchanged), so no new Kaggle panel was available and D174's ladder is used as
  registered. Per-season MEASURED/EXTRAPOLATED labels are taken from
  `bkp_ladder.json`'s `repriced.rows` (**10 MEASURED / 4 EXTRAPOLATED**), **not**
  from D174 §15's prose, which the artifact contradicts — the primary frame's five
  scored seasons are 2021-22 EXTRAPOLATED, 2022-23 EXTRAPOLATED, 2023-24 MEASURED,
  2024-25 MEASURED, 2025-26 MEASURED. Level twin with the k=5 outlier-realism
  haircut (dROI = 1.909 * 0.0317276 = 0.06057 per spread point): PRIMARY A0
  +7.83 -> +9.43, A1 +6.76 -> +8.29, A2 +6.46 -> +8.00, A3 +1.44 -> +3.05;
  SECONDARY A0 +3.95 -> +5.14, A1 +2.90 -> +3.93, A2 +0.28 -> +1.49, A3 -1.24 ->
  -0.22. **Uniform across arms; it cannot and does not move any paired contrast.**
  (13) **WHAT ACTUALLY CHANGED IS THE BASELINE, NOT THE FRONTIER.** On the fixed
  data the INCUMBENT edge selector, walk-forward at q=0.10, returns **+7.83%** on
  the report era (K=5, CI [-6.11,+19.16]) and **+3.95%** on 19 seasons (K=14, CI
  [-2.20,+7.34]) — positive, ERA-SPECIFIC, rolling-origin sign-inconsistent, and
  underpowered on both. D173 registered that transition; **this entry adds that
  the space immediately around it does not contain a detectable improvement.**
  (14) **WHAT THIS ENTRY DOES NOT CLAIM AND DOES NOT DO.** It does NOT claim the
  three hypotheses are false — only A3 is affirmatively wrong-signed. It does NOT
  claim A1 is worthless; it claims A1's +1.30pp sits below the family burden and
  the MDE80 and flips under one-fold deletion. **A null result at K=5 with a
  paired MDE80 of 4.51-13.58 points is a statement about DETECTABILITY, not a
  proof of absence; the binding constraint is K and the only instrument that
  relaxes it is fresh seasons (§9.4).** No production default changed. No gate
  run. No re-certification. The eval corpus is not widened. No table created or
  altered. `data/capstone_pergame.csv` md5 `695d40a3545e889267cad403b7acdce8`
  VERIFIED UNCHANGED before and after. `nbapred/` NOT MODIFIED — `anchored.py`
  and `oc_capacity.py` were IMPORTED, not edited. DB `data/nba.duckdb` READ-ONLY
  throughout (`read_only=True`), ZERO writes.
  [code scripts/ns_features.py, ns_score.py, ns_battery.py (all NEW);
   data/newstrat_prereg.md (+ .sha256
   db293ad8548a2e7099586a66847d6d1a71d8e83e4ab171e0917584980e9f1fc7),
   data/newstrat_notes.md (full working, checkpointed as the run proceeded,
   including the MDE80 table fixed BEFORE scoring);
   data/ns_features.csv.gz (md5 531ad048ff5ca02d763d65d130753499),
   data/ns_mde80.json, data/ns_score.json, data/ns_battery.json;
   data/logs/ns_score.log, data/logs/ns_battery.log;
   INPUTS ats19_frame.csv.gz 78bbba1fa9cf14e6dc9563aab73bdc0f,
   ats19_frame_D173blind.csv.gz 154226d46e1b4937089a4f68e8060de9,
   k19_d171_t2_pergame.csv 657458e52980aad702cc3b63766fa2f0;
   NO nbapred/ FILE MODIFIED; NO GATE RUN; NO PRODUCTION DEFAULT CHANGED;
   NO CERTIFIED ARTIFACT TOUCHED; NOTHING SHIPS]

- D177 **KAGGLE SWEPT TO EXHAUSTION FOR THE LAST BOOK-PANEL GAP, AND THE ANSWER
  IS NO. 1,613 DATASETS ENUMERATED; NOT ONE CARRIES PER-BOOKMAKER NBA ODDS FOR
  2019-20..2022-23. THE HOLE IS NOT CLOSED AND ON THIS ROUTE IT CANNOT BE.**
  WHAT KAGGLE DID YIELD IS **ONE REAL BOOK** — BetMGM's close, 2021-22..2025-26
  (`caseydurfee`) — WHICH IS **NOT A PANEL AND IS NEVER CALLED ONE**, BUT WHICH
  ANCHORS THE EXTRAPOLATION FROM INSIDE THE HOLE FOR THE FIRST TIME:
  **D166's FORWARD-CARRIED LAW (k=5+HC = 0.4261) WAS OPTIMISTIC BY ~1.25x ON
  2021-22/2022-23 (ANCHORED 0.3397 / 0.3398), THE SAME DIRECTION AS D174's
  1.7-2.0x AT THE MODERN END BUT MILDER — SO THE LAW WAS TOO GENEROUS AT BOTH
  ENDS OF THE RECORD AND THE MIDDLE IS THE LEAST WRONG PART OF IT.**
  **THE EQUITY PATH DOES NOT MOVE: NO SEASON BECOMES MEASURED, SO THE OFFICIAL
  FIGURE STAYS D174's +3.13% / +48.6u AND THE RECORD STAYS 9 OF 14 MEASURED.**
  A FOURTH INSTANCE OF THE D171 NAME BUG CLASS WAS CAUGHT — LOUDLY, BY DESIGN.
  DIAGNOSTIC. No production default changed, no gate run, eval corpus NOT
  widened, `scripts/bet_engine.py` and `scripts/prod_by_season.py` UNTOUCHED and
  NOT RUN, no walk-forward re-run. DB `data/nba.duckdb` **READ-ONLY THROUGHOUT**
  — `read_only=True` with `retry_s=60`, ZERO writes, no table created or
  altered. `nbapred.threads.pin(1)` before numpy. (2026-08-05)
  (1) **THE BRIEF, AND WHY KAGGLE WAS THE ONLY ROUTE LEFT.** D174 §11 put
  2019-20..2022-23 on the record as EXTRAPOLATED for a **legal, not technical**
  reason: Action Network's robots.txt is `Disallow: /`, and ESPN's core API has
  the data but Disney's ToS forbids extraction while `espn.com/robots.txt` names
  **`AI-crawler user-agents / Disallow: /`**. Kaggle is public, freely downloadable and
  permitted, so it is the remaining route. **IT WAS TAKEN, IT WAS SWEPT TO
  EXHAUSTION, AND IT DOES NOT CONTAIN THE DATA.**
  (2) **ACCESS: THE CHROME ROUTE IS DEAD, THE ANONYMOUS PUBLIC API IS NOT.**
  `kaggle_web.py::_find_session` raises `no logged-in Kaggle Chrome profile
  found`. But Kaggle's own `/api/v1/` needs **no auth at all** — `datasets/list`
  (search), `datasets/view` (metadata) and the **full `datasets/download` zip**
  all return 200 anonymously (72 MB pulled with no cookie). `kaggle_web.py` now
  falls back to it (`_anon_session`/`_session`) and exposes `search()`/`view()`
  so a candidate is SCREENED BEFORE it is downloaded. **Two gotchas recorded so
  nobody re-finds them:** the documented `datasets/list/files/{owner}/{slug}`
  endpoint **404s**, and `view`'s `files` array comes back **EMPTY** — the only
  reliable way to see file names is to download the zip and read its namelist.
  kaggle.com serves **no robots.txt** (`/robots.txt` soft-404s to the SPA shell,
  HTTP 404) and `/api/v1/` is Kaggle's own documented API whose entire purpose
  is dataset download. No other host was contacted by this entry.
  (3) **THE SWEEP.** 88 search terms; deep paging on `nba`/`basketball`/`odds`/
  `betting`/`sportsbook`/`bookmaker`/`spread`/`moneyline`/`wager` to exhaustion;
  and a **per-user enumeration of all 17 publishers** who have ever posted an
  odds file (`christophertreasure, thedevastator, oliviersportsdata, eddieglush,
  erichqiu, ehallmar, cviaxmiwnptr, chevronronson, zachht, ronaldoaf,
  caseydurfee, visualize25, giardinidavide, changewire, sharpapi, austro,
  mexwell`). **1,613 distinct datasets; 140 betting-related; 32 both-basketball-
  and-betting, and every one of those 32 was opened or its header read.**
  (4) **THE EVALUATION TABLE. THE DECISIVE COLUMN IS PER-BOOK, AND THE DECISIVE
  TRAP IS THAT AN AVERAGE OF BOOKS IS NOT A PANEL OF BOOKS.**
```
  dataset                                  seasons        per-BOOK?  OPEN? lic       verdict
  thedevastator/uncovering-hidden-…-20     2014-10..22-12 PROPS only  no   CC0       NO
  caseydurfee/mgm-grand-nba-betting-data   2021-22..25-26 1 book      no   CC BY-SA  USED, not a panel
  zachht/wnba-odds-history                 2025-09+       1 book(Pinn) n/a CC0       NO
  visualize25/basketball-betting-dataset   ..2023         no         YES   Unknown   NO
  giardinidavide/nba-odds                  2021-2022      no          no   Unknown   NO (oddsportal)
  bhavishsalia/nba2019odds                 2018-19        no          no   Unknown   NO
  hardikbhalekar/global-sports-…-2026      2026-06+       YES         no   CC BY-SA  NO (zero NBA rows)
  oliviersportsdata/us-sports-master…      1998-2026      no          no   CC BY-NC  NO (50-row teaser)
  dontefarquharson / cactusmann / marcuslin                                          NO
  ehallmar/nba-historical-stats-…          2006-07..17-18 YES         no   Unknown   held; FROZEN v1
  erichqiu/nba-odds-and-scores             2012-13..18-19 YES        YES   CC0       held; FROZEN v3
  chevronronson / cviaxmiwnptr / christophertreasure — re-verified single-consensus
```
  **THE NEAR MISS, AND IT IS INSTRUCTIVE.**
  `thedevastator/uncovering-hidden-trends-in-nba-betting-lines-20` (CC0, 72 MB,
  **Dec-2021 to Dec-2022 — squarely inside the hole**) advertises comparing
  "betting lines from a variety of sports books", ships a file literally called
  `slackr_output_arbs.csv`, and **does** carry per-book columns
  (`br_/bs_/csr_/dk_/fd_/mgm_/pb_` = BetRivers/…/Caesars/DraftKings/FanDuel/
  BetMGM/PointsBet). **All of it is on FANTASY-POINTS and FIRST-TEAM-TO-SCORE
  PROP markets, and all of it is ONE SINGLE DAY, 2022-12-12 (70 rows).** Its
  game-level `nba_YYYYMMDD_spread.csv` files carry one `home_line` per game
  whose values give the game away — `-503.3333`, `219.8333`, `7.66667` — **they
  are cross-book MEANS.** A consensus computed FROM books is still a consensus.
  (5) **THE BEST LEAD IN THE HUNT, KILLED ON EVIDENCE RATHER THAN ASSUMPTION.**
  The only two NBA sets on Kaggle with a real per-book schema are the two we
  already own. `ehallmar` is **v1, 2018-08-26, never updated**. `erichqiu` is
  **v3, 2020-05-02 — SEVEN WEEKS AFTER THE COVID SHUTDOWN**, so a scrape made
  then could plausibly have carried the truncated 2019-20 season and would have
  closed a quarter of the hole for free. **It was re-downloaded and diffed:
  BYTE-IDENTICAL ZIP (1,961,317 B), same 21-file namelist, last folder
  `2018-19/`.** Dead, and dead for a stated reason.
  (6) **A LIMIT OF THE SEARCH, RECORDED RATHER THAN GLOSSED.** Kaggle's
  `datasets/list?search=` indexes **title and subtitle ONLY, not descriptions** —
  probes for `jimtheflash`, `gambling_stuff` and `Open_Line_Spread` (strings
  present only in description bodies) each return **0 hits**. A per-book NBA
  panel published under a title naming neither the sport nor the market would be
  invisible to this sweep. Mitigated by the per-user enumeration and by paging
  the broad terms to exhaustion; **not eliminated, and the entry does not
  pretend otherwise.**
  (7) **WHAT WAS INGESTED, AND THE SENTENCE THAT GOVERNS IT: IT IS ONE BOOK.**
  `caseydurfee/mgm-grand-nba-betting-data` (CC BY-SA 4.0, 6,081 rows,
  2021-10-19..2026-02-12) is **BetMGM only**, scraped from Yahoo's internal JSON.
  **k=1 COMPUTES NO LADDER. THE BRIEF'S "5 BOOKS THAT ARE ONE FEED IS WORTH
  NOTHING" QUESTION DOES NOT ARISE, BECAUSE THERE ARE NOT 5 BOOKS — THERE IS 1,
  AND THIS ENTRY NEVER CALLS IT A PANEL.** It is ingested because it is the only
  free object that places a NAMED, VERIFIED operator inside the hole.
  (8) **THE FOURTH INSTANCE OF THE D171 BUG CLASS, AND THE DESIGN WORKED.** The
  feed names every side by **BARE CITY** ("Atlanta", "Utah", "Golden State").
  Through `nbapred/teams.py` as it stood, **28 of 30 franchises were
  UNRESOLVABLE** — and `resolve_map` REPORTED all 28 with row counts instead of
  silently deleting ~5,700 rows, which is precisely what D171 built it to do.
  Fix: a **CITY rule added LAST in `resolve()`**, so it is **strictly additive**
  — it can only fire where the three older rules already returned None and
  therefore cannot change any name this module resolved before.
```
  30 / 30 distinct strings resolved, to 30 DISTINCT franchises, 0 unresolved.
  'Los Angeles' STILL returns None, deliberately — two franchises share it.
  'Golden State' needs an explicit alias: nba_api's `city` for GSW is
     "San Francisco", so the city rule alone would have missed exactly one team.
```
  Regressions: `tests/test_teams_canon.py::test_city_rule_resolves_every_yahoo_mgm_spelling`
  (asserts **30 DISTINCT** abbrevs, because a rule that collapsed two cities
  would still "resolve" 30 strings), `::test_ambiguous_city_stays_unresolvable`,
  `::test_city_rule_is_strictly_additive`.
  (9) **MATCH RATE, BOTH DIRECTIONS, AND A MEASURED NO-OP.**
```
  MGM -> spine   6,077 / 6,079 rows carrying a spread            99.97%
  spine -> MGM   6,077 / 6,606 odds_market games 2021-22..25-26  92.0%
                 (shortfall is 2025-26 stopping at the All-Star break)
  per season  2021-22 1,329 | 2022-23 1,302 | 2023-24 1,318
              2024-25 1,320 | 2025-26   808
```
  **D174's +/-1 DAY TOLERANCE CONTRIBUTED EXACTLY ZERO — all 6,077 matched at
  offset 0.** Unlike ESPN (UTC), Yahoo's dates are already ET. The tolerance is
  kept in the code because it is free; that it is a no-op **here** is now
  measured rather than assumed.
  (10) **VALIDATION AGAINST WHAT WE ALREADY HOLD.**
```
  season   n     corr(mkt) MAD(mkt) bias   tie(mkt) MAD(open) tie(open) tie(close)
  2021-22 1329   0.9975    0.2961  +0.006   49.66%   1.5740    14.04%    49.74%
  2022-23 1302   0.9928    0.3184  +0.079   58.83%   1.7891    13.90%    53.04%
  2023-24 1318   0.9959    0.3001  +0.003   73.22%   1.3367    19.20%    53.72%
  2024-25 1320   0.9952    0.2924   0.000   75.23%   1.9129    22.65%    74.77%
  2025-26  808   0.9863    0.5186  -0.019   56.68%   1.4709    25.37%    84.41%
```
  corr 0.986-0.998 and |systematic offset| <= 0.08 pt: a real NBA spread on our
  games, not mis-keyed and not sign-flipped (a flip would show ~5+).
  **WHICH PHASE? MEASURED, NOT ASSUMED: MAD to our CLOSE is 0.18-0.30 and to our
  OPEN 1.34-1.91. IT IS THE CLOSE.**
  (11) **INDEPENDENCE — THE TIE-RATE LADDER APPLIED TO THE ONLY QUESTION IT CAN
  ANSWER HERE.** With k=1 there is no cross-operator question, so the test run
  is D174 §3's **same-operator control**: this scrape against the panel's OWN
  `mgm` rows — one book, two independent scrapers.
```
  2023-24 CLOSE  n=1229  tie 89.10%  MAD 0.2107
  2025-26 CLOSE  n= 100  tie 89.00%  MAD 0.1600
  2024-25 CLOSE  n=  75  tie 77.33%  MAD 1.2667   (thin; AN's MGM is sparse)
  2023-24 OPEN   n=1196  tie 33.03%  MAD 1.3094   <- the phase control
```
  Against D163/D174's ladder — **known duplicate 100.00% with MAD exactly
  0.0000; Caesars state skins 91-94%; genuine cross-operator 36.52%** — 89.10%
  and 89.00% sit squarely in D174's "independent scrapers of the SAME operator"
  band (81-96%). **VERDICT: A GENUINE BetMGM CLOSE, INDEPENDENTLY CAPTURED;
  NEITHER A RELABELLED COPY OF A FEED WE ALREADY HOLD NOR A DIFFERENT BOOK
  WEARING MGM's NAME.**
  (12) **THE LADDER (TASK 4): NO SEASON BECOMES MEASURED, AND NO LADDER IS
  INVENTED.** A best-of-k ladder needs >= 2 books. What one book gives against a
  consensus is a SCALE statistic `m1 = mean|book - consensus close|`, and the
  ladder is a function of scale — so the ratio is FITTED where both are MEASURED
  and only then read off. `ladder_vals` is **IMPORTED from `scripts/bkp_ladder.py`,
  not reimplemented**, so every cell is D163/D174-comparable.
  **HARNESS ANCHOR: THE CALIBRATION REPRODUCES D174 EXACTLY** — 2023-24 k2/k3/k5/k8
  = 0.2569 / 0.3854 / 0.5569 / 0.7265 (D174 §6, identical); 2024-25 k5+HC 0.2527
  and 2025-26 k5+HC 0.2154 (D174 §7, identical); 2018-19 k2 0.1107 (D174 §10).
```
  season  games books  m1_mean m1_trim |  k2     k3     k5     k8  |  k2_hc  k3_hc  k5_hc  k8_hc
  2012-13  1313   5    0.1813  0.1552   0.1175 0.1762 0.2610 0.2610  0.1013 0.1519 0.2211 0.2211
  2013-14  1319   5    0.1632  0.1556   0.1013 0.1519 0.2206 0.2206  0.0925 0.1388 0.2000 0.2000
  2014-15  1309   5    0.1647  0.1527   0.1136 0.1704 0.2511 0.2511  0.1045 0.1567 0.2294 0.2294
  2015-16  1315   5    0.2058  0.1940   0.1186 0.1778 0.2587 0.2587  0.1118 0.1677 0.2428 0.2428
  2016-17  1309   5    0.2317  0.2014   0.1223 0.1834 0.2678 0.2678  0.1110 0.1664 0.2401 0.2401
  2017-18  1312   5    0.2563  0.1952   0.1631 0.2446 0.3706 0.3706  0.1052 0.1577 0.2269 0.2269
  2018-19  1307   5    0.1929  0.1849   0.1107 0.1660 0.2366 0.2366  0.1040 0.1559 0.2223 0.2223
  2023-24  1319  11    0.4428  0.3096   0.2569 0.3854 0.5569 0.7265  0.1610 0.2415 0.3381 0.4221
  2024-25  1291   5    0.2840  0.2461   0.1702 0.2550 0.3118 0.3118  0.1435 0.2147 0.2527 0.2527
  2025-26  1265   6    0.4131  0.2311   0.1431 0.2140 0.2350 0.2350  0.1331 0.1985 0.2154 0.2154
```
  **THE TRIMMED SCALE HAD TO BE USED, AND THAT WAS NOT ASSUMED — IT IS THE ONE
  PLACE THIS METHOD NEARLY BROKE.** With a plain mean, 2025-26's `m1` blows up
  to 0.4131 against a MEASURED k5+HC of 0.2154 and **the harness misses by
  147%**. Applying **D163's OWN 1.5-pt outlier-haircut rule to `m1`** — the same
  rule the ladder it predicts already uses, so numerator and denominator are
  haircut alike — collapses that: `k5_hc/m1_trim` **1.2071, cv 0.142, K=10, 95%
  CI [1.0842, 1.3300]** against `k5_hc/m1_mean` 1.0265, **cv 0.254 — rejected on
  every count**.
  **HARNESS CHECK, THE TEST THAT DECIDES WHETHER ANY OF THIS MAY BE QUOTED:**
```
  season   k5_hc MEASURED   k5_hc ANCHORED   ratio
  2023-24     0.3381           0.3064        0.907
  2024-25     0.2527           0.2822        1.117
  2025-26     0.2154           0.2662        1.236
  WORST CONTROL-SEASON ERROR  23.6%
```
  **NO ANCHORED CELL BELOW MAY BE READ WITHOUT THAT +/-24% IN FRONT OF IT.**
  (13) **WAS THE EXTRAPOLATION OPTIMISTIC, PESSIMISTIC OR RIGHT? LAW vs LAW —
  D174 §7's comparison, chosen because it does not depend on a back-out.** D166
  carried D163's pooled ESPN23 law, k=5 + outlier haircut = **0.4261**, into
  every extrapolated season.
```
  season    EXTRAPOLATED   ANCHORED k5+HC   [95% CI]           ratio
  2021-22      0.4261         0.3397     [0.3051, 0.3743]   **1.254x**
  2022-23      0.4261         0.3398     [0.3052, 0.3744]   **1.254x**
  carrying the harness's own 23.6%:                          1.02x - 1.38x
  2019-20 / 2020-21  0.4261   NOTHING — MGM's file starts 2021-10-19
  raw twins:  k5 raw 0.4128 / 0.4129 | k2 raw 0.1955 | k2+HC 0.1643 / 0.1644
              | k3+HC 0.2463 | k8+HC 0.3474
```
  **OPTIMISTIC, BY ~1.25x. SO THE FORWARD-CARRIED LAW WAS TOO GENEROUS AT BOTH
  ENDS OF THE RECORD — 1.7-2.0x on 2024-25/2025-26 (D174 §7) and ~1.25x here —
  AND THE MIDDLE OF THE RECORD IS THE LEAST WRONG PART OF IT.** Note what this
  does NOT say: it is an anchored EXTRAPOLATION, not a measurement, and half the
  hole (**2019-20 and 2020-21, both COVID seasons**) is untouched by it.
  (14) **THE RE-PRICE (TASK 5): THE PATH DOES NOT MOVE, AND SAYING SO IS THE
  RESULT.** No season becomes MEASURED, so the official figure is unchanged.
  Reproduced from `data/bkp_ladder.json` to the cent: pooled **+3.1305% /
  +48.62u**, season-clustered **K=14 +2.54% [-3.75, +8.82]**.
  A SENSITIVITY was computed and is labelled as one — D174's path with 2021-22
  and 2022-23's applied gain swapped for the anchored estimate:
```
  pooled          +3.13% (+48.6u)  ->  +3.40% (+52.8u)
  clustered K=14  +2.54% [-3.75,+8.82] -> +2.81% [-3.34,+8.96]
```
  **AND THE DIRECTION MUST BE EXPLAINED OR IT READS AS A CONTRADICTION OF §13.**
  The law comparison says D166 was too GENEROUS while the path sensitivity moves
  UP. Both are right and they are different quantities: D166's per-season
  **APPLIED** gains for those two seasons, backed out of its own dcover, were
  **0.0174 (2021-22) and 0.0785 (2022-23)** — far BELOW its own 0.4261 law.
  That is exactly D174 §12's recorded caveat that the back-out is **~8x noisier
  than the law**. **SO THE +0.27pp IS NOISE CORRECTION, NOT EVIDENCE THAT
  EXECUTION IN THE HOLE WAS BETTER THAN ASSUMED, AND IT IS NOT PROMOTED TO A
  RE-PRICE.**
  (15) **THE COUNT.** **9 of 14 scored seasons MEASURED, 5 EXTRAPOLATED —
  UNCHANGED from D174.** A one-book file does not make a season measured. What
  changes is only the QUALITY of the label on two of the five: 2021-22 and
  2022-23 move from "EXTRAPOLATED, law carried forward from another era" to
  "EXTRAPOLATED, ANCHORED to a verified operator inside the season, +/-24%".
  **2019-20 and 2020-21 remain wholly EXTRAPOLATED with no anchor of any kind,
  and per D174 §15 the COVID seasons are never pooled into a headline.**
  (16) **ERA / CLUSTERING / AVAILABILITY (GATE_POLICY_V2 §10).** Calibration
  universe: the ten seasons with a MEASURED close panel (2012-13..2018-19
  erichqiu + 2023-24..2025-26 AN/ESPN), **K=10**, and the ratio CI above is the
  9-dof cluster-mean interval. Harness control: **K=3**, and its 23.6% worst
  error is quoted rather than a CI, because K=3 does not support one. The
  re-priced path is **K=14**. ERA-AVAILABILITY: unchanged everywhere; improved in
  LABEL QUALITY only for 2021-22/2022-23. AVAILABILITY TIER: not applicable — no
  arm here consumes an out-set.
  (17) **UNPRICED BIAS, RESTATED ONCE AS REQUIRED.** Best-of-k **always**
  transacts at the most offside book in the panel, and **nothing here, in D163,
  D166 or D174 charges anything for being LIMITED, RESTRICTED OR VOIDED.** Every
  number above is gross of that cost and it remains the largest unmodelled risk
  in the execution layer. It applies to the anchored cells exactly as it applies
  to the measured ones, and the anchored cells are if anything MORE exposed,
  since BetMGM is a retail operator of precisely the kind that limits winners.
  (18) **WHAT THIS ENTRY DOES NOT CLAIM.** It does **not** close
  2019-20..2022-23 and says plainly that Kaggle cannot. It does **not** present
  a panel — there is one operator and the word is refused throughout. It does
  **not** re-price the walk-forward (§14 is a labelled sensitivity, and the
  official path is D174's, reproduced not recomputed). It does **not** widen the
  eval corpus, re-certify, gate, ship, or change any default. It does **not**
  contact any host other than kaggle.com, and it does not revisit ESPN, Action
  Network or oddsportal — **D174's robots/ToS stops are upheld in full.** It
  does **not** delete or re-scrape the pre-existing ESPN/AN files, and D174's
  flagged **pre-existing compliance exposure remains open and remains the
  owner's call.**
  [code scripts/d177_mgm_probe.py (NEW, not under `nbapred/`);
   nbapred/ingest/kaggle_web.py MODIFIED (anonymous public-API fallback +
   `search()`/`view()`; the Chrome path is untouched and still tried first);
   nbapred/teams.py MODIFIED (unique-city rule, LAST in `resolve()`, strictly
   additive; `_static()` now returns a third map; `known_report_names` updated
   for the new arity; "Golden State" alias added);
   tests/test_teams_canon.py EXTENDED (3 new regressions, 13 pass);
   data/kaggle_odds_notes.md (full working, checkpointed as the run proceeded —
   the dataset evaluation table, the search limits, and the rejected m1_mean
   variant that missed by 147%);
   data/d177_mgm_probe.json, data/d177_mgm_rows.csv.gz (the 6,077 joined BetMGM
   closes on our team keys), data/logs/d177_mgm_probe.log;
   data/raw/kaggle/caseydurfee__mgm-grand-nba-betting-data/ (NEW, 6,081 rows),
   data/raw/kaggle/thedevastator__uncovering-hidden-trends-in-nba-betting-lines-20/,
   data/raw/kaggle/zachht__wnba-odds-history/,
   data/raw/kaggle/visualize25__basketball-betting-dataset/ (all NEW, all
   EVALUATED AND REJECTED, kept so the verdicts are re-checkable);
   docs/OPENING_LINES.md updated (new §9; §4's Kaggle verdicts and the access
   note rewritten; the phase-coverage table now separates 2019-20/2020-21 from
   2021-22/2022-23 and states the 9-of-14 record);
   inputs data/bkp_panel_rows.csv.gz, data/bkp_ladder.json (D174's, READ ONLY),
   scripts/bkp_ladder.py (IMPORTED for `ladder_vals`/`clustered`, NOT EDITED),
   scripts/build_odds_open.py (`_pair_join`'s rule reused, file NOT EDITED),
   data/nba.duckdb (read_only=True retry_s=60; queries: `odds_market`,
   `odds_open`);
   scripts/bet_engine.py UNTOUCHED, scripts/prod_by_season.py NOT RUN, no gate
   re-run, no default changed, eval corpus unchanged, no walk-forward re-run,
   no chart written; DB READ-ONLY THROUGHOUT — ZERO writes, no table created
   or altered]

- D178 **THE OCTOBER-BLOCKING LIVE-PATH FLAGS, CLOSED — AND THE CLV BANDS
  RE-DERIVED. THE HEADLINE IS NOT THE FILTER: IT IS THAT THE >=2-BOOK RULE
  D142's EVIDENCE DEMANDS CANNOT BE SATISFIED BY EITHER FEED WE ACTUALLY OWN
  AT THE OPEN, SO OCTOBER'S CLV SCORING SET LIVES OR DIES ON A SYSTEMD UNIT
  THAT HAS NEVER RUN IN-SEASON.** (2026-08-05)
  SHIP + DIAGNOSTIC. Four production files changed
  (`nbapred/engine/slate.py`, `nbapred/ingest/injury_pdf.py`,
  `scripts/bet_engine.py`, `scripts/predict_today.py`), two test files re-pinned
  or extended, one new derivation script, one new runbook. NO GATE RUN, NO
  MODEL RE-CERTIFIED, EVAL CORPUS UNCHANGED, `data/capstone_pergame.csv`
  NOT WRITTEN (md5 `695d40a3545e889267cad403b7acdce8` before and after).
  DB `data/nba.duckdb` **READ-ONLY THROUGHOUT** — `read_only=True` with
  `retry_s=60` on every connect, ZERO writes, no table created or altered.
  `nbapred.threads.pin(1)` before numpy in the new script.
  (1) **FIX 1 — THE GAME-TYPE FILTER, AND THE HOLE WAS BIGGER THAN D172 SAID.**
  D172 found `slate.py` was the one production consumer lacking
  `game_id LIKE '002%'`. It is worse than one query: **`todays_games()` had no
  filter at all**, so the nba_api scoreboard — which serves `001` PRESEASON in
  early October and `003` ALL-STAR in February — handed exhibition games
  straight to `fit_production`, the tank priming and the bet engine regardless
  of what any downstream SQL did. Spine census re-confirmed: **315 rows / 97
  non-franchise team codes = 148 `001` international preseason + 167 `003`
  All-Star.** Fixed at FOUR chokepoints so no caller can smuggle one in:
  `slate.todays_games()` (`slate.py:55`), `slate.slate_context()`
  (`slate.py:97`), `bet_engine.emit()`/`scan_open()` (`bet_engine.py:748`,
  `:811`), `predict_today.main()` (`predict_today.py:42`). New helpers
  `is_regular_season`/`filter_regular_season`/`GAME_TYPE` (`slate.py:30-52`)
  REPORT every drop rather than swallowing it (D171's `teams.py` law).
  **THE bet_engine FILTER IS NOT BELT-AND-BRACES:** both emitters keep their
  own copy of `games` and index `ctx["outs"][gid]`, so an unfiltered list would
  have raised KeyError the first October night a preseason game appeared.
  **THREE MORE UNFILTERED QUERIES FOUND BY THE AUDIT, ALL FIXED:** the b2b
  lookup (now `slate.b2b_teams`, `slate.py:73`) and BOTH props CTEs in
  `predict_today.py` (`:118`, `:125`) — the second of which, `current_team`'s
  `arg_max`, could resolve a player's CURRENT team to an All-Star or exhibition
  side and drop that entire roster from the printout. b2b impact MEASURED
  across the whole spine: **1 date x 2 teams** (2025-12-17, the day after the
  2025 NBA Cup final) — small, but `prod_by_season.py:111` and
  `production.fit_schedule_layer:131` are both 002-only, so it was a live/
  backtest PARITY break and it was free. FALSE POSITIVES the audit correctly
  left alone: `october_bridge.py:118` and `production.py:334` filter `001%`
  DELIBERATELY (the D84-A preseason bridge); `tanking.py:194` is a
  team_id->abbrev map; `bet_engine.py` settle/dry-run key on a specific
  game_id. REGRESSION TEST as asked: `test_emit_never_books_a_non_regular_
  season_game` pushes an All-Star + a preseason game through the LIVE emission
  path beside a real one and asserts (a) the model layer is never asked about
  them and (b) neither `bet_paper` nor `bet_quotes_panel` carries their id —
  plus four unit tests covering all six prefixes, the scoreboard chokepoint and
  the b2b query (`test_bet_engine.py:399-552`).
  (2) **FIX 2 — >=2 BOOKS AT THE OPEN, AND THE FEED FINDING THAT MATTERS MORE
  THAN THE RULE.** D142 measured the whole shop asset: best-of-2 lifts CLV
  ~49%, and taking the WORSE book erases essentially all of it (+0.0092 ->
  -0.0007). D125 shipped `bet_quotes_panel` but nothing REQUIRED two books.
  Now `MIN_BOOKS_OPEN = 2` (`bet_engine.py:237`): an OPEN row is BOOKED only
  with >=2 distinct two-sided books; with one book the observation is **STILL
  WRITTEN** — all four stakes zeroed, `single_book=TRUE`, `clv_eligible=FALSE`,
  `detail` carrying `single_book=1` (`bet_engine.py:670`) — and `--report` /
  `--monthly-report` score CLV on `clv_eligible` rows only. **A THIN-BOOK NIGHT
  DEGRADES VOLUME, VISIBLY, RATHER THAN SILENTLY POLLUTING THE MEAN.** Three
  new columns (`n_books`, `single_book`, `clv_eligible`) migrate in place;
  `clv_eligible` DEFAULTS TRUE so **no pre-D178 row is retroactively removed
  from a measurement this rule did not govern**. The gate binds AT THE OPEN
  ONLY — POST_REPORT/PRETIP record `n_books` for telemetry but keep eligibility,
  because narrowing them too would silently redefine the two comparison views
  the whole three-view design exists to measure.
  **NOW THE PART THAT SHOULD WORRY THE OWNER. D174 said Action Network has no
  per-book opener; we checked what the OTHER feeds actually return instead of
  assuming, and the answer is that NEITHER LOCAL FEED CAN SERVE THE OPEN.**
  Share of games with >=2 distinct operators, MEASURED off D174's own
  `data/bkp_panel_rows.csv.gz`:
```
      source  phase    2023-24   2024-25   2025-26
      espn    open       94.7%      0.0%      3.4%
      espn    close      95.2%      0.0%      1.5%
      an      close     100.0%     99.6%     99.1%
```
  **ESPN's live poller is ONE BOOK BY CONSTRUCTION** (`espn_lines.py` reads the
  public scoreboard = ESPN BET), and its historical open panel collapsed to one
  operator when ESPN BET became the house book. **Action Network carries NO
  per-book opening price at all** (D174 (2)(c): `book_id 30` is a consensus
  opener and its per-book snapshot is the CLOSE). **THE ODDS API
  (`odds_logger` -> `ev["bookmakers"][]`, regions=us; `load_odds.flatten`
  emits one row per book) IS THE ONLY LIVE SOURCE WHOSE RESPONSE SHAPE CAN
  SATISFY THE RULE — AND IT HAS NEVER RUN IN-SEASON IN THIS REPO.**
  `data/raw/odds/` holds exactly ONE offseason day whose `data` is `[]`, and
  **`odds_quotes` is EMPTY, 0 rows.** If that systemd unit is not up and
  authenticated on opening night, **every OPEN row will be single-book and the
  OPEN CLV scoring set will be EMPTY.** That is now the loudest failure the
  system has, by design, and it is §3 of the runbook. NOTE what this rule does
  NOT claim: it does not claim the Odds API will deliver k>=2 in practice —
  that is UNMEASURED here because no in-season capture exists. It claims only
  that the shape supports it and that the other two feeds provably do not.
  (3) **FIX 3 — THE CLV BANDS, RE-DERIVED FROM SCRATCH, NOT PASTED.**
  `scripts/d178_clvbands.py` (NEW) -> `data/d178_clvbands.json`. FIDELITY
  ANCHOR FIRST, per D171's discipline: the LEAKY_REG arm reproduces the
  registered D155/D159 ML@open UNION digits **n=1378, CLV=+0.01590 EXACT**, so
  the harness is the registered one before anything else is believed.
  **WHICH SPACE — STATED, BECAUSE D173 PUBLISHES TWO CLV NUMBERS AND THEY ARE
  NOT INTERCHANGEABLE.** `bet_engine.settle` computes
  `clv = close_implied - implied_p` where `implied_p` is the de-vigged
  CONSENSUS MONEYLINE probability on our side. **The engine scores in
  PROBABILITY space on the REAL MONEYLINE**, so the ML frame is used
  (+0.01228, essentially unchanged from D159's +0.01197). D173's ATS
  SPREAD-POINT CLV, which DOUBLED to +0.320, is in POINTS; the engine never
  sees a spread point and using it would have been a **~26x unit error**.
```
    frame|arm      nbets  mo  med/mo    centre    sd_bet        se       RED       GOOD
    ML|HONEST       1386  21      67  +0.01228   0.05123  0.006258  -0.000236  +0.024797  <- INSTALLED
    SP|HONEST       1657  35      47  +0.01692   0.06006  0.008760  -0.000600  +0.034450
    ML|LEAKY_REG    1378  21      66  +0.01590   0.04882  0.006010  +0.003880  +0.027920
    SP|LEAKY_REG    1304  28      44  +0.01944   0.05795  0.008790  +0.001870  +0.037010
```
  **DERIVATION INPUTS (ML|HONEST, `capstone_pergame.csv`, union of the 4 F4
  rules, unique games, @OPEN, 2023-24..2025-26 — the three full-T2 seasons, the
  only ones with a real moneyline open): CENTRE = the UNION mean CLV
  +0.012280275 (D120/D121 used the ALL-SAME-SIDE UNIVERSE, +0.0048 on this
  frame — defect 1); PER-BET SD 0.051226; n = 1,386 bets over 21 months, MEDIAN
  67/month; MONTHLY SE 0.006258 = sd/sqrt(67); BANDS = centre +- 2 se ->
  RED -0.000236 / GOOD +0.024797.** Installed as a DICT `CLV_BAND`
  (`bet_engine.py:282`) with `CLV_MONTH_RED`/`GOOD` **COMPUTED FROM IT**
  (`:317-318`), so the constants cannot drift from their inputs; verified
  installed == derived to **0.00e+00**. `--monthly-report` now PRINTS frame,
  arm, seasons, n, centre, per-bet sd, n/month, se, k, source, anchor and what
  it supersedes, beside the verdict.
  **TWO CONSEQUENCES THAT ARE NOT SIDE-EFFECTS AND ARE PRINTED EVERY RUN.**
  **(a) RED IS NOW MUCH TIGHTER (-0.0002 vs -0.0131).** Correctly centred on
  +0.0123, "2 sigma below" is essentially "any negative month". On the 21
  in-frame months it flags THREE — 2024-10, 2025-04, 2025-10 — and **TWO OF THE
  THREE ARE OCTOBERS** (n=29 both times). Expect an October RED; it means the
  month ran below the historical rate, not that the engine is broken.
  **(b) THE TRIGGER IS UNREACHABLE BY CONSTRUCTION, AND ALWAYS WAS. Asking for
  2 consecutive months above a +2-sigma line is E[wait] = 1/0.02275^2 = ~1,932
  MONTHS.** D159 reported ~98 months for D120/D121 and read that as
  mis-specification; the deeper point is that **the number was ~98 only because
  their GOOD line happened to sit at +1.23 sigma OF THIS FRAME BY ACCIDENT — a
  correctly-specified symmetric 2-sigma band makes the gate WORSE, not better.**
  **FIXING THE BAND DOES NOT FIX THE TRIGGER.** This entry deliberately does
  NOT move `TRIGGER_MONTHS` or re-specify the gate: that is a product decision
  with its own D-line. It makes the arithmetic impossible to overlook instead.
  (4) **CLV IS THE MONITOR, NOT THE OBJECTIVE — D176's FINDING, FOLDED INTO THE
  INSTRUMENT IT CONSTRAINS.** D176 measured CLV and ROI apart: the
  availability-divergence selector bought MORE CLV (6/6 cells, +0.143 pts) and
  LESS ROI (1/6, -1.16pp), while the explicitly CLV-TARGETED selector bought
  essentially NO extra CLV (+0.004) and the MOST ROI. **CLV IS NOT A SUFFICIENT
  STATISTIC FOR BET SELECTION, SO A PURELY CLV-TUNED BAND CAN GREEN-LIGHT A
  SELECTOR THAT IS LOSING MONEY.** These bands are kept because CLV RESOLVES
  FAST — an early-warning read on execution and timing. **A GREEN CLV MONTH IS
  NOT EVIDENCE OF PROFITABILITY; ROI AGAINST THE INCUMBENT IS THE OBJECTIVE.**
  Stated in `CLV_BAND`, in `d178_clvbands.py`'s header, in runbook §4, and
  PRINTED by `--monthly-report` on every run. **ON NULLS (D176's second
  lesson):** D176 found all three new arms beat their own permutation nulls at
  p<=0.048 and survived BH **and all three still lost to the incumbent** —
  "beats a scrambled copy of itself" is true of the incumbent too. **CHECKED
  AND STATED: nothing in this derivation is a net-of-null statistic.** Centre,
  sd and se are plain sample moments of the union CLV, so there is no null to
  over-read; any future re-derivation that introduces one must report against
  the INCUMBENT as well.
  (5) **FIX 4 — LIVE DRY RUN, BOTH BRANCHES.** New `--dry-books {1,2}`.
  On the real 2026-03-27 slate (10 games): **books=2** — OPEN 11 bet rows
  (11 booked, 0 excluded), POST_REPORT 8, PRETIP 8, 40 panel rows per view, 27
  settled; staked flat 11.00/8.00/8.00, raw_kelly 53.03/13.91/13.91,
  shrunk_kelly 0.00 everywhere (the D112/D117 result — the calibrated CLOSE edge
  does not clear the vig), open_shrunk 8.03/0.02/0.02; OPEN mean CLV +0.0200
  (flag `ok`), POST_REPORT/PRETIP +0.0000 by construction (both priced at the
  close). **books=1, the HONEST historical shape** — OPEN 11 rows, **0 BOOKED,
  11 single-book/CLV-EXCLUDED, n_clv = 0, `1book` = 11, meanK = 1.00**, while
  POST_REPORT/PRETIP book 14 each. Both branches of the >=2-book rule exercised
  end to end including settle, `--report` and `--monthly-report`. NO ERROR.
  `bet_paper` was NEVER touched (temp DuckDB). `books=2` adds a CLEARLY-LABELLED
  SYNTHETIC second book priced off the same row's spread via the program's own
  `sigmoid(margin/6.96)` map — it exists so the BOOKED branch and all four arms
  are exercised, and it is **plumbing, never a measurement**: D174 (5)
  established no historical per-book ML OPEN panel exists for 2024-25/2025-26.
  `scripts/predict_today.py` runs CLEAN (offseason no-op, exit 0).
  **`docs/OCTOBER_RUNBOOK.md`** (NEW): cron layout with ordering constraints,
  what each view books, the >=2-book rule and its feed reality, the bands and
  their derivation, the game-type filter, a **ranked what-breaks-first table**
  with first symptom / check / blast radius, all eleven kill switches with
  DEFAULTS (`LATE_STATE` 0-OFF, `TANK_TERM` 1-ON, `OCT_BRIDGE` 1-ON,
  `OCT_BRIDGE_TRAIL` "2", `COVID_GUARD` 0-OFF, `PROPS_MIN_RAMP` 1-ON,
  `PROPS_ABSENCE_RAMP` 1-ON, `PROPS_KALMAN_FWD` unset-OFF, `ORACLE_PLAYED_OUTS`
  0-OFF **and LEAKAGE — ceiling runs only**, `TANK_SEASON_FLOOR` pinned
  2020-21), and a "things that are NOT broken" section. **It also records that
  `predict_today.py` IS NOT IN CRON** — deliberate (it is off the write path)
  but it means a break in it is only ever found by running it.
  (6) **THE TWO RED TANKING FIXTURES, RE-PINNED.** Both were pinned to a DB
  vintage that no longer exists. `k_26` at the old floor drifted **-2.26990
  (pre-D112) -> -2.17831 (after D170's 97-report-day backfill) -> -2.08251
  (after D171's Clippers join fix)** — a DATA change, not a code regression, on
  a term fit on the availability-driven composite. Re-pinned to
  **-2.08251078599815** with the tolerance TIGHTENED 0.01 -> **1e-4**, so a
  future drift is again detectable and LOUDER than before. The gate-table
  fixture no longer reads the stale `data/apr_tank_stats.csv` at all: 18
  `(tank_score, gp_before)` pairs are pinned INLINE (`test_tanking.py:102`)
  with the superseded CSV value in a trailing comment on every line. **`gp` is
  unchanged on all 18 teams; only the scores moved** — consistent with an
  availability-input shift through a POOLED EXPANDING z, which moves every team
  once it moves any. `test_tanking.py` now **6 passed** (was 4 passed / 2
  failed). `data/apr_tank_stats.csv` NOT regenerated.
  (7) **`da Silva, Tristan` — THE ROOT CAUSE IS WORSE THAN THE ARTEFACT, AND
  THE ARTEFACT IS THE ONLY ONE LEFT.** `injury_pdf.parse_pdf`'s `PLAYER` regex
  required an INITIAL CAPITAL, so a lowercase nobiliary particle never matched
  "Last, First" and fell through to an **UNGUARDED `else: team = f`**. TWO
  effects, both measured: **(a) the player's OWN row was dropped entirely** —
  no `player`, so no row is appended, i.e. **Tristan da Silva has never had an
  injury status in this system at all**; and (b) the poisoned value
  FORWARD-FILLED to every later row in that team's block, leaving **30 rows of
  `injury_reports_pit.team` holding a player name**, which `slate_context`'s
  `team_id_for` resolves to None — silently emptying those Orlando out-sets.
  Fixed AT THE PARSER: `PLAYER_RX` lifted to module level and widened to accept
  a 1-3-letter lowercase particle (`injury_pdf.py:54`), plus a REPORTING
  backstop — `elif f in _team_names()` and otherwise `log.warning` and KEEP the
  previous team (`:260-273`) — so an unrecognised fragment can never again
  become a team. Bounded so no team name can match: team names carry no comma
  and none begins with a short lowercase word; asserted against all 31
  `known_report_names()` in `test_teams_canon.py:136`.
  **RE-PARSED ALL 1,259 ARCHIVED PDFs IN MEMORY (no DB write): 126,378 rows,
  0 bad files, distinct `team` 31 -> 30 real names.** ANSWERING THE QUESTION
  ASKED: **NO other non-team string survives.** The only remaining non-name
  values are **6 rows with `team=None`**, all from report 2023-01-05, which
  also have `game_date=None` and are therefore dropped by `load_all`'s
  `dropna(subset=["game_date"])` — they never reach the table and are
  PRE-EXISTING, unchanged by this fix.
  **THE TABLE WAS DELIBERATELY NOT REWRITTEN.** Reload delta MEASURED: **30
  rows out (all `team='da Silva, Tristan'`), 39 rows in (all Orlando Magic; 17
  Out / 16 Questionable / 3 Available / 2 Doubtful / 1 Probable, of which 1 is
  da Silva's own Out).** That is a change to the availability inputs the
  certified model reads, so re-loading `injury_reports_pit` would move
  `capstone_pergame.csv` and is a re-cert, not a bookkeeping edit — **THE
  OWNER'S CALL, NOT THIS ENTRY'S.** The LIVE path is nevertheless fixed from
  the next PDF onward, which is the October-blocking half.
  (8) **D-NUMBER COLLISION, HANDLED TWICE.** D176 was claimed in flight by the
  strategy-space study and D177 by the Kaggle book-panel sweep. Every citation
  in this entry's code, tests and docs was renumbered **D176 -> D177 -> D178**
  (77 references across 10 files) and `d176_clvbands.*` renamed `d178_clvbands.*`.
  Reason it was worth doing properly rather than leaving stale comments: D141
  hall-of-shame 15 — **a production constant whose comment points at an
  unrelated study is worse than no comment.** No file collision occurred
  (D177's artifacts are `d177_mgm_probe.*`).
  (9) **WHAT THIS ENTRY DOES NOT CLAIM.** It does NOT claim the model got
  better — no model coefficient, gate, corpus or certified artifact moved. It
  does not claim the >=2-book rule will PRODUCE volume in October; on the
  evidence in hand it may produce none, and that is the point of logging the
  refusals. It does not re-specify the real-stakes TRIGGER. It does not reload
  `injury_reports_pit`, regenerate `apr_tank_stats.csv`, or move
  `PINNED_SEASON_FLOOR`. It does not measure the Odds API's live per-book depth
  — that is unmeasurable until the season starts. **FULL TEST SUITE: 153
  passed, 0 FAILED, 0 errors** (226s). The two known-red `test_tanking.py`
  fixtures are now GREEN — `test_tanking.py` alone is 6/6 — and this entry adds
  **13 new tests**: 5 for the game-type filter (incl. the emit-path regression),
  3 for the >=2-book rule, 3 for the bands, 2 for the parser.
  [code nbapred/engine/slate.py (game-type filter x2 chokepoints, `b2b_teams`
   extracted and filtered), nbapred/ingest/injury_pdf.py (`PLAYER_RX` widened
   + reporting backstop), scripts/bet_engine.py (`MIN_BOOKS_OPEN`, `CLV_BAND`,
   3 new bet_paper columns + migration, clv_eligible scoring in `report`/
   `monthly_report`, derivation-input printout, `--dry-books`),
   scripts/predict_today.py (filter + both props CTEs);
   scripts/d178_clvbands.py (NEW) -> data/d178_clvbands.json (NEW);
   docs/OCTOBER_RUNBOOK.md (NEW); data/livepath_notes.md (NEW, the checkpointed
   working record incl. the two D-number collisions);
   tests/test_bet_engine.py +11 tests, tests/test_teams_canon.py +2 tests,
   tests/test_tanking.py 2 fixtures re-pinned;
   data/capstone_pergame.csv NOT WRITTEN (md5 695d40a3545e889267cad403b7acdce8
   before and after); injury_reports_pit NOT RELOADED; apr_tank_stats.csv NOT
   REGENERATED; NO GATE RUN; NO PRODUCTION MODEL DEFAULT CHANGED; eval corpus
   unchanged; DB data/nba.duckdb READ-ONLY THROUGHOUT — read_only=True with
   retry_s=60 on every connect, ZERO writes, no table created or altered]

- D179 **WHICH GAMES ARE IN THE MODEL — A CENSUS, AND AN NBA CUP MOTIVATION
  TEST. ALL-STAR WAS NEVER IN THE MODEL; CUP GROUP PLAY ALWAYS WAS, AND IS
  MEASURABLY HARMLESS. THE REPORTING FRAMES ARE RE-CUT TO POST-2018 (MODEL) AND
  2023-26 (BETTING), BOTH CHOSEN BY DATA AVAILABILITY AND BOTH LESS FLATTERING
  THAN WHAT THEY REPLACE.** Owner asked three things: whether the All-Star game
  is in the model, which games we use at all, and whether in-season-tournament
  motivation needs a test.

  **Q1 CENSUS (`scripts/d179_cup_motivation.py` -> `data/d179_gametypes.json`).**
  Spine holds 40,128 games across six `game_id` prefixes: `002` regular season
  35,546 (88.58%), `004` playoffs 2,440, `001` preseason 2,019, `003` All-Star
  weekend 83, `005` play-in 37, `006` NBA Cup FINAL 3 (2023-12-09, 2024-12-17,
  2025-12-16). **Every model surface filters `002%`** — verified by grep across
  `nbapred/` and `scripts/`: 23 files touch `nba_games` without the filter and
  ALL are ingest, audit or utility paths. The one that needed checking was
  `nbapred/model/rapm.py`, which has no prefix constraint of its own — but it is
  not in the production margin (no `rapm` reference in `production.py`,
  `slate.py` or `prod_by_season.py`) and its live consumer
  `nbapred/features/defense_zone.py:113` applies `002%` itself, as does
  `def_rapm.collect_shots(only_002=)`. **ANSWER: the All-Star game has never
  been in the model.** The exposure was live-only and is already closed by D178
  FIX 1 (`todays_games()` had NO filter, so a February All-Star date would have
  handed the bet engine exhibition games with non-franchise team ids; never
  fired, entry point has only run in the offseason).

  **Q2 CUP MOTIVATION — ns, AND THE FRAMING MATTERS.** Cup group-stage and
  quarter/semi-final games carry `002` **because they count in the standings**,
  so they are in the model and cannot be removed without removing real
  regular-season games. Only the final is exempt (`006`) and is already out.
  Pre-registered diff-in-diff, design stated before scoring: TREAT window
  Nov 1-Dec 20, CONTROL window Jan 15-Mar 31, Cup seasons 2023-26 vs pre-Cup
  2018-23 (COVID seasons naturally absent from the pre-Cup arm's Nov window).
  n = 1,013 / 1,598 / 1,446 / 2,435.

  | statistic | Cup Nov-Dec less Jan-Mar | pre-Cup same | DiD | se | z |
  |---|---|---|---|---|---|
  | signed home margin | +0.694 | +0.849 | **-0.156** | 0.789 | -0.20 |
  | mean abs margin | -1.075 | -0.147 | **-0.927** | 0.789 | -1.18 |
  | sd of margin | -1.229 | -0.125 | **-1.104** | 0.789 | -1.40 |

  **MDE80 = 2.209 pts, stated before scoring.** All three ns and all three
  inside the noise floor. All three share a sign — Cup-window games marginally
  TIGHTER than the seasonal norm, the direction raised effort predicts — but
  **the three statistics are computed on the same games, so the agreement is one
  piece of evidence, not three**, and the honest statement is that we can rule
  out a Cup effect larger than ~2.2 points and nothing smaller. **DECISION: Cup
  games stay in, unflagged. No default changed.**

  **Q3 REPORTING FRAMES RE-CUT.** Owner asked to report the model post-2018
  (injury reports begin 2018-12-17) and betting on 2023-26 or 2024-26. Both
  re-cuts were made on data availability and **both are worse than what they
  replace**, which is the reason to trust them.
  - **Model frame -> 2018-19 onward.** Pooled normalized gap **13.22%** (K=6,
    n=7,378: 2018-19, 2021-22..2025-26). Pre-injury-feed seasons pool to
    **6.81%** (K=10) and the all-poolable blend to **9.05%**. The post-2018
    figure is the WORSE one. It is nonetheless the right one because the two
    windows measure different models: before the feed exists the availability
    leg runs on inputs it was never designed to have, so those seasons score a
    crippled variant, and the 9.05% blend averages two systems and flatters the
    one we would deploy. Per-season post-2018: 14.98 / 16.95 / 13.21 / 16.34 /
    6.43 / 12.43.
  - **Betting frame -> 2023-26, NOT 2024-26.** This is the only window with a
    MEASURED multi-book panel (earlier seasons infer it from a shopping law;
    D177 showed that law was ~1.25x too generous mid-corpus). At the firm
    default (k=5 +haircut): 2023-24 +0.37%, 2024-25 +12.12%, 2025-26 +3.11%;
    pooled **+5.48%** on 500 bets, **season-clustered 95% CI [-9.79%, +20.75%],
    MDE80 18.3pp.** Dropping 2023-24 raises the point estimate to +8.33% and
    widens the interval to **[-48.89%, +65.55%], MDE80 61.0pp** — an interval
    that cannot detect a 60-point edge is not a measurement, so **K=2 is
    rejected on power, not on preference.** Recorded plainly in the README:
    the interval contains zero and **2024-25 alone supplies 82% of the pooled
    P&L**, so this remains a candidate, not a result.

  **HALL OF SHAME ADDITION (methodological, applies backwards):** a window
  chosen after seeing which one scores best is a selection, however principled
  the story told afterwards. Both frames here were fixed by asking "where does
  the input exist?" before looking at the endpoint, and both landed on the less
  flattering answer. That is the only evidence that the question was asked in
  the right order.

  [SCOPE: `scripts/d179_cup_motivation.py` NEW (read-only);
  `data/d179_gametypes.json` NEW; README reporting frames re-cut; NO GATE RUN;
  NO PRODUCTION MODEL DEFAULT CHANGED; no feature added or removed; Cup games
  explicitly LEFT IN; DB data/nba.duckdb READ-ONLY THROUGHOUT, zero writes]

- D180 **CHART STALENESS SWEEP — 7 OF 35 PUBLISHED CHARTS WERE BUILT ON
  SUPERSEDED DATA AND NOBODY NOTICED. TWO NEW FRAME CHARTS ADDED; HEADLINE
  BETTING TIER MOVED TO k=8 (MAX BOOKS).** Owner flagged a published chart
  showing the pre-D158 pooled figures (LL 0.6036 / mkt 0.5924, gap 11.13%) and
  said "many such cases".

  **THE SPECIFIC CHART WAS ALREADY FIXED, THE GENERAL CLAIM WAS RIGHT.** The
  flagged render carried the OLD suptitle "certified production stack
  (LATE_STATE=0, TANK_TERM=1)"; the on-disk file has since been rebuilt and now
  reads "CERTIFIED D171" with computed values. But that was luck, not process:
  there was no mechanism that would have caught it.

  **SWEEP METHOD (the reusable part).** mtime is worthless here — a `cp` resets
  it — and the numbers are pixels, so they cannot be grepped. The only sound
  test is **re-run every chart script and diff the md5**: a chart whose bytes
  change on a no-op rerun was, by definition, built on data that has since moved.
  18 scripts, all exited 0.

  | result | charts |
  |---|---|
  | already current | 28 |
  | **STALE -> refreshed** | **7**: `ats19_open`, `history_logloss_by_season`, `history_normalized_gap`, `honest_clv`, `k19_model_and_rules`, `status_trading_h2h`, `walkforward_equity` |

  **20% of the published chart set was wrong.** Every one is a D171/D173
  consumer, i.e. exactly the charts the re-certification should have
  invalidated — the re-cert updated the numbers and the register but never
  re-rendered everything downstream. **HALL OF SHAME: a re-certification is not
  complete until every artifact that consumes it has been re-run and diffed.
  Publishing is a consumer.**

  **PRESERVATION (owner instruction: do not delete or edit, archive instead).**
  All 8 superseded renders (the 7 above + the explicitly-named
  `_superseded_logloss_4season_normalized.png`, retired from the public repo)
  copied to `charts_archive/` with a `__preD179_YYYYMMDD` suffix BEFORE the
  refreshed versions were published. Nothing overwritten in place, nothing lost.

  **TWO NEW FRAME CHARTS (`scripts/d179_frame_charts.py`)** on the D179 frames:
  `charts/frame_model_post2018.png` (per-season normalized gap 2018-19 onward,
  pooled 13.22%, with pre-feed 6.81% / blended 9.05% beside it) and
  `charts/frame_betting_k8_2023_26.png` (per-bet equity on the measured panel,
  per-season split, 95% CI). Palette validated rather than eyeballed: categorical
  light-mode, worst adjacent CVD dE 6.6 (deutan) which is the 6-8 floor band ->
  **every mark carries a direct value label** as the required secondary encoding,
  which also discharges the contrast WARN on the green. Rendered and inspected;
  four label collisions and two title overflows fixed before shipping.

  **HEADLINE BETTING TIER -> k=8 (MAX BOOKS), per owner.** Supersedes the
  k=5 +haircut headline in D179.

  | tier | 2023-24 | 2024-25 | 2025-26 | pooled | 95% CI (K=3) |
  |---|---|---|---|---|---|
  | k=8 raw | +3.09% | +15.30% | +7.14% | **+8.72%** (+43.62u/500) | [-6.72, +24.17] |
  | k=8 +haircut | +0.49% | +12.63% | +4.03% | +5.96% (+29.82u) | [-9.54, +21.47] |

  **All three seasons positive at k=8** (at k=5 +haircut 2023-24 was +0.37%,
  effectively flat). The interval still contains zero and **2024-25 supplies 65%
  of the pooled P&L**, so the reading is unchanged: a candidate, not a result.
  Both raw and haircut are published; reporting raw alone would drop the charge
  for the 8.1% of best-of-N prices that get limited or voided.

  [SCOPE: `scripts/d179_frame_charts.py` NEW; 7 charts re-rendered from existing
  scripts (no script logic changed); 8 superseded renders ARCHIVED not deleted;
  README headline tier -> k=8; NO GATE RUN; NO PRODUCTION MODEL DEFAULT CHANGED;
  no model refit; DB READ-ONLY THROUGHOUT]

- D181 **k=9 TIER ADDED (MAX BOOKS OBSERVED AT THE OPEN) AND A
  SIMULATION-PERFORMANCE REPORT WRITTEN IN INSTITUTIONAL FORMAT. THE WORK
  UNCOVERED A FALSE CLAIM I HAD PUT IN THE README TWO PUSHES EARLIER.** Owner
  supplied a US Treasury cash/futures basis-RV sim report and asked for the same
  format at max k with the haircut removed.

  **MAX k IS 9, NOT 8.** The panel holds 16 distinct operators. At the OPEN the
  observed books-per-game distribution tops out at **9** (256 games), so
  `TIERS = [1, 2, 5, 8]` was leaving the true ceiling untested. Added 9 and
  re-ran (`WF_TAG=_D181 AS_TAG=_D173`; the run aborts on its own reproduction
  guard if the adaptive tag is not matched to the D173 arms — that guard worked).

  | tier | ROI | cum | 95% CI |
  |---|---|---|---|
  | k=1 raw | +1.83% | +31.6u | [-2.44, +6.63] |
  | k=5 raw | +3.96% | +68.6u | [-0.69, +8.60] |
  | k=8 raw | +4.58% | +79.3u | [-0.25, +9.24] |
  | **k=9 raw** | **+4.63%** | **+80.3u** | **[-0.20, +9.30]** |
  | k=9 +haircut | +3.65% | +63.3u | [-0.86, +8.37] |
  | exchange c=2% | +5.58% | +96.7u | **[+1.15, +10.56] SIG** |

  **THE 9th BOOK IS WORTH +0.05 ROI POINTS.** The ladder has flattened out well
  before the ceiling; 8 and 9 are the same number. Also worth stating plainly:
  **~60% of the reported edge is execution, not model** — the identical bets at
  one retail book return +1.83%.

  **CORRECTION TO MY OWN README (D179).** I wrote that 2023-24..2025-26 was "the
  only window with a MEASURED multi-book price panel". **That is false and I
  should have checked the panel before asserting it.** Measured books/game at the
  OPEN: **2023-24 = 7.74** (max 9, 94.7% of games with >=2), **2024-25 = 1.00**
  (0.0% with >=2), **2025-26 = 1.03** (3.4%). `wf_equity.py:46` says so directly
  — `RETAIL_MEASURED = ("2023-24",)`, `RETAIL_EXTRAP = ("2024-25", "2025-26")` —
  and I did not read it. Across the 14-season frame the split is **7 measured**
  (2012-13..2017-18 offshore + 2023-24) **/ 7 modelled**. README corrected.

  **THE EXTRAPOLATION IS NOT WHAT IS CARRYING THE RESULT** (the check that had to
  be run once the error was found): measured seasons return **+5.65%** (K=7,
  n=780) against **+3.79%** (K=7, n=953) on modelled ones. The error was in the
  claim, not in the direction of the number.

  **REPORT (`docs/SIM_REPORT.md`, `charts/sim_report_equity.png`).** Full window
  758 sessions / 1,733 bets / 14 seasons, k=9 raw, no haircut, flat $10k stake:
  net **$802,501**, $1,059/day, **Sharpe 0.5**, win days **47%**, max drawdown
  **-$305,240**, edge **463 bps**. 10/14 seasons profitable; two seasons
  (2014-15, 2024-25) supply **61%** of net PnL; season-clustered 95% CI
  **[-0.12%, +9.38%]** contains zero.

  **THREE FORMAT-DRIVEN HONESTY POINTS** the source document's schema forced into
  the open, all of which belong in the register:
  1. **Sharpe must not be annualised at sqrt(252).** This strategy trades ~54
     sessions a season. sqrt(252) reports 1.2; the honest sqrt(54) reports
     **0.5**. Any future Sharpe in this project uses realised session count.
  2. **Win days are 47%, below half.** Profit comes from winning days being
     bigger, not more frequent — the opposite profile to a market-making book,
     and a fatter-tailed one.
  3. **A 463 bps "edge" is not comparable to a market-maker's sub-bps edge.** It
     is large only because turnover is ~$23k/day rather than ~$1B/day. Quoting
     the bps figure alongside an institutional report without that sentence would
     be actively misleading.

  [SCOPE: `scripts/wf_equity.py` TIERS 8 -> 9 (additive only, no selection rule,
  band or existing tier touched; backup at scratchpad/wf_equity.py.bak);
  `scripts/d181_report_tables.py` NEW; `scripts/d181_sim_chart.py` NEW;
  `data/wf_equity_D181.json` + `wf_perbet_D181.json` + `d181_report.json` NEW;
  `docs/SIM_REPORT.md` NEW; `charts/sim_report_equity.png` NEW; README betting-
  frame claim CORRECTED; NO GATE RUN; NO PRODUCTION MODEL DEFAULT CHANGED; DB
  READ-ONLY THROUGHOUT]

- D182 **"SHOULD WE NOT TRADE IN APRIL OR DECEMBER THEN?" — NO. THE MONTH FILTER
  IS 85% SEARCH ARTIFACT, APRIL IS NOT EVEN A LOSING MONTH, AND DECEMBER IS ONE
  BAD DECEMBER.** Owner read the D181 monthly table (Dec -$127,073, Apr -$13,756)
  and asked whether to stop trading those months. Four tests declared before
  scoring; decision rule declared before scoring (ship only if walk-forward beats
  the incumbent AND the manufacturing null does not explain the in-sample gain —
  net-of-null alone is insufficient per D176).

  **T1 THE TEMPTATION.** Dropping Dec+Apr moves 14-season ROI from **+4.631% to
  +7.103%, a +2.473-point gain** on n=1328. This is the number that makes the
  idea look obvious.

  **T2 THE EFFECT IS NOT WHAT THE TABLE SUGGESTS.** Pooled monthly dollars weight
  seasons by how many bets they happened to contribute, and coverage is wildly
  uneven — **December appears in only 5 of 14 seasons**, April in 11, March in 12.
  Equal-weighting seasons:
  - **APRIL: mean season ROI +3.78% (K=11, t=+0.59).** April is a POSITIVE month.
    Its negative dollar total is a composition artifact, not a signal. The owner's
    question named a month that does not lose money.
  - **DECEMBER: mean season ROI -18.92% (K=5, t=-1.39, ns).** By season:
    2022-23 **-66.1%**, 2020-21 -23.6%, 2021-22 -14.2%, 2025-26 -8.2%,
    2013-14 +17.5%. **Excluding 2022-23 alone, December is -1.56% on n=94.**
    One season is the entire effect.

  **T3 MANUFACTURING NULL (2,000 draws, month labels shuffled WITHIN season, so
  season and bet composition are preserved).** The same "drop the worst 2 of 7
  months" search run on label-shuffled data manufactures **+2.090 points on
  average** and +3.725 at the 95th percentile. Real gain +2.473.
  **p = 0.2800. Net of null +0.382 points.** ~85% of the apparent gain is the
  search itself. This is the D165/D166 capacity result reproducing at a smaller
  scale on a 21-cell space.

  **T4 WALK-FORWARD (the only test that decides).** Drop every month with
  negative ROI on seasons 1..k, freeze, score k+1:
  **filtered +5.423% (n=1277) vs unfiltered incumbent +4.997% (n=1639) =
  +0.426 points, season-clustered t=+0.51 (K=13), ns.** Per-season deltas swing
  +0.00/-0.31/+0.00/+0.00/+1.49/-2.52/+0.00/**+9.20**/+1.87/-3.28/+5.42/-0.95/
  -4.33 — the single large win is 2020-21, i.e. the COVID August calendar, and
  the rule is **negative in the two most recent seasons**. The rule does not even
  select December until 2023-24, because December is absent from most prior
  seasons.

  **VERDICT: DO NOT SHIP.** Fails both conditions.

  **HALL OF SHAME (the transferable part):** a monthly/subgroup breakdown printed
  for reporting is not a menu of filters. Pooled subgroup dollars conflate effect
  with coverage — here it labelled a **positive** month (April, +3.78%/season) as
  a loser purely because fewer seasons contributed bets to it. Any future subgroup
  exclusion in this project must clear (a) equal-weighted per-season inference,
  (b) a within-season label-shuffle null, and (c) walk-forward against the
  incumbent. D181's own monthly table has been annotated in `docs/SIM_REPORT.md`
  so it cannot be misread the same way again.

  [SCOPE: `scripts/d182_month_filter.py` NEW (read-only);
  `data/d182_month_filter.json` NEW; `docs/SIM_REPORT.md` monthly section
  annotated; NO GATE RUN; NO PRODUCTION MODEL DEFAULT CHANGED; no month filter
  added anywhere; DB not touched]

- D183 **THE SHARPE ANNUALISATION BASIS — OWNER PROPOSED TRADES RATHER THAN
  SESSIONS. HE IS RIGHT THAT IT IS A VALID (ARGUABLY BETTER) BASIS, IT GIVES THE
  SAME ANSWER, AND THE FACT THAT IT DOES IS WHAT PROVES sqrt(252) WRONG.** Owner
  challenged D181's sqrt(54): "i dont see why not sharpe sqrt(252) - we should do
  on amount of trades, not sessions traded, right?"

  **THE QUESTION IS CHECKABLE, NOT A CONVENTION.** Annualising = per-period Sharpe
  x sqrt(periods per YEAR). Two natural bases must therefore agree, since both
  estimate the same annual quantity:

  | route | per-period Sharpe | x sqrt(periods/yr) | annualised |
  |---|---|---|---|
  | per SESSION | +0.07458 | sqrt(54.1) | **0.549** |
  | per TRADE | +0.04949 | sqrt(123.8) | **0.551** |

  **They agree to 0.0019 (0.3%).** They agree because the implied mean
  intra-session correlation is **+0.0054** — var(session) observed 2.01519 against
  2.00127 predicted under independence, ratio 1.0070, on 456/758 sessions carrying
  more than one bet. Same-night bets are effectively independent, so trades and
  sessions are two ways of counting the same year.

  **THE DECISIVE POINT: sqrt(252) BREAKS THE AGREEMENT.** Applied per-session it
  gives **1.184**; applied per-trade it gives **0.786**. If 252 were the right
  period count both routes would STILL have to agree — annualisation cannot depend
  on which valid basis you pick. That they disagree by 1.5x is a self-contained
  proof that a year of this strategy does not contain 252 periods. It contains
  ~54 sessions, equivalently ~124 trades. sqrt(252) overstates the session route
  by 2.16x and the trade route by 1.43x.

  **OWNER'S BASIS PREFERENCE ADOPTED where it is finer-grained**: per-trade is the
  better unit (it does not throw away within-night dispersion, and it is robust to
  slate size drifting). It changes the reported number by +0.002. `docs/SIM_REPORT.md`
  now shows BOTH routes side by side rather than one, because the agreement is
  itself the evidence.

  **ADDITIONAL, AND IT CUTS THE OTHER WAY FROM THE OWNER'S PROPOSAL:** the betting
  window spans a mean **88 calendar days of 365**, so capital is idle **76%** of
  the year. An annualised Sharpe computed only over active periods does not charge
  for that idleness, so 0.55 is itself generous as a description of what a
  full-year capital allocation would experience.

  **HALL OF SHAME (methodological):** an annualisation factor is not a style
  choice, and the way to catch a wrong one is to recompute on a second valid basis
  and check the two agree. Any Sharpe quoted in this project must state its period
  count, and where two bases exist, both.

  [SCOPE: `scripts/d183_sharpe_basis.py` NEW (read-only);
  `data/d183_sharpe.json` NEW; `docs/SIM_REPORT.md` Sharpe caveat rewritten to
  show both routes; headline Sharpe UNCHANGED at 0.5; NO GATE RUN; NO PRODUCTION
  MODEL DEFAULT CHANGED; DB not touched]

- D184 **DECEMBER, ERA-RESTRICTED: THE OWNER'S ARGUMENT SURVIVES FURTHER THAN
  D182's DID, BUT DIES ON OUR OWN SHIPPING STATISTIC AND FINDS NO MECHANISM.
  APRIL'S PREMISE IS FACTUALLY WRONG. AND THE SELECTOR ALREADY DECLINES DECEMBER
  IN 9 OF 14 SEASONS.** Owner: "if training to try to profit on post 2023 (or
  2018?) markets, i think we should not rely on the 2013-14 result, there must be
  something structurally wrong with december and april in recent seasons if all
  negative."

  **THE ERA POINT IS CONCEDED AND IS CORRECT IN PRINCIPLE.** D182 let a
  **2013-14** December (+17.5%) sit in the same average as the modern ones. If
  the deployment target is the post-2018 market, a pre-injury-feed season should
  not be allowed to rescue a month. Re-scored on the reported frame:

  **DECEMBER (2018-19+): NEGATIVE IN 4/4 SEASONS** — 2020-21 -23.61% (n=5),
  2021-22 -14.15% (n=29), 2022-23 -66.11% (n=17), 2025-26 -8.21% (n=26).
  Pooled **-24.23% on n=77**.
  - bet-level bootstrap 95% CI **[-44.78, -3.74] — excludes zero**
  - **season-clustered 95% CI [-69.67, +13.63] — CONTAINS zero.** Per
    GATE_POLICY_V2 §8 the season-clustered interval is the shipping statistic and
    D139 established the i.i.d. bet-level bootstrap is the one that lies. So the
    honest read is NOT significant.
  - exact sign test 4/4 gives **p = 0.0625**, and **0.0625 is the minimum
    attainable p at K=4** — a perfect run of four cannot reach 0.05. The design
    cannot deliver significance here regardless of how real the effect is.
  - excluding 2022-23 (n=17, -66%): December is **-12.36% on n=60**.

  **APRIL: THE PREMISE IS WRONG.** Modern April is negative in **2 of 6** seasons,
  not all — 2018-19 +3.69, 2020-21 -19.83, 2021-22 +6.15, 2022-23 +43.18,
  2023-24 -8.46, **2024-25 +18.74**. Pooled **+0.29%** on n=177, CI
  [-13.16, +14.22], sign-test p = 0.89. April's best modern showing is in the
  best modern season. **April is not dropped and the question should not be asked
  of it again.**

  **NO MECHANISM IN THE INPUTS (H3).** The availability leg is the model's
  injury-reactive half, so a structurally mispriced month should show anomalous
  absence load. Mean daily Out+Doubtful listings vs the all-month mean:
  Oct -36.0%, Nov -1.0%, **Dec -4.3%**, Jan +1.5%, Feb +0.7%, Mar +3.7%,
  **Apr +35.1%**. **December is unremarkable; APRIL is the anomalous month** — and
  April is the one that performs fine. The mechanism story points at the wrong
  month.

  **THE FINDING THAT ACTUALLY CHANGES THE DECISION (H4).** The walk-forward
  selector **already places zero December bets in 9 of 14 seasons** (2012-13,
  2014-15, 2015-16, 2016-17, 2017-18, 2018-19, 2019-20, 2023-24, 2024-25).
  December is **6.4% of all volume, 6.8% in the modern frame**. We are already
  mostly not trading December; the proposal is to hard-code what the selector
  does on its own most years.

  **COST/BENEFIT.** Modern frame with December **+3.69%** (n=1132) vs without
  **+5.73%** (n=1055): **+2.04 points for 6.8% of volume.** That +2.04 is
  IN-SAMPLE and carries D182's manufacturing critique; it must not be booked as
  expected gain.

  **DECISION — a judgment override, logged as such, NOT a gated finding.** The
  statistical bar is not cleared and cannot be at K=4. But the decision is cheap
  in both directions: at 6.8% of volume the cost of being wrong is bounded, the
  sign is consistent 4/4, and October deployment will meet December before any
  live evidence exists. **A December skip is therefore a defensible
  risk-management choice, and it is registered as a PRE-COMMITMENT to be scored
  live — not as a result.** April is NOT included. If it is adopted the live
  scoring must compare against the unfiltered incumbent, per D176.

  **HALL OF SHAME:** D182 pooled across eras and let a 2013-14 season into a
  judgement about a post-2018 market. Era-restriction should have been the FIRST
  cut, not the owner's correction. Subgroup questions inherit the frame of the
  deployment target.

  [SCOPE: `scripts/d184_december_modern.py` NEW (read-only);
  `data/d184_december.json` NEW; NO GATE RUN; NO PRODUCTION MODEL DEFAULT
  CHANGED; no month filter implemented in code — this entry records a
  pre-commitment, not a change; DB READ-ONLY]

- D186 **THE FRAME WAS WRONG BY ONE SEASON: 2018-19 IS ONLY 63.7% INJURY-COVERED.
  CORRECTED TO 2019-20+ AND STAMPED ON EVERY PUBLISHED DOC.** Owner: "only post
  2018 is trustworthy (actually should be 2019-26 then, since injury reports
  happened in 2018-2019 season, right?)". **Correct, and D179's frame was sloppy.**

  Measured coverage — share of regular-season game DATES carrying any injury
  report: 2017-18 **0.0%**; **2018-19 63.7%** (first report 2018-12-17, i.e. a
  third of the way into the season); 2019-20 100%; 2020-21 96.4%; 2021-22 99.4%;
  2022-23 99.4%; 2023-24 100%; 2024-25 100%; 2025-26 95.1%. **2019-20 is the
  first fully-covered season.** Including 2018-19 put a partially-blind season
  inside the very frame that exists to guarantee the model is not blind.

  | frame | K | n | normalized gap |
  |---|---|---|---|
  | pre-feed 2007-08..2017-18 | 10 | 12,298 | 6.81% |
  | ~~2018-19+ (D179, one partial season)~~ | 8 | 9,516 | ~~13.78%~~ |
  | **2019-20+ FULLY COVERED (corrected)** | **7** | **8,286** | **13.59%** |
  | 2019-20+ excluding COVID (= certified corpus) | 5 | 6,148 | 12.88% |

  Every doc in the public repo now carries a DATA-COVERAGE CAVEAT header naming
  the 2018-12-17 start, the per-season coverage, and the fact that any 14- or
  19-season figure blends two different models.

- D187 **THE POST-2019 FRAME IS TOO SHORT TO OPTIMISE ON — AND THAT KILLS THE
  DECEMBER SKIP, WHICH I HAD RECOMMENDED ONE ENTRY EARLIER. RETRACTION OF D184's
  RECOMMENDATION.** Owner asked to re-optimise the trading strategy on the
  corrected frame, dropping high-drawdown periods with structural explanations,
  and pre-committed to skipping December.

  **(1) THE DRAWDOWNS ARE NOT PERIODS.** On the corrected frame the two deepest
  underwater stretches are **2022-02-26 -> 2025-03-08 (-30.52u over 401 bets)**
  and one spanning 83 bets; together the top two cover **49% of all bets**. These
  are multi-SEASON underwater stretches, not excisable events. "Remove the
  high-drawdown periods" is not an available lever — it would remove half the
  sample, and the periods are identified by the endpoint anyway.

  **(2) THE MEASUREMENT INSTRUMENT IS TOO BLUNT — THIS IS THE HEADLINE.** The
  2019-20+ frame yields **7 seasons and only 4 scoreable walk-forward steps**.
  A null that takes the **best of 5 RANDOM game subsets** of comparable size
  (subsets chosen at random, never by outcome) buys **+2.54 ROI points on average,
  +5.46 at the 95th percentile**. Every filter tested lands inside that band:

  | arm | ROI | n | cum | +/- (95%) |
  |---|---|---|---|---|
  | INCUMBENT no filter | **+5.50%** | 610 | +33.5u | 24.21 |
  | F1 drop first 5 games | +1.60% | 641 | +10.3u | 20.75 |
  | F1 drop first 10 games | +2.82% | 663 | +18.7u | 22.89 |
  | F1 drop first 15 games | +6.55% | 526 | +34.5u | 17.16 |
  | F0 skip December | +2.44% | 604 | +14.7u | 19.82 |

  Best arm beats incumbent by +1.05 points against a null mean of +2.54,
  p(null >= real) = 1.000. **Nothing survives. Every interval is +/-17 to +/-24
  points wide.** The frame we adopted for data integrity is too short to tune on;
  correctness and tunability are in direct tension here, and correctness wins.

  **(3) DECEMBER SKIP: RETRACTED.** Decomposed on the corrected frame:
  - incumbent, no filter **+5.50%**
  - incumbent configs, December dropped at SCORING only **+4.96%**
    -> **direct effect of not taking December bets = -0.54 pts**
  - December excluded during SELECTION as well **+2.44%**
    -> **selection-change effect = -2.52 pts**

  **Total cost of the December skip on the corrected frame: -3.06 points.**
  D184's recommendation rested on the 2018-19+ frame and on the December bets the
  ALL-ERA selection happened to place (n=77, dominated by 2022-23 at -66% on
  n=17). Re-optimised on 2019-20+ the selector places 56 December bets that are
  only mildly negative, and removing them is harmful. **I recommended this one
  entry ago and it does not survive the frame the owner correctly insisted on.**

  **(4) THE GENERAL METHODOLOGICAL FINDING, WHICH IS THE PART WORTH KEEPING:**
  the selection-change channel (-2.52) is **five times larger** than the direct
  channel (-0.54). A filter does not merely remove bets — it changes WHICH
  CONFIGURATION the walk-forward selects, and that indirect effect dominates.
  **No filter in this project may be evaluated by looking at the bets it removes;
  selection must be re-run under the filter.** D182 and D184 both failed to do
  this and are superseded on that point.

  **(5) OWNER'S BROADER HYPOTHESIS ("something wrong with that part of the
  season — trades? break?") HAS INPUT-SIDE SUPPORT BUT CANNOT BE RESOLVED ON
  P&L.** Mean Out+Doubtful listings by half-month, 2019-20+, vs the season mean:
  Oct-b -29.0%, Nov-a -3.1%, Nov-b +6.4%, **Dec-a +14.3% (peak)**, Dec-b +9.0%,
  **Jan-a +9.5%**, Jan-b -0.5%, Feb-a +2.3%, Feb-b -0.5%, Mar +6.9/+6.5%,
  Apr-a +3.5%, Apr-b -25.2%. So absence load DOES peak in early December and
  stays elevated into early January — the availability leg is working hardest
  exactly there. But the effect is modest (+14% at peak) and, per (2), the P&L
  instrument cannot resolve a window effect of any plausible size on 4 steps.
  **Recorded as a live-monitoring target, not a filter.**

  **VERDICT: NO FILTER ADOPTED. The shipped strategy remains the unfiltered
  incumbent.** December is NOT skipped.

  [SCOPE: `scripts/d185_post2018_strategy.py` NEW (read-only);
  `data/d185_post2018.json` NEW; two bugs found and fixed in it before scoring —
  s_i not re-indexed after slicing (scattered every bet into the original
  19-season columns, so the walk-forward read empty columns and returned nothing)
  and injury_reports_pit storing FULL team names against an abbrev frame (5th
  instance of the team-name bug class; nbapred/teams.py REPORTED the one
  unresolvable name, `da Silva, Tristan`, which is a known un-reloaded parser
  artefact from D178); NO GATE RUN; NO PRODUCTION MODEL DEFAULT CHANGED; no
  filter implemented; D184's December pre-commitment RETRACTED; DB READ-ONLY]

- D188 **THE DATA-LIMITATION NOTICE IS NOW THE FIRST THING ON THE REPO, BACKED BY
  A MEASURED COVERAGE FIGURE RATHER THAN PROSE.** Owner: "make sure that it is
  clear on the github that we are working with limited data until 2018."

  **MEASURED COVERAGE OF ALL FOUR MODEL INPUTS, BY SEASON**
  (`scripts/d188_coverage_chart.py` -> `charts/data_coverage.png`,
  `data/d188_coverage.csv`). The point the figure makes that the prose did not:
  **three of the four inputs are complete back to 2007-08.** DARKO player talent
  100% every season (backfilled, D170) and the opening spread 100% every season.
  **Only the injury report is missing — and it is missing ENTIRELY for 11 full
  seasons.**

  | input | 2007-08..2017-18 | 2018-19 | 2019-20+ |
  |---|---|---|---|
  | daily injury report | **0% (11 seasons)** | 63.7% | 95-100% |
  | DARKO player talent | 100% | 100% | 100% |
  | opening spread | 100% | 100% | 100% |
  | multi-book panel AT THE OPEN | none | none | **2023-24 only** (7.74 books/game; 2024-25 = 1.00, 2025-26 = 1.03) |

  This sharpens the framing considerably. The pre-2019 seasons are not "sparse
  data" in general — they are **specifically and only missing the input that the
  availability leg, i.e. half the production margin, is built from**. That is why
  those seasons score a crippled variant rather than a noisier version of the
  same thing, and it is a cleaner justification for the 2019-20 frame than the
  one D186 gave.

  **README RESTRUCTURED.** The limitation notice is now the first section, above
  the model description, with the figure inline and four consequences stated
  plainly: (1) every 14- and 19-season figure measures a different model;
  (2) the honest frame is 7 seasons and its intervals are tens of ROI points
  wide; (3) per D187 the frame is too short to tune on (best-of-5 RANDOM subsets
  buy +2.54 points); (4) multi-book execution is counterfactual outside 2023-24.
  All 23 docs carry the D186 coverage header.

  **CHART DESIGN NOTE:** coverage is a magnitude, so the encoding is a single
  sequential hue light->dark with an explicit distinct 'none' state — never a
  rainbow — and **every cell carries its number**, so nothing depends on colour
  alone.

  [SCOPE: `scripts/d188_coverage_chart.py` NEW; `charts/data_coverage.png` NEW;
  `data/d188_coverage.csv` NEW; README restructured; NO GATE RUN; NO PRODUCTION
  MODEL DEFAULT CHANGED; DB READ-ONLY]

- D189 **CONTINUOUS LOG-LOSS CHARTS REBUILT ON THE CORRECTED 2019-26 FRAME.**
  Owner: "update continuous log loss graphs from 2019" / "from 19-26". The
  existing continuous charts covered either the certified 5-season corpus
  (2021-22..2025-26) or all 19 seasons; neither is the frame this project now
  reports on after D186.

  Source `data/k19_d171_t2_pergame.csv` (the D171 re-certified 19-season run)
  sliced to 2019-20+. **Reproduces the registered pooled figure EXACTLY**:
  n=8,286, ll_us 0.61113, ll_mkt 0.59823, **gap 13.59%**.

  | season | n | gap |
  |---|---|---|
  | 2019-20 | 1,058 | **6.10%** (best) |
  | 2020-21 | 1,080 | **26.98%** (worst) |
  | 2021-22 | 1,228 | 16.95% |
  | 2022-23 | 1,230 | 13.21% |
  | 2023-24 | 1,230 | 16.34% |
  | 2024-25 | 1,230 | 6.43% |
  | 2025-26 | 1,230 | 12.43% |
  | **pooled** | **8,286** | **13.59%** |

  **THE TWO COVID SEASONS ARE THE EXTREMES IN BOTH DIRECTIONS** — 2019-20 is the
  best cell on the frame and 2020-21 the worst, and they are adjacent. They are
  KEPT because full injury coverage is the frame's only criterion and both meet
  it; removing them would be exactly the endpoint-driven subgroup selection D182
  and D187 rejected. They are flagged red in both charts so the reader can see
  how much of the frame's dispersion they carry.

  NEW files, nothing overwritten in place:
  `charts/logloss_continuous_2019_26.png` (rolling-100, one panel per season) and
  `charts/frame_model_2019_26.png` (per-season gap). The superseded
  `charts/frame_model_post2018.png` — built on the wrong 2018-19+ frame — is
  ARCHIVED to `charts_archive/frame_model_post2018__supersededByD189_20260805.png`
  and removed from the public repo, per the owner's standing preserve-don't-delete
  rule.

  [SCOPE: `scripts/d189_logloss_2019.py` NEW (read-only); 2 charts NEW; 1 chart
  archived + retired from repo; README charts section updated and both figures
  embedded inline; NO GATE RUN; NO PRODUCTION MODEL DEFAULT CHANGED; DB not
  touched]

- D191 **THE 2023-26 EQUITY PATH, PUBLISHED WITH THE HAIRCUT LINE ON THE SAME
  AXES.** Owner: "push this graph with haircut version as well."

  `charts/equity_2023_26_k8_haircut.png` — the 500-bet walk-forward path over
  2023-24..2025-26 at k=8, drawn twice:

  | arm | cum | ROI |
  |---|---|---|
  | best of 8 books, raw | **+43.62u** | **+8.72%** |
  | after outlier-realism haircut | **+29.82u** | **+5.96%** |
  | **charged by the haircut** | **13.80u** | **2.76 pts** |

  **BOTH ON ONE AXES, DELIBERATELY.** D181 removed the haircut from the headline
  by instruction, which is legitimate — but publishing the two paths as separate
  figures would let either be quoted alone, and the gap between them IS the
  execution assumption. Drawn together, the 13.80u the haircut charges is
  impossible to miss, and it is annotated in-figure rather than left to a
  caption. The haircut charges for the 8.1% of best-of-N prices sitting >1.5
  points off the next book — precisely the prices a book limits, lowers or voids.

  Figure also carries, in the subtitle rather than a footnote, the two facts that
  bound the reading: the season-clustered 95% CI on the raw path is
  **[-6.72%, +24.17%]** and therefore contains zero, and **2024-25 alone supplies
  65% of the P&L**.

  [SCOPE: `scripts/d190_equity_haircut.py` NEW (read-only); one chart NEW;
  no existing chart altered; NO GATE RUN; NO PRODUCTION MODEL DEFAULT CHANGED;
  DB not touched]

- D192 **CAN THE HAND-SET CONSTANTS BE CALIBRATED EMPIRICALLY? YES FOR SOME, AND
  THE USEFUL AXIS IS NOT TUNED-VS-FIXED BUT DERIVABLE-VS-SEARCHABLE. ALSO
  CONTAINS A RETRACTION OF THIS ENTRY'S OWN FIRST ATTEMPT.** Owner relayed an
  outside critique of architecture-level selection bias and asked whether
  constants could be calibrated instead of hand-set at round values.

  **THE CRITIQUE'S CENTRAL POINT IS CONCEDED AND ALREADY IN THE REGISTER.** The
  architecture was developed on 2021-26 and inserted into earlier walk-forward
  runs; the walk-forward tests parameter refitting and betting-rule selection but
  NOT the architecture. The ablation flipping +3.54% -> -3.70% is exactly that
  evidence and is in the README. Nothing to relitigate.

  **THE FRAMEWORK (`docs/CONSTANTS.md`, new).**
  - **TYPE A — DERIVABLE**: a deterministic function of a quantity estimable on
    the training fold. Recomputed by formula inside every fold, consumes ~ZERO
    degrees of freedom, out-of-sample BY CONSTRUCTION, needs no held-out data.
  - **TYPE B — SEARCHABLE**: no closed form, chosen by held-out comparison.
    Consumes DOF, needs the nested design the critique describes, and on 7
    seasons mostly manufactures noise (D165: 600 cells buy +16.92 ROI points from
    nothing).
  The programme is to move constants B -> A where a derivation exists, not to
  tune everything and not to leave everything alone.

  **C1 LINK SCALE 7.2 IS NOT A FREE PARAMETER — TYPE A, CONFIRMED.** A logistic
  of scale s has SD s*pi/sqrt(3); matching the margin-residual normal gives the
  plug-in s = sigma*sqrt(3)/pi using training residuals only. Margin-residual SD
  **13.653** -> plug-in **7.527**; full-frame 1-D MLE **7.721**; shipped **7.2**.
  Walk-forward (estimate on 1..k, score k+1): mean delta **-0.000212 nats**,
  season-clustered **t=-1.58 (K=5), ns**; fitted scale stable at **7.81-7.98**
  across folds. So 7.2 is slightly tight, costs **1.6% of the model-market gap**,
  and re-deriving it per fold is free and marginally better. **The single
  clearest B->A move available.**

  **C2 THE 50/50 BLEND IS SUBOPTIMAL, NEARLY COSTLESSLY — AND TWO INDEPENDENT
  METHODS AGREE.** Inverse-variance w = (var_b - cov)/(var_a + var_b - 2cov) is
  Type A. Residual var ff **211.52**, comp **203.43**, cov **196.09**
  (**corr 0.945**) -> **w = 0.322**, bootstrap 95% CI **[0.225, 0.418]**, which
  **EXCLUDES 0.5** (P(w>0.5)=0.000). The register's earlier held-out log-loss
  SEARCH independently preferred **~0.30**. **A closed form and a search agreeing
  means the 30/70 preference was never a search artifact.**
  But the objective is flat: blend residual RMSE **14.1796 at w=0.322** vs
  **14.2050 at w=0.5** — the entire distance is **0.025 pts, 0.18%**. Cause: at
  corr 0.945 the denominator var_a+var_b-2cov = **22.8**, only **10.8%** of a leg
  variance, so the optimum sits in a very shallow bowl. **0.5 is genuinely
  suboptimal and nearly costless; keeping it is defensible, and the defensible
  REASON is the flatness, not the round number.**

  **C3 RETRACTION OF MY OWN FIRST PASS.** The first attempt estimated k for
  n/(n+600) via a between/within variance decomposition on TEAM MARGINS, got
  **k~9**, and was about to report the shipped 600 as **65x too large**. **That
  was wrong and is retracted — it compared two different objects.** n/(n+600)
  (`latestate.py:70` C_SHRINK, `tanking.py:46`) shrinks a **FITTED COEFFICIENT**
  where **n = active fit rows**, not a team's game count; it is a burn-in guard
  (0.143 at n=100, 0.500 at n=600, 0.952 at n=12,000). At realistic row counts it
  is ~0.9-0.95, a mild residual shrink. What IS free is where burn-in releases,
  and the Type-A alternative is `b^2/(b^2+se^2)` from the fit's own standard
  error. **Not implemented; recorded as the strongest remaining B->A candidate.**

  **HALL OF SHAME:** I nearly published "the shipped shrinkage is 65x too large"
  off a variance decomposition that measured a different quantity than the
  constant governs. **Before comparing an estimate to a shipped constant, read
  what the constant is applied to.** A units/object mismatch produces a
  confident, precise, entirely wrong number — and this one would have been
  published in response to an outside critique, which is the worst possible
  moment to be wrong.

  **STILL UNVALIDATED, ranked as the next B->A candidates:** ridge 25 (standard
  derivations exist — generalised cross-validation or marginal likelihood, both
  fold-internal), team_home_ridge 200, lookback and roster windows.

  [SCOPE: `scripts/d192_empirical_constants.py` NEW (read-only);
  `data/d192_constants.json` NEW; `docs/CONSTANTS.md` NEW; NO GATE RUN; NO
  PRODUCTION MODEL DEFAULT CHANGED; no constant altered anywhere in the model;
  DB READ-ONLY]

- D193 **THE IDENTICAL-FRAME MODEL / OPEN / CLOSE BENCHMARK — THE MODEL DOES NOT
  BEAT THE OPENING LINE AS A GENERAL FORECASTER, AND OUR ONE PROFITABLE SEASON
  WAS THE MARKET'S WORST OPENING SEASON RATHER THAN OUR BEST MODELLING.**
  Priority 1 of an outside roadmap: the repo reported model LL and CLOSING LL on
  the 2019-26 frame but OPENING LL only on a different 1,892-game moneyline
  frame. Those cannot be combined. They are now computed on identical games:
  **n=8,237** carrying model, open AND close (of 8,239 in frame).

  **(A) MARGIN SPACE — no devig, no link, no conversion of any kind:**

  | source | RMSE | MAE | bias |
  |---|---|---|---|
  | model | **13.6533** | 10.6943 | -0.2718 |
  | **opening line** | **13.5958** | 10.5950 | -0.1677 |
  | closing line | **13.3692** | 10.4201 | -0.1354 |

  **THE OPENING LINE IS MORE ACCURATE THAN OUR MODEL** (-0.0575 pts RMSE).
  open->close improvement +0.2266 pts; open->model **-0.0575**.
  **CAPTURE FRACTION (margin) = -0.254.**

  **(B) PROBABILITY SPACE — each source gets its OWN walk-forward logistic
  scale, so none is handicapped by another's calibration.** Pooled: model
  **0.60524**, open **0.60501**, close **0.59257**. Fitted scales 7.887 / 7.834 /
  7.759.
  - LL(open) - LL(close) = **+0.01244**  (all the open->close information)
  - LL(open) - LL(model) = **-0.00023**  (what we recover)
  - **CAPTURE FRACTION = -0.019.** Per season **[-0.173, -0.169, -0.663, +0.670,
    -0.862]**, mean -0.240, 95% CI [-0.975, +0.496], **positive in 1 of 5**.

  **THE 2024-25 EXPLANATION, WHICH REFRAMES THE ENTIRE BETTING RESULT.** The one
  season with positive capture is 2024-25 (+0.670) — the same season supplying
  **65% of the walk-forward P&L**. Its open->close gap was **+0.02311**, against
  0.01308 / 0.00863 / 0.00979 / 0.00748 in the others: **roughly 3x the typical
  season, i.e. the openers that year were unusually uninformative.** Our model's
  own 2024-25 log loss (0.58857) is ordinary — essentially the same as 2025-26's
  0.58659. **So our best season was the market's worst opening season, not our
  best modelling.** That is a market-side explanation for the profit
  concentration, and it predicts the edge does not persist unless openers stay
  soft.

  **WHAT THIS DOES AND DOES NOT OVERTURN.** It does NOT contradict the ATS
  result (50.65% cover vs a coin flip, significant): that is a SELECTED subset,
  and a selector can be right about which games to bet while the general
  forecast is not better than the opener. It DOES mean the README's framing must
  distinguish the two, and it means **the deployable model should take the
  opening line as its prior rather than ignoring it** — the roadmap's item 2 is
  now motivated by measurement rather than principle.

  **CANARY SHIPPED (`scripts/canary.py`).** Nine checks, every one traceable to a
  failure that actually happened here: odds_quotes populated (FAIL in-season —
  the logger has never run and the open-price CLV record is unrecoverable if it
  is down on opening night), >=2 books at the open, injury-feed freshness,
  DARKO freshness, team-name resolvability (the four-instance bug class),
  non-002 contamination in bet_paper, tank-floor drift, CLV inside the D178
  band, and calibration sanity. Exit 1 on any FAIL so cron can alert.
  First run: 4 pass / 3 warn / 0 fail. Two of the warns are known and open —
  `odds_quotes` empty, and `da Silva, Tristan` still in the injury table because
  the D178 parser fix was never followed by a reload.

  **ONE CANARY BUG CAUGHT BY WRITING IT:** I first classified tank-floor drift as
  FAIL. `tanking.floor_audit`'s own contract says drift does NOT mean anything is
  broken — it means the data now supports a different floor and moving there is a
  gated model change. Downgraded to WARN. **A monitor that cries wolf on a
  working safeguard is worse than no monitor**, and the derived floor is now
  1996-97 vs pinned 2020-21 precisely because D170's backfill extended box-score
  coverage — the pin is doing its job.

  [SCOPE: `scripts/d193_open_benchmark.py` NEW (read-only); `scripts/canary.py`
  NEW (read-only, never writes); `data/d193_open_benchmark.json` NEW; NO GATE
  RUN; NO PRODUCTION MODEL DEFAULT CHANGED; DB READ-ONLY]

- D194 **RECERTIFIED ON THE RELOADED INJURY TABLE — AND IT IS A NO-OP, FOR AN
  INSTRUCTIVE REASON.** Owner: "recertify". The D178 parser fix had never been
  followed by a table reload.

  **RELOAD.** `scripts/bf_injury_load.py` reparsed all 1,259 archived PDFs, 0
  failures -> 126,378 rows -> 125,704 after dedup (was 125,695, **+9**).
  **Distinct team values 31 -> 30, unresolvable 1 -> 0**: `da Silva, Tristan` is
  gone and now appears correctly as a PLAYER with 15 rows (7 Questionable + 1 Out
  + 1 Available for Orlando, 6 Out for New Orleans — he was traded).

  **RECERTIFICATION: IDENTICAL.** Pooled certified gap **12.87% before and
  after**; ll_us 0.60541 and ll_mkt 0.59244 both unchanged; **zero of 6,148 games
  changed p_us** (max |delta| 0.00000). The capstone md5 moved
  (695d40a3 -> 2c4feac6) **only because row order changed** — every numeric column
  is byte-identical. DB backed up to `nba.duckdb.pre_D194_backup` first.

  **WHY IT IS A NO-OP, AND THIS IS THE PART WORTH KEEPING.** The T2-HONEST
  availability tier is `5PM official injury report UNION official pregame
  inactives`. All **5/5** of da Silva's Out dates are covered by the inactives
  feed, so the UNION masked the parse error completely. **The certified backtest
  is structurally robust to injury-report parse errors, because a second feed
  carries the same information.**

  **THE ASYMMETRY THAT MATTERS FOR TRADING.** That robustness does NOT extend to
  the live path. Official inactives are released ~30 minutes before tip; **they do
  not exist when the opener is posted.** At the OPEN the injury report is the only
  availability feed, so a parse error there is NOT masked by anything. **The
  backtest is protected where the live path is exposed** — which means injury-PDF
  parse health is a LIVE-PATH concern that no backtest number will ever surface.
  It is in `scripts/canary.py`.

- D195 **THE MARKET-OFFSET MODEL — THE SIGN FLIPS. SAME INFORMATION, EXPRESSED AS
  A RESIDUAL ON THE OPENER, GOES FROM -0.019 TO +0.282 CAPTURE.** D193 showed the
  market-blind model does not beat the opening line as a general forecaster. This
  builds the consequence: `m_final = open_margin + f(x)`.

  Declared before scoring: baseline to beat is **f=0, i.e. THE OPENER ITSELF**
  (D176 — the incumbent, not a null). Five features, all knowable at the open:
  model_edge (= m_us - open_margin), rest_diff, TRAILING absence_diff (never
  tonight's 5pm report, which does not exist at the open), gidx, |open_margin|.
  Ridge with lambda by **generalised cross-validation inside the training fold**
  — Type-A per D192, no grid, no held-out data. Walk-forward: fit 1..k, score k+1.

  | source | RMSE | log loss |
  |---|---|---|
  | market-blind model | 13.7294 | 0.60524 |
  | opener | 13.6898 | 0.60501 |
  | **offset model** | **13.6320** | **0.60150** |
  | close | 13.4329 | 0.59257 |

  | | market-blind | **offset** |
  |---|---|---|
  | capture (log loss) | -0.019 | **+0.282** |
  | capture (margin) | -0.254 | **+0.225** |

  **Per-season LL gain over the opener: [+0.00228, +0.00034, +0.00263, +0.01070,
  +0.00149] — POSITIVE IN 5/5.** Mean +0.00349, t=+1.89 (K=5) **ns by t**, but a
  one-sided sign test on 5/5 is **p=0.031**. Mixed but leaning, and far stronger
  than anything the market-blind stack has shown against the opener.

  **THE COEFFICIENT THAT MATTERS: model_edge +0.351.** Our market-blind
  disagreement with the opener is worth **~35% of its face value** — when the
  model says the line is 3 points wrong, about 1 point of that is real and the
  ridge shrinks the rest away. That is the model earning its keep as a *residual
  signal* while failing as a *standalone forecast*, and it is exactly what a
  strongly-regularised offset architecture is for.

  **CAVEAT, STATED PLAINLY:** the largest single-season gain is 2024-25
  (+0.01070), which D193 identified as the season the OPENERS were unusually bad
  (open-close gap 3x normal). 5/5 positive means it is not solely that, but the
  magnitude is not evenly earned.

  **NOT SHIPPED.** This is a challenger. The production stack is untouched and
  remains market-blind. `nbapred/market/anchored.py` already carries the MOVEMENT
  head (D147); this is the missing OUTCOME head, and the two are the pieces of a
  market-aware deployable model.

- D196 **odds_quotes: THE LOGGER WAS DEAD, THE WRITE PATH WAS NEVER PROVEN, AND
  BOTH ARE NOW FIXED.** Owner: "set up odds quotes".

  **DIAGNOSIS.** `odds_quotes` at 0 rows had never been distinguished from
  "broken". It is both. (a) The `@reboot ... nohup` cron daemon **stopped on
  2026-07-27** and nothing restarted it — 10 days dead. (b) When it WAS running it
  logged `0 events, 500 credits left`, which is CORRECT offseason behaviour: no
  NBA events -> no rows. So the empty table was expected, but the process being
  dead was not, and nothing would have caught it.

  **WRITE PATH PROVEN END-TO-END** rather than assumed: a synthetic snapshot in
  the exact API shape (sentinel date 1999-01-01, `event_id='CANARY_TEST_EVENT'`,
  two bookmakers) was loaded by `scripts/load_odds.py` -> **4 quote rows, 2
  distinct bookmakers, i.e. the >=2-book rule would be SATISFIED**. Test rows and
  the sentinel file were then deleted; `odds_quotes` back to 0.

  **DURABILITY.** Replaced the `@reboot nohup` entry with a **systemd user
  service** (`~/.config/systemd/user/nba-odds-logger.service`, `Restart=always`,
  `RestartSec=30`), `loginctl enable-linger` set so it runs without a login
  session. **Verified by `kill -9`: the service came back with a new PID within
  35 seconds.** The superseded cron line was removed so there is exactly one
  owner of this process.

  [SCOPE D194/D195/D196: injury_reports_pit RELOADED (DB write, backed up first);
  capstone re-run, outputs numerically identical; `scripts/d195_market_offset.py`
  NEW; `data/d195_offset.json` NEW; systemd user unit NEW; crontab @reboot line
  REMOVED; `REPO_NOTE.md` NEW (remote removed from nba_model); NO GATE RUN; NO
  PRODUCTION MODEL DEFAULT CHANGED; the offset model is a CHALLENGER, not shipped]

- D197 **THE CANONICAL POINT-IN-TIME FRAME (roadmap item 1).**
  `data/pit_frame.csv.gz`, 8,239 games 2019-20..2025-26, 26 columns, every one
  labelled with the information tier at which it becomes knowable: **T0** before
  the opener (gidx, rest, TRAILING absence load), **T1** the opener, **T2** the
  same-day 5PM report, **T3** pregame inactives, **T4** the close, **T5** the
  outcome. Coverage T0 100/95.7%, T1 100%, T2 95.7%, T3 99.8%, T4 100%, T5 100%.
  Open/close probabilities use ONE shared logistic transform so neither source is
  advantaged (D193's rule). **Two bugs found while building it, both silent:**
  the oc frame stores `game_id` as an INT (`20700001`) while the DB uses the
  zero-padded VARCHAR (`0022400211`), so every DB join returned nothing (T3 read
  0.0%); and the inactives merge added 10 rows through duplicate keys. Both are
  now guarded by row-count assertions.

- D198 **ITEMS 3 AND 5.**
  **STACK (item 3): no change earned.** Ridge shrunk toward the incumbent (0.5,
  0.5) on m_ff/m_comp: unshrunk OLS is (+0.337, +0.739) at RMSE 14.2147 against
  the incumbent's 14.2316 — **0.12%**. Consistent with D192 (components correlate
  0.945; flat loss surface). Component coverage is 3 seasons, so there is no
  held-out season to earn a change with. **Incumbent retained.**
  **CONDITIONAL VARIANCE (item 5): SHIPS THE CONTROL TEST.** Student-t(nu=5) with
  log sigma linear in T0/T1 uncertainty inputs, walk-forward, against the shipped
  sigmoid(mu/7.2): pooled LL **0.60518 vs 0.60549**, mean delta **-0.000316**,
  **t=-4.08 (K=5) SIGNIFICANT, better in 5/5 seasons.** But the effect is tiny and
  the mechanism is NOT what was hypothesised: fitted sigma varies only ~0.5% per
  SD of its strongest input (sd 0.11-0.38 around a mean of ~11.3). **Essentially
  all of the gain is the intercept — the model wants sigma ~11.3, i.e. a link
  scale nearer 7.8 than 7.2, which is exactly D192's finding arriving by a second
  route.** The conditional part contributes almost nothing. Recorded as
  confirmation of the link-scale result, NOT as evidence for state-dependent
  variance.

- D199 **BET-TIME LEAKAGE: THE OWNER ASKED THE RIGHT QUESTION AND THE ANSWER IS
  YES, WORTH 33% OF OUR ENTIRE DEFICIT TO THE MARKET.** Owner: "do we have
  lookahead leakage because we are oracling injuries/lineups but betting at open?"

  **THE MISMATCH.** `docs/LEAKAGE.md` licenses the 5PM report and pregame
  inactives as LEGITIMATE inputs, justified by "close forms after these". **That
  is a CLOSE-TIME licence.** Every open-priced betting study in this repo uses a
  model built on those inputs while transacting at a price posted BEFORE them.
  The policy was never wrong; it was applied to the wrong bet time.

  **EXPOSURE, MEASURED.** Comparing each team-day's out-set to the previous
  report: **81.9% of tonight's out-set was already known, 18.1% is new on the
  day**; minutes-weighted the new share is **19.3%**; **44.3%** of team-days carry
  at least one late scratch, and only **35.3%** are unchanged. Late scratches are
  marginally heavier than known absences (18.93 vs 17.87 MPG), so talent-weighting
  does not rescue it.

  **COST, MEASURED.** New `OPEN_TIME_OUTS=1` in `prod_by_season.py` rebuilds the
  out-set BY CARRY-FORWARD from the most recent report strictly before game day,
  and drops the inactives union entirely.

  | availability | ll_us | normalized gap |
  |---|---|---|
  | CERTIFIED (same-day + inactives) | 0.60541 | **12.87%** |
  | **AS-OF-OPEN (carry-forward)** | **0.60973** | **17.17%** |

  **+0.00433 nats, gap widens 4.30 points. The late information is worth 33% of
  the model's entire deficit to the market.**

  **A CONSTRUCTION ERROR I MADE AND CAUGHT.** My first as-of-open query used only
  rows with `report_date < game_date`. The archive holds 93,110 same-day rows and
  32,594 previous-evening rows, so that query covered a minority of game-days and
  outs/team collapsed to 0.22-0.35 against a true ~1.2. **A bettor at the open
  knows the LAST PUBLISHED STATUS CARRIED FORWARD, not only explicitly-advance
  reports.** Rebuilt as carry-forward: coverage 100%, outs/team 0.40-0.59.

  **THE RESULT THAT MATTERS — THE OFFSET ARCHITECTURE SURVIVES, THE STANDALONE
  MODEL DOES NOT.** Same games, same protocol, only the availability inputs
  change:

  | | contaminated | **honest** |
  |---|---|---|
  | market-blind capture vs opener | +0.075 | **-0.104** |
  | **offset-model capture vs opener** | +0.313 | **+0.222** |

  **The market-blind model's apparent edge over the opener FLIPS NEGATIVE once it
  can only see what a bettor could see. The offset model degrades ~29% and stays
  clearly positive.** That is the strongest argument yet for the market-anchored
  architecture: because its ridge already shrinks the model edge to ~35% of face
  value (D195), it was never leaning on the contaminated signal.

  **WHAT THIS INVALIDATES.** Every open-priced ROI/ATS/CLV figure in this
  repository was produced with a model that saw ~19% of its availability signal
  after the price it transacted at. They are OPTIMISTIC by an unquantified amount
  and must be re-run on `OPEN_TIME_OUTS=1` before being quoted again.

  [SCOPE: `scripts/d197_pit_frame.py`, `scripts/d198_stack_and_variance.py` NEW;
  `data/pit_frame.csv.gz`, `data/capstone_pergame_opentime.csv`,
  `data/prod_by_season_opentime.json` NEW; `scripts/prod_by_season.py` gains
  `OPEN_TIME_OUTS` (additive switch, DEFAULT UNCHANGED, backup in scratchpad);
  NO GATE RUN; NO PRODUCTION MODEL DEFAULT CHANGED; DB READ-ONLY]

- D200 **THE PARTICIPATION MODEL — THE DIRECT FIX FOR D199, AND IT WORKS.**
  D199 established that at the open we know only the LAST PUBLISHED status, and
  that the missing 18-19% is worth 33% of our deficit to the market. The response
  is not to accept a degraded hard out-set but to **forecast it**: replace the 0/1
  carry-forward flag with a calibrated `P(out tonight | as-of-open information)`.

  Design declared before scoring. Unit: player-team-gameday with a same-day label.
  Features strictly T0: last published status, days since it, consecutive prior
  Out reports, consecutive prior Questionable/Doubtful reports, team rest.
  L2 logistic, Newton-IRLS, walk-forward 1..k -> k+1. **Incumbent to beat (D176):
  the hard carry-forward rule "out tonight iff out on the last report".**

  | | Brier | log loss |
  |---|---|---|
  | hard carry-forward (incumbent) | 0.13145 | 0.53177 |
  | **participation model** | **0.10437** | **0.34924** |

  **Brier -20.6%, better in 4/4 seasons.**

  **WHERE THE GAIN COMES FROM — the two numbers that matter:**
  - players last listed **QUESTIONABLE** (2,006 rows in the last scored season)
    are **actually out 28.9%** of the time. The model says **26.5%**. **The hard
    rule says 0.000** — it treats every one of them as definitely available.
  - players last listed **OUT** are **actually out 91.5%**. The model says 92.4%.
    **The hard rule says 1.000** — it over-commits on the 8.5% who do play.

  So the carry-forward out-set is wrong in BOTH directions, and a calibrated
  probability fixes both. Fitted coefficients are sensible and monotone in
  severity: ls_Out +2.06, ls_Doubtful -0.08, ls_Questionable -0.60,
  ls_Probable -1.38; streak_out +0.06 (a long-standing absence persists),
  streak_q -0.10 (a chronic Questionable tends to play), rest -0.09.

  **BLOWOUT TRAP — MY DECISION, AS ASKED.** Participation is fitted with NO
  game-state or competitiveness feature. A player's minutes are truncated by a
  blowout and the blowout is caused by the margin being predicted, so any
  expected-margin, pace or total input would leak the outcome backwards into the
  availability leg. Rest and schedule are admissible (fixed before tip);
  competitiveness is not, and none appears. **Validation is on the availability
  label directly (Brier/LL), never on downstream game log loss** — otherwise an
  improvement here is indistinguishable from a change anywhere else in the stack.

  **NOT YET INTEGRATED.** The composition leg currently takes a hard OUT set;
  consuming `1 - P(out)` as a weight is a production change requiring a gate and a
  re-certification. **The measured prize is D199's 4.30 points of normalized gap;
  this entry establishes that the input needed to chase it is materially better
  than the incumbent, not that the gap has been recovered.**

  [SCOPE: `scripts/d200_participation.py` NEW (read-only);
  `data/d200_participation.json` NEW; NO GATE RUN; NO PRODUCTION MODEL DEFAULT
  CHANGED; composition leg UNCHANGED; DB READ-ONLY]

- D201 **SOFT AVAILABILITY INTEGRATED.** `CompositionModel.strength()` now accepts
  either a SET of out player-ids (hard; byte-identical to prior behaviour — set
  form and `dict{p:1.0}` agree to 1e-12, `dict{p:0.0}` equals no-one-out,
  `dict{p:0.5}` sits exactly halfway) or a DICT `{player_id: P(out)}` weighting
  each player by `1-P(out)`. `prod_by_season.py` gains `SOFT_AVAIL=1` (implies
  `OPEN_TIME_OUTS=1`), consuming `data/p_out.csv.gz` from
  `scripts/d201_pout_artifact.py`. The artifact is walk-forward by construction:
  season k+1's probabilities come from a model fitted on 1..k only; the first
  season falls back to the hard rule so it is never worse than status quo there.
  **42.8% of artifact rows sit in (0.1, 0.9)** — the region a hard 0/1 discards.
  **CONTROL: the default path reproduces the certification to max |dp| = 9.8e-15.**

- D202 **GATE: SHIP.** Spec SHA-256 `25d1f85f18069f48b484bc0f6c078f89`, MDE80
  stated before the endpoint was read. Incumbent = HONEST-HARD carry-forward;
  challenger = HONEST-SOFT.

  | | |
  |---|---|
  | season-clustered mean delta | **-0.002265 nats** |
  | 95% CI (4 dof) | **[-0.004118, -0.000412] — EXCLUDES ZERO** |
  | t | **-3.39** |
  | MDE80 (stated first) | 0.002415 |
  | better in | **5/5 seasons** |
  | calibration veto | PASS (mean p 0.5587 vs base 0.5534) |

  | availability | log loss | normalized gap |
  |---|---|---|
  | contaminated | 0.60541 | 12.87% |
  | honest-hard | 0.60973 | **17.17%** |
  | **honest-soft (SHIPPED)** | **0.60747** | **14.92%** |

  **Soft availability recovers 52% of D199's leakage penalty using only what is
  public at the open.** Expected outs/team rises 0.40-0.59 -> 0.91-1.10 against
  the contaminated 1.09-1.38.

- D203 **HONEST RERUN OF EVERY PUBLISHED FIGURE, AND WHAT IT COSTS.** Trading
  frame spliced (6,108 of 22,742 games carry the honest margin, mean |delta m_us|
  0.42 pts), adaptive arms regenerated, `wf_equity.py` rerun. **The reproduction
  guard correctly refused the first attempt** until the arms matched the new
  frame — the guard working exactly as designed.

  | tier | contaminated | **HONEST** | delta |
  |---|---|---|---|
  | k=1 raw | +1.83% / +31.6u | +1.83% / +32.5u | 0.00 |
  | k=5 raw | +3.96% / +68.6u | +3.73% / +66.3u | -0.23 |
  | **k=9 raw (14-season)** | **+4.63% / +80.3u** | **+4.33% / +77.0u** | **-0.30** |
  | k=9 +haircut | +3.65% / +63.3u | +3.47% / +61.6u | -0.19 |
  | exchange c=2% | +5.58% / +96.7u | +5.58% / +99.2u | 0.00 |

  **THE DIRECTION DEPENDS ON THE WINDOW, AND BOTH ARE PUBLISHED.** On the
  14-season frame the honest result is LOWER (+4.33% vs +4.63%). On the 2023-26
  window it is HIGHER (**+10.15% vs +8.72%** at k=8 raw) — but with a far wider
  interval (**[-18.79, +39.09]** vs [-6.72, +24.17]) and **worse concentration:
  2024-25's share of the window's P&L rises from 65.2% to 84.4%.** The honest
  2023-24 season goes NEGATIVE (-0.65% vs +3.09%). Reporting only the 3-season
  improvement would be the flattering half of a two-sided correction.

  **MODEL vs OPENER vs CLOSE, identical games, each source its own walk-forward
  scale** — the comparison that decides whether any of this is information:

  | model variant | model LL | opener | close | capture | offset-capture |
  |---|---|---|---|---|---|
  | contaminated | 0.59127 | 0.59228 | 0.57870 | +0.075 | +0.313 |
  | honest-hard | 0.59370 | 0.59228 | 0.57870 | **-0.104** | +0.222 |
  | **honest-soft (shipped)** | **0.59276** | 0.59228 | 0.57870 | **-0.035** | **+0.267** |

  **Standalone we still do not beat the opening line** (-0.035), though the gated
  soft model recovers two thirds of the way back from the hard rule's -0.104.
  **Against the close we are well behind**: 0.59276 vs 0.57870, and capture of 1.0
  would mean matching it. **The offset architecture closes 26.7% of the
  open-to-close gap** and is the only construction that is positive at all.

  **SIM REPORT rerun:** 861 sessions, 1,777 bets, net **$769,855** (was $802,501),
  $894/day, Sharpe 0.5, win days 48%, max drawdown **-$343,893** (was -$305,240),
  edge **433 bps**, **9/14 seasons profitable** (was 10/14), pooled ROI +4.33%
  with a season-clustered 95% CI of **[-1.10, +9.77]**.

  **PUBLICATION.** The owner's standing rule was to publish only if PnL improved.
  On the primary 14-season frame it did not. **The owner overrode that rule and
  published in both directions**, which is the right call: the README already
  stated these figures were being re-run, the correction is 0.30 ROI points
  against an interval +/-15 points wide, and a register that publishes only
  favourable corrections is worth less than one that publishes all of them. The
  five contaminated chart renders are archived under `charts_archive/` with a
  `__contaminated_20260806` suffix rather than deleted.

  [SCOPE: `nbapred/model/composition.py` soft-availability support (ADDITIVE,
  default byte-identical); `scripts/prod_by_season.py` gains `SOFT_AVAIL`;
  `scripts/d201_pout_artifact.py` NEW; `data/p_out.csv.gz`,
  `data/capstone_pergame_soft.csv`, `data/ats19_frame_honest.csv.gz`,
  `data/wf_equity_HONEST.json`, `data/as_adaptive_HONEST.json`,
  `data/d202_gate.json`, `data/d203_report_honest.json` NEW; 5 charts
  regenerated, contaminated renders ARCHIVED; GATE RUN AND PASSED (D202);
  PRODUCTION DEFAULT STILL UNCHANGED — SOFT_AVAIL is opt-in pending the owner's
  re-certification call]

- D204 **THE OFFSET CONSTRUCTION, TRADED THROUGH PRODUCTION MACHINERY FOR THE
  FIRST TIME — AND IT IS BETTER ON EVERY DIMENSION, NOT JUST THE HEADLINE.**
  Owner: "have we tried to trade on offset construction?" Honest answer at the
  time: only in a scratch harness (REVIEW_v2 Appendix A), never through
  `as_adaptive` -> `wf_equity`, the pipeline that produced every published
  trading figure. Now it has been.

  **CONSTRUCTION.** `scripts/d204_offset_frame.py` writes
  `data/ats19_frame_offset.csv.gz`, whose `m_us` IS `open_margin + f(x)` with `f`
  a ridge (lambda=3000, hard shrink toward "the opener is right") **fitted
  walk-forward** — season k+1's margins come from a fit on seasons 1..k only.
  Features knowable at the open: edge vs the opener, rest differential,
  |open_margin|. Base margins are the D203 honest frame. Fitted edge coefficient
  is stable at **0.33-0.37** across every fold, and the ridge shrinks mean
  disagreement with the opener from **2.085 to 0.960 points (46%)**.

  **SCORED ON 2023-26 ONLY**, per the owner: that is the measured multi-book
  window (2023-24 has 7.74 books/game at the open; the two after it have 1.00 and
  1.03). Trained on all prior history.

  | tier | BLIND (honest, D203) | **OFFSET** | delta |
  |---|---|---|---|
  | k=1 raw | +4.44% / +23.6u | **+10.63% / +48.9u** | +6.19 |
  | k=5 raw | +8.90% / +47.3u | **+15.31% / +70.4u** | +6.41 |
  | k=9 raw | +10.21% / +54.3u | **+16.62% / +76.4u** | +6.40 |
  | k=9 +haircut | +7.87% / +41.9u | **+14.36% / +66.0u** | +6.48 |
  | exchange c=2% | +8.31% / +44.2u | **+14.72% / +67.7u** | +6.41 |

  **+6.4 ROI points at EVERY execution tier**, including k=1 — the gain is model,
  not shopping, and multi-book adds to it rather than explaining it.

  **THE QUALITY IMPROVES MORE THAN THE HEADLINE DOES, WHICH IS THE PART THAT
  MATTERS:**

  | | BLIND | **OFFSET** |
  |---|---|---|
  | per-season ROI | −0.60 / +22.20 / +6.64 | **+13.77 / +24.00 / +5.05** |
  | positive seasons | **2/3** | **3/3** |
  | best-season share of P&L | **84%** | **66%** |
  | 95% CI (K=3) | [−18.73, +39.16] | **[−6.95, +40.18]** |
  | bets | 532 | 460 |

  The blind arm is NEGATIVE in 2023-24 and leans 84% on 2024-25 — the season D193
  identified as having the worst openers. **The offset arm is positive in all
  three, is markedly less concentrated, and has a tighter interval.** It also bets
  LESS (460 vs 532): more selective, not more active.

  **STILL NOT SIGNIFICANT.** K=3, and the interval contains zero. This is the
  strongest trading result in the register and it is still one season of bad luck
  from nothing. On the full 14-season frame the same construction returns
  **+7.57% / +121.8u at k=9** against the blind arm's +4.33% / +77.0u, with cover
  **56.4%** against a 52.38% breakeven.

  **WHY IT WORKS, in one number:** the fitted edge coefficient is ~0.35, so the
  construction spends the market-blind model's disagreement at about a third of
  face value. That is also why it survived D199's leakage correction when the
  standalone model did not — it was never leaning on the contaminated signal.

  [SCOPE: `scripts/d204_offset_frame.py` NEW; `data/ats19_frame_offset.csv.gz`,
  `data/as_adaptive_OFFSET.json`, `data/wf_equity_OFFSET.json`,
  `data/wf_perbet_OFFSET.json` NEW; NO PRODUCTION MODEL DEFAULT CHANGED — the
  offset construction remains a CHALLENGER; the shipped stack is still
  market-blind; DB READ-ONLY]

- D205 **CAN WE SHIP BOTH AS COMPETITORS? NO — THEY ARE NESTED, NOT INDEPENDENT.
  BUT BOTH MUST RUN ANYWAY, FOR REASONS THAT ARE NOT DIVERSIFICATION.** Owner
  asked what best-season share is, which model is truly better, and whether both
  can ship as competitors.

  **BEST-SEASON SHARE** = the fraction of a window's total P&L contributed by its
  single best season. At K=3 no interval can separate "an edge" from "one lucky
  season", but concentration reads on it directly: 84% means the other two
  seasons produced 16% between them, i.e. the result is largely one season
  wearing a three-season costume. Lower is better. Blind 84%, offset 66%.

  **WHICH IS BETTER: THE OFFSET MODEL, ON EVERY MEASURED DIMENSION.** ROI +6.4
  points at every tier including k=1; positive 3/3 seasons vs 2/3; best-season
  share 66% vs 84%; interval [-6.95,+40.18] vs [-18.73,+39.16]; fewer bets (460
  vs 532, i.e. more selective); stable fitted coefficient (0.33-0.37 every fold);
  interior threshold selection where the blind arm ran to the grid boundary; and
  it survived D199's leakage correction (capture +0.267) where the blind arm
  flipped negative (-0.104). On the 14-season frame it is also better and better
  powered: +7.57% / +121.8u vs +4.33% / +77.0u, cover 56.4% vs a 52.38%
  breakeven. **There is no dimension on which the blind arm wins.**

  **BOTH AS COMPETITORS: NO. MEASURED.**
  - On 2023-26 k=9 they share **297 games** — 65% of offset's bets, 56% of
    blind's. On every one of those shared games they take the **same side**:
    identical P&L sign 297/297, **per-bet correlation +1.000**.
  - Daily return correlation across 325 sessions: **+0.573**.
  - A 50/50 portfolio at the same total stake returns **+13.18%, sd 1.012**
    against offset-alone's **+16.62%, sd 1.093**. It buys **-7.4% volatility for
    -20.7% return.**

  | arm | ROI | daily sd | return per unit risk |
  |---|---|---|---|
  | blind | +10.21% | 1.188 | 8.6 |
  | 50/50 blend | +13.18% | 1.012 | 13.0 |
  | **offset alone** | **+16.62%** | 1.093 | **15.2** |

  **The blend is strictly worse than offset alone on risk-adjusted return.**
  Running both as a portfolio is not diversification, it is diluting the better
  signal with a worse, highly correlated one.

  **BUT BOTH MUST RUN, FOR THREE NON-PORTFOLIO REASONS:**
  1. **The blind model is an INPUT to the offset model.** Its disagreement with
     the opener is the offset model's dominant term (coefficient ~0.35). There is
     no version of this where offset runs and blind does not. They are nested.
  2. **Different failure modes.** The blind model needs no price at bet time; the
     offset model has no anchor if the odds feed dies. Blind is the degraded-mode
     fallback, and D196 established the odds logger has already died once.
  3. **It is the live control.** K=3 cannot settle this retrospectively. Running
     both prospectively in 2026-27, scoring the same games, is the only way the
     +6.4-point difference gets resolved — and D176's rule requires a challenger
     be measured against the incumbent, not a null.

  **DECISION: offset as PRIMARY challenger, blind as its required input,
  degraded-mode fallback, and live control. Not a portfolio. Neither is shipped
  as the production default yet — that is a re-certification.**

  [SCOPE: analysis only; no code changed; no default changed; DB not touched]

- D206 **TWO SUMMARY REPORTS, ONE PER ARM — AND A CONFIDENCE-INTERVAL BUG I
  CAUGHT WHILE WRITING THEM THAT WOULD HAVE CLAIMED FALSE SIGNIFICANCE.**
  Owner asked for the summary in both a market-offset and an updated
  market-blind version. `docs/SIM_REPORT_OFFSET.md` NEW,
  `docs/SIM_REPORT.md` updated, cross-linked, identical machinery and games.

  **THE BUG.** `scripts/d181_report_tables.py` built its season-clustered
  interval by centring on the **n-WEIGHTED pooled ROI** while taking the standard
  deviation from the **UNWEIGHTED per-season ROIs**. Those are different
  quantities. For the offset arm it produced **[+1.43, +13.71] — excluding
  zero** — and I was one paragraph from publishing "the first statistically
  significant trading result in this project."

  **CAUGHT BY DISAGREEMENT WITH THE PRODUCTION PIPELINE**, which reported
  **[-0.62, +11.66]** for the same tier. `wf_equity` uses `oc.cluster_mean_t`,
  which centres on the unweighted mean of per-cluster estimates — the definition
  GATE_POLICY_V2 §8 specifies. The pipeline was right; the report generator was
  wrong. Corrected:

  | arm | pooled (n-wtd) | clustered mean | 95% CI (K=14) | |
  |---|---|---|---|---|
  | market-blind | +4.33% | +4.12% | [-1.32, +9.55] | contains zero |
  | market-offset | +7.57% | +5.52% | [-0.62, +11.66] | **contains zero** |

  **NEITHER ARM IS SIGNIFICANT.** The offset arm is better on every point
  estimate — net $1,217,630 vs $769,855, Sharpe 0.9 vs 0.5, 10/14 vs 9/14
  profitable seasons, edge 757 vs 433 bps, at comparable drawdown — and it is
  closer to clearing zero, but it does not clear it.

  **HALL OF SHAME:** a confidence interval must be centred on the SAME quantity
  it takes its dispersion from. Mixing a weighted point estimate with an
  unweighted sd is not a rounding difference — here it moved the interval far
  enough to flip the verdict. **Two independent implementations disagreeing is
  what caught it; had the report generator been the only path, it would have
  shipped.** Both published reports now carry the correction inline rather than
  silently showing the fixed number.

  [SCOPE: `docs/SIM_REPORT_OFFSET.md` NEW; `docs/SIM_REPORT.md` corrected;
  `scripts/d181_report_tables.py` CI bug FIXED; `scripts/d206_offset_chart.py`
  and `charts/sim_report_equity_offset.png` NEW; no model default changed]

- D207 **THE TRADING FRAME WAS WRONG BY SEVEN SEASONS — SAME ERROR AS D186, AND
  IT WAS COSTING US. RE-CUT TO 2019-26. MARKET-OFFSET PROMOTED TO PRIMARY.**
  Owner: "why are we doing 14 season? we only get proper injury report data from
  2019 onwards, should only be those seasons, otherwise we are killing
  ourselves." Correct on both counts.

  D186 established the MODEL frame as 2019-20+ because the injury report begins
  2018-12-17 and the availability leg — half the production margin — is empty
  before it. **The TRADING frame was never re-cut to match**, so every published
  trading figure pooled 7 fully-covered seasons with 7 that measure a crippled
  variant. Re-cut, at k=9 raw:

  | frame | market-blind | **market-offset** |
  |---|---|---|
  | ALL 14 (7 with no feed) | +4.33% / +77.0u | +7.57% / +121.8u |
  | pre-2019 (no feed at all) | +3.77% / +28.0u | +5.67% / +40.8u |
  | **2019-26 (fully covered)** | **+4.74% / +49.0u** | **+9.11% / +80.9u** |
  | 2023-26 (measured panel) | +10.21% / +54.3u | +16.62% / +76.4u |

  **The dilution was real: the offset arm reads +9.11% on the honest frame
  against +7.57% on the 14-season blend.** Including seasons where the model
  cannot run as designed was costing ~1.5 ROI points, in addition to being wrong.

  **PRIMARY ARM CHANGED TO MARKET-OFFSET, per the owner.** On 2019-26 (K=7,
  n=888): pooled **+9.11%**, cum **+80.9u**, season-clustered mean +6.67%, 95% CI
  **[-2.75, +16.08]** — still CONTAINS ZERO — positive **5/7** seasons. Sim
  report: 565 sessions, net **$809,335**, $1,432/day, **Sharpe 1.1**, win days
  53%, max drawdown **-$240,301**, edge **911 bps**. The blind arm on the same
  frame: +4.74%, $490,095, Sharpe 0.6, 4/7, edge 474 bps.

  **HONEST DISTRIBUTION WITHIN THE FRAME:** the offset arm's first half
  (2019-22) is **-$30,637 at Sharpe -0.1**; its second half (2022-26) is
  **+$839,972 at Sharpe 2.0**. The result is concentrated in the recent block,
  which is also the block the architecture was developed on. Not evenly earned.

  **DOES THE BLIND MODEL STILL LOSE TO THE CLOSE? YES, DECISIVELY.** Honest
  inputs, identical games: blind 0.59276, opener 0.59228, close 0.57870. We are
  **+0.01406 worse than the close — 12.3% of its skill-above-coinflip** — and
  **+0.00048 worse than the opener (0.5%)**. The offset construction closes
  **26.7%** of the open-to-close gap; the blind model closes **-3.5%** of it, i.e.
  moves the wrong way.

  **DO WE STILL MENTION THE MARKET-BLIND MODEL? YES, PROMINENTLY — IT IS NOT A
  DISCARDED ATTEMPT, IT IS THE OFFSET MODEL'S DOMINANT INPUT** (edge coefficient
  ~0.35, stable 0.33-0.37 across every fold). Removing it from the story would
  make the offset construction look like it appeared from nowhere, and would bury
  what is arguably the register's most useful negative result: a carefully built,
  market-blind forecasting model does not beat the opening line, and the value it
  does carry is only extractable when spent at a third of face value against a
  market anchor.

  [SCOPE: `scripts/d181_report_tables.py` gains `RPT_FROM` (default 2019-20);
  both SIM_REPORTs and the README re-cut to 2019-26 with the offset arm primary;
  NO PRODUCTION MODEL DEFAULT CHANGED — promoting the offset construction to the
  shipped default remains a re-certification]

- D208 **SHOULD THE SUMMARY REPORT 3 OR 4 SEASONS? NEITHER. THE Sharpe-2.0 THAT
  PROMPTED THE QUESTION IS WHAT WINDOW SELECTION BUYS FROM NOTHING.** Owner:
  "should we give summary trades on last 3 or 4 seasons? if sharpe 2, lets make
  it 4." The Sharpe 2.0 came from a first-half/second-half split *I had just
  shown him*, so choosing the window on it is endpoint-driven selection — the
  exact procedure D182 and D187 rejected.

  **EVERY CONTIGUOUS WINDOW INSIDE THE HONEST FRAME** (offset arm, k=9):

  | length | best window | best ROI | mean of ALL windows of that length | **selection buys** |
  |---|---|---|---|---|
  | 3 | 2022-23..2024-25 | +16.95% | +9.03% | **+7.92 pts** |
  | 4 | 2022-23..2025-26 | +14.92% | +9.33% | **+5.59 pts** |
  | 5 | 2021-22..2025-26 | +13.47% | +9.94% | +3.53 pts |
  | 6 | 2020-21..2025-26 | +11.08% | +10.34% | +0.74 pts |
  | **7 (the frame)** | 2019-20..2025-26 | **+9.11%** | +9.11% | **0.00** |

  **THE AVERAGE WINDOW IS ~+9% AT EVERY LENGTH.** +9.03 / +9.33 / +9.94 / +10.34
  / +9.11. There is no recency signal here at all — shortening the window does
  not find a better regime, it finds a better *draw*. The entire apparent gain
  from "report the last 3 or 4" is the +5.6 to +7.9 points that picking the best
  window manufactures.

  **AND THE ONE 'SIGNIFICANT' WINDOW IS SELECTED.** 2021-22..2025-26 shows
  CI [+1.49, +21.05], excluding zero — and it is the best of only three
  five-season windows. That is not a finding, it is the maximum of three draws.

  **DECISION: the summary stays on the full 2019-26 frame.** It is the only
  window whose selection cost is zero by construction, and its +9.11% is
  indistinguishable from the average of every sub-window at every length — which
  is the strongest evidence available that it is the unbiased estimate.

  The recent block is still reported, but as a LABELLED SUB-ANALYSIS with its
  selection cost stated inline, never as the headline. Same treatment D182 gave
  the month filter.

  **HALL OF SHAME:** I produced the first-half/second-half split as honest
  disclosure of uneven earning, and it immediately became a candidate headline.
  **Disclosing a favourable subset creates the temptation to promote it.** When
  reporting a split, report what selecting on it would cost in the same breath.

  [SCOPE: analysis only; no code changed; no default changed; summary frame
  UNCHANGED at 2019-26]

- D210 **REVIEW PDF REDONE IN THE ORIGINAL LAYOUT.** Owner: "i strongly dislike
  the new format, keep to the old one as much as possible... can have all the
  extraneous details in readme but not here." The D209 rebuild had restructured
  the document into seven numbered sections with the leakage narrative, the
  window analysis and the frame justification inline. That is a different
  document, not a refresh of the owner's.

  **RESTORED FROM THE ORIGINAL:** unnumbered bold headings (Strategy / Headline
  results / The model); a dense opening Strategy paragraph rather than a
  bulleted breakdown; a **per-season headline table carrying books/game at open,
  bets, P&L in units, ROI and Sharpe** with pooled and post-haircut rows; a
  single-panel equity chart on the **sequential bet index** with dashed season
  dividers and a break-even rule (`scripts/d210_review_equity.py`, replacing the
  two-panel date-axis figure); an italic caption under it; one paragraph
  carrying the interval, MDE80 and concentration; and the component table.

  **MOVED OUT to the README and `docs/`:** the frame justification beyond one
  sentence, the leakage narrative, the window-selection table, the gate detail,
  the portfolio analysis. The PDF is 2 pages, down from 3, and 111 KB from 326.

  Numbers are current throughout: 888 bets, 7 seasons, pooled **+9.11%**
  (+80.93u), clustered 95% CI **[-2.75, +16.08]**, MDE80 12.7pp, 2024-25 supplying
  62% of P&L, trough -23.1u at bet 161.

  **PROCESS NOTE:** the PDF generator is a RENDERER over `docs/REVIEW.md`, so the
  document cannot drift from the repository. Three rendering defects were caught
  only by looking at the output rather than trusting the build: a 14-season equity
  figure under 7-season text, multi-line bullets breaking out of their bullets,
  and the two-panel chart itself. **A document generator's output must be read,
  not just built.**

  [SCOPE: `docs/REVIEW.md` restructured to the original shape;
  `scripts/d210_review_equity.py` NEW; `charts/review_equity.png` NEW;
  `nba_model_and_strategy_review.pdf` rebuilt; no numbers changed, no model
  default changed]

- D211 **THE FOUR-WAY LOG-LOSS COMPARISON — OPEN, CLOSE, BLIND AND OFFSET — AND
  WHY IT HAD TO BE PLOTTED AS A DIFFERENCE.** Owner asked to keep the log-loss
  graph and put all four series on it. The first build plotted the four LEVELS
  and was unreadable: the series differ by **~0.01 nats** while any one of their
  rolling-100 paths swings **~0.30**, so the signal is 3% of the noise and all
  four lines overlap into a single tangle. **Replotted as a difference from the
  OPENING LINE**, which is now the zero axis — the shared game-to-game difficulty
  cancels and only the quantity of interest remains.

  Pooled, 2019-26, each source with its own walk-forward logistic scale:

  | source | log loss | vs the opener |
  |---|---|---|
  | opening line | 0.60839 | 0 (the baseline) |
  | closing line | 0.59799 | **-0.01040** |
  | offset construction | 0.60589 | **-0.00250** |
  | market-blind model | 0.61217 | **+0.00379** |

  **The market-blind model is the ONLY series above the opener.** Offset recovers
  **24.1%** of the open-to-close gap; the blind model recovers **-36.4%**, i.e.
  moves the wrong way.

  Palette validated rather than eyeballed (`dataviz/validate_palette.js`): the
  first candidate FAILED the chroma floor (navy read as grey); the shipped set
  passes all six checks with worst adjacent CVD dE 22.9, and every series is
  additionally direct-labelled so identity never depends on colour.

- D212 **A COMPACT FORM OF THE SAME CHART, BECAUSE THE FULL ONE WAS DECORATIVE AT
  REPORT WIDTH.** The 7-panel D211 figure is 19 inches wide and correct at full
  size, but reduced 2.75x into the report's 6.9-inch column its 7pt annotations
  render near 2.5pt — present but unreadable, which is worse than absent.
  `charts/logloss_compact_2019_26.png` carries the identical measurement as a
  per-season grouped bar chart against the opener at zero. **The full-width
  version stays for the repository; the compact one goes in the report.** A
  figure that cannot be read at the size it is published at is not a figure.

- D213 **THE REPORT RESTORED TO THREE PAGES; BOLD DENSITY MATCHED TO THE OWNER'S
  SAMPLE; HEADLINE CUT TO THE RECENT THREE SEASONS WITH THE FULL FRAME KEPT
  BESIDE IT.** Owner: "i liked all the info in the 3 page report - what was taken
  away?" and asked whether the heavy bolding belonged.

  **RESTORED** (D210 had cut these to reach 2 pages): the bet-time leakage
  section in full — the 81.9/18.1/19.3/44.3% exposure, the 12.87% -> 17.17% cost,
  the gated fix and its 52% recovery, and the fact that the correction flipped the
  blind arm negative (+0.075 -> -0.104) while the offset arm held (+0.313 ->
  +0.267); the window-selection table; "what would change the assessment"; and the
  fuller caveats.

  **BOLD DENSITY.** Checked against the owner's original: it bolds the
  `Strategy:`/`Simulation:` labels, the pooled table row, and two or three figures
  in the notes — perhaps six inline bolds in the whole document. The D209/D210
  drafts bolded nearly every number, which flattens emphasis to nothing. Cut to
  the sample's density: labels, the pooled row, and three claims that carry the
  argument ("Both intervals contain zero", "the late information was worth 33%",
  "the only series above the opener").

  **HEADLINE = 2023-24..2025-26 per owner**, pooled +16.62% on 460 bets, CI
  [-9.29, +37.83], with the post-haircut row (+14.36%) and **the full seven-season
  frame (+9.11%, 888 bets, CI [-2.75, +16.08]) kept as the last row of the same
  table.** The caption states that the seven-season row is the honest denominator
  and the three-season block is its recent end, not a separate result — and the
  body still carries D208's finding that short windows buy +5.6 to +7.9 points of
  pure selection. That is how a recency headline can be shown without it becoming
  a claim.

  [SCOPE: `scripts/d211_logloss_4way.py`, `scripts/d212_logloss_compact.py` NEW;
  2 charts NEW; `docs/REVIEW.md` restored and re-emphasised; PDF rebuilt to 3
  pages; no numbers changed, no model default changed]

- D214/D215 **LOG-LOSS CHART BACK TO A LINE FORM; EQUITY CUT TO THE HEADLINE
  WINDOW; OUTLIER HAIRCUT MOVED OUT OF THE TABLE.** Owner: "change bar chart to
  graph, and headline results graph with pnl should only be last 3 or 4 years,
  your call but i think 3" and "get rid of outlier haircut i think and maybe
  mention in caveats."

  **LINE, NOT BARS, AND NOT SEVEN PANELS.** D211's 7-panel line chart was right
  at full width and illegible in the report column; D212 solved that with bars,
  which the owner did not want. `scripts/d214_logloss_line.py` is the form that
  satisfies both: ONE continuous panel across all seven scored seasons, dashed
  season dividers, still plotted as a difference from the opening line because
  the four levels differ by ~0.01 nats against ~0.30 of rolling swing. Legible at
  9.6 inches and at 6.9.

  **EQUITY CUT TO 2023-26 (460 bets, +76.44u)** so the figure and the headline
  table describe the same block — they disagreed before. The annotation changed
  from raw trough to **max drawdown (-7.75u at bet 372)**: on a block that rises
  from bet 1 the raw minimum is -0.09u at bet 2 and says nothing. The caption
  still carries the full-frame path (+80.93u on 888 bets, -23.1u drawdown) so the
  recent block is never mistaken for the whole record. **The log-loss figure
  deliberately keeps all seven seasons** — forecast accuracy has no reason to be
  cut to the recent block, and more seasons sharpen the comparison.

  **OUTLIER HAIRCUT OUT OF THE HEADLINE TABLE, INTO THE CAVEATS**, and stated
  more honestly there than the row did. The row implied the haircut was the
  execution correction; it is a partial one. What the caveat now says: always
  transacting at the best of nine books means always transacting at whichever book
  is furthest offside, and those are exactly the numbers unavailable in size,
  moved before the bet lands, or withdrawn from an account that keeps taking them.
  The outlier charge costs ~2.3 ROI points and **does not capture staking limits,
  reduced maximums on the best number, or account restriction — which in practice
  arrive faster than any price effect.**

  **OWNER ASKED WHETHER A PROFESSIONAL WOULD RECOGNISE THIS.** Yes, instantly,
  though not under this name — "outlier realism haircut" is our term. A pro would
  call it not being able to get down at the outlier, or simply getting limited,
  and would likely say the price-based framing understates it, because
  restriction arrives before the number moves. The caveat is written to be
  recognisable in those terms rather than ours.

  **GITHUB HEADER** added to the running page header
  (`github.com/sean-qin-usa/nba-prediction-model`, left) with the frame and the
  simulated disclaimer right. First attempt overlapped the two strings; the repo
  reference now replaces the redundant title, which duplicated the H1 below it.

  [SCOPE: `scripts/d214_logloss_line.py` NEW; `scripts/d210_review_equity.py`
  window 2023-24+ and drawdown annotation; `scripts/d209_make_review_pdf.py`
  header; `docs/REVIEW.md` haircut row removed and caveat expanded; PDF rebuilt,
  3 pages; no numbers changed, no model default changed]

- D216 **THE REPORT REBUILT TO COPY THE OWNER'S ORIGINAL, AND CUT BACK TO A
  RESULTS DOCUMENT.** Owner supplied the original PDF as reference and asked for
  the format copied, the log-loss figure returned to plain levels, and the
  self-critical material removed.

  **STRUCTURE NOW MIRRORS THE ORIGINAL:** title, then the github subtitle line
  ("Code, data pipeline and the full research register: ...") which is what the
  owner meant by the github header — the D209 build had put it in a running page
  header instead, which the original does not have. Section headings with a rule
  beneath. Strategy / Headline results / The model / Considerations, with
  Reporting frames, Execution, Methodology, Caveats and Next steps as
  subsections. Page number only in the footer. Typography tightened (body 7.5pt,
  tables 6.8pt) to reach the original's 3-page density.

  **LOG LOSS BACK TO LEVELS (`scripts/d216_logloss_season.py`).** My rolling-path
  versions had forced a difference-from-opener workaround because 8,000 points of
  a ~0.30 swing hide a ~0.01 gap. **The original's own answer was better: plot
  per-season means.** At 7 points per series the levels are legible directly and
  no baseline trick is needed. Four series separated by line STYLE as well as
  colour (solid / dashed / dash-dot / dotted) plus distinct markers, so the figure
  survives greyscale and colour-vision deficiency. Thin lines (1.15pt) per the
  owner's request.

  | season | open | close | offset | blind |
  |---|---|---|---|---|
  | pooled (n=8,286) | 0.6084 | **0.5980** | 0.6059 | 0.6122 |

  **CUT FROM THE REPORT, MOVED TO THE README** (owner: "i dont like how we have so
  much area on this report talking about how our model isnt sustainable... can be
  in readme or elsewhere, but not here"): the whole Research record section,
  manufacturing capacity, the permutation-null result, the window-selection
  analysis, the era-specificity ablation, CLV-is-not-an-objective, no-market-
  impact, and the list of corrected errors. All of it now lives under a new
  README section, **Limits of the result — the material kept out of the summary**,
  rather than being deleted.

  **FOUR THINGS KEPT, AND THE REASON.** A results document that drops every limit
  becomes a claim rather than a result, so these stayed: **the interval contains
  zero** (without it +16.62% reads as an established edge); **simulated, no
  capital deployed**; **best-of-9 observed in one season and inferred in two**
  (the headline number depends on it); and **2026-27 is the first genuinely
  out-of-sample season**. Everything removed is presentational emphasis;
  everything kept is load-bearing for reading the numbers correctly.

  **ALSO REMOVED:** the leak-and-fix section (owner: results doc, belongs in the
  repo) — retained in `docs/DECISIONS.md` D199-D202 and summarised in the README.

  [SCOPE: `scripts/d216_logloss_season.py` NEW; `charts/logloss_season_4way.png`
  NEW; `scripts/d209_make_review_pdf.py` header/rules/typography;
  `docs/REVIEW.md` restructured; README gains the Limits section; PDF rebuilt to
  3 pages; no numbers changed]

- D217 **AN OUTSIDE REVIEW CAUGHT A REAL RENDERING DEFECT AND THREE STALE CLAIMS.
  EVERY CHECKABLE ASSERTION IT MADE WAS CORRECT.** Reviewer compared the
  published PDF against the owner's original and raised: unembedded fonts, README
  staleness, a duplicated heading, and report scripts referenced by the register
  but absent from the repository. **All four verified and all four true.**

  **FONT EMBEDDING — the serious one.** `pdffonts` reported **emb=no on all five
  faces** (Helvetica, Helvetica-Bold, Helvetica-Oblique, Courier, Symbol).
  ReportLab's base-14 fonts are legal PDF but carry no outlines, so the viewer
  substitutes; on systems without metric-compatible faces that renders as merged
  or wildly-spaced glyphs. **The text extracted cleanly the whole time, which is
  exactly why I had not caught it** — I verified extraction, not rendering.
  Fixed by registering DejaVu as TrueType. Three attempts were needed and the
  sequence is worth recording:
  1. registering the faces and setting every `fontName` -> still one Helvetica;
  2. `bulletFontName` and an explicit table `FONTNAME` -> still one Helvetica;
  3. **`rl_config.canvas_basefontname = "DejaVu"`** -> zero. ReportLab registers
     its base font before any document object exists, so `setFont()` in a page
     callback is always too late.
  `pdffonts` now reports **0 unembedded faces**, and that check is the acceptance
  criterion for this artifact from now on.

  **STALE CLAIMS, all corrected:** README said the register "runs to D207" while
  `DECISIONS.md` reached D216; README described the PDF as "retained for history
  and predates three corrections" when it is in fact regenerated from
  `docs/REVIEW.md` on every build; and a sed edit had left the literal duplicated
  heading `Why the model looks era-specific### Why the model looks era-specific`.

  **SCRIPTS NOW PUBLISHED.** The register cited `d209_make_review_pdf.py`,
  `d210_review_equity.py`, `d214_logloss_line.py` and others that existed only in
  the private tree. 17 report/analysis scripts copied across, so the register's
  references resolve and the figures are reproducible.

  **ADOPTED FROM THE REVIEW:** body type raised 7.5 -> 8.3pt with 11pt leading;
  the offset equation written out explicitly as
  `m_offset = m_open + f(m_blind − m_open, rest differential, |m_open|)` so a
  reader can see immediately what is learned versus handcrafted; and **a
  single-book execution row placed beside the headline** rather than buried in a
  caveat — best-of-nine is the largest practical weakness in the number, so the
  observed one-book figure (+10.63%, about two thirds of the reported ROI) now
  sits next to it.

  **NOT ADOPTED:** the suggestion to re-add the full research-record section. The
  owner had just removed it deliberately; it lives in the README under "Limits of
  the result". Splitting the difference by restoring a "compact evidence /
  limitation table" would reintroduce what was cut.

  **HALL OF SHAME:** I verified this PDF by extracting its text and by rendering
  it to PNG on a machine that happens to have Helvetica metrics. **Neither test
  can see an unembedded font.** Checking that a document *renders here* is not
  checking that it renders anywhere; for any published artifact, verify the
  embedded resources, not just the output.

  [SCOPE: `scripts/d209_make_review_pdf.py` font embedding via rl_config +
  TrueType registration, body size, table FONTNAME; `scripts/d210_review_equity.py`
  and `scripts/d216_logloss_season.py` flatter aspect for page balance;
  `docs/REVIEW.md` explicit equation + execution row; README three corrections;
  17 scripts published; no numbers changed]

- D218 **SECOND REVIEW PASS: FOUR CONTENT ERRORS FOUND, ALL FOUR REAL, ONE OF
  THEM A CLAIM I NEVER CHECKED.** The reviewer was reading D216; D217 had already
  fixed the fonts, the README staleness and the missing scripts. The new material
  was four content corrections, and every one held up.

  **(1) A FALSE EQUIVALENCE I ASSERTED WITHOUT CHECKING.** The report said the
  genuine content of our disagreement, "0.206 points, 8.4% of what we claim", was
  "close to the third of face value the offset ridge independently arrives at."
  **8.4% is not close to 33%.** Worse, the two are measured on different
  denominators AND different frames: 8.4% is the real share of a 2.455-point RAW
  disagreement derived from cover rate on the 19-season contaminated ATS test;
  0.33-0.37 is a ridge coefficient on margin residuals over 2019-26 on honest
  inputs. A third denominator exists too — 0.206/0.751 = 27% of the break-even
  requirement. **I wrote a corroboration between two numbers that do not measure
  the same thing.** Now reported separately with that stated.

  **(2) 7.2 vs 7.53 LOOKED LIKE A CONTRADICTION.** The document gave the link as
  7.2 and then said the training residuals imply 7.53 without saying why the
  formula does not use 7.53. Clarified: 7.2 is the frozen production constant,
  7.53 is the plug-in estimate, re-deriving per fold is worth ~0.0002 nats and is
  not significant, and **every log-loss comparison in the document recalibrates
  each source walk-forward** so no forecast is advantaged by another's scale.

  **(3) "THE BEST OF THE 9 BOOKS HELD" OVERSTATED WHAT WE HAVE.** We hold a
  measured panel in 2023-24 only. Reworded to a modelled best-of-nine tier,
  observed in 2023-24 at 7.74 books/game and inferred elsewhere.

  **(4) "THE ONLY ONE OF THE FOUR ABOVE THE OPENING LINE" IS A POOLED CLAIM AND
  THE FIGURE BESIDE IT CONTRADICTED IT.** Checked per season:

  | season | above the opener |
  |---|---|
  | 2019-20, 2021-22, 2022-23, 2023-24, 2025-26 | blind |
  | **2020-21** | **blind AND offset** (offset 0.6266 vs open 0.6257) |
  | **2024-25** | **neither** (blind 0.5915 vs open 0.6040) |

  So the blind model is above the opener in **six** of seven, not all seven, and
  the offset construction is above it once. **A reader could have falsified the
  sentence from the chart printed directly beneath it.** Now stated as pooled,
  with the per-season exceptions named.

  **ALSO ADOPTED:** the player-level feature inventory condensed to one sentence
  with the detail left in the repository (owner had already asked for less of
  this), which also fixed the page-2 density the reviewer flagged.

  **TRIED AND REVERTED:** a forced page break before "The model". It produced 5
  pages with a short page 1; without it the condensing alone balances the
  document at 4. The layout complaint was real but the proposed fix was not the
  right one.

  **HALL OF SHAME:** the "close to a third" clause was rhetorical glue between two
  numbers I liked the look of together. **Nothing in the pipeline can catch a
  false equivalence between two correctly-computed figures** — no gate, no null,
  no test. It survived because it sounded like corroboration, and it took an
  outside reader to ask what the denominators were.

  [SCOPE: `docs/REVIEW.md` four corrections + condensed feature paragraph;
  `scripts/d209_make_review_pdf.py` PageBreak support added (currently unused);
  PDF rebuilt, 4 pages, 0 unembedded fonts; no computed number changed]

- D219 **CLICKABLE REPO LINK AND THE RULE UNDER THE TITLE.** Owner: "make sure
  link is clickable. i also want the line under the title again, looks better."

  **LINK.** The github reference was plain text. The markdown renderer now emits
  a real PDF `<link>` annotation for markdown links with an http/mailto target,
  and additionally auto-links bare `github.com/...` strings (guarded so it never
  double-wraps an existing link). Verified structurally rather than visually:
  `/URI` actions in the file = **2**, resolving to
  `https://github.com/sean-qin-usa/nba-prediction-model`. A blue underline that
  is not an annotation is not a link, so the check is on the annotation.

  **TITLE RULE.** Restored the navy rule directly beneath the H1, matching the
  owner's original. Section headings keep their lighter rule; the title's is
  heavier (1.1pt navy vs 0.7pt pale) so the hierarchy reads.

  [SCOPE: `scripts/d209_make_review_pdf.py` inline-link handling and H1 rule;
  PDF rebuilt, 4 pages, 0 unembedded fonts, 2 URI annotations; no content or
  numbers changed]

- D220 **BACK TO THE SMALLER TYPE AND 3 PAGES.** Owner: "font is quite big - i
  actually like it before, can we go back? 3 pages is good."

  Reverted body 8.3 -> 7.5pt on 9.9 leading, tables 7.2 -> 6.8pt, and the code and
  quote styles with them. **The font EMBEDDING is untouched and unrelated** — that
  was the actual defect the outside review found (`pdffonts` emb=no on every
  face), and it is independent of point size. Post-build check still reports
  **0 unembedded faces** and **2 URI annotations**.

  Reverting the size alone left the document at 4 pages, because content had been
  added since the smaller type was last in use (the execution ladder, the explicit
  offset equation, the four D218 corrections). Spacing was tightened first
  (heading space-before 12 -> 7, list spacing, figure and table spacers, margins),
  then figures narrowed 7.1 -> 6.55 inches. That still left **three lines** on a
  fourth page, so three low-value clauses were cut rather than shrink the artwork
  further or squeeze the leading again: the exchange-row restatement, a redundant
  "free, and the highest-value operational change" tag, and one spelled-out
  "closing-line value" where CLV had already been defined.

  **Note on the tension with the outside review**, which argued for 8-8.5pt body:
  it was right that the type is small for something meant to be read, and the
  owner's preference is for the denser page. The review's real finding — the
  unembedded fonts — is fixed either way, and that is what was making the document
  look broken rather than the point size.

  [SCOPE: `scripts/d209_make_review_pdf.py` sizes and spacing; `docs/REVIEW.md`
  three clauses trimmed; PDF 3 pages, 0 unembedded fonts, 2 URI annotations; no
  numbers or findings changed]

- D221 **EDITORIAL PASS: THE REPORT READ LIKE THE REGISTER IT WAS EXTRACTED
  FROM.** Owner, on a language critique: "many extraneous comments that a reader
  can infer... many short 5 word sentences that are abrupt and do not seem
  professional" and "i like mathy language so im not sure about following these
  to a t."

  **THE DIAGNOSIS WAS RIGHT AND MORE PRECISE THAN "SENTENCES ARE TOO SHORT."**
  Mean sentence length was already ~19 words. The defect was RHYTHM: runs of 3-7
  word fragments (`Simulated.` / `Settlement-only fills.` / `Props before
  sides.`) alternating with 40-46 word multi-clause sentences. That is the shape
  of an audit log, which is literally what the document was extracted from.

  **WHAT WAS CHANGED.** Bold-fragment-plus-compressed-explanation converted to
  full sentences throughout Methodology, Caveats and Next steps; the two longest
  sentences (46 and 40 words) split; and meta-commentary a reader can infer
  removed — "Series are distinguished by line style as well as colour" describes
  the figure the reader is looking at.

  **ONE ITEM WAS A CORRECTNESS FIX, NOT TONE.** The report said "What is learned
  is only the correction; everything else is handcrafted and declared." **That is
  false.** The team ratings, availability probabilities, schedule coefficients and
  the offset coefficients are all *estimated*; what is fixed ex ante is the
  architecture and its transformations. Reworded accordingly. The sentence was
  written to answer the handcrafted-architecture criticism and in doing so
  undersold what the model actually fits.

  **WHERE THE CRITIQUE WAS NOT FOLLOWED.** Several proposed terminology swaps
  trade precision for blandness and were declined: "honest denominator" became
  "primary estimate" (accepted — clearer), but "Not next" -> "Current priority"
  inverts the meaning and was rejected, and the mathematical register the owner
  prefers was kept throughout — interval notation `[0.33, 0.37]`, `m_blind −
  m_open`, seasons `1…k`.

  **TWO LAYOUT DUPLICATIONS REMOVED, which also paid for the longer prose.** A
  small execution table on page 1 restated the full ladder on page 3; it is now
  one sentence carrying the single-book figure with a pointer. The seven-row
  seasonal log-loss table restated the chart directly above it; collapsed to the
  pooled row. Document back to **3 pages**, and the page-2/3 boundary no longer
  splits the central market-comparison conclusion mid-sentence.

  [SCOPE: `docs/REVIEW.md` editorial pass and two table removals; PDF rebuilt,
  3 pages, 0 unembedded fonts, 2 URI annotations; no computed number changed]

- D222 **THE REPORT SPENT MORE SPACE ON THE MARKET-BLIND MODEL'S FAILURE THAN ON
  THE SHIPPED CONSTRUCTION'S RESULT.** Owner: "seems like we focus heavily on how
  bad the market blind model is instead of how good our offset is, are they?" and
  then, on my first fix, "do not add extra to say that offset is good, we have
  enough as it — just need to not rip on the market blind model so much."

  **THE OBSERVATION WAS CORRECT.** Of the market-comparison section, the majority
  of the prose described the market-blind model's shortfall: it was the only one
  above the opener, it recovered −36.4% of the open-to-close gap, its normalized
  gap was 13.59%, and a further paragraph gave its cover rate (50.65%), its raw
  disagreement (2.455 pts) and the 8.4% genuine-content derivation. The offset
  construction — the thing actually being reported — got one clause.

  **MY FIRST ATTEMPT OVERCORRECTED** by adding offset-positive material (six of
  seven seasons better than the opener, 0.0024 nats pooled, six of seven better
  than the blind model). The owner rejected that: the fix was to stop dwelling on
  the negative, not to start advocating. **Reverted the additions and cut the
  criticism instead** — the market-blind model is now named once, as the reason
  the architecture is a correction to the line rather than a forecast compared
  against it, and nothing more.

  **"TWO FURTHER DIAGNOSTICS" REMOVED ENTIRELY.** Cover rate 50.65%, the
  2.455-point disagreement and the 8.4%-genuine-content arithmetic were all
  market-blind measurements on a nineteen-season contaminated frame — a different
  model on a different frame from everything else in the document, and
  contributing nothing to how good the shipped construction is. The one sentence
  worth keeping (CLV positive and significant across 19 seasons) was retained
  because it bears on the live plan.

  **ALSO CUT:** the walk-forward-scale clause, stated in both the body and the
  figure caption. A reader does not need the calibration convention twice, or
  arguably at all, in a results document.

  **LINK VERIFIED PROPERLY, NOT BY STRING COUNT.** Owner: "am not convinced link
  is clickable, check that it is." Counting `/URI` occurrences proves nothing —
  a clickable link requires an object with `/Subtype /Link`, a `/Rect`, an `/A`
  action of `/S /URI`, AND attachment to a page via `/Annots`. Checked all four:
  **object 2 carries `/Subtype /Link`, `Rect [237.4 718.3 436.2 727.3]`,
  `/URI (https://github.com/sean-qin-usa/nba-prediction-model)`, and page 1
  declares `/Annots [ 2 0 R ]`.** It is a real annotation. **My earlier
  verification — counting `/URI` strings — would have passed on a document with
  the URI present but no annotation and no page attachment**, which is exactly
  the failure mode the owner suspected.

  [SCOPE: `docs/REVIEW.md` market-comparison section rebalanced and diagnostics
  paragraph removed; PDF 3 pages, 0 unembedded fonts, 1 verified link annotation;
  no computed number changed]

- D223 **FINAL EDITORIAL CYCLE ON THE REPORT.** Reviewer's remaining items, of
  which the owner asked for all but the Execution-to-page-3 move.

  **PAGE FLOW.** Forced a break before "The model", which produced the intended
  split — page 1 strategy/headline/equity, page 2 the model and accuracy, page 3
  considerations onward — and stayed at 3 pages this time. **The break alone
  relocated the orphan rather than removing it:** page 2 then ended with the
  "Considerations" heading stranded above its own content. Fixed generally with
  `keepWithNext=1` on both heading styles AND on the rule that follows a heading,
  so no heading can strand at a page bottom regardless of where content falls.
  All three pages now end mid-paragraph rather than on a heading.

  **THREE WORDING CORRECTIONS.**
  - "honest denominator" -> "the full seven-season frame ... serves as the primary
    reference". The original was self-conscious and grammatically awkward.
  - "the era of the measured multi-book panel" was **too broad and I had already
    corrected this class of overstatement once in D218**. Only 2023-24 carries a
    genuine panel. Now: "the recent execution-study window. Multi-book prices are
    observed directly in 2023-24 and inferred in the two seasons after it."
  - "worth +0.0012 of log loss" was **ambiguous by sign convention**, since higher
    log loss is worse. Checked the register's convention (positive denotes
    improvement, e.g. PAID_RETROSPECTIVE's "+0.0005-0.0015") and restated
    explicitly: "improves log loss by 0.0012 nats combined, an effect whose
    interval contains zero."

  **TWO LONG SENTENCES SPLIT** (the 45-word architecture sentence and the
  best-of-N caveat).

  **SEVEN-SEASON ROW GIVEN EQUAL VISUAL WEIGHT.** Previously the 3-season row was
  bold and the 7-season row plain, which invited exactly the cherry-picking read
  the document argues against. Both rows are now bold and labelled by role —
  "recent execution headline" and "primary full-frame reference".

  **NOT DONE, per the owner:** moving Execution wholly onto page 3.

  [SCOPE: `docs/REVIEW.md` wording and table emphasis;
  `scripts/d209_make_review_pdf.py` keepWithNext on headings and heading rules;
  PDF 3 pages, 0 unembedded fonts, 1 verified link annotation on page 1; no
  computed number changed]

- D224 **GATE: THE OFFSET CONSTRUCTION, ON PREDICTION POWER. SHIP.** Spec
  SHA-256 `2ecc47402ee3f021b59ab6c0b0f94b88`, MDE80 stated before the endpoint.
  Incumbent = the shipped market-blind margin carrying D202 soft availability;
  challenger = `m_open + f(m_blind − m_open, rest_diff, |m_open|)`, `f` refitted
  walk-forward. Each source keeps its own walk-forward logistic scale.

  | | |
  |---|---|
  | pooled log loss | blind 0.61222 -> **offset 0.60595** |
  | season-clustered mean delta | **-0.006378 nats** |
  | 95% CI (6 dof) | **[-0.010621, -0.002134] — EXCLUDES ZERO** |
  | t | **-3.68** |
  | MDE80 (stated first) | 0.005703 |
  | better in | **6/7 seasons** (only 2024-25 worse) |
  | calibration veto | PASS (mean p 0.5577 vs base 0.5521) |

- D225/D226 **BOTH PROMOTED TO PRODUCTION.** Owner: "promote both to production
  if they have higher prediction power." Both cleared a pre-registered
  season-clustered gate against the incumbent — soft availability at D202
  (t=-3.39, 5/5 seasons), the offset construction at D224 (t=-3.68, 6/7) — so
  both are promoted rather than left opt-in.

  **1. SOFT AVAILABILITY IS NOW THE DEFAULT.** `SOFT_AVAIL` flips to on in
  `prod_by_season.py`; `SOFT_AVAIL=0` restores the hard out-set. Verified: the
  new default reproduces the gated soft run to **8.99e-15**. The certified figure
  moves from the leak-contaminated 12.87% to **14.92%** normalized gap — a
  *worse-looking* number that is the honest one, since the incumbent 12.87% was
  built on information published after the price it transacts at (D199).

  **2. THE OFFSET LAYER IS NOW IN THE LIVE PATH.**
  `nbapred/market/offset.py` NEW, with coefficients frozen by
  `scripts/d225_fit_offset_prod.py` into `data/offset_coefs.json`
  (edge +0.3564, rest_diff +0.0417, abs_open -0.0114; n=22,742, fitted through
  2026-04-12). Wired into BOTH `bet_engine.py` model call sites.

  **THREE PROPERTIES ENFORCED BY CONSTRUCTION, not by convention:**
  - **Market-blindness survives.** `apply()` takes the blind margin as an
    ARGUMENT and cannot reach back into the model that produced it, so the blind
    model still never sees a price. That boundary is what makes D193/D224 mean
    anything.
  - **Degraded mode is the incumbent.** With no opening price, or with
    `OFFSET_LAYER=0`, `apply()` returns the blind margin unchanged. A dead odds
    feed costs the correction, never the prediction — which matters because D196
    established the odds logger has already died once.
  - **The blind model is not retired.** It is the offset layer's dominant input
    and its fallback.

  **A REAL BEHAVIOURAL CONSEQUENCE, caught by a test rather than reasoned about.**
  `test_emit_never_books_a_non_regular_season_game` failed after wiring — not on
  its game-type assertion but on `assert n > 0, "the regular-season game must
  still book"`. **The offset shrinks disagreement with the market by ~65%, so a
  margin that used to clear the betting threshold no longer does.** That is the
  layer working as designed, and it means live bet VOLUME will fall. The stub was
  recalibrated (blind margin 6.85 -> 10.67, the value leaving post-offset p at
  0.72 against that test's quotes) so the rule assertions still test what they
  were written to test. Two smaller stub gaps surfaced the same way: the fake
  model had no `margin()` method, and `bet_engine.py` imported neither `math` nor
  `sigmoid`.

  **VERIFICATION:** 153/153 tests pass; live dry run emits with the offset
  applied (edges now +0.047/+0.048 where the raw model would have shown more);
  `predict_today.py` exits clean; canary 5 pass / 2 warn / 0 fail.

  [SCOPE: `nbapred/market/offset.py` NEW; `scripts/d225_fit_offset_prod.py` NEW;
  `data/offset_coefs.json` NEW; `scripts/prod_by_season.py` SOFT_AVAIL default
  ON; `scripts/bet_engine.py` offset wired at both model call sites plus math /
  sigmoid imports; `tests/test_bet_engine.py` stub given `margin()` and
  recalibrated; PRODUCTION DEFAULTS CHANGED — both gated]

- D227 **README REWRITTEN AROUND THE SHIPPED SYSTEM.** Owner: "redo readme
  content (especially charts) to reflect what we have been working on in the
  summary — offset values, new timeframe/prediction power, etc."

  **THE PROBLEM WAS ACCRETION.** The README had grown to **715 lines** across ~28
  headings, most of them added as findings landed and none removed when they were
  superseded. It described a market-blind win-probability model as the product —
  which stopped being true at D226 — and carried whole sections ("We beat the
  opening line...", "The fix that follows: anchor on the opener", "Traded through
  production machinery") that were the *narrative of arriving* at the offset
  construction rather than a description of it. **It referenced 3 of 46 charts,
  and none of the ones the current argument rests on.**

  **REWRITTEN TO 251 LINES.** Leads with what the system now is and why the
  architecture takes that shape. Prediction power gets its own section with the
  four-way comparison and both promotion gates side by side. The betting record
  compares the two arms on one table rather than across four sections. The
  historical narrative — how the leak was found, how the frames were corrected —
  is compressed into "What we got wrong, and caught" rather than being re-argued.

  **CHARTS RE-POINTED.** Now embeds `data_coverage`, `logloss_season_4way`,
  `frame_model_2019_26` and `review_equity`, and the charts table names the six
  that carry the argument. The stale references to `logloss_continuous_2019_26`
  (superseded by the per-season levels chart) are gone.

  **VERIFIED RATHER THAN ASSUMED:** every referenced chart file exists (6/6),
  every referenced doc exists (6/6), and a sweep for figures superseded by the
  offset promotion (+8.72%, +4.63%, a bare 12.87%) returns only two hits, both
  correct in context — the leak narrative, where 12.87% -> 17.17% IS the finding,
  and a register citation.

  [SCOPE: README.md rewritten, 715 -> 251 lines; no numbers changed, no model
  default changed; all links and chart paths checked to resolve]

- D228 **THE ODDS LOGGER WOULD NOT HAVE PRODUCED THE DATASET IT EXISTS FOR.**
  Owner asked for a roadmap audit; the logger is the only irreversible item on
  it, so it was audited against the four-snapshot spec before any modelling.

  **FINDING 1 — PROPS WERE UNREACHABLE, NOT MERELY UNUSED.** `.env` sets
  `ODDS_MONTHLY_BUDGET=500`; `odds_logger.py` gated prop polling on
  `MONTHLY_BUDGET == 0`, so `poll_props()` was **dead code in the shipped
  configuration**. Documented as deliberate ("they'd eat it in one evening") and
  correct on a 500-credit tier — but the consequence is that 2026-27 would have
  yielded ZERO prop prices, leaving "market-offset props" unexplored in 2027 for
  exactly the reason it is unexplored today. **The gap was a live config
  decision reproducing itself, not a historical accident.**

  **FINDING 2 — BOTH CREDIT GUARDS WERE INERT.** `ODDS_CREDIT_FLOOR=50` gated
  only the (already dead) props branch, and the main-polling brake sat in an
  `elif` AFTER `if MONTHLY_BUDGET:` — unreachable whenever a budget is set.

  **FINDING 3 — THE BURST BYPASSED THE PACER, AND THE ARITHMETIC DID NOT CLOSE.**
  `sleep_min = min(paced, CLOSE_WINDOW_MIN/3)` applied whenever any tip was
  within 90 min. At 3 credits/poll (h2h,spreads,totals x us) and ~9 polls on a
  staggered 10-game night that is 27 credits/night, **675 across 25 game nights
  against a 500 budget — a 35% overdraw.** On exhaustion `_get` raises, the
  handler retries every 2 min forever, and the unit stays `active` while
  capturing nothing. The paced cadence was confirmed by observation, not only
  arithmetic: the log ran 08:32 -> 12:04 -> 15:35 -> 19:05 -> 22:34, ~3.5h apart,
  matching `(498/24)/3` polls/day exactly.

  **THE SPEC ITSELF WAS ANCHORED TO THE WRONG CLOCK.** The proposed "pre-report /
  post-report" pair is clock-anchored, but `scripts/wlm_chart.py` measured the
  within-day move: **76% complete at T-4h, 80% at T-2h, 91% at T-1h, and ~75%
  ALREADY GONE at the 5PM ET report for early AND late games**, with the only
  burst at T-2h -> tip. Tips are staggered 19:00-22:30 ET, so a 5PM-anchored pair
  straddles a different completion fraction for every game. **The ladder is
  therefore TIP-relative** (`nbapred/ingest/odds_sched.py`, pure and unit-tested):
  OPEN / T-4h / T-2h / T-1h / CLOSE(T-15m), generated per game and MERGED, since
  one `/odds` call serves the whole slate — ~40 marks collapse to ~14 calls.

  **THE LEVER WAS MARKET TIERING, NOT MONEY.** Credits are priced per market per
  region and the sides strategy transacts on SPREADS ONLY (nothing in `nbapred/`
  or `bet_engine.py` reads h2h or totals from the live feed). Core poll 3 -> **1
  credit**, extras ride along once a day. **A 3x cadence increase for no money**,
  which is what makes the full ladder affordable.

  **MEASURED ON A SIMULATED MONTH, 28 DENSE 10-GAME NIGHTS FROM 500 CREDITS:**
      old policy   27 cr/night   ran dry ~day 18, remaining nights lost
      **new        16-18 cr/night   483/500 spent, 28/28 nights captured, 17 left**
  Degradation is graceful and self-correcting (allowance is recomputed from the
  live `x-requests-remaining`, not the nominal budget): at 480 credits the full
  ladder plus extras, at 250 the extras and a rung drop, at 90 it falls back to
  **open + close only** and stops there. Trimming is priority-ordered and can
  never drop OPEN or CLOSE — the transaction price and the CLV reference.

  **TWO OF MY OWN BUGS WERE CAUGHT BY THE SIMULATION, BOTH THE CLASS BEING
  FIXED.** (a) `want_extra` was gated on `allowance >= 999`, silently dropping
  h2h/totals whenever any budget was set — the same silent-config-disable as
  finding 1. (b) The nightly prop cap was applied PER ITERATION, so each main
  poll sampled that many MORE events: 2/night became 10/night and **a 2.4x
  overdraw**. Neither was visible by reading; both appeared on the first
  end-to-end run. A third defect — polling on a CAPPED wake when no target was
  due — broke the invariant the budget argument rests on (polls == plan targets)
  and cost a credit per long quiet stretch.

  **PROPS ARE NOW RATIONED, NOT SWITCHED OFF:** one market (`player_points`, 1
  credit/event) under a nightly event cap against genuine leftover allowance, so
  they degrade first and can never starve the sides ladder. On the maximally
  dense simulated month that is 15 prop events across 8 of 28 nights — small, but
  the first prop prices this project would ever hold.

  **`scripts/logger_canary.py`** checks the OUTPUT, not the process, because
  `systemctl is-active` stays green through every failure that matters —
  exhausted credits, revoked key, wrong sport key, a season that started while
  the poller sat in its offseason idle. Current: 3 pass / 5 warn / 0 fail, the
  warns all "offseason, no events". Tests: `tests/test_odds_sched.py`, 14 cases,
  the central one being that a real night fits the real budget.

- D229 **THE PUBLIC MIRROR SHIPPED HARD AVAILABILITY WHILE THE README CLAIMED
  SOFT. THE GATES STAND; THE PUBLICATION DID NOT.** Raised by an external reader
  against `origin/main`, with a reproduction. **THE CLAIM WAS CORRECT.**

  **WHAT WAS ACTUALLY BROKEN.** `nba_model/` (private, where every gate ran) and
  `nba-prediction-model/` (public) are two SEPARATE DIRECTORIES, not a worktree
  pair — the public repo is a curated copy, so code changes never propagate
  unless copied by hand. `prod_by_season.py` had been pushed (it passes
  `{player_id: p_out}`) but `composition.py` had not (it still did `pid in out`).
  **That combination is worse than either policy**: it is not the pre-D202 hard
  rule, which received a set of ACTUAL outs, and it is not soft availability —
  it drops every player carrying ANY probability, at any probability.

      input            private (gates)   public (mirror)
      nobody out           144.0             144.0
      P(out)=0             144.0 OK           96.0  WRONG — dropped at 0%
      P(out)=0.5           120.0 OK           96.0  WRONG
      P(out)=1              96.0 OK           96.0  right by accident
      hard set {p}          96.0 OK           96.0  right by accident

  **THE GATES ARE UNAFFECTED, AND THIS WAS CHECKED RATHER THAN ASSERTED.** The
  soft implementation landed in the private tree at **2026-08-06 22:18**;
  `data/offset_coefs.json`, the D225 production fit that D224 gates, was written
  **2026-08-07 00:42** — after it. D202/D224, the log-loss table, ROI and Sharpe
  all stand. **The defect was in what a reader could reproduce, not in what was
  measured** — which is the less alarming failure but the more embarrassing one,
  since the repository is the evidence.

  **THE READER WAS ALSO RIGHT THAT THE IDENTITY CHECKS WERE NEVER COMMITTED.**
  D202 recorded that 0 / 0.5 / 1.0 had been verified; they had been, but
  interactively. `tests/test_soft_availability.py` now asserts them, 13 cases.
  **The test that catches this class of bug is the INTERIOR probability** — a
  membership check passes 0.0 and 1.0 by accident and can only fail at 0.5. Run
  against the pre-fix public code the file gives **10 failed / 3 passed**, and
  the 3 that pass are exactly the accidental ones.

  **THE DIVERGENCE WAS WIDER THAN THE FILE REPORTED**, which is the part worth
  remembering: `slate.py` was missing the D178 live-path game-type filter (the
  All-Star/preseason exclusion the owner explicitly asked for), `injury_pdf.py`
  the D178 modern-layout parser (+204 lines), `tanking.py`/`teams.py` the D171
  abbreviation fix, and 70 scripts including **`d224_gate_offset.py` — the gate
  behind a headline claim.** All synced; `nbapred/` and `tests/` now diff clean.
  **A SECOND PUBLICATION DEFECT, FOUND BY READING THE FULL TEST RUN RATHER THAN
  ITS TAIL — AND MY FIRST NUMBER FOR IT WAS WRONG.** I initially recorded "7
  DB-backed tests cannot run in a clone, so a cloner gets 27 pass / 7 error."
  That came from a command piped through `tail -8`, which discarded the failure
  list before it reached the log; the true full-suite result on the synced public
  tree was **35 failed / 141 passed / 4 skipped**. The extra failures were not
  the DB. `.gitignore` blanket-excludes `data/`, so the public tree shipped **2
  data items against the private tree's 527** — and `nbapred/model/travel.py`
  reads `data/arenas.csv` at import, giving `FileNotFoundError` rather than the
  clean skip the tests were written to take (`test_travel_neutral.py` guards on
  `nba.duckdb` being absent, which is the wrong guard for a missing CSV).
  **THE CLONE EXPERIENCE, BEFORE AND AFTER:**
      before   35 failed / 141 passed /  4 skipped   in 43 min
      **after    0 failed / 151 passed / 29 skipped   in 33 s**
  Four distinct causes, none of them a code defect:
  (1) **REFERENCE DATA THE PACKAGE NEEDS TO IMPORT AT ALL** — `arenas.csv` and
      `xwalk_overrides.csv`, 12 KB, read by `nbapred/model/travel.py`. Shipped.
  (2) **GOLDEN FIXTURES FOR REGISTERED CONSTRUCTIONS** — `rw_early_decomp_pergame`,
      `rw_early_signals`, `rw_week1_psroster`, ~190 KB, the corpora the October
      bridge assertions check a construction against. Shipped, so a reader can
      verify the claim rather than take it. The 262 MB `nba.duckdb` and every
      computed frame stay out.
  (3) **A 120-SECOND RETRY ON A FILE THAT WILL NEVER APPEAR.** `db.connect()`
      retries `duckdb.IOException` for 120 s (readers) because the recurring
      real-world fault is a lock another process holds — but a MISSING database
      is not contention, and each guarded test burned the full deadline before
      its `except Exception: pytest.skip` could fire. **That single wait was
      essentially the whole 43 minutes.** `connect(read_only=True)` now fails
      immediately when `DB_PATH` is absent; writers are exempt, since for them a
      missing file is the ordinary create-on-first-use path.
  (4) **PRECONDITIONS WRITTEN AS ASSERTIONS.** `tests/conftest.py` converts the
      absent-database sentinel into a SKIP — narrowly, matching only that error
      and only where the DB is genuinely missing, so it cannot mask a
      regression and is unreachable in the working tree. `test_basics.py`'s 2K
      parser test skipped rather than asserting on an uncommitted scrape
      archive. **Both were reporting missing data as broken code.**
  **THE LESSON IS THE SAME ONE AS THE MIRROR ITSELF: a check whose output is
  truncated is not a check.** The first number here came from a command piped
  through `tail -8`; the second came from reading the whole run.

  **THE PROCESS FAILURE, STATED PLAINLY:** two directories, one remote, and a
  manual copy step with nothing asserting they agree. The identity tests now fail
  loudly in whichever tree is wrong, which is the only durable fix.

- D230 **CHANNEL-WISE TRUST IN THE OFFSET LAYER — NO SHIP, AS PRE-REGISTERED.
  THE OPENER PRICES SCHEDULE AND UNDER-PRICES AVAILABILITY, EXACTLY AS
  PREDICTED, AND THE EFFECT IS STILL TOO SMALL TO SHIP.** Owner relayed an
  external proposal: stop spending the whole model-market disagreement at one
  trust coefficient and let each channel earn its own. **Prereg
  `data/d230_prereg.md` sha256 `6ce46dcfbc6d34fbf74c4bbfa9643f10076de3178e
  ea9f37cd9cf1016963c23a`, hashed before any challenger log loss existed**, with
  four numbered predictions including a headline prediction of NO-SHIP.

  **THE NESTING, WHICH IS WHAT MAKES THIS A TEST RATHER THAN A SEARCH.**
  `m_off = m_open + b(m_blind - m_open) + SUM_k d_k c_k + g*rest + d*|m_open|`
  with the L2 penalty on the `d_k` ONLY. At lam -> inf every `d_k` -> 0 and the
  challenger IS the shipped model, so the null is the incumbent rather than
  zero (D198's rule). `lam` is picked on an inner split of the training block;
  both arms are refitted by the same code on the same folds.

  | | |
  |---|---|
  | season-clustered mean delta | **-0.000093 nats** |
  | 95% CI (5 dof) | **[-0.000346, +0.000161] — CONTAINS ZERO** |
  | t | -0.938 |
  | better in | 4/6 seasons |
  | MDE80 (stated first, placebo null) | 0.00104 |

  **STOP CLAUSE MET: `nbapred/market/offset.py` NOT TOUCHED, no ship diff
  written, the single-b offset stands.** MDE80 came from a WITHIN-SEASON
  PERMUTATION of the channel block refitted through the same folds — the null's
  own dispersion — not from a guessed sd; my first version guessed 0.60 nats of
  per-game loss and produced 0.01763, which was 17x too large and would have
  made any result look underpowered.

  **THE PREDICTIONS, SCORED.**
      T1 d_sched < 0   **CONFIRMED  mean -0.0285, negative in 4/4 live folds**
      T2 d_comp  > 0   **CONFIRMED in sign, mean +0.1154 — but see below**
      T3 |d_k| small   PARTIAL: lam spans 1e2..1e9; d_tank reaches +0.56
      T4 NO-SHIP       **CONFIRMED, and the magnitude too (|delta| << 0.001)**
  **T1 IS THE ONE TO TRUST AND T2 IS NOT SEPARATELY IDENTIFIED.** `m_sched`
  is orthogonal to the other channels (r = -0.015 vs m_ff, -0.001 vs m_comp),
  so its coefficient is cleanly estimated: **the opener really does price rest,
  back-to-backs and home edge, and gives them LESS than the common trust.**
  But `m_ff` and `m_comp` correlate **0.800**, and their fitted deviations come
  out near-equal and opposite (-0.1160 / +0.1154) — the signature of a ridge
  trading one collinear column against the other, not of two separately
  measured trusts. The honest reading is a JOINT ff->comp tilt, and it is not
  evidence that availability specifically earns +0.115.

  **THE FRAME, AND A CHANNEL THAT IS DEAD.** `PROD_SEASONS` + `COMPONENT_OUT`
  (new, gated, default off) dump the five channels per game with
  `sum(channels) == margin()` ASSERTED on every game; measured max deviation
  7.11e-15 over 8,286 games, 2019-20..2025-26. **`m_late` (the D90 late-state
  layer) is IDENTICALLY ZERO on 100% of games in this frame** — so the margin is
  four live channels, not five, and `d_late` is an unidentifiable zero column.
  **ATTRIBUTION CORRECTED (D233): that is not a discovery, it is the documented
  D112 state.** D112 demoted the layer on a pre-registered rule (held-out
  21-23 +0.00014 ns against dev 24-26 +0.00267, DiD +0.00253 SIG — it helped
  measurably more on the seasons it was developed on), and `LATE_STATE` has
  defaulted to off ever since, so `late is None` and the term is exactly 0.0 by
  construction. The original wording here read like a find; it was a
  confirmation. `m_tank` is nonzero on only 27.5%. The fitted `b` averages 0.346
  across folds, independently reproducing the shipped 0.3564.

  **WHY THIS WAS WORTH RUNNING EVEN THOUGH THE NULL WAS PREDICTED.** A NO-SHIP
  that was predicted in advance is evidence about the architecture; the same
  result found afterwards is only a null. D157 said the lever is the penalty,
  not the basis. This is the same finding on a different object: re-parameterising
  one signal into four does not create information, and the one channel whose
  trust is cleanly identified moves in the direction theory said it would.

- D230b **THE CERTIFICATION PIPELINE WAS NEVER BIT-REPRODUCIBLE, SO EVERY
  `max|dp| = 0` CONTROL IN THIS REGISTER HAS BEEN MEASURING A NOISE FLOOR.**
  Found while trying to prove a refactor inert.

  **THE MEASUREMENT.** Running `prod_by_season.py` TWICE, same code, same
  settings, same machine: **max|d p_us| = 1.577e-14 on 5,616 of 6,148 games.**
  The refactor under test differed by 1.038e-14 — **LESS than the pipeline
  differs from itself.** A control that cannot reach zero cannot distinguish
  "inert" from "small", and D134's discipline asks for exactly that zero.

  **CAUSE: unordered aggregation feeding a floating-point reduction.** Float
  addition is not associative, and several DuckDB scans that feed numeric
  reductions never pinned their outer row order (the `ORDER BY` clauses sit
  inside window functions, and DuckDB parallelises the scan). Two fixes landed:
  `composition.strength()` now sums in `player_id` order, and the four-factors
  aggregate got `ORDER BY 1,2`. **Floor 1.577e-14 -> 3.220e-15, games affected
  5,616 -> 1,666.** NOT ZERO, and I stopped there rather than guess a third
  time: per-channel localisation (using D230's new dump) shows the residue is
  spread across ALL FOUR live channels — m_sched on 3,609 games, m_tank 1,937,
  m_ff 571 at 2.13e-13, m_comp 238 — i.e. systemic, several more call sites, a
  project rather than a patch.

  **WHAT IT DOES AND DOES NOT THREATEN.** Gate deltas in this register are
  ~1e-3 nats; the floor is ~1e-13 on a margin. **No conclusion anywhere is at
  risk.** What IS wrong is the stated meaning of the controls: the reproduction
  figure cited as `9.8e-15` was not evidence that a change was inert, it was the
  noise floor being reported as if it were a measurement. The right claim for
  such controls is "below the pipeline's reproducibility floor of ~1e-14", and
  D230's own refactor control is now stated that way.

- D231 **THE MINUTES-TO-POINTS BRIDGE, AND A DIAGNOSTIC THAT POINTED THE WRONG
  WAY.** Owner relayed a research program built on D144's mechanism: promoted
  players do not preserve per-minute production, so the remaining error is in
  the minutes-to-points bridge. Tested on 190,197 player-games, 1,063 players,
  2018-19..2025-26.

  **(A) DIAGNOSTIC — and it came out with the OPPOSITE SIGN to D144.** Building
  each player's trailing minutes and points as EWMA (half-life 8 games, strictly
  prior) and scoring the shipped bridge `pts_hat = minutes x trailing_rate`, the
  residual is MONOTONE INCREASING in the minutes deviation across all ten
  deciles (-0.704 at the bottom to +1.066 at the top; OLS slope **+0.0787 pts
  per extra minute above a player's trailing norm**). Read naively this says
  players who exceed their usual minutes produce MORE per minute, not less.

  **THAT READING IS WRONG AND I ACTED ON IT FOR ONE STEP BEFORE CATCHING IT.**
  Minutes and production are jointly determined inside a game: blowouts, foul
  trouble and mid-game injuries all produce low minutes at depressed value,
  which manufactures the whole gradient without any promotion effect existing.
  The endogeneity was flagged in the script's own docstring and I still drew a
  causal inference from it. **A second measurement in the same script (B) was
  ALSO mislabelled**: it reports a walk-forward improvement of -0.028 RMSE,
  7/7 seasons, CI excluding zero — but it conditions on ACTUAL minutes, so it
  answers "if the minutes forecast were perfect" and is **NOT USABLE AT BET
  TIME.** It is retained as a bridge diagnostic and must never be quoted as a
  shippable gain.

  **THE CLEAN TEST IS D231b, AND IT VINDICATES D144.** See below: measured
  bet-time, the model UNDER-penalises absences, which is exactly what "promoted
  replacements underperform" predicts. The lesson is that (A)'s association and
  the causal question have opposite signs, and only the second one is decidable
  from bet-time data.

- D231b **THE MODEL UNDER-PENALISES ABSENCES, 7/7 SEASONS.** Regressing the
  production margin residual on the expected-absence differential, with expected
  outs taken from the walk-forward as-of-open P(out) artifact (D201) so nothing
  reads tonight's minutes:

      pooled     residual = -0.2577 - 0.5653 * out_diff   t = -7.39, n = 8,239
      clustered  slope -0.5367, 95% CI [-0.7348, -0.3385], SAME SIGN 7/7

  Monotone across buckets: at out_diff <= -2 the residual is +1.373, at
  out_diff > +1 it is -2.040. **An absence costs about half a point of margin
  MORE than the departing player's own `talent x minutes`, which the composition
  leg has already removed.** That second-order cost is exactly what D133 arm C
  and D144 diagnosed and never priced.

- D232 **GATE: SHIP. THE ABSENCE-RESPONSE TERM.** Prereg `data/d232_prereg.md`
  sha256 `fe77ff1eba18b1dba5740be59fc53803941ecc49b87b2f1e2435b41f6b6b5cc8`,
  hashed before any challenger log loss existed, with four numbered predictions.

      margin += beta * (E[absences_home] - E[absences_away])

  beta = 0 IS the shipped model, so the null is the incumbent (D198's rule).

  | | main arm | + strength control |
  |---|---|---|
  | season-clustered mean delta | **-0.002174** | -0.002318 |
  | 95% CI (5 dof) | **[-0.003299, -0.001048]** | [-0.003610, -0.001027] |
  | t | **-4.96** | -4.62 |
  | better in | **6/6** | 6/6 |
  | mean beta | -0.6448 | **-0.7035** |
  | calibration veto | PASS | PASS |
  | MDE80 (stated first) | 0.00107 | — |

  **ALL FOUR PREDICTIONS CONFIRMED, INCLUDING THE ONE THAT COULD HAVE KILLED
  IT.** T3 asked whether `out_diff` was really TEAM STRENGTH in disguise — bad
  teams rest and tank more. Under an arm that also regresses on the model's own
  margin, beta does not shrink, **it grows** (-0.645 -> -0.704). The effect is
  availability.

  **A FEATURE-DEFINITION BUG I SHIPPED INTO MY OWN FIRST RUN.** The prereg
  specifies EXPECTED absences (sum of P(out)). `prod_by_season.py` emitted
  `n_out = len(outs[t])`, and under SOFT_AVAIL `outs[t]` is a DICT over every
  rostered player carrying ANY out-probability — so len() is a HEADCOUNT (mean
  1.70/team) not an expectation (mean 0.96/team). **The code comment asserting
  it "is the EXPECTED number out" was wrong: len != sum.** The first gate run
  used the headcount and was materially weaker (t -2.82, CI upper -0.000221)
  than the pre-registered quantity (t -4.96, CI upper -0.001048). Both are now
  emitted (`eo_*` alongside `n_out_*`) and both are reported; the prereg governs
  which is the arm.

  **SHIPPED** as `nbapred/model/absence.py`, wired as a sixth channel of
  `margin_components()`. Frozen coefficient **-0.8284** (full-frame fit, which is
  what the walk-forward rule "fit on all prior seasons" prescribes for 2026-27);
  per-fold betas -0.28/-0.62/-0.71/-0.71/-0.81/-0.74 are kept in the artifact so
  the upward trend stays visible rather than being averaged away. `ABSENCE_TERM=0`
  restores the pre-D232 margin. **CONTROL: with the switch off, max|d p_us| =
  3.830e-15 against the pre-D232 run — at the D230b reproducibility floor, which
  is now the correct form for such a claim.** With it on, 7,349 of 8,286 games
  move, mean |dp| 0.0215. NOTE: the certified per-season numbers from a
  production run use the FROZEN full-frame coefficient and are therefore
  IN-SAMPLE; the walk-forward gate above is the evidence, not those.

- D233 **MULTI-STATE AVAILABILITY: THE MINUTES LADDER IS REAL, MONOTONE, STABLE
  8/8 SEASONS — AND 200x TOO SMALL FOR THE SIDES MODEL.** For 187,261
  player-games that were actually PLAYED, attenuation = actual minutes /
  trailing minutes, conditioned on the last status published STRICTLY BEFORE
  game day:

      (none)        n=182,623   1.0086
      Available     n=    366   1.0043   -0.4%
      Probable      n=  1,691   0.9805   -2.8%
      Questionable  n=  2,404   0.9490   **-6.0%**
      Doubtful      n=     60   0.8824   -12.6%
      Out (played)  n=    117   0.8204   -18.8%

  Perfectly ordered by severity. The Questionable effect is season-clustered
  **-0.0629, 95% CI [-0.0870, -0.0388], same sign 8/8.** So `weight = 1 - P(out)`
  really does over-credit a player who plays through doubt: the shipped leg has
  no way to express "plays, but restricted".

  **AND THEN THE EXPOSURE ARITHMETIC KILLS IT FOR SIDES.** Minutes-weighted
  across the whole status distribution the effect is **0.127% of team
  minutes-value = ~0.0041 pts of margin**, against the D232 absence term's
  -0.83 pts per expected absence — **200x smaller.** Questionable player-games
  are 1.28% of the sample. NOT GATED into the sides model: an effect three
  orders of magnitude under the D232 term cannot clear a log-loss gate, and
  running one anyway would only manufacture a second-look opportunity.

  **WHERE IT DOES BELONG IS PROPS**, which is also where the owner's own roadmap
  puts validation: a 6% minutes haircut on a Questionable starter moves HIS
  points line directly, un-diluted by nine team-mates. Recorded as a props input
  with the ladder above ready to use.

  **NOT BUILT, AND WHY.** The constrained replacement graph needs rotation
  transitions; there is NO `game_rotation` table, and `lineup_stints` covers
  48% / 20% / 27% / 47% / 103% / 66% / 107% of games by season — too patchy to
  learn who absorbs whose minutes without the coverage itself selecting the
  sample. The owner's own note that backfilling those seasons "would be
  worthwhile" is confirmed, and it is a data-acquisition task, not a modelling
  one.

- D233b **CERTIFICATION MANIFEST — the instrument D230b said was needed.**
  `scripts/cert_manifest.py` records the code rollup hash over 81 files, hashes
  of every input artifact, the shipped coefficients, library and python versions,
  git HEAD, and **the 14 environment switches that change the numbers** — because
  an unset switch is a silent default and D229 is an entire entry about silent
  defaults. `--check` re-derives and diffs. Crucially it records the MEASURED
  NUMERIC FLOOR (p_us 3.3e-15, margin 2.2e-13) and states in the artifact that
  controls must be compared against that floor and never against zero. This is
  the honest replacement for a `max|dp| = 0` control that was never achievable.
  `docs/OCTOBER_CAPTURE_SPEC.md` freezes the 2026-27 capture schema, the
  30-minute event-pairing window, the RESPONSE-vs-CATCH-UP split that quote age
  now makes possible, four pre-registered predictions, and the go/no-go rule —
  all decided while no 2026-27 price exists.

- D234 **"CAN WE USE THE TIMESTAMPS?" — THE INVENTORY, NOT THE INTENT. TWO OF
  THREE PROSPECTIVE CLOCKS ALREADY WORK; NOTHING HISTORICAL IS RECOVERABLE; AND
  THE ONE THAT IS MISSING IS A DOWNLOAD, NOT A MODEL.**

  | clock | status |
  |---|---|
  | our receipt time (`snapshot_ts`) | **WORKS** — written per poll, fsync'd |
  | book post time (`book_last_update`) | **WORKS** — in `odds_quotes`, populated by `load_odds.py` |
  | injury event time | **MISSING** — see below |
  | anything historical | **ABSENT** |

  **HISTORICAL: NO, AND NOT FOR A MODELLING REASON.** `odds_quotes` and
  `bet_quotes_panel` are empty; the entire historical odds stack (`odds_open`,
  `odds_hist_sbr`, `bkp_panel_rows`) is one row per game with a TWO-VALUE
  open/close `phase` categorical and **no timestamp column anywhere**. On the
  availability side `injury_reports_pit.report_date` is a `DATE`, and
  **`edition` has cardinality 1 — all 125,704 rows are `05PM`.** There is no
  timestamp PAIR in held data that could measure a market response to an injury
  event, so that channel is a 2027 result no matter what is modelled.

  **AND THE RAW ARCHIVE CONFIRMS IT IS AN ACQUISITION GAP, NOT AN INGEST BUG:
  all 1,260 injury-report PDFs on disk are 5PM editions.** The NBA publishes
  several editions a day; this project has only ever downloaded one of them.

  **THIS FOUND A DEFECT IN MY OWN OCTOBER SPEC, TWO DAYS OLD.**
  `docs/OCTOBER_CAPTURE_SPEC.md` §2 assumed the report poller would produce
  status TRANSITIONS with times attached. With one edition per day every event
  carries the same clock time, so §3's 30-minute pairing window would compare an
  edition against itself and §5 would return a confident null that means
  nothing. **The prerequisite is now stated ahead of everything else in that
  document: fetch EVERY published edition and stamp each with `observed_ts`.**
  Cheap — the same endpoint with a different edition key — and without it the
  price capture cannot be paired with anything.

  **ONE OPEN INCONSISTENCY, STATED WITHOUT OVERCLAIMING.** The trading frame
  `ats19_frame_honest.csv.gz` was written 2026-08-06 and therefore predates the
  D232 absence term by construction, so **every published betting figure excludes
  it.** Whether re-running raises or lowers them is unknown and untested. I also
  checked whether that frame matches a current-code run and it matches NEITHER
  honest-hard (mean |dp| 0.0142) nor honest-soft (0.0111) — but that is NOT
  evidence of staleness: the 19-season trading frame and the 5-season
  certification artifact are built by different scripts with different refit
  regimes, and a difference of that size is expected between them. Recorded as
  an open item rather than a defect, because I have not demonstrated it is one.

- D235 **INTEGRATION AUDIT: D232 WAS GATED ON THE WRONG LAYER. THE TERM
  SURVIVES, AT HALF THE CLAIMED SIZE.** Raised by an external reader against the
  shipped implementation. **THE CRITICISM IS CORRECT AND THE CLAIM WAS
  OVERSCOPED.**

  **THE ORDERING PROBLEM.** D232 gated the absence term on the MARKET-BLIND
  margin. Production does not forecast with that margin — `bet_engine` passes it
  through the offset layer. Layers do not commute:
  `offset(m + absence) != offset(m) + absence`. The term is added BEFORE the
  offset, so its effect on the forecast actually emitted is multiplied by the
  edge coefficient: **-0.8284 pts per absence becomes -0.2953, a 2.81x
  attenuation.** Purely additive channels commute and their ordering is a
  ~1e-13 floating-point question; a SHRINKAGE layer above them is not, and that
  distinction is what I missed.

  **RE-GATED THROUGH THE COMPLETE STACK**, with the offset REFIT PER ARM — a
  downstream learned layer fitted on the old upstream representation is not the
  right comparator, and holding it fixed would credit or penalise the challenger
  for a stale fit rather than for its own content.

      layer                             mean delta   95% CI (5 dof)       t     better
      BLIND (what D232 gated)           -0.002174  [-0.003299,-0.001048] -4.96   6/6
      **FINAL STACK (what ships)**      **-0.001018** **[-0.001728,-0.000309]** **-3.69**  **6/6**

  Calibration PASS. **So the term survives integration at roughly HALF the
  blind-layer figure, and -0.001018 is the number that describes the shipped
  system.** `nbapred/model/absence.py` and `data/absence_coefs.json` now carry
  both, with an instruction to quote the final-stack one.

  **AN UNEXPECTED PARTIAL OFFSET TO THE ATTENUATION.** The offset's fitted edge
  coefficient RISES for the challenger in ALL SIX seasons (mean 0.306 -> 0.346;
  2024-25 0.274 -> 0.343). A better market-blind margin earns MORE trust from
  the offset, which recovers part of what the 2.81x attenuation takes away. That
  is only visible because the offset was refit per arm.

  **THE BAD-CONTROL POINT WAS ALSO RIGHT, AND IT CLEARS.** D232's confound arm
  controlled for `m_total`, which already contains the availability-sensitive
  composition channel — i.e. partly the treatment. Redone against `m_ff`, the
  four-factors channel, which takes NO availability input:

      control m_total (bad control)  beta -0.8907  CI [-1.2887,-0.4926]  7/7
      control m_ff (pre-availability) beta -0.8634  CI [-1.2492,-0.4775]  7/7

  Materially unchanged. **The honest claim is that the coefficient survives
  control for team strength and is consistent with an additional
  replacement/lineup cost — NOT that the experiment causally identifies the cost
  of an absence.** Team reporting behaviour, injury severity, roster depth,
  late-season resting and schedule congestion are not separated here.

  **STATUS RECLASSIFIED: gated for the market-blind layer, integration-audited
  through the final stack, and NOT independently confirmed.** These seasons
  developed D232, so this audit is an integration check, not fresh evidence. The
  decisive test is the frozen 2026-27 on/off shadow, and it must not be retuned
  mid-season.

- D236 **THE SINGLE-BOOK COVER RATE, AND WHY THE BEST-OF-NINE TIER HAS NONE.**
  An external reader inferred a 265-192-3 record from the published 460 bets and
  +48.9u, and asked that it be REGENERATED before being presented as verified
  rather than derived. Regenerated from `data/wf_perbet_OFFSET.json`, and the
  inference was exact:

      1 book (observed)   265 W  192 L  3 push   457 graded
                          **57.99% cover**  +48.91u  **+10.63% ROI**
                          break-even at -110 = 52.38%

  **THE BEST-OF-NINE TIER CANNOT BE EXPRESSED THE SAME WAY, AND A NAIVE COUNT
  WOULD MISSTATE IT.** k=1 has exactly three distinct per-bet values (-1.0, 0.0,
  +0.9091), which is what makes a genuine W-L-push count possible. **k=9 has 417
  of 460 bets at FRACTIONAL values** — that tier is a modelled distribution of
  available numbers, not settled outcomes, so its per-bet entries are
  expectations. Counting `ev > 0` there returns **61.96%**; the ROI-implied
  effective figure is **61.09%**. The two differ because partial expectations do
  not resolve to wins, and quoting the count as a cover rate would overstate the
  observed record by ~0.9pp against a number that is not an observed record at
  all.

  **DOC DECISION (owner): the results doc carries the single-book sentence and
  NOTHING ELSE.** The k=9 reasoning above is exactly the kind of methodological
  aside that belongs in the register — the owner's standing rule for
  `docs/REVIEW.md` is that it is a RESULTS document and extraneous explanation
  goes elsewhere (D218-D223 editorial cycle). So the review states 265-192-3 /
  58.0% / 52.38% break-even and stops.

  **AND THE READER FOUND A REAL PUBLICATION GAP**: the repo published aggregate
  P&L but never the per-bet artifact, so no reader could check the record.
  `data/wf_perbet_OFFSET.json.gz` (116 KB) is now shipped and cited in the
  review's header, and the re-derivation of 265-192-3 / 57.99% / +10.63% FROM
  THE SHIPPED FILE is asserted at commit time.

- D237 **REGIME-CONDITIONED BETTING — NOT SUPPORTED, AND NOT RESOLVABLE ON THIS
  DATA. DIAGNOSTIC ONLY; NOTHING SHIPPED.** Owner: "in a period where all models
  are wonky, may be profitable to capture there... lets try more sophisticated
  trading strategies". A genuinely untested axis — every prior gate conditions
  the FORECAST; none conditions the BET DECISION on the state of the market.

  Three strictly-prior regime states, each computed from games settled before the
  bet's own date: trailing market log loss (the owner's "wonky market"), trailing
  |actual - open| (outcome unpredictability), and days into the season.

  **(a) ON THE 888-BET BOOK (2019-26, observed single-book tier), EVERY REGIME
  IS NON-MONOTONE WITH A HOT BUCKET IN THE MIDDLE** — mkt_ll_tr runs
  -2.77 / +5.47 / **+13.96** / +4.10 across quartiles; mkt_ae_tr runs
  +0.74 / **+14.86** / +2.78 / +2.33. All three season-clustered slopes ns.
  **THE SEARCH NULL SETTLES IT: observed best spread across the three candidates
  is 16.74 ROI points against a permutation null whose MEDIAN is 17.28 and whose
  95th percentile is 26.52, p = 0.553.** The structure is SMALLER than what
  searching three candidates buys from pure noise. Bucket SE is +/-6.3 ROI
  points, so this design cannot resolve below ~25 points of spread.

  **(b) THE POWERED FORM IS ALSO NULL, AND THE CENTRAL HYPOTHESIS POINTS THE
  WRONG WAY.** The claim "when the market is wonky our edge is larger" is about
  the FORECAST advantage, which is observable on all 8,239 games rather than the
  12% that clear a betting threshold, and whose per-game quantity is a log-loss
  difference rather than a +0.91/-1.00 lottery ticket. Measured as
  `edge = ll_open - ll_offset`:

      trailing market log loss   Q1 +0.00572   Q4 +0.00158    gap -0.00414
      trailing |actual-open|     Q1 +0.00023   Q4 +0.00274    gap +0.00251
      days into season           Q1 +0.00444   Q4 +0.00250    gap -0.00193

  **When the market forecasts WORST our edge is SMALLEST** — the opposite of the
  hypothesis. All three season-clustered slopes ns.

  **(c) THE FINDING THAT MATTERS IS THE POWER CEILING, NOT THE NULL.** MDE80 on
  a Q4-Q1 edge gap is **0.00571 nats, which is 2.2x the entire pooled edge of
  +0.00259.** Seven seasons cannot resolve a regime effect smaller than twice the
  total effect it would be modulating. So the honest statement is NOT "regimes do
  not exist" but **"this corpus cannot answer the question, and no amount of
  re-slicing it will."** A regime rule is precisely a subset selector, and the
  register's manufacturing-capacity result (best-of-N random subsets buy +2.54
  ROI points from nothing) says a subset selector searched against realised
  returns on the seasons that built the strategy is the single most dangerous
  thing this project can do.

  **NOT SHIPPED, NOT PRE-REGISTERED FOR A GATE, AND NOT TO BE RETRIED ON THESE
  SEASONS.** If the idea is to be tested it needs either a lower-variance
  endpoint (CLV resolves in weeks and needs no de-vigging, which is why D-series
  already names it the primary live diagnostic) or the 2026-27 season as fresh
  data. The one directional hint worth carrying forward, at ns: early season
  (Q1 days_in, median 20 days) shows the largest edge of any bucket in (b),
  which is at least consistent with unsettled rosters being harder for the
  market to price.

- D238 **WHEN DO WE WIN — THE SIGNAL IS REAL, GRADED, AND ENTIRELY IN THE TAIL;
  WINS ARE THE MARKET COMING TO US; NO REGIME VARIABLE (PCA INCLUDED) ADDS
  ANYTHING; AND THE PROFITABILITY QUESTION IS ANSWERABLE IN 45 BETS ON CLV OR
  20 SEASONS ON ROI.** Owner: "think of multiple other ways to condition betting
  on the right game. do we simply not have good enough signal? very interested
  in regime gates — maybe pca. when to determine when market is truly
  profitable? analyze when we win and dont." Diagnostic only; nothing shipped.

  **(A) THE SIGNAL QUESTION HAS A SHARP ANSWER: YES, AND ONLY IN THE TAIL.**
  P(model side covers) by decile of |stated edge| on ALL 8,237 games 2019-26:
  flat at 48.7-51.2% through D1-D8 (|edge| up to ~1.1 pts), then **54.4% at D9
  and 57.4% at D10** against a 52.38% break-even. Season-clustered slope
  **+3.49 pp per point of |edge|, 95% CI [+1.35, +5.63], K=7, SIGNIFICANT** —
  the first significant dose-response this project has measured on the betting
  signal. Below ~1.2 pts of disagreement there is NOTHING; the walk-forward
  threshold already harvests the tail (book median |edge| 1.25 vs slate 0.64).

  **(B) WHEN WE WIN: WHEN THE MARKET LATER AGREES.** On the 888-bet observed
  book, the line moved TOWARD our side on **62.2%** of bets (mean CLV **+1.07
  pts**, positive in ALL 7 seasons):

      CLV > 0 (market later agreed)   n=545  cover 59.3%  ROI +12.98%
      CLV = 0                         n=133  cover 53.4%  ROI  +1.91%
      CLV < 0 (market later opposed)  n=200  cover 45.0%  ROI -13.88%
  **DENOMINATOR NOTE (D239b), because a reader flagged 545+133+200=878 against a
  stated 888:** both figures are right and the block mixed them. The bucket rows
  are GRADED-only (878 = 888 minus 10 pushes); the 62.2% share is over ALL 888
  bets (552 of 888 had CLV>0, of which 545 were graded). Presentation defect,
  not arithmetic.

  Diagnostic, since the close is not knowable at bet time — but it is the
  healthiest possible decomposition: the edge is information the close
  subsequently prices, not noise that settlement happens to reward.

  **(C) THE BET-TIME CONDITIONER FAMILY: ONE LOOK, ONE NULL.** Six candidates
  (edge quintile, home/away side, favorite/dog side, line magnitude,
  absence-side, season third) examined together; family permutation null over
  all of them: **observed max spread 27.3 ROI pts vs null median 17.7, 95th pct
  27.5 — p = 0.056, AT the boundary and not past it.** The spread is carried
  entirely by edge_quintile (bottom 48.3%/-7.7%, top 62.7%/+19.6%), which is
  the same phenomenon as (A) measured on the weaker endpoint. **No other
  conditioner clears its own noise.** Two hypotheses recorded for 2026-27
  pre-registration, NOT for retry on these seasons: (1) bets on the
  absence-heavy side underperform (52.4%/-0.0% vs 58.7%/+11.9% on the healthy
  side, n=319/403, mechanically consistent with D232's under-penalisation,
  which production now corrects); (2) mid-season bets outperform (ns).

  **(D) PCA REGIME, AS ASKED: RAN, INTERPRETABLE, NULL.** Eight strictly-prior
  market-state variables (trailing market LL, |actual-open|, open->close
  movement, favorite-cover rate, home-cover rate, |line|, totals, days-in),
  rotated on the 2017-19 lead-in only, applied forward. PC1-3 hold 91%:
  PC1 = general scoring/volatility state, PC2 = line-magnitude regime,
  PC3 = market-accuracy regime. **All three ns against the powered all-games
  edge endpoint.** D157's lesson holds on a new object: rotation does not
  create information the variables do not carry.

  **(E) "WHEN TO DETERMINE WHEN THE MARKET IS TRULY PROFITABLE" — NOW A
  NUMBER.** Per-bet return sd 0.945 at ~127 bets/season: distinguishing the
  pooled +5.2% ROI from zero needs **~2,600 bets = 20 seasons**; detecting a
  subgroup 5 ROI pts better than the rest needs **88 seasons**. The same
  question on CLV (mean +1.07, sd 2.56) resolves in **~45 bets — under half a
  season.** This is the quantitative form of the standing rule that CLV is the
  primary live diagnostic: profitability is a CLV question in-season and an ROI
  question only across eras.

  **NOT DONE, DELIBERATELY:** no threshold change off table (C) — the bottom
  book quintile sitting below break-even is an observation ABOUT the
  walk-forward selector, and only that selector, run walk-forward, may act on
  it. Choosing a cutoff from this table is tuning on realised returns.

- D239 **THE SIGNAL IS RANKABLE — AND IT SURVIVES PROPER SMALL-K INFERENCE.
  NO OBSERVABLE STATE MODIFIES THAT TRUST.** A reviewer proposed the fundamental
  diagnostic this project had never run: regress the SIGNED advantage on the
  stated edge, once against the opener's residual and once against CLV. Run on
  8,237 games, 2019-26.

      sign(d)*(Y-O) = a + b|d|    pooled b **+1.1975**   mean +0.792 pts
      sign(d)*(C-O) = a + b|d|    pooled b **+0.6735**   mean +0.483 pts

  Monotone across every |d| quintile in BOTH (opener residual +0.275 -> +2.311;
  CLV +0.172 -> +1.097). **Both slopes positive, both CIs excluding zero: the
  model's claimed edge is RANKABLE, not merely present on average.**

  **THE INTERACTION MODEL REPLACES D238's BUCKETS AND FINDS NOTHING.** Fitting
  `sclv = a + b|d| + g*z + delta*(|d| x z)` — with the |d| main effect as the
  control, so a state cannot "win" by selecting larger edges — over a pre-listed
  family of three: roster-transition factor (the MECHANISM behind D238's
  early-season hint, built as 1 - returning-minutes share, strictly prior),
  trailing market log loss, and total expected absences. **delta is ns in all
  three** (-0.0685 / +0.1730 / -0.0267; |t| <= 1.18). Combined with D237,
  D237b and D238 that is 0/3 continuous interactions after 0/3 bucket tests
  across four independent attempts. **The edge is graded but state-invariant on
  everything observable in this corpus.**

- D239b **THE REVIEWER WAS RIGHT ABOUT THE INFERENCE, RIGHT ABOUT THE
  POST-SELECTION, WRONG ABOUT THE ARITHMETIC — AND MY POWER NUMBER WAS WORSE
  THAN EITHER OF US SAID.**

  **(1) K=7 CANNOT CARRY AN ORDINARY CLUSTER INTERVAL. RE-TESTED FOUR WAYS AND
  IT HOLDS.**
      signed opener residual  CR1 [+0.344,+2.051]  CR2 [+0.240,+2.155]
                              **wild cluster boot-t p=0.0233**  LOSO all
                              positive (min +0.852)  sign-flip perm p=0.0318
      signed CLV              CR1 [+0.169,+1.178]  CR2 [+0.103,+1.244]
                              **wild cluster boot-t p=0.0082**  LOSO all
                              positive (min +0.428)  sign-flip perm p=0.0158
  No single season carries either result, and the bootstrap-t — the standard
  few-cluster remedy — clears both.

  **(2) "SIGNAL ONLY IN THE TAIL" IS POST-SELECTION AND IS WITHDRAWN AS A
  FINDING.** D238 pooled D1-D8 after seeing them and located the break near 1.2
  pts by eye. The defensible claim is the CONTINUOUS SLOPE. The hinge
  `b2(|d|-1.25)+` is now frozen at tau=1.25 in `docs/SHADOW_2026_27.md` as a
  prospective test (C0), not asserted historically. The related "the selector
  harvests exactly that tail" was also overstated — book median |d| is 1.25, so
  about half the book sits BELOW the putative breakpoint.

  **(3) THE CLV>0 SPLIT IS DIAGNOSTIC AND WAS LABELLED SO.** It conditions on the
  close and can never be a gate; reaffirmed.

  **(4) THE COUNTS DO RECONCILE — 552 vs 545 IS THE ANSWER.** 888 bets, 10
  pushes, 878 graded. The bucket rows are graded-only (545+133+200=878); the
  62.2% share is over all 888 (552 of 888 had CLV>0, 545 of them graded). Both
  correct, two denominators in one block. **Presentation defect, not arithmetic;
  D238 is annotated in place.**

  **(5) MY POWER FIGURE WAS WRONG BY 3-6x, AND WORSE THAN THE REVIEWER'S
  CORRECTION.** D238's "~45 bets to confirm CLV" assumed independence and took
  the observed +1.07 as truth. Block bootstrap by WEEK gives a **design effect
  of 2.91**, and a winner's-curse shrink is mandatory on an effect discovered
  in-sample:
      no shrinkage 131 bets  ·  30% shrink 267  ·  50% shrink 523
  The reviewer suggested 75-100; the honest range is **131-267**, i.e. one to two
  seasons at ~127 bets/season, not half a season. `SHADOW_2026_27.md` inherits
  that horizon.

  **SHADOW SPEC EXTENDED TO FIVE PRE-REGISTERED CANDIDATES**, all constants
  frozen before any 2026-27 price exists: C0 tail hinge (tau=1.25), C1
  uncertainty-adjusted lower-bound edge, C2 post-open information shock, C3
  perturbation robustness + price stress, **C4 paired pre-D232 vs production
  forecast** — the last being the cleanest causal-style test available, because
  the absence correction was designed and shipped BEFORE the depleted-side
  subgroup result that now motivates testing it. Primary endpoint throughout is
  incremental signed CLV **matched on |d| decile**, which is what stops a
  candidate winning by merely retaining larger edges — a real risk now that
  D239 has established the slope.

- D240 **PARTICIPATION-V2: THE LABEL IS MISALIGNED, THE MISALIGNMENT IS LARGE,
  AND FIXING IT MAKES THE MODEL SIGNIFICANTLY WORSE.** Prereg
  `data/d240_prereg.md` sha256 `30c8dbd2cb2effe5700e7f4a731431478a1dbbdee4153
  919d558f41b27c442f1`. Reviewer's #1 recommendation: `d200_participation.py`
  predicts whether the NBA will publish an administrative "Out", while the
  composition leg spends `1 - p_out` as an EXPECTED PARTICIPATION weight. Two
  different questions.

  **T1 CONFIRMED, AND FAR EXCEEDED.** Over 209,393 rotation-candidate
  player-games (appeared for the team within 21 days, trailing minutes >= 8),
  2019-26, absence rate 24.4%:
      not-yet-Out at the pre-game-day cutoff     **76.3%**  (prereg said >= 25%)
      minutes-weighted                           72.6%
      by trailing-minute band  8-15 86.2% · 15-22 79.0% · 22-30 66.9% · 30+ 64.9%
      P(absent | prior-report Out)               0.929
  The gradient is sensible and the gap persists at 64.9% even for 30+ minute
  players, so it is not purely a fringe-roster artefact. **WORDING CORRECTED
  (reader):** this measures the share whose last status STRICTLY BEFORE GAME DAY
  was not "Out". It does NOT show they were never designated Out — a same-day
  edition after the cutoff may have done so. The claim is about what was
  knowable at the bet-time cutoff, which is the only thing that matters for the
  leg, but the earlier phrasing overstated it. Exact opener timestamps will
  replace the coarse "strictly before game day" rule. **The shipped artifact
  predicts mean 0.1118 against an actual absence rate of 0.2435 — it
  under-predicts participation loss by more than 2x, because it was trained on
  a rarer event.**

  **T4 REFUTED — AND THIS IS THE RESULT THAT MATTERS.** Swapping the v2
  artifact through the FULL production stack (`POUT_ARTIFACT`, new env switch,
  default unchanged):
      GENUINE v2 folds (6)  mean **+0.002982**  CI **[+0.000642, +0.005322]**
                            worse in **6/6**
      all 7 rows            mean +0.002714  CI [+0.000722, +0.004707]  worse 7/7
  Not inconclusive: significantly worse, every scored season.
  **FOLD-COUNT CORRECTION (reader, verified):** `p_out_v2.csv.gz` starts
  2020-12-25 because 2019-20 is the initial training season, so the 2019-20 row
  had `eo_v = 0.00000` — an availability-BLIND fallback, not a v2 forecast. The
  honest statement is "worse in 6/6 genuine walk-forward folds; the unscored
  burn-in fallback was also worse." Dropping it makes the result slightly
  STRONGER.

  **THE MECHANISM IS NOT ESTABLISHED, AND MY FIRST EXPLANATION WAS
  MATHEMATICALLY WRONG.** I originally wrote that the leg DOUBLE-DISCOUNTS,
  because trailing minutes supposedly already absorbs routine non-participation.
  **That is incorrect.** `trail_min` is an EWMA over GAMES PLAYED ONLY, so it
  estimates `E[M | plays]`; multiplying by `P(plays)` is precisely how one
  obtains the unconditional `E[M]`. The two terms compose correctly and there is
  no double-count. A reader caught this and is right.

  A proposed replacement — that v2 leaks team minute mass because independent
  per-player removals violate `sum_i M_i = 240` — is CONCEPTUALLY sound but is
  NOT confirmed by the direct measurement:
      raw sum of trailing minutes        295.4 /team-game
      incumbent  sum (1-p_out)*tr_min    259.8   (**+19.8 OVER** 240)
      v2         sum (1-p_v2)*tr_min     232.7   (**-7.3 UNDER** 240)
  **v2 lands CLOSER to the 240 constraint than the incumbent does**, so "v2
  loses mass" is not what the arithmetic shows. Whatever the incumbent gains, it
  is not mass conservation.

  **WHAT IS ESTABLISHED:** the empirical result (worse 6/6, CI excluding zero)
  and the fact that expected outs/team rises 0.957 -> 1.658. **WHAT IS NOT:**
  why. The strongest surviving hypothesis is the reader's endogenous/exogenous
  split — a coach's DNP REALLOCATES minutes rather than reducing team strength,
  while an injury genuinely removes them, and the leg cannot tell these apart
  because it has no reallocation step. That is a hypothesis for a successor
  experiment, not a finding of this one, and it is recorded as such.

  **THE SUCCESSOR THIS POINTS TO** is a joint team-level minutes allocator that
  conserves the constraint by construction — `u_i = P_i(play)*E[M_i|play]` then
  `M_hat_i = 240 * u_i / sum_j u_j`, with role/position capacity — scored FIRST
  on player-minute allocation and props, not on sides. Because that architecture
  was suggested by inspecting this failure, any historical result on it is
  development evidence and 2026-27 is the clean confirmation.

  **STOP CLAUSE HONOURED**: `data/p_out.csv.gz` byte-unchanged, `nbapred/`
  untouched. `data/p_out_v2.csv.gz` is retained as EVIDENCE only and is
  explicitly NOT a 2026-27 shadow candidate — an arm that is significantly worse
  in 7/7 seasons does not earn shadow capacity.

  **TWO ERRORS OF MINE, BOTH CAUGHT BEFORE REPORTING.** (1) The first run joined
  status on `report_date < game_date`, keeping only the 25.9% advance rows; only
  891 of 4,752 game dates carry any advance row, so **81% of candidates were
  assigned status 'none' BY CONSTRUCTION** and T1 read 86.3%. That is D199's
  original defect reproduced verbatim. Rebuilt with carry-forward (last edition
  strictly before game day, as `report_out_map` does) and T1 fell to 76.3%.
  The tell was an 80% unlabelled rate persisting into the 30+ minute band —
  starters do not miss games unannounced at that rate. (2) Part 2's head-to-head
  reported a -1.30 nat "win" for v2, which is almost entirely the incumbent
  being scored at p~0 on players it never models; on the common report-listed
  universe its log loss is 0.681, not 1.66. The prereg anticipated an asymmetry
  but not one that large, and the figure is withdrawn as a model comparison.

- D241 **JOINT-MARKET DISTRIBUTION: NO SHIP. THE MECHANISM IS REAL, THE
  ESTIMATOR IS NOT.** Prereg `data/d241_prereg.md` sha256 `6c4d7890073fca92e4a
  187aa2ab28373ec2db3bf82f3ed7ca28a2feb1ebd30d0`. The offset layer reads the
  spread but never the TOTAL, and D198's variance null never used the one
  variance proxy priced by someone better informed than us.

      arm A (linear scale)  +0.001483  CI [-0.003649, +0.006614]  better 2/6
      arm B (power scale)   +0.000850  CI [-0.002614, +0.004313]  better 2/6
      arm C (+ de-vigged ML) +0.00198  CI [-0.01284, +0.01680]    K=3 only
      MDE80 (permutation, stated first) 0.00086

  **FOUR OF FIVE PREDICTIONS CONFIRMED; THE ONE THAT FAILED IS "IT SHIPS".**
  T1 confirmed — s1 > 0 in ALL SIX folds, so a higher total genuinely does imply
  a wider margin scale. T4 confirmed exactly: **0.00% side flips**, the scale
  term never repriced a direction. T3 confirmed (B ~ A). T5 confirmed: the
  moneyline adds nothing beyond the total, with a coefficient that swings
  -1.4 / -6.7 / -9.9 across its three folds.
  **THE VARIANCE MECHANISM IS DIRECTIONALLY SUPPORTED, BUT ITS INCREMENTAL
  FORECASTING VALUE IS UNPROVEN; unstable early estimates produce the observed
  harm.** The fitted s1 decays 0.236 -> 0.036 as training grows and the thin
  early folds over-extrapolate so badly that 2021-22 costs +0.0109 nats alone.
  **I FIRST WROTE "the failure is overfitting, not absence of signal" AND THAT
  OVERCLAIMS**: the scored evidence (better 2/6, CI straddling zero, later folds
  indistinguishable from the incumbent) is equally consistent with genuine
  coefficient decay, era variation, an effect too small to move log loss, or the
  offset layer having already absorbed it. A shrunk estimator would very likely
  prevent the +0.0109 disaster, but its first-order effect is to pull the
  challenger toward the incumbent — turning "harmful" into "approximately zero"
  is not the same as finding value.
  A SHRUNK VARIANT IS THE OBVIOUS FIX AND WAS DELIBERATELY NOT RUN — it is not
  in the prereg, and adding an arm after seeing A fail is the second-look trap.
  Recorded as a candidate for a future pre-registration.

  **COVERAGE FINDING THAT SHAPED ARM C:** moneyline data is 100% present for
  season_end 2019-2022, 50% in 2023, and **ZERO from 2024 onward** in both
  `odds_market` and `odds_hist_sbr`. The reviewer called this "an immediate
  experiment using data you already hold"; half of it is not held. Opening
  totals are 93.5% covered and were fine.

- D242 **JOINT MINUTE-CONSERVING COMPOSITION: THE HYPOTHESIS IS REFUTED, AND SO
  ARE BOTH PROPOSED MECHANISMS — INCLUDING MINE.** Prereg `data/d242_prereg.md`
  sha256 `4d7e98a44a71ee028b043a56fcfe7d0cf2cc1b9f43dedcdde5c8acf22afd7aa1`.
  Built after an external review argued that D240 was CONSTRUCTION_INVALID: it
  tested a broad participation probability as an INDEPENDENT per-player
  discount, and NBA teams allocate ~240 minutes regardless of absence, so a
  routine DNP REALLOCATES minutes rather than destroying them.

  **THE SCALE TRAP, HANDLED BEFORE ANY SCORING.** Under any 240-conserving
  allocation `sum_i m_i / 48 = 5` EXACTLY, so team strength stops being a sum of
  contributions and becomes 5 x (minutes-weighted mean talent). A naive
  full-stack comparison would have measured a broken 0.5 blend weight, not the
  hypothesis. Stage 1 therefore AFFINE-RECALIBRATES every arm per fold, which
  isolates the shape change from the scale artefact.

  **SIX ARMS x FOUR CONSTRUCTIONS, ALL DECLARED IN ADVANCE AND ALL REPORTED**
  (N1 proportional, N2 capped water-fill at 40 min, N3 role-tiered 160/80 by
  minutes rank since `nba_players` has NO position column, N4 half-conserved):

      arm        mean d-RMSE   95% CI                better   verdict
      A  broad, independent  **+0.0451**  [+0.0161,+0.0742]  0/6  **WORSE**
      B  narrow + alloc      **+0.0004**  [-0.0226,+0.0234]  4/6  neutral
      C_N1 broad + alloc       +0.0316    [-0.0093,+0.0724]  1/6  ns
      C_N2 broad + capped      +0.0258    [-0.0151,+0.0667]  2/6  ns
      C_N3 broad + role-tier   +0.0065    [-0.0311,+0.0442]  3/6  ns
      C_N4 broad + half-cons   +0.0323    [-0.0028,+0.0675]  1/6  ns
      B_N2 narrow + capped     -0.0044    [-0.0222,+0.0134]  4/6  neutral
      **ORACLE (actual minutes) -0.0827  [-0.1280,-0.0374]  6/6  BETTER**

  **T1 CONFIRMED, AND IT IS WHAT MAKES THE REST INTERPRETABLE: the minutes
  pathway is REAL.** Perfect minutes knowledge is worth **-0.0827 RMSE, 6/6
  seasons**. So there is genuine signal in who plays — and **none of our
  estimators capture any of it.**

  **T2 CONFIRMED** — arm A reproduces D240's harm at this layer (+0.0451, 0/6).
  **T3 CONFIRMED** — allocation ALONE is neutral (+0.0004, 4/6).
  **T4 REFUTED** — C vs A is **-0.0136, CI [-0.0326,+0.0055], ns**. Allocation
  moves in the right direction but recovers only a fraction of A's harm and not
  significantly.
  **AND THE DECISIVE ROW: C vs B = +0.0312 (t +2.00).** With allocation held
  FIXED, the broad label still hurts.

  **SO MINUTE CONSERVATION IS NOT WHY BROAD PARTICIPATION FAILED.** That
  refutes my D240 "double-discount" explanation AND the reviewer's "minutes
  vanish" replacement. The six-arm decomposition is what separates them; D240
  alone could not, which is exactly why it was CONSTRUCTION_INVALID rather than
  a refutation. **Narrowest supported conclusion: a broad non-appearance
  probability is a worse estimator of EXOGENOUS unavailability than the narrow
  administrative label, and reallocating endogenous DNPs injects noise rather
  than repairing an accounting error.**

  **AN UNPREDICTED MONOTONE PATTERN, WORTH CARRYING.** Harm falls as the
  allocation is constrained: N1 +0.0316 -> N2 +0.0258 -> N3 +0.0065, with the
  half-conserved N4 at +0.0323 — no better than FULL conservation, so the gain
  is from CONSTRAINT, not from conserving less. The narrow-label arms are
  neutral under every construction (B_N1 +0.0004, B_N2 -0.0044), confirming
  T3 twice. Tightening
  role capacity converges toward neutral, which is precisely the failure the
  reviewer anticipated — unconstrained reallocation floods minutes to
  deep-bench players. It converges to NEUTRAL, not to positive, so it does not
  rescue the arm; but any future allocator should be role-constrained from the
  start.

  **A HARNESS BUG I CAUGHT IN MY OWN T1 GATE.** The first ORACLE run returned
  EXACTLY +0.0000 with a zero-width CI. `SOFT_AVAIL` defaults on and takes
  precedence in the out-set branch, so `ORACLE_PLAYED_OUTS` was never reached
  and I had re-run the control. Had that been reported it would have declared
  the entire minutes pathway worthless on a config-precedence error — the exact
  opposite of the true result. Rerun with `SOFT_AVAIL=0`.

  **CORRECTION, POSTED AFTER THE FIRST REGISTRATION OF THIS ENTRY: ONE ARM DID
  WIN STAGE 1, AND I HAD ALREADY WRITTEN "no arm won".** `B_N3` and `B_N4`
  completed after the entry was pushed, and B_N3 is not marginal:

      **B_N3  narrow out-set + role-tiered allocation
              -0.0562  CI [-0.0676, -0.0448]  t -12.69  better 6/6  BETTER**

  That is **68% of the ORACLE ceiling** (-0.0562 / -0.0827). The original
  sentence was written from an incomplete grid and is wrong; it is retained
  above only so the sequence of claims is visible.

  **AND THE CONTROLLED TEST SAYS THE TIERING CARRIES EVERYTHING — THIS IS NOT
  AN AVAILABILITY RESULT.** Under the narrow label mean p_out is ~0.11, so
  `u_i ~ trail_min_i` and N3 reduces to **"replace trailing minutes with a
  canonical 160/80 starter/bench split"**. Re-running the same tiering on the
  HARD T2 out-set:

      hard T2 + N3 tiering   **-0.1275**  CI [-0.1827,-0.0722]  t -5.93  6/6
      ORACLE (perfect outs, NO tiering)   -0.0827  CI [-0.1280,-0.0374]  6/6
      soft p_out + N3 tiering (B_N3)      -0.0562  CI [-0.0676,-0.0448]  6/6
      **B_N3 minus hard-T2-N3 = +0.0712, t +3.76**

  **Tiering with a HARD out-set beats knowing exactly who played with NO
  tiering**, and layering soft availability on top of tiering makes it WORSE by
  +0.0712. So the composition leg's MINUTES WEIGHTING is materially
  miscalibrated, and that miscalibration costs more than perfect availability
  information is worth. A t of -12.69 at K=6 is far outside anything else in
  this register and is treated as a warning, not a trophy.

  **LABELLING ERROR, RECORDED: I launched that control describing it as "no
  availability weighting" and it is not that.** `SOFT_AVAIL=0` with neither
  `REPORT_OUTS` nor `INACTIVE_OUTS` yields the T2-HONEST HARD out-set, as the
  run log states. It remains the decisive shape-vs-availability test, but it is
  not the control I described when I started it.

  **THE HYPOTHESIS FOR ITS OWN PREREG, STATED NOW SO IT CANNOT DRIFT:** the leg
  weights players by trailing minutes-when-playing, which over-weights
  high-minute players relative to how a real 240-minute rotation distributes;
  a canonical starter/bench split corrects it. Predicted failure mode: 160/80 is
  a HAND-CHOSEN free parameter pair, so a prereg must either freeze it from an
  independent source (realised starter/bench minute shares on training folds
  only) or fit it walk-forward — and the t must face a wild cluster bootstrap,
  because every previous t of that size in this register has been an artefact.

  **NOTHING SHIPS ON THIS.** B_N3 was NOT the pre-registered primary arm (C_N1
  was), it emerged from a four-way construction sweep, and the prereg's stage-2
  clause names the primary arm only. Promoting it now would be exactly the
  second-look selection this project has rejected elsewhere. It is recorded as
  a HYPOTHESIS about the composition leg's minutes weighting — not about
  availability — and it needs its own pre-registration.

  **STOP CLAUSE HONOURED:** `COMP_ALLOC` unset is byte-identical to the
  pre-D242 default (control run: max|dp| **3.39e-15**, at the D230b
  reproducibility floor). No stage-2 full-stack run was performed. Nothing
  ships.

- D243 **ONE FROZEN REGULARISED TOTAL-CONDITIONED SCALE, PROSPECTIVE ONLY.**
  D241's unrestricted arm was `UNDERPOWERED`/inconclusive, not refuted: s1 > 0
  in 6/6 folds (the mechanism) but better in only 2/6 with a CI straddling zero
  and a coefficient decaying 0.236 -> 0.036. Those folds are now SPENT, so a
  shrinkage grid on them would be second-look optimisation. **`sigma_i =
  sigma_0 (T_i/Tbar)^gamma` with a ridge penalty lambda = 50 FIXED NOW** is
  frozen as shadow candidate C5 in `docs/SHADOW_2026_27.md`, with four arms
  including a within-season-and-spread-bucket placebo, and — the gap D241
  had — **a declared minimum practically-relevant effect of 0.0005 nats**,
  because "statistically nonzero" was never the right bar for a variance term.
  Queued BELOW the minutes and role-transition work.

- D244 **FAILURE LOG RESTRUCTURED: EVIDENCE STATUS IS NOW SEPARATE FROM SHIP
  VERDICT.** "NO SHIP" was overloaded across ~30 negative results. `docs/
  FAILURES.md` now carries an eight-value status (`VALID_REFUTATION`,
  `UNDERPOWERED`, `CONSTRUCTION_INVALID`, `INTEGRATION_INVALID`,
  `DATA_BLOCKED`, `FRESH_CONFIRMATION_PENDING`, `ACCEPTED_PROSPECTIVELY`,
  `SUPERSEDED`) plus per-entry estimand, information timestamp, populations,
  endpoint, MDE80, whether downstream layers were refitted, whether the holdout
  is spent, and the NARROWEST supported conclusion.

  **APPLYING IT RECLASSIFIED MY OWN ENTRIES, WHICH IS THE POINT:**
  * D237-D239 regime gates: logged as nulls, actually **UNDERPOWERED** — MDE80
    is 2.2x the pooled edge, so the corpus cannot address the hypothesis. That
    implies "change the endpoint", not "closed".
  * D230's ff/composition split: **UNDERPOWERED**, not refuted — r = 0.800
    means unidentifiable at this n, i.e. unmeasurable rather than measured.
  * D240: **CONSTRUCTION_INVALID**, and the entry now records that downstream
    layers were NOT refitted, a second reason the verdict is construction-bound.

  **TWO ENTRIES ADDED FROM THE REVIEW, BOTH VERIFIED AGAINST THE REGISTER
  BEFORE BEING WRITTEN:**
  * **D144 role transitions — CONSTRUCTION_INVALID.** The archive records
    `PROMOTED -2.2078 / DEMOTED +2.3497` and `ARM R | PROMOTED — PASS-WEAK and
    on the HARM side` against `DEMOTED — ERA-CONDITIONAL`. The intervention
    changed minutes but left per-minute RATE unchanged, and the POOLED veto was
    misaligned with a heterogeneous treatment whose halves have opposite sign
    and partially cancel. Retest alive with direction-stratified gates.
  * **D151 props channel ramp — CONSTRUCTION_INVALID for rebounds.** The
    coefficient was fitted against an ANALYTIC expectation while the scored
    system samples minutes empirically, clips and simulates. The archive
    measures the generator directly: **simulated 4.7760 vs realized 4.7777,
    bias -0.0017** — already unbiased, so the correction necessarily overshot.
    Rebounds CLOSED; assists FRESH_CONFIRMATION_PENDING, and any retest must fit
    against simulated output from the exact production generator under common
    random numbers.

- D242c **FORENSIC COMPLETION OF D242: PROTOCOL DEVIATION ON T1, A SILENTLY
  MISSING PRE-REGISTERED ARM, AND THE FULL FACTORIAL THAT REFUTES MY OWN
  "TIERING CARRIES EVERYTHING".** Prompted by external review; every claim below
  was verified in code or by a new run, and nothing here promotes anything.

  **(1) PROTOCOL DEVIATION — T1 WAS NOT EXECUTED AS REGISTERED. The hashed
  prereg is NOT edited; this entry supersedes the language.** `d242_prereg.md`
  calls the ORACLE arm "actual minutes played" and the first D242 entry said
  "perfect minutes knowledge". **Neither is what ran.**
  `ORACLE_PLAYED_OUTS` is documented in `prod_by_season.py` as "roster window
  minus who actually played" — it reveals APPEARANCE and then weights the
  survivors by their unchanged trailing minutes. So **-0.0827 measures perfect
  APPEARANCE information with imperfect minute weights**, and the phrase "68% of
  the ORACLE ceiling" in the previous entry is withdrawn: that artifact is not a
  minutes ceiling.

  **AND THE TRUE MINUTES ORACLE STILL CANNOT BE MEASURED AT THIS LAYER.**
  `ORACLE_MINUTES=1` does exist and does use realised minutes — but it rewrites
  the PROBABILITY path (`pp.append`), not `margin_components()`, so it never
  reaches the `m_comp` dump. Measured: `ORACLE_MIN` vs `HARD_RAW` **max|d m_comp|
  = 3.55e-15**, i.e. bit-identical at the D230b floor. T1 as written requires
  instrumentation that does not exist, and that is now a known gap rather than a
  reported result.

  **(2) A PRE-REGISTERED ARM WENT MISSING AND THE HARNESS ALLOWED IT.** The
  prereg names PLACEBO a HARNESS-VALIDITY gate. `d242_alloc_stage1.py` selected
  arms with `if file.exists()`, so the placebo silently vanished and the run
  reported as though the grid were complete — the same class as reading a result
  through `tail`. The script now ASSERTS a required set and fails loudly;
  `D242_ALLOW_PARTIAL` exists only for an explicitly partial diagnostic.

  **PLACEBO, NOW RUN** (p_out permuted within season, marginal preserved —
  verified mean 0.7573 both): **+0.0369, CI [-0.0000, +0.0739], better 1/6.**
  The harness does NOT manufacture improvement from noise; shuffled availability
  is WORSE than real, which is the correct direction. But it is not
  "indistinguishable from CONTROL" as T6 predicted, so **T6 is recorded as
  PARTIALLY REFUTED**: no leak, but real p_out carries measurable signal that
  shuffling destroys.

  **(3) THE COMPLETE 3x2 AVAILABILITY-BY-WEIGHTING FACTORIAL, AND IT REFUTES MY
  PREVIOUS CLAIM.**

      availability regime      raw weights    N3 weights     N3 effect
      soft opener-time           0 (ctrl)      -0.0562        -0.0562
      hard T2 (POST-OPEN)         -0.0465      -0.1275        -0.0810
      played-set oracle           -0.0827      -0.1490        -0.0663
      availability effect     -0.0465 then    -0.0713 then
      (soft->hard->oracle)        -0.0362        -0.0215

  **BOTH AXES MATTER AND THEY ARE ROUGHLY ADDITIVE.** N3 helps by 0.056-0.081
  under EVERY availability regime, and better availability helps by 0.036-0.071
  under EVERY weighting. **"Tiering carries everything" — my claim in the
  previous entry — is WRONG and is withdrawn.** The earlier B_N3-minus-hard-T2-N3
  comparison changed two axes at once and could not identify either.

  **(4) EVERY HARD-T2 ROW IS A POST-OPEN DIAGNOSTIC, NOT A DEPLOYABLE RESULT.**
  T2 unions the official pregame inactive list, released ~30 min before tip, so
  those arms use information a bettor does not hold at the open. They bound the
  mechanism; they cannot bound a shippable gain.

  **(5) INFERENCE, SEPARATED PROPERLY.** The exact sign test needs no
  distributional assumption: **6/6 gives one-sided p = 1/64 = 0.0156**, so the
  DIRECTION is robust without leaning on t = -12.69 at all. But B_N3 was the
  max of an 8-cell construction sweep (|t| 12.69 against next-largest 2.36), and
  **a single-arm wild bootstrap does not answer the selection question** — that
  needs a max-statistic permutation over the whole sweep, which is a separate
  test and is NOT yet run.

  **STATUS: `CONSTRUCTION_INVALID` -> the honest formulation is that N3
  discovered a large, season-stable benefit from RANK-CONSTRAINED MINUTE
  WEIGHTING, selected from a sweep, confounding several mechanisms
  simultaneously (rank ordering, a rank-5/6 discontinuity, fixed tier totals,
  separate within-tier normalisation, extreme-weight shrinkage, roster-size
  insensitivity), and evaluated on spent data. It is DEVELOPMENT EVIDENCE, not
  an identified effect and not a shipping result.** The successor is one
  tightly-specified role-share allocator whose single structural parameter is
  estimated from PRIOR REALISED MINUTE SHARES rather than chosen by hand, with
  the causal path verified from player minutes upward — not another sweep.

- D242d **THREE FINAL FORENSIC CORRECTIONS, AND THE SELECTION-ADJUSTED TEST.**

  **(1) T6 IS INCONCLUSIVE, NOT "PARTIALLY REFUTED" — my previous wording was
  wrong.** The placebo is +0.0369 with 95% CI **[-0.00002, +0.0739]**, which
  CONTAINS ZERO. One cannot conclude it differs from control; nor can one
  conclude equivalence, because no equivalence margin was pre-registered.
  Better in 1/6 means worse in 5/6, whose one-sided sign probability is
  **7/64 = 0.109**. So "real p_out carries measurable signal that shuffling
  destroys" is withdrawn as overstated. **The defensible statement: the placebo
  did NOT improve on control, so there is no evidence of harness-manufactured
  gain; its mostly-harmful direction is consistent with real availability being
  useful, but the placebo-control difference is statistically inconclusive.**
  Recorded separately: the gate itself was mis-specified. A shuffled-information
  placebo should be required NOT TO BEAT control, not required to be equivalent
  to a control that contains real information.

  **(2) THE GRID GUARD WAS STILL TOO NARROW, AND HAD A TRUTHINESS BUG.**
  `REQUIRED` listed only five arms, so `B_N3` — the arm that actually moved —
  could have gone missing again while the run called itself complete. It now
  requires the full pre-registered grid (six arms x four constructions, plus
  PLACEBO and ORACLE), with the four hard/oracle cells separated as
  `FORENSIC_DIAGNOSTICS`. And **`if not os.environ.get("D242_ALLOW_PARTIAL")`
  was a real defect: the string "0" is TRUTHY in Python, so setting the flag to
  0 would have ENABLED partial runs.** Now compares explicitly to `"1"`.
  Verified: the guard fires on an empty grid under both no flag and the "0"
  value.

  **(3) THE EXACT MAX-STATISTIC SIGN-FLIP TEST, NOW RUN.** All 2^6 = 64
  season-level sign vectors, applied to EVERY arm simultaneously so cross-arm
  dependence is preserved, statistic `max_a(-t_a)` over the eight searched
  B/C x N1-N4 cells:

      observed max -t   **12.69**  (B_N3)
      null max -t       median 1.01, 95th pct 3.91, maximum 12.69
      **EXACT selection-adjusted p = 1/64 = 0.0156**

  The observed value IS the null maximum, i.e. no sign vector produces a larger
  max statistic, so the adjusted p equals the unadjusted 1/64 here — the eight-
  arm search costs nothing because the other seven arms are nowhere near. **The
  full 64-row null distribution is persisted in `data/d242_maxstat.json`, not
  just the p-value.** This handles small K and the eight-arm selection together;
  it still does not make spent data confirmatory.

  **(4) MY AVAILABILITY RANGE WAS WRONG.** I wrote "availability helps by
  0.036-0.071", which covered only the raw-weight pair. Across all four
  transitions the range is **-0.0215 to -0.0713**. Over the complete
  soft->oracle transition: raw weights -0.0827, N3 weights -0.0928,
  **difference-in-differences only -0.0101**. So "roughly additive" is a fair
  DESCRIPTIVE summary of the endpoints but has NOT been inferentially tested,
  and is now written as: both axes improve every observed comparison; their
  interaction appears modest over the full transition but remains untested.

  **STRONGEST DEFENSIBLE CONCLUSION, superseding all earlier D242 language:**
  rank-constrained minute weighting improves composition-margin RMSE under soft
  opener-time, post-open hard, and oracle-appearance information; appearance
  information also improves results under both raw and N3 weighting; and N3's
  causal mechanism remains UNIDENTIFIED because the construction simultaneously
  changes tier shares, rank discontinuity, within-tier normalisation, extreme-
  weight shrinkage and roster-size sensitivity.

- D245 **ROLE-SHARE MINUTE ALLOCATION: ALL SIX PREDICTIONS CONFIRMED AT L1-L2,
  BUT THE MECHANISM IS NOT THE ONE THE HYPOTHESIS NAMED.** Prereg
  `data/d245_prereg.md` sha256 `1d145732c8d7c29f2349da451125f180d9c84dca32a65
  808633f1801fb3c1201`. One learned parameter, estimated from realised minute
  SHARES and never from margin or log loss, replacing D242's hand-chosen
  160/80. Causal path verified from player minutes upward. 16,345 team-games.

  **ALPHA IS REMARKABLY STABLE, AND THE HAND-CHOSEN SPLIT WAS WRONG.**
      per-fold alpha  0.6262 0.6249 0.6243 0.6254 0.6274 0.6269
      mean **0.6259**, spread **0.0031**, against 160/240 = 0.6667
  **P1 CONFIRMED** (|alpha - 2/3| = 0.0408 < 0.05) and **P2 CONFIRMED**. So the
  true top-five share is ~0.626 — i.e. **150/90, not the 160/80 I chose by hand
  in D242.** The construction D242 stumbled on is corroborated in KIND but its
  parameter was off by four points of share.

  **P6 CONFIRMED — COVERAGE IS NOT 1, AND THIS QUALIFIES D242.** Candidate
  players account for **94.41%** of actual minutes; the OTHER bucket is
  **5.59% = 13.49 minutes per team-game**, and **64.6% of team-games contain
  some non-candidate minutes.** So part of any rank-tiering gain is roster
  REGULARISATION rather than role allocation, exactly as flagged. Those minutes
  are bucketed explicitly, never dropped.

  **L1 (MINUTE-SHARE TOTAL VARIATION) — P3 CONFIRMED.**
      PRIMARY (alpha)   **-0.0043**  CI [-0.0058, -0.0028]  better **6/6**
      FIXED 160/80       -0.0043  CI [-0.0070, -0.0016]  better 6/6
      PLACEBO           **+0.1287**  CI [+0.1246, +0.1327]  better **0/6**
  **The placebo is a decisive validity check: shuffling ranks while preserving
  weights costs +0.1287, thirty times the size of the real effect.** The harness
  does not manufacture gains from structure alone.

  **L2 (STRENGTH vs S* = sum_i v_i * M_ACTUAL_i / 48, the TRUE minutes target
  D242's "oracle" never was) — P4 CONFIRMED.**
      PRIMARY  **-0.1917**  CI [-0.2391, -0.1442]  better **6/6**
      FIXED     -0.2331  CI [-0.2848, -0.1813]  better 6/6

  **P5 CONFIRMED, AND IT PUTS A NUMBER ON D242'S SELECTION BIAS: FIXED beats
  PRIMARY by -0.0414 at L2** despite using a share that is demonstrably wrong
  (0.667 against a measured 0.626). That gap is the advantage of having been
  chosen on these seasons, and it is the first direct estimate of selection bias
  this register has produced.

  **THE RANK-BAND DECOMPOSITION CONTRADICTS THE STATED MECHANISM.** Player-minute
  MAE by band:
      ranks 1-5   ctrl 6.077   primary 6.256   (**primary WORSE by 0.18**)
      ranks 6-8   ctrl 8.405   primary 8.719   (**primary WORSE by 0.31**)
      ranks 9+    ctrl 8.083   primary 7.269   (**primary BETTER by 0.81**)
  **The entire gain is in the DEEP BENCH, and the core rotation gets slightly
  worse.** The hypothesis was about correctly sharing minutes between starters
  and reserves; what the allocator actually does is stop over-crediting rank-9+
  players. Combined with P6's 5.59% coverage gap, the honest reading is that
  this is largely a REGULARISATION of noisy low-minute estimates, not a role
  discovery. That is still a real and useful effect — it improves both L1 and
  L2 in 6/6 seasons — but it is not what the prereg claimed.

  Fallbacks, all counted and emitted: 233 too-few-candidates, 160 over-48
  clips, 9 fewer-than-six-candidates.

  **L3/L4 NOT YET RUN.** The prereg requires the mechanism at L1-L2 before
  advancing, and that bar is met, so composition-margin and full-stack runs are
  earned. **Nothing ships regardless of their outcome:** the rank-tier
  architecture was learned on these seasons, so a favourable L4 freezes the
  construction for the 2026-27 shadow rather than promoting it.

- D245b **REPAIR GATE: EIGHT VERIFIED CONSTRUCTION DEFECTS FIXED, AND THE
  REPAIR REVERSES ONE OF D245's HEADLINE CLAIMS.** External review identified
  load-bearing implementation deviations; all eight were confirmed against
  source before repair, and the repaired run changes several conclusions.

  **THE WORST DEFECT: D245's CONTROL WAS NEVER THE PRODUCTION CONTROL.**
  `rolling(10, min_periods=3).mean().shift(1)` combined with a strictly-prior
  row filter DOUBLE-EXCLUDES the most recent prior game and requires four
  qualifying appearances where production requires one. Demonstrated on
  `[10,20,30,40]`: **production 25.0, D245 20.0**. Every downstream quantity —
  coverage, ranks, alpha, L1, L2 — rested on it. **Repaired and IDENTITY-TESTED
  against `CompositionModel(con, before=date)`: 400/400 exact matches on each of
  two sampled dates, zero mismatches.**

  **ALPHA MOVES, AND P1 IS NOW REFUTED.** D245 divided top-five actual minutes
  by CANDIDATE minutes, not total team minutes. Repaired:
      per-fold  0.5964 0.5937 0.5923 0.5947 0.5963 0.5954
      **mean 0.5948** (was 0.6259), spread 0.0041, vs 160/240 = 0.6667
  **The implied split is 143/97 — not the "150/90" D245 claimed, and further
  still from the hand-chosen 160/80.** |alpha - 2/3| = **0.0719 > 0.05**, so
  **P1 is REFUTED** on the repaired denominator, where D245 recorded it
  confirmed. P2 (stability) survives easily. Coverage 94.41% -> **95.34%**.

  **L1 WITH `OTHER` INSIDE THE SHARE VECTOR:**
      n1     -0.0000  CI [-0.0000, +0.0000]   <- see below, a CORRECTNESS CHECK
      prim   **-0.0020**  CI [-0.0034, -0.0006]  better **6/6**
      fix     -0.0023  CI [-0.0058, +0.0012]  better 5/6  **now ns**
      plac   **+0.1233**  CI [+0.1193, +0.1273]  better 0/6
  **N1 is EXACTLY zero at L1 by construction, not by null result:** proportional
  rescaling cannot change shares, so `share(240u/sum u) == share(u)`. Its
  appearance as -0.0000 confirms the harness. Note also that FIXED 160/80, which
  D245 reported as significant here, is **ns after repair**.

  **L2 AGAINST S* SPANNING ALL PARTICIPANTS:**
      n1     **-0.0781**  CI [-0.1213, -0.0349]  6/6
      prim   **-0.1808**  CI [-0.2317, -0.1300]  6/6
      fix     -0.2435  CI [-0.2957, -0.1914]  6/6
      plac   **+0.6514**  CI [+0.5554, +0.7475]  0/6
  **THE N1 DIAGNOSTIC EARNS ITS PLACE: 43% of the L2 gain (-0.0781 of -0.1808)
  is PURE MASS NORMALISATION**, i.e. merely forcing predicted minutes to total
  240. The incremental **TIER BENEFIT is -0.1027, CI [-0.1274, -0.0780], 6/6**.
  Without N1 the whole -0.1808 would have been attributed to rank tiering.

  **THE ADDITIVE DECOMPOSITION REVERSES D245's MECHANISM CLAIM.** D245 said "the
  entire gain is in the deep bench", from an MAE-by-band diagnostic that does
  not decompose anything. The genuinely additive TV contributions, which sum
  exactly to the total:
      prim   ranks 1-5 **-0.00502**   ranks 6-8 **+0.00424**   ranks 9+ -0.00122
             sum -0.00200 == total -0.00200
  **The gain is concentrated in the TOP FIVE and is partly given back at ranks
  6-8; the deep bench contributes least.** That is the opposite of what D245
  asserted, and it was an artefact of averaging MAE within bands. Claim
  withdrawn.

  **P5 REMAINS PENDING, NOT CONFIRMED.** The prereg scores it at L3; D245
  declared it confirmed from an L2 difference. Protocol mismatch, corrected. And
  the -0.0414 gap was never an identified selection-bias estimate: alpha is
  fitted to mean minute SHARE while L2 scores talent-weighted STRENGTH error, so
  the comparison conflates selection, objective mismatch, talent correlation and
  coverage. An identified estimate needs nested selection or the fresh 2026-27
  difference. Repaired gap (fix minus prim at L2) is -0.0627.

  **A COVERAGE FACT WORTH RECORDING: `p_out` is absent for ~81% of candidate
  player-games** (artifact has 87,218 rows against ~196,000 lookups). This is
  FAITHFUL to production — `outs[t]` only carries players present in the map and
  the rest weigh 1.0 — but it means opener-time soft availability touches under
  a fifth of candidates, which bounds how much any availability refinement can
  move this leg.

  **VERDICT ON THE REVIEWER'S THREE-BRANCH RULE: the third branch.** The
  repaired arm beats CONTROL and beats N1 at both L1 and L2, 6/6 each, with a
  placebo that is catastrophically worse. That is genuine evidence for
  rank-constrained regularisation — **and per the same rule it is named a
  RANK-CONSTRAINED MINUTE REGULARISER, not a role-share model**, since nothing
  here demonstrates role discovery. L3/L4 remain NOT RUN pending the owner's
  instruction.

- D245c **VALIDATION PASS: ALL FOUR CLARIFICATIONS RESOLVE, AND THE COVERAGE
  STRATIFICATION REFUTES THE REGULARISATION READING — IN THE OPPOSITE
  DIRECTION.**

  **IDENTITY TEST, NOW PERSISTED** (`data/d245c_validation.json`, four dates
  rather than the two quoted in prose):
      2022-12-01  2,322 comparisons  0 mismatches  max|d| 1.42e-14
      2023-01-15  2,334              0             1.42e-14
      2024-03-01  2,434              0             1.42e-14
      2025-02-10  2,522              0             1.42e-14
  The reconstructed trailing-minute table is the production table to float
  noise, across 9,612 player-date comparisons.

  **(1) THE CAP CONTRIBUTES NOTHING; THE "43% IS MASS NORMALISATION" STATEMENT
  STANDS.** D245b's single shared `over_48_iter` counter could not attribute the
  clip. Arm-specific counters show it binds rarely (n1 17 games, prim 46, fix
  12, plac 46 of ~14,000), and **N1(capped) minus N1(uncapped) at L2 is
  -0.000027, CI [-0.000070, +0.000016]** — indistinguishable from zero. So N1's
  -0.0781 really is total-minute normalisation.

  **(3) THE TIER BENEFIT IS NOT AN OVERTIME ARTEFACT.** Raw S* uses realised
  minutes, which exceed 240 in overtime while every allocator predicts exactly
  240. Scored against both:
      raw S*            prim-ctrl -0.1808   **prim-n1 -0.1027**  6/6
      OT-normalised S*  prim-ctrl -0.1846   **prim-n1 -0.1036**  6/6
  Essentially unchanged. Game-length variation is not carrying the result.

  **(4) COVERAGE STRATIFICATION — THE DIAGNOSTIC D245b PROMISED AND DID NOT
  IMPLEMENT, AND IT OVERTURNS MY OWN SPECULATION.**
      OTHER == 0  (n=5,669)  tier benefit dTV **-0.00959**  dL2 **-0.1989**
      OTHER >  0  (n=8,610)  tier benefit dTV **+0.00304**  dL2  -0.0387
      slope on OTHER share:  L1 **+0.06556**   L2 **+0.51830**  (both POSITIVE)

  **The benefit is STRONGEST where the candidate pool is COMPLETE and turns
  HARMFUL at L1 where it is not.** On stable rosters the L1 tier benefit is
  -0.00959, nearly FIVE TIMES the pooled -0.0020; on rosters with non-candidate
  minutes it is +0.00304, i.e. the allocator makes minute shares WORSE. The
  pooled figure was averaging a strong effect and a harmful one.

  **THIS KILLS THE ROSTER-REGULARISATION READING.** D245 speculated — and I
  wrote into the register — that the gain was regularisation arising from the
  12-day window's incompleteness. The opposite is true: incompleteness DESTROYS
  the gain. Of the reviewer's three interpretations this is the third: **the
  allocator is a genuine rank-weight correction that BREAKS on roster churn**
  (returns after >12 days, trades, new acquisitions).

  **DIRECTLY ACTIONABLE, AND IT MUST BE PRE-REGISTERED NOT PATCHED IN:** any
  shipped allocator needs an explicit guard when the candidate pool does not
  cover the roster — fall back to the control weighting rather than allocate
  240 minutes across a pool known to be missing real players. Choosing that
  threshold on these seasons would be selection; it belongs in the 2026-27
  freeze.

  **LANGUAGE CORRECTIONS ADOPTED.** `pout_missing` -> `pout_default_zero`
  (absence from the artifact is a structural default under production
  semantics, not a feed failure). The sparse-support claim is narrowed: the
  artifact carries modelled probabilities for ~19% of candidate player-games,
  which bounds PROBABILITY RECALIBRATION ON ITS CURRENT SUPPORT — it does not
  bound models that predict unreported absences, use timestamped transitions, or
  widen coverage with more report editions. And the placebo is recorded as a
  SANITY CHECK: it shuffles u while talents and outcomes stay with original
  identities, so catastrophic failure is expected; validity rests instead on the
  production identity, N1's exactly-zero L1 delta, the additive decomposition
  summing to total, and six-season consistency.

  **L3/L4 ARE NOW UNBLOCKED** by the reviewer's own condition (validation pass
  complete, no redesign, no tuning). The decisive historical comparison at both
  layers is **PRIMARY vs N1**, not PRIMARY vs CONTROL, or normalisation collects
  credit as tier alpha.

- D245d **L3/L4: THE MINUTES IMPROVEMENT IS REAL AND SURVIVES TO THE
  COMPOSITION MARGIN, THEN DIES AT THE FULL STACK. NOTHING SHIPS.** Prereg
  sha256 `1d145732...`. Run only after the D245c validation pass cleared all
  four conditions. Decisive comparison at both layers is PRIMARY vs N1, not
  PRIMARY vs CONTROL, so total-minute normalisation cannot collect credit as
  tier alpha.

  **L3 — COMPOSITION MARGIN, each arm affine-recalibrated on training folds:**
      PRIMARY vs CONTROL   **-0.04406**  CI [-0.06695, -0.02117]  6/6  BETTER
      N1 vs CONTROL         +0.00040  CI [-0.02261, +0.02341]  4/6  **ns**
      **PRIMARY vs N1**    **-0.04446**  CI [-0.07143, -0.01749]  6/6  BETTER
      FIXED vs PRIMARY      -0.01214  CI [-0.03241, +0.00813]  5/6  ns
  **At L3 the tier constraint carries the entire effect and 240-normalisation
  contributes NOTHING** — a different split from L2, where N1 supplied 43%. So
  normalisation helps an internal strength estimate but not margin prediction;
  only the rank constraint reaches the margin.

  **L4 — FULL STACK, with the blend weight, offset ridge AND probability link
  refitted per arm per fold (D235's rule: an arm whose upstream representation
  changed cannot be judged against a downstream fitted on the old one):**
      PRIMARY vs CONTROL   -0.00032  CI [-0.00065, +0.00002]  t -2.43  6/6  **ns**
      N1 vs CONTROL        +0.00007  CI [-0.00020, +0.00034]  3/6  ns
      **PRIMARY vs N1**    **-0.00038**  CI [-0.00083, +0.00006]  t -2.22  5/6  **ns**
      FIXED vs PRIMARY     -0.00018  CI [-0.00039, +0.00002]  5/6  ns
  **EVERY L4 COMPARISON IS NULL.** Point estimates are favourable and PRIMARY
  vs CONTROL is better in 6/6, but both CIs contain zero. **The tier increment
  at the layer production actually emits is -0.00038 nats, ns — and that is the
  number that matters.**

  **THE ATTENUATION IS ARITHMETIC, NOT MYSTERIOUS.** The composition channel is
  0.652 of the fitted blend and the offset spends the blind margin at 0.3413,
  so a -0.04406-point composition gain becomes
  `-0.04406 x 0.652 x 0.3413 = -0.00981` points on the emitted margin — about
  0.0014 in logit units at scale 7.2. **A ~14x two-stage attenuation before the
  forecast is made.** This is the second arrow the reviewer warned could break:
  L1/L2/L3 all improved significantly and L4 still did not.

  **TWO SECONDARY FINDINGS.** The fitted four-factor blend share FALLS as the
  composition channel improves (0.382 ctrl -> 0.371 n1 -> 0.348 primary ->
  0.335 fixed): the stack reallocates weight toward the better channel, which is
  the refit doing its job. And the offset edge coefficient RISES (0.3236 ->
  0.3413 -> 0.3623), the same "a better blind margin earns more trust" pattern
  D235 found — which is why refitting downstream per arm was necessary rather
  than pedantic.

  **STATUS: `VALID_REFUTATION` AT L4, `CONFIRMED` AT L1-L3.** The construction
  genuinely improves player-minute allocation (L1), talent-weighted lineup
  strength (L2) and composition-margin RMSE (L3), each 6/6 with CIs excluding
  zero. It does not measurably improve the opening-line forecast. Nothing is
  promoted; `nbapred/` production defaults are untouched and `COMP_ALLOC`
  remains unset by default.

  **WHAT THIS DOES NOT SAY.** It does not say the minutes pathway is worthless —
  L1-L3 are unambiguous, and the same improved minutes feed the props engine,
  where there is no offset layer to attenuate them by 14x. The right reading is
  that a composition-channel improvement of this size cannot survive the blend
  and the offset, so future minutes work should be evaluated on props first, or
  be large enough to clear a 14x haircut.

- D246 **THE MOVEMENT TEACHER: THE MODEL CAN ANTICIPATE THE MARKET BUT CANNOT
  BEAT IT. NO SHIP, AND THE DECOMPOSITION IS THE RESULT.** Prereg
  `data/d246_prereg.md` sha256 `661404b64a32ba8d82a292748debbb02ab7a68c8496a31
  c5766a65682b48531e`. Chosen ahead of the other queued items because D245d
  measured a **~14x two-stage attenuation** on anything entering through the
  blend (share 0.652 x offset edge 0.3413); a market-quantity target bypasses it.
  The target is also **5.9x less noisy** than the outcome residual (sd 2.303
  against 13.60).

      Y - O = (C - O) + (Y - C)
              head M     head F

  Both heads use SEVEN opener-time features only. **LEAKAGE CHECK PASSED: the
  feature matrix is bit-identical (max|d| 0.0e+00) under a within-season
  permutation of the closing line**, so the close is a training target and
  provably never an input.

      season   r2_M     r2_F     ll_offset  ll_teacher   d_ll
      2020-21  0.1351  -0.0109    0.62656    0.62754   +0.00098
      2021-22  0.1424  -0.0005    0.61534    0.61366   -0.00168
      2022-23  0.1227  -0.0035    0.63041    0.63187   +0.00145
      2023-24  0.0878  -0.0002    0.59107    0.58942   -0.00165
      2024-25  0.2062  +0.0044    0.59473    0.59327   -0.00146
      2025-26  0.0873  +0.0015    0.57862    0.57720   -0.00142

  **T1 CONFIRMED AND BEAT MY OWN PREDICTION: head M reaches +0.1302 mean OOS
  R^2.** I predicted 3-10% from seven features against D147's 17.1% with a
  richer live set; it delivered 13.0%. Open-to-close movement is substantially
  predictable from opener-time information alone.

  **T2 CONFIRMED, AND IT IS THE REAL FINDING: head F reaches -0.0015 OOS R^2 —
  ZERO.** The same features that explain 13% of what the market will do explain
  NOTHING of what the market gets wrong. **The model can anticipate the close
  and cannot beat it.** That is the sharpest statement this register has of
  where the edge lives, and it independently corroborates the whole offset
  architecture: a correction TO the line is the right shape precisely because
  there is no measurable signal in the post-close residual.

  **T3 NO SHIP AS PREDICTED.** Full-stack vs the shipped offset: **-0.000630,
  CI [-0.002141, +0.000881], t -1.07, better 4/6.** Favourable point estimate,
  CI contains zero. The prereg predicted exactly this (< 0.001 nats, CI probably
  containing zero), so it is a scored prediction rather than a disappointment.

  **T4 heads are distinct** (corr of head M with head F's residual +0.198), so
  the two-head split is not one signal counted twice. Fitted gamma 1.007 and eta
  1.004 — essentially no shrinkage, i.e. the combination is close to
  `O + predicted movement + predicted residual`.

  **WHAT THIS REDIRECTS.** Items still queued that act on the composition or
  four-factor channels (probabilistic candidate/return model, box-score
  replacement allocator, matchup interactions, joint role transitions) all
  inherit D245d's 14x haircut for SIDES. D246 says the other route — predicting
  market movement — is real but already largely captured by the offset. **The
  honest implication is that neither route is likely to move the sides forecast
  materially on this corpus, and the remaining upside is in props (no offset
  attenuation) and in the timestamped 2026-27 channel (genuinely new
  information).**

- D246b **CLEAN COMPLETION: MY D246 HEADLINE WAS AN ALGEBRAIC IDENTITY, AND THE
  PRE-REGISTERED ENDPOINT I NEVER COMPUTED IS THE ONE THAT PASSES.**

  **THE IDENTITY.** Ridge is linear in the target, so
  `beta_M + beta_F = R_lam[(C-O)+(Y-C)] = R_lam(Y-O)`. **Verified numerically:
  max|(beta_M+beta_F) - beta_direct| = 2.7e-15.** With gamma ~ eta ~ 1 the D246
  "teacher" was algebraically a plain ridge on (Y-O), the closing-line
  decomposition CANCELLED, and the fitted gamma 1.007 / eta 1.004 were that
  identity asserting itself — not two distilled signals. **D246's T3 therefore
  never tested whether learning close movement improves the offset.** With
  CROSS-FITTED meta-weights the identity breaks and eta collapses toward zero
  (gamma 1.69-1.98, eta -0.06 to +0.11), which is the honest picture.

  **THE TEST D246 NEVER RAN — M-ONLY vs THE SHIPPED OFFSET:**
      M-ONLY vs OFFSET        **-0.000788**  CI [-0.002507, +0.000932]  4/6  ns
      M+F vs M-ONLY            +0.000070  CI [-0.002746, +0.002887]  4/6  ns
      DIRECT ridge(Y-O)        -0.000245  CI [-0.002249, +0.001759]  4/6  ns
  Log loss remains null. **But that is not the endpoint this head targets.**

  **T5 — PRE-REGISTERED IN D246 AND NEVER COMPUTED — PASSES.**
      offset mean signed CLV   **+0.5251 pts**
      M-only mean signed CLV   **+0.6367 pts**
      difference **+0.1117, CI [+0.0528, +0.1706], BETTER IN 6/6 SEASONS**
  **The CI excludes zero.** This is the "M-only improves CLV but not log loss"
  branch: a useful EXECUTION AND TIMING signal, not a superior outcome forecast.
  It is also the first significant positive result in this line of work, and it
  was invisible because the script omitted the endpoint its own prereg named.

  **"THE MODEL CANNOT BEAT THE MARKET" IS WITHDRAWN — HEAD F WAS UNDERPOWERED,
  NOT REFUTED.** With standardised ridge (D246 penalised raw columns spanning
  points to hundredths, so the small-scale features were shrunk far harder),
  head F still gives mean r2 -0.00278. But the paired dMSE against predicting
  zero is **+0.4923 with CI [-1.0196, +2.0041] and MDE80 2.0526** — the design
  cannot resolve anything below dMSE ~2.05, and the observed effect is a quarter
  of that. **The correct statement is "no point-estimate improvement under this
  specification", not "the market cannot be beaten".**

  **"ALREADY CAPTURED BY THE OFFSET" IS REFUTED BY ABLATION.**
      full seven features      r2 **+0.1299**
      edge only                r2 **+0.0669**
      everything BUT edge      r2 **+0.0556**
  Edge alone supplies about half, and the other six opener-time features supply
  nearly as much again, roughly additively (0.0669 + 0.0556 = 0.1225 vs 0.1299).
  **Movement prediction is NOT a relabelling of the offset's own input** — rest,
  availability, totals, season timing and market reliability carry genuinely
  incremental movement information.

  **AND THE 14x HAIRCUT IS LOCAL, NOT UNIVERSAL — my queue-wide claim was too
  broad.** D245d measured 0.652 x 0.3413 for one perturbation entering through
  the existing composition channel. A feature that enters the offset directly,
  predicts movement, is orthogonal to the blind margin, or concentrates in
  high-impact games does not inherit it. Items 2, 3, 5 and 6 are lower priority
  for historical sides, but they do not mathematically inherit an identical
  attenuation, and D246's closing paragraph overstated that.

  **STATUS: `FRESH_CONFIRMATION_PENDING` on CLV.** M-ONLY is a genuine
  CLV/timing candidate with a CI excluding zero across 6/6 seasons, discovered
  on spent data. It ships nothing; it becomes shadow candidate C7 for 2026-27,
  where D238's power work says CLV needs ~131-267 bets to confirm.

- D246c **THE +0.1117 WAS LINE-DIRECTION FORECASTING, NOT TIMING — AND THE
  PROPER TIMING TEST PASSES ANYWAY.**

  **THE DEFECT.** D246b compared `sign(shipped_margin - open)` against
  `sign(predicted movement)`. Those are DIFFERENT SIDE-SELECTION RULES, and M is
  trained on `close - open`, so positive signed movement is exactly what it must
  produce. Measured directly: **side-agreement rate 0.743, and on agreement
  games the two rules are IDENTICAL BY CONSTRUCTION** (baseline CLV +0.7923 on
  agreement, -0.2124 on disagreement). **The entire D246b gap came from flipping
  the side**, so it established line-direction prediction, not execution
  improvement. Claim corrected.

  **THE REAL TIMING TEST — BASELINE SIDE FROZEN, never flipped:**
      s = sign(shipped_margin - open)      z = s * predicted(close - open)

      season   n(z>0)  CLV(z>0)   n(z<=0)  CLV(z<=0)
      2020-21     670   +1.0351      405    -0.2062
      2021-22     894   +0.8238      330    -0.3758
      2022-23     915   +0.7115      277    -0.1841
      2023-24     954   +0.4751      276    -0.2129
      2024-25     942   +1.1778      288    -0.1684
      2025-26     978   +0.5307      252    -0.1270

      **CLV(z>0) - CLV(z<=0) = +1.0047, CI [+0.6923, +1.3172], 6/6 SEASONS**

  **Holding the wager side fixed, movement alignment separates good execution
  from bad by a full point of CLV, every season.** Games where the line is
  predicted to move AGAINST a waiter carry +0.79 CLV; games where it is
  predicted to move TOWARD the baseline side carry -0.21. **Retention is 74.5%**
  of games, so this is not a thin-slice rule.

  **THE ABLATION, NOW INFERENTIAL RATHER THAN DESCRIPTIVE.** R^2 values do not
  add, so D246b's `0.0669 + 0.0556 ~ 0.1299` demonstrated nothing. Paired
  out-of-sample squared errors:
      full 7 vs EDGE-ONLY   dMSE **-0.3437  CI [-0.4708, -0.2166]  6/6  SIG**
      full 7 vs NO-EDGE     dMSE  -0.4924  CI [-1.0490, +0.0643]  6/6  ns
  **The six non-edge features add SIGNIFICANTLY beyond the offset's own edge.**
  So the movement head is genuinely incremental to the offset, confirmed
  inferentially rather than by comparing R^2 values.

  **C7 REFRAMED BEFORE ANY 2026-27 OBSERVATION.** Its historical evidence came
  from `sign(pM)` as a side rule, which is NOT the policy being frozen. The
  frozen policy is now explicit: the baseline side is the frozen production
  side, C7 may NEVER flip it, alignment `z = baseline_side x predicted movement`
  uses a threshold of exactly zero, and `z > 0` executes now while `z <= 0`
  waits. Primary prospective endpoint is incremental signed CLV ON THE FROZEN
  BASELINE SIDE, matched on |edge| decile, with retention reported.

  **REMAINING KNOWN DEFECT, RECORDED NOT HIDDEN:** D246b's M+F arm fitted its
  probability scale on the DIRECT training forecast rather than the
  corresponding M+F one, and the earliest outer fold falls back to
  (gamma, eta) = (1, 1) because it has too few inner seasons to cross-fit. Both
  affect the M+F comparison and T4 only — not the M-only CLV or timing results —
  and M+F was already null.

- D246d **THE +1.0047 IS GROUP SEPARATION ON THE FULL SLATE, NOT EXECUTION
  ALPHA. ON THE ACTUAL BET LEDGER THE POLICY DEFERS ONLY 12% OF BETS AND THE
  ILLUSTRATIVE BENEFIT IS ~0.03 POINTS.**

  **THREE DEFECTS IN D246c, ALL CONFIRMED.** (a) It scored ALL 7,181 games
  without the production betting threshold, while the strategy takes ~12% of the
  slate. (b) `z<=0 -> wait` had NO DELAYED FILL — the script compared the
  open-to-close CLV of two GROUPS, it never executed anything later. (c) "74.5%
  retention" was the wrong word: it is the IMMEDIATE-EXECUTION rate. If `z<=0`
  never executes at all then C7 is a bet SELECTOR, not a timing strategy.

  **RERUN ON THE ACTUAL PRODUCTION BET LEDGER (769 bets, not 7,181 games):**
      season  n_bets  exec_rate  CLV(z>0)  CLV(z<=0)
      2020-21     94     0.553     +0.962    -0.345
      2021-22    112     0.946     +1.269    -1.250
      2022-23    103     0.932     +1.255    -0.571
      2023-24    154     0.935     +0.993    +0.100
      2024-25    210     0.933     +2.094    -0.357
      2025-26     96     0.969     +0.548    +1.000

      separation **+1.4242, CI [+0.2528, +2.5957], 5/6** (full slate: +1.0047,
      CI [+0.6923, +1.3172], 6/6)

  **The point estimate is larger on the ledger but the CI is FOUR TIMES WIDER
  and it is 5/6 not 6/6**, because n falls from 7,181 to 769. Two seasons show
  POSITIVE CLV(z<=0) on 3-10 deferred bets — the split does not survive small
  samples, which is exactly what the ledger exposes and the slate hid.

  **AND THE IMMEDIATE-EXECUTION RATE IS 87.8% ON THE LEDGER, NOT 74.5%.** Only
  **12.2%** of actual bets would be deferred. So the illustrative slate-level
  benefit, IF a deferred bet fills at the close, is
  `0.122 x 0.237 = 0.0289 points of CLV per bet` — not +1.42, and smaller even
  than the ~0.054 the full-slate numbers implied.

  **CORRECTED HEADLINE, superseding D246c's:** *side-frozen movement
  stratification is confirmed — the model separates favourable from unfavourable
  open-to-close movement while holding the wager side fixed, across six seasons.
  That is NECESSARY evidence for C7 and is NOT realised timing alpha, because
  the historical test contains no delayed executable quotes and the deferred
  fraction of real bets is only 12%.*

  **CALIBRATION DRIFT, FOR PROSPECTIVE MONITORING NOT RETROSPECTIVE REPAIR.**
  The zero-threshold action rate on the full slate drifts 62.3% -> 73.0% ->
  76.8% -> 77.6% -> 76.6% -> **79.5%**. A rule whose action rate moves 17 points
  across the sample needs its action rate watched live; the threshold stays at
  exactly zero and is NOT re-fitted.

- D247 **BETWEEN-SEASON ADAPTATION: THE ALTITUDE PREMISE IS INVERTED, 2024-25 IS
  NOT AN ANOMALY, AND THE ONE APPARENT OUTLIER IS AN ODDS-FEED ARTEFACT.**

  Sean's hypothesis, three limbs: (a) after DEN won 2022-23, opponents adopted
  altitude prep and DEN's edge collapsed; (b) teams retool to counter whoever
  beat them or won the league; (c) teams fix last year's problems and the market
  underprices the change — which is why 2024-25 was our best betting season.

  Prereg first: a 19-season x 30-team panel with a story I already found
  plausible is the exact configuration D239 measured as manufacturing +2.54 ROI
  points from nothing. Arms fixed before any was read; champion/finalist lists
  are public record; the outcome is the SAME for every arm and is always a
  MARKET RESIDUAL vs the CLOSE, never raw performance. A team can collapse
  entirely and still score 0 if the market saw it coming — that refutes the
  hypothesis rather than confirming it. [scripts/d247_adaptation.py, d247b, d247c, d247d]

  **(a) ALTITUDE — REFUTED, AND THE SIGN RUNS THE OTHER WAY.** DEN home, 19
  seasons, raw = home margin minus league-average home margin, res = vs CLOSE:

      DEN home        pre-title      title-onward     delta
      raw              +2.951          +5.694        **+2.743**
      market residual  +0.336          +0.922        **+0.586**

      pooled DEN home residual +0.459 CI(-0.512,+1.431) k=19 **ns**

  Denver's home edge ROSE after the championship on both measures. The archive
  series that looked like a post-title collapse (+8.67 -> +5.23 -> +0.21 ->
  -3.50) is the STRENGTH-CONTROLLED deviation d_t — a different quantity from
  the bettable one, and D96/D70 already closed altitude on two constructions.
  Altitude is also REDISTRIBUTIVE: E[altitude gain] over all home games = 0.22 m.
  UTA is the instructive mirror — raw home collapsed to -10.49 in 2024-25, but
  the market residual was only -1.95: **the market priced the collapse.**

  **(b) CHAMPION / FINALIST ADAPTATION — UNDERPOWERED, NOT REFUTED.**
      defending champion vs CLOSE  -0.5321 CI(-1.1882,+0.1241) 13/19 negative
      beaten finalist  vs CLOSE    -0.4191 CI(-1.0776,+0.2394) 11/19 negative
      MDE80 **0.875 pts/game** — 1.6x the point estimate.
  Both are directionally consistent with the hypothesis and neither resolves at
  k=19. Status UNDERPOWERED (taxonomy in FAILURES.md), not VALID_REFUTATION.

  **(c) "MARKET ANCHORS ON LAST SEASON" — REAL, CORRECTLY SIZED, UNBETTABLE.**
  Slope of market residual on prior-season margin: **-0.0232 at game level,
  -0.0241 at team-season** — consistent across aggregation levels. Realised
  total **+0.1306 pts/game CI(-0.0219,+0.2831) ns**, matching -0.0232 x E|dprev|
  5.589 = 0.130 almost exactly. Spent as a bet (back last season's worse team):

      close, all games       **49.06%**  CI(48.19,49.93)   -3.32pp vs break-even
      close, top tercile       49.81%    CI(48.63,50.98)   -2.57pp
      open,  all games         48.78%    CI(47.87,49.68)   -3.60pp

  Note the shape: mean residual is **+0.1306 yet cover is BELOW 50%**. The
  effect exists in expectation and is absent at the margin, which is where the
  bet settles. Two different functionals; only the second one pays.

  **T6 RETRACTED — THE -0.1608 WAS AN AGGREGATION ARTEFACT.** Controlling for
  the team's current-season market price, team-season prev_act gave -0.16080
  (t -9.46), implying a +-1.6 pt/game edge from two public numbers. VIF was only
  2.0 so it was suppression, not collinearity — but at GAME level the same
  contrast is **-0.00507 (t -0.24) ns**. Collapsing 82 games into one point per
  team, with corr(prev_act, mkt) = +0.705, produced a partial coefficient with
  no game-level counterpart. The bet losing 49.06% is the same fact measured
  in money.

  **WITHIN-SEASON DECAY — REFUTED, and this one is decisive on its own.** An
  unpriced offseason change must be largest in October and shrink by January:

      first 2 wks  n=1,932  market |err| 9.907  open->close 1.233
      wks 3-4      n=2,268               9.412               1.233
      month 2      n=4,036               9.711               1.241
      months 3-4   n=7,881               9.805               1.218
      months 5+    n=6,623               9.565               1.330

  Flat. The market is no worse in the first two weeks than in April, so the
  "hasn't caught up to the offseason yet" mechanism is dead independent of
  (a), (b) and (c).

  **THERE WAS NO 2024-25 ANOMALY TO EXPLAIN.** +24.00% vs the 7-season pooled
  +9.11% is +2.29 SE; as the MAX of 7 seasons **p 0.144**. On the full
  1,230-game slate — 6x the bet ledger and not conditioned on the selector —
  2024-25's normalised gap ranks **7/19 (z -0.50)**. The market was not soft.

  **AND THE ONE REAL OUTLIER IS THE ODDS FEED, NOT THE MARKET.** 2024-25's CLV
  ranked 1/19 at z +2.80. But **corr(season CLV, season mean |close-open|) =
  +0.917** — CLV is mechanically proportional to how far the RECORDED line
  travels. 2024-25 has the largest movement of 19 seasons (1.907), the 3rd-most
  UNCHANGED lines (22.3%) and the 2nd-widest opener dispersion (7.713). More
  never-moving lines AND bigger moves when they move AND a wider opener spread
  is a RECORDING-SOURCE signature, and REVIEW.md already documents the cause:
  books/game at open falls 7.74 (2023-24) -> 1.00 (2024-25). Feed eras:

      <=2023-24   CLV 0.2464   move 1.1993   capture 0.1914
      >=2024-25   CLV 0.6293   move 1.6988   capture 0.3590

  Rerun on CAPTURE FRACTION (CLV / mean|close-open|), which is invariant to the
  recorded scale, and corrected for 2024-25 having been SELECTED as the best:

      measure                      2024-25  rank      z   p(max19)  p(x4meas)
      CLV (raw)                     0.8626  1/19  +2.80    0.0938     0.3256
      capture fraction              0.4523  1/19  +2.42    0.2581     0.6971
      direction hit | line moved    0.6266  1/19  +1.98    0.6041     0.9754

  **Nothing survives selection correction.** D246d's 2024-25 CLV(z>0) of +2.094
  — the largest of the six seasons — is the same artefact seen through the
  timing split, and should be read as feed scale rather than model skill.

  **NET.** Limb (a) refuted with the sign inverted; limb (b) underpowered;
  limb (c) real at 0.13 pts/game and unbettable; the mechanism that would make
  any of them exploitable (early-season mispricing) refuted flat; and the
  premise — that 2024-25 needs explaining — does not survive its own
  selection correction. **No production change.** The durable finding is
  methodological: **CLV is not comparable across seasons whose odds feeds
  differ, and every cross-season CLV claim in this repo must be read on capture
  fraction instead.** That applies retroactively to D246c/D246d and prospectively
  to the C7 shadow test, where the October capture spec fixes the feed.

- D248 **THE ODDS FEED HAS THREE RECORDING REGIMES, NOT ONE. THE 2024-25 CLV
  RESULT WAS A LINE-GRID ARTEFACT, AND A CONVENTION-CONSISTENT FEED NOW EXISTS
  FOR 2007-08..2024-25.**

  D247 traced 2024-25's rank-1 CLV to recorded line travel (corr +0.917) but
  stopped at "the feed changed". This asks where else it changed, why, and
  whether it can be repaired from held data.
  [scripts/d248_feed_audit.py, build_odds_unified.py, d248b_rescore.py]

  **THREE REGIMES.** `docs/OPENING_LINES.md` documents one splice; there are two.

      2007-08..2022-23   SBR composite        ~50% integer, ~10% on 3/7
      2023-24            ESPN, 15-book mean   43.5% integer, 2.85% QUARTER-points
      2024-25..2025-26   ESPN BET alone       0.7% integer, 99.3% half-points

  **CHANGEPOINTS** (best single split; p from 20,000 season-order shuffles):

      fingerprint          split                pre     post    p
      pct on key numbers   2023-24|2024-25     9.739   0.285   0.0049 ***
      pct half-point       2023-24|2024-25    47.395  98.902   0.0083 ***
      pct integer          2023-24|2024-25    52.428   1.098   0.0091 ***
      sd(open)             2023-24|2024-25     6.313   7.864   0.0112 *
      n distinct opens     2023-24|2024-25    64.353  52.500   0.0240 *
      mean |close-open|    2013-14|2014-15     0.945   1.431   0.0147 *

  Granularity was chosen as the discriminator BEFORE reading it, because a
  market can change how far lines move but cannot change the rate of half-point
  use. A changepoint there is necessarily a recording change.

  **ROOT CAUSE, VERIFIED IN THE RAW FEED — NOT OUR BUILDER.** ESPN BET posts
  **100.0% half-point spreads and 0.00% key numbers in every season it appears**
  (n=1178 / 1204 / 291). It is that book's house convention: half-points make
  spread pushes impossible. In 2024-25 it was the ONLY book carrying opens
  (ESPN 2 providers; Action Network carries no opens at all), so the frame
  inherited the convention wholesale.

  **THE MECHANISM.** A half-points-only feed has a line grid spaced **1.0 apart,
  not 0.5**, so the smallest observable move DOUBLES. That produces 2024-25's
  otherwise contradictory signature — simultaneously the MOST unchanged lines
  (22.3%) and the BIGGEST moves when they move (1.907) — and inflates CLV
  proportionally, since CLV is measured in points of line travel.

  **2023-24 IS DISTORTED THE OPPOSITE WAY.** Averaging 15 books produced 2.85%
  quarter-point values, 89 distinct opens (the most of any season), and SMOOTHED
  MOVEMENT AWAY. Two adjacent seasons, two opposite distortions, both pushing
  toward the same false conclusion.

  **THE REPAIR.** TeamRankings book1 holds the historical convention and, unlike
  every other source, OVERLAPS BOTH ERAS (2021-22..2025-26):

      TR book1 integer share  52.3 / 51.0 / 50.9 / 49.7 / **0.0** (2025-26)
      TR key-number rate     10.59 / 9.22 / 9.74 / 8.67 / **3.36**

  Splice validated on the two overlapping seasons rather than assumed:
  **bias -0.037 and +0.008, corr 0.9694 and 0.9777.** No level shift.

  `data/odds_unified.csv.gz` — 21,828/22,742 games (96.0%), per-game `feed_src`,
  no silent fallback. SBR 2007-08..2022-23 at 100%; TR book1 2023-24 (94.7%) and
  2024-25 (84.0%). **2024-25 granularity restored: integer 0.7% -> 49.2%,
  key numbers 0.33% -> 9.10%.**

  **2025-26 CANNOT BE UNIFIED AND IS NOT FORCED.** By that season every book in
  every held source posts half-points only — TR book1 included (0.0% integer).
  That is an industry shift, not a collection failure. Emitted with
  `convention_break=True`, coverage 47.0%, and EXCLUDED from every cross-season
  comparison rather than quietly included.

  **CONSEQUENCE — 2024-25 STOPS BEING AN OUTLIER** (offset refit walk-forward on
  each feed, since the side `sign(m_us - open)` depends on the opener;
  `m_us_blind` is feed-invariant by G2 and is reused):

      measure    OLD (stitched)              NEW (unified)
      CLV        +0.8602 rank  1/17 z +2.75  +0.5329 rank 3/17 z +1.37
      capture    +0.4510 rank  1/17 z +2.44  +0.3071 rank 3/17 z +1.08
      abs_move   +1.9073 rank  1/17 z +2.03  +1.7352 rank 2/17 z +1.49

  PAIRED on identical game sets, so this is not TR's coverage selecting games
  (covered 2024-25 games do move more on the old feed, 1.9985 vs 1.4289):

      season   n      OLD CLV   NEW CLV
      2023-24  1165    0.3172    0.5906   <- up: averaging had smoothed it
      2024-25  1033    0.9676    0.5329   <- down: the grid had inflated it

  **corr(CLV, abs_move) is +0.925 on the NEW feed too** — the relationship is
  structural, not a defect of the old data, which is why **capture fraction
  stays the required cross-season metric even now.**

  **CORRECTIONS TO `docs/OPENING_LINES.md`.** (i) "Movement has roughly doubled
  over the sample, 0.87 -> 1.86" — the doubling is real but happens at the
  **2014-15** changepoint, and the 1.86 endpoint was grid-inflated; on the
  unified feed 2024-25 is 1.735. (ii) The move-toward% series quoted there
  (60.1% in 2024-25) is computed on the distorted feed and needs recomputation.
  (iii) The doc describes ONE splice; there are two.

  **NOT A PRODUCTION CHANGE, AND DELIBERATELY SO.** Research wants a consistent
  convention; execution wants the actual price at the book you can bet, which is
  a half-point line if that is what your book posts. Those are different
  requirements and should not share a table. The unified feed is for cross-season
  DIAGNOSTICS. Switching production to it would need a full stack refit, would
  inherit 84%/47% coverage where the frame currently has 100%, and would price
  games at a book we do not bet.

- D249 **P&L PROVENANCE: 72% OF THE HEADLINE COMES FROM MODELLED PRICES. AND
  THREE DATASETS WE ALREADY OWN ARE UNUSED, ONE OF WHICH THE FAILURE LOG
  RECORDS AS UNOBTAINABLE.** [scripts/d249_pnl_provenance.py]

  **(1) IS THE P&L ACCURATE?** It is arithmetically correct at its stated
  assumptions, and REVIEW.md already discloses that best-of-nine is observed in
  one season and inferred in two. What was never quantified is how much weight
  that assumption carries.

  First, what the `tag` in `wf_perbet_OFFSET.json` actually means
  (`scripts/wf_equity.py::era_of`): it is a SEASON-level label for which
  multi-book PANEL priced the shopping tier, **not** a claim about the base
  price. **k=1 is a real observed price in every season.**

      season   n    tag            k=1 ROI   k=9 ROI   uplift   pushes
      2023-24  154  MEASURED         +9.80    +13.77    +3.97      3
      2024-25  210  EXTRAPOLATED    +18.18    +24.00    +5.82      0
      2025-26   96  EXTRAPOLATED     -4.55     +5.05    +9.59      0

      shopping uplift where the panel EXISTS (7 seasons): +2.20 [+0.97, +3.97]
      shopping uplift where it is MODELLED  (7 seasons): +3.38 [+0.50, +9.59]
      2024-25 is **2.6x** the measured mean; 2025-26 is **4.4x**.

  Of the 460-bet block, **306 bets (67%) carry an extrapolated panel and supply
  +55.24u of the +76.44u total (72%)**. The headline under three assumptions:

      AS REPORTED (modelled k=9)              +76.44u   **+16.62%**
      measured 2023-24, historical uplift
        elsewhere (+2.20)                     +61.74u   **+13.42%**
      SINGLE OBSERVED BOOK ONLY (k=1)         +48.91u   **+10.63%**

  **The headline moves 5.99 ROI points between "one book we can see" and "nine
  books we model."** Every interval spans zero regardless. The defensible floor
  is the +10.63% single-book figure, which REVIEW.md already reports.

  **BEST-OF-NINE IS UNMEASURABLE IN 2024-25, NOT MERELY UNMEASURED.** D174
  counted ESPN providers at 16 -> 2 -> 4 and its own fixed-basket ladder stops
  at **k=3**, because only three books span the seasons. There are not nine
  books in 2024-25 to measure.

  **PUSH ASYMMETRY (new).** A half-point line cannot push. The ledger carries
  0-5 pushes per season historically and **exactly zero in 2024-25 and 2025-26**
  — D248's granularity finding visible in the P&L itself. Any shopping model
  that converts line improvement into cover probability is flattered by a grid
  where no improvement is ever absorbed by a push.

  **(2) DATA WE ALREADY HOLD AND DO NOT USE.** Counted from the raw JSONL.

  **(a) MONEYLINE POST-2023 IS NOT BLOCKED — FAILURES.md §E IS WRONG.** The
  entry says ML is "zero from 2024 onward". That is true of `odds_market` and
  `odds_hist_sbr` and false of the Action Network raw sitting in
  `data/raw/sbr_ext/`, which carries moneyline for **99.8% / 98.5% / 95.9%** of
  regular-season games in 2023-24 / 2024-25 / 2025-26 across **6-7 books**,
  plus totals at the same coverage. D241's joint-market work was abandoned
  partly for want of this.

  **(b) PUBLIC BET-SPLIT DATA, NEVER ONCE USED.** Every AN market entry carries
  `bet_info` with `money.percent` and `tickets.percent` — the money-versus-
  tickets divergence that is the standard sharp-money indicator:

      season   bet_info   money%!=tickets%   mean|diff|   max|diff|
      2023-24    99.8%          83.3%          15.67         87
      2024-25    98.4%          91.0%           4.38         58
      2025-26    95.9%          93.0%           4.85         54

  Percentages only; the absolute `value` fields are 0 throughout, so handle is
  unavailable. **2023-24's divergence is ~3x the later seasons**, so this field
  needs the same cross-season caution D248 imposed on CLV and must not be
  pooled across the feed regimes without a check.

  **(c) REALISED PARTICIPATION RUNS TO 2000-01, TWELVE SEASONS BEFORE THE
  INJURY REPORT.** `player_game_stats` holds 1,005,376 rows covering game-id
  prefixes 00200..00225 — **2000-01 through 2025-26 at full regular-season
  counts**. The D186 caveat is that the availability leg only works from
  2019-20 because `injury_reports_pit` starts 2018-12-17. That bounds the
  REPORT, not participation. A durability/absence prior built from realised
  participation history is PIT-legal and available for every season in the
  frame. It is a DIFFERENT model from the report-driven one and would have to
  be validated as such, not swapped in.

  **NOT A PRODUCTION CHANGE.** This entry counts what exists; it builds nothing.

- D250 **AVAILABILITY CANNOT BE BACKFILLED FROM DATA ON DISK. THE FAILURE IS
  STRUCTURAL, AND basketball-reference IS THE ONLY REACHABLE GROUND TRUTH.**
  [scripts/d250_availability_backfill.py]

  **REACHABILITY, TESTED BEFORE ANYTHING WAS PLANNED AROUND IT.** Every claim
  below is a live HTTP result from this machine, not an assumption:

      basketball-reference.com          **200**, 469 KB box score — WORKS
      kaggle.com                        200 (landing only; datasets need auth)
      prosportstransactions.com         403 — Cloudflare "Just a moment" JS challenge
      stats.nba.com                     **connection timeout** (not an HTTP error)
      cdn.nba.com                       403 "Access Denied" — egress proxy
      site.api.espn.com                 403 "Access Denied" — egress proxy
      api.balldontlie.io                401 — now key-gated
      api.sportsdata.io                 401 — paid

  The three NBA/ESPN hosts fail at the proxy with an identical body, so that is
  an egress blocklist here rather than the sites refusing. **b-ref is the only
  free, reachable source of historical availability.** Its box scores publish a
  per-team `Inactive:` block plus "Did Not Dress" markers, verified by fetch.

  **THE FREE SHORTCUT, AND WHY IT FAILS.** `player_game_stats` covers
  2000-01..2025-26 but holds only DRESSED players, so an inactive player's
  absence is the signal. Two roster rules were tested against b-ref ground truth
  on 35 games x 2 teams spread over six seasons (2008-09 .. 2024-25):

      arm                        precision  recall     F1   exact-set
      A  +-10 team-game window      0.741    0.716   0.728    35.7%
      B  tenure brackets the date   **0.941**  0.560   0.703    24.3%

  Arm B was not a retune: arm A's errors were dominated by mid-season TRADES
  (one team-game derived seven false inactives, all departed players), so B
  requires the player's first and last appearance for that team to bracket the
  date. That fixed precision as predicted, 0.741 -> **0.941**.

  **RECALL IS THE WALL AND IT CANNOT BE TUNED AWAY.** B's misses are
  season-ending injuries — `adonal foyle; jeremy richardson; mike wilks`,
  `martell webster; raef lafrentz`. **A player who never plays again leaves no
  trace in a table of who played.** Those are exactly the absences a margin
  model most needs. No roster rule over played-games data can recover them; the
  information exists only in a roster or transaction feed, which we do not hold
  and cannot reach (prosportstransactions is Cloudflare-blocked).

  **VERDICT: the on-disk derivation is NOT usable** as an availability source.
  F1 is ~0.71 either way; the arms trade precision for recall without either
  reaching a level that could feed E[outs].

  **THE SCRAPE IS FEASIBLE BUT IS NOT RECOMMENDED YET.** Backfilling
  2007-08..2018-19 is ~13,500 box scores; at b-ref's stated 20 req/min ceiling
  (this script uses 15/min and caches every response) that is **~11 hours** of
  someone else's bandwidth. Before spending it, note what it would and would not
  buy: **production only BETS in seasons where the injury report already exists**
  (2019-20+, per D186). Backfilling would widen the EVALUATION corpus, not
  improve any shipped forecast. That may still be worth it — every inference in
  this register is season-clustered and starved of k — but it should be decided
  as "buy more validation seasons", not as "fix the availability leg".

  Cost of this entry: 36 fetches, cached under `data/raw/bref_boxscores/`.
  Two b-ref codes 404'd on the first pass because team codes are ERA-DEPENDENT
  (CHA->CHO only from 2014-15; BKN is NJN before 2012-13); `bref_code()` now
  handles this.

- D251 **THE PUBLIC BET SPLIT: REAL, UNBETTABLE, AND ITS MECHANISM IS REFUTED.**
  [scripts/build_an_market.py, scripts/d251_betsplit_signal.py]

  First test of a factor held on disk since August and never opened.
  `build_an_market.py` extracts the Action Network panel — per-book closing
  spread, moneyline, total, and the money/tickets split — into
  `data/an_market.csv.gz`: **19,351 game-book rows, 3,607/3,690 games (97.8%)**,
  with ML, totals and bet-split each at 95.7-99.3% by season.

  **AN HOLDS ONE SNAPSHOT PER BOOK AND IT IS THE CLOSE.** There is no opening
  price and no history, so this panel can test whether the split predicts the
  OUTCOME and can never test line MOVEMENT. Any "reverse line movement"
  construction from it would be leakage.

  **THE BOOK BASKET FOR THIS FIELD IS NOT D248's.** D248's core basket
  (15/68/69) was validated for SPREAD coverage. The split populates differently:
  books 68/69/71/75 report `money% == tickets%` — a placeholder, not a
  measurement — on 69-97% of rows in 2023-24 and 2024-25, and only fill it
  properly in 2025-26. **Book 15 is the only book populated across all three
  seasons** (83.3% / 91.4% / 92.8% on 1,221 / 1,205 / 1,174 rows) and is
  therefore the whole usable panel. Choosing a basket once and reusing it for
  every field is a mistake this nearly made.

  **RESULTS** (weekly block bootstrap; k=3 seasons cannot resolve anything, so
  season clustering is reported alongside, not as the primary):

      T1  back the side with more MONEY than TICKETS, vs the CLOSE
          n=3,200   cover **51.69%**   CI [50.05%, 53.30%]
          vs 50%: +1.69pp    vs 52.38% break-even: **-0.69pp**
          per season 51.24 / 50.41 / 53.39 — no single season drives it
          >>> **REAL BUT UNBETTABLE**

      T2  monotonicity in |money% - tickets%|, quintiles:
          49.38 / 54.17 / 54.22 / 47.61 / 53.99, rank corr **+0.133**
          >>> **NOT MONOTONE — the mechanism is REFUTED.** If the split worked
          because large wagers are informed, the edge would grow with the
          divergence. It does not, so whatever T1 detects is not that.

      T3  model side, split AGREES     n=1648  cover 51.58% [49.24, 53.96]
          model side, split DISAGREES  n=1552  cover 48.20% [45.87, 50.60]

  T3 is the only part worth returning to — a ~3.4pp gap in the model's own cover
  rate — but the intervals overlap and it is an unpaired comparison on three
  seasons. **Not a filter yet, and not claimed as one.**

  **A CONSTRUCTION BUG, CAUGHT AND RECORDED.** The first run aggregated with
  `median(money) - median(tickets)`, which is not `median(money - tickets)`.
  Independent medians coincided and the nonzero-divergence rate collapsed from
  ~90% to 36.9%, leaving a sample that was 82% one season. The per-book
  divergence is now computed first and aggregated second.

  **NO PRODUCTION CHANGE.** The lasting deliverable is `an_market.csv.gz`, which
  also unblocks the post-2023 moneyline work D249 found was never blocked.

- D252 **NO, STAR ABSENCES ARE NOT OUR EDGE. THE EDGE IS AGAINST THE OPENER
  ONLY, IT IS BROAD RATHER THAN POCKETED, AND NO SLICE BEATS A FAMILY NULL.**
  [scripts/d252_where_is_the_edge.py, d252b_edge_offset.py]

  **THE OFFSET, DECOMPOSED.** `m_offset = m_open + 0.3564*(m_blind - m_open)
  + 0.0417*rest_diff - 0.0114*|m_open|`, equivalently
  **`0.6436*m_open + 0.3564*m_blind`** plus two nudges. Measured on 22,740 games:

      term                          mean|.|   sd     p95|.|
      0.3564*(m_blind - m_open)      0.743   0.976   1.923
      0.0417*rest_diff               0.009   0.025   0.042
      -0.0114*|m_open|               0.068   0.043   0.148

  The rest and size terms are **rounding error** — 0.009 and 0.068 points. The
  layer is one thing: take our disagreement (mean 2.085 pts) and keep 36% of it.
  The correction exceeds a full point on 26.7% of games.

      MAE vs the realised margin:
        opening line  9.7964      blind model  9.8954   <- WORSE than the line
        shipped offset 9.7760     closing line 9.6876

  **Our own model is worse than the opening line it corrects.** The offset works
  by shrinking it, not by trusting it.

  **Q1 STARS OUT — THE ANSWER IS NO.** Star = top-3 by PIT prior-season-to-date
  minutes; absence is realised non-appearance, used only to PARTITION the
  evaluation, never as an input. 21,234 games, 19 seasons.

      stars out    n     ll_mkt   offset vs close   offset vs open
      0          8992    0.6029      +0.00334          -0.00112
      1          7266    0.5789      +0.00510          **-0.00315 SIG**
      2          3366    0.5748      +0.00431          +0.00044
      3+         1610    0.5611      +0.01024          +0.00030

      gap(>=1 out) - gap(none) vs CLOSE  +0.00240 CI [-0.00127, +0.00606] ns
      gap(>=1 out) - gap(none) vs OPEN   -0.00106 CI [-0.00323, +0.00111] ns

  Neither is significant and the vs-close sign is the WRONG way. **Correction to
  the intuition behind the question:** market log loss FALLS as absences rise
  (0.6029 -> 0.5611) — star-out games are MORE predictable, not less, because
  they are more lopsided. The market handles them fine. D77's "star-out-favorite
  wins" reading is not reproduced at the layer production emits.

  **Q2 WHERE IS THE EDGE.** Five pre-registered slice families, fixed before
  reading, scored against a family-wide permutation null (labels shuffled within
  season, max |gap| recomputed 2,000 times):

      versus the CLOSE: **every single slice is positive** (we are behind).
        observed max |gap| 0.01017; null median 0.01319, 95th 0.02768
        **p = 0.7370 — NO SLICE BEATS THE FAMILY NULL.**

      versus the OPEN: several slices negative, the largest being
        |our edge| Q5 (biggest disagreement)  -0.00577 CI [-0.01069,-0.00086]
        |open| 2-5 band                       -0.00340 CI [-0.00603,-0.00076]
        exactly one star out                  -0.00315 CI [-0.00470,-0.00160]

  Note the reversal between layers: on the BLIND model the biggest-disagreement
  quintile is where we are WORST against the close (+0.02363), while on the
  OFFSET margin it is where we beat the opener MOST (-0.00577). That is the
  offset doing its job — the shrinkage protects exactly where the raw model is
  most wrong.

  **NET: the edge is ~-0.0017 nats against the OPENER, spread broadly, and
  there is no pocket.** Against the close we are +0.0050 behind and no slice
  escapes it.

  **THREE CORRECTIONS TO THE RECORD.**

  **(a) `p_us` IN THE FRAME IS THE BLIND MODEL, NOT THE SHIPPED ONE.** It equals
  `sigmoid(m_us696/6.96)` to 2.2e-16 and equals `sigmoid(m_us_blind/7.2)` on
  **75.2%** of games. Any analysis using `p_us` as "our forecast" has been
  scoring the market-blind model. D252's first pass did exactly that; D252b
  rescores on `m_us`. **Every earlier entry that read `p_us` as the shipped
  probability needs the same check.**

  **(b) THE CALIBRATION VEIN IS STILL CLOSED, NOW CONFIRMED FOR THE OFFSET
  ARCHITECTURE.** D74 closed it (scale stays 7.2; nulls at D48, D61, T1). A
  walk-forward scale refit, same margin on both sides, reproduces that verdict:

      BLIND margin   walk-forward minus fixed-7.2  +0.00010 CI [-0.00025,+0.00045]
      OFFSET margin  walk-forward minus fixed-7.2  -0.00009 CI [-0.00066,+0.00048]

  The blind figure matches D74's +0.0001 almost exactly. The offset compresses
  the margin (sd 6.695 -> 6.523) so its optimal scale drifts to 6.38-6.84, but
  the gain from chasing it is still nil. **D74 holds and now covers D224.**

  **(c) THE "~14x TWO-STAGE ATTENUATION" IS ~4.5x.** D245d's own worked example
  is `0.04406 x 0.652 x 0.3413 = 0.00981`, which is a factor of
  **0.04406/0.00981 = 4.49**, and `1/(0.652 x 0.3413) = 4.49`. The arithmetic
  reproduces to 1e-5; only the label is wrong. This matters because **D246 chose
  its research direction citing "~14x"** — a market-quantity target still
  bypasses the blend, so the DIRECTION stands, but the stated size of the prize
  was overstated threefold. Anything gated on "must clear a 14x haircut" should
  be re-read against 4.5x.

  **NO PRODUCTION CHANGE.**

- D253 **NO COMBINATION OF OBSERVABLE CONDITIONS PREDICTS WHERE WE BEAT THE
  MARKET. THREE OF FOUR CELLS ARE FLAT AND THE FOURTH IS OUR OWN EDGE SIZE.**
  [scripts/d253_differentiators.py, d253b_probe.py]

  Sean asked for season length, every production and rejected feature, and any
  other differentiator, swept together rather than one slice at a time. D252
  tested five slice families one variable at a time and none beat a family null;
  a one-at-a-time test cannot find a condition that exists only as a
  COMBINATION. An elastic net over everything at once is the right tool and also
  the most dangerous thing in this repo (D239: best-of-N random subsets buy
  +2.54 ROI points from nothing), so the headline is **walk-forward OOS R^2
  against a permutation null over the ENTIRE procedure** — outcome shuffled
  within season, the whole fit rerun end to end including standardisation, the
  alpha/l1 search and the fold structure.

  Outcome is a DIFFERENCE, not our error: `ll(offset margin) - ll(price)`.
  Predicting our own error would only rediscover that some games are less
  predictable. Leakage asserted, not assumed: the closing line is an input to
  `y_close` by construction, so it and everything derived from it is banned from
  the feature set.

  **COVERAGE TIERS, MEASURED NOT ASSUMED** — the first run collapsed 19 seasons
  to 3 because `dropna` hit features that do not exist in the early era:

      Tier A  19 seasons  line, total, rest, b2b, our disagreement, days-in,
                          season length, shortened-season flag, PIT star
                          absences, national TV                      (18 feats)
      Tier B   7 seasons  + injury-report outs, inactive counts,
                          absence_tr, expected outs                   (26 feats)
      Tier C   3 seasons  m_ff / m_comp — NO walk-forward fold structure
                          exists, so it is reported unavailable, not fitted

  Tier-B features are **not zero-filled outside their era**: a 0 would mean
  "nobody out" in 2021 and "we have no idea" in 2010, which would teach the
  model an era rather than an absence.

  **RESULTS**

      model                 target     OOS R^2    null median   p
      A  19 seasons         vs close   -0.00086     -0.00057   0.850
      A  19 seasons         vs open    **-0.02001** -0.00096   1.000
      B   7 seasons         vs close   -0.00428     -0.00191   0.925
      B   7 seasons         vs open    **+0.00170** -0.00200   0.000*

  *40 draws cannot resolve p below 1/40; re-probed in D253b.

  **THE STRUCTURAL CEILING, WHICH MAKES THE NULL RESULT UNSURPRISING.** For
  `y_open`, between-season sd of the mean is **0.00286** against a within-season
  sd of **0.08947** — a ratio of **0.032**. Even a PERFECT season-level
  predictor could explain at most ~0.1% of the variance in our per-game edge.
  Almost everything is game-level noise, so any regime or condition story is
  bounded before it starts.

  **WHY MODEL A ACTIVELY HURTS (-0.02001, worse than its own null).** The
  outcome's dispersion is itself regime-dependent: within-season sd of `y_open`
  runs **0.162 in 2007-08..2009-10 and 0.059-0.087 thereafter**. Every fold
  trains across that boundary, so a fit learned on the high-variance era is
  worse than predicting the mean. This is a variance regime, not a signal
  regime, and it is a third era boundary on top of D248's two feed regimes.

  **NOTE ON 2024-25.** Mean `y_open` is -0.01095, four times any other season,
  with the second-highest dispersion. That is the D248 feed artefact showing up
  once more, and it is the single season most responsible for any apparent
  "we beat the opener" trend.

  **CORRECTION TO D252's ACCOUNT OF THE OFFSET.** D252 described lambda=3000 as
  a hard shrink that cuts the edge coefficient to 0.3564. That is wrong. The
  penalty is nearly inert at the shipped value:

      lambda        0      1000     3000     10000    30000
      edge coef   0.3608   0.3590   0.3554   0.3425   0.3090

  The UNPENALISED fit already gives 0.3608. **The 36% is the regression's
  verdict on our disagreement, not an artefact of the penalty** — the data say
  our edge over the opener is worth about a third of face value, and the ridge
  only trims the last half point.

  **NO PRODUCTION CHANGE.**

- D253b **THE ONE POSITIVE CELL IS OUR OWN EDGE SIZE. THE OTHER 24 FEATURES SIT
  EXACTLY ON THE NULL, AND CARRYING THEM COSTS 5x THE OOS R^2.**
  [scripts/d253b_probe.py]

  D253's Model B / `y_open` returned +0.00170 at p = 0.000 on 40 draws — a
  resolution floor of 1/40, across four cells, so it could not be accepted as
  written. Re-probed with **400 draws** and an ablation:

      arm                                features   OOS R^2
      FULL                                  26      +0.00170
      **our_edge + our_edge_abs ONLY**       2      **+0.00837**
      FULL minus the two edge columns       24      **-0.00185**

      400-draw null: median -0.00204, 95th -0.00130, **max -0.00098**
      observed +0.00170, p = 0.0000, family-adjusted over 4 cells p = 0.0000

  **The result is real and it is entirely the two edge columns.** The 24-feature
  arm lands at -0.00185 against a null median of -0.00204 — indistinguishable
  from noise. Season length, the shortened-season flag, rest, back-to-backs,
  line level, total, days into season, national TV, star absences, injury-report
  outs, inactive counts, absence_tr and expected outs **contribute nothing,
  jointly or severally.**

  And carrying them is not free: the same target, same folds, same procedure
  gives **+0.00837 on two features and +0.00170 on twenty-six**. Twenty-four
  uninformative columns cost four-fifths of the out-of-sample signal. That is
  the cleanest demonstration in this register of why the feature pile keeps
  failing (D157, D237, D238) — it is not that the extra features are neutral,
  it is that they are actively expensive.

  **WHAT THE SURVIVING SIGNAL SAYS, AND WHY IT IS NOT NEW.** The coefficients
  are `our_edge_abs` negative and `our_edge` positive: the larger our
  disagreement with the opener, the more we beat the opener. D252b found the
  same thing by slicing (|our edge| Q5 at -0.00577, the largest of any slice).
  **The offset layer already prices exactly this, at 0.3564.** So the answer to
  "is there a condition telling us when to trust ourselves" is: yes, one — the
  size of our own disagreement — and it is already in production.

  **THE ONE ACTIONABLE READ.** The offset is LINEAR in the edge. If beating the
  opener genuinely grows with |edge| faster than linearly, a convex spending
  rule — a quadratic or spline term in `edge` — would capture something the
  current layer cannot. That is a real, narrow, testable follow-up, and it is
  the only shippable idea to come out of the whole sweep. It needs its own
  pre-registered gate and an MDE80 stated in advance; **R^2 0.008 on a log-loss
  difference is a small prize and the two-stage attenuation (4.5x, corrected in
  D252) applies to anything entering through the blend but NOT to this, which
  sits in the offset itself.**

  **NO PRODUCTION CHANGE.**

- D254 **THE CONVEX SPENDING RULE IS REFUTED, AND THE SHAPE EVIDENCE LEANS
  CONCAVE. MY INFERENCE FROM D253b CONFLATED TWO DIFFERENT QUANTITIES.**
  [scripts/d254_convex_offset.py]

  D253b's only actionable read was that the offset is LINEAR in the edge while
  two independent findings suggested the payoff grows with |edge| — D252b's
  largest-|edge| quintile beating the opener most (-0.00577) and D253b's
  `our_edge_abs` carrying a negative coefficient in 100% of folds. Four arms,
  all refit walk-forward at the production ridge with the probability scale
  refit per arm per fold, knot fixed in advance at 3.0 points (~65th percentile
  of |edge|) and NOT tuned. The convex term is `edge * |edge|`, never `edge^2`,
  because the correction must be ODD in the edge.

      arm  vs LINEAR        CI                    better   MDE80
      Q  quadratic  +0.000076  [-0.000049,+0.000200]  9/18  0.00017 (2.2x)
      S  spline     +0.000018  [-0.000054,+0.000089]  6/18  0.00010 (5.3x)
      C  clipped    +0.000110  [-0.000087,+0.000307]  6/18  0.00026 (2.4x)

  **All three are ns and all three point the WRONG WAY** (positive = worse than
  linear). Status `UNDERPOWERED` on the strict taxonomy — every MDE80 is 2-5x
  its own estimate — but there is no hint of a prize here to be underpowered
  against.

  **THE FITTED SHAPE IS CONCAVE, NOT CONVEX.**

      C  inner slope +0.3958   outer slope +0.2051   (outer is HALF the inner)
      S  edge +0.3703, beyond-3pt -0.0458  ->  outer slope +0.3245
      Q  edge*|edge| +0.00301  (weakly convex, and the odd one out)

  S and C agree that large disagreements deserve a SMALLER fraction, not a
  larger one; Q disagrees with both. Three arms that cannot agree on the sign of
  the curvature are describing a shape the data do not determine.

  **WHERE MY INFERENCE WENT WRONG, because the distinction generalises.**
  "The offset beats the opener MOST when |edge| is large" and "the offset should
  SPEND MORE when |edge| is large" are different claims. The first is about
  where the layer adds value; the second is about the optimal coefficient. Under
  a correctly specified LINEAR model the first is expected anyway — a large edge
  produces a large correction, and a large correction has more room to improve
  on the opener. D252b and D253b were both consistent with linearity all along,
  and I read a shape claim out of a magnitude observation.

  **The linear offset stands.** D253's sweep therefore yields no shippable
  change at all: the 24 non-edge features sit on the null, and the two edge
  columns that carry the signal are already priced correctly at 0.3564.

  **NO PRODUCTION CHANGE.**

- D255 **THE RATINGS ARE RANK-1 AND SEAN IS RIGHT THAT THIS IS A REAL
  LIMITATION. PAIR EFFECTS DO PERSIST ACROSS SEASONS — AND ARE WORTH 0.03-0.10
  POINTS, ns.** [scripts/d255_matchup_residual.py, d255b_matchup_value.py;
  external review: an independent external review]

  **HOW THE OPPONENT-ADJUSTED RATINGS ACTUALLY WORK.** One ridge regression over
  every team-game to date, `ortg = 100*pts/poss`,
  `poss = fga + 0.44*fta - oreb + tov`:

      ortg_ij = mu + off_i - def_j + home*is_home + home_dev_i*is_home

  ridge 25 on {off, def}, 200 on the per-team home deviations, sum-to-zero
  identification, leakage-safe (`fit_through(date)` uses only prior games).
  **Each team is exactly two numbers.** Team i's offence is assumed to perform
  the same amount better or worse against EVERY defence — there is no
  `off_i x def_j` term anywhere in the stack. A persistent stylistic propensity
  is precisely what this cannot express, so the question is well posed.

  **BEFORE PARAMETERISING AN INTERACTION, TEST WHETHER ONE EXISTS.** A free
  interaction is 870 ordered pairs on ~2.75 meetings each; it would fit noise
  perfectly. So the prior question is whether the pair-level residual REPLICATES:

      R1  within season, first meeting vs later   r **-0.0183** CI [-0.0356,-0.0009]  5/19 positive
      R2  across seasons, pair in s vs s+1        r **+0.0303** CI [+0.0122,+0.0484] 14/18 positive

  **That split is exactly the signature a stylistic propensity would leave** —
  teams adjust to each other between meetings within a season, while the
  underlying scheme mismatch recurs the following year. (R1's negative sign is
  partly mechanical: off/def are fitted on the same games, inducing a small
  negative residual correlation. That bias works AGAINST R2, so R2 is the safe
  direction.)

  **TWO CONFOUNDS TESTED AND BOTH REFUTED.**
  (a) *Scheduling.* If persistence came from better-estimated means or shared
      divisional travel, it would be strongest where teams meet most. It is the
      opposite: same-division (3.88 meetings) **+0.0252 ns**, same-conference
      **-0.0076 ns**, **cross-conference (1.94 meetings) +0.0461 SIG**.
  (b) *Conference imbalance.* Removing the conference-pair mean changes almost
      nothing: 0.0303 -> **0.0291** overall, 0.0461 -> **0.0450** cross-conference.
      (The imbalance itself is real and persistent — West-offence-vs-East-defence
      is positive in 16/19 seasons, mean EW-WE gap -0.4444 ortg — it just is not
      what R2 is measuring.)

  Persistence in the FEWEST-meeting group remains unexplained. A pair-specific
  travel/venue effect that repeats annually is the obvious candidate and is NOT
  stylistic; it is left open because the value test below makes it moot.

  **THE ONLY NUMBER THAT MATTERS: DOES LAST SEASON'S PAIR RESIDUAL PREDICT THIS
  SEASON'S MARGIN AGAINST THE MARKET?** Strictly out-of-sample, 21,502 games,
  18 seasons:

      vs CLOSE  slope +0.00080 pts  CI [-0.01910,+0.02069]  ns  MDE80 **33.2x** est
      vs OPEN   slope +0.00317 pts  CI [-0.01679,+0.02313]  ns  MDE80 **8.4x** est

      implied margin swing across +-2sd of the signal:
        **+0.025 points vs the close, +0.100 points vs the open**

  **Real, persistent, significant in R2 — and worth a tenth of a point at
  best, indistinguishable from zero.** This is the register's recurring shape
  once more: an effect that exists and cannot be traded.

  **EXTERNAL REVIEW (external review, high reasoning) — INDEPENDENTLY REACHED THE SAME
  PLACE.** Verdict: *"worth one falsification attempt, not worth a search
  programme."* Its reasoning, which I am recording because two points are
  sharper than mine:
  - A free `off_i x def_j` layer is dead on arrival, and a LATENT bilinear term
    learned from `ortg` alone is also not credible — **without exogenous style
    anchors it mostly re-expresses mean team strength and schedule noise in a
    rotated basis.**
  - The only defensible form is `(s_i^O)' Lambda (s_j^D)` with **K <= 3**
    exogenous style axes built from trailing box-score rates (3PA/FGA, AST/FGM;
    FTA/FGA, OREB rate; TOV/poss, STL, PF), diagonal Lambda, league-wide slopes,
    ridge heavier than on off/def — **and the style axes ORTHOGONALISED to the
    team's estimated mean off/def first, or the "interaction" degenerates into
    "good offences punish weak defences", which the model already has.**
  - Required placebo: **shuffle opponent style vectors within season/date
    buckets; if the effect survives, it is schedule/quality leakage.**
  - Effect-size prior: 0.3-0.8 pts raw in the extreme tail, ~0-0.2 typical;
    after the 4.5x attenuation, **0.07-0.18 emitted in the tail and
    economically irrelevant typically.** That brackets my measured 0.03-0.10.

  **COACH ROTATION FLEXIBILITY: NOT IDENTIFIABLE, AND I AGREE.** Minutes
  concentration, game-to-game minutes reshuffling and role volatility are all
  computable from box scores, but none separates coach flexibility from roster
  depth, star dependence, injuries, blowouts, back-to-backs or opponent
  strength. Without lineup continuity or usable play-by-play (`lineup_stints`
  covers 20-107% by season, i.e. unusable) this is a **team-level responsiveness
  index at best, not a coach effect**. `DATA_BLOCKED`, not refuted.

  **THE MORE USEFUL CRITICISM.** The external review’s closing point is one this register has
  not stated plainly: the stack is **mean-state** — season-average team identity
  — while the residual edge against an opener most plausibly lives in **state
  changes**: who is effectively available, what the current minute allocation
  implies, short-horizon roster configuration. That is an argument for the
  queued availability/minutes items over any matchup work, and it agrees with
  D252's finding that the shipped edge is broad rather than pocketed.

  **NO PRODUCTION CHANGE. Item 5 of the queue (regularised matchup
  interactions) is now bounded rather than open**: if it is ever run it must be
  the K<=3 orthogonalised style form with the shuffle placebo, and it must clear
  a bar of ~0.1 points that the direct measurement says it will not.

- D256 **D255's OPEN QUESTION, CLOSED: THE PAIR PERSISTENCE WAS MOSTLY A *TEAM*
  EFFECT, AND THAT TEAM EFFECT IS RIDGE SHRINKAGE. THE GENUINE INTERACTION IS
  ns.** [scripts/d256_pair_interaction.py]

  D255 measured across-season persistence of the ratings residual on RAW pair
  means and left the cross-conference concentration unexplained, guessing at a
  travel/venue effect. The guess was unnecessary. A pair mean decomposes as

      grand + row_i + col_j + interaction_ij

  where `row_i` is anything persistently unmodelled about team i's offence and
  `col_j` the same for team j's defence. Both recur next season for reasons
  that have nothing to do with the opponent, and **every pair (i, .) inherits
  row_i** — so team-level persistence masquerades as pair persistence. D255
  never removed them. Double-centring the 30x30 matrix within season does:

      L0  raw pair means (what D255 used)      r **+0.0303** CI [+0.0122,+0.0484] SIG
      L1  row-centred                          r  +0.0227  CI [+0.0042,+0.0412] SIG
      L2  DOUBLE-CENTRED = interaction only    r **+0.0170** CI [-0.0016,+0.0356] **ns**

  **About 44% of D255's headline was team-level, and what remains does not
  reach significance.** The matchup question is therefore weaker than D255
  reported, and it was already worth only 0.03-0.10 points there.

  **THE TEAM EFFECT IS THE ESTIMATOR, NOT THE MODEL.** Row and column
  persistence are large — r +0.4386 and +0.4126, 18/18 seasons — but they are
  ridge shrinkage bias, and the scale proves it:

      ridge   sd(row means), ortg pts
        25    **7.404e-01**   <- shipped
         5     1.812e-01
         1     3.794e-02
         0    **5.274e-14**   <- OLS: numerically zero

  Ridge pulls `off_i` toward the mean, so strong offences are persistently
  under-fitted and weak ones over-fitted; team strength persists across seasons;
  the residual persists with it. Under OLS the residuals are orthogonal to the
  team dummies by construction and the effect vanishes entirely. **This is the
  intended bias half of the bias-variance trade the ridge exists to make, not a
  deficiency to fix.** Including the per-team `home_dev` term (which production
  carries and D255's fit omitted) changes nothing: 0.5502 -> 0.5498.

  **AN ERROR OF MINE, RECORDED BECAUSE IT NEARLY BECAME A FINDING.** I first ran
  this comparison reporting only correlations and got r ~ 0.55 for **every**
  spec including OLS — which I was about to read as "the persistence is not
  shrinkage after all". It was correlating values of order **1e-14**. A
  correlation is scale-free, so it happily reports structure in float noise. The
  check that settled it was one line of `sd()`.

  **WHAT REMAINS.** The cross-conference interaction survives double-centring
  (L2 +0.0364, SIG; divisional L2 +0.0032, ns), so something pair-specific and
  cross-conference-concentrated is still there. It is one cell of several, it is
  the group with the FEWEST meetings, and D255b already measured the whole pair
  signal at +0.025 points vs the close and +0.100 vs the open, both ns with
  MDE80 33x and 8x the estimate. **Not worth further pursuit on these data**;
  travel/venue remains the most plausible candidate and is not stylistic.

  **NO PRODUCTION CHANGE — and nothing from D247 onward has shipped.** The last
  production change in this register is D232 (`nbapred/model/absence.py`).
  Everything since has been diagnostic, corrective, or refuted, which is the
  honest state of the work.

- D257 **TENDENCIES ARE REAL AND HIGHLY RELIABLE. THEY DO *NOT* TAKE A SEASON TO
  LAND — BUT HOW FAST DEPENDS ON WHICH TENDENCY, AND THE SPLIT IS THE USEFUL
  PART.** [scripts/d257_tendency_persistence.py]

  The composition model reduces each player to ONE scalar (`talent`, minutes-
  weighted). A pass-first guard and a rim-runner with equal talent are
  interchangeable in it. Before building any style representation, two questions
  had to be answered: are tendencies measurable at all, and are they estimable
  POINT-IN-TIME? A tendency that resets on opening night cannot be carried in
  from prior games, and a PIT estimate of it would be stale exactly when it
  matters.

  Five rate axes, all computable from `player_game_stats` fields the model never
  touches (`rima`, `mida`, `thra`, `ast`, `tov`, `fta`). 45,484 team-games,
  19 seasons. Rates, so team quality divides out.

  **A. RELIABILITY — odd vs even games within a season:**

      fg3_rate  **+0.953** [+0.943,+0.963]      ast_rate  +0.836
      rim_rate  **+0.927** [+0.916,+0.939]      ftr       +0.732
      tov_rate  +0.692

  These are not noise. Shot composition is measured about as reliably as
  anything in this project.

  **B. PERSISTENCE across the offseason:** fg3 +0.636, ast +0.623, rim +0.609,
  tov +0.509, ftr +0.501. Substantial carry-over, well short of identity.

  **C. WHAT PREDICTS THE REST OF THE SEASON — last year, or the first 10 games?**

      axis       prior season   first 10 games   verdict
      fg3_rate      +0.629         **+0.791**    first 10 wins clearly
      rim_rate      +0.594         **+0.719**    first 10 wins clearly
      ast_rate      +0.604           +0.645      tie
      ftr           +0.496           +0.502      tie
      tov_rate      +0.481           +0.538      first 10

  **D. BIG MOVERS — the actual regime changes.** Top decile of |this season −
  prior season|, n=54 each. Fraction of the eventual change already present in
  games 1-10, with a **disjoint denominator** (games 21+ only, so the numerator
  is not inside its own denominator — the naive version shared sample and read
  0.81-0.97, which was inflated):

      fg3_rate  **0.79**  [0.53, 1.07]      ftr       0.65  [0.17, 0.96]
      rim_rate    0.62    [0.27, 0.88]      ast_rate  **0.57**  [0.22, 0.99]
      tov_rate    0.47    [0.13, 0.91]

  **THE ANSWER TO THE QUESTION AS ASKED.** Sean's intuition was that a new
  coach's change to shooting composition "takes a while to get down". It is
  **half right, and the half that is wrong is the one he named**:

  * **Shot selection changes almost immediately.** ~79% of the eventual
    three-point-rate change is present within ten games. A coach can order more
    threes on night one, and the data says they do.
  * **Ball movement and ball security change slowly.** ast_rate 0.57 and
    tov_rate 0.47 — barely half the eventual change has landed by game ten.
    Habit is slower than instruction.

  **DESIGN CONSEQUENCE, and it is specific.** A PIT tendency estimator should
  NOT use one window for all axes. Spacing and rim pressure should be read from
  a SHORT current-season window (they are reliable at r 0.93-0.95 and land
  fast); playmaking and turnover rate need a prior-season prior blended in,
  because a short current window measures a transition rather than a level.
  Using one trailing window for everything would be wrong in opposite directions
  on the two groups.

  **WHAT THIS DOES NOT ESTABLISH.** That any of it predicts the market. Tendency
  reliability is a property of the measurement, not evidence of edge; D253b's
  finding stands that 24 plausible features cost 80% of out-of-sample R^2. The
  next test is whether the composition residual relates to lineup tendency
  COMPLEMENTARITY, and it must clear the same bars as everything else.

  **NO PRODUCTION CHANGE.**

- D258 **THE PER-AXIS TENDENCY ESTIMATOR, BUILT AND ITS CONSTANTS FITTED. THE
  SPREAD IS 16x IN GAMES, SO ONE WINDOW REALLY IS WRONG BOTH WAYS.**
  [scripts/d258_tendency_estimator.py]

  **FIRST, THE ANSWER TO "ARE WE DOING SO": NO.** Production carries exactly two
  things per player — the DARKO scalar and `trailing_min`, a fixed 10-game
  minutes average — and estimates no tendency at all. D257's design consequence
  was a specification for something that did not exist.

  **THE ESTIMATOR** is empirical-Bayes shrinkage in the natural denominator
  rather than a window:

      tendency_hat = (numerator_to_date + k * base) / (denominator_to_date + k)

  `base` is the player's prior-season rate where it exists, else the league rate;
  both strictly PIT. `k` is the prior's weight in denominator units, i.e. exactly
  how much current-season evidence it takes to match the prior. A window is
  worse on two counts: it discards everything past its edge and weights
  everything inside it equally, while shrinkage degrades smoothly and handles a
  player with 4 attempts and one with 400 in the same expression.

  **k IS FITTED, NOT ASSERTED**, walk-forward — chosen on seasons strictly before
  s, scored on s — so D257's reading was falsifiable. Endpoint is
  attempt-weighted squared error on the next game's observed rate.

      axis       k        denom   games-equiv   vs prior-only   vs current-only
      fg3_rate     8      fga        **0.7**       +24.4%           +3.6%
      rim_rate    16      fga          1.4         +20.7%           +3.8%
      ast_rate    32      fgm          6.2         +10.2%          +10.0%
      ftr        128      fga         11.6          +4.4%           +8.3%
      tov_rate   128      poss        10.0          +3.5%           +8.2%

  **The prediction held.** fg3 and rim came back fastest, tov and ftr slowest,
  in the order D257 required. The fitted k is also remarkably stable — ast_rate
  chose 32 in all 16 walk-forward seasons, tov_rate 128 in all 16 — which is
  itself evidence the constants are structural rather than fitted noise.

  **THE GAIN PATTERN IS THE INTERESTING PART** and it is internally coherent:
  for the FAST axes the current season carries the information (+24% over
  prior-only, only +3.6% over current-only), while for the SLOW axes the prior
  does (+8% over current-only, only +3.5% over prior-only). The two groups want
  opposite estimators, which is precisely why one window cannot serve both.

  **CONCRETELY.** Three-point rate reaches the prior's weight in **0.7 games** —
  a coach's instruction is in the data essentially immediately. Turnover rate
  takes **10 games** and free-throw rate **11.6**. Production's 10-game window,
  if it were reused for tendencies, would be ~14x too slow for spacing and about
  right for turnovers.

  **WHAT THIS IS NOT.** It is a measurement instrument, not an edge. Nothing here
  touches the market. D253b's result stands — 24 plausible features cost 80% of
  out-of-sample R^2 — so the estimator earns its place only if the
  complementarity test that follows finds the composition residual actually
  depends on lineup tendency mix.

  **NO PRODUCTION CHANGE.**

- D259 **TENDENCY MIX DOES NOT EXPLAIN THE ADDITIVE MODEL'S ERROR USABLY, AND IS
  WORTH NOTHING AGAINST THE MARKET. THE SYNERGY LINE CLOSES.**
  [scripts/d259_complementarity.py]

  The payoff test for D257/D258. Features are MIX, not level — level is already
  carried by `talent`, so only dispersion and fit are new information: weighted
  spread of three-point rate, playmaking and rim pressure across the available
  lineup; minutes concentration (HHI); `fit_sr`, the weighted correlation between
  a player's spacing and his rim pressure across the lineup, where NEGATIVE is
  the classic complementary shape (the spacers and the rim attackers are
  different people); and playmaking supply minus shot-creation demand.

  Strictly PIT: tendencies from D258's shrinkage on prior games only, weights
  from TRAILING minutes, never minutes actually played tonight — realised minutes
  would let the outcome choose its own weights. 43,750 team-games, 19 seasons.

  **BAR 1 — does mix explain the additive model's own error?**

      level only (w_fg3, w_ast, w_rim)   OOS R^2 **+0.00042**
      MIX only   (dispersion + fit)      OOS R^2 **-0.00007**
      level + mix                        OOS R^2  +0.00054
      permutation null (60 draws): median -0.00035, 95th -0.00015, p = 0.017

  **A DEFECT IN MY OWN VERDICT LOGIC, RECORDED BECAUSE IT ALMOST BECAME A
  FINDING.** The script printed "MIX MATTERS" on the null comparison alone. But
  the MIX-only R^2 is **NEGATIVE**. All p = 0.017 establishes is that mix is
  LESS BAD than shuffled mix. **Beating a permutation null is necessary, not
  sufficient**; a model with negative out-of-sample R^2 has no predictive value
  however it scores against its own shuffle. The honest statement is that mix
  carries a trace of real information and not enough to reach the mean.

  Mix adds **+0.00012** on top of level. Level itself is +0.00042 — positive,
  and about four ten-thousandths of the variance in a team's offensive residual.

  **BAR 2 — against the market:**

      vs CLOSE  OOS R^2 **-0.000229**      vs OPEN  OOS R^2 **-0.000082**

  Both negative: worse than predicting the mean. Nothing here is tradable.

  **WHAT THE WHOLE ARC ESTABLISHES.** Sean's structural criticism was correct at
  every level and it does not pay:

      D255  team pair effects persist across seasons (+0.0303) ...
      D256  ... but 44% was a TEAM effect and the interaction is ns (+0.0170);
            the team part is ridge shrinkage bias, zero at lambda = 0
      D257  tendencies are real and reliable (fg3 split-half +0.953) and switch
            at 16x different speeds by axis
      D258  a per-axis PIT estimator, constants fitted and stable
      D259  and mix explains neither the additive error nor the market

  **The rank-1 additive form is genuinely mis-specified, and closing that
  mis-specification does not help.** That is worth more than another null: it
  bounds an entire family — matchup interactions, style complementarity, lineup
  fit — rather than one feature. Item 5 of the research queue (regularised
  matchup interactions) is now CLOSED rather than merely bounded, on measurement
  rather than on prior.

  **WHAT SURVIVES.** D258's estimator is a correct instrument and stays in the
  repo; if a future question needs a PIT tendency it should use it rather than a
  window. And the honest reading of the arc is the one an external review gave
  at D255: this stack is mean-state, and the residual edge — if any — lives in
  state CHANGES, not in a better description of the average team.

  **NO PRODUCTION CHANGE.**

- D260 **THE SHIPPED MINUTES WINDOW IS 16th OF 32 AND A FITTED ESTIMATOR BEATS
  IT IN 16/16 SEASONS — AND IT BUYS NOTHING, FOR A REASON MORE GENERAL THAN
  ATTENUATION.** [scripts/d260_minutes_estimator.py, d260b_propagate.py]

  `composition.py` weights every player by `trailing_min`, "avg of last 10 games
  played (>=12 min)". Ten appears nowhere in the register as a measured choice.
  It is the SECOND of the two things production carries per player and it
  multiplies talent directly, so it was worth fitting.

  **L1 — THE ESTIMATOR.** 392,083 player-games, 32 arms (flat windows 3-40,
  EWMA half-lives 2-30, empirical-Bayes shrinkage k 1-64, and EWMA+shrinkage
  hybrids), walk-forward, predicting the next game's minutes given the player
  plays:

      winner **es3_2** — EWMA half-life 3, shrunk to the prior-season mean at k=2
      chosen in **16 of 16** walk-forward seasons
      MSE reduction vs shipped **+5.80%**, CI [+5.48%, +6.12%], better 16/16
      RMSE 5.781 -> 5.611 minutes
      **the shipped 10-game window ranks 16th of 32**

  Minutes are far more recency-driven than a flat 10-game mean allows: every one
  of the top ten arms uses a half-life of 3-5 games.

  **L2 — THE PROPAGATION, MEASURED (my arithmetic bound was wrong).** I estimated
  ~0.037 margin points by sqrt(n) scaling across players. Measured on 21,587
  games with PIT DARKO talent, the composition margin actually moves **0.1697
  points** on average (median 0.1355, p90 0.3598) — **4.61x my envelope**. The
  sqrt(n) assumption was wrong; minutes errors do not cancel across a roster the
  way independent errors would.

  **L3 — AND IT DOES NOT IMPROVE THE MARGIN.** Affine-recalibrated per season, as
  D245d requires:

      RMSE change **-0.00016 points**, CI [-0.00037, +0.00005], 11/19, **ns**
      composition RMSE is ~14.06 points, so this is a **0.001% change**
      **L1 5.80% -> L3 0.0011% — a ~5,100x attenuation**

  **THE MECHANISM, AND IT IS NOT THE 4.5x BLEND ATTENUATION.** The margin moves
  by 0.17 points and that movement is **almost orthogonal to what the margin got
  wrong**:

      implied corr(margin shift, margin error) = **+0.0086**

  Nine-tenths of one percent aligned. For scale: had the shift been orthogonal,
  RMSE would have WORSENED by +0.00173 (its own variance, delta^2/2R); had it
  been perfectly aligned it would have improved by 0.22. The observed -0.00016
  is a shift whose alignment barely exceeds its own variance.

  **THE DURABLE LESSON, which generalises past this entry.** An input
  improvement's value at the outcome depends on its ALIGNMENT with the outcome's
  error, not on the size of the input improvement. The usable form:

      dRMSE  ~  (var(shift) - 2*cov(shift, error)) / (2 * RMSE)

  so an input change is WORTH NOTHING unless `cov(shift, error) > var(shift)/2`,
  and is HARMFUL below that. This is computable in advance from a proposed
  change and the outcome's RMSE, and it explains D245d's L1/L2/L3-improve-L4-null
  pattern more fundamentally than the blend arithmetic does: the composition
  margin's 14-point error is dominated by irreducible game noise, not by input
  precision, so almost any input refinement lands orthogonal to it.

  **L4 NOT RUN, deliberately.** The implied log-loss effect is ~1.5e-05 nats
  against a typical MDE80 of ~1e-3 — about 68x below resolvable. A log-loss run
  would return a null describing the TEST's power rather than the estimator, and
  D245d already documented that failure mode.

  **WHAT SURVIVES.** `es3_2` is a strictly better minutes estimator and the
  finding that the shipped window ranks 16th of 32 stands on its own. But under
  G1 there is nothing to ship: the gain is unmeasurable at the emitted layer.
  Recorded so nobody re-fits this window expecting it to pay.

  **NO PRODUCTION CHANGE.**
