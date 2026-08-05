# Complete Stat & Feature Inventory
Everything measured, per entity, with source table/file and model use.
Companion: FEATURE_LEDGER.md (accept/reject verdicts), DECISIONS.md (register).

## PER-PLAYER — OFFENSE
| stat | source | granularity | used in |
|---|---|---|---|
| rim/mid/3PT attempt rates (per min) | player_game_stats (PBP zones) | per game, trailing | prop sim volume |
| rim/mid/3PT accuracy (FG%) | same | trailing pooled | prop sim makes |
| FT rate (FTA/min), FT% | same | trailing | prop sim |
| assists, rebounds (per min) | same | trailing | prop sim (ast decoupled D30) |
| turnovers (per min) | same | trailing | v1 fit dim |
| usage propensity u_i | v2_usage.npz (softmax fit) | season | star-out redistribution (attempts) |
| catch-and-shoot vs pull-up freq + 3P% | ptshot cache ('Pullups'!) | season | tested: splits dead (D37); C&S freq = dependency signal (G3) |
| Synergy play-type freq + PPP ×10 types | synergy cache | season | ingested; priors untested (queued) |
| drives, touches (paint/post/elbow), passes | ptstats cache | season | ingested; untested |
| points/game distribution | prop sim output | simulated | props (PIT 0.507) |
| 2K attributes ×38 | ratings_2k + crosswalk | per scrape | v1 priors (β-trusted per dim) |
| DARKO o_dpm/d_dpm/net | darko_dpm (daily snapshots) | daily | composition talent; validation ref |
| v1 Bayesian skill posteriors ×10 dims | v1_posterior.npz | season fit | skill backbone |
| age | darko_dpm | current | tested: curves real, no predictive gain (D28) |
| trailing minutes (avg/sd/hist) | player_game_stats | rolling 10 | prop exposure; composition weights |
| star-out redistribution (softmax attempts lift + positional minutes tilt, RESIDUAL-calibrated: ATT 0.16 / MIN 0.39 of full effect) | nbapred/engine/starout.py (D33/D39 science) | LIVE props path (D83; attempts LL +0.049 held-out); vs-EWMA-baseline double-count gotcha documented |
| draft slot / draft year | draft_history table (landed 2026-07-31, 8,434 rows) | rookie draft-slot prior REJECTED at incremental bridge gate (D85: -0.00222, 0/3 seasons); slot→DPM curve + PIT minutes map kept as assets |

## PER-PLAYER — DEFENSE (three orthogonal axes, D36)
| axis | stat | source | notes |
|---|---|---|---|
| 1. Rim/help | zone def-RAPM (rim/mid/3 suppressed) | def_rapm.py on PBP+lineups | **BROKEN: def_rapm.py:44 still has the D81 `homeTeamId` bug -> 100% of shots attributed to the home five, 49.9% wrong. Any published rating from it is invalid (D99). Clean re-fit still fails the props gate: +0.0011 NS.** |
| 1. Rim/help | blocks, DREB per min | player_game_stats | v1 dims |
| 1. Rim/help | DARKO d_dpm | darko_dpm | loads on rim/help |
| 2. On-ball | matchup-def rating (pts allowed vs opp baseline) | matchups cache (who-guarded-whom) | Amen Thompson/Dort top; orthogonal to axis 1 |
| 2. On-ball | steals per min | player_game_stats | v1 dim |
| 2. On-ball | defended-FG% by category (overall/3PT/2PT/<6ft/>15ft) | defended_fg table | individual contest quality |
| 3. Team context | hustle: contested shots (2/3PT), deflections, screen assists | hustle cache | ingested; untested |
| — | Synergy defensive play-types ×10 | synergy cache | ingested; untested |

## PER-TEAM
| stat | source | used in |
|---|---|---|
| four factors (eFG/TOV/OREB/FTr), opponent-adjusted | four_factors.py | production margin (50%) |
| single ortg/drtg ratings + team home-dev | team_ratings.py | early-season fallback; home edge |
| composition strength (avail × trail-min × DARKO) | composition.py | production margin (50%) |
| zone defense allowed (rim/mid/3 logit shifts) | props.team_zone_defense | **D99: this is a BOX path (player_game_stats zone columns are shooter-attributed) so it was NEVER D81-poisoned — no rebuild needed. Re-gated clean: NS (+0.0037 CI(-0.0027,+0.0098) on D58's construction), and >half its pooled gain is a stale-league-constant LEVEL effect. Stays out of live.** |
| pace (poss/game) | player_game_stats agg | tested: dead for props |
| cold-start prior (last season ×0.75) | production.py | early-season margin |
| play-type frequency profile | synergy cache | untested (queued) |
| cross-season FF carry (prior-season factor rows ×0.3×continuity, HARD STOP at 200 current rows) | production.py continuity_map + four_factors.py carry_rows | production margin, early season (D62/D63; +0.00097 pooled, gp[0,5) +0.0154) |
| tank composite (vet-min shift + rotation experimentation + shutdown listings + standings incentive; expanding-z, PIT) | nbapred/model/tanking.py (verbatim from apr_program.py) | production margin += k·tank_diff, gp≥55 only (D73, +0.00418) |
| urgency flag (within 3 GB of play-in cutoff, gp≥60) | BUILD B (scripts, mr_* results) | NS at gate (+0.00028 pooled) — FROZEN F5; re-gate unblocked feature-side by 2021-22 landing (2026-07-31) |
| October composition bridge cm_ps (001 preseason rosters + prior-season trailing min + PIT DARKO + outs) | scripts/rw_early_v1_gate.py; live-package cm_ps path | replaces the dead week-1 comp leg; market parity wk1 (D84-A) — FROZEN F6 Oct-2026 one-shot |
| late-state layer v1 (k·tsd + form5 + outs diff, active window) | scripts/rw_regimeD_latelayer.py | spec'd D84-D (active +0.0168→+0.0112); NOT shipped |

## GAME CONTEXT
schedule_features: days_rest, B2B, 3-in-4, games-last-7, travel km, TZ shift,
rest/travel advantage (tested: dead), home/away, opponent id.

## AVAILABILITY / OFFICIALS (PIT)
national-TV star-DNP suppression (D144, MEASURED FACT, not a shipped term):
core players are absent from **.1746** of national-TV games vs **.2010** of
local ones; within player-season **-0.02189 CI(-0.03030,-0.01295)**, with
rest/b2b/home/calendar/opponent controls **-0.02466**, season-mean t at 3 dof
**-0.02189 CI(-0.03505,-0.00872)**. The explicitly-labelled "Rest" DNP is
suppressed ~2.7x (.00114 vs .00305) but is only ~6% of the total. Panel:
`data/ad_natltv_panel.csv.gz`.
injury_reports_pit (official NBA injury-PDF feed **2018-12-17→2026-04-12,
125,695 rows, 8 seasons** — D49, backfilled to the SOURCE FLOOR by D170; the
standardised PDF genuinely does not exist before 2018-12-17, probed daily
against a control URL; per season report-days 110/156/139/168/169/168/167/159
for 2018-19..2025-26, 2018-19 partial by source) — feeds tank shutdown
listings + live OUT sets; game_inactives (official retro inactive lists,
**2006-07→2025-26, D170**; V2's `InactivePlayers` was populated the whole time
and the gap was INGEST, not source — it is empty for 2005-06 and earlier, which
IS the source floor. V2 endpoint broken ≥4/10/2025 so 2024-25's last 35 games
and 2025-26 come from BoxScoreSummaryV3 — see scripts/backfill_inactives_v3.py
and scripts/bf_inactives_hist_{fetch,load}.py, whose loader is STRICTLY
ADDITIVE so it cannot overwrite V3 rows with V2 blanks. Correctness: **0 of
77,313+ inactive rows appear in player_game_stats with minutes>0**);
game_officials (retro crews via boxscoresummary; same coverage gap, same
chained fix; pregame assignments poller in nbapred/ingest/referees.py —
ingested, no term shipped).
**UNOFFICIAL AVAILABILITY SOURCES — HUNTED AND REFUSED (D175).** Three free
surfaces were verified rather than assumed; none is ingested and no table was
created. (a) `boxscoretraditionalv3.comment` — ALREADY CACHED on disk
(`data/raw/nba_api/boxscoretraditionalv3/`, 36,595 games), carries the NBA's own
post-game DNP/DND/NWT memo at body-part granularity on **all 30 seasons,
93.3-97.5% of games, back to 1996-97**, and is 99.99% consistent with minutes
(9 of 125,728 DNP rows logged time; DND/NWT 0 of 26,453). **It attaches a reason
to 0.00% of `game_inactives` rows in all 20 seasons** — the two populations are
structurally disjoint, because from 2006-07 the NBA moved did-not-dress players
out of the box-score `players` array into `InactivePlayers`. Also note the
league changed convention at 2017-18: specific body-part memos run
643-1,434/season BEFORE it and 57-147/season after, so the OLD era has better
box-score reason granularity than the modern one. (b)
`boxscoresummaryv2.InactivePlayers` carries **no reason column** (headers are
PLAYER_ID/FIRST_NAME/LAST_NAME/JERSEY_NUM/TEAM_ID/TEAM_CITY/TEAM_NAME/
TEAM_ABBREVIATION) — confirmed, it is exactly `game_inactives`. (c) Wayback
snapshots of CBS/USA Today NBA injury pages are systematic on **2013-14..2017-18
only** (C48 80.9-96.0%) and are the only surface carrying all three of advance
notice, reasons and probabilistic status: news-break `Updated` date >=1 day
before the game on **90.6%** of rows, body-part reason on **100%**, and a
monotone status ladder (PROBABLE 88.1% played -> GTD 44.3% -> QUESTIONABLE
35.8% -> DOUBTFUL 12.5% -> OUT 7.1% -> OUT_SEASON 0.0%). Agreement on
OUT+OUT_SEASON **94.6% (2015-16) / 96.3% (2016-17)**; reason coverage of
`game_inactives` **36.4%/48.9%** overall and **56.4%/75.1%** on mpg>=25
players. **Piloted on three seasons it still does not pay: +0.100pp (2014-15),
+0.220pp (2015-16), -2.030pp (2016-17); mean -0.570pp, t(2)=-0.78, p=0.52,
wrong sign on 2 of 3, sign flips on dropping 2016-17.** NOT ingested. Raw pilot
artifacts under `data/raw/unofficial/`; full working in
`data/unofficial_injury_notes.md`.
**prosportstransactions.com is 403/Cloudflare-challenged on every path
including robots.txt, and Basketball-Reference's `*/gamelog/` is robots.txt
`Disallow`ed — neither may be scraped.**
**KNOWN DEFECT, NOT FIXED (D170, owner's call):** the PDFs name the Clippers
"LA Clippers" but `report_out_map()` maps through nba_api's full_name
"Los Angeles Clippers", so **all 2,514 Clippers report rows are silently
dropped from every T1/T2 OUT-set, in the certified seasons too**; 30 more rows
are lost to the heuristic parser's upper-case-surname assumption
("da Silva, Tristan" lands in the `team` column). Fixing either changes the
certified baseline's inputs.

## MARKET / CAPTURE (never model inputs — benchmark & edge measurement)
odds_market (spreads/totals/ML 2008-2026), odds_quotes (live logger),
softbook_props (Underdog lines), clv_log, flagged_windows (W1-W4),
injury_reports (poller armed, Oct).

## ENGINE / MODEL ARTIFACTS
v1_posterior.npz (10-dim skills + net), v2_usage.npz (usage u),
possessions_v2 (375k possessions — **DO NOT USE: D99 measured def_team=0 on 100% of rows and off_lineup/off_team agreement 0.5004; build() will never rebuild them**), lineup_stints (**151,914** stints, verified CLEAN of D81),
defended_fg, matchups, synergy/ptstats/ptshot caches.
player_game_stats coverage (**D152, 2026-08-02 — HISTORICAL BACKFILL**):
complete and verified (`scripts/backfill_history.py verify`: 0 missing games,
**0 zone-dead games** anywhere) for

| season | 002 games | expected | pgs rows |
|---|---|---|---|
| 2010-11 | 1230 | 1230 | 29,475 |
| 2011-12 | **990** | 990 (lockout) | 25,455 |
| 2012-13 | 1229 | 1229 (see note) | 31,347 |
| 2015-16 | 1230 | 1230 | 31,423 |
| 2016-17 | 1230 | 1230 | 31,564 |
| 2017-18 | 1230 | 1230 | 30,980 |
| 2018-19 | 1230 | 1230 | 30,801 |
| 2019-20 | **1059** | 1059 | **26,538** |
| 2020-21 | 1080 | 1080 | 28,859 |
| 2021-22 | 1230 | 1230 | 31,321 |
| 2022-23 | 1230 | 1230 | 31,542 |
| 2023-24 | 1230 | 1230 | 32,385 |
| 2024-25 | 1230 | 1230 | 32,515 |
| 2025-26 | 1230 | 1230 | 32,179 |

2019-20 was 24,281 rows / 971 games; the 88-game hole was **exactly** the
bubble seeding games `0021901231`..`0021901318` (no cached artifact of any
kind), now filled — the E1 bubble has player-level data for the first time.
2012-13's 1,229 is CORRECT: `0021201214` (BOS @ IND 2013-04-16) was cancelled
after the Boston Marathon bombing; BOS and IND have 81 GP, everyone else 82.
Further seasons (2014-15, 2013-14, 2009-10, 2008-09 ... back toward 1996-97)
were still landing when this was written — re-run
`python scripts/backfill_history.py verify` for the live picture.

**Source depth, probed not assumed (36 seasons, D152).** boxscoretraditionalv3
works to at least 1983-84; **playbyplayv3 is a HARD FLOOR at 1996-97** — for
1995-96 and every season below it the endpoint returns HTTP 200 with ZERO
actions (verified on 14 games). Since `possessions.py` only defers a game when
the PBP *file* is missing, an empty-but-present PBP would insert rows with all
zone counts 0 and silently degrade `four_factors` eFG to fgm/fga. **Do not pull
below 1996-97 without a code change.** `plusMinusPoints` is likewise 0.0 in
every season 1995-96 and below. GameRotation is partial at ALL depths (not a
season cutoff) and costs **31 s per absent game** (server hangs, then empty
body), so it is excluded from backfills — `lineup_stints` gains no new
coverage from this work.

## ADAPTATION SPEED (how fast rates absorb a playstyle change)
All trailing rates use recency-leaning weights over last ~15-20 games
(≥12-min games only): a TRUE sustained change reaches ~50% absorption in
~4-6 games, ~full in 10-15 (2-4 weeks). Applies equally to every action rate
(shot mix, dribble-heavy → TOV rate, getting stripped → TOV, steals, drives).
Faster adaptation (changepoint detection) is untested — and the Kalman/EWMA
gates warn: chasing change faster than noise allows COSTS accuracy. Empirics
(D38): 88% of players show ≥1 sustained 10pt+ zone-mix shift per ~season;
median window-to-window drift 0.07 — most "shifts" are noise-sized.

## GAME-STATE / QUARTER SCORES (added D135) — HELD LOCALLY, ZERO API COST
| stat | source | granularity | status |
|---|---|---|---|
| score at any instant / any period boundary | `data/raw/nba_api/playbyplayv3/` (8,964 games) — actions carry `scoreHome`/`scoreAway`/`period` as LABELLED fields, so this path is **immune to the D81/D99 `homeTeamId is None` bug** | per event | derived on demand; 99.77% reconciliation with `nba_games` finals |
| quarter scores PTS_QTR1..4 + PTS_OT1..10 | `boxscoresummaryv2` cache `LineScore` (4,914 games) | per game × team | second, independent path |
| conditional win rate given lead entering period N (any margin bucket) | computed from the above + `nba_games` | any | **measured and REJECTED at team level — see D135 / LEAGUE_STAT_GUIDE.md (c). Residual variance below the binomial floor; OOS +0.00011 CI(-0.00015,+0.00036)** |
| per-quarter net margin as a pregame team trait | same | trailing, expanding | tested: +0.011 RMSE on a 14.28 baseline (0.08%) = noise. Consistent with D84-C |
| PTS_PAINT / PTS_2ND_CHANCE / PTS_FB / PTS_OFF_TO / LARGEST_LEAD / LEAD_CHANGES / TIMES_TIED | `boxscoresummaryv2` cache `OtherStats` (4,914 games) | per game × team | **cached, NOT in DuckDB, NOT used** — shortlist items 6 & 7 |
| ATTENDANCE, GAME_TIME (duration) | `boxscoresummaryv2` `GameInfo` | per game | cached, unused |
| NATL_TV_BROADCASTER_ABBREVIATION | `boxscoresummaryv2` `GameSummary`; 1,134 of 4,914 = **23.08%** | per game | **INGESTED D144** -> `data/ad_natl_tv.csv`. **ERA-AVAILABLE E4+E5+E6 ONLY** (cache starts 2022-23; structurally absent on E3 = half the legacy holdout). MEASURED: within-player-season core-DNP **-0.02189 CI(-0.03030,-0.01295) SIG** on national TV, robust to placebo/weekday/calendar. **NOT the Player Participation Policy** — largest in the PRE-policy season, DiD at the 2023-24 boundary +0.00979 ns, marquee gradient ABC/ESPN+ -0.062 -> NBA TV -0.012. **NOT GATED (V2 §5.5)**: the props universe conditions on having PLAYED, and the minutes shift conditional on playing is ~0.007 min |
| GameRotation stints — `USG_PCT`, `PT_DIFF`, `IN/OUT_TIME_REAL` | `data/raw/nba_api/gamerotation/` — 5,028 files, **336,823 stints / 5,026 games / 107,584 player-games** | per player-stint | **INGESTED + VALIDATED D144** -> `data/ad_rotation.csv.gz`, `ad_rotation_pg.csv.gz`. Rotation seconds vs box seconds **99.79% within 5 s** (MAE 5.17 s); `PLAYER_PTS` exact **99.80%**. Regular-season coverage 2019-20 498 / 2020-21 218 / 2021-22 328 / 2022-23 566 / 2023-24 1195 / 2024-25 765 / 2025-26 1230 |
| **STARTER FLAG** (`first_in <= 0.5` tenths) — derived; `player_game_stats` has NO starter column | same | per player-game | **NEW INFORMATION, D144.** Exactly 5 starters on **99.682%** of 10,052 cached team-games. Frozen artifact `data/ad_role_flags.npz` (102,331 rows / 1,117 players), keyed `(player_id, ord)`, PIT-safe by construction and pinned by `tests/test_role_state.py`. Carries a **2.4-minute** projection error the minutes history does not see (see FEATURE_LEDGER) |
| garbage-time share of a player's minutes | rotation stints x `lineup_stints` running margin (\|margin\|>=20 and t>=Q4) | per player-game | **BUILT D144, NOT GATED** -> `data/ad_garbage_pg.csv.gz`. Mean share 0.0931; 14.2% of player-games >20% garbage. This is D135 shortlist item 6 at PLAYER rather than game granularity |

## LEAGUE SURFACE NOT YET INGESTED (D135 sweep)
nba_api 1.11.4 = **137 endpoint classes / 373 result sets / 9,315 columns**.
Full dump in scratchpad `endpoints.json`; family map in LEAGUE_STAT_GUIDE.md (a).
Families where we hold **nothing or only season-level aggregates**:
- **Per-game tracking** `BoxScorePlayerTrackV3` — speed, distance, touches,
  passes, contested/uncontested FGA+FG%, rebound chances, defendedAtRim. LIVE-VERIFIED.
- **Per-game hustle** `BoxScoreHustleV2` — deflections, contested shots, charges,
  screen assists, loose balls, box-outs. We hold **2** season-level files only.
  `HUSTLE_STATUS=1` on 74.8% of cached games.
- **Per-game usage** `BoxScoreUsageV3` — realized share of team FGA/AST/REB/PTS.
  (We currently FIT this latent via `v2_usage.npz`.)
- **Per-game advanced / misc / scoring** `BoxScore{Advanced,Misc,Scoring}V3` —
  per-game ORtg/DRtg/pace/PIE, points off TO / 2nd chance / fast break / paint
  with OPPONENT columns, % of points by zone and % assisted.
- **Team splits** `TeamDashboardByGeneralSplits` (Overall/Location/Month/DaysRest/
  PrePostAllStar/WinsLosses); `LeagueDashTeamClutch` (params `ahead_behind`,
  `point_diff`, `clutch_time`, `period`). Not wrapped by nba_api: team
  ByGameSplits / ByClutch (raw URL only).
- **Standings splits** `LeagueStandingsV3` (115 cols) — AheadAtHalf/AheadAtThird/
  BehindAt*, HOME/ROAD/L10, ThreePTSOrLess, LeadInFGPCT, FewerTurnovers, monthly,
  streaks, GB, clinch. **All measured worthless net of team strength (D135).**
- **Schedule** `ScheduleLeagueV2` (139 cols) — isNeutral, arenaName/City/State,
  gameTimeEst, gameLabel, weekNumber, full national/home/away broadcast tree.
- **Roster/coach** `CommonTeamRoster` (has a `Coaches` result set — head coach
  identity and change dates are absent from our DB entirely); `PlayerIndex`
  (ROSTER_STATUS); `CommonPlayerInfo`.
- **Derived** `{Player,Team}EstimatedMetrics` (E_OFF/DEF_RATING, E_PACE, E_USG) —
  LIVE-VERIFIED, but bounded dead by D97's talent oracle (+0.004 for the whole family).
- **DEAD ENDPOINTS (verified live, return non-JSON):** `WinProbabilityPBP`,
  `GravityLeaders`. Library metadata is NOT a live-availability guarantee.

## DUCKDB TABLES (28, as of D135)
bet_paper, bet_quotes_panel, clv_log, darko_dpm, darko_history, defended_fg,
draft_history, epm_history, epm_history_daily, game_inactives, game_officials,
injury_reports, injury_reports_pit, lineup_stints, nba_games, nba_players,
odds_hist_sbr, odds_market, odds_open, odds_quotes, player_game_stats,
player_states, player_xwalk, possessions_v2, ratings_2k, schedule_features,
state_shocks, v3_predictions.
`nba_games` 18,938 rows / 9,467 games, seasons 2019-20..2025-26.
`player_game_stats` 234,308 rows / 27 cols. `schedule_features` 2,460 rows
(**one season only** — 2025-26; the ledger's "sched features absent for 24-25"
note generalises: this table is far thinner than the game corpus).

## D171 COVERAGE CENSUS — every feed, per season, 2007-08..2025-26

Measured by `scripts/d171_gap_census.py` (read-only), 2026-08-04, AFTER D170's
backfill. `inact%` = share of the season's games with an inactive list;
`report` = report-days / game-days; `darko%` = PIT minute coverage; `poss%` /
`stints%` / `refs%` = share of games present in `possessions_v2` /
`lineup_stints` / `game_officials`.

```
season   games  inact%  report_days  darko%  poss%  stints%  refs%  sched  odds_mkt  sbr
2007-08   1230  100.0     0/162      100.0    0.0     0.0     0.0      0     1316   1316
2008-09   1230  100.0     0/163      100.0    0.0     0.1     0.0      0     1315   1315
2009-10   1230  100.0     0/164      100.0    0.0     0.1     0.0      0     1312   1312
2010-11   1230  100.0     0/164      100.0    0.0     0.0     0.0      0     1311   1311
2011-12    990   99.0     0/120      100.0    0.0     0.0     0.0      0     1074   1074
2012-13   1229  100.0     0/163      100.0    0.0     0.0     0.0      0     1313   1314
2013-14   1230  100.0     0/163      100.0    0.0     0.0     0.0      0     1319   1319
2014-15   1230   99.9     0/162      100.0    0.0     0.0     0.0      0     1311   1311
2015-16   1230  100.0     0/161      100.0    0.0     0.1     0.0      0     1316   1316
2016-17   1230  100.0     0/162      100.0    0.0     0.0     0.0      0     1309   1309
2017-18   1230  100.0     0/168      100.0    0.0     0.0     0.0      0     1312   1312
2018-19   1230  100.0   107/168      100.0    0.0     0.6     0.0      0     1312   1312
2019-20   1059  100.0   150/150      100.0   46.8    48.2     0.0      0     1142   1143
2020-21   1080  100.0   135/140      100.0   20.2    20.2     0.0      0     1171   1171
2021-22   1230  100.0   164/165      100.0   26.5    26.5     0.0      0     1321   1323
2022-23   1230  100.0   163/164      100.0   46.0    46.0   100.0      0     1320    664
2023-24   1230  100.0   160/160      100.0   96.8    96.8   100.0      0     1319      0
2024-25   1230  100.0   162/163      100.0   62.1    62.1   100.0      0     1321      0
2025-26   1230   99.8   155/164      100.0   99.6    99.6    99.8   1230     1322      0
```

**THE ROW THAT MATTERS MOST IS WHICH OF THESE THE MODEL ACTUALLY OPENS.**
Grepped over the shipped path (`fit_production` + `CompositionModel` +
`four_factors` / `travel` / `october_bridge` / `latestate` / `tanking`), the
certified model reads **exactly seven tables**:
`nba_games`, `player_game_stats`, `nba_players`, `darko_history`, `darko_dpm`,
`injury_reports_pit`, `odds_market` — plus `game_inactives`, which is read
NOWHERE inside `nbapred/` and enters only through the caller's out-set.

Consequences, and this is the whole answer to "what else is missing from the
old seasons":
* `game_inactives` (99.0-100% everywhere) and `darko_history` (100.0%
  everywhere) are **CLOSED** by D170. There is no DARKO ceiling.
* `injury_reports_pit` before **2018-12-17** is a **PERMANENT SOURCE FLOOR** and
  is the ONLY remaining model-visible asymmetry. D171 prices it at
  **-0.741pp normalized** (paired T2 vs T2i, n=9,516, season-clustered
  t = -5.02, K=8, 7/8 seasons improve), with an old-era range of
  **-0.25pp to -0.74pp**.
* `possessions_v2`, `lineup_stints`, `game_officials`, `schedule_features`,
  `defended_fg`, `epm_history*`, `odds_open`, `odds_hist_sbr`, `ratings_2k`
  are **NOT read by the certified model**. Their (large) coverage asymmetries
  bound FUTURE work; they are not part of any current per-season gap. In
  particular `schedule_features` is still 2025-26-only (D135 confirmed), and
  `possessions_v2`/`lineup_stints` are absent before 2018-19 and only 62.1%
  complete even on 2024-25.
* `odds_market` is complete on all 19 seasons (~1,310/season) — the benchmark
  itself has no era gap. Panel DEPTH does (D163: 2 books early vs up to 9
  operators modern), but that is an EXECUTION-study asymmetry, not a model one.
