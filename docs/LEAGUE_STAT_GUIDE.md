# The League's Statistical Surface — a feature-selection guide
Written 2026-08-01 for Sean. Research memo; no production file touched.
Companions: STAT_INVENTORY.md (what we already measure), FEATURE_LEDGER.md
(verdicts), DECISIONS.md (register; this memo is **D135**).

**Read this first, because it is the answer to the question you asked.**
The broadcast stat you described — "this team's chance of winning if they enter
the quarter ahead" — is real, the league publishes it, and we can compute every
variant of it ourselves for free. It is also, at the team-specific level,
**almost entirely noise**, and the part that is not noise is just "this team is
good", which our model already knows. The numbers are in section (c). I would
not build a gate on it, and section (d) says what I would build instead.

---

## (a) THE MAP — where the league keeps its statistics

`nba_api` 1.11.4 wraps **137 endpoint classes**, exposing **373 result sets** and
**9,315 columns**. Enumerated from each class's own `expected_data` +
`__init__` signature (library metadata, no API hammering). Twenty-one live calls
were made to spot-check actual returned columns; all rate-limit-clean, no backoff
needed. Two endpoints came back **non-functional** against the live API (below).

| # | Family | Endpoints | Granularity | Key required param |
|---|---|---|---|---|
| 20 | **A. Per-game box scores** | BoxScore{Traditional,Advanced,FourFactors,Misc,Scoring,Usage,PlayerTrack,Hustle,Defensive,Matchups,Summary}{V2,V3}, HustleStatsBoxScore, InfographicFanDuelPlayer | per game × player AND per game × team | `game_id` |
| 9 | **B. Event / rotation** | PlayByPlay{,V2,V3}, GameRotation, WinProbabilityPBP, Video* | per event / per stint | `game_id` |
| 17 | **C. League dashboards** | LeagueDash{Team,Player}Stats, …Clutch, …ShotLocations, …Lineups, …PlayerBioStats, LeagueHustleStats{Player,Team}, SynergyPlayTypes, LeagueDashPt{Stats,Defend,TeamDefend}, LeagueDash{Player,Team,Opp}PtShot | season aggregate, ~30 filter axes each | none (season default) |
| 11 | **D. Player split dashboards** | PlayerDashboardBy{GeneralSplits,GameSplits,Clutch,LastNGames,ShootingSplits,TeamPerformance,YearOverYear}, PlayerDashPt{Pass,Reb,Shots,ShotDefend} | season × split | `player_id` |
| 10 | **E. Team split / on-off** | TeamDashboardBy{GeneralSplits,ShootingSplits}, TeamDashPt{Pass,Reb,Shots}, TeamPlayerOnOff{Details,Summary}, TeamPlayerDashboard, TeamDashLineups, LeaguePlayerOnDetails | season × split, per lineup, per on/off | `team_id` |
| 21 | **J. Standings / schedule / logs** | **LeagueStandingsV3**, LeagueStandings, PlayoffPicture, ISTStandings, ScheduleLeagueV2, Scoreboard{V2,V3}, League{GameLog,GameFinder}, {Player,Team}GameLog{,s}, CumeStats*, streak finders | season / per game row | none |
| 18 | **K. Reference / bio / career** | PlayerIndex, CommonAllPlayers, CommonPlayerInfo, **CommonTeamRoster** (incl. a `Coaches` set), TeamDetails, PlayerCareerStats, PlayerProfileV2, Franchise*, PlayerAwards | per player / per team / per season | mixed |
| 13 | **L. Derived metrics / leaders** | {Player,Team}EstimatedMetrics, DefenseHub, GravityLeaders, DunkScoreLeaders, LeagueLeaders, HomePage*, FantasyWidget | season | none |
| 7 | **G. Matchups** | LeagueSeasonMatchups, MatchupsRollup, PlayerVsPlayer, TeamVsPlayer, PlayerCompare, TeamAndPlayersVsPlayers | season × defender-offender pair | mixed |
| 7 | **I. Draft / combine** | DraftHistory, DraftBoard, DraftCombine{Stats,PlayerAnthro,DrillResults,SpotShooting,NonStationaryShooting} | per draft class | none |
| 2+2 | **F/H. Lineups, shot charts** | LeagueLineupViz, ShotChartLineupDetail, ShotChartDetail, ShotChartLeagueWide | per 5-man / per shot | mixed |

Machine-readable dump: scratchpad `endpoints.json`; readable: `sv_full_meta.txt`;
family map: `sv_families.json`.

### Two gaps worth knowing
1. **`WinProbabilityPBP` is DEAD.** Every `run_type` (`each second`, `default`,
   omitted) returns non-JSON — the endpoint no longer serves. So the league's own
   broadcast win-probability *curve* is not retrievable, even though the library
   still advertises its 15 columns (HOME_PCT, VISITOR_PCT, HOME_SCORE_MARGIN,
   PERIOD, SECONDS_REMAINING, HOME_POSS_IND). `GravityLeaders` fails identically.
   **Library metadata is not a live-availability guarantee — verify before you plan
   around an endpoint.**
2. **nba_api wraps only 2 team dashboards** (`ByGeneralSplits`,
   `ByShootingSplits`) against 7 on the player side. There is no wrapped
   `TeamDashboardByGameSplits` (by-period / by-score-margin) or
   `TeamDashboardByClutch`. Those URLs exist on stats.nba.com; using them means a
   raw request, not a library call.

---

## (b) YOUR BROADCAST STAT — found, and we can compute all of it for free

### It is published, as a standings column
`LeagueStandingsV3` (`leaguestandingsv3`, 115 columns, 1 call per season) carries
per-team, per-season **W-L record strings** for exactly the conditional you
described:

`AheadAtHalf` · `BehindAtHalf` · `TiedAtHalf` · `AheadAtThird` · `BehindAtThird` · `TiedAtThird`

plus a long tail of other broadcast splits: `HOME`, `ROAD`, `L10`, `Last10Home`,
`Last10Road`, `OT`, `ThreePTSOrLess` (games decided by ≤3), `TenPTSOrMore`,
`Score100PTS`, `OppScore100PTS`, `OppOver500`, `LeadInFGPCT`, `LeadInReb`,
`FewerTurnovers`, month-by-month records, streaks, games-back, clinch flags,
vs-conference and vs-division records.

Live-verified, 2024-25:
```
Cavaliers 64-18 | AheadAtHalf 50-9 | AheadAtThird 55-3 | BehindAtThird  9-15
Thunder   68-14 | AheadAtHalf 53-5 | AheadAtThird 61-2 | BehindAtThird  6-10
Lakers    50-32 | AheadAtHalf 43-7 | AheadAtThird 46-4 | BehindAtThird  4-27
```
So that is where the broadcast gets it. Note what the league does **not**
publish: margin buckets. "Up 5–9 entering the fourth" is not a column anywhere.
Nor is any of it shrunk, opponent-adjusted, or given a standard error.

### Everything else in the family we compute ourselves, at zero API cost
We already hold what is needed:

- **`data/raw/nba_api/playbyplayv3/` — 8,964 cached games.** Every action carries
  `scoreHome`, `scoreAway` and `period` as *explicitly labelled* fields. So the
  score at any instant is a direct read. Critically, **this path never touches
  `homeTeamId`**, so the D81/D99 `pbp['game'].get('homeTeamId') is None` landmine
  (which poisoned `defense_zone`, `possessions_v2` and `fit_v2_usage`) does not
  apply here. That is a rare clean surface in this codebase.
- **`data/raw/nba_api/boxscoresummaryv2/` — 4,914 games.** `LineScore` gives
  `PTS_QTR1..4` + `PTS_OT1..10` directly; a second path that cross-checks the first.
- **`nba_games`** (18,938 rows / 9,467 games) for outcome and home flag.
- **`lineup_stints`** (151,914 rows, with `t_start`/`t_end`/`margin`) for score
  state at arbitrary sub-quarter times.

I built the whole thing in one pass: 8,874 games parsed from PBP → 8,839 whose
period-boundary scores reconcile with `nba_games` finals (**99.77% agreement**,
20 mismatches dropped) → **8,181 regular-season games / 16,362 team-games,
2019-20 through 2025-26**. Zero API calls.

League base rates from our own data (these are the numbers the broadcast quotes):
| entering | win rate | n |
|---|---|---|
| ahead after Q1 | **.6636** | 7,817 |
| ahead at half | **.7413** | 7,945 |
| ahead after Q3 | **.8252** | 7,940 |

**So: yes. Any conditional-win-rate variant you want — any period, any margin
bucket, any rest/home/opponent interaction, at team or league level — is a
`GROUP BY` away, for free, forever.** The API adds nothing except the
pre-formatted record string.

---

## (c) THE CAUTION, QUANTIFIED — the team-specific version is mostly noise

This is the part that matters. A team-season gives you only ~38 games in which
the team led after three. That is not many coin flips.

### Raw team-specific conditional (the broadcast number)
`obs SD` = spread of the rate across team-seasons. `binomial-noise SD` = the
spread you would see if every team were identical and only luck differed.

| conditional | n/team-season | obs SD | noise SD | true SD | **noise share** | season-to-season r | split-half (Spearman-Brown) |
|---|---|---|---|---|---|---|---|
| ahead after Q1 | 37.2 | .1469 | .0797 | .1235 | 29.4% | +0.357 | 0.706 |
| ahead at half | 37.8 | .1231 | .0739 | .0984 | 36.1% | +0.333 | 0.621 |
| ahead after Q3 | 37.8 | .0996 | .0652 | .0754 | 42.8% | +0.296 | 0.529 |
| behind after Q3 | 37.8 | .0910 | .0649 | .0638 | 50.9% | +0.297 | 0.464 |
| ahead Q3 by **1–5** | 12.4 | .1650 | .1397 | .0878 | **71.7%** | +0.103 | **−0.388** |
| ahead Q3 by **6–10** | 10.7 | .1403 | .1265 | .0606 | **81.4%** | −0.033 | n/a |
| ahead Q3 by **11+** | 19.4 | .0510 | .0469 | .0201 | **84.4%** | +0.118 | **−0.186** |
| *(ref)* overall win% | 77.9 | .1505 | .0569 | .1393 | 14.3% | +0.519 | 0.868 |

The margin-bucketed version — the one a broadcast actually puts on screen — is
**72–84% pure sampling noise** and has **negative** split-half reliability. A
negative split-half means the first half of a team's season tells you *nothing*
about the second half of the same season on the same conditional.

### The kill shot: it is team quality wearing a costume
corr(conditional win rate, season win%): ahead-Q1 **+0.90**, ahead-Q3 **+0.81**,
behind-Q3 **+0.73**. Strip out season win% and what is left — the thing people
mean by "this team closes well" — is:

| conditional | residual SD | binomial noise SD | true residual SD | residual r (year-over-year) | residual split-half |
|---|---|---|---|---|---|
| ahead after Q1 | .0638 | .0797 | **0** (variance < 0) | −0.088 | −0.465 |
| ahead after Q3 | .0582 | .0652 | **0** (variance < 0) | −0.079 | −0.299 |
| behind after Q3 | .0622 | .0649 | **0** (variance < 0) | +0.026 | −0.132 |

The residual spread is **below the binomial floor in all three cases**. There is
no measurable team-to-team difference in closing ability once you know how good
the team is. This is the same arithmetic that killed team-specific home
advantage in D70, arriving at the same place from a different direction.

### Independent cross-check on the league's own published records
Not our reconstruction — the NBA's numbers. `leaguestandingsv3` for 2013-14
through 2024-25 = **360 team-seasons**, 12 calls, all clean. Residual
year-over-year correlation, net of season win%:

| split | residual r_yy |
|---|---|
| AheadAtHalf | +0.047 |
| **AheadAtThird** | **−0.019** |
| BehindAtThird | +0.012 |
| BehindAtHalf | +0.003 |
| **HOME** | **+0.000** ← re-derives D70 straight from league data |
| ThreePTSOrLess (games decided by ≤3) | +0.112, and **97.2% of its variance is noise** |
| LeadInFGPCT | +0.188 |
| FewerTurnovers | +0.112 |

The only two splits with any residual persistence are `LeadInFGPCT` and
`FewerTurnovers` — which are eFG% and TOV%, i.e. **two of the four factors we
already carry at 50% weight in production**. Nothing new anywhere in the family.

### Out-of-sample gate arithmetic
Walk-forward, 6 test seasons, n = 7,005. Task: predict the win **given** the team
is ahead after Q3 — the friendliest possible framing, because here the
conditional is actually observed.

| model | log loss |
|---|---|
| A. margin entering Q4 only | 0.38157 |
| B. + team strength | 0.37158 |
| C. + prior-season team closing residual | 0.37148 |

- B − A = **+0.00999** — team strength helps a lot, so the test has power.
- **C − B = +0.00011, bootstrap 95% CI (−0.00015, +0.00036).** A precise null.
  For scale, our sides gates have a realized MDE80 around +0.0024 (D127); this
  effect is an order of magnitude below the floor of what we can even detect.

### Which parts of the family are usable, and how
Our product is **pregame**. A pregame model cannot condition on "entered Q4
ahead" — that information does not exist when the bet is placed. So sort the
family three ways:

**(a) Usable pregame, as a team TRAIT — and measured worthless.**
The legitimate pregame translation is "does this team's *quarter-by-quarter*
scoring profile add anything to its overall scoring profile". I tested it:
trailing per-quarter net margin (expanding, min 10 games, opponent's too) added
to a trailing full-game net margin baseline, walk-forward over 6 seasons,
n = 14,178.

| model | out-of-sample RMSE (margin points) | Δ vs baseline |
|---|---|---|
| full-game margin only | 14.2785 | — |
| + Q1 split | 14.2676 | +0.0108 |
| + Q4 split | 14.2737 | +0.0047 |
| + all four quarter splits | 14.2717 | +0.0068 |

**+0.011 points of margin on a 14.28-point baseline = 0.08%**, and this against
a crude standalone baseline, not against production. Consistent with D84-C: at
the team-margin level the surface is exhausted.

**(b) Only usable for a live / in-game product, which we do not have.**
Everything genuinely conditional — win probability given the current score and
clock, `LeagueDashTeamClutch` (which does expose `ahead_behind`, `point_diff`,
`clutch_time` and `period` parameters and returns W/L), the whole
`PlayerDashboardByGameSplits` `ByScoreMargin`/`ByPeriod` family. These are
well-formed questions for a live model. We do not run one, and note that the
league's own live win-prob endpoint is dead, so we would be building the model
too, not just consuming it.

**(c) Pure broadcast garnish.** The margin-bucketed team-specific conditionals —
"the Lakers are 23-1 when leading by 10+ after three". 84% noise, negative
split-half, zero residual persistence. It is a sentence, not a statistic.

---

## (d) THE RANKED GATE SHORTLIST

Ranked by expected value per unit of effort, weighted (as instructed) toward the
two places we actually have live edges: **props** and **availability/effort**.
Calibration anchors: D133 props ramp **+0.039 CRPS** (our biggest recent win);
D46 schedule layer **+0.00598** held-out log loss (the only sides term that
survives out-of-sample); sides gate realized MDE80 ≈ **+0.0024** (D127). Any
item below +0.0024 on sides is, by construction, a long shot.

### Tier 1 — free, causal mechanism, availability/effort family

**1. National-TV flag → star availability.** `NATL_TV_BROADCASTER_ABBREVIATION`
in `BoxScoreSummaryV2.GameSummary`, **already sitting in our cache** on 4,914
games; present on **23.1%** of them. `ScheduleLeagueV2` gives it prospectively
for future slates.
*Hypothesis:* the NBA Player Participation Policy (2023-24 onward) explicitly
prohibits resting healthy star players in nationally televised games. That is a
hard written rule, not a behavioural regularity — so on ~23% of the slate the
DNP-rest hazard for stars is mechanically suppressed. Our availability model
currently has no such term.
*Gated on:* props `proj_min` / points CRPS (D133 harness), and DNP prediction.
*Expected:* **+0.003 to +0.010 points CRPS** on the affected subset; small
pooled. *Cost:* zero for history, one call/season prospectively.
*Not dead per ledger:* availability is the ACCEPTED family; nothing in the ledger
touches broadcast designation. **Best risk-adjusted item on the list.**

**2. Per-game hustle box score.** `BoxScoreHustleV2` / `HustleStatsBoxScore`
(`game_id`): deflections, contested shots (2PT/3PT), charges drawn, screen
assists, loose balls recovered, box-outs. We hold **2 files total** — season-level
league aggregates. Per-game hustle does not exist in our DB.
*Hypothesis:* deflections + loose balls per minute is the closest thing the
league publishes to a direct **effort** measurement, and effort is exactly what
the tank term (D73) is currently trying to infer *indirectly* from roster
composition and vet-minute shifts. Measuring the thing beats inferring it.
*Gated on:* first the D73 tank window (sides log loss, gp≥55); secondarily props.
*Expected:* **+0.002 to +0.005** on the tank-active subset; ~+0.0005 pooled.
*Cost:* 1 call/game; `HUSTLE_STATUS=1` on **74.8%** of cached games (so ~3,676
games are known-available before spending a call).
*Not dead:* STAT_INVENTORY lists hustle as "ingested; untested", and only at
season granularity — the per-game axis has never been built.

**3. Per-game physical load.** `BoxScorePlayerTrackV3` → `distance` (miles) and
`speed` (avg mph) per player per game. Live-verified working.
*Hypothesis:* minutes is the props bottleneck (D12: 11.2% of prop error; D133
paid +0.039 CRPS for a minutes *location* fix alone). Trailing minutes measures
exposure but not **load**. Cumulative distance over a trailing window is the
league's own load metric and should predict both next-game minutes and rest-DNP
risk beyond minutes.
*Gated on:* props minutes CRPS then points CRPS, D133 harness, pre-registered.
*Expected:* **+0.005 to +0.015 points CRPS** if load carries anything beyond
trailing minutes. *Honest caveat:* D133 ARM C is the cautionary precedent — a
richer two-axis minutes model **won on minutes and lost on points**. Pre-register
points as primary or this repeats that mistake.

**4. Published per-game usage share.** `BoxScoreUsageV3`: `usagePercentage`,
`percentageFieldGoalsAttempted`, `percentageAssists`, `percentageReboundsTotal`,
`percentagePoints` — the *realized* share, per player per game.
*Hypothesis:* we currently **fit** a latent softmax usage model (`v2_usage.npz`)
to drive star-out redistribution (D33/D39/D83, live, residual-calibrated at ATT
0.16 / MIN 0.39). The league publishes the measured quantity. Swapping an
estimated latent for a measured one is unlikely to produce a large point gain,
but it removes an entire artifact class — recall D99, where `fit_v2_usage`
silently discarded 49.9% of shots through the `homeTeamId` bug and the run looked
healthy. A measured share cannot have that failure mode.
*Gated on:* props attempts log-likelihood (the D83 endpoint, which held +0.049).
*Expected:* **+0.000 to +0.004** direct; the real return is robustness.
*Cost:* 1 call/game.

**5. `GameRotation.USG_PCT` and `PT_DIFF` — free, already cached.**
**5,028 cached `gamerotation` files** already on disk, and STAT_INVENTORY does
not list `USG_PCT` or `PT_DIFF` as used anywhere — we appear to consume only the
in/out times.
*Hypothesis:* per-stint usage-while-on-floor is a finer-grained star-out signal
than a per-game share, and it is already paid for.
*Gated on:* props attempts LL. *Expected:* **+0.001 to +0.005**.
*Cost:* **zero.** Do this one first purely because it is free.

### Tier 2 — free from cache, plausible, but on the crowded sides surface

**6. Blowout-intensity reweighting of trailing rates.** `LARGEST_LEAD`,
`LEAD_CHANGES`, `TIMES_TIED` from `BoxScoreSummaryV2.OtherStats` — **free, 4,914
games already cached.**
*Hypothesis:* garbage-time minutes contaminate trailing per-minute rates for
starters (deflated) and bench (inflated) alike. A game-level blowout index would
let the trailing estimators down-weight those minutes.
*Not dead — read the ledger carefully:* the rejected "blowout throttle" was a
throttle applied inside the **simulator** ("real in minutes, −0.14 corr, too
small for points"). Reweighting the **estimator's training minutes** is a
different construction on a different object. State this explicitly in the
pre-registration or a reviewer will correctly call it a re-run.
*Gated on:* props points CRPS. *Expected:* **+0.002 to +0.008**.

**7. Margin-VARIANCE model from possession-mix.** `PTS_FB`, `PTS_2ND_CHANCE`,
`PTS_PAINT`, `PTS_OFF_TO` (free, cached, 4,914 games; and per-player via
`BoxScoreMiscV3` with opponent columns).
*Hypothesis:* transition and second-chance share drive the **dispersion** of
game margin, not its mean. Every battery that has been run — D84-C's 49 features,
D127, D88 — targeted the **mean**. A calibrated per-game sigma is a structurally
different lever on log loss: sharpening the variance improves calibration even
when the mean is untouchable.
*Gated on:* sides log loss via a per-game sigma. *Expected:* **+0.001 to +0.003**
— borderline against MDE80 +0.0024, so power it properly or do not start.
*Why not already dead:* it is the one axis D84-C did not cover. Genuinely open.

**8. `ScheduleLeagueV2` — `isNeutral`, `arenaName`, `gameTimeEst`, `gameLabel`.**
1 call/season, 139 columns.
*Hypothesis:* mostly a **correctness** item. A neutral-site game modelled as a
home game is simply wrong, and we currently have no `isNeutral` flag. Tip-off
local time is an untested schedule axis, and D46's schedule layer is our only
term that survives out-of-sample — that is where marginal schedule information
has historically paid.
*Expected:* ~+0.0005 pooled, but it removes a small mis-specification.
*Cost:* trivial. Do it with the national-TV pull.

**9. Head coach identity and change dates.** `CommonTeamRoster` → `Coaches`
result set. Not in our DB at all.
*Hypothesis:* a mid-season coaching change is a genuine regime break that our
expanding-window estimators absorb only over ~4–6 games (per STAT_INVENTORY's
adaptation-speed section). An explicit reset would sharpen the window.
*Honest:* ~5–8 in-season changes per season league-wide. **n is too small to
clear a 95% gate.** Build it as a diagnostic and a regime marker, not as a
gated feature.

**10. `PlayerIndex.ROSTER_STATUS` + two-way/G-League assignment churn.**
1 call, availability family.
*Hypothesis:* two-way assignment and roster churn is availability information
that never appears in the injury PDF (a player is not "injured", he is in Sioux
Falls). Availability is our best-performing family.
*Expected:* small, concentrated on deep-bench props where our minutes model is
already weakest. *Cost:* trivial.

### Tier 3 — long shots, named so you know they were considered

**11. Contested vs uncontested FG% per game** (`BoxScorePlayerTrackV3`). A shot-
quality axis our `player_game_stats` lacks (we hold zone, not contest level).
**Long shot:** this is adjacent to D58/D99 defender-aware, which re-gated clean
at **+0.0037, CI (−0.0027, +0.0098), NS**, with more than half of the raw pooled
gain shown to be a level effect. Same neighbourhood, likely same answer.

**12. `LeagueDashPlayerPtShot` filtered by `touch_time_range` / `dribble_range` /
`close_def_dist_range` / `shot_clock_range`.** Season-level shot-quality
decomposition. **Long shot:** D37 already killed catch-and-shoot vs pull-up
splits. Different slicing of substantially the same information.

**13. `{Player,Team}EstimatedMetrics` (`E_OFF_RATING`, `E_PACE`, `E_USG_PCT`).**
Live-verified working. **Mostly dead on arrival:** D86-ENSEMBLE killed the talent
ensemble (rho 0.909 redundancy) and **D97's perfect-talent oracle bounds the
entire talent family at +0.004**. The only defensible use is as an early-season
stabiliser inside the October bridge (F6), where DARKO is thinnest.

**14. `LeagueDashLineups` / `TeamDashLineups` — as a VALIDATOR, not a feature.**
We hold 151,914 verified-clean `lineup_stints`, so the published lineup nets are
redundant as inputs. They are valuable as an **independent cross-check** against
the D81 bug class: if our stint-derived lineup net diverges from the league's
published net, we have a construction bug. Worth one call per season purely as a
standing regression test.

**15. `ISTStandings` — in-season tournament stakes.** Knockout games carry real
prize money (~$500k/player) and are a rare, dated, *positive*-motivation shock —
the mirror of the tank term. **Long shot on n:** roughly 50 knockout games per
season league-wide. Interesting; not gateable for several more seasons.

### Explicitly NOT on this list, and why
`WinProbabilityPBP` and `GravityLeaders` — both non-functional live (section a).
`DraftCombine*` and anything draft-slot-derived — D85 rejected the rookie
draft-slot prior at the incremental bridge gate (−0.00222, 0/3 seasons).

---

## (e) ALREADY DEAD — do not re-propose

Re-proposing any of these without a **named construction artifact** (per V2 §5.3)
is a waste of a gate slot.

| family | verdict | register |
|---|---|---|
| **Team-margin feature hunt generally** | 49-feature battery NULL — the surface is near-exhausted at the team-margin level | D84-C |
| **Talent ensembles** (DARKO+EPM+BPM, any weighting) | pooled −0.00006; error rho-bar 0.874, rho 0.909 redundancy | D86-ENSEMBLE, D94 |
| **The entire talent family** | a *perfect-talent oracle* bounds the whole family at **+0.004** — no talent metric can be worth more than that | D97 |
| **Pace** (team or opponent, props or sides) | dead; and its apparent gain was a **level** effect against a hardcoded constant | D88 |
| **Team-specific home advantage** | noise; re-derived here from league standings at residual r_yy **+0.000** | D70, D96 |
| **Altitude** | dead | D70, D96 |
| **Possession-level defence-conditioning** | killed at the pre-registered dev gate: −0.00108 NS, offence-only and team-def arms SIG HARMFUL | D127 |
| **Defender-aware props** | +0.0037 NS; >half of raw pooled gain is a level effect | D58, D99 |
| **def-RAPM team-aggregate (props)** | +0.0011 NS even after the D107 fix | D99, D107 |
| **Kalman rates (props)** | EWMA better by 0.90% rate-WMAE, CI[−0.00166,−0.00107], 3/3 seasons | D99, D128 |
| **Catch-and-shoot vs pull-up splits** | dead | D37 |
| **Opponent-pace, rest/B2B, teammate-out lift (props)** | all confirmed dead on retest | FEATURE_LEDGER |
| **Rookie draft-slot prior** | −0.00222, 0/3 seasons | D85-ROSTER |
| **Event-recency window blend** | retired; a zero measured four ways | D124 |
| **Minutes uncertainty-only widening (props)** | SIG **harmful**, 5/5 seasons | D133 ARM B |
| **Cold-start prior** | moves 0 of 6,148 games — structurally unreachable | D110, D123 |
| **Late-state layer** | reverted; held-out +0.00014 | D112 |
| ***and now:*** **team-specific in-game conditional win rates** | **residual variance below the binomial floor; OOS +0.00011, CI (−0.00015,+0.00036)** | **D135 (this memo)** |

### One standing methodological note this exercise reinforced
Every one of the broadcast splits in `leaguestandingsv3` is a **raw record with
no shrinkage and no standard error**, computed on n ≈ 12–40. Publication by the
league is not evidence of signal. Before any published split enters a gate, run
the three-line check used in section (c): observed SD, binomial-noise SD, and
residual-after-team-strength. If the residual variance sits below the binomial
floor, stop — there is nothing there to model.
