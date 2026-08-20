# ERAS — the measured era table (D139)

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

Status: REFERENCE. Every future gate cites an era code from this file.
Produced by `scripts/era_measure.py` (read-only on data/nba.duckdb) →
`data/era_signatures.json`. Encoded for code use in
`nbapred/eval/splits.py::ERAS`.

**Why this file exists.** Our whole campaign used ONE partition — dev
2023-24..2025-26, holdout 2021-22..2022-23 — whose boundary (2023-10-24) is
ALSO the start of the Player Participation Policy, the In-Season Tournament and
the new CBA apron regime. "Passed dev, failed holdout" was therefore ambiguous
between OVERFITTING (D111) and a real ERA EFFECT (D70). Nothing in the register
could tell them apart. This table is the missing reference.

**Rule of construction:** every claim below is a number computed from
`nba_games` / `player_game_stats`, not a recollection of NBA news. Where the
data contradicted the received story, the data won and the correction is
flagged **CORRECTION**.

---

## 1. The table

| era | window | seasons | scorable? | what changed |
|---|---|---|---|---|
| **E-3** | ≤ 2010-11 | **ALL 15 landed (D160): 1996-97..2010-11** | **2007-08..2010-11 YES (D160)**; 1996-97..2006-07 NO — see below | pre-lockout CBA; **fewest players used (9.99, 1996-97) and lowest core-DNP (.1051, 2000-01) measured**; **home margin +2.57..+3.88 in all 15 seasons, no overlap with the current corpus**; **highest travel measured anywhere, 941.6 km in 2003-04** |
| **E-2** | 2011-12 → 2013-14 | **all 3 landed** (2011-12/2012-13 D152, **2013-14 D160**) | all three **YES** | post-lockout CBA; **pre-3PT-boom** (3PA share .226/.243/.260 vs .384 in 2019-20); pace trough (93.9-96.5 poss); **home margin +2.6 to +3.2, above anything in the current corpus** |
| **E-1** | 2014-15 → 2018-19 | **all 5 landed** (2015-16..2018-19 D152, **2014-15 D160**) | **all five YES (D160 completes it)** | 3PT ramp (.268 → .285 → .316 → .337 → .359 share); pace recovery to 102.5; travel 868-904 km — **no longer the highest measured (D160: 2003-04 is 941.6)** |
| **E0** | 2019-10-22 → 2020-03-11 | 2019-20 pre-shutdown | **YES (new, D152)** | normal league; last pre-COVID basketball |
| **E0H** | 2020-03-12 → 2020-07-29 | — | n/a | hiatus, zero games |
| **E1** | 2020-07-30 → 2020-10-11 | 2019-20 bubble | **NO** | 88 seeding games at ONE site, no crowds |
| **E2** | 2020-12-22 → 2021-05-16 | 2020-21 | **NO** | 72-game compressed season, no/limited crowds, play-in pilot |
| **E3** | 2021-10-19 → 2022-04-10 | 2021-22 | yes | crowds return; **Omicron** (sub-era E3-OMI) |
| **E4** | 2022-10-18 → 2023-04-09 | 2022-23 | yes | no COVID distortion — but the corpus's **home-advantage outlier** |
| **E5** | 2023-10-24 → 2024-04-14 | 2023-24 | yes | **PPP + In-Season Tournament + new CBA**. THE LEGACY DEV/HOLDOUT BOUNDARY |
| **E6** | 2024-10-22 → 2026-04-30 | 2024-25, 2025-26 | yes | apron regime maturing |

Sub-era: **E3-OMI** 2021-12-13 → 2022-01-02 (ISO 2021-W50..W52).

**Certified eval corpus = 5 seasons, 6,148 games (E3, E4, E5, E6).** Unchanged
by this file and by D160.

**SCORABLE (D160, data sufficiency per `scripts/history_scorable.py`) = 19
seasons, 2007-08..2025-26, CONTIGUOUS, and every one of them has market odds.**
16 poolable + 3 separate strata (1998-99 is a fourth lockout stratum but has no
odds and is not scorable). Below 2007-08 the binding constraint is no longer
box scores — all 30 seasons back to 1996-97 are now at 100% coverage — it is
(a) **`odds_market` starts at 2007-08**, and (b) ~~**`darko_history` starts
2003-10-29**~~ **`darko_history` now starts 1996-11-01 (D170, 2026-08-04)**,
below which `fit_production` REFUSES outright ("darko_history empty before
<date>; refusing snapshot fallback"). **The DARKO floor is therefore NO LONGER
BINDING anywhere in the box-score corpus: all 30 seasons back to 1996-97 are
now fittable.** `odds_market` at 2007-08 is the only remaining constraint on
the SCORABLE frame, so the scorable frame is unchanged at 19 seasons and this
entry does not widen it. See §7.

---

## 2. Measured signatures, per season

All figures regular season only (`game_id LIKE '002%'`).

| season | era | n | home margin | home WR | travel km/team-game | b2b | 3-in-4 | mean rest (d) | rest≥3 | repeat-opp | poss/team-g | pts/team-g | players used | core-DNP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1996-97 | **E-3** | 1189 | +2.573 | .5753 | **911.1** | .245 | .305 | 2.09 | .227 | .014 | 92.84 | 96.90 | **9.99** | .1418 |
| 1997-98 | **E-3** | 1189 | +2.946 | .5955 | 898.7 | .233 | .289 | 2.09 | .226 | .015 | 93.06 | 95.57 | 10.09 | .1396 |
| 1998-99 | **E-3** | **725**¶ | **+3.434** | **.6234** | 861.8 | **.335** | **.494** | **1.81** | **.114** | .030 | **91.61** | **91.58** | 10.29 | .1171 |
| 1999-00 | **E-3** | 1189 | **+3.540** | **.6106** | **919.4** | .248 | .315 | 2.08 | .218 | .020 | 95.67 | 97.47 | 10.22 | .1064 |
| 2000-01 | **E-3** | 1189 | +2.916 | .5980 | 892.1 | .241 | .307 | 2.08 | .224 | .010 | 93.95 | 94.81 | 10.08 | **.1051** |
| 2001-02 | **E-3** | 1189 | **+3.399** | .5913 | 892.0 | .251 | .319 | 2.08 | .227 | .020 | 93.33 | 95.48 | 10.05 | .1508 |
| 2002-03 | **E-3** | 1189 | **+3.884** | **.6283** | 885.2 | .251 | .333 | 2.08 | .214 | .027 | 93.76 | 95.08 | 10.07 | .1130 |
| 2003-04 | **E-3** | 1189 | **+3.599** | **.6140** | **941.6** | .248 | .303 | 2.07 | .227 | .013 | 92.62 | **93.40** | 10.05 | .1397 |
| 2004-05 | **E-3** | 1230 | +3.126 | .6049 | 886.9 | .248 | .317 | 2.07 | .218 | .019 | 93.65 | 97.20 | 10.14 | .1519 |
| 2005-06 | **E-3** | 1230 | +3.368 | .6033 | 868.5 | .235 | .314 | 2.08 | .217 | .019 | 93.16 | 97.01 | 10.13 | .1266 |
| 2006-07 | **E-3** | 1230 | +2.986 | .5911 | 902.6 | .248 | .320 | 2.07 | .221 | .017 | 94.57 | 98.74 | 10.20 | .1614 |
| 2007-08 | **E-3** | 1230 | +3.406 | .6008 | 879.1 | .243 | .324 | 2.07 | .213 | .018 | 94.81 | 99.92 | 10.12 | .1341 |
| 2008-09 | **E-3** | 1230 | +3.251 | .6081 | 882.0 | .236 | .303 | 2.08 | .224 | .018 | 94.20 | 99.95 | **10.01** | .1589 |
| 2009-10 | **E-3** | 1230 | +2.732 | .5943 | 881.7 | .245 | .308 | 2.08 | .224 | .009 | 95.10 | 100.45 | 10.09 | .1371 |
| 2010-11 | **E-3** | 1230 | **+3.167** | **.6041** | 872.2 | .244 | .321 | 2.08 | .216 | .011 | 94.61 | 99.55 | **10.22** | **.1451** |
| 2011-12 | **E-2** | **990**§ | +2.820 | .5859 | 878.0 | **.310** | **.463** | **1.88** | **.123** | .019 | **93.89** | **96.26** | 10.48 | **.1436** |
| 2012-13 | **E-2** | 1229‡ | **+3.226** | **.6119** | 863.4 | .233 | .319 | 2.07 | .210 | .018 | 94.57 | 98.14 | 10.48 | .1507 |
| 2013-14 | **E-2** | 1230 | +2.597 | .5805 | 867.1 | .229 | .311 | 2.08 | .200 | .018 | 96.51 | 101.01 | 10.41 | .1365 |
| 2014-15 | **E-1** | 1230 | +2.407 | .5748 | 888.1 | .236 | .326 | 2.08 | .189 | .013 | 96.44 | 100.01 | 10.56 | .1980 |
| 2015-16 | **E-1** | 1230 | +2.675 | .5886 | 868.3 | .217 | .300 | 2.08 | .191 | .016 | 98.23 | 102.67 | 10.60 | .1488 |
| 2016-17 | **E-1** | 1230 | +3.149 | .5837 | **903.7** | .200 | .292 | 2.07 | .176 | .017 | 98.82 | 105.59 | 10.62 | **.1300** |
| 2017-18 | **E-1** | 1230 | +2.106 | .5789 | 889.9 | .176 | .233 | 2.16 | .220 | .012 | 99.56 | 106.33 | 10.61 | .1693 |
| 2018-19 | **E-1** | 1230 | +2.724 | .5927 | **877.5** | .162 | .222 | 2.16 | .223 | .020 | 102.52 | 111.18 | 10.61 | .1876 |
| 2019-20 | E0+E1 | 1059 | +2.130 | .5515 | 820.1† | .149 | .223 | 3.65† | .194 | .011 | 102.82 | 111.80 | 10.57 | .1836 |
| 2020-21 | E2 | 1080 | **+0.944** | .5435 | 830.2 | **.212** | **.337** | **2.03** | **.138** | **.086** | 101.42 | 112.09 | 10.67 | .2271 |
| 2021-22 | E3 | 1230 | +1.723 | .5439 | 842.1 | .172 | .253 | 2.12 | .193 | .035 | 100.44 | 110.62 | 10.58 | .2394 |
| 2022-23 | E4 | 1230 | **+2.500** | **.5805** | 800.8 | .163 | .248 | 2.12 | .193 | .058 | 101.65 | 114.69 | 10.53 | .2023 |
| 2023-24 | E5 | 1230 | +2.146 | .5431 | 826.3 | .172 | .250 | 2.12 | .195 | .049 | 100.81 | 114.21 | 10.73 | .2063 |
| 2024-25 | E6 | 1230 | +1.692 | .5423 | 850.9 | .183 | .268 | 2.12 | .187 | .048 | 101.16 | 113.83 | 10.69 | **.2502** |
| 2025-26 | E6 | 1230 | +1.726 | .5520 | 855.1 | .179 | .268 | 2.12 | .193 | .040 | 101.84 | 115.61 | 10.83 | .2435 |

† 2019-20 pooled figures mix E0 and E1; E0 alone is home margin **+2.174**,
travel **894.4** km/team-game. The table previously read 945.2, computed
BEFORE the D140 neutral-site fix and fictitious — E1's true travel is 0. The
figure is now the re-measured **820.1** (which is exactly the predicted
894.4 × 971/1059). See §3.

§ **2011-12 is the lockout season — 990 games (66 per team)**, b2b **.310**
(vs .212 in 2020-21, .172 in 2021-22), 3-in-4 **.463** (vs .337), mean rest
**1.88 d** (vs 2.03), rest≥3 **.123** (vs .138). Treat exactly like 2020-21: a
separate stratum, never pooled.

**CORRECTION (D160).** This footnote used to call 2011-12 "the most
schedule-distorted season we hold" and "the pace/scoring trough of the whole
corpus". Both statements were true of the corpus as it stood and are **false
now that 1996-97..2009-10 have landed**, and both are superseded by the SAME
season: ¶ **1998-99, the 50-game lockout, is worse than 2011-12 on EVERY load
channel** — b2b **.335** vs .310, 3-in-4 **.494** vs .463, mean rest **1.81 d**
vs 1.88, rest≥3 **.114** vs .123 — and it is also the pace and scoring trough
of the entire 30-season series (**91.61** poss, **91.58** pts/team-game, against
93.89 / 96.26). 1998-99 gets the same treatment 2011-12 gets: a separate
stratum, never pooled. It has no odds, so this costs the betting lane nothing.
Two more superlatives in this table moved as well: the lowest core-DNP measured
anywhere is now **.1051 (2000-01)**, not 2016-17's .1300, and the fewest
players used is **9.99 (1996-97)**, not 2010-11's 10.22.

‡ 2012-13 played **1,229** games, not 1230, and that is correct, not a gap:
`0021201214` (BOS @ IND, 2013-04-16) was cancelled after the Boston Marathon
bombing and never made up. Verified in-data — 1214 is the only gap in the id
sequence and exactly BOS and IND have 81 GP.

**D152 UPDATE (2026-08-02).** Rows for **2012-13, 2017-18, 2018-19** are new
(seasons landed by the historical backfill). Two corrections to rows that were
already here:
* **2019-20 changed** because the 88 bubble games now have box data for the
  first time (`box_games` 971 → 1059). poss 102.65 → **102.82**, pts 111.45 →
  **111.80**, players-used 10.55 → **10.57**, core-DNP .1660 → **.1836**, and
  travel picks up the D140 fix (945.2 → 820.1). The E1 bubble sub-row in §3
  is now measurable: 88 games, home margin **+1.648**, home WR .5568,
  travel **0.0** as constructed.
* **2024-25 / 2025-26 travel in the rows above are STALE**, independently of
  this backfill: re-measuring with today's `era_measure.py` gives **844.4** and
  **848.3** (not 850.9 / 855.1). Verified by a controlled re-run with the
  pre-D152 `arenas.csv`, which reproduces 844.4/848.3 — so this is pre-existing
  drift in the registered table, NOT an effect of the new arena rows. The
  registered D139 values are left in place; treat the re-measured numbers as
  authoritative and re-register them with the next era decision.
* `data/arenas.csv` gained 7 historical franchises (NOH, NJN, SEA, NOK, VAN,
  CHH, WSB). This was load-bearing, not cosmetic: `era_measure.py` sums
  `st["travel_km"]` WITHOUT checking `travel_valid`, so a franchise missing
  from arenas.csv silently contributes travel 0.0. 2012-13 measured **802.7**
  km/team-game before NOH was added and **863.4** after — a 60.7 km artefact
  from one team's 82 games. Any season pulled below 2013-14 needs its
  franchises present before its travel figure means anything.

**D160 UPDATE (2026-08-03) — 16 NEW ROWS, AND THE TABLE IS NOW COMPLETE FROM
THE PBP FLOOR.** Rows for **1996-97, 1997-98, 1998-99, 1999-00, 2000-01,
2001-02, 2002-03, 2003-04, 2004-05, 2005-06, 2006-07, 2007-08, 2008-09,
2009-10, 2013-14, 2014-15** are new. §2 now covers **all 30 seasons
1996-97..2025-26 with ZERO gaps**, every one at 100% box coverage and zero
zone-dead games. Arena coverage was checked BEFORE measuring, per the D152
warning: all 34 team codes appearing in 002 games below 2009-10 are present in
`arenas.csv`, so no season's travel figure is silently zeroed.

What the new rows say, and it is the loudest thing in this file:

1. **HOME ADVANTAGE IS NOT A CONSTANT AND THE OLD ERA IS A DIFFERENT SPORT ON
   THIS AXIS.** Home margin runs **+2.57 to +3.88 in every single season
   1996-97..2010-11 (15/15)** against **+1.69 to +2.50 across the whole current
   corpus**. The maximum ever measured is **2002-03 at +3.884 / .6283 home WR**;
   the minimum is 2020-21 at +0.944. There is **no overlap at all** between the
   pre-2011 range and the 2021-26 range. D139/D152's regime warning is now
   supported by 15 seasons instead of 3, and it is a monotone decline, not noise.
2. **THE 3PT RAMP IS CONTINUOUS BACK TO 1997-98, WITH ONE CLEAN STEP.**
   3PA share: .160 (1997-98) → .167 → .170 → .181 → .182 → .187 → .196 → .202 →
   .212 → .222 (2007-08) → .222 (2010-11) → .243 → .260 → .268 → .285 → .316 →
   .337 → .359 → .384 → .399 → .421 (2024-25). **The single exception is
   1996-97 at .212** — out of sequence by +5.2pp against 1997-98, the largest
   one-season move in the entire 30-season series, and it lands exactly on the
   **shortened 3-point line (22 ft, 1994-95..1996-97, restored to 23'9" in
   1997-98)**. Measured here, not recalled; added to §7.
3. **PACE**: the series is U-shaped, not monotone — 92.84 (1996-97), trough
   **91.61 (1998-99)**, 92.62 (2003-04), then a slow climb to 102.5 (2018-19)
   and a plateau ~101 since. Scoring bottoms at **91.58 pts/team-game in
   1998-99** and **93.40 in 2003-04**, i.e. the pre-hand-check-ban trough.
4. **TRAVEL**: the registered claim that E-1 has the "highest travel measured
   anywhere (868-904 km)" is **CORRECTED** — the maximum is **941.6 km/team-game
   in 2003-04**, with 1999-00 at 919.4 and 1996-97 at 911.1 also above 2016-17's
   903.7. Travel has fallen roughly 90 km/team-game since 2003-04.
5. **CORE-DNP / players-used**: availability was far tighter in the old era —
   core-DNP .1051-.1614 for 1996-97..2008-09 against .2023-.2502 for 2022-26,
   and players used 9.99-10.20 against 10.53-10.83. Whatever the availability
   feeds would have said about those seasons, the underlying rate of stars
   sitting was roughly HALF what it is now.

**E-3 IS NOW TOO COARSE, AND THAT IS AN OWNER DECISION, NOT AN AGENT'S.**
E-3 as coded ("≤ 2010-11") now spans **15 seasons** with a 1.3-point spread in
home margin, an 11-point spread in pts/team-game and a 6.2pp spread in 3PA
share. The measured internal boundaries are already in §7 (shortened 3PT line
1996-97, zone defense legalised 2001-02, hand-check ban + 29→30 teams 2004-05,
1998-99 lockout). D160 deliberately did **NOT** re-code the era boundaries or
touch `nbapred/eval/splits.py::ERAS`, because that is a model-facing change
requiring a gate; the rows are landed under the existing codes and the
sub-structure is named for whoever re-registers.

---

## 2b. NEW SIGNATURE ROWS (D153) — model-facing, not sport-facing

The §2 table measures the SPORT. These two blocks measure what our own stack
can SEE in each season, and they explain more of our historical deficit than
any sport-side feature does. Produced by `scripts/history_killed.py` and
`scripts/history_analyze.py`; read them before attributing anything to an era.

### 2b.1 DARKO MINUTE COVERAGE — the availability ramp inside our own stack

`CompositionModel` sets `talent = darko.get(player_id, 0.0)`, so a player with
no `darko_history` row is silently scored LEAGUE-AVERAGE. Coverage is the share
of actually-played minutes belonging to players who have any prior DARKO row:

**D170 (2026-08-04) CLOSED THIS RAMP. THE TABLE BELOW IS HISTORY, NOT CURRENT
STATE.** The ramp was never a DARKO limit — darko.app server-renders the full
daily series back to **1996-11-01**, for retired players too (Kobe
1996-11-01..2016-04-13, n=1777; Duncan, Iverson, Nash, Dirk, Yao, KG likewise).
We had only ever fetched 1,009 of the 3,934 player_ids in `player_game_stats`,
i.e. the modern roster universe. After backfilling, `darko_history` holds
**1,103,818 rows / 2,909 players / 1996-11-01..2026-07-26** (was 354,600 / 837 /
2003-10-29) and PIT minute coverage is **98.7-99.95% in EVERY season
1996-97..2025-26**. Read the "was" column only when interpreting a result
computed BEFORE 2026-08-04.

| season | was → is | season | was → is | season | was → is |
|---|---|---|---|---|---|
| 2007-08 | **3.6%** → **99.95%** | 2014-15 | 28.0% → 99.90% | 2021-22 | 88.8% → 99.89% |
| 2008-09 | 6.9% → 99.91% | 2015-16 | 36.9% → 99.90% | 2022-23 | 96.9% → 99.91% |
| 2009-10 | 9.5% → 99.90% | 2016-17 | 44.1% → 99.90% | 2023-24 | 99.9% → 99.91% |
| 2010-11 | **11.2%** → **99.94%** | 2017-18 | 50.5% → 99.87% | 2024-25 | 99.9% → 99.91% |
| 2011-12 | 12.7% → 99.89% | 2018-19 | 62.1% → 99.92% | 2025-26 | 99.9% → 99.87% |
| 2012-13 | 18.6% → 99.93% | 2019-20 | 73.0% → 99.87% | | |
| 2013-14 | 23.0% → 99.93% | 2020-21 | 81.2% → 99.89% | | |

**WHAT THAT COST US, MEASURED (D170).** Re-running D161's 19-season model
availability-blind on the backfilled feed, changing nothing else: 2007-08's
normalized gap **+26.87% → +7.18%**, 2009-10 **+25.69% → +4.24%**, 2010-11
**+23.61% → +4.22%**, 2008-09 **+17.85% → −2.44% (the model BEATS the market)**.
The three seasons that already had ~100% coverage moved by +0.48 / +0.08 /
−0.09 pp — a placebo the design did not have to be given. **Most of what this
file called the "oldest-season deficit" was our own starved talent feed.**

Measured consequences over the 11 poolable seasons (D153, on the PRE-D170 feed):
corr(coverage, **D19 composition-leg** effect) = **+0.793**;
corr(coverage, **D21 four-factors-leg** effect) = **−0.605**;
corr(coverage, normalized market gap) = −0.391;
corr(coverage, the shrunk per-team-home benefit) = **−0.774**.

**This is the §5 trap at corpus scale.** On 2010-11 the composition leg is
significantly HARMFUL (−0.01384) and the arm that does not consume DARKO scores
a 12.88% normalized gap against the shipped 24.67% — i.e. about HALF the
oldest-season deficit is our own starved talent feed, not the era. Any
historical claim about D19/D21 must cite this row.
**D170 UPDATE: those four correlations were computed against a coverage
variable that is now CONSTANT at ~99.9%, so they can no longer be recomputed
and must not be quoted as current. D153's inference was right and the
correction is larger than D153 estimated: it is not "about half" the
oldest-season deficit, it is ~73% of it on 2007-08 and ~82% on 2010-11.**

### 2b.2 TEAM-HOME STRUCTURE, per season (D137 method, D153 extension)

league HFA is the fitted mean home effect with schedule controls; τ and signal
share are method-of-moments EB on the per-team deviations d_t; both DEN and UTA
are their own d_t in points.

| season | era | league HFA | sd(d) | rms se(d) | τ | signal share | DEN d | UTA d |
|---|---|---|---|---|---|---|---|---|
| 2010-11 | E-3 | +2.588 | 2.599 | 2.453 | 0.861 | 11.0% | +5.65 | −1.20 |
| 2011-12 | E-2 | +2.332 | 3.715 | 2.918 | 2.300 | 38.3% | −3.80 | +4.48 |
| 2012-13 | E-2 | +2.551 | 3.023 | 2.588 | 1.564 | 26.8% | +4.73 | +3.94 |
| 2015-16 | E-1 | +1.887 | 2.336 | 2.567 | 0.000 | **0.0%** | −4.36 | +2.19 |
| 2016-17 | E-1 | +2.670 | 2.641 | 2.781 | 0.000 | **0.0%** | −1.98 | −3.03 |
| 2017-18 | E-1 | +1.063 | 2.856 | 2.756 | 0.749 | 6.9% | +4.36 | +3.01 |
| 2018-19 | E-1 | +2.725 | 3.532 | 2.856 | 2.079 | 34.6% | +7.64 | −0.56 |
| 2019-20 | E0 | +2.157 | 3.678 | 3.141 | 1.913 | 27.1% | +1.88 | −1.84 |
| 2020-21 | E2 | **−0.309** | 3.892 | 3.266 | 2.117 | 29.6% | +0.97 | +6.67 |
| 2021-22 | E3 | +1.225 | 3.525 | 3.091 | 1.694 | 23.1% | −3.63 | +4.72 |
| 2022-23 | E4 | +2.548 | 3.643 | 2.792 | 2.339 | 41.2% | +8.73 | −0.80 |
| 2023-24 | E5 | +2.394 | 4.423 | 3.034 | 3.219 | **53.0%** | +5.26 | +9.36 |
| 2024-25 | E6 | +2.171 | 3.314 | 3.056 | 1.284 | 15.0% | +0.42 | −2.13 |
| 2025-26 | E6 | +3.187 | 2.717 | 3.193 | 0.000 | **0.0%** | −3.55 | +2.95 |

(These league HFAs are schedule-CONTROLLED and are not the raw home margins in
§2; the E2 sign flip to −0.309 is the no-crowd season with controls applied.)

**Cross-season persistence of d_t, the number D70 and D137 turned on:**
lag-1 **+0.0839** over 359 team-pairs, lag-2 −0.0779;
**historical block (2010-2019) +0.1098 vs modern block (2021-2026) +0.0264.**
At sd(d) ≈ 3.3 pts, r = +0.110 is a next-season forecast SD of **0.36 pts**.
**Team home advantage has essentially ZERO cross-season memory in EVERY era we
can measure** — D70's kill and D137's quantification survive a nine-season
extension, and within-season signal share is episodic with no era pattern.

### 2b.3 ALTITUDE AND TRAVEL, per era (D153)

Visitor's elevation gain, points per km climbed, schedule-controlled:
E-3 +0.411 (t=+0.56) · E-2 +0.702 (+1.19) · E-1 +0.522 (+1.30) · E0 +0.433
(+0.45) · E2 +1.410 (+1.44) · **E3 +2.983 (t=+3.29)** · E4 −0.456 (−0.56) ·
E5 +0.098 (+0.11) · E6 −0.321 (−0.47). **The only significant altitude
coefficient anywhere is in 2021-22 — a season INSIDE the certified corpus.**

Margin-frame travel terms (home minus away, pre-registered signs
dtrav NEG / dtz POS / d3in4 NEG):

| frame | dtrav_kkm | dtz_east | d3in4 |
|---|---|---|---|
| HISTORICAL (2010-2019, no lockout) | +0.0067 (t=+0.05) | −0.0744 (−0.52) | −0.5394 (−1.83) |
| CERTIFIED 5 | −0.2269 (−1.28) | −0.2164 (−1.19) | −0.7333 (−1.91) |
| **ALL n=16,544** | −0.0793 (−0.79) | −0.0336 (−0.33) | **−0.5156 (t=−2.44)** |

**Travel is WEAKER historically, not stronger** — despite E-1 carrying the most
travel ever measured (868-904 km/team-game). Circadian sign-flips and its only
significant coefficient (E6 −0.7337 t=−2.41) has the WRONG registered sign.
**`d3in4` is the one travel-family term with a correct, era-robust sign**, and
it is significant on the widest frame the project has ever assembled and in
E-1, an era no gate selected on.

**`travel_valid` COVERAGE IS 1.0000 ON EVERY HISTORICAL SEASON** after D152's
7 franchise additions — the only invalid rows corpus-wide are 2024-25/2025-26
at 0.9894 (the D140 unknown-venue neutrals plus the following team-game).

---

Definitions: *core player* = played ≥20 games in that season at ≥28.0 min/g,
assigned to the team he played the most games for. *core-DNP* = fraction of
(core player, his team's games) pairs with no box-score row. *repeat-opp* =
fraction of team-games whose opponent was also the previous game's opponent.
*poss* = FGA + 0.44·FTA − OREB + TOV.

---

## 3. Era by era, with the corrections

### E0 — pre-COVID normal (2019-20 to 2020-03-11) — **NOW SCORABLE (D152)**
971 games, home margin **+2.174**, normal crowds. Reproduces D131 exactly.

**STATUS CHANGED 2026-08-02.** D131 recorded this as "NOT SCORABLE and this is
structural, not a backlog item" — `fit_production` raised
`RuntimeError: FourFactors not ready ... no carry rows from 2018-19`. That
diagnosis was right about the cause and wrong about it being structural: it was
a missing-data problem, and D152 backfilled 2018-19 (1230/1230 games, 30,801
`player_game_stats` rows). Probed at 07:52 on 2026-08-02,
`fit_production(con,'2019-20',before=2019-10-22)` **returns**. 2018-19 itself is
also now scorable (2017-18 landed).

**Scorable does NOT mean admissible.** 2019-20 is still E0+E1 pooled — 88 of its
1,059 games are the one-site, zero-travel, no-crowd bubble — and E0 sits on the
far side of the COVID break from the E3-E6 corpus. Admitting it is a separate
decision the owner must take; D152 changed the data, not the eval universe.

### E1 — the Orlando bubble (88 games) — NOT SCORABLE (travel bug FIXED at D140)
* True travel for every one of these team-games is **0 km** — one site.
* `nbapred/model/travel.py` USED to derive the host from `matchup` and so
  assigned **1,505.5 km per team-game** of travel that never happened. **FIXED
  at D140**: neutral-site games (the bubble window + the feed's
  `is_home`-FALSE-on-both marker) now chain on the VENUE, and a >14-day hiatus
  resets acute load. Measured after the fix: **0.0 km, 0.0 h tz, 0.0 m
  elevation on all 176 bubble team-games.**
* **D140 also measured what the fiction was worth, and it was NOT the cause of
  D136's significant coefficients**: dtrav_kkm on the FULL frame moves only
  −0.3088 (t=−2.21) → −0.3061 (t=−2.17), still SIG, and d3in4 is bit-identical
  (is_3in4 is a pure date computation). The bubble is 88/8,279 = 1.06% of that
  frame. What d3in4's FULL-frame significance DOES depend on is the bubble's
  presence at all (drop the 88 games: t=−2.01 → −1.88 ns) — via schedule
  density, not via travel.
* `player_game_stats` has **zero rows for all 88 games** (the season's box
  coverage is 971/1059), so E1 cannot even be measured at player level.
* Home margin +1.648 on n=88 (se ≈1.35) — statistically nothing, as D131 said.

### E2 — no-crowd compressed season (2020-21) — NOT SCORABLE (excluded by D131)
* Home margin **+0.944** vs the +1.958 E3-E6 baseline = **−1.014 pts, z=−2.02**.
  D131's number reproduces exactly.
* **CORRECTION — "deliberately reduced travel" is NOT what the data shows.**
  Travel per team-game is **830.2 km**, which sits INSIDE the normal range
  (800.8 in 2022-23, 842.1 in 2021-22, 855.1 in 2025-26). Season-total travel
  is down only ~9% (59.8k vs 65.7k km/team) and that is almost entirely the
  72-game schedule, not shorter trips.
* **What actually compressed was REST, not distance**: b2b rate **.212**
  (vs .163-.183 elsewhere), 3-in-4 **.337** (vs .248-.268), mean rest **2.03 d**
  (vs 2.12), rest≥3 days **.138** (vs .187-.195). E2 is the densest schedule in
  the corpus by a wide margin.
* The series-style scheduling IS visible but is a small effect:
  repeat-opponent **.086** vs .011 (2019-20) and .035-.058 later. Note the
  league has drifted UP on repeat-opponent scheduling generally, so E2 is
  ~1.5-2x the modern norm, not 8x.
* Verdict unchanged from D131: stays out of the eval corpus. If ever added it
  is a SEPARATE STRATUM, never pooled.

### E3 — re-entry + Omicron (2021-22) — SCORABLE, holdout half #1
* Crowds back; home margin +1.723.
* **Omicron confirmed and dated from the data.** ISO weeks 2021-W50..W52
  (2021-12-13 → 2022-01-02) carry core-DNP **0.3033** against a pre-wave
  baseline of **0.1332** (2021-W42..W49) — a **2.28x** jump. The same calendar
  weeks a year later are flat (0.1696 vs 0.1614 pre). Monthly: Nov-21 .1323 →
  Dec-21 .2480, the largest Nov→Dec jump in the corpus (+11.6pp vs +0.4 /
  +1.2 / −3.4pp in 2022-23 / 2023-24 / 2024-25).
* Absence stays elevated post-wave (2022-W01..W05 = .2098), so E3 is a
  high-absence season overall (.2394 for the year).
* ~~**Our injury feed cannot see any of this**: `injury_reports_pit` starts
  2023-10, `game_inactives` starts 2022-23. Every E3 absence number in this
  file is derived from `player_game_stats` non-appearance. Any feature that
  consumes the injury feed is STRUCTURALLY INERT on half the legacy holdout.~~
  **NO LONGER TRUE — D170 (2026-08-04).** E3 (2021-22) now carries BOTH feeds:
  `injury_reports_pit` 168 report-days (2021-10-19..2022-04-10) and
  `game_inactives` on the full schedule. The absence numbers in this file were
  still derived from `player_game_stats` non-appearance and are unchanged, but
  a feature that consumes the injury feed is **no longer structurally inert**
  on the legacy holdout and any claim that it is must be re-tested.

### E4 — post-COVID baseline (2022-23) — SCORABLE, holdout half #2
* **CORRECTION — this is NOT "the only clean normal season".** It is the
  corpus's **home-advantage outlier**: home WR **.5805** against .5453 pooled
  over the other four scorable seasons (**z = +2.23**), home margin **+2.500**
  vs +1.822 (z = +1.51).
* It is also the least predictable season for the market: `ll_mkt` 0.62437,
  the corpus maximum (D132).
* Consequence for every past gate: **half the legacy holdout is an outlier
  regime on exactly the axis (home advantage) that our schedule layer, our
  home-edge fit and D20/D70 all live on.** A feature failing on the holdout may
  be failing on E4's peculiarity rather than failing out-of-sample.
* Scoring steps up here and stays: 110.62 → 114.69 pts/team-game.

### E5 — PPP + In-Season Tournament + new CBA (2023-24) — the dev boundary
* **In-Season Tournament confirmed in the data**: `game_id` prefix **006**
  (the NBA Cup final, which is not a regular-season game) first appears in
  2023-24 and in every season since. Measured, not asserted.
* **CORRECTION — the Player Participation Policy did not reduce measured star
  absence.** core-DNP: .2023 (E4) → .2063 (E5) → .2502 / .2435 (E6). If the PPP
  moved anything in our data it moved it the wrong way.
* Games used per team-game rise slightly (10.53 → 10.73), consistent with
  deeper rotations.
* This is the boundary the legacy split sits on. Any dev-vs-holdout comparison
  is confounded with all three E5 changes simultaneously.

### E6 — apron maturing (2024-25, 2025-26)
* core-DNP at its corpus maximum (.2502 / .2435).
* The pool of qualifying core players SHRINKS monotonically: 136 (E4) → 128
  (E5) → 119 → 115. Fewer players clear ≥28 mpg over ≥20 games.
* Travel per team-game at its corpus maximum (850.9 / 855.1).
* Pace flat. Scoring flat-to-up (115.61 in 2025-26, corpus max).
* Attribution to the apron is a STORY, not a measurement — the shrinking core
  pool and rising absence are consistent with thinner rotations and load
  management, and this file does not claim to have identified the cause.

---

## 4. What is NOT an era effect

Two things the data says are stable, so no future gate should invoke them:

* **Pace.** 100.44 - 102.65 possessions/team-game across all seven seasons.
  There is no pace regime in this corpus.
* **Rest structure outside E2.** mean rest is **2.12 days in every single
  season from 2021-22 to 2025-26**, to two decimal places. b2b and 3-in-4 move
  by ≤2pp. The schedule is remarkably stationary once E2 is removed.

---

## 5. Feature-availability by era (the trap list)

| channel | E-3/E-2/E-1 | E0/E1 | E2 | E3 | E4 | E5 | E6 |
|---|---|---|---|---|---|---|---|
| `nba_games` (schedule, scores) | yes (to 1996-97) | yes | yes | yes | yes | yes | yes |
| `player_game_stats` | yes where landed (D152/D153) | E0 only | yes | yes | yes | yes | yes |
| `odds_market` closes | **from 2007-08; THIN pre-2013** (162/131/83 games unmatched in 2010-11/2011-12/2012-13) | yes | yes | yes | yes | yes | yes |
| **`darko_history` MINUTE coverage** | **~99.9% (D170; was 11-62%) — see §2b.1** | ~99.9% (was 73%) | ~99.9% (was 81%) | ~99.9% (was 89%) | ~99.9% (was 97%) | 100% | 100% |
| `game_inactives` | **YES (D170; was no)** | yes | yes | **yes** | yes | yes | yes |
| `injury_reports_pit` | **no before 2018-12-17 — SOURCE FLOOR, not an ingest gap** | **yes (D170; was no)** | **yes** | **yes** | **yes** | yes | yes |
| `schedule_features` table | no | no | no | no | no | no | 2025-26 only |
| travel/venue features meaningful? | **yes — `travel_valid`=1.0000 (D152 arenas fix verified, D153)** | E0 yes, **E1 all-zero by construction (D140)** | yes | yes | yes | yes | yes |
| D73 tank term active? | **NO below `tanking.season_floor()` (=2014-15 as of D153) — UNTESTABLE, not failed** | no | no | yes | yes | yes | yes |

E1's travel/venue columns are no longer FALSE, but they are still degenerate —
every value is exactly 0 — so E1 contributes no travel identification. The 10
feed-flagged neutral games in E5/E6 have UNKNOWN venue coordinates and are
marked `travel_valid=False` (with the next team-game after each, 36 team-games
total): DROP them, do not score them.

Read this table before claiming a feature "does not transfer to the holdout".
~~Two of the four availability channels are simply ABSENT on the holdout.~~
**SUPERSEDED BY D170 (2026-08-04): they were absent from OUR DB, not from the
world.** `game_inactives` now covers **2006-07 onward** (BoxScoreSummaryV2's
`InactivePlayers` result set was populated the whole time; probed live, it is
empty for 2005-06 and earlier and non-empty from 2006-07 — that IS a source
floor) and `injury_reports_pit` covers **2018-12-17 onward** (probed daily; the
league's standardised PDF does not exist before that date, which IS a source
floor). The only genuinely absent channel on the old eras is the injury report
before 2018-12-17.

**D153 ADDS THE ROW THAT MATTERS MOST AND IT IS NOT BINARY.** `darko_history`
coverage is a RAMP, not a switch, so the composition leg is not "off" pre-2019 —
it is progressively degraded, and its measured effect tracks the ramp at
r = +0.79. A ramped input produces what looks exactly like a smooth era effect.
Before writing "ERA-SPECIFIC" about any term, check §2b.1 for the term's
inputs.

---

## 6. How to cite

* In a pre-registration: name the eras in the eval universe, e.g.
  "dev = E5+E6, confirm = E3+E4", and state whether the hypothesis is expected
  to be era-stable.
* In a result: report the per-era estimate and the heterogeneity statistic from
  `nbapred.eval.splits.era_decomposition` (I², τ, Q).
* Never write "the holdout" without naming which eras it contains.

---

## 7. GOING BACK — the real regime boundaries below E-2 (D152)

Added when the historical backfill established that game data is available to
**1996-97** and no further (PBP is empty below it — see D152 / STAT_INVENTORY).
These are the boundaries any future decision to widen the corpus has to cross.
They are listed oldest-first with what breaks at each.

| boundary | season | what changes | consequence for us |
|---|---|---|---|
| **PBP floor** | **1996-97** | `playbyplayv3` returns ZERO actions for 1995-96 and every season below it (verified on 14 games, 1995-96 → 1983-84) | **HARD FLOOR.** Below it every zone feature (`rima/mida/thra`...) is silently 0 and `four_factors` eFG degenerates to fgm/fga. Boxscores still load — which is the trap. **NOW ENFORCED IN CODE (D160)**: `possessions.load_corpus` REFUSES any game with `fga>0` and zero zone attempts, and `backfill_history.py pgs` refuses any season below 1996-97 outright. 13 such games had already leaked into `player_game_stats` and were purged |
| plus-minus floor | 1996-97 | `plusMinusPoints` is 0.0 in every probed season 1995-96 and below | same floor, second reason |
| **DARKO floor** | ~~2003-04 (2003-10-29)~~ → **1996-97 (1996-11-01), D170** | `darko_history` has no rows before it | ~~**THE BINDING FLOOR FOR THE MODEL, measured D160.**~~ **NO LONGER BINDING (D170, 2026-08-04).** The 2003-10-29 floor was an artefact of having fetched only 837 of 3,934 darko.app player pages; the site serves the full daily series back to 1996-11-01 for retired players too. `darko_history` is now 1,103,818 rows / 2,909 players / 1996-11-01.., PIT minute coverage ~99.9% in every season, and **1996-97..2003-04 are now fittable.** They remain UNSCORABLE against the market (`odds_market` starts 2007-08), so the 19-season scorable frame is unchanged |
| **odds floor** | **2007-08** | `odds_market` / `odds_open` start at season_end 2008 | **THE BINDING FLOOR FOR EVALUATION.** 2004-05/2005-06/2006-07 fit but cannot be scored against a market, so they are training tail and carry only. This is why the scorable run starts at 2007-08 and not at 2004-05 |
| **shortened 3PT line** | **1994-95 → 1996-97** (restored 1997-98) | 3PT line moved in to a uniform 22 ft, then back to 23'9" | **MEASURED, not recalled (D160):** 3PA share is **.212 in 1996-97 and .160 in 1997-98** — a −5.2pp step, the largest single-season move in the whole 30-season series, and the ONLY break in an otherwise continuous .160 → .421 ramp. 1996-97 is therefore a shot-mix regime of its own and must never be pooled with 1997-2000 on any 3PT-sensitive measure |
| **zone defense legalised** | **2001-02** | illegal-defense rules repealed; help defense becomes legal | shot-mix and rim-finishing priors are not transportable across this line |
| **hand-check ban** | **2004-05** | perimeter hand-checking outlawed | the largest single scoring/pace discontinuity in the modern era; also the season the league went 29 → 30 teams (002 count 1189 → 1230) |
| **lockout** | **2011-12** | 66-game season, compressed | expected 002 count **990**, not 1230. Same distortion class as 2020-21 (b2b/3-in-4 spike): must be a separate stratum, never pooled |
| cancelled game | 2012-13 | BOS @ IND never played | expected 002 count **1229** |
| **3PT-era ramp** | continuous, **1997-98 → 2024-25** (D160 extends it back) | measured 3PA share: 1997-98 **.160**, 2003-04 .187, 2007-08 .222, 2012-13 .243, 2013-14 .260, 2017-18 .337, 2018-19 .359, 2019-20 **.384**, 2024-25 **.421** | NOT a step change — a **28-season monotone ramp** with exactly one break (1996-97's shortened line, row above). Any pre-2014 season is a different offensive regime by this measure alone, which is why E-2 is coded separately from E-1 |
| **lockout** | 1998-99 | 50-game season | expected 002 count **725**. **MEASURED D160: this is the most schedule-distorted season we hold, worse than 2011-12 on EVERY channel** (b2b .335 vs .310, 3-in-4 .494 vs .463, mean rest 1.81 vs 1.88 d, rest≥3 .114 vs .123) and the pace/scoring trough of the entire series (91.61 poss, 91.58 pts). Separate stratum, never pooled — it has no odds so this costs nothing |
| 29 → 30 teams | 2004-05 | Charlotte Bobcats | 002 count 1189 → 1230 |
| 27 → 29 teams | 1995-96 | Grizzlies, Raptors | below the PBP floor anyway |

Franchise codes that appear below 2013-14 and did NOT exist in `arenas.csv`
until D152: **NOH** (→NOP 2013-14), **NJN** (→BKN 2012-13), **SEA** (→OKC
2008-09), **NOK** (2005-07), **VAN** (→MEM 2001-02), **CHH** (→NOH 2002-03),
**WSB** (→WAS 1997-98). A missing code costs that team's whole season of
travel silently (measured 60.7 km/team-game error on 2012-13 from NOH alone).

## D171 — ABSENCE DENSITY IS AN ERA PROPERTY, NOT AN INGEST GAP

D170 measured that the availability feed is worth more to a modern season than
an old one (1.32 OUT/team-game in 2024-25 vs 0.31-0.48 in 2012-16) and left the
interpretation open. **D171 settles it: the difference is REAL, not a data
gap.** Measured by `scripts/d171_era_density.py` (read-only):

```
season   inactive rows  raw/team-gm  scored OUT/tm  kept%  played/team-gm  viol
2007-08          5,985         4.87          0.42    8.7%          10.12     0
2009-10          5,423         4.41          0.39    8.8%          10.08     0
2012-13          4,454         3.62          0.32    8.8%          10.48     0
2013-14          4,675         3.80          0.31    8.1%          10.41     0
2015-16          5,121         4.16          0.38    9.0%          10.60     0
2016-17          5,143         4.18          0.44   10.5%          10.62     0
2017-18          9,705         7.89          0.70    8.9%          10.61     0
2018-19          9,599         7.80          0.83   10.6%          10.61     0
2020-21          6,809         6.30          1.13   17.9%          10.67     0
2021-22         10,625         8.64          1.38   16.0%          10.58     0
2023-24         11,010         8.95          1.09   12.2%          10.73     0
2024-25         10,854         8.82          1.32   14.9%          10.69     0
2025-26         11,059         9.01          1.33   14.7%          10.83     0
```

Three independent reasons this is an era property:
1. **`viol` = 0 on every one of the 19 seasons.** `viol` counts players listed
   inactive who nevertheless logged minutes; zero everywhere means the official
   list is complete and correct in BOTH eras. The thin old-era out-set is not
   us missing rows.
2. **The RAW official density roughly DOUBLES at exactly 2017-18** (3.6-4.2 →
   7.8-9.0), the season the NBA instituted its injury-report policy and load
   management became general. That number is the league's own list, not ours.
3. **Rotation depth is FLAT** (players used per team-game 10.0-10.6 old vs
   10.5-10.8 modern), so it is not a roster-size artefact either.

The one genuine change in OUR filter's behaviour points the same way: `kept%`
(the share of the inactive list surviving the 12-day roster window) rises from
**8.1-10.5%** to **12-18%**. Modern absences are concentrated among players who
ARE in the active window — rotation players resting — which is the
load-management signature precisely. **So an honest availability feed helps a
modern season more, and that is a property of the ERA, not a defect in the old
seasons' data.**

**ERA-AVAILABILITY ROW (GATE_POLICY_V2 §10), CURRENT.** `game_inactives` and
`darko_history` cover E-3 through E6 in full. The only model-visible channel
still absent on the old eras is the 5PM injury report before **2018-12-17**,
which is a SOURCE floor; D171 prices it at **-0.741pp** at modern density and
**-0.25pp..-0.74pp** at old-era density. AVAILABILITY TIER remains era-varying
and must be labelled per season: **T2i on 2007-08..2017-18, T2 on
2018-19..2025-26.**

## D175 — THE UNOFFICIAL BACKFILL IS REFUSED, AND OUT-SET VALUE IS NON-MONOTONE

D175 hunted free unofficial availability data for the pre-2018-12-17 era and
**recommends NOT closing the gap.** Three era facts land here:

1. **A free, systematic, permitted archive exists for 5 of the 11 old seasons
   and not the other 6.** Wayback snapshots of `cbssports.com/nba/injuries`
   and `usatoday.com/sports/nba/injuries/` (web.archive.org publishes no
   robots.txt; 404 = unrestricted). C48 = share of game days with a
   distinct-content snapshot in the prior 48h:
```
        07-08 08-09 09-10 10-11 11-12 12-13 13-14 14-15 15-16 16-17 17-18
 union     0%    2%    0%    2%  6.8% 11.1% 80.9% 83.4% 90.0% 93.5% 96.0%
```
   So **2013-14..2017-18 are archivable and 2007-08..2012-13 are not.**
   `prosportstransactions.com` is behind a Cloudflare managed challenge (403 on
   every path incl. robots.txt) and BBRef's `*/gamelog/` is `Disallow`ed, so
   neither is available at all.

2. **PILOTED ON THREE OF THE FIVE ARCHIVABLE SEASONS, IT DOES NOT PAY —
   BUT THE REASON IS VARIANCE, NOT A CLEAN NEGATIVE.** CBS OUT-set unioned
   with inactives, fed to `k19_t2.season_run` UNMODIFIED (so it is the exact
   analogue of `T2 = report UNION inactives`); baseline `t2i` = inactives only,
   same process, same DB state, and it reproduces the certified per-season
   numbers exactly:
```
  season   CBS snaps  C48     t2i     CBS arm    DELTA      outs/tm base->CBS
  2014-15     109    67.3%   +9.80%   +9.90%   +0.100pp       0.415 -> 0.528
  2015-16     120    85.6%   +5.07%   +5.29%   +0.220pp       0.375 -> 0.526
  2016-17     189    91.0%   +7.91%   +5.88%   -2.030pp       0.437 -> 0.546

  mean -0.570pp  sd 1.266  t(2) = -0.780  p = 0.5171  95% CI [-3.714,+2.574]
  WRONG SIGN on 2 of 3;  leave-one-out means -0.905 / -0.965 / +0.160
```
   **The CI spans the benchmark, zero and the opposite sign, and the whole
   pooled effect is one season.** Not actionable at K=3.

3. **A PERFECT AVAILABILITY ORACLE IS WORTH FAR MORE THAN THE REPORT, AND
   THE OUT-SET IS NON-MONOTONE IN COMPLETENESS.** Out-set = every player who
   logged 0 minutes in that exact game (maximal leakage; a bound, not a result):
```
  season     t2i     outs/tm    PERFECT-ORACLE   outs/tm    DELTA
  2012-13   +7.06%    0.319         +5.30%       1.306     -1.760pp
  2015-16   +5.07%    0.375         +2.46%       1.339     -2.610pp
  2023-24  +16.97%    0.974        +13.22%       1.794     -3.750pp
```
   Old-era headroom is **2.4-3.5x the report's -0.741pp**. But a PARTIAL feed
   landing at 0.526-0.546 outs/team, between a 0.375-0.437 baseline and a
   ~1.3 optimum, can be WORSE THAN NOTHING. **A half-filled out-set is not
   half a good** — which is precisely the "sporadic data is worse than a clean
   gap" risk, now measured rather than asserted.

4. **MARGINAL-SET PRECISION IS THE AXIS THAT SEPARATES THE TWO FEEDS.** Because
   `game_inactives` is already complete and correct, a pregame source can only
   add names NOT on it, and those marginal names carry the entire effect:
```
                    report-OUT rows  already in inactives  marginal  of marginal, PLAYED
  CBS      2015-16          2,129         74.78%            25.22%        21.42%
  OFFICIAL 2022-23          8,049         94.47%             5.53%         3.37%
  OFFICIAL 2023-24          8,928         92.79%             7.21%         1.24%
```
   The report adds a SMALL, ~98%-true increment on top of a DENSE modern base.
   The archive adds a LARGE, ~79%-true increment on top of a THIN old base.

**CONSEQUENCE FOR THE ERA ROW: unchanged.** The tier labels stay **T2i on
2007-08..2017-18, T2 on 2018-19..2025-26**. D175 creates no T2-unofficial tier
and recommends against one on the current evidence. **The pre-2018-12-17 floor
stands, but the reason is now precise: for 2007-08..2012-13 no free systematic
archive exists at all; for 2013-14..2017-18 one does and it is not measurably
worth ingesting at K=3.** 2013-14 and 2017-18 remain unpiloted and would take
K to 5 — the owner's call, not a conclusion of the entry.
