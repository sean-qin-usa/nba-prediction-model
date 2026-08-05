# HOME ADVANTAGE — decomposition, stationarity, and whether any of it is forecastable

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

2026-08-01. Diagnostic-only investigation. **No production file was touched**;
every number below comes from new read-only scratch scripts `scripts/ha_*.py`.
Corpus: all 7 seasons of regular-season games in `data/nba.duckdb`
(2019-20..2025-26, 8,191 non-neutral games), strata per D131.

> **ESTIMATION DISCIPLINE.** Sections 1-3 and 5 are **DESCRIPTIVE and
> FULL-SAMPLE**. They are not forecastable edges and none of them may be cited
> as one. Section 4d is the only point-in-time construction: the roster
> aggregate for season *y* uses player splits estimated strictly from seasons
> before *y*. Every headline number carries a bootstrap CI.

---

## THE VERDICT, up front

**Sean's hypothesis is half right, and the half that is right is the half that
cannot be traded.**

1. **Team-specific home advantage is REAL within a season.** Of the observed
   3.53-point spread in team home deviations, 26.1% is true signal
   (tau = 1.80 pts, CI 0.77-2.45), and a parametric null of zero true spread is
   rejected at p = 0.0031. D70's "dead vein" was too strong as a statement
   about the *existence* of the effect.
2. **It has essentially ZERO carry into the next season.** Lag-1 correlation
   of team home deviation across the 5 normal seasons is **+0.021**
   CI(-0.120, +0.155) — NS. Lag-2 is **-0.140** CI(-0.350, +0.086) — NS.
   Rank-order lag-1 Spearman is +0.033.
3. So the best possible forecast of a team's next-season home deviation has a
   standard deviation of **0.075 points** (upper end of the lag-1 CI: 0.55
   points). Against a league home edge of ~1.96 points and a market that prices
   the total game to ~0.5 points, that is nothing. **D70's conclusion stands;
   its stated mechanism is now measured rather than asserted.**
4. **The Lakers are not special.** LAL's 5-season pooled home deviation is
   +2.08 pts (rank 3/30) but z = 1.53 — not distinguishable from league
   average. And LAL's single best home season in the corpus is **2023-24, under
   Darvin Ham, before Redick arrived**. The coach-era story does not survive
   contact with the year-by-year numbers.
5. **Where the ~1.96 points actually comes from**: about half is crowd
   (+0.99 pts, causally identified by the 2020-21 no-crowd season), essentially
   none is travel/rest asymmetry (-0.11 pts, CI spans zero), and the rest is an
   unexplained residual (+1.07 pts) that survives an empty arena.
6. **Home advantage is flat over the season, with a mild downward tilt that
   does not reach significance** (Sean's second hypothesis). The fitted trend is
   -0.38 pts per half-season CI(-0.99, +0.26) — *his sign*, not his
   significance — and it does not replicate in 2019-20. **Opening night
   specifically is the LOWEST point estimate in the table** (n = 67, so it
   settles nothing either way).
7. **The one live thread is players, not teams.** Player home-sensitivity has a
   small but statistically SIGNIFICANT lag-1 persistence (r = +0.066,
   CI +0.014..+0.115, n = 1,371 player-pairs) where the team-level test has
   none. Roster-aggregating it produces a *very* stationary team signal
   (lag-1 autocorrelation +0.486 vs +0.039 for team identity) that predicts the
   team's realised home advantage **not at all** (r = -0.03, NS). Sean's
   mechanism reasoning was structurally right and the quantity it produces is
   empty. Section 4.

**One-line summary for the register:** team home advantage is real within a
season (tau = 1.80 pts, p = 0.003) and has no memory across seasons
(lag-1 +0.021 NS); ~51% of the league edge is crowd, ~0% is travel; LAL is
rank 3/30 at +2.09 but the biggest apparent 5-season home edge under a
zero-effect null is +2.74, LARGER than the +2.63 observed.

---

## 0. Method, and two traps that change the answer by 2x

### The identifying regression
One regression per season over non-neutral regular-season games:

```
margin_g = sum_t s_t * (1{home=t} - 1{away=t})    # team strength
         + sum_t a_t * 1{home=t}                  # team's TOTAL home advantage
         + C_g . beta                             # schedule controls
```

The home block spans the intercept, so no separate intercept is fitted and
`a_t` is the team's own home advantage in points. The strength block has an
exact null direction (`sum_t z_t == 0` identically), so the minimum-norm lstsq
solution automatically gives `sum_t s_t = 0`: strength is measured against the
league mean and `a_t` is **fully opponent- and own-quality-controlled**. Then

```
league HFA   = mean_t(a_t)
team home DEVIATION  d_t = a_t - mean_t(a_t),   sum_t d_t = 0
```

Schedule controls (both sides, so the confound Sean flagged is handled):
`h_b2b, a_b2b, h_3in4, a_3in4, rest_diff, h_travel_km, a_travel_km, |h_tz|,
|a_tz|`. `schedule_features` in the DB only covers 2025-26, so all of this was
rebuilt for 7 seasons from `nba_games` + the static `ARENAS` geo
(`scripts/ha_panel.py`).

### Trap 1 — split-half must not share the fitted home effect
Residualising each game with `a_hat_t` added back gives both halves a common
term and inflates the half-half correlation **mechanically**: 0.47 instead of
0.13. Fixed by residualising on team strength + controls only, and corroborated
with a fully independent refit of the whole regression on each random half.

### Trap 2 — bootstrapping games cannot estimate tau
Resampling games with replacement puts a **second** layer of sampling noise on
each `d_hat`, so `var(d_hat_boot) ~ tau^2 + 2*se^2` while `mean(se^2_boot) ~
se^2` — method-of-moments then returns `sqrt(tau^2 + se^2)`. It reported
tau = 3.5 instead of 1.8. Fixed by bootstrapping the **second stage** (resample
the `(d_hat, se)` pairs, which propagates the finite number of team-seasons
without adding noise) plus a parametric null for the p-value.

### A data finding worth keeping
The NBA feed sets **`is_home = FALSE` on BOTH team-rows** for neutral-court
regular-season games — Mexico City, Paris, and the NBA Cup semifinals in Las
Vegas. There are **10** such games in the corpus (5 in 2024-25, 5 in 2025-26).
Dropping them reproduces D131's realised home margin exactly (2024-25 +1.6922
vs D131's +1.692; keeping them gives +1.6325). The flag is incomplete for
earlier seasons (the Dec-2023 Cup semifinals in Vegas are not marked), leaving
~2-4 games/season, i.e. <= 0.01 pt on a 1,230-game mean.
`nbapred/features/schedule.py` parses the host from `matchup` and would treat
these as ordinary home games. **Not fixed here** (diagnostic task) — flagged
for the schedule-layer owner. This is the concrete resolution of D136's
registered follow-up (3) ("NBA global games are attributed to the nominal host
arena... worth fixing before any margin-endpoint retest"): **the games are
already identifiable from the DB with no new data**, via
`GROUP BY game_id HAVING SUM(is_home) = 0`.

---

## 1. THE STATIONARITY VERDICT

### 1a. Per-season fit

| season | n | home g/team | league HFA (raw) | resid SD | sd(d_hat) | rms SE(d_hat) | tau | signal share |
|---|---|---|---|---|---|---|---|---|
| 2019-20 (pre-shutdown) |  971 | 32.4 | +2.291 | 12.45 | 3.650 | 3.144 | 1.856 | 25.8% |
| 2020-21 (no crowd)     | 1080 | 36.0 | +0.944 | 13.74 | 3.902 | 3.269 | 2.131 | 29.8% |
| 2021-22                | 1230 | 41.0 | +1.723 | 13.86 | 3.572 | 3.091 | 1.790 | 25.1% |
| 2022-23                | 1230 | 41.0 | +2.500 | 12.53 | 3.634 | 2.792 | 2.325 | 40.9% |
| 2023-24                | 1230 | 41.0 | +2.133 | 13.62 | 4.423 | 3.035 | 3.217 | 52.9% |
| 2024-25                | 1225 | 40.8 | +1.666 | 13.64 | 3.279 | 3.043 | 1.223 | 13.9% |
| 2025-26                | 1225 | 40.8 | +1.737 | 14.22 | 2.749 | 3.177 | **0.000** | **0.0%** |

The league-HFA column is the no-schedule-control estimate; it reproduces D131
exactly. (The *controlled* per-season HFA is not reported here because 9
control coefficients estimated on 1,230 games are too noisy to pin an
intercept — the controlled league level is taken from the pooled fit in
section 2, where the coefficients see 8,191 games.)

**Note the last row.** In the most recent season the observed spread of team
home deviations is *smaller* than the sampling noise: there is zero evidence of
any true team-specific home advantage in 2025-26.

### 1b. n is brutal, and that is the whole story
41 home games per team-season. The residual SD of a game margin is ~13.6
points. `d_t` is identified as (home-game evidence about `s_t + a_t`) minus
(road-game evidence about `s_t`), so its sampling SE is roughly
`sqrt(2) * 13.6 / sqrt(41) = 3.0 points` — and that is exactly what the
regression reports (rms SE 3.03).

**A 3.0-point measurement error on a 1.8-point effect.** Everything else
follows from that line.

### 1c. Pooled EB over the 5 normal seasons (150 team-seasons)

```
sd(d_hat) observed              3.525 pts
rms sampling SE of d_hat        3.030 pts
tau = sqrt(var_obs - var_noise) 1.800 pts    CI (0.768, 2.453)
SIGNAL SHARE                    26.1%        CI (6.0%, 39.7%)
parametric null (tau = 0)       p = 0.0031   -> the within-season spread IS real
```

### 1d. Reliability

| | 2019-20 | 2020-21 | 2021-22 | 2022-23 | 2023-24 | 2024-25 | 2025-26 |
|---|---|---|---|---|---|---|---|
| conditional split-half r (optimistic) | 0.490 | 0.506 | 0.479 | 0.565 | 0.635 | 0.420 | 0.239 |
| **independent** half-refit r | 0.121 | 0.191 | 0.131 | 0.261 | 0.350 | 0.115 | -0.105 |
| independent, Spearman-Brown to full season | 0.217 | 0.320 | 0.231 | 0.414 | 0.519 | 0.206 | -0.234 |

Mean over the 5 normal seasons: **independent SB = 0.227**. That agrees with
the EB signal share of 26.1% by a completely different route — two independent
methods, same answer. The conditional number (0.625) is the trap-1 artefact and
should be ignored.

### 1e. Season-to-season persistence — the number that decides it

| | lag-1 | lag-2 |
|---|---|---|
| all 7 seasons, pooled | +0.045 (n=180) | -0.104 (n=150) |
| **normal 5, pooled** | **+0.0212 CI(-0.120, +0.155) NS** | **-0.1404 CI(-0.350, +0.086) NS** |
| per-pair lag-1 | 21->22 +0.036, 22->23 +0.021, 23->24 -0.042, 24->25 +0.104 | |
| rank-order (Spearman) lag-1 | +0.033 mean | |

CIs are cluster bootstraps over teams (8,000x, seed 20260801).

### 1f. Signal share, and what is actually forecastable

```
observed sd(d_hat) per season             3.525 pts
TRUE within-season spread tau             1.800 pts   (26.1% of the variance)
within-season reliability (independent)   0.227
lag-1 slope of the optimal predictor      +0.0212
SD OF THE BEST FORECAST OF NEXT-SEASON d  0.075 pts
   ... using the UPPER end of the lag-1 CI 0.546 pts
disattenuated true persistence            +0.093
```

**Read that as: a team's home advantage is genuinely different from the
league's in a given season, and knowing it tells you essentially nothing about
the next season.** That is the precise sense in which the vein is dead — not
that the effect does not exist, but that it does not survive the calendar. This
is D70's diagnosis, now measured.

### 1g. Is that within-season tau actually home advantage? (two robustness kills)
Before the 26% is called real, two alternative explanations had to die
(`scripts/ha_tau_check.py`).

**(A) Are the analytic SEs honest?** Residual-bootstrap null: simulate seasons
in which every team has *exactly* the league home edge and no team-specific
deviation, keeping the real schedule, real team strengths and real residual
distribution.

| season | observed sd(d) | NULL sd(d) | null [p5,p95] | analytic rms SE | p |
|---|---|---|---|---|---|
| 2019-20 | 3.650 | 3.045 | 2.42-3.67 | 3.144 | 0.063 |
| 2020-21 | 3.902 | 3.186 | 2.57-3.89 | 3.269 | 0.047 |
| 2021-22 | 3.572 | 3.058 | 2.49-3.66 | 3.091 | 0.070 |
| 2022-23 | 3.634 | 2.722 | 2.14-3.31 | 2.792 | **0.007** |
| 2023-24 | 4.423 | 2.938 | 2.41-3.57 | 3.035 | **0.000** |
| 2024-25 | 3.279 | 2.986 | 2.41-3.63 | 3.043 | 0.213 |
| 2025-26 | 2.749 | 3.128 | 2.48-3.88 | 3.177 | 0.800 |

The simulated null matches the analytic SE season by season — **the SEs are
honest** (if anything mildly conservative). Pooled (Fisher over the 5 normal
seasons) p ~ 0.0008, so the excess spread is real. **But look at where it comes
from**: 2022-23 and 2023-24 carry all of it; 2024-25 and 2025-26 show nothing
at all. Even the *within-season* signal is episodic.

**(B) Is it just form x schedule timing?** A season-constant team FE cannot
tell "this team is better at home" from "this team's home games fell in its
healthy stretch". Letting strength drift:

| specification | sd(d) | rms SE | tau | signal share |
|---|---|---|---|---|
| season-constant team FE | 3.525 | 3.030 | 1.800 | 26.1% |
| team x HALF-season FE | 3.471 | 3.024 | 1.702 | 24.1% |
| team x THIRD-of-season FE | 3.531 | 3.043 | 1.790 | 25.7% |
| team x MONTH FE | 3.543 | 3.173 | 1.575 | 19.8% |

**tau survives.** The within-season spread is not an artefact of form timing;
it is genuinely venue-linked (or at least home/road-linked) within the season.
It just does not persist to the next one.

### 1h. Power — what this test could NOT have found
The lag-1 CI half-width is ~0.137, so the smallest persistence this design
could resolve corresponds to a forecast SD of `0.137 * 3.525 = 0.48 points`.
**A genuinely forecastable team home effect smaller than about half a point is
invisible to this test and always will be at 30 teams x 5 seasons.** That is
not a loophole to reopen the vein: D70 tested exactly that region at the
logloss endpoint with the shipped stack and returned an exact null
(-0.00002 pooled). The descriptive test lacks power below ~0.5 pt; the endpoint
test says that even if something lives there it is worth zero.

### 1i. How many teams survive 5 seasons of pooling?
Pooled over all 5 normal seasons (205 home games each, SE ~1.35):

| team | pooled d | z | | team | pooled d | z |
|---|---|---|---|---|---|---|
| UTA | +2.734 | **2.02** | | PHI | -3.379 | **-2.50** |
| MEM | +2.354 | 1.74 | | WAS | -2.149 | -1.59 |
| IND | +2.130 | 1.57 | | BKN | -1.917 | -1.42 |
| LAL | +2.078 | 1.53 | | CHA | -1.641 | -1.22 |
| MIL | +1.633 | 1.20 | | SAC | -1.299 | -0.96 |

**2 of 30 teams exceed |z| = 2. With 30 tests you expect ~1.4 by chance.**
Nothing here survives a multiple-comparison correction.

Season ranks for the eight largest pooled home edges show the same thing —
there is no team that is consistently at the top:

| team | 21-22 | 22-23 | 23-24 | 24-25 | 25-26 |
|---|---|---|---|---|---|
| UTA | 2 | 19 | 1 | 22 | 6 |
| MEM | 5 | 1 | 27 | 7 | 10 |
| IND | 6 | 13 | 10 | 8 | 7 |
| LAL | 8 | 16 | 5 | 6 | 16 |
| MIL | 20 | 11 | 6 | 5 | 13 |
| OKC | 19 | 6 | 2 | 12 | 24 |
| DEN | 25 | 2 | 4 | 18 | 27 |
| HOU | 18 | 8 | 8 | 16 | 11 |

---

## 2. WHERE THE ~1.96 POINTS COMES FROM

Pooled regression, all 7 seasons, **season-specific team fixed effects** (so
opponent and own quality are controlled within season), a crowd-regime-specific
home intercept, and the schedule state of both sides. `HFA_r` is then the home
edge at zero schedule asymmetry ("pure" HFA), and
`crowd = HFA_pure(normal) - HFA_pure(nocrowd)`.

### 2a. The schedule asymmetry the visitor actually faces

| | home team | away team | away - home |
|---|---|---|---|
| b2b rate | 0.1601 | 0.1924 | +0.0324 |
| 3-in-4 rate | 0.2565 | 0.2716 | +0.0151 |
| days rest | 2.159 | 2.074 | -0.085 |
| travel km | 572 | 1,110 | +538 |
| \|tz shift\| | 0.315 | 0.569 | +0.253 |
| prior consecutive road games | 0.978 | 1.084 | +0.105 |

### 2b. The decomposition table

| component | points | 95% CI |
|---|---|---|
| **RAW home edge, normal seasons (6,140 g)** | **+1.958** | |
| travel/rest schedule terms, net | -0.111 | (-0.645, +0.444) |
| **crowd** (2020-21 natural experiment) | **+0.991** | **(+0.092, +1.911)** |
| residual (empty arena, symmetric schedule) | +1.074 | (-0.001, +2.125) |
| *check: sum* | *+1.954* | |

Pure HFA by crowd stratum: `nocrowd +1.074`, `normal +2.065`,
`2019-20 pre-shutdown +2.372`.

**Crowd is the single largest identified mechanism and it is ~half the total.**
It is also the only one with a genuine causal design behind it. Note that
"crowd" here means *everything mediated by the crowd*, which includes
crowd-driven referee bias — this design cannot separate the two.

### 2c. Travel and rest are NOT the mechanism
The table above measures the schedule terms against a *zero*-schedule baseline
(nobody has played or travelled). The question Sean actually asked —
*"how much of the home edge is just that the visitor is more often tired?"* —
is contrastive, so the counterfactual has to be **give the visitor the host's
schedule state**: `asymmetry_j = b_(away,j) * (E[X_away,j] - E[X_home,j])`.
Travel-km and |tz shift| are 0.77-0.86 correlated so they are reported as a
joint block (their individual coefficients are not separately interpretable).

Normal 5 seasons, `scripts/ha_decomp2.py`:

| component | points of home edge | 95% CI | |
|---|---|---|---|
| b2b asymmetry | +0.0564 | (+0.0174, +0.1054) | SIG |
| 3-in-4 asymmetry | +0.0090 | (-0.0069, +0.0301) | NS |
| rest differential | +0.0179 | (-0.0229, +0.0588) | NS |
| travel-km | -0.2071 | (-0.5684, +0.1442) | NS |
| timezone shift | +0.1211 | (-0.0674, +0.3103) | NS |
| **FATIGUE block** (b2b + 3in4 + rest) | **+0.0833** | (+0.0374, +0.1358) | **SIG — 4.3%** |
| **TRAVEL block** (km + tz, joint) | -0.0860 | (-0.3352, +0.1670) | NS — -4.4% |
| **TOTAL schedule asymmetry** | **-0.0027** | (-0.2585, +0.2595) | NS — **-0.1%** |

**The visitor's schedule disadvantage is worth statistically nothing. About 4%
of the home edge is the visitor being on a back-to-back more often; the travel
block is a non-significant *negative*.** Whatever home advantage is, it is not
tired legs and it is not the flight.

This is independently consistent with D17 (rest rejected, empirical B2B ratio
0.9973) and with D46's shape: the b2b effect is real and large *for the team on
the b2b* (`h_b2b -2.246`, `a_b2b +2.111` — ~2.2 pts either way), but it
contributes almost nothing to *home advantage* because the asymmetry is only
3.2 percentage points (host 16.0% vs visitor 19.2%). The visitor does fly
538 km more than the host; it does not cost them points once rest is
controlled.

> **CONVERGENCE WITH D136** (landed while this was being written; entirely
> different construction — differential regressors, the full shipped D46 layer
> as control, a wpct control, walk-forward refits). D136's margin-scale
> asymmetry diagnostic: **`trav_h -0.4568 CI(-0.812,-0.102) SIG` vs
> `trav_a +0.1264 ns`** — "the measurable travel cost sits on the HOME team
> that just flew home, not on the visitor, because away travel is the routine
> state." That is the mechanism behind the number in this table. My travel
> block is small and NS for the same reason, and my fit puts most of it on the
> home side too once travel-km and |tz| are read jointly (`h_travel -0.036`
> plus `h_tz -0.690`, i.e. a -0.238 pt contribution from the host's own recent
> travel). **Two independent constructions, same conclusion: travel is a real
> ~0.2-0.6 pt physical effect that does not produce home advantage, because it
> is not asymmetric in the direction people assume.**

### 2d. Altitude
`E[altitude gain]` over all home games is **0.22 m** — every team hosts and
visits a balanced slate, so altitude is purely **redistributive** and cannot
explain a single point of the *league-wide* home edge. What it can do is make
DEN/UTA's home edge bigger than average, and descriptively it does:

| | points | 95% CI | |
|---|---|---|---|
| DEN + UTA home, pooled | +2.179 | (+0.590, +3.651) | SIG |
| DEN home | +1.571 | (-0.880, +3.758) | NS |
| UTA home | +2.784 | (+0.334, +5.071) | SIG |

**Do not read this as a contradiction of D96.** It is a full-sample descriptive
estimate, and it is nonstationary in exactly the way section 1 predicts — the
season-by-season deviations are:

| | 19-20 | 20-21 | 21-22 | 22-23 | 23-24 | 24-25 | 25-26 |
|---|---|---|---|---|---|---|---|
| DEN | +2.08 | +0.91 | -3.58 | +8.67 | +5.23 | +0.21 | -3.50 |
| UTA | -1.84 | +6.71 | +4.75 | -0.85 | +9.34 | -2.41 | +2.84 |

DEN swings 12 points across adjacent seasons. A quantity with a +2.2 pt
full-sample mean and a 12-point season-to-season swing is not a forecastable
feature, which is precisely what D70 and D96 measured directly and got nulls
for. **Altitude stays closed.** The full-sample significance here is the
descriptive shadow of an effect that cannot be estimated in time to use it.

### 2e. What the decomposition implies about where a team-specific lever COULD live
Reading the three components against section 1:

- **Crowd (+0.99, ~51%)** — this is the mechanism most people mean by "home
  aura", and it is the biggest single piece. But a crowd effect that is
  *team-specific and stable* would show up as a stable `d_t`, and section 1
  says `d_t` has zero carry. Whatever crowds do, they do it to roughly the same
  degree for everyone, or they do it differently each year.
- **Travel/rest (~0.00)** — cannot be a team-specific lever because it barely
  exists at the league level. It is also already priced: D46 ships the b2b
  term, which is where the real (2.2-pt) effect lives.
- **Residual (+1.07, ~55%)** — familiarity with the rims/floor/sightlines,
  sleeping at home, no hotel, routine. This is the only bucket with the right
  *shape* for a team-specific effect (it is literally venue-specific), and it
  is also the bucket we have no instrument for. Altitude lives here and is
  measured above: descriptively real, temporally useless.

---

## 3. THE LAKERS / COACH-ERA QUESTION

JJ Redick became LAL head coach for 2024-25. Sean's claim is "the Lakers at
home under JJ Redick".

### 3a. LAL raw, by season (neutral games dropped)

| season | coach | home g | home margin | home W% | road margin | road W% | overall margin | home-road split |
|---|---|---|---|---|---|---|---|---|
| 2019-20 | pre | 31 | +9.32 | .742 | +5.56 | .812 | +7.41 | 3.76 |
| 2020-21 | pre | 36 | +3.53 | .583 | +2.06 | .583 | +2.79 | 1.47 |
| 2021-22 | pre | 41 | -0.54 | .512 | -5.56 | .293 | -3.05 | 5.02 |
| 2022-23 | pre | 41 | +3.22 | .561 | -2.07 | .488 | +0.57 | 5.29 |
| 2023-24 | pre (Ham) | 42 | +4.43 | .667 | -3.43 | .475 | +0.60 | **7.85** |
| 2024-25 | **Redick** | 41 | +4.83 | .756 | -2.39 | .463 | +1.22 | 7.22 |
| 2025-26 | **Redick** | 41 | +4.02 | .683 | -0.51 | .610 | +1.76 | 4.54 |

The raw picture is where the intuition comes from: LAL's home win rate under
Redick is .756 and .683 while their road margin is negative. **But the largest
home-road split in the entire corpus is 2023-24 — Darvin Ham's last season,
before Redick arrived.**

### 3b. Controlled home deviation, LAL vs the league

| season | LAL d_t | SE | z | league rank (1 = biggest) |
|---|---|---|---|---|
| 2019-20 | -2.063 | 3.191 | -0.65 | 22 |
| 2020-21 | +0.833 | 3.276 | +0.25 | 15 |
| 2021-22 | +2.563 | 3.105 | +0.83 | 8 |
| 2022-23 | -0.560 | 2.803 | -0.20 | 16 |
| 2023-24 | +4.748 | 3.043 | +1.56 | 5 |
| 2024-25 | +3.069 | 3.045 | +1.01 | 6 |
| 2025-26 | +0.567 | 3.182 | +0.18 | 16 |

**No single LAL season reaches z = 2.** The two best are 2023-24 (pre-Redick)
and 2024-25 (Redick's first), and 2025-26 — Redick's second — is rank 16 of 30,
i.e. dead average.

### 3c. The coach-era contrast, tested directly

Bootstrap (1,200x, era-pooled regression with season-specific team FE):

| team | pre-era d | Redick-era d | delta | 95% CI | |
|---|---|---|---|---|---|
| **LAL** | +1.245 | +1.921 | **+0.676** | (-4.345, +5.741) | **NS** |
| UTA | +3.795 | +0.106 | -3.689 | (-9.148, +1.502) | NS |
| MEM | +0.623 | +1.827 | +1.203 | (-4.125, +6.745) | NS |
| IND | +0.213 | +2.365 | +2.152 | (-2.859, +7.050) | NS |
| MIL | +1.752 | +1.904 | +0.152 | (-5.004, +5.792) | NS |

The analytic version of the same contrast, from the per-season estimates
(`z = delta / se`):

| team | Redick era (2 seas) | pre-Redick (5 seas) | pre-Redick, normal only (21-22..23-24) | era delta | z |
|---|---|---|---|---|---|
| **LAL** | +1.818 (z 0.83) | +1.104 (z 0.80) | **+2.250 (z 1.31)** | +0.714 | **+0.27 NS** |
| UTA | +0.215 | +3.621 (z 2.64) | +4.414 | -3.406 | -1.32 NS |
| MEM | +2.002 | +0.339 | +2.589 | +1.663 | +0.64 NS |
| IND | +2.463 | +0.209 | +1.907 | +2.254 | +0.87 NS |
| MIL | +1.995 | +1.836 | +1.392 | +0.159 | +0.06 NS |
| DEN | -1.641 | +2.661 | +3.439 | -4.303 | -1.66 NS |
| PHI | -4.828 | +1.759 | -2.413 | -6.587 | -2.55 SIG* |

\* PHI is the only |z| > 2 era delta in the set, and its pre-era number is
dominated by a single outlier season (2019-20 d = +10.87 on 31 home games).
With this many contrasts, one hit at p ~ 0.011 is what chance produces.

### 3d. The direct answer to Sean's question

**Is LAL's home advantage distinguishable from league average?**

| window | LAL d | 95% CI | n home | verdict |
|---|---|---|---|---|
| Redick era (24-25, 25-26) | +1.921 | (-2.718, +6.185) | 82 | **NS** |
| 5 normal seasons pooled | +2.091 | (-0.431, +4.696) | 206 | **NS** |

**Plain terms.** Yes, the Lakers are near the top of the apparent home-advantage
table — rank 3 of 30 over five seasons at +2.08 pts. No, that is not
distinguishable from league average (z = 1.53, and with 30 teams you expect
someone to be at +2 by chance). And the coach-era story specifically fails on
its own terms: **LAL's home advantage in the three normal seasons *before*
Redick (+2.25) is slightly LARGER than in the two seasons *under* him (+1.82)**,
and the era delta is +0.68 with a CI spanning ±5 points. The one number that
looks like the story — 2024-25's d = +3.07 — is a 1.0-sigma result sitting next
to a 1.6-sigma result from the year before, under a different coach.

The honest reading is the one Sean asked me to give if it were true: **"Lakers
are great at home" is mostly "Lakers were a good team with a bad road record",
plus a small sample.** Their overall margin in the Redick era is +1.22 and
+1.76 — a decent team. Split 41/41 with a 13.6-point per-game residual SD, the
home-road split of such a team wanders by ±3 points a season for free.

### 3e. The decisive test: is the TOP OF THE TABLE bigger than chance?
Residual-bootstrap null over the 5 normal seasons pooled (205 home games per
team): every team gets *exactly* the league home edge and no team-specific
deviation; real schedule, real strengths, real residuals; 2,000 draws.

| | observed | null | null [p5, p95] | p |
|---|---|---|---|---|
| biggest apparent home edge (max d) | **+2.627** (UTA) | **+2.738** | (+1.847, +3.832) | **0.519** |
| spread of home edges, sd(d) | 1.481 | 1.335 | (1.052, 1.628) | 0.203 |

**Under a world with literally zero persistent team-specific home advantage,
the biggest apparent home edge across 5 seasons would be +2.74 points — LARGER
than the +2.63 we actually observe.** The entire top of the table, LAL's +2.09
included, is what chance produces at 30 teams x 205 games.

**This is not in tension with section 1c** (`p = 0.003` that the within-season
spread is real). They test different things: `d_t` is genuinely non-zero
*within* a season and the season-specific effects average out across five
seasons because they are independent draws. Section 1c detects a season-
specific effect; section 3e tests for a persistent one and finds nothing.
Together they are the whole finding in one line: **real within a season,
zero across seasons.**

> **INDEPENDENT CONFIRMATION FROM D135**, on data this investigation never
> touched. D135 pulled the league's own published `leaguestandingsv3` HOME/ROAD
> split records for **360 team-seasons, 2013-14..2024-25** — 12 seasons, more
> than double this corpus — and measured the residual year-over-year
> persistence after removing team strength: **HOME +0.000, which D135 itself
> describes as "re-derives D70 from league data."** A completely different
> source, a completely different construction, twice the sample, same answer.
> Between D70 (endpoint null), D96 (altitude null), D135 (league-data
> persistence +0.000) and this document (lag-1 +0.021 NS, permutation p=0.519),
> the persistent-team-home-advantage hypothesis has now failed four independent
> tests. **It should not be re-opened without a fresh corpus and a new
> mechanism.**

---

## 4. PLAYER-LEVEL HOME SENSITIVITY (Sean's idea, built for the first time)

Per player-season, minutes-weighted WLS of a per-minute rate on a home dummy:
`rate_i = mu_p + delta_p * 1{home} + e_i`, weights = minutes (which is the
correct weighting: a counting stat divided by minutes has
`var ∝ 1/minutes`). Metrics: `pts36`, `gmsc36` (Hollinger Game Score per 36),
`ts` (true shooting). Inclusion: >= 5 min in a game, >= 250 minutes on **each**
side of the split. 160,477 player-games; 1,759 usable player-seasons over the
5 normal seasons.

### 4-0. The league-wide per-minute home effect (context)

| metric | home | road | diff |
|---|---|---|---|
| pts / 36 | 17.088 | 16.801 | **+0.288** |
| GameScore / 36 | 13.895 | 13.419 | +0.476 |
| true shooting % | .5790 | .5695 | **+0.95 pp** |

Sanity check: `0.288 pts/36 * (240 team minutes / 36) = +1.92 points` of team
scoring — which is the league home edge (+1.96) almost exactly. That is an
accounting identity rather than independent evidence, but it confirms the panel
reconciles. The `+0.95 pp` shooting-efficiency gap is the classic finding and
is where the points come from.

### 4a. Spread of player home sensitivity, and how much is noise

| metric | n (player-seasons) | mean delta | sd(delta_hat) | rms SE | tau | **signal share** |
|---|---|---|---|---|---|---|
| pts36 | 1,759 | +0.239 | 2.030 | 1.950 | 0.562 | **7.7%** |
| gmsc36 | 1,759 | +0.441 | 2.163 | 2.080 | 0.594 | **7.5%** |

### 4b. Reliability

| metric | split-half r | Spearman-Brown | lag-1 r | lag-1 95% CI | n pairs | lag-2 r |
|---|---|---|---|---|---|---|
| pts36 | +0.059 | +0.111 | **+0.066** | **(+0.014, +0.115) SIG** | 1,371 | +0.034 |
| gmsc36 | +0.067 | +0.126 | +0.047 | (-0.006, +0.099) NS | 1,371 | +0.006 |

**This is the one place in the whole investigation where a persistence
coefficient's CI excludes zero.** It is not because the effect is bigger — it
is because n is 1,371 player-pairs instead of 120 team-pairs. The team-level
test could never have found something this small (section 1h: MDE ~0.14).

### 4c. Is the true spread of player home-sensitivity > 0?
Parametric null (`delta_p ~ N(mean, se_p^2)`, i.e. zero true spread), 4,000
draws:

| metric | observed sd | null sd | null [p5,p95] | p | verdict |
|---|---|---|---|---|---|
| pts36 | 2.0295 | 1.9500 | 1.891-2.011 | **0.0163** | true spread REAL |
| gmsc36 | 2.1632 | 2.0796 | 2.016-2.145 | **0.0132** | true spread REAL |

So: some players really are more home-sensitive than others, by about
tau = 0.56-0.59 points per 36 minutes, and it carries weakly across seasons.
**That is a genuine positive finding and it is Sean's, not the literature's.**

### 4d. THE PRIZE — does roster aggregation beat team identity? **NO.**

Point-in-time construction: for each team-season *y*, EB-shrink every player's
home split using seasons strictly before *y*, then weight by that player's
share of the team's minutes in *y*. Head-to-head against the team's own
prior-season home deviation, both predicting the realised `d_t`:

| predictor of realised d_t | normal seasons (n=150) | 95% CI | |
|---|---|---|---|
| **roster-aggregated player sensitivity (PIT)** | **r = -0.0325** | (-0.171, +0.108) | NS |
| team identity (prior-season d_t) | r = +0.0243 | (-0.139, +0.194) | NS |
| difference (roster - identity) | -0.0568 | (-0.262, +0.134) | NS |

All-seasons version (n=180): roster -0.0065, identity +0.0451, difference
-0.0516 — same picture.

**And here is the twist worth remembering.** The roster aggregate IS far more
stationary than team identity — that part of Sean's intuition is correct:

| signal | lag-1 autocorrelation | per-pair |
|---|---|---|
| **roster-aggregated home edge** | **+0.486** | .566 / .304 / .315 / .745 / .497 |
| team identity (d_t) | +0.039 | .144 / .036 / .036 / .021 / -.042 |

So Sean's mechanism argument was structurally right — *aggregating a
player-level property does produce a stable team-level quantity where team
identity produces noise* — and the stable quantity turns out to have **no
relationship at all** with the team's realised home advantage. Stationarity of
a predictor is not predictive power.

Two reasons it dies:
1. **Scale.** The spread of the roster aggregate across teams is 0.043 per-36
   units = **0.286 points** at team scale (1 SD). Even a perfect signal of that
   size is below anything the stack can use.
2. **Attenuation ceiling.** Realised `d_t` has reliability 0.26, so the maximum
   correlation ANY perfect predictor of the true home deviation could achieve
   is `sqrt(0.26) = 0.51`. Observed is -0.03 with a CI half-width of 0.14, so
   the true correlation is bounded at roughly `|rho| < 0.28` — i.e. the roster
   aggregate explains under ~8% of the true within-season signal, and the point
   estimate is on the wrong side of zero.

**Verdict on (4): the player-level effect is real and mildly persistent; the
team-level aggregation of it is not a forecastable home-advantage feature.**
The place the player result might still pay is where it was measured — in
PROPS, as a per-player home/road prior on a per-36 rate — not in the sides
model. See the pre-registered gate in section 6.

---

## 5. WITHIN-SEASON TIME PROFILE (Sean's second hypothesis)

Sean's claim: home advantage "typically diminishes over the season" and
"should be extra strong on opening night and then weaken."

Every estimate below controls opponent/own quality the same way as section 1
(season-specific team FE) **and** the schedule confounds that move with the
calendar — because B2B rate, travel load and the rest distribution all drift
through the season, and without them you measure the schedule, not the crowd.
The realised schedule asymmetry per bucket is printed alongside so the drift is
visible. 5 normal seasons.

### 5a. The shape, by team-games-played

| bucket | n | raw | controlled | 95% CI | b2b asym | travel asym (km/1000) |
|---|---|---|---|---|---|---|
| gp 0-5 | 388 | +1.915 | +2.390 | (+0.944, +3.721) | +0.003 | +0.469 |
| gp 5-10 | 379 | +3.309 | +3.375 | (+1.922, +4.803) | +0.003 | +0.510 |
| gp 10-20 | 750 | +2.557 | +2.593 | (+1.525, +3.693) | +0.045 | +0.529 |
| gp 20-30 | 742 | +1.589 | +2.283 | (+1.078, +3.495) | +0.047 | +0.533 |
| gp 30-41 | 825 | +2.205 | +2.370 | (+1.108, +3.545) | +0.034 | +0.557 |
| gp 41-52 | 828 | +1.906 | +2.323 | (+1.167, +3.460) | +0.027 | +0.566 |
| gp 52-62 | 751 | +0.879 | +1.313 | (+0.128, +2.476) | +0.031 | +0.631 |
| gp 62-72 | 748 | +1.973 | +2.606 | (+1.482, +3.733) | +0.019 | +0.599 |
| gp 72+ | 729 | +1.914 | +2.005 | (+0.591, +3.318) | +0.032 | +0.469 |

Robustness with half-season team FE (so late-season strength drift and tanking
cannot masquerade as a home-edge change) gives the same shape:
2.32 / 3.31 / 2.84 / 2.20 / 2.59 / 2.33 / 1.68 / 2.49 / 2.08.

By calendar month:

| month | n | raw | controlled | 95% CI |
|---|---|---|---|---|
| Oct | 400 | +2.072 | +2.547 | (+1.190, +3.983) |
| Nov | 1105 | +2.637 | +2.690 | (+1.755, +3.746) |
| Dec | 1022 | +1.897 | +2.373 | (+1.356, +3.465) |
| Jan | 1140 | +2.020 | +2.492 | (+1.546, +3.513) |
| Feb | 842 | +1.232 | +1.501 | (+0.457, +2.589) |
| Mar | 1165 | +1.760 | +2.316 | (+1.254, +3.367) |
| Apr | 466 | +2.039 | +1.957 | (+0.326, +3.419) |

### 5b. The opening-night claim, tested directly

| window | n | EARLY | REST | DIFF | 95% CI | |
|---|---|---|---|---|---|---|
| opening night (day 0-1) | **67** | +0.454 | +2.407 | **-1.953** | (-5.377, +1.201) | NS |
| opening week (day <= 7) | 277 | +2.258 | +2.313 | -0.055 | (-1.779, +1.561) | NS |
| first 5 team-games | 388 | +2.404 | +2.297 | +0.108 | (-1.376, +1.550) | NS |
| first 10 team-games | 767 | +2.856 | +2.195 | +0.662 | (-0.365, +1.686) | NS |
| first 20 team-games | 1517 | +2.720 | +2.141 | +0.580 | (-0.189, +1.360) | NS |

**Opening night is n = 67 games across the whole corpus.** Its CI is ±3 points
wide — this window can never settle anything, and I am saying so plainly rather
than reporting the point estimate as a result. For what it is worth the point
estimate goes the *wrong* way for the hypothesis (opening night +0.45 vs +2.41
for the rest of the season).

### 5c. The trend test, and a fair hearing for the hypothesis

Linear in team-games-played (centred at game 41), same controls:

| specification | term | estimate | 95% CI | |
|---|---|---|---|---|
| linear | pts per half-season | **-0.377** | (-0.989, +0.255) | **NS** |
| | intercept (HFA at mid-season) | +2.280 | | |
| linear + quadratic | linear | -0.371 | (-0.960, +0.311) | NS |
| | quadratic | +0.260 | (-0.929, +1.451) | NS |

And the coarse 3-bucket version, which is the cleanest picture:

| stratum | early (gp<10) | mid | late (gp>=50) |
|---|---|---|---|
| **normal 5 seasons** | **+2.855** (+1.89,+3.96) | +2.371 (+1.49,+3.17) | **+1.970** (+1.12,+2.88) |
| 2020-21 NO CROWD | -0.740 (-3.32,+1.79) | +0.519 (-1.33,+2.33) | -0.249 (-2.82,+1.98) |
| 2019-20 pre-shutdown | +1.071 (-1.10,+3.24) | +2.065 (+0.34,+3.97) | +2.283 (-0.00,+4.70) |

**Prior-free description of the observed shape: essentially FLAT, with a mild
downward tilt that does not reach significance.** In the 5 normal seasons the
profile *is* monotone declining across three buckets (+2.86 -> +2.37 -> +1.97,
a drop of 0.89 pts end to end), and the fitted linear trend is -0.38 pts per
half-season, i.e. about -0.75 pts from opening night to game 82. **The sign is
Sean's.** The magnitude is not resolvable: the trend CI spans zero
(-0.99, +0.26), every bucket CI contains the league level, and the contrast
that would matter most (first-10 vs rest, +0.66) is NS.

Two things weaken it further:
- It does **not** replicate in the other crowd-normal stratum. 2019-20
  pre-shutdown goes the *opposite* way (+1.07 -> +2.07 -> +2.28).
- The largest single wiggle in the fine-grained table is the gp 52-62 / February
  dip (+1.31 / +1.50), which is the All-Star-break region and is a level shift,
  not a trend.

**And it specifically is NOT an opening-night effect.** Opening night itself
(n = 67) is the *lowest* point estimate in the entire table. If there is
anything real here it is a broad first-quarter-of-the-season elevation worth
well under a point, not a spike that decays.

### 5d. The no-crowd cross-check — this is the interesting bit
Sean asked for it, and it is the one part of section 5 that carries mechanism
information rather than just a null.

If the early-season elevation were crowd-driven (novelty, energy), **2020-21
should show a flatter or absent early bump.** It does: 2020-21's early bucket
is **-0.740**, the only negative early value in the table, against +2.855 for
the normal seasons — and 2020-21's profile has no early peak at all. Directional
support for a crowd reading of whatever early elevation exists.

The honest qualifier: 2020-21's three buckets have CIs of ±2.5 points each, so
this cross-check cannot carry weight on its own. It is consistent, not
probative. Combined with the section 2e dose-response (DiD +0.611, NS, right
sign) there are now two independent, underpowered, same-signed pieces of
evidence that what changes with crowd presence is the *level* of home
advantage, and there is no evidence that its *shape* over the season is
anything but flat.

### 5e. What the shape says about the mechanism

This is the diagnostic value of (5) and it lines up with section 2.

- A **travel/fatigue** mechanism should ACCUMULATE: visitors are more worn down
  in March than in November, so home advantage should RISE late. **It does not
  — the point estimate goes the other way.** The travel asymmetry genuinely does
  rise through the season (0.47 -> 0.63 -> 0.47 thousand km, peaking in the
  gp 52-62 band) while the home edge does not follow it up. **This
  independently corroborates section 2c from a completely different direction:
  travel is not the mechanism.**
- A **crowd/novelty** mechanism that faded would produce exactly the mild
  decline observed, and section 5d shows the decline is absent in the no-crowd
  season. That is the reading most consistent with the data — but the trend is
  NS and does not replicate in 2019-20, so it stays a hypothesis.
- Net: the season profile is dominated by **level** mechanisms — crowd presence
  and venue familiarity — not by anything that accumulates. Same conclusion the
  decomposition reaches by a different route.

---

## 6. FOLLOW-UP GATES — PRE-REGISTERED, NOT RUN, NOT SHIPPED

Nothing in this document ships. Two candidates emerged; both are written down
here as specifications so that if either is ever run, it is run once, against a
test written before the data was looked at.

### GATE HA-1 (from section 4) — PLAYER HOME/ROAD PRIOR IN **PROPS**, not sides
**Rationale.** Section 4 found the only significant persistence in the whole
investigation: player home-sensitivity in pts/36 carries at lag-1 r = +0.066
CI(+0.014, +0.115), with a true spread of tau = 0.56 pts/36 (p = 0.016 against
zero spread). Section 4d killed the *team-margin* use of it. It was measured on
a per-player per-36 rate, which is exactly a props object.

**Exact construction (one config, no sweeps).** At each props refit, for each
player, EB-shrink the home-minus-road split of the modelled per-36 rate using
only seasons strictly before the current one, with the shrinkage weight
`w_p = tau^2 / (tau^2 + se_p^2)` and `tau` re-estimated by method-of-moments on
the same prior window. Apply `+/- w_p * delta_p / 2` to the player's projected
rate for home/road respectively. Players with < 250 prior minutes on either
side get `delta_p = 0`.

**Endpoint and gate.** Props CRPS on the certified props harness, paired
bootstrap 2000x vs a same-run control, ships only if the pooled 95% CI excludes
zero (G1 / GATE_POLICY_V2). **MDE must be stated before scoring.**

**Adverse prior, recorded now.** I expect this to FAIL or land hairline. The
effect is `0.56 * 0.11 (reliability) ~ 0.06 pts/36` after shrinkage, which on a
30-minute projection is ~0.05 points of a scoring line. The honest reason to
run it anyway is that props endpoints have far more n than sides and this is
the only place the corpus showed a persistent signal at all.

**What would make me withdraw it:** if the props pipeline already conditions on
home/away anywhere in the rate construction, this is double-counting and the
gate should not be run at all. **Check that first.**

### GATE HA-2 (from section 5) — SEASON-PHASE INTERACTION ON THE D46 HOME EDGE
**Status: registered NO-GO on this corpus.** Sean asked for the gate to be
specified if the profile turned out real. The point estimate has his sign
(-0.38 pts per half-season) but the CI spans zero and the pattern does not
replicate in 2019-20. Writing the test down so the question is closed with a
reactivation condition rather than left dangling.

**Exact test, if it is ever run.** In `fit_schedule_layer`, replace the scalar
`he_global` with `he_global + phase_beta * ((mean_gp - 41) / 41)`, with
`phase_beta` fitted walk-forward on the same trailing 730-day window that
already fits `he_global`. ONE config, no sweeps, no bucket search. Endpoint:
pooled logloss on the certified 5-season corpus, paired bootstrap 2000x vs a
same-run control, ship only if the pooled 95% CI excludes zero. Control-hash
field per D134(3): name the control CSV, the env vars, and max|dp|.

**Why it must NOT be run now.** The descriptive precursor is a null
(CI -0.989 to +0.255), so running the endpoint gate would be a second look at
spent data — the exact failure mode GATE_POLICY_V2 exists to prevent. Note also
that the effect is bounded above by ~0.75 pts of home edge spread across a whole
season, applied to a term whose total size is ~2 pts; D46's whole schedule layer
was worth +0.0054 logloss, so a quarter-scale modulation of one of its terms is
comfortably below MDE on 6,148 games.

**Reactivation condition (write it down, honour it):** eligible on a fresh
corpus (2026-27 onward, i.e. seasons not used here), and only if the linear
phase trend reappears with the SAME NEGATIVE SIGN and a CI excluding zero in the
descriptive test first. Two conditions, both pre-stated.

---

## 7. CAVEATS, AND WHAT WOULD CHANGE MY MIND

1. **Everything except 4d is full-sample.** These are descriptions of the past,
   not walk-forward estimates. The only walk-forward statement in the document
   is section 4d's head-to-head, and it is a null.
2. **Power.** The team-level lag-1 test cannot resolve a persistence
   corresponding to a forecast SD below ~0.48 points (section 1h). This
   document *cannot* prove there is no small persistent team home effect. What
   it can say — combined with D70's endpoint null — is that if one exists it is
   below the size that moves logloss.
3. **The crowd estimate leans on one season.** 2020-21 is 1,080 games with a
   compressed 72-game schedule and series-style repeat visits. The schedule
   controls absorb the travel/b2b differences (2020-21's h_b2b 0.203 vs normal
   0.159 is explicitly in the fit), but 2020-21 also had COVID protocols,
   unusual absences and a different competitive environment. **+0.99 pts should
   be read as an upper bound on crowd, not a point estimate.** The one internal
   check available — the within-2020-21 dose response as limited attendance
   returned in March 2021 — gives DiD **+0.611 CI(-1.203, +2.515) NS**: the
   right sign, no power.
4. **"Crowd" includes crowd-mediated referee bias.** The no-crowd design cannot
   separate them, and probably nothing available to us can.
5. **The specification choice moves lag-1 a little, in the favourable
   direction.** Under team x half-season FE lag-1 is +0.057 and under team x
   month FE it is +0.097, versus +0.021 with a season-constant FE. All are NS
   and all imply a forecast SD under 0.35 points, but the direction is
   consistent and worth noting rather than hiding: controlling within-season
   strength drift makes team home advantage look *slightly* more persistent,
   not less.
6. **What would change my mind.** A lag-1 correlation above ~0.25 on a fresh
   corpus, or a DEN/UTA-specific walk-forward estimate that stays positive
   across an era boundary. Neither has happened in three attempts (D20 -> D70 ->
   D96) and this document explains why: the estimator has a 3.0-point standard
   error on a 1.8-point quantity that resets every October.

---

## 8. FILES

| file | what |
|---|---|
| `scripts/ha_panel.py` | builds the game panel (home/away from `matchup`, neutral flag, rest/b2b/travel/tz rebuilt for 7 seasons, strata per D131) |
| `scripts/ha_core.py` | shared estimator: the identifying regression, EB shrinkage, bootstrap helpers |
| `scripts/ha_stationarity.py` | section 1 |
| `scripts/ha_tau_check.py` | section 1g (SE validation + strength-drift kill) |
| `scripts/ha_decomp.py` | section 2 (crowd, altitude, dose response) |
| `scripts/ha_decomp2.py` | section 2c (schedule-asymmetry counterfactual) |
| `scripts/ha_did21.py` | section 2 caveat 3 (DiD CI) |
| `scripts/ha_lakers.py` | section 3 |
| `scripts/ha_player.py` | section 4 |
| `scripts/ha_timeprofile.py` | section 5 |
| `data/homeadv_notes.md` | running checkpoint notes |

All read-only against `data/nba.duckdb` (`read_only=True`, 60s retry on lock).
Seed 20260801 throughout. Intermediate artifacts (the `d_t` matrix, its SEs,
the shrunk version, player splits, roster panel) are in the session scratchpad,
not in `data/`, because none of them is a production input.
