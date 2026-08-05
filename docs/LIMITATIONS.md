# Limitations & caveats (read before trusting any number)

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

Consolidated list of what is NOT yet true, what the current numbers do and don't
mean, and where the known weaknesses are. Cross-refs: COMPLEXITY, LEAKAGE,
DATA_QUALITY, PRIORS, SIGNALS, GAME_MODEL_SCOPE.

## Status honesty
- **No production model exists.** The possession engine is a v0 skeleton and the
  Bayesian fit currently covers only the shooting dimensions. The system can
  simulate a game and score itself, but it is not yet a trustworthy forecaster.
- **FIRST BACKTEST FAILED (expected).** On 1110 walk-forward 2025-26 games the v0
  engine scored log loss 0.6925 — a coin flip — LOSING to the home base-rate
  (0.6882) and badly to a simple Elo (0.6182) on the same games. Team-strength
  signal is real and Elo grabs it; the bottom-up engine currently captures none
  of it (win probs collapse to ~0.5). Root causes: rates composed from RAW,
  opponent-UNadjusted trailing box stats; shooting-only; over-shrunk team defense
  so teams barely differ; no RAPM/pace. This is I.5 internal-validity failing on
  a SKELETON — not a verdict on the full model. Fix = opponent-adjusted full fit
  (RAPM defense + pace + TOV/reb), then re-test vs Elo. Do not paper over it by
  tuning shrinkage on partial data.
- **All current model numbers are on a PARTIAL, SINGLE season (2025-26).** The
  backfill is still running (~500-600/1361 games as of 2026-07-27). Team/player
  estimates on 3-15 games are noisy; projections are pipeline smoke tests, not
  verdicts.

## Data coverage
- **No market odds for 2025-26 games.** SBR historical ends 2022-23; the live
  Odds-API logger has been in the offseason with nothing to capture. So the
  engine cannot yet be scored against the market or a well-calibrated Elo on
  current games — only against naive baselines (coin flip, home base-rate).
- **Market/Elo baselines (0.589 / 0.617) were measured on 2007-2023 SBR data**,
  a different game set than the model is currently fit on. Not a controlled
  head-to-head yet.
- **Odds free-tier ($0) coverage gap 2023-24 & 2024-25** (SBR stopped, others
  blocked). See PAID_OPTIONS.md.
- **Ratings snapshots are current-only.** DARKO/2K are today's snapshots →
  FORWARD-ONLY, not usable as backtest priors for past seasons (would leak).
  Historical rating snapshots not yet collected. See LEAKAGE.md.
- **Injury reports: none in offseason.** The minutes model (II.4) depends on them
  and is deliberately deferred until October data exists.

## Model scope not yet built
- **Defense is team-level only.** The engine applies the defending TEAM's
  suppression to shots (incl. 3PT), but individual "who guards the shooter"
  matchup weights (II.3.4 `Σ wᵢⱼ·defender`) need fitted individual defensive
  skill (from the RAPM/stint term) + tracking defended-FG% (v1.5). Not in yet.
- **Fit covers shooting only.** Turnover, rebounding, foul-draw, and the
  identification-critical stint-margin (RAPM) likelihood are not yet fit.
- **Engine v0 known miscalibration:** simulated game-to-game variance is too high
  (margin SD ~22 vs real ~13) because pace is modeled as Poisson (over-dispersed)
  and the two teams' paces are independent. Fix with the clock model (II.3.7).
- **Timeouts, momentum, clutch/score-state behavior, coach in-game adjustments**
  are out of v1 scope (GAME_MODEL_SCOPE.md).
- **No age curves, no rookie college-translation priors, no play-type (Synergy)
  tendencies, no tracking/shot-quality layer** yet (all v1.5).

## Methodological caveats
- **β trust weights are attenuated** by the small outcome sample; they will grow
  and stabilize with the full backfill. The FT-highest-β result is a valid
  ordering check, not a final magnitude.
- **Era comparability:** cross-era pooling of results/ratings is biased (rule
  regimes, pace, 3PT era); must use season/regime fixed effects + recency
  weighting before any multi-season fit. Not enforced in code yet (moot while
  single-season). See DATA_QUALITY.md.
- **Team defensive factor is EB-shrunk but crude** (opponent FG% vs league); it is
  a stand-in until real defensive skills are fit.
- **Every future feature must pass the walk-forward ablation gate** before it
  enters the model (COMPLEXITY.md). Nothing here is exempt.

## Infra / ops
- **Runs on Sean's workstation, not a VPS.** Home-box downtime = capture holes
  once the season starts. The backfill has already been killed once mid-run
  (resumable via cache, but a reminder uptime isn't guaranteed).
- **numpyro is CPU-only** here (no CUDA jaxlib for the RTX 2070); fine for current
  model sizes, may bound the full possession-level MCMC (v2).
