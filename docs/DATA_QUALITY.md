# Data quality & era effects on training

Answers two risks Sean raised: (1) does poor/missing recording bias the model,
and (2) does the changing quality/style of play across eras affect training.
Both are real; both are managed by design, not luck.

## 1. Missing / poor recording

**Coverage degrades going back in time** — richer features have shorter
histories (approximate reliable spans; verify empirically when we backfill
history):

| source | reliable from | notes |
|---|---|---|
| Traditional box scores | ~1946 | robust for decades |
| Play-by-play (events) | ~1996-97 | our event tree's floor |
| Shot location (x/y, distance) | ~1996-97 | improves over time; modern = complete |
| GameRotation (stints) | ~2007-08 | spotty earlier |
| Player tracking (defended-FG%, speed/dist, touches) | 2013-14 | SportVU/Second Spectrum |
| Hustle stats (box-outs, deflections) | 2015-16 | |
| Synergy play-types | 2015-16 | v1.5 tendency priors |
| Official injury-report PDFs | ~2017 | standardized league feed |

**The failure mode is treating "not recorded" as "zero."** A player with no
recorded three-point attempts is *unknown*, not a 0% shooter. Defenses:

- The EB skill-prior estimator (`skill_priors.py`) keys on attempts/exposure, so
  a 0-attempt player shrinks to the **league mean**, never to 0. Verified: a 0/2
  shooter lands at ~league average, not 0.00.
- The Bayesian model treats unobserved dimensions as *unobserved*: the posterior
  stays near the (wide) prior rather than being pulled to a spurious value. Poor
  data → wider uncertainty, not bias — exactly the property H-A relies on.
- Each feature layer trains only over its reliable span (shot-quality/tracking
  on 2013+, Synergy on 2015+); the core event tree uses the longer PBP history.
- **Systematic** recording bias (e.g. era differences in how assists/blocks were
  scored) is worse than random gaps and is absorbed by the era/regime fixed
  effects below — never pooled as if comparable.

**Current corpus (2025-26, via `scripts/audit_coverage.py`)**: 0% shots missing
coordinates, every boxscore has matching PBP, empty-minutes rows are DNPs. Modern
data is clean; the audit script is the standing check to run on any historical
backfill before fitting.

## 2. Era / quality-of-play effects

**Yes — the game itself changed, and pooling eras naively biases the fit.**
Pace, three-point rate (~15 3PA/game in the 1990s vs ~35 now), hand-check
(removed 2004-05), defensive-three-seconds (2001-02), the take-foul rule (2022),
athleticism and spacing — all shift what raw stats *mean*. League baselines
(pace, ORtg, 3PA rate) drift every season.

Handling:

- **Season/regime fixed effects on league baselines.** The prior intercepts
  `α_k` are era-specific, so "average" is measured against contemporaries, not a
  pooled cross-era mean. Without this, a modern average scorer looks
  above-average vs a 1999 baseline — a bias.
- **Regime guardrail (handoff I.6):** never train across a rule-regime break;
  freeze/segment calibration at each break.
- **Recency weighting:** recent seasons weigh more — today's game is what we
  predict, so today's style dominates.
- **Predict within-era, relative to contemporaries.** Game prediction only needs
  a player's skill *relative to who he plays against this season*. We do NOT
  attempt cross-era absolute talent comparison (is 2024-LeBron > 1999-Jordan) —
  it's unnecessary for pricing games and is a modeling trap.
- **Pool cross-era only for era-invariant parameters.** Some effects are ~physics
  (a height/length edge at the rim, a speed edge on drives) and can borrow
  strength across eras; the hierarchical structure lets the data separate them
  from era-relative quantities if they aren't actually invariant.

## Practical status
We currently ingest 2025-26 only, so era-mixing is not yet an active risk — but
the skill model must be regime-aware before ANY multi-season fit. The reliable-
span table above dictates which feature layers can use which years.
