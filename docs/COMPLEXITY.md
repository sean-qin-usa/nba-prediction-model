# Managing model complexity — the practical mechanisms

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

Sean's question: how do we stop ourselves adding things that hurt the model
through complexity, *besides thinking rationally?* Rational thought doesn't
survive contact with 100 feature ideas. These are the mechanisms — tooling and
rules that make the decision for us. (Sean's own ERRORS.md A4/A5/A6 are the
failure modes this is built to prevent.)

## The single biggest structural defense: prefer structure over parameters

Complexity is dangerous in proportion to **free parameters relative to data**.
This project has two levers most ML projects lack:

1. **The simulator's synergy/complementarity terms cost ZERO free parameters**
   — they're interactions of existing player ratings inside the possession
   logits (passer × finisher, gravity-sum spacing). Adding "synergy" doesn't add
   parameters, so it can't overfit the way a `player_i×player_j` regression term
   would. *Rule: express a new effect structurally (as a function of parameters
   we already have) before ever adding a free parameter for it.*
2. **The Bayesian priors ARE regularization, automatically.** A weak new term
   with a sensible prior gets a posterior near zero — it costs a little variance,
   not a blow-up. Residual synergy `ρ_ij` is shrunk hard by construction. *Rule:
   every new free parameter ships WITH a prior tight enough that it must overcome
   the prior to move. No flat priors on speculative terms.*

## The gate every addition must pass (tooling: nbapred/eval/)

No feature/term enters the production model on the strength of a story or an
in-sample gain. It must pass, in order:

1. **Walk-forward OOS improvement, bootstrap-gated.** `eval/ablate.py` computes
   the OOS log-loss change vs the current model and block-bootstraps its 95% CI.
   **Keep only if the CI clears zero.** This is the mechanical version of "does
   it help?" — it judges the gain against its own sampling noise, so we can't
   fool ourselves with a lucky season. (Demonstrated live in
   `scripts/eval_baselines.py`: the positive-control feature passes with CI
   (0.0085, 0.0146); a null feature would straddle zero and be cut.)
2. **Calibration battery (III.2) as a veto.** A more complex model that improves
   log loss but breaks marginal calibration (pace, ORtg, 3PA rate, totals sd,
   foul-out rate…) is REJECTED regardless. `eval/metrics.py` tracks ECE +
   reliability alongside log loss precisely because they can disagree — in the
   demo, the market blend improved log loss but *worsened* ECE. Two metrics, two
   gates.
3. **Deflated significance across the whole test family (I.6).** We will try
   many features; the more we try, the more one passes by luck. Correct the
   keep-threshold for the number of things tested this cycle (deflated-Sharpe /
   Bonferroni-style). *Rule: log every feature tested, including the losers, so
   the deflation denominator is honest.*
4. **Predictive information criteria for the Bayesian fits.** Compare model
   variants by PSIS-LOO / WAIC (expected log predictive density), which penalize
   effective parameter count directly — not by in-sample likelihood. A term that
   raises in-sample fit but lowers ELPD is complexity, not signal.

## Process rules (prevent the churn failure mode, ERRORS A6)

- **Ablation is pairwise and cumulative.** Sum-of-parts must reconcile with the
  whole (ERRORS A5): when two features are added, also run each-disabled to
  catch interactions that a solo test misses.
- **One change per evaluation.** Never bundle features into a version; you can't
  attribute the delta. (This is why `ablate.py` compares one variant to base.)
- **Hard complexity budget, set from data.** ~30 teams × ~82 games/season. The
  effective sample for any context bucket must justify its parameters; thin
  buckets get pooled, not fit. Track parameter count per model version.
- **The holdout is sacred.** Walk-forward by season; the most recent season(s)
  are never looked at during development (III.3). Regime breaks freeze fits
  (I.6) — never train across a rule change.
- **Removal is a first-class action.** Re-run the gate on existing features
  periodically; anything whose CI no longer clears zero gets cut. Complexity
  decays in value as the league changes.

## Default posture

When unsure, DON'T add it. The null zone (I.3 item 7) is a feature of the
design, not a gap. A simpler model that we can calibrate and trust beats a
richer one we can't — and the whole edge thesis (H-A) is a *conditional* claim
in a few flagged windows, not a general "we model basketball better than
everyone" claim that would tempt us to bolt on everything.
