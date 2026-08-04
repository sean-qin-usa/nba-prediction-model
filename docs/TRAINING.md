# How training works (and how priors get updated)

Short version: "training" here is **Bayesian posterior estimation of latent
player skills**, not gradient descent on a loss. The priors (2K / DARKO /
trailing stats) are the *starting belief* about each player; the game data is
the *likelihood* that updates that belief into a posterior; new games keep
updating it. So yes — we are continuously updating player-skill priors, in the
precise Bayesian sense.

## The objects

- `θ_{i,k}` — player i's latent skill on dimension k (3PT, rim finish, rim
  protection, TO propensity, …). Unobserved; what we estimate.
- **Prior:** `θ_{i,k} ~ Normal(α_k + β_k · rating_{i,k}, σ_k²)`, hierarchical
  (position → league). `rating` is the external prior center (2K z-score /
  DARKO / trailing stat); `(α_k, β_k)` are learned league-wide, so the model
  *learns how much to trust each rating source per dimension* (β).
- **Likelihood — two, jointly:**
  1. *Event likelihood* — the possession outcomes (makes/misses by zone, TOs,
     rebounds, fouls) as binomial/multinomial counts, with each probability
     produced by composing the on-court players' `θ` through the possession
     engine's logits. This is what `player_game_stats` feeds.
  2. *Stint-margin likelihood* — a RAPM-style term on lineup point margins that
     pins each player's TOTAL two-way impact (defense is underidentified by
     events alone). This is what `lineup_stints` feeds. Identification-critical.

## The training loop (v1)

```
1. Pick a walk-forward cutoff (train on games strictly before it).
2. Assemble sufficient statistics up to the cutoff  (pit.trailing_* — no leak).
3. Set priors from as-of ratings (pit.*_asof — never a future snapshot).
4. MCMC (NUTS; PyMC/numpyro) samples the joint posterior over every θ and the
   shared α, β, σ, given prior × (event likelihood × stint likelihood).
   → VI for fast iteration, full MCMC for the production fit.
5. Output: a POSTERIOR DISTRIBUTION per player per skill — wide where data is
   thin (rookies), tight where it's rich (veterans).
```

There is no single "accuracy" being maximized in step 4 — MCMC explores which
skill values are *consistent with the observed events and margins*. Predictive
accuracy is judged separately, out-of-sample, by the walk-forward harness
(log loss / calibration vs the market, per docs/COMPLEXITY.md).

## Updating priors as the season goes (your question, directly)

Three coupled mechanisms:

1. **Posterior → next prior.** After a fit, each player's posterior becomes the
   prior for the next fit. New games' likelihood shifts it. That *is* "updating
   the player stat priors."
2. **Form / recency.** Players drift, so the update isn't a flat pool: an AR(1)
   term pulls the skill toward the player's own mean between updates, and the
   likelihood carries an exponential time-decay so recent games weigh more
   (II.1; v1 uses the decay shortcut).
3. **External anchors refresh.** DARKO re-ingests daily and 2K periodically;
   as-of joins mean tonight's fit uses the latest snapshot *available before
   tip*, never a future one.

Cadence: a cheap incremental update daily (decay + new games), a full MCMC
refit on a slower cycle (e.g. weekly), refits frozen across rule-regime breaks.

## Are substitutions modeled? Yes.

Subs are not an afterthought — they're the mechanism behind the whole edge.
The engine's state includes the current 5-man lineups; the **minutes/rotation
model (II.4)** projects who is on the floor and for how long, and **stint
assembly** builds the sequence of lineups a game runs through. Different
lineups → different `θ` compositions → different outcome probabilities. That is
exactly what `lineup_stints` trains and what H-A monetizes (pricing
lineup-composition changes). Foul-trouble and injury/ejection triggers that
force subs are in-engine contingencies.

## Timeouts, and everything else about a game

Timeouts are **not** in v1 (a second-order clock/momentum effect). Every game
element — modeled, planned, or deliberately out of scope — is cataloged in
docs/GAME_MODEL_SCOPE.md so nothing is lost track of.
