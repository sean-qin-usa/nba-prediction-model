# D232 PRE-REGISTRATION — ABSENCE-RESPONSE CORRECTION

Hashed before any challenger log loss existed.

## 1. THE FINDING THIS ACTS ON (D231b)

Regressing the production model's margin residual on the expected-absence
differential, bet-time clean (expected outs come from the walk-forward as-of-open
P(out) artifact, D201):

    residual = -0.2577 - 0.5653 * out_diff      pooled, t = -7.39, n = 8,239
    season-clustered slope -0.5367, 95% CI [-0.7348, -0.3385], same sign 7/7

**The model UNDER-penalises absences by roughly half a point of margin per
expected absence.** The composition leg already removes each absent player's own
`talent x minutes`, so this is a SECOND-ORDER effect: an absence costs more than
the departing player's individual contribution, which is what D133 arm C and
D144 predicted when they found promoted replacements underperforming the
production implied by their bench rates.

## 2. THE CHALLENGER

    m' = m_total + beta * out_diff        out_diff = E[outs_home] - E[outs_away]

`beta` fitted walk-forward on prior seasons only. **beta = 0 IS the shipped
model**, so the null is the incumbent, not zero (D198's rule, as in D230).

Nothing new enters the information set: `out_diff` is a function of the SAME
P(out) artifact the composition leg already consumes. This is a recalibration of
how the model aggregates availability, not a new feature.

## 3. THE CONFOUND THAT WOULD KILL IT, AND THE CONTROL

Bad teams may rest and tank more, so `out_diff` could proxy TEAM STRENGTH rather
than availability. **A second arm therefore controls for the model's own margin:**

    m' = m_total + beta * out_diff + gamma * m_total

If beta survives with the control, the effect is availability. If beta collapses,
it was strength wearing availability's clothes and the arm is a NO-SHIP whatever
the log loss says.

## 4. PREDICTIONS, BEFORE SCORING

- **T1  beta < 0**, near the -0.54 measured on margins.
- **T2  The arm SHIPS.** Unlike D230 this is not a re-parameterisation of the
  same signal: it corrects a measured, monotone, 7/7-season bias whose t is
  -6.63. A half-point of margin is ~0.07 of a sigmoid link scale, which is a
  real probability move, not a rounding.
- **T3  beta SURVIVES the strength control** with less than half its magnitude
  lost. Absences are only weakly correlated with team quality within a season.
- **T4  The gain is SMALLER than the margin arithmetic suggests**, because
  out_diff is symmetric and near-zero on most games (sd 1.97, median 0).

## 5. GATE

- Season-clustered mean delta in log loss vs the shipped production margin,
  unweighted mean of per-season estimates, 95% CI at K-1 dof must EXCLUDE ZERO.
- K = 7 (2019-20..2025-26); better in >= 5/7.
- Calibration veto: mean forecast probability must not drift further from base
  rate than the incumbent's own drift.
- MDE80 from a within-season permutation null of `out_diff`, stated before the
  endpoint is read.
- Walk-forward: beta fitted on seasons 1..k, scored on k+1 alone.

## 6. STOP CLAUSE

If the CI contains zero, or if beta does not survive §3's strength control, the
arm is NO-SHIP and `nbapred/model/production.py` is not touched.
