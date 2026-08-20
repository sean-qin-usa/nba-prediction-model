# D230 PRE-REGISTRATION — CHANNEL-WISE TRUST IN THE OFFSET LAYER

Hashed before any challenger log loss existed. Written against the frame
produced by `PROD_SEASONS=2019-20..2025-26 COMPONENT_OUT=...
scripts/prod_by_season.py`, whose per-game channel sums are asserted equal to
`margin()` on every game.

## 1. THE QUESTION

The shipped offset (D224/D226) spends the WHOLE model-market disagreement at a
single trust coefficient:

    m_offset = m_open + b*(m_blind - m_open) + g*rest_diff + d*|m_open|
    b = +0.3564, g = +0.0417, d = -0.0114

But `m_blind` is a sum of five channels with very different relationships to
what a bookmaker prices:

    m_blind = c_ff + c_comp + c_sched + c_tank + c_late

A single `b` asserts every channel deserves identical trust. That is an
assumption, never tested.

## 2. THE CHALLENGER, AND WHY IT IS NESTED

    m_offset = m_open + b*(m_blind - m_open) + SUM_k d_k*c_k
                      + g*rest_diff + d*|m_open|

with an L2 penalty `lam * SUM_k d_k^2` on the DEVIATIONS ONLY; b, g, d are
unpenalised. `d_k` is the extra trust channel k earns beyond the common b.

**At lam -> infinity every d_k -> 0 and this is EXACTLY the shipped model.**
The incumbent is therefore a point in the challenger's family, not a separate
fit, and `lam` is selected walk-forward on prior seasons only. This is D198's
"shrink toward the incumbent" rule, which the register requires: the null is
the shipped model, never zero.

## 3. PREDICTIONS, WITH DIRECTIONS, BEFORE SCORING

- **T1  d_sched < 0.** Rest, back-to-backs and home edge are the most public,
  most heavily modelled inputs in the market. The opener should already price
  them, so this channel should earn LESS than the common trust.
- **T2  d_comp > 0.** Availability is the least-priced channel at the open —
  D199 measured 18.1% of the minutes-weighted out-set as posting AFTER the
  line, and D202 showed soft availability alone was worth -0.0023 nats. If any
  channel deserves extra trust it is this one.
- **T3  |d_k| small; the fitted lam large.** D157's finding was that the lever
  is the PENALTY, not the basis. A re-parameterisation of the same information
  should not create any.
- **T4  THE HEADLINE PREDICTION IS NO-SHIP.** Pooled gain over the incumbent
  < 0.001 nats, with a season-clustered CI containing zero. Reasons stated up
  front: three of the five channels (sched, tank, late) are near-constant or
  exactly zero on most games, so they carry almost no cross-game variance to
  earn a coefficient with; and D198 already fitted 2 of these 5 channels
  (m_ff/m_comp) freely and moved RMSE by 0.12%, ns.

Recording T4 in advance because a NO-SHIP that was predicted is evidence about
the architecture, while a NO-SHIP found after the fact is only a null result.

## 4. GATE (GATE_POLICY_V2 §8)

- Statistic: season-clustered mean delta in log loss vs the shipped offset,
  centred on the UNWEIGHTED mean of per-season estimates (`oc.cluster_mean_t`).
- 95% CI at K-1 dof must EXCLUDE ZERO. K = 7 seasons (2019-20..2025-26).
- Better in >= 5 of 7 seasons.
- Calibration veto: mean forecast probability must not drift from base rate by
  more than the incumbent's own drift.
- MDE80 stated BEFORE the endpoint is read, from the frame's own dispersion.
- Walk-forward: `lam` and all coefficients fitted on seasons 1..k, scored on
  k+1 only. Nothing refitted on the season being scored.

## 5. STOP CLAUSE

If the CI contains zero, the arm is NO-SHIP, `nbapred/market/offset.py` is NOT
touched, no ship diff is written, and the incumbent single-b offset stands. A
per-channel coefficient that is merely INTERESTING does not ship.

## 6. WHAT IS DIAGNOSTIC AND CANNOT SHIP

The fitted `d_k` signs are reported as diagnosis of what the opener prices
regardless of the gate outcome, because that is informative for the props and
conditional-trust work even when the arm itself is inert. They are labelled
diagnostic and are NOT a licence to hand-pick a channel subset and re-gate it
(the second-look trap, D229 §5.2 precedent).
