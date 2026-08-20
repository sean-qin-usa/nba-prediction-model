# D246 PRE-REGISTRATION — OPENER-TO-CLOSE MOVEMENT TEACHER (two heads)

Hashed before any challenger score existed.

## WHY THIS AND WHY NOW

D245d measured a ~14x two-stage attenuation: a composition-channel improvement
is multiplied by the blend share (0.652) and then by the offset edge coefficient
(0.3413) before reaching the forecast. Items acting on that channel inherit the
haircut. **The movement teacher bypasses the blend entirely** — it predicts a
market quantity directly.

The target is also far less noisy: `close - open` has sd 2.303 against 13.60 for
the outcome residual `Y - O`, a **5.9x variance reduction**.

## DECOMPOSITION

    Y - O = (C - O) + (Y - C)
            movement    residual-after-close

Two separately regularised heads, both fitted on OPENER-TIME FEATURES ONLY:

    head M  predicts (C - O)   — information the market will absorb by close
    head F  predicts (Y - C)   — information that may remain mispriced at close

**THE CLOSING LINE IS A TRAINING TARGET ONLY AND IS NEVER AN INPUT.** Asserted
in code: the feature matrix is built before `close_margin` is read, and a
leakage check permutes the close within season and requires every feature to be
bit-identical.

Combined forecast:  `M_hat = O + gamma*head_M + eta*head_F`, gamma and eta
fitted walk-forward on prior seasons.

## FEATURES (opener-time, all already in the frame)

edge = m_blind - open · |open| · rest differential · expected absence
differential · total · days into season · trailing market log loss

## PREDICTIONS

- T1  head M achieves positive out-of-sample R^2 against the no-movement
      baseline. D147 measured 17.1% with a richer live feature set; with these
      seven features I predict LESS, 3-10%.
- T2  head F is WEAKER than head M. Beating the close is harder than
      anticipating it.
- T3  The combined forecast beats the shipped offset on full-stack log loss by
      a point estimate < 0.001 nats, and the CI probably contains zero.
- T4  head M's contribution SURVIVES controlling for head F, and vice versa —
      if either collapses, the two heads are the same signal twice.
- T5  Signed CLV improves. This is the endpoint the movement head should move
      most, and by D238's measurement CLV needs ~131-267 bets to resolve.

## GATE

Season-clustered, K=7, CI at K-1 dof. Primary endpoint full-stack log loss;
secondary signed CLV; diagnostic close-movement RMSE. Nothing ships on
historical data — a favourable result freezes the construction for 2026-27.
