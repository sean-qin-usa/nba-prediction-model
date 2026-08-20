# D245 PRE-REGISTRATION — ROLE-SHARE MINUTE ALLOCATION

Hashed before any challenger score existed. Separate from D242; `COMP_ALLOC`
is left untouched and the D242 code path is not reused for the primary arm.

## HYPOTHESIS

A team's top-five share of expected minutes is stable enough to estimate from
prior games, and imposing that walk-forward share improves (a) player-minute
allocation, (b) talent-weighted lineup strength, and (c) full-stack
opening-line forecasts.

D242 found that a hand-chosen 160/80 tiering (N3) improved composition-margin
RMSE under every availability regime. That construction is INELIGIBLE here: it
was selected from an eight-cell sweep on these seasons and confounds tier
shares, a rank-5/6 discontinuity, within-tier normalisation, extreme-weight
shrinkage and roster-size sensitivity. D245 replaces the hand-chosen pair with
ONE parameter estimated from realised minute shares.

## THE SINGLE LEARNED PARAMETER

    alpha_f = mean over TRAINING games of
              ( sum of actual minutes for the top-5 by u_i )
              / ( total actual team minutes in that game )

with u_i = (1 - p_out_i) * trailing_minutes_i, ranked using information
available at the opener cutoff. Actual minutes are normalised by the team-game
total so overtime does not distort the share.

**alpha is estimated from MINUTE ALLOCATION ONLY. It is never fitted against
margin, RMSE, or log loss.** Explicitly forbidden: tuning the number of tiers,
the rank cutoff, any shrinkage strength, or using evaluation-fold shares.

## ALLOCATION

    M_hat_i = 240 * alpha_f * u_i / sum_{top5} u_j        i in top 5
    M_hat_i = 240 * (1-alpha_f) * u_i / sum_{rest} u_j    otherwise

Deterministic fallbacks, each COUNTED and emitted in the artifact: fewer than
six candidates; zero within-tier weight; missing availability; predicted
minutes above the physical 48; insufficient training games.

## CANDIDATE COVERAGE — A LOAD-BEARING DIAGNOSTIC

    coverage = (actual minutes belonging to candidate players)
             / (total actual team minutes)

Non-candidate minutes go in an explicit OTHER bucket and are NEVER silently
discarded. If coverage is materially below 1, then any gain from rank-tiering
may be roster REGULARISATION rather than role allocation, and that alternative
must be reported alongside the result.

## ARMS (five; only ONE can advance)

    CONTROL      opener-time soft availability x raw trailing minutes
    PRIMARY      same information, training-estimated alpha_f allocation
    FIXED_160_80 D242's discovery specification — DIAGNOSTIC, PERMANENTLY
                 INELIGIBLE TO SHIP
    ORACLE_MIN   actual minutes, diagnostic ceiling, not deployable
    PLACEBO      ranks shuffled within team-game, weights preserved

## FOUR ORDERED GATES

  L1 minute allocation   total-variation distance between predicted and actual
                         minute SHARES; player-minute MAE; error by rank band
                         1-5 / 6-8 / 9+; calibration of predicted vs actual
                         top-5 share; count of >48-minute predictions
  L2 lineup strength     against S*_g = sum_i v_i * M_actual_i / 48, which is
                         the TRUE minutes target D242's "oracle" never was
  L3 composition margin  home-minus-away vs realised margin, training-only
                         affine recalibration
  L4 full stack          blend, offset, link and calibration all refitted
                         within each training fold; opening-market log loss

An arm advances only by showing the mechanism at L1-L2, not by L3 alone.

## PREDICTIONS

- P1  alpha_f lands near 2/3 (i.e. 160/240). If it does, that independently
      corroborates the construction D242 stumbled on; if it lands far away,
      N3's benefit was NOT the tier share and is more likely shrinkage or
      roster regularisation.
- P2  alpha_f is stable across folds (spread < 0.05).
- P3  PRIMARY improves L1 minute-share TV distance vs CONTROL.
- P4  PRIMARY improves L2 strength error vs CONTROL.
- P5  FIXED_160_80 >= PRIMARY at L3, because it was selected on this data —
      and that gap is a DIRECT ESTIMATE OF THE SELECTION BIAS in D242.
- P6  Candidate coverage < 1.00, so part of D242's N3 gain is regularisation.

## STOP CLAUSE

Historical data cannot ship this: the rank-tier architecture was learned on
these seasons. A favourable result freezes the construction for the 2026-27
shadow. `nbapred/` production defaults are not touched by this entry.
