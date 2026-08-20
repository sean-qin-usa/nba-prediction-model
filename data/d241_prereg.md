# D241 PRE-REGISTRATION — JOINT-MARKET DISTRIBUTION (TOTAL, THEN MONEYLINE)

Hashed before any challenger log loss existed.

## THE GAP

Production converts a margin to a probability with a FIXED scale:

    P(home win) = sigmoid(margin / 7.2)

The offset layer reads the spread but never the TOTAL or the MONEYLINE. Yet a
240-total game and a 205-total game do not have the same margin variance —
more possessions, more variance — so the same corrected margin should imply a
probability nearer 0.5 in the high-total game. D198's variance model used
availability, early-season, rest and favourite size, and was a precise null;
**it never used the market total**, which is the one variance proxy priced by
someone with more information than we have.

Reviewer's framing, adopted: the total should affect the SCALE, not the
expected margin. This may improve log loss while never changing a side.

## ARMS

Incumbent      p = sigmoid(m_offset / s),        s scalar, walk-forward
A  linear      s_i = s0 + s1*(total_i - tbar)
B  power       s_i = s0 * (total_i / tbar)^gamma
C  joint       arm A plus the de-vigged opening moneyline as a second scale
               term, ON THE SUBSET WHERE ML EXISTS

All parameters fitted on seasons 1..k, applied to k+1 unchanged. Tested at the
FULL STACK — m_offset, the margin production actually emits — because D235
established that gating an intermediate layer overstates the shipped effect.

## COVERAGE, MEASURED BEFORE FITTING

Opening total: 93.5% of 2019-26 (7,704 / 8,239); the only real gap is 2022-23
opens (661/1194). Rows with a missing or non-positive total fall back to the
incumbent scale, so the arm is defined everywhere.

**Moneyline: present for season_end 2019-2022 at 100%, 50% in 2023, and ZERO
from 2024 onward** in both `odds_market` and `odds_hist_sbr`. Arm C is
therefore restricted to 2019-20..2022-23 and can be scored on at most three
walk-forward seasons. It is reported as a bounded side-result, NOT as a
shippable arm, and its failure or success cannot promote anything.

## PREDICTIONS

- T1  s1 > 0 in arm A: higher total implies a wider scale.
- T2  Arm A improves full-stack log loss with a season-clustered CI excluding
      zero. Magnitude prediction: small, ~0.0005-0.0015 nats, because the
      offset already compresses margins toward the opener.
- T3  Arm B ~ arm A. If a power law materially beats a line, the effect is
      being driven by the tails and should be distrusted.
- T4  The side changes on < 1% of games. A scale term must not reprice
      directions; if it does, it is not doing what it claims.
- T5  Arm C adds nothing beyond arm A. The moneyline and spread are two views
      of the same first moment; any real gain would be a devig artefact.

## GATE

Season-clustered mean delta in full-stack log loss vs the incumbent, 95% CI at
K-1 dof excluding zero, better in >= 5/7 seasons, calibration veto, MDE80 from
a within-season permutation of the total stated before the endpoint is read.

STOP CLAUSE: on failure `nbapred/model/production.py` and the offset layer are
untouched. Arm C never ships regardless of outcome.
