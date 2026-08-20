# D240 PRE-REGISTRATION — PARTICIPATION-V2: REALIGN THE TARGET

Hashed before any challenger score existed.

## THE DEFECT UNDER TEST

`d200_participation.py:124` defines the incumbent's label as

    y_out = (status_today == "Out")

i.e. it predicts whether the NBA's 5PM report will carry an administrative
"Out" label — not whether the player contributes tonight. The composition leg
spends `1 - p_out` as an expected-participation weight, so the model is being
asked a question its label never answered. Non-appearances with no same-day Out
label (G-League assignments, coach DNPs, late scratches) are invisible to it,
and the 28,881 G-League reason rows suggest that blind spot is large.

## THE CHALLENGER

Same walk-forward discipline, new LABEL and new features, all as-of-open:

    label     player did NOT appear tonight (seconds == 0 / absent)
    universe  rotation candidates: appeared for the team within 21 days with
              trailing minutes >= 8, UNION report-listed players that day
    features  last status before game day (one-hot), status transition
              direction, consecutive Out reports, consecutive Q/D reports,
              days since last report, days since last appearance, played the
              team's prior game, trailing minutes, REASON CATEGORY (G-League /
              protocols / rest-personal / lower-body injury / other injury /
              illness / surgery / none)

Ridge logistic, fitted on seasons before the scored one, 2019-20..2025-26.
Incumbent comparator: the shipped `p_out` artifact, with p=0 for players it
does not list — that IS its implicit prediction for them.

## PREDICTIONS

- T1  Label misalignment is MATERIAL: >= 25% of true non-appearances by
      rotation-candidate players carry no same-day administrative Out label.
- T2  v2 beats the incumbent on the participation label (log loss + Brier),
      season-clustered CI excluding zero, >= 6/7 seasons. Stated openly: part
      of this is the label realignment itself, since the incumbent is scored
      against a target it was never trained for — that asymmetry is the point
      of the audit, not a flaw in it.
- T3  The dominant new coefficients are the G-League reason and days since
      last appearance — roster-state, not injury state.
- T4  DOWNSTREAM (the gate that matters): swapping the composition leg's
      artifact for v2 improves full-stack production log loss by an amount
      whose season-clustered CI may include zero. Prediction: direction
      negative (better), magnitude near D202's -0.0023 or below.

## GATE FOR THE DOWNSTREAM SWAP (house standard)

Season-clustered mean delta vs the incumbent artifact through the FULL
`prod_by_season` stack, 95% CI at K-1 dof excluding zero, better in >= 5/7,
calibration veto, MDE80 from a within-season permutation of the artifact
stated before the endpoint. STOP CLAUSE: on any failure `data/p_out.csv.gz`
and production are untouched; v2 ships only as `data/p_out_v2.csv.gz` for the
2026-27 shadow.
