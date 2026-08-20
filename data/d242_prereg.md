# D242 PRE-REGISTRATION — JOINT MINUTE-CONSERVING COMPOSITION

Hashed before any challenger score existed.

## HYPOTHESIS

D240 showed that a broad participation probability applied as an INDEPENDENT
per-player discount is harmful (+0.002982 full-stack, worse 6/6). The proposed
explanation is that NBA teams allocate ~240 regulation minutes regardless of who
is absent, so a routine DNP REALLOCATES minutes rather than destroying them,
and the current construction

    S = SUM_i P(play_i) * E[M_i | plays] * v_i / 48

lets minutes vanish. **The hypothesis under test: broad participation helps only
when minutes are conserved and reallocated.**

D240 does NOT refute a joint expected-minutes model. It refutes one particular
independent-discount construction.

## THE SCALE PROBLEM, STATED UP FRONT

Under any 240-conserving allocation, `SUM_i m_i / 48 = 5` EXACTLY. So team
strength stops being a sum of contributions and becomes 5 x (minutes-weighted
mean talent). Its level and dispersion change completely, which would make the
fixed 0.5 blend weight in the production margin wrong for reasons that have
nothing to do with the hypothesis.

**Therefore the primary comparison is run at the COMPOSITION-MARGIN level with a
per-fold AFFINE RECALIBRATION** (`a + b*cm`, a and b fitted on training seasons
only). That isolates the SHAPE change — mass conservation — from a pure scale
artefact. Only an arm that wins there is promoted to a full-stack run, where the
blend and link are refitted per fold as well.

## ARMS (six; the whole family, declared now)

    CONTROL   current composition leg, narrow out-set, no allocation
    A         broad participation x conditional minutes, INDEPENDENT weights
              -- reproduces the D240 loser, included so the reproduction is
              visible rather than assumed
    B         narrow out-set + joint 240 allocation
    C         broad participation + joint 240 allocation        <- PRIMARY
    PLACEBO   arm C with participation probabilities shuffled within
              (team, season) -- must be indistinguishable from CONTROL
    ORACLE    actual minutes played, NOT DEPLOYABLE, diagnostic ceiling only

Arm B isolates whether ALLOCATION alone helps. C minus B is the incremental
value of the broader participation model. That decomposition is the point of the
design: D240 could not separate them.

## FOUR ALLOCATION CONSTRUCTIONS (declared now; all four reported, no picking)

Let u_i = P(play_i) * E[M_i | plays].

    N1 proportional   m_i = 240 * u_i / SUM_j u_j
    N2 capped         N1 then iterative water-fill with a 40-minute individual
                      cap, so displaced minutes cannot pile onto one player
    N3 role-tiered    top-5 by u_i receive 160 minutes pro rata, the remainder
                      80 pro rata -- a starter/bench capacity proxy, since
                      `nba_players` carries NO position column
    N4 shrunk         m_i = 0.5*N1_i + 0.5*u_i -- partial conservation, the
                      honest middle between the incumbent and full allocation

All four are scored for every arm. **Reporting all four is what stops this from
being a four-way search dressed as one test**; the primary is N1, the simplest,
declared here.

## PREDICTIONS

- T1  ORACLE beats CONTROL substantially. If perfect minutes do not help, the
      whole minutes pathway is worthless and T2-T5 are uninterpretable.
- T2  Arm A reproduces D240's harm at the composition-margin level.
- T3  Arm B (allocation alone, narrow labels) is roughly NEUTRAL vs CONTROL.
      Conserving minutes should not by itself add information.
- T4  **THE HYPOTHESIS: arm C beats arm A decisively.** If broad participation
      is only harmful because minutes vanish, restoring them should recover it.
- T5  Arm C vs CONTROL is the shipping question and I predict it is NS or
      marginal. Reason stated in advance: the incumbent's narrow label already
      approximates exogenous absence, and allocation mostly re-adds minutes the
      incumbent never removed.
- T6  PLACEBO is indistinguishable from CONTROL. If it is not, the harness
      leaks and every other row is void.

## GATE

Stage 1 (composition margin, all arms x all four constructions): season-clustered
mean delta in RMSE against actual margin, each arm affine-recalibrated per fold.
Stage 2 (full stack, primary arm only if stage 1 favourable): season-clustered
log-loss delta, 95% CI at K-1 dof excluding zero, better in >= 5/7, calibration
veto, MDE80 from a within-season permutation stated before the endpoint.

## STOP CLAUSE

`nbapred/model/composition.py` default path stays byte-identical (env-gated,
`COMP_ALLOC` unset = current behaviour, asserted by control run). On stage-1
failure no stage-2 run happens and nothing ships. Historical data is used to
CONSTRUCT and DEBUG this architecture; because the architecture was suggested by
inspecting D240's failure, a favourable historical result is DEVELOPMENT
evidence and 2026-27 is the clean confirmation.
