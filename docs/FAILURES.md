# WHAT WE TRIED THAT DIDN'T WORK, AND WHY

The register (`DECISIONS.md`, `DECISIONS_ARCHIVE.md`) records everything in
order. This document is the failures pulled out and organised by *reason for
failure*, because that is the useful axis: the same trap catches different
ideas, and the point of writing them down is that the next idea gets checked
against the trap rather than rediscovering it.

## EVIDENCE STATUS IS SEPARATE FROM SHIP VERDICT

"NO SHIP" is overloaded. Some experiments refute a hypothesis; others test the
wrong construction, lack the power to resolve the effect they target, or
identify a real mechanism without establishing incremental predictive value.
**Those imply different next actions**, so every entry carries an explicit
status:

| status | meaning | next action |
|---|---|---|
| `VALID_REFUTATION` | correct construction, adequate power, hypothesis is dead | closed |
| `UNDERPOWERED` | the design cannot resolve an effect of the plausible size | change the endpoint or stop |
| `CONSTRUCTION_INVALID` | the thing built was not the thing hypothesised | rebuild, hypothesis alive |
| `INTEGRATION_INVALID` | gated at the wrong layer of the stack | re-gate at the emitted layer |
| `DATA_BLOCKED` | the measurement does not exist | acquisition, not modelling |
| `FRESH_CONFIRMATION_PENDING` | favourable, but the holdout is spent | prospective only |
| `ACCEPTED_PROSPECTIVELY` | gated and shipped, awaiting out-of-sample | shadow |
| `SUPERSEDED` | replaced by a later, better test | historical |

**Where a mechanism is claimed it is labelled CONFIRMED or HYPOTHESIS.** Several
entries record that the *first* explanation offered was wrong, because a wrong
mechanism attached to a correct null is how a research programme walks into the
same wall twice — and two such cases below (D240, D241) were caught only by
outside review, after being written into the register as if established.

**Fields recorded per entry:** estimand · information timestamp · train and
evaluation populations · primary endpoint · MDE80 and whether a minimum
practically-relevant effect was declared · whether downstream layers were
refitted · whether the holdout is spent · construction validity · retest
eligibility · and the *narrowest* causal conclusion the result supports.

---

## A. THE DOMINANT PATTERN: re-describing information the model already has

Six independent attempts, all null. If a proposal amounts to a new *basis* for
`m_blind − m_open`, this section already answers it.

### A1. PCA / low-rank compression of the rejected feature pile (D157)

**Status: `VALID_REFUTATION`.** Estimand: log-loss cost of carrying the
rejected block under a rotated basis. Endpoint: season-clustered pooled log
loss. MDE80 0.00452 declared before scoring. Holdout: spent. Retest: closed —
rotation cannot create information a full-rank design does not carry.

**Tried.** Principal-component and supervised-ordering compression of the 15
schedule/motivation/travel columns that had been rejected individually, plus the
30 team-home dummies, plus the joint 43. Three pre-registered arms.

**Predicted.** That compression would buy back parameter noise.

**Result.** All three NO-SHIP. PCA at the pre-registered 90%-variance rule moved
the carry-all cost from −0.00571 to −0.00552: paired **+0.00019, CI (−0.00009,
+0.00051), ns**.

**Why — CONFIRMED.** The pile is not low-rank. The top component holds 13.6%;
10/12/14 of 15 components are needed for 80/90/95% of variance; effective rank
13.46 of 15. The team-home dummies are *exactly* incompressible (29 eigenvalues
of 1.035 and one zero) because one-hot dummies are orthogonal by construction.

**The transferable lesson.** The lever is the **penalty, not the basis**. The
same 15 terms under plain ridge at matched effective df cost −0.00120 against
PCA's −0.00335 — 2.8× cheaper — and are free (−0.00009) at edf 2.19. But
*nothing is ever positive on either family*: "keep everything" can be made free
only by making it inert (rms 0.458 vs 2.170 raw, i.e. 79% switched off).

### A2. Channel-wise trust in the offset layer (D230)

**Status: `VALID_REFUTATION` for the schedule channel; `UNDERPOWERED` for the
ff/composition pair** — those two correlate 0.800 and are not separately
identifiable at this n, so their split is unmeasurable rather than measured
and null. Endpoint: season-clustered log loss, MDE80 0.00104 (permutation).
Downstream refitted: yes (lam per inner split). Holdout: spent.

**Tried.** Letting each of the four live margin channels earn its own trust
coefficient instead of one shared 0.356, nested so λ→∞ reproduces the shipped
model exactly.

**Predicted (in advance, and this is why it was worth running).** NO-SHIP, with
`d_sched < 0` and `d_comp > 0`.

**Result.** Mean delta −0.000093, CI [−0.000346, +0.000161], 4/6 seasons.
NO-SHIP as predicted. `d_sched` = −0.0285 CONFIRMED.

**Why — CONFIRMED for the schedule channel, NOT for the rest.** `m_sched` is
orthogonal to the other channels (r = −0.015, −0.001), so its coefficient is
cleanly estimated: **the opener genuinely does price rest, back-to-backs and
home edge, and gives them less than the common trust.** But `m_ff` and `m_comp`
correlate 0.800 and their deviations came out near-equal and opposite
(−0.1160 / +0.1154) — a ridge trading collinear columns, not two measured
trusts. "Availability earns +0.115" is **not** supported.

### A3-A6. Regime gates: buckets, continuous interactions, PCA states (D237, D237b, D238, D239)

**Status: `UNDERPOWERED`, not `VALID_REFUTATION`.** This distinction is the
whole point: MDE80 on a regime gap is 2.2x the pooled edge itself, so the
design cannot resolve any plausible effect. The nulls are real but they do
not refute the hypothesis — they show the corpus cannot address it. Retest
eligibility: **only on a lower-variance endpoint (CLV) or fresh seasons.**

**Tried, in four independent forms.** (a) quartile buckets on trailing market
log loss, outcome unpredictability, days-into-season, scored on 888 bets;
(b) the same on all 8,239 games using edge-over-opener; (c) six bet-time
conditioners under one family null; (d) continuous `|d| × z` interactions with
the `|d|` main effect as control, on three states including a roster-transition
factor built as the *mechanism* behind the early-season hint.

**Result.** **0/3 bucket tests, 0/3 continuous interactions.** The bet-level
search null is decisive: observed best spread 16.74 ROI points against a
permutation null whose **median is 17.28** — the structure is smaller than what
searching three candidates buys from noise (p = 0.553). PCA on eight
strictly-prior state variables (lead-in rotation, PC1–3 = 91% of variance,
cleanly interpretable) — all three ns.

**Why — CONFIRMED.** MDE80 on a regime gap is **0.00571 nats, 2.2× the entire
pooled edge of +0.00259**. Seven seasons cannot resolve a regime effect smaller
than twice the effect it would modulate. The honest statement is not "regimes do
not exist" but **"this corpus cannot answer it, and re-slicing will not help."**

**The trap this section exists to name.** A regime rule *is* a subset selector,
and the register's manufacturing-capacity result is that best-of-N random
subsets of this same book buy **+2.54 ROI points from nothing**. Searching
subset selectors against realised returns on the seasons that built the strategy
is the most dangerous single activity available here.

---

## B. FAILURES OF TARGET DEFINITION — the model answered the wrong question

### B1. Participation-v2: realigning the availability label (D240)

**Status: `CONSTRUCTION_INVALID`.** The experiment tested a broad
participation probability applied as an INDEPENDENT per-player discount. It
does NOT refute a joint expected-minutes model, and D242 is the rebuild.
Estimand: full-stack log loss under an artifact swap. Information timestamp:
last report strictly before game day. Population: 209,393 rotation-candidate
player-games. Downstream refitted: NO — the blend and link were held fixed,
which is a second reason to treat the verdict as construction-limited.
Holdout: spent. **Narrowest supported conclusion: a standalone non-appearance
probability, applied without modelling minute substitution, is harmful.**

**Tried.** `d200_participation.py` predicts whether the NBA will publish an
administrative "Out"; the composition leg spends `1 − p_out` as an *expected
participation* weight. Those are different questions. So: relabel to actual
non-appearance over 209,393 rotation-candidate player-games, add parsed reason
category, status-transition direction, days since last appearance, and swap the
artifact through the full production stack.

**Predicted.** T1: ≥25% of non-appearances unlabelled. T4: downstream direction
negative (better), magnitude near D202's −0.0023.

**Result.** **T1 confirmed and far exceeded — 76.3%** not-yet-Out at the
pre-game-day cutoff (72.6% minutes-weighted; still 64.9% among 30+ minute
players). The shipped artifact predicts mean 0.1118 against an actual absence
rate of 0.2435 — it under-predicts by more than 2×.
**T4 REFUTED.** Full-stack: **+0.002982, CI [+0.000642, +0.005322], worse in
6/6 genuine folds.** Significantly worse, every scored season.

**Why — NOT ESTABLISHED, and the first explanation was wrong.** I initially
wrote that the leg *double-discounts*, because trailing minutes supposedly
already absorbs routine non-participation. **That is mathematically incorrect.**
`trail_min` is an EWMA over games *played*, so it estimates `E[M | plays]`, and
multiplying by `P(plays)` is exactly how one gets `E[M]`. The terms compose
correctly.

A proposed replacement — that v2 leaks team minute mass by violating
`Σ M_i = 240` — is conceptually sound but **is not what the measurement shows**:

    raw sum of trailing minutes       295.4 per team-game
    incumbent  Σ (1−p_out)·trail_min  259.8   (+19.8 OVER 240)
    v2         Σ (1−p_v2)·trail_min   232.7   (−7.3 UNDER 240)

v2 lands *closer* to the constraint. Whatever the incumbent gains, it is not
mass conservation.

**Surviving HYPOTHESIS (not a finding).** Non-appearance mixes two states:
*exogenous* absence, which genuinely removes strength, and *endogenous* rotation
choice, where a coach's DNP **reallocates** minutes to whoever plays instead.
The leg has no reallocation step, so it treats both as independent losses. The
narrow administrative label may work better because it approximates the
exogenous state only.

**Successor.** A joint team-level minutes allocator that conserves the
constraint by construction — `u_i = P_i(play)·E[M_i|play]`, `M̂_i = 240·u_i/Σu_j`,
with role/position capacity — scored **first on player-minute allocation and
props**, not on sides. Because that architecture was suggested by inspecting
this failure, historical results on it are development evidence; 2026-27 is the
clean confirmation.

---

## C. FAILURES OF ESTIMATION — the mechanism is real, the estimator is not

### C1. Joint-market distribution: total and moneyline into the scale (D241)

**Status: `UNDERPOWERED` / inconclusive — NOT `VALID_REFUTATION`.** The
mechanism is supported (s1 > 0 in 6/6) but incremental value is unproven.
Endpoint: full-stack log loss. MDE80 0.00086 (permutation, declared first).
No minimum practically-relevant effect was declared, which in hindsight is a
gap — 'statistically nonzero' was never going to be the right bar for a
variance term. Holdout: spent, so a shrinkage grid on these folds would be a
second-look optimisation. Retest: ONE frozen regularised specification,
prospective only.

**Tried.** The offset reads the spread but never the total. A 240-total game has
more margin variance than a 205-total game, so the same corrected margin should
imply a probability nearer 0.5. Three arms: linear scale in the total, power
scale, and a joint arm adding the de-vigged moneyline.

**Predicted.** T1 `s1 > 0`; T2 ships; T3 B≈A; T4 side flips < 1%; T5 ML adds
nothing.

**Result. Four of five confirmed — the one that failed is "it ships."**

    arm A linear   +0.001483  CI [−0.003649, +0.006614]  better 2/6
    arm B power    +0.000850  CI [−0.002614, +0.004313]  better 2/6
    arm C + ML     +0.00198   CI [−0.01284, +0.01680]    K=3 only
    MDE80 (permutation, stated first) 0.00086

T1 confirmed — **s1 > 0 in all six folds**. T4 confirmed exactly — **0.00% side
flips**, the scale term never repriced a direction. T5 confirmed: the ML
coefficient swings −1.4 / −6.7 / −9.9 across its three folds.

**Why — the mechanism is directionally supported; its incremental forecasting
value is UNPROVEN.** Fitted `s1` decays 0.236 → 0.036 as training grows and thin
early folds over-extrapolate so badly that 2021-22 costs **+0.0109 nats alone**.
I first wrote "the failure is overfitting, not absence of signal" and **that
overclaims**: better-2/6 with a CI straddling zero is equally consistent with
genuine coefficient decay, era variation, an effect too small to move log loss,
or the offset having already absorbed it.

**What was deliberately NOT done.** A shrunk variant is the obvious fix and is
not in the prereg; adding an arm after seeing A fail is the second-look trap. It
is frozen as a low-priority *prospective* candidate. Note its first-order effect
is to pull the challenger toward the incumbent — turning "harmful" into
"approximately zero" is not finding value.

### C2. Per-term empirical-Bayes weighting of the rejected pile (D157 arm C)

**Status: `VALID_REFUTATION`** — pre-registered, wrong-sided, mechanism
identified and confirmed.

**Predicted** −0.0025 to 0.000. **Measured −0.00634 — refuted, wrong side.**

**Why — CONFIRMED, with a clean mechanism.** The EB rule `w = t²/(1+t²)` exceeds
the layer's global 0.798 whenever `|t| > 2.02`, and the expensive terms are
exactly the high-in-window-|t| ones. So the rule **shrinks the harmless columns
and amplifies the harmful ones**. The transferable lesson: *in-window evidence
is not out-of-sample stability.*

### B2. Role-transition minutes correction (D144)

**Status: `CONSTRUCTION_INVALID`. One of the best retest candidates.**

**Tried.** Adjust projected minutes when a player moves into or out of a
starting role, using the GameRotation starter flag.

**Result.** Minutes improved in BOTH directions; points improved for demoted
players and WORSENED for promoted ones. The pre-registered PIT veto failed and
the arm was correctly rejected.

**Why — CONFIRMED, and it is an incomplete intervention rather than a dead
hypothesis.** The model changed minutes but left per-minute RATE unchanged,
i.e. it assumed bench production transfers intact to starter minutes. The
register records the resulting split directly: `PROMOTED -2.2078 / DEMOTED
+2.3497`, and `ARM R | PROMOTED — PASS-WEAK and on the HARM side` against
`DEMOTED — ERA-CONDITIONAL`. **The pooled veto was misaligned with a
heterogeneous treatment whose halves have opposite sign and partially cancel.**

**Narrowest supported conclusion.** Correcting minutes alone, for a transition
that also changes usage and efficiency, is harmful for promotions. The full
pathway `role -> minutes AND rate -> production` is untested.

**Retest eligibility.** Alive, with gates stratified by transition direction —
minutes CRPS must improve in both strata and points CRPS must not worsen in
either. A demotion-only rule cannot be justified from the old result; that
would be post-selection and must be declared before fresh outcomes.

### B3. Props channel ramp calibration (D151)

**Status: `CONSTRUCTION_INVALID` for rebounds — estimand mismatch.**

**Tried.** A channel-specific minutes-ramp correction, with the coefficient
fitted against an ANALYTIC expectation (rate x projected minutes).

**Why — CONFIRMED.** The scored system is not the analytic model: it samples
minutes from an empirical histogram, clips, and simulates per channel. The
register measures the generator directly — **simulated mean rebounds 4.7760 vs
realized 4.7777, bias -0.0017** — i.e. the production generator was already
approximately unbiased, so a correction fitted to an analytic bias that did not
exist necessarily overshot.

**Narrowest supported conclusion.** *Rebounds are closed*: there was no
scored-generator bias to correct. The favourable assists result was found after
splitting channels and is `FRESH_CONFIRMATION_PENDING`.

**Retest eligibility.** Assists only, prospectively, with the coefficient fitted
against SIMULATED OUTPUT from the exact production generator under common
random numbers — never against an analytic moment. Points/rebounds/threes serve
as invariance controls.

---

### C3. Between-season adaptation: champion response and altitude decay (D247)

**Estimand.** For each defending champion (and separately, each beaten
finalist), the mean of `signed(margin_actual - close_margin)` from that team's
perspective across its following season. Against the CLOSE, deliberately: the
hypothesis is not "the champion declines" — that is well known and priced — but
"the champion declines and the market does not know."

**Endpoint frozen before reading:** season-clustered mean over k=19, plus MDE80
computed from the panel's own dispersion.

| arm | estimate | 95% CI | sign count | MDE80 |
|---|---|---|---|---|
| defending champion vs close | **-0.5321** | (-1.1882, +0.1241) | 13/19 negative | **0.875** |
| beaten finalist vs close | -0.4191 | (-1.0776, +0.2394) | 11/19 negative | — |

**Status: `UNDERPOWERED`, not `VALID_REFUTATION`.** Both arms point the way the
hypothesis predicts and neither resolves. MDE80 is **1.6x the point estimate**,
so a real league-adapts-to-the-champion effect of the size actually observed is
undetectable in this panel by construction. Nineteen champion-seasons is the
entire NBA history available in the frame; this arm cannot be rescued by more
careful analysis, only by ~40 more seasons that do not exist.

**Narrowest defensible conclusion:** the data neither support nor exclude a
champion-adaptation effect smaller than ~0.9 pts/game.

**The altitude limb of the same hypothesis IS refuted, and the sign is
inverted.** Denver's home advantage *rose* after the 2023 title — raw +2.951 ->
+5.694, market residual +0.336 -> +0.922, with a pooled 19-season home residual
of +0.459 CI(-0.512,+1.431) ns. The archive series that reads as a post-title
collapse (+8.67 -> -3.50) is the **strength-controlled** deviation `d_t`, which
is not the bettable quantity. Utah is the control that makes the point: its raw
home edge collapsed to **-10.49** in 2024-25 while its market residual was only
**-1.95**. A real decline that the market prices is worth nothing.

**Why the whole family fails regardless of any single arm.** Every limb requires
the market to be slow to absorb an offseason change. That mechanism has its own
signature — mispricing largest in October, decaying by January — and the
signature is absent:

| window | n | market abs error | open->close move |
|---|---|---|---|
| first 2 weeks | 1,932 | 9.907 | 1.233 |
| weeks 3-4 | 2,268 | 9.412 | 1.233 |
| month 2 | 4,036 | 9.711 | 1.241 |
| months 3-4 | 7,881 | 9.805 | 1.218 |
| months 5+ | 6,623 | 9.565 | 1.330 |

Flat. The market is no worse in the first two weeks of a season than in April,
which kills the transmission mechanism for (a), (b) and (c) at once —
independent of whether any individual arm reaches significance.

**Holdout spent:** none. Every arm ran on the historical frame with no
production change and no live data consumed.

---

## D. FAILURES OF MEASUREMENT — the number was an artifact of how it was computed

These are mine, not the ideas'. They are recorded because each was caught only
by a specific check, and the check is the reusable part.

| what looked true | what was actually happening | what caught it |
|---|---|---|
| "86.3% of absences carry no Out label" (D240 first pass) | The status join kept only `report_date < game_date` rows. The archive is **74.1% same-day**; only 891 of 4,752 game dates carry any advance row, so **81% of candidates got status 'none' by construction.** D199's defect reproduced verbatim. | An 80% unlabelled rate persisting into the 30+ minute band. Starters do not miss games unannounced at that rate — the number was too extreme to be behavioural. |
| "v2 beats the incumbent by 1.30 nats" (D240 Part 2) | The incumbent was being scored at p≈0 on players it never models. On the common report-listed universe its log loss is **0.681, not 1.66.** | Checking the comparison on a *common* universe rather than the challenger's own. |
| "the pipeline reproduces to 9.8e-15" (register-wide) | That was the **noise floor**, not a measurement. The pipeline differs from *itself* by 1.577e-14 across runs — more than the refactor under test changed it. `max\|dp\| = 0` was never achievable. | Running the same code twice, which no control had ever done. |
| "signal exists only in the tail" (D238) | Post-selection: D1–D8 were pooled *after* their results were seen and the breakpoint placed at ~1.2 by eye. The defensible claim is the continuous slope. | A reader. Now frozen as a prospective hinge at τ=1.25 rather than asserted. |
| "~45 bets to confirm CLV" (D238) | Assumed independence and took the observed +1.07 as truth. Weekly block bootstrap gives a **design effect of 2.91**; with winner's-curse shrinkage the honest range is **131–267 bets**. | Recomputing with block resampling instead of an iid formula. |
| "27 pass / 7 error for a cloner" (D229) | Came from a command piped through `tail -8` that discarded the failure list. True result: **35 failed / 141 passed.** | Reading the whole run instead of its tail. |
| "prior-season margin predicts the market residual at -0.16 pts/pt" (D247 T6) | Estimated at **team-season level**, collapsing 82 games into one point per team. `corr(prev_act, current mkt) = +0.705` produced a suppression pair (prev -0.1608, mkt +0.2175) with **no game-level counterpart: -0.00507, t -0.24 ns**. It implied a +-1.6 pt/game edge from two public numbers. | Trying to **spend it**. The implied rule — back last season's worse team — covers **49.06%** at the close vs 52.38% break-even. A coefficient that cannot be bet was not measuring a market. |
| "2024-25 CLV ranked 1/19, z +2.80 — the model read that season" (D247) | **CLV is mechanically proportional to how far the RECORDED line travels**, and `corr(season CLV, season mean\|close-open\|) = +0.917`. 2024-25 has the largest recorded movement of 19 seasons (1.907) *and* the 3rd-most **unchanged** lines (22.3%) *and* the 2nd-widest opener dispersion — a feed signature, not a market one. REVIEW.md already recorded the cause: books/game at open falls **7.74 -> 1.00**. | Normalising to **capture fraction** (CLV / mean\|close-open\|) and correcting for 2024-25 having been *selected* as the best season: p(max19) 0.094 -> 0.258, p across the 4 pre-declared measures 0.326 -> 0.697. **Nothing survives.** |
| "2024-25's +24.00% ROI needs a causal explanation" (D247, the premise itself) | +24.00% vs the 7-season pooled +9.11% is **+2.29 SE**; as the **max of 7 seasons, p 0.144**. On the full 1,230-game slate the normalised gap ranks **7/19 (z -0.50)** — the market was ordinary. The question presupposed a fact that selection correction removes. | Testing the premise before testing the hypothesis. The full slate is 6x the bet ledger and is not conditioned on the selector. |
| "market line movement has roughly doubled since 2008-09" (OPENING_LINES.md) | Two problems. The rise is a **step at 2014-15**, not a drift. And the 2024-25 endpoint is **grid-inflated**: that season's opens come from ESPN BET alone, which posts **100% half-point spreads**, so the line grid is spaced 1.0 instead of 0.5 and the smallest observable move mechanically doubles. | Measuring **granularity** rather than movement. A market can change how far lines move; it cannot change the rate of half-point use from 47% to 99% or delete key numbers from 9.7% to 0.3%. |
| "TeamRankings and our frame disagree by sd 5-6 pts" (D248 first pass, mine) | TeamRankings uses its own abbreviations — `BK/GS/NO/NY/PHO/SA`. `fav_team` decides the **sign** of the spread, so six unmapped codes silently **flipped 20% of games** rather than dropping them. Corrected bias/sd: -0.001/1.585 and -0.008/0.780. | The disagreement sd (≈5) was nearly the sd of the lines themselves (≈6.4), i.e. the two sources were almost uncorrelated. **Two feeds quoting the same game cannot be uncorrelated** — that is a join failure, never a data finding. Now an assertion: every joined row's `fav_team` must match one of the two sides. |
| "inactive players can be derived from who didn't appear in the box score" (D250) | `player_game_stats` holds only DRESSED players, so absence looks like the signal. But a **season-ending injury leaves no trace in a table of who played** — the player simply stops appearing, indistinguishable from having left the team. Precision reached 0.941 with a tenure rule; **recall stalled at 0.560** and the misses were exactly the long-term injuries a margin model most needs. | Validating against basketball-reference's published `Inactive:` block on 35 games across six seasons, instead of shipping the derivation on the strength of its plausibility. The failure list named its own cause. |
| "the bet split works because large wagers are informed" (D251) | Backing the side with more money% than tickets% covers **51.69%** — above 50%, below the 52.38% break-even. But the edge is **NOT monotone in the divergence** (quintile covers 49.4 / 54.2 / 54.2 / 47.6 / 54.0, rank corr +0.133). If informed money drove it, a bigger divergence would mean a bigger edge. | Pre-registering monotonicity as a separate test from the headline. T1 alone would have been reported as a weak positive; T2 shows the stated mechanism is not what produces it. |
| "use the D248 core book basket for the bet split too" (D251, mine) | The basket 15/68/69 was validated for **spread coverage**. For the split, books 68/69/71/75 report `money% == tickets%` on 69–97% of rows in the first two seasons — a placeholder, not a measurement. Only **book 15** is populated across all three. | Checking the population rate of the specific field per book per season, rather than inheriting a basket validated for a different column. |
| "median(money) − median(tickets)" (D251, mine) | That is not `median(money − tickets)`. Independent medians coincided, the nonzero-divergence rate collapsed from ~90% to 36.9%, and the surviving sample was **82% a single season**. | The per-season counts (108 / 124 / 1,092) were impossible for a field documented at ~90% coverage in all three. A sample that lopsided is an aggregation bug before it is a data fact. |
| "the shipped p_us is badly calibrated — a walk-forward logistic beats it in 17/18 seasons" (D252, mine) | I compared `p_us` against a logistic on `m_us`. **Those are different margins.** `p_us` is `sigmoid(m_us696/6.96)` exactly, which equals `sigmoid(m_us_blind/7.2)` on 75.2% of games — the BLIND model. The +0.00504 was the offset layer's own contribution, not a calibration defect. | Testing calibration with the margin held FIXED on both sides. Same margin, fixed-7.2 vs walk-forward: **+0.00010** on blind and **−0.00009** on offset — both null, reproducing D74. |
| "a ~14x two-stage attenuation before the forecast is made" (D245d) | The entry's own worked example is `0.04406 × 0.652 × 0.3413 = 0.00981`, a factor of **4.49**, and `1/(0.652 × 0.3413) = 4.49`. The multiplication is right to 1e-5; the summary label is 3× too large. **D246 chose its research direction citing the 14×.** | Recomputing the stated arithmetic instead of quoting the headline. A summary figure that does not follow from the numbers directly above it is the cheapest possible check and it went unrun for six entries. |
| "star-out games are chaotic, so everyone predicts them worse" (D252, my framing) | Backwards. Market log loss **falls** as absences rise — 0.6029 (none) → 0.5611 (3+). A missing star makes the game more lopsided and therefore MORE predictable. My script printed a hardcoded narrative line asserting the opposite of its own table. | Reading the table rather than the sentence above it. Prose asserting a direction the adjacent numbers contradict is a failure mode of narrated analysis, not of the analysis. |
| "lambda=3000 is a hard shrink; 0.3564 is what survives it" (D252, mine) | The penalty is nearly inert at the shipped value. Edge coefficient by lambda: **0.3608 (unpenalised)**, 0.3590 (1e3), 0.3554 (3e3), 0.3090 (3e4). The unregularised fit already says 36%. | Refitting at lambda=0. **The 36% is the regression's verdict on our disagreement, not the penalty's** — a claim about a regulariser is checkable by turning it off. |
| "an elastic net over every feature will find the conditions where we beat the market" (D253) | Twenty-four of twenty-six features sit **exactly on the permutation null** (-0.00185 vs null median -0.00204). Worse, carrying them costs four-fifths of the signal: the same target gives **+0.00837 on two features and +0.00170 on twenty-six**. | Ablating rather than reading coefficients. A regularised fit that is "significant" can be significant entirely through two columns while the other twenty-four actively dilute it. |
| "dropna on the feature matrix" (D253, mine) | Collapsed 19 seasons to **3** in one line, because the injury-report and component features do not exist before 2019-20 and 2023-24. The run reported OOS R^2 = nan on 0 folds. | The printed season count. Any assembly step that silently changes the corpus size needs the corpus printed after it, and features need explicit COVERAGE TIERS rather than a shared dropna. |
| "we beat the opener most on large disagreements, so the offset should spend convexly" (D254, mine) | Two different claims. "Where the layer adds most value" is not "where the coefficient should be larger" — under a correctly specified LINEAR model a large edge produces a large correction, which has more room to improve on the opener. All three convex arms came back ns and pointing the wrong way, and the fitted shape leans **concave** (clipped arm: inner slope +0.3958, outer +0.2051). | Building the arms instead of reasoning from the slice. A magnitude observation and a curvature claim look alike in prose and are separated only by fitting the curve. |
| "persistent team-pair residual means a stylistic matchup effect worth modelling" (D255) | It persists across seasons (r +0.0303, 14/18) and survives both the scheduling and conference-imbalance confounds. It is still worth **+0.025 points vs the close and +0.100 vs the open**, both ns with MDE80 **33x and 8x** the estimate. Persistence and tradability are different properties. | Going straight to the out-of-sample value test — does LAST season's pair residual predict THIS season's margin against the market — instead of parameterising an interaction first. The existence question and the money question have different answers. |
| "every signal in the sweep passes the alignment screen at exactly 2.000" (D262, mine) | Scaling a signal by the **in-sample** OLS slope of error-on-signal forces `cov == var` algebraically, so alignment = cov/(var/2) = 2.000 for every signal regardless of content. D260's criterion presumes the shift is determined INDEPENDENTLY of the error it is scored against. | Fourteen identical rows. **A screen that passes everything is measuring itself.** Fixed by fitting the slope walk-forward and measuring var/cov on held-out seasons; D261 was unaffected because its signal was already in margin units and never fitted. |
| "the replacement-quality signal shows no effect" (D261 first pass) | It showed **var(signal) = 0.00000** — the signal was identically zero, because the rotation table was built from games players actually PLAYED, so an absent player had no row and every player looked present. A degenerate signal and a genuine null produce the same endpoint. | Printing `var(signal)` next to `cov(signal, error)`. **A signal's own variance is the cheapest possible sanity check** and it distinguishes "no effect" from "no signal". |
| "MIX MATTERS — it beats the permutation null at p=0.017" (D259, mine) | The MIX-only out-of-sample R^2 is **-0.00007**, i.e. worse than predicting the mean. The null was merely *more* negative (-0.00035), so p=0.017 establishes only that mix is less bad than shuffled mix. The verdict line tested the null comparison and never checked the sign. | Reading the R^2 before the p-value. **Beating a permutation null is necessary, not sufficient.** A verdict string that keys on significance alone will call a model that cannot reach the mean a finding. |
| "the team-level residual persists at r=0.55 even under OLS, so it is not shrinkage" (D256, mine, caught before it was written down) | Under OLS the row means are **5.3e-14** — numerically zero, as orthogonality to the team dummies requires. The r=0.55 was correlating float noise. A correlation is scale-free and will happily report structure in values of order 1e-14. | One line of `sd()`. **Never report a correlation without the scale of what is being correlated.** At ridge 25 the same quantity has sd 0.740 ortg points; at ridge 0 it has none. |
| "pair residuals persist across seasons, so there is a matchup effect" (D255) | A pair mean is `grand + row_i + col_j + interaction_ij`, and every pair (i,·) carries team i's own persistent residual. Double-centring the matrix drops the persistence from **+0.0303 SIG to +0.0170 ns** — 44% of the headline was team-level, not pair-level. | The two-way decomposition. Any "pair effect" measured on uncentred pair means is contaminated by both teams' main effects, and those persist for ordinary reasons. |

**The reusable checks, in order of how often they have paid.**
1. Run the same code twice before attributing a difference to a change.
2. Score a challenger and an incumbent on a *common* universe.
3. Compare an observed split against a permutation null **over the whole family
   searched**, not against zero.
4. State MDE80 before reading the endpoint, from the null's own dispersion.
5. Never read a result through `tail`.
6. Pre-register the *quantity*, not just the hypothesis — D232's `len(dict)` vs
   `sum(dict)` bug was invisible to unit tests (both are valid Python) and was
   caught only because the prereg named a quantity the code did not compute.
7. **Try to spend a coefficient before believing it.** D247's -0.16 survived
   every statistical check available — t -9.46, VIF 2.0, Type M 1.00x — and died
   the moment it was written as a bet. An effect that cannot be converted into a
   position was measuring the aggregation, not the market.
8. **Estimate at the level the decision is made at.** The same contrast was
   -0.1608 per team-season and -0.00507 per game. Bets are placed per game.
9. **Normalise a market metric by its own scale before comparing across
   periods.** CLV, capture, and hit rate all move with the recorded line's
   travel distance, which is a property of the odds feed. Only capture fraction
   is comparable across feeds.
10. **Fingerprint the feed, not just the market.** Granularity, key-number rate,
    and distinct-value count are recording conventions. A changepoint in one of
    them dates a data change precisely, and no market story can explain it.
11. **A cross-source comparison whose disagreement sd approaches the sd of the
    quantity itself is a broken join.** Two books pricing the same game are
    ~0.97 correlated; anything far below that is a key or sign problem.
12. **Test reachability before designing around a source.** Of eight candidate
    endpoints, only basketball-reference and Kaggle's landing page returned 200;
    stats.nba.com timed out and cdn.nba.com / site.api.espn.com were blocked at
    the egress proxy with an identical body. A plan built on an unreachable
    source is not a plan.
13. **Validate a derived field against published truth before shipping it**,
    and read the error list — D250's misses named their own cause in one line.
14. **A book basket is validated per FIELD, not once per source.** Coverage of
    the spread says nothing about coverage of the bet split.
15. **Check which column a metric actually reads.** `p_us` has been available
    for the whole project and is the BLIND model's probability, not the shipped
    one. A column name is not a contract.
16. **Recompute a summary figure from the numbers it summarises.** D245d's
    "14x" did not follow from its own worked example and was cited for six
    entries.
17. **Ablate before believing a regularised fit.** D253's significant model was
    significant through 2 of 26 columns, and the other 24 cost it 80% of its
    out-of-sample R^2. Uninformative features are not neutral.
18. **State a feature's coverage era before pooling it.** Zero-filling a count
    outside the era that records it teaches the model an era, not a fact.
19. **"Where an effect is largest" is not "where the coefficient should be
    larger."** The first is a magnitude observation, the second a claim about
    functional form. Only fitting the alternative shape separates them.
20. **Test that an effect REPLICATES before parameterising it.** D255 asked
    whether pair residuals recur across seasons before fitting any interaction;
    870 pair effects on 2.75 meetings each would have fitted noise perfectly.
21. **Persistent is not tradable.** A significant across-season correlation and
    a zero out-of-sample slope against the market are entirely compatible.
22. **Never quote a correlation without the scale of its inputs.** r is
    scale-free and reports structure in 1e-14 float noise as readily as in real
    signal. D256 was one `sd()` away from a fabricated finding.
23. **Decompose before attributing.** A pair statistic contains both main
    effects; a regime statistic contains the era; double-centre or demean first,
    then ask what is left.
24. **Check the sign of R^2 before the p-value.** A model can beat its own
    shuffled labels and still be worse than the mean. Significance answers "is
    there signal"; the sign answers "can it predict", and only the second one
    decides anything.
25. **Screen an input change for ALIGNMENT before building it.** `cov(shift,
    error) > var(shift)/2` is computable from a crude version of the signal and
    costs one script. D261 killed a queued project on a 15x miss.
26. **Print a signal's own variance.** A degenerate signal reads exactly like a
    null result at the endpoint.
27. **Alignment is necessary, not sufficient.** `cov > var/2` says a change
    helps rather than harms; it says nothing about size. D262's best signal was
    well-aligned and worth 0.007% of RMSE. Report dRMSE beside the ratio.
28. **A screen that passes every candidate is measuring itself,** not the
    candidates. Identical scores across a heterogeneous family is the tell.

---

## E. `DATA_BLOCKED` — not disproven

Distinguishing these from failures matters: nothing here was tested and lost.
Each is an acquisition task.


- **Constrained replacement graph.** No `game_rotation` table exists, and
  `lineup_stints` covers 48% / 20% / 27% / 47% / 103% / 66% / 107% of games by
  season. Too patchy to learn who absorbs whose minutes without coverage
  selecting the sample. **Data acquisition, not modelling.**
- **Injury-event → market response.** `edition` has cardinality **1** across all
  125,704 rows and all 1,260 raw PDFs are 5PM editions. There is no timestamp
  *pair* in held data. The NBA publishes several editions a day; this project has
  only ever downloaded one.
- **Anything cross-book or price-before-point.** `odds_quotes` is empty; the
  historical stack is one row per game with a two-value open/close `phase` and no
  timestamp column.
- ~~**Moneyline-based work after 2022-23.**~~ **RESOLVED — D249. This entry was
  wrong.** ML is indeed zero from 2024 onward in `odds_market` and
  `odds_hist_sbr`, but those are not the only sources held. The Action Network
  raw in `data/raw/sbr_ext/` carries moneyline for **99.8% / 98.5% / 95.9%** of
  regular-season games in 2023-24 / 2024-25 / 2025-26 across 6–7 books, plus
  totals at the same coverage. The lesson is the reusable part: **this was
  recorded as blocked after checking two tables, not after checking the files on
  disk.** A `DATA_BLOCKED` verdict must name every source inspected.
