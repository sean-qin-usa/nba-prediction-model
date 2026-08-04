# GATE POLICY V3 — shipping rules for the small-effect era, with split rigor

> **V3 BANNER (D139, 2026-08-01).** This file is now the V3 policy. It still
> lives at `docs/GATE_POLICY_V2.md` **on purpose**: every existing citation in
> DECISIONS.md, FEATURE_LEDGER.md and the scripts resolves to this path, and the
> codex product-pass rule ("if cleanup breaks old citations, do not do it")
> applies. `docs/GATE_POLICY_V3.md` is a pointer to here.
>
> **Sections 1-7 below are V2 VERBATIM and UNRENUMBERED.** T0/T1/T2/T3 keep
> their meanings; the power arithmetic, the FDR family register, the
> NS-portfolio and the selection floor are unchanged. V3 **ADDS** sections
> 8-12: a multi-split requirement, a clustered-inference requirement, an era
> statement, and a mechanical rule for adjudicating disagreeing folds. Nothing
> in 1-7 is repealed. A gate that passed V2 does not automatically pass V3 —
> see section 12 for the retroactive audit.

Status: PROPOSED (Task C methodology review, 2026-07-30). Supersedes the
single-test clause G1 of COMPLEXITY.md **if adopted**; every other COMPLEXITY.md
mechanism (calibration veto, one-change-per-eval, ledger-the-losers, removal as
first-class) stands unchanged. Numbers: `scripts/ba_gatepower.py` (read-only on
data/capstone_pergame_*.csv; paired bootstrap 2000x seed 20260730).

## 1. Why V1 no longer fits: the measured power problem

V1 rule (G1): ship iff the paired-bootstrap 95% CI on the pooled per-game
log-loss delta excludes zero. That rule was calibrated in the era of
0.005-0.02 effects (D16/D21/D33/D46). Current true effects are 0.0005-0.002
(D55 +0.00099, D62 +0.00097, the NS-portfolio 0.0002-0.0024).

Observed per-game paired loss-delta sd (real artifacts):

| change | footprint | pooled sd | changed-game sd |
|---|---|---|---|
| D46 sched layer (legacy→sched) | 100% of games | 0.096 | 0.096 |
| D47 dead-team (sched→dead) | 98.6% | 0.078 | 0.078 |
| D62 carry (csfix→carry) | 35.0% | 0.030 | 0.051 |

MDE at 80% power for the 95% pass rule, MDE80 = 2.802·sd/√n:

| sd | n=1230 (one season) | n=3690 (pooled) |
|---|---|---|
| 0.030 (carry-like local) | 0.00240 | 0.00138 |
| 0.050 | 0.00399 | 0.00231 |
| 0.096 (observed global) | 0.00767 | 0.00443 |
| 0.150 | 0.01198 | 0.00692 |

Power to pass at n=3690, observed global sd 0.096: **5.0% at Δ=0.0005, 9.2% at
0.001, 24.4% at 0.002**. A single season (n=1230) has 3.8-10.9% power across
the whole true-effect band — indistinguishable from the 2.5% false-pass rate.
Conclusion: for GLOBAL-footprint features the V1 gate's MDE80 (0.0044) sits
ABOVE the entire current effect band. Only local-footprint changes
(carry-like, pooled sd 0.030 → MDE80 0.0014) are still adequately testable.

**Footprint fact (measured, not assumed):** unchanged games contribute exactly
zero delta AND zero noise to the paired test, so conditioning on
"changed games" does NOT raise power — carry's pooled z (1.66) equals its
changed-subset z (1.66) identically. Power gains come only from pre-registered
EFFECT-CONCENTRATION windows that exclude changed-but-diluted games: carry's
gp[0,5) basis z=2.61 vs pooled z=2.20.

## 2. What the V1 rule has already cost us

The NS-portfolio (BRIEF4): six positive-direction results below single-test
resolution, with implied z from their registered CIs:

| member | delta | SE | z | power if true |
|---|---|---|---|---|
| 3P-luck defense-only | +0.00041 | 0.00042 | 0.98 | 16.3% |
| event-recency (isolation gate) | +0.00138 | 0.00112 | 1.24 | 23.4% |
| comp-heavy 60/40 | +0.00100 | ~0.00051 | ~1.96 | 50.0% |
| dead-team term (recomputed vs sched) | +0.00038 | 0.00128 | 0.30 | 4.8% |
| continuity prior | +0.00020 | ~0.00040 | ~0.50 | 7.2% |
| carry ship-confirm (D63 addendum) | +0.00083 | 0.00051 | 1.63 | 37.2% |

- If ALL six were true at their point estimates, the V1 gate would be expected
  to pass **1.39 of 6** — i.e. it MISSES ~4.6 true effects of this size.
  Observing zero/one passes among them is exactly what "all real" predicts.
- All-six-positive under a global null: sign-test p = 0.016. Naive Stouffer
  z = 2.70 (p = 0.0035) — but the members were CHOSEN for positive sign, and
  after half-normal selection correction the combined evidence is z = 1.23,
  **p = 0.109**. Honest read: the portfolio likely contains 1-3 real effects
  worth ~0.001-0.003 pooled combined (point-estimate sum 0.0042, selection-
  shrunk), but this data cannot say WHICH members are real — and cannot fully
  rule out that none are.
- Best estimate of V1's cost to date: **~2-4 true small effects false-rejected,
  ~0.001-0.003 pooled log-loss left unshipped.**

## 3. The V2 policy — tiers

### T0 — Correctness fixes (exempt from the effect gate)
A change qualifies iff the code was demonstrably wrong against its own spec
INDEPENDENT of any OOS delta (D54 precedent: cold-start `wh=0` dead-code bug).
Ships on construction evidence; the paired delta is measured and REGISTERED but
is not the shipping criterion. Guardrail: a "fix" whose only justification is a
positive delta is a feature and goes through T1/T2. (This retroactively
formalizes D55.)

### T1 — Solo-ship
Pre-registered single config. Requirements (all of):
1. Paired 2000x bootstrap 95% CI vs the SAME-RUN control excludes zero
   (control built via fit_production imports; per-season lls cross-checked
   against the current headline artifact, today `capstone_pergame_carry.csv`).
2. Point estimate ≥ 0.002 pooled, OR ≥ the MDE80 of its pre-registered
   effect-concentration window (computed BEFORE the run, section 5.5).
3. Survives Benjamini-Hochberg FDR at q = 0.10 across the running gate family
   (K = 57 enumerated as of 2026-07-30, `scripts/audit_gate_stats.py`;
   family grows append-only, section 4).
4. Calibration battery veto unchanged (COMPLEXITY.md gate 2).

### T2 — Monitored portfolio (0.0005 ≤ estimate < 0.002, or CI straddles zero)
Individually NS positive effects with ≥2/3 season sign-consistency enter an
append-only NS-PORTFOLIO ledger (this file, section 6). They do NOT ship solo.
Bundle mechanics:
- **Freeze first.** Bundle membership is fixed by a pre-set objective rule
  (registered in DECISIONS.md before the joint run), e.g. "all ledgered members
  with z ≥ 0.5 and mechanically independent channels". No hand-picking after
  seeing joint numbers; no adding/dropping members between freeze and gate.
- **Joint gate.** ONE new paired run of the full bundle vs same-run control at
  95%. Because members were sign-selected on the SAME 3690 games, a joint pass
  on this data is biased upward by the selection floor
  b_sel = Σ 0.8·SE_i (half-normal null mean). For the current 5 shippable
  members b_sel ≈ 0.0030 vs point-sum 0.0034 — **the same-data joint gate
  cannot cleanly separate selection artifact from truth.** Therefore:
- **Provisional ship + fresh-data confirmation.** If joint CI_lo > 0 AND joint
  point ≥ 0.002, the bundle ships PROVISIONAL with a mandatory shadow control
  (the no-bundle model keeps running all season) and a pre-registered 2026-27
  kill rule: evaluate paired delta at n≈615 (mid-season) and n=1230
  (season end); KILL if the mid-season point estimate < 0 with CI excluding
  +0.002, or if the season-end 95% CI excludes zero on the harm side; PROMOTE
  to permanent only if the season-end fresh-data point estimate > 0. Fresh
  games are the only unbiased test the selection left us.
- **Future portfolios**: select members on all-but-latest season, confirm the
  joint bundle on the held-out latest season before any provisional ship (the
  current portfolio cannot do this — all 3 seasons were used in selection —
  which is why the fresh-season kill rule above is mandatory).

### T3 — Reject
Point estimate < 0.0005 pooled-equivalent, sign-inconsistent across seasons, or
autopsied as construction/confound artifact (homesens venue-proxy precedent).
Goes to FEATURE_LEDGER.md with false-rejection risk per existing protocol.
Underpowered NS results are ledgered as "underpowered-NS", NOT "rejected"
(section 5.5).

## 4. Family bookkeeping (FDR)
- Every CI-gated comparison is appended to the family register AT
  PRE-REGISTRATION TIME (not at result time), winners and losers alike —
  COMPLEXITY.md gate 3 already requires logging losers; V2 makes the log the
  denominator of a computed correction instead of a vibe.
- Record the bootstrap one-sided p (`p_wrongside`, already emitted by
  stat_link-style gates) for every comparison.
- At each ship decision, recompute BH at q = 0.10 over the full family.
  T1 requires BH pass; T2 members are exempt (they never ship solo);
  parameter sweeps enter as ONE family member but gate internally at
  Bonferroni alpha/m (the fadeshape 7-config ci99.2857 precedent, codified).

## 5. Pre-registration rules (anti-selection, all mandatory)
1. BEFORE running any gate, register in DECISIONS.md (or the journal):
   exact config, same-run control definition, eval universe and any
   effect-concentration window, bootstrap spec (paired, 2000x, seed),
   pass rule and tier claimed, sweep size m (1 if none).
2. Subgroup results only count if the subgroup was registered before the run.
   Post-hoc subgroup passes (e.g. eventrecency 2024-25) are exploratory and
   require a fresh pre-registered confirmation to count for anything.
3. One config per hypothesis. A failed gate may be re-run ONLY with a
   documented construction artifact (FEATURE_LEDGER hall-of-shame classes);
   the retest supersedes the original and is a NEW family member.
4. Baseline discipline: gate against the SHIPPED model's same-run control
   (D45 rule), never a strawman; cross-check control per-season lls vs the
   current headline artifact before trusting any delta.
5. **Power floor:** compute MDE80 for the registered design first
   (`ba_gatepower.py` section B; MDE80 = 2.802·sd/√n with sd taken from the
   nearest measured analogue in section 1). If MDE80 > 3x the plausible
   effect, the test as designed is uninformative — localize the window,
   redesign, or defer; do not run it and call the NS a rejection.
6. Market data (p_mkt) may define analysis subsets, never features (G2).

## 6. NS-portfolio ledger — RESOLVED ON FRESH DATA (D138, 2026-08-01)

### 6.0 What section 6 said on 2026-07-30 (kept for the record)
Shippable members (5): 3P-luck defense-only, event-conditional recency,
comp-heavy 60/40, dead-team term, continuity prior. (The 6th positive-NS
result, the D63 carry ship-confirm, is already in production and is listed in
section 2 as evidence of the resolution problem, not as a candidate.)
Bundle freeze rule proposed for registration: all five, joint gate per T2.
Expected joint z if all point estimates are real and channels independent:
Σd/√(ΣSE²) ≈ 0.0034/0.0019 ≈ 1.8 — i.e. even the joint gate is only ~46%
powered; the fresh-season confirmation is doing the real work.

### 6.1 The fresh data arrived and the test was run — D138
This section's "fresh-season confirmation" requirement was satisfied WITHOUT
waiting for 2026-27: D101 unlocked 2021-22 + 2022-23 (2,458 games), which
D110 certifies "no gate from D46 through D102 could select on", and D132
certified all five seasons. Pre-registration **data/nsport_prereg.md sha256
2cec470fa252b27a1cf8adf72cc15ac3d79ba5abd19aff423764f62db42f686e** was frozen before
any holdout number existed. Control-hash 6,148/6,148 vs the D132 certified CSV,
max|dp| 1.366e-14, 0.0000 of games moved.

**RESULT — HOLDOUT JOINT +0.001979 CI(+0.000283,+0.003652) n=2,458, PASS**
(p_wrongside 0.011). Pre-registered predictions: all-members-real = +0.00200
(**observed = 99.0% of it**), none-real = 0 (**rejected, z = +2.29**).
Dev 23-26 +0.00137 NS; all five seasons +0.00161 CI(+0.00049,+0.00270) PASS;
per-season +0.00227/+0.00169/+0.00233/+0.00106/+0.00072 = **5/5 positive**;
DiD holdout−dev +0.00061 NS, i.e. the effect is LARGER out of sample.

### 6.2 Final per-member outcome
| member | ledgered Δ | outcome | fresh-data Δ (holdout) |
|---|---|---|---|
| 3P-luck defense-only | +0.00041 | **CONFIRMED, in bundle** | **+0.00072 CI(+0.00005,+0.00136) PASS** |
| comp-heavy 60/40 | +0.00100 | **CONFIRMED but DECAYING** | **+0.00129 CI(+0.00010,+0.00240) PASS** |
| D130 ARM A urgency | +0.00048 | in bundle, ATTENUATED (cold k_u) | −0.00021 NS — uninformative |
| D136 ARM A travel | +0.00011 | in bundle, positive-NS | +0.00042 NS; dev reproduces D136 exactly |
| event-recency (F2) | +0.00138 | **REFUTED** — struck (D124) | −0.00087 on the same holdout |
| dead-team term | +0.00038 | dropped, z = 0.30 < 0.5 | not scored |
| continuity prior | +0.00020 | **VOID — moves 0 of 6,148 games** | not scorable |
| carry ship-confirm | +0.00083 | SHIPPED (D62/D63), in the control | n/a |

**ROW CORRECTION (do not propagate the old numbers).** Section 2's continuity
row "+0.00020, SE ~0.00040, z ~0.50" is wrong twice: the measured values are
pooled **+0.00005 CI(−0.00001,+0.00010)** and early-window **+0.00019
CI(−0.00002,+0.00039)**, both against a **PRE-CARRY** control, and the term is
now structurally unreachable (the `not ff.ready` ratings-fallback branch was
deleted from production.py; D62 carry makes ff.ready true at every opener).

### 6.3 Section 2's cost estimate, now measured
Section 2 estimated "~2-4 true small effects false-rejected, ~0.001-0.003
pooled log-loss left unshipped". **CONFIRMED and sharpened: 2 confirmed
members, +0.0016 to +0.0020 backward-looking (interval +0.0005 to +0.0037) =
14-18% of D132's remaining +0.01120 raw gap.** Forward-looking it is
**+0.0007-0.0011**, because comp-heavy 60/40 is measurably fading
(per-season +0.00147/+0.00112/+0.00164/+0.00062/**−0.00037**, correlating
r = 0.946 with D102 RT4's independently-built fitted-blend profile, which is
also negative on 2025-26). A meta-analysis of all 25 rejected sides channels
(D138 §9) pools to +0.000049 ± 0.000072; with these four members removed the
remaining 18 pool to **+0.000012 ± 0.000081 — flat zero**.

### 6.4 Corrections to this policy's own mechanics, earned by the run
1. **"Mechanically independent channels" must be MEASURED, not judged.**
   3P-luck and comp-heavy were frozen as independent and are partial
   SUBSTITUTES: Σmembers +0.00420 vs joint +0.00198 = **47.1%** on the holdout
   (dev 92.5%). Improving the FF leg and then down-weighting the FF leg
   overlap by construction. Any future T2 bundle must report joint/Σ and treat
   a ratio far from 1 as a membership error, not a curiosity. D64's
   "members stack additively" does not generalise.
2. **A T2 member with a COLD walk-forward coefficient on the confirmation set
   is untestable, not refuted.** D130 ARM A ran at k_u 0.95/0.69 on the
   holdout vs 1.84/2.59/2.87 on dev because `pg_urgency2.py` carries a
   `season >= '2021-22'` corpus-floor literal — hall-of-shame #8, the exact
   D110/D112 pathology. Check every member's estimator warmth on the
   confirmation set BEFORE freezing, and record it in the pre-registration.
3. **Every dev/holdout verdict must carry a between- vs within-era
   decomposition.** Our split is exactly the pre/post Player-Participation-
   Policy boundary (2023-24 also brought the In-Season Tournament and the
   apron regime), and 2021-22 contains the Omicron wave, so "fails holdout" is
   confounded with "is an era effect" unless tested. For the bundle:
   between-era share 21.9%, F(1,3) = 0.840 — noise, not era. Same test clears
   D124's F2 profile (17.5%, F = 0.634) and D102 RT4's (15.3%, F = 0.541).
4. **LOSO is the right stability diagnostic for a 5-season corpus, and it is
   only a diagnostic.** With frozen configs, "fit on 4 / score 1" reduces to
   the per-season estimate; the folds reuse data and are NOT five independent
   confirmations. Report sign-consistency and spread, never a fold-count p.
5. **The selection floor b_sel is a DEV-ONLY quantity.** On a selection-clean
   holdout it is identically irrelevant, which is why the fresh-data clause in
   section 3 was worth writing and worth honouring.

### 6.5 Status
**PROVISIONAL PASS. NOTHING SHIPPED.** Remaining ship conditions (D138 §12):
calibration-battery veto, BH q=0.10 over the family, resolution of the
comp-heavy decay, the 2026-27 shadow control with its kill rules, and a human
ship decision as its own D-line. Recommended next step is a SOLO T1 gate on
3P-luck defense-only, the only member that is both fresh-data confirmed and
non-decaying (5/5 seasons positive, between-era share 27.2%).
**THIS HOLDOUT IS NOW SPENT for all four members** — no win-probability retest
on these 6,148 games; the next unbiased read is 2026-27.

### 6.6 D141 UPDATE — the solo M1 gate was run, and it is a NO-SHIP
Two of D138 §12's five conditions are now DISCHARGED and one of them KILLS the
solo ship. M1's status is UNCHANGED: T2 monitored portfolio member.

| condition | outcome |
|---|---|
| calibration-battery veto | **PASS** — ECE10 −0.00104, ECE20 −0.00402, Brier −0.00017, link slope 0.9725→0.9817 (toward 1), HL p 0.467→0.301, 4/5 seasons better. Nothing degrades. |
| BH q=0.10 over the family | **FAIL** — family recounted append-only to **K=106**, +1 for the gate = 107. On the §9-mandated clustered p (cluster-mean t, 4 dof, **0.0378**) M1 ranks 28 against a threshold of 0.02617. It survives only on the i.i.d. p (0.0155, rank 26, thr 0.02430), which §9.1 demotes to a secondary. Enumeration: `data/bh_family.csv`. |
| M2 decay | **UNRESOLVED — M2 HELD.** Trend −0.00042/season, 2025-26 **−0.00037**, corroborated at r=0.946 by D102 RT4. Resolution is the 2026-27 one-shot **as a kill rule**, since n=1230 gives MDE80 ~0.0024 and cannot confirm a +0.001 effect. |
| 2026-27 shadow control | not yet run |
| human ship decision | **recorded: NO-SHIP (D141).** |

**THE V3 BATTERY ON M1 SOLO** (re-scoring of the D138 artifact, no model
re-run, so it adds no new data and cannot resolve selection): pooled +0.00047,
season-cluster CI(+0.00024,+0.00087) SIG, **cluster-mean t at 4 dof
CI(−0.00008,+0.00101) ns**, block bootstrap ns, month-cluster ns, ICC −0.00020
(DEFF 0.755), rolling-origin **4/4 positive no sign flip**, LOSO 5/5,
**ERA-STABLE** (I²=5%, Q p=0.371). `adjudicate()` = MULTI-SPLIT PASS (with
notes). **It still cannot ship**, because §11 is an overlay on the V2 tier and
the V2 tier fails: point **+0.00047** is below its own **MDE80 0.00063** and
below the 0.002 T1 floor, so §3's requirement 2 is unmet — and BH is unmet too.

**A SIXTH CONDITION, ADDED BY D141 AND NOW MANDATORY FOR ANY NS-PORTFOLIO
SHIP: IMPLEMENTATION IDENTITY.** M1 has NO code path in `nbapred/`. The
`FF_LUCK` switch that looks like it implements it wires the BLUNT both-sides
variant — a different construction, registered in this family as a LOSER.
Before any member ships, diff the code the gate RAN against the code the
production switch REACHES, and record the diff in the ship entry.

## 7. Retroactive audit of past ships under V2

| ship | authorizing number | z | one-sided p | V2 verdict |
|---|---|---|---|---|
| D46 sched layer | +0.00539 CI(+0.0024,+0.0085) | 3.46 | 2.7e-4 | **T1 PASS clean** — ≥0.002, BH rank-3 pass, no action |
| D55 cold-start fix | +0.00099 CI(+0.00013,+0.00185) | 2.26 | 0.012 | **T0 exempt (bug fix)** — ship stands; as a FEATURE it would NOT solo-ship (T2 band; fails BH q=0.10 at rank 6: 0.012 > 0.0105) |
| D62 carry | pooled +0.00097 CI(+0.000085,+0.001816) | 2.20 | 0.014 | **T1 PASS via localized-window clause** — pooled gate alone fails BH (rank 7: 0.014 > 0.0123) and sits in the T2 band, but the pre-registered gp[0,5) basis (+0.0154, z=2.61, p=0.0045) passes BH, the early window (+0.00387, z=2.24) is registered, no slice harmed, single config, no sweep. Closest call of the three. |
| D63 ship-confirm | +0.00083 CI(−0.00018,+0.00181) NS | 1.66 | 0.049 | Confirm-regression: reproduces sign and 86% of authorizing estimate → acceptable under V2's confirm rule (≥50% and same sign). **Open item:** carry2 is currently bitwise-identical to carry (0 changed games) — the 2022-23 refill re-confirm promised in D63 has NOT yet landed; keep open. |

Net: no past ship is reversed under V2. D55 survives on the correctness
exemption, D62 on the localized-window clause — and both exemptions exist
precisely because those ships were, on reflection, correctly decided for
reasons the V1 rule couldn't articulate. What V2 changes going forward: small
global effects (0.0005-0.002) stop being tested one-at-a-time at a threshold
where the test is a coin with 5-25% heads probability, and start being pooled
into a jointly-gated, fresh-data-confirmed portfolio.

---

# V3 ADDITIONS (D139, 2026-08-01)

Sections 8-12 are new. They add SPLIT RIGOR and CLUSTERED INFERENCE to the V2
statistics. They do not replace anything above.

## 8. The multi-split requirement

**The problem V3 fixes.** Every gate from D46 through D136 was adjudicated on
ONE partition: dev = 2023-24..2025-26, holdout = 2021-22..2022-23. That
boundary, 2023-10-24, is ALSO the start of the Player Participation Policy, the
In-Season Tournament and the new CBA apron regime (docs/ERAS.md era **E5**). So
"passed dev, failed holdout" was AMBIGUOUS between overfitting (D111) and a real
era effect (D70) — and the register contains one documented instance of each,
with no way to tell them apart at gate time.

**8.1 What every gate must now report.** Use
`nbapred.eval.splits.full_report()`; do not hand-roll. Required, all four:

| split | what it is | what it proves |
|---|---|---|
| **ROLLING-ORIGIN** (expanding window: train ≤ k, test k+1) | the only causal split; mirrors live use | per-fold sign consistency is the closest thing this corpus has to REPLICATION |
| **LOSO** (delete-one-season) | stability / influence diagnostic | how much the pooled verdict leans on any one season |
| **LEGACY dev/holdout** | the pre-V3 partition, verbatim | continuity of citation only |
| **BLOCK BOOTSTRAP** (7-day calendar blocks) | temporally honest resample | whether significance survives within-week correlation |

Plus the **ERA DECOMPOSITION**: per-era estimate, between-era vs within-era
variance share, and DerSimonian-Laird Q / I² / τ on the era means.

**8.2 LOSO IS A STABILITY DIAGNOSTIC, NOT k CONFIRMATIONS.** On a 5-season
corpus any two LOSO folds share 75% of their data. `loso()` therefore returns
`independent_folds = 1` and an interpretation string, and **it is a policy
violation to present k folds as k proofs.** Report the fold SPREAD and each
season's INFLUENCE on the pooled estimate; never a fold count as evidence
strength.

**8.3 What rolling-origin does and does not buy us.** Our production stack is
already walk-forward at the PARAMETER level (weekly refits on a trailing 730d
window), so for a walk-forward artifact the per-season fold estimate is already
free of future information — rolling-origin adds the chronological reading and
the cumulative drift curve, not a new guarantee. What is **NOT** walk-forward is
HYPOTHESIS SELECTION (D111: "pre-registration protected our PARAMETER choices,
never our HYPOTHESIS choices"). No re-scoring of a fixed artifact can undo that.
Rolling-origin folds are honest about parameters and silent about selection.

**8.4 Fresh-season confirmation is still the only clean test.** Unchanged from
V2 §6 and D106's corollary. Multi-split raises the bar; it does not create new
data.

## 9. Clustered inference (mandatory)

**The defect, measured.** Every SIDES gate in the register resampled INDIVIDUAL
GAME deltas i.i.d. — `rng.integers(0, len(delta), size=(B, len(delta)))` — in
`of_transfer_ablation.py:90`, `apr_program.py:643` (D73), `es2_hardstop.py:319`
(D62), `ov_latestate_gate.py` (D90), `pg_urgency2.py:330` (D130),
`tv_gate.py:149` (D136) and in `ba_gatepower.py:44`, i.e. in the script that
produced §1's own power table. A sweep of the repo finds 98 files resampling
i.i.d. against 6 clustering. Per-game deltas inside a season are NOT independent:
schedule-layer betas, tank k, carry weights and the props ramp table are
estimated from shared data and move every game in the season together. The
i.i.d. bootstrap therefore UNDERSTATES the SE and the gate is ANTI-CONSERVATIVE.
The sister project `/hdd/steveqin/sean_dev/football_exercise` RETRACTED a
headline "we beat the market" claim for exactly this reason (0.9540 vs 0.9572
looked like a win; clustered by season it was −0.0031 CI[−0.0069,+0.0008] ns).

**9.1 REQUIRED.** Every gate reports, side by side:
1. the i.i.d. CI (secondary, for continuity with the register);
2. a CLUSTERED CI whose cluster level is JUSTIFIED BY THE MECHANISM through
   which the feature's coefficients are shared — season for anything fit
   per-season or on an expanding window, player for props row-level effects,
   7-day block for schedule-driven effects;
3. the intra-cluster ICC and the implied design effect
   (`splits.icc_oneway`), and
4. the CLUSTER-MEAN t INTERVAL at K−1 dof (`splits.cluster_mean_t_interval`).

The **shipping CI is the clustered one.** The i.i.d. CI may be quoted only as a
secondary.

**9.2 Measured design effects on our own arms** (DEFF_boot = SE_seasonCluster /
SE_iid; DEFF_anova = 1+(n0−1)·ICC; `scripts/cv_cluster_audit.py`):

| arm | intra-season ICC | DEFF_boot | DEFF_anova |
|---|---|---|---|
| D46 schedule layer | +0.00048 | 1.13 | 1.58 |
| D73 tank (old floor) | +0.00082 | 1.27 | 2.01 |
| D73 tank (warm floor) | +0.00003 | 0.93 | 1.03 |
| D90 late-state | +0.00105 | 1.39 | 2.29 |
| D62 carry | −0.00009 | 0.85 | 0.89 |
| D91 October bridge | −0.00008 | 0.86 | 0.90 |
| D124 F2 | +0.00008 | 0.91 | 1.10 |
| D132 headline vs market | −0.00032 | 0.67 | 0.61 |
| **D133 props ramp ARM A** | +0.00063 | **1.68** | **3.78** |
| D133 ARM A0 (level control) | +0.00097 | 1.96 | 5.29 |

**Rule of thumb for the register: our historical SIDES CIs were 0-40% too
narrow; our PROPS CIs (player-clustered but not season-clustered) were 68-96%
too narrow.** Sides deltas are dominated by irreducible outcome noise (y−p),
which really is independent across games, which is why the sides defect is
modest. Props deltas are not.

**9.3 SMALL-K WARNING — do not over-trust the cluster bootstrap.** With K=5
scorable seasons the between-cluster variance is estimated on 4 dof and the
cluster bootstrap is unreliable **in both directions**. Measured example: D130
ARM A has 3 seasons whose means are +0.00053 / +0.00047 / +0.00045, and the
season-cluster bootstrap returns SE = 0.00002 and a spuriously SIGNIFICANT
interval on a result that is plainly null. **Always report the ANOVA design
effect alongside the bootstrap, and treat the cluster-mean t interval as the
conservative bound.** Never ship on a K<10 cluster bootstrap alone.

**9.4 This does NOT resolve the "we are too conservative" question — both are
true.** V2 §1 measured that MDE80 for a global-footprint change (0.0044) sits
ABOVE the entire plausible effect band (0.0005-0.002), i.e. 5-24% power: we
false-REJECT small true effects. §9 measures that our SEs were understated: we
false-ACCEPT terms whose coefficients are season-shared. These act on different
quantities and correcting the SE makes the power problem WORSE. There is no
single knob. The only instrument that fixes both is MORE DATA (fresh seasons),
which is why V2 §6's fresh-season confirmation survives into V3 unchanged.

## 10. The era statement (mandatory)

Every gate must state, in its DECISIONS.md entry, in these words:

1. **Eval universe by ERA CODE**, not by season list
   (e.g. "dev = E5+E6, confirm = E3+E4"), per docs/ERAS.md.
2. **Era-availability check.** Does every input the feature consumes exist in
   every era being scored? docs/ERAS.md §5 is the trap list.
   `game_inactives` starts 2022-23 and `injury_reports_pit` starts 2023-10 — a
   feature that consumes either is STRUCTURALLY INERT on half or all of the
   legacy holdout, and calling that "does not transfer" is a category error
   (this is the D110 §1a cold-estimator mistake, generalised).
3. **Era-stability verdict**, in one of exactly three forms:
   * **ERA-STABLE** — I² < 50% and Q not significant at p=0.10. The effect may
     be quoted as one number and extrapolated to a new era.
   * **ERA-CONDITIONAL** — heterogeneous in MAGNITUDE, sign-consistent across
     eras. Ships, but the entry must state the per-era estimates and the live
     expectation must use the MOST RECENT era's estimate, not the pooled one.
   * **ERA-SPECIFIC** — sign flips across eras, or the effect is confined to one
     era. Does not ship. This is the D70 class.
4. **COVID-frame check.** If any fit frame includes E0/E1/E2, say so explicitly
   and report the estimate with and without. Precedent for why:
   **D136's two "SIG MATCH" margin coefficients (dtrav_kkm −0.3088 t=−2.21,
   d3in4 −0.6144 t=−2.00) exist ONLY in its FULL 2019-20..2025-26 frame; on the
   scorable 2021-26 frame they are −0.2903 t=−1.75 ns and −0.5827 t=−1.61 ns.**
   Including the COVID era MANUFACTURED both. Worse, the Orlando bubble's true
   travel is 0 km/team-game and `nbapred/model/travel.py` assigns 1,505.5 — the
   feature is not merely noisy there, it is fictitious.

## 11. Adjudication when the folds disagree

Mechanical, computed by `splits.adjudicate()`, so it cannot be argued after the
fact. Read it as an OVERLAY on the V2 tier: a change must clear its V2 tier
**and** land in a shippable V3 row.

| V3 row | condition | consequence |
|---|---|---|
| **MULTI-SPLIT PASS** | clustered pooled CI excludes 0, rolling-origin sign-consistent in all folds, ERA-STABLE, block bootstrap agrees | ships at its V2 tier |
| **MULTI-SPLIT PASS (with notes)** | as above but with a non-fatal flag (e.g. cluster-mean t straddles 0 at K=5) | ships at its V2 tier; the flag goes in the entry verbatim |
| **ERA-CONDITIONAL** | clustered pooled CI excludes 0 but I² ≥ 50% with consistent sign | ships ONLY with per-era estimates published; live expectation = most recent era; mandatory re-read next season |
| **PASS-WEAK** | clustered pooled CI excludes 0, ERA-STABLE, folds noisy (sd of folds > point estimate) | T2 monitored portfolio, not T1 solo-ship |
| **NO-PASS** | clustered pooled CI straddles 0, or rolling-origin sign flips, or a single season flips the pooled sign | no ship; ledger as underpowered-NS or era-specific per V2 §5.5 |

**Tie-breaks, in priority order:**
1. **Rolling-origin beats LOSO beats legacy.** If rolling-origin says no and
   LOSO says yes, the answer is no: LOSO folds are not independent (§8.2).
2. **Clustered beats i.i.d.** always (§9.1).
3. **A sign flip in ANY rolling-origin fold is disqualifying for T1** and sends
   the change to T2 at best.
4. **The most recent fold is the live forecast.** Where the folds trend, quote
   the trend and the last fold, not the mean. Measured precedent: D133's
   rolling-origin folds run +0.05639 → +0.03321 → +0.02851 (drift −0.0065 per
   season), so the honest 2026-27 expectation is ~+0.028, not the pooled
   +0.0404.
5. **A disagreement that coincides with an era boundary is an ERA finding until
   proven otherwise** — check docs/ERAS.md §5 for availability first, then the
   era decomposition, and only then reach for "overfitting".

## 12. V3 retroactive audit (`scripts/cv_rescore.py`, `data/cv_rescore.json`)

Every arm below is an already-registered on-disk artifact re-analysed under
§§8-11. No model was re-run: the deltas are bit-identical to what the original
gates saw, only the inference moved.

| ship / verdict | V2 status | V3 verdict | action |
|---|---|---|---|
| **D46 schedule layer** | T1 PASS clean | **MULTI-SPLIT PASS** — +0.00597, RO 4/4 positive, I²=27% ERA-STABLE, season-cluster CI(+0.00288,+0.00907) SIG, cluster-mean t CI(+0.00104,+0.01091) SIG | none. The one term that survives everything. |
| **D73 tank** | shipped, UNCONFIRMED (D112) | **PASS (with notes) / borderline** — pooled +0.00199 SIG i.i.d. and season-cluster, but cluster-mean t CI(−0.00036,+0.00434) ns; RO 4/4 positive, only 2025-26 SIG; drift +0.00140/season | stays shipped (positive in every era, harms nothing); D73's "PASSES, DECISIVELY" language is DEAD; re-read after 2026-27 |
| **D62 carry** | T1 via localized window | **NO-PASS** +0.00011 ns, RO 3/4, single season flips the pooled sign | confirms D111's "contributes ~0"; re-label, do not re-gate on this corpus |
| **D91 October bridge** | never a solo ship | **NO-PASS** +0.00012 ns | confirms |
| **D90 late-state** | REVERTED at D112 | **NO-PASS**, and the largest sides design effect measured (1.39 boot / 2.29 ANOVA) | revert vindicated twice over |
| **D124 F2 retirement** | RETIRED on pooled NS | **NO-PASS**, and ERA-STABLE (I²=19%) — so the retirement was NOT era-driven | retirement upheld |
| **D130 ARM A urgency** | unconfirmed-not-refuted | **NO-PASS**; and its artifact contains NO holdout seasons at all, so era-stability is structurally uncheckable | verdict unchanged; any 2026-27 one-shot must score all eras |
| **D133 props ramp (SHIPPED)** | T1, dev+holdout | **ERA-CONDITIONAL, SURVIVES** — RO 4/4 all SIG, 5/5 seasons SIG, season-cluster SIG, cluster-mean t CI(+0.02728,+0.05378) SIG; I²=70% is MAGNITUDE heterogeneity, no sign flip | **NO REVERT.** Publish per-era estimates; live expectation +0.028 (last fold), not +0.040 |
| **D138 NS-portfolio joint** | PROVISIONAL PASS on the holdout, +0.001979 CI(+0.000283,+0.003652), i.i.d. bootstrap | **SURVIVES CLUSTERING — and its i.i.d. CI was if anything CONSERVATIVE.** Intra-season ICC is NEGATIVE (−0.00054), season DEFF 0.50; all-5 pooled +0.00161 season-cluster CI(+0.00105,+0.00218) SIG, cluster-mean t CI(+0.00073,+0.00250) SIG, RO 4/4 positive, I²=0% ERA-STABLE. **MULTI-SPLIT PASS** | provisional status unchanged; §9 does not weaken it. One caution: on the 2-season holdout arm alone the MONTH-cluster CI straddles zero (−0.00020,+0.00425) and the t interval has 1 dof — read the 5-season arm |
| **D132 headline gap** | certified | robust to every split and every clustering (−0.01120, season-cluster CI(−0.01437,−0.00782)) | none |
| **D136 travel margin readout** | "well-powered readout", 2 SIG terms | **BOTH SIG TERMS ARE COVID-FRAME ARTEFACTS** (§10.4) | annotate D136; any retest uses the E3-E6 frame |

**Net: no shipped term is reverted by V3.** D46 is confirmed, D133 is confirmed
with an era caveat and a decay warning, D73 keeps its ship on a demoted claim,
and the two convicted terms (D90, D62) are re-convicted independently. The two
things V3 actually overturns are a **claim** (D136's significant travel/density
coefficients) and a **methodology** (i.i.d. gating).

**Does clustered inference explain D110/D111's evaporation? PARTIALLY.**
Applying the measured design effect to the registered z's: D46 3.46→3.05 PASS,
D73 2.81→2.22 (with its t-interval straddling zero), D90 2.68→1.93 FAIL. That
reproduces D111's per-term verdict from the INFERENCE ALONE — clustered CIs
would have caught the two convicted terms at gate time rather than 100 entries
later. But the selection channel is bigger and untouched by any SE correction:
E[max z] at our search depth (96-511 comparisons) is 2.34-2.91 all by itself.
Both channels are real; neither alone accounts for the shortfall.

## 13. What V3 does not change

The calibration-battery veto, one-change-per-eval, ledger-the-losers, removal as
first-class (COMPLEXITY.md); the T0-T3 tier definitions; the MDE80 power floor
and §5.5; the BH FDR family register and its append-only rule; the NS-portfolio,
its freeze rule, the selection floor b_sel = Σ 0.8·SE_i and the mandatory
fresh-season kill rule; the D45 same-run-control rule and the D134 control-hash
field; G2 (market data may define subsets, never features). All stand verbatim.
