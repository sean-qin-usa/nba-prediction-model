# Hand-set constants: which are derivable, which are free

> **DATA-COVERAGE CAVEAT — READ BEFORE QUOTING ANY NUMBER IN THIS FILE.**
> The daily NBA injury report, which the model's availability leg depends on,
> begins **2018-12-17 — mid-way through 2018-19**. Only **2019-20 → 2025-26
> (7 seasons)** is fully covered, and that is the only frame in which the model
> runs as designed. Figures spanning earlier seasons blend two different models.
> (`D186`)

This document answers a fair criticism: the architecture is handcrafted, and it
carries fixed constants at suspiciously round values — ridge 25, home-effect
ridge 200, link scale 7.2, shrinkage `n/(n+600)`, a 50/50 blend. The criticism
asks whether these should be calibrated rather than chosen.

**The answer is not "tune everything."** With seven reliable seasons, a large
hyperparameter search manufactures more than it finds — measured in this
repository at **+16.92 ROI points from a 600-cell search on pure noise**
(`D165`). But "leave them alone" is also wrong. The useful distinction is not
*tuned vs fixed*; it is **derivable vs searchable**.

## The two kinds of constant

| | **Type A — derivable** | **Type B — searchable** |
|---|---|---|
| definition | a deterministic function of a quantity estimable on the training fold | no closed form; can only be chosen by comparing held-out performance |
| how it is set | recomputed by formula inside every fold | grid search against a validation split |
| degrees of freedom consumed | **≈ zero** | one per grid dimension |
| out-of-sample status | **out of sample by construction** | requires nested CV: inner validation, untouched outer fold |
| behaviour on 7 seasons | safe | mostly manufactures noise |

**The right programme is to move constants from Type B to Type A wherever a
derivation exists, and to leave the irreducibly-Type-B ones fixed and declared.**
That is a stronger claim than "we tuned it," because a Type-A constant needs no
held-out data at all — it is a plug-in estimate, not a selection.

What follows is that audit, run on the corrected 2019-26 frame (`D192`).

## C1 — link scale 7.2 is **not a free parameter**

`P(home win) = sigmoid(margin / 7.2)`. The 7.2 looks hand-picked. It is not:
it is pinned by the residual spread of NBA margins.

A logistic with scale `s` has SD `sπ/√3`. Matching it to the margin-residual
normal gives the plug-in `s = σ·√3/π`, which uses **only training residuals and
no outcome search at all**:

| | value |
|---|---|
| margin-residual SD on the frame | **13.653 pts** |
| Type-A plug-in `σ√3/π` | **7.527** |
| full-frame 1-D MLE | **7.721** |
| **shipped** | **7.200** |

Walk-forward — estimate the scale on seasons 1..k, score k+1:

| test season | MLE on prior | ll(fitted) | ll(7.2) | delta |
|---|---|---|---|---|
| 2021-22 | 7.979 | 0.61922 | 0.61953 | −0.00031 |
| 2022-23 | 7.895 | 0.63417 | 0.63461 | −0.00044 |
| 2023-24 | 7.895 | 0.59877 | 0.59921 | −0.00044 |
| 2024-25 | 7.864 | 0.58857 | 0.58872 | −0.00014 |
| 2025-26 | 7.806 | 0.58659 | 0.58631 | **+0.00028** |

Mean delta **−0.000212 nats**, season-clustered **t = −1.58 (K=5), ns**. The
fitted scale is remarkably stable across folds (**7.81–7.98**).

**Reading.** 7.2 is slightly too tight — the data wants ~7.8–7.9 — but the cost
is 0.0002 nats against a model-market gap of 0.0129, i.e. **1.6% of the gap**,
and it is not significant. The honest statement is not "7.2 was tuned"; it is
**"7.2 is a plug-in value implied by a property of the sport, and re-deriving it
inside each fold is free and marginally better."** That is a real upgrade
available at zero degrees of freedom, and it is the single clearest Type-B →
Type-A move in the model.

## C2 — the 50/50 blend is suboptimal, nearly costlessly

For two unbiased estimates of the same margin, the variance-minimising weight is
the inverse-variance rule `w = (var_b − cov) / (var_a + var_b − 2cov)`. That is
Type A: closed form, computable on the training fold.

| | value |
|---|---|
| residual variance, four-factors leg | 211.52 |
| residual variance, composition leg | 203.43 |
| covariance | 196.09 (**corr 0.945**) |
| **inverse-variance weight on four-factors** | **0.322** |
| bootstrap 95% CI | **[0.225, 0.418]** |
| **shipped** | **0.500** |

**Two independent methods agree.** The register's earlier fitted search — a
completely different procedure, minimising held-out log loss — preferred **≈0.30**.
The closed-form derivation lands at **0.322**, and the bootstrap interval
**excludes 0.5** (P(w > 0.5) = 0.000). So the 30/70 preference recorded earlier
was not a search artifact; it is what the variance structure implies.

**And yet moving there buys almost nothing**, because the objective is flat:

| weight on four-factors | blend residual RMSE |
|---|---|
| 0.300 | 14.1800 |
| **0.322 (optimal)** | **14.1796** |
| 0.400 | 14.1845 |
| **0.500 (shipped)** | **14.2050** |
| 1.000 | 14.5438 |

The whole distance from optimal to shipped is **0.025 points of margin RMSE —
0.18%**. That is why the register's log-loss test measured only +0.00077 nats
with an interval crossing zero.

**The reason is the 0.945 correlation between the legs.** When two estimators
agree that closely, the denominator of the inverse-variance rule
(`var_a + var_b − 2cov = 22.8`) is **10.8% of a single leg's variance**, so the
optimal weight sits at the bottom of a very shallow bowl. It is identifiable but
barely consequential.

**Reading.** 0.5 is genuinely suboptimal, and we can now say so with a derivation
rather than a search. It is also nearly costless, and 0.5 is the robust choice
when the objective is this flat and the two inputs this collinear. Keeping it is
defensible; the defensible *reason* is the flatness, not "we picked a round
number."

## C3 — `n/(n+600)`: a correction to this audit's own first attempt

The first pass at this document estimated `k` as a team-mean shrinkage from a
between/within variance decomposition on team margins, got `k ≈ 9`, and was about
to report the shipped 600 as **65× too large**. That comparison was wrong and is
retracted — it compared two different objects.

`n/(n+600)` (`latestate.py:70`, `tanking.py:46`) is applied to a **fitted
coefficient** (`beta_form`, `beta_out`, `beta_tank`), where **n is the number of
active fit rows**, not a team's game count. It is a burn-in guard: it holds a
coefficient near zero while the fit is thin and releases it as rows accumulate.

| n rows | n/(n+600) | |
|---|---|---|
| 100 | 0.143 | hard burn-in |
| 600 | 0.500 | half released |
| 3,000 | 0.833 | mostly released |
| 12,000 | 0.952 | effectively off |

At the row counts these fits actually reach, the factor is ~0.9–0.95 — a mild
residual shrink whose real work is the early-sample guard.

**What is genuinely free here is *where the burn-in releases*.** The Type-A
alternative exists: for a coefficient with estimate `b` and standard error `se`,
the James–Stein / empirical-Bayes shrink is `b²/(b² + se²)` — estimable from the
fit itself, no grid, no held-out data. That would replace a chosen 600 with a
quantity the fit reports about itself. **Not implemented; recorded as the
strongest remaining Type-B → Type-A candidate.**

## Where this leaves the criticism

The criticism's central point stands and is not disputed anywhere in this
repository: **the architecture was developed on 2021–26 data and then inserted
into earlier walk-forward simulations.** The walk-forward honestly tests
parameter refitting and betting-rule selection; it does not make the architecture
historically out of sample. The ablation that removes the era-specific terms
flips the simulated strategy from **+3.54% to −3.70%**, which is direct evidence
of era-tailoring, and it is reported in the README rather than buried.

What this audit adds is that **the round numbers are not the weak point.** Of the
three examined, one is a plug-in implied by the sport (7.2), one is suboptimal by
an amount too small to matter and for a reason we can now name (0.5, on legs
correlated 0.945), and one was mis-audited by us before being corrected (600).
The remaining unvalidated constants — ridge 25, the home-effect ridge 200,
lookback and roster windows — are the ones that deserve the Type-A treatment
next, and ridge in particular has a standard one (generalised cross-validation or
marginal likelihood, both computable inside a fold).

**None of that changes the bottom line.** The model is frozen. 2026-27 is the
first decisive prospective evaluation. Until it is scored, this is a research
system and a candidate strategy, not a validated edge.
