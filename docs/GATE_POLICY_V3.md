# GATE POLICY V3 — pointer

> **DATA-COVERAGE CAVEAT — READ BEFORE QUOTING ANY NUMBER IN THIS FILE.**
> The daily NBA injury report, which the model's availability leg depends on,
> begins **2018-12-17 — mid-way through 2018-19**. Coverage of regular-season
> game dates is **0% before that, 63.7% in 2018-19, and 95–100% from 2019-20
> onward**. Only **2019-20 → 2025-26 (7 seasons)** is fully covered, and that is
> the only frame in which the model runs as designed. Earlier seasons score a
> *crippled variant* whose availability leg is fed inputs it was never meant to
> have. Any figure here spanning seasons before 2019-20 — including every
> 14-season and 19-season figure — blends two different models and should be
> read as historical context, not as a description of the shipped system.
> (`D186`)

**The V3 policy lives at [`docs/GATE_POLICY_V2.md`](GATE_POLICY_V2.md).**

It was kept at the V2 path on purpose. Every citation in DECISIONS.md,
FEATURE_LEDGER.md, CONTAMINATED_GATES.md and the scripts already resolves to
that path, and the standing cleanup rule is "if cleanup breaks old citations,
do not do it" (D134). Renaming the file would have broken ~40 live references
to buy nothing.

What is in there:

* **Sections 1-7 — V2, verbatim and unrenumbered.** Power arithmetic
  (MDE80 = 2.802·sd/√n), tiers T0/T1/T2/T3, the BH FDR family register, the
  NS-portfolio and its selection floor b_sel = Σ 0.8·SE_i, the
  pre-registration rules, the V2 retroactive audit.
* **Section 8 — the multi-split requirement** (rolling-origin, LOSO, legacy,
  block bootstrap; and why LOSO is a stability diagnostic, not k proofs).
* **Section 9 — clustered inference**, mandatory, with the measured design
  effects for our own arms and the small-K warning.
* **Section 10 — the era statement**, citing [`docs/ERAS.md`](ERAS.md).
* **Section 11 — adjudication when the folds disagree** (the mechanical table
  and the tie-break order).
* **Section 12 — the V3 retroactive audit** of every ship where the single
  split was load-bearing.
* **Section 13 — what V3 does not change.**

Implementation: `nbapred/eval/splits.py`. Tests: `tests/test_splits.py`.
Re-scoring: `scripts/cv_rescore.py` → `data/cv_rescore.json`.
Clustering audit: `scripts/cv_cluster_audit.py` → `data/cv_cluster_audit.json`.
Era measurement: `scripts/era_measure.py` → `data/era_signatures.json`.

Registered as **D139**.
