# Feature ledger — every factor tested, verdict, and false-rejection risk

Sean's principle (2026-07-28): rejections can be construction artifacts too
(the frozen-asof bug nearly killed the composition model). Every rejected
feature gets a construction re-review. Risk = chance the rejection was an
artifact of HOW it was tested rather than the feature being genuinely useless.

## ACCEPTED (in production)
| feature | evidence | notes |
|---|---|---|
| Opponent-adjusted team ratings | 0.601 vs Elo 0.618 | core |
| **Availability-composition (DARKO×trailing-min)** | 0.5465 vs refit-ratings 0.5611 | Sean's idea; NEARLY false-rejected 3× in earlier constructions |
| Conditioning fix + truncated minutes (props) | CRPS 5.40→5.11; PIT 0.507 | audit find, not a feature |
| **D133 early-minutes ramp (props `proj_min`)** | **dev Oct-Nov points dCRPS +0.03909 CI(+0.03161,+0.04697) SIG, 3.5x realized MDE80, n=12,618/481 players; holdout 21-23 +0.04227 CI(+0.03449,+0.04979) SIG = 108% of dev; PIT 0.4820→0.4979 (October 0.4579→0.5013); 5/5 seasons SIG; Dec-Jun +0.00423 SIG better** | the fix D79 queued / D86 named twice / D128 quantified. **Mechanism is NOT calendar** — the same gp decay appears in Feb-Mar at team-gp>=30; the estimator conditions on `seconds>=720` and is blind to ABSENCE. Adversarial control passed: a matched single-constant level knob captures only 42.6% dev / 29.3% holdout (A−A0 +0.02245 SIG). Exactly zero at gp>=20 (4,337 rows bitwise unchanged). Env `PROPS_MIN_RAMP=0` restores pre-D133 |
| **D145 absence ramp (props `proj_min`, `PROPS_ABSENCE_RAMP`)** | **points CRPS +0.06001 on the pre-registered window `miss10>=5` (n=8,924 / 591 players, 5 seasons), SEASON-CLUSTERED CI(+0.04743,+0.07505) SIG — the §9.1 shipping CI — cluster-mean t at 4 dof CI(+0.03845,+0.08328) SIG, block bootstrap SIG, rolling-origin 4/4 SIG with drift +0.00366/season, LOSO 5/5 SIG, ERA-STABLE (I²=41%, Q p=0.166), BH rank 15/111 p=0.000875, PIT 0.4741→0.5036, 4.6x the pre-registered MDE80 of 0.0131** | **D133 fixed the PROXY, not the AXIS.** D133's ramp is keyed on games-PLAYED and is identically zero at gp>=20 = 65% of the universe, where the absence bias it named is still +0.83/+2.97 min at miss10 5-7/8-10 (and identical at team-gp>=30, so not calendar). The pre-registered discriminating test was the gp>=20 stratum: **+0.05755 CI(+0.04285,+0.07205) SIG**. Exactly zero on miss10<=4 (89.7% of rows, bitwise). §6.6 identity confirmed by RUNNING the switch: shipped-OFF reproduces the gate control bitwise (0/15,005), shipped-ON reproduces 105.3%. Complementary to D133, not a substitute (A−C = +0.00407 SIG; replacing the gp ramp HARMS untouched rows). **Caveat stated up front: a matched constant on the same window captures 82.5% (A−A0 = +0.01051 SIG) — most of this is the absence INDICATOR, a minority is the shape.** Env `PROPS_ABSENCE_RAMP=0` restores pre-D145 |
| D46 schedule layer | **the ONLY term that individually passes out-of-sample (D110/D112): +0.00598 held-out 21-23 CI(+0.00117,+0.01076) PASS** | walk-forward home edge + b2b + dead-team; ~all of the held-out campaign gain |
| D73 April tank term | **DEMOTED (D112)** — dev 24-26 +0.00334 PASS but held-out 21-23 **+0.00147 CI(−0.00036,+0.00329) NS** with a WARM coefficient; 23-24 +0.00032 NS | still enabled (`TANK_TERM=1` default): the held-out point estimate is positive and removing it is worse in all five seasons. But the registered "PASSES DECISIVELY" is a dev-season claim and D110's "efficacy transfers at 99%" was a cold-coefficient artifact (44% warm). UNCONFIRMED, not proven |

## DEAD CODE / REVERTED (registered as shipped, contributing nothing today)
Two ways a row in ACCEPTED goes stale: the term is switched off by a LATER
feature and nobody re-measures it (D16 — structurally unreachable), or its
acceptance evidence never covered a season the model had not been developed on
and the out-of-sample test, when it finally became possible, came back zero
(D90 — reverted). These rows exist so the ledger stops claiming credit the
shipped model does not earn. Entry criterion: a MEASURED number — games moved,
or a held-out effect with a CI — not an argument.

| feature | was | measured now | why it is dead |
|---|---|---|---|
| D90 late-state layer (form5 + outs, gp≥55) | ACCEPTED — pre-registered gate, pooled +0.00189 CI(+0.00053,+0.00329); active window +0.00560 | **held-out 21-23 +0.00014 CI(−0.00085,+0.00108)** (precise null, 5% of its dev +0.00267 PASS); **23-24 −0.00085** with the layer active on 33.6% of games; **DiD dev−held +0.00253 CI(+0.00005,+0.00492) SIG** | **REVERTED 2026-08-01 (D112): `LATE_STATE` defaults to "0".** Its gate was measured entirely on 2023-26; the corpus-floor literal made it identically 0 on 2021-22 and cold on 2022-23, so nobody could test it out-of-sample. Once the floor was derived from data and the coefficients ran warm, the layer's out-of-sample value is zero and its development-season advantage is the only one in the whole battery that survives at CI strength. Code + tests intact; `LATE_STATE=1` restores it |
| Cold-start prior (regress 0.75, D16/D55) | ACCEPTED — "early-season 0.669→0.644; wins every season" (Sean's idea) | **moves 0 of 6,148 games, all five seasons (D110)**; per-term ablation effect EXACTLY 0.00000 on DEV, HELD-OUT and GATE alike | it lives only in the pre-`ff.ready` ratings fallback, and **D62 cross-season carry makes `ff.ready` true from opening night**, so the fallback branch never executes. D67-R3 predicted this exactly ("carry makes ff.ready → ENTIRE fallback branch dead again; D55 attribution stale"). The D54/D55 construction fix (+0.00099, T0-shipped) was measured BEFORE carry shipped and contributes exactly zero today. The original 0.669→0.644 evidence is not wrong — it is **stale**: it describes a stack that no longer exists. To revive the claim the prior must be wired into the MAIN path behind its own gate; until then it is not a shipped feature. |
| `player_rates_kalman` (props) | — | see REJECTED row below | EWMA is the shipped estimator; the Kalman module is unreferenced by production |

## REJECTED — construction re-reviewed
| feature | verdict | false-rejection risk | why |
|---|---|---|---|
| Composition (RAPM-sum) | 0.608 worse | **VINDICATED** | noisy talent was the flaw; DARKO version works |
| Injury adjustment (RAPM / DARKO ×2) | worse on changed games | **VINDICATED-adjacent** | 'adjustment' double-counts; composition-with-OUT-sets is the correct injury model and it works |
| 3P-luck defence-only (M1 / `defonly`) | **NO-SHIP at D141 under V3.** 5-season pooled **+0.00047** i.i.d. CI(+0.00004,+0.00091) SIG, season-cluster CI(+0.00024,+0.00087) SIG, but **cluster-mean t at 4 dof CI(−0.00008,+0.00101) ns**, block bootstrap ns, month-cluster ns | **MED — the strongest NS-portfolio member** | Fails V2 T1 req.2 outright: point +0.00047 < its own MDE80 **0.00063** and < the 0.002 floor; fails **BH q=0.10** at K=107 on the clustered p (0.0378, rank 28, thr 0.02617) though it survives on the i.i.d. p (0.0155, rank 26, thr 0.02430). What PASSED: rolling-origin **4/4 positive no sign flip**, LOSO 5/5, **ERA-STABLE** (I²=5%, Q p=0.371), **calibration strictly improves** (ECE10 −0.00104, link slope 0.9725→0.9817), zero fitted parameters. **NOT IMPLEMENTED IN nbapred/** — `FF_LUCK=1` is the blunt variant, a different term (hall-of-shame 15). Needs a real implementation + the 2026-27 one-shot, NOT more analysis of the spent 6,148 games |
| Comp-heavy 60/40 blend (M2) | **HELD at D141 — not shipped, not gated.** Per-season +0.00147/+0.00112/+0.00164/+0.00062/**−0.00037**, OLS trend **−0.00042/season**, negative on the most recent season | MED | D102 RT4's independently-built fitted-weight version gives the same shape (+0.00106/+0.00163/+0.00223/+0.00012/**−0.00120**, trend −0.00060, **r=0.946**): two constructions of one channel agreeing on level, shape and sign flip. 5 points cannot separate decay from noise. **Resolution = the 2026-27 one-shot as a KILL rule** (n=1230 → MDE80 ~0.0024, so a season cannot CONFIRM +0.001): negative-or-flat retires M2 permanently; only ≥ +0.002 revives a ship conversation. May not be bundled into any joint gate as evidence until then |
| Recency-weighted ratings | 0.624 vs 0.601 | RETESTED 60d+rescaled-ridge: 0.5989 vs 0.6005, CI (−0.001,+0.004) NS → stays out; direction FLIPPED under proper construction, revisit with more seasons |
| Fitted FF/comp blend weight (D22) | **RE-TESTED at 5-season power (D102): +0.00077 CI(−0.00036,+0.00187) NS** | MED→LOW | original was fitted on ONE season (n~500) = textbook starvation; at n=6148 the walk-forward sign-constrained MLE is stable at w_ff 0.27–0.34 (likelihood wants ~30/70, not 50/50) so D22's DIRECTION was a small-sample artifact — but the per-season delta decays monotonically (+0.0011/+0.0016/+0.0022/+0.0001/−0.0012) and is NEGATIVE on the newest season. Fixed 50/50 retained |
| Event-recency window blend (F2/D52) | **RETIRED (D124): settled on the CURRENT certified stack — pooled +0.00002 CI(−0.00152,+0.00166) NS n=6,148; dev 23-26 +0.00061 NS, holdout 21-23 −0.00087 NS (negative point)** | — | point estimate walked +0.00138 → +0.00037 → −0.00013 (D102) → +0.00002 as power/control grew: a zero measured four ways. Only positive season is dev 2024-25 (+0.0029 NS) = season-lottery signature. Struck from the freeze list; control fidelity vs certified stack 2.5e-14 |
| Late-gated form term solo (F1/D71) | **RETIRED (D143). Re-gated SOLO against the D132 certified control after D90 was reverted: pooled −0.00000076, i.i.d. CI(−0.00090,+0.00085), season-cluster CI(−0.00099,+0.00099), cluster-mean t 4 dof CI(−0.00159,+0.00159), realized MDE80 0.00126.** Window either-gp≥55 (n=2,066) −0.0000023 ns. Earlier: D71 isolation +0.00178 CI(+0.00076,+0.00275) z=3.51; D102 RT2 on top of D90 −0.00012 NS | LOW | The D102 "D90 absorbs it" verdict is SUPERSEDED — D112/D118 reverted D90, so the re-gate was against a genuinely changed baseline (D45 rule). Construction identity proven: `fit_form_k` imported verbatim from `ba_windowed.py`, walk-forward k **bit-identical to RT2's on every shared refit date**, zero-outside-window 1.1e-16, estimator WARM on all 5 seasons (n_form 503→1054). The null is INFORMATIVE: the design's MDE80 (0.00126) is below D71's registered +0.00178. On D71's own dev seasons the term is +0.00052 (29% of registered), on the never-selected-on seasons −0.00078; rolling-origin 2/4, drift +0.00109/season = the "value only in the newest seasons" signature D139 used to demote D73. D71's own "form dominates, carry only one late-season term" lesson is REFUTED: form 0.0000 vs D73 tank +0.00199. Struck from the 2026-27 freeze list — nothing left to confirm |
| Season-phase adaptive memory / effective learning rate (D143) | **NO-SHIP, and the premise is measured FALSE.** Optimal exponential half-life by phase is **34/21/21/55/21 games on gp[10,20)/[20,30)/[30,41)/[41,55)/[55,82)** — FLAT at ~21, not falling; per-phase tuning buys ≤0.011 held-out RMSE over one global h=21. Past-vs-future 10-game persistence +0.575/+0.491/+0.487/+0.482/**+0.593** = flat-to-rising late. Endpoint: phase-varying (h=34/13/8) **+0.00023 CI(−0.00118,+0.00165)**; constant (h=21) **+0.00013 CI(−0.00119,+0.00150)**; **contrast B−C +0.00010 (MDE80 0.00082) pooled but −0.00019 at both-gp≥41 and −0.00046 at both-gp≥55** — varying the memory is worse exactly where it varies | LOW | The owner's observation is real and its mechanism is not: the fitted per-game recency premium does climb **1.09→1.37→1.66→2.20→2.74**, but a CONSTANT h=21 predicts 1.28/1.46/1.71/2.01/2.58 — the climb is a fixed memory meeting a lengthening history. Actionable residue: production runs at **h = INFINITY** in-season (`FourFactors.fit` accepts `half_life_days`; nothing passes it; D62 carry hard-stops at 200 rows), whose margin-level RMSE cost rises +0.0044 → **+0.1330** from early to gp≥55 — but the correction converts to only **+0.00013** of win-probability skill. Closes D48 ungated-constant #(3)/(5) for the FF memory channel with a number, and explains D18's old 60d null (60d ≈ 28 games ≈ near the optimum; the whole prize is ~0.0001). All three arms also FAIL the calibration veto (each moves the link slope AWAY from 1: 0.9725 → 0.9404/0.9518/0.9540) and BH q=0.10 |
| Opponent-pace (props) | NS | RETESTED w/ EB-shrunk pace: still worse (CI −0.037,−0.011) → CONFIRMED dead |
| Teammate-out lift (props) | worse on star-out | RETESTED minutes-only (×1.038): delta −0.015 CI (−0.044,+0.013) NS → stays out (much milder than rate-lift; revisit vs real prop lines) |
| Rest advantage (win-prob) | neutral | LOW | join verified; empirical B2B effect ~0 |
| Rest/B2B (props) | 0.997 ratio | LOW | empirically dead (n=3,905; 2025-26 only — sched features absent for 24-25) |
| Minutes UNCERTAINTY-only widening (props, D133 ARM B) | **SIG HARMFUL: dev Oct-Nov −0.00657 CI(−0.00731,−0.00582), holdout −0.00488, 5/5 seasons; −16.8% of the location fix; A−B = +0.04566 SIG** | **NONE** | pre-registered magnitude-matched ablation (spread widened by exactly the sd that would absorb ARM A's location correction, mean untouched). Settles that the props minutes defect is **LOCATION, not spread**. GOTCHA found in the process: `sd_min` is DEAD on the live path — `simulate_player` takes the empirical `minutes_hist` branch on 100% of the scored universe and never reads it, so the naive "widen sd_min" would have been a silent no-op |
| Two-axis (gp × availability) minutes ramp (props, D133 ARM C) | **not rejected — UNSHIPPED**: better on MINUTES (Oct-Nov minutes CRPS +0.07331 vs A's +0.04757, minutes MAE +0.13106 vs +0.07373) but NOT on points (A−C +0.00007 ns dev; holdout A +0.04227 vs C +0.03678) | LOW | passes its own gate; loses to the simpler single-axis arm on the pre-registered primary endpoint and needs a team-schedule query. Diagnosis: points need a per-minute EFFICIENCY correction the minutes fix does not supply. Next candidate, not a rejection |
| Kalman rates (props) | **RE-GATED CLEAN (D99): EWMA better by −0.90% rate-WMAE, CI[−0.00166,−0.00107], n=63,393, 3/3 seasons** | **NONE** | old "wash" was measured pre-D79 (contaminated universe + `predict(0)` no-op forward step + only the EWMA arm had `minutes_hist`); with all three fixed the rejection is clean and decisive. `player_rates_kalman` is dead code in production |
| EWMA-hl5 minutes | CRPS worse | LOW | chases within-role noise; conditioned eval fair |
| Empirical-minutes sampling | CI negative | LOW | retested after conditioning fix; still worse |
| Blowout throttle | neutral | LOW | real in minutes (−0.14 corr), too small for points |
| Overdispersion (game heat) | CRPS worse | LOW | binomial variance verified correct |
| def-RAPM team-aggregate (props) | **RE-GATED CLEAN (D99): +0.0011 CI(−0.0004,+0.0027) NS** | LOW | original fit had ALL shots attributed to the home five; **`def_rapm.py` FIXED 2026-08-01 (D107) — runtime-probed 50.3%/49.7% home/away**, and it gained `only_002`/`before` filters (defaults OFF so the D99 re-run stays bit-reproducible). The fix is worth +0.0016 CI(+0.0002,+0.0029) and moves ratings a lot (corr 0.55/0.44/0.54), yet the arm still does not beat the raw box allowance |
| Defender-aware (props) | **RE-GATED CLEAN (D99): +0.0037 CI(−0.0027,+0.0098) NS** on D58's own 2025-26 construction; pooled 3-season RAW +0.0080 PASS is **>half level-effect** (LEVEL arm, zero defensive info, +0.0043 CI(+0.0026,+0.0060)) and the pure-defense REL arm is NS pooled and 3/3 seasons | LOW (was "provisional") | D79 contamination owned none of it. NEW: hardcoded league constants {rim .613, mid .44, thr .359} are stale by −0.030..−0.038 logit — but a level knob would cancel the known D86 minutes over-projection, so it is a diagnostic, not a feature |
| Naive totals model | market +1pt MAE | LOW-MED | model was crude; but blend weight ~0 says no info |
| Betting model-vs-close | 29% hit | GENUINE | construction thrice-fixed; result robust |
| Talent ensemble (DARKO+EPM+BPM z-avg, D86) | pooled −0.00006 CI(−0.0014,+0.0013); gp<20 +0.0011 NS; mid-dist +0.0002 NS | LOW | pre-registered 1 config, 0 fitted params, same-run control; error rho-bar 0.874 predicted ~no headroom; swap was substantive (mean |Δcm| 1.04) so the null is informative — legs redundant with DARKO, best single leg (skill 0.374) diluted by equal weights |
| **Travel fatigue — great-circle km (D136 ARM A)** | **UNDERPOWERED-NS, not rejected: dev +0.00011 CI(−0.00028,+0.00050) n=3,690, MDE80 0.00055.** But SIG at the margin scale on the full 8,279-game frame: **−0.3088 pts/1,000 km CI(−0.5826,−0.0350)**, sign-correct in **98.7%** of the 75 walk-forward refits | **HIGH — this is a power limit, not a null effect** | The term is real and stationary by construction (BOS→LAX is 4,169 km in every season, so no D20/D70 drift is possible). It moves margin by rms **0.195 pts** against a pre-registered break-even of **1.485 pts**, and its observed endpoint delta sits essentially ON the theoretical value for a TRUE term of that size (+0.00007). An 80%-powered pass needs **88,526 games ≈ 72 NBA seasons**. ASYMMETRY: the cost is on the HOME team that just flew home (trav_h −0.4568 CI(−0.8120,−0.1016) SIG), NOT on the visitor (trav_a +0.1264 ns) — away travel is the routine state and adds nothing once b2b is controlled. Eligible retest: fresh corpus, or a MARGIN/spread endpoint where it is 2–7× better resolved. NOT re-runnable on these 6,148 games (second-look trap) |
| Circadian / signed timezone crossings (D136 ARM B) | **KILLED ON SIGN, not on p.** dev −0.00009 CI(−0.00068,+0.00054) ns; margin coefficient NEGATIVE in all 4 universes and SIG on dev (−0.5212, t=−2.33) — the **opposite** of the pre-registered prediction | LOW | The prereg committed to acute phase-at-tip (eastward travel HELPS: a 3-zone eastward trip puts a 19:30 ET tip at 16:30 body time, inside the circadian peak) over re-entrainment cost. The data leans the other way, but not stably: sign-correct in only **38.7%** of refits and the dev "significance" collapses on the holdout margin frame (−0.0233, t=−0.09). A significant wrong-signed term is overfitting, not physiology — killed by the pre-registered rule regardless of its p-value |
| Road-trip state — nth road game + homestand return (D136 ARM C) | dev **−0.00038** CI(−0.00116,+0.00034) ns; margins hret_h −0.4356 ns, rlen_extra_a +0.0513 ns, both sign-correct in 90.7% / 96.0% of refits | MED | Signs are right and stable; magnitudes are ~0.2–0.5 pts, i.e. below resolution (needs 51,134 games). Negative endpoint point estimate = adding a noisy coefficient of the right sign but over-estimated size. The mirrored columns (aret_a, rlen_extra_h) come back identically zero, confirming these regressors are structurally one-sided |
| Schedule density beyond b2b — 3-in-4 / 5-in-7 (D136 ARM D) | dev **−0.00020** CI(−0.00088,+0.00043) ns, and **SIG HARMFUL on 2024-25 alone** (−0.00133 CI(−0.00241,−0.00026)) while positive on the other two seasons | LOW | **It mostly RE-LABELS the already-shipped b2b terms — measured, not asserted:** the joint refit moves home-b2b **+0.3346** and away-b2b **−0.3363**, while arms A/B/C move them by ≤0.06 (corr(d3in4,hb2b)=+0.30, corr(d3in4,ab2b)=−0.34). d3in4 is sign-correct in **100%** of refits and SIG at the margin scale (−0.6144 CI(−1.2150,−0.0138)) — it is simply information D46's b2b already prices. Season-lottery signature per D101/D102 |
| Travel / density MARGIN coefficients (D136 readout) | **CLAIM CORRECTED at D140.** After the neutral-site fix: FULL frame dtrav_kkm −0.3061 t=−2.17 SIG (was −0.3088 t=−2.21), d3in4 −0.6144 t=−2.005 **bit-identical**; SCORED 2021-26 dtrav_kkm −0.2843 t=−1.72 ns, d3in4 −0.5827 t=−1.61 ns | — | D139's verdict ("both SIG terms are COVID-frame artefacts") STANDS, its implied MECHANISM does not. The 1,505.5 km bubble fiction moved dtrav_kkm by 0.9% and t by 0.04, and could not touch d3in4 at all (`is_3in4` is a pure date computation). **d3in4's** FULL-frame significance does depend on the bubble (drop the 88 games → t=−1.88 ns) but via its schedule DENSITY; **dtrav_kkm's** is a POWER story — the COVID seasons add +35% n at a larger same-signed estimate (−0.358 COVID-only vs −0.284 scorable). Honest phrasing: on the scorable era the travel coefficient is the same sign and order of magnitude but UNDERPOWERED, not manufactured. D136 shipped nothing then and ships nothing now |
| Travel/circadian/density PORTFOLIO ABCD (D136) | **T2 FAIL: dev −0.00066 CI(−0.00179,+0.00047) ns — the joint point estimate is NEGATIVE** (T2 needs CI_lo>0 AND point ≥0.002) | LOW | Combined rms(dm) 0.609 pts vs the 1.485-pt break-even; 20,857 games (17 seasons) for an 80%-powered pass. Holdout endpoint deliberately NEVER scored (pre-registered: once, only if dev passes) so 2021-22/2022-23 stay clean |
| **GameRotation role-transition minutes correction (props, D144 ARM R)** | **NO-SHIP, and NOT a rejection.** Passes the full V3 battery with **zero flags**: 5-season pooled **+0.03989 CI(+0.02009,+0.05810) SIG** on the pre-registered ROLE-ACTIVE window, block bootstrap SIG (temporal DEFF 1.04), RO **4/4 positive no sign flip**, LOSO 5/5, dev +0.03600 SIG / holdout +0.05119 SIG (**transfer 1.42**), **ERA-STABLE I²=0% Q p=0.450**, season-cluster CI(+0.02957,+0.05525) SIG, **cluster-mean t at 4 dof CI(+0.01666,+0.07363) SIG**, BH q=0.10 rank 20/110 SURVIVES | **HIGH — this is a veto failure and a construction defect, not a null** | Killed by its OWN pre-registered veto V1 (PIT on the primary stratum must move toward 0.5) which fails in **both** splits (dev .5014→.4967, holdout .4924→.4888) — the pooled PIT is a cancellation of two opposite-signed sub-strata and the veto was written on the wrong statistic, but softening it after seeing +0.036 is the selection the policy exists to prevent. **And the substantive reason:** the term is ASYMMETRIC — DEMOTED (43% of treated) **+0.11895 dev / +0.14600 holdout SIG**, PROMOTED (57%) **−0.02730 dev / −0.02681 holdout**, replicating to within 0.0005. Minutes improve in BOTH halves (+0.19482 CRPS active, +0.06532 pooled = **1.4× D133's entire pooled minutes gain**); points improve only on demotions. **This is D133 ARM C's failure mode on a new axis** and the diagnosis is the same: promotions need a per-minute EFFICIENCY correction, because a promoted player's rates were earned in a bench role. Demoted-only was NOT gated (second-look trap, and its own battery is ERA-CONDITIONAL I²=81%). **No code path was added to nbapred/** — per hall-of-shame 15, a default-OFF switch for a construction that failed its own veto is the hazard, not the remedy |
| **Role-transition read from MINUTES instead of the rotation sheet (D144 ARM M, adversarial control)** | **The control that proves the source matters:** +0.00198 dev / +0.00164 holdout against ARM R's +0.03600 / +0.05119; **R − M = +0.03402 CI(+0.01188,+0.05744) dev and +0.04954 CI(+0.01476,+0.08359) holdout, both SIG** | **NONE** | Same bucket machinery, last-game minutes vs the player's own trailing 20-game median, **no rotation input**. At the minutes level: minutes-only partition sees residuals −0.3039/+0.3345 where the rotation partition sees −2.2078/+2.3497, and **inside the minutes-STABLE cell the rotation flag still gives −2.5427/+2.5234** (n=2,475/1,903). **ROLE IS NOT VOLUME** — a newly promoted starter is not identifiable from his minutes history |
| **National-TV flag as a props term (D144)** | **NOT RUN, under V2 §5.5 — registered as structurally unreachable, NOT as a null** | — | The DNP suppression is real (−0.02189 within player-season, SIG, placebo-clean) but the props eval universe conditions on `seconds>=720`, i.e. on the player HAVING PLAYED, so a DNP-hazard term is outside the scored set. The only route to points CRPS is a minutes shift conditional on playing: FE **+0.1041 ns**, walk-forward **+0.0888 / −0.0063 / −0.0069** min — sign-unstable, ~0.007 min = 0.02% of proj_min, implying ~1e-4 CRPS against MDE80 ~0.011. §5.5 forbids running that and calling the NS a rejection. Becomes gateable the day an availability/slate endpoint exists |
| Defence-conditioned possession margin (D127) | **KILLED at the pre-registered dev gate: S1 (player off+def, D113 spec) −0.00108 CI(−0.00278,+0.00062) NS n=3,690, 0/3 seasons positive; offence-only and team-def arms SIG HARMFUL (−0.00225 / −0.00237)** | LOW-MED | possession RAPM swapped for half the comp slot reproduces DARKO-comp's ordering (corr 0.868, near-equal spread) plus noise; harm concentrates in the pre-registered disagreement window (S3 −0.0042 SIG). Defence-conditioning IS positive vs offence-only (+0.00116 NS, p_ws 0.085 — D113's game-level echo) but both arms sit below production. CI_hi +0.0006 excludes the claimed +0.001..+0.003 band (realized MDE80 0.00239). Holdout 21-23 never scored. Retry requires a construction artifact per V2 5.3; effects ≤ +0.0006 not excluded |
| **Absence-aware `trail_min` for the star-out detector (D149 ARM A, `STAROUT_TRAIL=played`)** | **NO-SHIP, and it is a DECISION not a null: the DNP dilution is LOAD-BEARING.** The arm does exactly what it was designed to do — recall 0.8121→0.9486, P(fire) by consecutive games missed goes from 0.864/0.790/0.750/0.7495/0.7947 to a FLAT 0.9525/0.9403/0.9426/0.9510/0.9632, firing team-dates +15.8% — and then LOSES points CRPS **−0.00441 CI[−0.00899,−0.00007] SIG** on its own pre-registered window and **−0.02355 CI[−0.03933,−0.00825] SIG** on the rows it newly fires, while gaining **+0.01900 SIG attempts LL on those same rows**. Season-clustered points CI(−0.00736,−0.00148) SIG NEGATIVE, rolling-origin 0/3. | **LOW** — the arm was implemented by the SHIPPED code under an env switch (no replica), the switch is proved a bitwise no-op at its default on 25,748 player cells, and the failure direction is predicted by D34/D35/D83 rather than surprising | The extra firings are LONG-ABSENCE stars, i.e. exactly the rows where the live EWMA baseline has fully re-equilibrated (D35 contamination at its maximum), so re-lifting there is the D83 double-count in pure form. The DNP zeros in `trail_min` act as an implicit absence prior that switches the term off precisely when the baseline no longer needs it. **Answers D146's registered open question: keep the current construction.** |
| **`null_u` (pool-arithmetic-only) star-out lift, replacing the fitted softmax (D150 ARM U, `STAROUT_USAGE=null_u`)** | **NO-SHIP — D129's registered hypothesis REFUTED out of sample.** Non-inferiority test with margins fixed in advance (points −0.002, attempts LL −0.005): points is a genuine tie (dev season-cluster +0.00010 CI(−0.00173,+0.00235) ns) but attempts LL is **−0.00667 CI(−0.00856,−0.00464) SIG WORSE**, cluster-mean t SIG, rolling-origin **0/3**, SIG worse in **5/5** seasons. **BH SURVIVES: p=0.01388, rank 26/113, thr 0.02301.** | **LOW** | D129 measured the opposite IN-SAMPLE, on one 9,315-row stratum, at FULL D33 magnitude. This gate is OOS, 27,950 rows, five seasons, at the shipped 0.16 residual scale, on the channel D33 was gated on. **`data/v2_usage.npz` stays; the player-specific weighting is load-bearing and the lift magnitude is not the whole signal.** Diagnostic: the module's documented trailing-attempt FALLBACK is SIG BETTER on attempts (+0.00304) and SIG WORSE on points (−0.00215) — the npz failure mode now has a measured cost |
| **Per-channel props ramp (D151 ARMS A/B, `PROPS_CHANNEL_RAMP`)** | **NO-SHIP, H3 REFUTED: both arms SIG HARM rebounds.** ARM A **−0.00263 CI[−0.00313,−0.00210]**, ARM B **−0.00347 CI[−0.00400,−0.00293]** on the 19,630 ramp-active rows; −0.00653/−0.00833 at delta≥2; SIG harmful in 4/5 seasons, positive in none; season-clustered SIG NEGATIVE, rolling-origin 0/2, ERA-UNSTABLE (I²=89-93%). Points and threes **bitwise unchanged** (max|dCRPS| = 0.0) so the veto passes by construction. | **LOW** — realized MDE80 0.00074 vs a pre-registered 0.00315, so this is a refutation and not a power failure | **The lam was fitted on a closed-form moment of a model that is never scored.** `simulate_player` draws minutes from the empirical histogram and CLIPS to [0,48]; that truncation already supplies the 0.031 rebounds the analytic moment thought were missing (simulated mean 4.7760 vs realized 4.7777 — already unbiased), so the correction over-shoots to 4.8065. D133's October rebound PIT overshoot is CONFIRMED (0.5209) and confirmed OCTOBER-LOCAL (whole window 0.4991) — real, but not reachable by a location knob. ARM B (dispersion-matched) is worse than ARM A everywhere: **D133's location-beats-spread replicates per channel.** Assists went the other way (MULTI-SPLIT PASS on dev, SIG positive in 4/4 non-COVID-fit seasons) and is deliberately NOT shipped — splitting the bundle post hoc is the §5.2 trap |

## Construction-artifact hall of shame (check these in every new test)
1. Frozen as-of state in walk-forward (rosters decay) — refit rolling.
2. fetchdf Timestamp vs fetchall date keys — str(d)[:10].
3. Eval/train universe conditioning mismatch (600s vs 720s).
4. Phantom rosters (ever-played vs recently-played).
5. Sign conventions (RAPM defense negation).
6. Silent .get() defaults hiding join failures.
7. Stride/order sampling changing the eval population.
8. **Trailing-corpus starvation** (D101): a 730-day/expanding trailing window
   is only as long as `nba_games` — check what the window ACTUALLY held at the
   earliest eval season before blaming a season. Quantifier:
   `scripts/ds_starvation_diag.py`. Also check for corpus-floor LITERALS
   (`season >= '2022-23'`) that were the corpus floor at the time of writing.
   **What they cost (D110 → D112):** `tanking.py` (4 queries) and
   `latestate.py` (`BURN_IN_SEASON`) both froze the 2022-23 floor into
   production. Effect: D73 and D90 were identically 0.0 on 2021-22 and COLD on
   2022-23 (tank k -0.23..-1.32 vs -2.0..-2.6 warm; late-state mean |term|
   0.15 pts vs 0.56) — so the two terms were **UNTESTABLE on the only genuine
   holdout the project has**, and the D110 audit had to leave both verdicts
   open. **Fix pattern (D112):** derive the floor from the data
   (`tanking.season_floor()` = earliest season in a contiguous run of ≥99%
   box-score-covered seasons) and expose an env override so the old floor
   stays reproducible for parity tests. Beware: relaxing a floor under an
   EXPANDING-window estimator is not a no-op on later seasons — it re-fits
   them too (see D112 for the measured size).
9. **Season-level verdicts from a single starved/odd season** (D101/D102/D98):
   per-season profiles are anti-correlated across seasons, so any pooled
   rejection whose sign is set by ONE season is suspect — but re-testing it
   with more data usually confirms the rejection rather than overturning it
   (4/4 in the D102 battery).
10. **A missing key that reads as a valid value.** `pbp['game'].get('homeTeamId')`
   returns `None` in 100% of cached games, so `x if team == home_id else y`
   silently collapses to one branch forever. Two symptoms, both measured in D99:
   *misattribution* (`defense_zone`/`possessions_v2`: 50% of lineups swapped)
   and *silent selection* (`fit_v2_usage`: `if pid in off5` then discards 49.9%
   of shots — the run looks healthy, it just halves n). Assert on the key, don't
   `.get()` it. Copies of the same lookup spread by copy-paste: fix them by grep,
   not by file.
11. **A "fix to design" is a hypothesis, not an improvement.** D79 restored the
   Kalman forward step from a `predict(0)` no-op to the intended behaviour and
   made the estimator measurably WORSE (−0.15%, SIG, 3/3 seasons). Same for the
   D79 002 filter on the incumbent EWMA (−0.36%). Gate hygiene fixes like any
   other change; correctness and accuracy are different claims.
12. **Level bias masquerading as a signal** (D88 pace, D99 zone defense). Any
   feature applied against a hardcoded league constant can pass by re-centering
   the model rather than by informing it. Always run a LEVEL arm (the
   cross-sectional mean of the adjustment, applied to everyone — carries zero
   information) alongside the real one; if LEVEL captures most of the gain, the
   feature is dead and you have found a calibration bug instead. Then check
   whether that calibration bug is itself cancelling a known upstream bias
   before "fixing" it.
13. **A stationary mechanism can still be unresolvable — say so with a number,
   before you score** (D136). Travel/circadian/road-trip terms are immune to the
   D20/D70 nonstationarity that killed team-identity home advantage, and their
   signs came back physiologically correct in 98.7%/96.0%/100% of walk-forward
   refits. They still returned a flat endpoint null, because at SCALE=7.2 the
   log-loss gain from a true margin term of rms `s` points is 0.5*E[p(1-p)]*
   s^2/SCALE^2 while its MDE80 is 2.802*sqrt(E[p(1-p)])*s/(SCALE*sqrt(n)) —
   break-even at n=3,690 is s=1.485 pts, and physiology only offers 0.2-0.6.
   Compute that break-even in the pre-registration. If the mechanism's plausible
   size sits below it, the honest deliverable is an UPPER BOUND plus the games
   needed (here: 72 seasons for travel), not a "rejection" — and the margin
   scale, where the same term IS resolvable, is the endpoint to report.
14. **"Incremental over the shipped term" must mean a JOINT REFIT, not a
   bolt-on** (D136). Schedule density looked like a new channel until the layer
   was refit jointly: it moved the already-shipped home-b2b coefficient +0.335
   and away-b2b -0.336 and returned nothing at the endpoint. Had it been applied
   on top of frozen b2b coefficients it would have double-counted the same
   nights twice. Keep the incumbent regressors in the design matrix AND let the
   fit re-partition, then report how far the incumbent coefficients moved — that
   shift is the diagnostic that tells you whether you found a channel or a
   synonym.

15. **A SWITCH NAMED AFTER A HYPOTHESIS IS NOT EVIDENCE THAT IT IMPLEMENTS IT**
   (D141). `FF_LUCK=1` reads like "the 3P-luck term" and was cited in a ship
   directive as "M1 is already implemented". It is not: `FF_LUCK` wires the
   BLUNT both-sides variant (`four_factors.py:81-92`, a REGISTERED LOSER that
   made 2023-24/2024-25 worse by +0.0024/+0.0035), while the gated M1 is the
   DEFENSE-ONLY hybrid that exists only in `scripts/scratch_nsport_joint.py`
   and has NO code path in `nbapred/` at all. Shipping by flipping the default
   would have shipped a different feature carrying none of the evidence.
   **Rule: before shipping "the term that already passed", diff the CODE THAT
   RAN IN THE GATE against the code the switch reaches — the D45/D134 same-run
   control discipline applied to IDENTITY, not just to numbers.** One-line
   check: `grep -rn <SWITCH> nbapred/` and confirm the callee is the gate's
   construction, not a namesake.
16. **A CLUSTERED SE THAT SHRINKS IS NOT A STRONGER RESULT** (D141, extending
   GATE_POLICY_V2 §9.3). At K=5 a season-cluster bootstrap can return an SE
   BELOW the i.i.d. one when the intra-season ICC comes back NEGATIVE (M1:
   ICC −0.00020, DEFF 0.755, `p_wrongside` 0.0000 on 2,000 draws). That is the
   D130 ARM A pathology in mirror image and it is a small-K artifact, not
   evidence. **Rule: whenever DEFF < 1, quote the cluster-mean t interval at
   K−1 dof as THE result** (M1: +0.00047 CI(−0.00008,+0.00101) ns, t=2.386 vs
   t_crit 2.776) and treat the shrunken bootstrap p as unusable — including
   when it feeds a BH family correction, where it flips the verdict.
17. **A VETO EVALUATED ON A POOLED STRATUM CAN BE A CANCELLATION** (D144). The
   D144 role arm pre-registered "PIT on the ROLE-ACTIVE stratum must move
   TOWARD 0.5". The control's pooled PIT was 0.5014 — apparently perfect — but
   that number is the average of **+0.0388 (PROMOTED) and −0.0476 (DEMOTED)**,
   two sub-strata mis-calibrated hard in OPPOSITE directions. The arm improved
   both by 4× and 9× and the pooled statistic still moved AWAY from 0.5, so the
   veto failed on a correct treatment. **Rule: when a term is applied with
   OPPOSITE SIGNS to disjoint sub-strata, every calibration veto must be
   registered PER STRATUM. A pooled calibration statistic over a
   sign-heterogeneous treatment measures the cancellation, not the
   calibration** — in either direction: it can pass a term that breaks both
   halves symmetrically just as easily as it can fail one that fixes both.
   Corollary, and the reason D144 still did not ship: this is a reason to write
   the NEXT veto correctly, never a licence to reinterpret the one already
   registered.
18. **MINUTES ARE NOT ROLE, AND A MINUTES WIN IS NOT A POINTS WIN** (D144,
   confirming D133 ARM C). Two independent props arms have now bought large
   MINUTES gains that failed to convert on points wherever per-minute
   efficiency changes with the new role: D133 ARM C (minutes CRPS +0.07331 vs
   the shipped +0.04757, points A−C +0.00007 ns) and D144 ARM R (minutes CRPS
   +0.19482 SIG on active rows, yet points **−0.02730** on the PROMOTED half,
   replicated at −0.02681 on the holdout). **Rule: any arm that moves
   `proj_min` for a REASON — role change, availability, load — must
   pre-register whether the player's per-minute RATE vector is still valid at
   the new minutes level, and gate points as primary. Scaling bench-earned
   rates to starter minutes over-projects points by construction.** The
   binding constraint in props is now the minutes→points bridge, not minutes.
17. **A FIX KEYED ON A PROXY LEAVES THE AXIS UNCORRECTED WHEREVER THE PROXY IS
   INERT** (D145, generalising D133). D133 correctly diagnosed the mechanism as
   ABSENCE and then shipped a correction keyed on GAMES-PLAYED, forced to zero
   at gp>=20 on an estimator-memory argument. gp is a proxy: early in a season
   everyone has low gp, so b(gp) absorbs the absence of a whole universe — but
   at gp>=20, **65% of the props universe, where the shipped term is identically
   zero, the absence bias is untouched and still +0.83 / +2.97 minutes at
   miss10 5-7 / 8-10.** Correcting the axis directly there is worth +0.05755
   points CRPS SIG, on rows the original term could not by construction have
   selected on. **Rule: when a diagnosis names mechanism M and the fix is keyed
   on proxy P, measure the residual on M INSIDE THE REGION WHERE P IS INERT —
   that region is both the cleanest test of the diagnosis and, if the diagnosis
   was right, unclaimed value.** Corollary: the inert region is a
   selection-clean holdout that costs nothing to construct.
18. **AN ESTIMATOR BUG AND A SHIPPABLE FIX ARE DIFFERENT CLAIMS; SIZE THE
   ENDPOINT BEFORE YOU GATE** (D145). `composition.trail_min` carries the SAME
   absence blindness as D133 and a LARGER bias in its own units (+6.05 min at
   gp 1-2 mid-season vs D133's +3.01), and the comp leg is half of every
   production margin — every heuristic said "gate it". It is nonetheless
   ungateable: the exact additive-margin identity lets the footprint be computed
   from `p` alone, and **best-case dLogLoss / MDE80 <= 0.80 in every window, at
   every window length, INCLUDING a perfect-minutes oracle.** A pass is
   structurally impossible, so running it would have manufactured a "null" that
   was really a power failure. **Rule: for any additive-margin change, compute
   `0.5*E[d^2 p(1-p)]` against MDE80 BEFORE pre-registering; if the ratio is
   below 1 the gate cannot pass even if the hypothesis is perfectly true, and
   §5.5 forbids running it.** This is the cheap version of the D136/D141 lesson
   and it costs one afternoon of arithmetic instead of one spent corpus.
19. **A COEFFICIENT FITTED ON A CLOSED-FORM MOMENT OF A MODEL YOU DO NOT SCORE
   IS NOT A CORRECTION TO THE MODEL YOU DO SCORE** (D151). The per-channel
   ramp solved `E[rate*(m0 - lam*D)] = E[y]` on the ANALYTIC prediction
   `rate * proj_min` and got a beautifully stable coefficient (lam_reb < 1 in
   5/5 walk-forward cutoffs). But the shipped generative model draws minutes
   from an empirical histogram and CLIPS them at [0,48]; the truncation
   already supplied the 0.031 rebounds the moment said were missing, so the
   simulated mean was ALREADY unbiased (4.7760 vs realized 4.7777) and the
   correction over-shot to 4.8065 for **−0.00263 CRPS SIG**. **Rule: fit any
   correction on the GENERATIVE output the gate will score — simulate, then
   take the moment — or gate the analytic model instead. Check the two agree
   before you spend an arm; it is one line.** This is D141's
   implementation-identity lesson moved down into the ESTIMAND.
20. **A DETECTOR BUG CAN BE LOAD-BEARING; FIX THE DETECTOR AND MEASURE THE
   ENDPOINT, NEVER JUST THE DETECTOR** (D149). `starout.team_context` averaged
   DNP zeros into `trail_min`, so a star de-starred himself while absent
   (P(fire) 0.864 → 0.7495 by 4-5 games missed, recall 0.789/0.812).
   Conditioning on played games only fixes the detector completely — P(fire)
   goes FLAT at ~0.95 — and **costs −0.024 points CRPS on exactly the rows it
   newly fires**, because those rows are long-absence stars whose
   redistribution the trailing baseline has already absorbed (D35/D83). The
   "bug" was functioning as an implicit absence prior. **Rule: when a guard
   looks wrong, ask what it is accidentally suppressing before you remove it,
   and gate the removal on the endpoint the term serves.**
