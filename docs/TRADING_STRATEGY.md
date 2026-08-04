# TRADING STRATEGY

This document has two parts.

* **PART I — a bet-at-close strategy designed around our losses.** Built,
  evaluated, **rejected for capital**. Sections 1-7.
* **PART II — the 2026-27 F4 PAPER-TRADE REGISTRY (SHIPPED).** The rules,
  the upper confidence-excess cap, the edge-shrinkage sizing, the
  pre-registration statement, the 4-season re-sim, and the October rule card.
  Sections 8-14. This is the live specification `scripts/bet_engine.py`
  implements.

---

# PART I — a bet-at-close strategy designed around our losses

**Status: BUILT, EVALUATED, AND REJECTED FOR CAPITAL.** Zero of eight
pre-registered configurations holds ROI at vig in both halves of the corpus.
The mechanical selection protocol, run in both directions, selects a
strategy that loses out-of-sample both times. Details, and the one genuinely
positive finding that survives, are below.

Scripts: `scripts/ts_strategy.py` (the strategy + evaluation),
`scripts/ts_openers.py` (the opening-line investigation).
Artifacts: `data/ts_strategy.json`, `data/ts_strategy_bets.csv`,
`data/ts_openers.json`, `data/ts_frame.csv`.

---

## 1. Information discipline (the rule this whole document obeys)

**The MODEL is market-blind.** `p_us` comes out of `nbapred/` having never
seen a price. That is ground rule G2 and it is untouched here.

**The STRATEGY is not market-blind, and must not be.** A strategy that
cannot see the price it is transacting against cannot compute an edge.
Comparing our probability to the price we are about to trade *is the job*.
That is not lookahead.

What *would* be lookahead is using any price knowable only **after** the
decision. Nothing here does that.

Our historical price series is the **CLOSE**
(`odds_market.p_home_spread` — a de-vigged closing-spread probability,
`p = sigmoid(home_expected_margin / 6.96)`). So the honest formulation is a
**BET-AT-CLOSE** strategy: it transacts at the close and uses *that* price as
its decision input.

This is deliberately the **hardest possible test**. The close is the sharpest
price of the day. Any edge that survives at the close is a *lower bound* on
the edge available earlier. Section 2 measures exactly how much this
formulation costs us.

---

## 2. Are opening lines obtainable? (`scripts/ts_openers.py`)

**Short answer: not for any season that matters.**

| source | open? | coverage |
|---|---|---|
| `odds_hist_sbr` (DuckDB) | **YES** — `spread_open` + `spread_close` | 2007-08 … **2022-23 (664 of 1230 games)**, then nothing |
| `odds_market` (DuckDB) | no | 2007-08 … 2025-26, CLOSE only — this is the `p_mkt` every backtest uses |
| kaggle `cviaxmiwnptr` | no | 2008-2026, one spread per game (verified = the SBR **close**) |
| kaggle `ehallmar` | no | per-book spreads/ML, no open/close flag, ends ~2018 |
| kaggle `christophertreasure` | no | one price per game |
| kaggle `erichqiu` | no | 2012-13 … 2018-19 |

The `data/raw/sbr/` "HTML-not-xlsx failures" named in the directive were
checked directly: `nba-odds-2008-09.html` (1315 games), `nba-odds-2009-10.html`
(1312), `nba-odds-2022-23.html` (664) all parse fine with `pd.read_html`, and
**all three are already in `odds_hist_sbr` at exactly those row counts.**
There is nothing to recover — the 2022-23 archive genuinely stops at 664 games
because SBR stopped publishing mid-season.

Moneylines tell the same story: `odds_market.ml_home` is populated through
2022-23 and is **NULL for all of 2023-24, 2024-25, 2025-26**. For the OOS
window we have exactly one number per game: the closing spread.

**Consequence:** bet-at-open is **untestable out-of-sample in this repo**.
It cannot be evaluated on a single OOS season. Bet-at-close is not a
conservative choice, it is the only available one.

> **Independent confirmation (concurrent thread).** While this was being
> built, another thread produced `scripts/build_odds_open.py` /
> `scripts/measure_line_movement.py` (`odds_open` table, `docs/OPENING_LINES.md`)
> and reached the same conclusion from the same source: *"sportsbookreviewsonline.com
> stopped publishing mid-2022-23, so there is NO opening line for 2023-24
> onward."* It further notes that SBR's ML column is the **closing** moneyline
> only — there is no opening ML in the source at all. Two independent audits,
> same verdict. Where that thread's numbers differ from §2.1 below, prefer its
> table for *coverage* questions (it builds a persistent `odds_open` on our
> team keys) and this one for the *CLV* result (it carries the
> favourite-drift control).

### 2.1 What the close costs us — and the one strong positive result

On the window where openers *do* exist (2021-22 + 2022-23 partial, n=1,892):

| statistic | value |
|---|---|
| log-loss of the OPEN | 0.62615 |
| log-loss of the CLOSE | 0.61702 |
| **open → close sharpening** | **+0.00913 nats/game** (close is sharper, as expected) |
| log-loss of our model | 0.62965 (worse than both) |
| our mean edge vs the OPEN | **-0.0079** |
| our mean edge vs the CLOSE | **-0.0192** |
| **CLV of betting our side at the OPEN** | **+0.01124 probability, t = +6.71**, 52.6% positive |
| CLV of betting the OPEN's own favourite (control) | **-0.00362, t = -2.14**, 47.6% positive |

Read the last two rows carefully, because this is the most valuable number in
the whole exercise. The obvious objection to any CLV claim is that lines drift
toward favourites and a model that backs the favourite 87% of the time
harvests that drift for free. **It is not free here — the drift runs the other
way.** Backing the open's favourite *loses* 0.36 points of CLV; backing our
side *gains* 1.12 points at t=+6.7. The closing line moves toward us.

Our model carries real information that the market has not yet priced at the
open and *has* priced by the close. It is a **market-blind model that beats
the opener and loses to the closer**, and we are structurally forced to
transact at the closer.

---

## 3. The strategy specification

Built directly from the loss forensics, not from a parameter search.

### Design inputs (given, not rediscovered)

1. Net deficit is a **thin uniform bleed** across every confidence bucket
   (+4…+10 nats each) **plus a fat tail**: worst 1% of games = 68% of the net
   deficit.
   → There is no "safe" confidence bucket. The only structural escape is to
   **remove the tail**. The tail veto is therefore the centrepiece.
2. Well calibrated at high confidence (85%+: predict 89.3%, hit 88.3%);
   useless at low (50-55%: predict 52.4%, hit 52.9%, market 56.8%).
   → Trade only where `|p_us − 0.5|` is large.
3. Confidence **above** the market's: we lose 0.0150/gm. **Below**: 0.0064/gm.
   → This one is uncomfortable. The textbook bet — back the side where our
   probability beats the price — lands in the *first* regime whenever we back
   the shared favourite, i.e. **exactly where we are worst**. The second
   regime is reached by backing the shared **dog**. So direction is a
   pre-registered **arm**, not an assumption: FAV vs DOG. The forensics
   predict DOG > FAV; D77/D82 predict the opposite ("fade favourites
   contraindicated", "pro-shaded-side structure"). The data arbitrates and
   the multiplicity is paid for in the family-wise noise test.
4. Late-season window was OOS-positive at vig in 3 consecutive sims;
   T20.D03-10 base (n=63) was −0.65% at vig / +3.83% fair.
   → Late window carried as a pre-registered overlay.

### Layer A — ELIGIBILITY (which games we consider at all)

| leg | rule | rationale | fires on |
|---|---|---|---|
| A1 confidence tier | `\|p_us − 0.5\| ≥ 0.20` | input (2) | 38.1% |
| A2 shared side | `(p_us−0.5)(p_mkt−0.5) > 0` | opposite-side games are known net-negative (D78) | 87% |
| **A3 tail veto** | veto if **any** leg below | input (1) — the fat-tail control | union 45% |
| — EARLY | `min(gp_home, gp_away) < 20` | D84-A early regime; week-1 comp leg is a literal dead zero | 25.1% |
| — CHAOS | `max(m5_abs_h, m5_abs_a) ≥ 18.0` | top quintile of trailing-5 mean \|margin\| — the "chaos team" proxy | 19.0% |
| — FRESH | `max_team #inactives with ≥24 min trailing-10 who played that team's PREVIOUS game ≥ 2` | multi-player *unmodelled* availability shock — the "post-event window" proxy | 6.9% |
| A4 late window | `max(gp_home, gp_away) ≥ 55` | input (4) | optional arm |

All three veto thresholds were fixed **a priori from covariate quantiles
only**. No outcome, price, or PnL was consulted to choose them. `FRESH`
deliberately requires the absent player to have played *the previous game* —
a long-term absence is already in the model's state, only a fresh one is an
uncertainty shock.

### Layer B — EDGE TEST vs the transacted price, including vig

Offered price, proportional overround `V = 1.045` (the D75/D78 convention):

```
q_side = p_mkt_side · V        dec = max(1/q_side, 1.01)
```

- **TEST:** `p_us_side > q_side` — strictly positive EV at the price we
  actually transact, equivalently Kelly `f* > 0`. This is side-asymmetric by
  construction and correctly so: the vig hurdle is `p_mkt_side·(V−1)`, worth
  3.4 points of probability on a 0.75 favourite but only 1.1 on a 0.25 dog.
- **CAP:** `p_us_side − p_mkt_side ≤ 0.10` — the adverse-selection cap. Large
  divergence means the market knows something structural (D13/H10), and input
  (3) says so directly.

### Layer C — SIZING

```
f*         = (p_us_side · dec − 1)/(dec − 1)      Kelly on the VIGGED odds
stake_frac = min(0.25 · f*, 0.02)                 quarter-Kelly, HARD 2% cap
day cap    = total staked on one date ≤ 6% of bankroll, pro-rata scaled
```

Bankroll compounds; bets settle end of day. The per-bet cap and the day cap
are the two sizing-level tail controls (same-day bets are the correlated-loss
channel); the A3 veto is the eligibility-level one. Flat 1u is reported
alongside for comparability with D75/D78.

### Layer D — BANKROLL SIM

Equity curve, max drawdown as a fraction of running peak, per-season
decomposition, bootstrap CIs, noise-compatibility.

### The pre-registered family (8 configs)

`{FAV, DOG} × {base, veto} × {all-season, late}` — fixed before scoring.

---

## 4. Does the filter even work? The structural diagnostic

Betting ROI on n≈50-200 bets is nearly powerless. Log-loss on n=4,920 games is
not. So before asking "does it make money", ask the question that actually has
power: **does eligibility + veto produce a stratum where our deficit vs the
close is gone?**

| stratum | n | net/gm (nats) | share of total deficit |
|---|---|---|---|
| ALL games | 4920 | **+9.29** | 100.0% |
| worst 1% of games | 49 | +630.79 | **67.6%** |
| A1+A2 eligible | 1945 | +9.75 | 41.5% |
| **A1+A2 + tail VETO** | **1016** | **+7.76** | 17.2% |
| vetoed only | 929 | +11.92 | 24.2% |
| conf_us > conf_mkt | 1650 | +15.04 | 54.3% |
| conf_us < conf_mkt | 3270 | +6.39 | 45.7% |

`net/gm > 0` means **we are worse than the de-vigged close** on that stratum.

Three things to note.

1. **The design inputs reproduce exactly.** Worst 1% = 67.6% of the deficit
   (given: 68%). conf_us>conf_mkt = +0.01504 (given: 0.0150). conf_us<conf_mkt
   = +0.00639 (given: 0.0064). The forensics are confirmed on this frame.
2. **The veto works — partially.** It cuts the eligible stratum's deficit from
   +9.75 to +7.76 nats/game, and the games it removes are worse (+11.92) than
   the ones it keeps. The tail control is doing real work.
3. **It is not nearly enough.** +7.76 nats/game is still a deficit. **There is
   no stratum in this table with `net/gm ≤ 0`.** We do not beat the de-vigged
   close anywhere we can define in advance — and a bet-at-close strategy can
   only make money where we beat the de-vigged close by more than the vig.

That is the answer, and everything in section 5 is confirmation of it.

---

## 5. Evaluation

IS = **2022-23 + 2023-24** (the less-developed seasons), OOS = **2024-25 +
2025-26** — deliberately **reversed** from the earlier sims, which selected on
2023-24/2024-25 and validated on 2025-26. Putting the development seasons in
the holdout is the harder arrangement. Both directions are reported.

2021-22 has no `game_inactives` coverage, so the FRESH veto leg cannot be
built there; veto configs are shown blank rather than silently run on a
degraded filter.

### 5.1 Direction 1 — IS 2022-23 + 2023-24 → OOS 2024-25 + 2025-26

**IS** (2,460 games), quarter-Kelly / flat:

| config | n | hit% | ROI% | ROI fair% | CI95 ROI% | maxDD% | P(noise) |
|---|---|---|---|---|---|---|---|
| FAV.base | 151 | 76.8 | −0.27 | +4.22 | [−10.18, +9.24] | 9.6 | 0.523 |
| FAV.base.late | 49 | 85.7 | +7.11 | +11.93 | [−7.71, +20.54] | 4.3 | 0.197 |
| **FAV.veto** | 73 | 83.6 | **+11.30** | +16.31 | [−0.82, +22.24] | 4.6 | **0.051** |
| **FAV.veto.late** | 38 | 89.5 | **+12.91** | +17.99 | [−2.49, +25.64] | 3.0 | 0.075 |
| DOG.base | 366 | 18.0 | +9.83 | +14.77 | [−22.50, +45.23] | 33.0 | 0.245 |
| **DOG.base.late** | 145 | 19.3 | **+23.60** | +29.16 | [−29.67, +86.76] | 21.2 | 0.167 |
| DOG.veto | 225 | 14.7 | −10.00 | −5.94 | [−45.72, +31.64] | 43.9 | 0.692 |
| DOG.veto.late | 109 | 14.7 | −23.17 | −19.72 | [−69.26, +38.61] | 36.4 | 0.791 |

Family-wise: best fair ROI +29.16%, **P(max ≥ that | market is truth) = 0.143**.
So the best IS result is *not even in-sample significant* once the 8-config
family is paid for. That alone should have set expectations.

**OOS** (2,460 games):

| config | n | hit% | ROI% | ROI fair% | CI95 ROI% | maxDD% | P(noise) |
|---|---|---|---|---|---|---|---|
| FAV.base | 206 | 75.2 | −6.16 | −1.94 | [−14.52, +2.31] | 26.8 | 0.932 |
| FAV.base.late | 85 | 77.6 | −2.20 | +2.20 | [−13.79, +9.27] | 12.6 | 0.648 |
| FAV.veto | 93 | 74.2 | −7.54 | −3.38 | [−20.30, +5.39] | 15.2 | 0.890 |
| FAV.veto.late | 56 | 76.8 | −4.08 | +0.24 | [−19.76, +10.61] | 9.0 | 0.716 |
| DOG.base | 464 | 17.0 | −7.01 | −2.82 | [−30.47, +18.65] | 56.0 | 0.691 |
| DOG.base.late | 165 | 10.3 | −35.22 | −32.31 | [−72.60, +8.95] | 38.4 | 0.937 |
| DOG.veto | 222 | 18.5 | +3.90 | +8.57 | [−31.59, +45.33] | 29.1 | 0.399 |
| DOG.veto.late | 93 | 9.7 | −25.66 | −22.31 | [−78.99, +48.17] | 20.7 | 0.791 |

**7 of 8 negative at vig.** Every IS winner reverses.

### 5.2 Direction 2 — reversed (IS 2024-25 + 2025-26 → OOS 2022-23 + 2023-24)

Same numbers with the labels swapped (half B is now IS, half A is now OOS).
Family-wise IS: best fair ROI +8.57%, **P = 0.626** — indistinguishable from
noise even in-sample.

### 5.3 The selection protocol (the decisive test)

Mechanical, no discretion: pick the single best config by IS ROI at vig among
those with IS n ≥ 30, then report its OOS.

| | selected | IS ROI | OOS n | OOS hit% | OOS ROI | OOS fair | CI95 | maxDD | P(noise) | swing |
|---|---|---|---|---|---|---|---|---|---|---|
| **Dir 1** | DOG.base.late | +23.60% | 165 | 10.3 | **−35.22%** | −32.31% | [−72.60,+8.95] | 38.4% | 0.937 | **−58.8 pts** |
| **Dir 2** | DOG.veto | +3.90% | 225 | 14.7 | **−10.00%** | −5.94% | [−45.72,+31.64] | 43.9% | 0.692 | −13.9 pts |

**In both directions the protocol selects a strategy that loses OOS.** Note
*what* it selects: the DOG arm both times. The dog arm has the highest
variance (longshots at dec ≈ 5), so it wins the in-sample maximum lottery and
then reverts. This is selection-on-variance in its purest textbook form, and
it is exactly the failure mode the reversed-split design was built to expose.

### 5.4 Per-season (quarter-Kelly, vig 4.5%) — n / ROI% / maxDD%

| config | 2021-22 | 2022-23 | 2023-24 | 2024-25 | 2025-26 |
|---|---|---|---|---|---|
| FAV.base | 63 / −4.81 / 5.3 | 51 / +6.91 / 5.2 | 100 / −3.55 / 9.6 | 113 / −11.08 / 21.4 | 93 / +0.51 / 11.1 |
| FAV.base.late | 15 / −3.04 / 2.4 | 14 / +32.68 / 0.0 | 35 / −1.50 / 4.3 | 46 / −6.95 / 7.7 | 39 / +3.51 / 7.4 |
| FAV.veto | *(no inactives)* | 22 / +12.16 / 2.7 | 51 / +11.02 / 4.6 | 49 / −4.57 / 5.9 | 44 / −11.10 / 11.4 |
| FAV.veto.late | *(no inactives)* | 10 / +35.10 / 0.0 | 28 / +6.59 / 3.0 | 33 / −5.72 / 5.9 | 23 / −1.62 / 5.9 |
| DOG.base | 184 / −3.92 / 30.6 | 169 / +29.10 / 17.7 | 197 / −6.50 / 23.5 | 208 / +5.59 / 32.1 | 256 / −20.75 / 43.0 |
| DOG.base.late | 74 / −16.73 / 22.1 | 64 / +48.73 / 12.3 | 81 / +8.29 / 21.2 | 69 / −44.38 / 32.1 | 96 / −27.06 / 20.3 |
| DOG.veto | *(no inactives)* | 111 / +12.71 / 16.0 | 114 / −38.80 / 33.2 | 98 / +33.92 / 17.9 | 124 / −17.69 / 29.1 |
| DOG.veto.late | *(no inactives)* | 50 / +4.17 / 16.0 | 59 / −53.26 / 26.0 | 38 / −11.20 / 17.7 | 55 / −35.96 / 16.4 |

2021-22 is an extra never-selected-on season (newly scorable per D101).
`FAV.base` and `FAV.base.late` are both **negative** there.

### 5.5 Full sample (4 seasons), break-even vig, sizing sensitivity

| config | n | hit% | ROI@4.5% | @3.0% | @2.0% | @fair | break-even V | maxDD% | bankroll 100 → |
|---|---|---|---|---|---|---|---|---|---|
| FAV.base | 357 | 75.9 | −3.53 | −1.26 | −0.29 | +0.81 | 1.0159 | 28.5 | 81.9 |
| FAV.base.late | 134 | 80.6 | +0.97 | +2.79 | +4.10 | +6.16 | 1.0509 | 13.2 | 102.1 |
| FAV.veto | 166 | 78.3 | +0.44 | +1.95 | +3.02 | +4.97 | 1.0477 | 15.2 | 101.2 |
| **FAV.veto.late** | 94 | 81.9 | **+2.11** | +4.38 | +5.74 | +7.76 | **1.0852** | **9.0** | 103.4 |
| DOG.base | 830 | 17.5 | +0.48 | +0.93 | +1.28 | +2.12 | 1.0530 | 56.0 | 105.4 |
| DOG.base.late | 310 | 14.5 | −2.46 | −2.30 | −2.19 | −1.92 | never (<fair) | 51.5 | 91.0 |
| DOG.veto | 447 | 16.6 | −3.36 | −2.96 | −2.61 | −1.70 | never (<fair) | 46.4 | 85.5 |
| DOG.veto.late | 202 | 12.4 | −24.10 | −23.58 | −23.21 | −22.29 | never (<fair) | 48.8 | 62.3 |

Sizing sensitivity on the two best FAV configs (full sample) — the result is
**stable across sizing**, so it is not a Kelly artifact (this is the D75
"Kelly-consistency failure" signature, and it is absent here):

| sizing | FAV.veto ROI% / DD% | FAV.veto.late ROI% / DD% |
|---|---|---|
| flat 1u | +0.83 / 8.2 | +3.19 / 5.1 |
| quarter-Kelly cap 2% | +0.44 / 15.2 | +2.11 / 9.0 |
| eighth-Kelly cap 2% | −1.12 / 13.6 | +1.63 / 6.9 |
| half-Kelly cap 2% | +0.24 / 16.3 | +2.56 / 9.9 |
| quarter-Kelly cap 1% | +0.35 / 8.5 | +2.65 / 5.1 |
| quarter-Kelly **no cap** | −0.24 / **25.8** | +2.00 / **13.6** |

The caps earn their keep on drawdown: removing them roughly doubles max DD
for no ROI gain. The tail controls work *as risk controls* even though they do
not manufacture an edge.

---

## 6. Verdict

### 6.1 Does any configuration hold ROI out of sample?

**No.** Operationalised without discretion — a config "holds" only if its ROI
at vig is positive in **both** halves of the corpus (one positive half is what
a coin does half the time):

| config | n_A | ROI_A% | n_B | ROI_B% | both > 0? | pooled% | pooled fair% | maxDD% |
|---|---|---|---|---|---|---|---|---|
| FAV.base | 151 | −0.27 | 206 | −6.16 | no | −3.53 | +0.81 | 28.5 |
| FAV.base.late | 49 | +7.11 | 85 | −2.20 | no | +0.97 | +5.52 | 13.2 |
| FAV.veto | 73 | +11.30 | 93 | −7.54 | no | +0.44 | +4.96 | 15.2 |
| FAV.veto.late | 38 | +12.91 | 56 | −4.08 | no | +2.11 | +6.70 | 9.0 |
| DOG.base | 366 | +9.83 | 464 | −7.01 | no | +0.48 | +5.00 | 56.0 |
| DOG.base.late | 145 | +23.60 | 165 | −35.22 | no | −2.46 | +1.93 | 51.5 |
| DOG.veto | 225 | −10.00 | 222 | +3.90 | no | −3.36 | +0.98 | 46.4 |
| DOG.veto.late | 109 | −23.17 | 93 | −25.66 | no | −24.10 | −20.69 | 48.8 |

**Configs positive at vig in both halves: 0 of 8.**

Full-sample family-wise noise test: best fair ROI +6.70%,
**P(max ≥ that | the de-vigged close is the truth) = 0.592.** The entire
four-season, eight-config result set is what a market-efficient world produces
**59% of the time**.

### 6.2 The ROI series is not tracking the edge — proof

If these ROIs were an edge, our structurally best seasons would be our best
betting seasons. They are the opposite:

| season | net/gm on the A1+A2+VETO stratum (lower = better) | FAV.base | FAV.base.late | FAV.veto | FAV.veto.late |
|---|---|---|---|---|---|
| 2022-23 | +0.00949 | +6.91 | +32.68 | +12.16 | +35.10 |
| 2023-24 | **+0.01471** (worst) | −3.55 | −1.50 | **+11.02** | +6.59 |
| 2024-25 | **+0.00073** (best — near parity!) | −11.08 | −6.95 | **−4.57** | −5.72 |
| 2025-26 | +0.00570 | +0.51 | +3.51 | −11.10 | −1.62 |

`corr(structural deficit, ROI)` per config: FAV.base +0.474, FAV.base.late
+0.289, **FAV.veto +0.749**, FAV.veto.late +0.470, DOG.base +0.025,
DOG.base.late +0.695, DOG.veto −0.790, DOG.veto.late −0.486. **Mean +0.178.**

A real edge would give a strongly *negative* correlation. Instead 2024-25 —
the one season where we reach effectively market parity on the eligible
stratum (+0.0007 nats/game) — is a **losing** betting season for all four FAV
configs, while 2023-24, our *worst* season structurally (+0.0147), is
`FAV.veto`'s second-best. The betting P&L is noise riding on top of a deficit
that never went away.

### 6.3 The power ceiling — this experiment cannot answer the question

Under the breakeven null a flat 1u bet has `E[pnl] = 0` and
`Var[pnl] = dec − 1` exactly, so `se(ROI) = sqrt(mean(dec−1)/n)`:

| config | n (4 seasons) | mean dec | se(ROI)% | ROI% needed for p<0.05 | observed (flat)% | n for a true +2% edge @80% power | ≈ seasons |
|---|---|---|---|---|---|---|---|
| FAV.base | 357 | 1.301 | 2.90 | **+4.78** | −2.25 | 4,653 | 52 |
| FAV.base.late | 134 | 1.265 | 4.45 | **+7.32** | +1.16 | 4,101 | 122 |
| FAV.veto | 166 | 1.299 | 4.24 | **+6.98** | +0.83 | 4,622 | 111 |
| FAV.veto.late | 94 | 1.270 | 5.36 | **+8.81** | +3.19 | 4,172 | **178** |
| DOG.base | 830 | 6.394 | 8.06 | +13.26 | −1.94 | 83,374 | 402 |
| DOG.base.late | 310 | 7.280 | 14.23 | +23.41 | −11.08 | 97,070 | 1,252 |
| DOG.veto | 447 | 6.208 | 10.79 | +17.76 | −9.85 | 80,497 | 720 |
| DOG.veto.late | 202 | 6.758 | 16.88 | +27.77 | −31.45 | 89,001 | 1,762 |

Look at the "ROI% needed" column against the "observed" column. **Not one
config observed even half of what it would need** — and the threshold itself
is absurd: a +7% ROI moneyline strategy would be one of the best sports
betting results ever documented.

The last column is the one that matters. `FAV.veto.late` — the best-behaved
config in the whole study — generates **24 bets per season**. Detecting a true
+2% ROI edge on it at 80% power would take **178 NBA seasons.** So the honest
statement is not "the strategy is proven unprofitable" — it is:

> **We can rule out a large edge. We cannot detect a small one, and we have
> no positive evidence for one. The observed numbers are fully explained by
> noise (P = 0.592), and the structural diagnostic — which *does* have power
> at n = 4,920 — says we are behind the de-vigged close on every stratum we
> can define in advance.**

### 6.4 What actually holds up

Three findings survive.

1. **The tail veto is real, and it is a risk control, not an alpha source.**
   It cuts the eligible stratum's deficit from +9.75 to +7.76 nats/game and
   the games it removes are genuinely worse (+11.92). It roughly halves max
   drawdown (25.8% → 15.2% on FAV.veto with caps). It does not turn a losing
   proposition into a winning one, and no sizing rule can.
2. **The DOG arm is dead — decisively, and this settles input (3).** The
   forensics predicted DOG > FAV because `conf_us < conf_mkt` is the
   less-bad regime. It is not: DOG pooled ROI is negative in 3 of 4 configs,
   DOG.veto.late loses 24% pooled with a 49% drawdown, and DOG configs carry
   3-5× the drawdown for worse returns. **"Less bad in log-loss" ≠
   "profitable at the price."** D77/D82's "fade favourites contraindicated"
   is upheld against the forensic prediction. Log-loss regimes do not
   translate into betting directions, because the price already contains the
   thing that makes the regime less bad.
3. **The model beats the OPEN and loses to the CLOSE (section 2.1).** This is
   the only place in the whole exercise where a real, significant, correctly
   controlled effect appears: +0.0112 CLV at t = +6.7 with the
   favourite-drift control running *negative* (−0.0036, t = −2.1). That is
   not a betting result — it is a **timing** result, and it says the value in
   this model is realised by transacting **early**, not by transacting
   better.

### 6.5 Cross-thread: reconciliation with D112 (landed the same day)

D112's W49 catastrophic-tail forensic is the thread that produced this
strategy's design inputs, and it finished independently. Four points of
contact, all of which strengthen the verdict:

1. **D112's structural identity confirms our edge test.** On a same-side bet,
   `edge ≡ conf_us − conf_mkt` exactly — so our `DIV_CAP` (gross-divergence
   cap) *is* D112's upper conf-excess cap under another name. D112 reported
   R4_LOWT going −4.51% → **+2.47%** pooled at X=0.08. We pre-registered
   X=0.10. Sweeping ours to match:

| config | cap | n_A | ROI_A% | n_B | ROI_B% | both>0 | pooled% | pooled fair% | P(noise) |
|---|---|---|---|---|---|---|---|---|---|
| FAV.base | 0.10 | 151 | −0.27 | 206 | −6.16 | no | −3.53 | +0.81 | 0.868 |
| FAV.base | 0.08 | 130 | +3.11 | 172 | −6.27 | no | −2.17 | +2.23 | 0.747 |
| FAV.base | 0.06 | 98 | +2.14 | 123 | −3.43 | no | −0.94 | +3.51 | 0.604 |
| FAV.base.late | 0.10 | 49 | +7.11 | 85 | −2.20 | no | +0.97 | +5.52 | 0.426 |
| FAV.base.late | 0.08 | 44 | +11.64 | 73 | −0.79 | no | +3.38 | +8.03 | 0.258 |
| FAV.base.late | 0.06 | 33 | +9.64 | 51 | −1.28 | no | +2.52 | +7.13 | 0.336 |
| FAV.veto | 0.10 | 73 | +11.30 | 93 | −7.54 | no | +0.44 | +4.96 | 0.468 |
| FAV.veto | 0.08 | 65 | +12.40 | 80 | −5.30 | no | +2.26 | +6.86 | 0.322 |
| **FAV.veto** | **0.06** | 51 | +10.78 | 59 | **+1.84** | **YES** | **+5.77** | +10.53 | 0.141 |
| FAV.veto.late | 0.10 | 38 | +12.91 | 56 | −4.08 | no | +2.11 | +6.70 | 0.365 |
| FAV.veto.late | 0.08 | 34 | +15.09 | 48 | −1.39 | no | +4.62 | +9.32 | 0.224 |
| FAV.veto.late | 0.06 | 27 | +16.33 | 33 | −4.89 | no | +3.51 | +8.17 | 0.307 |

   The threads **converge in direction**: tightening the cap improves pooled
   ROI monotonically in 3 of 4 configs, exactly as D112's cap sweep found.
   And **one config finally holds both halves** — `FAV.veto` at cap 0.06,
   pooled +5.77% at vig, +10.53% fair, P(noise) = 0.141.

2. **But that survivor is fewer than chance produces.** Null check, 4,000
   game-level replicates with `y ~ Bernoulli(p_mkt)` and fair pricing, so
   every config is *exactly* breakeven by construction:

   > Expected number of the 12 configs holding both halves **by chance = 3.13**
   > (median 2, 90th pct 9). **Observed = 1.**
   > **P(chance produces ≥ 1) = 0.651.**

   We found *less* than a market-efficient world hands out for free. There is
   no survivor to promote. This is the cleanest single refutation in the
   study, and it is the reason the cap sweep does not reopen the verdict.

3. **D112's Kelly-slope result is a direct critique of our Layer C, and it
   holds.** D112 measured `realised_excess = −0.0140 + 0.184 × claimed_excess`
   (se 0.097) over 4 seasons, n=4,367 — **82-89% of our claimed edge is
   illusory**, so Kelly stakes on the claimed edge are 5-9× oversized. Our
   Layer C sizes Kelly on exactly that claimed edge, so quarter-Kelly is
   still ~1.4-2.3× too large by D112's slope. Our sizing sensitivity (§5.5)
   is the independent check and it agrees that this is not the binding
   problem: eighth-Kelly (which *is* roughly D112-corrected) gives
   **−1.12%** on FAV.veto against +0.44% at quarter-Kelly. Correcting the
   sizing does not rescue the strategy, because the strategy's problem is
   that the edge is not there, not that it is mis-sized.

4. **D112 independently validates two of our eligibility legs and challenges
   a third.** Validated: 57% of catastrophes are *opposite-side* games
   (same_side 0.429 vs 0.893 base) — our A2 filter excludes all of them by
   construction; and conf_gap is the dominant catastrophe separator — our
   DIV_CAP bounds it. Challenged: D112's walk-forward market-blind classifier
   for top-1% membership scores OOS AUC **0.654 vs 0.640 for `conf_us`
   alone**, i.e. *nothing* beyond our own confidence identifies the tail
   ex ante. That is a direct prediction that our EARLY/CHAOS/FRESH veto
   should help only marginally — and it is precisely what we measured
   (+9.75 → +7.76 nats/gm, real but nowhere near sufficient). **Two threads,
   different methods, same conclusion.**

### 6.6 Recommendation

**Do not deploy capital against the close.** The bet-at-close formulation is
structurally losing and the evidence for any configuration is noise.

The actionable path is the one section 2.1 points at: our edge exists against
the *opener*, and we cannot test it here because no opener exists for any OOS
season. That is a **data acquisition problem, not a modelling problem**, and
it is exactly what D95's paid retrospective already ranked first (Odds API
20K tier + historical snapshots + VPS, ~$520/yr — "buys PRICES, not model").
The 2026-27 paper-trade registry (`scripts/bet_engine.py`, D91) with CLV
logging from opening night is the correct instrument: it captures the price
path we are missing, and CLV is measurable at ~40× the rate of ROI, so it can
answer in one season what ROI cannot answer in eighty-five.

Concretely, replace the ROI target with a **CLV target**: log every candidate
at the earliest available price, settle CLV against the close, and gate any
future capital deployment on demonstrated positive CLV — not on backtested
ROI, which this document shows is uninformative at achievable sample sizes.

---

## 7. Honesty ledger

- **The single "surviving" config (FAV.veto @ cap 0.06) is not a result.** It
  came out of a 12-cell post-hoc sweep run *after* seeing D112, it has
  P(noise) = 0.141 per-comparison, and the null check says chance hands out
  3.13 such survivors per 12 cells. It is reported because suppressing it
  would be dishonest, not because it should be traded.

- **The eligibility thresholds** (EARLY < 20, CHAOS ≥ 18.0, FRESH ≥ 2) were
  fixed from covariate quantiles before any PnL was scored, but they were
  chosen by *me*, informed by prior register knowledge of which regimes are
  bad. They are not adversarially clean.
- **The FAV/DOG direction axis** was added because the forensics implied it.
  That is a genuine pre-registered arm, but it doubles the family and the
  family-wise numbers reflect that.
- **The T-tier (0.20) and divergence cap (0.10)** are inherited from D78,
  which selected them on data that includes 2023-24 and 2024-25. This
  evaluation is therefore clean w.r.t. *this* family's selection but **not**
  clean w.r.t. the D77/D78 discovery that motivated the tier. Same caveat
  class as bet_sim3.
- **Pricing** assumes a 4.5% proportional overround on a de-vigged
  closing-spread probability. Real NBA moneyline vig is asymmetric
  (favourites juiced harder), which if anything makes the FAV arm *optimistic*
  here. No line shopping is modelled in the headline; the break-even-V column
  shows the sensitivity.
- **Moneylines do not exist in our data after 2022-23.** Every OOS price is a
  spread converted through the fixed logistic `sigmoid(margin/6.96)`. The
  conversion reproduces the SBR close to MAE 2.5e-4 where both exist, but it
  is a conversion, not a quote.
- **2021-22 lacks `game_inactives`**, so veto configs cannot be built there
  and are reported blank rather than run on a degraded filter.
- **`nbapred/` was not touched.** DuckDB opened `read_only=True` throughout.


---
---

# PART II — the 2026-27 F4 PAPER-TRADE REGISTRY (SHIPPED)

Implemented in `scripts/bet_engine.py` (+ `scripts/f4_shrinkage.py`), re-simulated
in `scripts/f4_resim.py`, artifacts `data/f4_shrinkage.json` /
`data/f4_resim.json`, tests `tests/test_bet_engine.py` (11 green).

Registers: D75 / D78 / D82 (the rules), **D112** (the cap + the Kelly slope),
D13 / D77 (the divergence-is-a-liability thesis).

---

## 8. PRE-REGISTRATION STATEMENT

> **Frozen 2026-08-01, before a single 2026-27 game exists.** Everything in
> sections 9-11 — the four rules and their operators, the upper
> confidence-excess cap `CAP = 0.08`, the three sizing arms, the shrinkage
> functional form `shrunk_edge = max(0, a + b·edge)`, and the annual refit
> protocol — is fixed now and will not be changed during the 2026-27 season on
> the basis of 2026-27 results. Deviations get their own D-line and are
> reported as deviations.
>
> **What is registered vs. what is estimated.** `(a, b)` are ESTIMATED, not
> chosen: they are the OLS coefficients of realised excess on claimed excess,
> refit each October 1 on all COMPLETED seasons and on nothing else
> (`f4_shrinkage.py --refit`, cron'd). The refit is a **calibration of our own
> estimator**, not a tuning knob — it is scored by whether the corrected edge
> is closer to realised (`|realised − shrunk|` 0.3874 vs `|realised − claimed|`
> 0.3997 on 4 seasons: better), never by PnL. Refitting mid-season, or
> refitting to improve ROI, voids the pre-registration.
>
> **What is TENTATIVE and declared as such.** The **cap VALUE of 0.08** came
> out of a sweep: D112 swept 8 caps × 6 rules × 2 frames with **no selection
> protection**, so its confidence intervals are per-comparison and the exact
> 0.08 is not defensible on its own. What IS supported, by two statistics
> computed with no reference to any rule — the reliability curve and the
> Kelly slope — is the **DIRECTION**: realised excess stops rising with
> claimed excess and turns negative above ~0.06, so an upper cap on claimed
> edge should help. We ship the direction and carry the value as a registered
> guess. **The parameter, not the direction, is what October is testing.**
>
> **What we are NOT doing.** We are not re-tuning R4's threshold. D112's
> headline used `t = 0.04`; the shipped rule stays at its registered
> `t = 0.02`. `t = 0.04` is reported below as a labelled diagnostic only —
> promoting it now would be exactly the post-hoc selection D111's process
> lesson warns about.

---

## 9. THE RULES (unchanged from D75 / D78 / D82)

`edge = p_us_side − p_mkt_side` on the picked side; the pick is our side of
0.5. `p_us` is market-blind (G2); using `p_mkt` here is **bet selection**, not
a model input, and is allowed.

Two registry-level vetoes run **before** any rule:

| # | veto | why |
|---|---|---|
| V1 | opposite side never bet — `(p_us−0.5)(p_mkt−0.5) ≤ 0` | known net-negative (D78); 57% of D112's catastrophes live here |
| **V2** | **`conf_us − conf_mkt > 0.08` → skip** | **D112, NEW.** On a same-side bet this quantity *is* the edge, so V2 is an upper edge cap |

| rule | operators | provenance |
|---|---|---|
| `R4_LOWT` | `edge > 0.02` AND late (`max gp ≥ 55`) | D75 primary — first flat-OOS-vig-positive band in program history |
| `T20_D03_10_W` | `\|p_us−0.5\| > 0.20` AND `0.03 ≤ edge ≤ 0.10` AND late | D78 — first SELECTION-PROTECTED OOS-vig-positive rule |
| `T20_D03_10` | same, no late overlay | D78 — closest-to-profitable at real n |
| `STAR_FAV_SHARPER` | `edge > 0` AND the market favourite has a star OUT (≥28 trailing min over last 10 games with 12+ min, ≥3 qualifying) | D82 |

V2 **subsumes** the old D13/D78 band cap of 0.10 (0.08 < 0.10) and, for the
first time, puts a cap on `R4_LOWT` and `STAR_FAV_SHARPER`, which had none.
30 of D112's worst-49 games carry `conf_gap > 0.06`.

`rules_fired(..., cap=float('inf'))` reproduces the pre-D112 registry exactly;
that is how the re-sim's UNCAPPED arm and the unit tests are built.

---

## 10. SIZING — three arms in parallel, and the edge-shrinkage ship

### 10.1 The Kelly slope (D112, rule-free)

Over 4 seasons of same-side games (n = 4,367):

```
realised_excess = -0.0140 + 0.184 x claimed_excess     (se 0.097, t=+1.90)
```

replication frame (capstone_tank, 3 seasons, n = 3,311): `−0.0161 + 0.106·x`
(se 0.110). **82-89% of the edge we claim is not there.** Kelly consumes the
claimed edge linearly, so raw-Kelly stakes have been **5-9× oversized**. This
is the mechanism behind D75's unexplained signature — *quarter-Kelly negative
on the same bets where flat was positive*. A Kelly bettor whose stated edge is
5× too large is not a bettor with a small edge; he is a bettor over-betting an
edge that is not there.

### 10.2 The three arms (all logged for every candidate, none selected)

| arm | stake | what it is |
|---|---|---|
| `flat` | 1.0u | the **honest control** — the only arm D75/D78's positive results were ever measured on |
| `raw_kelly` | `min(0.25·f*(p_us_side), 10u)` on a 100u reference bankroll | what D75 ran; kept so October can price the correction |
| `shrunk_kelly` | same but `f*` computed from `p_mkt_side + max(0, a + b·edge)` | **the D112 ship** |

`f* ≤ 0` stakes 0 — never a negative stake, never a forced bet. All three are
staked on the **best shopped decimal** across books, and all three settle off
that one price, so the arms differ only in size.

**Cold start.** The walk-forward estimator needs ≥1 completed season. Where
none exists the shrunk arm stakes **0**, never falls back to raw Kelly.

### 10.3 The load-bearing consequence: at vig, the calibrated bettor bets nothing

Break-even claimed edge (where `shrunk_edge` first exceeds 0) is
`−a/b = **0.0758**`. The registered cap is **0.0800**. Those two numbers are
28 basis points apart, and that is the whole story:

* Only **2.2 – 5.8%** of capped bets have a positive calibrated edge at all.
* The largest overround at which quarter-Kelly on the calibrated edge still
  stakes is `V_max = (p_mkt_side + shrunk_edge)/p_mkt_side`, whose **maximum
  over every capped bet in 4 seasons is 1.0011** — against the sims' 1.045.
* Therefore **`shrunk_kelly` stakes exactly 0 on every registered rule, in
  every window, in both frames.** That is not a bug; it is the answer.

**Slope-noise sensitivity** (a fixed, b swept; % of capped registry bets a
quarter-Kelly bettor would stake at V = 1.045):

| b | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| primary | 0.0% | 0.0% | 0.0% | 0.0% | 0.2% | 5.3% | 11.0% | 17.3% | 24.5% | 30.8% |
| replication | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 3.7% | 9.4% | 16.4% | 22.6% | 29.8% |

`b` would have to exceed ~0.5-0.6 — **3.5 to 4.5 standard errors above the
point estimate of 0.184** — before a single capped bet clears the vig. The
sizing conclusion does not depend on the exact slope, only on `b ≪ 1`.

**So when does the shrunk arm ever fire?** Only when the best shopped decimal
beats `1/(p_mkt_side + shrunk_edge)` — i.e. only when line shopping delivers a
price **at or better than consensus fair**. The shrunk arm is therefore also a
**line-shopping detector**, and this is the precise, quantified form of D75's
hand-wave that "line shopping (~2-5c on favourites) covers the remaining gap".
It converts that claim into a nightly, falsifiable measurement.

**Corroboration from Part I.** §6.5(3), computed independently on a different
family, found eighth-Kelly (≈ the D112-corrected size) gives −1.12% against
+0.44% at quarter-Kelly on FAV.veto: correcting the sizing does not rescue a
strategy whose edge is absent. Two threads, same conclusion.

### 10.4 A warning the re-sim produced: shrinkage WITHOUT the cap is dangerous

With the cap lifted, the only way a bet clears the shrinkage intercept is to
carry a very large claimed edge — which is precisely D112's catastrophe
signature. Uncapped, the shrunk arm bets 1-5 games in 4 seasons and returns
**−24.6% to −100%**. **Shrinkage and the cap must ship together**; shipping
(a) without (b) would concentrate the entire bankroll on the worst bets we
make.

---

## 11. THE 4-SEASON RE-SIM (`scripts/f4_resim.py`)

Primary frame `data/ds_rt1_pergame.csv` (`p_full`, 4 seasons, n=4,920);
replication frame `data/capstone_pergame_tank.csv` (3 seasons, n=3,690).
Pricing/vig/Kelly conventions imported verbatim from `bet_sim3.py`
(proportional overround 1.045, decimal floored at 1.01, quarter-Kelly on a
100u reference bankroll capped at 10u).

**Both IS/OOS directions.** The two directions asked for are the same
partition scored once — only the labels swap, and the table says so rather
than printing each number twice:

* `DEV` = 2023-24 + 2024-25 — the seasons **every rule was developed on**.
* `NONDEV` = 2022-23 + 2025-26.
* **DEV-IN direction:** IS = `DEV`, OOS = `NONDEV`.
* **DEV-OUT direction:** IS = `NONDEV`, OOS = `DEV` — *the harder
  arrangement*, development seasons in the holdout.
* `REG:IS`/`REG:OOS` (2022-23…2024-25 / 2025-26) carried for continuity with
  the D75/D78/D112 tables.

### 11.1 PRIMARY frame — FLAT arm, n / hit% / ROI% / maxDD

| rule | window | UNCAPPED | CAP=0.08 | null p (capped) |
|---|---|---|---|---|
| **R4_LOWT** | POOL | 336 / 71.4 / **−4.81** / 25.70 | 267 / 76.0 / **−0.82** / 13.84 | — |
| | REG:IS | 255 / 69.4 / −7.08 / 23.04 | 202 / 73.3 / −4.00 / 11.96 | — |
| | REG:OOS | 81 / 77.8 / +2.33 / 7.13 | 65 / 84.6 / **+9.09** / 5.52 | 0.027 |
| | DEV | 207 / 69.1 / −9.09 / 20.45 | 164 / 72.6 / −6.63 / 11.96 | — |
| | NONDEV | 129 / 75.2 / +2.05 / 7.13 | 103 / 81.6 / **+8.44** / 5.52 | 0.016 |
| **T20_D03_10_W** | POOL | 155 / 80.6 / −0.27 / 11.16 | 138 / 82.6 / **+1.33** / 8.35 | 0.097 |
| | REG:IS | 111 / 80.2 / −1.73 / 8.51 | 100 / 83.0 / +1.36 / 5.73 | 0.131 |
| | REG:OOS | 44 / 81.8 / +3.40 / 4.70 | 38 / 81.6 / +1.23 / 4.49 | 0.265 |
| | DEV | 96 / 77.1 / −7.09 / 8.51 | 85 / 80.0 / −4.15 / 5.73 | — |
| | NONDEV | 59 / 86.4 / +10.82 / 4.70 | 53 / 86.8 / +10.11 / 4.49 | 0.024 |
| **T20_D03_10** | POOL | 408 / 76.2 / −3.23 / 21.81 | 353 / 77.6 / −2.46 / 18.85 | — |
| | REG:IS | 305 / 75.1 / −5.20 / 20.36 | 265 / 77.0 / −3.63 / 16.70 | — |
| | REG:OOS | 103 / 79.6 / +2.62 / 4.97 | 88 / 79.5 / +1.06 / 4.91 | 0.178 |
| | DEV | 249 / 74.3 / −7.35 / 20.36 | 216 / 75.9 / −6.18 / 16.70 | — |
| | NONDEV | 159 / 79.2 / +3.22 / 4.97 | 137 / 80.3 / +3.40 / 4.91 | 0.055 |
| **STAR_FAV_SHARPER** | POOL | 938 / 69.6 / −4.91 / 59.07 | 776 / 71.3 / −4.51 / 48.08 | — |
| | REG:IS | 671 / 69.4 / −4.95 / 38.02 | 554 / 70.9 / −4.68 / 28.24 | — |
| | REG:OOS | 267 / 70.0 / −4.80 / 23.57 | 222 / 72.1 / −4.09 / 23.31 | — |
| | DEV | 485 / 70.7 / −4.95 / 25.29 | 398 / 72.6 / −4.04 / 17.55 | — |
| | NONDEV | 453 / 68.4 / −4.87 / 35.08 | 378 / 69.8 / −5.00 / 31.98 | — |
| *[diag] R4_LOWT(t=.04)* | POOL | 225 / 69.3 / −4.51 / 15.57 | 156 / 76.3 / **+2.47** / 4.93 | 0.092 |
| | REG:OOS | 58 / 74.1 / +1.51 / 5.49 | 42 / 83.3 / **+11.66** / 3.56 | 0.042 |
| | NONDEV | 88 / 71.6 / −0.02 / 5.49 | 62 / 80.6 / +9.73 / 3.56 | 0.040 |

The bolded R4_LOWT(t=.04) cells reproduce D112 to the decimal
(−4.51 → +2.47 pooled; +1.51 → +11.66 OOS), so the re-sim is a faithful
carry-forward, not a re-derivation.

### 11.2 PRIMARY frame — RAW-KELLY arm (`shrunk_kelly` is 0 in every cell)

| rule | window | UNCAPPED | CAP=0.08 | null p |
|---|---|---|---|---|
| R4_LOWT | POOL | 255 / 69.0 / −8.71 / **99.26** | 186 / 74.7 / +4.10 / **16.11** | 0.045 |
| R4_LOWT | REG:OOS | 61 / 73.8 / −11.72 / 47.82 | 45 / 82.2 / +8.17 / 11.11 | 0.099 |
| R4_LOWT | DEV | 158 / 66.5 / −6.40 / 47.08 | 115 / 70.4 / +1.46 / 11.62 | 0.183 |
| R4_LOWT | NONDEV | 97 / 73.2 / −12.91 / 62.79 | 71 / 81.7 / +9.05 / 11.11 | 0.050 |
| T20_D03_10_W | POOL | 134 / 80.6 / +0.35 / 25.69 | 117 / 82.9 / +5.39 / 13.35 | 0.033 |
| T20_D03_10 | POOL | 357 / 75.9 / −2.83 / 50.48 | 302 / 77.5 / −1.38 / 36.77 | — |
| STAR_FAV_SHARPER | POOL | 510 / 66.1 / −5.71 / **143.65** | 348 / 68.1 / −3.21 / **33.54** | — |

### 11.3 The cap's most robust effect is DRAWDOWN, not ROI

This is the finding the ROI tables bury. Max drawdown falls in **essentially
every cell**, and by far more than ROI moves:

| rule / arm | maxDD uncapped → capped | factor |
|---|---|---|
| R4_LOWT, raw-Kelly, POOL | 99.26 → 16.11 | **6.2×** |
| STAR_FAV_SHARPER, raw-Kelly, POOL | 143.65 → 33.54 | **4.3×** |
| R4_LOWT, raw-Kelly, NONDEV | 62.79 → 11.11 | 5.7× |
| R4_LOWT, flat, POOL | 25.70 → 13.84 | 1.9× |
| [diag] R4(.04), flat, POOL | 15.57 → 4.93 | 3.2× |

A 4-6× drawdown reduction is not a noise-limited quantity the way a 3-point
ROI move is — it follows mechanically from deleting the fattest-tailed bets,
which is exactly what D112 says those bets are. **If only one thing in this
ship survives 2026-27, it should be this.**

### 11.4 The registered statistic: paired dPnL/bet (flat, capped − uncapped)

Evaluated on the UNCAPPED bet set so the pairing is exact; the capped arm
scores 0 on a skipped bet (capital not deployed). Positive = the cap helps.

| rule | PRIMARY POOL | REG:IS | REG:OOS | DEV | NONDEV | REPLICATION POOL |
|---|---|---|---|---|---|---|
| R4_LOWT | **+0.0417 [+0.0059,+0.0809] SIG** | +0.0391 NS | +0.0497 NS | +0.0384 NS | +0.0469 NS | +0.0273 NS |
| T20_D03_10_W | +0.0145 NS | +0.0295 NS | −0.0234 NS | +0.0342 NS | −0.0174 NS | +0.0035 NS |
| T20_D03_10 | +0.0110 NS | +0.0205 NS | −0.0171 NS | +0.0198 NS | −0.0029 NS | +0.0080 NS |
| STAR_FAV_SHARPER | +0.0118 NS | +0.0109 NS | +0.0140 NS | +0.0163 NS | +0.0070 NS | +0.0064 NS |
| *[diag] R4(t=.04)* | **+0.0622 [+0.0077,+0.1180] SIG** | +0.0597 NS | +0.0694 NS | +0.0580 NS | +0.0687 NS | +0.0431 NS |

**10 of 10 pooled cells positive across both frames.** Only the two
R4-family POOL cells are individually significant, and both were already
reported by D112 — so this is sign consistency, not new significance.

**ROI direction is weaker than dPnL direction, and the difference is real,
not cosmetic.** dPnL measures absolute PnL improvement; ROI divides by a stake
base the cap also shrinks. Capping improves POOL ROI on 4/5 primary rules but
only 2/5 replication rules (`T20_D03_10_W` −2.53→−2.76, `T20_D03_10`
−6.44→−6.77, `STAR_FAV_SHARPER` −5.22→−5.51 all get slightly *worse* on ROI
while their PnL improves). The cap's ROI benefit is concentrated in the
**R4 / late-window family**; elsewhere it buys drawdown, not return.

---

## 12. NOISE-COMPATIBILITY — the numbers that decide this

Null model: we have **no edge**, outcomes drawn `Bernoulli(p_mkt_side)` at the
same prices with the same stakes, 20,000 replicates. `null p` = P(ROI ≥
observed | no edge).

**12.1 Family-wise count.** 200 rule × variant × window × arm cells are
printed per frame.

> Expected cells with null p < 0.05 under a **global null** = **10.0**.
> Observed: **11** (primary), **8** (replication).

**Nothing in the ROI tables is distinguishable from chance at family level.**
Every positive cell in §11.1-11.2 is inside what a market-efficient world
hands out for free. (Cells overlap heavily — the same bets are re-scored
across windows and arms — so 10.0 is an upper bound on the surprise, not a
test. The bound is not binding: we are *at* it, not above it.)

**12.2 The power ceiling.** Flat per-bet PnL has sd ≈ 0.48-0.63 on these
rules. To detect a **true +2.5% ROI** at 80% power, two-sided 5%:

| rule | capped bets/season | sd | n needed | **seasons needed** |
|---|---|---|---|---|
| R4_LOWT | 66.8 | 0.586 | 4,315 | **65** |
| T20_D03_10_W | 34.5 | 0.481 | 2,900 | **84** |
| T20_D03_10 | 88.2 | 0.535 | 3,600 | **41** |
| STAR_FAV_SHARPER | 194.0 | 0.632 | 5,024 | **26** |
| *[diag] R4(t=.04)* | 39.0 | 0.602 | 4,552 | **117** |

Union of the four registered rules: 1,149 games fire uncapped (287/season),
**963 capped (241/season)** — the cap removes 16.2% of the nightly card. Union
flat ROI −5.35% → −4.54%.

**ROI cannot settle this in a human career.** That is why §6.6's
recommendation stands and is now the operating instruction: **the 2026-27
target is CLV, not ROI.** CLV is measurable at ~40× the rate of ROI, it is
logged on every candidate already, and it is the only statistic this registry
can actually move in one season.

---

## 13. STOP RULES AND WHAT WOULD FALSIFY THIS

Declared now, in advance:

1. **Mid-season refits are forbidden.** `f4_shrinkage.py --refit` runs
   October 1 on completed seasons only (cron'd). Any other invocation voids
   the pre-registration for that season.
2. **The cap value is on trial, the direction is not.** If 2026-27 shows the
   skipped band (`conf_excess > 0.08`) outperforming the kept band on CLV, the
   cap is wrong and gets a D-line reversing it. A null result does NOT license
   sweeping the cap again.
3. **No arm selection before the season ends.** The three sizing arms are
   reported side by side. Picking one from an in-season scorecard is itself a
   selection and must be pre-declared; the default at season end is *flat*,
   because flat is the only arm any historical positive was measured on.
4. **`shrunk_kelly` staking > 0 on many nights is itself a finding** — it
   would mean our best shopped price routinely beats consensus fair, which is
   the line-shopping premium D75 assumed and never measured. Log it, do not
   suppress it. **[D142 has now measured it historically: best-of-2 at the open
   is worth +0.94pp of breakeven and +0.0094 of CLV per bet. See §15.]**
5. **What would make us deploy capital: sustained positive CLV**, not
   backtested ROI. §12.2 is the reason.
6. **Line shopping may improve EXECUTION on a bet the model already wants; it
   may never QUALIFY a bet** (D142 §7). Firing the rules against the shopped
   price instead of the consensus adds bets that hit 68.02% against a 71.88%
   breakeven (-3.86pp, ROI -6.30%) versus the kept bets' -1.69pp — the extra
   volume a shop unlocks is negative selection. Edge is computed against the
   CONSENSUS price; the shopped price is used to transact and to score.

---

## 14. THE OCTOBER RULE CARD

*(Everything the engine does on a game night, in one page.)*

```
FOR EACH GAME ON TONIGHT'S SLATE
  p_us   <- nbapred slate model              (MARKET-BLIND; G2)
  p_mkt  <- MEDIAN de-vig implied across books     (consensus)
  price  <- BEST decimal on our side across books  (line shopping)

  pick side = our side of 0.5
  edge        = p_us_side - p_mkt_side              (vs CONSENSUS, never vs
                                                     the shopped price — D142)
  conf_excess = |p_us - 0.5| - |p_mkt - 0.5|        (== edge, same-side)

  V1  if not same side ................................ SKIP GAME
  V2  if conf_excess > 0.08 ........................... SKIP GAME   <- D112

  FIRE:  R4_LOWT           edge > 0.02  AND max(gp) >= 55
         T20_D03_10_W      |p_us-.5| > .20 AND .03 <= edge <= .10 AND late
         T20_D03_10        |p_us-.5| > .20 AND .03 <= edge <= .10
         STAR_FAV_SHARPER  edge > 0 AND market favourite has a star OUT

  SIZE (all three, on `price`, logged side by side):
         flat          1.0u
         raw_kelly     min(0.25 * f*(p_us_side), 10u)
         shrunk_kelly  min(0.25 * f*(p_mkt_side + max(0, a + b*edge)), 10u)
                       a = -0.0140, b = +0.184   (data/f4_shrinkage.json,
                       refit 2026-10-01 on completed seasons only)
                       EXPECT 0 most nights; > 0 means the shopped price
                       beat consensus fair.

  LOG: candidate_ts, p_us, p_mkt_side, edge, conf_excess, cap_in_force,
       shrunk_edge, shrink_a/b, all three stakes, price, book.
  SETTLE (next morning): outcome, PnL per arm, and CLV vs the last pre-tip
       snapshot.  CLV IS THE 2026-27 TARGET.
```

**One-line summary of the ship.** Our divergence from the market is a
liability that grows with its own size, and our claimed edge is ~5× larger
than the edge we actually realise. So: refuse the biggest divergences
(`CAP = 0.08`), size on the calibrated edge rather than the claimed one — which
at a 4.5% overround means *do not size at all* — keep flat as the honest
control, and measure CLV, because ROI cannot answer in 65 seasons what CLV can
answer in one.

---

## 15. LINE SHOPPING AT THE OPEN — WHAT THE BOOK PANEL IS ACTUALLY WORTH (D142)

Every betting result in Parts I and II priced **one number per game**, which is
not how anyone bets. D142 is the first test that prices a **book panel** at the
open, using the per-book `open_books` row in the TeamRankings spread-movement
scrape. It answers the question D121 left hanging: *does line shopping close the
1.47pp deficit?*

**PLAIN ANSWER: partially, and not enough.** Best-of-2 buys **+0.94pp** of
breakeven on the rule union (**+0.97pp** on the universe) = **64-66%** of the
1.47pp gap. Applied to D121's own registered number: `-1.47 + 0.97 = -0.50pp`.
After the outlier-realism haircut (drop or cap "best" prices more than 1.5 pts
from the other book, which is 8.1% of them) it is **+0.68 to +0.77pp**, i.e.
46-52%. **The recommendation of D121 is unchanged: no capital at open or
close.**

What the data supports:

| quantity | measured |
|---|---|
| books in the panel | **2** (the "Book 3" column is empty for every NBA game) |
| games with 2 opening quotes | 4,639 of 5,932; **73.6%** of the model frame |
| best-worst dispersion at the open | mean **0.649 pts**, median 0.50, 36.2% exact ties, 13.1% >= 1.5 pts |
| shop gain over a one-book bettor | **+0.331 pts** = +0.94..0.97pp of breakeven |
| adverse bound (WORST of 2) | **-1.02pp** of breakeven, ROI -4.10% vs -2.68% |
| CLV gain, best-of-2, rule union | **+0.0094/bet, +49%**, SIG in 5 of 5 sets |
| 5-8 book shop (EXTRAPOLATION, ceiling) | +1.94 to +2.38pp — would clear 1.47pp |

### 15.1 The consequence for October: the book panel is now REQUIRED

D125 shipped `bet_quotes_panel` (every book's two-sided quote at every emission)
and every bet row's `book` / `best_price` / `consensus_price`, noting the engine
"logs a book panel but has never been shown to NEED one". **It is now shown.**

The panel is the difference between **+0.0193 and +0.0287 of CLV per bet** on the
union — on the exact metric the 2026-27 programme is scored on — and between a
one-book bettor at **+0.0092** and one holding the wrong single account at
**-0.0007** on the universe. Capturing **>= 2 books at the open** is therefore a
**data-collection requirement, not telemetry**. An OPEN-view scan that books a
game off a single book's quote is a defective observation, and a month in which
the median slate game has fewer than 2 book quotes at the OPEN view should be
flagged in `--monthly-report` alongside the CLV bands.

D125's 2026-12-31 decision rule ("if one book family supplies >= 75% of positive
CLV, line shopping is the product; flat across books => model timing is") stands,
and should now be read with a prior: best-of-N is worth **~+0.009 CLV at 2 books**
and materially more at 5-8, so a flat-across-books reading would be the
surprising outcome, not the default.

### 15.2 What line shopping is NOT allowed to do

**Shop the price; do not shop for more bets** (now §13 stop rule 6). Re-firing
the rules against the shopped price grows the union from 938 to 1,033 bets and
makes the arm *worse* (-2.05pp / ROI -3.29% versus pure execution's -0.91pp /
-1.27%); the added bets alone hit 68.02% against a 71.88% breakeven. Edge is
computed against the consensus; the shopped price is used to transact and score.

**Do not read the CLV gain as a second confirmation.** It is arithmetic — CLV is
`p_close_side - p_open_side`, so a better open raises CLV by exactly the price
gain. It is bankable in a CLV-scored programme; it is not new information.

### 15.3 Caveats that bind

- **2 books, one vendor, spreads only** (no moneylines), so every number goes
  through `p = sigmoid(margin/6.96)` + a 1.045 overround. That map is ~1.9pp
  pessimistic on breakeven versus real MLs — a LEVEL bias that cancels in the
  policy deltas. Spread-scale sensitivity moves the gain only to 0.87-1.01pp.
- **The 2-book subsample is not a random half.** On the ML frame, 2-book games
  run at -3.47pp and games without 2 books at +3.48pp, because TR's coverage
  collapses in 2025-26 (561/1,227), the season the model did best in. The shop
  GAIN transfers; the ROI LEVEL of that subsample does not, and must not be
  quoted as the programme's ROI.
- **Opening lines carry the day's lowest limits**, and that bites hardest on
  precisely the outlier book a shop wants to hit.
- Family-wise: 81 cells scored, 4.0 expected significant under a global null,
  **0 observed**. Nothing here beats breakeven; the result is a price mechanic,
  not an edge.

---

## 16. THE MOVEMENT MODEL AND THE ARBITRAGE QUESTION (D147 / D148)

Two changes to the October plan, one addition and one prohibition. Neither
touches the frozen D75/D78/D82 registry, the sizing arms, or the stop rules.

### 16.1 What changed

The owner's question was: *"if we predict that the closing prices significantly
differ from the opening, we should have some arbitrage, right?"*

**The answer is: there is a lock, and you should not take it.** A closeable
round trip (bet side A at the open, take side B at the moved price) locks on
**20-34%** of games on REAL moneylines — but hedging it turns a **+9.51%**
directional position into a **+4.55%** one. It is not arbitrage: a true arb is
riskless *at inception*, and this is a directional bet plus an option to hedge
that only exercises when we were already right. The lock rate IS the option's
exercise rate.

The threshold arithmetic is the whole story. Two retail books at the validated
1.043 overround lock iff **CLV on the entry side exceeds `1 - 1/1.043 =
0.04123`**. Our measured CLV per bet runs +0.009 (universe, D121) to +0.054
(D147's best rule). Only the best rule's *mean* clears it, and a mean above a
threshold is not per-bet clearance.

### 16.2 ADDITION — a second CLV selector in the paper book

D147's movement model predicts open->close movement at **OOS R^2 +0.171**
(rolling-origin, live features only), direction correct **65.0%** of all movers
and **84.0%** in the top bucket, against a within-date permutation placebo of
+0.005 (p=0.0000). Its CLV per bet at matched volume is ~3x the frozen union's:

| set | n | CLV/bet | season-clustered CI |
|---|---|---|---|
| universe (side = p_us) | 4882 | +0.00907 | [+0.00797,+0.00989] |
| UNION (4 frozen rules) | 1318 | +0.01833 | [+0.01495,+0.02218] |
| MOVEMENT \|pred\|>1.00 | 1783 | +0.03995 | [+0.02891,+0.05067] |
| MOVEMENT \|pred\|>1.50 | 920 | +0.05409 | [+0.03929,+0.07184] |

Monthly, `|pred|>0.50` gives 28 months at a median 123 bets/month,
mean-of-months **+0.02983, 100% of months positive, zero below the -0.0131 red
flag, 23/28 above the +0.0200 good line**.

**It ships to PAPER as a T2 monitored second selector, not as a replacement.**
It is ERA-CONDITIONAL (I^2=76%, p_Q=0.039) and its K-1 season-mean t interval
straddles zero, so it does not clear a T1 row.

**Decision rule, pre-registered here:** if the OPEN-view CLV of `|pred|>1.00`
does not beat the frozen union by **>= +0.010/bet over two completed calendar
months**, drop it and say so in a new D-line.

### 16.3 PROHIBITION — the D142 re-fire rule extends verbatim

D142 (7) established: *shop the price, never shop for more bets.* The same
applies to the movement model. It may be used to **improve execution and to
rank bets the frozen rules already want**; it must **never qualify new bets**.
D142 measured the added-volume penalty at -3.86pp on the shopped-price version,
and there is no reason to expect a different sign here.

### 16.4 What does NOT change, and why

**No capital at open or close.** Unchanged from D121/D126/D142. Every
naked-entry cell is **ns** under the K-1 cluster-mean t interval that
GATE_POLICY_V2 §9.3 makes binding at K=4 seasons — including the eye-catching
`|pred|>1.50` cell at cover 57.99% / ROI +10.70%, whose cluster-mean t interval
is [-2.83%, +23.88%]. The season-cluster bootstrap says SIG; the bound that
counts says no.

And the decisive check: regressing (realised margin - REAL closing line) on the
predicted movement gives **i.i.d. t = +2.17** — which would read as "we beat the
close" — but **season-clustered t = +1.68, not significant**. That is the exact
pattern the sister football project had to retract. **We beat the OPEN. We do
not beat the CLOSE.**

### 16.5 The one thing that survives the conservative bound

The **ex-ante middle**: bet side S at the opening number at -110, and buy the
other side at the close only if the line moved >= W points in S's favour.

| side rule | W | n | 2nd-leg fires | P(mid \| 2 legs) | ROI/entry | cluster-mean t CI |
|---|---|---|---|---|---|---|
| p_us (incumbent) | 1.0 | 4849 | 42.2% | 7.28% | -4.71% | [-9.19,-0.17] SIG NEG |
| pred_dm \|pred\|>1.0 | 2.0 | 1771 | 35.5% | 10.33% | **+4.62%** | [+2.56,+6.65] SIG |
| pred_dm \|pred\|>1.5 | 2.0 | 914 | 42.3% | 11.63% | **+6.79%** | [+0.88,+12.18] SIG |

Positive in 4/4 seasons. It survives the K-1 bound precisely *because* the
middle cuts season-to-season dispersion by more than it cuts the mean — **the
statistical case strengthens exactly as the economic case weakens.** Per
GATE_POLICY_V2 §11 tie-break 4 the live expectation is the most recent fold:
**2025-26 gives +3.71%**, not the pooled +4.62%.

This is a PAPER measurement. It is not a recommendation to deploy capital.

### 16.6 Caveats that bind (D148 §10)

- **$0 budget — FATAL.** A round trip needs capital on both legs at two venues
  simultaneously. The lock rate is irrelevant if the first leg cannot be placed.
- **US exchange access — BINDS HARD.** The exchange is the only cost model under
  which this is comfortable (commission on net winnings only: `|pred|>1.5` at 5%
  sweep / 5% commission still locks 34.02% for +34.08%). Betfair excludes US
  customers; Prophet X / Novig / Sporttrade are few-state with thin NBA side
  liquidity at the open.
- **Opening-line limits — BINDS** exactly on the big-predicted-movement games a
  book prices small and moves fast (D120's standing caveat).
- **Price you see != price you get — BINDS MEASURABLY.** Exiting at the close
  locks 28.51%; exiting at the single best quote on the TeamRankings intraday
  path locks 43.86%. Roughly a third of measured "locks" exist only at one
  moment's quote, and those are exactly the ones stale-line voids apply to.
- **A PRICING-FRAME CORRECTION that partly revises §15.3 above.** The SP map
  `p = sigmoid(margin/6.96)` is **not** a level bias. It is monotone in the side:
  pessimistic on favourites (SP/ML up to 1.05) and **optimistic on dogs** (down
  to 0.91 at p~0.35), quoting dog decimals ~10% longer than any book offers.
  §15.3's cancellation argument holds for the favourite-heavy frozen rules and
  FAILS for any dog-inclusive set — the movement model's picks are ~50/50, and
  the same bet set reads **+23.49% on SP versus +9.93% on real moneylines**.
  Any future work on a dog-inclusive bet set must price on real MLs.

---

## 17. THE HONEST RE-CERTIFICATION OF THE CLV PROGRAMME (D159, 2026-08-03)

**Everything in §§8-16 above was computed from model probabilities produced by
the availability-LEAKY path D158 found** (`ds_rt1_capstone.py:117-122` builds OUT
sets from tonight's box score, the same defect `prod_by_season.py` had). The
rules select on model-vs-market divergence, so a model that secretly knows who
played selects better bets. D156 Part B re-measured ROI honestly; **CLV — the
declared October target — had never been measured honestly until D159.**

Full working: `data/honest_trading_notes.md`. Artifact `data/hc_honestclv.json`,
chart `charts/honest_clv.png`.

### 17.1 The CLV survives, 13% smaller

Real opening moneylines @ open, n=3,682 universe = 2023-24..2025-26 = **exactly
the three FULL-T2 (full injury feed) seasons**, K=3.

| set | n | **CLV HONEST** | K−1 cluster-mean t | CLV LEAKY (same corpus) | **leak share** | registered (D155) |
|---|---|---|---|---|---|---|
| R4_LOWT | 484 | **+0.01488** | [+0.01055,+0.01951] SIG | +0.01499 | +0.7% | +0.01961 |
| T20_D03_10_W | 216 | +0.00815 | [−0.00671,+0.02406] ns | +0.00734 | −11.0% | +0.01358 |
| T20_D03_10 | 574 | **+0.01392** | [+0.00872,+0.01926] SIG | +0.01551 | +10.2% | +0.01785 |
| STAR_FAV_SHARPER | 1036 | **+0.00781** | [+0.00537,+0.01011] SIG | +0.00983 | +20.5% | +0.01215 |
| **UNION** | **1398** | **+0.01197** | **[+0.01099,+0.01295] SIG** | +0.01381 | **+13.4%** | +0.01590 |

**13.4% of the measured union CLV was the leak**, not most of it. Honest is
86.6% of the same-corpus leaky level and 75.3% of the registered one — the rest
of the gap is corpus drift (D152 backfill + D153 tank floor landed after
`ds_rt1_pergame.csv` was built). D121's selection placebo, re-run honest, comes
back at **−0.0008 [−0.0038,+0.0039] on the union — clean**: this is information,
not open-price mean reversion.

§15's line-shopping asset is almost untouched, because the best-of-2 gain is
ARITHMETIC: UNION honest ONEBOOK **+0.01721**, BEST2 **+0.02628** (leak shares
8.7% and 6.6%; registered +0.01943 / +0.02883).

### 17.2 The rules fire MORE honestly, not fewer

Union bets, ML frame: **1,398 honest vs 1,355 leaky (+3.2%), Jaccard 0.886.** A
less-confident model disagrees with the market MORE, so more games clear the
edge thresholds. All four rules still fire, all still 100% favourite-side, mean
implied price unchanged (0.688 vs 0.684), per-season volume flat. The selection
has not degenerated. **October volume assumption: 66 union bets/month** (median
over 21 months on the ML frame the engine actually prices), not the 44 the D121
bands assumed.

### 17.3 The matched-control alpha loses significance

D155 §5's alpha (rule ROI minus a bin-matched market-favourite control),
re-measured honest:

| when | set | rule ROI | ctrl | **alpha honest** | K−1 t | registered |
|---|---|---|---|---|---|---|
| OPEN | UNION | **−0.66%** | −5.58% | **+4.93%** | [−8.30,+17.44] **ns** | +8.22% **SIG** |
| CLOSE | UNION | −1.41% | −4.82% | **+3.41%** | [−5.97,+14.39] ns | +6.51% ns |

Alpha is positive in **10/10** cells and significant in **0/10** (D155 had 2/10).
The favourite-headwind story stands; its magnitude is ~40% smaller and the
register must stop citing +6.51/+8.22 as measured alpha. **Honest bet-at-open
union ROI is NEGATIVE (−0.66%)** — independently confirming D156 Part B on the
real-moneyline frame. **No capital, at the open or the close. D121's
recommendation is unchanged and better supported.**

### 17.4 THE BANDS ARE REPLACED

The registered **−0.0131 / +0.0200** were recovered exactly from
`data/bo_openbacktest.json` and have **two defects, only one of which is the
leak**:

1. they are **CENTRED ON THE ALL-SAME-SIDE UNIVERSE (+0.00350)** but **WIDTHED
   BY THE UNION** (±2×0.05521/√44), while `bet_engine.py --monthly-report`
   scores the **UNION** against them. Under their own (leaky) calibration GOOD
   sat only **+0.24σ** above the union's monthly mean — a coin flip, never a
   2-sigma gate;
2. they are framed on the **SP spread convention**, while the engine prices on
   real book decimals (the ML convention, `bet_engine.py:505-516`).

**NEW CALIBRATION — honest, union-centred, real-moneyline frame, 66 bets/month:**

> **CENTRE (honest expectation) +0.01197  ·  per-bet CLV sd 0.05111  ·  monthly
> se 0.00629**
>
> | band | value |
> |---|---|
> | **RED FLAG** | month mean CLV **< −0.0006** |
> | **GOOD** | month mean CLV **> +0.0246** |
>
> At other volumes: `band = 0.01197 ± 2 × 0.05111/√n` → 44/mo −0.0034/+0.0274,
> 100/mo +0.0018/+0.0222.

Plain reading: **at 66 bets/month a NEGATIVE month is a red flag** (zero is
−1.90σ), and a month must beat **+0.025** to be good news.

**PRICE-PANEL RULE, DECLARED IN ADVANCE.** +0.01197 is a SINGLE-QUOTE number.
§15 measured that best-of-2 adds +0.0094 arithmetically and that the shop GAIN
transfers even where the LEVEL does not — so a ≥2-book live capture runs ~+0.021
and would clear a +0.0200 GOOD line **on line shopping alone, with zero model
skill**. Therefore: **score the monthly CLV against these bands at the CONSENSUS
price, and report the best-price CLV separately as the shop premium.**
`bet_quotes_panel` already logs both (§14 / D125 §2); this needs discipline, not
new data.

**D125's real-stakes trigger** ("2 consecutive months OPEN CLV > +0.0200, none <
−0.0131") is **mis-specified, not merely mis-levelled**. On the honest ML frame
the old +0.0200 line is a +1.28σ monthly event: P(month) 10.1%, P(2 consecutive)
1.0%, **expected wait ≈ 98 months**. Under the new bands it becomes a genuine 2σ
gate (P(month) 2.3%, P(2 consecutive) 0.05%) — the correct property for a gate
on real capital. **The constants in `scripts/bet_engine.py`
(`CLV_MONTH_RED` / `CLV_MONTH_GOOD`) are NOT changed by D159; changing a frozen
live constant is its own D-line and its own decision.**

### 17.5 THE OPERATIONAL FACT THAT MATTERS MOST

Union CLV by season and availability tier (SP frame, 5 seasons — the one place
2021-22 is used, deliberately, because it is the season with **no injury feed in
existence** and is therefore the natural experiment for a feed outage):

| season | tier | CLV honest | CLV leaky | **leak share** |
|---|---|---|---|---|
| **2021-22** | **BLIND (no feed)** | **+0.00859** | +0.02970 | **+71%** |
| 2022-23 | inactives-only T2 | +0.02115 | +0.02464 | +14% |
| 2023-24 | full T2 | +0.01606 | +0.01907 | +16% |
| 2024-25 | full T2 | +0.01723 | +0.01711 | −1% |
| 2025-26 | full T2 | +0.01140 | +0.01325 | +14% |

**On full-feed seasons the leak was worth 10-20% of the CLV; on the no-feed
season it was worth 71%.** The availability feed is roughly two-thirds of what
the CLV asset is made of. **A feed outage in October is not a data-quality
incident — it is an asset outage, and the bands above become wrong the day it
happens.** Feed liveness is now a first-class operational dependency of the
paper book.
