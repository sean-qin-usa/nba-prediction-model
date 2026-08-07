# NBA opening-spread walk-forward — model and strategy review

**Strategy:** Directional value-taking against the opening point spread in NBA
regular-season games. A market-blind margin model prices every listed game; its
disagreement with the opening spread is then spent, at about a third of face
value, as a correction *to* the opening spread — `m_final = open_margin + f(x)`,
with `f` a ridge shrunk hard toward zero and fitted walk-forward. A single side
is taken where the corrected margin still disagrees with the line by more than a
walk-forward selected threshold, at the best number available across books. Every
position is one side of one game at −110, entered at the open and held to
settlement: no hedge, no offsetting leg, no in-game management, no exit. Flat 1
unit per bet, so nothing compounds. No calendar filter.

Selection is the strategy. The book is roughly 10% of the slate; the cover rate
on the selected book is not the cover rate on the slate.

**Frame:** 888 bets, 7 seasons, 2019-20 → 2025-26. Earlier seasons are excluded
on principle: the daily injury report begins 2018-12-17 and the availability leg
is half the production margin, so before 2019-20 the model cannot run as designed.
The configuration is selected on seasons 1..k, frozen, scored on k+1, and rolled
forward; the selection history reaches back to 2012-13.

## Headline results

| season | books/game at open | bets | P&L (u) | ROI | Sharpe (ann.) |
|---|---|---|---|---|---|
| 2019-20 | — | 119 | −4.28u | −3.60% | −0.45 |
| 2020-21 | — | 94 | −5.71u | −6.07% | −0.60 |
| 2021-22 | — | 112 | +6.93u | +6.18% | 0.63 |
| 2022-23 | — | 103 | +7.55u | +7.33% | 0.90 |
| 2023-24 | 7.74 | 154 | +21.20u | +13.77% | 1.90 |
| 2024-25 | 1.00 | 210 | +50.40u | +24.00% | 3.81 |
| 2025-26 | 1.03 | 96 | +4.85u | +5.05% | 0.54 |
| **2019-26 pooled, best of 9 books** | — | **888** | **+80.93u** | **+9.11%** | **1.10** |
| 2019-26 pooled, after outlier haircut | — | 888 | +69.56u | +7.83% | — |

ROI is P&L per unit staked. Sharpe is annualized by √(sessions per season), not
√252, which would inflate it 1.8×. Prices are the best of 9 books, but that panel
is measured only in 2023-24, so the multi-book figures are modelled for every
other season. The haircut row charges for the 8.1% of best-of-N prices that sit
more than 1.5 points off the next book.

![equity](../charts/review_equity.png)

*All 888 bets in date order at the opening spread and −110; flat 1u, so the path
is a running sum and nothing is re-selected within it. Trough is −23.1u at bet
161. The first three seasons lose money; the last four make +83.5u.*

Pooled ROI is **+9.11%** over 888 bets, with a season-clustered 95% interval of
**[−2.75%, +16.08%]** and an MDE80 of 12.7pp. **The interval contains zero** and
2024-25 alone supplies **62%** of the P&L. The frame is also too short to tune on:
measured, a null taking the best of five randomly chosen game subsets buys **+2.54
ROI points** on average, and every strategy filter tested lands inside that band.
Shortening the window does not help — the best 4-season window returns +14.92%
against a +9.33% average across all 4-season windows, so **+5.6 of those points
are the selection, not a regime.**

## The model

The model outputs a margin, not a probability. A spread is itself a margin
forecast, so the sides test involves no devig or probability conversion.

```
margin = 0.5·four_factors + 0.5·availability_composition + schedule_layer + tank_term
P(home win) = sigmoid(margin / 7.2)
```

Two independent estimates of team strength, averaged, plus additive context. The
market-blind model never sees market odds, enforced structurally rather than by
convention; only the offset layer above it sees the price.

| component | what it is | how it is fit |
|---|---|---|
| Four factors | opponent-adjusted ratings on shooting, turnovers, rebounding and free-throw rate; a decomposition of points per possession rather than a feature shortlist | one L2 ridge solve per factor (`ridge=25`), then the four adjusted values mapped to points by a fitted linear map. An 8-factor extension tied, so 4 is kept for parsimony |
| Availability composition | Σ over available players of DARKO talent × trailing minutes / 48; the leg that reacts to injuries | each player weighted by 1 − P(out), forecast from as-of-open information only. Requires the daily injury report; this is why the frame starts 2019-20 |
| Schedule layer | home edge, back-to-backs, dead-team flags | estimated walk-forward, EB shrinkage `n/(n+600)` toward a prior, `team_home_ridge=200` on per-team home deviations. The only component that has survived strict out-of-sample testing on every split tried |
| Tank term | late-season effort | exactly zero outside its window |
| Offset layer | the correction to the opening line | ridge on three features knowable at the open, shrunk hard toward zero. Fitted edge coefficient 0.33–0.37 in every fold |

## How it compares to the market

Model, opening line and closing line scored on identical games:

| source | log loss | |
|---|---|---|
| market-blind model | 0.59276 | beaten by both |
| opening line | 0.59228 | |
| offset construction | 0.58865 | recovers 26.7% of open→close |
| closing line | 0.57870 | |

The market-blind model does not beat the opening line standing alone — it is
0.00048 worse than the open and 0.01406 worse than the close. It is not a
discarded attempt: it is the offset layer's dominant input, and the finding that
its disagreement must be spent at a third of face value against a market anchor,
rather than trusted on its own, is the strategy.

## Notes and caveats

- Results are **simulated** on recorded opening prices with a walk-forward
  selection rule. **No capital has ever been deployed.**
- Multi-book pricing is measured in 2023-24 only and modelled elsewhere.
- The architecture was developed on 2021-26 data, so retrospective performance is
  not an untouched test. **2026-27 is the first decisive prospective evaluation.**
- Neither arm is the production default; promoting one is a re-certification.
- Full method, the register of 209 decisions (most of them rejections), the
  bet-time information leak and its fix, and the window-selection analysis are in
  the repository README and `docs/`.
