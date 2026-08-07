# NBA opening-spread walk-forward — model and strategy review

Code, data pipeline and the full research register: **github.com/sean-qin-usa/nba-prediction-model**

## Strategy

Directional value-taking against the opening point spread in NBA regular-season games. A market-blind margin model prices every listed game; its disagreement with the opening spread is then spent, at about a third of face value, as a correction *to* the opening spread. What is learned is only the correction; everything else is handcrafted and declared:

```
m_offset = m_open + f( m_blind − m_open ,  rest differential ,  |m_open| )
```

`f` is a ridge shrunk hard toward zero, fitted walk-forward. Its coefficient on the fundamental disagreement is 0.33–0.37 in every fold. A single side is taken when the corrected margin still disagrees with the line by more than a walk-forward selected threshold, at the best number available across books. Every position is one side of one game at −110, entered at the open and held to settlement: no hedge, no offsetting leg, no in-game management, no exit. Flat 1 unit per bet, so nothing compounds. No calendar filter.

Selection is the strategy. The book below is 460 bets out of roughly 3,690 regular-season games in the window, about 12% of the slate. A side is taken only where the corrected margin disagrees with the opening spread by more than the walk-forward threshold, so the cover rate on the selected book is not the cover rate on the slate.

Simulation: 460 bets, 3 seasons, 2023-24 → 2025-26. The configuration is selected on seasons 1..k from a pre-declared space, frozen, scored on season k+1, and rolled forward; nothing is refitted on the season being scored. The selection history reaches back to 2012-13, so the three seasons here are scored against a decade of prior data rather than against each other. Results are reported at a modelled best-of-nine execution tier. That tier is observed directly in 2023-24, where the panel averages 7.74 books per game, and inferred from the measured shopping relationship in 2024-25 and 2025-26.

## Headline results

| season | books/game at open | bets | P&L (u) | ROI | Sharpe (ann.) |
|---|---|---|---|---|---|
| 2023-24 | 7.74 | 154 | +21.20u | +13.77% | 1.90 |
| 2024-25 | 1.00 | 210 | +50.40u | +24.00% | 3.81 |
| 2025-26 | 1.03 | 96 | +4.85u | +5.05% | 0.54 |
| **2023-26 pooled, modelled best of 9 books** | **—** | **460** | **+76.44u** | **+16.62%** | **2.25** |
| 2019-26, all seven scored seasons | — | 888 | +80.93u | +9.11% | 1.10 |

ROI is P&L per unit staked; Sharpe is annualized on realised sessions. The multi-book panel is measured only in 2023-24, so the pooled figures are modelled for 2024-25 and 2025-26. The last row is the full scored frame, the honest denominator against which the three-season block is its recent end.

| execution | 2023-26 ROI |
|---|---|
| 1 book, observed | +10.63% |
| 2 books | +12.91% |
| **best of 9 books, the reported tier** | **+16.62%** |

Best-of-nine is the largest practical weakness in the headline, so the single-book number sits beside it: about two-thirds of the reported ROI survives with no shopping at all.

![equity](../charts/review_equity.png)

*All 460 bets in date order at the opening spread and −110; flat 1u, so the path is a running sum and nothing is re-selected within it. Max drawdown is −7.75u at bet 372. Over the full seven-season frame the path is +80.93u on 888 bets with a −23.1u drawdown, the first three of those seasons losing money and the last four making +83.5u.*

Pooled ROI is +16.62% over 460 bets, with a season-clustered 95% interval of [−9.29%, +37.83%]; on the full seven-season frame it is +9.11% with an interval of [−2.75%, +16.08%]. Three seasons is a small sample and both intervals contain zero. 2024-25 supplies 66% of the three-season P&L.



## The model

The model outputs a margin, not a probability. A spread is itself a margin forecast, so the sides test involves no devig or probability conversion.

```
margin = 0.5·four_factors + 0.5·availability_composition + schedule_layer + tank_term
P(home win) = sigmoid(margin / 7.2)
```

Two independent estimates of team strength, averaged, plus additive context. The market-blind model never sees market odds, enforced structurally rather than by convention. Only the offset layer above it sees the price.

| component | what it is | how it is fit |
|---|---|---|
| Four factors | opponent-adjusted ratings on shooting, turnovers, rebounding and free-throw rate; a decomposition of points per possession rather than a feature shortlist | one L2 ridge solve per factor (`ridge=25`), then the four adjusted values mapped to points by a fitted linear map. An 8-factor extension tied, so 4 is kept for parsimony. |
| Availability composition | Σ over available players of DARKO talent × trailing minutes / 48; the leg that reacts to injuries | each player weighted by 1 − P(out), forecast from information available at the open. Requires the daily injury report; this is why the model frame starts 2019-20 |
| Schedule layer | home edge, back-to-backs, dead-team flags | estimated walk-forward, EB shrinkage `n/(n+600)` toward a prior, `team_home_ridge=200` on per-team home deviations. The only component that has survived strict out-of-sample testing on every split tried. |
| Tank term | late-season effort | exactly zero outside its window |
| Offset layer | the correction applied to the opening line | ridge on three features knowable at the open, shrunk hard toward zero. Fitted edge coefficient 0.33–0.37 in every fold |

Regularisation, in four forms: L2 ridge on the ratings solve; empirical-Bayes shrinkage toward a prior in the schedule and tank layers; data augmentation via prior-season pseudo-observations; and Bayesian/EB priors in the usage and props fits. The link scale 7.2 is the frozen production constant. It is a plug-in rather than a tuned one — matching a logistic to the margin-residual normal gives 7.53 from training residuals alone, and re-deriving it inside each fold is worth about 0.0002 nats, not significant. Every log-loss comparison in this document recalibrates each source walk-forward on prior seasons, so no forecast is advantaged by another's scale.

Inputs are four, and all public: `nba_api` box scores, the daily NBA injury report (PDF ingest), DARKO talent ratings, and historical odds, the last used by the offset layer and for scoring but never by the market-blind model. Measured: the entire purchasable stack (professional minutes projections, tracking feeds, premium talent ratings) is worth +0.0012 of log loss combined, not significant. Fitting is restricted to the `002` prefix, 35,546 regular-season games.

Player-, lineup- and matchup-level alternatives were built and gated rather than skipped — defensive RAPM, defended-FG% by shot category, Synergy play-types, a player skill posterior and 151,914 lineup stints among them. None improved significantly on the shipped DARKO-based availability composition, which is the one player-level term that survived. The full inventory and each gate result are in the repository.

**Accuracy.** Normalized gap is `(ll_us − ll_mkt) / (ln2 − ll_mkt)`, the share of the closing line's skill-above-a-coinflip the model fails to capture; zero is parity. On the fully-covered frame (2019-20 → 2025-26, K=7, n=8,286) the market-blind model sits 13.59% behind the market closing line, and the closing line beats it in every season of the frame.

![log loss](../charts/logloss_season_4way.png)

*Log loss by season, all four forecasts on identical games, each converted to a probability with its own walk-forward scale. Lower is better. The closing line is below everything else in six of seven seasons; the market-blind model is above the opening line in six of seven. Series are distinguished by line style as well as colour.*

| season | opening line | closing line | offset construction | market-blind model |
|---|---|---|---|---|
| 2019-20 (COVID) | 0.6102 | 0.6096 | 0.6082 | 0.6158 |
| 2020-21 (COVID) | 0.6257 | 0.6174 | 0.6266 | 0.6376 |
| 2021-22 | 0.6170 | 0.6039 | 0.6153 | 0.6220 |
| 2022-23 | 0.6326 | 0.6239 | 0.6304 | 0.6359 |
| 2023-24 | 0.5923 | 0.5825 | 0.5911 | 0.5993 |
| 2024-25 | 0.6040 | 0.5809 | 0.5947 | 0.5915 |
| 2025-26 | 0.5801 | 0.5727 | 0.5786 | 0.5875 |
| **pooled (n=8,286)** | **0.6084** | **0.5980** | **0.6059** | **0.6122** |

**Pooled, the market-blind model is the only one of the four with worse log loss than the opener.** Per season the ordering is not uniform: the blind model is above the opener in six of seven, and the offset construction is above it in 2020-21. It does not beat the price it would transact at, and it is beaten decisively by the close. Expressed as the share of open-to-close information recovered, the offset construction captures +24.1% and the market-blind model −36.4%, moving the wrong way. That result is the reason the strategy is built as a correction to the line rather than as a forecast compared against it. Measured against the closing line alone the normalized gap is 13.59%, and the two COVID seasons are the extremes in both directions — 2019-20 the model's best on this frame and 2020-21 its worst. Both are kept because they are fully injury-covered, which is the frame's only criterion.

Two further diagnostics, both on all games rather than the selected book. Against the opening spread the market-blind margin covers 50.65%, above a coin flip and significant, and below the 52.38% a −110 book charges, also significant. We disagree with the opening line by 2.455 points on average; if that disagreement were entirely real information we would cover 57.6%, so the genuine content is 0.206 points — 8.4% of the raw disagreement, and about 27% of the 0.751 points needed to clear the vig. That figure and the offset ridge's 0.33–0.37 coefficient are measured on different denominators and different frames, so they are reported separately rather than as corroboration. Closing-line value is positive and significant across 19 seasons, so the disagreement does carry information relative to the market's final price, by a margin smaller than the cost of transacting it.

## Considerations

### Reporting frames

Two frames, each defined by what data exists. **Model accuracy:** 2019-20 → 2025-26 — the daily injury report the availability leg is built on begins 2018-12-17, so 2019-20 is the first fully covered season. **Betting:** headline on 2023-24 → 2025-26, the era of the measured multi-book panel, with the full seven-season frame reported beside it in the same table.

### Execution

Model held fixed and only the transacted price varied, on the 2023-26 book.

| execution | 2023-26 ROI | status |
|---|---|---|
| 1 retail book | +10.63% | degenerate reference |
| 2 books | +12.91% | measured in 2023-24 only |
| 5 books | +15.31% | firm baseline |
| **9 books** | **+16.62%** | **max books observed, the reported tier** |
| exchange, 2% commission | +14.72% | arithmetic, no exchange data held |

The ladder flattens fast: 36% of the time two books post the same number, so extra books duplicate rather than add.

### Methodology

- **Walk-forward, no lookahead.** Configuration chosen on seasons 1..k from a pre-declared space, scored on k+1 only.
- **Settlement-only fills.** Priced at the recorded opening spread, held to settlement at −110. No queue model, no partial fill, no re-pricing.
- **Bet-time information only.** Availability is built from the last injury report published *before* game day, never the 5PM report or the pregame inactive list, both of which post after the opening line.
- **Regular season only.** Filtered on the `002` prefix. NBA Cup group-stage games are in because they count in the standings; a pre-registered difference-in-differences found no detectable effect (MDE 2.21 points).
- **Season-clustered inference,** checked across rolling-origin, leave-one-season-out, block-bootstrap and legacy splits, corrected for multiple comparisons across the running family.

### Caveats

- **Simulated.** The model is not in production and no capital has been deployed.
- **Best-of-9 pricing is observed in one season and inferred in two.** A measured multi-book panel exists for 2023-24 only, at 7.74 books per game; 2024-25 observes 1.00 and 2025-26 observes 1.03, so their price comes from a shopping law applied to one recorded book.
- **Best-of-N transacts at whichever book is furthest offside.** 8.1% of best prices sit more than 1.5 points off the next book — the ones that get limited, lowered or voided. Charging for them costs about 2.3 ROI points, and that charge captures none of the staking limits or account restrictions that arrive faster in practice.
- **2026-27 is the first genuinely out-of-sample season.** The betting configuration is re-selected walk-forward, but the model architecture was chosen on a 2021-26 corpus and handed to every step as fixed.
- Exchange fees, state taxes and account limits are not modelled. Full method, gates and the register of rejected work are in the repository.

### Next steps

- **Capture at least two books at the open, from opening night.** Best-of-two lifts CLV by about 49%, and taking the worse book erases nearly all of it. The logger has never run in-season, so if it is not up on opening night the season's open-price CLV record is lost and cannot be reconstructed.
- **Run both arms prospectively in 2026-27.** The offset construction is the primary and the market-blind model is its live control; they share 65% of their bets and take the same side on every shared game, so this is a controlled comparison rather than a portfolio.
- **CLV as the live yardstick.** Significant across 19 seasons and positive in 17, in units that need no devig convention, and it resolves in weeks where ROI needs decades.
- **Exchange access, if it becomes available.** The largest single lever measured here and an access problem rather than a modelling one.
- **Props before sides.** Both shipped improvements of the last cycle came from the props engine, and soft books are softer on props.

Not next: further feature search on the sides model. The question the project turns on is whether live CLV against opening prices, with two books captured, behaves as the backtest says.
