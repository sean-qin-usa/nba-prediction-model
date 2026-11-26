# NBA Prediction Model

A market-anchored NBA model for spreads and totals, together with the research record behind it. The shipped spread forecast is an ensemble: about two-thirds the market's opening line and one-third a market-blind model built from player ratings, minutes and availability. The model does not out-forecast the market on average. What it does is pick out the games where the opening line is furthest off, and those are the only games it bets.

## Backtest results

| | |
|---|---|
| +65% | Median season return of the full betting engine at quarter Kelly, across 11 seasons it had not seen; all 11 were profitable, the weakest at +5.9%. |
| 56.3% | Share of the engine's bets won, against a break-even of 52.4%, for a return of about +10% per unit staked on 336 bets a season. |
| 56.9% | Share of spread-rule bets won when the model is 2 or more points from the opening spread, over 829 bets from 2012 to 2026, with every weight in the model learned from earlier seasons. |
| 55.3% | Share of totals-rule bets won when the model is 2 or more points from the opening total, over 2,380 bets; it beat break-even in 12 of 14 seasons and won 57.4% on five further seasons it was never tuned on. |

These are backtests that assume the opening line is bet, which has not been shown live, and at the prices available later in the day the edge disappears. No capital has been deployed on any of it.

Every season is scored with weights learned only from the seasons before it, including the weights that combine the spread model's pieces, so no weight was fitted on the season it scores; the choice of those pieces was still made with recent seasons in view. Per unit staked the engine earned between +1% and +16% in each of them. Compounded at quarter Kelly, 2023-24 and 2024-25 come out much the largest (+207% and +447%) because the engine placed twice as many bets in them; the [engine page](docs/ENGINE.md#why-recent-seasons-are-larger) sets that out, along with a repair made to the 2024-25 opening lines.

![season return by season](charts/engine_by_season.png)

![return per unit staked by season](charts/engine_per_unit.png)

## This season, 2026-27

Paper bets logged as they happened, through November 25 (269 games played), with no real money staked.

| Bet type | Record | Win rate | Units | Per bet | Beat the close by |
|---|---|---|---|---|---|
| Totals (since Nov 16) | 4-2 | 66.7% | +1.65 | +0.27 | -0.3 points |
| Day-before spread (since Nov 16) | 2-4 | 33.3% | -2.25 | -0.38 | +1.0 points |
| Moneyline (since Oct 21) | 167-52 | 76.3% | +16.42 | +0.07 | none |

Break-even is 52.4% for totals and spreads and 71.7% for the moneyline book, which backs favourites. These samples are too small to settle anything; the [season page](docs/SEASON_2026_27.md#live-paper-views-from-november-16) lists every bet, along with replays of the weeks before the live views started.

## Pages

| Page | What it covers |
|---|---|
| [Betting engine](docs/ENGINE.md) | Every rule combined, each season bet with rules learned from earlier seasons |
| [Strategies](docs/STRATEGIES.md) | Each rule's results by month and by size of disagreement, with tested explanations |
| [2026-27 season](docs/SEASON_2026_27.md) | This season out of sample: log loss against the market, every bet type, the live book |
| [Model overview](docs/OVERVIEW.md) | What the model is, data limits, the 2019-26 record, how to run it |
| [Three-page review (PDF)](nba_model_and_strategy_review.pdf) | Summary of the system and its limits |
| [Decision register](docs/DECISIONS.md) | Every decision and rejection, in order |
| [Limitations](docs/LIMITATIONS.md) | What the results do not show |

## Status

Two paper views began on 2026-11-16, and for every spread and total they record the opening price, the price actually available and the close. A second totals arm, with coefficients refitted on the repaired 2024-25 opening lines, is logged beside the frozen one, and the day-before spread view logs a second arm that combines the model's pieces with the walk-forward weights used in the backtest. Nothing has been bet with real money.

## Licence & disclaimer

For research and educational purposes. Nothing here is betting advice, and the project's own conclusion is that nothing in it is statistically resolved.
