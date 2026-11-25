# NBA Prediction Model

A market-anchored NBA model for spreads and totals, together with the research record behind it. The shipped spread forecast is an ensemble: about two-thirds the market's opening line and one-third a market-blind model built from player ratings, minutes and availability. The model does not out-forecast the market on average. What it does is pick out the games where the opening line is furthest off, and those are the only games it bets.

## Backtest results

| | |
|---|---|
| +57% | Median season return of the full betting engine at quarter Kelly, across 11 seasons it had not seen, ten of them profitable. |
| 56.1% | Share of the engine's bets won, against a break-even of 52.4%, for a return of about +9% per unit staked on 344 bets a season. |
| 58.3% | Share of spread-rule bets won when the model is 2 or more points from the opening spread, over 753 bets from 2012 to 2026. |
| 55.3% | Share of totals-rule bets won when the model is 2 or more points from the opening total, over 2,380 bets; it beat break-even in 12 of 14 seasons and won 57.4% on five further seasons it was never tuned on. |

These are backtests that assume the opening line is bet, which has not been shown live, and at the prices available later in the day the edge disappears. No capital has been deployed on any of it.

The last three seasons are much the largest because the engine placed twice as many bets in them, not because it was fitted to them; the [engine page](docs/ENGINE.md#why-the-last-three-seasons-are-so-large) explains this, along with a repair made to the 2024-25 opening lines.

![season return by season](charts/engine_by_season.png)

## This season, 2026-27

The table runs through November 24, after 260 games; live rows are paper bets logged as they happened, and replay rows were reconstructed afterwards from the odds and injury reports stored at the time.

| Rule | How recorded | Record | Win rate | Units | Units per bet | Points ahead of the close |
|---|---|---|---|---|---|---|
| Totals, 2+ points from the line on screen | Live since Nov 16 | 4-2 | 66.7% | +1.65 | +0.27 | -0.3 |
| Totals, 2+ points from the opening total | Replay, Oct 20 to Nov 15 | 28-17 | 62.2% | +8.45 | +0.19 | +1.9 |
| Totals, 2+ points from the first line seen | Replay, Oct 20 to Nov 15 | 18-17 | 51.4% | -0.63 | -0.02 | +0.1 |
| Day-before spread, 5+ points from the line | Live since Nov 16 | 1-4 | 20.0% | -3.06 | -0.61 | +1.2 |
| Day-before spread, 5+ points from the line | Replay, Nov 5 to Nov 15 | 2-4 | 33.3% | -2.10 | -0.35 | +1.9 |
| Day-before spread, 5+ points from the line | Replay, Oct 21 to Nov 4; the rule was chosen on these games | 7-2 | 77.8% | +4.36 | +0.48 | +3.3 |
| Moneyline book, three views of 74 games | Live since Oct 21 | 160-52 | 75.5% | +13.82 | +0.07 | none |

A spread or totals bet at the usual price breaks even at 52.4%, while the moneyline book backs favourites and needs 71.7% at the prices taken, so its win rate is not comparable with the others. The season so far repeats the backtest's warning: the totals rule wins at the opening number and breaks even at the number actually available, and the spread rule is 3-8 on games it was not chosen on. The moneyline book has profited from favourites winning, without beating the closing price. Every sample here is too small to settle anything, and the live views will be read at 60 totals bets and 40 spread bets; the bet-by-bet record is on the [season page](docs/SEASON_2026_27.md#live-paper-views-from-november-16).

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

Two paper views began on 2026-11-16, and for every spread and total they record the opening price, the price actually available and the close. A second totals arm, with coefficients refitted on the repaired 2024-25 opening lines, is logged beside the frozen one, and nothing has been bet with real money.

## Licence & disclaimer

For research and educational purposes. Nothing here is betting advice, and the project's own conclusion is that nothing in it is statistically resolved.
