# NBA Prediction Model

A market-anchored NBA model for spreads and totals, together with the research record behind it. The model does not out-forecast the market on average. What it does is pick out the games where the opening line is furthest off, and those are the only games it bets.

## Results

| | |
|---|---|
| +54% | Median season return of the full betting engine at quarter Kelly, across 11 seasons it had not seen, none of which lost. |
| 56.1% | Share of the engine's bets won, against a break-even of 52.4%, for a return of about +10% per unit staked on 304 bets a season. |
| 57.4% | Share of spread-rule bets won when the model is 2 or more points from the opening spread, over 800 bets from 2012 to 2026 (56.4% without the mislabeled openers described below). |
| 55.6% | Share of totals-rule bets won when the model is 2 or more points from the opening total, over 2,449 bets; it beat break-even in 12 of 14 seasons and won 55.8% on five further seasons it was never tuned on (55.2% on the main sample without the mislabeled openers). |
| 28-17 | Totals rule this season, at the opening total, through 2026-11-15. |

These are backtests and replays that assume the opening line is bet, which has not been shown live, and at the prices available later in the day the edge disappears. No capital has been deployed on any of it.

One season carries a correction. Opening lines for October and November 2024 were partly mislabeled in the source feed, which had inflated the engine's 2024-25 return to +1,124%; with those games removed it is +215%, and the median season is unchanged. The [engine page](docs/ENGINE.md#why-2024-25-looked-so-large) shows why this was a data fault and not overfitting.

![season return by season](charts/engine_by_season.png)

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

Two paper views began on 2026-11-16, and for every spread and total they record the opening price, the price actually available and the close. They will be read at 40 spread bets and 60 totals bets.

## Licence & disclaimer

For research and educational purposes. Nothing here is betting advice, and the project's own conclusion is that nothing in it is statistically resolved.
