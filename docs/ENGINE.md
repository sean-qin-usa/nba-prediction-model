# The betting engine: every rule combined, tested on seasons it had not seen

This page asks what happens when every rule on the [strategies page](STRATEGIES.md) is used at once. The engine bets by size of disagreement, by time of season and by side, and sizes each stake to the estimated edge. Each season is bet with rules learned from earlier seasons.

The median season return is +54% of bankroll at quarter Kelly, with no losing season in 11. Most of that comes from staking more on the same edge. All of it assumes the opening line can be bet.

![season return by season](../charts/engine_by_season.png)

## How it works

The candidates are every game from 2012-13 to 2025-26 where the spread model or the totals model is 1.5 or more points from the opening line. There are 6,378. Each model's prediction for a season comes from a fit on earlier seasons.

For each test season, a regression is fitted on the candidates of earlier seasons. It estimates whether a bet wins from the size of the gap, the market, the part of the season and the side taken. The stake is a quarter of the Kelly stake that probability implies. There is no bet at or below break-even, and no more than 25% of the bankroll is at risk on one night.

The test seasons are 2015-16 to 2025-26. Nothing from a test season is used to bet it.

## Results at quarter Kelly

| Engine | Median season | Worst season | Largest drawdown | ROI per unit staked | Bets per season |
|---|---|---|---|---|---|
| Gap size only | +40.2% | +4.7% | 30% | +11.0% | 340 |
| + time of season | +54.3% | +5.3% | 27% | +10.0% | 323 |
| + side | +36.7% | +15.7% | 31% | +10.8% | 325 |
| All rules | +53.8% | +13.7% | 32% | +10.0% | 314 |

The seasonal rules beat the gap-only engine in 9 of 11 seasons, and all rules together beat it in 8. One season distorts the averages. 2024-25 returned over +1,100% and its opening lines look stale. Without it the all-rules median is +50.2% and the mean +62.0%.

Larger stakes raise both the return and the risk. At half Kelly the all-rules median is +99% with a 55% drawdown. At full Kelly it is +153%, with a 75% drawdown and a worst season of -30%.

| Season | Bets | Cover | ROI per unit staked | Season return | Largest drawdown |
|---|---|---|---|---|---|
| 2015-16 | 211 | 58.3% | +8.1% | +26.8% | 12% |
| 2016-17 | 228 | 55.7% | +8.6% | +41.2% | 21% |
| 2017-18 | 248 | 52.4% | +3.0% | +13.7% | 17% |
| 2018-19 | 160 | 56.2% | +5.7% | +15.2% | 26% |
| 2019-20 | 195 | 56.9% | +23.1% | +87.6% | 10% |
| 2020-21 | 251 | 59.4% | +10.7% | +53.8% | 11% |
| 2021-22 | 349 | 57.9% | +7.3% | +60.2% | 32% |
| 2022-23 | 219 | 58.0% | +12.6% | +46.6% | 20% |
| 2023-24 | 514 | 54.5% | +12.2% | +204.6% | 18% |
| 2024-25 | 497 | 61.4% | +14.7% | +1,123.5% | 23% |
| 2025-26 | 585 | 53.5% | +4.5% | +70.6% | 30% |

## What the engine learned

Fitted on all seasons, the engine assigns these win rates to typical bets.

| Bet | Win rate | Bet? |
|---|---|---|
| Totals over, gap 2, Dec-Jan | 54.0% | Yes |
| Totals over, gap 2, Oct-Nov | 50.0% | No |
| Totals under, gap 2, Dec-Jan | 57.7% | Yes |
| Totals over, gap 4, Dec-Jan | 60.9% | Yes |
| Spread, gap 2, Oct-Nov | 56.1% | Yes |
| Spread, gap 3, Oct-Nov | 61.3% | Yes |
| Spread, gap 2, April | 48.2% | No |

These are the patterns on the strategies page. Here each was learned from earlier seasons and then applied to the next one.

## A simpler engine for comparison

The simpler engine groups bets by market and gap size. It estimates each group's win rate from earlier seasons, pulls that estimate strongly toward break-even, and bets the groups that still clear it. It stakes less and places 182 bets a season.

| | Median season | Mean | Volatility (sd) | Worst | Losing seasons | ROI per unit staked |
|---|---|---|---|---|---|---|
| Totals + spread | +17.3% | +40.3% | 73.4% | -10.2% | 2 of 11 | +12.6% |
| Totals only | +9.0% | +25.8% | 41.5% | -3.1% | 2 of 11 | +10.0% |
| Spread only | +4.1% | +8.2% | 16.3% | -9.6% | 3 of 11 | +20.1% |

## How to read +54%

The figure reflects larger stakes more than extra skill. ROI per unit staked is about +10% in the full engine and +12.6% in the simple one. The full engine returns more because it places 314 bets at about twice the stake. Adding the seasonal and side rules barely improves the win-probability forecast itself, whose log loss moves from 0.68690 to 0.68673.

More bets did not mean more profit per bet. Bets per season tripled after 2020, as opening totals began moving more and the lower-gap groups came into play. Those groups win about 54%.

The figure also assumes every opening line is bet at -110 with 2 to 4% of the bankroll. First lines carry low limits. The most mispriced ones move within minutes.

The most recent season was the weakest for skill. In 2025-26 the engine won 53.5% of its bets and returned +4.5% per unit staked, with a 30% drawdown.

None of this has been shown live. The paper views began recording the price actually available on 2026-11-16.
