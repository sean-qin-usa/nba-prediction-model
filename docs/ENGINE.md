# The betting engine

This page asks what happens when every rule on the [strategies page](STRATEGIES.md) is used at once. The engine bets by size of disagreement, by time of season and by side, and sizes each stake to the estimated edge. Each season is bet with rules learned from earlier seasons.

The median season return is +54% of bankroll at quarter Kelly, with no losing season in 11. Most of that comes from staking more on the same edge. All of it assumes the opening line can be bet.

![season return by season](../charts/engine_by_season.png)

## How it works

The candidates are the 6,230 games from 2012-13 to 2025-26 where the spread model or the totals model is 1.5 or more points from the opening line. Each model's prediction for a season comes from a fit on earlier seasons.

For each test season, a regression is fitted on the candidates of earlier seasons. It estimates whether a bet wins from the size of the gap, the market, the part of the season and the side taken. The stake is a quarter of the Kelly stake that probability implies. There is no bet at or below break-even, and no more than 25% of the bankroll is at risk on one night.

The test seasons are 2015-16 to 2025-26, and nothing from a test season is used to bet it.

## Results at quarter Kelly

| Engine | Median season | Worst season | Largest drawdown | ROI per unit staked | Bets per season |
|---|---|---|---|---|---|
| Gap size only | +40.2% | +4.7% | 29% | +10.7% | 332 |
| + time of season | +57.1% | +5.3% | 27% | +9.9% | 310 |
| + side | +36.7% | +15.7% | 31% | +10.5% | 316 |
| All rules | +53.8% | +13.7% | 32% | +10.0% | 302 |

The seasonal rules beat the gap-only engine in 9 of 11 seasons, and all rules together beat it in 8. The means sit well above the medians because the last three seasons were large: the all-rules mean is +84.6%, and without 2024-25 the mean is +63.0% and the median +50.2%.

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
| 2023-24 | 514 | 54.5% | +12.3% | +205.3% | 18% |
| 2024-25 | 429 | 58.5% | +12.4% | +300.4% | 23% |
| 2025-26 | 518 | 54.2% | +5.9% | +80.2% | 27% |

## Why the last three seasons are so large

The three largest returns are the three most recent seasons, which can look like a model fitted to recent data. It is not that: every weight used to bet a season was learned from earlier seasons only, and the return per unit staked in 2023-24 and 2024-25, at about +12%, is in line with 2022-23 and well below 2019-20.

What changed is volume. The engine placed 487 bets a season from 2023-24 on, against 233 before, as opening totals began to sit further from the model, and it staked about 2% of the bankroll on each where it had staked about 1.6%. Total stakes over a season therefore rose from roughly four times the bankroll to about ten, and because each stake is a share of the current bankroll, the same edge compounds into a far larger season. The stake sizes are also why these figures should not be read as achievable: opening lines carry low limits, and nobody gets ten times their bankroll down on them in a season.

## A repair to the 2024-25 opening lines

The opening lines for 2024-25 come from a single book, read through ESPN's odds feed. In October and November 2024 the feed's opening field often held an alternate line instead of the main one, and the price attached to it gives this away: a main line opens near -110 on both sides, while these opened anywhere from -450 to +290. By that test 40% of the opening totals and 21% of the opening spreads in those two months were not the main line, against under 1% from December on and none in 2025-26.

A model sides against a line that is 7 points off and usually wins, so left alone those games would have made 2024-25 look several times better than it was. Each one (121 totals and 70 spreads) was replaced with the opening line from a second source, Action Network, which sits within about a point of ESPN's on ordinary games, and the few with no second opener were left out. The models were then refitted season by season on the repaired lines, and the figures on this page and the strategies page come from that rerun.

## What it learned

Fitted on all seasons, the engine assigns these win rates to typical bets.

| Bet | Win rate | Bet? |
|---|---|---|
| Totals over, gap 2, Dec-Jan | 54.3% | Yes |
| Totals over, gap 2, Oct-Nov | 50.3% | No |
| Totals under, gap 2, Dec-Jan | 57.4% | Yes |
| Totals over, gap 4, Dec-Jan | 60.4% | Yes |
| Spread, gap 2, Oct-Nov | 55.8% | Yes |
| Spread, gap 3, Oct-Nov | 60.1% | Yes |
| Spread, gap 2, April | 48.0% | No |

These are the patterns on the strategies page, except that here each was learned from earlier seasons and then applied to the next one.

## A simpler engine

The simpler engine groups bets by market and gap size. It estimates each group's win rate from earlier seasons, pulls that estimate strongly toward break-even, and bets the groups that still clear it. It stakes less and places 172 bets a season.

| | Median season | Mean | Volatility (sd) | Worst | Losing seasons | ROI per unit staked |
|---|---|---|---|---|---|---|
| Totals + spread | +19.6% | +27.1% | 31.7% | -10.2% | 2 of 11 | +12.5% |
| Totals only | +9.0% | +19.8% | 26.7% | -3.1% | 2 of 11 | +9.8% |
| Spread only | +4.1% | +5.6% | 9.6% | -9.6% | 3 of 11 | +19.6% |

## Where the +54% comes from

The figure is leverage, not extra skill: ROI per unit staked is about +10% in the full engine and +12.5% in the simple one. The full engine returns more because it places 302 bets a season at larger stakes. Adding the seasonal and side rules barely improves the win-probability forecast itself, whose log loss moves from 0.68770 to 0.68759.

More bets did not mean more profit per bet. Bets per season more than doubled after 2022, as opening totals began moving more and the lower-gap groups, which win about 54%, came into play.

The figure also assumes every opening line is bet at -110 with 2 to 4% of the bankroll. First lines carry low limits, and the most mispriced ones move within minutes.

The most recent season was among the weakest for skill. In 2025-26 the engine won 54.2% of its bets and returned +5.9% per unit staked, with a 27% drawdown.

None of this has been shown live; the paper views began recording the price actually available on 2026-11-16.
