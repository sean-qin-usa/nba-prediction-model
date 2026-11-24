# The betting engine

This page asks what happens when every rule on the [strategies page](STRATEGIES.md) is used at once. The engine bets by size of disagreement, by time of season and by side, and sizes each stake to the estimated edge. Each season is bet with rules learned from earlier seasons.

The median season return is +57% of bankroll at quarter Kelly, with ten profitable seasons in 11. Most of that comes from staking more on the same edge. All of it assumes the opening line can be bet.

![season return by season](../charts/engine_by_season.png)

## How it works

The candidates are the 6,248 games from 2012-13 to 2025-26 where the spread model or the totals model is 1.5 or more points from the opening line. Each model's prediction for a season comes from a fit on earlier seasons. The spread model knows who is expected to play from the injury-report feed where one exists (December 2018 on) and from a synthetic feed built from box scores before that; the [strategies page](STRATEGIES.md#knowing-who-plays) shows what each is worth.

For each test season, a regression is fitted on the candidates of earlier seasons. It estimates whether a bet wins from the size of the gap, the market, the part of the season and the side taken. The stake is a quarter of the Kelly stake that probability implies. There is no bet at or below break-even, and no more than 25% of the bankroll is at risk on one night.

The test seasons are 2015-16 to 2025-26, and nothing from a test season is used to bet it.

## Results at quarter Kelly

| Engine | Median season | Worst season | Largest drawdown | ROI per unit staked | Bets per season |
|---|---|---|---|---|---|
| Gap size only | +42.2% | -7.7% | 30% | +10.4% | 402 |
| + time of season | +75.0% | -14.1% | 36% | +9.6% | 364 |
| + side | +44.9% | +1.9% | 34% | +9.6% | 368 |
| All rules | +57.3% | -10.8% | 34% | +9.0% | 344 |

The seasonal rules beat the gap-only engine in 8 of 11 seasons, and all rules together beat it in 7. The means sit well above the medians because the last three seasons were large: the all-rules mean is +102.5%, and without 2024-25 the mean is +69.8% and the median +55.0%.

Larger stakes raise both the return and the risk. At half Kelly the all-rules median is +127% with a 59% drawdown. At full Kelly it is +216%, with an 82% drawdown and a worst season of -66%.

| Season | Bets | Cover | ROI per unit staked | Season return | Largest drawdown |
|---|---|---|---|---|---|
| 2015-16 | 218 | 57.3% | +7.2% | +21.6% | 12% |
| 2016-17 | 266 | 55.6% | +10.3% | +52.7% | 13% |
| 2017-18 | 332 | 52.1% | -1.9% | -10.8% | 34% |
| 2018-19 | 229 | 53.3% | +3.0% | +9.3% | 25% |
| 2019-20 | 250 | 56.4% | +20.2% | +91.9% | 14% |
| 2020-21 | 270 | 58.1% | +11.1% | +57.3% | 11% |
| 2021-22 | 379 | 58.8% | +9.9% | +107.6% | 29% |
| 2022-23 | 245 | 57.6% | +8.9% | +41.5% | 22% |
| 2023-24 | 559 | 54.2% | +11.2% | +204.2% | 19% |
| 2024-25 | 479 | 58.2% | +12.4% | +429.3% | 26% |
| 2025-26 | 554 | 55.4% | +6.8% | +123.1% | 28% |

## Why the last three seasons are so large

The three largest returns are the three most recent seasons, which can look like a model fitted to recent data. It is not that: every weight used to bet a season was learned from earlier seasons only, and the return per unit staked in 2023-24 and 2024-25, at 11 to 12%, is in line with 2020-21 and well below 2019-20.

What changed is volume. The engine placed 531 bets a season from 2023-24 on, against 274 before, as opening totals began to sit further from the model, and it staked about 2.1% of the bankroll on each where it had staked about 1.5%. Total stakes over a season therefore rose from roughly four times the bankroll to about eleven, and because each stake is a share of the current bankroll, the same edge compounds into a far larger season. The stake sizes are also why these figures should not be read as achievable: opening lines carry low limits, and nobody gets eleven times their bankroll down on them in a season.

Profit per bet barely changed. Across the two flat-stake rules, bets covered 55.8% for +6.5% per bet up to 2022-23 and 56.3% for +7.4% from 2023-24 on, while the number of bets a season went from 177 to 396, most of the increase in totals. Two checks rule out the obvious objections. Rerunning the totals model on opening lines from a second source gives the same volume and a 55.4% cover rate, so the extra bets are not a quirk of one data feed; and a totals model fitted on 2007-17 and never updated still covers 56.1% in 2023-26, on 227 bets a season against 157 before, so they are not a fit to recent seasons either. The volume is still the part least likely to carry over: so far this season 11% of totals reach the 2-point threshold at the line actually on screen, against 25% in the backtest's recent seasons.

## A repair to the 2024-25 opening lines

The opening lines for 2024-25 come from a single book, read through ESPN's odds feed. In October and November 2024 the feed's opening field often held an alternate line instead of the main one, and the price attached to it gives this away: a main line opens near -110 on both sides, while these opened anywhere from -450 to +290. By that test 40% of the opening totals and 21% of the opening spreads in those two months were not the main line, against under 1% from December on and none in 2025-26.

A model sides against a line that is 7 points off and usually wins, so left alone those games would have made 2024-25 look several times better than it was. Each one (121 totals and 70 spreads) was replaced with the opening line from a second source, Action Network, which sits within about a point of ESPN's on ordinary games, and the few with no second opener were left out. The models were then refitted season by season on the repaired lines, and the figures on this page and the strategies page come from that rerun.

## What it learned

Fitted on all seasons, the engine assigns these win rates to typical bets.

| Bet | Win rate | Bet? |
|---|---|---|
| Totals over, gap 2, Dec-Jan | 54.4% | Yes |
| Totals over, gap 2, Oct-Nov | 50.3% | No |
| Totals under, gap 2, Dec-Jan | 57.5% | Yes |
| Totals over, gap 4, Dec-Jan | 60.4% | Yes |
| Spread, gap 2, Oct-Nov | 59.1% | Yes |
| Spread, gap 3, Oct-Nov | 63.1% | Yes |
| Spread, gap 2, April | 52.1% | No |

These are the patterns on the strategies page, except that here each was learned from earlier seasons and then applied to the next one.

## A simpler engine

The simpler engine groups bets by market and gap size. It estimates each group's win rate from earlier seasons, pulls that estimate strongly toward break-even, and bets the groups that still clear it. It stakes less and places 185 bets a season.

| | Median season | Mean | Volatility (sd) | Worst | Losing seasons | ROI per unit staked |
|---|---|---|---|---|---|---|
| Totals + spread | +13.7% | +35.9% | 58.2% | -1.5% | 1 of 11 | +11.4% |
| Totals only | +9.0% | +19.8% | 26.7% | -3.1% | 2 of 11 | +9.8% |
| Spread only | +4.0% | +11.6% | 26.3% | -6.9% | 1 of 11 | +14.7% |

## Where the +57% comes from

The figure is leverage, not extra skill: ROI per unit staked is about +9% in the full engine and +11.4% in the simple one. The full engine returns more because it places 344 bets a season at larger stakes. Adding the seasonal and side rules does not improve the win-probability forecast itself, whose log loss goes from 0.68647 to 0.68680.

More bets did not mean more profit per bet. Bets per season more than doubled after 2022, as opening totals began moving more and the lower-gap groups, which win about 54%, came into play.

The figure also assumes every opening line is bet at -110 with 2 to 4% of the bankroll. First lines carry low limits, and most of their move toward the close happens overnight and on game day.

The engine is not safe from a bad year. In 2017-18 it won 52.1% of its bets, lost 1.9% per unit staked and finished down 10.8% after a 34% drawdown, and the gap-only and seasonal versions each have a losing season too.

None of this has been shown live; the paper views began recording the price actually available on 2026-11-16.
