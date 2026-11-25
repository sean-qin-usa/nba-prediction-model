# The betting engine

This page asks what happens when every rule on the [strategies page](STRATEGIES.md) is used at once. The engine bets by size of disagreement, by time of season and by side, and sizes each stake to the estimated edge. Each season is bet with rules learned from earlier seasons.

Per unit staked the engine returns about +10%, and between +1% and +16% in every one of the 11 seasons. Compounded at quarter Kelly that is a median season of +65% of bankroll, with all 11 seasons profitable and the weakest at +5.9%. The season returns are the same edge multiplied by stake size, and all of it assumes the opening line can be bet.

![return per unit staked by season](../charts/engine_per_unit.png)

![season return by season](../charts/engine_by_season.png)

## How it works

The candidates are the 6,442 games from 2012-13 to 2025-26 where the spread model or the totals model is 1.5 or more points from the opening line. Each model's prediction for a season comes from a fit on earlier seasons, and that includes the weights that combine the spread model's five pieces, which are refitted before each season on the seasons before it ([how the pieces are weighted](STRATEGIES.md#how-the-pieces-are-weighted)). The spread model knows who is expected to play from the injury-report feed where one exists (December 2018 on) and from a synthetic feed built from box scores before that; the [strategies page](STRATEGIES.md#knowing-who-plays) shows what each is worth.

For each test season, a regression is fitted on the candidates of earlier seasons. It estimates whether a bet wins from the size of the gap, the market, the part of the season and the side taken. The stake is a quarter of the Kelly stake that probability implies. There is no bet at or below break-even, and no more than 25% of the bankroll is at risk on one night.

The test seasons are 2015-16 to 2025-26, and nothing from a test season is used to bet it.

## Results at quarter Kelly

| Engine | Median season | Worst season | Largest drawdown | ROI per unit staked | Bets per season |
|---|---|---|---|---|---|
| Gap size only | +39.8% | +1.0% | 30% | +9.9% | 404 |
| + time of season | +63.8% | -0.1% | 31% | +9.0% | 347 |
| + side | +47.4% | +19.7% | 32% | +11.1% | 363 |
| All rules | +64.6% | +5.9% | 32% | +10.0% | 336 |

The seasonal rules beat the gap-only engine in 7 of 11 seasons, and all rules together beat it in 8. The means sit well above the medians because 2023-24 and 2024-25 were large: the all-rules mean is +106.4%, and without 2024-25 the mean is +72.3% and the median +58.1%.

Larger stakes raise both the return and the risk. At half Kelly the all-rules median is +117% with a 55% drawdown. At full Kelly it is +256%, with a 74% drawdown and a worst season of -47%.

| Season | Bets | Cover | ROI per unit staked | Season return | Largest drawdown |
|---|---|---|---|---|---|
| 2015-16 | 207 | 58.5% | +13.2% | +51.6% | 13% |
| 2016-17 | 227 | 56.8% | +8.7% | +51.0% | 13% |
| 2017-18 | 296 | 51.0% | +1.0% | +5.9% | 17% |
| 2018-19 | 204 | 58.8% | +14.2% | +48.1% | 22% |
| 2019-20 | 251 | 53.8% | +9.6% | +42.2% | 27% |
| 2020-21 | 317 | 58.0% | +12.4% | +93.4% | 12% |
| 2021-22 | 393 | 56.5% | +6.8% | +75.9% | 32% |
| 2022-23 | 252 | 57.1% | +12.2% | +64.6% | 19% |
| 2023-24 | 545 | 54.3% | +10.7% | +207.0% | 18% |
| 2024-25 | 440 | 58.4% | +15.6% | +447.3% | 20% |
| 2025-26 | 569 | 55.5% | +5.1% | +83.2% | 28% |

## Why recent seasons are larger

The two largest returns are 2023-24 and 2024-25, and in a backtest that deserves suspicion. An earlier version of this page had a real problem here: the spread model's pieces were combined as they had been designed on 2021-22 to 2025-26, and the spread rule covered 60.2% in those seasons against 56.2% before. The pieces are now weighted from earlier seasons only, and that gap is gone, with the rule covering 56.2% from 2021-22 on and 57.7% in the nine seasons before. What still carries hindsight is the choice of the pieces themselves, which was made with recent seasons in view. The totals model was specified in November 2026 before its results were looked at.

What is left is volume, not a higher return per bet: per unit staked, 2023-24 earned 10.7% and 2024-25 15.6%, against 13.2% in 2015-16 and 14.2% in 2018-19.

![bets and return per unit staked by season](../charts/engine_bets_and_roi.png)

The engine placed 518 bets a season from 2023-24 on, against 268 before, as opening totals began to sit further from the model, at a slightly larger stake on each (2.1% of the bankroll against 1.7%). Total stakes over a season therefore rose from roughly five times the bankroll to about eleven, and because each stake is a share of the current bankroll, the same edge compounds into a far larger season. The stake sizes are also why these figures should not be read as achievable: opening lines carry low limits, and nobody gets eleven times their bankroll down on them in a season.

Profit per bet barely changed. Across the two flat-stake rules, bets covered 55.9% for +6.7% per bet up to 2022-23 and 55.3% for +5.6% from 2023-24 on, while the number of bets a season went from 184 to 395, most of the increase in totals. Two checks rule out the obvious objections. Rerunning the totals model on opening lines from a second source gives the same volume and a 55.4% cover rate, so the extra bets are not a quirk of one data feed; and a totals model fitted on 2007-17 and never updated still covers 56.1% in 2023-26, on 227 bets a season against 157 before, so they are not a fit to recent seasons either. The volume is still the part least likely to carry over: so far this season 11% of totals reach the 2-point threshold at the line actually on screen, against 25% in the backtest's recent seasons.

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
| Spread, gap 2, Oct-Nov | 56.9% | Yes |
| Spread, gap 3, Oct-Nov | 60.2% | Yes |
| Spread, gap 2, April | 54.5% | Yes |

These are the patterns on the strategies page, except that here each was learned from earlier seasons and then applied to the next one.

## A simpler engine

The simpler engine groups bets by market and gap size. It estimates each group's win rate from earlier seasons, pulls that estimate strongly toward break-even, and bets the groups that still clear it. It stakes less and places 195 bets a season.

| | Median season | Mean | Volatility (sd) | Worst | Losing seasons | ROI per unit staked |
|---|---|---|---|---|---|---|
| Totals + spread | +12.8% | +28.3% | 34.3% | -12.4% | 1 of 11 | +10.2% |
| Totals only | +9.0% | +19.8% | 26.7% | -3.1% | 2 of 11 | +9.8% |
| Spread only | +7.2% | +6.5% | 11.8% | -9.5% | 3 of 11 | +8.8% |

## Where the +65% comes from

The figure is leverage, not extra skill: ROI per unit staked is about +10% in both the full engine and the simple one. The full engine returns more because it places 336 bets a season at larger stakes. Adding the seasonal and side rules does not improve the win-probability forecast itself, whose log loss goes from 0.68729 to 0.68735.

More bets did not mean more profit per bet. Bets per season more than doubled after 2022, as opening totals began moving more and the lower-gap groups, which win about 54%, came into play.

The figure also assumes every opening line is bet at -110 with 2 to 4% of the bankroll. First lines carry low limits, and most of their move toward the close happens overnight and on game day.

The engine is not safe from a bad year. In 2017-18 it won 51.0% of its bets, earned 1.0% per unit staked and finished up only 5.9% after a 17% drawdown; the seasonal version lost 0.1% that season and the simpler engine lost 12.4%.

None of this has been shown live; the paper views began recording the price actually available on 2026-11-16.
