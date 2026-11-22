# The betting engine

This page asks what happens when every rule on the [strategies page](STRATEGIES.md) is used at once. The engine bets by size of disagreement, by time of season and by side, and sizes each stake to the estimated edge. Each season is bet with rules learned from earlier seasons.

The median season return is +54% of bankroll at quarter Kelly, with no losing season in 11. Most of that comes from staking more on the same edge. All of it assumes the opening line can be bet.

![season return by season](../charts/engine_by_season.png)

## How it works

The candidates are the 6,378 games from 2012-13 to 2025-26 where the spread model or the totals model is 1.5 or more points from the opening line. Each model's prediction for a season comes from a fit on earlier seasons.

For each test season, a regression is fitted on the candidates of earlier seasons. It estimates whether a bet wins from the size of the gap, the market, the part of the season and the side taken. The stake is a quarter of the Kelly stake that probability implies. There is no bet at or below break-even, and no more than 25% of the bankroll is at risk on one night.

The test seasons are 2015-16 to 2025-26, and nothing from a test season is used to bet it.

## Results at quarter Kelly

| Engine | Median season | Worst season | Largest drawdown | ROI per unit staked | Bets per season |
|---|---|---|---|---|---|
| Gap size only | +40.2% | +4.7% | 30% | +11.0% | 340 |
| + time of season | +54.3% | +5.3% | 27% | +10.0% | 323 |
| + side | +36.7% | +15.7% | 31% | +10.8% | 325 |
| All rules | +53.8% | +13.7% | 32% | +10.0% | 314 |

The seasonal rules beat the gap-only engine in 9 of 11 seasons, and all rules together beat it in 8. One season needed a correction. As first recorded, 2024-25 returned over +1,100%, because a fifth of its bets were graded against opening lines that were never the main line; [the section below](#why-2024-25-looked-so-large) explains what went wrong and why it was a data fault, not overfitting. With those games removed the season returns +215%, and the all-rules median is unchanged at +53.8%. The summary table above is as first recorded: for the all-rules engine the correction lowers ROI per unit staked from +10.0% to +9.7% and bets per season from 314 to 304, and leaves the worst season and the largest drawdown where they were.

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
| 2024-25 (mislabeled openers removed) | 396 | 57.8% | +11.2% | +215.2% | 23% |
| 2025-26 | 585 | 53.5% | +4.5% | +70.6% | 30% |

## Why 2024-25 looked so large

As first recorded, 2024-25 returned +1,124%. That was not overfitting, and it was not the model knowing more than usual: every weight used to bet that season was learned from earlier seasons, and the same models, with one more season of data behind them, had their weakest year in 2025-26. The cause was bad data in the opening lines.

The opening lines for 2024-25 come from a single book, read through ESPN's odds feed. In October and November 2024 the feed's opening field often held an alternate line instead of the main one, and the price attached to it gives this away: a main line opens near -110 on both sides, while these opened anywhere from -450 to +290. By that test 40% of the opening totals and 21% of the opening spreads in those two months were not the main line, against under 1% from December on and none in 2025-26. The mislabeled totals sat 7.5 points from the opening total at a second source, where ordinary ones sit about 1 point away.

A model anchored on how teams have actually been scoring will side against a line that is 7 points off, and it will usually win. The engine placed 101 bets on those openers and won 75% of them, and quarter-Kelly staking compounded the run.

| 2024-25 | Bets | Cover | ROI per unit staked | Season return |
|---|---|---|---|---|
| As first recorded | 497 | 61.4% | +14.7% | +1,123.5% |
| Mislabeled openers removed | 396 | 57.8% | +11.2% | +215.2% |

The season return falls much further than the cover rate because stakes compound. With the same bets and stakes, a 56% cover rate gives a median season of +89% and a 61.4% cover rate gives +514%, so a few extra points of cover multiply the result several times over.

The correction removes the mislabeled games both from the seasons the engine learns from and from the season it bets. It leaves the median season at +53.8% and the worst at +13.7%, lowers the mean season from +158.5% to +76.2%, and moves 2025-26 from +70.6% to +72.9%, since that season's weights were learned partly from 2024-25. A few alternate lines priced close to -110 probably remain in those two months, so +215% is better read as an upper bound for 2024-25 than as a clean figure.

## What it learned

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

These are the patterns on the strategies page, except that here each was learned from earlier seasons and then applied to the next one.

## A simpler engine

The simpler engine groups bets by market and gap size. It estimates each group's win rate from earlier seasons, pulls that estimate strongly toward break-even, and bets the groups that still clear it. It stakes less and places 182 bets a season. The table is as first recorded and still includes the mislabeled 2024-25 openers, which lift the means far more than the medians: without that season the two-market mean is +19.2% and the median +14.1%.

| | Median season | Mean | Volatility (sd) | Worst | Losing seasons | ROI per unit staked |
|---|---|---|---|---|---|---|
| Totals + spread | +17.3% | +40.3% | 73.4% | -10.2% | 2 of 11 | +12.6% |
| Totals only | +9.0% | +25.8% | 41.5% | -3.1% | 2 of 11 | +10.0% |
| Spread only | +4.1% | +8.2% | 16.3% | -9.6% | 3 of 11 | +20.1% |

## Where the +54% comes from

The figure is leverage, not extra skill: ROI per unit staked is about +10% in the full engine and +12.6% in the simple one. The full engine returns more because it places 314 bets at about twice the stake. Adding the seasonal and side rules barely improves the win-probability forecast itself, whose log loss moves from 0.68690 to 0.68673.

More bets did not mean more profit per bet. Bets per season tripled after 2020, as opening totals began moving more and the lower-gap groups, which win about 54%, came into play.

The figure also assumes every opening line is bet at -110 with 2 to 4% of the bankroll. First lines carry low limits, and the most mispriced ones move within minutes.

The most recent season was the weakest for skill. In 2025-26 the engine won 53.5% of its bets and returned +4.5% per unit staked, with a 30% drawdown.

None of this has been shown live; the paper views began recording the price actually available on 2026-11-16.
