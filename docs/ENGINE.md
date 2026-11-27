# The betting engine

This page asks what happens when every rule on the [strategies page](STRATEGIES.md) is used at once. The engine bets by size of disagreement, by time of season and by side, and sizes each stake to the estimated edge. Each season is bet with rules learned from earlier seasons.

Per unit staked the engine returns about +9%, and between -3% and +18% across the 11 seasons. Compounded at quarter Kelly that is a median season of +53% of bankroll, with 10 of the 11 seasons profitable and 2017-18 down 15.9%. The season returns are the same edge multiplied by stake size, and all of it assumes the opening line can be bet.

![season return by season](../charts/engine_by_season.png)

![return per unit staked by season](../charts/engine_per_unit.png)

## How it works

The candidates are the 6,009 games from 2012-13 to 2025-26 where the spread model or the totals model is 1.5 or more points from the opening line. Each model's prediction for a season comes from a fit on earlier seasons, and that includes the weights that combine the spread model's five pieces, which are refitted before each season on the seasons before it ([how the pieces are weighted](STRATEGIES.md#how-the-pieces-are-weighted)). The spread model knows who is expected to play from the injury-report feed where one exists (December 2018 on) and from a synthetic feed built from box scores before that; the [strategies page](STRATEGIES.md#knowing-who-plays) shows what each is worth. Both models price a game with what was known the evening before it, when the opening line is posted, so the previous night's results are not used; an earlier version of this page did use them, and [the strategies page](STRATEGIES.md#priced-the-evening-before) shows what that was worth.

For each test season, a regression is fitted on the candidates of earlier seasons. It estimates whether a bet wins from the size of the gap, the market, the part of the season and the side taken. The stake is a quarter of the Kelly stake that probability implies. There is no bet at or below break-even, and no more than 25% of the bankroll is at risk on one night.

The test seasons are 2015-16 to 2025-26, and nothing from a test season is used to bet it.

## Results at quarter Kelly

| Engine | Median season | Worst season | Largest drawdown | ROI per unit staked | Bets per season |
|---|---|---|---|---|---|
| Gap size only | +35.2% | -1.4% | 34% | +9.0% | 365 |
| + time of season | +51.1% | -16.2% | 37% | +8.8% | 326 |
| + side | +46.2% | +4.1% | 40% | +10.1% | 327 |
| All rules | +53.2% | -15.9% | 36% | +9.4% | 311 |

The seasonal rules beat the gap-only engine in 8 of 11 seasons, and all rules together beat it in 9. The means sit well above the medians because 2023-24 and 2024-25 were large: the all-rules mean is +84.3%, and without 2024-25 the mean is +59.6% and the median +52.1%.

Larger stakes raise both the return and the risk. At half Kelly the all-rules median is +101% with a 59% drawdown. At full Kelly it is +125%, with an 83% drawdown and a worst season of -78%.

| Season | Bets | Cover | ROI per unit staked | Season return | Largest drawdown |
|---|---|---|---|---|---|
| 2015-16 | 176 | 59.7% | +12.3% | +51.0% | 11% |
| 2016-17 | 244 | 55.3% | +6.3% | +47.9% | 22% |
| 2017-18 | 268 | 50.7% | -3.4% | -15.9% | 22% |
| 2018-19 | 182 | 59.9% | +18.1% | +59.2% | 13% |
| 2019-20 | 216 | 55.6% | +14.4% | +77.1% | 29% |
| 2020-21 | 288 | 55.9% | +9.2% | +53.2% | 15% |
| 2021-22 | 380 | 58.2% | +7.1% | +79.4% | 36% |
| 2022-23 | 242 | 53.3% | +10.5% | +45.4% | 22% |
| 2023-24 | 493 | 53.5% | +10.2% | +153.5% | 22% |
| 2024-25 | 409 | 59.2% | +15.4% | +331.7% | 19% |
| 2025-26 | 528 | 53.6% | +3.5% | +44.7% | 32% |

The 2022-23 row holds totals bets for only the first half of that season. The source of opening totals stopped on January 16, 2023, so 658 of the 1,230 games have one, and spreads for the rest of that season come from a different source of opening lines.

## Why recent seasons are larger

The two largest returns are 2023-24 and 2024-25, and in a backtest that deserves suspicion. An earlier version of this page had a real problem here: the spread model's pieces were combined as they had been designed on 2021-22 to 2025-26, and the spread rule covered 60.2% in those seasons against 56.2% before. The pieces are now weighted from earlier seasons only, and the recent seasons no longer lead: the rule covers 54.4% from 2021-22 on and 58.0% in the nine seasons before. What still carries hindsight is the choice of the pieces themselves, which was made with recent seasons in view. The totals model was specified in November 2026 before its results were looked at.

What is left is volume, not a higher return per bet: per unit staked, 2023-24 earned 10.2% and 2024-25 15.4%, against 12.3% in 2015-16 and 18.1% in 2018-19.

![bets and return per unit staked by season](../charts/engine_bets_and_roi.png)

The engine placed 477 bets a season from 2023-24 on, against 250 before, as opening totals began to sit further from the model, at a slightly larger stake on each (2.1% of the bankroll against 1.9%). Total stakes over a season therefore rose from roughly five times the bankroll to about ten, and because each stake is a share of the current bankroll, the same edge compounds into a far larger season. The stake sizes are also why these figures should not be read as achievable: opening lines carry low limits, and nobody gets ten times their bankroll down on them in a season.

At flat stakes the recent seasons are in fact the weaker ones. Across the two flat-stake rules, bets covered 56.0% for +6.9% per bet up to 2022-23 and 53.4% for +1.9% from 2023-24 on, while the number of bets a season went from 163 to 358, most of the increase in totals. The engine holds up better than that because it stakes more where earlier seasons won more, which means larger gaps and few early-season totals. Most of the extra totals volume comes from the model, not from the market. A totals model fitted on 2007-17 and never updated places 193 bets a season in 2023-26 against 138 before, a rise of 40%, and covers 56.2%; the model refitted each season goes from 126 to 276 and covers 53.4%. The difference is mostly one input, how far league-wide totals have lately been running over or under, which carried little weight before 2022 and now moves the prediction by half a point per standard deviation. About 90 bets a season in 2023-26 qualify only because of it, nearly all of them overs, and they covered 49.6%. Dropping the input is not the obvious fix, because on 2007-12 the same kind of bet covered 57.3%. The feed is not the cause: a second source of opening lines gives more bets, not fewer (878 against 732 on the same games), at a 54.6% cover rate.

Spread volume rose for a plainer reason. Spreads themselves have widened, with double-digit lines in 21% of games against 16%, so the model and the opener differ by more points while agreeing as closely as before. Scaled to the width of spreads, the share of games past the threshold is 5% in both periods. The volume is still the part least likely to carry over: so far this season 15% of totals reach the 2-point threshold at the first line we saw, against 23% in the backtest's recent seasons.

## A repair to the 2024-25 opening lines

The opening lines for 2024-25 come from a single book, read through ESPN's odds feed. In October and November 2024 the feed's opening field often held an alternate line instead of the main one, and the price attached to it gives this away: a main line opens near -110 on both sides, while these opened anywhere from -450 to +290. By that test 40% of the opening totals and 21% of the opening spreads in those two months were not the main line, against under 1% from December on and none in 2025-26.

A model sides against a line that is 7 points off and usually wins, so left alone those games would have made 2024-25 look several times better than it was. Each one (121 totals and 70 spreads) was replaced with the opening line from a second source, Action Network, which sits within about a point of ESPN's on ordinary games, and the few with no second opener were left out. The models were then refitted season by season on the repaired lines, and the figures on this page and the strategies page come from that rerun.

## What it learned

Fitted on all seasons, the engine assigns these win rates to typical bets.

| Bet | Win rate | Bet? |
|---|---|---|
| Totals over, gap 2, Dec-Jan | 54.1% | Yes |
| Totals over, gap 2, Oct-Nov | 49.3% | No |
| Totals under, gap 2, Dec-Jan | 57.5% | Yes |
| Totals over, gap 4, Dec-Jan | 60.8% | Yes |
| Spread, gap 2, Oct-Nov | 57.1% | Yes |
| Spread, gap 3, Oct-Nov | 60.4% | Yes |
| Spread, gap 2, April | 53.2% | Yes |

These are the patterns on the strategies page, except that here each was learned from earlier seasons and then applied to the next one.

## A simpler engine

The simpler engine groups bets by market and gap size. It estimates each group's win rate from earlier seasons, pulls that estimate strongly toward break-even, and bets the groups that still clear it. It stakes less and places 216 bets a season.

| | Median season | Mean | Volatility (sd) | Worst | Losing seasons | ROI per unit staked |
|---|---|---|---|---|---|---|
| Totals + spread | +14.0% | +26.9% | 32.6% | -0.3% | 1 of 11 | +9.5% |
| Totals only | +8.7% | +20.4% | 24.3% | -2.8% | 1 of 11 | +10.5% |
| Spread only | +5.4% | +4.9% | 9.7% | -18.5% | 1 of 11 | +6.5% |

## Where the +53% comes from

The figure is leverage, not extra skill: ROI per unit staked is about +9% in both the full engine and the simple one. The full engine returns more because it places 311 bets a season at larger stakes. Adding the seasonal and side rules does not improve the win-probability forecast itself, whose log loss is 0.68821 without them and 0.68822 with them.

More bets did not mean more profit per bet. Bets per season nearly doubled after 2022, as opening totals began moving more and the lower-gap groups, which win about 52%, came into play.

The figure also assumes every opening line is bet at -110 with 2 to 4% of the bankroll. First lines carry low limits, and most of their move toward the close happens overnight and on game day.

The engine is not safe from a bad year. In 2017-18 it won 50.7% of its bets, lost 3.4% per unit staked and finished down 15.9% after a 22% drawdown; the gap-only version lost 1.4% that season, while the simpler engine, which stakes less, made 7.3%.

None of this has been shown live; the paper views began recording the price actually available on 2026-11-16.
