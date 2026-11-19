# Betting strategies

This page lists every strategy tried on top of the model, what it returned, and why. Each explanation comes with the number that backs it, and the headings say where one is partial or missing. Where a plausible story failed its check, it appears under rejected.

Unless stated, results are the walk-forward backtest for 2012-13 to 2025-26. Each season is scored by a model fitted on earlier seasons, bets are priced at the opening line at -110, and stakes are flat. This season's results are in [SEASON_2026_27.md](SEASON_2026_27.md).

## ROI and cover rate

ROI is per bet: it is profit divided by the amount staked, so +6% means each 1-unit bet returned 0.06 units on average. The return over a season then depends on how much is staked: ROI times the number of bets times the stake size. The totals rule earns +6.1% over about 175 bets, which at 1% of the starting bankroll per bet comes to 0.061 x 175 x 1%, or about +10.7% of bankroll.

Cover rate is the share of bets won, and at -110 the break-even is 52.4%.

Everything here assumes the opening line is actually bet, which has not been shown live.

## Summary

| Strategy | Bets per season | Cover | ROI per bet | Season return at 1% stakes | Status |
|---|---|---|---|---|---|
| Spread, model gap 2+ points | 57 | 57.4% | +9.5% | +5.4% | Backtest only |
| Totals, model gap 2+ points | 175 | 55.6% | +6.1% | +10.7% | Passed a pre-declared test |
| Totals, same rule on 2007-12 (never tuned on) | 154 | 55.8% | +6.6% | +10.2% | Holdout |
| Totals, bet only within 1 point of the opener | about 145 | about 55.6% | about +6% | about +9% | Under live test |
| Either rule, stake in proportion to the gap | same bets | same | +11.4% spread, +7.6% totals | +8.5%, +18.5% | Logged beside the flat stake |
| Moneyline, four rules frozen before the season | about 75 | not comparable | no line value | not estimated | Live; likely no edge |
| Player points | not a rule yet | 52.6% at a 5+ point probability gap | +0.7% | not estimated | Not bettable |

The gap is how far the model's number sits from the line, in points.

![cover by gap](../charts/strategy_cover_by_gap.png)

![cover by month](../charts/strategy_cover_by_month.png)

All of the rules combined into one engine, tested on seasons it had not seen, are on a separate page: [ENGINE.md](ENGINE.md).

## Spread rule

The rule bets the model's side when its corrected margin is 2 or more points from the opening spread.

### Where the edge comes from

Across 15,911 games the final margin moves 0.35 points for every point the model disagrees with the opener. A plain ratings model built from results in 20 lines of code earns 0.11, and it adds nothing once the full model is included. The player ratings, minutes and availability inputs are therefore doing the work.

### Bigger gaps cover more

A gap of a given size, set against an outcome spread of 12.6 points, implies a cover rate, and actual results track it.

| Gap | Implied by the model | Actual | Bets |
|---|---|---|---|
| 2 to 2.5 | 56.9% | 54.7% | 453 |
| 2.5 to 3 | 58.5% | 58.1% | 179 |
| 3+ | 61.6% | 63.7% | 168 |

### By month

| Month | Cover | Bets |
|---|---|---|
| Oct-Nov | 62.4% | 271 |
| Dec | 57.0% | 86 |
| Jan | 53.4% | 73 |
| Feb | 54.7% | 75 |
| Mar | 60.5% | 129 |
| Apr | 49.3% | 142 |

### Best in October and November

The model is no more accurate per point early in the season. It earns 0.34 points of result per point of gap in October and November, against 0.41 from December to March. What changes is how far it disagrees with the opener: the spread of that disagreement is 3.2 points early and 2.5 later. As a result 7.3% of early games reach the 2-point threshold, compared with 3.2% later, and they clear it by more.

### April loses (small sample)

April is where the model disagrees with the opener most, at 3.5 points, and is right least, at 0.12 points of result per point of gap. April games also carry more expected absences: 2.7 a game against 1.9. On 2019-26, April bets in games with three or more expected absences covered 35.9% (39 bets), while the rest covered 58.1% (31 bets). The model is weakest when many regular players sit, although rest and tanking were not separated.

## Totals rule

The rule bets over or under when the model's predicted final total is 2 or more points from the opening total. This model uses no player data; it reads how the market has been pricing the two teams and the league.

### What drives the bets

Each bet was assigned to the input that contributed most to it.

| Largest contributor | Cover | Bets |
|---|---|---|
| Each team's previous game went over or under its closing total | 58.3% | 362 |
| Opening total differs from where recent closing totals put these teams | 56.1% | 1,336 |
| Each team's last five games ran over or under | 55.7% | 296 |
| League-wide totals have been running over or under | 54.3% | 186 |
| Days of rest | 51.0% | 239 |

Most bets, and most of the profit, come from an opener that sits away from where the market itself has recently closed these teams, while bets driven mainly by rest do not pay.

### Bigger gaps cover more

The implied rates use an outcome spread of 18 points.

| Gap | Implied by the model | Actual | Bets |
|---|---|---|---|
| 2 to 3 | 55.3% | 54.0% | 1,783 |
| 3 to 4 | 57.4% | 57.7% | 468 |
| 4+ | 60.8% | 65.2% | 198 |

### The edge dies when the line moves

Priced at the close, the bets covered 56.4% when the line stayed within 1 point of the opener. Once the line had moved with the model they covered 51.8%, and once it had moved against the model, 48.8%. The market corrects an opener within hours, sometimes minutes, so a late bet goes into a line that already holds what the model knows.

![totals by line movement](../charts/totals_line_moved.png)

### By month

| Month | 2012-26 | Bets | 2007-12 holdout | Bets |
|---|---|---|---|---|
| Oct-Nov | 52.4% | 699 | 53.9% | 219 |
| Dec | 56.3% | 426 | 51.4% | 107 |
| Jan | 55.5% | 337 | 53.4% | 116 |
| Feb | 57.8% | 282 | 60.9% | 110 |
| Mar | 54.2% | 389 | 55.9% | 127 |
| Apr | 59.2% | 289 | 62.6% | 91 |

### April is strong (partly explained)

April games finish 2.1 points above the opening total on average, against 0.4 in other months. Of the model's 289 April bets, 279 were overs. The model leans to the over by about half a point and follows the league when totals run high, which puts it on the right side. Why April scoring beats the opener was not tested.

### Weak in October and November (unexplained)

One story is that the model has too little history early, but it holds only in part. Bets placed before team ratings existed covered 49.2% (124 bets), but bets placed with ratings available still covered only 53.0% (575 bets).

### February (unexplained)

February is strong in both samples, and the All-Star break is not the reason, since February bets on long-rested teams covered 56.0% and the others 58.6%.

## Sizing and thresholds

Staking in proportion to the gap (1 unit at 2 points, capped at 2.5) raises ROI from +9.5% to +11.4% on spreads and from +6.1% to +7.6% on totals. This follows from the gap tables above, because bigger gaps cover more. It does not smooth the seasons much: losing seasons go from 3 of 14 to 4 on spreads, two of them by under half a unit, and from 2 to 1 on totals.

A higher threshold gives more per bet and less in total. Spread bets at a gap of 2.5+ cover 60.8% for +16.1%, on 25 bets a season. Totals bets at 3+ cover 59.9% for +14.4%, on 48 a season.

On the totals rule, quarter Kelly returned +13.0% a season on average, with a median of +4.2% and a standard deviation of 25.1%. Full Kelly reached a 66% drawdown; the detail is in [SEASON_2026_27.md](SEASON_2026_27.md).

The monthly tables suggest spreads through November and totals from December. Those cuts were made after seeing the results, so this is a hypothesis for next season.

## Tried and rejected

| Idea | Result |
|---|---|
| Train the spread model on line moves instead of final scores | Worse in 11 of 14 seasons |
| Trust the spread model more early in the season | Trust is the same early and late |
| Add rest and rookie minutes to the spread model | Better in only 5 of 14 seasons |
| Refit the spread model on recent seasons only | No gain |
| Bet against big line moves at the close | 48% |
| Bet with big line moves at the close | 51.5%, under break-even |
| Drop the totals model's standing lean to the over | Looked good on 2012-26 (57.0%), failed on 2007-12 (52.8%) |
| Totals bets relative to the season's running average call | 54.4% against 55.8% on 2007-12 |
| Add expected absences to the totals model | The opener already prices them |
| Perfect knowledge of who plays, for spreads | 58.0% becomes 58.4% |

Two of these looked like improvements until they were run on seasons they had not been found on. The monthly and by-type cuts on this page have had no such check, so they only describe the backtest.

## Open questions

The first unknown is whether the opening line can be bet in practice, since both rules lose their edge at later prices. The second is whether any of this holds live, which the paper views began recording on 2026-11-16. One season, 2024-25, also supplies about a third of the backtest profit, and its opening lines look stale.
