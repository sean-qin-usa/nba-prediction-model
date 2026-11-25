# Betting strategies

This page lists every strategy tried on top of the model, what it returned, and why. Each explanation comes with the number that backs it, and the headings say where one is partial or missing. Where a plausible story failed its check, it appears under rejected.

Unless stated, results are the walk-forward backtest for 2012-13 to 2025-26. Each season is scored by a model fitted on earlier seasons, bets are priced at the opening line at -110, and stakes are flat. This season's results are in [SEASON_2026_27.md](SEASON_2026_27.md).

## ROI and cover rate

ROI is per bet: it is profit divided by the amount staked, so +6% means each 1-unit bet returned 0.06 units on average. The return over a season then depends on how much is staked: ROI times the number of bets times the stake size. The totals rule earns +5.5% over about 170 bets, which at 1% of the starting bankroll per bet comes to 0.055 x 170 x 1%, or about +9.3% of bankroll.

Cover rate is the share of bets won, and at -110 the break-even is 52.4%.

Everything here assumes the opening line is actually bet, which has not been shown live.

## Summary

| Strategy | Bets per season | Cover | ROI per bet | Season return at 1% stakes | Status |
|---|---|---|---|---|---|
| Spread, model gap 2+ points | 54 | 58.3% | +11.3% | +6.1% | Backtest only |
| Totals, model gap 2+ points | 170 | 55.3% | +5.5% | +9.3% | Passed a pre-declared test |
| Totals, same rule on 2007-12 (never tuned on) | 137 | 57.4% | +9.5% | +13.0% | Holdout |
| Totals, bet only within 1 point of the opener | about 140 | about 55.3% | about +5.5% | about +8% | Under live test |
| Either rule, stake in proportion to the gap | same bets | same | +12.2% spread, +6.9% totals | +8.4%, +16.1% | Logged beside the flat stake |
| Moneyline, four rules frozen before the season | about 75 | not comparable | no line value | not estimated | Live; likely no edge |
| Player points | not a rule yet | 52.6% at a 5+ point probability gap | +0.7% | not estimated | Not bettable |

The gap is how far the model's number sits from the line, in points.

The spread model was developed on 2021-22 to 2025-26, so those seasons flatter it. On the nine seasons before, which had no part in building it, the spread rule covers 56.2% on 361 bets for +7.3% per bet, against 60.2% on 392 bets since; the earlier figure is the cleaner one. The totals model was specified before its results were looked at.

The 2024-25 opening lines behind these figures were repaired after a fault in the source feed, which the [engine page](ENGINE.md#a-repair-to-the-2024-25-opening-lines) describes.

![cover by gap](../charts/strategy_cover_by_gap.png)

![cover by month](../charts/strategy_cover_by_month.png)

All of the rules combined into one engine, tested on seasons it had not seen, are on a separate page: [ENGINE.md](ENGINE.md).

## Spread rule

The rule bets the model's side when its corrected margin is 2 or more points from the opening spread.

### Where the edge comes from

Across 15,911 games the final margin moves 0.36 points for every point the model disagrees with the opener. A plain ratings model built from results in 20 lines of code earns 0.09, and it adds nothing once the full model is included. The player ratings, minutes and availability inputs are therefore doing the work.

### Bigger gaps cover more

A gap of a given size, set against an outcome spread of 12.6 points, implies a cover rate, and actual results follow it loosely: the middle band falls short and the top band overshoots.

| Gap | Implied by the model | Actual | Bets |
|---|---|---|---|
| 2 to 2.5 | 56.9% | 57.8% | 460 |
| 2.5 to 3 | 58.5% | 55.2% | 181 |
| 3+ | 61.7% | 65.2% | 112 |

### By month

| Month | Cover | Bets |
|---|---|---|
| Oct-Nov | 63.6% | 236 |
| Dec | 56.3% | 87 |
| Jan | 52.4% | 63 |
| Feb | 52.3% | 65 |
| Mar | 60.1% | 153 |
| Apr | 53.8% | 130 |

### Best in October and November

The model is no more accurate per point early in the season. It earns 0.30 points of result per point of gap in October and November, against 0.42 from December to March. What changes is how far it disagrees with the opener: the spread of that disagreement is 3.1 points early and 2.6 later. As a result 6.4% of early games reach the 2-point threshold, compared with 3.2% later, and they clear it by more.

### April is weakest

April is where the model disagrees with the opener most, at 3.4 points, and is right least, at 0.22 points of result per point of gap, so its bets cover only 53.8%. April games also carry more expected absences: 2.7 a game against 1.9. An earlier version of this page blamed those absences, but the split does not hold on the rebuilt model: on 2019-26, April bets in games with three or more expected absences covered 55.0% (40 bets) and the rest 57.1% (28 bets). Giving April its own, lower trust did not help out of sample either, so the weak month is unexplained.

## Totals rule

The rule bets over or under when the model's predicted final total is 2 or more points from the opening total. This model uses no player data; it reads how the market has been pricing the two teams and the league.

### What drives the bets

Each bet was assigned to the input that contributed most to it.

| Largest contributor | Cover | Bets |
|---|---|---|
| Each team's previous game went over or under its closing total | 58.0% | 362 |
| Opening total differs from where recent closing totals put these teams | 55.7% | 1,272 |
| Each team's last five games ran over or under | 54.8% | 301 |
| League-wide totals have been running over or under | 54.3% | 186 |
| Days of rest | 51.3% | 238 |

Most bets, and most of the profit, come from an opener that sits away from where the market itself has recently closed these teams, while bets driven mainly by rest do not pay.

### Bigger gaps cover more

The implied rates use an outcome spread of 18 points.

| Gap | Implied by the model | Actual | Bets |
|---|---|---|---|
| 2 to 3 | 55.3% | 53.7% | 1,760 |
| 3 to 4 | 57.4% | 57.3% | 450 |
| 4+ | 60.4% | 65.9% | 170 |

### The edge dies when the line moves

Priced at the close, the bets covered 56.2% when the line stayed within 1 point of the opener. Once the line had moved with the model they covered 51.4%, and once it had moved against the model, 49.3%. The market corrects an opener within hours, sometimes minutes, so a late bet goes into a line that already holds what the model knows.

![totals by line movement](../charts/totals_line_moved.png)

### By month

| Month | 2012-26 | Bets | 2007-12 holdout | Bets |
|---|---|---|---|---|
| Oct-Nov | 50.6% | 664 | 54.4% | 204 |
| Dec | 56.0% | 411 | 52.7% | 91 |
| Jan | 56.0% | 332 | 57.1% | 98 |
| Feb | 58.0% | 276 | 63.4% | 101 |
| Mar | 54.6% | 383 | 55.5% | 110 |
| Apr | 59.6% | 287 | 65.4% | 81 |

### April is strong (partly explained)

April games finish 2.1 points above the opening total on average, against 0.4 in other months. Of the model's 287 April bets, 277 were overs. The model leans to the over by about half a point and follows the league when totals run high, which puts it on the right side. Why April scoring beats the opener was not tested.

### Weak in October and November (unexplained)

One story is that the model has too little history early, but it holds only in part. Bets placed before team ratings existed covered 49.2% (124 bets), but bets placed with ratings available still covered only 50.9% (540 bets).

### February (unexplained)

February is strong in both samples, and the All-Star break is not the reason, since February bets on long-rested teams covered 56.6% and the others 58.5%.

## Sizing and thresholds

Staking in proportion to the gap (1 unit at 2 points, capped at 2.5) raises ROI from +11.3% to +12.2% on spreads and from +5.5% to +6.9% on totals. This follows from the gap tables above, because bigger gaps cover more. It trims the losing seasons from 2 of 14 to 1 on each rule.

A higher threshold gives more per bet and less in total. Spread bets at a gap of 2.5+ cover 59.0% for +12.7%, on 21 bets a season. Totals bets at 3+ cover 59.7% for +13.9%, on 44 a season.

On the totals rule, quarter Kelly returned +9.6% a season on average, with a median of +4.7% and a standard deviation of 14.0%. Full Kelly reached a 68% drawdown; the detail is in [SEASON_2026_27.md](SEASON_2026_27.md).

The monthly tables suggest spreads through November and totals from December. Those cuts were made after seeing the results, so this is a hypothesis for next season.

### How big a stake

The table runs both rules together with sized stakes, where the base stake is a share of the current bankroll and each of the 14 seasons starts fresh. No more than 25% of the bankroll is at risk on one night.

| Base stake | Median season | Worst season | Losing seasons | Worst drawdown within a season |
|---|---|---|---|---|
| 1% | +20.5% | -7.3% | 2 of 14 | 21% |
| 2% | +37.8% | -17.1% | 2 of 14 | 42% |
| 3% | +54.9% | -30.1% | 2 of 14 | 55% |
| 5% | +79.4% | -57.5% | 3 of 14 | 70% |
| 8% | +89.3% | -75.1% | 3 of 14 | 81% |

Drawdown grows almost in step with the stake, and the median return stops growing before the drawdown does: at a 12% base stake the median is +75.7%, below the 8% row, with a 91% drawdown and five losing seasons. The longest losing run was 10 bets, and the busiest night carried 17 bets at once. Mean returns run far above the medians at the larger stakes, because a few seasons compound to several times the bankroll, which is why the table leaves them out.

## Why the rules win

The working thesis is that the model is faster than the opening line and no smarter than the market. On spreads it prices at the opener what the market prices over the following day, which is who will play and how good they are, and the line then moves toward it. The totals model uses no player data and bets when the opener sits away from where the market itself recently closed those teams. Neither rule has an edge at later prices, and the model does not beat the closing line on any market.

### Knowing who plays

The spread model is built on player ratings, so it matters how much it knows about who will play. The table reruns the same model over the same games with five levels of that knowledge, on the rule the live view uses: bet when the market-blind model is 5 or more points from the opening spread.

| What the model knows about who plays | 2012-18 cover | 2018-26 cover | All 14 seasons | Bets |
|---|---|---|---|---|
| Nothing | 53.2% | 55.4% | 54.8% | 1,315 |
| A synthetic feed built from box scores | 54.5% | 56.9% | 56.2% | 1,319 |
| The injury-report feed, as of the open | not available | 58.6% | 56.9% | 1,260 |
| Official inactive lists, 30 minutes before tip | 55.2% | 60.0% | 58.6% | 1,274 |
| Perfect hindsight | 58.0% | 60.1% | 59.5% | 1,403 |

Cover rises with every step, and so does the amount by which the bets beat the closing line, from 0.6 points with nothing to 1.8 with hindsight. The injury-report feed only exists from December 2018, so for earlier seasons the backtest uses the synthetic feed, which estimates each rotation player's chance of sitting from box scores before game day: whether he missed the last game, how many in a row, days since he played, his minutes and back-to-backs. It recovers about half of what the real feed adds, and its estimate of minutes missing per team tracks reality about as well (a correlation of 0.68 against 0.72). The last two rows cannot be bet a day ahead and are there as a ceiling.

The older seasons stay behind the recent ones at every level, so the missing feed explains part of that gap and not all of it.

### Retuned on other seasons

The spread model was designed on 2021-22 to 2025-26, so two checks ask how much of its result depends on that. In the first, the model's five pieces (two team-strength estimates, a schedule term, a late-season tanking term and the availability adjustments) were re-weighted by a regression fitted on a random half of the 19 seasons and scored on the other half, 100 times over.

| Spread rule, gap 2+ | 2012-21 | 2021-26 | All 14 seasons |
|---|---|---|---|
| The model as designed | 55.5% | 60.4% | 58.0% |
| Weights retuned on random other seasons | 54.6% | 57.2% | 55.9% |
| As designed, without the tanking term | 55.5% | 54.8% | 55.2% |

Nearly all of the advantage in the development seasons comes from one piece: the tanking term, which the shipped model only switches on from 2020-21. Without it those seasons cover 54.8%, the same as the earlier ones. The second check therefore switched that term on for 2012-20 as well, seasons it was never designed on, with its coefficient learned from earlier seasons. It lifted the spread rule there from 55.9% to 57.0% on 337 bets, a gain inside the noise and far short of the five points it adds where it was built.

The fair reading is a spread rule that covers about 56%, not 58%, and whose best seasons owe a good deal to a term tuned on those seasons.

### Not roster or league change

A model built on player ratings might be expected to win most when rosters change, since a market anchored on last season's results would be slow to adjust. That is not what the backtest shows. Turnover here is the share of a team's minutes so far played by players who were not on that team last season, which counts rookies and trades alike, and the table splits games into thirds by the two teams' combined turnover.

| Combined turnover | Spread cover | Bets | Totals cover | Bets |
|---|---|---|---|---|
| Low | 57.4% | 237 | 54.9% | 887 |
| Middle | 57.4% | 270 | 54.9% | 827 |
| High | 60.2% | 246 | 56.2% | 666 |

In a regression of win on gap, turnover and early season, the turnover coefficient is about zero for both rules (p = 0.48 on spreads and 0.73 on totals). The model's disagreement with the opener is also no wider when turnover is high. The early-season spread edge does not come from new rosters either, since October and November bets covered 67.1% in low-turnover games (85 bets) and 66.7% in high-turnover games (78 bets). Backing the team that was worse last season covered 58.6% (290 bets), against 58.0% for backing the better one (462 bets).

Season by season the picture is the same. Spread cover is uncorrelated with league-wide turnover (-0.03) and with the year-on-year change in league scoring (+0.19, on 14 seasons). Totals cover falls with league turnover (-0.56), which is most likely a shared time trend, because turnover has drifted down over the years in which totals bets have grown.

### Not periods of change either

The same question was asked of the stretches when rosters shift during a season, and of the particular teams that changed.

| Cut | Spread cover | Bets | Totals cover | Bets |
|---|---|---|---|---|
| Rotation stable over the last 15 games | 62.2% | 119 | 56.5% | 503 |
| Rotation churning | 56.1% | 262 | 57.1% | 694 |
| 28 days before the trade deadline | 56.9% | 51 | 53.1% | 254 |
| 21 days after the trade deadline | 48.4% | 64 | 61.8% | 254 |
| First 20 games, a rebuilt team in the game | 63.9% | 119 | 50.2% | 263 |
| First 20 games, no rebuilt team | 60.6% | 137 | 52.6% | 456 |
| 30 days after the deadline, a traded team in the game | 48.5% | 66 | 57.6% | 210 |
| 30 days after the deadline, no traded team | 62.5% | 32 | 61.1% | 131 |

Churn compares who played a team's last 5 games with the 10 before. A rebuilt team is in the top quarter for offseason turnover, and a traded team is one whose deadline arrivals went on to play 10% or more of its minutes. When rotations are churning the spread model fires on more games (5.9% against 2.7%) but is right less per point of gap (1.02 against 1.43), so cover does not improve. The jump in totals cover after the deadline looked like a finding until it was run on the 2007-12 holdout, where it reversed to 59.8% before the deadline and 50.6% after.

One pattern points the other way and is recorded here as a lead. On spreads, bets where the model backs a rebuilt team covered 63.2% (125 bets), while bets where it fades one covered 54.6% (141 bets), and the same gap appears early in the season though not after the deadline. The samples overlap, though, and a single test with no thresholds is positive without being significant (p = 0.16), so it waits for a season it was not found on.

If the edge came from understanding change it should rise with turnover, and it does not. What the bets have in common is a rough opening number: bet counts more than doubled from 2023, as opening totals began to sit further from the model.

## Tried and rejected

| Idea | Result |
|---|---|
| Train the spread model on line moves instead of final scores | Worse in 12 of 14 seasons |
| Trust the spread model more early in the season | Trust is the same early and late |
| Add rest and rookie minutes to the spread model | Better in only 6 of 14 seasons |
| Refit the spread model on recent seasons only | No gain |
| Bet against big line moves at the close | 48% |
| Bet with big line moves at the close | 51 to 52%, under break-even |
| Drop the totals model's standing lean to the over | Looked good on 2012-26 (56.6%), failed on 2007-12 (51.7%) |
| Totals bets relative to the season's running average call | 54.2% against 57.4% on 2007-12 |
| Add expected absences to the totals model | The opener already prices them |
| A different trust in the model by gap size, by month, or fixed at another value | None beats the fitted one third |

Two of these looked like improvements until they were run on seasons they had not been found on. The monthly and by-type cuts on this page have had no such check, so they only describe the backtest.

## Open questions

The first unknown is whether the opening line can be bet in practice, since both rules lose their edge at later prices. The second is whether any of this holds live, which the paper views began recording on 2026-11-16. The last three seasons also supply most of the backtest profit, because they hold far more bets than the earlier ones. One of them, 2024-25, accounts for 38% of the spread rule's profit on its own (68.9% on 103 bets); without it the rule covers 56.6% on 650 bets. Its rules were learned from earlier seasons and the result holds at two other sources of opening lines, but the choice of which rules to report was made with that season in view.
