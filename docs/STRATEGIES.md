# Betting strategies: results and what explains them

Every strategy tried on top of the model, what it returned, and why, where the why could
be checked against the model's own inputs. An explanation is listed as **tested** only if
a number backs it. Where a plausible story failed its check, it is listed as rejected.

Unless stated, results are the walk-forward backtest for 2012-13 to 2025-26: each season
is scored by a model fitted only on earlier seasons, bets are priced at the opening line
at -110, and stakes are flat. This season's results are in
[SEASON_2026_27.md](SEASON_2026_27.md).

## How to read the numbers

- **ROI is per bet.** It is profit divided by the amount staked: +6% means each 1-unit
  bet returned 0.06 units on average.
- **Season return depends on how much is staked.** It is ROI x number of bets x stake
  size. The totals rule at +6.1% over about 175 bets, staking 1% of the starting
  bankroll each time, returns 0.061 x 175 x 1% = about +10.7% of bankroll in a season.
- **Cover rate** is the share of bets won. At -110 the break-even is 52.4%.
- **All of it assumes the opening line is actually bet.** That has not been shown live.

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

"Gap" is how far the model's number is from the line, in points.

## Spread rule

Bet the model's side when its corrected margin is 2+ points from the opening spread.

**Why it works at all (tested).** Across 15,911 games the final margin moves 0.35 points
for every point the model disagrees with the opener. A plain 20-line ratings model built
from results earns 0.11, and adds nothing once the full model is included. So the player
ratings, minutes and availability inputs are doing real work.

**Cover rises with the gap as the model implies (tested).** A gap of g points against an
outcome spread of 12.6 points implies a cover rate. Actual results track it:

| Gap | Implied by the model | Actual | Bets |
|---|---|---|---|
| 2 to 2.5 | 56.9% | 54.7% | 453 |
| 2.5 to 3 | 58.5% | 58.1% | 179 |
| 3+ | 61.6% | 63.7% | 168 |

**By month.**

| Month | Cover | Bets |
|---|---|---|
| Oct-Nov | 62.4% | 271 |
| Dec | 57.0% | 86 |
| Jan | 53.4% | 73 |
| Feb | 54.7% | 75 |
| Mar | 60.5% | 129 |
| Apr | 49.3% | 142 |

**Why October and November are best (tested).** The model is not more accurate per point
early: 0.34 points of result per point of gap in Oct-Nov against 0.41 from December to
March. It simply disagrees with the opener by more. The spread of its disagreement is 3.2
points early against 2.5 later, so 7.3% of early games reach the 2-point threshold
against 3.2% later, and they reach it by more.

**Why April loses (tested, small sample).** In April the model disagrees with the opener
most (3.5 points) and is right least (0.12 points of result per point of gap). April
games carry more expected absences, 2.7 a game against 1.9. On 2019-26, April bets in
games with 3+ expected absences covered 35.9% (39 bets) and the rest 58.1% (31 bets).
The model is weakest when many regular players sit. Whether that is rest, tanking or
both was not separated.

## Totals rule

Bet over or under when the model's predicted final total is 2+ points from the opening
total. This model uses no player data. It reads how the market has been pricing the two
teams and the league.

**What drives the bets (tested).** Each bet was assigned to the input that contributed
most to it:

| Largest contributor | Cover | Bets |
|---|---|---|
| Each team's previous game went over or under its closing total | 58.3% | 362 |
| Opening total differs from where recent closing totals put these teams | 56.1% | 1,336 |
| Each team's last five games ran over or under | 55.7% | 296 |
| League-wide totals have been running over or under | 54.3% | 186 |
| Days of rest | 51.0% | 239 |

Most bets, and most of the profit, come from the opener sitting away from where the
market itself has recently closed these teams. Bets driven mainly by rest do not pay.

**Cover rises with the gap as the model implies (tested).** Against an outcome spread of
18 points:

| Gap | Implied by the model | Actual | Bets |
|---|---|---|---|
| 2 to 3 | 55.3% | 54.0% | 1,783 |
| 3 to 4 | 57.4% | 57.7% | 468 |
| 4+ | 60.8% | 65.2% | 198 |

**The edge exists only while the line has not moved (tested).** Bets priced at the close:
56.4% when the line stayed within 1 point of the opener, 51.8% once it had moved with
the model, 48.8% once it had moved against it. The opener is corrected by the market
within hours, sometimes minutes, so a late bet is a bet into a line that already knows
what the model knows.

**By month.**

| Month | 2012-26 | Bets | 2007-12 holdout | Bets |
|---|---|---|---|---|
| Oct-Nov | 52.4% | 699 | 53.9% | 219 |
| Dec | 56.3% | 426 | 51.4% | 107 |
| Jan | 55.5% | 337 | 53.4% | 116 |
| Feb | 57.8% | 282 | 60.9% | 110 |
| Mar | 54.2% | 389 | 55.9% | 127 |
| Apr | 59.2% | 289 | 62.6% | 91 |

**Why April is strong (partly tested).** April games finish 2.1 points above the opening
total on average, against 0.4 in other months, and 279 of the model's 289 April bets
were overs. The model leans to the over by about half a point and follows the league
when totals run high, so it is on the right side of that. Why April scoring beats the
opener was not tested.

**Why October and November are weak (not explained).** One story is that the model has
too little history early. It only partly holds: bets placed before team ratings existed
covered 49.2% (124 bets), but bets placed with ratings available still covered only
53.0% (575).

**February (not explained).** It is strong in both samples. The All-Star break is not
the reason: February bets on long-rested teams covered 56.0%, the others 58.6%.

## Variations

- **Stake in proportion to the gap** (1 unit at 2 points, capped at 2.5). ROI rises from
  +9.5% to +11.4% on spreads and +6.1% to +7.6% on totals, and losing seasons fall from
  3 and 2 of 14 to none. This follows from the gap tables above: bigger gaps cover more.
- **Higher threshold.** Spread gap 2.5+: 60.8%, +16.1%, 25 bets a season. Totals gap 3+:
  59.9%, +14.4%, 48 a season. More ROI per bet, fewer units in total.
- **Fraction of Kelly.** On the totals rule, quarter Kelly returned +13.0% a season on
  average with a median of +4.2% and a standard deviation of 25.1%. Full Kelly reached a
  66% drawdown. Details in [SEASON_2026_27.md](SEASON_2026_27.md).
- **By calendar.** The monthly tables suggest spreads through November and totals from
  December. The monthly cuts were made after seeing the results, so this is a
  hypothesis for next season, not a rule.

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

Two of these looked like improvements until they were run on seasons they had not been
found on. That is the reason every monthly and by-type cut on this page should be read
as a description of the backtest, not a promise.

## What is not known

- Whether the opening line can be bet in practice. Both rules lose their edge at later
  prices.
- Whether any of this holds live. The paper views began recording on 2026-11-16.
- One season, 2024-25, supplies about a third of the backtest profit, and its opening
  lines look stale.
