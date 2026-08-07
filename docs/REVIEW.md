# NBA opening-spread relative value — model and strategy review

**Supersedes `nba_model_and_strategy_review.pdf`**, whose figures predate three
corrections: the injury-report coverage frame (`D186`), the bet-time information
leak (`D199`), and the trading frame (`D207`). The PDF is retained for history.

---

## 1. What this is

A market-anchored NBA sides strategy. A market-blind margin model prices every
regular-season game; that model's **disagreement with the opening line** is then
spent, at about a third of face value, as a correction to the opening line
itself. Bets are one side of one game at −110, entered at the open, held to
settlement. Flat 1 unit. No hedge, no exit, no calendar filter.

```
m_final = open_margin + f(x)
```

`f` is a ridge shrunk hard toward zero — toward *"the opener is right"* — fitted
walk-forward on three features knowable at the open: the market-blind model's
edge vs the opener, rest differential, and |open_margin|. Its edge coefficient is
**0.33–0.37 in every fold**.

## 2. The frame, and why it is only seven seasons

**Everything here is 2019-20 → 2025-26. Earlier seasons are excluded on
principle, not on results.**

The model has four inputs. Three go back to 2007-08. The fourth — the **daily NBA
injury report**, which the availability leg is built from — **does not exist
before 2018-12-17**, and covers only 63.7% of 2018-19. Availability composition
is *half the production margin*, so before 2019-20 the model cannot run as
designed: those seasons score a different, crippled system.

![coverage](../charts/data_coverage.png)

Pooling them was costing us in both directions. It diluted the result — the
strategy reads **+9.11%** on the honest frame against **+7.57%** on the 14-season
blend — and it made every interval a mixture of two different models.

**The cost of that honesty is power.** Seven seasons is not many, and it is why
nothing below is statistically significant.

A second, tighter window matters for execution: a genuine **multi-book panel at
the open exists for 2023-24 only** (7.74 books/game; 2024-25 and 2025-26 observe
1.00 and 1.03). Multi-book figures outside that season are modelled uplift.

## 3. Results

Priced at the opening spread, k=9 (max books observed), walk-forward selection.

| | market-blind | **market-offset** |
|---|---|---|
| pooled ROI | +4.74% | **+9.11%** |
| cumulative | +49.0u | **+80.9u** |
| bets | 1,035 | 888 |
| season-clustered mean | +3.14% | **+6.67%** |
| **95% CI (K=7)** | [−5.79, +12.07] | **[−2.75, +16.08]** |
| profitable seasons | 4/7 | **5/7** |

**Neither is significant. Both intervals contain zero.**

At a flat $10,000 stake, the offset arm: **565 sessions, net $809,335, $1,432/day,
Sharpe 1.1, win days 53%, max drawdown −$240,301, edge 911 bps.** The blind arm:
$490,095, Sharpe 0.6, edge 474 bps.

![equity](../charts/sim_report_equity_offset.png)

**The result is not evenly earned.** The offset arm's first three seasons
(2019-22) return **−$30,637 at Sharpe −0.1**; the last four (2022-26) return
**+$839,972 at Sharpe 2.0** — and that recent block is also the block the
architecture was developed on.

### On the measured-panel window (2023-26)

| tier | market-blind | **market-offset** | delta |
|---|---|---|---|
| k=1 raw | +4.44% | **+10.63%** | +6.19 |
| k=9 raw | +10.21% | **+16.62%** | +6.40 |
| k=9 +haircut | +7.87% | **+14.36%** | +6.48 |

**+6.4 ROI points at every tier including a single book** — the gain is model,
not shopping. And the quality improves more than the headline: positive **3/3**
seasons against 2/3, best-season concentration **66%** against 84%, on **fewer**
bets (460 vs 532).

## 4. Yes, we built our own forecasting model. It loses to the market.

This is the most useful negative result in the project, and the offset
construction does not make sense without it.

Model, opening line and closing line on **identical games**, honest inputs:

| source | log loss | vs the model |
|---|---|---|
| our market-blind model | 0.59276 | — |
| **opening line** | **0.59228** | we are 0.00048 worse |
| **closing line** | **0.57870** | we are 0.01406 worse |

**Beaten by the close decisively — 12.3% of its skill-above-a-coinflip — and by
the opener narrowly, 0.5%.** Expressed as the share of open-to-close information
recovered: the blind model captures **−3.5%** (it moves the wrong way); the
offset construction captures **+26.7%**.

So the market-blind model is not a discarded attempt. **It is the offset model's
dominant input**, and the finding that it must be spent at a third of face value
against a market anchor — rather than trusted on its own — *is* the strategy.

## 5. The bet-time information problem, and the fix

The availability leg was built on the **5PM injury report** and **pregame
inactives**. Both publish *after* the opening line. Every open-priced figure
predating `D199` therefore gave the model information the bettor did not have.

Measured: **81.9%** of a team's out-set is already known from the prior report,
but **18.1% is new on the day** (19.3% minutes-weighted). Rebuilding the out-set
by carry-forward from the last report *strictly before* game day widened the
model's gap to the market from **12.87% to 17.17%** — the late information was
worth **33% of our entire deficit**.

**The fix, gated and shipped.** The composition leg now weights each player by
**1 − P(out)**, forecast from as-of-open information only. Players last listed
*Questionable* are out **28.9%** of the time; the old hard rule scored them 0.000
and was wrong in both directions. Gate: season-clustered delta −0.002265,
**CI [−0.0041, −0.0004] excludes zero**, t = −3.39, better **5/5 seasons**,
calibration veto passed. **It recovers 52% of the leakage penalty** using only
what is public at the open.

Critically, **the offset architecture survived this correction and the standalone
model did not**: blind capture fell from +0.075 to **−0.104**; offset fell from
+0.313 to **+0.267**. Because the ridge already spent the model edge at a third
of face value, it was never leaning on the contaminated signal.

## 6. What would change our mind

- **2026-27, scored prospectively.** The architecture was developed on 2021-26;
  no retrospective procedure can make it out-of-sample. Both arms will run, on
  identical games, with the blind arm as the live control.
- **Two books captured at the open, from opening night.** The logger has never
  run in-season and `odds_quotes` is empty. If it is down on opening night the
  season's open-price record is unrecoverable.
- **Seven more seasons.** Not available. This is why the intervals are what they
  are.

## 7. Caveats

- Simulated on recorded prices. **No capital has ever been deployed.**
- Multi-book execution is counterfactual outside 2023-24.
- Sharpe is annualised on realised session count (~80/season), never √252, which
  would inflate it 1.8×.
- The 2019-26 frame is too short to tune on: a best-of-five **random** game subset
  buys **+2.54 ROI points**.
- Neither arm is the production default. Promoting the offset construction is a
  re-certification and has not been done.
- The register (`DECISIONS.md`, 207 entries) records the rejections, the four
  team-name join bugs, the availability leak, two wrong frames, and a confidence
  interval that briefly claimed false significance. Most entries are failures.
