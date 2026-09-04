# NBA Prediction Model

A market-anchored NBA sides model and the research record behind it: a decision register that runs to D247, most entries rejections.

The three-page summary is [Model and Strategy Review (PDF)](nba_model_and_strategy_review.pdf), rendered from [docs/REVIEW.md](docs/REVIEW.md) so the two stay in sync. It covers what the system is, what it returns, and where it is weakest.

## Overview

A market-blind margin model prices every regular-season game. Its disagreement with the opening line is then spent, at about a third of face value, as a correction to that line:

```
m_offset = m_open + f( m_blind − m_open ,  rest differential ,  |m_open| )
```

The market-blind model does not beat the opening line on its own, but its disagreement with the line still carries information: shrunk hard toward the market anchor, it improves on the opener in six of seven scored seasons. This asymmetry is the project's central finding and the reason the system is built as a correction rather than a standalone forecast.

The offset layer and the soft-availability leg were promoted to production only after clearing pre-registered, season-clustered gates against the incumbent (`D202`, `D224`).

## Data coverage before 2019 is severely limited

The model has four inputs. One of them, the daily NBA injury report that the availability leg is built on, did not exist before 2018-12-17.

![data coverage](charts/data_coverage.png)

| | |
|---|---|
| Seasons with zero injury-report coverage | 11 (2007-08 … 2017-18) |
| First season with partial coverage | 2018-19, at 63.7% |
| First fully covered season | 2019-20 |
| Fully covered seasons available | 7 (2019-20 → 2025-26) |
| Seasons with a measured multi-book price at the open | 1 (2023-24) |

Four consequences follow. First, any figure spanning seasons before 2019-20 measures a different, degraded model: availability composition is half the production margin, and before the feed exists that leg runs on inputs it was never designed to have. Second, the defensible evaluation frame is seven seasons, and season-clustered intervals on it are tens of ROI points wide, so essentially everything tested is statistically indistinguishable from noise. Third, the frame is too short to tune on: a null that takes the best of five randomly chosen game subsets buys +2.54 ROI points (`D187`). Fourth, multi-book execution is largely counterfactual: the best-of-nine tier is observed in 2023-24 only (7.74 books/game), while 2024-25 and 2025-26 observe 1.00 and 1.03.

Each document in `docs/` repeats this caveat in its header, and numbers quoted from this project should carry their frame with them.

## The two reporting frames

Each frame is defined by what data exists, not by which window scored best.

| | frame | why |
|---|---|---|
| Model accuracy | 2019-20 → 2025-26 | the first fully injury-covered season onward |
| Betting headline | 2023-24 → 2025-26 | the recent execution-study window, with the full seven-season frame reported alongside |

Excluding the pre-2019 seasons raised the betting headline (+9.11% against +7.57% on the 14-season blend), which is why the cut had to be made on principle before the numbers were looked at.

## Prediction power

All four forecasts on identical games, 2019-20 → 2025-26 (n=8,286), each converted to a probability with its own walk-forward scale.

![log loss by season](charts/logloss_season_4way.png)

| | opening line | closing line | offset construction | market-blind model |
|---|---|---|---|---|
| pooled log loss | 0.6084 | 0.5980 | 0.6059 | 0.6122 |
| vs the opener | — | −0.0104 | −0.0025 | +0.0038 |

The offset construction improves on the opening line in six of the seven scored seasons and recovers 24.1% of the information the market adds between the open and the close. The market-blind model does not improve on the opener; it recovers −36.4%, moving the wrong way.

Against the closing line, the market-blind normalized gap `(ll_us − ll_mkt) / (ln2 − ll_mkt)` (the share of the closing line's skill above a coin flip left uncaptured) is 13.59%.

![normalized gap by season](charts/frame_model_2019_26.png)

### The two production gates

| | soft availability (`D202`) | offset construction (`D224`) |
|---|---|---|
| season-clustered delta | −0.002265 nats | −0.006378 nats |
| 95% CI | [−0.0041, −0.0004] | [−0.0106, −0.0021] |
| t | −3.39 | −3.68 |
| better in | 5/5 seasons | 6/7 seasons |
| calibration veto | pass | pass |
| verdict | SHIP | SHIP |

Both specs were hashed, with minimum detectable effects stated, before any endpoint was scored.

## Betting record

Priced at the opening spread at −110, flat 1 unit, walk-forward selection, best of nine books.

![equity](charts/review_equity.png)

| | market-blind | offset construction |
|---|---|---|
| 2023-26 pooled ROI | +10.21% | +16.62% |
| cumulative | +54.3u | +76.4u |
| bets | 532 | 460 |
| 95% CI (K=3) | [−18.73, +39.16] | [−6.95, +40.18] |
| positive seasons | 2/3 | 3/3 |
| best-season share of P&L | 84% | 66% |
| 2019-26 pooled ROI | +4.74% | +9.11% |

Both intervals contain zero. The offset arm is better on every dimension (higher ROI, positive in all three seasons, less concentrated, tighter interval, on fewer bets), but three seasons is a small sample and nothing here is statistically resolved.

### Execution

The gain comes from the model rather than line shopping; it holds at a single book.

| execution | 2023-26 ROI |
|---|---|
| 1 book, observed | +10.63% |
| 2 books | +12.91% |
| 5 books | +15.31% |
| 9 books, the reported tier | +16.62% |

The ladder flattens quickly because 36% of the time two books post the same number, so extra books duplicate rather than add.

## What the model is

```
margin = 0.5·four_factors + 0.5·availability_composition + schedule_layer + tank_term
P(home win) = sigmoid(margin / 7.2)
m_offset = m_open + f(m_blind − m_open, rest_diff, |m_open|)
```

The architecture and its transformations are fixed ex ante. Within each walk-forward fold the team ratings, availability probabilities, schedule effects and offset coefficients are estimated from prior seasons only.

| component | what it is | how it is fit |
|---|---|---|
| Four factors | opponent-adjusted ratings on shooting, turnovers, rebounding, free-throw rate | one L2 ridge solve per factor (`ridge=25`), mapped to points by a fitted linear map |
| Availability composition | Σ over players of DARKO talent × trailing minutes / 48, each weighted by 1 − P(out) forecast from information available at the open | requires the daily injury report; this is why the frame starts 2019-20 |
| Schedule layer | home edge, back-to-backs, dead-team flags | walk-forward, EB shrinkage `n/(n+600)`, `team_home_ridge=200`; the only component to survive strict out-of-sample testing on every split tried |
| Tank term | late-season effort | exactly zero outside its window |
| Offset layer | the correction applied to the opening line | ridge shrunk hard toward zero; edge coefficient 0.33–0.37 in every fold |

The link scale 7.2 is derived rather than hand-set: matching a logistic to the margin-residual normal gives 7.53 from training residuals alone. [docs/CONSTANTS.md](docs/CONSTANTS.md) separates the constants that are derivable from those that are genuinely free.

Market-blindness is enforced by construction. `nbapred/market/offset.py` takes the blind margin as an argument and cannot reach back into the model that produced it, so the blind model never sees a price. With no opening price available the layer returns the blind margin unchanged; a dead odds feed costs the correction but not the prediction.

## Which games are in the model

Every model surface filters on the `002` game-id prefix.

| prefix | | games | in? |
|---|---|---|---|
| `002` | regular season | 35,546 | yes |
| `004` | playoffs | 2,440 | no |
| `001` | preseason | 2,019 | no |
| `003` | All-Star weekend | 83 | no |
| `005` | play-in | 37 | no |
| `006` | NBA Cup final | 3 | no; does not count in the standings |

NBA Cup group-stage games are in, because they do count in the standings. A pre-registered difference-in-differences found no detectable effect from them (MDE 2.21 points, `D179`).

## Limitations

Manufacturing capacity. Tuning to a single season yields positive in-sample ROI on 19 of 19 seasons: mean +15.79% in sample against −1.13% out. The identical procedure on pure noise manufactures +17.46. Net of noise this project's capacity is −0.55 (p = 0.685). Every development-versus-out-of-sample gap in the register is smaller than what a modest grid search produces from nothing.

Short-window headlines are mostly selection. The best 3-season window returns +16.95% against a +9.03% average across all 3-season windows; the best 4-season window +14.92% against +9.33%. Roughly +5.6 to +7.9 points of any short-window headline comes from the act of choosing it (`D208`).

Beating a permutation null is not enough on its own. Three selectors each beat their own null (p ≤ 0.048 after correction) and all three lost to the incumbent (`D176`).

The model is era-specific by construction. Ablating the era-specific context terms flips the sign of the betting result, and zero shipped components can be dated to a gate using only pre-2021 data. 2026-27 is the first genuinely prospective test.

CLV is used as a monitor rather than an objective. A divergence selector bought more CLV and less ROI; a CLV-targeted selector bought no extra CLV and the most ROI (`D176`).

Bet volume will fall. The offset layer shrinks disagreement with the market by ~65%, so margins that used to clear the betting threshold no longer do. This was caught by a test assertion rather than anticipated (`D226`).

## Errata: errors made and caught

- An availability leak in the certified backtest: the capstone built injury lists from that night's box score. The live path was clean, so no prediction was ever wrong, but the published expectation was too good by 3.8 points of normalized gap (`D158`).
- A bet-time information leak worth 33% of the model's deficit: the availability leg used the 5PM report and pregame inactives, both published after the opening line it transacts at. Rebuilding by carry-forward widened the gap 12.87% → 17.17%; forecasting `P(out)` instead recovered 52% of that (`D199`–`D202`).
- A team-name join that silently dropped rows, four times. The fix was a resolver that reports unresolvable names; on the fourth instance it caught 28 of 30 franchises failing in a new feed (`D171`, `D177`).
- Two reporting frames corrected in public: the model frame was off by one season, the trading frame by seven (`D186`, `D207`).
- A confidence interval that briefly claimed false significance, centred on an n-weighted mean while taking its dispersion from an unweighted one (`D216`).
- A false equivalence between two correctly computed numbers. No gate, null or test can catch that; it took an outside reader asking what the denominators were (`D218`).
- A chart axis that hid the one season the model won: a hard-coded `ylim(0, …)` clipped 2008-09 off the bottom of the frame (`D171`).

## Charts

`charts/` holds current renders. The ones that carry the results:

| chart | contents |
|---|---|
| `data_coverage.png` | measured coverage of all four model inputs, by season |
| `logloss_season_4way.png` | log loss by season: opener, close, offset construction, market-blind model |
| `frame_model_2019_26.png` | per-season normalized gap on the corrected frame |
| `review_equity.png` | the 460-bet equity path behind the headline |
| `sim_report_equity_offset.png` | full-frame equity for the offset arm |
| `sim_report_equity.png` | the same for the market-blind arm |

Superseded renders are moved to `charts_archive/` with a datestamp rather than deleted, so a number that changed can be traced to the run that changed it.

## Layout

```
nbapred/            the model
  model/            four factors, composition, schedule layer, tanking, bridge
  market/           offset layer, anchored CLV model, windows
  engine/           props simulator, star-out redistribution, slate assembly
  eval/splits.py    rolling-origin / LOSO / block / era / clustered inference
  ingest/           odds, nba_api, DARKO, injury-report PDFs, historical odds
docs/               the research record — start with DECISIONS.md (D170+;
                    earlier entries in DECISIONS_ARCHIVE.md, split at 1 MB
                    because GitHub stops rendering markdown above that)
scripts/            gate scripts, backtests, backfills, the paper bet engine
charts/             current results only
tests/              153 tests, including leakage and reproducibility guards
```

Suggested reading order: [docs/REVIEW.md](docs/REVIEW.md) (results), [docs/FAILURES.md](docs/FAILURES.md) (what was tried and why it failed, grouped by cause of failure), [docs/DECISIONS.md](docs/DECISIONS.md) (the register, D170+) and [docs/DECISIONS_ARCHIVE.md](docs/DECISIONS_ARCHIVE.md) (D1-D169), [docs/GATE_POLICY_V2.md](docs/GATE_POLICY_V2.md) (how a change gets promoted), [docs/LEAKAGE.md](docs/LEAKAGE.md), [docs/CONSTANTS.md](docs/CONSTANTS.md), [docs/OCTOBER_RUNBOOK.md](docs/OCTOBER_RUNBOOK.md).

## Method

Every gate is pre-registered with a SHA-256 hash written before any endpoint is scored. Evaluation is out-of-sample with season-clustered confidence intervals, checked across rolling-origin, leave-one-season-out, block-bootstrap and legacy splits, decomposed by era, and corrected for multiple comparisons across the running family. A change ships only if it clears all of that and a calibration veto.

Two working rules: hypotheses are stated before configurations are tried (no blind grids), and a term whose fitted sign contradicts its stated mechanism is treated as a null regardless of how significant it looks.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env        # add your own keys; .env is gitignored
python scripts/pull_nba_daily.py
python scripts/build_features.py
python scripts/prod_by_season.py       # the capstone backtest
python -m pytest tests/ -q
python scripts/canary.py               # pre-flight / in-season tripwire
```

`scripts/canary.py` runs nine checks, each traceable to a failure that actually happened here: feed staleness, the multi-book logger never having run in-season, the team-name join bug class, non-regular-season contamination, tank-floor drift, CLV band breach and calibration sanity. It exits non-zero on any hard failure so it can be wired to cron.

Not included: the 13 GB DuckDB corpus, raw API captures, and all credentials. The ingest scripts rebuild the corpus from public sources.

## Next steps

1. Capture at least two books at the open, from opening night. Best-of-two lifts CLV by about 49%, and taking the worse book erases nearly all of it. The logger has never run in-season and `odds_quotes` is empty. If it is down on opening night, that season's open-price record cannot be reconstructed, which makes this the only irreversible item on this list.
2. Run both arms prospectively in 2026-27, offset as primary and market-blind as its live control. They share 65% of their bets and take the same side on every overlapping game, so this is a controlled comparison rather than a portfolio (`D205`).
3. Use CLV as the live yardstick. It is positive in 17 of 19 seasons, needs no devig convention, and resolves in weeks where ROI needs decades.
4. Exchange access, if it becomes available. It is the largest cost lever measured here, and an access problem rather than a modelling one.
5. Props before sides: both shipped improvements of the last cycle came from the props engine.

Further feature search on the sides model is not planned. Nineteen seasons, an exhaustion audit, a 49-feature battery and a possession-level rebuild all point the same way, and the capacity number above explains why marginal features keep looking real and then failing to transfer.

## Status

The offset construction and soft availability are both in production, both gated. No capital has been deployed. The open question is whether live CLV against opening prices, with two books captured, reproduces the backtested relationship.

## Licence & disclaimer

For research and educational purposes. Nothing here is betting advice, and the project's own conclusion is that nothing in it is statistically resolved.
