# NBA Prediction Model

A market-blind NBA win-probability model, and the full research record behind
it: **161 registered decisions, most of them rejections.**

The model is the smaller half of this repository. The larger half is
`docs/DECISIONS.md` — an append-only register in which every experiment was
pre-registered, gated out-of-sample, and written down whether it worked or not.
Several of the most useful entries document mistakes we made and caught.

---

## What the model is

```
margin = 0.5 · four_factors + 0.5 · availability_composition
       + schedule_layer + tank_term
P(home win) = sigmoid(margin / 7.2)
```

Two independent estimates of team strength, averaged, plus additive context:

- **Four factors** — opponent-adjusted ridge ratings on shooting, turnovers,
  rebounding and free-throw rate, mapped to points by a fitted linear map.
- **Availability composition** — Σ over *available* players of
  `DARKO_talent × trailing_minutes / 48`. This is the leg that reacts to
  injuries, and it is why the model is market-blind but not uninformed.
- **Schedule layer** — home edge, back-to-backs, dead-team flags, estimated
  walk-forward with shrinkage toward a prior. The only component that has ever
  survived strict out-of-sample testing on every split we have tried.
- **Tank term** — late-season effort, exactly zero outside its window.

Regularisation is used throughout, in four forms: L2 ridge on the ratings solve
(`ridge=25`; `team_home_ridge=200` on per-team home deviations), empirical-Bayes
shrinkage toward a prior in the schedule and tank layers (`n/(n+600)`), data
augmentation via prior-season pseudo-observations, and Bayesian/EB priors in the
usage and props fits.

**The model never sees market odds.** Bet selection may use the price available
at bet time; the model may not. That rule is what makes the comparisons below
mean anything.

## What the record actually shows

Stated plainly, because the point of the register is that it does not flatter
the project.

| | |
|---|---|
| Seasons evaluated | **19** (2007-08 … 2025-26), contiguous |
| Market beats the model in | **19 of 19 seasons** |
| Normalized gap, live injury feed | **11.45%** of the market's skill-above-coinflip |
| Normalized gap, availability-blind | **20.88%** (K=19, significant) |
| Frozen betting rules, 15 unseen seasons | **−5.60% ROI, significantly negative** |
| Closing-line value (CLV) | **+0.0066/bet, significant, era-stable, placebo p=0.000** |
| Model's worth over a no-information bettor | **+1.67pp of ROI** |
| Needed to break even after vig | **≈ +3.4pp more** |

The last two lines are the whole finding: **the information is real, it is
measurable, and it is smaller than the vig.**

### We beat the opening line. We do not beat the price of betting it.

Tested on **22,742 games across 19 seasons** against the real opening spread —
the largest and cleanest test in this repository, and one with no probability
conversion anywhere in it (see *For non-bettors* below for what these terms
mean):

| our margin vs the opening spread | result |
|---|---|
| beats a coin flip (50%)? | **yes — 50.65%, +0.65pp, significant** |
| beats the break-even a bookmaker charges (52.38%)? | **no — short by 1.73pp, significant** |

Both statements are true at once, and the second one is what decides whether
money is made. On the 14 seasons the model has never seen, the edge over a coin
flip is positive but **not** statistically significant, so the honest version of
"we beat the open" is: *demonstrated over the full 19 seasons, not demonstrated
on out-of-sample data alone, and never large enough to pay for itself.*

**How small is the edge, concretely?** We disagree with the opening line by
**2.455 points** per game on average. If that disagreement were entirely real
information, we would cover 57.6% of the time. We cover 50.65%. So the genuine
content of our disagreement is **0.206 points — 8.4% of what we claim, and the
rest is noise.** Breaking even requires 0.751 points. **We deliver 27% of the
edge needed.**

For scale, a fair comparison nobody asked for and we ran anyway: *betting every
road team* returns −3.26%. We return −3.25%. Our selection genuinely adds
+1.18pp over a composition-matched random selector — it simply spends that
advantage buying back the road exposure the trivial strategy gets for free.

### For non-bettors: the four terms that matter

- **The spread** is a handicap that makes an uneven game a coin flip. "Lakers
  −6.5" means back the Lakers and they must win by 7+ for you to collect. It is
  a *prediction of the margin*, which is exactly what this model outputs — so
  comparing our margin to the spread is the most direct test of the model there
  is, with no probability maths in between.
- **The vig** (or juice) is the bookmaker's fee, charged by making both sides
  pay less than even money. Standard is −110: risk $110 to win $100 on either
  side. That's why break-even is **52.38%**, not 50% — you must be right 52.38%
  of the time just to stand still. The vig is why a real edge can still lose
  money, and it is the single most important number in this repository.
- **Line shopping** is placing each bet at whichever bookmaker offers the best
  number. Books disagree — one may post −6.5 while another posts −6 — and since
  our edge is roughly the size of the fee, a half-point of price is material.
  More books means more chances to find the best number. *Caveat measured here:*
  36% of the time our two books post exactly the same number, so extra books
  duplicate rather than add, and the benefit flattens fast.
- **An exchange** (Betfair, Sporttrade, Prophet X) is a marketplace where you
  bet against other people instead of against the house. The difference is how
  it charges: a bookmaker bakes its fee into every price, win or lose, while an
  exchange takes commission **only on net winnings**. That cuts the cost of
  trading from about 3.00 points to about 0.41 — which is why it is the largest
  single lever in this project, and it is an access problem rather than a
  modelling one.

**What data we actually hold for those last two, stated plainly:** two books,
historically, for 2021-26. Our own multi-book logger can pull eight or more US
books but **has never run during a season**. We hold **no exchange data at
all** — the exchange figures below are our existing bets repriced under an
assumed commission, never executed. Treat both as arithmetic, not evidence.

### Closing prices vs opening prices

Those headline ROIs are struck at the **close**, where market odds exist for all
19 seasons. You would never actually bet there — the close is the market's final
and best answer. The interesting price is the **open**, and it is a weaker test
for an unavoidable reason: real opening moneylines only exist in our data for
**three seasons**, which is two degrees of freedom and cannot reject anything.

| frame | seasons | union ROI | verdict |
|---|---|---|---|
| **Close**, real moneylines | **19** | **−3.40%** | **significantly negative** |
| Close, 15 seasons no gate ever saw | 15 | **−5.60%** | **significantly negative** |
| **Open, real spreads (ATS)** | **19** | **−3.25%** | **significantly negative** |
| Open, ATS, 14 seasons never seen | 14 | **−3.64%** | **significantly negative** |
| Open, real moneylines, live injury feed | 3 | −0.66% | not significant (2 dof) |

So: **at the close we can prove the rules lose. At the open we can prove
nothing** — the point estimate is roughly breakeven and the interval swallows it.

Execution is the larger lever, and it is an access problem rather than a
modelling one. Holding the model fixed and varying only where the bet is placed
(fully-equipped model tier, at the open):

| execution | cost over fair | union ROI |
|---|---|---|
| 1 retail book (our access) | 3.00 pts | −0.95% |
| 2 books | — | +0.39% |
| 5 books | 1.10 pts | +1.76% |
| 8 books | 0.67 pts | +2.39% |
| exchange, 2% commission | **0.41 pts** | **+2.69%** |

A single exchange account structurally beats an extrapolated eight-book shop,
because commission scales with the payout and these rules are 68% favourites.
**None of these cells survives the three-season significance bound**, so they
are reported as point estimates and a direction, not as a result.

The honest summary of the trading work: *the rules are rejected where we have
power to test them; at the price we would actually pay they are indistinguishable
from breakeven; and the only lever large enough to matter is transaction cost.*

The betting rules are *rejected*, not merely unconfirmed — they look least bad
precisely on the seasons they were selected from, which is the overfitting
signature. Closing-line value survives everything we have thrown at it,
including a two-decade out-of-sample frame and a within-date permutation
placebo, so it is the only quantity here worth tracking live.

## What we got wrong, and caught

These are in the register with numbers; they are listed here because they are
the most transferable part of the project.

- **An availability leak in the certified backtest.** The capstone built injury
  lists from *tonight's box score*. The live path was clean, so this never made
  a prediction wrong — it made our published expectation too good, by 3.8
  points of normalized gap. Found by a ceiling study, not by a test. (`D158`)
- **A switch named after a hypothesis it did not implement.** We nearly shipped
  a registered *loser* because an environment flag's name matched a term that
  had passed, while the code behind it was a different construction. (`D141`)
- **A "significant" coefficient that existed only because of COVID.** Travel and
  schedule-density effects were significant on a frame that included the 2020
  bubble — where travel was structurally zero and our code assigned 1,505 km.
  On scorable seasons they are null. (`D136`, `D140`)
- **Confidence intervals that were too narrow.** Every sides gate used an i.i.d.
  bootstrap where per-game deltas share fitted coefficients. Correcting to
  season-clustered inference reproduces, from inference alone, the two terms we
  had already reverted for other reasons. (`D139`)
- **Season literals and derived floors that moved under us.** A corpus backfill
  silently changed a data-derived constant and invalidated a certified table
  twice. It is now pinned, with a drift detector. (`D131`, `D153`, `D155`)

## Layout

```
nbapred/            the model
  model/            four factors, composition, schedule layer, tanking, bridge
  engine/           props simulator, star-out redistribution, slate assembly
  eval/splits.py    rolling-origin / LOSO / block / era / clustered inference
  ingest/           odds, nba_api, DARKO, injury-report PDFs, historical odds
docs/               the research record — start with DECISIONS.md
scripts/            gate scripts, backtests, backfills, the paper bet engine
charts/             current results only
tests/              129 tests, including leakage and reproducibility guards
```

Suggested reading order: `docs/DECISIONS.md` (the register),
`docs/GATE_POLICY_V2.md` (how a change earns its way in), `docs/LEAKAGE.md`
(what counts as legitimate information and what does not), and
`docs/LIMITATIONS.md`.

## Method

Every gate is pre-registered with a SHA-256 hash written before any endpoint is
scored. Evaluation is out-of-sample with **season-clustered** confidence
intervals, checked across rolling-origin, leave-one-season-out, block-bootstrap
and legacy splits, decomposed by era, and corrected for multiple comparisons
across the running family. A change ships only if it clears all of that *and* a
calibration veto — and several changes that passed six of seven conditions were
still declined.

Two rules carry most of the weight: **hypotheses come before configurations**
(no blind grids), and **a term whose fitted sign contradicts its stated
mechanism is a null, however significant it looks.**

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env        # add your own keys; .env is gitignored
python scripts/pull_nba_daily.py
python scripts/build_features.py
python scripts/prod_by_season.py       # the capstone backtest
python -m pytest tests/ -q
```

**Not included:** the 13 GB DuckDB corpus, raw API captures, and all
credentials. The ingest scripts rebuild the corpus from public sources; raw
files are ground truth and the database is derived and rebuildable by design.

## Status

Research is complete; the model is not in production and no capital has been
deployed. The open question is whether closing-line value measured live —
against opening prices, with at least two books captured — behaves the way
nineteen backtested seasons say it should.

## Licence & disclaimer

For research and educational purposes. Nothing here is betting advice, and the
project's own conclusion is that these rules lose money at retail prices.
