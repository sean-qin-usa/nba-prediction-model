# Paid options — deferred, not forgotten

> **DATA-COVERAGE CAVEAT — READ BEFORE QUOTING ANY NUMBER IN THIS FILE.**
> The daily NBA injury report, which the model's availability leg depends on,
> begins **2018-12-17 — mid-way through 2018-19**. Coverage of regular-season
> game dates is **0% before that, 63.7% in 2018-19, and 95–100% from 2019-20
> onward**. Only **2019-20 → 2025-26 (7 seasons)** is fully covered, and that is
> the only frame in which the model runs as designed. Earlier seasons score a
> *crippled variant* whose availability leg is fed inputs it was never meant to
> have. Any figure here spanning seasons before 2019-20 — including every
> 14-season and 19-season figure — blends two different models and should be
> read as historical context, not as a description of the shipped system.
> (`D186`)

Standing constraint (Sean, 2026-07-26): **$0 budget. Nothing paid.** This file
marks exactly what each paid item would unlock, so upgrades are a decision,
not a rediscovery. Free substitutes in use are listed with their gaps.

## 1. The Odds API paid tier ($30–99/mo)
- Free tier: 500 credits/mo. One main-lines snapshot (h2h+spreads+totals, us)
  costs 3 credits → ~5 snapshots/day and **zero props**. No line-movement
  resolution, no close capture on multi-game nights, H-B data starves.
- 20K tier ($30/mo): ~10-min main-line cadence + props inside 24h of tip —
  the cadence the handoff assumes. 100K tier adds us2 region (sharper books)
  and SGP-adjacent markets.
- **Free substitute:** budget-paced logger (see `ODDS_MONTHLY_BUDGET`) +
  ESPN scoreboard capture (ESPN BET lines, unlimited, one book, no props).
  Gap: line-path granularity, prop distributions, multi-book de-vig quality.

## 2. The Odds API historical snapshots (one-off purchase, ~$0.001/credit)
- 10-min historical odds (incl. props since 2023) back to 2020. Would let the
  H-A open→close backtest and H-B prop calibration run THIS offseason on real
  multi-book line paths.
- **Free substitute:** sportsbookreviewsonline archives (2007-08 → 2022-23):
  open, close, ML, totals, 2H — but only open/close (no path), one composite
  book, no props, and nothing after 2022-23. H-A backtest is possible on
  open→close movement; H-B is not.

## 3. VPS (~$5/mo)
- Handoff I says logger uptime matters from day one; a VPS survives home
  power/net/reboots.
- **Free substitute:** this machine + systemd + cron with `timeout` guards.
  Gap: home-box downtime = permanent capture holes. Acceptable in offseason;
  revisit before October.

## 4. Synergy play-type data (institutional; effectively unavailable)
- v1.5 tendency priors (PnR/iso/post-up frequencies + PPP).
- **Free substitute (and likely permanent choice):** derive action-mix and
  efficiency splits from our own play-by-play corpus — real NBA frequencies,
  no 2K bias. Decision journaled 2026-07-26.

## 5. Betting exchange / sharp book access (state-dependent, funded account)
- Live-money execution and exchange market-making (I.3). Blocked anyway on
  Illinois legality check — genuinely a later decision, not just budget.

## 6. Second odds aggregator (OddsJam/DonBest-class, $$$)
- Injury-timestamped line audit trails, more books, faster push feeds.
  Only relevant if CLV measurement becomes the binding constraint.

## Money-no-object spec (external review xhigh, 2026-07-30; full text data/logs/review_gpt2.md)
Buy order for a DK-tier sides model (annual, budgeting estimates):
1. Second Spectrum/Genius tracking $300k-1.5M → +0.003-0.006 (ONLY item that materially moves sides alone)
2. Sportradar official NBA push feeds $150k-500k → +0.001-0.002 (PIT state QA, latency)
3. One tagging product (Stats Perform/SIS) → small; 4. One news/human-info product → +0.0005-0.0015
5. Don Best screen → 0 model value (execution only). Diminishing returns after top-4.
Channel decomposition of our remaining ~+0.012 vs close: October state 0.0025-0.0035 (free),
April intent 0.002-0.003 (free), role topology 0.0015-0.0025 (free+news), matchup geometry
0.0015-0.0025 (tracking), win-map/tails 0.001-0.0015 (free), private+aggregation 0.0015-0.0025
(unreachable). Free-extractable ≈ 0.005-0.007 via ARCHITECTURE: (1) hierarchical player-lineup
state-space replacing late 50/50 fusion (+0.003-0.005), (2) multi-task side/total/props shared
latents (+0.0015-0.003), (3) matchup-conditioned game distribution replacing sigmoid/7.2
(+0.001-0.0025), (4) regime-switching uncertainty head for Oct/Apr/chaos (+0.0015-0.0025).
HARD CALL: free-data ceiling ≈ 0.581 pooled vs close 0.5778 — sides = match-close-minus-epsilon;
durable edge lives in props/derivatives/timing. (Aligns with D13/D14.)
