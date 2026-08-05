# Paid retrospective — what money would actually have bought (2026-07-31)

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

Sean's ask: "figure out what we could have paid for." This consolidates every
paid option ever logged (docs/PAID_OPTIONS.md, docs/PAID_ORACLES.md, the codex
money-no-object spec, DECISIONS.md oracle measurements) into one final
worth-it table. The unusual asset here: we MEASURED most of these with oracle
stand-ins instead of guessing, so verdicts cite D-numbers, not vibes.

Units: sides log-loss delta per game per season unless noted. Context for
scale: total remaining gap to close ≈ +0.010-0.019 by season; D73/D90-class
shipped wins are +0.002-0.006; the T2 registration band starts at +0.0005.

## The worth-it table

| # | Product (vendor class) | Est. annual cost | Measured / bounded value | Verdict |
|---|---|---|---|---|
| 1 | Real-time scratch/availability wire (Sportradar live injury wire, FantasyLabs-class alerts) | $5-20k | **+0.0037 measured** — report-based OUT-sets cost exactly this vs the oracle played-set (PAID_ORACLES #8; D63/D68 two-tier gap ~0.0035/season). The wire recovers most but not all (last-second scratches are irreducible). | **WOULD-BUY** — the largest measured model delta per dollar; already Sean's committed would-buy (2026-07-30) |
| 2 | Second Spectrum / Genius full tracking license | $300k-1.5M | **Pooled +0.0004 NS; heavy-favorite subset +0.0059 CI(+0.0026,+0.0090), positive all 3 seasons (D72 oracle)** — closes ~22% of the heavy-fav gap; chaos teams NS wrong side. Codex blind estimate +0.003-0.006 — our measurement says the truth is the conditional pocket, not the pooled number. | **CONDITIONAL** — institutional bankroll only; per-dollar absurd for solo. If free, wire it as a conditional heavy-fav term (needs own gate) |
| 3 | The Odds API 20K tier | $360 ($30/mo) | No direct model delta (market-blind rule) — but D69's verdict is that the remaining money is in PRICES, not model: earlier lines, soft books, props, multi-book de-vig. This is the tier the bet-engine handoff assumes (10-min cadence + props). | **WOULD-BUY** — first dollar spent; unlocks the actual money loop that D91's paper-trade engine is built for |
| 4 | The Odds API historical snapshots (one-off) | ~$50-200 one-off | Unlocks H-A open-to-close and H-B prop-calibration backtests on real 10-min multi-book paths incl. props since 2023 (PAID_OPTIONS #2). Free substitute (SBR archives) has no path, no props, ends 2022-23. | **WOULD-BUY** — one-off, small, converts two blocked backtests into runnable ones |
| 5 | VPS | $60 ($5/mo) | Prevents permanent odds-capture holes from home-box downtime (PAID_OPTIONS #3); matters from opening night. | **WOULD-BUY** before October |
| 6 | dunksandthrees EPM premium | ~$60-100 | **~0 measured, twice.** D86 pre-registered talent-ensemble gate: NULL (-0.00006 pooled). D94 second look with the re-identified DAILY grid (median EPM age 1 day vs 2-13 weekly): still NULL (+0.00021 CI(-0.00106,+0.00142)); fresher EPM raises its skill (0.341->0.355) but ALSO its error correlation with DARKO (0.889->0.909, over the 0.90 no-op line) — redundancy, not staleness. And the daily history is free anyway post-D94 re-id (epm_history_daily, top-5 validation 2690/2690). | **SKIP** — measured ~0 for the model; sub would be charity/UI only |
| 7 | Projection/minutes services (DARKO Patreon tier, props-projection sites) | $60-600 | Bounded above by the ORACLE-MINUTES ceiling: **perfect minutes worth only ~0.003 sides** (BRIEF_UPDATE2 #4), and 11.2% of prop error (D12). DARKO's own x_minutes REJECTED as props projector (D45) — our EWMA/Kalman path already sits near the free frontier. | **SKIP** — the oracle bound says even perfection is small; a paid projection is not perfection |
| 8 | Synergy play-type data | institutional | Not sold at our scale; own-play-by-play action-mix chosen as the permanent substitute (decision journaled 2026-07-26; real NBA frequencies, no 2K bias). | **SKIP** (closed) |
| 9 | Sportradar official NBA push feeds | $150-500k | Codex est +0.001-0.002 (PIT state QA, latency). Unmeasured by us; class overlaps #1's value. | **SKIP** at our scale — #1 captures the same channel cheaper |
| 10 | Tagging products (Stats Perform / SIS) | $$$ | Codex: "small". No oracle built; nothing in our residual decomposition points here. | **SKIP** |
| 11 | News/human-info product | $$ | Codex est +0.0005-0.0015; official injury-report PDFs (free, 45k PIT rows) already feed the OUT-sets that get within +0.0037 of the played-set oracle. | **SKIP** — marginal over free PDFs + wire (#1) |
| 12 | Don Best screen / second aggregator (OddsJam-class) | $$$ | **0 model value** (market inputs FORBIDDEN by the market-blind rule); execution/CLV audit only (PAID_OPTIONS #6, PAID_ORACLES #11). | **SKIP** unless CLV measurement becomes the binding constraint |
| 13 | Biometric load data (Kinexon/Catapult) | n/a | League-restricted, not sold to bettors (PAID_ORACLES #9). | closed — not purchasable |
| 14 | True insider info | n/a | Not a legal product; shops buy SPEED not secrets (PAID_ORACLES #10); bounded above by #1-class oracles anyway. | closed |
| 15 | Betting exchange / sharp book funded account | state-dep. | Execution venue, not data. Blocked on the Illinois legality check before any budget question (PAID_OPTIONS #5). | **CONDITIONAL** — legality first; required eventually for live money |
| 16 | Charter/flight tracking (FlightAware) | $ | Schedule-derived travel (free) is the upper bound; delays marginal (PAID_ORACLES #13). | **SKIP** (tiny) |
| 17 | Synergy college / G-League | $$ | Free sports-reference CBB covers rookie-translation needs (PAID_ORACLES #12). | **SKIP** |

## The codex money-no-object spec (2026-07-30; full text data/logs/review_gpt2.md)

Buy order for a DK-tier sides model (annual): (1) Second Spectrum/Genius
tracking $300k-1.5M -> +0.003-0.006 (only item that materially moves sides
alone — our D72 oracle refines this to "heavy-fav pocket +0.0059, pooled NS");
(2) Sportradar official push feeds $150k-500k -> +0.001-0.002; (3) one tagging
product -> small; (4) one news/human-info product -> +0.0005-0.0015; (5) Don
Best screen -> 0 model value (execution only). Diminishing returns after
top-4. Channel decomposition of the remaining ~+0.012 vs close: October state
0.0025-0.0035 (free — since banked by D84-A/D91), April intent 0.002-0.003
(free — banked by D73/D90), role topology 0.0015-0.0025 (free+news), matchup
geometry 0.0015-0.0025 (tracking), win-map/tails 0.001-0.0015 (free),
private+aggregation 0.0015-0.0025 (unreachable). Free-extractable ~0.005-0.007
via architecture. HARD CALL: free-data ceiling ~ match-close-minus-epsilon on
sides; durable edge lives in props/derivatives/timing (aligns D13/D14/D69).

## Bottom line — the ranked shopping list

If Sean ever spends money: buy in this order. (1) **The Odds API 20K tier +
the one-off historical snapshots + a VPS, ~$520 first year total** — zero
model delta but it purchases the thing the whole program is now bottlenecked
on per D69/D13/D14: better PRICES (props, line paths, multi-book de-vig,
uptime) for the already-built D91 paper-trade -> live loop. (2) **A real-time
availability wire, $5-20k/yr** — the only paid MODEL input with a clean
measured payoff (+0.0037/season, the entire free-vs-bought tier gap), worth it
the moment real stakes exceed ~$5k/yr of expected edge. (3) **Nothing else.**
Tracking (+0.0059 but only in heavy-fav pockets, pooled NS) waits for an
institutional bankroll; EPM subs, projection services, talent metrics, news
products, and odds screens are all measured-or-bounded at ~0 given what the
free stack already extracts. Expected total model gain from the full sane list
(#1 wire + conditional tracking pocket): **~+0.004-0.006 pooled sides**, on
top of which the cheap execution tier converts modeled edge into collectable
edge — everything beyond that is either free-reachable by architecture
(~+0.005-0.007 per codex) or unreachable market aggregation (~+0.002).
