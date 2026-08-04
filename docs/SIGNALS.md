# Prediction signals (model inputs, NOT the market)

The simulator is market-blind (handoff I.6): odds are benchmark + CLV KPI +
betting-blend only, never a forecast input. Everything below is a legitimate
*forecast* signal. Status: [x] captured, [~] partial, [ ] planned.

## In the handoff, being built
- [x] Player skill priors — nba_api box/PBP stats, DARKO, 2K (docs/PRIORS.md)
- [x] Sufficient statistics per player-game (possessions.py)
- [x] Lineup-stint margins / RAPM substrate (stints.py)
- [~] Minutes & availability — injury-report PDF poller armed; minutes GBM (II.4)
      DEFERRED on purpose (see note below)
- [ ] Shot-quality / defender distance (v1.5; x/y coords already captured in PBP)
- [ ] Age curves per dimension (II.1)
- [ ] College-to-NBA translation priors for rookies (v1.5)

## Added beyond the handoff (this session)
- [x] **Referee crew** (referees.py) — refs measurably shift foul rate, FT
  volume, pace, totals; assignments are published PREGAME so they're knowable
  before the line settles. Retrospective crews (BoxScoreSummaryV2) build per-ref
  tendencies from our PBP; the pregame feed is the forward signal. Feeds the
  foul-draw logit (II.3.3) as a crew fixed-effect + a standalone totals signal.

## Other candidate signals — ranked by expected value, all free

1. **Rest / schedule fatigue** — [x] BUILT (`schedule_features`, schedule.py).
   Back-to-backs, 3-in-4, 4-in-5, days-rest, travel_km (haversine), tz_shift,
   and the opponent-differenced rest_adv / travel_adv (the edge is in the
   asymmetry). 2,460 team-game rows for 2025-26; 441 back-to-backs. NOT yet
   wired into a forecast — it's a feature table awaiting the minutes model +
   a team fatigue term, each of which must pass the docs/COMPLEXITY.md gate.
2. **Confirmed starting lineups** (HIGH, in-season only). Beat writers / league
   feed post starters ~30 min pre-tip; collapses availability uncertainty right
   when the closing line firms. Pairs with the injury poller.
3. **Travel/altitude** (LOW-MED). Denver/Utah altitude has a small, real home
   edge and visitor-fatigue effect; cheap to add as a venue covariate.
4. **Player tracking fatigue proxies** (MED, v1.5). nba_api tracking endpoints:
   speed/distance load over recent games as a fatigue regressor for minutes and
   efficiency — the handoff lists these under II.4 already.
5. **Roster continuity / games-together** (MED). How long the current rotation
   has played together — proxies chemistry the residual synergy term can't yet
   identify; derivable from `lineup_stints` history.
6. **Coaching (restricted, per II.5)**. Rotations/minutes are modeled directly;
   scheme is team fixed-effects + covariates; in-game adjustment skill is
   deliberately ignored in v1. No change recommended.
7. **Public bet %/splits** — NOT a forecast input. Logged only as a model-free
   benchmark stream (steam-chase / public-fade), per I.5.

## Why the minutes model (II.4) is deferred, not skipped

It's next on the roadmap and H-A-critical, but building it *now* would violate
docs/COMPLEXITY.md: (1) its strongest features are injury-report status ×
team-specific resolution rates — and there are **zero injury PDFs in the
offseason** to fit or validate against; (2) the season backfill isn't finished,
so EWMA-minutes / rotation history is partial; (3) with no way to run the
walk-forward ablation gate yet, we'd be shipping an unvalidated GBM — exactly
the "added a thing we can't prove helps" trap. Prerequisites: full backfill
complete + first real injury reports (October). The `schedule_features` it
depends on are already built and waiting.

## Explicitly excluded as forecast inputs
- Any market line (I.6 market-blind rule) — benchmark/CLV/blend only.
- Narrative/"motivation" without a rule (revenge games, etc.) — no PIT-clean,
  non-hindsight definition; skip until one exists.
